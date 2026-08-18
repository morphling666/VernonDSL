#include "native_execution_graph_autodiff.h"

#include "execution_graph/execution_graph_internal.h"
#include "native_operator.h"
#include "native_pipeline_autodiff.h"

#include <nanobind/stl/unique_ptr.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {

nb::object plannedDeviceValue(nb::object value, vernon::execution::detail::RhiCommandPlanSink &sink) {
    if (nb::hasattr(value, "_planned_device_write_pending") &&
        nb::cast<bool>(value.attr("_planned_device_write_pending")()))
        return value;
    if (nb::hasattr(value, "_device_dirty") && nb::cast<bool>(value.attr("_device_dirty")))
        return value;
    nb::object storage;
    nb::object result;
    if (nb::hasattr(value, "_begin_planned_upload") && nb::hasattr(value, "_planned_buffer")) {
        storage = value;
        result = value;
    } else if (nb::hasattr(value, "_resident_buffer")) {
        throw std::runtime_error("planned graph cotangent does not support transactional residency");
    } else {
        nb::object array = nb::module_::import_("numpy").attr("ascontiguousarray")(value);
        storage = nb::module_::import_("vernon_dsl").attr("storage").attr("from_numpy")(array);
        result = storage;
    }
    nb::tuple residency = nb::cast<nb::tuple>(storage.attr("_begin_planned_upload")());
    if (residency.size() != 3)
        throw std::runtime_error("planned graph cotangent residency transaction is invalid");
    nb::object bufferObject = nb::borrow<nb::object>(residency[0]);
    nb::object transaction = nb::borrow<nb::object>(residency[1]);
    auto *buffer = nb::cast<RhiBuffer *>(bufferObject);
    if (!buffer || !buffer->host)
        throw std::runtime_error("cannot allocate planned graph cotangent buffer");
    struct UploadContext {
        VernonRhiDevice device{};
        VernonRhiBuffer buffer{};
        std::vector<std::pair<uint64_t, std::string>> uploads;
    };
    auto context = std::make_shared<UploadContext>();
    context->device = buffer->host->device;
    context->buffer = buffer->handle;
    for (nb::handle item : nb::cast<nb::list>(residency[2])) {
        nb::tuple upload = nb::cast<nb::tuple>(item);
        if (upload.size() != 2)
            throw std::runtime_error("planned graph cotangent upload range is invalid");
        const uint64_t offset = nb::cast<uint64_t>(upload[0]);
        nb::bytes raw = nb::cast<nb::bytes>(upload[1]);
        if (offset > buffer->size || raw.size() > buffer->size - offset)
            throw std::runtime_error("planned graph cotangent upload range exceeds its buffer");
        context->uploads.emplace_back(offset, std::string(raw.c_str(), raw.size()));
    }
    const auto encode = [](void *opaque, VernonRhiCommandEncoder encoder) {
        auto &state = *static_cast<UploadContext *>(opaque);
        for (const auto &[offset, bytes] : state.uploads) {
            const VernonRhiStatus status = vernonRhiCommandEncoderUploadBuffer(state.device, encoder, state.buffer,
                                                                               offset, bytes.data(), bytes.size());
            if (status != VERNON_RHI_STATUS_OK)
                return status;
        }
        return VERNON_RHI_STATUS_OK;
    };
    if (!context->uploads.empty()) {
        vernon::execution::detail::RhiCommandExecutionPlan plan;
        vernon::execution::detail::CommandNode upload;
        upload.kind = vernon::execution::detail::CommandNodeKind::Transfer;
        upload.queue = vernon::execution::detail::CommandQueueClass::Transfer;
        for (const auto &[offset, bytes] : context->uploads)
            upload.accesses.push_back(vernon::execution::detail::rhiBufferAccess(
                buffer->handle, offset, bytes.size(), vernon::execution::AccessMode::Write,
                VERNON_RHI_STATE_TRANSFER_DESTINATION));
        plan.commands.nodes.push_back(std::move(upload));
        plan.encoders.push_back({encode, context.get()});
        plan.retainedContexts.push_back(std::move(context));
        vernon::execution::detail::appendRhiBufferBinding(plan.bindings, buffer->handle);
        if (sink.append(std::move(plan)) != VERNON_RHI_STATUS_OK) {
            transaction.attr("_rollback_planned_state")();
            throw std::runtime_error("cannot append planned graph cotangent upload");
        }
    }
    try {
        retainPythonCommandCompletion(sink, transaction);
    } catch (...) {
        transaction.attr("_rollback_planned_state")();
        throw;
    }
    return result;
}

uint64_t metadataBytes(const PythonAdMetadata &metadata) {
    uint64_t scalarBytes = 0;
    switch (metadata.dtype) {
    case VERNON_DATA_BOOL:
    case VERNON_DATA_U8:
        scalarBytes = 1;
        break;
    case VERNON_DATA_F16:
        scalarBytes = 2;
        break;
    case VERNON_DATA_I32:
    case VERNON_DATA_U32:
    case VERNON_DATA_F32:
        scalarBytes = 4;
        break;
    case VERNON_DATA_F64:
        scalarBytes = 8;
        break;
    }
    for (uint64_t extent : metadata.shape) {
        if (extent && scalarBytes > UINT64_MAX / extent)
            return UINT64_MAX;
        scalarBytes *= extent;
    }
    return scalarBytes;
}

uint64_t pythonValueAllocationBytes(const nb::object &value) {
    nb::object storage = nb::hasattr(value, "_array") ? value.attr("_array") : value;
    Py_buffer view{};
    if (PyObject_GetBuffer(storage.ptr(), &view, PyBUF_SIMPLE) != 0) {
        PyErr_Clear();
        return 0;
    }
    const uint64_t bytes = view.len < 0 ? 0 : static_cast<uint64_t>(view.len);
    PyBuffer_Release(&view);
    return bytes;
}

struct PythonCheckpointResource final : vernon::execution::GraphCheckpointResource {
    explicit PythonCheckpointResource(nb::object value)
        : value(std::move(value)), byteAlignment(nb::hasattr(this->value, "_element_alignment")
                                                     ? nb::cast<uint64_t>(this->value.attr("_element_alignment"))
                                                     : 1) {}

    uint64_t byteSize() const override {
        Py_buffer view{};
        if (PyObject_GetBuffer(value.ptr(), &view, PyBUF_CONTIG_RO) != 0)
            throw nb::python_error();
        const uint64_t result = static_cast<uint64_t>(view.len);
        PyBuffer_Release(&view);
        return result;
    }
    uint64_t alignment() const override { return byteAlignment; }

    bool copyTo(void *destination, uint64_t byteSize, std::string &error) const override {
        Py_buffer view{};
        if (PyObject_GetBuffer(value.ptr(), &view, PyBUF_CONTIG_RO) != 0) {
            error = "checkpoint resource does not expose contiguous bytes";
            PyErr_Clear();
            return false;
        }
        const bool valid = static_cast<uint64_t>(view.len) == byteSize;
        if (valid && byteSize)
            std::memcpy(destination, view.buf, byteSize);
        PyBuffer_Release(&view);
        if (!valid)
            error = "checkpoint resource byte size changed after graph compilation";
        return valid;
    }

    bool copyFrom(const void *source, uint64_t byteSize, std::string &error) override {
        Py_buffer view{};
        if (PyObject_GetBuffer(value.ptr(), &view, PyBUF_CONTIG | PyBUF_WRITABLE) != 0) {
            error = "checkpoint resource does not expose writable contiguous bytes";
            PyErr_Clear();
            return false;
        }
        const bool valid = static_cast<uint64_t>(view.len) == byteSize;
        if (valid && byteSize)
            std::memcpy(view.buf, source, byteSize);
        PyBuffer_Release(&view);
        if (!valid)
            error = "checkpoint resource byte size changed after graph compilation";
        return valid;
    }

    bool copyRangeTo(uint64_t offset, void *destination, uint64_t rangeByteSize, std::string &error) const override {
        Py_buffer view{};
        if (PyObject_GetBuffer(value.ptr(), &view, PyBUF_CONTIG_RO) != 0) {
            error = "checkpoint resource does not expose contiguous bytes";
            PyErr_Clear();
            return false;
        }
        const uint64_t size = view.len < 0 ? 0 : static_cast<uint64_t>(view.len);
        const bool valid = offset <= size && rangeByteSize <= size - offset;
        if (valid && rangeByteSize)
            std::memcpy(destination, static_cast<const std::byte *>(view.buf) + offset, rangeByteSize);
        PyBuffer_Release(&view);
        if (!valid)
            error = "checkpoint write footprint exceeds the bound resource";
        return valid;
    }

    bool copyRangeFrom(uint64_t offset, const void *source, uint64_t rangeByteSize, std::string &error) override {
        Py_buffer view{};
        if (PyObject_GetBuffer(value.ptr(), &view, PyBUF_CONTIG | PyBUF_WRITABLE) != 0) {
            error = "checkpoint resource does not expose writable contiguous bytes";
            PyErr_Clear();
            return false;
        }
        const uint64_t size = view.len < 0 ? 0 : static_cast<uint64_t>(view.len);
        const bool valid = offset <= size && rangeByteSize <= size - offset;
        if (valid && rangeByteSize)
            std::memcpy(static_cast<std::byte *>(view.buf) + offset, source, rangeByteSize);
        PyBuffer_Release(&view);
        if (!valid)
            error = "checkpoint write footprint exceeds the bound resource";
        return valid;
    }

    bool copyRangesTo(const std::vector<vernon::execution::GraphByteRange> &ranges, void *packedDestination,
                      std::string &error) const override {
        Py_buffer view{};
        if (PyObject_GetBuffer(value.ptr(), &view, PyBUF_CONTIG_RO) != 0) {
            error = "checkpoint resource does not expose contiguous bytes";
            PyErr_Clear();
            return false;
        }
        const uint64_t size = view.len < 0 ? 0 : static_cast<uint64_t>(view.len);
        auto *destination = static_cast<std::byte *>(packedDestination);
        bool valid = true;
        for (const vernon::execution::GraphByteRange &range : ranges) {
            if (range.offset > size || range.byteSize > size - range.offset) {
                valid = false;
                break;
            }
            if (range.byteSize)
                std::memcpy(destination, static_cast<const std::byte *>(view.buf) + range.offset, range.byteSize);
            destination += range.byteSize;
        }
        PyBuffer_Release(&view);
        if (!valid)
            error = "checkpoint write footprint exceeds the bound resource";
        return valid;
    }

    bool copyRangesFrom(const std::vector<vernon::execution::GraphByteRange> &ranges, const void *packedSource,
                        std::string &error) override {
        Py_buffer view{};
        if (PyObject_GetBuffer(value.ptr(), &view, PyBUF_CONTIG | PyBUF_WRITABLE) != 0) {
            error = "checkpoint resource does not expose writable contiguous bytes";
            PyErr_Clear();
            return false;
        }
        const uint64_t size = view.len < 0 ? 0 : static_cast<uint64_t>(view.len);
        const auto *source = static_cast<const std::byte *>(packedSource);
        bool valid = true;
        for (const vernon::execution::GraphByteRange &range : ranges) {
            if (range.offset > size || range.byteSize > size - range.offset) {
                valid = false;
                break;
            }
            if (range.byteSize)
                std::memcpy(static_cast<std::byte *>(view.buf) + range.offset, source, range.byteSize);
            source += range.byteSize;
        }
        PyBuffer_Release(&view);
        if (!valid)
            error = "checkpoint write footprint exceeds the bound resource";
        return valid;
    }

    nb::object value;
    uint64_t byteAlignment;
};

struct PythonGraphAutodiffValue final : vernon::execution::GraphAutodiffValue {
    struct AddExpression {
        std::shared_ptr<PythonGraphAutodiffValue> left;
        std::shared_ptr<PythonGraphAutodiffValue> right;
    };

    PythonGraphAutodiffValue(nb::object value, std::shared_ptr<PythonGraphCallbackState> callbackState)
        : value(std::move(value)), allocationBytesValue(pythonValueAllocationBytes(this->value)),
          valueBytesValue(allocationBytesValue), callbackState(std::move(callbackState)) {}

    PythonGraphAutodiffValue(nb::object value, std::shared_ptr<PythonGraphCallbackState> callbackState,
                             uint64_t allocationBytes)
        : value(std::move(value)), allocationBytesValue(allocationBytes), valueBytesValue(allocationBytes),
          callbackState(std::move(callbackState)) {}

    PythonGraphAutodiffValue(std::shared_ptr<AddExpression> expression,
                             std::shared_ptr<PythonGraphCallbackState> callbackState, uint64_t allocationBytes,
                             uint64_t valueBytes)
        : expression(std::move(expression)), allocationBytesValue(allocationBytes), valueBytesValue(valueBytes),
          callbackState(std::move(callbackState)) {}

    uintptr_t logicalIdentity() const override {
        return expression ? reinterpret_cast<uintptr_t>(expression.get()) : reinterpret_cast<uintptr_t>(value.ptr());
    }
    uint64_t allocationBytes() const override { return allocationBytesValue; }

    bool materialize(vernon::execution::detail::RhiCommandPlanSink *sink, std::string &error) override {
        try {
            materialize(sink);
            return true;
        } catch (const std::exception &exception) {
            error = exception.what();
        } catch (...) {
            error = "planned graph autodiff value materialization failed";
        }
        return false;
    }

    std::shared_ptr<PythonGraphAutodiffValue> clone() const {
        return expression ? std::make_shared<PythonGraphAutodiffValue>(expression, callbackState, allocationBytesValue,
                                                                       valueBytesValue)
                          : std::make_shared<PythonGraphAutodiffValue>(value, callbackState, allocationBytesValue);
    }

    nb::object materialize(vernon::execution::detail::RhiCommandPlanSink *sink = nullptr) const {
        if (!expression) {
            if (sink)
                value = plannedDeviceValue(std::move(value), *sink);
            return value;
        }
        nb::list nodes;
        appendExpression(nodes, sink);
        nb::object materialize =
            nb::module_::import_("vernon_dsl._runtime.operators.gradient_expression").attr("materialize");
        value = sink ? materialize(nodes, nb::cast(sink, nb::rv_policy::reference)) : materialize(nodes);
        expression.reset();
        allocationBytesValue = valueBytesValue;
        return value;
    }

    size_t appendExpression(nb::list &nodes, vernon::execution::detail::RhiCommandPlanSink *sink) const {
        struct Frame {
            const PythonGraphAutodiffValue *value;
            bool expanded;
        };
        std::vector<Frame> frames{{this, false}};
        std::vector<size_t> indices;
        while (!frames.empty()) {
            const Frame frame = frames.back();
            frames.pop_back();
            if (!frame.value->expression) {
                if (sink)
                    frame.value->value = plannedDeviceValue(std::move(frame.value->value), *sink);
                nodes.append(nb::make_tuple("leaf", frame.value->value));
                indices.push_back(nodes.size() - 1);
                continue;
            }
            if (!frame.expanded) {
                frames.push_back({frame.value, true});
                frames.push_back({frame.value->expression->right.get(), false});
                frames.push_back({frame.value->expression->left.get(), false});
                continue;
            }
            const size_t right = indices.back();
            indices.pop_back();
            const size_t left = indices.back();
            indices.pop_back();
            nodes.append(nb::make_tuple("add", left, right));
            indices.push_back(nodes.size() - 1);
        }
        return indices.back();
    }

    std::shared_ptr<vernon::execution::GraphAutodiffValue> add(const vernon::execution::GraphAutodiffValue &other,
                                                               std::string &error) const override {
        const auto *python = dynamic_cast<const PythonGraphAutodiffValue *>(&other);
        if (!python) {
            error = "graph cotangent contributions have incompatible native value types";
            return {};
        }
        try {
            if (callbackState)
                callbackState->recordReverseCallback();
            auto node = std::make_shared<AddExpression>();
            node->left = clone();
            node->right = python->clone();
            const uint64_t resultBytes = std::max(valueBytesValue, python->valueBytesValue);
            if (python->allocationBytesValue > std::numeric_limits<uint64_t>::max() - allocationBytesValue ||
                resultBytes >
                    std::numeric_limits<uint64_t>::max() - allocationBytesValue - python->allocationBytesValue) {
                error = "graph cotangent accumulation size overflows";
                return {};
            }
            return std::make_shared<PythonGraphAutodiffValue>(
                std::move(node), callbackState, allocationBytesValue + python->allocationBytesValue + resultBytes,
                resultBytes);
        } catch (...) {
            if (callbackState)
                callbackState->captureException(std::current_exception());
            error = "graph cotangent accumulation failed";
            return {};
        }
    }

    mutable nb::object value;
    mutable std::shared_ptr<AddExpression> expression;
    mutable uint64_t allocationBytesValue;
    uint64_t valueBytesValue;
    std::shared_ptr<PythonGraphCallbackState> callbackState;
};

struct PythonPassPullback final : vernon::execution::PassPullback {
    PythonPassPullback(nb::object value, std::shared_ptr<PythonGraphCallbackState> callbackState)
        : value(std::move(value)), native(nb::cast<PythonPullback *>(this->value.attr("native"))),
          gradientGroups(this->value.attr("gradient_groups")), cotangentGroups(this->value.attr("cotangent_groups")),
          carrierShape(nb::cast<nb::tuple>(this->value.attr("carrier_shape"))),
          estimatedTapeBytesValue(nb::cast<uint64_t>(this->value.attr("estimated_tape_bytes"))),
          activeOperationCountValue(nb::cast<uint64_t>(this->value.attr("active_operation_count"))),
          recomputationCostValue(nb::cast<uint64_t>(this->value.attr("recomputation_cost"))),
          residualSourceKindValue(nb::cast<std::string>(this->value.attr("residual_source_kind"))),
          controlHistoryKindValue(nb::cast<std::string>(this->value.attr("control_history_kind"))),
          callbackState(std::move(callbackState)) {}

    bool apply(const vernon::execution::NamedGraphAutodiffValues &cotangents,
               vernon::execution::NamedGraphAutodiffValues &gradients,
               const vernon::execution::PassPullbackApplyOptions &options,
               vernon::execution::detail::RhiCommandPlanSink *sink, std::string &error) override {
        const VernonPullbackApplyOptions runtimeOptions{sizeof(VernonPullbackApplyOptions),
                                                        VERNON_PULLBACK_APPLY_OPTIONS_VERSION,
                                                        options.maximumTemporaryBytes,
                                                        options.maximumReusableConstructionBytes,
                                                        {}};
        return applyImpl(cotangents, gradients, &runtimeOptions, error, sink);
    }

    bool applyImpl(const vernon::execution::NamedGraphAutodiffValues &cotangents,
                   vernon::execution::NamedGraphAutodiffValues &gradients, const VernonPullbackApplyOptions *options,
                   std::string &error, vernon::execution::detail::RhiCommandPlanSink *sink = nullptr) {
        try {
            nb::dict values;
            for (const auto &cotangent : cotangents) {
                auto python = std::dynamic_pointer_cast<PythonGraphAutodiffValue>(cotangent.second);
                if (!python)
                    throw std::runtime_error("graph autodiff value has an incompatible native type");
                values[cotangent.first.c_str()] = python->materialize(sink);
            }
            if (callbackState)
                callbackState->recordReverseCallback();
            nb::dict result;
            const bool appliedOnDevice =
                sink ? native->applyGroupedDevicePlanned(values, gradientGroups, cotangentGroups, carrierShape, true,
                                                         options, *sink, result)
                     : native->applyGroupedDeviceWithOptions(values, gradientGroups, cotangentGroups, carrierShape,
                                                             true, options, result);
            if (!appliedOnDevice && sink)
                throw std::runtime_error("planned graph pullback requires device-resident Runtime AD lowering");
            if (!appliedOnDevice)
                result = native->applyGroupedWithOptions(values, gradientGroups, cotangentGroups, carrierShape, true,
                                                         options);
            else if (sink)
                sink->retain(std::make_shared<nb::object>(value));
            const auto &metadataValues = native->gradientMetadata();
            for (auto item : result) {
                const std::string path = nb::cast<std::string>(item.first);
                const auto metadata =
                    std::find_if(metadataValues.begin(), metadataValues.end(),
                                 [&](const PythonAdMetadata &candidate) { return candidate.path == path; });
                nb::object gradient = nb::borrow<nb::object>(item.second);
                gradients.emplace_back(path, std::make_shared<PythonGraphAutodiffValue>(
                                                 gradient, callbackState,
                                                 metadata == metadataValues.end() ? pythonValueAllocationBytes(gradient)
                                                                                  : metadataBytes(*metadata)));
            }
            return true;
        } catch (...) {
            if (callbackState)
                callbackState->captureException(std::current_exception());
            error = "pipeline pullback application failed";
            return false;
        }
    }

    uint64_t estimatedTapeBytes() const override { return estimatedTapeBytesValue; }
    uint64_t logicalResidualBytes() const override { return native->logicalResidualBytes(); }
    uint64_t residentTapeBytes() const override { return native->residentBytes(); }
    uint64_t allocatedTapeBytes() const override { return native->allocatedBytes(); }
    uint64_t retainedAllocationBytes() const override { return native->retainedAllocationBytes(); }
    uint64_t peakTemporaryTapeBytes() const override { return native->peakTemporaryBytes(); }
    uint64_t submissionCount() const override { return native->submissionCount(); }
    uint64_t waitCount() const override { return native->waitCount(); }
    uint64_t readbackCount() const override { return native->readbackCount(); }
    uint64_t atomicPublicationCount() const override { return native->atomicPublicationCount(); }
    uint64_t temporaryAllocationTrafficBytes() const override { return native->temporaryAllocationTrafficBytes(); }
    uint64_t deviceWaitNanoseconds() const override { return native->deviceWaitNanoseconds(); }
    uint64_t activeOperationCount() const override { return activeOperationCountValue; }
    uint64_t recomputationCost() const override { return recomputationCostValue; }
    uint64_t tapeContextLimitBytes() const override { return native->tapeContextLimitBytes(); }
    std::string residualSourceKind() const override { return residualSourceKindValue; }
    std::string controlHistoryKind() const override { return controlHistoryKindValue; }

    nb::object value;
    PythonPullback *native;
    nb::object gradientGroups;
    nb::object cotangentGroups;
    nb::tuple carrierShape;
    uint64_t estimatedTapeBytesValue;
    uint64_t activeOperationCountValue;
    uint64_t recomputationCostValue;
    std::string residualSourceKindValue;
    std::string controlHistoryKindValue;
    std::shared_ptr<PythonGraphCallbackState> callbackState;
};

} // namespace

std::shared_ptr<vernon::execution::GraphCheckpointResource> makePythonCheckpointResource(nb::object value) {
    return std::make_shared<PythonCheckpointResource>(std::move(value));
}

std::shared_ptr<vernon::execution::GraphAutodiffValue>
makePythonGraphAutodiffValue(nb::object value, std::shared_ptr<PythonGraphCallbackState> callbackState) {
    return std::make_shared<PythonGraphAutodiffValue>(std::move(value), std::move(callbackState));
}

nb::object pythonGraphAutodiffValue(const std::shared_ptr<vernon::execution::GraphAutodiffValue> &value) {
    auto python = std::dynamic_pointer_cast<PythonGraphAutodiffValue>(value);
    if (!python)
        throw std::runtime_error("graph autodiff value has an incompatible native type");
    return python->materialize();
}

std::unique_ptr<vernon::execution::PassPullback>
makePythonPassPullback(nb::object value, std::shared_ptr<PythonGraphCallbackState> callbackState) {
    return std::make_unique<PythonPassPullback>(std::move(value), std::move(callbackState));
}

PythonGraphBackwardSubmission::PythonGraphBackwardSubmission(
    std::shared_ptr<vernon::execution::GraphBackwardSubmission> value, std::exception_ptr retainedException)
    : submission(std::move(value)), exception(std::move(retainedException)) {}

void PythonGraphBackwardSubmission::wait() {
    std::string error;
    if (!submission->wait(error)) {
        if (exception)
            std::rethrow_exception(exception);
        throw std::runtime_error(error);
    }
}

uint32_t PythonGraphBackwardSubmission::state() const { return static_cast<uint32_t>(submission->state()); }

nb::dict PythonGraphBackwardSubmission::gradients() const {
    nb::dict result;
    for (const auto &gradient : submission->gradients())
        result[gradient.first.c_str()] = pythonGraphAutodiffValue(gradient.second);
    return result;
}

PythonGraphPullback::PythonGraphPullback(std::shared_ptr<vernon::execution::GraphPullback> value,
                                         std::shared_ptr<PythonGraphCallbackState> retainedCallbackState,
                                         std::vector<std::string> retainedObjectiveNames,
                                         std::shared_ptr<std::vector<nb::object>> retainedOwners)
    : pullback(std::move(value)), callbackState(std::move(retainedCallbackState)),
      objectiveNames(std::move(retainedObjectiveNames)), owners(std::move(retainedOwners)) {}

std::unique_ptr<PythonGraphBackwardSubmission> PythonGraphPullback::submit(const nb::object &cotangent) {
    vernon::execution::NamedGraphAutodiffValues values;
    const bool implicit = cotangent.is_none();
    if (!implicit) {
        if (nb::isinstance<nb::dict>(cotangent)) {
            nb::dict supplied = nb::cast<nb::dict>(cotangent);
            for (auto item : supplied)
                values.emplace_back(nb::cast<std::string>(item.first),
                                    makePythonGraphAutodiffValue(nb::borrow<nb::object>(item.second), callbackState));
        } else if (objectiveNames.size() == 1) {
            values.emplace_back(objectiveNames.front(),
                                makePythonGraphAutodiffValue(nb::borrow<nb::object>(cotangent), callbackState));
        } else {
            throw std::invalid_argument("graph pullback requires exactly one cotangent per objective");
        }
    }
    callbackState->beginOperation(true);
    std::shared_ptr<vernon::execution::GraphBackwardSubmission> submission;
    try {
        submission = pullback->submit(values, implicit);
    } catch (...) {
        (void)callbackState->endOperation();
        throw;
    }
    PythonGraphCallbackState::OperationFrame frame = callbackState->endOperation();
    lastReversePythonCallbackCount = frame.callbacks;
    auto retainedException = std::move(frame.exception);
    return std::make_unique<PythonGraphBackwardSubmission>(std::move(submission), std::move(retainedException));
}

uint32_t PythonGraphPullback::forwardState() const {
    return static_cast<uint32_t>(pullback->forwardSubmission().state());
}

void PythonGraphPullback::waitForward() {
    if (pullback->forwardSubmission().wait() != VERNON_RHI_STATUS_OK)
        throw std::runtime_error("graph VJP forward submission failed");
}

uint64_t PythonGraphPullback::reversePythonCallbackCount() const { return lastReversePythonCallbackCount; }

void bindExecutionGraphAutodiff(nb::module_ &module) {
    nb::class_<PythonGraphPullback>(module, "_GraphPullback")
        .def("submit", &PythonGraphPullback::submit, nb::arg("cotangent") = nb::none())
        .def("wait_forward", &PythonGraphPullback::waitForward)
        .def_prop_ro("forward_state", &PythonGraphPullback::forwardState)
        .def_prop_ro("estimated_tape_bytes",
                     [](const PythonGraphPullback &value) { return value.pullback->estimatedTapeBytes(); })
        .def_prop_ro("logical_residual_bytes",
                     [](const PythonGraphPullback &value) { return value.pullback->logicalResidualBytes(); })
        .def_prop_ro("resident_tape_bytes",
                     [](const PythonGraphPullback &value) { return value.pullback->residentTapeBytes(); })
        .def_prop_ro("allocated_tape_bytes",
                     [](const PythonGraphPullback &value) { return value.pullback->allocatedTapeBytes(); })
        .def_prop_ro("retained_allocation_bytes",
                     [](const PythonGraphPullback &value) { return value.pullback->retainedAllocationBytes(); })
        .def_prop_ro("checkpoint_bytes",
                     [](const PythonGraphPullback &value) { return value.pullback->checkpointBytes(); })
        .def_prop_ro("peak_runtime_managed_bytes",
                     [](const PythonGraphPullback &value) { return value.pullback->peakRuntimeManagedBytes(); })
        .def_prop_ro("submission_count",
                     [](const PythonGraphPullback &value) { return value.pullback->submissionCount(); })
        .def_prop_ro("wait_count", [](const PythonGraphPullback &value) { return value.pullback->waitCount(); })
        .def_prop_ro("readback_count", [](const PythonGraphPullback &value) { return value.pullback->readbackCount(); })
        .def_prop_ro("atomic_publication_count",
                     [](const PythonGraphPullback &value) { return value.pullback->atomicPublicationCount(); })
        .def_prop_ro("temporary_allocation_traffic_bytes",
                     [](const PythonGraphPullback &value) { return value.pullback->temporaryAllocationTrafficBytes(); })
        .def_prop_ro("device_wait_nanoseconds",
                     [](const PythonGraphPullback &value) { return value.pullback->deviceWaitNanoseconds(); })
        .def_prop_ro("tape_context_limit_bytes",
                     [](const PythonGraphPullback &value) { return value.pullback->tapeContextLimitBytes(); })
        .def_prop_ro("recomputation_factor",
                     [](const PythonGraphPullback &value) { return value.pullback->recomputationFactor(); })
        .def_prop_ro("pass_telemetry",
                     [](const PythonGraphPullback &value) {
                         nb::list result;
                         for (const vernon::execution::GraphAutodiffPassTelemetry &item :
                              value.pullback->passTelemetry()) {
                             nb::dict telemetry;
                             telemetry["schedule_offset"] = item.scheduleOffset;
                             telemetry["pass_name"] = item.passName;
                             telemetry["residual_source_kind"] = item.residualSourceKind;
                             telemetry["control_history_kind"] = item.controlHistoryKind;
                             telemetry["estimated_tape_bytes"] = item.estimatedTapeBytes;
                             telemetry["logical_residual_bytes"] = item.logicalResidualBytes;
                             telemetry["resident_tape_bytes"] = item.residentTapeBytes;
                             telemetry["allocated_tape_bytes"] = item.allocatedTapeBytes;
                             telemetry["retained_allocation_bytes"] = item.retainedAllocationBytes;
                             telemetry["peak_temporary_tape_bytes"] = item.peakTemporaryTapeBytes;
                             telemetry["checkpoint_bytes"] = item.checkpointBytes;
                             telemetry["active_operation_count"] = item.activeOperationCount;
                             telemetry["recomputation_cost"] = item.recomputationCost;
                             if (item.controlHistoryBytes)
                                 telemetry["control_history_bytes"] = *item.controlHistoryBytes;
                             else
                                 telemetry["control_history_bytes"] = nb::none();
                             result.append(std::move(telemetry));
                         }
                         return result;
                     })
        .def_prop_ro("reverse_python_callback_count", &PythonGraphPullback::reversePythonCallbackCount);
    nb::class_<PythonGraphBackwardSubmission>(module, "_GraphBackwardSubmission")
        .def("wait", &PythonGraphBackwardSubmission::wait)
        .def_prop_ro("state", &PythonGraphBackwardSubmission::state)
        .def_prop_ro("gradients", &PythonGraphBackwardSubmission::gradients);
}
