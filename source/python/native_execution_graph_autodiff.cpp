#include "native_execution_graph_autodiff.h"

#include "native_pipeline_autodiff.h"

#include <nanobind/stl/unique_ptr.h>

#include <cstring>
#include <stdexcept>
#include <utility>

namespace {

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

    nb::object value;
    uint64_t byteAlignment;
};

struct PythonGraphAutodiffValue final : vernon::execution::GraphAutodiffValue {
    PythonGraphAutodiffValue(nb::object value, std::shared_ptr<PythonGraphCallbackState> callbackState)
        : value(std::move(value)), callbackState(std::move(callbackState)) {}

    uintptr_t logicalIdentity() const override { return reinterpret_cast<uintptr_t>(value.ptr()); }

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
            nb::object numpy = nb::module_::import_("numpy");
            nb::object tensorStorageType = nb::module_::import_("vernon_dsl._runtime.resources").attr("TensorStorage");
            const bool leftStorage = nb::isinstance(value, tensorStorageType);
            const bool rightStorage = nb::isinstance(python->value, tensorStorageType);
            nb::object result;
            if (leftStorage || rightStorage)
                result = tensorStorageType.attr("_add_gradients")(leftStorage ? value : python->value,
                                                                  leftStorage ? python->value : value);
            else {
                nb::object sum = numpy.attr("asarray")(value).attr("__add__")(numpy.attr("asarray")(python->value));
                result = numpy.attr("ascontiguousarray")(sum);
            }
            return std::make_shared<PythonGraphAutodiffValue>(std::move(result), callbackState);
        } catch (...) {
            if (callbackState)
                callbackState->captureException(std::current_exception());
            error = "graph cotangent accumulation failed";
            return {};
        }
    }

    nb::object value;
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
               vernon::execution::NamedGraphAutodiffValues &gradients, std::string &error) override {
        try {
            nb::dict values;
            for (const auto &cotangent : cotangents)
                values[cotangent.first.c_str()] = pythonGraphAutodiffValue(cotangent.second);
            if (callbackState)
                callbackState->recordReverseCallback();
            nb::dict result = native->applyGrouped(values, gradientGroups, cotangentGroups, carrierShape, true);
            for (auto item : result)
                gradients.emplace_back(
                    nb::cast<std::string>(item.first),
                    makePythonGraphAutodiffValue(nb::borrow<nb::object>(item.second), callbackState));
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
    return python->value;
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
        .def_prop_ro("checkpoint_bytes",
                     [](const PythonGraphPullback &value) { return value.pullback->checkpointBytes(); })
        .def_prop_ro("peak_runtime_managed_bytes",
                     [](const PythonGraphPullback &value) { return value.pullback->peakRuntimeManagedBytes(); })
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
