#ifndef VERNON_PYTHON_NATIVE_PROGRAM_AUTODIFF_H
#define VERNON_PYTHON_NATIVE_PROGRAM_AUTODIFF_H

#include "native_command_retention.h"
#include "native_program.h"
#include "runtime/autodiff/runtime_autodiff_telemetry.h"
#include "runtime/autodiff/runtime_forward_plan.h"
#include "runtime/program_boundary_view.h"

#include <deque>
#include <limits>
#include <unordered_map>

struct PythonAdViewDescriptor {
    uintptr_t allocationBegin{};
    size_t allocationSize{};
    size_t byteOffset{};
    VernonDataType dtype{};
    bool writable{};
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;

    VernonTensorView tensorView() const {
        VernonTensorView tensor{};
        tensor.struct_size = sizeof(VernonTensorView);
        tensor.storage = VERNON_TENSOR_HOST;
        tensor.host_data = reinterpret_cast<const void *>(allocationBegin);
        tensor.element_layout = vernonRuntimeGetScalarValueLayout(dtype);
        tensor.access = writable ? VERNON_ACCESS_READ_WRITE : VERNON_ACCESS_READ;
        tensor.rank = static_cast<uint32_t>(shape.size());
        tensor.shape = shape.data();
        tensor.byte_strides = strides.data();
        tensor.byte_offset = byteOffset;
        tensor.byte_size = allocationSize;
        return tensor;
    }
};

PythonAdViewDescriptor validatePythonAdOriginalView(const std::string &path, VernonDataType dtype,
                                                    const std::vector<uint64_t> &expectedShape, const nb::object &array,
                                                    bool writable);

inline nb::object pythonAdArraySource(const nb::object &source) {
    return nb::hasattr(source, "_native_host_array") ? source.attr("_native_host_array")() : source;
}

struct PythonAdValue {
    std::string path;
    nb::object source;
    nb::object array;
    std::vector<uint64_t> shape;
    bool writable{};
    PythonAdViewDescriptor originalView;
    VernonAdValue value{};

    PythonAdValue(std::string path, VernonDataType dtype, std::vector<uint64_t> shape, const nb::object &source,
                  uint32_t access = VERNON_ACCESS_READ)
        : path(std::move(path)), source(nb::module_::import_("numpy").attr("asarray")(pythonAdArraySource(source))),
          array(this->source), writable(access != VERNON_ACCESS_READ) {
        originalView = validatePythonAdOriginalView(this->path, dtype, shape, this->source, writable);
        this->shape = originalView.shape;
        if (!nb::cast<bool>(array.attr("flags").attr("c_contiguous"))) {
            nb::object numpy = nb::module_::import_("numpy");
            array = access == VERNON_ACCESS_WRITE ? numpy.attr("empty_like")(this->source)
                                                  : numpy.attr("ascontiguousarray")(array);
        }
        const size_t itemSize = nb::cast<size_t>(array.attr("dtype").attr("itemsize"));
        size_t bytes = itemSize;
        for (uint64_t extent : this->shape) {
            if (extent && bytes > std::numeric_limits<size_t>::max() / static_cast<size_t>(extent))
                throw std::invalid_argument("Python autodiff Value byte size overflows");
            bytes *= static_cast<size_t>(extent);
        }
        if (!itemSize)
            throw std::invalid_argument("Python autodiff Value byte size overflows");
        if (this->shape.size() > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("Python autodiff Value rank overflows");
        value.struct_size = sizeof(value);
        value.path = {this->path.data(), this->path.size()};
        value.dtype = dtype;
        value.data = reinterpret_cast<void *>(nb::cast<uintptr_t>(array.attr("ctypes").attr("data")));
        value.size = bytes;
        value.rank = static_cast<uint32_t>(this->shape.size());
        value.shape = this->shape.empty() ? nullptr : this->shape.data();
    }

    PythonAdValue(PythonAdValue &&other) noexcept
        : path(std::move(other.path)), source(std::move(other.source)), array(std::move(other.array)),
          shape(std::move(other.shape)), writable(other.writable), originalView(std::move(other.originalView)),
          value(other.value) {
        value.path = {path.data(), path.size()};
        value.shape = shape.empty() ? nullptr : shape.data();
    }
    void commit() {
        if (writable && source.ptr() != array.ptr())
            nb::module_::import_("numpy").attr("copyto")(source, array);
    }
    PythonAdValue(const PythonAdValue &) = delete;
    PythonAdValue &operator=(const PythonAdValue &) = delete;
};

struct PythonAdMetadata {
    std::string path;
    std::string binding;
    VernonDataType dtype{};
    std::vector<uint64_t> shape;
};

inline size_t pythonAdMetadataBytes(const PythonAdMetadata &metadata) {
    size_t bytes = autodiffDtypeSize(metadata.dtype);
    for (uint64_t extent : metadata.shape) {
        if (extent > std::numeric_limits<size_t>::max() ||
            (extent && bytes > std::numeric_limits<size_t>::max() / static_cast<size_t>(extent)))
            throw std::length_error("Python autodiff metadata byte size overflows");
        bytes *= static_cast<size_t>(extent);
    }
    return bytes;
}

inline nb::object pythonAdGradientBuffer(const PythonAdMetadata &gradient, const nb::object &binding) {
    nb::object numpy = nb::module_::import_("numpy");
    nb::object expectedDtype = numpy.attr("dtype")(numpyDtypeName(gradient.dtype));
    if (!binding.is_none() && nb::hasattr(binding, "_native_host_array")) {
        nb::object host = binding.attr("_native_host_array")();
        const std::vector<uint64_t> hostShape = nb::cast<std::vector<uint64_t>>(host.attr("shape"));
        if (hostShape.size() == gradient.shape.size() &&
            nb::cast<bool>(host.attr("dtype").attr("__eq__")(expectedDtype))) {
            bool compatible = true;
            for (size_t dimension = 0; dimension < hostShape.size(); ++dimension) {
                if (gradient.shape[dimension] && gradient.shape[dimension] != hostShape[dimension]) {
                    compatible = false;
                    break;
                }
            }
            if (compatible)
                return numpy.attr("zeros_like")(host);
        }
    }
    return numpy.attr("zeros")(gradient.shape, numpy.attr(numpyDtypeName(gradient.dtype)));
}

inline void instantiatePythonAdMetadata(PythonAdMetadata &leaf, const std::deque<PythonAdValue> &bound) {
    for (const PythonAdValue &value : bound) {
        if (value.path == leaf.path) {
            leaf.shape = value.shape;
            return;
        }
    }
}

PythonAdMetadata adInputLeafMetadata(VernonProgramExecutable *pipeline, const ProgramParameterMetadata &parameter,
                                     size_t leafIndex, VernonProgramValueLeafView *reflected = nullptr);

nb::object resolveProgramInputLeaf(const nb::dict &inputs, const std::string &leafPath);

struct PythonPullback {
    PythonPullback(VernonRuntimeContext *runtime, VernonPullback *handle, std::vector<PythonAdMetadata> gradients,
                   std::vector<PythonAdMetadata> cotangents, bool hasCarrierDimensions, nb::object pipelineOwner,
                   nb::dict bindings)
        : runtime(runtime), handle(handle), gradients(std::move(gradients)), cotangents(std::move(cotangents)),
          hasCarrierDimensions(hasCarrierDimensions), pipelineOwner(std::move(pipelineOwner)),
          bindings(std::move(bindings)) {}
    ~PythonPullback() { vernonPullbackDestroy(handle); }

    nb::dict apply(const nb::object &cotangent) { return applyImpl(cotangent, false); }
    nb::dict applyLogical(const nb::object &cotangent) { return applyImpl(cotangent, true); }
    nb::dict applyGrouped(const nb::object &cotangent, const nb::object &gradientGroups,
                          const nb::object &cotangentGroups, const nb::object &carrierShape, bool logical) {
        return applyGroupedWithOptions(cotangent, gradientGroups, cotangentGroups, carrierShape, logical, nullptr);
    }
    nb::dict applyGroupedWithOptions(const nb::object &cotangent, const nb::object &gradientGroups,
                                     const nb::object &cotangentGroups, const nb::object &carrierShape, bool logical,
                                     const VernonPullbackApplyOptions *options);
    bool applyGroupedDeviceWithOptions(const nb::object &cotangent, const nb::object &gradientGroups,
                                       const nb::object &cotangentGroups, const nb::object &carrierShape, bool logical,
                                       const VernonPullbackApplyOptions *options, nb::dict &result,
                                       vernon::execution::detail::RhiCommandPlanSink *sink = nullptr);
    bool applyGroupedDevicePlanned(const nb::object &cotangent, const nb::object &gradientGroups,
                                   const nb::object &cotangentGroups, const nb::object &carrierShape, bool logical,
                                   const VernonPullbackApplyOptions *options,
                                   vernon::execution::detail::RhiCommandPlanSink &sink, nb::dict &result) {
        return applyGroupedDeviceWithOptions(cotangent, gradientGroups, cotangentGroups, carrierShape, logical, options,
                                             result, &sink);
    }
    size_t logicalResidualBytes() const { return memoryUsage().logicalResidualBytes; }
    size_t residentBytes() const { return memoryUsage().residentBytes; }
    size_t allocatedBytes() const { return memoryUsage().allocatedBytes; }
    size_t retainedAllocationBytes() const { return memoryUsage().retainedAllocationBytes; }
    size_t peakTemporaryBytes() const { return memoryUsage().peakTemporaryBytes; }
    uint64_t submissionCount() const { return controlPlaneUsage().submissions; }
    uint64_t waitCount() const { return controlPlaneUsage().waits; }
    uint64_t readbackCount() const { return controlPlaneUsage().readbacks; }
    uint64_t atomicPublicationCount() const { return controlPlaneUsage().atomicPublications; }
    uint64_t temporaryAllocationTrafficBytes() const { return controlPlaneUsage().temporaryAllocationBytes; }
    uint64_t deviceWaitNanoseconds() const { return controlPlaneUsage().deviceWaitNanoseconds; }
    size_t tapeContextLimitBytes() const { return vernon::runtime::autodiffHostTapeContextLimit(runtime); }
    uint64_t peakRuntimeManagedBytes() const {
        return vernon::runtime::autodiffPullbackPeakRuntimeManagedBytes(handle);
    }
    nb::object checkpointPlan() const {
        const vernon::runtime::AutodiffPullbackCheckpointPlan plan =
            vernon::runtime::autodiffPullbackCheckpointPlan(handle);
        if (!plan.present)
            return nb::none();
        nb::dict result;
        result["peak_bytes"] = plan.peakBytes;
        result["memory_budget"] = plan.memoryBudget;
        result["logical_residual_bytes"] = plan.logicalResidualBytes;
        result["retained_allocation_bytes"] = plan.retainedAllocationBytes;
        result["initial_state_bytes"] = plan.initialStateBytes;
        result["restoration_bytes"] = plan.restorationBytes;
        result["transaction_bytes"] = plan.transactionBytes;
        result["persistent_checkpoint_bytes"] = plan.persistentCheckpointBytes;
        result["backward_value_bytes"] = plan.backwardValueBytes;
        result["replay_cost"] = plan.replayCost;
        result["recomputation_cost"] = plan.recomputationCost;
        result["selected_policy"] = plan.selectedPolicy;
        return result;
    }
    nb::list passTelemetry() const {
        nb::list result;
        for (const vernon::runtime::AutodiffPullbackPassTelemetry &item :
             vernon::runtime::autodiffPullbackPassTelemetry(handle)) {
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
            result.append(std::move(telemetry));
        }
        return result;
    }
    const std::vector<PythonAdMetadata> &gradientMetadata() const { return gradients; }

private:
    vernon::runtime::AutodiffPullbackMemoryUsage memoryUsage() const {
        return vernon::runtime::autodiffPullbackMemoryUsage(handle);
    }
    vernon::runtime::AutodiffPullbackControlPlaneUsage controlPlaneUsage() const {
        return vernon::runtime::autodiffPullbackControlPlaneUsage(handle);
    }

    bool applyDeviceImpl(const nb::object &cotangent, bool logicalCotangent, const VernonPullbackApplyOptions *options,
                         nb::dict &result, vernon::execution::detail::RhiCommandPlanSink *sink = nullptr) {
        std::vector<nb::object> retainedBuffers;
        std::vector<nb::object> retainedViews;
        std::vector<std::vector<uint64_t>> retainedShapes;
        std::vector<std::vector<int64_t>> retainedStrides;
        retainedShapes.reserve(cotangents.size());
        retainedStrides.reserve(cotangents.size() + gradients.size());
        std::vector<VernonAdDeviceValue> seeds;
        seeds.reserve(cotangents.size());
        VernonAdDeviceValueSet seedSet{};
        const VernonAdDeviceValueSet *seedView = nullptr;
        if (!cotangent.is_none()) {
            nb::dict supplied;
            if (cotangents.size() == 1 && !nb::isinstance<nb::dict>(cotangent))
                supplied[nb::str(cotangents.front().path.c_str())] = cotangent;
            else if (nb::isinstance<nb::dict>(cotangent))
                supplied = nb::cast<nb::dict>(cotangent);
            else
                return false;
            if (supplied.size() != cotangents.size())
                return false;
            for (const PythonAdMetadata &sourceMetadata : cotangents) {
                PythonAdMetadata metadata = sourceMetadata;
                if (logicalCotangent && hasCarrierDimensions) {
                    if (metadata.shape.size() < 3)
                        throw std::runtime_error("autodiff cotangent has no invocation carrier dimensions");
                    metadata.shape.erase(metadata.shape.begin(), metadata.shape.begin() + 3);
                }
                nb::str path(sourceMetadata.path.c_str());
                if (!supplied.contains(path))
                    return false;
                nb::object value = nb::borrow<nb::object>(supplied[path]);
                if (!nb::hasattr(value, "_resident_buffer") || !nb::hasattr(value, "layout") ||
                    nb::cast<std::vector<uint64_t>>(value.attr("shape")) != metadata.shape)
                    return false;
                nb::object bufferObject = sink ? value.attr("_planned_buffer")() : value.attr("_resident_buffer")();
                auto *buffer = nb::cast<RhiBuffer *>(bufferObject);
                nb::object layout = value.attr("layout");
                std::vector<int64_t> strides = nb::cast<std::vector<int64_t>>(layout.attr("byte_strides"));
                const size_t offset = nb::cast<size_t>(layout.attr("byte_offset"));
                const size_t bytes = pythonAdMetadataBytes(metadata);
                if (!buffer || strides.size() != metadata.shape.size() || offset > buffer->size)
                    return false;
                retainedBuffers.push_back(std::move(bufferObject));
                retainedViews.push_back(std::move(value));
                retainedStrides.push_back(std::move(strides));
                retainedShapes.push_back(std::move(metadata.shape));
                seeds.push_back({sizeof(VernonAdDeviceValue),
                                 {sourceMetadata.path.data(), sourceMetadata.path.size()},
                                 metadata.dtype,
                                 buffer->handle,
                                 offset,
                                 buffer->size,
                                 bytes,
                                 static_cast<uint32_t>(retainedShapes.back().size()),
                                 retainedShapes.back().data(),
                                 retainedStrides.back().data(),
                                 {}});
            }
            seedSet = {sizeof(VernonAdDeviceValueSet), seeds.data(), seeds.size(), {}};
            seedView = &seedSet;
        }

        std::unordered_map<PyObject *, nb::object> gradientsByOwner;
        std::unordered_map<PyObject *, nb::object> plannedWritesByOwner;
        std::vector<nb::object> plannedWrites;
        struct PlannedWriteRollback {
            std::vector<nb::object> &transactions;
            bool released{};
            ~PlannedWriteRollback() {
                if (released)
                    return;
                for (nb::object &transaction : transactions)
                    try {
                        transaction.attr("_rollback_planned_state")();
                    } catch (...) {
                    }
            }
        } plannedWriteRollback{plannedWrites};
        std::vector<nb::object> materializedGradients;
        std::vector<VernonAdDeviceValue> gradientViews;
        materializedGradients.reserve(gradients.size());
        gradientViews.reserve(gradients.size());
        for (const PythonAdMetadata &gradient : gradients) {
            nb::str bindingPath(gradient.binding.c_str());
            nb::object binding =
                bindings.contains(bindingPath) ? nb::borrow<nb::object>(bindings[bindingPath]) : nb::none();
            if (binding.is_none() || !nb::hasattr(binding, "_materialize_gradient") ||
                !nb::hasattr(binding, "_gradient_device_view"))
                return false;
            nb::object owner = nb::hasattr(binding, "owner") ? binding.attr("owner") : binding;
            auto existing = gradientsByOwner.find(owner.ptr());
            nb::object zeros = pythonAdGradientBuffer(gradient, binding);
            nb::object materialized =
                existing == gradientsByOwner.end()
                    ? binding.attr("_materialize_gradient")(zeros, gradient.path)
                    : binding.attr("_materialize_gradient")(zeros, gradient.path, existing->second);
            if (existing == gradientsByOwner.end())
                gradientsByOwner.emplace(owner.ptr(), materialized);
            nb::object deviceView = binding.attr("_gradient_device_view")(gradient.path, materialized, gradient.shape);
            if (sink && plannedWritesByOwner.find(owner.ptr()) == plannedWritesByOwner.end()) {
                nb::object transaction = materialized.attr("_begin_planned_device_write")();
                plannedWritesByOwner.emplace(owner.ptr(), transaction);
                plannedWrites.push_back(std::move(transaction));
            }
            nb::object bufferObject =
                sink ? deviceView.attr("_planned_buffer")() : deviceView.attr("_resident_buffer")();
            auto *buffer = nb::cast<RhiBuffer *>(bufferObject);
            nb::object layout = deviceView.attr("layout");
            std::vector<int64_t> strides = nb::cast<std::vector<int64_t>>(layout.attr("byte_strides"));
            const size_t offset = nb::cast<size_t>(layout.attr("byte_offset"));
            const size_t bytes = pythonAdMetadataBytes(gradient);
            if (!buffer || strides.size() != gradient.shape.size() || offset > buffer->size)
                return false;
            retainedBuffers.push_back(std::move(bufferObject));
            retainedViews.push_back(std::move(deviceView));
            materializedGradients.push_back(materialized);
            retainedStrides.push_back(std::move(strides));
            gradientViews.push_back({sizeof(VernonAdDeviceValue),
                                     {gradient.path.data(), gradient.path.size()},
                                     gradient.dtype,
                                     buffer->handle,
                                     offset,
                                     buffer->size,
                                     bytes,
                                     static_cast<uint32_t>(gradient.shape.size()),
                                     gradient.shape.data(),
                                     retainedStrides.back().data(),
                                     {}});
        }
        VernonAdDeviceValueSet gradientSet{
            sizeof(VernonAdDeviceValueSet), gradientViews.data(), gradientViews.size(), {}};
        if (!sink)
            throw std::invalid_argument("device pullback requires canonical command-plan execution");
        const VernonStatus status =
            vernon::runtime::ad::applyPullbackDeviceWithPlanSink(*handle, seedView, gradientSet, options, *sink);
        if (status != VERNON_STATUS_OK) {
            throw std::runtime_error("device pullback application failed: " +
                                     nativeStringView(vernonRuntimeGetLastError(runtime)));
        }
        if (sink) {
            for (const nb::object &buffer : retainedBuffers)
                retainPythonObject(*sink, buffer);
            for (const nb::object &view : retainedViews)
                retainPythonObject(*sink, view);
            for (nb::object &transaction : plannedWrites)
                retainPythonCommandCompletion(*sink, transaction);
            plannedWriteRollback.released = true;
        }
        for (size_t index = 0; index < gradients.size(); ++index) {
            if (!sink)
                materializedGradients[index].attr("_mark_device_dirty")();
            result[nb::str(gradients[index].path.c_str())] = materializedGradients[index];
        }
        return true;
    }

    nb::dict applyImpl(const nb::object &cotangent, bool logicalCotangent,
                       const VernonPullbackApplyOptions *options = nullptr) {
        std::deque<PythonAdValue> gradientValues;
        std::vector<VernonAdValue> gradientViews;
        nb::dict result;
        for (const PythonAdMetadata &gradient : gradients) {
            nb::str bindingPath(gradient.binding.c_str());
            nb::object binding =
                bindings.contains(bindingPath) ? nb::borrow<nb::object>(bindings[bindingPath]) : nb::none();
            nb::object zeros = pythonAdGradientBuffer(gradient, binding);
            gradientValues.emplace_back(gradient.path, gradient.dtype, gradient.shape, zeros);
            gradientViews.push_back(gradientValues.back().value);
        }
        VernonAdValueSet gradientSet{sizeof(VernonAdValueSet), gradientViews.data(), gradientViews.size(), {}};

        std::deque<PythonAdValue> seeds;
        std::vector<VernonAdValue> seedViews;
        VernonAdValueSet seedSet{};
        const VernonAdValueSet *seedView = nullptr;
        const auto seedMetadata = [&](const PythonAdMetadata &metadata) {
            PythonAdMetadata logical = metadata;
            if (logicalCotangent && hasCarrierDimensions) {
                if (logical.shape.size() < 3)
                    throw std::runtime_error("autodiff cotangent has no invocation carrier dimensions");
                logical.shape.erase(logical.shape.begin(), logical.shape.begin() + 3);
            }
            return logical;
        };
        nb::object suppliedCotangent = cotangent;
        if (suppliedCotangent.is_none() && cotangents.size() == 1) {
            const PythonAdMetadata metadata = seedMetadata(cotangents.front());
            suppliedCotangent = nb::module_::import_("numpy").attr("ones")(
                metadata.shape, nb::module_::import_("numpy").attr("dtype")(numpyDtypeName(metadata.dtype)));
        }
        if (!suppliedCotangent.is_none()) {
            if (cotangents.size() == 1) {
                const PythonAdMetadata metadata = seedMetadata(cotangents.front());
                seeds.emplace_back(metadata.path, metadata.dtype, metadata.shape, suppliedCotangent);
            } else {
                if (!nb::isinstance<nb::dict>(suppliedCotangent))
                    throw std::invalid_argument("aggregate pullback cotangent must be a leaf-path dictionary");
                nb::dict values = nb::cast<nb::dict>(suppliedCotangent);
                if (values.size() != cotangents.size())
                    throw std::invalid_argument("aggregate pullback requires every output cotangent leaf");
                for (const PythonAdMetadata &carriedMetadata : cotangents) {
                    const PythonAdMetadata metadata = seedMetadata(carriedMetadata);
                    nb::str path(metadata.path.c_str());
                    if (!values.contains(path))
                        throw std::invalid_argument("missing output cotangent leaf '" + metadata.path + "'");
                    seeds.emplace_back(metadata.path, metadata.dtype, metadata.shape,
                                       nb::borrow<nb::object>(values[path]));
                }
            }
            for (PythonAdValue &seed : seeds)
                seedViews.push_back(seed.value);
            seedSet = {sizeof(VernonAdValueSet), seedViews.data(), seedViews.size(), {}};
            seedView = &seedSet;
        }
        const VernonStatus status = options ? vernonPullbackApplyWithOptions(handle, seedView, &gradientSet, options)
                                            : vernonPullbackApply(handle, seedView, &gradientSet);
        if (status != VERNON_STATUS_OK)
            throw std::runtime_error("pullback application failed: " +
                                     nativeStringView(vernonRuntimeGetLastError(runtime)));
        std::unordered_map<PyObject *, nb::object> gradientsByOwner;
        for (size_t index = 0; index < gradientValues.size(); ++index) {
            PythonAdValue &gradient = gradientValues[index];
            const PythonAdMetadata &metadata = gradients[index];
            nb::str path(gradient.path.c_str());
            nb::str bindingPath(metadata.binding.c_str());
            nb::object binding =
                bindings.contains(bindingPath) ? nb::borrow<nb::object>(bindings[bindingPath]) : nb::none();
            if (binding.is_none() || !nb::hasattr(binding, "_materialize_gradient")) {
                result[path] = gradient.array;
                continue;
            }
            nb::object owner = nb::hasattr(binding, "owner") ? binding.attr("owner") : binding;
            auto existing = gradientsByOwner.find(owner.ptr());
            nb::object materialized =
                existing == gradientsByOwner.end()
                    ? binding.attr("_materialize_gradient")(gradient.array, path)
                    : binding.attr("_materialize_gradient")(gradient.array, path, existing->second);
            if (existing == gradientsByOwner.end())
                gradientsByOwner.emplace(owner.ptr(), materialized);
            result[path] = std::move(materialized);
        }
        return result;
    }

    VernonRuntimeContext *runtime{};
    VernonPullback *handle{};
    std::vector<PythonAdMetadata> gradients;
    std::vector<PythonAdMetadata> cotangents;
    bool hasCarrierDimensions{};
    nb::object pipelineOwner;
    nb::dict bindings;
};

struct PythonProgramExecutable {
    PythonProgramExecutable(Runtime *owner, VernonRuntimeContext *runtime, VernonProgramBundle *bundle,
                            VernonProgramExecutable *pipeline, std::vector<SharedCompileResult> retainedResults = {},
                            std::vector<std::pair<std::string, VernonCpuEntryPoint>> registeredCpuEntries = {})
        : owner(owner), runtime(runtime), bundle(bundle), pipeline(pipeline),
          retainedResults(std::move(retainedResults)), registeredCpuEntries(std::move(registeredCpuEntries)) {}
    ~PythonProgramExecutable() {
        vernonRuntimeProgramExecutableDestroy(pipeline);
        for (const auto &[symbol, entry] : registeredCpuEntries)
            vernonRuntimeUnregisterCpuEntry(runtime, {symbol.data(), symbol.size()}, entry);
        vernonRuntimeProgramBundleDestroy(bundle);
    }

    std::unique_ptr<ProgramInvocationBuilder> invocationBuilder() {
        return std::make_unique<ProgramInvocationBuilder>(owner, runtime, pipeline);
    }

    std::array<uint32_t, 3> workgroupSize() const { return {0, 0, 0}; }

    nb::dict programAdSignature() const {
        if (!vernonRuntimeProgramExecutableHasProgramAutodiff(pipeline))
            throw std::runtime_error("pipeline has no Program autodiff signature");
        nb::dict signature;
        const std::pair<const char *, VernonProgramAdBoundary> boundaries[] = {
            {"inputs", VERNON_PROGRAM_AD_INPUT},         {"outputs", VERNON_PROGRAM_AD_OUTPUT},
            {"cotangents", VERNON_PROGRAM_AD_COTANGENT}, {"gradients", VERNON_PROGRAM_AD_GRADIENT},
            {"captures", VERNON_PROGRAM_AD_CAPTURE},
        };
        for (const auto &[name, boundary] : boundaries) {
            nb::list rows;
            const size_t count = vernonRuntimeProgramExecutableGetProgramAdValueCount(pipeline, boundary);
            for (size_t index = 0; index < count; ++index) {
                VernonProgramAdValueView value{};
                value.struct_size = sizeof(value);
                if (vernonRuntimeProgramExecutableGetProgramAdValueByIndex(pipeline, boundary, index, &value) !=
                    VERNON_STATUS_OK)
                    throw std::runtime_error("cannot read Program autodiff signature");
                nb::dict row;
                row["path"] = nativeStringView(value.path);
                row["value_id"] = value.value_id;
                row["external"] = value.external != 0;
                row["output"] = value.output != 0;
                rows.append(std::move(row));
            }
            signature[name] = std::move(rows);
        }
        return signature;
    }

    nb::dict programAbi() const {
        nb::list slots;
        for (const vernon::runtime::ProgramBoundaryView &slot : vernon::runtime::programBoundaryViews(*pipeline)) {
            nb::dict row;
            row["slot"] = slot.slot;
            row["path"] = slot.path;
            row["value"] = slot.value;
            row["role"] = slot.role;
            row["category"] = slot.category;
            slots.append(std::move(row));
        }
        nb::dict result;
        result["boundary_slots"] = std::move(slots);
        return result;
    }

    void programForwardBound(ProgramInvocationBuilder &builder) {
        if (builder.pipeline != pipeline)
            throw std::invalid_argument("Program invocation builder belongs to another pipeline");
        VernonPullback *pullback = nullptr;
        const VernonStatus status = builder.forwardProgram(&pullback);
        if (pullback)
            vernonPullbackDestroy(pullback);
        if (status != VERNON_STATUS_OK)
            throw std::runtime_error("Program forward failed: " + nativeStringView(vernonRuntimeGetLastError(runtime)));
    }

    nb::tuple programVjpBound(ProgramInvocationBuilder &builder, const nb::dict &programBindings,
                              nb::object pipelineOwner, const nb::object &checkpointMemoryBudget,
                              const std::string &checkpointPolicy) {
        if (!vernonRuntimeProgramExecutableHasProgramAutodiff(pipeline))
            throw std::runtime_error("pipeline has no Program autodiff signature");
        if (checkpointMemoryBudget.is_none())
            vernon::runtime::autodiffSetProgramCheckpointPlan(pipeline, nullptr, checkpointPolicy);
        else {
            const uint64_t budget = nb::cast<uint64_t>(checkpointMemoryBudget);
            vernon::runtime::autodiffSetProgramCheckpointPlan(pipeline, &budget, checkpointPolicy);
        }
        const auto leafMetadata = [&](const char *kind, size_t index) {
            VernonAdValueMetadataView value{};
            value.struct_size = sizeof(value);
            VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT;
            if (std::string_view(kind) == "input")
                status = vernonRuntimeProgramExecutableGetAdInputByIndex(pipeline, index, &value);
            else if (std::string_view(kind) == "output")
                status = vernonRuntimeProgramExecutableGetAdOutputByIndex(pipeline, index, &value);
            else if (std::string_view(kind) == "cotangent")
                status = vernonRuntimeProgramExecutableGetAdCotangentByIndex(pipeline, index, &value);
            else if (std::string_view(kind) == "gradient")
                status = vernonRuntimeProgramExecutableGetAdGradientByIndex(pipeline, index, &value);
            if (status != VERNON_STATUS_OK)
                throw std::runtime_error("cannot read Program autodiff leaf metadata");
            PythonAdMetadata result;
            result.path = nativeStringView(value.path);
            result.binding = result.path;
            result.dtype = value.dtype;
            if (value.rank)
                result.shape.assign(value.shape, value.shape + value.rank);
            return result;
        };
        const auto declaredDerivativePath = [&](VernonAdDerivativeRole role, const std::string &leafPath) {
            const size_t count = vernonRuntimeProgramExecutableGetAdDerivativeGroupCount(pipeline);
            for (size_t groupIndex = 0; groupIndex < count; ++groupIndex) {
                VernonAdDerivativeGroupView group{};
                group.struct_size = sizeof(group);
                if (vernonRuntimeProgramExecutableGetAdDerivativeGroupByIndex(pipeline, groupIndex, &group) !=
                        VERNON_STATUS_OK ||
                    group.role != role)
                    continue;
                for (size_t leafIndex = 0; leafIndex < group.leaf_count; ++leafIndex) {
                    VernonStringView leaf{};
                    if (vernonRuntimeProgramExecutableGetAdDerivativeGroupLeaf(pipeline, groupIndex, leafIndex,
                                                                               &leaf) == VERNON_STATUS_OK &&
                        nativeStringView(leaf) == leafPath)
                        return nativeStringView(group.declared_path);
                }
            }
            throw std::runtime_error("Program derivative leaf has no declared group");
        };
        nb::dict outputs;
        if (builder.pipeline != pipeline)
            throw std::invalid_argument("Program invocation builder belongs to another pipeline");
        VernonPullback *pullback = nullptr;
        const VernonStatus status = builder.forwardProgram(&pullback);
        if (status != VERNON_STATUS_OK) {
            const std::string error = nativeStringView(vernonRuntimeGetLastError(runtime));
            if (status == VERNON_STATUS_INVALID_ARGUMENT || error.find("memory budget") != std::string::npos)
                throw std::invalid_argument(error);
            throw std::runtime_error("Program autodiff forward failed: " + error);
        }
        std::unique_ptr<VernonPullback, decltype(&vernonPullbackDestroy)> pullbackOwner(pullback,
                                                                                        &vernonPullbackDestroy);

        std::vector<PythonAdMetadata> cotangents;
        const auto instantiateBoundMetadata = [&](PythonAdMetadata &leaf) {
            nb::object value = resolveProgramInputLeaf(programBindings, leaf.path);
            if (nb::hasattr(value, "_native_host_array"))
                value = value.attr("_native_host_array")();
            nb::object array = nb::module_::import_("numpy").attr("asarray")(value);
            leaf.shape = nb::cast<std::vector<uint64_t>>(array.attr("shape"));
        };
        const size_t cotangentCount = vernonRuntimeProgramExecutableGetAdCotangentCount(pipeline);
        for (size_t index = 0; index < cotangentCount; ++index) {
            PythonAdMetadata leaf = leafMetadata("cotangent", index);
            leaf.binding = declaredDerivativePath(VERNON_AD_DERIVATIVE_COTANGENT, leaf.path);
            if (const size_t separator = leaf.binding.find('.'); separator != std::string::npos)
                leaf.binding.resize(separator);
            instantiateBoundMetadata(leaf);
            cotangents.push_back(std::move(leaf));
        }
        std::vector<PythonAdMetadata> gradients;
        const size_t gradientCount = vernonRuntimeProgramExecutableGetAdGradientCount(pipeline);
        for (size_t index = 0; index < gradientCount; ++index) {
            PythonAdMetadata leaf = leafMetadata("gradient", index);
            leaf.binding = declaredDerivativePath(VERNON_AD_DERIVATIVE_GRADIENT, leaf.path);
            if (const size_t separator = leaf.binding.find('.'); separator != std::string::npos)
                leaf.binding.resize(separator);
            instantiateBoundMetadata(leaf);
            gradients.push_back(std::move(leaf));
        }
        return nb::make_tuple(outputs,
                              std::make_unique<PythonPullback>(runtime, pullbackOwner.release(), std::move(gradients),
                                                               std::move(cotangents), false, std::move(pipelineOwner),
                                                               nb::dict(programBindings)));
    }

    nb::list writeFootprints() const {
        nb::list result;
        return result;
    }

    nb::list readFootprints() const {
        nb::list result;
        return result;
    }

    nb::list derivativeGroups() const {
        nb::list result;
        const size_t groupCount = vernonRuntimeProgramExecutableGetAdDerivativeGroupCount(pipeline);
        for (size_t groupIndex = 0; groupIndex < groupCount; ++groupIndex) {
            VernonAdDerivativeGroupView group{};
            group.struct_size = sizeof(group);
            if (vernonRuntimeProgramExecutableGetAdDerivativeGroupByIndex(pipeline, groupIndex, &group) !=
                VERNON_STATUS_OK)
                throw std::runtime_error("cannot read autodiff derivative group metadata");
            nb::list leaves;
            for (size_t leafIndex = 0; leafIndex < group.leaf_count; ++leafIndex) {
                VernonStringView leafPath{};
                if (vernonRuntimeProgramExecutableGetAdDerivativeGroupLeaf(pipeline, groupIndex, leafIndex,
                                                                           &leafPath) != VERNON_STATUS_OK)
                    throw std::runtime_error("cannot read autodiff derivative group leaf");
                leaves.append(nativeStringView(leafPath));
            }
            result.append(nb::make_tuple(group.role == VERNON_AD_DERIVATIVE_GRADIENT ? "gradient" : "cotangent",
                                         nativeStringView(group.declared_path), leaves));
        }
        return result;
    }

    std::vector<ProgramParameterMetadata> parameters() const {
        std::vector<ProgramParameterMetadata> result;
        const size_t count = vernonRuntimeProgramExecutableGetParameterCount(pipeline);
        result.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            VernonProgramParameterView view{};
            if (vernonRuntimeProgramExecutableGetParameterByIndex(pipeline, index, &view) != VERNON_STATUS_OK)
                throw std::runtime_error("cannot read loaded pipeline parameter");
            ProgramParameterMetadata parameter = parameterMetadata(view);
            parameter.elementLeafPaths.reserve(parameter.elementLeaves.size());
            parameter.elementLeafShapes.reserve(parameter.elementLeaves.size());
            for (size_t leafIndex = 0; leafIndex < parameter.elementLeaves.size(); ++leafIndex) {
                VernonProgramValueLeafView leaf{};
                leaf.struct_size = sizeof(leaf);
                const VernonStringView name{parameter.name.data(), parameter.name.size()};
                if (vernonRuntimeProgramExecutableGetParameterValueLeaf(pipeline, name, leafIndex, &leaf) !=
                    VERNON_STATUS_OK)
                    throw std::runtime_error("cannot read loaded Program parameter leaf path");
                std::vector<ProgramParameterMetadata::PathComponent> path;
                path.reserve(leaf.path_count);
                for (size_t componentIndex = 0; componentIndex < leaf.path_count; ++componentIndex) {
                    const VernonValuePathComponentView &component = leaf.path[componentIndex];
                    path.push_back({component.kind == VERNON_VALUE_PATH_FIELD,
                                    component.kind == VERNON_VALUE_PATH_FIELD ? nativeStringView(component.field) : "",
                                    component.index});
                }
                parameter.elementLeafPaths.push_back(std::move(path));
                std::vector<uint64_t> shape;
                if (leaf.static_shape)
                    shape.assign(leaf.static_shape, leaf.static_shape + leaf.static_rank);
                parameter.elementLeafShapes.push_back(std::move(shape));
            }
            result.push_back(std::move(parameter));
        }
        return result;
    }

    std::vector<ProgramOutputMetadata> outputs() const {
        std::vector<ProgramOutputMetadata> result;
        const size_t count = vernonRuntimeProgramExecutableGetOutputCount(pipeline);
        result.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            VernonProgramOutputView view{};
            if (vernonRuntimeProgramExecutableGetOutputByIndex(pipeline, index, &view) != VERNON_STATUS_OK)
                throw std::runtime_error("cannot read loaded pipeline output");
            result.push_back(outputMetadata(view));
        }
        return result;
    }

    Runtime *owner{};
    VernonRuntimeContext *runtime{};
    VernonProgramBundle *bundle{};
    VernonProgramExecutable *pipeline{};
    // ORC entry pointers are valid only while an interned compile result owns the JIT.
    std::vector<std::shared_ptr<InternedCpuJit>> internedCpuJits;
    std::vector<SharedCompileResult> retainedResults;
    std::vector<std::pair<std::string, VernonCpuEntryPoint>> registeredCpuEntries;
};

#endif
