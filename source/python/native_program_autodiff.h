#ifndef VERNON_PYTHON_NATIVE_PROGRAM_AUTODIFF_H
#define VERNON_PYTHON_NATIVE_PROGRAM_AUTODIFF_H

#include "native_command_retention.h"
#include "native_program.h"
#include "runtime/autodiff/runtime_autodiff_telemetry.h"
#include "runtime/program_boundary_view.h"

#include <deque>
#include <limits>
#include <unordered_map>

inline nb::object pythonAdArraySource(const nb::object &source) {
    return nb::hasattr(source, "_native_host_array") ? source.attr("_native_host_array")() : source;
}

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

PythonAdMetadata adInputLeafMetadata(VernonProgramExecutable *executable, const ProgramParameterMetadata &parameter,
                                     size_t leafIndex, VernonProgramValueLeafView *reflected = nullptr);

nb::object resolveProgramInputLeaf(const nb::dict &inputs, const std::string &leafPath);

struct PythonPullback {
    PythonPullback(Runtime *owner, VernonRuntimeContext *runtime, VernonProgramExecutable *executable,
                   VernonPullback *handle, std::vector<PythonAdMetadata> gradients,
                   std::vector<PythonAdMetadata> cotangents, bool hasCarrierDimensions, nb::object executableOwner,
                   nb::dict bindings)
        : owner(owner), runtime(runtime), executable(executable), handle(handle), gradients(std::move(gradients)),
          cotangents(std::move(cotangents)), hasCarrierDimensions(hasCarrierDimensions),
          executableOwner(std::move(executableOwner)), bindings(std::move(bindings)) {}
    ~PythonPullback() { vernonProgramPullbackDestroy(handle); }

    nb::dict apply(const nb::object &cotangent) { return applyImpl(cotangent, false); }
    nb::dict applyLogical(const nb::object &cotangent) { return applyImpl(cotangent, true); }
    nb::dict applyGrouped(const nb::object &cotangent, const nb::object &gradientGroups,
                          const nb::object &cotangentGroups, const nb::object &carrierShape, bool logical) {
        return applyGroupedWithOptions(cotangent, gradientGroups, cotangentGroups, carrierShape, logical, nullptr);
    }
    nb::dict applyGroupedWithOptions(const nb::object &cotangent, const nb::object &gradientGroups,
                                     const nb::object &cotangentGroups, const nb::object &carrierShape, bool logical,
                                     const VernonPullbackApplyOptions *options);
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

    nb::dict applyImpl(const nb::object &cotangent, bool logicalCotangent,
                       const VernonPullbackApplyOptions *options = nullptr) {
        (void)logicalCotangent;
        ProgramInvocationBuilder builder(owner, runtime, executable);
        nb::dict result;
        std::unordered_map<PyObject *, nb::object> gradientsByOwner;
        for (const PythonAdMetadata &gradient : gradients) {
            nb::str bindingPath(gradient.binding.c_str());
            nb::object binding =
                bindings.contains(bindingPath) ? nb::borrow<nb::object>(bindings[bindingPath]) : nb::none();
            nb::object zeros = pythonAdGradientBuffer(gradient, binding);
            nb::object materialized = zeros;
            if (!binding.is_none() && nb::hasattr(binding, "_materialize_gradient")) {
                nb::object resourceOwner = nb::hasattr(binding, "owner") ? binding.attr("owner") : binding;
                auto existing = gradientsByOwner.find(resourceOwner.ptr());
                materialized = existing == gradientsByOwner.end()
                                   ? binding.attr("_materialize_gradient")(zeros, gradient.path)
                                   : binding.attr("_materialize_gradient")(zeros, gradient.path, existing->second);
                if (existing == gradientsByOwner.end())
                    gradientsByOwner.emplace(resourceOwner.ptr(), materialized);
            }
            result[nb::str(gradient.path.c_str())] = std::move(materialized);
        }

        const auto boundaryMetadata = [&](VernonProgramBoundaryRole role) {
            std::vector<ProgramParameterMetadata> reflected;
            const size_t count = vernonRuntimeProgramExecutableGetBoundaryCount(executable, role);
            for (size_t index = 0; index < count; ++index) {
                VernonProgramParameterView view{};
                if (vernonRuntimeProgramExecutableGetBoundaryByIndex(executable, role, index, &view) !=
                    VERNON_STATUS_OK)
                    throw std::runtime_error("cannot reflect canonical derivative boundary");
                ProgramParameterMetadata parameter = parameterMetadata(view);
                for (size_t leafIndex = 0; leafIndex < parameter.elementLeaves.size(); ++leafIndex) {
                    VernonProgramValueLeafView leaf{};
                    leaf.struct_size = sizeof(leaf);
                    if (vernonRuntimeProgramExecutableGetBoundaryValueLeaf(executable, role, parameter.slot, leafIndex,
                                                                           &leaf) != VERNON_STATUS_OK)
                        throw std::runtime_error("cannot reflect canonical derivative boundary leaf");
                    std::vector<ProgramParameterMetadata::PathComponent> path;
                    for (size_t component = 0; component < leaf.path_count; ++component)
                        path.push_back({leaf.path[component].kind == VERNON_VALUE_PATH_FIELD,
                                        leaf.path[component].kind == VERNON_VALUE_PATH_FIELD
                                            ? nativeStringView(leaf.path[component].field)
                                            : "",
                                        leaf.path[component].index});
                    parameter.elementLeafPaths.push_back(std::move(path));
                    parameter.elementLeafShapes.emplace_back();
                    if (leaf.static_rank)
                        parameter.elementLeafShapes.back().assign(leaf.static_shape,
                                                                  leaf.static_shape + leaf.static_rank);
                }
                reflected.push_back(std::move(parameter));
            }
            return reflected;
        };
        const auto firstGroupLeaf = [&](VernonAdDerivativeRole role, const std::string &declaredPath) {
            const size_t count = vernonRuntimeProgramExecutableGetAdDerivativeGroupCount(executable);
            for (size_t groupIndex = 0; groupIndex < count; ++groupIndex) {
                VernonAdDerivativeGroupView group{};
                group.struct_size = sizeof(group);
                if (vernonRuntimeProgramExecutableGetAdDerivativeGroupByIndex(executable, groupIndex, &group) !=
                        VERNON_STATUS_OK ||
                    group.role != role || nativeStringView(group.declared_path) != declaredPath || !group.leaf_count)
                    continue;
                VernonStringView leaf{};
                if (vernonRuntimeProgramExecutableGetAdDerivativeGroupLeaf(executable, groupIndex, 0, &leaf) ==
                    VERNON_STATUS_OK)
                    return nativeStringView(leaf);
            }
            throw std::runtime_error("canonical derivative boundary has no derivative group");
        };
        struct PackedPublication {
            nb::object bytes;
            size_t elementByteSize{};
            std::vector<std::tuple<nb::object, size_t, size_t>> destinations;
        };
        std::vector<PackedPublication> packedPublications;
        const auto leafPath = [](const ProgramParameterMetadata &parameter, size_t leafIndex) {
            std::string leaf;
            for (const auto &component : parameter.elementLeafPaths[leafIndex]) {
                if (!leaf.empty())
                    leaf.push_back('.');
                leaf += component.field ? component.name : std::to_string(component.index);
            }
            if (leaf.empty() || parameter.name == leaf ||
                (parameter.name.size() > leaf.size() &&
                 parameter.name.compare(parameter.name.size() - leaf.size(), leaf.size(), leaf) == 0 &&
                 parameter.name[parameter.name.size() - leaf.size() - 1] == '.'))
                return parameter.name;
            return parameter.name + "." + leaf;
        };
        const auto packBoundary = [&](const ProgramParameterMetadata &parameter, const nb::dict &values, bool publish) {
            if (parameter.elementLeaves.size() != parameter.elementLeafPaths.size())
                throw std::runtime_error("canonical aggregate boundary reflection is incomplete");
            nb::object numpy = nb::module_::import_("numpy");
            std::vector<uint64_t> outerShape = parameter.shape;
            for (size_t leafIndex = 0; leafIndex < parameter.elementLeaves.size(); ++leafIndex) {
                const std::string firstPath = leafPath(parameter, leafIndex);
                nb::str firstKey(firstPath.c_str());
                if (!values.contains(firstKey))
                    continue;
                nb::object first = numpy.attr("asarray")(pythonAdArraySource(nb::borrow<nb::object>(values[firstKey])));
                const std::vector<uint64_t> shape = nb::cast<std::vector<uint64_t>>(first.attr("shape"));
                if (shape.size() < parameter.elementLeafShapes[leafIndex].size())
                    throw std::invalid_argument("aggregate derivative leaf rank is too small");
                outerShape.assign(shape.begin(),
                                  shape.end() - static_cast<ptrdiff_t>(parameter.elementLeafShapes[leafIndex].size()));
                break;
            }
            std::vector<uint64_t> packedShape = outerShape;
            packedShape.push_back(parameter.elementByteSize);
            nb::object packed = numpy.attr("zeros")(packedShape, numpy.attr("uint8"));
            nb::object packedBytes = packed.attr("reshape")(-1, parameter.elementByteSize);
            PackedPublication publication{packed, parameter.elementByteSize, {}};
            for (size_t leafIndex = 0; leafIndex < parameter.elementLeaves.size(); ++leafIndex) {
                const std::string path = leafPath(parameter, leafIndex);
                nb::str key(path.c_str());
                if (!values.contains(key))
                    continue;
                nb::object destination =
                    numpy.attr("asarray")(pythonAdArraySource(nb::borrow<nb::object>(values[key])));
                const size_t bytes =
                    autodiffDtypeSize(static_cast<VernonDataType>(parameter.elementLeaves[leafIndex].dtype)) *
                    parameter.elementLeaves[leafIndex].scalar_count;
                nb::object region = packedBytes.attr("__getitem__")(nb::make_tuple(
                    nb::slice(nb::none(), nb::none(), nb::none()),
                    nb::slice(static_cast<size_t>(parameter.elementLeaves[leafIndex].byte_offset),
                              static_cast<size_t>(parameter.elementLeaves[leafIndex].byte_offset) + bytes, size_t{1})));
                if (publish)
                    publication.destinations.emplace_back(destination, parameter.elementLeaves[leafIndex].byte_offset,
                                                          bytes);
                else {
                    nb::object leafBytes = numpy.attr("ascontiguousarray")(destination)
                                               .attr("reshape")(-1)
                                               .attr("view")(numpy.attr("uint8"))
                                               .attr("reshape")(-1, bytes);
                    numpy.attr("copyto")(region, leafBytes);
                }
            }
            if (publish)
                packedPublications.push_back(std::move(publication));
            return packed;
        };
        const auto addBoundary = [&](const ProgramParameterMetadata &parameter, const nb::object &source,
                                     VernonValueAccess access) {
            if (parameter.kind != VERNON_PROGRAM_TENSOR)
                throw std::invalid_argument("Program derivative boundary must be a Tensor");
            if (nb::hasattr(source, "_resident_buffer") && nb::hasattr(source, "layout") &&
                vernon::runtime::autodiffRhiDevice(runtime).index != VERNON_RHI_INVALID_HANDLE_INDEX) {
                nb::object bufferObject = source.attr("_resident_buffer")();
                auto *buffer = nb::cast<RhiBuffer *>(bufferObject);
                nb::object layout = source.attr("layout");
                builder.ownedArgument(builder.prepareRhiTensor(
                    nb::int_(parameter.slot), buffer, access, nb::cast<std::vector<uint64_t>>(source.attr("shape")),
                    nb::cast<std::vector<int64_t>>(layout.attr("byte_strides")),
                    nb::cast<size_t>(layout.attr("byte_offset"))));
            } else {
                nb::object array = nb::module_::import_("numpy").attr("asarray")(pythonAdArraySource(source));
                builder.ownedArgument(builder.prepareHostTensor(nb::int_(parameter.slot), array));
            }
        };
        const auto derivativeMetadata = [&](VernonAdDerivativeRole role,
                                            const ProgramParameterMetadata &parameter) -> const PythonAdMetadata & {
            const std::string leaf = firstGroupLeaf(role, parameter.name);
            const std::vector<PythonAdMetadata> &metadata =
                role == VERNON_AD_DERIVATIVE_COTANGENT ? cotangents : gradients;
            const auto found = std::find_if(metadata.begin(), metadata.end(),
                                            [&](const PythonAdMetadata &candidate) { return candidate.path == leaf; });
            if (found == metadata.end())
                throw std::runtime_error("canonical derivative boundary has no Python metadata");
            return *found;
        };
        const auto derivativeBinding = [&](const PythonAdMetadata &metadata) {
            nb::str path(metadata.binding.c_str());
            return bindings.contains(path) ? nb::borrow<nb::object>(bindings[path]) : nb::none();
        };
        std::vector<PyObject *> gradientSources;
        std::vector<PyObject *> deviceGradientBuffers;
        const auto injectiveLayout = [](const std::vector<uint64_t> &shape, const std::vector<int64_t> &strides,
                                        size_t elementBytes) {
            std::vector<std::pair<uint64_t, uint64_t>> axes;
            for (size_t index = 0; index < shape.size(); ++index)
                if (shape[index] > 1)
                    axes.emplace_back(static_cast<uint64_t>(std::abs(strides[index])), shape[index]);
            std::sort(axes.begin(), axes.end());
            uint64_t span = elementBytes;
            for (const auto &[stride, extent] : axes) {
                if (stride < span)
                    return false;
                if (extent - 1 > (std::numeric_limits<uint64_t>::max() - span) / stride)
                    return false;
                span += (extent - 1) * stride;
            }
            return true;
        };
        const auto addDerivativeBoundary = [&](const ProgramParameterMetadata &parameter, nb::object source,
                                               VernonValueAccess access, VernonAdDerivativeRole role) {
            const PythonAdMetadata &metadata = derivativeMetadata(role, parameter);
            if (!nb::hasattr(source, "_resident_buffer"))
                return addBoundary(parameter, source, access);
            nb::object binding = derivativeBinding(metadata);
            if (binding.is_none())
                return addBoundary(parameter, source, access);
            nb::object destination = source;
            bool sharedGradient = false;
            if (role == VERNON_AD_DERIVATIVE_GRADIENT) {
                sharedGradient =
                    std::find(gradientSources.begin(), gradientSources.end(), source.ptr()) != gradientSources.end();
                if (!sharedGradient)
                    gradientSources.push_back(source.ptr());
            }
            bool directDevice = vernon::runtime::autodiffRhiDevice(runtime).index != VERNON_RHI_INVALID_HANDLE_INDEX;
            nb::object bufferObject = nb::none();
            nb::tuple boundaryLayout;
            if (nb::hasattr(binding, "_gradient_boundary_layout") && directDevice) {
                boundaryLayout =
                    nb::cast<nb::tuple>(binding.attr("_gradient_boundary_layout")(source, parameter.elementByteSize));
                const std::vector<uint64_t> shape = nb::cast<std::vector<uint64_t>>(boundaryLayout[0]);
                const std::vector<int64_t> strides = nb::cast<std::vector<int64_t>>(boundaryLayout[1]);
                directDevice &= injectiveLayout(shape, strides, parameter.elementByteSize);
                bufferObject = source.attr("_resident_buffer")();
                if (role == VERNON_AD_DERIVATIVE_GRADIENT) {
                    if (std::find(deviceGradientBuffers.begin(), deviceGradientBuffers.end(), bufferObject.ptr()) !=
                        deviceGradientBuffers.end())
                        directDevice = false;
                    else if (directDevice)
                        deviceGradientBuffers.push_back(bufferObject.ptr());
                }
            }
            if (!sharedGradient && parameter.elementLeaves.size() == 1 &&
                parameter.elementLeaves.front().scalar_count == 1 &&
                (parameter.elementLeafShapes.empty() || parameter.elementLeafShapes.front().empty()) &&
                nb::hasattr(binding, "_gradient_device_view")) {
                source = binding.attr("_gradient_device_view")(metadata.path, source, metadata.shape);
                nb::object layout = source.attr("layout");
                const std::vector<uint64_t> shape = nb::cast<std::vector<uint64_t>>(source.attr("shape"));
                const std::vector<int64_t> strides = nb::cast<std::vector<int64_t>>(layout.attr("byte_strides"));
                const bool injective =
                    injectiveLayout(shape, strides, nb::cast<size_t>(source.attr("dtype").attr("itemsize")));
                if (injective)
                    return addBoundary(parameter, source, access);
                source = destination;
            }
            if (directDevice && !boundaryLayout.is_none()) {
                auto *buffer = nb::cast<RhiBuffer *>(bufferObject);
                builder.ownedArgument(builder.prepareRhiTensor(
                    nb::int_(parameter.slot), buffer, access, nb::cast<std::vector<uint64_t>>(boundaryLayout[0]),
                    nb::cast<std::vector<int64_t>>(boundaryLayout[1]), nb::cast<size_t>(boundaryLayout[2])));
                return;
            }
            nb::dict leaves;
            for (size_t leafIndex = 0; leafIndex < parameter.elementLeafPaths.size(); ++leafIndex) {
                const std::string path = leafPath(parameter, leafIndex);
                const auto found =
                    std::find_if((role == VERNON_AD_DERIVATIVE_COTANGENT ? cotangents : gradients).begin(),
                                 (role == VERNON_AD_DERIVATIVE_COTANGENT ? cotangents : gradients).end(),
                                 [&](const PythonAdMetadata &candidate) { return candidate.path == path; });
                if (found == (role == VERNON_AD_DERIVATIVE_COTANGENT ? cotangents : gradients).end())
                    continue;
                leaves[nb::str(path.c_str())] = binding.attr("_gradient_device_view")(path, source, found->shape);
            }
            addBoundary(parameter, packBoundary(parameter, leaves, access == VERNON_ACCESS_WRITE), access);
        };

        nb::dict supplied;
        const auto cotangentBoundaries = boundaryMetadata(VERNON_PROGRAM_BOUNDARY_COTANGENT);
        if (cotangent.is_none()) {
            if (cotangentBoundaries.size() != 1 || cotangents.size() != 1)
                throw std::invalid_argument("implicit Program cotangent requires exactly one boundary");
            const PythonAdMetadata &metadata = cotangents.front();
            supplied[nb::str(cotangentBoundaries.front().name.c_str())] = nb::module_::import_("numpy").attr("ones")(
                metadata.shape, nb::module_::import_("numpy").attr("dtype")(numpyDtypeName(metadata.dtype)));
        } else if (nb::isinstance<nb::dict>(cotangent)) {
            supplied = nb::cast<nb::dict>(cotangent);
        } else {
            if (cotangentBoundaries.size() != 1)
                throw std::invalid_argument("pullback requires one cotangent per canonical boundary");
            supplied[nb::str(cotangentBoundaries.front().name.c_str())] = cotangent;
        }
        for (const ProgramParameterMetadata &parameter : cotangentBoundaries) {
            nb::str path(parameter.name.c_str());
            if (!supplied.contains(path))
                throw std::invalid_argument("missing canonical cotangent boundary '" + parameter.name + "'");
            nb::object source = nb::borrow<nb::object>(supplied[path]);
            if (nb::isinstance<nb::dict>(source))
                source = packBoundary(parameter, nb::cast<nb::dict>(source), false);
            addDerivativeBoundary(parameter, source, VERNON_ACCESS_READ, VERNON_AD_DERIVATIVE_COTANGENT);
        }
        std::vector<nb::object> devicePublishedGradients;
        for (const ProgramParameterMetadata &parameter : boundaryMetadata(VERNON_PROGRAM_BOUNDARY_GRADIENT)) {
            const std::string leaf = firstGroupLeaf(VERNON_AD_DERIVATIVE_GRADIENT, parameter.name);
            nb::str path(leaf.c_str());
            if (!result.contains(path))
                throw std::runtime_error("missing canonical gradient publication owner");
            nb::object source = nb::borrow<nb::object>(result[path]);
            if (parameter.elementLeaves.size() > 1 && !nb::hasattr(source, "_native_host_array"))
                source = packBoundary(parameter, result, true);
            if (nb::hasattr(source, "_resident_buffer")) {
                devicePublishedGradients.push_back(nb::borrow<nb::object>(result[path]));
            }
            addDerivativeBoundary(parameter, source, VERNON_ACCESS_WRITE, VERNON_AD_DERIVATIVE_GRADIENT);
        }
        std::vector<VernonProgramArgument> arguments;
        builder.collectArguments(arguments);
        const VernonStatus status =
            options ? vernonProgramPullbackApplyWithOptions(handle, arguments.data(), arguments.size(), options)
                    : vernonProgramPullbackApply(handle, arguments.data(), arguments.size());
        if (status != VERNON_STATUS_OK)
            throw std::runtime_error("pullback application failed: " +
                                     nativeStringView(vernonRuntimeGetLastError(runtime)));
        nb::object numpy = nb::module_::import_("numpy");
        for (nb::object &gradient : devicePublishedGradients)
            if (nb::hasattr(gradient, "_mark_device_dirty"))
                gradient.attr("_mark_device_dirty")();
        for (PackedPublication &publication : packedPublications) {
            nb::object packedBytes = publication.bytes.attr("reshape")(-1, publication.elementByteSize);
            for (auto &[destination, offset, bytes] : publication.destinations) {
                nb::object region = packedBytes.attr("__getitem__")(nb::make_tuple(
                    nb::slice(nb::none(), nb::none(), nb::none()), nb::slice(offset, offset + bytes, size_t{1})));
                nb::object typed = region.attr("copy")()
                                       .attr("reshape")(-1)
                                       .attr("view")(destination.attr("dtype"))
                                       .attr("reshape")(destination.attr("shape"));
                numpy.attr("add")(destination, typed, nb::arg("out") = destination);
            }
        }
        return result;
    }

    Runtime *owner{};
    VernonRuntimeContext *runtime{};
    VernonProgramExecutable *executable{};
    VernonPullback *handle{};
    std::vector<PythonAdMetadata> gradients;
    std::vector<PythonAdMetadata> cotangents;
    bool hasCarrierDimensions{};
    nb::object executableOwner;
    nb::dict bindings;
};

struct PythonProgramExecutable {
    PythonProgramExecutable(Runtime *owner, VernonRuntimeContext *runtime, VernonProgramBundle *bundle,
                            VernonProgramExecutable *executable, std::vector<SharedCompileResult> retainedResults = {},
                            std::vector<std::pair<std::string, VernonCpuEntryPoint>> registeredCpuEntries = {})
        : owner(owner), runtime(runtime), bundle(bundle), executable(executable),
          retainedResults(std::move(retainedResults)), registeredCpuEntries(std::move(registeredCpuEntries)) {}
    ~PythonProgramExecutable() {
        vernonRuntimeProgramExecutableDestroy(executable);
        for (const auto &[symbol, entry] : registeredCpuEntries)
            vernonRuntimeUnregisterCpuEntry(runtime, {symbol.data(), symbol.size()}, entry);
        vernonRuntimeProgramBundleDestroy(bundle);
    }

    std::unique_ptr<ProgramInvocationBuilder> invocationBuilder() {
        return std::make_unique<ProgramInvocationBuilder>(owner, runtime, executable);
    }

    std::array<uint32_t, 3> workgroupSize() const { return {0, 0, 0}; }

    nb::dict programAdSignature() const {
        if (!vernonRuntimeProgramExecutableHasProgramAutodiff(executable))
            throw std::runtime_error("executable has no Program autodiff signature");
        nb::dict signature;
        const std::pair<const char *, VernonProgramBoundaryRole> boundaries[] = {
            {"inputs", VERNON_PROGRAM_BOUNDARY_INPUT},
            {"outputs", VERNON_PROGRAM_BOUNDARY_OUTPUT},
            {"cotangents", VERNON_PROGRAM_BOUNDARY_COTANGENT},
            {"gradients", VERNON_PROGRAM_BOUNDARY_GRADIENT},
        };
        for (const auto &[name, boundary] : boundaries) {
            nb::list rows;
            const size_t count = vernonRuntimeProgramExecutableGetBoundaryCount(executable, boundary);
            for (size_t index = 0; index < count; ++index) {
                VernonProgramParameterView value{};
                if (vernonRuntimeProgramExecutableGetBoundaryByIndex(executable, boundary, index, &value) !=
                    VERNON_STATUS_OK)
                    throw std::runtime_error("cannot read Program autodiff signature");
                nb::dict row;
                row["path"] = nativeStringView(value.name);
                row["slot"] = value.slot;
                rows.append(std::move(row));
            }
            signature[name] = std::move(rows);
        }
        return signature;
    }

    nb::dict programAbi() const {
        nb::list slots;
        for (const vernon::runtime::ProgramBoundaryView &slot : vernon::runtime::programBoundaryViews(*executable)) {
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
        if (builder.executable != executable)
            throw std::invalid_argument("Program invocation builder belongs to another executable");
        VernonPullback *pullback = nullptr;
        const VernonStatus status = builder.forwardProgram(&pullback);
        if (pullback)
            vernonProgramPullbackDestroy(pullback);
        if (status != VERNON_STATUS_OK)
            throw std::runtime_error("Program forward failed: " + nativeStringView(vernonRuntimeGetLastError(runtime)));
    }

    nb::tuple programVjpBound(ProgramInvocationBuilder &builder, const nb::dict &programBindings,
                              nb::object executableOwner, const nb::object &checkpointMemoryBudget,
                              const std::string &checkpointPolicy) {
        if (!vernonRuntimeProgramExecutableHasProgramAutodiff(executable))
            throw std::runtime_error("executable has no Program autodiff signature");
        if (checkpointMemoryBudget.is_none())
            vernon::runtime::autodiffSetProgramCheckpointPlan(executable, nullptr, checkpointPolicy);
        else {
            const uint64_t budget = nb::cast<uint64_t>(checkpointMemoryBudget);
            vernon::runtime::autodiffSetProgramCheckpointPlan(executable, &budget, checkpointPolicy);
        }
        const auto declaredDerivativePath = [&](VernonAdDerivativeRole role, const std::string &leafPath) {
            const size_t count = vernonRuntimeProgramExecutableGetAdDerivativeGroupCount(executable);
            for (size_t groupIndex = 0; groupIndex < count; ++groupIndex) {
                VernonAdDerivativeGroupView group{};
                group.struct_size = sizeof(group);
                if (vernonRuntimeProgramExecutableGetAdDerivativeGroupByIndex(executable, groupIndex, &group) !=
                        VERNON_STATUS_OK ||
                    group.role != role)
                    continue;
                for (size_t leafIndex = 0; leafIndex < group.leaf_count; ++leafIndex) {
                    VernonStringView leaf{};
                    if (vernonRuntimeProgramExecutableGetAdDerivativeGroupLeaf(executable, groupIndex, leafIndex,
                                                                               &leaf) == VERNON_STATUS_OK &&
                        nativeStringView(leaf) == leafPath)
                        return nativeStringView(group.declared_path);
                }
            }
            throw std::runtime_error("Program derivative leaf has no declared group");
        };
        const auto reflectedDerivativeLeaves = [&](VernonProgramBoundaryRole boundaryRole,
                                                   VernonAdDerivativeRole derivativeRole) {
            std::vector<PythonAdMetadata> result;
            const size_t groupCount = vernonRuntimeProgramExecutableGetAdDerivativeGroupCount(executable);
            for (size_t groupIndex = 0; groupIndex < groupCount; ++groupIndex) {
                VernonAdDerivativeGroupView group{};
                group.struct_size = sizeof(group);
                if (vernonRuntimeProgramExecutableGetAdDerivativeGroupByIndex(executable, groupIndex, &group) !=
                        VERNON_STATUS_OK ||
                    group.role != derivativeRole)
                    continue;
                const std::string declaredPath = nativeStringView(group.declared_path);
                VernonProgramParameterView boundary{};
                if (vernonRuntimeProgramExecutableFindBoundary(executable, boundaryRole,
                                                               {declaredPath.data(), declaredPath.size()},
                                                               &boundary) != VERNON_STATUS_OK)
                    throw std::runtime_error("Program derivative group has no canonical boundary");
                for (size_t groupLeafIndex = 0; groupLeafIndex < group.leaf_count; ++groupLeafIndex) {
                    VernonStringView groupLeaf{};
                    if (vernonRuntimeProgramExecutableGetAdDerivativeGroupLeaf(executable, groupIndex, groupLeafIndex,
                                                                               &groupLeaf) != VERNON_STATUS_OK)
                        throw std::runtime_error("cannot read Program derivative group leaf");
                    const std::string expectedPath = nativeStringView(groupLeaf);
                    bool matched = false;
                    for (size_t boundaryLeafIndex = 0; boundaryLeafIndex < boundary.element_layout.leaf_count;
                         ++boundaryLeafIndex) {
                        VernonProgramValueLeafView leaf{};
                        leaf.struct_size = sizeof(leaf);
                        if (vernonRuntimeProgramExecutableGetBoundaryValueLeaf(
                                executable, boundaryRole, boundary.slot, boundaryLeafIndex, &leaf) != VERNON_STATUS_OK)
                            throw std::runtime_error("cannot read canonical Program boundary leaf");
                        std::string relativePath;
                        for (size_t component = 0; component < leaf.path_count; ++component) {
                            if (!relativePath.empty())
                                relativePath.push_back('.');
                            relativePath += leaf.path[component].kind == VERNON_VALUE_PATH_FIELD
                                                ? nativeStringView(leaf.path[component].field)
                                                : std::to_string(leaf.path[component].index);
                        }
                        std::string reflectedPath = declaredPath;
                        const bool alreadyQualified =
                            !relativePath.empty() &&
                            (declaredPath == relativePath ||
                             (declaredPath.size() > relativePath.size() &&
                              declaredPath.compare(declaredPath.size() - relativePath.size(), relativePath.size(),
                                                   relativePath) == 0 &&
                              declaredPath[declaredPath.size() - relativePath.size() - 1] == '.'));
                        if (!relativePath.empty() && !alreadyQualified)
                            reflectedPath += "." + relativePath;
                        if (reflectedPath != expectedPath)
                            continue;
                        PythonAdMetadata metadata;
                        metadata.path = expectedPath;
                        metadata.binding = declaredPath;
                        metadata.dtype = static_cast<VernonDataType>(leaf.value.dtype);
                        if (boundary.rank)
                            metadata.shape.assign(boundary.static_shape, boundary.static_shape + boundary.rank);
                        if (leaf.static_rank)
                            metadata.shape.insert(metadata.shape.end(), leaf.static_shape,
                                                  leaf.static_shape + leaf.static_rank);
                        result.push_back(std::move(metadata));
                        matched = true;
                        break;
                    }
                    if (!matched)
                        throw std::runtime_error("Program derivative group leaf is absent from its canonical boundary");
                }
            }
            return result;
        };
        nb::dict outputs;
        if (builder.executable != executable)
            throw std::invalid_argument("Program invocation builder belongs to another executable");
        VernonPullback *pullback = nullptr;
        const VernonStatus status = builder.forwardProgram(&pullback);
        if (status != VERNON_STATUS_OK) {
            const std::string error = nativeStringView(vernonRuntimeGetLastError(runtime));
            if (status == VERNON_STATUS_INVALID_ARGUMENT || error.find("memory budget") != std::string::npos)
                throw std::invalid_argument(error);
            throw std::runtime_error("Program autodiff forward failed: " + error);
        }
        std::unique_ptr<VernonPullback, decltype(&vernonProgramPullbackDestroy)> pullbackOwner(
            pullback, &vernonProgramPullbackDestroy);

        std::vector<PythonAdMetadata> cotangents =
            reflectedDerivativeLeaves(VERNON_PROGRAM_BOUNDARY_COTANGENT, VERNON_AD_DERIVATIVE_COTANGENT);
        const auto instantiateBoundMetadata = [&](PythonAdMetadata &leaf) {
            nb::object value = resolveProgramInputLeaf(programBindings, leaf.path);
            if (nb::hasattr(value, "_native_host_array"))
                value = value.attr("_native_host_array")();
            nb::object array = nb::module_::import_("numpy").attr("asarray")(value);
            leaf.shape = nb::cast<std::vector<uint64_t>>(array.attr("shape"));
        };
        for (PythonAdMetadata &leaf : cotangents) {
            leaf.binding = declaredDerivativePath(VERNON_AD_DERIVATIVE_COTANGENT, leaf.path);
            if (const size_t separator = leaf.binding.find('.'); separator != std::string::npos)
                leaf.binding.resize(separator);
            instantiateBoundMetadata(leaf);
        }
        std::vector<PythonAdMetadata> gradients =
            reflectedDerivativeLeaves(VERNON_PROGRAM_BOUNDARY_GRADIENT, VERNON_AD_DERIVATIVE_GRADIENT);
        for (PythonAdMetadata &leaf : gradients) {
            leaf.binding = declaredDerivativePath(VERNON_AD_DERIVATIVE_GRADIENT, leaf.path);
            if (const size_t separator = leaf.binding.find('.'); separator != std::string::npos)
                leaf.binding.resize(separator);
            instantiateBoundMetadata(leaf);
        }
        return nb::make_tuple(outputs,
                              std::make_unique<PythonPullback>(owner, runtime, executable, pullbackOwner.release(),
                                                               std::move(gradients), std::move(cotangents), false,
                                                               std::move(executableOwner), nb::dict(programBindings)));
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
        const size_t groupCount = vernonRuntimeProgramExecutableGetAdDerivativeGroupCount(executable);
        for (size_t groupIndex = 0; groupIndex < groupCount; ++groupIndex) {
            VernonAdDerivativeGroupView group{};
            group.struct_size = sizeof(group);
            if (vernonRuntimeProgramExecutableGetAdDerivativeGroupByIndex(executable, groupIndex, &group) !=
                VERNON_STATUS_OK)
                throw std::runtime_error("cannot read autodiff derivative group metadata");
            nb::list leaves;
            for (size_t leafIndex = 0; leafIndex < group.leaf_count; ++leafIndex) {
                VernonStringView leafPath{};
                if (vernonRuntimeProgramExecutableGetAdDerivativeGroupLeaf(executable, groupIndex, leafIndex,
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
        const size_t count = vernonRuntimeProgramExecutableGetParameterCount(executable);
        result.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            VernonProgramParameterView view{};
            if (vernonRuntimeProgramExecutableGetParameterByIndex(executable, index, &view) != VERNON_STATUS_OK)
                throw std::runtime_error("cannot read loaded executable parameter");
            ProgramParameterMetadata parameter = parameterMetadata(view);
            parameter.elementLeafPaths.reserve(parameter.elementLeaves.size());
            parameter.elementLeafShapes.reserve(parameter.elementLeaves.size());
            for (size_t leafIndex = 0; leafIndex < parameter.elementLeaves.size(); ++leafIndex) {
                VernonProgramValueLeafView leaf{};
                leaf.struct_size = sizeof(leaf);
                const VernonStringView name{parameter.name.data(), parameter.name.size()};
                if (vernonRuntimeProgramExecutableGetParameterValueLeaf(executable, name, leafIndex, &leaf) !=
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
        const size_t count = vernonRuntimeProgramExecutableGetOutputCount(executable);
        result.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            VernonProgramOutputView view{};
            if (vernonRuntimeProgramExecutableGetOutputByIndex(executable, index, &view) != VERNON_STATUS_OK)
                throw std::runtime_error("cannot read loaded executable output");
            result.push_back(outputMetadata(view));
        }
        return result;
    }

    Runtime *owner{};
    VernonRuntimeContext *runtime{};
    VernonProgramBundle *bundle{};
    VernonProgramExecutable *executable{};
    // ORC entry pointers are valid only while an interned compile result owns the JIT.
    std::vector<std::shared_ptr<InternedCpuJit>> internedCpuJits;
    std::vector<SharedCompileResult> retainedResults;
    std::vector<std::pair<std::string, VernonCpuEntryPoint>> registeredCpuEntries;
};

#endif
