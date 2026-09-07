#include "runtime_autodiff_internal.h"

#include "runtime/pipeline_metadata.h"
#include "runtime/program_execution_manifest.h"
#include "runtime/resolved_execution_plan.h"
#include "runtime/runtime_state.h"
#include "runtime_autodiff_telemetry.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

VernonDifferentiatedProgram *differentiatedPipeline(VernonProgramExecutable *pipeline) {
    return pipeline ? &pipeline->autodiff : nullptr;
}

const VernonDifferentiatedProgram *differentiatedPipeline(const VernonProgramExecutable *pipeline) {
    return pipeline ? &pipeline->autodiff : nullptr;
}

} // namespace

namespace vernon::runtime::ad {

size_t dtypeSize(VernonDataType dtype) {
    switch (dtype) {
    case VERNON_DATA_BOOL:
    case VERNON_DATA_U8:
        return 1;
    case VERNON_DATA_F16:
        return 2;
    case VERNON_DATA_I32:
    case VERNON_DATA_U32:
    case VERNON_DATA_F32:
        return sizeof(float);
    case VERNON_DATA_F64:
        return sizeof(double);
    default:
        return 0;
    }
}

bool materializeDerivativeValueAbi(ValueAbi &derivative, const std::vector<ValueAbi> &sources) {
    const auto source = std::find_if(sources.begin(), sources.end(),
                                     [&](const ValueAbi &value) { return value.path == derivative.path; });
    if (source == sources.end())
        return false;
    const size_t sourceScalar = dtypeSize(source->dtype);
    const size_t derivativeScalar = dtypeSize(derivative.dtype);
    if (!sourceScalar || !derivativeScalar || source->byteSize % sourceScalar ||
        (source->byteSize / sourceScalar &&
         derivativeScalar > std::numeric_limits<size_t>::max() / (source->byteSize / sourceScalar)))
        return false;
    derivative.byteSize = (source->byteSize / sourceScalar) * derivativeScalar;
    derivative.logicalShape = source->logicalShape;
    return true;
}

bool appendValueLayoutAbi(const ValueLayout &layout, const std::vector<uint64_t> &shape, const std::string &rootPath,
                          std::vector<ValueAbi> &values, std::string &error) {
    if (layout.leaves.empty())
        return error = "autodiff parameter has no canonical Value layout", false;
    size_t elementCount = 1;
    bool dynamicShape = false;
    for (uint64_t extent : shape) {
        if (!extent) {
            dynamicShape = true;
            continue;
        }
        if (elementCount > std::numeric_limits<size_t>::max() / static_cast<size_t>(extent))
            return error = "autodiff parameter shape overflows", false;
        elementCount *= static_cast<size_t>(extent);
    }
    for (const ValueLeaf &leaf : layout.leaves) {
        const std::optional<VernonDataType> dtype = pipelineDataType(leaf.dtype);
        size_t bytes = dtype ? dtypeSize(*dtype) : 0;
        if (!bytes || !leaf.scalarCount ||
            bytes > std::numeric_limits<size_t>::max() / static_cast<size_t>(leaf.scalarCount))
            return error = "autodiff parameter leaf ABI is invalid", false;
        bytes *= static_cast<size_t>(leaf.scalarCount);
        if (!dynamicShape) {
            if (elementCount && bytes > std::numeric_limits<size_t>::max() / elementCount)
                return error = "autodiff parameter leaf byte size overflows", false;
            bytes *= elementCount;
        }
        std::string path = canonicalValueLeafPath(rootPath, leaf);
        std::vector<uint64_t> logicalShape = shape;
        logicalShape.insert(logicalShape.end(), leaf.shape.begin(), leaf.shape.end());
        ValueAbi candidate{std::move(path), *dtype, bytes, std::max<size_t>(layout.alignment, 1),
                           std::move(logicalShape)};
        const auto existing = std::find_if(values.begin(), values.end(),
                                           [&](const ValueAbi &value) { return value.path == candidate.path; });
        if (existing != values.end()) {
            if (!sameValueAbi(*existing, candidate))
                return error = "autodiff parameters define conflicting ABI values for '" + candidate.path + "'", false;
        } else {
            values.push_back(std::move(candidate));
        }
    }
    return true;
}

bool appendParameterValueAbi(const Parameter &parameter, const std::string &rootPath, std::vector<ValueAbi> &values,
                             std::string &error) {
    const ValueLayout &layout = parameter.valueLayout ? *parameter.valueLayout : parameter.elementLayout;
    return appendValueLayoutAbi(layout, parameter.shape, rootPath, values, error);
}

bool validLaunchSize(VernonLaunchSize grid) { return grid.x != 0 && grid.y != 0 && grid.z != 0; }

bool invocationExtent(VernonLaunchSize grid, VernonLaunchSize workgroup, VernonLaunchSize &extent) {
    if (!validLaunchSize(grid) || !validLaunchSize(workgroup) ||
        grid.x > std::numeric_limits<uint32_t>::max() / workgroup.x ||
        grid.y > std::numeric_limits<uint32_t>::max() / workgroup.y ||
        grid.z > std::numeric_limits<uint32_t>::max() / workgroup.z)
        return false;
    extent = {grid.x * workgroup.x, grid.y * workgroup.y, grid.z * workgroup.z};
    return true;
}

bool carrierCount(VernonLaunchSize extent, size_t &count) {
    if (!validLaunchSize(extent) || extent.x > SIZE_MAX / extent.y ||
        static_cast<size_t>(extent.x) * extent.y > SIZE_MAX / extent.z)
        return false;
    count = static_cast<size_t>(extent.x) * extent.y * extent.z;
    return true;
}

bool materializeCarrierValue(ValueAbi &abi, VernonLaunchSize extent) {
    size_t count = 0;
    if (!carrierCount(extent, count) || (count && abi.byteSize > SIZE_MAX / count))
        return false;
    abi.byteSize *= count;
    if (count > 1)
        abi.logicalShape.insert(abi.logicalShape.begin(), {extent.z, extent.y, extent.x});
    return true;
}

bool validSet(const VernonAdValueSet *set, bool required) {
    return (!required && !set) ||
           (set && set->struct_size >= sizeof(VernonAdValueSet) && (!set->value_count || set->values));
}

VernonAdValue *findValue(VernonAdValueSet &set, const std::string &path) {
    for (size_t index = 0; index < set.value_count; ++index) {
        VernonAdValue &value = set.values[index];
        if (value.struct_size >= sizeof(VernonAdValue) && value.path.data && value.path.size == path.size() &&
            std::memcmp(value.path.data, path.data(), path.size()) == 0)
            return &value;
    }
    return nullptr;
}

const VernonAdValue *findValue(const VernonAdValueSet &set, const std::string &path) {
    for (size_t index = 0; index < set.value_count; ++index) {
        const VernonAdValue &value = set.values[index];
        if (value.struct_size >= sizeof(VernonAdValue) && value.path.data && value.path.size == path.size() &&
            std::memcmp(value.path.data, path.data(), path.size()) == 0)
            return &value;
    }
    return nullptr;
}

bool valueMatches(const VernonAdValue &value, const ValueAbi &abi) {
    if (value.dtype != abi.dtype || !value.data || value.size != abi.byteSize ||
        value.rank != abi.logicalShape.size() || (value.rank && !value.shape))
        return false;
    return value.rank == 0 || std::equal(abi.logicalShape.begin(), abi.logicalShape.end(), value.shape);
}

bool derivativeAbiMatches(const ValueAbi &primal, const ValueAbi &derivative) {
    if (primal.dtype != VERNON_DATA_F16 && primal.dtype != VERNON_DATA_F32 && primal.dtype != VERNON_DATA_F64)
        return false;
    const VernonDataType expected = primal.dtype == VERNON_DATA_F64 ? VERNON_DATA_F64 : VERNON_DATA_F32;
    if (derivative.dtype != expected || derivative.logicalShape != primal.logicalShape)
        return false;
    const size_t primalScalarSize = dtypeSize(primal.dtype);
    const size_t derivativeScalarSize = dtypeSize(derivative.dtype);
    return primalScalarSize && derivativeScalarSize && primal.byteSize % primalScalarSize == 0 &&
           derivative.byteSize == primal.byteSize / primalScalarSize * derivativeScalarSize;
}

bool makeCotangentBytes(const VernonAdValueSet *cotangents, const ValueAbi &abi, std::vector<uint8_t> &bytes,
                        std::string &error) {
    bytes.resize(abi.byteSize);
    if (cotangents) {
        if (cotangents->value_count != 1) {
            error = "pullback requires one output cotangent";
            return false;
        }
        const VernonAdValue *cotangent = findValue(*cotangents, abi.path);
        if (!cotangent || !valueMatches(*cotangent, abi)) {
            error = "output cotangent does not match backward reflection";
            return false;
        }
        std::memcpy(bytes.data(), cotangent->data, abi.byteSize);
        return true;
    }
    if (abi.byteSize != dtypeSize(abi.dtype)) {
        error = "a non-scalar output requires an explicit cotangent";
        return false;
    }
    if (abi.dtype == VERNON_DATA_F32) {
        const float one = 1.0f;
        std::memcpy(bytes.data(), &one, sizeof(one));
        return true;
    }
    if (abi.dtype == VERNON_DATA_F64) {
        const double one = 1.0;
        std::memcpy(bytes.data(), &one, sizeof(one));
        return true;
    }
    error = "implicit cotangents require an f32 or f64 scalar output";
    return false;
}

bool validateDerivativeGroupsAgainstSignature(VernonRuntimeContext &context,
                                              const std::vector<AutodiffDerivativeGroup> &groups,
                                              const Signature &signature) {
    const auto matches = [&](AutodiffDerivativeRole role, const std::vector<ValueAbi> &values) {
        std::vector<std::string> expected = autodiffDerivativeLeafPaths(groups, role);
        if (expected.size() != values.size())
            return false;
        std::vector<std::string> actual;
        actual.reserve(values.size());
        for (const ValueAbi &value : values)
            actual.push_back(value.path);
        std::sort(expected.begin(), expected.end());
        std::sort(actual.begin(), actual.end());
        return expected == actual;
    };
    if (!matches(AutodiffDerivativeRole::Gradient, signature.gradients)) {
        invocationDiagnostic(context) = "autodiff gradient groups do not match the executable signature";
        return false;
    }
    if (!matches(AutodiffDerivativeRole::Cotangent, signature.cotangents)) {
        invocationDiagnostic(context) = "autodiff cotangent groups do not match the executable signature";
        return false;
    }
    return true;
}

VernonStatus preparePipelineForwardCommandPlan(VernonProgramExecutable &pipeline,
                                               const VernonStageInvocationDescriptor &invocation,
                                               const VernonAdValueSet &inputs,
                                               execution::detail::RhiCommandExecutionPlan &plan,
                                               VernonPullback *&pullback) {
    pullback = nullptr;
    const auto *differentiated = differentiatedPipeline(&pipeline);
    auto *executable = differentiated ? differentiated->executable.get() : nullptr;
    if (!executable || invocation.struct_size < sizeof(VernonStageInvocationDescriptor) ||
        invocation.abi_version != VERNON_PROGRAM_VERSION || !validLaunchSize(invocation.compute_grid) ||
        (invocation.argument_count && !invocation.arguments) || !validSet(&inputs, true)) {
        invocationDiagnostic(*pipeline.context) = "invalid deferred autodiff forward invocation";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    auto result = std::make_unique<VernonPullback>();
    result->contextLease = vernon::runtime::acquireContextLease(*pipeline.context);
    std::unique_ptr<PullbackExecution> execution;
    const ForwardExecutionTarget target{{}, &invocation, &plan};
    const VernonStatus status = executable->forward(target, invocation.compute_grid, inputs, nullptr, execution);
    if (status != VERNON_STATUS_OK)
        return status;
    if (!execution) {
        invocationDiagnostic(*pipeline.context) = "deferred autodiff forward produced no pullback";
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    result->execution = std::move(execution);
    pullback = result.release();
    return VERNON_STATUS_OK;
}

} // namespace vernon::runtime::ad

namespace vernon::runtime::program_execution {

VernonStatus forwardProgramInvocation(VernonProgramExecutable &pipeline,
                                      const VernonStageInvocationDescriptor &invocation, VernonPullback *&pullback,
                                      const ProgramInvocationContext *programContext) {
    pullback = nullptr;
    const auto *differentiated = differentiatedPipeline(&pipeline);
    auto *executable = differentiated ? differentiated->executable.get() : nullptr;
    if (!executable || invocation.struct_size < sizeof(VernonStageInvocationDescriptor) ||
        invocation.abi_version != VERNON_PROGRAM_VERSION || (invocation.argument_count && !invocation.arguments)) {
        invocationDiagnostic(*pipeline.context) = "invalid canonical Program invocation";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), nullptr, 0, {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), nullptr, 0, {}};
    std::unique_ptr<ad::PullbackExecution> execution;
    const ad::ForwardExecutionTarget target{{}, &invocation, nullptr, programContext};
    const VernonStatus status = executable->forward(target, {1, 1, 1}, inputs, &outputs, execution);
    if (status != VERNON_STATUS_OK)
        return status;
    if (!execution)
        return VERNON_STATUS_OK;
    auto result = std::make_unique<VernonPullback>();
    result->contextLease = vernon::runtime::acquireContextLease(*pipeline.context);
    result->execution = std::move(execution);
    pullback = result.release();
    return VERNON_STATUS_OK;
}

void attachProgramSnapshot(VernonPullback &pullback, std::shared_ptr<const program::InvocationSnapshot> snapshot) {
    pullback.programSnapshot = std::move(snapshot);
}

} // namespace vernon::runtime::program_execution

namespace {

VernonStatus fail(VernonRuntimeContext *context, std::string_view message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) noexcept {
    try {
        if (context)
            vernon::runtime::invocationDiagnostic(*context) = message;
    } catch (...) {
    }
    return status;
}

const vernon::runtime::ad::Executable *autodiffExecutable(const VernonProgramExecutable *pipeline) {
    const auto *differentiated = differentiatedPipeline(pipeline);
    return differentiated ? differentiated->executable.get() : nullptr;
}

bool validDeviceFootprint(const VernonAdDeviceValue &value) {
    const size_t scalarBytes = vernon::runtime::ad::dtypeSize(value.dtype);
    if (!scalarBytes || !value.buffer_size || value.buffer_size > std::numeric_limits<size_t>::max() ||
        value.offset > value.buffer_size || value.size > value.buffer_size || (value.rank && !value.byte_strides))
        return false;
    uint64_t before = 0;
    uint64_t after = 0;
    for (uint32_t dimension = 0; dimension < value.rank; ++dimension) {
        if (!value.shape[dimension])
            return false;
        const int64_t stride = value.byte_strides[dimension];
        const uint64_t magnitude =
            stride < 0 ? static_cast<uint64_t>(-(stride + 1)) + 1 : static_cast<uint64_t>(stride);
        const uint64_t count = value.shape[dimension] - 1;
        if (count && magnitude > std::numeric_limits<uint64_t>::max() / count)
            return false;
        const uint64_t span = magnitude * count;
        uint64_t &side = stride < 0 ? before : after;
        if (span > std::numeric_limits<uint64_t>::max() - side)
            return false;
        side += span;
    }
    return before <= value.offset && after <= value.buffer_size - value.offset &&
           scalarBytes <= value.buffer_size - value.offset - after;
}

bool validDeviceSet(const VernonAdDeviceValueSet *set, bool required) {
    if (!set)
        return !required;
    if (set->struct_size < sizeof(*set) || (set->value_count && !set->values) ||
        std::any_of(std::begin(set->reserved), std::end(set->reserved), [](uint32_t value) { return value != 0; }))
        return false;
    for (size_t index = 0; index < set->value_count; ++index) {
        const VernonAdDeviceValue &value = set->values[index];
        if (value.struct_size < sizeof(value) || !value.path.data || !value.path.size || !value.size ||
            value.buffer.index == VERNON_RHI_INVALID_HANDLE_INDEX || (value.rank && !value.shape) ||
            !validDeviceFootprint(value) ||
            std::any_of(
                std::begin(value.reserved), std::end(value.reserved), [](uint32_t reserved) { return reserved != 0; }))
            return false;
    }
    return true;
}

} // namespace

namespace {

bool copyMetadata(const std::vector<vernon::runtime::ad::ValueAbi> &values, size_t index,
                  VernonAdValueMetadataView *metadata) {
    if (!metadata || metadata->struct_size < sizeof(*metadata) || index >= values.size())
        return false;
    const vernon::runtime::ad::ValueAbi &value = values[index];
    *metadata = {sizeof(*metadata),
                 {value.path.data(), value.path.size()},
                 value.dtype,
                 static_cast<uint32_t>(value.logicalShape.size()),
                 value.logicalShape.empty() ? nullptr : value.logicalShape.data(),
                 {}};
    return true;
}

const vernon::runtime::program::Program *programExecution(const VernonProgramExecutable *pipeline) {
    if (!pipeline)
        return nullptr;
    const vernon::runtime::program::Program &program = pipeline->executionPlan->resolvedProgram->program;
    return vernon::runtime::program::findGraph(program, "backward") ? &program : nullptr;
}

std::optional<vernon::runtime::program::BoundaryRole> programBoundaryRole(VernonProgramAdBoundary boundary) {
    using vernon::runtime::program::BoundaryRole;
    switch (boundary) {
    case VERNON_PROGRAM_AD_INPUT:
        return BoundaryRole::Input;
    case VERNON_PROGRAM_AD_OUTPUT:
        return BoundaryRole::Output;
    case VERNON_PROGRAM_AD_COTANGENT:
        return BoundaryRole::Cotangent;
    case VERNON_PROGRAM_AD_GRADIENT:
        return BoundaryRole::Gradient;
    default:
        return std::nullopt;
    }
}

const vernon::runtime::program::BoundarySlot *programBoundaryAt(const vernon::runtime::program::Program &program,
                                                                vernon::runtime::program::BoundaryRole role,
                                                                size_t index) {
    for (const vernon::runtime::program::BoundarySlot &slot : program.abi.boundarySlots)
        if (slot.role == role && index-- == 0)
            return &slot;
    return nullptr;
}

} // namespace

vernon::runtime::AutodiffPullbackMemoryUsage
vernon::runtime::autodiffPullbackMemoryUsage(const VernonPullback *pullback) {
    if (!pullback || !pullback->execution)
        return {};
    const ad::PullbackMemoryUsage usage = pullback->execution->memoryUsage();
    return {usage.logicalResidualBytes, usage.residentBytes, usage.allocatedBytes, usage.retainedAllocationBytes,
            usage.peakTemporaryBytes};
}

vernon::runtime::AutodiffPullbackControlPlaneUsage
vernon::runtime::autodiffPullbackControlPlaneUsage(const VernonPullback *pullback) {
    if (!pullback || !pullback->execution)
        return {};
    const program_execution::ExecutionControlPlaneUsage usage = pullback->execution->controlPlaneUsage();
    return {usage.submissions,
            usage.waits,
            usage.readbacks,
            usage.atomicPublications,
            usage.temporaryAllocationBytes,
            usage.deviceWaitNanoseconds};
}

uint64_t vernon::runtime::ad::PullbackExecution::peakRuntimeManagedBytes() const {
    const PullbackMemoryUsage usage = memoryUsage();
    return std::max(std::max(usage.retainedAllocationBytes, usage.allocatedBytes), usage.peakTemporaryBytes);
}

vernon::runtime::AutodiffPullbackCheckpointPlan
vernon::runtime::autodiffPullbackCheckpointPlan(const VernonPullback *pullback) {
    if (!pullback || !pullback->execution)
        return {};
    return pullback->execution->checkpointPlan();
}

std::vector<vernon::runtime::AutodiffPullbackPassTelemetry>
vernon::runtime::autodiffPullbackPassTelemetry(const VernonPullback *pullback) {
    if (!pullback || !pullback->execution)
        return {};
    return pullback->execution->passTelemetry();
}

uint64_t vernon::runtime::autodiffPullbackPeakRuntimeManagedBytes(const VernonPullback *pullback) {
    if (!pullback || !pullback->execution)
        return 0;
    return pullback->execution->peakRuntimeManagedBytes();
}

void vernon::runtime::autodiffSetProgramCheckpointPlan(VernonProgramExecutable *pipeline, const uint64_t *memoryBudget,
                                                       std::string_view policy) {
    if (!pipeline)
        return;
    if (memoryBudget)
        pipeline->autodiff.checkpointMemoryBudget = *memoryBudget;
    else
        pipeline->autodiff.checkpointMemoryBudget.reset();
    pipeline->autodiff.checkpointPolicy = std::string(policy);
}

size_t vernon::runtime::autodiffHostTapeContextLimit(const VernonRuntimeContext *context) {
    return context ? ad::autodiffMemoryContextLimit(context->autodiffMemoryPolicy) : 0;
}

VernonRhiDevice vernon::runtime::autodiffRhiDevice(const VernonRuntimeContext *context) {
    return context ? context->rhiDevice : VernonRhiDevice{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
}

extern "C" {

uint8_t vernonRuntimeProgramExecutableHasProgramAutodiff(const VernonProgramExecutable *pipeline) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    return programExecution(pipeline) ? 1 : 0;
}

size_t vernonRuntimeProgramExecutableGetProgramAdValueCount(const VernonProgramExecutable *pipeline,
                                                            VernonProgramAdBoundary boundary) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const vernon::runtime::program::Program *execution = programExecution(pipeline);
    if (!execution)
        return 0;
    if (boundary == VERNON_PROGRAM_AD_CAPTURE)
        return execution->residualContract ? execution->residualContract->captures.size() : 0;
    const std::optional<vernon::runtime::program::BoundaryRole> role = programBoundaryRole(boundary);
    if (!role)
        return 0;
    return static_cast<size_t>(
        std::count_if(execution->abi.boundarySlots.begin(), execution->abi.boundarySlots.end(),
                      [&](const vernon::runtime::program::BoundarySlot &slot) { return slot.role == *role; }));
}

VernonStatus vernonRuntimeProgramExecutableGetProgramAdValueByIndex(const VernonProgramExecutable *pipeline,
                                                                    VernonProgramAdBoundary boundary, size_t index,
                                                                    VernonProgramAdValueView *view) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const vernon::runtime::program::Program *execution = programExecution(pipeline);
    if (!execution || !view || view->struct_size < sizeof(*view))
        return fail(pipeline ? pipeline->context : nullptr, "invalid Program autodiff value query");
    const vernon::runtime::program::BoundarySlot *binding = nullptr;
    uint32_t capture = UINT32_MAX;
    if (boundary == VERNON_PROGRAM_AD_CAPTURE) {
        if (execution->residualContract && index < execution->residualContract->captures.size())
            capture = execution->residualContract->captures[index].value;
    } else if (const std::optional<vernon::runtime::program::BoundaryRole> role = programBoundaryRole(boundary)) {
        binding = programBoundaryAt(*execution, *role, index);
    }
    const uint32_t valueId = binding ? binding->value : capture;
    if (valueId >= execution->values.size())
        return fail(pipeline ? pipeline->context : nullptr, "invalid Program autodiff value query");
    const vernon::runtime::program::Value &slot = execution->values[valueId];
    const std::string &path = binding ? binding->path : slot.name;
    const bool external = std::any_of(
        execution->graphs.begin(), execution->graphs.end(), [&](const vernon::runtime::program::Graph &graph) {
            return std::any_of(
                graph.inputs.begin(), graph.inputs.end(), [&](const vernon::runtime::program::GraphInput &input) {
                    return input.kind == vernon::runtime::program::GraphInputKind::UserInput && input.value == valueId;
                });
        });
    const bool output = std::any_of(execution->graphs.begin(), execution->graphs.end(),
                                    [&](const vernon::runtime::program::Graph &graph) {
                                        return std::any_of(graph.outputs.begin(), graph.outputs.end(),
                                                           [&](const vernon::runtime::program::GraphOutput &candidate) {
                                                               return candidate.value == valueId;
                                                           });
                                    });
    *view = {sizeof(*view),
             {path.data(), path.size()},
             valueId,
             static_cast<uint8_t>(external),
             static_cast<uint8_t>(output),
             {}};
    return VERNON_STATUS_OK;
}

size_t vernonRuntimeProgramExecutableGetAdInputCount(const VernonProgramExecutable *pipeline) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *executable = autodiffExecutable(pipeline);
    return executable ? executable->signature().inputs.size() : 0;
}

VernonStatus vernonRuntimeProgramExecutableGetAdInputByIndex(const VernonProgramExecutable *pipeline, size_t index,
                                                             VernonAdValueMetadataView *metadata) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *executable = autodiffExecutable(pipeline);
    if (!executable || !copyMetadata(executable->signature().inputs, index, metadata))
        return fail(pipeline ? pipeline->context : nullptr, "invalid autodiff input query");
    return VERNON_STATUS_OK;
}

size_t vernonRuntimeProgramExecutableGetAdOutputCount(const VernonProgramExecutable *pipeline) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *executable = autodiffExecutable(pipeline);
    return executable ? executable->signature().outputs.size() : 0;
}

VernonStatus vernonRuntimeProgramExecutableGetAdOutputByIndex(const VernonProgramExecutable *pipeline, size_t index,
                                                              VernonAdValueMetadataView *metadata) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *executable = autodiffExecutable(pipeline);
    if (!executable || !copyMetadata(executable->signature().outputs, index, metadata))
        return fail(pipeline ? pipeline->context : nullptr, "invalid autodiff output query");
    return VERNON_STATUS_OK;
}

size_t vernonRuntimeProgramExecutableGetAdCotangentCount(const VernonProgramExecutable *pipeline) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *executable = autodiffExecutable(pipeline);
    return executable ? executable->signature().cotangents.size() : 0;
}

VernonStatus vernonRuntimeProgramExecutableGetAdCotangentByIndex(const VernonProgramExecutable *pipeline, size_t index,
                                                                 VernonAdValueMetadataView *metadata) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *executable = autodiffExecutable(pipeline);
    if (!executable || !copyMetadata(executable->signature().cotangents, index, metadata))
        return fail(pipeline ? pipeline->context : nullptr, "invalid autodiff cotangent query");
    return VERNON_STATUS_OK;
}

size_t vernonRuntimeProgramExecutableGetAdGradientCount(const VernonProgramExecutable *pipeline) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *executable = autodiffExecutable(pipeline);
    return executable ? executable->signature().gradients.size() : 0;
}

VernonStatus vernonRuntimeProgramExecutableGetAdGradientByIndex(const VernonProgramExecutable *pipeline, size_t index,
                                                                VernonAdValueMetadataView *metadata) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *executable = autodiffExecutable(pipeline);
    if (!executable || !copyMetadata(executable->signature().gradients, index, metadata))
        return fail(pipeline ? pipeline->context : nullptr, "invalid autodiff gradient query");
    return VERNON_STATUS_OK;
}

size_t vernonRuntimeProgramExecutableGetAdDerivativeGroupCount(const VernonProgramExecutable *pipeline) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *differentiated = differentiatedPipeline(pipeline);
    return differentiated ? differentiated->derivativeGroups.size() : 0;
}

VernonStatus vernonRuntimeProgramExecutableGetAdDerivativeGroupByIndex(const VernonProgramExecutable *pipeline,
                                                                       size_t groupIndex,
                                                                       VernonAdDerivativeGroupView *view) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *differentiated = differentiatedPipeline(pipeline);
    if (!differentiated || !view || view->struct_size < sizeof(*view) ||
        groupIndex >= differentiated->derivativeGroups.size())
        return fail(pipeline ? pipeline->context : nullptr, "invalid autodiff derivative group query");
    const vernon::runtime::AutodiffDerivativeGroup &group = differentiated->derivativeGroups[groupIndex];
    *view = {sizeof(*view),
             group.role == vernon::runtime::AutodiffDerivativeRole::Gradient ? VERNON_AD_DERIVATIVE_GRADIENT
                                                                             : VERNON_AD_DERIVATIVE_COTANGENT,
             {group.declaredPath.data(), group.declaredPath.size()},
             group.leafPaths.size(),
             {}};
    return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeProgramExecutableGetAdDerivativeGroupLeaf(const VernonProgramExecutable *pipeline,
                                                                    size_t groupIndex, size_t leafIndex,
                                                                    VernonStringView *leafPath) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *differentiated = differentiatedPipeline(pipeline);
    if (!differentiated || !leafPath || groupIndex >= differentiated->derivativeGroups.size()) {
        return fail(pipeline ? pipeline->context : nullptr, "invalid autodiff derivative group leaf query");
    }
    const vernon::runtime::AutodiffDerivativeGroup &group = differentiated->derivativeGroups[groupIndex];
    if (leafIndex >= group.leafPaths.size())
        return fail(pipeline->context, "invalid autodiff derivative group leaf query");
    const std::string &path = group.leafPaths[leafIndex];
    *leafPath = {path.data(), path.size()};
    return VERNON_STATUS_OK;
}

VernonStatus vernonPullbackApplyWithOptions(VernonPullback *pullback, const VernonAdValueSet *cotangents,
                                            VernonAdValueSet *gradients, const VernonPullbackApplyOptions *options) {
    VernonRuntimeContext *context = pullback && pullback->contextLease ? &pullback->contextLease->get() : nullptr;
    try {
        using namespace vernon::runtime::ad;
        vernon::runtime::RuntimeDiagnosticScope diagnostic(context);
        if (!pullback || !pullback->execution || !validSet(cotangents, false) || !validSet(gradients, true) ||
            !options || options->struct_size != sizeof(VernonPullbackApplyOptions) ||
            options->abi_version != VERNON_PULLBACK_APPLY_OPTIONS_VERSION ||
            std::any_of(std::begin(options->reserved), std::end(options->reserved),
                        [](uint32_t value) { return value != 0; }))
            return fail(context, "invalid pullback invocation");
        const auto boundedSize = [](uint64_t value) {
            return value > std::numeric_limits<size_t>::max() ? std::numeric_limits<size_t>::max()
                                                              : static_cast<size_t>(value);
        };
        const PullbackApplyOptions runtimeOptions{boundedSize(options->maximum_temporary_bytes),
                                                  boundedSize(options->maximum_reusable_construction_bytes)};
        return pullback->execution->apply(cotangents, *gradients, runtimeOptions);
    } catch (const std::bad_alloc &) {
        return fail(context, "cannot allocate pullback state", VERNON_STATUS_INTERNAL_ERROR);
    } catch (const std::length_error &) {
        return fail(context, "pullback allocation is too large", VERNON_STATUS_INTERNAL_ERROR);
    } catch (...) {
        return fail(context, "unexpected pullback failure", VERNON_STATUS_INTERNAL_ERROR);
    }
}

VernonStatus vernonPullbackApply(VernonPullback *pullback, const VernonAdValueSet *cotangents,
                                 VernonAdValueSet *gradients) {
    const VernonPullbackApplyOptions options{sizeof(VernonPullbackApplyOptions),
                                             VERNON_PULLBACK_APPLY_OPTIONS_VERSION,
                                             std::numeric_limits<uint64_t>::max(),
                                             0,
                                             {}};
    return vernonPullbackApplyWithOptions(pullback, cotangents, gradients, &options);
}

void vernonPullbackDestroy(VernonPullback *pullback) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(
        pullback && pullback->contextLease ? &pullback->contextLease->get() : nullptr);
    delete pullback;
}

} // extern "C"

VernonStatus vernon::runtime::ad::applyPullbackDeviceWithPlanSink(VernonPullback &pullback,
                                                                  const VernonAdDeviceValueSet *cotangents,
                                                                  VernonAdDeviceValueSet &gradients,
                                                                  const VernonPullbackApplyOptions *options,
                                                                  execution::detail::RhiCommandPlanSink &sink) {
    VernonRuntimeContext *context = pullback.contextLease ? &pullback.contextLease->get() : nullptr;
    try {
        vernon::runtime::RuntimeDiagnosticScope diagnostic(context);
        auto *deviceExecution =
            pullback.execution ? dynamic_cast<DevicePullbackExecution *>(pullback.execution.get()) : nullptr;
        if (!deviceExecution || !validDeviceSet(cotangents, false) || !validDeviceSet(&gradients, true) || !options ||
            options->struct_size != sizeof(VernonPullbackApplyOptions) ||
            options->abi_version != VERNON_PULLBACK_APPLY_OPTIONS_VERSION ||
            std::any_of(std::begin(options->reserved), std::end(options->reserved),
                        [](uint32_t value) { return value != 0; }))
            return fail(context, "invalid planned device pullback invocation");
        const auto boundedSize = [](uint64_t value) {
            return value > std::numeric_limits<size_t>::max() ? std::numeric_limits<size_t>::max()
                                                              : static_cast<size_t>(value);
        };
        const PullbackApplyOptions runtimeOptions{boundedSize(options->maximum_temporary_bytes),
                                                  boundedSize(options->maximum_reusable_construction_bytes)};
        return deviceExecution->applyDevice(cotangents, gradients, runtimeOptions, &sink);
    } catch (const std::bad_alloc &) {
        return fail(context, "cannot allocate planned device pullback state", VERNON_STATUS_INTERNAL_ERROR);
    } catch (const std::length_error &) {
        return fail(context, "planned device pullback allocation is too large", VERNON_STATUS_INTERNAL_ERROR);
    } catch (...) {
        return fail(context, "unexpected planned device pullback failure", VERNON_STATUS_INTERNAL_ERROR);
    }
}
