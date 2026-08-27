#include "runtime_pipeline_backend.h"

#include "backend_cpu.h"
#include "compute_launch_planner.h"
#include "runtime/autodiff/host_tape_allocator.h"
#include "runtime/autodiff/tape_allocator_abi.h"

#include "VernonCpuWorkgroupABI.h"

#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vernon::runtime {
namespace {

VernonStatus fail(VernonRuntimeContext &context, std::string error,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    invocationDiagnostic(context) = std::move(error);
    return status;
}

bool hasTapeBuiltins(const CpuPipelineState &state) {
    for (const std::string &builtin : state.layoutBuiltins) {
        if (builtin == VERNON_AD_TAPE_ALLOCATOR_BUILTIN || builtin == VERNON_AD_TAPE_ROOT_REGION_BUILTIN)
            return true;
    }
    return false;
}

bool cpuDispatchVolume(const uint32_t groupCount[3], const uint32_t workgroup[3], size_t &volume) {
    volume = 1;
    for (int axis = 0; axis < 3; ++axis) {
        const size_t groups = groupCount[axis] ? groupCount[axis] : 1;
        const size_t wg = workgroup[axis] ? workgroup[axis] : 1;
        if (!groups || !wg || groups > std::numeric_limits<size_t>::max() / wg)
            return false;
        const size_t axisVolume = groups * wg;
        if (volume > std::numeric_limits<size_t>::max() / axisVolume)
            return false;
        volume *= axisVolume;
    }
    return volume != 0;
}

VernonStatus packCpuInvocation(const CpuPipelineState &state, std::vector<unsigned char> &packed, std::string &error) {
    if (!state.packedSize || state.packedOffsets.size() != state.layout.size() ||
        state.packedFieldSizes.size() != state.layout.size()) {
        error = "CPU tape dispatch packed layout is incomplete";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    try {
        packed.assign(state.packedSize, 0);
    } catch (const std::bad_alloc &) {
        error = "cannot allocate CPU tape packed arguments";
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    for (size_t index = 0; index < state.layout.size(); ++index) {
        const auto &layout = state.layout[index];
        const auto &value = state.values[index];
        const size_t offset = state.packedOffsets[index];
        const size_t size = state.packedFieldSizes[index];
        if (offset > packed.size() || size > packed.size() - offset) {
            error = "CPU tape dispatch packed field is out of range";
            return VERNON_STATUS_INVALID_ARGUMENT;
        }
        if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            const auto &resource = value.payload.buffer.resource;
            if (!resource.resource.value || !resource.size) {
                error = "CPU storage binding requires a host Tensor";
                return VERNON_STATUS_INVALID_ARGUMENT;
            }
            const auto *storage =
                reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(resource.resource.value) + resource.offset);
            if (size == sizeof(uintptr_t)) {
                const uintptr_t pointer = reinterpret_cast<uintptr_t>(storage);
                std::memcpy(packed.data() + offset, &pointer, sizeof(pointer));
            } else {
                if (resource.size < size) {
                    error = "CPU storage binding is smaller than the inline argument";
                    return VERNON_STATUS_INVALID_ARGUMENT;
                }
                std::memcpy(packed.data() + offset, storage, size);
            }
        } else {
            if (!value.payload.inline_value.data || value.payload.inline_value.size != size) {
                error = "CPU inline binding is invalid";
                return VERNON_STATUS_INVALID_ARGUMENT;
            }
            std::memcpy(packed.data() + offset, value.payload.inline_value.data, size);
        }
    }
    return VERNON_STATUS_OK;
}

VernonStatus dispatchCpuTapedCompute(VernonRuntimeContext &context, CpuPipelineState &state, const uint32_t groups[3]) {
    if (!state.entry)
        return fail(context, "CPU tape dispatch has no entry");
    if (!state.tapeAllocator)
        return fail(context, "CPU tape dispatch has no allocator");
    ad::HostStaticTapeBatch *batch = ad::HostStaticTapeBatch::fromWriteDescriptor(state.tapeAllocator);
    if (!batch)
        return fail(context, "CPU tape allocator is not a host write descriptor");
    size_t volume = 0;
    if (!cpuDispatchVolume(groups, state.workgroup, volume))
        return fail(context, "CPU tape dispatch volume overflows");
    if (batch->size() < volume)
        return fail(context, "CPU tape dispatch has fewer lanes than the compute grid");
    if (state.tapeAllocatorOffset == std::numeric_limits<size_t>::max() ||
        state.tapeAllocatorOffset > state.packedSize ||
        sizeof(VernonAdTapeAllocator *) > state.packedSize - state.tapeAllocatorOffset)
        return fail(context, "CPU tape allocator packed offset is invalid");
    std::vector<unsigned char> packed;
    std::string packError;
    if (const VernonStatus status = packCpuInvocation(state, packed, packError); status != VERNON_STATUS_OK)
        return fail(context, std::move(packError), status);
    std::vector<std::vector<unsigned char>> lanePacked;
    std::vector<const void *> laneArguments;
    std::vector<ad::HostStaticTapeBatch::Reader> readers;
    try {
        lanePacked.assign(volume, packed);
        laneArguments.resize(volume);
        if (batch->isCompacted())
            readers.assign(volume, {});
    } catch (const std::bad_alloc &) {
        return fail(context, "cannot allocate CPU tape lane frames", VERNON_STATUS_INTERNAL_ERROR);
    }
    for (size_t lane = 0; lane < volume; ++lane) {
        VernonAdTapeAllocator *allocator = nullptr;
        VernonAdRegionHandle root = VERNON_AD_INVALID_REGION_HANDLE;
        if (batch->isCompacted()) {
            if (!batch->initializeReader(lane, readers[lane]))
                return fail(context, "CPU tape dispatch cannot initialize a sealed reader");
            allocator = readers[lane].descriptor();
            root = readers[lane].rootRegion();
        } else {
            allocator = batch->descriptor(lane);
            if (state.tapeRootOffset != std::numeric_limits<size_t>::max())
                root = batch->rootRegion(lane);
        }
        if (!allocator)
            return fail(context, "CPU tape dispatch is missing a lane allocator");
        std::memcpy(lanePacked[lane].data() + state.tapeAllocatorOffset, &allocator, sizeof(allocator));
        if (state.tapeRootOffset != std::numeric_limits<size_t>::max()) {
            if (batch->isCompacted() && !root)
                return fail(context, "CPU tape dispatch has no sealed root region");
            if (state.tapeRootOffset > packed.size() || sizeof(root) > packed.size() - state.tapeRootOffset)
                return fail(context, "CPU tape root packed offset is invalid");
            std::memcpy(lanePacked[lane].data() + state.tapeRootOffset, &root, sizeof(root));
        }
        laneArguments[lane] = lanePacked[lane].data();
    }
    CpuWorkgroupScheduler &scheduler = cpuWorkgroupScheduler(context);
    const VernonStatus status = scheduler.dispatch(groups, state.workgroup, [&](VernonCpuRangeV1 &range) {
        range.arguments = laneArguments.front();
        range.arguments_size = packed.size();
        range.results = nullptr;
        range.results_size = 0;
        range.textures = nullptr;
        range.lane_arguments = laneArguments.data();
        range.lane_table_count = laneArguments.size();
        const VernonCpuInvocation invocation{&range, VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1, nullptr, 0, nullptr};
        return state.entry(&invocation);
    });
    if (status != VERNON_STATUS_OK)
        return fail(context,
                    scheduler.lastDiagnostic().empty()
                        ? (status == VERNON_STATUS_INVALID_ARGUMENT ? "CPU builtin reflection or dispatch is invalid"
                                                                    : "CPU provider entry invocation failed")
                        : scheduler.lastDiagnostic(),
                    status);
    return status;
}

} // namespace

bool resolveCpuPipeline(VernonPipelineBundle &bundle, const Variant &variant, VernonLoadedPipeline &pipeline) {
    auto state = std::make_unique<CpuPipelineState>();
    CpuKernelState kernel;
    ReflectedEntry reflection;
    if (!loadCpuNativeArtifact(*bundle.context, *bundle.stages.at(variant.compute).cpuArtifact, kernel, reflection,
                               invocationDiagnostic(*bundle.context)) ||
        !prepareCpuComputePipeline(*bundle.context, std::move(kernel), std::move(reflection), *state))
        return false;
    installRuntimeBackendState(pipeline, state.release());
    return true;
}

void destroyCpuPipeline(VernonLoadedPipeline &pipeline) {
    CpuPipelineState &state = runtimeBackendState<CpuPipelineState>(pipeline);
    vernonRuntimeCoreBindingsDestroy(state.bindings);
    vernonRuntimeCorePipelineDestroy(state.pipeline);
}

VernonStatus invokeCpuComputePipeline(VernonLoadedPipeline &pipeline, const PlannedComputeLaunch &launch) {
    CpuPipelineState &state = runtimeBackendState<CpuPipelineState>(pipeline);
    for (size_t index = 0; index < state.layout.size(); ++index) {
        const auto &layout = state.layout[index];
        auto &value = state.values[index];
        value = {};
        value.slot = layout.slot;
        value.kind = layout.kind;
        if (index < state.layoutBuiltins.size() && !state.layoutBuiltins[index].empty()) {
            const std::string &builtin = state.layoutBuiltins[index];
            if (builtin == VERNON_AD_TAPE_ALLOCATOR_BUILTIN) {
                if (!state.tapeAllocator)
                    return fail(*pipeline.context, "CPU tape dispatch has no allocator");
                value.payload.inline_value.data = &state.tapeAllocator;
                value.payload.inline_value.size = sizeof(state.tapeAllocator);
            } else if (builtin == VERNON_AD_TAPE_ROOT_REGION_BUILTIN) {
                if (!state.tapeRoot)
                    return fail(*pipeline.context, "CPU tape dispatch has no sealed root region");
                value.payload.inline_value.data = &state.tapeRoot;
                value.payload.inline_value.size = sizeof(state.tapeRoot);
            } else {
                return fail(*pipeline.context, "CPU tape dispatch has an unsupported builtin ABI");
            }
            if (value.payload.inline_value.size != layout.element_size)
                return fail(*pipeline.context, "CPU tape builtin size does not match its packed ABI");
            continue;
        }
        if (layout.argument_index >= launch.arguments.size())
            return fail(*pipeline.context, "CPU prepared argument index is invalid");
        const ComputeLaunchArgument &argument = launch.arguments[layout.argument_index];
        if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            const auto *tensor = std::get_if<ComputeTensorArgument>(&argument);
            if (!tensor || !tensor->hostData || !tensor->hostSize)
                return fail(*pipeline.context, "CPU storage binding requires a host Tensor");
            value.payload.buffer.resource.identity = cpuProviderResourceIdentity(*pipeline.context);
            value.payload.buffer.resource.resource.value = reinterpret_cast<uintptr_t>(tensor->hostData);
            value.payload.buffer.resource.size = tensor->hostSize;
        } else {
            const auto *scalar = std::get_if<ComputeScalarArgument>(&argument);
            const auto *tensor = std::get_if<ComputeTensorArgument>(&argument);
            if (scalar && scalar->data && scalar->size) {
                value.payload.inline_value.data = scalar->data;
                value.payload.inline_value.size = scalar->size;
            } else if (tensor && tensor->tensorViewData && tensor->tensorViewSize == layout.element_size) {
                value.payload.inline_value.data = tensor->tensorViewData;
                value.payload.inline_value.size = tensor->tensorViewSize;
            } else {
                return fail(*pipeline.context, "CPU inline binding requires host data");
            }
        }
    }
    const uint32_t groups[3]{launch.grid.x, launch.grid.y, launch.grid.z};
    if (hasTapeBuiltins(state))
        return dispatchCpuTapedCompute(*pipeline.context, state, groups);
    VernonStatus status =
        state.bindings ? vernonRuntimeCoreUpdateBindings(state.bindings, state.values.data(), state.values.size())
                       : vernonRuntimeCoreCreateBindings(state.pipeline, state.values.data(), state.values.size(),
                                                         &state.bindings);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError = cpuProviderLastError(*pipeline.context);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to prepare CPU provider bindings",
                    status);
    }
    status = vernonRuntimeCoreEncodeDispatch(state.pipeline, state.bindings, launch.commandEncoder, groups, nullptr, 0);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError = cpuProviderLastError(*pipeline.context);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to encode CPU provider dispatch",
                    status);
    }
    return VERNON_STATUS_OK;
}

} // namespace vernon::runtime
