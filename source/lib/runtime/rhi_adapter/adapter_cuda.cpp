#include "adapter_common.h"

#if defined(VERNON_HAS_CUDA_RHI)

#include "rhi/cuda_backend.h"
#include "rhi/rhi_internal.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

namespace vernon::runtime::rhi_adapter {
namespace {

using rhi::cuda::DevicePointer;
using rhi::cuda::DeviceState;
using rhi::cuda::PreparedFunction;

constexpr size_t kMaxGroups = VERNON_RUNTIME_PROVIDER_MAX_GROUPS;
// A storage buffer passes as the five fields of a memref descriptor; an inline value passes as itself.
constexpr size_t kMemRefParameters = 5;
// cuMemAlloc's alignment, which kernels may assume of every buffer they receive.
constexpr uint64_t kDeviceAllocationAlignment = 256;

struct CudaAdapterState {
    DeviceState *device{};
    std::atomic<uint64_t> nextSerial{};
};

CudaAdapterState &cudaState(VernonRuntimeRhiAdapter &adapter) {
    assert(adapter.rhiBackend == VERNON_RHI_BACKEND_CUDA);
    assert(adapter.backend.state);
    return *static_cast<CudaAdapterState *>(adapter.backend.state);
}

const CudaAdapterState &cudaState(const VernonRuntimeRhiAdapter &adapter) {
    assert(adapter.rhiBackend == VERNON_RHI_BACKEND_CUDA);
    assert(adapter.backend.state);
    return *static_cast<const CudaAdapterState *>(adapter.backend.state);
}

DeviceState &cudaDevice(VernonRuntimeRhiAdapter &adapter) { return *cudaState(adapter).device; }
const DeviceState &cudaDevice(const VernonRuntimeRhiAdapter &adapter) { return *cudaState(adapter).device; }

void destroyBackend(void *state) noexcept { delete static_cast<CudaAdapterState *>(state); }

RhiAdapterResult<void> synchronizeBackend(void *state, std::string &error) noexcept {
    try {
        const auto result = static_cast<CudaAdapterState *>(state)->device->synchronize();
        if (result == rhi::cuda::kSuccess)
            return RhiAdapterResult<void>{vernon::ok()};
        error = rhi::cuda::describeResult(result, "cuStreamSynchronize");
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "cuStreamSynchronize",
                               static_cast<uint64_t>(result));
    } catch (...) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "cuStreamSynchronize");
    }
}

uint64_t resourceIdentity(const void *state) noexcept {
    return reinterpret_cast<uintptr_t>(static_cast<const CudaAdapterState *>(state)->device);
}

const RhiAdapterBackendOps backendOps{destroyBackend, synchronizeBackend, resourceIdentity};

struct PreparedShader {
    std::vector<uint8_t> artifact;
    std::string entry;
};

// Retained by the groups and pipeline layouts made from it, since commands that used those may still be recording.
struct GroupLayout : RhiRetainedObject {
    std::vector<VernonRuntimeProviderBindingLayoutEntry> entries;
};

// One kernel parameter run: the entry of the group at a group index that supplies it.
struct KernelParameter {
    uint32_t group{};
    uint32_t entry{};
    size_t offset{};
};

// Kernel parameters follow entry bindings, which number the parameters of every group contiguously.
struct PreparedLayout : RhiRetainedObject {
    std::array<GroupLayout *, kMaxGroups> groups{};
    uint32_t groupCount{};
    std::array<std::vector<KernelParameter>, kMaxGroups> parameters;
    size_t parameterCount{};

    ~PreparedLayout() override {
        for (uint32_t group = 0; group < groupCount; ++group)
            releaseRetainedObject(groups[group]);
    }
};

struct PreparedPipeline : RhiRetainedObject {
    DeviceState *device{};
    PreparedLayout *layout{};
    PreparedFunction function;
    uint32_t workgroup[3]{1, 1, 1};

    ~PreparedPipeline() override {
        function.destroy(*device);
        releaseRetainedObject(layout);
    }
};

struct CudaGroup : RhiGroup {
    struct MemRefDescriptor {
        DevicePointer allocated{};
        DevicePointer aligned{};
        uint64_t offset{};
        uint64_t size{};
        uint64_t stride{1};
    };
    // Indexed by entry: the memref of a storage buffer and the bytes of an inline value.
    struct Value {
        MemRefDescriptor memref;
        std::vector<uint8_t> inlineStorage;
        DevicePointer hostStorage{};
    };
    DeviceState *device{};
    GroupLayout *layout{};
    uint64_t serial{};
    std::vector<Value> values;

    ~CudaGroup() override {
        for (const Value &value : values)
            if (value.hostStorage)
                device->free(value.hostStorage);
        releaseRetainedObject(layout);
    }
};

// Applied state holds group serials rather than pointers, because addresses are reused.
struct CudaCommands : RhiCommands {
    PreparedPipeline *pipeline{};
    std::array<CudaGroup *, kMaxGroups> groups{};
    // The kernel parameter pointers of `parameterLayout`, and the group serial each index filled in.
    const PreparedLayout *parameterLayout{};
    std::vector<void *> parameters;
    std::array<uint64_t, kMaxGroups> parameterGroups{};
};

RhiAdapterResult<void> cudaResult(rhi::cuda::Result result, const char *operation) {
    return result == rhi::cuda::kSuccess
               ? RhiAdapterResult<void>{vernon::ok()}
               : providerFailure(vernon::ProviderErrorCode::BackendFailure, operation, static_cast<uint64_t>(result));
}

uint32_t getCapabilities(void *) { return VERNON_RUNTIME_PROVIDER_COMPUTE; }

VernonRuntimeProviderDeviceIdentity getDeviceIdentity(void *data) {
    const auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    const auto &device = cudaDevice(adapter);
    return {0x43554441u, static_cast<uint64_t>(device.computeCapabilityMajor) << 32 | device.computeCapabilityMinor,
            device.driverVersion};
}

RhiAdapterResult<void> prepareShaderResult(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        descriptor->stage != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE || !descriptor->data || descriptor->size == 0 ||
        !descriptor->entry.data || descriptor->entry.size == 0)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "cuda_adapter_received_an_invalid_compute_shader_descriptor");
    try {
        auto shader = std::make_unique<PreparedShader>();
        const auto *bytes = static_cast<const uint8_t *>(descriptor->data);
        shader->artifact.assign(bytes, bytes + descriptor->size);
        shader->entry.assign(descriptor->entry.data, descriptor->entry.size);
        *output = toHandle(shader.release());
        adapter.shaderPreparations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "cuda_shader_preparation_ran_out_of_memory");
    }
}

RhiAdapterResult<void> prepareGroupLayoutResult(void *, const VernonRuntimeProviderGroupLayoutDescriptor *descriptor,
                                                VernonRuntimeProviderObject *output) {
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        (descriptor->entry_count != 0 && !descriptor->entries))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "cuda_adapter_received_an_invalid_group_layout");
    for (size_t index = 0; index < descriptor->entry_count; ++index) {
        const auto &entry = descriptor->entries[index];
        if ((entry.kind != VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
             entry.kind != VERNON_RUNTIME_PROVIDER_INLINE_VALUE) ||
            entry.stage_mask != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE || entry.array_count != 1 ||
            entry.element_size == 0)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "cuda_group_layout_contains_an_unsupported_binding", index);
    }
    try {
        auto layout = std::make_unique<GroupLayout>();
        layout->entries.assign(descriptor->entries, descriptor->entries + descriptor->entry_count);
        *output = toHandle(layout.release());
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                               "cuda_group_layout_preparation_ran_out_of_memory");
    }
}

RhiAdapterResult<void> prepareLayoutImplResult(VernonRuntimeRhiAdapter &adapter,
                                               const VernonRuntimeProviderPipelineLayoutDescriptor &descriptor,
                                               VernonRuntimeProviderObject *output) {
    auto layout = std::make_unique<PreparedLayout>();
    std::vector<KernelParameter *> ordered;
    for (size_t group = 0; group < descriptor.group_layout_count; ++group) {
        auto *groupLayout = fromHandle<GroupLayout>(descriptor.group_layouts[group]);
        if (!groupLayout)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "cuda_pipeline_layout_group_layout_is_invalid", group);
        groupLayout->references.fetch_add(1, std::memory_order_relaxed);
        layout->groups[group] = groupLayout;
        layout->groupCount = static_cast<uint32_t>(group + 1);
        auto &parameters = layout->parameters[group];
        parameters.resize(groupLayout->entries.size());
        for (uint32_t entry = 0; entry < parameters.size(); ++entry) {
            parameters[entry] = {static_cast<uint32_t>(group), entry, 0};
            ordered.push_back(&parameters[entry]);
        }
    }
    const auto binding = [&](const KernelParameter *parameter) {
        return layout->groups[parameter->group]->entries[parameter->entry].binding;
    };
    std::sort(ordered.begin(), ordered.end(), [&](const KernelParameter *left, const KernelParameter *right) {
        return binding(left) < binding(right);
    });
    for (size_t index = 0; index < ordered.size(); ++index) {
        if (binding(ordered[index]) != index)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "cuda_adapter_abi_bindings_must_be_contiguous", index);
        ordered[index]->offset = layout->parameterCount;
        const auto kind = layout->groups[ordered[index]->group]->entries[ordered[index]->entry].kind;
        layout->parameterCount += kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ? kMemRefParameters : 1;
    }
    *output = toHandle(layout.release());
    adapter.layoutPreparations.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> prepareLayoutResult(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                                           VernonRuntimeProviderObject *output) {
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        descriptor->group_layout_count > kMaxGroups ||
        (descriptor->group_layout_count != 0 && !descriptor->group_layouts) || descriptor->vertex_attribute_count != 0)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "cuda_adapter_received_an_invalid_pipeline_layout");
    if (descriptor->push_constant_size != 0)
        return providerFailure(vernon::ProviderErrorCode::Unsupported, "cuda_dispatch_push_constants_are_unsupported");
    try {
        return prepareLayoutImplResult(*static_cast<VernonRuntimeRhiAdapter *>(data), *descriptor, output);
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "cuda_layout_preparation_ran_out_of_memory");
    }
}

RhiAdapterResult<void> preparePipelineResult(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                                             VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        descriptor->kind != VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE || descriptor->shader_count != 1 ||
        !descriptor->shaders || !fromHandle<PreparedShader>(descriptor->shaders[0]) || !layout ||
        descriptor->workgroup_size[0] == 0 || descriptor->workgroup_size[1] == 0 || descriptor->workgroup_size[2] == 0)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "cuda_adapter_received_an_invalid_compute_pipeline");
    try {
        auto pipeline = std::make_unique<PreparedPipeline>();
        PreparedShader &shader = *fromHandle<PreparedShader>(descriptor->shaders[0]);
        auto &device = cudaDevice(adapter);
        pipeline->device = &device;
        layout->references.fetch_add(1, std::memory_order_relaxed);
        pipeline->layout = layout;
        auto created = cudaResult(
            pipeline->function.create(device, shader.artifact.data(), shader.artifact.size(), shader.entry.c_str()),
            "CUDA pipeline preparation");
        if (!created)
            return created;
        std::copy_n(descriptor->workgroup_size, 3, pipeline->workgroup);
        *output = toHandle(pipeline.release());
        adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                               "cuda_pipeline_preparation_ran_out_of_memory");
    }
}

// Host values bound as storage need device memory: a transient group's comes from its encoder's transient memory, and a
// persistent group allocates its own.
RhiAdapterResult<DevicePointer> placeHostStorage(VernonRuntimeRhiAdapter &adapter, CudaGroup &group,
                                                 const VernonRuntimeProviderGroupDescriptor &descriptor,
                                                 const VernonRuntimeProviderInlineValue &host,
                                                 CudaGroup::Value &value) {
    if (descriptor.lifetime == VERNON_RUNTIME_PROVIDER_GROUP_TRANSIENT) {
        auto written = writeCommandTransient(adapter, descriptor.command_encoder, host.data, host.size,
                                             kDeviceAllocationAlignment);
        if (!written)
            return RhiAdapterResult<DevicePointer>{vernon::err(std::move(written).error())};
        return RhiAdapterResult<DevicePointer>{
            vernon::ok(static_cast<DevicePointer>(written.value().buffer + written.value().offset))};
    }
    if (auto allocated =
            cudaResult(group.device->allocate(value.hostStorage, host.size), "cuMemAlloc host storage binding");
        !allocated)
        return RhiAdapterResult<DevicePointer>{vernon::err(std::move(allocated).error())};
    if (auto uploaded = cudaResult(group.device->upload(value.hostStorage, host.data, host.size),
                                   "cuMemcpyHtoDAsync host storage binding");
        !uploaded)
        return RhiAdapterResult<DevicePointer>{vernon::err(std::move(uploaded).error())};
    return RhiAdapterResult<DevicePointer>{vernon::ok(value.hostStorage)};
}

RhiAdapterResult<void> initializeGroupValue(VernonRuntimeRhiAdapter &adapter, CudaGroup &group,
                                            const VernonRuntimeProviderGroupDescriptor &descriptor,
                                            const VernonRuntimeProviderBindingLayoutEntry &entry,
                                            const VernonRuntimeProviderBindingValue &source, CudaGroup::Value &value) {
    if (source.kind != entry.kind)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "cuda_group_value_kind_is_invalid",
                               entry.slot);
    if (entry.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE) {
        if (!source.payload.inline_value.data || source.payload.inline_value.size == 0)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "cuda_inline_binding_requires_initial_storage", entry.slot);
        const auto *bytes = static_cast<const uint8_t *>(source.payload.inline_value.data);
        value.inlineStorage.assign(bytes, bytes + source.payload.inline_value.size);
        return RhiAdapterResult<void>{vernon::ok()};
    }
    if ((source.flags & ~VERNON_RUNTIME_PROVIDER_BINDING_HOST_STORAGE) != 0)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "cuda_storage_buffer_flags_are_invalid",
                               entry.slot);
    DevicePointer pointer{};
    uint64_t byteSize = 0;
    uint64_t byteStride = entry.element_size;
    if ((source.flags & VERNON_RUNTIME_PROVIDER_BINDING_HOST_STORAGE) != 0) {
        const auto &inlineValue = source.payload.inline_value;
        if (!inlineValue.data || !inlineValue.size || inlineValue.size % entry.element_size != 0)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "cuda_host_storage_binding_is_invalid",
                                   inlineValue.size, entry.element_size);
        auto placed = placeHostStorage(adapter, group, descriptor, inlineValue, value);
        if (!placed)
            return RhiAdapterResult<void>{vernon::err(std::move(placed).error())};
        pointer = placed.value();
        byteSize = inlineValue.size;
    } else {
        const auto &reference = source.payload.buffer.resource;
        if ((reference.identity & kRhiResourceKindMask) != kRhiBufferResource || reference.size == 0)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "cuda_storage_buffer_binding_is_invalid",
                                   entry.slot);
        auto acquired = acquireGroupResource(adapter, group, reference);
        if (!acquired)
            return RhiAdapterResult<void>{vernon::err(std::move(acquired).error())};
        if (!acquired.value())
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "cuda_storage_buffer_binding_is_stale",
                                   entry.slot);
        if (writesBinding(entry))
            group.writes.push_back(reference);
        pointer = static_cast<DevicePointer>(acquired.value()) + reference.offset;
        byteSize = reference.size;
        if (source.payload.buffer.stride)
            byteStride = source.payload.buffer.stride;
    }
    if (byteStride < entry.element_size || byteStride % entry.element_size != 0)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "cuda_storage_buffer_stride_is_invalid",
                               byteStride, entry.element_size);
    value.memref = {pointer, pointer, 0, byteSize / byteStride, byteStride / entry.element_size};
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> createGroupImplResult(VernonRuntimeRhiAdapter &adapter,
                                             const VernonRuntimeProviderGroupDescriptor &descriptor,
                                             VernonRuntimeProviderObject *output) {
    auto *layout = fromHandle<GroupLayout>(descriptor.group_layout);
    if (!layout || descriptor.value_count != layout->entries.size() ||
        (descriptor.value_count != 0 && !descriptor.values))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "cuda_adapter_received_an_invalid_group_descriptor");
    auto group = std::make_unique<CudaGroup>();
    group->device = &cudaDevice(adapter);
    layout->references.fetch_add(1, std::memory_order_relaxed);
    group->layout = layout;
    group->values.resize(layout->entries.size());
    for (size_t index = 0; index < layout->entries.size(); ++index)
        if (auto initialized = initializeGroupValue(adapter, *group, descriptor, layout->entries[index],
                                                    descriptor.values[index], group->values[index]);
            !initialized)
            return initialized;
    group->serial = cudaState(adapter).nextSerial.fetch_add(1, std::memory_order_relaxed) + 1;
    *output = toHandle(group.release());
    adapter.groupCreations.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> createGroupResult(void *data, const VernonRuntimeProviderGroupDescriptor *descriptor,
                                         VernonRuntimeProviderObject *output) {
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "cuda_adapter_received_an_invalid_group_descriptor");
    try {
        return createGroupImplResult(*static_cast<VernonRuntimeRhiAdapter *>(data), *descriptor, output);
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "cuda_group_creation_ran_out_of_memory");
    }
}

RhiAdapterResult<void> beginCommandsResult(void *data, VernonRuntimeProviderObject encoder,
                                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!output)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "cuda_begin_commands_output_is_null");
    std::unique_ptr<CudaCommands> commands(new (std::nothrow) CudaCommands());
    if (!commands)
        return providerFailure(vernon::ProviderErrorCode::ResourceExhausted, "cuda_commands_allocation");
    if (auto begun = beginRhiCommands(adapter, encoder, *commands); !begun)
        return begun;
    if (commands->native != reinterpret_cast<uintptr_t>(&cudaDevice(adapter)))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "cuda_command_encoder_is_invalid");
    *output = toHandle(commands.release());
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> endCommandsResult(void *, VernonRuntimeProviderObject handle) {
    std::unique_ptr<CudaCommands> commands(fromHandle<CudaCommands>(handle));
    return commands
               ? RhiAdapterResult<void>{vernon::ok()}
               : providerFailure(vernon::ProviderErrorCode::InvalidArgument, "cuda_end_commands_handle_is_invalid");
}

RhiAdapterResult<void> bindPipelineResult(void *data, VernonRuntimeProviderObject handle,
                                          VernonRuntimeProviderObject pipelineHandle) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<CudaCommands>(handle);
    auto *pipeline = fromHandle<PreparedPipeline>(pipelineHandle);
    if (!commands || !pipeline)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "cuda_bind_pipeline_is_invalid");
    if (auto retained = retainForCommands(adapter, commands->encoder, *pipeline); !retained)
        return retained;
    const PreparedLayout &layout = *pipeline->layout;
    for (uint32_t group = 0; group < kMaxGroups; ++group)
        if (commands->groups[group] &&
            (group >= layout.groupCount || commands->groups[group]->layout != layout.groups[group]))
            commands->groups[group] = nullptr;
    commands->pipeline = pipeline;
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> bindGroupResult(void *data, VernonRuntimeProviderObject handle, uint32_t index,
                                       VernonRuntimeProviderObject groupHandle) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<CudaCommands>(handle);
    auto *group = fromHandle<CudaGroup>(groupHandle);
    if (!commands || !group || !commands->pipeline || index >= commands->pipeline->layout->groupCount ||
        group->layout != commands->pipeline->layout->groups[index])
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "cuda_bind_group_is_invalid", index);
    if (auto retained = retainForCommands(adapter, commands->encoder, *group); !retained)
        return retained;
    commands->groups[index] = group;
    return RhiAdapterResult<void>{vernon::ok()};
}

// Points the kernel parameters at the values of the bound groups whose parameters the array lacks.
RhiAdapterResult<void> fillParameters(CudaCommands &commands) {
    const PreparedLayout &layout = *commands.pipeline->layout;
    if (commands.parameterLayout != &layout) {
        commands.parameters.assign(layout.parameterCount, nullptr);
        commands.parameterLayout = &layout;
        commands.parameterGroups.fill(0);
    }
    for (uint32_t index = 0; index < layout.groupCount; ++index) {
        CudaGroup *group = commands.groups[index];
        if (!group)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "cuda_command_group_is_unbound", index);
        if (commands.parameterGroups[index] == group->serial)
            continue;
        for (const KernelParameter &parameter : layout.parameters[index]) {
            CudaGroup::Value &value = group->values[parameter.entry];
            void **target = commands.parameters.data() + parameter.offset;
            if (group->layout->entries[parameter.entry].kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
                target[0] = &value.memref.allocated;
                target[1] = &value.memref.aligned;
                target[2] = &value.memref.offset;
                target[3] = &value.memref.size;
                target[4] = &value.memref.stride;
            } else {
                target[0] = value.inlineStorage.data();
            }
        }
        commands.parameterGroups[index] = group->serial;
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> encodeDispatchResult(void *data, VernonRuntimeProviderObject handle,
                                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<CudaCommands>(handle);
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !commands || !commands->pipeline ||
        commands->rendering || descriptor->push_constant_size != 0)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "cuda_adapter_received_an_invalid_dispatch");
    if (auto filled = fillParameters(*commands); !filled)
        return filled;
    PreparedPipeline &pipeline = *commands->pipeline;
    if (auto launched = cudaResult(pipeline.function.launch(*pipeline.device, descriptor->group_count,
                                                            pipeline.workgroup, commands->parameters.data()),
                                   "cuLaunchKernel");
        !launched)
        return launched;
    for (uint32_t index = 0; index < pipeline.layout->groupCount; ++index)
        if (!commands->groups[index]->writes.empty())
            if (auto recorded = recordGroupWrites(adapter, commands->encoder, *commands->groups[index]); !recorded)
                return recorded;
    if (auto recorded = recordProviderCommand(adapter, commands->encoder, false); !recorded)
        return recorded;
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

void destroyShader(void *, VernonRuntimeProviderObject handle) { delete fromHandle<PreparedShader>(handle); }
void destroyGroupLayout(void *, VernonRuntimeProviderObject handle) {
    releaseRetainedObject(fromHandle<GroupLayout>(handle));
}
void destroyLayout(void *, VernonRuntimeProviderObject handle) {
    releaseRetainedObject(fromHandle<PreparedLayout>(handle));
}
void destroyPipeline(void *, VernonRuntimeProviderObject handle) {
    releaseRetainedObject(fromHandle<PreparedPipeline>(handle));
}
void destroyGroup(void *, VernonRuntimeProviderObject handle) { releaseRetainedObject(fromHandle<CudaGroup>(handle)); }

VernonStatus prepareShader(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, prepareShaderResult(data, descriptor, output), "CUDA shader preparation failed");
}
VernonStatus prepareGroupLayout(void *data, const VernonRuntimeProviderGroupLayoutDescriptor *descriptor,
                                VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, prepareGroupLayoutResult(data, descriptor, output),
                          "CUDA group layout preparation failed");
}
VernonStatus prepareLayout(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, prepareLayoutResult(data, descriptor, output), "CUDA layout preparation failed");
}
VernonStatus preparePipeline(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                             VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, preparePipelineResult(data, descriptor, output), "CUDA pipeline preparation failed");
}
VernonStatus createGroup(void *data, const VernonRuntimeProviderGroupDescriptor *descriptor,
                         VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, createGroupResult(data, descriptor, output), "CUDA group creation failed");
}
VernonStatus beginCommands(void *data, VernonRuntimeProviderObject encoder, VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, beginCommandsResult(data, encoder, output), "CUDA commands could not begin");
}
VernonStatus endCommands(void *data, VernonRuntimeProviderObject commands) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, endCommandsResult(data, commands), "CUDA commands failed");
}
VernonStatus bindPipeline(void *data, VernonRuntimeProviderObject commands, VernonRuntimeProviderObject pipeline) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, bindPipelineResult(data, commands, pipeline), "CUDA pipeline binding failed");
}
VernonStatus bindGroup(void *data, VernonRuntimeProviderObject commands, uint32_t index,
                       VernonRuntimeProviderObject group) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, bindGroupResult(data, commands, index, group), "CUDA group binding failed");
}
VernonStatus encodeDispatch(void *data, VernonRuntimeProviderObject commands,
                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, encodeDispatchResult(data, commands, descriptor), "CUDA dispatch encoding failed");
}

} // namespace

void initializeCudaProvider(VernonRuntimeRhiAdapter &adapter) {
    adapter.provider.struct_size = sizeof(adapter.provider);
    adapter.provider.abi_version = VERNON_PROGRAM_VERSION;
    adapter.provider.user_data = &adapter;
    adapter.provider.get_capabilities = getCapabilities;
    adapter.provider.get_device_identity = getDeviceIdentity;
    adapter.provider.prepare_shader = prepareShader;
    adapter.provider.prepare_group_layout = prepareGroupLayout;
    adapter.provider.prepare_pipeline_layout = prepareLayout;
    adapter.provider.prepare_pipeline = preparePipeline;
    adapter.provider.create_group = createGroup;
    adapter.provider.begin_commands = beginCommands;
    adapter.provider.end_commands = endCommands;
    adapter.provider.encode_bind_pipeline = bindPipeline;
    adapter.provider.encode_bind_group = bindGroup;
    adapter.provider.encode_dispatch = encodeDispatch;
    adapter.provider.destroy_shader = destroyShader;
    adapter.provider.destroy_group_layout = destroyGroupLayout;
    adapter.provider.destroy_pipeline_layout = destroyLayout;
    adapter.provider.destroy_pipeline = destroyPipeline;
    adapter.provider.destroy_group = destroyGroup;
}

} // namespace vernon::runtime::rhi_adapter

namespace vernon::runtime {

namespace {

VernonRuntimeRhiAdapter *createCudaAdapter(std::unique_ptr<rhi_adapter::CudaAdapterState> state) {
    auto adapter = std::unique_ptr<VernonRuntimeRhiAdapter>(new (std::nothrow) VernonRuntimeRhiAdapter());
    if (!adapter || !state || !state->device)
        return nullptr;
    adapter->rhiBackend = VERNON_RHI_BACKEND_CUDA;
    if (!adapter->backend.adopt(state.get(), &rhi_adapter::backendOps))
        return nullptr;
    [[maybe_unused]] auto *adoptedState = state.release();
    rhi_adapter::initializeCudaProvider(*adapter);
    return adapter.release();
}

} // namespace

VernonRuntimeRhiAdapter *createCudaRhiAdapter(VernonRhiDevice device, VernonRhiBackend backend) {
    if (backend != VERNON_RHI_BACKEND_CUDA)
        return nullptr;
    auto resolvedDeviceState = rhi::deviceState(device, backend);
    if (resolvedDeviceState.isErr())
        return nullptr;
    auto *deviceState = static_cast<rhi::cuda::DeviceState *>(resolvedDeviceState.value());
    auto state = std::unique_ptr<rhi_adapter::CudaAdapterState>(new (std::nothrow) rhi_adapter::CudaAdapterState());
    if (!state)
        return nullptr;
    state->device = deviceState;
    return createCudaAdapter(std::move(state));
}

} // namespace vernon::runtime

#endif
