#include "adapter_common.h"

#if defined(VERNON_HAS_CUDA_RHI)

#include <algorithm>
#include <cstring>
#include <mutex>
#include <new>
#include <vector>

namespace vernon::runtime::rhi_adapter {
namespace {

using rhi::cuda::DevicePointer;
using rhi::cuda::DeviceState;
using rhi::cuda::PreparedFunction;

struct PreparedShader {
    std::vector<uint8_t> artifact;
    std::string entry;
};

struct PreparedLayout {
    std::vector<VernonRuntimeProviderBindingLayoutEntry> entries;
};

struct PreparedPipeline {
    PreparedFunction function;
    DeviceState *device{};
    uint32_t workgroup[3]{1, 1, 1};
};

struct PreparedBindingSet {
    struct MemRefDescriptor {
        DevicePointer allocated{};
        DevicePointer aligned{};
        uint64_t offset{};
        uint64_t size{};
        uint64_t stride{1};
    };
    struct Slot {
        VernonRuntimeProviderBindingLayoutEntry layout{};
        size_t parameterOffset{};
        size_t descriptorOffset{};
        std::vector<uint8_t> inlineStorage;
    };
    std::vector<Slot> slots;
    std::vector<MemRefDescriptor> descriptors;
    std::vector<void *> parameters;
    std::mutex mutex;
};

VernonStatus cudaStatus(VernonRuntimeRhiAdapter &adapter, rhi::cuda::Result result, const char *operation) {
    return result == rhi::cuda::kSuccess
               ? VERNON_STATUS_OK
               : fail(adapter, rhi::cuda::describeResult(result, operation), VERNON_STATUS_INTERNAL_ERROR);
}

uint32_t getCapabilities(void *) { return VERNON_RUNTIME_PROVIDER_COMPUTE; }

VernonRuntimeProviderDeviceIdentity getDeviceIdentity(void *data) {
    const auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return {0x43554441u,
            static_cast<uint64_t>(adapter.device->computeCapabilityMajor) << 32 |
                adapter.device->computeCapabilityMinor,
            adapter.device->driverVersion};
}

VernonStatus prepareShader(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        descriptor->stage != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE || !descriptor->data || descriptor->size == 0 ||
        !descriptor->entry.data || descriptor->entry.size == 0)
        return fail(adapter, "CUDA adapter received an invalid compute shader descriptor");
    try {
        auto shader = std::make_unique<PreparedShader>();
        const auto *bytes = static_cast<const uint8_t *>(descriptor->data);
        shader->artifact.assign(bytes, bytes + descriptor->size);
        shader->entry.assign(descriptor->entry.data, descriptor->entry.size);
        *output = toHandle(shader.release());
        adapter.shaderPreparations.fetch_add(1, std::memory_order_relaxed);
        return VERNON_STATUS_OK;
    } catch (const std::bad_alloc &) {
        return fail(adapter, "CUDA shader preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    }
}

VernonStatus prepareLayout(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        (descriptor->binding_count != 0 && !descriptor->bindings))
        return fail(adapter, "CUDA adapter received an invalid pipeline layout");
    try {
        auto layout = std::make_unique<PreparedLayout>();
        layout->entries.assign(descriptor->bindings, descriptor->bindings + descriptor->binding_count);
        for (const auto &entry : layout->entries)
            if ((entry.kind != VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
                 entry.kind != VERNON_RUNTIME_PROVIDER_INLINE_VALUE) ||
                entry.stage_mask != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE || entry.array_count != 1 ||
                (entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER && entry.element_size == 0))
                return fail(adapter, "CUDA adapter pipeline layout contains an unsupported binding");
        std::sort(layout->entries.begin(), layout->entries.end(),
                  [](const auto &left, const auto &right) { return left.binding < right.binding; });
        for (size_t index = 0; index < layout->entries.size(); ++index)
            if (layout->entries[index].binding != index)
                return fail(adapter, "CUDA adapter ABI bindings must be contiguous");
        *output = toHandle(layout.release());
        adapter.layoutPreparations.fetch_add(1, std::memory_order_relaxed);
        return VERNON_STATUS_OK;
    } catch (const std::bad_alloc &) {
        return fail(adapter, "CUDA layout preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    }
}

VernonStatus preparePipeline(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                             VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        descriptor->kind != VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE || descriptor->shader_count != 1 ||
        !descriptor->shaders || !fromHandle<PreparedShader>(descriptor->shaders[0]) ||
        !fromHandle<PreparedLayout>(descriptor->layout) || descriptor->workgroup_size[0] == 0 ||
        descriptor->workgroup_size[1] == 0 || descriptor->workgroup_size[2] == 0)
        return fail(adapter, "CUDA adapter received an invalid compute pipeline");
    try {
        auto pipeline = std::make_unique<PreparedPipeline>();
        PreparedShader &shader = *fromHandle<PreparedShader>(descriptor->shaders[0]);
        const VernonStatus status = cudaStatus(adapter,
                                               pipeline->function.create(*adapter.device, shader.artifact.data(),
                                                                         shader.artifact.size(), shader.entry.c_str()),
                                               "CUDA pipeline preparation");
        if (status != VERNON_STATUS_OK)
            return status;
        pipeline->device = adapter.device;
        std::copy_n(descriptor->workgroup_size, 3, pipeline->workgroup);
        *output = toHandle(pipeline.release());
        adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
        return VERNON_STATUS_OK;
    } catch (const std::bad_alloc &) {
        return fail(adapter, "CUDA pipeline preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    }
}

VernonStatus retainResource(void *data, VernonRuntimeProviderResourceReference resource) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return resource.identity != 0 && resource.resource.value != 0
               ? VERNON_STATUS_OK
               : fail(adapter, "CUDA adapter received an invalid resource reference");
}

void releaseResource(void *, VernonRuntimeProviderResourceReference) {}

VernonStatus updateBindingsImpl(VernonRuntimeRhiAdapter &adapter, PreparedBindingSet &bindings,
                                const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    if (valueCount != bindings.slots.size() || (valueCount != 0 && !values))
        return fail(adapter, "CUDA binding values do not match the prepared layout");
    for (PreparedBindingSet::Slot &slot : bindings.slots) {
        const auto value = std::find_if(values, values + valueCount,
                                        [&slot](const auto &candidate) { return candidate.slot == slot.layout.slot; });
        if (value == values + valueCount || value->kind != slot.layout.kind)
            return fail(adapter, "CUDA binding slot or kind does not match the prepared layout");
        if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            if (!value->resource.resource.value || value->resource.size == 0)
                return fail(adapter, "CUDA storage-buffer binding is invalid");
            const DevicePointer pointer = value->resource.resource.value + value->resource.offset;
            bindings.descriptors[slot.descriptorOffset] = {pointer, pointer, 0,
                                                           value->resource.size / slot.layout.element_size, 1};
        } else {
            if (!value->inline_data || value->inline_size != slot.inlineStorage.size())
                return fail(adapter, "CUDA inline binding size changed after preparation");
            std::memcpy(slot.inlineStorage.data(), value->inline_data, value->inline_size);
        }
    }
    return VERNON_STATUS_OK;
}

VernonStatus createBindingSet(void *data, const VernonRuntimeProviderBindingSetDescriptor *descriptor,
                              VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    PreparedLayout *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !layout)
        return fail(adapter, "CUDA adapter received an invalid binding-set descriptor");
    try {
        auto bindings = std::make_unique<PreparedBindingSet>();
        size_t descriptorCount = 0;
        size_t parameterCount = 0;
        bindings->slots.reserve(layout->entries.size());
        for (const auto &entry : layout->entries) {
            PreparedBindingSet::Slot slot;
            slot.layout = entry;
            slot.parameterOffset = parameterCount;
            if (entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
                slot.descriptorOffset = descriptorCount++;
                parameterCount += 5;
            } else {
                const auto value =
                    std::find_if(descriptor->values, descriptor->values + descriptor->value_count,
                                 [&entry](const auto &candidate) { return candidate.slot == entry.slot; });
                if (value == descriptor->values + descriptor->value_count || !value->inline_data ||
                    value->inline_size == 0)
                    return fail(adapter, "CUDA inline binding requires initial storage");
                slot.inlineStorage.resize(value->inline_size);
                ++parameterCount;
            }
            bindings->slots.push_back(std::move(slot));
        }
        bindings->descriptors.resize(descriptorCount);
        bindings->parameters.resize(parameterCount);
        for (PreparedBindingSet::Slot &slot : bindings->slots)
            if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
                auto &memref = bindings->descriptors[slot.descriptorOffset];
                bindings->parameters[slot.parameterOffset] = &memref.allocated;
                bindings->parameters[slot.parameterOffset + 1] = &memref.aligned;
                bindings->parameters[slot.parameterOffset + 2] = &memref.offset;
                bindings->parameters[slot.parameterOffset + 3] = &memref.size;
                bindings->parameters[slot.parameterOffset + 4] = &memref.stride;
            } else {
                bindings->parameters[slot.parameterOffset] = slot.inlineStorage.data();
            }
        const VernonStatus status = updateBindingsImpl(adapter, *bindings, descriptor->values, descriptor->value_count);
        if (status != VERNON_STATUS_OK)
            return status;
        *output = toHandle(bindings.release());
        adapter.bindingCreations.fetch_add(1, std::memory_order_relaxed);
        return VERNON_STATUS_OK;
    } catch (const std::bad_alloc &) {
        return fail(adapter, "CUDA binding preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    }
}

VernonStatus updateBindingSet(void *data, VernonRuntimeProviderObject handle,
                              const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    PreparedBindingSet *bindings = fromHandle<PreparedBindingSet>(handle);
    if (!bindings)
        return fail(adapter, "CUDA adapter received an invalid binding set");
    std::lock_guard<std::mutex> guard(bindings->mutex);
    return updateBindingsImpl(adapter, *bindings, values, valueCount);
}

VernonStatus encodeDispatch(void *data, VernonRuntimeProviderObject,
                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    PreparedPipeline *pipeline = descriptor ? fromHandle<PreparedPipeline>(descriptor->pipeline) : nullptr;
    PreparedBindingSet *bindings = descriptor ? fromHandle<PreparedBindingSet>(descriptor->bindings) : nullptr;
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !pipeline ||
        (!bindings && descriptor->bindings.value != 0))
        return fail(adapter, "CUDA adapter received an invalid dispatch");
    std::unique_lock<std::mutex> guard;
    if (bindings)
        guard = std::unique_lock<std::mutex>(bindings->mutex);
    const VernonStatus status =
        cudaStatus(adapter,
                   pipeline->function.launch(*pipeline->device, descriptor->group_count, pipeline->workgroup,
                                             bindings ? bindings->parameters.data() : nullptr),
                   "cuLaunchKernel");
    if (status == VERNON_STATUS_OK)
        adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return status;
}

VernonStatus encodeDraw(void *data, VernonRuntimeProviderObject, const VernonRuntimeProviderDrawDescriptor *) {
    return fail(*static_cast<VernonRuntimeRhiAdapter *>(data), "CUDA does not support graphics",
                VERNON_STATUS_UNSUPPORTED_TARGET);
}

void destroyShader(void *, VernonRuntimeProviderObject handle) { delete fromHandle<PreparedShader>(handle); }
void destroyLayout(void *, VernonRuntimeProviderObject handle) { delete fromHandle<PreparedLayout>(handle); }
void destroyBindingSet(void *, VernonRuntimeProviderObject handle) { delete fromHandle<PreparedBindingSet>(handle); }
void destroyPipeline(void *, VernonRuntimeProviderObject handle) {
    auto *pipeline = fromHandle<PreparedPipeline>(handle);
    if (!pipeline)
        return;
    pipeline->function.destroy(*pipeline->device);
    delete pipeline;
}

} // namespace

void initializeCudaProvider(VernonRuntimeRhiAdapter &adapter) {
    adapter.provider.struct_size = sizeof(adapter.provider);
    adapter.provider.abi_version = VERNON_RUNTIME_DEVICE_PROVIDER_ABI_VERSION;
    adapter.provider.user_data = &adapter;
    adapter.provider.get_capabilities = getCapabilities;
    adapter.provider.get_device_identity = getDeviceIdentity;
    adapter.provider.prepare_shader = prepareShader;
    adapter.provider.prepare_pipeline_layout = prepareLayout;
    adapter.provider.prepare_pipeline = preparePipeline;
    adapter.provider.retain_resource = retainResource;
    adapter.provider.release_resource = releaseResource;
    adapter.provider.create_binding_set = createBindingSet;
    adapter.provider.update_binding_set = updateBindingSet;
    adapter.provider.encode_dispatch = encodeDispatch;
    adapter.provider.encode_draw = encodeDraw;
    adapter.provider.destroy_shader = destroyShader;
    adapter.provider.destroy_pipeline_layout = destroyLayout;
    adapter.provider.destroy_pipeline = destroyPipeline;
    adapter.provider.destroy_binding_set = destroyBindingSet;
}

} // namespace vernon::runtime::rhi_adapter

#endif
