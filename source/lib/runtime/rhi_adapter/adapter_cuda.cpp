#include "adapter_common.h"

#if defined(VERNON_HAS_CUDA_RHI)

#include <algorithm>
#include <cstring>
#include <mutex>
#include <new>
#include <unordered_map>
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
    std::atomic<uint32_t> references{1};
    PreparedFunction function;
    DeviceState *device{};
    uint32_t workgroup[3]{1, 1, 1};
};

struct PreparedBindingSet {
    std::atomic<uint32_t> references{1};
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
        VernonRuntimeProviderResourceReference resourceReference{};
    };
    std::vector<Slot> slots;
    std::unordered_map<uint32_t, size_t> slotIndices;
    std::vector<size_t> valueIndices;
    std::vector<uint8_t> seenSlots;
    std::vector<DevicePointer> resolvedValues;
    std::vector<MemRefDescriptor> descriptors;
    std::vector<void *> parameters;
    std::mutex mutex;
};

void releaseCommandBindings(void *context, uint64_t);
void releaseCommandPipeline(void *context, uint64_t);

bool retainCommandObjects(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                          PreparedPipeline &pipeline, PreparedBindingSet *bindings) {
    pipeline.references.fetch_add(1, std::memory_order_relaxed);
    if (!deferCommandCleanup(adapter, encoder, &pipeline, 0, releaseCommandPipeline)) {
        releaseCommandPipeline(&pipeline, 0);
        return false;
    }
    if (bindings) {
        bindings->references.fetch_add(1, std::memory_order_relaxed);
        if (!deferCommandCleanup(adapter, encoder, bindings, 0, releaseCommandBindings)) {
            releaseCommandBindings(bindings, 0);
            return false;
        }
    }
    return true;
}

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
    return retainRhiResource(adapter, resource) ? VERNON_STATUS_OK
                                                : fail(adapter, "CUDA adapter received a stale resource");
}

void releaseResource(void *data, VernonRuntimeProviderResourceReference resource) {
    releaseRhiResource(*static_cast<VernonRuntimeRhiAdapter *>(data), resource);
}

VernonStatus updateBindingsImpl(VernonRuntimeRhiAdapter &adapter, PreparedBindingSet &bindings,
                                const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    if (valueCount != bindings.slots.size() || (valueCount != 0 && !values))
        return fail(adapter, "CUDA binding values do not match the prepared layout");
    std::fill(bindings.seenSlots.begin(), bindings.seenSlots.end(), uint8_t{0});
    for (size_t index = 0; index < valueCount; ++index) {
        const auto found = bindings.slotIndices.find(values[index].slot);
        if (found == bindings.slotIndices.end() || bindings.seenSlots[found->second])
            return fail(adapter, "CUDA binding slot is invalid or duplicated");
        bindings.seenSlots[found->second] = 1;
        bindings.valueIndices[found->second] = index;
    }
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        const auto &slot = bindings.slots[index];
        const auto *value = &values[bindings.valueIndices[index]];
        if (value->kind != slot.layout.kind)
            return fail(adapter, "CUDA binding slot or kind does not match the prepared layout");
        if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            const DevicePointer resource = resolveRhiResource(adapter, value->resource);
            if (!resource || value->resource.size == 0)
                return fail(adapter, "CUDA storage-buffer binding is invalid");
            bindings.resolvedValues[index] = resource + value->resource.offset;
        } else if (!value->inline_data || value->inline_size != slot.inlineStorage.size())
            return fail(adapter, "CUDA inline binding size changed after preparation");
    }
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        auto &slot = bindings.slots[index];
        const auto *value = &values[bindings.valueIndices[index]];
        slot.resourceReference = {};
        if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            const DevicePointer pointer = bindings.resolvedValues[index];
            bindings.descriptors[slot.descriptorOffset] = {pointer, pointer, 0,
                                                           value->resource.size / slot.layout.element_size, 1};
            slot.resourceReference = value->resource;
        } else
            std::memcpy(slot.inlineStorage.data(), value->inline_data, value->inline_size);
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
        bindings->slotIndices.reserve(layout->entries.size());
        bindings->valueIndices.resize(layout->entries.size());
        bindings->seenSlots.resize(layout->entries.size());
        bindings->resolvedValues.resize(layout->entries.size());
        for (size_t index = 0; index < layout->entries.size(); ++index) {
            const auto &entry = layout->entries[index];
            PreparedBindingSet::Slot slot;
            slot.layout = entry;
            if (!bindings->slotIndices.emplace(entry.slot, index).second)
                return fail(adapter, "CUDA binding layout contains duplicate slots");
            slot.parameterOffset = parameterCount;
            if (entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
                slot.descriptorOffset = descriptorCount++;
                parameterCount += 5;
            } else
                ++parameterCount;
            bindings->slots.push_back(std::move(slot));
        }
        if (descriptor->value_count != bindings->slots.size() || (descriptor->value_count != 0 && !descriptor->values))
            return fail(adapter, "CUDA binding values do not match the prepared layout");
        std::fill(bindings->seenSlots.begin(), bindings->seenSlots.end(), uint8_t{0});
        for (size_t index = 0; index < descriptor->value_count; ++index) {
            const auto found = bindings->slotIndices.find(descriptor->values[index].slot);
            if (found == bindings->slotIndices.end() || bindings->seenSlots[found->second])
                return fail(adapter, "CUDA binding slot is invalid or duplicated");
            bindings->seenSlots[found->second] = 1;
            bindings->valueIndices[found->second] = index;
        }
        for (size_t index = 0; index < bindings->slots.size(); ++index) {
            auto &slot = bindings->slots[index];
            if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER)
                continue;
            const auto &value = descriptor->values[bindings->valueIndices[index]];
            if (!value.inline_data || value.inline_size == 0)
                return fail(adapter, "CUDA inline binding requires initial storage");
            slot.inlineStorage.resize(value.inline_size);
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

VernonStatus encodeDispatch(void *data, VernonRuntimeProviderObject commandEncoder,
                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    PreparedPipeline *pipeline = descriptor ? fromHandle<PreparedPipeline>(descriptor->pipeline) : nullptr;
    PreparedBindingSet *bindings = descriptor ? fromHandle<PreparedBindingSet>(descriptor->bindings) : nullptr;
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !pipeline ||
        (!bindings && descriptor->bindings.value != 0))
        return fail(adapter, "CUDA adapter received an invalid dispatch");
    if (nativeCommandEncoder(adapter, commandEncoder) != reinterpret_cast<uintptr_t>(pipeline->device) ||
        commandEncoderRendering(adapter, commandEncoder))
        return fail(adapter, "CUDA dispatch command encoder is invalid");
    if (!retainCommandObjects(adapter, commandEncoder, *pipeline, bindings))
        return fail(adapter, "CUDA dispatch could not retain provider objects", VERNON_STATUS_INTERNAL_ERROR);
    std::unique_lock<std::mutex> guard;
    if (bindings)
        guard = std::unique_lock<std::mutex>(bindings->mutex);
    if (bindings)
        for (const auto &slot : bindings->slots)
            if (slot.resourceReference.resource.value &&
                !retainCommandResource(adapter, commandEncoder, slot.resourceReference))
                return fail(adapter, "CUDA dispatch could not retain its resources", VERNON_STATUS_INTERNAL_ERROR);
    const VernonStatus status =
        cudaStatus(adapter,
                   pipeline->function.launch(*pipeline->device, descriptor->group_count, pipeline->workgroup,
                                             bindings ? bindings->parameters.data() : nullptr),
                   "cuLaunchKernel");
    if (status == VERNON_STATUS_OK && !recordProviderCommand(adapter, commandEncoder, false))
        return fail(adapter, "CUDA dispatch command encoder state changed");
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
void releaseCommandBindings(void *context, uint64_t) {
    auto *bindings = static_cast<PreparedBindingSet *>(context);
    if (bindings && bindings->references.fetch_sub(1, std::memory_order_acq_rel) == 1)
        delete bindings;
}
void destroyBindingSet(void *, VernonRuntimeProviderObject handle) {
    releaseCommandBindings(fromHandle<PreparedBindingSet>(handle), 0);
}
void releaseCommandPipeline(void *context, uint64_t) {
    auto *pipeline = static_cast<PreparedPipeline *>(context);
    if (!pipeline || pipeline->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
        return;
    pipeline->function.destroy(*pipeline->device);
    delete pipeline;
}
void destroyPipeline(void *, VernonRuntimeProviderObject handle) {
    releaseCommandPipeline(fromHandle<PreparedPipeline>(handle), 0);
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
