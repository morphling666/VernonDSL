#include "adapter_common.h"

#if defined(VERNON_HAS_CUDA_RHI)

#include "rhi/cuda_backend.h"
#include "rhi/rhi_internal.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

namespace vernon::runtime::rhi_adapter {
namespace {

using rhi::cuda::DevicePointer;
using rhi::cuda::DeviceState;
using rhi::cuda::PreparedFunction;

struct CudaAdapterState {
    DeviceState *device{};
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
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"cuStreamSynchronize", static_cast<uint64_t>(result), 0}})};
    } catch (...) {
        return RhiAdapterResult<void>{vernon::err(
            vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure, {"cuStreamSynchronize", 0, 0}})};
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
        DevicePointer hostStorage{};
        size_t hostStorageSize{};
    };
    DeviceState *device{};
    std::vector<Slot> slots;
    std::vector<MemRefDescriptor> descriptors;
    std::vector<void *> parameters;

    ~PreparedBindingSet() {
        if (!device)
            return;
        for (const Slot &slot : slots)
            if (slot.hostStorage)
                device->free(slot.hostStorage);
    }
};

void releaseCommandBindings(void *context, uint64_t);
void releaseCommandPipeline(void *context, uint64_t);

RhiAdapterResult<void> retainCommandObjects(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                            PreparedPipeline &pipeline, PreparedBindingSet *bindings) {
    pipeline.references.fetch_add(1, std::memory_order_relaxed);
    if (!deferCommandCleanup(adapter, encoder, &pipeline, 0, releaseCommandPipeline)) {
        releaseCommandPipeline(&pipeline, 0);
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"cuda_command_pipeline_cleanup_registration_failed", 0, 0}})};
    }
    if (bindings) {
        bindings->references.fetch_add(1, std::memory_order_relaxed);
        if (!deferCommandCleanup(adapter, encoder, bindings, 0, releaseCommandBindings)) {
            releaseCommandBindings(bindings, 0);
            return RhiAdapterResult<void>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                                  {"cuda_command_bindings_cleanup_registration_failed", 0, 0}})};
        }
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> cudaResult(rhi::cuda::Result result, const char *operation) {
    return result == rhi::cuda::kSuccess
               ? RhiAdapterResult<void>{vernon::ok()}
               : RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                     vernon::ProviderErrorCode::BackendFailure, {operation, static_cast<uint64_t>(result), 0}})};
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
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                              {"cuda_adapter_received_an_invalid_compute_shader_descriptor", 0, 0}})};
    try {
        auto shader = std::make_unique<PreparedShader>();
        const auto *bytes = static_cast<const uint8_t *>(descriptor->data);
        shader->artifact.assign(bytes, bytes + descriptor->size);
        shader->entry.assign(descriptor->entry.data, descriptor->entry.size);
        *output = toHandle(shader.release());
        adapter.shaderPreparations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"cuda_shader_preparation_ran_out_of_memory", 0, 0}})};
    }
}

RhiAdapterResult<void> prepareLayoutResult(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        (descriptor->binding_count != 0 && !descriptor->bindings))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"cuda_adapter_received_an_invalid_pipeline_layout", 0, 0}})};
    try {
        auto layout = std::make_unique<PreparedLayout>();
        layout->entries.assign(descriptor->bindings, descriptor->bindings + descriptor->binding_count);
        for (const auto &entry : layout->entries)
            if ((entry.kind != VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
                 entry.kind != VERNON_RUNTIME_PROVIDER_INLINE_VALUE) ||
                entry.stage_mask != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE || entry.array_count != 1 ||
                (entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER && entry.element_size == 0))
                return RhiAdapterResult<void>{vernon::err(
                    vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                          {"cuda_adapter_pipeline_layout_contains_an_unsupported_binding", 0, 0}})};
        std::sort(layout->entries.begin(), layout->entries.end(),
                  [](const auto &left, const auto &right) { return left.binding < right.binding; });
        for (size_t index = 0; index < layout->entries.size(); ++index)
            if (layout->entries[index].binding != index)
                return RhiAdapterResult<void>{
                    vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                      {"cuda_adapter_abi_bindings_must_be_contiguous", 0, 0}})};
        *output = toHandle(layout.release());
        adapter.layoutPreparations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"cuda_layout_preparation_ran_out_of_memory", 0, 0}})};
    }
}

RhiAdapterResult<void> preparePipelineResult(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                                             VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        descriptor->kind != VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE || descriptor->shader_count != 1 ||
        !descriptor->shaders || !fromHandle<PreparedShader>(descriptor->shaders[0]) ||
        !fromHandle<PreparedLayout>(descriptor->layout) || descriptor->workgroup_size[0] == 0 ||
        descriptor->workgroup_size[1] == 0 || descriptor->workgroup_size[2] == 0)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"cuda_adapter_received_an_invalid_compute_pipeline", 0, 0}})};
    try {
        auto pipeline = std::make_unique<PreparedPipeline>();
        PreparedShader &shader = *fromHandle<PreparedShader>(descriptor->shaders[0]);
        auto &device = cudaDevice(adapter);
        auto created = cudaResult(
            pipeline->function.create(device, shader.artifact.data(), shader.artifact.size(), shader.entry.c_str()),
            "CUDA pipeline preparation");
        if (!created)
            return created;
        pipeline->device = &device;
        std::copy_n(descriptor->workgroup_size, 3, pipeline->workgroup);
        *output = toHandle(pipeline.release());
        adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"cuda_pipeline_preparation_ran_out_of_memory", 0, 0}})};
    }
}

RhiAdapterResult<void> initializeBindingsResult(VernonRuntimeRhiAdapter &adapter, PreparedBindingSet &bindings,
                                                const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    if (valueCount != bindings.slots.size() || (valueCount != 0 && !values))
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                              {"cuda_binding_values_do_not_match_the_prepared_layout", 0, 0}})};
    std::unordered_map<uint32_t, size_t> slotIndices;
    std::vector<size_t> valueIndices(bindings.slots.size());
    std::vector<uint8_t> seenSlots(bindings.slots.size());
    std::vector<DevicePointer> resolvedValues(bindings.slots.size());
    slotIndices.reserve(bindings.slots.size());
    for (size_t index = 0; index < bindings.slots.size(); ++index)
        if (!slotIndices.emplace(bindings.slots[index].layout.slot, index).second)
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::InvalidArgument, {"cuda_binding_layout_contains_duplicate_slots", 0, 0}})};
    for (size_t index = 0; index < valueCount; ++index) {
        const auto found = slotIndices.find(values[index].slot);
        if (found == slotIndices.end() || seenSlots[found->second])
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::InvalidArgument, {"cuda_binding_slot_is_invalid_or_duplicated", 0, 0}})};
        seenSlots[found->second] = 1;
        valueIndices[found->second] = index;
    }
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        auto &slot = bindings.slots[index];
        const auto *value = &values[valueIndices[index]];
        if (value->kind != slot.layout.kind)
            return RhiAdapterResult<void>{vernon::err(
                vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                      {"cuda_binding_slot_or_kind_does_not_match_the_prepared_layout", 0, 0}})};
        if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            const bool hostStorage = (value->flags & VERNON_RUNTIME_PROVIDER_BINDING_HOST_STORAGE) != 0;
            if ((value->flags & ~VERNON_RUNTIME_PROVIDER_BINDING_HOST_STORAGE) != 0)
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::InvalidArgument, {"cuda_storage_buffer_flags_are_invalid", 0, 0}})};
            if (hostStorage) {
                const auto &inlineValue = value->payload.inline_value;
                if (!inlineValue.data || !inlineValue.size || inlineValue.size % slot.layout.element_size != 0)
                    return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                        vernon::ProviderErrorCode::InvalidArgument,
                        {"cuda_host_storage_binding_is_invalid", inlineValue.size, slot.layout.element_size}})};
                if (!slot.hostStorage) {
                    auto allocated = cudaResult(cudaDevice(adapter).allocate(slot.hostStorage, inlineValue.size),
                                                "cuMemAlloc host storage binding");
                    if (!allocated)
                        return allocated;
                    slot.hostStorageSize = inlineValue.size;
                } else if (slot.hostStorageSize != inlineValue.size) {
                    return RhiAdapterResult<void>{
                        vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                          {"cuda_host_storage_binding_size_changed", inlineValue.size,
                                                           static_cast<uint32_t>(slot.hostStorageSize)}})};
                }
                auto uploaded =
                    cudaResult(cudaDevice(adapter).upload(slot.hostStorage, inlineValue.data, inlineValue.size),
                               "cuMemcpyHtoD host storage binding");
                if (!uploaded)
                    return uploaded;
                resolvedValues[index] = slot.hostStorage;
            } else {
                const auto &reference = value->payload.buffer.resource;
                auto resolved = resolveRhiResource(adapter, reference);
                if (!resolved)
                    return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
                const DevicePointer resource = std::move(resolved).value();
                if (reference.size == 0)
                    return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                        vernon::ProviderErrorCode::InvalidArgument, {"cuda_storage_buffer_binding_is_invalid", 0, 0}})};
                resolvedValues[index] = resource + reference.offset;
            }
        } else {
            if (!value->payload.inline_value.data || value->payload.inline_value.size == 0)
                return RhiAdapterResult<void>{
                    vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                      {"cuda_inline_binding_requires_initial_storage", 0, 0}})};
            slot.inlineStorage.resize(value->payload.inline_value.size);
        }
    }
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        auto &slot = bindings.slots[index];
        const auto *value = &values[valueIndices[index]];
        slot.resourceReference = {};
        if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            const DevicePointer pointer = resolvedValues[index];
            const bool hostStorage = (value->flags & VERNON_RUNTIME_PROVIDER_BINDING_HOST_STORAGE) != 0;
            const uint64_t byteSize =
                hostStorage ? value->payload.inline_value.size : value->payload.buffer.resource.size;
            const uint64_t byteStride =
                hostStorage || !value->payload.buffer.stride ? slot.layout.element_size : value->payload.buffer.stride;
            if (byteStride < slot.layout.element_size || byteStride % slot.layout.element_size != 0)
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::InvalidArgument, {"cuda_storage_buffer_stride_is_invalid", 0, 0}})};
            bindings.descriptors[slot.descriptorOffset] = {pointer, pointer, 0, byteSize / byteStride,
                                                           byteStride / slot.layout.element_size};
            if (!hostStorage)
                slot.resourceReference = value->payload.buffer.resource;
        } else
            std::memcpy(slot.inlineStorage.data(), value->payload.inline_value.data, value->payload.inline_value.size);
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> createBindingSetResult(void *data, const VernonRuntimeProviderBindingSetDescriptor *descriptor,
                                              VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    PreparedLayout *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !layout)
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                              {"cuda_adapter_received_an_invalid_binding_set_descriptor", 0, 0}})};
    try {
        auto bindings = std::make_unique<PreparedBindingSet>();
        bindings->device = &cudaDevice(adapter);
        size_t descriptorCount = 0;
        size_t parameterCount = 0;
        bindings->slots.reserve(layout->entries.size());
        for (size_t index = 0; index < layout->entries.size(); ++index) {
            const auto &entry = layout->entries[index];
            PreparedBindingSet::Slot slot;
            slot.layout = entry;
            slot.parameterOffset = parameterCount;
            if (entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
                slot.descriptorOffset = descriptorCount++;
                parameterCount += 5;
            } else
                ++parameterCount;
            bindings->slots.push_back(std::move(slot));
        }
        bindings->descriptors.resize(descriptorCount);
        bindings->parameters.resize(parameterCount);
        auto initialized = initializeBindingsResult(adapter, *bindings, descriptor->values, descriptor->value_count);
        if (!initialized)
            return initialized;
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
        *output = toHandle(bindings.release());
        adapter.bindingCreations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"cuda_binding_preparation_ran_out_of_memory", 0, 0}})};
    }
}

RhiAdapterResult<void> encodeDispatchResult(void *data, VernonRuntimeProviderObject commandEncoder,
                                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    PreparedPipeline *pipeline = descriptor ? fromHandle<PreparedPipeline>(descriptor->pipeline) : nullptr;
    PreparedBindingSet *bindings = descriptor ? fromHandle<PreparedBindingSet>(descriptor->bindings) : nullptr;
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !pipeline ||
        (!bindings && descriptor->bindings.value != 0))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"cuda_adapter_received_an_invalid_dispatch", 0, 0}})};
    auto native = nativeCommandEncoder(adapter, commandEncoder);
    if (!native)
        return RhiAdapterResult<void>{vernon::err(std::move(native).error())};
    const uint64_t nativeValue = std::move(native).value();
    auto rendering = commandEncoderRendering(adapter, commandEncoder);
    if (!rendering)
        return RhiAdapterResult<void>{vernon::err(std::move(rendering).error())};
    if (nativeValue != reinterpret_cast<uintptr_t>(pipeline->device) || std::move(rendering).value())
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"cuda_dispatch_command_encoder_is_invalid", 0, 0}})};
    if (!retainCommandObjects(adapter, commandEncoder, *pipeline, bindings))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"cuda_dispatch_could_not_retain_provider_objects", 0, 0}})};
    if (bindings)
        for (const auto &slot : bindings->slots)
            if (slot.resourceReference.resource.value &&
                !retainCommandResource(adapter, commandEncoder, slot.resourceReference))
                return RhiAdapterResult<void>{
                    vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                                      {"cuda_dispatch_could_not_retain_its_resources", 0, 0}})};
    auto launched =
        cudaResult(pipeline->function.launch(*pipeline->device, descriptor->group_count, pipeline->workgroup,
                                             bindings ? bindings->parameters.data() : nullptr),
                   "cuLaunchKernel");
    if (!launched)
        return launched;
    if (!recordProviderCommand(adapter, commandEncoder, false))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"cuda_dispatch_command_encoder_state_changed", 0, 0}})};
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> encodeDrawResult(void *data, VernonRuntimeProviderObject,
                                        const VernonRuntimeProviderDrawDescriptor *) {
    return RhiAdapterResult<void>{vernon::err(
        vernon::ProviderError{vernon::ProviderErrorCode::Unsupported, {"cuda_does_not_support_graphics", 0, 0}})};
}

RhiAdapterResult<void> destroyShaderResult(void *, VernonRuntimeProviderObject handle) {
    delete fromHandle<PreparedShader>(handle);
    return RhiAdapterResult<void>{vernon::ok()};
}
RhiAdapterResult<void> destroyLayoutResult(void *, VernonRuntimeProviderObject handle) {
    delete fromHandle<PreparedLayout>(handle);
    return RhiAdapterResult<void>{vernon::ok()};
}
void releaseCommandBindings(void *context, uint64_t) {
    auto *bindings = static_cast<PreparedBindingSet *>(context);
    if (bindings && bindings->references.fetch_sub(1, std::memory_order_acq_rel) == 1)
        delete bindings;
}
RhiAdapterResult<void> destroyBindingSetResult(void *, VernonRuntimeProviderObject handle) {
    releaseCommandBindings(fromHandle<PreparedBindingSet>(handle), 0);
    return RhiAdapterResult<void>{vernon::ok()};
}
void releaseCommandPipeline(void *context, uint64_t) {
    auto *pipeline = static_cast<PreparedPipeline *>(context);
    if (!pipeline || pipeline->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
        return;
    pipeline->function.destroy(*pipeline->device);
    delete pipeline;
}
RhiAdapterResult<void> destroyPipelineResult(void *, VernonRuntimeProviderObject handle) {
    releaseCommandPipeline(fromHandle<PreparedPipeline>(handle), 0);
    return RhiAdapterResult<void>{vernon::ok()};
}

VernonStatus prepareShader(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, prepareShaderResult(data, descriptor, output), "CUDA shader preparation failed");
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
VernonStatus createBindingSet(void *data, const VernonRuntimeProviderBindingSetDescriptor *descriptor,
                              VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, createBindingSetResult(data, descriptor, output),
                          "CUDA binding-set creation failed");
}
VernonStatus encodeDispatch(void *data, VernonRuntimeProviderObject commandEncoder,
                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, encodeDispatchResult(data, commandEncoder, descriptor),
                          "CUDA dispatch encoding failed");
}
VernonStatus encodeDraw(void *data, VernonRuntimeProviderObject commandEncoder,
                        const VernonRuntimeProviderDrawDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, encodeDrawResult(data, commandEncoder, descriptor), "CUDA draw encoding failed");
}
void destroyShader(void *data, VernonRuntimeProviderObject handle) {
    auto result = destroyShaderResult(data, handle);
    if (!result)
        recordProviderError(*static_cast<VernonRuntimeRhiAdapter *>(data), std::move(result).error(),
                            "CUDA shader destruction failed");
}
void destroyLayout(void *data, VernonRuntimeProviderObject handle) {
    auto result = destroyLayoutResult(data, handle);
    if (!result)
        recordProviderError(*static_cast<VernonRuntimeRhiAdapter *>(data), std::move(result).error(),
                            "CUDA layout destruction failed");
}
void destroyPipeline(void *data, VernonRuntimeProviderObject handle) {
    auto result = destroyPipelineResult(data, handle);
    if (!result)
        recordProviderError(*static_cast<VernonRuntimeRhiAdapter *>(data), std::move(result).error(),
                            "CUDA pipeline destruction failed");
}
void destroyBindingSet(void *data, VernonRuntimeProviderObject handle) {
    auto result = destroyBindingSetResult(data, handle);
    if (!result)
        recordProviderError(*static_cast<VernonRuntimeRhiAdapter *>(data), std::move(result).error(),
                            "CUDA binding-set destruction failed");
}

} // namespace

void initializeCudaProvider(VernonRuntimeRhiAdapter &adapter) {
    adapter.provider.struct_size = sizeof(adapter.provider);
    adapter.provider.abi_version = VERNON_PROGRAM_VERSION;
    adapter.provider.user_data = &adapter;
    adapter.provider.get_capabilities = getCapabilities;
    adapter.provider.get_device_identity = getDeviceIdentity;
    adapter.provider.prepare_shader = prepareShader;
    adapter.provider.prepare_pipeline_layout = prepareLayout;
    adapter.provider.prepare_pipeline = preparePipeline;
    adapter.provider.retain_resource = retainRhiResourceCallback;
    adapter.provider.release_resource = releaseRhiResourceCallback;
    adapter.provider.describe_image = describeProviderImageCallback;
    adapter.provider.create_binding_set = createBindingSet;
    adapter.provider.encode_dispatch = encodeDispatch;
    adapter.provider.encode_draw = encodeDraw;
    adapter.provider.destroy_shader = destroyShader;
    adapter.provider.destroy_pipeline_layout = destroyLayout;
    adapter.provider.destroy_pipeline = destroyPipeline;
    adapter.provider.destroy_binding_set = destroyBindingSet;
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
