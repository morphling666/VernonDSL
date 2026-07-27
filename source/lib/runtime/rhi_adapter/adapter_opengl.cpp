#include "../vertex_attribute_capabilities.h"
#include "adapter_common.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

namespace vernon::runtime::rhi_adapter {
namespace {

struct PreparedShader {
    uint32_t stage{};
    std::string source;
};

struct PreparedLayout {
    struct Entry {
        VernonRuntimeProviderBindingLayoutEntry layout{};
        std::string name;
    };
    std::vector<Entry> entries;
    std::vector<VernonRuntimeProviderVertexAttribute> vertexAttributes;
};

struct PreparedPipeline {
    struct Binding {
        uint32_t slot{};
        VernonRuntimeProviderBindingKind kind{VERNON_RUNTIME_PROVIDER_INLINE_VALUE};
        rhi::opengl::Int location{-1};
        uint32_t valueCount{};
        uint32_t columnCount{};
        uint32_t binding{};
        uint32_t access{};
        uint32_t divisor{};
    };
    struct VertexAttribute {
        VernonRuntimeProviderVertexAttribute layout{};
        uint32_t divisor{};
    };
    std::atomic<uint32_t> references{1};
    VernonRuntimeRhiAdapter *adapter{};
    rhi::opengl::DeviceState *device{};
    rhi::opengl::Uint program{};
    rhi::opengl::Uint vertexArray{};
    rhi::opengl::Uint framebuffer{};
    std::vector<Binding> bindings;
    std::vector<VertexAttribute> vertexAttributes;
    bool compute{};
};

struct PreparedBindingSet {
    std::atomic<uint32_t> references{1};
    struct Slot {
        uint32_t slot{};
        VernonRuntimeProviderBindingKind kind{VERNON_RUNTIME_PROVIDER_INLINE_VALUE};
        uint32_t valueCount{};
        uint32_t columnCount{};
        uint32_t flags{};
        uint64_t resource{};
        VernonRuntimeProviderResourceReference resourceReference{};
        uint64_t resourceOffset{};
        uint32_t resourceTarget{};
        uint32_t resourceStride{};
        uint32_t binding{};
        uint32_t stageMask{};
        size_t byteSize{};
        std::vector<float> storage;
        rhi::opengl::Buffer inlineBuffer;
    };
    rhi::opengl::DeviceState *device{};
    std::vector<Slot> slots;
    std::unordered_map<uint32_t, size_t> slotIndices;
    std::unordered_map<uint32_t, size_t> vertexSlotByBinding;
    std::vector<size_t> valueIndices;
    std::vector<uint8_t> seenSlots;
    std::vector<uint64_t> resolvedValues;
    std::mutex mutex;

    ~PreparedBindingSet() {
        if (!device)
            return;
        for (auto &slot : slots)
            if (slot.inlineBuffer.name)
                device->destroyBuffer(slot.inlineBuffer);
    }
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

uint32_t getCapabilities(void *) {
    return VERNON_RUNTIME_PROVIDER_COMPUTE | VERNON_RUNTIME_PROVIDER_GRAPHICS | VERNON_RUNTIME_PROVIDER_NATIVE_INTEROP;
}

VernonRuntimeProviderDeviceIdentity getDeviceIdentity(void *data) {
    const auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    const uint64_t identity = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(adapter.openGLDevice));
    return {0x4f50454e474cu, identity,
            static_cast<uint64_t>(adapter.openGLDevice->callbacks.api_version_major) << 32 |
                adapter.openGLDevice->callbacks.api_version_minor};
}

VernonStatus prepareShader(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !descriptor->data ||
        descriptor->size == 0 ||
        (descriptor->stage != VERNON_RUNTIME_PROVIDER_STAGE_VERTEX &&
         descriptor->stage != VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT &&
         descriptor->stage != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE))
        return fail(adapter, "OpenGL adapter received an invalid shader");
    try {
        auto shader = std::make_unique<PreparedShader>();
        shader->stage = descriptor->stage;
        shader->source.assign(static_cast<const char *>(descriptor->data), descriptor->size);
        *output = toHandle(shader.release());
        adapter.shaderPreparations.fetch_add(1, std::memory_order_relaxed);
        return VERNON_STATUS_OK;
    } catch (const std::bad_alloc &) {
        return fail(adapter, "OpenGL shader preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    }
}

VernonStatus prepareLayout(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        (descriptor->binding_count != 0 && !descriptor->bindings) ||
        (descriptor->vertex_attribute_count != 0 && !descriptor->vertex_attributes))
        return fail(adapter, "OpenGL adapter received an invalid pipeline layout");
    auto layout = std::unique_ptr<PreparedLayout>(new (std::nothrow) PreparedLayout());
    if (!layout)
        return fail(adapter, "OpenGL layout preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    try {
        layout->entries.reserve(descriptor->binding_count);
        for (size_t index = 0; index < descriptor->binding_count; ++index) {
            const auto &source = descriptor->bindings[index];
            const bool inlineValue =
                source.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
                ((source.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE && source.binding != UINT32_MAX &&
                  source.element_size != 0) ||
                 (source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM && source.name.data &&
                  source.name.size != 0 && source.vector_count != 0 &&
                  ((source.vector_count == 1 && source.element_count >= 1 && source.element_count <= 4) ||
                   (source.vector_count >= 2 && source.vector_count <= 4 &&
                    source.element_count % source.vector_count == 0 &&
                    source.element_count / source.vector_count >= 2 &&
                    source.element_count / source.vector_count <= 4)) &&
                  source.element_size == source.element_count * sizeof(float)));
            const bool storageBuffer = source.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
                                       (source.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE ||
                                        source.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX ||
                                        source.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT) &&
                                       source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE &&
                                       source.binding != UINT32_MAX && source.element_size != 0;
            const bool uniformBuffer = source.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER &&
                                       source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM &&
                                       source.binding != UINT32_MAX && source.element_size != 0;
            const bool sampledImage = source.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE &&
                                      source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE &&
                                      source.binding != UINT32_MAX && source.name.data && source.name.size != 0;
            const bool sampler = source.kind == VERNON_RUNTIME_PROVIDER_SAMPLER &&
                                 source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE &&
                                 source.binding != UINT32_MAX;
            const bool vertexBuffer = source.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER &&
                                      source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_VERTEX_INPUT &&
                                      source.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX &&
                                      source.binding != UINT32_MAX && source.element_size != 0;
            if (!inlineValue && !uniformBuffer && !storageBuffer && !sampledImage && !sampler && !vertexBuffer)
                return fail(adapter, "OpenGL adapter pipeline layout contains an unsupported binding");
            PreparedLayout::Entry entry;
            entry.layout = source;
            if (source.name.data && source.name.size)
                entry.name.assign(source.name.data, source.name.size);
            layout->entries.push_back(std::move(entry));
        }
        rhi::opengl::Int maximumLocations = 0;
        adapter.openGLDevice->driver.getIntegerv(rhi::opengl::kMaxVertexAttribs, &maximumLocations);
        for (size_t index = 0; index < descriptor->vertex_attribute_count; ++index) {
            const auto &attribute = descriptor->vertex_attributes[index];
            const bool bindingExists =
                std::any_of(layout->entries.begin(), layout->entries.end(), [&](const PreparedLayout::Entry &entry) {
                    return entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER &&
                           entry.layout.binding == attribute.binding;
                });
            std::string capabilityDiagnostic;
            const uint32_t locationLimit = maximumLocations > 0 ? static_cast<uint32_t>(maximumLocations) : 0;
            if (!bindingExists)
                return fail(adapter, "OpenGL vertex attribute references an unknown binding");
            if (!validateVertexAttributeCapability(VertexAttributeBackend::OpenGL, attribute, locationLimit,
                                                   adapter.openGLDevice->driver.vertexAttribLPointer != nullptr,
                                                   capabilityDiagnostic))
                return fail(adapter, std::move(capabilityDiagnostic));
            layout->vertexAttributes.push_back(attribute);
        }
    } catch (const std::bad_alloc &) {
        return fail(adapter, "OpenGL layout preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    }
    *output = toHandle(layout.release());
    adapter.layoutPreparations.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
}

VernonStatus preparePipeline(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                             VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    const bool compute = descriptor && descriptor->kind == VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        (compute ? descriptor->shader_count != 1
                 : descriptor->kind != VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE || descriptor->shader_count != 2) ||
        !descriptor->shaders || !layout)
        return fail(adapter, "OpenGL adapter received an invalid pipeline");
    std::array<rhi::opengl::Uint, 2> compiled{};
    bool hasVertex = false;
    bool hasFragment = false;
    bool hasCompute = false;
    for (size_t index = 0; index < descriptor->shader_count; ++index) {
        const auto *shader = fromHandle<PreparedShader>(descriptor->shaders[index]);
        if (!shader)
            return fail(adapter, "OpenGL pipeline contains an invalid shader");
        const auto kind = shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX     ? rhi::opengl::kVertexShader
                          : shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT ? rhi::opengl::kFragmentShader
                                                                                    : rhi::opengl::kComputeShader;
        hasVertex |= shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX;
        hasFragment |= shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT;
        hasCompute |= shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
        compiled[index] = adapter.openGLDevice->compileShader(kind, shader->source, adapter.error);
        if (!compiled[index]) {
            for (size_t previous = 0; previous < index; ++previous)
                adapter.openGLDevice->driver.deleteShader(compiled[previous]);
            return VERNON_STATUS_INTERNAL_ERROR;
        }
    }
    if ((compute && !hasCompute) || (!compute && (!hasVertex || !hasFragment))) {
        for (size_t index = 0; index < descriptor->shader_count; ++index)
            adapter.openGLDevice->driver.deleteShader(compiled[index]);
        return fail(adapter, "OpenGL pipeline shader stages are incomplete");
    }
    try {
        auto pipeline = std::make_unique<PreparedPipeline>();
        pipeline->adapter = &adapter;
        pipeline->device = adapter.openGLDevice;
        pipeline->compute = compute;
        pipeline->bindings.reserve(layout->entries.size());
        pipeline->program = adapter.openGLDevice->linkProgram(
            {compiled.begin(), compiled.begin() + descriptor->shader_count}, adapter.error);
        if (!pipeline->program)
            return VERNON_STATUS_INTERNAL_ERROR;
        if (!compute &&
            !adapter.openGLDevice->createGraphicsObjects(pipeline->vertexArray, pipeline->framebuffer, adapter.error)) {
            adapter.openGLDevice->destroyProgram(pipeline->program);
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        if (!compute)
            adapter.openGLFramebufferSignatures.emplace(pipeline->framebuffer, OpenGLFramebufferSignature{});
        for (const auto &entry : layout->entries) {
            rhi::opengl::Int location = -1;
            if (!compute && entry.layout.kind != VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER &&
                entry.layout.kind != VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
                entry.layout.kind != VERNON_RUNTIME_PROVIDER_SAMPLER &&
                entry.layout.kind != VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER)
                location = adapter.openGLDevice->driver.getUniformLocation(pipeline->program, entry.name.c_str());
            if (!compute && entry.layout.kind != VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER &&
                entry.layout.kind != VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
                entry.layout.kind != VERNON_RUNTIME_PROVIDER_SAMPLER &&
                entry.layout.kind != VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && location < 0) {
                adapter.openGLDevice->destroyGraphicsObjects(pipeline->vertexArray, pipeline->framebuffer);
                adapter.openGLDevice->destroyProgram(pipeline->program);
                return fail(adapter, "OpenGL uniform location is missing");
            }
            const uint32_t valueCount =
                entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER
                    ? 0
                    : (entry.layout.element_count ? entry.layout.element_count : entry.layout.element_size / 4);
            pipeline->bindings.push_back({entry.layout.slot, entry.layout.kind, location, valueCount,
                                          entry.layout.vector_count, entry.layout.binding, entry.layout.access,
                                          entry.layout.divisor});
        }
        for (const VernonRuntimeProviderVertexAttribute &attribute : layout->vertexAttributes) {
            auto binding = std::find_if(layout->entries.begin(), layout->entries.end(), [&](const auto &entry) {
                return entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER &&
                       entry.layout.binding == attribute.binding;
            });
            if (binding == layout->entries.end())
                return fail(adapter, "OpenGL vertex attribute references a missing binding");
            pipeline->vertexAttributes.push_back({attribute, binding->layout.divisor});
        }
        *output = toHandle(pipeline.release());
        adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
        return VERNON_STATUS_OK;
    } catch (const std::bad_alloc &) {
        return fail(adapter, "OpenGL pipeline preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    }
}

VernonStatus retainResource(void *data, VernonRuntimeProviderResourceReference resource) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return retainRhiResource(adapter, resource) ? VERNON_STATUS_OK
                                                : fail(adapter, "OpenGL adapter received a stale resource");
}

void releaseResource(void *data, VernonRuntimeProviderResourceReference resource) {
    releaseRhiResource(*static_cast<VernonRuntimeRhiAdapter *>(data), resource);
}

VernonStatus updateBindingsImpl(VernonRuntimeRhiAdapter &adapter, PreparedBindingSet &bindings,
                                const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    if (valueCount != bindings.slots.size() || (valueCount != 0 && !values))
        return fail(adapter, "OpenGL binding values do not match the prepared layout");
    std::fill(bindings.seenSlots.begin(), bindings.seenSlots.end(), uint8_t{0});
    for (size_t index = 0; index < valueCount; ++index) {
        const auto found = bindings.slotIndices.find(values[index].slot);
        if (found == bindings.slotIndices.end() || bindings.seenSlots[found->second])
            return fail(adapter, "OpenGL binding slot is invalid or duplicated");
        bindings.seenSlots[found->second] = 1;
        bindings.valueIndices[found->second] = index;
    }
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        const auto &slot = bindings.slots[index];
        const auto *value = &values[bindings.valueIndices[index]];
        if (value->kind != slot.kind)
            return fail(adapter, "OpenGL binding slot or kind is invalid");
        if (slot.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE || slot.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER) {
            if (!value->inline_data || value->inline_size != slot.byteSize ||
                (value->flags & ~VERNON_RUNTIME_PROVIDER_BINDING_TRANSPOSE) != 0 ||
                (slot.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER &&
                 (value->flags & VERNON_RUNTIME_PROVIDER_BINDING_TRANSPOSE) != 0) ||
                (slot.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
                 (value->flags & VERNON_RUNTIME_PROVIDER_BINDING_TRANSPOSE) != 0 && slot.columnCount <= 1))
                return fail(adapter, "OpenGL inline binding is invalid");
            bindings.resolvedValues[index] = 0;
        } else {
            const bool defaultResource = (value->flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0;
            const uint64_t expectedKind = slot.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ? kDirectX12ImageResource
                                          : slot.kind == VERNON_RUNTIME_PROVIDER_SAMPLER     ? kDirectX12SamplerResource
                                                                                             : kDirectX12BufferResource;
            const uint64_t native = defaultResource ? 0 : resolveRhiResource(adapter, value->resource);
            if ((value->flags & ~VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0 ||
                (!defaultResource &&
                 ((value->resource.identity & kDirectX12ResourceKindMask) != expectedKind || native == 0)) ||
                (slot.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE && value->stride > 2) ||
                (slot.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && !defaultResource && value->stride == 0))
                return fail(adapter, "OpenGL resource binding is invalid");
            bindings.resolvedValues[index] = native;
        }
    }
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        auto &slot = bindings.slots[index];
        const auto *value = &values[bindings.valueIndices[index]];
        if (slot.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE || slot.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER) {
            std::memcpy(slot.storage.data(), value->inline_data, value->inline_size);
            if (slot.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
                slot.stageMask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE) {
                if (!bindings.device->uploadBuffer(slot.inlineBuffer, 0, value->inline_data, value->inline_size,
                                                   adapter.error))
                    return VERNON_STATUS_INTERNAL_ERROR;
                slot.resource = slot.inlineBuffer.name;
            }
        } else {
            const bool defaultResource = (value->flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0;
            const uint64_t native = bindings.resolvedValues[index];
            slot.resource = native;
            slot.resourceReference = defaultResource ? VernonRuntimeProviderResourceReference{} : value->resource;
            slot.resourceOffset = value->resource.offset;
            slot.resourceTarget = value->stride;
            slot.resourceStride = value->stride;
        }
        slot.flags = value->flags;
    }
    return VERNON_STATUS_OK;
}

bool retainBindingResources(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                            const PreparedBindingSet *bindings) {
    if (!bindings)
        return true;
    for (const auto &slot : bindings->slots)
        if (slot.resourceReference.resource.value && !retainCommandResource(adapter, encoder, slot.resourceReference))
            return false;
    return true;
}

VernonStatus createBindings(void *data, const VernonRuntimeProviderBindingSetDescriptor *descriptor,
                            VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !layout)
        return fail(adapter, "OpenGL adapter received an invalid binding-set descriptor");
    try {
        auto bindings = std::make_unique<PreparedBindingSet>();
        bindings->device = adapter.openGLDevice;
        bindings->slots.reserve(layout->entries.size());
        bindings->slotIndices.reserve(layout->entries.size());
        bindings->vertexSlotByBinding.reserve(layout->entries.size());
        bindings->valueIndices.resize(layout->entries.size());
        bindings->seenSlots.resize(layout->entries.size());
        bindings->resolvedValues.resize(layout->entries.size());
        for (size_t index = 0; index < layout->entries.size(); ++index) {
            const auto &entry = layout->entries[index];
            PreparedBindingSet::Slot slot;
            slot.slot = entry.layout.slot;
            if (!bindings->slotIndices.emplace(slot.slot, index).second)
                return fail(adapter, "OpenGL binding layout contains duplicate slots");
            if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER &&
                !bindings->vertexSlotByBinding.emplace(entry.layout.binding, index).second)
                return fail(adapter, "OpenGL binding layout contains duplicate vertex-buffer bindings");
            slot.kind = entry.layout.kind;
            slot.valueCount =
                entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER
                    ? 0
                    : (entry.layout.element_count ? entry.layout.element_count : entry.layout.element_size / 4);
            slot.columnCount = entry.layout.vector_count;
            slot.binding = entry.layout.binding;
            slot.stageMask = entry.layout.stage_mask;
            slot.byteSize = entry.layout.element_size;
            if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
                entry.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER) {
                slot.storage.resize((entry.layout.element_size + sizeof(float) - 1) / sizeof(float));
                if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
                    entry.layout.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE) {
                    if (!bindings->device->createBuffer(slot.inlineBuffer, entry.layout.element_size, adapter.error))
                        return VERNON_STATUS_INTERNAL_ERROR;
                }
            }
            bindings->slots.push_back(std::move(slot));
        }
        const VernonStatus status = updateBindingsImpl(adapter, *bindings, descriptor->values, descriptor->value_count);
        if (status != VERNON_STATUS_OK)
            return status;
        *output = toHandle(bindings.release());
        adapter.bindingCreations.fetch_add(1, std::memory_order_relaxed);
        return VERNON_STATUS_OK;
    } catch (const std::bad_alloc &) {
        return fail(adapter, "OpenGL binding preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    }
}

VernonStatus updateBindings(void *data, VernonRuntimeProviderObject handle,
                            const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *bindings = fromHandle<PreparedBindingSet>(handle);
    if (!bindings)
        return fail(adapter, "OpenGL adapter received an invalid binding set");
    std::lock_guard<std::mutex> guard(bindings->mutex);
    return updateBindingsImpl(adapter, *bindings, values, valueCount);
}

VernonStatus encodeDispatch(void *data, VernonRuntimeProviderObject commandEncoder,
                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *pipeline = descriptor ? fromHandle<PreparedPipeline>(descriptor->pipeline) : nullptr;
    auto *bindings = descriptor ? fromHandle<PreparedBindingSet>(descriptor->bindings) : nullptr;
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !pipeline || !pipeline->compute ||
        (!bindings && !pipeline->bindings.empty()) ||
        (bindings && bindings->slots.size() != pipeline->bindings.size()) || descriptor->group_count[0] == 0 ||
        descriptor->group_count[1] == 0 || descriptor->group_count[2] == 0 || descriptor->push_constant_size != 0)
        return fail(adapter, "OpenGL adapter received an invalid compute dispatch");
    std::unique_lock<std::mutex> guard;
    if (bindings)
        guard = std::unique_lock<std::mutex>(bindings->mutex);
    auto &device = *pipeline->device;
    if (nativeCommandEncoder(adapter, commandEncoder) != reinterpret_cast<uintptr_t>(&device) ||
        commandEncoderRendering(adapter, commandEncoder))
        return fail(adapter, "OpenGL dispatch command encoder is invalid");
    if (!retainCommandObjects(adapter, commandEncoder, *pipeline, bindings))
        return fail(adapter, "OpenGL dispatch could not retain provider objects", VERNON_STATUS_INTERNAL_ERROR);
    if (!retainBindingResources(adapter, commandEncoder, bindings))
        return fail(adapter, "OpenGL dispatch could not retain its resources", VERNON_STATUS_INTERNAL_ERROR);
    device.makeCurrent();
    device.driver.useProgram(pipeline->program);
    for (size_t index = 0; index < pipeline->bindings.size(); ++index) {
        const auto &binding = pipeline->bindings[index];
        const auto &slot = bindings->slots[index];
        if (binding.slot != slot.slot || binding.kind != slot.kind ||
            (binding.kind != VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
             binding.kind != VERNON_RUNTIME_PROVIDER_INLINE_VALUE))
            return fail(adapter, "OpenGL compute binding set does not match its pipeline");
        device.driver.bindBufferBase(rhi::opengl::kShaderStorageBuffer, binding.binding,
                                     static_cast<rhi::opengl::Uint>(slot.resource));
    }
    device.driver.dispatchCompute(descriptor->group_count[0], descriptor->group_count[1], descriptor->group_count[2]);
    for (size_t index = 0; index < pipeline->bindings.size(); ++index)
        if (pipeline->bindings[index].access != 0 && bindings->slots[index].resourceReference.resource.value &&
            !recordCommandWriteResource(adapter, commandEncoder, bindings->slots[index].resourceReference))
            return fail(adapter, "OpenGL dispatch could not track a written resource", VERNON_STATUS_INTERNAL_ERROR);
    if (!recordProviderCommand(adapter, commandEncoder, false))
        return fail(adapter, "OpenGL dispatch command encoder state changed");
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
}

VernonStatus encodeDraw(void *data, VernonRuntimeProviderObject commandEncoder,
                        const VernonRuntimeProviderDrawDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *pipeline = descriptor ? fromHandle<PreparedPipeline>(descriptor->pipeline) : nullptr;
    auto *bindings = descriptor ? fromHandle<PreparedBindingSet>(descriptor->bindings) : nullptr;
    constexpr size_t maxAttachments = 8;
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !pipeline || pipeline->compute ||
        descriptor->vertex_count == 0 || descriptor->instance_count == 0 ||
        (!bindings && descriptor->bindings.value != 0) ||
        (pipeline->bindings.empty() ? bindings != nullptr : bindings == nullptr) ||
        descriptor->color_attachment_count == 0 || descriptor->color_attachment_count > maxAttachments ||
        !descriptor->color_attachments ||
        ((descriptor->index_count != 0) != (descriptor->index_buffer.resource.value != 0)) ||
        (descriptor->index_count != 0 && descriptor->index_type != 0))
        return fail(adapter, "OpenGL adapter received an invalid draw");
    std::unique_lock<std::mutex> bindingGuard;
    if (bindings) {
        bindingGuard = std::unique_lock<std::mutex>(bindings->mutex);
        if (bindings->slots.size() != pipeline->bindings.size())
            return fail(adapter, "OpenGL draw binding set does not match its pipeline");
    }
    auto &device = *pipeline->device;
    const int claim = claimCommandRendering(adapter, commandEncoder, vernon::rhi::CommandRenderingStateless);
    if (nativeCommandEncoder(adapter, commandEncoder) != reinterpret_cast<uintptr_t>(&device) || claim < 0)
        return fail(adapter, "OpenGL draw command encoder is invalid");
    const bool firstDraw = claim != 0;
    const bool standaloneRendering = !commandEncoderHasRenderingDescriptor(adapter, commandEncoder);
    const uint64_t renderingFramebuffer = commandRenderingObject(adapter, commandEncoder, pipeline->framebuffer);
    if (!renderingFramebuffer)
        return fail(adapter, "OpenGL command framebuffer state is invalid");
    if (!retainCommandObjects(adapter, commandEncoder, *pipeline, bindings))
        return fail(adapter, "OpenGL draw could not retain provider objects", VERNON_STATUS_INTERNAL_ERROR);
    if (!retainBindingResources(adapter, commandEncoder, bindings))
        return fail(adapter, "OpenGL draw could not retain its bindings", VERNON_STATUS_INTERNAL_ERROR);
    std::array<VernonRhiLoadOperation, maxAttachments> colorLoads{};
    std::array<std::array<float, 4>, maxAttachments> colorClears{};
    for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
        colorLoads[index] = static_cast<VernonRhiLoadOperation>(descriptor->color_attachments[index].load_operation);
        std::copy(std::begin(descriptor->color_attachments[index].clear_color),
                  std::end(descriptor->color_attachments[index].clear_color), colorClears[index].begin());
        VernonRhiStoreOperation ignoredStore =
            static_cast<VernonRhiStoreOperation>(descriptor->color_attachments[index].store_operation);
        (void)commandColorOperations(adapter, commandEncoder, index, colorLoads[index], ignoredStore,
                                     colorClears[index].data());
    }
    VernonRhiLoadOperation depthLoad = static_cast<VernonRhiLoadOperation>(descriptor->depth_load_operation);
    VernonRhiStoreOperation ignoredDepthStore = static_cast<VernonRhiStoreOperation>(descriptor->depth_store_operation);
    VernonRhiLoadOperation ignoredStencilLoad = VERNON_RHI_LOAD_DISCARD;
    VernonRhiStoreOperation ignoredStencilStore = VERNON_RHI_STORE_DISCARD;
    float clearDepth = descriptor->clear_depth;
    uint32_t ignoredClearStencil = 0;
    (void)commandDepthOperations(adapter, commandEncoder, depthLoad, ignoredDepthStore, ignoredStencilLoad,
                                 ignoredStencilStore, clearDepth, ignoredClearStencil);
    auto &driver = device.driver;
    std::array<rhi::opengl::Enum, maxAttachments> drawBuffers{};
    drawBuffers.fill(rhi::opengl::kNone);
    size_t drawBufferCount = 0;
    device.makeCurrent();
    if (adapter.openGLFramebufferGeneration != device.framebufferGeneration) {
        adapter.openGLFramebufferValid = false;
        adapter.openGLFramebufferSignatures.clear();
        adapter.openGLFramebufferGeneration = device.framebufferGeneration;
    }
    if (!adapter.openGLProgramValid || adapter.openGLProgram != pipeline->program) {
        driver.useProgram(pipeline->program);
        adapter.openGLProgram = pipeline->program;
        adapter.openGLProgramValid = true;
    }
    if (!adapter.openGLVertexArrayValid || adapter.openGLVertexArray != pipeline->vertexArray) {
        driver.bindVertexArray(pipeline->vertexArray);
        adapter.openGLVertexArray = pipeline->vertexArray;
        adapter.openGLVertexArrayValid = true;
    }
    for (size_t index = 0; index < pipeline->bindings.size(); ++index) {
        const auto &binding = pipeline->bindings[index];
        const auto &slot = bindings->slots[index];
        if (slot.slot != binding.slot || slot.kind != binding.kind || slot.valueCount != binding.valueCount ||
            slot.columnCount != binding.columnCount)
            return fail(adapter, "OpenGL draw binding order does not match its pipeline");
        if (binding.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER) {
            continue;
        } else if (binding.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER) {
            driver.bindBufferBase(rhi::opengl::kUniformBuffer, binding.binding,
                                  static_cast<rhi::opengl::Uint>(slot.resource));
        } else if (binding.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            driver.bindBufferBase(rhi::opengl::kShaderStorageBuffer, binding.binding,
                                  static_cast<rhi::opengl::Uint>(slot.resource));
        } else if (binding.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE) {
            const auto target = slot.resourceTarget == 0   ? rhi::opengl::kTexture2D
                                : slot.resourceTarget == 1 ? rhi::opengl::kTexture3D
                                                           : rhi::opengl::kTextureCubeMap;
            driver.activeTexture(rhi::opengl::kTexture0 + binding.binding);
            driver.bindTexture(target, static_cast<rhi::opengl::Uint>(slot.resource));
            driver.uniform1i(binding.location, static_cast<rhi::opengl::Int>(binding.binding));
        } else if (binding.kind == VERNON_RUNTIME_PROVIDER_SAMPLER) {
            driver.bindSampler(binding.binding, static_cast<rhi::opengl::Uint>(slot.resource));
        } else if (binding.columnCount <= 1) {
            const float *value = slot.storage.data();
            if (binding.valueCount == 1)
                driver.uniform1fv(binding.location, 1, value);
            else if (binding.valueCount == 2)
                driver.uniform2fv(binding.location, 1, value);
            else if (binding.valueCount == 3)
                driver.uniform3fv(binding.location, 1, value);
            else if (binding.valueCount == 4)
                driver.uniform4fv(binding.location, 1, value);
            else
                driver.uniform4fv(binding.location, 1, value);
        } else {
            const float *value = slot.storage.data();
            const auto transpose =
                static_cast<rhi::opengl::Boolean>((slot.flags & VERNON_RUNTIME_PROVIDER_BINDING_TRANSPOSE) != 0);
            const uint32_t columns = binding.columnCount;
            const uint32_t rows = binding.valueCount / columns;
            if (columns == 2 && rows == 2)
                driver.uniformMatrix2fv(binding.location, 1, transpose, value);
            else if (columns == 2 && rows == 3)
                driver.uniformMatrix2x3fv(binding.location, 1, transpose, value);
            else if (columns == 2 && rows == 4)
                driver.uniformMatrix2x4fv(binding.location, 1, transpose, value);
            else if (columns == 3 && rows == 2)
                driver.uniformMatrix3x2fv(binding.location, 1, transpose, value);
            else if (columns == 3 && rows == 3)
                driver.uniformMatrix3fv(binding.location, 1, transpose, value);
            else if (columns == 3 && rows == 4)
                driver.uniformMatrix3x4fv(binding.location, 1, transpose, value);
            else if (columns == 4 && rows == 2)
                driver.uniformMatrix4x2fv(binding.location, 1, transpose, value);
            else if (columns == 4 && rows == 3)
                driver.uniformMatrix4x3fv(binding.location, 1, transpose, value);
            else
                driver.uniformMatrix4fv(binding.location, 1, transpose, value);
        }
    }
    for (const PreparedPipeline::VertexAttribute &attribute : pipeline->vertexAttributes) {
        const auto found = bindings->vertexSlotByBinding.find(attribute.layout.binding);
        if (found == bindings->vertexSlotByBinding.end())
            return fail(adapter, "OpenGL vertex attribute binding is missing");
        const auto &slot = bindings->slots[found->second];
        driver.bindBuffer(rhi::opengl::kArrayBuffer, static_cast<rhi::opengl::Uint>(slot.resource));
        driver.enableVertexAttribArray(attribute.layout.location);
        const auto pointer = reinterpret_cast<const void *>(
            static_cast<uintptr_t>(slot.resourceOffset + attribute.layout.relative_offset));
        if (attribute.layout.dtype == VERNON_RUNTIME_PROVIDER_I32 ||
            attribute.layout.dtype == VERNON_RUNTIME_PROVIDER_U32) {
            driver.vertexAttribIPointer(
                attribute.layout.location, attribute.layout.component_count,
                attribute.layout.dtype == VERNON_RUNTIME_PROVIDER_I32 ? rhi::opengl::kInt : rhi::opengl::kUnsignedInt,
                slot.resourceStride, pointer);
        } else if (attribute.layout.dtype == VERNON_RUNTIME_PROVIDER_F64) {
            driver.vertexAttribLPointer(attribute.layout.location, attribute.layout.component_count,
                                        rhi::opengl::kDouble, slot.resourceStride, pointer);
        } else {
            driver.vertexAttribPointer(attribute.layout.location, attribute.layout.component_count,
                                       attribute.layout.dtype == VERNON_RUNTIME_PROVIDER_F16 ? rhi::opengl::kHalfFloat
                                                                                             : rhi::opengl::kFloat,
                                       0, slot.resourceStride, pointer);
        }
        driver.vertexAttribDivisor(attribute.layout.location, attribute.divisor);
    }
    OpenGLFramebufferSignature framebufferSignature{};
    std::array<uint64_t, maxAttachments> colorImages{};
    for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
        const auto &attachment = descriptor->color_attachments[index];
        if (!retainCommandResource(adapter, commandEncoder, attachment.image))
            return fail(adapter, "OpenGL draw could not retain its color attachment", VERNON_STATUS_INTERNAL_ERROR);
        colorImages[index] = resolveRhiResource(adapter, attachment.image);
        if (attachment.location >= maxAttachments || colorImages[index] == 0)
            return fail(adapter, "OpenGL draw contains an invalid color attachment");
        framebufferSignature.values[framebufferSignature.count++] = attachment.location;
        framebufferSignature.values[framebufferSignature.count++] = attachment.image.resource.value;
        const auto target = rhi::opengl::kColorAttachment0 + attachment.location;
        drawBuffers[attachment.location] = target;
        drawBufferCount = std::max(drawBufferCount, static_cast<size_t>(attachment.location) + 1);
    }
    const bool hasDepth = descriptor->depth_stencil_attachment.resource.value != 0;
    if (hasDepth && !retainCommandResource(adapter, commandEncoder, descriptor->depth_stencil_attachment))
        return fail(adapter, "OpenGL draw could not retain its depth attachment", VERNON_STATUS_INTERNAL_ERROR);
    const uint64_t depthImage = hasDepth ? resolveRhiResource(adapter, descriptor->depth_stencil_attachment) : 0;
    if (hasDepth && depthImage == 0)
        return fail(adapter, "OpenGL depth attachment is stale");
    framebufferSignature.values[framebufferSignature.count++] = hasDepth;
    framebufferSignature.values[framebufferSignature.count++] = descriptor->depth_stencil_attachment.resource.value;
    if (!adapter.openGLFramebufferValid || adapter.openGLFramebuffer != renderingFramebuffer) {
        driver.bindFramebuffer(rhi::opengl::kFramebuffer, static_cast<rhi::opengl::Uint>(renderingFramebuffer));
        adapter.openGLFramebuffer = static_cast<rhi::opengl::Uint>(renderingFramebuffer);
        adapter.openGLFramebufferValid = true;
    }
    auto cachedFramebuffer =
        adapter.openGLFramebufferSignatures.find(static_cast<rhi::opengl::Uint>(renderingFramebuffer));
    if (cachedFramebuffer == adapter.openGLFramebufferSignatures.end() ||
        !(cachedFramebuffer->second == framebufferSignature)) {
        for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
            const auto &attachment = descriptor->color_attachments[index];
            driver.framebufferTexture2D(rhi::opengl::kFramebuffer, rhi::opengl::kColorAttachment0 + attachment.location,
                                        rhi::opengl::kTexture2D, static_cast<rhi::opengl::Uint>(colorImages[index]), 0);
        }
        driver.framebufferTexture2D(rhi::opengl::kFramebuffer, rhi::opengl::kDepthAttachment, rhi::opengl::kTexture2D,
                                    hasDepth ? static_cast<rhi::opengl::Uint>(depthImage) : 0, 0);
        driver.drawBuffers(static_cast<rhi::opengl::Size>(drawBufferCount), drawBuffers.data());
        if (driver.checkFramebufferStatus(rhi::opengl::kFramebuffer) != rhi::opengl::kFramebufferComplete)
            return fail(adapter, "OpenGL draw framebuffer is incomplete", VERNON_STATUS_INTERNAL_ERROR);
        adapter.openGLFramebufferSignatures.insert_or_assign(static_cast<rhi::opengl::Uint>(renderingFramebuffer),
                                                             framebufferSignature);
    }
    if (!adapter.openGLViewportValid ||
        !std::equal(std::begin(descriptor->viewport), std::end(descriptor->viewport), adapter.openGLViewport.begin())) {
        driver.viewport(static_cast<rhi::opengl::Int>(descriptor->viewport[0]),
                        static_cast<rhi::opengl::Int>(descriptor->viewport[1]),
                        static_cast<rhi::opengl::Size>(descriptor->viewport[2]),
                        static_cast<rhi::opengl::Size>(descriptor->viewport[3]));
        std::copy(std::begin(descriptor->viewport), std::end(descriptor->viewport), adapter.openGLViewport.begin());
        adapter.openGLViewportValid = true;
    }
    for (size_t index = 0; index < descriptor->color_attachment_count; ++index)
        if (firstDraw && colorLoads[index] == VERNON_RHI_LOAD_CLEAR)
            driver.clearBufferfv(rhi::opengl::kColor,
                                 static_cast<rhi::opengl::Int>(descriptor->color_attachments[index].location),
                                 colorClears[index].data());
    if (hasDepth) {
        driver.enable(rhi::opengl::kDepthTest);
        driver.depthFunc(rhi::opengl::kLess);
        if (firstDraw && depthLoad == VERNON_RHI_LOAD_CLEAR)
            driver.clearBufferfv(rhi::opengl::kDepth, 0, &clearDepth);
    } else {
        driver.disable(rhi::opengl::kDepthTest);
    }
    if (driver.scissor) {
        if (!adapter.openGLScissorValid || !std::equal(std::begin(descriptor->scissor), std::end(descriptor->scissor),
                                                       adapter.openGLScissor.begin())) {
            driver.enable(rhi::opengl::kScissorTest);
            driver.scissor(static_cast<rhi::opengl::Int>(descriptor->scissor[0]),
                           static_cast<rhi::opengl::Int>(descriptor->scissor[1]),
                           static_cast<rhi::opengl::Size>(descriptor->scissor[2]),
                           static_cast<rhi::opengl::Size>(descriptor->scissor[3]));
            std::copy(std::begin(descriptor->scissor), std::end(descriptor->scissor), adapter.openGLScissor.begin());
            adapter.openGLScissorValid = true;
        }
    } else if (!std::equal(std::begin(descriptor->viewport), std::end(descriptor->viewport),
                           std::begin(descriptor->scissor))) {
        return fail(adapter, "OpenGL context does not expose glScissor", VERNON_STATUS_UNSUPPORTED_TARGET);
    }
    rhi::opengl::Enum topology = rhi::opengl::kTriangles;
    if (descriptor->topology == 1)
        topology = rhi::opengl::kLines;
    else if (descriptor->topology == 2)
        topology = rhi::opengl::kPoints;
    else if (descriptor->topology != 0)
        return fail(adapter, "OpenGL draw topology is invalid");
    if (descriptor->index_count != 0) {
        if (!retainCommandResource(adapter, commandEncoder, descriptor->index_buffer))
            return fail(adapter, "OpenGL draw could not retain its index buffer", VERNON_STATUS_INTERNAL_ERROR);
        const uint64_t indexBuffer = resolveRhiResource(adapter, descriptor->index_buffer);
        if (!indexBuffer)
            return fail(adapter, "OpenGL draw index buffer is stale");
        driver.bindBuffer(rhi::opengl::kElementArrayBuffer, static_cast<rhi::opengl::Uint>(indexBuffer));
        driver.drawElementsInstanced(
            topology, static_cast<rhi::opengl::Size>(descriptor->index_count), rhi::opengl::kUnsignedInt,
            reinterpret_cast<const void *>(static_cast<uintptr_t>(descriptor->index_buffer.offset)),
            static_cast<rhi::opengl::Size>(descriptor->instance_count));
    } else if (descriptor->instance_count == 1)
        driver.drawArrays(topology, static_cast<rhi::opengl::Int>(descriptor->first_vertex),
                          static_cast<rhi::opengl::Size>(descriptor->vertex_count));
    else
        driver.drawArraysInstanced(topology, static_cast<rhi::opengl::Int>(descriptor->first_vertex),
                                   static_cast<rhi::opengl::Size>(descriptor->vertex_count),
                                   static_cast<rhi::opengl::Size>(descriptor->instance_count));
    if (standaloneRendering && driver.invalidateFramebuffer) {
        std::array<rhi::opengl::Enum, maxAttachments + 1> discarded{};
        size_t discardedCount = 0;
        for (size_t index = 0; index < descriptor->color_attachment_count; ++index)
            if (descriptor->color_attachments[index].store_operation == VERNON_RHI_STORE_DISCARD)
                discarded[discardedCount++] =
                    rhi::opengl::kColorAttachment0 + descriptor->color_attachments[index].location;
        if (hasDepth && descriptor->depth_store_operation == VERNON_RHI_STORE_DISCARD)
            discarded[discardedCount++] = rhi::opengl::kDepthAttachment;
        if (discardedCount)
            driver.invalidateFramebuffer(rhi::opengl::kFramebuffer, static_cast<rhi::opengl::Size>(discardedCount),
                                         discarded.data());
    }
    if (!recordProviderCommand(adapter, commandEncoder, true))
        return fail(adapter, "OpenGL draw command encoder state changed");
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
}

void destroyShader(void *, VernonRuntimeProviderObject handle) { delete fromHandle<PreparedShader>(handle); }
void destroyLayout(void *, VernonRuntimeProviderObject handle) { delete fromHandle<PreparedLayout>(handle); }
void releaseCommandBindings(void *context, uint64_t) {
    auto *bindings = static_cast<PreparedBindingSet *>(context);
    if (!bindings || bindings->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
        return;
    delete bindings;
}
void destroyBindings(void *, VernonRuntimeProviderObject handle) {
    releaseCommandBindings(fromHandle<PreparedBindingSet>(handle), 0);
}
void releaseCommandPipeline(void *context, uint64_t) {
    auto *pipeline = static_cast<PreparedPipeline *>(context);
    if (!pipeline || pipeline->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
        return;
    if (pipeline->framebuffer) {
        pipeline->adapter->openGLFramebufferSignatures.erase(pipeline->framebuffer);
        if (pipeline->adapter->openGLFramebufferValid && pipeline->adapter->openGLFramebuffer == pipeline->framebuffer)
            pipeline->adapter->openGLFramebufferValid = false;
    }
    if (pipeline->adapter->openGLProgramValid && pipeline->adapter->openGLProgram == pipeline->program)
        pipeline->adapter->openGLProgramValid = false;
    if (pipeline->adapter->openGLVertexArrayValid && pipeline->adapter->openGLVertexArray == pipeline->vertexArray)
        pipeline->adapter->openGLVertexArrayValid = false;
    pipeline->device->destroyGraphicsObjects(pipeline->vertexArray, pipeline->framebuffer);
    pipeline->device->destroyProgram(pipeline->program);
    delete pipeline;
}
void destroyPipeline(void *, VernonRuntimeProviderObject handle) {
    releaseCommandPipeline(fromHandle<PreparedPipeline>(handle), 0);
}

} // namespace

void initializeOpenGLProvider(VernonRuntimeRhiAdapter &adapter) {
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
    adapter.provider.create_binding_set = createBindings;
    adapter.provider.update_binding_set = updateBindings;
    adapter.provider.encode_dispatch = encodeDispatch;
    adapter.provider.encode_draw = encodeDraw;
    adapter.provider.destroy_shader = destroyShader;
    adapter.provider.destroy_pipeline_layout = destroyLayout;
    adapter.provider.destroy_pipeline = destroyPipeline;
    adapter.provider.destroy_binding_set = destroyBindings;
}

} // namespace vernon::runtime::rhi_adapter
