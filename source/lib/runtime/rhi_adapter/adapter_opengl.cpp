#include "adapter_common.h"
#include "rhi/opengl_backend.h"
#include "rhi/rhi_internal.h"
#include "runtime/vertex_attribute_capabilities.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace vernon::runtime::rhi_adapter {
namespace {

struct OpenGLFramebufferSignature {
    std::array<uint64_t, 18> values{};
    size_t count{};

    bool operator==(const OpenGLFramebufferSignature &other) const {
        return count == other.count && std::equal(values.begin(), values.begin() + count, other.values.begin());
    }
};

struct OpenGLAdapterState {
    rhi::opengl::DeviceState *device{};
    uint64_t program{};
    uint64_t vertexArray{};
    uint64_t framebuffer{};
    uint64_t framebufferGeneration{};
    std::array<uint32_t, 4> viewport{};
    std::array<uint32_t, 4> scissor{};
    std::unordered_map<uint64_t, OpenGLFramebufferSignature> framebufferSignatures;
    bool programValid{};
    bool vertexArrayValid{};
    bool framebufferValid{};
    bool viewportValid{};
    bool scissorValid{};

    void invalidate() {
        programValid = false;
        vertexArrayValid = false;
        framebufferValid = false;
        viewportValid = false;
        scissorValid = false;
        framebufferSignatures.clear();
    }
};

OpenGLAdapterState &openGLState(VernonRuntimeRhiAdapter &adapter) {
    assert(adapter.rhiBackend == VERNON_RHI_BACKEND_OPENGL || adapter.rhiBackend == VERNON_RHI_BACKEND_OPENGL_ES);
    assert(adapter.backend.state);
    return *static_cast<OpenGLAdapterState *>(adapter.backend.state);
}

const OpenGLAdapterState &openGLState(const VernonRuntimeRhiAdapter &adapter) {
    assert(adapter.rhiBackend == VERNON_RHI_BACKEND_OPENGL || adapter.rhiBackend == VERNON_RHI_BACKEND_OPENGL_ES);
    assert(adapter.backend.state);
    return *static_cast<const OpenGLAdapterState *>(adapter.backend.state);
}

void destroyBackend(void *state) noexcept { delete static_cast<OpenGLAdapterState *>(state); }

VernonStatus synchronizeBackend(void *state, std::string &error) noexcept {
    try {
        auto &device = *static_cast<OpenGLAdapterState *>(state)->device;
        device.makeCurrent();
        device.driver.finish();
        return VERNON_STATUS_OK;
    } catch (...) {
        setBackendError(error, "OpenGL synchronization threw an exception");
        return VERNON_STATUS_INTERNAL_ERROR;
    }
}

uint64_t resourceIdentity(const void *state) noexcept {
    return reinterpret_cast<uintptr_t>(static_cast<const OpenGLAdapterState *>(state)->device);
}

void invalidateBackend(void *state) noexcept { static_cast<OpenGLAdapterState *>(state)->invalidate(); }

const RhiAdapterBackendOps backendOps{destroyBackend, synchronizeBackend, resourceIdentity, invalidateBackend};

struct PreparedLayout;

struct GraphicsBinding {
    uint32_t slot{};
    VernonRuntimeProviderBindingKind kind{VERNON_RUNTIME_PROVIDER_INLINE_VALUE};
    rhi::opengl::Int location{-1};
    uint32_t valueCount{};
    uint32_t columnCount{};
    uint32_t binding{};
    uint32_t access{};
    uint32_t divisor{};
    uint32_t numericType{};
    bool active{};
};

struct GraphicsVertexAttribute {
    VernonRuntimeProviderVertexAttribute layout{};
    uint32_t divisor{};
};

struct NativeGraphicsBundle {
    VernonRuntimeRhiAdapter *adapter{};
    rhi::opengl::DeviceState *device{};
    rhi::opengl::Uint program{};
    rhi::opengl::Uint vertexArray{};
    rhi::opengl::Uint framebuffer{};
    PreparedLayout *layout{};
    std::vector<GraphicsBinding> bindings;
    std::vector<GraphicsVertexAttribute> vertexAttributes;

    ~NativeGraphicsBundle();
};

struct PreparedShader {
    uint32_t stage{};
    std::string source;
    std::shared_ptr<NativeGraphicsBundle> graphicsBase;
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
    std::atomic<uint32_t> references{1};
    VernonRuntimeRhiAdapter *adapter{};
    std::shared_ptr<NativeGraphicsBundle> native;
    VernonRasterizationState rasterization{};
    VernonDepthStencilState depthStencil{};
    std::vector<VernonColorBlendState> colorBlends;
    uint32_t depthStencilFormat{};
    PreparedLayout *layout{};
    bool compute{};

    ~PreparedPipeline() {
        if (adapter)
            adapter->livePreparedPipelines.fetch_sub(1, std::memory_order_relaxed);
    }
};

struct PreparedBindingSet {
    std::atomic<uint32_t> references{1};
    struct Slot {
        uint32_t slot{};
        VernonRuntimeProviderBindingKind kind{VERNON_RUNTIME_PROVIDER_INLINE_VALUE};
        VernonRuntimeProviderBindingInterface interfaceKind{VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM};
        uint32_t valueCount{};
        uint32_t columnCount{};
        uint32_t flags{};
        uint64_t resource{};
        VernonRuntimeProviderResourceReference resourceReference{};
        uint64_t resourceOffset{};
        VernonRhiImageDimension resourceTarget{VERNON_RHI_IMAGE_2D};
        uint32_t resourceStride{};
        VernonRhiFormat textureFormat{VERNON_RHI_FORMAT_UNDEFINED};
        uint32_t binding{};
        uint32_t stageMask{};
        uint32_t numericType{};
        size_t byteSize{};
        std::vector<uint8_t> storage;
        rhi::opengl::Buffer inlineBuffer;
    };
    rhi::opengl::DeviceState *device{};
    std::vector<Slot> slots;
    std::unordered_map<uint32_t, size_t> slotIndices;
    std::unordered_map<uint32_t, size_t> vertexSlotByBinding;
    std::vector<size_t> valueIndices;
    std::vector<uint8_t> seenSlots;
    std::vector<uint64_t> resolvedValues;
    std::vector<VernonRhiImageDescriptor> resolvedImageDescriptors;
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

uint32_t blendFactor(uint32_t value) {
    constexpr uint32_t factors[]{0, 1, 0x0300, 0x0301, 0x0306, 0x0307, 0x0302, 0x0303, 0x0304, 0x0305};
    return value < std::size(factors) ? factors[value] : 0;
}

uint32_t blendOperation(uint32_t value) {
    constexpr uint32_t operations[]{0x8006, 0x800A, 0x800B, 0x8007, 0x8008};
    return value < std::size(operations) ? operations[value] : 0;
}

uint32_t stencilOperation(uint32_t value) {
    constexpr uint32_t operations[]{0x1E00, 0, 0x1E01, 0x1E02, 0x1E03, 0x150A, 0x8507, 0x8508};
    return value < std::size(operations) ? operations[value] : 0;
}

VernonStatus configureGraphicsState(VernonRuntimeRhiAdapter &adapter,
                                    const VernonRuntimeProviderPipelineDescriptor &descriptor,
                                    PreparedPipeline &pipeline) {
    if (!descriptor.color_format_count)
        return VERNON_STATUS_OK;
    const auto &rasterization = descriptor.rasterization;
    const auto &depthStencil = descriptor.depth_stencil;
    if (descriptor.sample_count != 1 || descriptor.color_format_count > 8 ||
        descriptor.color_blend_count != descriptor.color_format_count ||
        (descriptor.color_blend_count && !descriptor.color_blends) || rasterization.cull_mode > VERNON_RHI_CULL_BACK ||
        rasterization.front_face > VERNON_RHI_FRONT_FACE_CLOCKWISE || rasterization.depth_clamp ||
        rasterization.depth_bias_enabled > 1 || !std::isfinite(rasterization.depth_bias_constant) ||
        !std::isfinite(rasterization.depth_bias_slope) || depthStencil.depth_test > 1 || depthStencil.depth_write > 1 ||
        depthStencil.depth_compare > VERNON_RHI_COMPARE_ALWAYS || depthStencil.stencil_test > 1 ||
        (descriptor.depth_stencil_format && descriptor.depth_stencil_format != rhi::opengl::kDepthComponent32f &&
         descriptor.depth_stencil_format != rhi::opengl::kDepth32fStencil8) ||
        (!descriptor.depth_stencil_format && (depthStencil.depth_test || depthStencil.depth_write)) ||
        (depthStencil.stencil_test && descriptor.depth_stencil_format != rhi::opengl::kDepth32fStencil8) ||
        depthStencil.stencil_read_mask > 0xff || depthStencil.stencil_write_mask > 0xff ||
        !openGLState(adapter).device->driver.blendFuncSeparate ||
        !openGLState(adapter).device->driver.blendEquationSeparate || !openGLState(adapter).device->driver.colorMask ||
        (descriptor.color_format_count > 1 &&
         (!openGLState(adapter).device->driver.enablei || !openGLState(adapter).device->driver.disablei ||
          !openGLState(adapter).device->driver.blendFuncSeparatei ||
          !openGLState(adapter).device->driver.blendEquationSeparatei ||
          !openGLState(adapter).device->driver.colorMaski)))
        return fail(adapter, "OpenGL graphics pipeline contains unsupported graphics state");
    const auto validFace = [](const VernonStencilFaceState &face) {
        return face.stencil_fail <= VERNON_RHI_STENCIL_DECREMENT_WRAP &&
               face.depth_fail <= VERNON_RHI_STENCIL_DECREMENT_WRAP && face.pass <= VERNON_RHI_STENCIL_DECREMENT_WRAP &&
               face.compare <= VERNON_RHI_COMPARE_ALWAYS;
    };
    if (!validFace(depthStencil.front) || !validFace(depthStencil.back))
        return fail(adapter, "OpenGL graphics pipeline contains invalid stencil state");
    for (size_t index = 0; index < descriptor.color_blend_count; ++index) {
        const auto &blend = descriptor.color_blends[index];
        if (blend.blend_enabled > 1 || blend.source_color_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.destination_color_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.source_alpha_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.destination_alpha_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.color_operation > VERNON_RHI_BLEND_MAXIMUM || blend.alpha_operation > VERNON_RHI_BLEND_MAXIMUM ||
            (blend.write_mask & ~VERNON_RHI_COLOR_WRITE_ALL))
            return fail(adapter, "OpenGL graphics pipeline contains invalid blend state");
    }
    pipeline.rasterization = rasterization;
    pipeline.depthStencil = depthStencil;
    pipeline.colorBlends.assign(descriptor.color_blends, descriptor.color_blends + descriptor.color_blend_count);
    pipeline.depthStencilFormat = descriptor.depth_stencil_format;
    return VERNON_STATUS_OK;
}

VernonRuntimeProviderDeviceIdentity getDeviceIdentity(void *data) {
    const auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    const auto &state = openGLState(adapter);
    const uint64_t identity = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(state.device));
    return {0x4f50454e474cu, identity,
            static_cast<uint64_t>(state.device->callbacks.api_version_major) << 32 |
                state.device->callbacks.api_version_minor};
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
            const bool nativeNumericType = source.numeric_type == VERNON_RUNTIME_PROVIDER_F32 ||
                                           source.numeric_type == VERNON_RUNTIME_PROVIDER_I32 ||
                                           source.numeric_type == VERNON_RUNTIME_PROVIDER_U32;
            const bool nativeUniformShape =
                source.vector_count != 0 &&
                ((source.vector_count == 1 && source.element_count >= 1 && source.element_count <= 4) ||
                 (source.numeric_type == VERNON_RUNTIME_PROVIDER_F32 && source.vector_count >= 2 &&
                  source.vector_count <= 4 && source.element_count % source.vector_count == 0 &&
                  source.element_count / source.vector_count >= 2 && source.element_count / source.vector_count <= 4));
            const bool inlineValue =
                source.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
                ((source.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE && source.binding != UINT32_MAX &&
                  source.element_size != 0) ||
                 (source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM && source.name.data &&
                  source.name.size != 0 && nativeNumericType && nativeUniformShape &&
                  source.element_size == source.element_count * sizeof(uint32_t)));
            const bool storageBuffer = source.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
                                       (source.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE ||
                                        source.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX ||
                                        source.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT) &&
                                       (source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE ||
                                        source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM) &&
                                       source.binding != UINT32_MAX && source.element_size != 0;
            const bool uniformBuffer = source.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER &&
                                       source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM &&
                                       source.binding != UINT32_MAX && source.element_size != 0;
            const bool sampledImage = source.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE &&
                                      source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE &&
                                      source.binding != UINT32_MAX && source.name.data && source.name.size != 0;
            const bool storageImage = source.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE &&
                                      source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE &&
                                      source.binding != UINT32_MAX && source.access != 0;
            const bool sampler = source.kind == VERNON_RUNTIME_PROVIDER_SAMPLER &&
                                 source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE &&
                                 source.binding != UINT32_MAX;
            const bool vertexBuffer = source.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER &&
                                      source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_VERTEX_INPUT &&
                                      source.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX &&
                                      source.binding != UINT32_MAX && source.element_size != 0;
            if (!inlineValue && !uniformBuffer && !storageBuffer && !sampledImage && !storageImage && !sampler &&
                !vertexBuffer)
                return fail(adapter, "OpenGL adapter pipeline layout binding " + std::to_string(index) +
                                         " is unsupported (kind=" + std::to_string(source.kind) +
                                         ", interface=" + std::to_string(source.interface_kind) +
                                         ", stage_mask=" + std::to_string(source.stage_mask) +
                                         ", binding=" + std::to_string(source.binding) +
                                         ", element_size=" + std::to_string(source.element_size) +
                                         ", element_count=" + std::to_string(source.element_count) +
                                         ", vector_count=" + std::to_string(source.vector_count) +
                                         ", numeric_type=" + std::to_string(source.numeric_type) +
                                         ", name_size=" + std::to_string(source.name.size) + ")");
            PreparedLayout::Entry entry;
            entry.layout = source;
            if (source.name.data && source.name.size)
                entry.name.assign(source.name.data, source.name.size);
            layout->entries.push_back(std::move(entry));
        }
        rhi::opengl::Int maximumLocations = 0;
        openGLState(adapter).device->driver.getIntegerv(rhi::opengl::kMaxVertexAttribs, &maximumLocations);
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
                                                   openGLState(adapter).device->driver.vertexAttribLPointer != nullptr,
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

std::optional<rhi::opengl::Enum> storageImageFormat(VernonRhiFormat format) {
    switch (format) {
    case VERNON_RHI_FORMAT_R8_UNORM:
        return 0x8229;
    case VERNON_RHI_FORMAT_R16_FLOAT:
        return 0x822D;
    case VERNON_RHI_FORMAT_R32_FLOAT:
        return 0x822E;
    case VERNON_RHI_FORMAT_RG8_UNORM:
        return 0x822B;
    case VERNON_RHI_FORMAT_RGBA8_UNORM:
        return 0x8058;
    case VERNON_RHI_FORMAT_RGBA16_FLOAT:
        return 0x881A;
    case VERNON_RHI_FORMAT_RGBA32_FLOAT:
        return 0x8814;
    default:
        return std::nullopt;
    }
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
    if (!compute && descriptor->color_format_count) {
        auto *first = fromHandle<PreparedShader>(descriptor->shaders[0]);
        auto *second = fromHandle<PreparedShader>(descriptor->shaders[1]);
        std::shared_ptr<NativeGraphicsBundle> base =
            first && second && first->graphicsBase == second->graphicsBase ? first->graphicsBase : nullptr;
        if (base && base->layout == layout) {
            try {
                auto pipeline = std::make_unique<PreparedPipeline>();
                pipeline->adapter = &adapter;
                adapter.livePreparedPipelines.fetch_add(1, std::memory_order_relaxed);
                pipeline->native = std::move(base);
                pipeline->layout = layout;
                const VernonStatus stateStatus = configureGraphicsState(adapter, *descriptor, *pipeline);
                if (stateStatus != VERNON_STATUS_OK)
                    return stateStatus;
                *output = toHandle(pipeline.release());
                adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
                return VERNON_STATUS_OK;
            } catch (const std::bad_alloc &) {
                return fail(adapter, "OpenGL graphics variant allocation failed", VERNON_STATUS_INTERNAL_ERROR);
            }
        }
    }
    std::array<rhi::opengl::Uint, 3> compiled{};
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
        compiled[index] = openGLState(adapter).device->compileShader(kind, shader->source, adapter.error);
        if (!compiled[index]) {
            for (size_t previous = 0; previous < index; ++previous)
                openGLState(adapter).device->driver.deleteShader(compiled[previous]);
            return VERNON_STATUS_INTERNAL_ERROR;
        }
    }
    if ((compute && !hasCompute) || (!compute && (!hasVertex || !hasFragment))) {
        for (size_t index = 0; index < descriptor->shader_count; ++index)
            openGLState(adapter).device->driver.deleteShader(compiled[index]);
        return fail(adapter, "OpenGL pipeline shader stages are incomplete");
    }
    try {
        auto pipeline = std::make_unique<PreparedPipeline>();
        pipeline->adapter = &adapter;
        adapter.livePreparedPipelines.fetch_add(1, std::memory_order_relaxed);
        pipeline->native = std::make_shared<NativeGraphicsBundle>();
        pipeline->native->adapter = &adapter;
        pipeline->native->device = openGLState(adapter).device;
        pipeline->native->layout = layout;
        pipeline->layout = layout;
        pipeline->compute = compute;
        pipeline->native->bindings.reserve(layout->entries.size());
        pipeline->native->program = openGLState(adapter).device->linkProgram(
            {compiled.begin(), compiled.begin() + descriptor->shader_count}, adapter.error);
        if (!pipeline->native->program)
            return VERNON_STATUS_INTERNAL_ERROR;
        const VernonStatus stateStatus = configureGraphicsState(adapter, *descriptor, *pipeline);
        if (stateStatus != VERNON_STATUS_OK)
            return stateStatus;
        if (!compute && !openGLState(adapter).device->createGraphicsObjects(
                            pipeline->native->vertexArray, pipeline->native->framebuffer, adapter.error)) {
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        if (!compute)
            openGLState(adapter).framebufferSignatures.emplace(pipeline->native->framebuffer,
                                                               OpenGLFramebufferSignature{});
        std::vector<rhi::opengl::Int> locations(layout->entries.size(), -1);
        std::vector<uint32_t> activeTextureBindings;
        for (size_t index = 0; index < layout->entries.size(); ++index) {
            const auto &entry = layout->entries[index];
            if (!compute && entry.layout.kind != VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER &&
                entry.layout.kind != VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
                entry.layout.kind != VERNON_RUNTIME_PROVIDER_SAMPLER &&
                entry.layout.kind != VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER) {
                locations[index] = openGLState(adapter).device->driver.getUniformLocation(pipeline->native->program,
                                                                                          entry.name.c_str());
                if (locations[index] >= 0 && entry.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE)
                    activeTextureBindings.push_back(entry.layout.binding);
            }
        }
        for (size_t index = 0; index < layout->entries.size(); ++index) {
            const auto &entry = layout->entries[index];
            const rhi::opengl::Int location = locations[index];
            const bool inactiveNamedUniform = !compute && entry.layout.kind != VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER &&
                                              entry.layout.kind != VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
                                              entry.layout.kind != VERNON_RUNTIME_PROVIDER_SAMPLER &&
                                              entry.layout.kind != VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER &&
                                              location < 0;
            const bool inactiveSampler = !compute && entry.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER &&
                                         std::find(activeTextureBindings.begin(), activeTextureBindings.end(),
                                                   entry.layout.binding) == activeTextureBindings.end();
            // Linked programs legally omit feature-disabled or otherwise dead
            // uniforms. Keep stable variant slots in the binding set, but do
            // not issue GL calls for resources absent from this program.
            const bool active = !inactiveNamedUniform && !inactiveSampler;
            const uint32_t valueCount =
                entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER
                    ? 0
                    : (entry.layout.element_count ? entry.layout.element_count : entry.layout.element_size / 4);
            pipeline->native->bindings.push_back({entry.layout.slot, entry.layout.kind, location, valueCount,
                                                  entry.layout.vector_count, entry.layout.binding, entry.layout.access,
                                                  entry.layout.divisor, entry.layout.numeric_type, active});
        }
        for (const VernonRuntimeProviderVertexAttribute &attribute : layout->vertexAttributes) {
            auto binding = std::find_if(layout->entries.begin(), layout->entries.end(), [&](const auto &entry) {
                return entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER &&
                       entry.layout.binding == attribute.binding;
            });
            if (binding == layout->entries.end())
                return fail(adapter, "OpenGL vertex attribute references a missing binding");
            pipeline->native->vertexAttributes.push_back({attribute, binding->layout.divisor});
        }
        if (!compute && descriptor->color_format_count == 0)
            for (size_t index = 0; index < descriptor->shader_count; ++index)
                fromHandle<PreparedShader>(descriptor->shaders[index])->graphicsBase = pipeline->native;
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
        if (packedUniformBytes(slot.kind, slot.interfaceKind)) {
            if (!value->payload.inline_value.data || value->payload.inline_value.size != slot.byteSize ||
                (value->flags & ~VERNON_RUNTIME_PROVIDER_BINDING_TRANSPOSE) != 0 ||
                (slot.kind != VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
                 (value->flags & VERNON_RUNTIME_PROVIDER_BINDING_TRANSPOSE) != 0) ||
                (slot.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
                 (value->flags & VERNON_RUNTIME_PROVIDER_BINDING_TRANSPOSE) != 0 && slot.columnCount <= 1))
                return fail(adapter, "OpenGL inline binding is invalid");
            bindings.resolvedValues[index] = 0;
        } else {
            const bool defaultResource = (value->flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0;
            const VernonRuntimeProviderResourceReference &resource =
                slot.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE || slot.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE
                    ? value->payload.image.view
                : slot.kind == VERNON_RUNTIME_PROVIDER_SAMPLER ? value->payload.sampler.resource
                                                               : value->payload.buffer.resource;
            const bool image = slot.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                               slot.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
            const uint64_t expectedKind =
                slot.kind == VERNON_RUNTIME_PROVIDER_SAMPLER ? kRhiSamplerResource : kRhiBufferResource;
            const uint64_t native = defaultResource ? 0 : resolveRhiResource(adapter, resource);
            const bool imageBinding = slot.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                                      slot.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
            if ((value->flags & ~VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0 ||
                (!defaultResource && ((image ? !isRhiImageReference(resource)
                                             : (resource.identity & kRhiResourceKindMask) != expectedKind) ||
                                      native == 0)) ||
                (slot.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && !defaultResource &&
                 value->payload.buffer.stride == 0))
                return fail(adapter, "OpenGL resource binding is invalid");
            if (imageBinding) {
                VernonRhiImageDescriptor imageDescriptor{};
                if (!defaultResource && !describeRhiImage(adapter, resource, imageDescriptor))
                    return fail(adapter, "OpenGL image binding metadata is invalid");
                bindings.resolvedImageDescriptors[index] = imageDescriptor;
            }
            bindings.resolvedValues[index] = native;
        }
    }
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        auto &slot = bindings.slots[index];
        const auto *value = &values[bindings.valueIndices[index]];
        if (packedUniformBytes(slot.kind, slot.interfaceKind)) {
            std::memcpy(slot.storage.data(), value->payload.inline_value.data, value->payload.inline_value.size);
            if (slot.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
                slot.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
                slot.stageMask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE) {
                if (!bindings.device->uploadBuffer(slot.inlineBuffer, 0, value->payload.inline_value.data,
                                                   value->payload.inline_value.size, adapter.error))
                    return VERNON_STATUS_INTERNAL_ERROR;
                slot.resource = slot.inlineBuffer.name;
            }
        } else {
            const bool defaultResource = (value->flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0;
            const uint64_t native = bindings.resolvedValues[index];
            const VernonRuntimeProviderResourceReference &resource =
                slot.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE || slot.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE
                    ? value->payload.image.view
                : slot.kind == VERNON_RUNTIME_PROVIDER_SAMPLER ? value->payload.sampler.resource
                                                               : value->payload.buffer.resource;
            slot.resource = native;
            slot.resourceReference = defaultResource ? VernonRuntimeProviderResourceReference{} : resource;
            slot.resourceOffset = resource.offset;
            slot.resourceStride = value->payload.buffer.stride;
            if (slot.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                slot.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE) {
                slot.resourceTarget = bindings.resolvedImageDescriptors[index].dimension;
                slot.textureFormat = bindings.resolvedImageDescriptors[index].format;
            }
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
        bindings->device = openGLState(adapter).device;
        bindings->slots.reserve(layout->entries.size());
        bindings->slotIndices.reserve(layout->entries.size());
        bindings->vertexSlotByBinding.reserve(layout->entries.size());
        bindings->valueIndices.resize(layout->entries.size());
        bindings->seenSlots.resize(layout->entries.size());
        bindings->resolvedValues.resize(layout->entries.size());
        bindings->resolvedImageDescriptors.resize(layout->entries.size());
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
            slot.interfaceKind = entry.layout.interface_kind;
            slot.valueCount =
                entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER
                    ? 0
                    : (entry.layout.element_count ? entry.layout.element_count : entry.layout.element_size / 4);
            slot.columnCount = entry.layout.vector_count;
            slot.binding = entry.layout.binding;
            slot.stageMask = entry.layout.stage_mask;
            slot.numericType = entry.layout.numeric_type;
            slot.byteSize = entry.layout.element_size;
            if (packedUniformBytes(entry.layout.kind, entry.layout.interface_kind)) {
                slot.storage.resize(entry.layout.element_size);
                if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
                    entry.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
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
        (!bindings && !pipeline->native->bindings.empty()) ||
        (bindings && bindings->slots.size() != pipeline->native->bindings.size()) || descriptor->group_count[0] == 0 ||
        descriptor->group_count[1] == 0 || descriptor->group_count[2] == 0 || descriptor->push_constant_size != 0)
        return fail(adapter, "OpenGL adapter received an invalid compute dispatch");
    std::unique_lock<std::mutex> guard;
    if (bindings)
        guard = std::unique_lock<std::mutex>(bindings->mutex);
    auto &device = *pipeline->native->device;
    if (nativeCommandEncoder(adapter, commandEncoder) != reinterpret_cast<uintptr_t>(&device) ||
        commandEncoderRendering(adapter, commandEncoder))
        return fail(adapter, "OpenGL dispatch command encoder is invalid");
    if (!retainCommandObjects(adapter, commandEncoder, *pipeline, bindings))
        return fail(adapter, "OpenGL dispatch could not retain provider objects", VERNON_STATUS_INTERNAL_ERROR);
    if (!retainBindingResources(adapter, commandEncoder, bindings))
        return fail(adapter, "OpenGL dispatch could not retain its resources", VERNON_STATUS_INTERNAL_ERROR);
    device.makeCurrent();
    device.driver.useProgram(pipeline->native->program);
    for (size_t index = 0; index < pipeline->native->bindings.size(); ++index) {
        const auto &binding = pipeline->native->bindings[index];
        const auto &slot = bindings->slots[index];
        if (binding.slot != slot.slot || binding.kind != slot.kind ||
            (binding.kind != VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
             binding.kind != VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE &&
             binding.kind != VERNON_RUNTIME_PROVIDER_INLINE_VALUE))
            return fail(adapter, "OpenGL compute binding set does not match its pipeline");
        if (binding.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE) {
            const auto format = storageImageFormat(slot.textureFormat);
            if (!slot.resource || !format || !device.driver.bindImageTexture)
                return fail(adapter, "OpenGL storage image format or entry point is unavailable");
            const rhi::opengl::Enum access = binding.access == 1   ? rhi::opengl::kReadOnly
                                             : binding.access == 2 ? rhi::opengl::kWriteOnly
                                                                   : rhi::opengl::kReadWrite;
            device.driver.bindImageTexture(binding.binding, static_cast<rhi::opengl::Uint>(slot.resource), 0,
                                           slot.resourceTarget == VERNON_RHI_IMAGE_3D, 0, access, *format);
        } else {
            device.driver.bindBufferBase(rhi::opengl::kShaderStorageBuffer, binding.binding,
                                         static_cast<rhi::opengl::Uint>(slot.resource));
        }
    }
    device.driver.dispatchCompute(descriptor->group_count[0], descriptor->group_count[1], descriptor->group_count[2]);
    for (size_t index = 0; index < pipeline->native->bindings.size(); ++index)
        if (pipeline->native->bindings[index].access != 0 && bindings->slots[index].resourceReference.resource.value &&
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
    if (!validCommonDrawDescriptor(descriptor) || !pipeline || pipeline->compute ||
        (!bindings && descriptor->bindings.value != 0) ||
        (pipeline->native->bindings.empty() ? bindings != nullptr : bindings == nullptr) ||
        (descriptor->index_count != 0 && descriptor->index_type != 0))
        return fail(adapter, "OpenGL adapter received an invalid draw");
    std::unique_lock<std::mutex> bindingGuard;
    if (bindings) {
        bindingGuard = std::unique_lock<std::mutex>(bindings->mutex);
        if (bindings->slots.size() != pipeline->native->bindings.size())
            return fail(adapter, "OpenGL draw binding set does not match its pipeline");
    }
    auto &device = *pipeline->native->device;
    const int claim = claimCommandRendering(adapter, commandEncoder, vernon::rhi::CommandRenderingStateless);
    if (nativeCommandEncoder(adapter, commandEncoder) != reinterpret_cast<uintptr_t>(&device) || claim < 0)
        return fail(adapter, "OpenGL draw command encoder is invalid");
    const bool firstDraw = claim != 0;
    const bool standaloneRendering = !commandEncoderHasRenderingDescriptor(adapter, commandEncoder);
    const uint64_t renderingFramebuffer =
        commandRenderingObject(adapter, commandEncoder, pipeline->native->framebuffer);
    if (!renderingFramebuffer)
        return fail(adapter, "OpenGL command framebuffer state is invalid");
    if (!retainCommandObjects(adapter, commandEncoder, *pipeline, bindings))
        return fail(adapter, "OpenGL draw could not retain provider objects", VERNON_STATUS_INTERNAL_ERROR);
    if (!retainBindingResources(adapter, commandEncoder, bindings))
        return fail(adapter, "OpenGL draw could not retain its bindings", VERNON_STATUS_INTERNAL_ERROR);
    std::array<VernonRhiLoadOperation, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> colorLoads{};
    std::array<std::array<float, 4>, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> colorClears{};
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
    VernonRhiStoreOperation depthStore = static_cast<VernonRhiStoreOperation>(descriptor->depth_store_operation);
    VernonRhiLoadOperation stencilLoad = static_cast<VernonRhiLoadOperation>(descriptor->stencil_load_operation);
    VernonRhiStoreOperation stencilStore = static_cast<VernonRhiStoreOperation>(descriptor->stencil_store_operation);
    float clearDepth = descriptor->clear_depth;
    uint32_t clearStencil = descriptor->clear_stencil;
    (void)commandDepthOperations(adapter, commandEncoder, depthLoad, depthStore, stencilLoad, stencilStore, clearDepth,
                                 clearStencil);
    auto &driver = device.driver;
    std::array<rhi::opengl::Enum, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> drawBuffers{};
    drawBuffers.fill(rhi::opengl::kNone);
    size_t drawBufferCount = 0;
    device.makeCurrent();
    auto &state = openGLState(adapter);
    if (state.framebufferGeneration != device.framebufferGeneration) {
        state.framebufferValid = false;
        state.framebufferSignatures.clear();
        state.framebufferGeneration = device.framebufferGeneration;
    }
    if (firstDraw || !state.programValid || state.program != pipeline->native->program) {
        driver.useProgram(pipeline->native->program);
        state.program = pipeline->native->program;
        state.programValid = true;
    }
    if (firstDraw || !state.vertexArrayValid || state.vertexArray != pipeline->native->vertexArray) {
        driver.bindVertexArray(pipeline->native->vertexArray);
        state.vertexArray = pipeline->native->vertexArray;
        state.vertexArrayValid = true;
    }
    if (pipeline->rasterization.cull_mode == VERNON_RHI_CULL_NONE)
        driver.disable(rhi::opengl::kCullFace);
    else {
        driver.enable(rhi::opengl::kCullFace);
        driver.cullFace(pipeline->rasterization.cull_mode == VERNON_RHI_CULL_FRONT ? 0x0404 : 0x0405);
    }
    driver.frontFace(pipeline->rasterization.front_face == VERNON_RHI_FRONT_FACE_CLOCKWISE ? 0x0900 : 0x0901);
    if (pipeline->rasterization.depth_bias_enabled) {
        driver.enable(0x8037);
        driver.polygonOffset(pipeline->rasterization.depth_bias_slope, pipeline->rasterization.depth_bias_constant);
    } else {
        driver.disable(0x8037);
    }
    if (pipeline->depthStencil.depth_test) {
        driver.enable(rhi::opengl::kDepthTest);
        driver.depthFunc(0x0200 + pipeline->depthStencil.depth_compare);
    } else {
        driver.disable(rhi::opengl::kDepthTest);
    }
    driver.depthMask(pipeline->depthStencil.depth_write != 0);
    if (pipeline->depthStencil.stencil_test) {
        driver.enable(rhi::opengl::kStencilTest);
        const auto applyStencilFace = [&](uint32_t face, const VernonStencilFaceState &state) {
            driver.stencilFuncSeparate(face, 0x0200 + state.compare,
                                       static_cast<rhi::opengl::Int>(descriptor->stencil_reference),
                                       pipeline->depthStencil.stencil_read_mask);
            driver.stencilOpSeparate(face, stencilOperation(state.stencil_fail), stencilOperation(state.depth_fail),
                                     stencilOperation(state.pass));
            driver.stencilMaskSeparate(face, pipeline->depthStencil.stencil_write_mask);
        };
        applyStencilFace(0x0404, pipeline->depthStencil.front);
        applyStencilFace(0x0405, pipeline->depthStencil.back);
    } else {
        driver.disable(rhi::opengl::kStencilTest);
    }
    for (size_t index = 0; index < pipeline->colorBlends.size(); ++index) {
        const auto &blend = pipeline->colorBlends[index];
        if (pipeline->colorBlends.size() == 1) {
            if (blend.blend_enabled)
                driver.enable(rhi::opengl::kBlend);
            else
                driver.disable(rhi::opengl::kBlend);
            driver.blendFuncSeparate(
                blendFactor(blend.source_color_factor), blendFactor(blend.destination_color_factor),
                blendFactor(blend.source_alpha_factor), blendFactor(blend.destination_alpha_factor));
            driver.blendEquationSeparate(blendOperation(blend.color_operation), blendOperation(blend.alpha_operation));
            driver.colorMask((blend.write_mask & VERNON_RHI_COLOR_WRITE_RED) != 0,
                             (blend.write_mask & VERNON_RHI_COLOR_WRITE_GREEN) != 0,
                             (blend.write_mask & VERNON_RHI_COLOR_WRITE_BLUE) != 0,
                             (blend.write_mask & VERNON_RHI_COLOR_WRITE_ALPHA) != 0);
            continue;
        }
        if (blend.blend_enabled)
            driver.enablei(rhi::opengl::kBlend, static_cast<rhi::opengl::Uint>(index));
        else
            driver.disablei(rhi::opengl::kBlend, static_cast<rhi::opengl::Uint>(index));
        driver.blendFuncSeparatei(static_cast<rhi::opengl::Uint>(index), blendFactor(blend.source_color_factor),
                                  blendFactor(blend.destination_color_factor), blendFactor(blend.source_alpha_factor),
                                  blendFactor(blend.destination_alpha_factor));
        driver.blendEquationSeparatei(static_cast<rhi::opengl::Uint>(index), blendOperation(blend.color_operation),
                                      blendOperation(blend.alpha_operation));
        driver.colorMaski(static_cast<rhi::opengl::Uint>(index), (blend.write_mask & VERNON_RHI_COLOR_WRITE_RED) != 0,
                          (blend.write_mask & VERNON_RHI_COLOR_WRITE_GREEN) != 0,
                          (blend.write_mask & VERNON_RHI_COLOR_WRITE_BLUE) != 0,
                          (blend.write_mask & VERNON_RHI_COLOR_WRITE_ALPHA) != 0);
    }
    for (size_t index = 0; index < pipeline->native->bindings.size(); ++index) {
        const auto &binding = pipeline->native->bindings[index];
        const auto &slot = bindings->slots[index];
        if (slot.slot != binding.slot || slot.kind != binding.kind || slot.valueCount != binding.valueCount ||
            slot.columnCount != binding.columnCount || slot.numericType != binding.numericType)
            return fail(adapter, "OpenGL draw binding order does not match its pipeline");
        if (!binding.active)
            continue;
        if (binding.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER) {
            continue;
        } else if (binding.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER) {
            driver.bindBufferBase(rhi::opengl::kUniformBuffer, binding.binding,
                                  static_cast<rhi::opengl::Uint>(slot.resource));
        } else if (binding.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            driver.bindBufferBase(rhi::opengl::kShaderStorageBuffer, binding.binding,
                                  static_cast<rhi::opengl::Uint>(slot.resource));
        } else if (binding.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE) {
            const auto target = slot.resourceTarget == VERNON_RHI_IMAGE_2D   ? rhi::opengl::kTexture2D
                                : slot.resourceTarget == VERNON_RHI_IMAGE_3D ? rhi::opengl::kTexture3D
                                                                             : rhi::opengl::kTextureCubeMap;
            driver.activeTexture(rhi::opengl::kTexture0 + binding.binding);
            driver.bindTexture(target, static_cast<rhi::opengl::Uint>(slot.resource));
            driver.uniform1i(binding.location, static_cast<rhi::opengl::Int>(binding.binding));
        } else if (binding.kind == VERNON_RUNTIME_PROVIDER_SAMPLER) {
            driver.bindSampler(binding.binding, static_cast<rhi::opengl::Uint>(slot.resource));
        } else if (binding.columnCount <= 1) {
            if (binding.numericType == VERNON_RUNTIME_PROVIDER_I32) {
                std::array<rhi::opengl::Int, 4> value{};
                std::memcpy(value.data(), slot.storage.data(), binding.valueCount * sizeof(value[0]));
                if (binding.valueCount == 1)
                    driver.uniform1iv(binding.location, 1, value.data());
                else if (binding.valueCount == 2)
                    driver.uniform2iv(binding.location, 1, value.data());
                else if (binding.valueCount == 3)
                    driver.uniform3iv(binding.location, 1, value.data());
                else
                    driver.uniform4iv(binding.location, 1, value.data());
            } else if (binding.numericType == VERNON_RUNTIME_PROVIDER_U32) {
                std::array<rhi::opengl::Uint, 4> value{};
                std::memcpy(value.data(), slot.storage.data(), binding.valueCount * sizeof(value[0]));
                if (binding.valueCount == 1)
                    driver.uniform1uiv(binding.location, 1, value.data());
                else if (binding.valueCount == 2)
                    driver.uniform2uiv(binding.location, 1, value.data());
                else if (binding.valueCount == 3)
                    driver.uniform3uiv(binding.location, 1, value.data());
                else
                    driver.uniform4uiv(binding.location, 1, value.data());
            } else {
                std::array<float, 4> value{};
                std::memcpy(value.data(), slot.storage.data(), binding.valueCount * sizeof(value[0]));
                if (binding.valueCount == 1)
                    driver.uniform1fv(binding.location, 1, value.data());
                else if (binding.valueCount == 2)
                    driver.uniform2fv(binding.location, 1, value.data());
                else if (binding.valueCount == 3)
                    driver.uniform3fv(binding.location, 1, value.data());
                else
                    driver.uniform4fv(binding.location, 1, value.data());
            }
        } else {
            std::array<float, 16> value{};
            std::memcpy(value.data(), slot.storage.data(), binding.valueCount * sizeof(value[0]));
            const auto transpose =
                static_cast<rhi::opengl::Boolean>((slot.flags & VERNON_RUNTIME_PROVIDER_BINDING_TRANSPOSE) != 0);
            const uint32_t columns = binding.columnCount;
            const uint32_t rows = binding.valueCount / columns;
            if (columns == 2 && rows == 2)
                driver.uniformMatrix2fv(binding.location, 1, transpose, value.data());
            else if (columns == 2 && rows == 3)
                driver.uniformMatrix2x3fv(binding.location, 1, transpose, value.data());
            else if (columns == 2 && rows == 4)
                driver.uniformMatrix2x4fv(binding.location, 1, transpose, value.data());
            else if (columns == 3 && rows == 2)
                driver.uniformMatrix3x2fv(binding.location, 1, transpose, value.data());
            else if (columns == 3 && rows == 3)
                driver.uniformMatrix3fv(binding.location, 1, transpose, value.data());
            else if (columns == 3 && rows == 4)
                driver.uniformMatrix3x4fv(binding.location, 1, transpose, value.data());
            else if (columns == 4 && rows == 2)
                driver.uniformMatrix4x2fv(binding.location, 1, transpose, value.data());
            else if (columns == 4 && rows == 3)
                driver.uniformMatrix4x3fv(binding.location, 1, transpose, value.data());
            else
                driver.uniformMatrix4fv(binding.location, 1, transpose, value.data());
        }
    }
    for (const GraphicsVertexAttribute &attribute : pipeline->native->vertexAttributes) {
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
    std::array<uint64_t, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> colorImages{};
    for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
        const auto &attachment = descriptor->color_attachments[index];
        if (!retainCommandResource(adapter, commandEncoder, attachment.view))
            return fail(adapter, "OpenGL draw could not retain its color attachment", VERNON_STATUS_INTERNAL_ERROR);
        colorImages[index] = resolveRhiResource(adapter, attachment.view);
        if (attachment.location >= VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS || colorImages[index] == 0)
            return fail(adapter, "OpenGL draw contains an invalid color attachment");
        framebufferSignature.values[framebufferSignature.count++] = attachment.location;
        framebufferSignature.values[framebufferSignature.count++] = attachment.view.resource.value;
        const auto target = rhi::opengl::kColorAttachment0 + attachment.location;
        drawBuffers[attachment.location] = target;
        drawBufferCount = std::max(drawBufferCount, static_cast<size_t>(attachment.location) + 1);
    }
    const bool hasDepth = descriptor->depth_stencil_view.resource.value != 0;
    if (hasDepth && !retainCommandResource(adapter, commandEncoder, descriptor->depth_stencil_view))
        return fail(adapter, "OpenGL draw could not retain its depth attachment", VERNON_STATUS_INTERNAL_ERROR);
    const uint64_t depthImage = hasDepth ? resolveRhiResource(adapter, descriptor->depth_stencil_view) : 0;
    if (hasDepth && depthImage == 0)
        return fail(adapter, "OpenGL depth attachment is stale");
    framebufferSignature.values[framebufferSignature.count++] = hasDepth;
    framebufferSignature.values[framebufferSignature.count++] = descriptor->depth_stencil_view.resource.value;
    if (firstDraw || !state.framebufferValid || state.framebuffer != renderingFramebuffer) {
        driver.bindFramebuffer(rhi::opengl::kFramebuffer, static_cast<rhi::opengl::Uint>(renderingFramebuffer));
        state.framebuffer = static_cast<rhi::opengl::Uint>(renderingFramebuffer);
        state.framebufferValid = true;
    }
    auto cachedFramebuffer = state.framebufferSignatures.find(static_cast<rhi::opengl::Uint>(renderingFramebuffer));
    if (cachedFramebuffer == state.framebufferSignatures.end() ||
        !(cachedFramebuffer->second == framebufferSignature)) {
        for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
            const auto &attachment = descriptor->color_attachments[index];
            driver.framebufferTexture2D(rhi::opengl::kFramebuffer, rhi::opengl::kColorAttachment0 + attachment.location,
                                        rhi::opengl::kTexture2D, static_cast<rhi::opengl::Uint>(colorImages[index]), 0);
        }
        const rhi::opengl::Enum depthAttachmentPoint = pipeline->depthStencilFormat == rhi::opengl::kDepth32fStencil8
                                                           ? rhi::opengl::kDepthStencilAttachment
                                                           : rhi::opengl::kDepthAttachment;
        driver.framebufferTexture2D(rhi::opengl::kFramebuffer, rhi::opengl::kDepthAttachment, rhi::opengl::kTexture2D,
                                    0, 0);
        driver.framebufferTexture2D(rhi::opengl::kFramebuffer, rhi::opengl::kDepthStencilAttachment,
                                    rhi::opengl::kTexture2D, 0, 0);
        driver.framebufferTexture2D(rhi::opengl::kFramebuffer, depthAttachmentPoint, rhi::opengl::kTexture2D,
                                    hasDepth ? static_cast<rhi::opengl::Uint>(depthImage) : 0, 0);
        driver.drawBuffers(static_cast<rhi::opengl::Size>(drawBufferCount), drawBuffers.data());
        if (driver.checkFramebufferStatus(rhi::opengl::kFramebuffer) != rhi::opengl::kFramebufferComplete)
            return fail(adapter, "OpenGL draw framebuffer is incomplete", VERNON_STATUS_INTERNAL_ERROR);
        state.framebufferSignatures.insert_or_assign(static_cast<rhi::opengl::Uint>(renderingFramebuffer),
                                                     framebufferSignature);
    }
    if (firstDraw || !state.viewportValid ||
        !std::equal(std::begin(descriptor->viewport), std::end(descriptor->viewport), state.viewport.begin())) {
        driver.viewport(static_cast<rhi::opengl::Int>(descriptor->viewport[0]),
                        static_cast<rhi::opengl::Int>(descriptor->viewport[1]),
                        static_cast<rhi::opengl::Size>(descriptor->viewport[2]),
                        static_cast<rhi::opengl::Size>(descriptor->viewport[3]));
        std::copy(std::begin(descriptor->viewport), std::end(descriptor->viewport), state.viewport.begin());
        state.viewportValid = true;
    }
    for (size_t index = 0; index < descriptor->color_attachment_count; ++index)
        if (firstDraw && colorLoads[index] == VERNON_RHI_LOAD_CLEAR)
            driver.clearBufferfv(rhi::opengl::kColor,
                                 static_cast<rhi::opengl::Int>(descriptor->color_attachments[index].location),
                                 colorClears[index].data());
    if (hasDepth && firstDraw) {
        if (pipeline->depthStencilFormat != rhi::opengl::kDepth32fStencil8 &&
            (stencilLoad != VERNON_RHI_LOAD_DISCARD || stencilStore != VERNON_RHI_STORE_DISCARD || clearStencil))
            return fail(adapter, "OpenGL draw requests stencil operations for a depth-only attachment");
        if (depthLoad == VERNON_RHI_LOAD_CLEAR)
            driver.depthMask(1);
        if (stencilLoad == VERNON_RHI_LOAD_CLEAR) {
            driver.stencilMaskSeparate(0x0404, UINT32_MAX);
            driver.stencilMaskSeparate(0x0405, UINT32_MAX);
        }
        if (depthLoad == VERNON_RHI_LOAD_CLEAR && stencilLoad == VERNON_RHI_LOAD_CLEAR)
            driver.clearBufferfi(rhi::opengl::kDepthStencil, 0, clearDepth,
                                 static_cast<rhi::opengl::Int>(clearStencil));
        else {
            if (depthLoad == VERNON_RHI_LOAD_CLEAR)
                driver.clearBufferfv(rhi::opengl::kDepth, 0, &clearDepth);
            if (stencilLoad == VERNON_RHI_LOAD_CLEAR) {
                const rhi::opengl::Int stencil = static_cast<rhi::opengl::Int>(clearStencil);
                driver.clearBufferiv(rhi::opengl::kStencil, 0, &stencil);
            }
        }
        driver.depthMask(pipeline->depthStencil.depth_write != 0);
        if (stencilLoad == VERNON_RHI_LOAD_CLEAR) {
            driver.stencilMaskSeparate(0x0404, pipeline->depthStencil.stencil_write_mask);
            driver.stencilMaskSeparate(0x0405, pipeline->depthStencil.stencil_write_mask);
        }
    }
    if (driver.scissor) {
        if (firstDraw || !state.scissorValid ||
            !std::equal(std::begin(descriptor->scissor), std::end(descriptor->scissor), state.scissor.begin())) {
            driver.enable(rhi::opengl::kScissorTest);
            driver.scissor(static_cast<rhi::opengl::Int>(descriptor->scissor[0]),
                           static_cast<rhi::opengl::Int>(descriptor->scissor[1]),
                           static_cast<rhi::opengl::Size>(descriptor->scissor[2]),
                           static_cast<rhi::opengl::Size>(descriptor->scissor[3]));
            std::copy(std::begin(descriptor->scissor), std::end(descriptor->scissor), state.scissor.begin());
            state.scissorValid = true;
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
        std::array<rhi::opengl::Enum, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS + 2> discarded{};
        size_t discardedCount = 0;
        for (size_t index = 0; index < descriptor->color_attachment_count; ++index)
            if (descriptor->color_attachments[index].store_operation == VERNON_RUNTIME_PROVIDER_STORE_DISCARD)
                discarded[discardedCount++] =
                    rhi::opengl::kColorAttachment0 + descriptor->color_attachments[index].location;
        if (hasDepth && descriptor->depth_store_operation == VERNON_RUNTIME_PROVIDER_STORE_DISCARD)
            discarded[discardedCount++] = rhi::opengl::kDepthAttachment;
        if (hasDepth && pipeline->depthStencilFormat == rhi::opengl::kDepth32fStencil8 &&
            descriptor->stencil_store_operation == VERNON_RUNTIME_PROVIDER_STORE_DISCARD)
            discarded[discardedCount++] = rhi::opengl::kStencilAttachment;
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
NativeGraphicsBundle::~NativeGraphicsBundle() {
    if (!device)
        return;
    auto &state = openGLState(*adapter);
    if (framebuffer) {
        state.framebufferSignatures.erase(framebuffer);
        if (state.framebufferValid && state.framebuffer == framebuffer)
            state.framebufferValid = false;
    }
    if (state.programValid && state.program == program)
        state.programValid = false;
    if (state.vertexArrayValid && state.vertexArray == vertexArray)
        state.vertexArrayValid = false;
    device->destroyGraphicsObjects(vertexArray, framebuffer);
    device->destroyProgram(program);
}
void releaseCommandPipeline(void *context, uint64_t) {
    auto *pipeline = static_cast<PreparedPipeline *>(context);
    if (!pipeline || pipeline->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
        return;
    delete pipeline;
}
void destroyPipeline(void *, VernonRuntimeProviderObject handle) {
    releaseCommandPipeline(fromHandle<PreparedPipeline>(handle), 0);
}

} // namespace

void initializeOpenGLProvider(VernonRuntimeRhiAdapter &adapter) {
    adapter.provider.struct_size = sizeof(adapter.provider);
    adapter.provider.abi_version = VERNON_PROGRAM_VERSION;
    adapter.provider.user_data = &adapter;
    adapter.provider.get_capabilities = getCapabilities;
    adapter.provider.get_device_identity = getDeviceIdentity;
    adapter.provider.prepare_shader = prepareShader;
    adapter.provider.prepare_pipeline_layout = prepareLayout;
    adapter.provider.prepare_pipeline = preparePipeline;
    adapter.provider.retain_resource = retainResource;
    adapter.provider.release_resource = releaseResource;
    adapter.provider.describe_image = describeProviderImageCallback;
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

namespace vernon::runtime {

VernonRuntimeRhiAdapter *createOpenGLRhiAdapter(VernonRhiDevice device, VernonRhiBackend backend) {
    if (backend != VERNON_RHI_BACKEND_OPENGL && backend != VERNON_RHI_BACKEND_OPENGL_ES)
        return nullptr;
    auto *deviceState = static_cast<rhi::opengl::DeviceState *>(rhi::deviceState(device, backend));
    if (!deviceState)
        return nullptr;
    auto state = std::unique_ptr<rhi_adapter::OpenGLAdapterState>(new (std::nothrow) rhi_adapter::OpenGLAdapterState());
    auto adapter = std::unique_ptr<VernonRuntimeRhiAdapter>(new (std::nothrow) VernonRuntimeRhiAdapter());
    if (!state || !adapter)
        return nullptr;
    state->device = deviceState;
    adapter->rhiBackend = backend;
    if (!adapter->backend.adopt(state.get(), &rhi_adapter::backendOps))
        return nullptr;
    [[maybe_unused]] auto *adoptedState = state.release();
    rhi_adapter::initializeOpenGLProvider(*adapter);
    return adapter.release();
}

} // namespace vernon::runtime
