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

RhiAdapterResult<void> synchronizeBackend(void *state, std::string &error) noexcept {
    try {
        auto &device = *static_cast<OpenGLAdapterState *>(state)->device;
        device.makeCurrent();
        device.driver.finish();
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (...) {
        return RhiAdapterResult<void>{vernon::err(
            vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure, {"opengl_synchronize", 0, 0}})};
    }
}

uint64_t resourceIdentity(const void *state) noexcept {
    return reinterpret_cast<uintptr_t>(static_cast<const OpenGLAdapterState *>(state)->device);
}

const RhiAdapterBackendOps backendOps{destroyBackend, synchronizeBackend, resourceIdentity};

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
    std::unordered_map<uint32_t, size_t> vertexSlotByBinding;

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

RhiAdapterResult<void> retainCommandObjects(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                            PreparedPipeline &pipeline, PreparedBindingSet *bindings) {
    pipeline.references.fetch_add(1, std::memory_order_relaxed);
    if (!deferCommandCleanup(adapter, encoder, &pipeline, 0, releaseCommandPipeline)) {
        releaseCommandPipeline(&pipeline, 0);
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"opengl_command_pipeline_cleanup_registration_failed", 0, 0}})};
    }
    if (bindings) {
        bindings->references.fetch_add(1, std::memory_order_relaxed);
        if (!deferCommandCleanup(adapter, encoder, bindings, 0, releaseCommandBindings)) {
            releaseCommandBindings(bindings, 0);
            return RhiAdapterResult<void>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                                  {"opengl_command_bindings_cleanup_registration_failed", 0, 0}})};
        }
    }
    return RhiAdapterResult<void>{vernon::ok()};
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

RhiAdapterResult<void> configureGraphicsStateResult(VernonRuntimeRhiAdapter &adapter,
                                                    const VernonRuntimeProviderPipelineDescriptor &descriptor,
                                                    PreparedPipeline &pipeline) {
    if (!descriptor.color_format_count)
        return RhiAdapterResult<void>{vernon::ok()};
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
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                              {"opengl_graphics_pipeline_contains_unsupported_graphics_state", 0, 0}})};
    const auto validFace = [](const VernonStencilFaceState &face) {
        return face.stencil_fail <= VERNON_RHI_STENCIL_DECREMENT_WRAP &&
               face.depth_fail <= VERNON_RHI_STENCIL_DECREMENT_WRAP && face.pass <= VERNON_RHI_STENCIL_DECREMENT_WRAP &&
               face.compare <= VERNON_RHI_COMPARE_ALWAYS;
    };
    if (!validFace(depthStencil.front) || !validFace(depthStencil.back))
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                              {"opengl_graphics_pipeline_contains_invalid_stencil_state", 0, 0}})};
    for (size_t index = 0; index < descriptor.color_blend_count; ++index) {
        const auto &blend = descriptor.color_blends[index];
        if (blend.blend_enabled > 1 || blend.source_color_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.destination_color_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.source_alpha_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.destination_alpha_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.color_operation > VERNON_RHI_BLEND_MAXIMUM || blend.alpha_operation > VERNON_RHI_BLEND_MAXIMUM ||
            (blend.write_mask & ~VERNON_RHI_COLOR_WRITE_ALL))
            return RhiAdapterResult<void>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                  {"opengl_graphics_pipeline_contains_invalid_blend_state", 0, 0}})};
    }
    pipeline.rasterization = rasterization;
    pipeline.depthStencil = depthStencil;
    pipeline.colorBlends.assign(descriptor.color_blends, descriptor.color_blends + descriptor.color_blend_count);
    pipeline.depthStencilFormat = descriptor.depth_stencil_format;
    return RhiAdapterResult<void>{vernon::ok()};
}

VernonRuntimeProviderDeviceIdentity getDeviceIdentity(void *data) {
    const auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    const auto &state = openGLState(adapter);
    const uint64_t identity = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(state.device));
    return {0x4f50454e474cu, identity,
            static_cast<uint64_t>(state.device->callbacks.api_version_major) << 32 |
                state.device->callbacks.api_version_minor};
}

RhiAdapterResult<void> prepareShaderResult(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !descriptor->data ||
        descriptor->size == 0 ||
        (descriptor->stage != VERNON_RUNTIME_PROVIDER_STAGE_VERTEX &&
         descriptor->stage != VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT &&
         descriptor->stage != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"opengl_adapter_received_an_invalid_shader", 0, 0}})};
    try {
        auto shader = std::make_unique<PreparedShader>();
        shader->stage = descriptor->stage;
        shader->source.assign(static_cast<const char *>(descriptor->data), descriptor->size);
        *output = toHandle(shader.release());
        adapter.shaderPreparations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"opengl_shader_preparation_ran_out_of_memory", 0, 0}})};
    }
}

RhiAdapterResult<void> prepareLayoutResult(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        (descriptor->binding_count != 0 && !descriptor->bindings) ||
        (descriptor->vertex_attribute_count != 0 && !descriptor->vertex_attributes))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"opengl_adapter_received_an_invalid_pipeline_layout", 0, 0}})};
    auto layout = std::unique_ptr<PreparedLayout>(new (std::nothrow) PreparedLayout());
    if (!layout)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"opengl_layout_preparation_ran_out_of_memory", 0, 0}})};
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
                return RhiAdapterResult<void>{vernon::err(
                    vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                          {"opengl_validate_pipeline_layout_binding", index, source.kind}})};
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
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::InvalidArgument,
                    {"opengl_validate_vertex_attribute_binding", attribute.binding, attribute.location}})};
            if (!validateVertexAttributeCapability(VertexAttributeBackend::OpenGL, attribute, locationLimit,
                                                   openGLState(adapter).device->driver.vertexAttribLPointer != nullptr,
                                                   capabilityDiagnostic))
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::InvalidArgument,
                    {"opengl_validate_vertex_attribute_capability", attribute.location, attribute.binding}})};
            layout->vertexAttributes.push_back(attribute);
        }
    } catch (const std::bad_alloc &) {
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"opengl_layout_preparation_ran_out_of_memory", 0, 0}})};
    }
    *output = toHandle(layout.release());
    adapter.layoutPreparations.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
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

RhiAdapterResult<void> preparePipelineResult(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                                             VernonRuntimeProviderObject *output) {
    std::string backendDiagnostic;
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    const bool compute = descriptor && descriptor->kind == VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        (compute ? descriptor->shader_count != 1
                 : descriptor->kind != VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE || descriptor->shader_count != 2) ||
        !descriptor->shaders || !layout)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"opengl_adapter_received_an_invalid_pipeline", 0, 0}})};
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
                auto stateStatus = configureGraphicsStateResult(adapter, *descriptor, *pipeline);
                if (!stateStatus)
                    return stateStatus;
                *output = toHandle(pipeline.release());
                adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
                return RhiAdapterResult<void>{vernon::ok()};
            } catch (const std::bad_alloc &) {
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::BackendFailure, {"opengl_graphics_variant_allocation_failed", 0, 0}})};
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
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::InvalidArgument, {"opengl_pipeline_contains_an_invalid_shader", 0, 0}})};
        const auto kind = shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX     ? rhi::opengl::kVertexShader
                          : shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT ? rhi::opengl::kFragmentShader
                                                                                    : rhi::opengl::kComputeShader;
        hasVertex |= shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX;
        hasFragment |= shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT;
        hasCompute |= shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
        compiled[index] = openGLState(adapter).device->compileShader(kind, shader->source, backendDiagnostic);
        if (!compiled[index]) {
            for (size_t previous = 0; previous < index; ++previous)
                openGLState(adapter).device->driver.deleteShader(compiled[previous]);
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::BackendFailure,
                {"opengl_compile_shader", static_cast<uint64_t>(kind), static_cast<uint32_t>(index)}})};
        }
    }
    if ((compute && !hasCompute) || (!compute && (!hasVertex || !hasFragment))) {
        for (size_t index = 0; index < descriptor->shader_count; ++index)
            openGLState(adapter).device->driver.deleteShader(compiled[index]);
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"opengl_pipeline_shader_stages_are_incomplete", 0, 0}})};
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
            {compiled.begin(), compiled.begin() + descriptor->shader_count}, backendDiagnostic);
        if (!pipeline->native->program)
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::BackendFailure, {"opengl_link_program", descriptor->shader_count, 0}})};
        auto stateStatus = configureGraphicsStateResult(adapter, *descriptor, *pipeline);
        if (!stateStatus)
            return stateStatus;
        if (!compute && !openGLState(adapter).device->createGraphicsObjects(
                            pipeline->native->vertexArray, pipeline->native->framebuffer, backendDiagnostic)) {
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                                                            {"opengl_create_graphics_objects", 0, 0}})};
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
                return RhiAdapterResult<void>{
                    vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                      {"opengl_vertex_attribute_references_a_missing_binding", 0, 0}})};
            pipeline->native->vertexAttributes.push_back({attribute, binding->layout.divisor});
        }
        if (!compute && descriptor->color_format_count == 0)
            for (size_t index = 0; index < descriptor->shader_count; ++index)
                fromHandle<PreparedShader>(descriptor->shaders[index])->graphicsBase = pipeline->native;
        *output = toHandle(pipeline.release());
        adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"opengl_pipeline_preparation_ran_out_of_memory", 0, 0}})};
    }
}

RhiAdapterResult<void> initializeBindingsResult(VernonRuntimeRhiAdapter &adapter, PreparedBindingSet &bindings,
                                                const std::unordered_map<uint32_t, size_t> &slotIndices,
                                                const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    std::string backendDiagnostic;
    if (valueCount != bindings.slots.size() || (valueCount != 0 && !values))
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                              {"opengl_binding_values_do_not_match_the_prepared_layout", 0, 0}})};
    std::vector<size_t> valueIndices(bindings.slots.size());
    std::vector<uint8_t> seenSlots(bindings.slots.size());
    std::vector<uint64_t> resolvedValues(bindings.slots.size());
    std::vector<VernonRhiImageDescriptor> resolvedImageDescriptors(bindings.slots.size());
    for (size_t index = 0; index < valueCount; ++index) {
        const auto found = slotIndices.find(values[index].slot);
        if (found == slotIndices.end() || seenSlots[found->second])
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::InvalidArgument, {"opengl_binding_slot_is_invalid_or_duplicated", 0, 0}})};
        seenSlots[found->second] = 1;
        valueIndices[found->second] = index;
    }
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        const auto &slot = bindings.slots[index];
        const auto *value = &values[valueIndices[index]];
        if (value->kind != slot.kind)
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::InvalidArgument, {"opengl_binding_slot_or_kind_is_invalid", 0, 0}})};
        if (packedUniformBytes(slot.kind, slot.interfaceKind)) {
            if (!value->payload.inline_value.data || value->payload.inline_value.size != slot.byteSize ||
                (value->flags & ~VERNON_RUNTIME_PROVIDER_BINDING_TRANSPOSE) != 0 ||
                (slot.kind != VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
                 (value->flags & VERNON_RUNTIME_PROVIDER_BINDING_TRANSPOSE) != 0) ||
                (slot.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
                 (value->flags & VERNON_RUNTIME_PROVIDER_BINDING_TRANSPOSE) != 0 && slot.columnCount <= 1))
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::InvalidArgument, {"opengl_inline_binding_is_invalid", 0, 0}})};
            resolvedValues[index] = 0;
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
            uint64_t native = 0;
            if (!defaultResource) {
                auto resolved = resolveRhiResource(adapter, resource);
                if (!resolved)
                    return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
                native = std::move(resolved).value();
            }
            const bool imageBinding = slot.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                                      slot.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
            if ((value->flags & ~VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0 ||
                (!defaultResource && ((image ? !isRhiImageReference(resource)
                                             : (resource.identity & kRhiResourceKindMask) != expectedKind) ||
                                      native == 0)) ||
                (slot.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && !defaultResource &&
                 value->payload.buffer.stride == 0))
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::InvalidArgument, {"opengl_resource_binding_is_invalid", 0, 0}})};
            if (imageBinding) {
                VernonRhiImageDescriptor imageDescriptor{};
                if (!defaultResource) {
                    auto described = describeRhiImage(adapter, resource, imageDescriptor);
                    if (!described)
                        return RhiAdapterResult<void>{vernon::err(std::move(described).error())};
                }
                resolvedImageDescriptors[index] = imageDescriptor;
            }
            resolvedValues[index] = native;
        }
    }
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        auto &slot = bindings.slots[index];
        const auto *value = &values[valueIndices[index]];
        if (packedUniformBytes(slot.kind, slot.interfaceKind)) {
            std::memcpy(slot.storage.data(), value->payload.inline_value.data, value->payload.inline_value.size);
            if (slot.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
                slot.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
                slot.stageMask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE) {
                if (!bindings.device->uploadBuffer(slot.inlineBuffer, 0, value->payload.inline_value.data,
                                                   value->payload.inline_value.size, backendDiagnostic))
                    return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                        vernon::ProviderErrorCode::BackendFailure,
                        {"opengl_upload_inline_binding", value->payload.inline_value.size, slot.slot}})};
                slot.resource = slot.inlineBuffer.name;
            }
        } else {
            const bool defaultResource = (value->flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0;
            const uint64_t native = resolvedValues[index];
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
                slot.resourceTarget = resolvedImageDescriptors[index].dimension;
                slot.textureFormat = resolvedImageDescriptors[index].format;
            }
        }
        slot.flags = value->flags;
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> retainBindingResources(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                              const PreparedBindingSet *bindings) {
    if (!bindings)
        return RhiAdapterResult<void>{vernon::ok()};
    for (const auto &slot : bindings->slots) {
        if (!slot.resourceReference.resource.value)
            continue;
        auto retained = retainCommandResource(adapter, encoder, slot.resourceReference);
        if (!retained)
            return retained;
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> createBindingsResult(void *data, const VernonRuntimeProviderBindingSetDescriptor *descriptor,
                                            VernonRuntimeProviderObject *output) {
    std::string backendDiagnostic;
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !layout)
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                              {"opengl_adapter_received_an_invalid_binding_set_descriptor", 0, 0}})};
    try {
        auto bindings = std::make_unique<PreparedBindingSet>();
        bindings->device = openGLState(adapter).device;
        std::unordered_map<uint32_t, size_t> slotIndices;
        bindings->slots.reserve(layout->entries.size());
        slotIndices.reserve(layout->entries.size());
        bindings->vertexSlotByBinding.reserve(layout->entries.size());
        for (size_t index = 0; index < layout->entries.size(); ++index) {
            const auto &entry = layout->entries[index];
            PreparedBindingSet::Slot slot;
            slot.slot = entry.layout.slot;
            if (!slotIndices.emplace(slot.slot, index).second)
                return RhiAdapterResult<void>{
                    vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                      {"opengl_binding_layout_contains_duplicate_slots", 0, 0}})};
            if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER &&
                !bindings->vertexSlotByBinding.emplace(entry.layout.binding, index).second)
                return RhiAdapterResult<void>{vernon::err(
                    vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                          {"opengl_binding_layout_contains_duplicate_vertex_buffer_bindings", 0, 0}})};
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
                    if (!bindings->device->createBuffer(slot.inlineBuffer, entry.layout.element_size,
                                                        backendDiagnostic))
                        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                            vernon::ProviderErrorCode::BackendFailure,
                            {"opengl_create_inline_binding_buffer", entry.layout.element_size, entry.layout.slot}})};
                }
            }
            bindings->slots.push_back(std::move(slot));
        }
        auto status =
            initializeBindingsResult(adapter, *bindings, slotIndices, descriptor->values, descriptor->value_count);
        if (!status)
            return status;
        *output = toHandle(bindings.release());
        adapter.bindingCreations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"opengl_binding_preparation_ran_out_of_memory", 0, 0}})};
    }
}

RhiAdapterResult<void> encodeDispatchResult(void *data, VernonRuntimeProviderObject commandEncoder,
                                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *pipeline = descriptor ? fromHandle<PreparedPipeline>(descriptor->pipeline) : nullptr;
    auto *bindings = descriptor ? fromHandle<PreparedBindingSet>(descriptor->bindings) : nullptr;
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !pipeline || !pipeline->compute ||
        (!bindings && !pipeline->native->bindings.empty()) ||
        (bindings && bindings->slots.size() != pipeline->native->bindings.size()) || descriptor->group_count[0] == 0 ||
        descriptor->group_count[1] == 0 || descriptor->group_count[2] == 0 || descriptor->push_constant_size != 0)
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                              {"opengl_adapter_received_an_invalid_compute_dispatch", 0, 0}})};
    auto &device = *pipeline->native->device;
    auto nativeCommand = nativeCommandEncoder(adapter, commandEncoder);
    if (!nativeCommand)
        return RhiAdapterResult<void>{vernon::err(std::move(nativeCommand).error())};
    if (std::move(nativeCommand).value() != reinterpret_cast<uintptr_t>(&device) ||
        commandEncoderRendering(adapter, commandEncoder))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"opengl_dispatch_command_encoder_is_invalid", 0, 0}})};
    if (!retainCommandObjects(adapter, commandEncoder, *pipeline, bindings))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"opengl_dispatch_could_not_retain_provider_objects", 0, 0}})};
    if (!retainBindingResources(adapter, commandEncoder, bindings))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"opengl_dispatch_could_not_retain_its_resources", 0, 0}})};
    device.makeCurrent();
    device.driver.useProgram(pipeline->native->program);
    for (size_t index = 0; index < pipeline->native->bindings.size(); ++index) {
        const auto &binding = pipeline->native->bindings[index];
        const auto &slot = bindings->slots[index];
        if (binding.slot != slot.slot || binding.kind != slot.kind ||
            (binding.kind != VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
             binding.kind != VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE &&
             binding.kind != VERNON_RUNTIME_PROVIDER_INLINE_VALUE))
            return RhiAdapterResult<void>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                  {"opengl_compute_binding_set_does_not_match_its_pipeline", 0, 0}})};
        if (binding.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE) {
            const auto format = storageImageFormat(slot.textureFormat);
            if (!slot.resource || !format || !device.driver.bindImageTexture)
                return RhiAdapterResult<void>{vernon::err(
                    vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                          {"opengl_storage_image_format_or_entry_point_is_unavailable", 0, 0}})};
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
            return RhiAdapterResult<void>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                                  {"opengl_dispatch_could_not_track_a_written_resource", 0, 0}})};
    if (!recordProviderCommand(adapter, commandEncoder, false))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"opengl_dispatch_command_encoder_state_changed", 0, 0}})};
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> encodeDrawResult(void *data, VernonRuntimeProviderObject commandEncoder,
                                        const VernonRuntimeProviderDrawDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *pipeline = descriptor ? fromHandle<PreparedPipeline>(descriptor->pipeline) : nullptr;
    auto *bindings = descriptor ? fromHandle<PreparedBindingSet>(descriptor->bindings) : nullptr;
    if (!validCommonDrawDescriptor(descriptor) || !pipeline || pipeline->compute ||
        (!bindings && descriptor->bindings.value != 0) ||
        (pipeline->native->bindings.empty() ? bindings != nullptr : bindings == nullptr) ||
        (descriptor->index_count != 0 && descriptor->index_type != 0))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"opengl_adapter_received_an_invalid_draw", 0, 0}})};
    if (bindings) {
        if (bindings->slots.size() != pipeline->native->bindings.size())
            return RhiAdapterResult<void>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                  {"opengl_draw_binding_set_does_not_match_its_pipeline", 0, 0}})};
    }
    auto &device = *pipeline->native->device;
    auto renderingClaim = claimCommandRendering(adapter, commandEncoder, vernon::rhi::CommandRenderingStateless);
    if (!renderingClaim)
        return RhiAdapterResult<void>{vernon::err(std::move(renderingClaim).error())};
    const int claim = std::move(renderingClaim).value();
    auto nativeCommand = nativeCommandEncoder(adapter, commandEncoder);
    if (!nativeCommand)
        return RhiAdapterResult<void>{vernon::err(std::move(nativeCommand).error())};
    if (std::move(nativeCommand).value() != reinterpret_cast<uintptr_t>(&device) || claim < 0)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"opengl_draw_command_encoder_is_invalid", 0, 0}})};
    const bool firstDraw = claim != 0;
    const bool standaloneRendering = !commandEncoderHasRenderingDescriptor(adapter, commandEncoder);
    auto renderingObject = commandRenderingObject(adapter, commandEncoder, pipeline->native->framebuffer);
    if (!renderingObject)
        return RhiAdapterResult<void>{vernon::err(std::move(renderingObject).error())};
    const uint64_t renderingFramebuffer = std::move(renderingObject).value();
    if (!retainCommandObjects(adapter, commandEncoder, *pipeline, bindings))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"opengl_draw_could_not_retain_provider_objects", 0, 0}})};
    if (!retainBindingResources(adapter, commandEncoder, bindings))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"opengl_draw_could_not_retain_its_bindings", 0, 0}})};
    std::array<VernonRhiLoadOperation, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> colorLoads{};
    std::array<std::array<float, 4>, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> colorClears{};
    for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
        colorLoads[index] = static_cast<VernonRhiLoadOperation>(descriptor->color_attachments[index].load_operation);
        std::copy(std::begin(descriptor->color_attachments[index].clear_color),
                  std::end(descriptor->color_attachments[index].clear_color), colorClears[index].begin());
        VernonRhiStoreOperation ignoredStore =
            static_cast<VernonRhiStoreOperation>(descriptor->color_attachments[index].store_operation);
        auto operations = commandColorOperations(adapter, commandEncoder, index, colorLoads[index], ignoredStore,
                                                 colorClears[index].data());
        if (!operations)
            return RhiAdapterResult<void>{vernon::err(std::move(operations).error())};
    }
    VernonRhiLoadOperation depthLoad = static_cast<VernonRhiLoadOperation>(descriptor->depth_load_operation);
    VernonRhiStoreOperation depthStore = static_cast<VernonRhiStoreOperation>(descriptor->depth_store_operation);
    VernonRhiLoadOperation stencilLoad = static_cast<VernonRhiLoadOperation>(descriptor->stencil_load_operation);
    VernonRhiStoreOperation stencilStore = static_cast<VernonRhiStoreOperation>(descriptor->stencil_store_operation);
    float clearDepth = descriptor->clear_depth;
    uint32_t clearStencil = descriptor->clear_stencil;
    auto depthOperations = commandDepthOperations(adapter, commandEncoder, depthLoad, depthStore, stencilLoad,
                                                  stencilStore, clearDepth, clearStencil);
    if (!depthOperations)
        return RhiAdapterResult<void>{vernon::err(std::move(depthOperations).error())};
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
            return RhiAdapterResult<void>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                  {"opengl_draw_binding_order_does_not_match_its_pipeline", 0, 0}})};
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
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::InvalidArgument, {"opengl_vertex_attribute_binding_is_missing", 0, 0}})};
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
            return RhiAdapterResult<void>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                                  {"opengl_draw_could_not_retain_its_color_attachment", 0, 0}})};
        auto resolved = resolveCommandRhiResource(adapter, commandEncoder, attachment.view);
        if (!resolved)
            return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
        colorImages[index] = std::move(resolved).value();
        if (attachment.location >= VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS || colorImages[index] == 0)
            return RhiAdapterResult<void>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                  {"opengl_draw_contains_an_invalid_color_attachment", 0, 0}})};
        framebufferSignature.values[framebufferSignature.count++] = attachment.location;
        framebufferSignature.values[framebufferSignature.count++] = attachment.view.resource.value;
        const auto target = rhi::opengl::kColorAttachment0 + attachment.location;
        drawBuffers[attachment.location] = target;
        drawBufferCount = std::max(drawBufferCount, static_cast<size_t>(attachment.location) + 1);
    }
    const bool hasDepth = descriptor->depth_stencil_view.resource.value != 0;
    if (hasDepth && !retainCommandResource(adapter, commandEncoder, descriptor->depth_stencil_view))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"opengl_draw_could_not_retain_its_depth_attachment", 0, 0}})};
    uint64_t depthImage = 0;
    if (hasDepth) {
        auto resolved = resolveCommandRhiResource(adapter, commandEncoder, descriptor->depth_stencil_view);
        if (!resolved)
            return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
        depthImage = std::move(resolved).value();
    }
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
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::BackendFailure, {"opengl_draw_framebuffer_is_incomplete", 0, 0}})};
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
            return RhiAdapterResult<void>{vernon::err(
                vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                      {"opengl_draw_requests_stencil_operations_for_a_depth_only_attachment", 0, 0}})};
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
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::Unsupported, {"opengl_context_does_not_expose_glscissor", 0, 0}})};
    }
    rhi::opengl::Enum topology = rhi::opengl::kTriangles;
    if (descriptor->topology == 1)
        topology = rhi::opengl::kLines;
    else if (descriptor->topology == 2)
        topology = rhi::opengl::kPoints;
    else if (descriptor->topology != 0)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                                        {"opengl_draw_topology_is_invalid", 0, 0}})};
    if (descriptor->index_count != 0) {
        if (!retainCommandResource(adapter, commandEncoder, descriptor->index_buffer))
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::BackendFailure, {"opengl_draw_could_not_retain_its_index_buffer", 0, 0}})};
        auto resolved = resolveCommandRhiResource(adapter, commandEncoder, descriptor->index_buffer);
        if (!resolved)
            return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
        const uint64_t indexBuffer = std::move(resolved).value();
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
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"opengl_draw_command_encoder_state_changed", 0, 0}})};
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
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
    if (!bindings || bindings->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
        return;
    delete bindings;
}
RhiAdapterResult<void> destroyBindingsResult(void *, VernonRuntimeProviderObject handle) {
    releaseCommandBindings(fromHandle<PreparedBindingSet>(handle), 0);
    return RhiAdapterResult<void>{vernon::ok()};
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
RhiAdapterResult<void> destroyPipelineResult(void *, VernonRuntimeProviderObject handle) {
    releaseCommandPipeline(fromHandle<PreparedPipeline>(handle), 0);
    return RhiAdapterResult<void>{vernon::ok()};
}

void destroyShader(void *data, VernonRuntimeProviderObject handle) {
    auto result = destroyShaderResult(data, handle);
    if (!result)
        recordProviderError(*static_cast<VernonRuntimeRhiAdapter *>(data), std::move(result).error(),
                            "OpenGL shader destruction failed");
}
void destroyLayout(void *data, VernonRuntimeProviderObject handle) {
    auto result = destroyLayoutResult(data, handle);
    if (!result)
        recordProviderError(*static_cast<VernonRuntimeRhiAdapter *>(data), std::move(result).error(),
                            "OpenGL layout destruction failed");
}
void destroyPipeline(void *data, VernonRuntimeProviderObject handle) {
    auto result = destroyPipelineResult(data, handle);
    if (!result)
        recordProviderError(*static_cast<VernonRuntimeRhiAdapter *>(data), std::move(result).error(),
                            "OpenGL pipeline destruction failed");
}
void destroyBindings(void *data, VernonRuntimeProviderObject handle) {
    auto result = destroyBindingsResult(data, handle);
    if (!result)
        recordProviderError(*static_cast<VernonRuntimeRhiAdapter *>(data), std::move(result).error(),
                            "OpenGL binding-set destruction failed");
}

VernonStatus prepareShader(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, prepareShaderResult(data, descriptor, output), "OpenGL shader preparation failed");
}
VernonStatus prepareLayout(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, prepareLayoutResult(data, descriptor, output), "OpenGL layout preparation failed");
}
VernonStatus preparePipeline(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                             VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, preparePipelineResult(data, descriptor, output),
                          "OpenGL pipeline preparation failed");
}
VernonStatus createBindings(void *data, const VernonRuntimeProviderBindingSetDescriptor *descriptor,
                            VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, createBindingsResult(data, descriptor, output),
                          "OpenGL binding-set creation failed");
}
VernonStatus encodeDispatch(void *data, VernonRuntimeProviderObject encoder,
                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, encodeDispatchResult(data, encoder, descriptor), "OpenGL dispatch encoding failed");
}
VernonStatus encodeDraw(void *data, VernonRuntimeProviderObject encoder,
                        const VernonRuntimeProviderDrawDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, encodeDrawResult(data, encoder, descriptor), "OpenGL draw encoding failed");
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
    adapter.provider.retain_resource = retainRhiResourceCallback;
    adapter.provider.release_resource = releaseRhiResourceCallback;
    adapter.provider.describe_image = describeProviderImageCallback;
    adapter.provider.create_binding_set = createBindings;
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
