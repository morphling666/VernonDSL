#include "VernonFormats.hpp"
#include "adapter_common.h"
#include "rhi/opengl_backend.h"
#include "rhi/rhi_internal.h"
#include "runtime/interface_names.h"
#include "runtime/vertex_attribute_capabilities.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vernon::runtime::rhi_adapter {
namespace {

namespace gl = rhi::opengl;

constexpr size_t kMaxGroups = VERNON_RUNTIME_PROVIDER_MAX_GROUPS;
constexpr uint32_t kMaxVertexBindings = VERNON_RUNTIME_PROVIDER_MAX_VERTEX_BINDINGS;
// Vertex attribute locations are tracked in a 64-bit enable mask.
constexpr uint32_t kMaxTrackedVertexLocations = 64;

// The buffer of a vertex binding, by resource key, since a vertex array keeps referring to a name after deletion and
// names are reused. `stride` is the stride the binding's attributes were specified with.
struct VertexSource {
    uint64_t key{};
    gl::Uint name{};
    uint64_t offset{};
    uint32_t stride{};
};

struct OpenGLAdapterState {
    gl::DeviceState *device{};
    // Changes whenever OpenGL binding state may have changed outside the open commands: when commands begin and when
    // preparation binds OpenGL objects.
    uint64_t epoch{};
    uint64_t nextSerial{};
    gl::Int uniformOffsetAlignment{};
    gl::Int storageOffsetAlignment{};
    // Every draw uses this vertex array, which no other OpenGL code binds, so its contents persist between commands.
    gl::Uint vertexArray{};
    uint64_t vertexProgram{};
    std::array<VertexSource, kMaxVertexBindings> vertexSources{};
    uint64_t enabledAttributes{};
    // Resource keys are not reused, unlike OpenGL names, which a vertex array keeps referring to after deletion.
    uint64_t indexBufferKey{};
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

void destroyBackend(void *data) noexcept {
    auto *state = static_cast<OpenGLAdapterState *>(data);
    if (state->vertexArray)
        state->device->destroyVertexArray(state->vertexArray);
    delete state;
}

RhiAdapterResult<void> synchronizeBackend(void *state, std::string &error) noexcept {
    try {
        auto &device = *static_cast<OpenGLAdapterState *>(state)->device;
        device.makeCurrent();
        device.driver.finish();
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (...) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "opengl_synchronize");
    }
}

uint64_t resourceIdentity(const void *state) noexcept {
    return reinterpret_cast<uintptr_t>(static_cast<const OpenGLAdapterState *>(state)->device);
}

const RhiAdapterBackendOps backendOps{destroyBackend, synchronizeBackend, resourceIdentity};

// How an entry reaches the program: context bindings are shared by every program, named uniforms belong to one.
enum class EntryApplication : uint8_t {
    UniformBuffer,
    StorageBuffer,
    SampledImage,
    Sampler,
    StorageImage,
    NamedUniform
};

struct GroupLayout {
    struct Entry {
        VernonRuntimeProviderBindingLayoutEntry layout{};
        std::string name;
        EntryApplication application{};
        // Inline bytes the group uploads into a buffer: a persistent block of the device when persistent, its
        // encoder's transient memory when transient.
        bool inlineBuffer{};
    };
    std::vector<Entry> entries;
};

struct VertexAttribute {
    VernonRuntimeProviderVertexAttribute layout{};
    uint32_t divisor{};
};

struct PreparedLayout {
    std::array<const GroupLayout *, kMaxGroups> groups{};
    uint32_t groupCount{};
    // Bit g is set when the group layout at index g has entries; an index without entries is never bound.
    uint32_t groupMask{};
    uint32_t vertexBindingMask{};
    uint64_t attributeMask{};
    std::vector<VertexAttribute> vertexAttributes;
};

struct NativeProgram {
    gl::DeviceState *device{};
    gl::Uint program{};
    uint64_t serial{};
    const PreparedLayout *layout{};
    // Per group index and entry; -1 for entries that are not named uniforms or that the program does not use.
    std::array<std::vector<gl::Int>, kMaxGroups> locations;
    // The serial of the group whose named uniforms the program holds at each group index. Uniform values are program
    // state that no other code changes, so they stay valid across commands.
    std::array<uint64_t, kMaxGroups> uniformGroups{};

    ~NativeProgram() {
        if (device)
            device->destroyProgram(program);
    }
};

struct PreparedShader {
    uint32_t stage{};
    std::string source;
    std::shared_ptr<NativeProgram> graphicsBase;
};

struct PreparedPipeline : RhiRetainedObject {
    VernonRuntimeRhiAdapter *adapter{};
    std::shared_ptr<NativeProgram> native;
    uint64_t serial{};
    const PreparedLayout *layout{};
    // By vertex binding, from the graphics variant; Core prepares variants with a stride for every binding.
    std::array<uint32_t, kMaxVertexBindings> vertexStrides{};
    bool compute{};
    VernonRasterizationState rasterization{};
    VernonDepthStencilState depthStencil{};
    std::vector<VernonColorBlendState> colorBlends;

    ~PreparedPipeline() override {
        if (adapter)
            adapter->livePreparedPipelines.fetch_sub(1, std::memory_order_relaxed);
    }
};

struct GroupSlot {
    EntryApplication application{};
    uint32_t binding{};
    gl::Uint name{};
    uint64_t offset{};
    uint64_t range{};
    gl::Enum textureTarget{};
    gl::Enum imageFormat{};
    gl::Enum imageAccess{};
    gl::Boolean layered{};
    gl::Boolean transpose{};
    uint32_t valueCount{};
    uint32_t columnCount{};
    uint32_t numericType{};
    size_t valueOffset{};
};

struct OpenGLGroup : RhiGroup {
    gl::DeviceState *device{};
    const GroupLayout *layout{};
    uint64_t serial{};
    std::vector<GroupSlot> slots;
    // Named-uniform bytes, at each slot's value offset.
    std::vector<uint8_t> values;
};

// Applied state holds serials rather than pointers or OpenGL names, because addresses and names are reused.
struct OpenGLCommands : RhiCommands {
    PreparedPipeline *pipeline{};
    std::array<OpenGLGroup *, kMaxGroups> groups{};
    uint64_t epoch{};
    uint64_t generation{};
    uint64_t appliedProgram{};
    uint64_t appliedPipeline{};
    std::array<uint64_t, kMaxGroups> appliedGroups{};
    // Per context binding namespace (each application before NamedUniform) and binding point, one more than the group
    // index whose value the point holds, or 0. Points are context state that every group index shares: a group applied
    // at one index takes its points from the other indices, whose groups apply again before their next use.
    std::array<std::vector<uint8_t>, static_cast<size_t>(EntryApplication::NamedUniform)> bindingOwners;
    // The buffers bound at each vertex binding; their strides come from the pipeline.
    std::array<VertexSource, kMaxVertexBindings> vertexBuffers{};
    bool vertexArrayBound{};
    bool viewportValid{};
    bool scissorValid{};
    uint32_t stencilReference{};
    std::array<uint32_t, 4> viewport{};
    std::array<uint32_t, 4> scissor{};
    // The index buffer the commands retained and resolved last.
    uint64_t indexBufferKey{};
    gl::Uint indexBuffer{};

    void invalidate(const OpenGLAdapterState &state) {
        epoch = state.epoch;
        generation = state.device->drawStateGeneration;
        appliedProgram = 0;
        appliedPipeline = 0;
        appliedGroups.fill(0);
        for (std::vector<uint8_t> &owners : bindingOwners)
            owners.clear();
        vertexArrayBound = false;
        viewportValid = false;
        scissorValid = false;
    }
};

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
    if (!descriptor.color_format_count && !descriptor.depth_stencil_format)
        return RhiAdapterResult<void>{vernon::ok()};
    const auto &rasterization = descriptor.rasterization;
    const auto &depthStencil = descriptor.depth_stencil;
    const gl::Driver &driver = openGLState(adapter).device->driver;
    if (descriptor.sample_count != 1 || descriptor.color_format_count > 8 ||
        descriptor.color_blend_count != descriptor.color_format_count ||
        (descriptor.color_blend_count && !descriptor.color_blends) || rasterization.cull_mode > VERNON_RHI_CULL_BACK ||
        rasterization.front_face > VERNON_RHI_FRONT_FACE_CLOCKWISE || rasterization.depth_clamp ||
        rasterization.depth_bias_enabled > 1 || !std::isfinite(rasterization.depth_bias_constant) ||
        !std::isfinite(rasterization.depth_bias_slope) || depthStencil.depth_test > 1 || depthStencil.depth_write > 1 ||
        depthStencil.depth_compare > VERNON_RHI_COMPARE_ALWAYS || depthStencil.stencil_test > 1 ||
        (descriptor.depth_stencil_format && descriptor.depth_stencil_format != gl::kDepthComponent32f &&
         descriptor.depth_stencil_format != gl::kDepth32fStencil8) ||
        (!descriptor.depth_stencil_format && (depthStencil.depth_test || depthStencil.depth_write)) ||
        (depthStencil.stencil_test && descriptor.depth_stencil_format != gl::kDepth32fStencil8) ||
        depthStencil.stencil_read_mask > 0xff || depthStencil.stencil_write_mask > 0xff || !driver.blendFuncSeparate ||
        !driver.blendEquationSeparate || !driver.colorMask ||
        (descriptor.color_format_count > 1 && (!driver.enablei || !driver.disablei || !driver.blendFuncSeparatei ||
                                               !driver.blendEquationSeparatei || !driver.colorMaski)))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "opengl_graphics_pipeline_contains_unsupported_graphics_state");
    const auto validFace = [](const VernonStencilFaceState &face) {
        return face.stencil_fail <= VERNON_RHI_STENCIL_DECREMENT_WRAP &&
               face.depth_fail <= VERNON_RHI_STENCIL_DECREMENT_WRAP && face.pass <= VERNON_RHI_STENCIL_DECREMENT_WRAP &&
               face.compare <= VERNON_RHI_COMPARE_ALWAYS;
    };
    if (!validFace(depthStencil.front) || !validFace(depthStencil.back))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "opengl_graphics_pipeline_contains_invalid_stencil_state");
    for (size_t index = 0; index < descriptor.color_blend_count; ++index) {
        const auto &blend = descriptor.color_blends[index];
        if (blend.blend_enabled > 1 || blend.source_color_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.destination_color_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.source_alpha_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.destination_alpha_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.color_operation > VERNON_RHI_BLEND_MAXIMUM || blend.alpha_operation > VERNON_RHI_BLEND_MAXIMUM ||
            (blend.write_mask & ~VERNON_RHI_COLOR_WRITE_ALL))
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "opengl_graphics_pipeline_contains_invalid_blend_state");
    }
    pipeline.rasterization = rasterization;
    pipeline.depthStencil = depthStencil;
    if (descriptor.color_blend_count)
        pipeline.colorBlends.assign(descriptor.color_blends, descriptor.color_blends + descriptor.color_blend_count);
    else
        pipeline.colorBlends.clear();
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
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_adapter_received_an_invalid_shader");
    try {
        auto shader = std::make_unique<PreparedShader>();
        shader->stage = descriptor->stage;
        shader->source.assign(static_cast<const char *>(descriptor->data), descriptor->size);
        *output = toHandle(shader.release());
        adapter.shaderPreparations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                               "opengl_shader_preparation_ran_out_of_memory");
    }
}

std::optional<EntryApplication> entryApplication(const VernonRuntimeProviderBindingLayoutEntry &source) {
    const bool nativeNumericType = source.numeric_type == VERNON_RUNTIME_PROVIDER_F32 ||
                                   source.numeric_type == VERNON_RUNTIME_PROVIDER_I32 ||
                                   source.numeric_type == VERNON_RUNTIME_PROVIDER_U32;
    const bool nativeUniformShape =
        source.vector_count != 0 &&
        ((source.vector_count == 1 && source.element_count >= 1 && source.element_count <= 4) ||
         (source.numeric_type == VERNON_RUNTIME_PROVIDER_F32 && source.vector_count >= 2 && source.vector_count <= 4 &&
          source.element_count % source.vector_count == 0 && source.element_count / source.vector_count >= 2 &&
          source.element_count / source.vector_count <= 4));
    const bool compute = source.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
    const bool named = source.name.data && source.name.size != 0;
    const bool bound = source.binding != UINT32_MAX;
    switch (source.kind) {
    case VERNON_RUNTIME_PROVIDER_INLINE_VALUE:
        if (compute && bound && source.element_size != 0)
            return EntryApplication::StorageBuffer;
        if (!compute && source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM && named &&
            nativeNumericType && nativeUniformShape && source.element_size == source.element_count * sizeof(uint32_t))
            return EntryApplication::NamedUniform;
        return std::nullopt;
    case VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER:
        if ((compute || source.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX ||
             source.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT) &&
            (source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE ||
             source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM) &&
            bound && source.element_size != 0)
            return EntryApplication::StorageBuffer;
        return std::nullopt;
    case VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER:
        if (source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM && bound && source.element_size != 0)
            return EntryApplication::UniformBuffer;
        return std::nullopt;
    case VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE:
        if (source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE && bound && (compute || named))
            return EntryApplication::SampledImage;
        return std::nullopt;
    case VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE:
        if (source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE && bound && source.access != 0)
            return EntryApplication::StorageImage;
        return std::nullopt;
    case VERNON_RUNTIME_PROVIDER_SAMPLER:
        if (source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE && bound)
            return EntryApplication::Sampler;
        return std::nullopt;
    }
    return std::nullopt;
}

RhiAdapterResult<void> prepareGroupLayoutResult(const VernonRuntimeProviderGroupLayoutDescriptor *descriptor,
                                                VernonRuntimeProviderObject *output) {
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        (descriptor->entry_count != 0 && !descriptor->entries))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "opengl_adapter_received_an_invalid_group_layout");
    try {
        auto layout = std::make_unique<GroupLayout>();
        layout->entries.reserve(descriptor->entry_count);
        for (size_t index = 0; index < descriptor->entry_count; ++index) {
            const auto &source = descriptor->entries[index];
            const auto application = entryApplication(source);
            if (!application)
                return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_validate_group_entry", index,
                                       static_cast<uint32_t>(source.kind));
            GroupLayout::Entry entry;
            entry.layout = source;
            if (source.name.data && source.name.size)
                entry.name.assign(source.name.data, source.name.size);
            entry.application = *application;
            entry.inlineBuffer = *application != EntryApplication::NamedUniform &&
                                 packedUniformBytes(source.kind, source.interface_kind);
            layout->entries.push_back(std::move(entry));
        }
        *output = toHandle(layout.release());
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                               "opengl_group_layout_preparation_ran_out_of_memory");
    }
}

RhiAdapterResult<void> validateComputeBufferLimits(gl::Driver &driver, const PreparedLayout &layout) {
    size_t uniformBuffers = 0;
    size_t storageBuffers = 0;
    for (uint32_t group = 0; group < layout.groupCount; ++group)
        for (const GroupLayout::Entry &entry : layout.groups[group]->entries) {
            if (entry.layout.stage_mask != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE)
                continue;
            uniformBuffers += entry.application == EntryApplication::UniformBuffer;
            storageBuffers += entry.application == EntryApplication::StorageBuffer;
        }
    if (!uniformBuffers && !storageBuffers)
        return RhiAdapterResult<void>{vernon::ok()};
    gl::Int maximumUniformBindings = 0;
    gl::Int maximumUniformBlockSize = 0;
    gl::Int maximumComputeUniformBlocks = 0;
    gl::Int maximumComputeStorageBlocks = 0;
    gl::Int maximumStorageBindings = 0;
    driver.getIntegerv(gl::kMaxUniformBufferBindings, &maximumUniformBindings);
    driver.getIntegerv(gl::kMaxUniformBlockSize, &maximumUniformBlockSize);
    driver.getIntegerv(gl::kMaxComputeUniformBlocks, &maximumComputeUniformBlocks);
    driver.getIntegerv(gl::kMaxComputeShaderStorageBlocks, &maximumComputeStorageBlocks);
    driver.getIntegerv(gl::kMaxShaderStorageBufferBindings, &maximumStorageBindings);
    for (uint32_t group = 0; group < layout.groupCount; ++group)
        for (const GroupLayout::Entry &entry : layout.groups[group]->entries) {
            if (entry.layout.stage_mask != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE)
                continue;
            if (entry.application == EntryApplication::UniformBuffer &&
                (maximumUniformBindings <= 0 || maximumUniformBlockSize <= 0 ||
                 entry.layout.binding >= static_cast<uint32_t>(maximumUniformBindings) ||
                 entry.layout.element_size > static_cast<uint32_t>(maximumUniformBlockSize)))
                return providerFailure(vernon::ProviderErrorCode::Unsupported,
                                       "opengl_compute_uniform_buffer_exceeds_device_limits", entry.layout.binding,
                                       entry.layout.element_size);
            if (entry.application == EntryApplication::StorageBuffer &&
                (maximumStorageBindings <= 0 || entry.layout.binding >= static_cast<uint32_t>(maximumStorageBindings)))
                return providerFailure(vernon::ProviderErrorCode::Unsupported,
                                       "opengl_compute_storage_buffer_binding_exceeds_device_limits",
                                       entry.layout.binding,
                                       maximumStorageBindings > 0 ? static_cast<uint32_t>(maximumStorageBindings) : 0u);
        }
    if (maximumComputeUniformBlocks < 0 || maximumComputeStorageBlocks < 0 ||
        uniformBuffers > static_cast<size_t>(maximumComputeUniformBlocks) ||
        storageBuffers > static_cast<size_t>(maximumComputeStorageBlocks))
        return providerFailure(vernon::ProviderErrorCode::Unsupported,
                               "opengl_compute_buffer_binding_count_exceeds_device_limits",
                               std::min<size_t>(storageBuffers, UINT32_MAX),
                               static_cast<uint32_t>(std::min<size_t>(uniformBuffers, UINT32_MAX)));
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> prepareLayoutResult(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        descriptor->group_layout_count > kMaxGroups ||
        (descriptor->group_layout_count != 0 && !descriptor->group_layouts) ||
        (descriptor->vertex_attribute_count != 0 && !descriptor->vertex_attributes) ||
        descriptor->push_constant_size != 0)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "opengl_adapter_received_an_invalid_pipeline_layout");
    auto layout = std::unique_ptr<PreparedLayout>(new (std::nothrow) PreparedLayout());
    if (!layout)
        return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                               "opengl_layout_preparation_ran_out_of_memory");
    for (size_t group = 0; group < descriptor->group_layout_count; ++group) {
        layout->groups[group] = fromHandle<GroupLayout>(descriptor->group_layouts[group]);
        if (!layout->groups[group])
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "opengl_pipeline_layout_group_layout_is_invalid", group);
        if (!layout->groups[group]->entries.empty())
            layout->groupMask |= 1u << group;
    }
    layout->groupCount = static_cast<uint32_t>(descriptor->group_layout_count);
    auto &device = *openGLState(adapter).device;
    if (!device.makeCurrent())
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "opengl_make_context_current_for_layout");
    if (auto limits = validateComputeBufferLimits(device.driver, *layout); !limits)
        return limits;
    gl::Int maximumLocations = 0;
    device.driver.getIntegerv(gl::kMaxVertexAttribs, &maximumLocations);
    const uint32_t locationLimit =
        std::min(maximumLocations > 0 ? static_cast<uint32_t>(maximumLocations) : 0u, kMaxTrackedVertexLocations);
    try {
        layout->vertexAttributes.reserve(descriptor->vertex_attribute_count);
        std::array<uint32_t, kMaxVertexBindings> divisors{};
        for (size_t index = 0; index < descriptor->vertex_binding_count; ++index) {
            const VernonRuntimeProviderVertexBinding &binding = descriptor->vertex_bindings[index];
            divisors[binding.binding] = binding.divisor;
            layout->vertexBindingMask |= 1u << binding.binding;
        }
        for (size_t index = 0; index < descriptor->vertex_attribute_count; ++index) {
            const auto &attribute = descriptor->vertex_attributes[index];
            std::string capabilityDiagnostic;
            if (!validateVertexAttributeCapability(VertexAttributeBackend::OpenGL, attribute, locationLimit,
                                                   device.driver.vertexAttribLPointer != nullptr, capabilityDiagnostic))
                return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                       "OpenGL vertex attribute exceeds device location or format capabilities",
                                       attribute.location, attribute.binding);
            layout->attributeMask |= uint64_t{1} << attribute.location;
            layout->vertexAttributes.push_back({attribute, divisors[attribute.binding]});
        }
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                               "opengl_layout_preparation_ran_out_of_memory");
    }
    *output = toHandle(layout.release());
    adapter.layoutPreparations.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

std::optional<gl::Enum> storageImageFormat(VernonRhiFormat format) {
    const vernon::formats::Format *entry = vernon::formats::find(format);
    if (!entry || !entry->allows(VERNON_RHI_FORMAT_STORAGE))
        return std::nullopt;
    return entry->glInternalFormat;
}

// Resolves the program's interface: block bindings, sampler units and named-uniform locations. Binds the program.
RhiAdapterResult<void> linkProgramInterface(OpenGLAdapterState &state, NativeProgram &native, bool compute) {
    auto &driver = state.device->driver;
    const PreparedLayout &layout = *native.layout;
    // A block left unassigned silently reads binding point 0, so every active block must have a planned binding.
    std::vector<gl::Uint> assignedBlocks;
    for (uint32_t group = 0; group < layout.groupCount; ++group)
        for (const GroupLayout::Entry &entry : layout.groups[group]->entries) {
            if (entry.layout.kind != VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER)
                continue;
            const std::string blockName = ::vernon::runtime::openGLUniformBlockName(entry.layout.binding);
            const gl::Uint block = driver.getUniformBlockIndex(native.program, blockName.c_str());
            if (block == gl::kInvalidIndex)
                continue;
            driver.uniformBlockBinding(native.program, block, entry.layout.binding);
            if (std::find(assignedBlocks.begin(), assignedBlocks.end(), block) == assignedBlocks.end())
                assignedBlocks.push_back(block);
        }
    gl::Int activeBlocks = 0;
    driver.getProgramiv(native.program, gl::kActiveUniformBlocks, &activeBlocks);
    if (activeBlocks < 0 || assignedBlocks.size() < static_cast<size_t>(activeBlocks))
        return providerFailure(
            vernon::ProviderErrorCode::InvalidArgument, "opengl_uniform_block_has_no_planned_binding",
            static_cast<uint32_t>(std::max(activeBlocks, 0)), static_cast<uint32_t>(assignedBlocks.size()));
    // Compute GLSL declares each sampler's texture unit as its binding; graphics samplers receive it here, once.
    if (compute)
        return RhiAdapterResult<void>{vernon::ok()};
    driver.useProgram(native.program);
    ++state.epoch;
    for (uint32_t group = 0; group < layout.groupCount; ++group) {
        const auto &entries = layout.groups[group]->entries;
        native.locations[group].assign(entries.size(), -1);
        for (size_t index = 0; index < entries.size(); ++index) {
            const GroupLayout::Entry &entry = entries[index];
            if (entry.application != EntryApplication::NamedUniform &&
                entry.application != EntryApplication::SampledImage)
                continue;
            const gl::Int location = driver.getUniformLocation(native.program, entry.name.c_str());
            if (location < 0)
                continue;
            if (entry.application == EntryApplication::SampledImage)
                driver.uniform1i(location, static_cast<gl::Int>(entry.layout.binding));
            else
                native.locations[group][index] = location;
        }
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> preparePipelineResult(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                                             VernonRuntimeProviderObject *output) {
    std::string backendDiagnostic;
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto &state = openGLState(adapter);
    auto *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    const bool compute = descriptor && descriptor->kind == VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        (compute ? descriptor->shader_count != 1
                 : descriptor->kind != VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE || descriptor->shader_count != 2) ||
        !descriptor->shaders || !layout)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "opengl_adapter_received_an_invalid_pipeline");
    std::unique_ptr<PreparedPipeline> pipeline(new (std::nothrow) PreparedPipeline());
    if (!pipeline)
        return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                               "opengl_pipeline_preparation_ran_out_of_memory");
    pipeline->adapter = &adapter;
    adapter.livePreparedPipelines.fetch_add(1, std::memory_order_relaxed);
    pipeline->serial = ++state.nextSerial;
    pipeline->layout = layout;
    pipeline->compute = compute;
    std::copy_n(descriptor->vertex_strides, std::min<size_t>(descriptor->vertex_stride_count, kMaxVertexBindings),
                pipeline->vertexStrides.begin());
    try {
        if (auto configured = configureGraphicsStateResult(adapter, *descriptor, *pipeline); !configured)
            return configured;
        if (!compute && !state.vertexArray && !state.device->createVertexArray(state.vertexArray, backendDiagnostic))
            return providerFailure(vernon::ProviderErrorCode::BackendFailure, "opengl_create_vertex_array");
        if (!compute && descriptor->color_format_count) {
            auto *first = fromHandle<PreparedShader>(descriptor->shaders[0]);
            auto *second = fromHandle<PreparedShader>(descriptor->shaders[1]);
            if (first && second && first->graphicsBase && first->graphicsBase == second->graphicsBase &&
                first->graphicsBase->layout == layout) {
                pipeline->native = first->graphicsBase;
                *output = toHandle(pipeline.release());
                adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
                return RhiAdapterResult<void>{vernon::ok()};
            }
        }
        std::array<gl::Uint, 3> compiled{};
        const auto deleteCompiled = [&] {
            for (gl::Uint shader : compiled)
                if (shader)
                    state.device->driver.deleteShader(shader);
        };
        bool hasVertex = false;
        bool hasFragment = false;
        bool hasCompute = false;
        for (size_t index = 0; index < descriptor->shader_count; ++index) {
            const auto *shader = fromHandle<PreparedShader>(descriptor->shaders[index]);
            if (!shader) {
                deleteCompiled();
                return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                       "opengl_pipeline_contains_an_invalid_shader");
            }
            const auto kind = shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX     ? gl::kVertexShader
                              : shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT ? gl::kFragmentShader
                                                                                        : gl::kComputeShader;
            hasVertex |= shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX;
            hasFragment |= shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT;
            hasCompute |= shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
            compiled[index] = state.device->compileShader(kind, shader->source, backendDiagnostic);
            if (!compiled[index]) {
                deleteCompiled();
                return providerFailure(vernon::ProviderErrorCode::BackendFailure, "opengl_compile_shader",
                                       static_cast<uint64_t>(kind), static_cast<uint32_t>(index));
            }
        }
        if ((compute && !hasCompute) || (!compute && (!hasVertex || !hasFragment))) {
            deleteCompiled();
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "opengl_pipeline_shader_stages_are_incomplete");
        }
        auto native = std::make_shared<NativeProgram>();
        native->program = state.device->linkProgram({compiled.begin(), compiled.begin() + descriptor->shader_count},
                                                    backendDiagnostic);
        if (!native->program)
            return providerFailure(vernon::ProviderErrorCode::BackendFailure, "opengl_link_program",
                                   descriptor->shader_count);
        native->device = state.device;
        native->serial = ++state.nextSerial;
        native->layout = layout;
        if (auto linked = linkProgramInterface(state, *native, compute); !linked)
            return linked;
        if (!compute && descriptor->color_format_count == 0)
            for (size_t index = 0; index < descriptor->shader_count; ++index)
                fromHandle<PreparedShader>(descriptor->shaders[index])->graphicsBase = native;
        pipeline->native = std::move(native);
        *output = toHandle(pipeline.release());
        adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                               "opengl_pipeline_preparation_ran_out_of_memory");
    }
}

RhiAdapterResult<gl::Int> offsetAlignment(OpenGLAdapterState &state, EntryApplication application) {
    gl::Int &alignment =
        application == EntryApplication::UniformBuffer ? state.uniformOffsetAlignment : state.storageOffsetAlignment;
    if (alignment == 0)
        state.device->driver.getIntegerv(application == EntryApplication::UniformBuffer
                                             ? gl::kUniformBufferOffsetAlignment
                                             : gl::kShaderStorageBufferOffsetAlignment,
                                         &alignment);
    if (alignment <= 0)
        return RhiAdapterResult<gl::Int>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"opengl_buffer_offset_alignment_is_unavailable", 0, 0}})};
    return RhiAdapterResult<gl::Int>{vernon::ok(alignment)};
}

RhiAdapterResult<void> writeInlineUniform(VernonRuntimeRhiAdapter &adapter, OpenGLGroup &group,
                                          const VernonRuntimeProviderGroupDescriptor &descriptor,
                                          const GroupLayout::Entry &entry, const void *bytes, size_t size,
                                          GroupSlot &slot) {
    auto alignment = offsetAlignment(openGLState(adapter), entry.application);
    if (!alignment)
        return RhiAdapterResult<void>{vernon::err(std::move(alignment).error())};
    const auto aligned = static_cast<uint64_t>(alignment.value());
    if (descriptor.lifetime == VERNON_RUNTIME_PROVIDER_GROUP_TRANSIENT) {
        auto written = writeCommandTransient(adapter, descriptor.command_encoder, bytes, size, aligned);
        if (!written)
            return RhiAdapterResult<void>{vernon::err(std::move(written).error())};
        slot.name = static_cast<gl::Uint>(written.value().buffer);
        slot.offset = static_cast<size_t>(written.value().offset);
    } else {
        auto placed = placePersistent(adapter, group, bytes, size, aligned);
        if (!placed)
            return RhiAdapterResult<void>{vernon::err(std::move(placed).error())};
        slot.name = static_cast<gl::Uint>(placed.value().buffer);
        slot.offset = static_cast<size_t>(placed.value().offset);
    }
    slot.range = size;
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> initializeInlineSlot(VernonRuntimeRhiAdapter &adapter, OpenGLGroup &group,
                                            const VernonRuntimeProviderGroupDescriptor &descriptor,
                                            const GroupLayout::Entry &entry,
                                            const VernonRuntimeProviderBindingValue &value, GroupSlot &slot) {
    const uint32_t flags = value.flags;
    if (!value.payload.inline_value.data || value.payload.inline_value.size != entry.layout.element_size ||
        (flags & ~VERNON_RUNTIME_PROVIDER_BINDING_TRANSPOSE) != 0 ||
        ((flags & VERNON_RUNTIME_PROVIDER_BINDING_TRANSPOSE) != 0 &&
         (entry.application != EntryApplication::NamedUniform || slot.columnCount <= 1)))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_inline_binding_is_invalid",
                               entry.layout.slot);
    slot.transpose = static_cast<gl::Boolean>((flags & VERNON_RUNTIME_PROVIDER_BINDING_TRANSPOSE) != 0);
    if (!entry.inlineBuffer) {
        slot.valueOffset = group.values.size();
        const auto *bytes = static_cast<const uint8_t *>(value.payload.inline_value.data);
        group.values.insert(group.values.end(), bytes, bytes + value.payload.inline_value.size);
        return RhiAdapterResult<void>{vernon::ok()};
    }
    return writeInlineUniform(adapter, group, descriptor, entry, value.payload.inline_value.data,
                              value.payload.inline_value.size, slot);
}

RhiAdapterResult<void> initializeResourceSlot(VernonRuntimeRhiAdapter &adapter, OpenGLGroup &group,
                                              const GroupLayout::Entry &entry,
                                              const VernonRuntimeProviderBindingValue &value, GroupSlot &slot) {
    const bool defaultResource = (value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0;
    const bool image =
        entry.application == EntryApplication::SampledImage || entry.application == EntryApplication::StorageImage;
    const bool sampler = entry.application == EntryApplication::Sampler;
    const bool indexedBuffer =
        entry.application == EntryApplication::UniformBuffer || entry.application == EntryApplication::StorageBuffer;
    const VernonRuntimeProviderResourceReference &resource = image     ? value.payload.image.view
                                                             : sampler ? value.payload.sampler.resource
                                                                       : value.payload.buffer.resource;
    slot.textureTarget = gl::textureTarget(VERNON_RHI_IMAGE_2D);
    if ((value.flags & ~VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0 ||
        (defaultResource && entry.application == EntryApplication::StorageImage))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_resource_binding_is_invalid",
                               entry.layout.slot);
    if (defaultResource)
        return RhiAdapterResult<void>{vernon::ok()};
    const uint64_t expectedKind = sampler ? kRhiSamplerResource : kRhiBufferResource;
    if ((image ? !isRhiImageReference(resource) : (resource.identity & kRhiResourceKindMask) != expectedKind) ||
        (!image && !sampler && resource.size == 0))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_resource_binding_is_invalid",
                               entry.layout.slot);
    RhiImageBinding imageBinding;
    auto acquired = acquireGroupResource(adapter, group, resource, image ? &imageBinding : nullptr);
    if (!acquired)
        return RhiAdapterResult<void>{vernon::err(std::move(acquired).error())};
    slot.name = static_cast<gl::Uint>(acquired.value());
    if (!slot.name)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_resource_binding_is_invalid",
                               entry.layout.slot);
    slot.offset = resource.offset;
    slot.range = resource.size;
    if (indexedBuffer && slot.offset != 0) {
        auto alignment = offsetAlignment(openGLState(adapter), entry.application);
        if (!alignment)
            return RhiAdapterResult<void>{vernon::err(std::move(alignment).error())};
        if (slot.offset % static_cast<uint64_t>(alignment.value()) != 0 ||
            slot.offset > static_cast<uint64_t>(std::numeric_limits<gl::IntPtr>::max()) ||
            slot.range > static_cast<uint64_t>(std::numeric_limits<gl::SizePtr>::max()))
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_buffer_range_binding_is_invalid",
                                   entry.layout.slot);
    }
    if (image) {
        slot.textureTarget = gl::textureTarget(imageBinding.dimension);
        slot.layered = static_cast<gl::Boolean>(imageBinding.dimension != VERNON_RHI_IMAGE_2D);
    }
    if (entry.application == EntryApplication::StorageImage) {
        const auto format = storageImageFormat(imageBinding.format);
        if (!format || !group.device->driver.bindImageTexture)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "opengl_storage_image_format_or_entry_point_is_unavailable", entry.layout.slot);
        slot.imageFormat = *format;
        slot.imageAccess = entry.layout.access == 1   ? gl::kReadOnly
                           : entry.layout.access == 2 ? gl::kWriteOnly
                                                      : gl::kReadWrite;
    }
    if (writesBinding(entry.layout))
        group.writes.push_back(resource);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> createGroupResult(void *data, const VernonRuntimeProviderGroupDescriptor *descriptor,
                                         VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto &state = openGLState(adapter);
    const auto *layout = descriptor ? fromHandle<GroupLayout>(descriptor->group_layout) : nullptr;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !layout ||
        descriptor->value_count != layout->entries.size() || (descriptor->value_count != 0 && !descriptor->values))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "opengl_adapter_received_an_invalid_group_descriptor");
    try {
        auto group = std::make_unique<OpenGLGroup>();
        group->device = state.device;
        group->layout = layout;
        group->slots.resize(layout->entries.size());
        for (size_t index = 0; index < layout->entries.size(); ++index) {
            const GroupLayout::Entry &entry = layout->entries[index];
            const VernonRuntimeProviderBindingValue &value = descriptor->values[index];
            if (value.kind != entry.layout.kind)
                return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                       "opengl_group_value_kind_does_not_match_its_layout", index);
            GroupSlot &slot = group->slots[index];
            slot.application = entry.application;
            slot.binding = entry.layout.binding;
            slot.valueCount = entry.layout.element_count ? entry.layout.element_count : entry.layout.element_size / 4;
            slot.columnCount = entry.layout.vector_count;
            slot.numericType = entry.layout.numeric_type;
            auto initialized = packedUniformBytes(entry.layout.kind, entry.layout.interface_kind)
                                   ? initializeInlineSlot(adapter, *group, *descriptor, entry, value, slot)
                                   : initializeResourceSlot(adapter, *group, entry, value, slot);
            if (!initialized)
                return initialized;
        }
        group->serial = ++state.nextSerial;
        *output = toHandle(group.release());
        adapter.groupCreations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "opengl_group_creation_ran_out_of_memory");
    }
}

RhiAdapterResult<void> beginCommandsResult(void *data, VernonRuntimeProviderObject encoder,
                                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto &state = openGLState(adapter);
    if (!output)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_begin_commands_output_is_null");
    std::unique_ptr<OpenGLCommands> commands(new (std::nothrow) OpenGLCommands());
    if (!commands)
        return providerFailure(vernon::ProviderErrorCode::ResourceExhausted, "opengl_commands_allocation");
    if (auto begun = beginRhiCommands(adapter, encoder, *commands); !begun)
        return begun;
    if (commands->native != reinterpret_cast<uintptr_t>(state.device))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_command_encoder_is_invalid");
    if (!state.device->makeCurrent())
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "opengl_make_context_current_for_commands");
    // Errors that other OpenGL code left pending would otherwise be reported by end_commands.
    if (state.device->driver.getError)
        for (int drained = 0; drained < 16 && state.device->driver.getError() != 0; ++drained) {
        }
    ++state.epoch;
    commands->invalidate(state);
    *output = toHandle(commands.release());
    return RhiAdapterResult<void>{vernon::ok()};
}

// Reports the OpenGL errors raised since the last check, which the commands just encoded raised.
RhiAdapterResult<void> pendingErrors(VernonRuntimeRhiAdapter &adapter, const char *operation) {
    const gl::Driver &driver = openGLState(adapter).device->driver;
    if (!driver.getError)
        return RhiAdapterResult<void>{vernon::ok()};
    const gl::Enum error = driver.getError();
    if (error == 0)
        return RhiAdapterResult<void>{vernon::ok()};
    for (int drained = 0; drained < 16 && driver.getError() != 0; ++drained) {
    }
    return providerFailure(vernon::ProviderErrorCode::BackendFailure, operation, error);
}

RhiAdapterResult<void> endCommandsResult(void *data, VernonRuntimeProviderObject handle) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    std::unique_ptr<OpenGLCommands> commands(fromHandle<OpenGLCommands>(handle));
    if (!commands)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_end_commands_handle_is_invalid");
    return pendingErrors(adapter, "opengl_commands");
}

// The RHI's own OpenGL calls may have changed any binding the commands applied.
RhiAdapterResult<void> resumeCommandsResult(void *data, VernonRuntimeProviderObject handle) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<OpenGLCommands>(handle);
    if (!commands || commands->rendering)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_resume_commands_is_invalid");
    commands->invalidate(openGLState(adapter));
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> bindPipelineResult(void *data, VernonRuntimeProviderObject handle,
                                          VernonRuntimeProviderObject pipelineHandle) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<OpenGLCommands>(handle);
    auto *pipeline = fromHandle<PreparedPipeline>(pipelineHandle);
    if (!commands || !pipeline)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_bind_pipeline_is_invalid");
    if (auto retained = retainForCommands(adapter, commands->encoder, *pipeline); !retained)
        return retained;
    for (uint32_t group = 0; group < kMaxGroups; ++group)
        if (commands->groups[group] && (group >= pipeline->layout->groupCount ||
                                        commands->groups[group]->layout != pipeline->layout->groups[group]))
            commands->groups[group] = nullptr;
    commands->pipeline = pipeline;
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> bindGroupResult(void *data, VernonRuntimeProviderObject handle, uint32_t index,
                                       VernonRuntimeProviderObject groupHandle) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<OpenGLCommands>(handle);
    auto *group = fromHandle<OpenGLGroup>(groupHandle);
    if (!commands || !group || !commands->pipeline || index >= commands->pipeline->layout->groupCount ||
        group->layout != commands->pipeline->layout->groups[index])
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_bind_group_is_invalid", index);
    if (auto retained = retainForCommands(adapter, commands->encoder, *group); !retained)
        return retained;
    commands->groups[index] = group;
    return RhiAdapterResult<void>{vernon::ok()};
}

void bindIndexedBuffer(gl::Driver &driver, gl::Enum target, const GroupSlot &slot) {
    if (slot.offset == 0)
        driver.bindBufferBase(target, slot.binding, slot.name);
    else
        driver.bindBufferRange(target, slot.binding, slot.name, static_cast<gl::IntPtr>(slot.offset),
                               static_cast<gl::SizePtr>(slot.range));
}

// Context bindings serve every program, so they apply whether or not the bound program reads them.
void applyGroupBindings(gl::Driver &driver, OpenGLCommands &commands, uint32_t index, const OpenGLGroup &group) {
    for (const GroupSlot &slot : group.slots) {
        if (slot.application < EntryApplication::NamedUniform) {
            std::vector<uint8_t> &owners = commands.bindingOwners[static_cast<size_t>(slot.application)];
            if (owners.size() <= slot.binding)
                owners.resize(slot.binding + 1, 0);
            const uint8_t owner = owners[slot.binding];
            if (owner != 0 && owner != index + 1)
                commands.appliedGroups[owner - 1] = 0;
            owners[slot.binding] = static_cast<uint8_t>(index + 1);
        }
        switch (slot.application) {
        case EntryApplication::UniformBuffer:
            bindIndexedBuffer(driver, gl::kUniformBuffer, slot);
            break;
        case EntryApplication::StorageBuffer:
            bindIndexedBuffer(driver, gl::kShaderStorageBuffer, slot);
            break;
        case EntryApplication::SampledImage:
            driver.activeTexture(gl::kTexture0 + slot.binding);
            driver.bindTexture(slot.textureTarget, slot.name);
            break;
        case EntryApplication::Sampler:
            driver.bindSampler(slot.binding, slot.name);
            break;
        case EntryApplication::StorageImage:
            driver.bindImageTexture(slot.binding, slot.name, 0, slot.layered, 0, slot.imageAccess, slot.imageFormat);
            break;
        case EntryApplication::NamedUniform:
            break;
        }
    }
}

void uploadNamedUniform(gl::Driver &driver, gl::Int location, const GroupSlot &slot, const uint8_t *bytes) {
    const uint32_t count = slot.valueCount;
    if (slot.columnCount <= 1) {
        if (slot.numericType == VERNON_RUNTIME_PROVIDER_I32) {
            std::array<gl::Int, 4> value{};
            std::memcpy(value.data(), bytes, count * sizeof(value[0]));
            (count == 1   ? driver.uniform1iv
             : count == 2 ? driver.uniform2iv
             : count == 3 ? driver.uniform3iv
                          : driver.uniform4iv)(location, 1, value.data());
        } else if (slot.numericType == VERNON_RUNTIME_PROVIDER_U32) {
            std::array<gl::Uint, 4> value{};
            std::memcpy(value.data(), bytes, count * sizeof(value[0]));
            (count == 1   ? driver.uniform1uiv
             : count == 2 ? driver.uniform2uiv
             : count == 3 ? driver.uniform3uiv
                          : driver.uniform4uiv)(location, 1, value.data());
        } else {
            std::array<float, 4> value{};
            std::memcpy(value.data(), bytes, count * sizeof(value[0]));
            (count == 1   ? driver.uniform1fv
             : count == 2 ? driver.uniform2fv
             : count == 3 ? driver.uniform3fv
                          : driver.uniform4fv)(location, 1, value.data());
        }
        return;
    }
    std::array<float, 16> value{};
    std::memcpy(value.data(), bytes, count * sizeof(value[0]));
    const uint32_t columns = slot.columnCount;
    const uint32_t rows = count / columns;
    const auto matrix = columns == 2   ? (rows == 2   ? driver.uniformMatrix2fv
                                          : rows == 3 ? driver.uniformMatrix2x3fv
                                                      : driver.uniformMatrix2x4fv)
                        : columns == 3 ? (rows == 2   ? driver.uniformMatrix3x2fv
                                          : rows == 3 ? driver.uniformMatrix3fv
                                                      : driver.uniformMatrix3x4fv)
                                       : (rows == 2   ? driver.uniformMatrix4x2fv
                                          : rows == 3 ? driver.uniformMatrix4x3fv
                                                      : driver.uniformMatrix4fv);
    matrix(location, 1, slot.transpose, value.data());
}

// Brings the bound program and every bound group's context bindings up to date; requires the program's groups.
RhiAdapterResult<void> applyProgramAndGroups(OpenGLAdapterState &state, OpenGLCommands &commands) {
    auto &driver = state.device->driver;
    if (commands.epoch != state.epoch || commands.generation != state.device->drawStateGeneration)
        commands.invalidate(state);
    PreparedPipeline &pipeline = *commands.pipeline;
    NativeProgram &native = *pipeline.native;
    if (commands.appliedProgram != native.serial) {
        driver.useProgram(native.program);
        commands.appliedProgram = native.serial;
    }
    for (uint32_t index = 0; index < pipeline.layout->groupCount; ++index) {
        if ((pipeline.layout->groupMask & (1u << index)) == 0)
            continue;
        const OpenGLGroup *group = commands.groups[index];
        if (!group)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_command_group_is_unbound",
                                   index);
        if (commands.appliedGroups[index] != group->serial) {
            applyGroupBindings(driver, commands, index, *group);
            commands.appliedGroups[index] = group->serial;
        }
        if (native.uniformGroups[index] == group->serial)
            continue;
        const std::vector<gl::Int> &locations = native.locations[index];
        for (size_t entry = 0; entry < locations.size(); ++entry)
            if (locations[entry] >= 0)
                uploadNamedUniform(driver, locations[entry], group->slots[entry],
                                   group->values.data() + group->slots[entry].valueOffset);
        native.uniformGroups[index] = group->serial;
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> recordBoundGroupWrites(VernonRuntimeRhiAdapter &adapter, const OpenGLCommands &commands) {
    const PreparedLayout &layout = *commands.pipeline->layout;
    for (uint32_t index = 0; index < layout.groupCount; ++index)
        if ((layout.groupMask & (1u << index)) != 0 && !commands.groups[index]->writes.empty())
            if (auto recorded = recordGroupWrites(adapter, commands.encoder, *commands.groups[index]); !recorded)
                return recorded;
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> encodeDispatchResult(void *data, VernonRuntimeProviderObject handle,
                                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto &state = openGLState(adapter);
    auto *commands = fromHandle<OpenGLCommands>(handle);
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !commands || !commands->pipeline ||
        !commands->pipeline->compute || commands->rendering || descriptor->push_constant_size != 0)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "opengl_adapter_received_an_invalid_compute_dispatch");
    if (auto applied = applyProgramAndGroups(state, *commands); !applied)
        return applied;
    state.device->driver.dispatchCompute(descriptor->group_count[0], descriptor->group_count[1],
                                         descriptor->group_count[2]);
    if (auto recorded = recordBoundGroupWrites(adapter, *commands); !recorded)
        return recorded;
    if (auto recorded = recordProviderCommand(adapter, commands->encoder, false); !recorded)
        return recorded;
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

void applyStencilFunction(gl::Driver &driver, const PreparedPipeline &pipeline, uint32_t reference) {
    const auto &depthStencil = pipeline.depthStencil;
    driver.stencilFuncSeparate(gl::kFront, 0x0200 + depthStencil.front.compare, static_cast<gl::Int>(reference),
                               depthStencil.stencil_read_mask);
    driver.stencilFuncSeparate(gl::kBack, 0x0200 + depthStencil.back.compare, static_cast<gl::Int>(reference),
                               depthStencil.stencil_read_mask);
}

void applyFixedFunctionState(gl::Driver &driver, const PreparedPipeline &pipeline, uint32_t stencilReference) {
    const auto &rasterization = pipeline.rasterization;
    const auto &depthStencil = pipeline.depthStencil;
    if (rasterization.cull_mode == VERNON_RHI_CULL_NONE) {
        driver.disable(gl::kCullFace);
    } else {
        driver.enable(gl::kCullFace);
        driver.cullFace(rasterization.cull_mode == VERNON_RHI_CULL_FRONT ? gl::kFront : gl::kBack);
    }
    // GL vertex stages negate y so that image row 0 is the top, which mirrors every primitive's window-space winding.
    driver.frontFace(rasterization.front_face == VERNON_RHI_FRONT_FACE_CLOCKWISE ? gl::kCounterClockwise
                                                                                 : gl::kClockwise);
    if (rasterization.depth_bias_enabled) {
        driver.enable(0x8037);
        driver.polygonOffset(rasterization.depth_bias_slope, rasterization.depth_bias_constant);
    } else {
        driver.disable(0x8037);
    }
    if (depthStencil.depth_test) {
        driver.enable(gl::kDepthTest);
        driver.depthFunc(0x0200 + depthStencil.depth_compare);
    } else {
        driver.disable(gl::kDepthTest);
    }
    driver.depthMask(depthStencil.depth_write != 0);
    if (depthStencil.stencil_test) {
        driver.enable(gl::kStencilTest);
        applyStencilFunction(driver, pipeline, stencilReference);
        const std::array<std::pair<gl::Enum, const VernonStencilFaceState *>, 2> faces{
            {{gl::kFront, &depthStencil.front}, {gl::kBack, &depthStencil.back}}};
        for (const auto &[face, faceState] : faces) {
            driver.stencilOpSeparate(face, stencilOperation(faceState->stencil_fail),
                                     stencilOperation(faceState->depth_fail), stencilOperation(faceState->pass));
            driver.stencilMaskSeparate(face, depthStencil.stencil_write_mask);
        }
    } else {
        driver.disable(gl::kStencilTest);
    }
    for (size_t index = 0; index < pipeline.colorBlends.size(); ++index) {
        const auto &blend = pipeline.colorBlends[index];
        const auto red = static_cast<gl::Boolean>((blend.write_mask & VERNON_RHI_COLOR_WRITE_RED) != 0);
        const auto green = static_cast<gl::Boolean>((blend.write_mask & VERNON_RHI_COLOR_WRITE_GREEN) != 0);
        const auto blue = static_cast<gl::Boolean>((blend.write_mask & VERNON_RHI_COLOR_WRITE_BLUE) != 0);
        const auto alpha = static_cast<gl::Boolean>((blend.write_mask & VERNON_RHI_COLOR_WRITE_ALPHA) != 0);
        if (pipeline.colorBlends.size() == 1) {
            if (blend.blend_enabled)
                driver.enable(gl::kBlend);
            else
                driver.disable(gl::kBlend);
            driver.blendFuncSeparate(
                blendFactor(blend.source_color_factor), blendFactor(blend.destination_color_factor),
                blendFactor(blend.source_alpha_factor), blendFactor(blend.destination_alpha_factor));
            driver.blendEquationSeparate(blendOperation(blend.color_operation), blendOperation(blend.alpha_operation));
            driver.colorMask(red, green, blue, alpha);
            continue;
        }
        const auto buffer = static_cast<gl::Uint>(index);
        if (blend.blend_enabled)
            driver.enablei(gl::kBlend, buffer);
        else
            driver.disablei(gl::kBlend, buffer);
        driver.blendFuncSeparatei(buffer, blendFactor(blend.source_color_factor),
                                  blendFactor(blend.destination_color_factor), blendFactor(blend.source_alpha_factor),
                                  blendFactor(blend.destination_alpha_factor));
        driver.blendEquationSeparatei(buffer, blendOperation(blend.color_operation),
                                      blendOperation(blend.alpha_operation));
        driver.colorMaski(buffer, red, green, blue, alpha);
    }
}

// The adapter's vertex array keeps the attributes of the program and vertex sources that last specified them.
void applyVertexAttributes(OpenGLAdapterState &state, const OpenGLCommands &commands) {
    const PreparedPipeline &pipeline = *commands.pipeline;
    const PreparedLayout &layout = *pipeline.layout;
    const bool programChanged = state.vertexProgram != pipeline.native->serial;
    uint32_t changed = programChanged ? layout.vertexBindingMask : 0;
    for (uint32_t binding = 0; binding < kMaxVertexBindings; ++binding) {
        if ((layout.vertexBindingMask & (1u << binding)) == 0)
            continue;
        const VertexSource &bound = commands.vertexBuffers[binding];
        const VertexSource &applied = state.vertexSources[binding];
        if (applied.key != bound.key || applied.offset != bound.offset ||
            applied.stride != pipeline.vertexStrides[binding])
            changed |= 1u << binding;
    }
    if (!changed)
        return;
    auto &driver = state.device->driver;
    uint64_t disabled = state.enabledAttributes & ~layout.attributeMask;
    for (gl::Uint location = 0; disabled; ++location, disabled >>= 1)
        if (disabled & 1)
            driver.disableVertexAttribArray(location);
    for (const VertexAttribute &attribute : layout.vertexAttributes) {
        if ((changed & (1u << attribute.layout.binding)) == 0)
            continue;
        const VertexSource &source = commands.vertexBuffers[attribute.layout.binding];
        const uint32_t location = attribute.layout.location;
        driver.bindBuffer(gl::kArrayBuffer, source.name);
        if ((state.enabledAttributes & (uint64_t{1} << location)) == 0)
            driver.enableVertexAttribArray(location);
        const auto pointer =
            reinterpret_cast<const void *>(static_cast<uintptr_t>(source.offset + attribute.layout.relative_offset));
        const auto stride = static_cast<gl::Size>(pipeline.vertexStrides[attribute.layout.binding]);
        if (attribute.layout.dtype == VERNON_RUNTIME_PROVIDER_I32 ||
            attribute.layout.dtype == VERNON_RUNTIME_PROVIDER_U32)
            driver.vertexAttribIPointer(
                location, attribute.layout.component_count,
                attribute.layout.dtype == VERNON_RUNTIME_PROVIDER_I32 ? gl::kInt : gl::kUnsignedInt, stride, pointer);
        else if (attribute.layout.dtype == VERNON_RUNTIME_PROVIDER_F64)
            driver.vertexAttribLPointer(location, attribute.layout.component_count, gl::kDouble, stride, pointer);
        else
            driver.vertexAttribPointer(location, attribute.layout.component_count,
                                       attribute.layout.dtype == VERNON_RUNTIME_PROVIDER_F16 ? gl::kHalfFloat
                                                                                             : gl::kFloat,
                                       0, stride, pointer);
        driver.vertexAttribDivisor(location, attribute.divisor);
    }
    state.enabledAttributes = layout.attributeMask;
    state.vertexProgram = pipeline.native->serial;
    for (uint32_t binding = 0; binding < kMaxVertexBindings; ++binding)
        if ((changed & (1u << binding)) != 0) {
            state.vertexSources[binding] = commands.vertexBuffers[binding];
            state.vertexSources[binding].stride = pipeline.vertexStrides[binding];
        }
}

RhiAdapterResult<void> bindVertexBuffersResult(void *data, VernonRuntimeProviderObject handle, uint32_t firstBinding,
                                               const VernonRuntimeProviderResourceReference *references, size_t count) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<OpenGLCommands>(handle);
    if (!commands || !references || count == 0 || firstBinding > kMaxVertexBindings ||
        count > kMaxVertexBindings - firstBinding)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_bind_vertex_buffers_is_invalid");
    for (size_t index = 0; index < count; ++index) {
        const VernonRuntimeProviderResourceReference &reference = references[index];
        const size_t binding = firstBinding + index;
        if ((reference.identity & kRhiResourceKindMask) != kRhiBufferResource || reference.offset >= reference.size)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_vertex_buffer_is_invalid",
                                   binding);
        if (auto retained = retainCommandResource(adapter, commands->encoder, reference); !retained)
            return retained;
        auto resolved = resolveCommandRhiResource(adapter, commands->encoder, reference);
        if (!resolved)
            return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
        const auto name = static_cast<gl::Uint>(resolved.value());
        if (!name)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_vertex_buffer_is_stale",
                                   binding);
        commands->vertexBuffers[binding] = {reference.resource.value, name, reference.offset, 0};
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> applyIndexBuffer(VernonRuntimeRhiAdapter &adapter, OpenGLAdapterState &state,
                                        OpenGLCommands &commands, VernonRuntimeProviderResourceReference indexBuffer) {
    if (commands.indexBufferKey != indexBuffer.resource.value) {
        if (auto retained = retainCommandResource(adapter, commands.encoder, indexBuffer); !retained)
            return retained;
        auto resolved = resolveCommandRhiResource(adapter, commands.encoder, indexBuffer);
        if (!resolved)
            return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
        commands.indexBufferKey = indexBuffer.resource.value;
        commands.indexBuffer = static_cast<gl::Uint>(resolved.value());
    }
    if (state.indexBufferKey != indexBuffer.resource.value) {
        state.device->driver.bindBuffer(gl::kElementArrayBuffer, commands.indexBuffer);
        state.indexBufferKey = indexBuffer.resource.value;
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> encodeDrawResult(void *data, VernonRuntimeProviderObject handle,
                                        const VernonRuntimeProviderDrawDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto &state = openGLState(adapter);
    auto *commands = fromHandle<OpenGLCommands>(handle);
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !commands || !commands->pipeline ||
        commands->pipeline->compute || !commands->rendering || descriptor->topology > 2 ||
        (descriptor->index_count != 0 && descriptor->index_type != 0))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_adapter_received_an_invalid_draw");
    // gl_InstanceID excludes the base instance, so a nonzero first instance would change `instance_index`.
    if (descriptor->first_instance != 0)
        return providerFailure(vernon::ProviderErrorCode::Unsupported, "opengl_draw_first_instance_is_unsupported");
    auto &driver = state.device->driver;
    if (descriptor->index_count != 0 && descriptor->base_vertex != 0 && !driver.drawElementsInstancedBaseVertex)
        return providerFailure(vernon::ProviderErrorCode::Unsupported, "opengl_draw_base_vertex_is_unsupported");
    if (auto opened = openCommandRendering(adapter, commands->encoder); !opened)
        return RhiAdapterResult<void>{vernon::err(std::move(opened).error())};
    if (auto applied = applyProgramAndGroups(state, *commands); !applied)
        return applied;
    PreparedPipeline &pipeline = *commands->pipeline;
    if (commands->appliedPipeline != pipeline.serial) {
        applyFixedFunctionState(driver, pipeline, descriptor->stencil_reference);
        commands->appliedPipeline = pipeline.serial;
        commands->stencilReference = descriptor->stencil_reference;
    } else if (pipeline.depthStencil.stencil_test && commands->stencilReference != descriptor->stencil_reference) {
        applyStencilFunction(driver, pipeline, descriptor->stencil_reference);
        commands->stencilReference = descriptor->stencil_reference;
    }
    if (!commands->vertexArrayBound) {
        driver.bindVertexArray(state.vertexArray);
        commands->vertexArrayBound = true;
    }
    applyVertexAttributes(state, *commands);
    const std::array<uint32_t, 4> viewport{descriptor->viewport[0], descriptor->viewport[1], descriptor->viewport[2],
                                           descriptor->viewport[3]};
    if (!commands->viewportValid || commands->viewport != viewport) {
        driver.viewport(static_cast<gl::Int>(viewport[0]), static_cast<gl::Int>(viewport[1]),
                        static_cast<gl::Size>(viewport[2]), static_cast<gl::Size>(viewport[3]));
        commands->viewport = viewport;
        commands->viewportValid = true;
    }
    const std::array<uint32_t, 4> scissor{descriptor->scissor[0], descriptor->scissor[1], descriptor->scissor[2],
                                          descriptor->scissor[3]};
    if (!commands->scissorValid || commands->scissor != scissor) {
        driver.enable(gl::kScissorTest);
        driver.scissor(static_cast<gl::Int>(scissor[0]), static_cast<gl::Int>(scissor[1]),
                       static_cast<gl::Size>(scissor[2]), static_cast<gl::Size>(scissor[3]));
        commands->scissor = scissor;
        commands->scissorValid = true;
    }
    constexpr gl::Enum topologies[]{gl::kTriangles, gl::kLines, gl::kPoints};
    const gl::Enum topology = topologies[descriptor->topology];
    const auto instances = static_cast<gl::Size>(descriptor->instance_count);
    if (descriptor->index_count != 0) {
        if (auto bound = applyIndexBuffer(adapter, state, *commands, descriptor->index_buffer); !bound)
            return bound;
        const auto *indices = reinterpret_cast<const void *>(static_cast<uintptr_t>(descriptor->index_buffer.offset));
        const auto count = static_cast<gl::Size>(descriptor->index_count);
        if (descriptor->base_vertex != 0)
            driver.drawElementsInstancedBaseVertex(topology, count, gl::kUnsignedInt, indices, instances,
                                                   static_cast<gl::Int>(descriptor->base_vertex));
        else
            driver.drawElementsInstanced(topology, count, gl::kUnsignedInt, indices, instances);
    } else if (descriptor->instance_count == 1) {
        driver.drawArrays(topology, 0, static_cast<gl::Size>(descriptor->vertex_count));
    } else {
        driver.drawArraysInstanced(topology, 0, static_cast<gl::Size>(descriptor->vertex_count), instances);
    }
    if (auto recorded = recordBoundGroupWrites(adapter, *commands); !recorded)
        return recorded;
    if (auto recorded = recordProviderCommand(adapter, commands->encoder, true); !recorded)
        return recorded;
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    adapter.lastStencilReference.store(descriptor->stencil_reference, std::memory_order_relaxed);
    adapter.lastDrawIndexed.store(descriptor->index_count != 0, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> beginRenderingResult(void *data, VernonRuntimeProviderObject handle,
                                            const VernonRuntimeProviderRenderingDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<OpenGLCommands>(handle);
    if (!commands || !descriptor || commands->rendering)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_begin_rendering_is_invalid");
    if (auto begun = beginEncoderRendering(adapter, commands->encoder, *descriptor); !begun)
        return begun;
    commands->rendering = true;
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> endRenderingResult(void *data, VernonRuntimeProviderObject handle) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<OpenGLCommands>(handle);
    if (!commands || !commands->rendering)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_end_rendering_is_invalid");
    commands->rendering = false;
    if (auto ended = endEncoderRendering(adapter, commands->encoder); !ended)
        return ended;
    return pendingErrors(adapter, "opengl_rendering");
}

void destroyShader(void *, VernonRuntimeProviderObject handle) { delete fromHandle<PreparedShader>(handle); }
void destroyGroupLayout(void *, VernonRuntimeProviderObject handle) { delete fromHandle<GroupLayout>(handle); }
void destroyLayout(void *, VernonRuntimeProviderObject handle) { delete fromHandle<PreparedLayout>(handle); }
void destroyPipeline(void *, VernonRuntimeProviderObject handle) {
    releaseRetainedObject(fromHandle<PreparedPipeline>(handle));
}
void destroyGroup(void *, VernonRuntimeProviderObject handle) {
    releaseRetainedObject(fromHandle<OpenGLGroup>(handle));
}

VernonStatus prepareShader(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, prepareShaderResult(data, descriptor, output), "OpenGL shader preparation failed");
}
VernonStatus prepareGroupLayout(void *data, const VernonRuntimeProviderGroupLayoutDescriptor *descriptor,
                                VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, prepareGroupLayoutResult(descriptor, output),
                          "OpenGL group layout preparation failed");
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
VernonStatus createGroup(void *data, const VernonRuntimeProviderGroupDescriptor *descriptor,
                         VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, createGroupResult(data, descriptor, output), "OpenGL group creation failed");
}
VernonStatus beginCommands(void *data, VernonRuntimeProviderObject encoder, VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, beginCommandsResult(data, encoder, output), "OpenGL commands could not begin");
}
VernonStatus endCommands(void *data, VernonRuntimeProviderObject commands) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, endCommandsResult(data, commands), "OpenGL commands failed");
}
VernonStatus resumeCommands(void *data, VernonRuntimeProviderObject commands) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, resumeCommandsResult(data, commands), "OpenGL commands could not resume");
}
VernonStatus bindPipeline(void *data, VernonRuntimeProviderObject commands, VernonRuntimeProviderObject pipeline) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, bindPipelineResult(data, commands, pipeline), "OpenGL pipeline binding failed");
}
VernonStatus bindGroup(void *data, VernonRuntimeProviderObject commands, uint32_t index,
                       VernonRuntimeProviderObject group) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, bindGroupResult(data, commands, index, group), "OpenGL group binding failed");
}
VernonStatus bindVertexBuffers(void *data, VernonRuntimeProviderObject commands, uint32_t firstBinding,
                               const VernonRuntimeProviderResourceReference *buffers, size_t count) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, bindVertexBuffersResult(data, commands, firstBinding, buffers, count),
                          "OpenGL vertex buffer binding failed");
}
VernonStatus encodeDispatch(void *data, VernonRuntimeProviderObject commands,
                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, encodeDispatchResult(data, commands, descriptor), "OpenGL dispatch encoding failed");
}
VernonStatus encodeDraw(void *data, VernonRuntimeProviderObject commands,
                        const VernonRuntimeProviderDrawDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, encodeDrawResult(data, commands, descriptor), "OpenGL draw encoding failed");
}
VernonStatus beginRendering(void *data, VernonRuntimeProviderObject commands,
                            const VernonRuntimeProviderRenderingDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, beginRenderingResult(data, commands, descriptor),
                          "OpenGL rendering could not begin");
}
VernonStatus endRendering(void *data, VernonRuntimeProviderObject commands) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, endRenderingResult(data, commands), "OpenGL rendering could not end");
}

} // namespace

void initializeOpenGLProvider(VernonRuntimeRhiAdapter &adapter) {
    adapter.provider.struct_size = sizeof(adapter.provider);
    adapter.provider.abi_version = VERNON_PROGRAM_VERSION;
    adapter.provider.user_data = &adapter;
    adapter.provider.get_capabilities = getCapabilities;
    adapter.provider.get_device_identity = getDeviceIdentity;
    adapter.provider.prepare_shader = prepareShader;
    adapter.provider.prepare_group_layout = prepareGroupLayout;
    adapter.provider.prepare_pipeline_layout = prepareLayout;
    adapter.provider.prepare_pipeline = preparePipeline;
    adapter.provider.describe_image = describeProviderImageCallback;
    adapter.provider.create_group = createGroup;
    adapter.provider.begin_commands = beginCommands;
    adapter.provider.end_commands = endCommands;
    adapter.provider.resume_commands = resumeCommands;
    adapter.provider.encode_bind_pipeline = bindPipeline;
    adapter.provider.encode_bind_group = bindGroup;
    adapter.provider.encode_bind_vertex_buffers = bindVertexBuffers;
    adapter.provider.encode_dispatch = encodeDispatch;
    adapter.provider.encode_draw = encodeDraw;
    adapter.provider.begin_rendering = beginRendering;
    adapter.provider.end_rendering = endRendering;
    adapter.provider.destroy_shader = destroyShader;
    adapter.provider.destroy_group_layout = destroyGroupLayout;
    adapter.provider.destroy_pipeline_layout = destroyLayout;
    adapter.provider.destroy_pipeline = destroyPipeline;
    adapter.provider.destroy_group = destroyGroup;
}

} // namespace vernon::runtime::rhi_adapter

namespace vernon::runtime {

VernonRuntimeRhiAdapter *createOpenGLRhiAdapter(VernonRhiDevice device, VernonRhiBackend backend) {
    if (backend != VERNON_RHI_BACKEND_OPENGL && backend != VERNON_RHI_BACKEND_OPENGL_ES)
        return nullptr;
    auto resolvedDeviceState = rhi::deviceState(device, backend);
    if (resolvedDeviceState.isErr())
        return nullptr;
    auto *deviceState = static_cast<rhi::opengl::DeviceState *>(resolvedDeviceState.value());
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
