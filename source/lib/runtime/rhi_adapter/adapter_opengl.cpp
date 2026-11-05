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
    // Whether the device specifies attribute formats apart from vertex buffer bindings (OpenGL 4.3, OpenGL ES 3.1),
    // which makes a vertex buffer change one call per binding.
    bool separateFormats{};
    // Whether separate formats specified the vertex array last. Attribute pointers bind buffers at the binding of
    // each attribute's location, so a change of mode respecifies every attribute and binding.
    bool vertexArraySeparate{};
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
    // By vertex binding; zero for a per-vertex binding.
    std::array<uint32_t, kMaxVertexBindings> bindingDivisors{};
    // Whether the device's separate attribute formats can specify every attribute: their relative offsets are bounded.
    bool separateFormats{};
};

struct NativeProgram {
    gl::DeviceState *device{};
    gl::Uint program{};
    uint64_t serial{};
    const PreparedLayout *layout{};
    // Per group index and entry; -1 for entries that are not named uniforms or that the program does not use.
    std::array<std::vector<gl::Int>, kMaxGroups> locations;
    // Per group index and entry, 1 when the program reads the entry's context binding: an inactive uniform block,
    // sampled image or sampler unit is left as the context holds it.
    std::array<std::vector<uint8_t>, kMaxGroups> reads;
    // The serial of the group whose named uniforms the program holds at each group index. Uniform values are program
    // state that no other code changes, so they stay valid across commands.
    std::array<uint64_t, kMaxGroups> uniformGroups{};
    // SPIRV-Cross adds the first instance to gl_InstanceID through this uniform when the driver lacks
    // GL_ARB_shader_draw_parameters; -1 when the program reads gl_BaseInstanceARB or no instance index.
    gl::Int baseInstanceLocation{-1};
    uint32_t baseInstance{};

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
    // The inline buffer of an updatable group's slot.
    vernon::rhi::PersistentBlock updatable{};
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
    // Per group index, the entries of the applied group whose context bindings were applied, as programs read them.
    std::array<std::vector<uint8_t>, kMaxGroups> appliedReads;
    // Per context binding namespace (each application before NamedUniform) and binding point, one more than the group
    // index whose value the point holds, or 0. Points are context state that every group index shares: a group applied
    // at one index takes its points from the other indices, whose groups apply again before their next use.
    std::array<std::vector<uint8_t>, static_cast<size_t>(EntryApplication::NamedUniform)> bindingOwners;
    // Per context binding namespace and point, the slot whose object the point holds, so that a group binding what a
    // point holds already leaves it alone. Names stand for objects here because every OpenGL call that could delete
    // or create one between two applications invalidates the commands.
    std::array<std::vector<GroupSlot>, static_cast<size_t>(EntryApplication::NamedUniform)> boundSlots;
    // The buffers bound at each vertex binding; their strides come from the pipeline.
    std::array<VertexSource, kMaxVertexBindings> vertexBuffers{};
    bool vertexArrayBound{};
    bool viewportValid{};
    bool scissorValid{};
    uint32_t stencilReference{};
    std::array<uint32_t, 4> viewport{};
    std::array<uint32_t, 4> scissor{};

    void invalidate(const OpenGLAdapterState &state) {
        epoch = state.epoch;
        generation = state.device->drawStateGeneration;
        appliedProgram = 0;
        appliedPipeline = 0;
        appliedGroups.fill(0);
        for (std::vector<uint8_t> &owners : bindingOwners)
            owners.clear();
        for (std::vector<GroupSlot> &slots : boundSlots)
            slots.clear();
        vertexArrayBound = false;
        viewportValid = false;
        scissorValid = false;
    }
};

bool openGLVersionAtLeast(const VernonRuntimeRhiAdapter &adapter, uint16_t major, uint16_t minor) {
    const VernonOpenGLContextCallbacks &callbacks = openGLState(adapter).device->callbacks;
    return adapter.rhiBackend == VERNON_RHI_BACKEND_OPENGL &&
           (callbacks.api_version_major > major ||
            (callbacks.api_version_major == major && callbacks.api_version_minor >= minor));
}

// Device streams are multi-draws, core in OpenGL 4.3; OpenGL ES has none.
bool deviceDrawStreams(const VernonRuntimeRhiAdapter &adapter) {
    return openGLVersionAtLeast(adapter, 4, 3) && openGLState(adapter).device->driver.multiDrawElementsIndirect;
}

uint32_t getCapabilities(void *data) {
    return VERNON_RUNTIME_PROVIDER_COMPUTE | VERNON_RUNTIME_PROVIDER_GRAPHICS | VERNON_RUNTIME_PROVIDER_NATIVE_INTEROP |
           VERNON_RUNTIME_PROVIDER_HOST_DRAW_STREAMS |
           (deviceDrawStreams(*static_cast<const VernonRuntimeRhiAdapter *>(data))
                ? VERNON_RUNTIME_PROVIDER_DEVICE_DRAW_STREAMS
                : 0u);
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
        std::array<uint32_t, kMaxVertexBindings> &divisors = layout->bindingDivisors;
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
        const OpenGLAdapterState &state = openGLState(adapter);
        layout->separateFormats = state.separateFormats;
        if (state.separateFormats && !layout->vertexAttributes.empty()) {
            gl::Int maximumOffset = 0;
            device.driver.getIntegerv(gl::kMaxVertexAttribRelativeOffset, &maximumOffset);
            for (const VertexAttribute &attribute : layout->vertexAttributes)
                if (attribute.layout.relative_offset > static_cast<uint32_t>(std::max(maximumOffset, 0)) ||
                    (attribute.layout.dtype == VERNON_RUNTIME_PROVIDER_F64 && !device.driver.vertexAttribLFormat))
                    layout->separateFormats = false;
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
    for (uint32_t group = 0; group < layout.groupCount; ++group)
        native.reads[group].assign(layout.groups[group]->entries.size(), 1);
    // A block left unassigned silently reads binding point 0, so every active block must have a planned binding.
    std::vector<gl::Uint> assignedBlocks;
    for (uint32_t group = 0; group < layout.groupCount; ++group)
        for (size_t index = 0; index < layout.groups[group]->entries.size(); ++index) {
            const GroupLayout::Entry &entry = layout.groups[group]->entries[index];
            if (entry.layout.kind != VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER)
                continue;
            const std::string blockName = ::vernon::runtime::openGLUniformBlockName(entry.layout.binding);
            const gl::Uint block = driver.getUniformBlockIndex(native.program, blockName.c_str());
            if (block == gl::kInvalidIndex) {
                native.reads[group][index] = 0;
                continue;
            }
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
    native.baseInstanceLocation = driver.getUniformLocation(native.program, "SPIRV_Cross_BaseInstance");
    // A separate sampler serves the texture unit of its binding, which only active sampled images read.
    std::vector<uint32_t> sampledUnits;
    for (uint32_t group = 0; group < layout.groupCount; ++group) {
        const auto &entries = layout.groups[group]->entries;
        native.locations[group].assign(entries.size(), -1);
        for (size_t index = 0; index < entries.size(); ++index) {
            const GroupLayout::Entry &entry = entries[index];
            if (entry.application != EntryApplication::NamedUniform &&
                entry.application != EntryApplication::SampledImage)
                continue;
            const gl::Int location = driver.getUniformLocation(native.program, entry.name.c_str());
            if (location < 0) {
                native.reads[group][index] = 0;
                continue;
            }
            if (entry.application == EntryApplication::SampledImage) {
                driver.uniform1i(location, static_cast<gl::Int>(entry.layout.binding));
                sampledUnits.push_back(entry.layout.binding);
            } else {
                native.locations[group][index] = location;
            }
        }
    }
    for (uint32_t group = 0; group < layout.groupCount; ++group) {
        const auto &entries = layout.groups[group]->entries;
        for (size_t index = 0; index < entries.size(); ++index)
            if (entries[index].application == EntryApplication::Sampler)
                native.reads[group][index] = std::find(sampledUnits.begin(), sampledUnits.end(),
                                                       entries[index].layout.binding) != sampledUnits.end();
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
    } else if (descriptor.lifetime == VERNON_RUNTIME_PROVIDER_GROUP_UPDATABLE) {
        auto placed = placeUpdatable(adapter, group, size, aligned);
        if (!placed)
            return RhiAdapterResult<void>{vernon::err(std::move(placed).error())};
        slot.updatable = placed.value();
        slot.name = static_cast<gl::Uint>(placed.value().buffer);
        slot.offset = static_cast<size_t>(placed.value().offset);
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

RhiAdapterResult<void> updateGroupResult(void *data, VernonRuntimeProviderObject commandsHandle,
                                         VernonRuntimeProviderObject groupHandle,
                                         const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<OpenGLCommands>(commandsHandle);
    auto *group = fromHandle<OpenGLGroup>(groupHandle);
    if (!commands || !group || commands->rendering || valueCount != group->slots.size() || (valueCount && !values))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_group_update_is_invalid");
    bool named = false;
    bool written = false;
    for (size_t index = 0; index < valueCount; ++index) {
        const GroupLayout::Entry &entry = group->layout->entries[index];
        if (!packedUniformBytes(entry.layout.kind, entry.layout.interface_kind))
            continue;
        const auto &bytes = values[index].payload.inline_value;
        if (!bytes.data || bytes.size != entry.layout.element_size)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_inline_binding_is_invalid",
                                   entry.layout.slot);
        const GroupSlot &slot = group->slots[index];
        if (entry.inlineBuffer) {
            if (auto write = writeUpdatable(adapter, commands->encoder, slot.updatable, bytes.data, bytes.size); !write)
                return write;
            written = true;
        } else {
            std::memcpy(group->values.data() + slot.valueOffset, bytes.data, bytes.size);
            named = true;
        }
    }
    // Commands and programs re-apply named uniforms only for a group serial they have not applied.
    if (named)
        group->serial = ++openGLState(adapter).nextSerial;
    // A buffer write rebinds the array buffer.
    if (written)
        commands->invalidate(openGLState(adapter));
    return RhiAdapterResult<void>{vernon::ok()};
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
    if (state.device->validation)
        for (int drained = 0; drained < 16 && state.device->readError() != 0; ++drained) {
        }
    ++state.epoch;
    commands->invalidate(state);
    *output = toHandle(commands.release());
    return RhiAdapterResult<void>{vernon::ok()};
}

// Reports the OpenGL errors raised since the last check, which the commands just encoded raised. Outside validation
// mode they fail the submission's completion instead.
RhiAdapterResult<void> pendingErrors(VernonRuntimeRhiAdapter &adapter, const char *operation) {
    gl::DeviceState &device = *openGLState(adapter).device;
    if (!device.validation)
        return RhiAdapterResult<void>{vernon::ok()};
    const gl::Enum error = device.readError();
    if (error == 0)
        return RhiAdapterResult<void>{vernon::ok()};
    for (int drained = 0; drained < 16 && device.readError() != 0; ++drained) {
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

void bindIndexedBuffer(gl::Driver &driver, gl::Enum target, const GroupSlot &slot) {
    if (slot.offset == 0)
        driver.bindBufferBase(target, slot.binding, slot.name);
    else
        driver.bindBufferRange(target, slot.binding, slot.name, static_cast<gl::IntPtr>(slot.offset),
                               static_cast<gl::SizePtr>(slot.range));
}

// Whether binding `slot` leaves its context binding point as `held` left it.
bool sameBinding(const GroupSlot &held, const GroupSlot &slot) {
    if (held.name != slot.name)
        return false;
    switch (slot.application) {
    case EntryApplication::UniformBuffer:
    case EntryApplication::StorageBuffer:
        return held.offset == slot.offset && (slot.offset == 0 || held.range == slot.range);
    case EntryApplication::SampledImage:
        return held.textureTarget == slot.textureTarget;
    case EntryApplication::Sampler:
        return true;
    case EntryApplication::StorageImage:
        return held.layered == slot.layered && held.imageAccess == slot.imageAccess &&
               held.imageFormat == slot.imageFormat;
    case EntryApplication::NamedUniform:
        break;
    }
    return false;
}

// Applies the context bindings of the group's entries that `reads` marks.
void applyGroupBindings(gl::Driver &driver, OpenGLCommands &commands, uint32_t index, const OpenGLGroup &group,
                        const std::vector<uint8_t> &reads) {
    for (size_t entry = 0; entry < group.slots.size(); ++entry) {
        const GroupSlot &slot = group.slots[entry];
        if (!reads[entry])
            continue;
        if (slot.application < EntryApplication::NamedUniform) {
            std::vector<uint8_t> &owners = commands.bindingOwners[static_cast<size_t>(slot.application)];
            if (owners.size() <= slot.binding)
                owners.resize(slot.binding + 1, 0);
            const uint8_t owner = owners[slot.binding];
            if (owner != 0 && owner != index + 1)
                commands.appliedGroups[owner - 1] = 0;
            owners[slot.binding] = static_cast<uint8_t>(index + 1);
            std::vector<GroupSlot> &bound = commands.boundSlots[static_cast<size_t>(slot.application)];
            if (bound.size() <= slot.binding)
                bound.resize(slot.binding + 1);
            GroupSlot &held = bound[slot.binding];
            if (held.name != 0 && sameBinding(held, slot))
                continue;
            held = slot;
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
        const std::vector<uint8_t> &reads = native.reads[index];
        std::vector<uint8_t> &applied = commands.appliedReads[index];
        const bool same = commands.appliedGroups[index] == group->serial;
        bool covered = same;
        for (size_t entry = 0; covered && entry < reads.size(); ++entry)
            covered = !reads[entry] || applied[entry];
        if (!covered) {
            applyGroupBindings(driver, commands, index, *group, reads);
            if (same)
                for (size_t entry = 0; entry < reads.size(); ++entry)
                    applied[entry] |= reads[entry];
            else
                applied = reads;
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
// `firstInstance` moves each per-instance binding of divisor d forward by firstInstance / d records, which emulates a
// first instance on draws without a base instance.
void applyVertexAttributes(OpenGLAdapterState &state, const OpenGLCommands &commands, uint32_t firstInstance = 0) {
    const PreparedPipeline &pipeline = *commands.pipeline;
    const PreparedLayout &layout = *pipeline.layout;
    // A zero stride packs a binding's attributes tightly for attribute pointers only.
    bool separate = layout.separateFormats;
    for (uint32_t binding = 0; separate && binding < kMaxVertexBindings; ++binding)
        separate = (layout.vertexBindingMask & (1u << binding)) == 0 || pipeline.vertexStrides[binding] != 0;
    if (state.vertexArraySeparate != separate) {
        state.vertexArraySeparate = separate;
        state.vertexProgram = 0;
        state.vertexSources.fill({});
    }
    const bool programChanged = state.vertexProgram != pipeline.native->serial;
    // Separate formats keep the buffers of the bindings across format changes.
    uint32_t changed = programChanged && !separate ? layout.vertexBindingMask : 0;
    std::array<uint64_t, kMaxVertexBindings> offsets{};
    for (uint32_t binding = 0; binding < kMaxVertexBindings; ++binding) {
        if ((layout.vertexBindingMask & (1u << binding)) == 0)
            continue;
        const VertexSource &bound = commands.vertexBuffers[binding];
        const VertexSource &applied = state.vertexSources[binding];
        offsets[binding] = bound.offset;
        if (layout.bindingDivisors[binding])
            offsets[binding] +=
                uint64_t{firstInstance / layout.bindingDivisors[binding]} * pipeline.vertexStrides[binding];
        if (applied.key != bound.key || applied.offset != offsets[binding] ||
            applied.stride != pipeline.vertexStrides[binding])
            changed |= 1u << binding;
    }
    if (!changed && !(separate && programChanged))
        return;
    auto &driver = state.device->driver;
    const auto record = [&] {
        for (uint32_t binding = 0; binding < kMaxVertexBindings; ++binding)
            if ((changed & (1u << binding)) != 0) {
                state.vertexSources[binding] = commands.vertexBuffers[binding];
                state.vertexSources[binding].offset = offsets[binding];
                state.vertexSources[binding].stride = pipeline.vertexStrides[binding];
            }
    };
    if (separate) {
        if (programChanged) {
            uint64_t disabled = state.enabledAttributes & ~layout.attributeMask;
            for (gl::Uint location = 0; disabled; ++location, disabled >>= 1)
                if (disabled & 1)
                    driver.disableVertexAttribArray(location);
            for (const VertexAttribute &attribute : layout.vertexAttributes) {
                const uint32_t location = attribute.layout.location;
                if ((state.enabledAttributes & (uint64_t{1} << location)) == 0)
                    driver.enableVertexAttribArray(location);
                const auto components = static_cast<gl::Int>(attribute.layout.component_count);
                const uint32_t dtype = attribute.layout.dtype;
                if (dtype == VERNON_RUNTIME_PROVIDER_I32 || dtype == VERNON_RUNTIME_PROVIDER_U32)
                    driver.vertexAttribIFormat(location, components,
                                               dtype == VERNON_RUNTIME_PROVIDER_I32 ? gl::kInt : gl::kUnsignedInt,
                                               attribute.layout.relative_offset);
                else if (dtype == VERNON_RUNTIME_PROVIDER_F64)
                    driver.vertexAttribLFormat(location, components, gl::kDouble, attribute.layout.relative_offset);
                else
                    driver.vertexAttribFormat(location, components,
                                              dtype == VERNON_RUNTIME_PROVIDER_F16 ? gl::kHalfFloat : gl::kFloat, 0,
                                              attribute.layout.relative_offset);
                driver.vertexAttribBinding(location, attribute.layout.binding);
            }
            for (uint32_t binding = 0; binding < kMaxVertexBindings; ++binding)
                if ((layout.vertexBindingMask & (1u << binding)) != 0)
                    driver.vertexBindingDivisor(binding, layout.bindingDivisors[binding]);
            state.enabledAttributes = layout.attributeMask;
            state.vertexProgram = pipeline.native->serial;
        }
        for (uint32_t binding = 0; binding < kMaxVertexBindings; ++binding)
            if ((changed & (1u << binding)) != 0)
                driver.bindVertexBuffer(binding, commands.vertexBuffers[binding].name,
                                        static_cast<gl::IntPtr>(offsets[binding]),
                                        static_cast<gl::Size>(pipeline.vertexStrides[binding]));
        record();
        return;
    }
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
        const auto pointer = reinterpret_cast<const void *>(
            static_cast<uintptr_t>(offsets[attribute.layout.binding] + attribute.layout.relative_offset));
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
    record();
}

struct OpenGLPacketDecoder {
    VernonRuntimeRhiAdapter &adapter;
    OpenGLAdapterState &state;
    OpenGLCommands &commands;
    const RhiPacketBatch &batch;

    RhiAdapterResult<void> beginRendering() {
        commands.rendering = true;
        return RhiAdapterResult<void>{vernon::ok()};
    }

    RhiAdapterResult<void> endRendering() {
        commands.rendering = false;
        return pendingErrors(adapter, "opengl_rendering");
    }

    RhiAdapterResult<void> bindPipeline(RhiRetainedObject &object) {
        auto &pipeline = static_cast<PreparedPipeline &>(object);
        for (uint32_t group = 0; group < kMaxGroups; ++group)
            if (commands.groups[group] && (group >= pipeline.layout->groupCount ||
                                           commands.groups[group]->layout != pipeline.layout->groups[group]))
                commands.groups[group] = nullptr;
        commands.pipeline = &pipeline;
        return RhiAdapterResult<void>{vernon::ok()};
    }

    RhiAdapterResult<void> bindGroup(uint32_t index, RhiGroup &object) {
        auto &group = static_cast<OpenGLGroup &>(object);
        if (!commands.pipeline || index >= commands.pipeline->layout->groupCount ||
            group.layout != commands.pipeline->layout->groups[index])
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_bind_group_is_invalid", index);
        commands.groups[index] = &group;
        return RhiAdapterResult<void>{vernon::ok()};
    }

    RhiAdapterResult<void> bindVertexBuffers(uint32_t first, const uint32_t *resources, size_t count) {
        for (size_t index = 0; index < count; ++index) {
            const VernonRuntimeProviderResourceReference &reference = batch.resources[resources[index]];
            const auto name = static_cast<gl::Uint>(batch.natives[resources[index]]);
            if (!name)
                return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_vertex_buffer_is_stale",
                                       first + index);
            commands.vertexBuffers[first + index] = {reference.resource.value, name, reference.offset, 0};
        }
        return RhiAdapterResult<void>{vernon::ok()};
    }

    RhiAdapterResult<void> dispatch(const uint32_t *groupCount) {
        if (!commands.pipeline || !commands.pipeline->compute || commands.rendering)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "opengl_adapter_received_an_invalid_compute_dispatch");
        if (auto applied = applyProgramAndGroups(state, commands); !applied)
            return applied;
        state.device->driver.dispatchCompute(groupCount[0], groupCount[1], groupCount[2]);
        return RhiAdapterResult<void>{vernon::ok()};
    }

    RhiAdapterResult<void> draw(uint32_t topologyIndex, uint32_t indexResource,
                                const VernonRuntimeProviderDrawPayload &draw, const vernon::rhi::OpenedRendering &) {
        if (auto prepared = prepareDraw(topologyIndex, draw.viewport, draw.scissor, draw.stencil_reference); !prepared)
            return prepared;
        return issueDraw(topologyIndex, indexResource, draw.vertex_count, draw.instance_count, draw.first_index,
                         draw.base_vertex, draw.first_instance);
    }

    // A host stream's records become one native draw each; a device stream is one multi-draw.
    RhiAdapterResult<void> drawStream(uint32_t topologyIndex, uint32_t indexResource, uint32_t recordsResource,
                                      uint32_t countsResource, const VernonRuntimeProviderStreamPayload &stream,
                                      const vernon::rhi::OpenedRendering &) {
        if (recordsResource != UINT32_MAX)
            return drawDeviceStream(topologyIndex, indexResource, recordsResource, countsResource, stream);
        const auto address = [](const uint32_t *words) {
            return static_cast<uintptr_t>(words[0] | (uint64_t{words[1]} << 32));
        };
        const auto *records = reinterpret_cast<const VernonRuntimeProviderDrawRecord *>(address(stream.host_records));
        const auto *counts = reinterpret_cast<const uint32_t *>(address(stream.host_counts));
        const uint32_t count = std::min(counts[stream.count_index], stream.max_records);
        // Streams whose records draw nothing leave the context alone, so a table that keeps an entry for every batch
        // costs only its drawn entries.
        bool prepared = false;
        for (uint32_t index = 0; index < count; ++index) {
            const VernonRuntimeProviderDrawRecord &record = records[stream.first_record + index];
            if (record.index_count == 0 || record.instance_count == 0)
                continue;
            if (!prepared) {
                if (auto state = prepareDraw(topologyIndex, stream.viewport, stream.scissor, stream.stencil_reference);
                    !state)
                    return state;
                prepared = true;
            }
            if (auto issued = issueDraw(topologyIndex, indexResource, record.index_count, record.instance_count,
                                        record.first_index, record.base_vertex, record.first_instance);
                !issued)
                return issued;
        }
        return RhiAdapterResult<void>{vernon::ok()};
    }

    // The records are DrawElementsIndirectCommands, whose base instance offsets the per-instance attributes. One
    // uniform cannot give each draw its base instance, so a program that emulates gl_BaseInstance through
    // SPIRV_Cross_BaseInstance cannot draw a device stream; nor can an index binding with an offset, since the
    // records' first indices count from the start of the bound buffer.
    RhiAdapterResult<void> drawDeviceStream(uint32_t topologyIndex, uint32_t indexResource, uint32_t recordsResource,
                                            uint32_t countsResource, const VernonRuntimeProviderStreamPayload &stream) {
        auto &driver = state.device->driver;
        const bool counted = countsResource != UINT32_MAX;
        if (!deviceDrawStreams(adapter) ||
            (counted && (!openGLVersionAtLeast(adapter, 4, 6) || !driver.multiDrawElementsIndirectCount)))
            return providerFailure(vernon::ProviderErrorCode::Unsupported,
                                   "opengl_device_draw_streams_are_unsupported");
        if (stream.max_records == 0)
            return RhiAdapterResult<void>{vernon::ok()};
        if (auto prepared = prepareDraw(topologyIndex, stream.viewport, stream.scissor, stream.stencil_reference);
            !prepared)
            return prepared;
        if (commands.pipeline->native->baseInstanceLocation >= 0)
            return providerFailure(vernon::ProviderErrorCode::Unsupported,
                                   "opengl_device_draw_streams_need_native_base_instances");
        const VernonRuntimeProviderResourceReference &indexBuffer = batch.resources[indexResource];
        if (indexBuffer.offset != 0)
            return providerFailure(vernon::ProviderErrorCode::Unsupported,
                                   "opengl_device_draw_streams_need_an_index_binding_at_offset_zero");
        applyVertexAttributes(state, commands, 0);
        if (state.indexBufferKey != indexBuffer.resource.value) {
            driver.bindBuffer(gl::kElementArrayBuffer, static_cast<gl::Uint>(batch.natives[indexResource]));
            state.indexBufferKey = indexBuffer.resource.value;
        }
        driver.bindBuffer(gl::kDrawIndirectBuffer, static_cast<gl::Uint>(batch.natives[recordsResource]));
        constexpr gl::Enum topologies[]{gl::kTriangles, gl::kLines, gl::kPoints};
        const auto *records = reinterpret_cast<const void *>(
            static_cast<uintptr_t>(batch.resources[recordsResource].offset +
                                   uint64_t{stream.first_record} * sizeof(VernonRuntimeProviderDrawRecord)));
        constexpr auto stride = static_cast<gl::Size>(sizeof(VernonRuntimeProviderDrawRecord));
        if (!counted) {
            driver.multiDrawElementsIndirect(topologies[topologyIndex], gl::kUnsignedInt, records,
                                             static_cast<gl::Size>(stream.max_records), stride);
            return RhiAdapterResult<void>{vernon::ok()};
        }
        driver.bindBuffer(gl::kParameterBuffer, static_cast<gl::Uint>(batch.natives[countsResource]));
        driver.multiDrawElementsIndirectCount(
            topologies[topologyIndex], gl::kUnsignedInt, records,
            static_cast<gl::IntPtr>(batch.resources[countsResource].offset + uint64_t{stream.count_index} * 4),
            static_cast<gl::Size>(stream.max_records), stride);
        return RhiAdapterResult<void>{vernon::ok()};
    }

    // Applies the program, groups, fixed-function and dynamic state of the draws that follow.
    RhiAdapterResult<void> prepareDraw(uint32_t topologyIndex, const uint32_t *viewportRect,
                                       const uint32_t *scissorRect, uint32_t stencilReference) {
        if (!commands.pipeline || commands.pipeline->compute || !commands.rendering || topologyIndex > 2)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "opengl_adapter_received_an_invalid_draw");
        auto &driver = state.device->driver;
        if (auto applied = applyProgramAndGroups(state, commands); !applied)
            return applied;
        PreparedPipeline &pipeline = *commands.pipeline;
        if (commands.appliedPipeline != pipeline.serial) {
            applyFixedFunctionState(driver, pipeline, stencilReference);
            commands.appliedPipeline = pipeline.serial;
            commands.stencilReference = stencilReference;
        } else if (pipeline.depthStencil.stencil_test && commands.stencilReference != stencilReference) {
            applyStencilFunction(driver, pipeline, stencilReference);
            commands.stencilReference = stencilReference;
        }
        if (!commands.vertexArrayBound) {
            driver.bindVertexArray(state.vertexArray);
            commands.vertexArrayBound = true;
        }
        const std::array<uint32_t, 4> viewport{viewportRect[0], viewportRect[1], viewportRect[2], viewportRect[3]};
        if (!commands.viewportValid || commands.viewport != viewport) {
            driver.viewport(static_cast<gl::Int>(viewport[0]), static_cast<gl::Int>(viewport[1]),
                            static_cast<gl::Size>(viewport[2]), static_cast<gl::Size>(viewport[3]));
            commands.viewport = viewport;
            commands.viewportValid = true;
        }
        const std::array<uint32_t, 4> scissor{scissorRect[0], scissorRect[1], scissorRect[2], scissorRect[3]};
        if (!commands.scissorValid || commands.scissor != scissor) {
            driver.enable(gl::kScissorTest);
            driver.scissor(static_cast<gl::Int>(scissor[0]), static_cast<gl::Int>(scissor[1]),
                           static_cast<gl::Size>(scissor[2]), static_cast<gl::Size>(scissor[3]));
            commands.scissor = scissor;
            commands.scissorValid = true;
        }
        return RhiAdapterResult<void>{vernon::ok()};
    }

    // One draw after prepareDraw. gl_InstanceID excludes the base instance, so a first instance reaches
    // `instance_index` through gl_BaseInstanceARB, which only base-instance draws set, or through the program's
    // SPIRV_Cross_BaseInstance uniform; per-instance attributes start at it through the base instance or, without
    // base-instance draws, through offset attribute pointers.
    RhiAdapterResult<void> issueDraw(uint32_t topologyIndex, uint32_t indexResource, uint32_t vertexCount,
                                     uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex,
                                     uint32_t firstInstance) {
        auto &driver = state.device->driver;
        PreparedPipeline &pipeline = *commands.pipeline;
        const bool indexed = indexResource != UINT32_MAX;
        if (indexed && baseVertex != 0 && !driver.drawElementsInstancedBaseVertex)
            return providerFailure(vernon::ProviderErrorCode::Unsupported, "opengl_draw_base_vertex_is_unsupported");
        const bool baseInstanceDraw =
            firstInstance != 0 && (indexed ? driver.drawElementsInstancedBaseVertexBaseInstance != nullptr
                                           : driver.drawArraysInstancedBaseInstance != nullptr);
        if (firstInstance != 0 && !baseInstanceDraw) {
            // An attribute of divisor d reads record (first + i) / d, which an offset reaches only when d divides
            // first.
            for (uint32_t binding = 0; binding < kMaxVertexBindings; ++binding)
                if (pipeline.layout->bindingDivisors[binding] > 1 &&
                    firstInstance % pipeline.layout->bindingDivisors[binding] != 0)
                    return providerFailure(vernon::ProviderErrorCode::Unsupported,
                                           "opengl_first_instance_is_not_a_multiple_of_the_divisor", binding);
        }
        applyVertexAttributes(state, commands, baseInstanceDraw ? 0 : firstInstance);
        NativeProgram &native = *pipeline.native;
        if (native.baseInstanceLocation >= 0 && native.baseInstance != firstInstance) {
            driver.uniform1i(native.baseInstanceLocation, static_cast<gl::Int>(firstInstance));
            native.baseInstance = firstInstance;
        }
        constexpr gl::Enum topologies[]{gl::kTriangles, gl::kLines, gl::kPoints};
        const gl::Enum topology = topologies[topologyIndex];
        const auto instances = static_cast<gl::Size>(instanceCount);
        if (indexed) {
            const VernonRuntimeProviderResourceReference &indexBuffer = batch.resources[indexResource];
            if (state.indexBufferKey != indexBuffer.resource.value) {
                driver.bindBuffer(gl::kElementArrayBuffer, static_cast<gl::Uint>(batch.natives[indexResource]));
                state.indexBufferKey = indexBuffer.resource.value;
            }
            const auto *indices = reinterpret_cast<const void *>(
                static_cast<uintptr_t>(indexBuffer.offset + uint64_t{firstIndex} * sizeof(uint32_t)));
            const auto count = static_cast<gl::Size>(vertexCount);
            if (baseInstanceDraw)
                driver.drawElementsInstancedBaseVertexBaseInstance(topology, count, gl::kUnsignedInt, indices,
                                                                   instances, static_cast<gl::Int>(baseVertex),
                                                                   firstInstance);
            else if (baseVertex != 0)
                driver.drawElementsInstancedBaseVertex(topology, count, gl::kUnsignedInt, indices, instances,
                                                       static_cast<gl::Int>(baseVertex));
            else
                driver.drawElementsInstanced(topology, count, gl::kUnsignedInt, indices, instances);
        } else if (baseInstanceDraw) {
            driver.drawArraysInstancedBaseInstance(topology, 0, static_cast<gl::Size>(vertexCount), instances,
                                                   firstInstance);
        } else if (instanceCount == 1) {
            driver.drawArrays(topology, 0, static_cast<gl::Size>(vertexCount));
        } else {
            driver.drawArraysInstanced(topology, 0, static_cast<gl::Size>(vertexCount), instances);
        }
        return RhiAdapterResult<void>{vernon::ok()};
    }
};

RhiRetainedObject *packetPipeline(VernonRuntimeProviderObject handle) noexcept {
    return fromHandle<PreparedPipeline>(handle);
}
RhiGroup *packetGroup(VernonRuntimeProviderObject handle) noexcept { return fromHandle<OpenGLGroup>(handle); }

// A batch starts with nothing bound, so the bindings of earlier batches, whose objects may be gone, are dropped.
RhiAdapterResult<void> encodePacketsResult(void *data, VernonRuntimeProviderObject handle,
                                           VernonRuntimeProviderObject batchHandle, const uint32_t *payload,
                                           size_t payloadWords) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<OpenGLCommands>(handle);
    auto *batch = fromHandle<RhiPacketBatch>(batchHandle);
    if (!commands || !batch || commands->rendering)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "opengl_encode_packets_is_invalid");
    commands->pipeline = nullptr;
    commands->groups.fill(nullptr);
    commands->vertexBuffers.fill({});
    OpenGLPacketDecoder decoder{adapter, openGLState(adapter), *commands, *batch};
    return decodeRhiPackets(adapter, commands->encoder, *batch, payload, payloadWords, decoder);
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
VernonStatus updateGroup(void *data, VernonRuntimeProviderObject commands, VernonRuntimeProviderObject group,
                         const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, updateGroupResult(data, commands, group, values, valueCount),
                          "OpenGL group update failed");
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
VernonStatus encodePackets(void *data, VernonRuntimeProviderObject commands, VernonRuntimeProviderObject batch,
                           const uint32_t *payload, size_t payloadWords) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, encodePacketsResult(data, commands, batch, payload, payloadWords),
                          "OpenGL packet encoding failed");
}
VernonStatus createPacketBatch(void *data, const VernonRuntimeProviderPacketBatchDescriptor *descriptor,
                               VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, createRhiPacketBatch(adapter, descriptor, {packetPipeline, packetGroup}, output),
                          "OpenGL packet batch creation failed");
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
    adapter.provider.create_packet_batch = createPacketBatch;
    adapter.provider.encode_packets = encodePackets;
    adapter.provider.update_group = updateGroup;
    adapter.provider.destroy_shader = destroyShader;
    adapter.provider.destroy_group_layout = destroyGroupLayout;
    adapter.provider.destroy_pipeline_layout = destroyLayout;
    adapter.provider.destroy_pipeline = destroyPipeline;
    adapter.provider.destroy_group = destroyGroup;
    adapter.provider.destroy_packet_batch = destroyRhiPacketBatch;
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
    const VernonOpenGLContextCallbacks &callbacks = deviceState->callbacks;
    const auto [major, minor] = backend == VERNON_RHI_BACKEND_OPENGL ? std::pair{4, 3} : std::pair{3, 1};
    const rhi::opengl::Driver &driver = deviceState->driver;
    state->separateFormats = (callbacks.api_version_major > major ||
                              (callbacks.api_version_major == major && callbacks.api_version_minor >= minor)) &&
                             driver.bindVertexBuffer && driver.vertexAttribFormat && driver.vertexAttribIFormat &&
                             driver.vertexAttribBinding && driver.vertexBindingDivisor;
    adapter->rhiBackend = backend;
    if (!adapter->backend.adopt(state.get(), &rhi_adapter::backendOps))
        return nullptr;
    [[maybe_unused]] auto *adoptedState = state.release();
    rhi_adapter::initializeOpenGLProvider(*adapter);
    return adapter.release();
}

} // namespace vernon::runtime
