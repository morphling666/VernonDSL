#include "adapter_common.h"
#include "adapter_internal.h"
#include "rhi/rhi_internal.h"

#if defined(VERNON_HAS_METAL_RHI)

#include "rhi/metal_backend.h"
#include "runtime/metal_runtime_capabilities.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

namespace vernon::runtime::rhi_adapter {
namespace {

constexpr uint32_t kMaxGroups = VERNON_RUNTIME_PROVIDER_MAX_GROUPS;
constexpr uint32_t kNone = UINT32_MAX;
// Stage indices of the argument buffers a group holds: compute, vertex and fragment.
constexpr uint32_t kStageCount = 3;
constexpr uint32_t kPushConstantBufferIndex = 15;
constexpr uint32_t kFirstVertexBufferIndex = 16;
constexpr uint32_t kMaxBufferIndex = 31;
// The largest block setBytes accepts.
constexpr uint32_t kMaxDirectBlockSize = 4096;
// Offsets of constant buffers bound from one MTLBuffer must be 256-byte aligned on macOS GPUs.
constexpr uint32_t kInlineAlignment = 256;

struct MetalAdapterState {
    rhi::metal::DeviceState *device{};
    id<MTLSamplerState> defaultSampler;
    std::atomic<uint64_t> nextSerial{};
};

MetalAdapterState &metalState(VernonRuntimeRhiAdapter &adapter) {
    assert(adapter.rhiBackend == VERNON_RHI_BACKEND_METAL);
    assert(adapter.backend.state);
    return *static_cast<MetalAdapterState *>(adapter.backend.state);
}

const MetalAdapterState &metalState(const VernonRuntimeRhiAdapter &adapter) {
    assert(adapter.rhiBackend == VERNON_RHI_BACKEND_METAL);
    assert(adapter.backend.state);
    return *static_cast<const MetalAdapterState *>(adapter.backend.state);
}

rhi::metal::DeviceState &metalDevice(VernonRuntimeRhiAdapter &adapter) { return *metalState(adapter).device; }
const rhi::metal::DeviceState &metalDevice(const VernonRuntimeRhiAdapter &adapter) {
    return *metalState(adapter).device;
}

void destroyBackend(void *state) noexcept { delete static_cast<MetalAdapterState *>(state); }
RhiAdapterResult<void> synchronizeBackend(void *state, std::string &error) noexcept {
    @try {
        try {
            if (static_cast<MetalAdapterState *>(state)->device->synchronize(error))
                return RhiAdapterResult<void>{vernon::ok()};
            return providerFailure(vernon::ProviderErrorCode::BackendFailure, "metal_synchronize");
        } catch (...) {
            return providerFailure(vernon::ProviderErrorCode::BackendFailure, "metal_synchronize");
        }
    } @catch (NSException *) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "metal_synchronize");
    }
}
uint64_t resourceIdentity(const void *state) noexcept {
    return reinterpret_cast<uintptr_t>(static_cast<const MetalAdapterState *>(state)->device);
}
const RhiAdapterBackendOps backendOps{destroyBackend, synchronizeBackend, resourceIdentity};

uint32_t stageIndex(uint32_t stageMask) {
    switch (stageMask) {
    case VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE:
        return 0;
    case VERNON_RUNTIME_PROVIDER_STAGE_VERTEX:
        return 1;
    case VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT:
        return 2;
    default:
        return kNone;
    }
}

constexpr uint32_t kComputeStage = 0;
constexpr uint32_t kVertexStage = 1;
constexpr uint32_t kFragmentStage = 2;

// Graphics inline values bind as bytes at their buffer index; every other entry except a vertex buffer is a member
// of the stage's argument buffer, compute inline values through a buffer the group owns.
bool directValue(const VernonRuntimeProviderBindingLayoutEntry &entry) {
    return entry.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
           entry.stage_mask != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
}

bool argumentMember(const VernonRuntimeProviderBindingLayoutEntry &entry) {
    return entry.kind != VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && !directValue(entry);
}

MTLResourceUsage resourceUsage(const VernonRuntimeProviderBindingLayoutEntry &entry) {
    MTLResourceUsage usage = MTLResourceUsageRead;
    if (writesBinding(entry))
        usage |= MTLResourceUsageWrite;
    return usage;
}

MTLRenderStages renderStages(uint32_t stage) {
    return stage == kVertexStage ? MTLRenderStageVertex : MTLRenderStageFragment;
}

struct PreparedShader {
    id<MTLLibrary> library;
    id<MTLFunction> function;
    uint32_t stage{};
};

struct DirectValue {
    uint32_t entry{};
    uint32_t dataOffset{};
    uint32_t size{};
};

// Retained by the groups and pipeline layouts made from it, since commands that used those may still be recording.
struct GroupLayout : RhiRetainedObject {
    struct Entry {
        VernonRuntimeProviderBindingLayoutEntry layout{};
        uint32_t stage{};
        // Offset of the value's bytes in the group's inline buffer, for argument members that hold packed bytes.
        uint32_t inlineOffset{kNone};
    };
    std::vector<Entry> entries;
    std::vector<DirectValue> directValues;
    uint32_t directDataSize{};
    uint32_t inlineSize{};
    // Bit s is set when stage s has argument members.
    uint32_t argumentStages{};
    bool hasVertexBuffers{};
    // An argument encoder is not thread-safe, so groups encode their argument buffers under the mutex.
    std::mutex mutex;
    std::array<id<MTLArgumentEncoder>, kStageCount> encoders;
};

// One block of graphics inline values at a buffer index of one stage; members follow argument order, which the
// compiled stage's block follows.
struct DirectBlock {
    struct Member {
        uint32_t group{};
        uint32_t dataOffset{};
        uint32_t offset{};
        uint32_t size{};
    };
    uint32_t stage{};
    uint32_t index{};
    // Offset of the block in the command state's scratch bytes.
    uint32_t scratchOffset{};
    uint32_t size{};
    uint32_t groupMask{};
    std::vector<Member> members;
};

struct PreparedLayout : RhiRetainedObject {
    std::array<GroupLayout *, kMaxGroups> groups{};
    uint32_t groupCount{};
    std::vector<DirectBlock> directBlocks;
    uint32_t directScratchSize{};
    uint32_t directGroupMask{};
    uint32_t vertexGroupMask{};
    uint32_t pushConstantSize{};
    std::vector<VernonRuntimeProviderVertexAttribute> vertexAttributes;

    ~PreparedLayout() override {
        for (uint32_t group = 0; group < groupCount; ++group)
            releaseRetainedObject(groups[group]);
    }
};

struct PreparedPipeline : RhiRetainedObject {
    VernonRuntimeRhiAdapter *adapter{};
    PreparedLayout *layout{};
    id<MTLComputePipelineState> compute;
    id<MTLRenderPipelineState> render;
    id<MTLDepthStencilState> depthStencil;
    std::array<MTLPixelFormat, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> colorFormats{};
    MTLPixelFormat depthFormat{MTLPixelFormatInvalid};
    size_t colorFormatCount{};
    uint32_t sampleCount{1};
    uint32_t workgroup[3]{1, 1, 1};
    MTLCullMode cullMode{MTLCullModeNone};
    MTLWinding frontFace{MTLWindingCounterClockwise};
    float depthBiasConstant{};
    float depthBiasSlope{};
    bool depthBiasEnabled{};
    bool graphics{};

    ~PreparedPipeline() override {
        adapter->livePreparedPipelines.fetch_sub(1, std::memory_order_relaxed);
        releaseRetainedObject(layout);
    }
};

struct MetalGroup : RhiGroup {
    // A resource an argument buffer of the group references, which every encoder that binds the group must use.
    struct ResourceUse {
        id<MTLResource> resource;
        MTLResourceUsage usage{};
        uint32_t stage{};
    };
    struct VertexBuffer {
        uint32_t index{};
        id<MTLBuffer> buffer;
        uint64_t offset{};
    };
    // A range of the group's own buffer, or of its encoder's transient memory for a transient group.
    struct Memory {
        id<MTLBuffer> buffer;
        uint64_t offset{};
        uint8_t *mapped{};
    };
    GroupLayout *layout{};
    uint64_t serial{};
    Memory inlineValues;
    std::array<Memory, kStageCount> arguments;
    std::vector<uint8_t> directData;
    std::vector<ResourceUse> uses;
    std::vector<VertexBuffer> vertexBuffers;

    ~MetalGroup() override { releaseRetainedObject(layout); }
};

// Applied state holds group serials rather than pointers, because addresses are reused. Render state belongs to the
// render encoder, which a clear inside a scope replaces, so all of it is reapplied when the encoder changes.
struct MetalCommands : RhiCommands {
    PreparedPipeline *pipeline{};
    std::array<MetalGroup *, kMaxGroups> groups{};
    id<MTLRenderCommandEncoder> renderEncoder;
    const PreparedPipeline *renderPipeline{};
    std::array<uint64_t, kMaxGroups> vertexArguments{};
    std::array<uint64_t, kMaxGroups> fragmentArguments{};
    std::array<uint64_t, kMaxGroups> usedGroups{};
    std::array<uint64_t, kMaxGroups> vertexGroups{};
    // The layout whose blocks `directScratch` follows, and the group serial each index copied in.
    const PreparedLayout *directLayout{};
    std::array<uint64_t, kMaxGroups> directGroups{};
    std::vector<uint8_t> directScratch;
    bool viewportValid{};
    bool scissorValid{};
    bool stencilValid{};
    std::array<uint32_t, 4> viewport{};
    std::array<uint32_t, 4> scissor{};
    uint32_t stencilReference{};
    uint64_t indexBufferKey{};
    id<MTLBuffer> indexBuffer;
};

bool stringEquals(VernonStringView value, const char *expected) {
    const size_t size = std::strlen(expected);
    return value.data && value.size == size && std::memcmp(value.data, expected, size) == 0;
}

uint32_t getCapabilities(void *) {
    return VERNON_RUNTIME_PROVIDER_COMPUTE | VERNON_RUNTIME_PROVIDER_GRAPHICS | VERNON_RUNTIME_PROVIDER_NATIVE_INTEROP;
}

VernonRuntimeProviderDeviceIdentity getDeviceIdentity(void *data) {
    const auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return {0x4d4554414cull, metalDevice(adapter).device.registryID, 0};
}

RhiAdapterResult<void> prepareShaderResult(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        stageIndex(descriptor->stage) == kNone || !stringEquals(descriptor->format, "msl") || !descriptor->data ||
        descriptor->size == 0 || !descriptor->entry.data || descriptor->entry.size == 0)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "metal_adapter_received_an_invalid_shader_descriptor");
    @autoreleasepool {
        NSString *source = [[NSString alloc] initWithBytes:descriptor->data
                                                    length:descriptor->size
                                                  encoding:NSUTF8StringEncoding];
        NSString *entry = [[NSString alloc] initWithBytes:descriptor->entry.data
                                                   length:descriptor->entry.size
                                                 encoding:NSUTF8StringEncoding];
        if (!source || !entry)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "metal_shader_source_or_entry_point_is_not_valid_utf8");
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion2_4;
        NSError *error = nil;
        id<MTLLibrary> library = [metalDevice(adapter).device newLibraryWithSource:source options:options error:&error];
        if (!library)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "metal_compile_shader_library");
        id<MTLFunction> function = [library newFunctionWithName:entry];
        if (!function)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "metal_shader_entry_point_was_not_found");
        auto shader = std::unique_ptr<PreparedShader>(new (std::nothrow) PreparedShader());
        if (!shader)
            return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                                   "metal_shader_preparation_ran_out_of_memory");
        shader->library = library;
        shader->function = function;
        shader->stage = descriptor->stage;
        *output = toHandle(shader.release());
        adapter.shaderPreparations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    }
}

bool supportedEntry(const VernonRuntimeProviderBindingLayoutEntry &entry) {
    const bool supportedKind =
        entry.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER || entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
        entry.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE || entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE ||
        entry.kind == VERNON_RUNTIME_PROVIDER_SAMPLER || entry.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
        entry.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER;
    if (!supportedKind || stageIndex(entry.stage_mask) == kNone || entry.array_count != 1)
        return false;
    if ((packedUniformBytes(entry.kind, entry.interface_kind) || entry.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER) &&
        entry.element_size == 0)
        return false;
    if (entry.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER)
        return entry.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX && entry.binding >= kFirstVertexBufferIndex &&
               entry.binding < kMaxBufferIndex;
    if (directValue(entry))
        return entry.binding <= kPushConstantBufferIndex && entry.element_alignment != 0 &&
               (entry.element_alignment & (entry.element_alignment - 1)) == 0 &&
               entry.element_size <= kMaxDirectBlockSize;
    return entry.binding != kNone;
}

bool argumentTextureType(VernonTextureDimension dimension, MTLTextureType &type) {
    switch (dimension) {
    case VERNON_TEXTURE_2D:
        type = MTLTextureType2D;
        return true;
    case VERNON_TEXTURE_3D:
        type = MTLTextureType3D;
        return true;
    case VERNON_TEXTURE_CUBE:
        type = MTLTextureTypeCube;
        return true;
    case VERNON_TEXTURE_2D_ARRAY:
        type = MTLTextureType2DArray;
        return true;
    case VERNON_TEXTURE_CUBE_ARRAY:
        type = MTLTextureTypeCubeArray;
        return true;
    }
    return false;
}

// Encoders built from the entries match the argument buffer the compiled stage declares at the group index, since
// member ids, types and access come from the same reflection; groups therefore never depend on a pipeline.
RhiAdapterResult<void> prepareArgumentEncoders(VernonRuntimeRhiAdapter &adapter, GroupLayout &layout) {
    if (layout.argumentStages != 0 && !metalDevice(adapter).argumentBufferEncodingSupported)
        return providerFailure(vernon::ProviderErrorCode::Unsupported,
                               "metal_argument_buffer_encoding_is_unavailable_on_this_device");
    @try {
        for (uint32_t stage = 0; stage < kStageCount; ++stage) {
            if ((layout.argumentStages & (1u << stage)) == 0)
                continue;
            NSMutableArray<MTLArgumentDescriptor *> *arguments = [NSMutableArray array];
            for (const GroupLayout::Entry &entry : layout.entries) {
                if (entry.stage != stage || !argumentMember(entry.layout))
                    continue;
                MTLArgumentDescriptor *argument = [MTLArgumentDescriptor argumentDescriptor];
                argument.index = entry.layout.binding;
                argument.arrayLength = 1;
                argument.access = !writesBinding(entry.layout) ? MTLBindingAccessReadOnly
                                  : (entry.layout.access & 1u) ? MTLBindingAccessReadWrite
                                                               : MTLBindingAccessWriteOnly;
                if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER)
                    argument.dataType = MTLDataTypeSampler;
                else if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                         entry.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE) {
                    MTLTextureType type{};
                    if (!argumentTextureType(entry.layout.image_dimension, type))
                        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                               "metal_group_layout_contains_an_unsupported_texture_dimension",
                                               entry.layout.slot);
                    argument.dataType = MTLDataTypeTexture;
                    argument.textureType = type;
                } else
                    argument.dataType = MTLDataTypePointer;
                [arguments addObject:argument];
            }
            [arguments
                sortUsingComparator:^NSComparisonResult(MTLArgumentDescriptor *left, MTLArgumentDescriptor *right) {
                    return left.index < right.index ? NSOrderedAscending : NSOrderedDescending;
                }];
            id<MTLArgumentEncoder> encoder = [metalDevice(adapter).device newArgumentEncoderWithArguments:arguments];
            if (!encoder || encoder.encodedLength == 0)
                return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                                       "metal_argument_encoder_creation_failed", stage);
            layout.encoders[stage] = encoder;
        }
        return RhiAdapterResult<void>{vernon::ok()};
    } @catch (NSException *) {
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "metal_argument_encoder_creation_raised_an_exception");
    }
}

RhiAdapterResult<void> prepareGroupLayoutImplResult(VernonRuntimeRhiAdapter &adapter,
                                                    const VernonRuntimeProviderGroupLayoutDescriptor &descriptor,
                                                    VernonRuntimeProviderObject *output) {
    auto layout = std::make_unique<GroupLayout>();
    layout->entries.reserve(descriptor.entry_count);
    for (size_t index = 0; index < descriptor.entry_count; ++index) {
        const auto &source = descriptor.entries[index];
        if (!supportedEntry(source))
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "metal_group_layout_contains_an_unsupported_binding", index);
        GroupLayout::Entry entry;
        entry.layout = source;
        entry.stage = stageIndex(source.stage_mask);
        const bool duplicate =
            argumentMember(source) &&
            std::any_of(layout->entries.begin(), layout->entries.end(), [&](const GroupLayout::Entry &existing) {
                return argumentMember(existing.layout) && existing.stage == entry.stage &&
                       existing.layout.binding == source.binding;
            });
        if (duplicate)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "metal_group_layout_contains_duplicate_argument_members", index);
        if (directValue(source)) {
            layout->directValues.push_back({static_cast<uint32_t>(index), layout->directDataSize, source.element_size});
            layout->directDataSize += source.element_size;
        } else if (source.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER) {
            layout->hasVertexBuffers = true;
        } else {
            layout->argumentStages |= 1u << entry.stage;
            if (packedUniformBytes(source.kind, source.interface_kind)) {
                layout->inlineSize = (layout->inlineSize + kInlineAlignment - 1) & ~(kInlineAlignment - 1);
                entry.inlineOffset = layout->inlineSize;
                layout->inlineSize += source.element_size;
            }
        }
        layout->entries.push_back(entry);
    }
    if (auto prepared = prepareArgumentEncoders(adapter, *layout); !prepared)
        return prepared;
    *output = toHandle(layout.release());
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> prepareGroupLayoutResult(void *data,
                                                const VernonRuntimeProviderGroupLayoutDescriptor *descriptor,
                                                VernonRuntimeProviderObject *output) {
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        (descriptor->entry_count != 0 && !descriptor->entries))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "metal_adapter_received_an_invalid_group_layout");
    try {
        return prepareGroupLayoutImplResult(*static_cast<VernonRuntimeRhiAdapter *>(data), *descriptor, output);
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                               "metal_group_layout_preparation_ran_out_of_memory");
    }
}

// Places the graphics inline values of every group into one block per stage and buffer index, in argument order.
RhiAdapterResult<void> placeDirectValues(PreparedLayout &layout) {
    struct Candidate {
        uint32_t group{};
        const DirectValue *value{};
        const VernonRuntimeProviderBindingLayoutEntry *entry{};
    };
    std::vector<Candidate> candidates;
    for (uint32_t group = 0; group < layout.groupCount; ++group)
        for (const DirectValue &value : layout.groups[group]->directValues)
            candidates.push_back({group, &value, &layout.groups[group]->entries[value.entry].layout});
    std::sort(candidates.begin(), candidates.end(), [](const Candidate &left, const Candidate &right) {
        if (left.entry->stage_mask != right.entry->stage_mask)
            return left.entry->stage_mask < right.entry->stage_mask;
        if (left.entry->binding != right.entry->binding)
            return left.entry->binding < right.entry->binding;
        return left.entry->argument_index < right.entry->argument_index;
    });
    for (const Candidate &candidate : candidates) {
        const uint32_t stage = stageIndex(candidate.entry->stage_mask);
        if (layout.directBlocks.empty() || layout.directBlocks.back().stage != stage ||
            layout.directBlocks.back().index != candidate.entry->binding)
            layout.directBlocks.push_back({stage, candidate.entry->binding, 0, 0, 0, {}});
        DirectBlock &block = layout.directBlocks.back();
        const uint32_t alignment = candidate.entry->element_alignment;
        const uint32_t offset = (block.size + alignment - 1) & ~(alignment - 1);
        if (offset > kMaxDirectBlockSize || candidate.value->size > kMaxDirectBlockSize - offset)
            return providerFailure(vernon::ProviderErrorCode::Unsupported,
                                   "metal_inline_values_exceed_the_set_bytes_limit", candidate.entry->binding);
        block.members.push_back({candidate.group, candidate.value->dataOffset, offset, candidate.value->size});
        block.size = offset + candidate.value->size;
        block.groupMask |= 1u << candidate.group;
        layout.directGroupMask |= 1u << candidate.group;
    }
    for (DirectBlock &block : layout.directBlocks) {
        block.scratchOffset = layout.directScratchSize;
        layout.directScratchSize += (block.size + 15u) & ~15u;
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> validateArgumentBufferUsage(const VernonRuntimeRhiAdapter &adapter,
                                                   const PreparedLayout &layout) {
    uint64_t buffers = 0;
    uint64_t textures = 0;
    uint64_t samplers = 0;
    bool writableTexture = false;
    for (uint32_t group = 0; group < layout.groupCount; ++group)
        for (const GroupLayout::Entry &entry : layout.groups[group]->entries) {
            if (!argumentMember(entry.layout))
                continue;
            if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER)
                ++samplers;
            else if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                     entry.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE) {
                ++textures;
                writableTexture |=
                    entry.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE && writesBinding(entry.layout);
            } else
                ++buffers;
        }
    if (metalDevice(adapter).device.argumentBuffersSupport < MTLArgumentBuffersTier2 &&
        (buffers > 31 || textures > 31 || samplers > 16 || writableTexture))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "metal_layout_requires_argument_buffers_tier_2");
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> prepareLayoutImplResult(VernonRuntimeRhiAdapter &adapter,
                                               const VernonRuntimeProviderPipelineLayoutDescriptor &descriptor,
                                               VernonRuntimeProviderObject *output) {
    auto layout = std::make_unique<PreparedLayout>();
    for (size_t group = 0; group < descriptor.group_layout_count; ++group) {
        auto *groupLayout = fromHandle<GroupLayout>(descriptor.group_layouts[group]);
        if (!groupLayout)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "metal_pipeline_layout_group_layout_is_invalid", group);
        groupLayout->references.fetch_add(1, std::memory_order_relaxed);
        layout->groups[group] = groupLayout;
        layout->groupCount = static_cast<uint32_t>(group + 1);
        if (groupLayout->hasVertexBuffers)
            layout->vertexGroupMask |= 1u << group;
    }
    if (auto placed = placeDirectValues(*layout); !placed)
        return placed;
    if (descriptor.push_constant_size != 0 &&
        std::any_of(layout->directBlocks.begin(), layout->directBlocks.end(),
                    [](const DirectBlock &block) { return block.index == kPushConstantBufferIndex; }))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "metal_pipeline_layout_mixes_push_constants_and_inline_values");
    if (descriptor.push_constant_size > kMaxDirectBlockSize)
        return providerFailure(vernon::ProviderErrorCode::Unsupported,
                               "metal_push_constants_exceed_the_set_bytes_limit");
    layout->pushConstantSize = descriptor.push_constant_size;
    if (auto valid = validateArgumentBufferUsage(adapter, *layout); !valid)
        return valid;
    for (size_t index = 0; index < descriptor.vertex_attribute_count; ++index) {
        const auto &attribute = descriptor.vertex_attributes[index];
        bool bindingExists = false;
        for (uint32_t group = 0; group < layout->groupCount; ++group)
            for (const GroupLayout::Entry &entry : layout->groups[group]->entries)
                bindingExists |= entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER &&
                                 entry.layout.binding == attribute.binding;
        if (!bindingExists || attribute.location >= 31 || attribute.component_count == 0 ||
            attribute.component_count > 4)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "metal_vertex_attribute_is_invalid_or_references_an_unknown_binding",
                                   attribute.binding, attribute.location);
        layout->vertexAttributes.push_back(attribute);
    }
    *output = toHandle(layout.release());
    adapter.layoutPreparations.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> prepareLayoutResult(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                                           VernonRuntimeProviderObject *output) {
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        descriptor->group_layout_count > kMaxGroups ||
        (descriptor->group_layout_count != 0 && !descriptor->group_layouts) ||
        (descriptor->vertex_attribute_count != 0 && !descriptor->vertex_attributes))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "metal_adapter_received_an_invalid_pipeline_layout");
    try {
        return prepareLayoutImplResult(*static_cast<VernonRuntimeRhiAdapter *>(data), *descriptor, output);
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "metal_layout_preparation_ran_out_of_memory");
    }
}

MTLVertexFormat vertexFormat(uint32_t dtype, uint32_t components) {
    if (components == 0 || components > 4)
        return MTLVertexFormatInvalid;
    if (dtype == VERNON_RUNTIME_PROVIDER_I32) {
        constexpr MTLVertexFormat formats[]{MTLVertexFormatInt, MTLVertexFormatInt2, MTLVertexFormatInt3,
                                            MTLVertexFormatInt4};
        return formats[components - 1];
    }
    if (dtype == VERNON_RUNTIME_PROVIDER_U32) {
        constexpr MTLVertexFormat formats[]{MTLVertexFormatUInt, MTLVertexFormatUInt2, MTLVertexFormatUInt3,
                                            MTLVertexFormatUInt4};
        return formats[components - 1];
    }
    if (dtype == VERNON_RUNTIME_PROVIDER_F16) {
        constexpr MTLVertexFormat formats[]{MTLVertexFormatHalf, MTLVertexFormatHalf2, MTLVertexFormatHalf3,
                                            MTLVertexFormatHalf4};
        return formats[components - 1];
    }
    if (dtype == VERNON_RUNTIME_PROVIDER_F32) {
        constexpr MTLVertexFormat formats[]{MTLVertexFormatFloat, MTLVertexFormatFloat2, MTLVertexFormatFloat3,
                                            MTLVertexFormatFloat4};
        return formats[components - 1];
    }
    return MTLVertexFormatInvalid;
}

bool compareFunction(uint32_t operation, MTLCompareFunction &output) {
    constexpr MTLCompareFunction functions[]{MTLCompareFunctionNever,        MTLCompareFunctionLess,
                                             MTLCompareFunctionEqual,        MTLCompareFunctionLessEqual,
                                             MTLCompareFunctionGreater,      MTLCompareFunctionNotEqual,
                                             MTLCompareFunctionGreaterEqual, MTLCompareFunctionAlways};
    if (operation >= std::size(functions))
        return false;
    output = functions[operation];
    return true;
}

bool stencilOperation(uint32_t operation, MTLStencilOperation &output) {
    constexpr MTLStencilOperation operations[]{MTLStencilOperationKeep,           MTLStencilOperationZero,
                                               MTLStencilOperationReplace,        MTLStencilOperationIncrementClamp,
                                               MTLStencilOperationDecrementClamp, MTLStencilOperationInvert,
                                               MTLStencilOperationIncrementWrap,  MTLStencilOperationDecrementWrap};
    if (operation >= std::size(operations))
        return false;
    output = operations[operation];
    return true;
}

bool blendFactor(uint32_t factor, MTLBlendFactor &output) {
    constexpr MTLBlendFactor factors[]{MTLBlendFactorZero,
                                       MTLBlendFactorOne,
                                       MTLBlendFactorSourceColor,
                                       MTLBlendFactorOneMinusSourceColor,
                                       MTLBlendFactorDestinationColor,
                                       MTLBlendFactorOneMinusDestinationColor,
                                       MTLBlendFactorSourceAlpha,
                                       MTLBlendFactorOneMinusSourceAlpha,
                                       MTLBlendFactorDestinationAlpha,
                                       MTLBlendFactorOneMinusDestinationAlpha};
    if (factor >= std::size(factors))
        return false;
    output = factors[factor];
    return true;
}

bool blendOperation(uint32_t operation, MTLBlendOperation &output) {
    constexpr MTLBlendOperation operations[]{MTLBlendOperationAdd, MTLBlendOperationSubtract,
                                             MTLBlendOperationReverseSubtract, MTLBlendOperationMin,
                                             MTLBlendOperationMax};
    if (operation >= std::size(operations))
        return false;
    output = operations[operation];
    return true;
}

MTLColorWriteMask colorWriteMask(uint32_t mask) {
    MTLColorWriteMask result = MTLColorWriteMaskNone;
    if (mask & VERNON_RHI_COLOR_WRITE_RED)
        result |= MTLColorWriteMaskRed;
    if (mask & VERNON_RHI_COLOR_WRITE_GREEN)
        result |= MTLColorWriteMaskGreen;
    if (mask & VERNON_RHI_COLOR_WRITE_BLUE)
        result |= MTLColorWriteMaskBlue;
    if (mask & VERNON_RHI_COLOR_WRITE_ALPHA)
        result |= MTLColorWriteMaskAlpha;
    return result;
}

bool configureStencilFace(const VernonStencilFaceState &source, MTLStencilDescriptor *destination) {
    MTLCompareFunction compare{};
    MTLStencilOperation stencilFail{}, depthFail{}, pass{};
    if (!compareFunction(source.compare, compare) || !stencilOperation(source.stencil_fail, stencilFail) ||
        !stencilOperation(source.depth_fail, depthFail) || !stencilOperation(source.pass, pass))
        return false;
    destination.stencilCompareFunction = compare;
    destination.stencilFailureOperation = stencilFail;
    destination.depthFailureOperation = depthFail;
    destination.depthStencilPassOperation = pass;
    return true;
}

RhiAdapterResult<void> prepareComputePipeline(VernonRuntimeRhiAdapter &adapter,
                                              const VernonRuntimeProviderPipelineDescriptor &descriptor,
                                              PreparedPipeline &pipeline) {
    PreparedShader *shader = descriptor.shader_count == 1 ? fromHandle<PreparedShader>(descriptor.shaders[0]) : nullptr;
    if (!shader || shader->stage != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE || descriptor.workgroup_size[0] == 0 ||
        descriptor.workgroup_size[1] == 0 || descriptor.workgroup_size[2] == 0)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "metal_adapter_received_an_invalid_compute_pipeline");
    NSError *error = nil;
    pipeline.compute = [metalDevice(adapter).device newComputePipelineStateWithFunction:shader->function error:&error];
    if (!pipeline.compute)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "metal_create_compute_pipeline");
    const uint64_t total = static_cast<uint64_t>(descriptor.workgroup_size[0]) * descriptor.workgroup_size[1] *
                           descriptor.workgroup_size[2];
    if (!total || total > pipeline.compute.maxTotalThreadsPerThreadgroup)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "metal_compute_workgroup_exceeds_pipeline_limits");
    std::copy_n(descriptor.workgroup_size, 3, pipeline.workgroup);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> configureColorAttachment(const VernonColorBlendState &blend, MTLPixelFormat format,
                                                MTLRenderPipelineColorAttachmentDescriptor *attachment) {
    MTLBlendFactor sourceColor{}, destinationColor{}, sourceAlpha{}, destinationAlpha{};
    MTLBlendOperation colorOperation{}, alphaOperation{};
    if (format == MTLPixelFormatInvalid || blend.blend_enabled > 1 ||
        (blend.write_mask & ~VERNON_RHI_COLOR_WRITE_ALL) || !blendFactor(blend.source_color_factor, sourceColor) ||
        !blendFactor(blend.destination_color_factor, destinationColor) ||
        !blendFactor(blend.source_alpha_factor, sourceAlpha) ||
        !blendFactor(blend.destination_alpha_factor, destinationAlpha) ||
        !blendOperation(blend.color_operation, colorOperation) ||
        !blendOperation(blend.alpha_operation, alphaOperation))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "metal_graphics_pipeline_contains_an_invalid_color_attachment");
    attachment.pixelFormat = format;
    attachment.blendingEnabled = blend.blend_enabled;
    attachment.sourceRGBBlendFactor = sourceColor;
    attachment.destinationRGBBlendFactor = destinationColor;
    attachment.rgbBlendOperation = colorOperation;
    attachment.sourceAlphaBlendFactor = sourceAlpha;
    attachment.destinationAlphaBlendFactor = destinationAlpha;
    attachment.alphaBlendOperation = alphaOperation;
    attachment.writeMask = colorWriteMask(blend.write_mask);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> prepareDepthStencilState(VernonRuntimeRhiAdapter &adapter,
                                                const VernonRuntimeProviderPipelineDescriptor &descriptor,
                                                PreparedPipeline &pipeline) {
    MTLDepthStencilDescriptor *depthDescriptor = [[MTLDepthStencilDescriptor alloc] init];
    const auto &state = descriptor.depth_stencil;
    MTLCompareFunction depthCompare{};
    if (state.depth_test > 1 || state.depth_write > 1 || state.stencil_test > 1 ||
        !compareFunction(state.depth_test ? state.depth_compare : VERNON_RHI_COMPARE_ALWAYS, depthCompare) ||
        (state.stencil_test && descriptor.depth_stencil_format != MTLPixelFormatDepth32Float_Stencil8))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "metal_graphics_pipeline_contains_an_invalid_depth_stencil_state");
    depthDescriptor.depthCompareFunction = depthCompare;
    depthDescriptor.depthWriteEnabled = state.depth_write;
    if (state.stencil_test) {
        MTLStencilDescriptor *front = [[MTLStencilDescriptor alloc] init];
        MTLStencilDescriptor *back = [[MTLStencilDescriptor alloc] init];
        if (!configureStencilFace(state.front, front) || !configureStencilFace(state.back, back) ||
            state.stencil_read_mask > UINT8_MAX || state.stencil_write_mask > UINT8_MAX)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "metal_graphics_pipeline_contains_an_unsupported_stencil_state");
        front.readMask = state.stencil_read_mask;
        front.writeMask = state.stencil_write_mask;
        back.readMask = state.stencil_read_mask;
        back.writeMask = state.stencil_write_mask;
        depthDescriptor.frontFaceStencil = front;
        depthDescriptor.backFaceStencil = back;
    }
    pipeline.depthStencil = [metalDevice(adapter).device newDepthStencilStateWithDescriptor:depthDescriptor];
    if (!pipeline.depthStencil)
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "metal_depth_state_creation_failed");
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> prepareGraphicsPipeline(VernonRuntimeRhiAdapter &adapter,
                                               const VernonRuntimeProviderPipelineDescriptor &descriptor,
                                               PreparedPipeline &pipeline) {
    std::array<PreparedShader *, kStageCount> shaders{};
    for (size_t index = 0; index < descriptor.shader_count; ++index) {
        auto *shader = fromHandle<PreparedShader>(descriptor.shaders[index]);
        if (shader && shader->stage != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE)
            shaders[stageIndex(shader->stage)] = shader;
    }
    PreparedShader *vertex = shaders[kVertexStage];
    PreparedShader *fragment = shaders[kFragmentStage];
    if (!vertex || !fragment)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "metal_graphics_pipeline_is_missing_a_vertex_or_fragment_shader");
    pipeline.graphics = true;
    if (descriptor.color_format_count == 0 && descriptor.depth_stencil_format == 0)
        return RhiAdapterResult<void>{vernon::ok()};
    if (descriptor.color_format_count > VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS ||
        (descriptor.color_format_count && !descriptor.color_formats) || descriptor.sample_count != 1 ||
        (descriptor.depth_stencil_format && descriptor.depth_stencil_format != MTLPixelFormatDepth32Float &&
         descriptor.depth_stencil_format != MTLPixelFormatDepth32Float_Stencil8) ||
        descriptor.rasterization.cull_mode > VERNON_RHI_CULL_BACK ||
        descriptor.rasterization.front_face > VERNON_RHI_FRONT_FACE_CLOCKWISE || descriptor.rasterization.depth_clamp ||
        descriptor.rasterization.depth_bias_enabled > 1 ||
        !std::isfinite(descriptor.rasterization.depth_bias_constant) ||
        !std::isfinite(descriptor.rasterization.depth_bias_slope) ||
        descriptor.color_blend_count != descriptor.color_format_count ||
        (descriptor.color_blend_count && !descriptor.color_blends))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "metal_graphics_pipeline_uses_an_unsupported_attachment_configuration");
    const PreparedLayout &layout = *pipeline.layout;
    MTLRenderPipelineDescriptor *nativeDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    nativeDescriptor.vertexFunction = vertex->function;
    nativeDescriptor.fragmentFunction = fragment->function;
    nativeDescriptor.rasterSampleCount = std::max(1u, descriptor.sample_count);
    if (!layout.vertexAttributes.empty()) {
        if (!descriptor.vertex_strides || descriptor.vertex_stride_count == 0)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "metal_graphics_pipeline_has_no_vertex_strides");
        MTLVertexDescriptor *vertexDescriptor = [MTLVertexDescriptor vertexDescriptor];
        for (uint32_t group = 0; group < layout.groupCount; ++group)
            for (const GroupLayout::Entry &entry : layout.groups[group]->entries) {
                if (entry.layout.kind != VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER)
                    continue;
                const uint32_t binding = entry.layout.binding;
                if (binding >= descriptor.vertex_stride_count || descriptor.vertex_strides[binding] == 0)
                    return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                           "metal_graphics_pipeline_has_an_invalid_vertex_stride", binding);
                vertexDescriptor.layouts[binding].stride = descriptor.vertex_strides[binding];
                vertexDescriptor.layouts[binding].stepFunction =
                    entry.layout.divisor ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
                vertexDescriptor.layouts[binding].stepRate = entry.layout.divisor ? entry.layout.divisor : 1;
            }
        for (const auto &attribute : layout.vertexAttributes) {
            const MTLVertexFormat format = vertexFormat(attribute.dtype, attribute.component_count);
            if (format == MTLVertexFormatInvalid)
                return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                       "metal_graphics_pipeline_has_an_unsupported_vertex_format", attribute.location);
            vertexDescriptor.attributes[attribute.location].format = format;
            vertexDescriptor.attributes[attribute.location].offset = attribute.relative_offset;
            vertexDescriptor.attributes[attribute.location].bufferIndex = attribute.binding;
        }
        nativeDescriptor.vertexDescriptor = vertexDescriptor;
    }
    for (size_t index = 0; index < descriptor.color_format_count; ++index)
        if (auto configured = configureColorAttachment(descriptor.color_blends[index],
                                                       static_cast<MTLPixelFormat>(descriptor.color_formats[index]),
                                                       nativeDescriptor.colorAttachments[index]);
            !configured)
            return configured;
    nativeDescriptor.depthAttachmentPixelFormat = static_cast<MTLPixelFormat>(descriptor.depth_stencil_format);
    if (descriptor.depth_stencil_format == MTLPixelFormatDepth32Float_Stencil8)
        nativeDescriptor.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    NSError *error = nil;
    pipeline.render = [metalDevice(adapter).device newRenderPipelineStateWithDescriptor:nativeDescriptor error:&error];
    if (!pipeline.render)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "metal_create_render_pipeline");
    if (descriptor.depth_stencil_format)
        if (auto prepared = prepareDepthStencilState(adapter, descriptor, pipeline); !prepared)
            return prepared;
    pipeline.colorFormatCount = descriptor.color_format_count;
    for (size_t index = 0; index < descriptor.color_format_count; ++index)
        pipeline.colorFormats[index] = static_cast<MTLPixelFormat>(descriptor.color_formats[index]);
    pipeline.depthFormat = static_cast<MTLPixelFormat>(descriptor.depth_stencil_format);
    pipeline.sampleCount = std::max(1u, descriptor.sample_count);
    pipeline.cullMode = static_cast<MTLCullMode>(descriptor.rasterization.cull_mode);
    pipeline.frontFace = descriptor.rasterization.front_face == VERNON_RHI_FRONT_FACE_CLOCKWISE
                             ? MTLWindingClockwise
                             : MTLWindingCounterClockwise;
    pipeline.depthBiasEnabled = descriptor.rasterization.depth_bias_enabled != 0;
    pipeline.depthBiasConstant = descriptor.rasterization.depth_bias_constant;
    pipeline.depthBiasSlope = descriptor.rasterization.depth_bias_slope;
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> preparePipelineResult(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                                             VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !layout || !descriptor->shaders ||
        descriptor->shader_count == 0 ||
        (descriptor->kind != VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE &&
         descriptor->kind != VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "metal_adapter_received_an_invalid_pipeline");
    @autoreleasepool {
        auto pipeline = std::unique_ptr<PreparedPipeline>(new (std::nothrow) PreparedPipeline());
        if (!pipeline)
            return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                                   "metal_pipeline_preparation_ran_out_of_memory");
        pipeline->adapter = &adapter;
        adapter.livePreparedPipelines.fetch_add(1, std::memory_order_relaxed);
        layout->references.fetch_add(1, std::memory_order_relaxed);
        pipeline->layout = layout;
        auto prepared = descriptor->kind == VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE
                            ? prepareComputePipeline(adapter, *descriptor, *pipeline)
                            : prepareGraphicsPipeline(adapter, *descriptor, *pipeline);
        if (!prepared)
            return prepared;
        *output = toHandle(pipeline.release());
        adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    }
}

RhiAdapterResult<void> validateGroupValue(const GroupLayout::Entry &entry,
                                          const VernonRuntimeProviderBindingValue &value) {
    const auto &layout = entry.layout;
    if (value.kind != layout.kind)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "metal_group_value_kind_is_invalid",
                               layout.slot);
    if (packedUniformBytes(layout.kind, layout.interface_kind)) {
        if (!value.payload.inline_value.data || value.payload.inline_value.size != layout.element_size)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "metal_inline_binding_has_an_invalid_physical_size", layout.slot);
        return RhiAdapterResult<void>{vernon::ok()};
    }
    if ((value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0)
        return layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER && !value.payload.sampler.resource.resource.value
                   ? RhiAdapterResult<void>{vernon::ok()}
                   : providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                     "metal_default_resource_binding_is_invalid", layout.slot);
    const auto *resource = providerBindingResource(value);
    const bool image =
        layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE || layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
    const bool sampler = layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER;
    const uint64_t expectedKind = sampler ? kRhiSamplerResource : kRhiBufferResource;
    if (!resource || !resource->resource.value ||
        (image ? !isRhiImageReference(*resource) : (resource->identity & kRhiResourceKindMask) != expectedKind) ||
        (!image && !sampler && resource->offset >= resource->size))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "metal_resource_binding_is_invalid",
                               layout.slot);
    if (layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && value.payload.buffer.stride == 0)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "metal_vertex_binding_has_no_stride",
                               layout.slot);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<MetalGroup::Memory> placeGroupMemory(VernonRuntimeRhiAdapter &adapter,
                                                      const VernonRuntimeProviderGroupDescriptor &descriptor,
                                                      uint64_t size, uint64_t alignment) {
    using Placed = RhiAdapterResult<MetalGroup::Memory>;
    if (descriptor.lifetime == VERNON_RUNTIME_PROVIDER_GROUP_TRANSIENT) {
        auto reserved = reserveCommandTransient(adapter, descriptor.command_encoder, size, alignment);
        if (!reserved)
            return Placed{vernon::err(std::move(reserved).error())};
        const vernon::rhi::CommandTransientReservation &reservation = reserved.value();
        return Placed{vernon::ok(MetalGroup::Memory{
            (__bridge id<MTLBuffer>)(reinterpret_cast<void *>(static_cast<uintptr_t>(reservation.transient.buffer))),
            reservation.transient.offset, static_cast<uint8_t *>(reservation.mapped)})};
    }
    id<MTLBuffer> buffer = [metalDevice(adapter).device newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (!buffer)
        return Placed{vernon::err(
            providerFailure(vernon::ProviderErrorCode::BackendFailure, "metal_group_buffer_allocation_failed", size)
                .error())};
    return Placed{vernon::ok(MetalGroup::Memory{buffer, 0, static_cast<uint8_t *>(buffer.contents)})};
}

// Encodes the argument buffers of the group; the caller holds the group layout's mutex.
RhiAdapterResult<void> encodeArguments(VernonRuntimeRhiAdapter &adapter, MetalGroup &group,
                                       const VernonRuntimeProviderGroupDescriptor &descriptor,
                                       const std::vector<uint64_t> &natives) {
    const GroupLayout &layout = *group.layout;
    for (uint32_t stage = 0; stage < kStageCount; ++stage) {
        if ((layout.argumentStages & (1u << stage)) == 0)
            continue;
        id<MTLArgumentEncoder> encoder = layout.encoders[stage];
        auto placed =
            placeGroupMemory(adapter, descriptor, encoder.encodedLength, std::max<uint64_t>(encoder.alignment, 1));
        if (!placed)
            return RhiAdapterResult<void>{vernon::err(std::move(placed).error())};
        const MetalGroup::Memory &memory = placed.value();
        [encoder setArgumentBuffer:memory.buffer offset:memory.offset];
        for (size_t index = 0; index < layout.entries.size(); ++index) {
            const GroupLayout::Entry &entry = layout.entries[index];
            if (entry.stage != stage || !argumentMember(entry.layout))
                continue;
            const uint32_t member = entry.layout.binding;
            void *native = reinterpret_cast<void *>(static_cast<uintptr_t>(natives[index]));
            if (entry.inlineOffset != kNone)
                [encoder setBuffer:group.inlineValues.buffer
                            offset:group.inlineValues.offset + entry.inlineOffset
                           atIndex:member];
            else if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER)
                [encoder
                    setSamplerState:native ? (__bridge id<MTLSamplerState>)native : metalState(adapter).defaultSampler
                            atIndex:member];
            else if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                     entry.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE)
                [encoder setTexture:(__bridge id<MTLTexture>)native atIndex:member];
            else
                [encoder setBuffer:(__bridge id<MTLBuffer>)native
                            offset:providerBindingResource(descriptor.values[index])->offset
                           atIndex:member];
        }
        group.arguments[stage] = memory;
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> createGroupImplResult(VernonRuntimeRhiAdapter &adapter,
                                             const VernonRuntimeProviderGroupDescriptor &descriptor,
                                             VernonRuntimeProviderObject *output) {
    auto *layout = fromHandle<GroupLayout>(descriptor.group_layout);
    if (!layout || descriptor.value_count != layout->entries.size() ||
        (descriptor.value_count != 0 && !descriptor.values))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "metal_adapter_received_an_invalid_group_descriptor");
    const VernonRuntimeProviderBindingValue *values = descriptor.values;
    for (size_t index = 0; index < layout->entries.size(); ++index)
        if (auto valid = validateGroupValue(layout->entries[index], values[index]); !valid)
            return valid;
    auto group = std::make_unique<MetalGroup>();
    layout->references.fetch_add(1, std::memory_order_relaxed);
    group->layout = layout;
    if (layout->inlineSize != 0) {
        auto placed = placeGroupMemory(adapter, descriptor, layout->inlineSize, kInlineAlignment);
        if (!placed)
            return RhiAdapterResult<void>{vernon::err(std::move(placed).error())};
        group->inlineValues = placed.value();
    }
    group->directData.resize(layout->directDataSize);
    for (const DirectValue &value : layout->directValues)
        std::memcpy(group->directData.data() + value.dataOffset, values[value.entry].payload.inline_value.data,
                    value.size);
    std::vector<uint64_t> natives(layout->entries.size());
    for (size_t index = 0; index < layout->entries.size(); ++index) {
        const GroupLayout::Entry &entry = layout->entries[index];
        if (directValue(entry.layout))
            continue;
        if (entry.inlineOffset != kNone) {
            std::memcpy(group->inlineValues.mapped + entry.inlineOffset, values[index].payload.inline_value.data,
                        values[index].payload.inline_value.size);
            group->uses.push_back({group->inlineValues.buffer, MTLResourceUsageRead, entry.stage});
            continue;
        }
        if ((values[index].flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0)
            continue;
        const VernonRuntimeProviderResourceReference &resource = *providerBindingResource(values[index]);
        auto acquired = acquireGroupResource(adapter, *group, resource);
        if (!acquired)
            return RhiAdapterResult<void>{vernon::err(std::move(acquired).error())};
        natives[index] = acquired.value();
        if (!natives[index])
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "metal_resource_binding_is_stale",
                                   entry.layout.slot);
        void *native = reinterpret_cast<void *>(static_cast<uintptr_t>(natives[index]));
        if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER) {
            group->vertexBuffers.push_back({entry.layout.binding, (__bridge id<MTLBuffer>)native, resource.offset});
            continue;
        }
        if (entry.layout.kind != VERNON_RUNTIME_PROVIDER_SAMPLER)
            group->uses.push_back({(__bridge id<MTLResource>)native, resourceUsage(entry.layout), entry.stage});
        if (writesBinding(entry.layout))
            group->writes.push_back(resource);
    }
    @try {
        std::lock_guard<std::mutex> guard(layout->mutex);
        if (auto encoded = encodeArguments(adapter, *group, descriptor, natives); !encoded)
            return encoded;
    } @catch (NSException *) {
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "metal_argument_buffer_resource_encoding_raised_an_exception");
    }
    group->serial = metalState(adapter).nextSerial.fetch_add(1, std::memory_order_relaxed) + 1;
    *output = toHandle(group.release());
    adapter.groupCreations.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> createGroupResult(void *data, const VernonRuntimeProviderGroupDescriptor *descriptor,
                                         VernonRuntimeProviderObject *output) {
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "metal_adapter_received_an_invalid_group_descriptor");
    try {
        return createGroupImplResult(*static_cast<VernonRuntimeRhiAdapter *>(data), *descriptor, output);
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "metal_group_creation_ran_out_of_memory");
    }
}

RhiAdapterResult<void> beginCommandsResult(void *data, VernonRuntimeProviderObject encoder,
                                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!output)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "metal_begin_commands_output_is_null");
    std::unique_ptr<MetalCommands> commands(new (std::nothrow) MetalCommands());
    if (!commands)
        return providerFailure(vernon::ProviderErrorCode::ResourceExhausted, "metal_commands_allocation");
    if (auto begun = beginRhiCommands(adapter, encoder, *commands); !begun)
        return begun;
    if (!commands->native)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "metal_command_encoder_is_invalid");
    *output = toHandle(commands.release());
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> endCommandsResult(void *, VernonRuntimeProviderObject handle) {
    std::unique_ptr<MetalCommands> commands(fromHandle<MetalCommands>(handle));
    return commands
               ? RhiAdapterResult<void>{vernon::ok()}
               : providerFailure(vernon::ProviderErrorCode::InvalidArgument, "metal_end_commands_handle_is_invalid");
}

RhiAdapterResult<void> bindPipelineResult(void *data, VernonRuntimeProviderObject handle,
                                          VernonRuntimeProviderObject pipelineHandle) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<MetalCommands>(handle);
    auto *pipeline = fromHandle<PreparedPipeline>(pipelineHandle);
    if (!commands || !pipeline || (!pipeline->compute && !pipeline->render))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "metal_bind_pipeline_is_invalid");
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
    auto *commands = fromHandle<MetalCommands>(handle);
    auto *group = fromHandle<MetalGroup>(groupHandle);
    if (!commands || !group || !commands->pipeline || index >= commands->pipeline->layout->groupCount ||
        group->layout != commands->pipeline->layout->groups[index])
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "metal_bind_group_is_invalid", index);
    if (auto retained = retainForCommands(adapter, commands->encoder, *group); !retained)
        return retained;
    commands->groups[index] = group;
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> requireBoundGroups(const MetalCommands &commands) {
    for (uint32_t index = 0; index < commands.pipeline->layout->groupCount; ++index)
        if (!commands.groups[index])
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "metal_command_group_is_unbound", index);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> recordBoundGroupWrites(VernonRuntimeRhiAdapter &adapter, const MetalCommands &commands) {
    for (uint32_t index = 0; index < commands.pipeline->layout->groupCount; ++index)
        if (!commands.groups[index]->writes.empty())
            if (auto recorded = recordGroupWrites(adapter, commands.encoder, *commands.groups[index]); !recorded)
                return recorded;
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> encodeDispatchResult(void *data, VernonRuntimeProviderObject handle,
                                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<MetalCommands>(handle);
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !commands || !commands->pipeline ||
        !commands->pipeline->compute || commands->rendering || descriptor->group_count[0] == 0 ||
        descriptor->group_count[1] == 0 || descriptor->group_count[2] == 0 ||
        descriptor->push_constant_size != commands->pipeline->layout->pushConstantSize ||
        (descriptor->push_constant_size != 0 && !descriptor->push_constants))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "metal_adapter_received_an_invalid_dispatch");
    if (auto bound = requireBoundGroups(*commands); !bound)
        return bound;
    const PreparedPipeline &pipeline = *commands->pipeline;
    // Copies record their own encoders on the command buffer between commands, so each dispatch has its encoder.
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)(reinterpret_cast<void *>(static_cast<uintptr_t>(commands->native)));
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder)
        return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                               "metal_compute_command_encoder_creation_failed");
    [encoder setComputePipelineState:pipeline.compute];
    for (uint32_t index = 0; index < pipeline.layout->groupCount; ++index) {
        const MetalGroup &group = *commands->groups[index];
        if (const MetalGroup::Memory &arguments = group.arguments[kComputeStage]; arguments.buffer)
            [encoder setBuffer:arguments.buffer offset:arguments.offset atIndex:index];
        for (const MetalGroup::ResourceUse &use : group.uses)
            [encoder useResource:use.resource usage:use.usage];
    }
    if (descriptor->push_constant_size != 0)
        [encoder setBytes:descriptor->push_constants
                   length:descriptor->push_constant_size
                  atIndex:kPushConstantBufferIndex];
    [encoder dispatchThreadgroups:MTLSizeMake(descriptor->group_count[0], descriptor->group_count[1],
                                              descriptor->group_count[2])
            threadsPerThreadgroup:MTLSizeMake(pipeline.workgroup[0], pipeline.workgroup[1], pipeline.workgroup[2])];
    [encoder endEncoding];
    if (auto recorded = recordBoundGroupWrites(adapter, *commands); !recorded)
        return recorded;
    if (auto recorded = recordProviderCommand(adapter, commands->encoder, false); !recorded)
        return recorded;
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

bool primitiveType(uint32_t topology, MTLPrimitiveType &output) {
    switch (topology) {
    case 0:
        output = MTLPrimitiveTypeTriangle;
        return true;
    case 1:
        output = MTLPrimitiveTypeLine;
        return true;
    case 2:
        output = MTLPrimitiveTypePoint;
        return true;
    default:
        return false;
    }
}

void resetRenderState(MetalCommands &commands, id<MTLRenderCommandEncoder> encoder) {
    commands.renderEncoder = encoder;
    commands.renderPipeline = nullptr;
    commands.vertexArguments.fill(0);
    commands.fragmentArguments.fill(0);
    commands.usedGroups.fill(0);
    commands.vertexGroups.fill(0);
    commands.directLayout = nullptr;
    commands.viewportValid = false;
    commands.scissorValid = false;
    commands.stencilValid = false;
}

// Metal requires the pipeline's attachment formats to match the render pass at every location.
RhiAdapterResult<void> checkAttachments(const PreparedPipeline &pipeline, const rhi::metal::RenderingState &rendering) {
    for (uint32_t location = 0; location < VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS; ++location) {
        id<MTLTexture> texture = rendering.colorTextures[location];
        const MTLPixelFormat format =
            location < pipeline.colorFormatCount ? pipeline.colorFormats[location] : MTLPixelFormatInvalid;
        if ((texture == nil) != (format == MTLPixelFormatInvalid) ||
            (texture && (texture.pixelFormat != format || texture.sampleCount != pipeline.sampleCount)))
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "metal_draw_contains_an_incompatible_color_attachment", location);
    }
    id<MTLTexture> depthTexture = rendering.depthStencilTexture;
    if ((depthTexture == nil) != (pipeline.depthStencil == nil) ||
        (depthTexture &&
         (depthTexture.pixelFormat != pipeline.depthFormat || depthTexture.sampleCount != pipeline.sampleCount)))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "metal_draw_contains_an_incompatible_depth_attachment");
    return RhiAdapterResult<void>{vernon::ok()};
}

void applyRenderPipeline(MetalCommands &commands, id<MTLRenderCommandEncoder> encoder) {
    const PreparedPipeline &pipeline = *commands.pipeline;
    if (commands.renderPipeline == &pipeline)
        return;
    [encoder setRenderPipelineState:pipeline.render];
    if (pipeline.depthStencil)
        [encoder setDepthStencilState:pipeline.depthStencil];
    [encoder setCullMode:pipeline.cullMode];
    [encoder setFrontFacingWinding:pipeline.frontFace];
    [encoder setDepthBias:pipeline.depthBiasEnabled ? pipeline.depthBiasConstant : 0.0f
               slopeScale:pipeline.depthBiasEnabled ? pipeline.depthBiasSlope : 0.0f
                    clamp:0.0f];
    commands.renderPipeline = &pipeline;
}

// Binds the argument buffers, resources, vertex buffers and inline-value blocks of the bound groups that the render
// encoder lacks.
void applyRenderGroups(MetalCommands &commands, id<MTLRenderCommandEncoder> encoder) {
    const PreparedLayout &layout = *commands.pipeline->layout;
    for (uint32_t index = 0; index < layout.groupCount; ++index) {
        const MetalGroup &group = *commands.groups[index];
        const MetalGroup::Memory &vertexArguments = group.arguments[kVertexStage];
        if (vertexArguments.buffer && commands.vertexArguments[index] != group.serial) {
            [encoder setVertexBuffer:vertexArguments.buffer offset:vertexArguments.offset atIndex:index];
            commands.vertexArguments[index] = group.serial;
        }
        const MetalGroup::Memory &fragmentArguments = group.arguments[kFragmentStage];
        if (fragmentArguments.buffer && commands.fragmentArguments[index] != group.serial) {
            [encoder setFragmentBuffer:fragmentArguments.buffer offset:fragmentArguments.offset atIndex:index];
            commands.fragmentArguments[index] = group.serial;
        }
        if (commands.usedGroups[index] != group.serial) {
            for (const MetalGroup::ResourceUse &use : group.uses)
                [encoder useResource:use.resource usage:use.usage stages:renderStages(use.stage)];
            commands.usedGroups[index] = group.serial;
        }
        if ((layout.vertexGroupMask & (1u << index)) != 0 && commands.vertexGroups[index] != group.serial) {
            for (const MetalGroup::VertexBuffer &buffer : group.vertexBuffers)
                [encoder setVertexBuffer:buffer.buffer offset:buffer.offset atIndex:buffer.index];
            commands.vertexGroups[index] = group.serial;
        }
    }
    if (commands.directLayout != &layout) {
        commands.directLayout = &layout;
        commands.directGroups.fill(0);
        commands.directScratch.resize(layout.directScratchSize);
    }
    uint32_t changed = 0;
    for (uint32_t index = 0; index < layout.groupCount; ++index)
        if ((layout.directGroupMask & (1u << index)) != 0 &&
            commands.directGroups[index] != commands.groups[index]->serial) {
            changed |= 1u << index;
            commands.directGroups[index] = commands.groups[index]->serial;
        }
    if (!changed)
        return;
    for (const DirectBlock &block : layout.directBlocks) {
        if ((block.groupMask & changed) == 0)
            continue;
        uint8_t *bytes = commands.directScratch.data() + block.scratchOffset;
        for (const DirectBlock::Member &member : block.members)
            if ((changed & (1u << member.group)) != 0)
                std::memcpy(bytes + member.offset, commands.groups[member.group]->directData.data() + member.dataOffset,
                            member.size);
        if (block.stage == kVertexStage)
            [encoder setVertexBytes:bytes length:block.size atIndex:block.index];
        else
            [encoder setFragmentBytes:bytes length:block.size atIndex:block.index];
    }
}

void applyDynamicState(MetalCommands &commands, id<MTLRenderCommandEncoder> encoder,
                       const VernonRuntimeProviderDrawDescriptor &descriptor) {
    const std::array<uint32_t, 4> viewport{descriptor.viewport[0], descriptor.viewport[1], descriptor.viewport[2],
                                           descriptor.viewport[3]};
    if (!commands.viewportValid || commands.viewport != viewport) {
        [encoder setViewport:{static_cast<double>(viewport[0]), static_cast<double>(viewport[1]),
                              static_cast<double>(viewport[2]), static_cast<double>(viewport[3]), 0.0, 1.0}];
        commands.viewport = viewport;
        commands.viewportValid = true;
    }
    const std::array<uint32_t, 4> scissor{descriptor.scissor[0], descriptor.scissor[1], descriptor.scissor[2],
                                          descriptor.scissor[3]};
    if (!commands.scissorValid || commands.scissor != scissor) {
        [encoder setScissorRect:{scissor[0], scissor[1], scissor[2], scissor[3]}];
        commands.scissor = scissor;
        commands.scissorValid = true;
    }
    if (!commands.stencilValid || commands.stencilReference != descriptor.stencil_reference) {
        [encoder setStencilReferenceValue:descriptor.stencil_reference];
        commands.stencilReference = descriptor.stencil_reference;
        commands.stencilValid = true;
    }
}

RhiAdapterResult<void> encodeDrawResult(void *data, VernonRuntimeProviderObject handle,
                                        const VernonRuntimeProviderDrawDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<MetalCommands>(handle);
    MTLPrimitiveType primitive{};
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !commands || !commands->pipeline ||
        !commands->pipeline->render || !commands->rendering || !primitiveType(descriptor->topology, primitive) ||
        (descriptor->index_count != 0 && descriptor->index_type != 0))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "metal_adapter_received_an_invalid_or_unsupported_draw");
    if (auto bound = requireBoundGroups(*commands); !bound)
        return bound;
    auto opened = openCommandRendering(adapter, commands->encoder);
    if (!opened)
        return RhiAdapterResult<void>{vernon::err(std::move(opened).error())};
    auto *rendering = reinterpret_cast<rhi::metal::RenderingState *>(static_cast<uintptr_t>(opened.value().object));
    if (!rendering || !rendering->encoder)
        return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                               "metal_render_command_encoder_is_unavailable");
    if (auto compatible = checkAttachments(*commands->pipeline, *rendering); !compatible)
        return compatible;
    id<MTLRenderCommandEncoder> encoder = rendering->encoder;
    if (commands->renderEncoder != encoder)
        resetRenderState(*commands, encoder);
    applyRenderPipeline(*commands, encoder);
    applyRenderGroups(*commands, encoder);
    applyDynamicState(*commands, encoder, *descriptor);
    if (descriptor->index_count != 0) {
        const auto &indexBuffer = descriptor->index_buffer;
        if (commands->indexBufferKey != indexBuffer.resource.value) {
            if (auto retained = retainCommandResource(adapter, commands->encoder, indexBuffer); !retained)
                return retained;
            auto resolved = resolveCommandRhiResource(adapter, commands->encoder, indexBuffer);
            if (!resolved)
                return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
            commands->indexBuffer =
                (__bridge id<MTLBuffer>)(reinterpret_cast<void *>(static_cast<uintptr_t>(resolved.value())));
            if (!commands->indexBuffer)
                return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                       "metal_draw_contains_a_stale_index_buffer");
            commands->indexBufferKey = indexBuffer.resource.value;
        }
        [encoder drawIndexedPrimitives:primitive
                            indexCount:descriptor->index_count
                             indexType:MTLIndexTypeUInt32
                           indexBuffer:commands->indexBuffer
                     indexBufferOffset:indexBuffer.offset
                         instanceCount:descriptor->instance_count
                            baseVertex:descriptor->base_vertex
                          baseInstance:descriptor->first_instance];
    } else {
        [encoder drawPrimitives:primitive
                    vertexStart:0
                    vertexCount:descriptor->vertex_count
                  instanceCount:descriptor->instance_count
                   baseInstance:descriptor->first_instance];
    }
    if (auto recorded = recordBoundGroupWrites(adapter, *commands); !recorded)
        return recorded;
    if (auto recorded = recordProviderCommand(adapter, commands->encoder, true); !recorded)
        return recorded;
    adapter.lastStencilReference.store(descriptor->stencil_reference, std::memory_order_relaxed);
    adapter.lastDrawIndexed.store(descriptor->index_count != 0, std::memory_order_relaxed);
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> beginRenderingResult(void *data, VernonRuntimeProviderObject handle,
                                            const VernonRuntimeProviderRenderingDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<MetalCommands>(handle);
    if (!commands || !descriptor || commands->rendering)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "metal_begin_rendering_is_invalid");
    if (auto begun = beginEncoderRendering(adapter, commands->encoder, *descriptor); !begun)
        return begun;
    commands->rendering = true;
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> endRenderingResult(void *data, VernonRuntimeProviderObject handle) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<MetalCommands>(handle);
    if (!commands || !commands->rendering)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "metal_end_rendering_is_invalid");
    commands->rendering = false;
    resetRenderState(*commands, nil);
    return endEncoderRendering(adapter, commands->encoder);
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
void destroyGroup(void *, VernonRuntimeProviderObject handle) { releaseRetainedObject(fromHandle<MetalGroup>(handle)); }

VernonStatus prepareShader(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, prepareShaderResult(data, descriptor, output), "Metal shader preparation failed");
}
VernonStatus prepareGroupLayout(void *data, const VernonRuntimeProviderGroupLayoutDescriptor *descriptor,
                                VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, prepareGroupLayoutResult(data, descriptor, output),
                          "Metal group layout preparation failed");
}
VernonStatus prepareLayout(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, prepareLayoutResult(data, descriptor, output), "Metal layout preparation failed");
}
VernonStatus preparePipeline(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                             VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, preparePipelineResult(data, descriptor, output),
                          "Metal pipeline preparation failed");
}
VernonStatus createGroup(void *data, const VernonRuntimeProviderGroupDescriptor *descriptor,
                         VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, createGroupResult(data, descriptor, output), "Metal group creation failed");
}
VernonStatus beginCommands(void *data, VernonRuntimeProviderObject encoder, VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, beginCommandsResult(data, encoder, output), "Metal commands could not begin");
}
VernonStatus endCommands(void *data, VernonRuntimeProviderObject commands) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, endCommandsResult(data, commands), "Metal commands failed");
}
VernonStatus bindPipeline(void *data, VernonRuntimeProviderObject commands, VernonRuntimeProviderObject pipeline) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, bindPipelineResult(data, commands, pipeline), "Metal pipeline binding failed");
}
VernonStatus bindGroup(void *data, VernonRuntimeProviderObject commands, uint32_t index,
                       VernonRuntimeProviderObject group) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, bindGroupResult(data, commands, index, group), "Metal group binding failed");
}
VernonStatus encodeDispatch(void *data, VernonRuntimeProviderObject commands,
                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, encodeDispatchResult(data, commands, descriptor), "Metal dispatch encoding failed");
}
VernonStatus encodeDraw(void *data, VernonRuntimeProviderObject commands,
                        const VernonRuntimeProviderDrawDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, encodeDrawResult(data, commands, descriptor), "Metal draw encoding failed");
}
VernonStatus beginRendering(void *data, VernonRuntimeProviderObject commands,
                            const VernonRuntimeProviderRenderingDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, beginRenderingResult(data, commands, descriptor), "Metal rendering could not begin");
}
VernonStatus endRendering(void *data, VernonRuntimeProviderObject commands) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, endRenderingResult(data, commands), "Metal rendering could not end");
}

} // namespace

void initializeMetalProvider(VernonRuntimeRhiAdapter &adapter) {
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
    adapter.provider.encode_bind_pipeline = bindPipeline;
    adapter.provider.encode_bind_group = bindGroup;
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

vernon::runtime::MetalRuntimeDeviceCapabilities
vernon::runtime::metalRhiAdapterDeviceCapabilities(const VernonRuntimeRhiAdapter &adapter) {
    const auto &device = rhi_adapter::metalDevice(adapter);
    MetalRuntimeDeviceCapabilities capabilities;
    capabilities.maxComputeInvocations = device.maxComputeInvocations;
    std::copy_n(device.maxComputeWorkGroupSize, 3, capabilities.maxComputeWorkGroupSize);
    std::copy_n(device.operatingSystemVersion, 2, capabilities.operatingSystemVersion);
    capabilities.argumentBuffersTier = device.argumentBuffersTier;
    capabilities.argumentBufferEncodingSupported = device.argumentBufferEncodingSupported;
    return capabilities;
}

VernonRuntimeRhiAdapter *vernon::runtime::createMetalRhiAdapter(VernonRhiDevice device, VernonRhiBackend backend) {
    if (backend != VERNON_RHI_BACKEND_METAL)
        return nullptr;
    auto resolvedDeviceState = rhi::deviceState(device, backend);
    if (resolvedDeviceState.isErr())
        return nullptr;
    auto *deviceState = static_cast<rhi::metal::DeviceState *>(resolvedDeviceState.value());
    auto state = std::unique_ptr<rhi_adapter::MetalAdapterState>(new (std::nothrow) rhi_adapter::MetalAdapterState());
    auto adapter = std::unique_ptr<VernonRuntimeRhiAdapter>(new (std::nothrow) VernonRuntimeRhiAdapter());
    if (!state || !adapter)
        return nullptr;
    state->device = deviceState;
    MTLSamplerDescriptor *sampler = [[MTLSamplerDescriptor alloc] init];
    sampler.minFilter = MTLSamplerMinMagFilterLinear;
    sampler.magFilter = MTLSamplerMinMagFilterLinear;
    sampler.mipFilter = MTLSamplerMipFilterLinear;
    sampler.supportArgumentBuffers = YES;
    state->defaultSampler = [deviceState->device newSamplerStateWithDescriptor:sampler];
    if (!state->defaultSampler)
        return nullptr;
    adapter->rhiBackend = backend;
    if (!adapter->backend.adopt(state.get(), &rhi_adapter::backendOps))
        return nullptr;
    [[maybe_unused]] auto *adoptedState = state.release();
    rhi_adapter::initializeMetalProvider(*adapter);
    return adapter.release();
}

#endif
