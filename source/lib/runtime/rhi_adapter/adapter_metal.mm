#include "rhi/rhi_internal.h"
#include "adapter_common.h"
#include "adapter_internal.h"

#if defined(VERNON_HAS_METAL_RHI)

#include "rhi/metal_backend.h"
#include "runtime/metal_runtime_capabilities.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace vernon::runtime::rhi_adapter {
namespace {

struct MetalAdapterState {
    rhi::metal::DeviceState *device{};
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
VernonStatus synchronizeBackend(void *state, std::string &error) noexcept {
    @try {
        try {
            return static_cast<MetalAdapterState *>(state)->device->synchronize(error)
                       ? VERNON_STATUS_OK
                       : VERNON_STATUS_INTERNAL_ERROR;
        } catch (...) {
            setBackendError(error, "Metal synchronization threw a C++ exception");
            return VERNON_STATUS_INTERNAL_ERROR;
        }
    } @catch (NSException *) {
        setBackendError(error, "Metal synchronization raised an Objective-C exception");
        return VERNON_STATUS_INTERNAL_ERROR;
    }
}
uint64_t resourceIdentity(const void *state) noexcept {
    return reinterpret_cast<uintptr_t>(static_cast<const MetalAdapterState *>(state)->device);
}
void invalidateBackend(void *) noexcept {}

const RhiAdapterBackendOps backendOps{destroyBackend, synchronizeBackend, resourceIdentity, invalidateBackend};

struct PreparedShader {
    id<MTLLibrary> library;
    id<MTLFunction> function;
    uint32_t stage{};
};

struct PreparedLayout {
    struct ArgumentGroup {
        uint32_t stage{};
        uint32_t index{};
        id<MTLFunction> function;
    };

    std::vector<VernonRuntimeProviderBindingLayoutEntry> entries;
    std::vector<VernonRuntimeProviderVertexAttribute> vertexAttributes;
    std::unordered_map<uint32_t, size_t> slotIndices;
    std::vector<ArgumentGroup> argumentGroups;
    uint32_t pushConstantSize{};
    std::mutex mutex;
};

struct PreparedPipeline {
    std::atomic<uint32_t> references{1};
    VernonRuntimeRhiAdapter *owner{};
    id<MTLComputePipelineState> compute;
    id<MTLRenderPipelineState> render;
    id<MTLDepthStencilState> depthStencil;
    std::array<MTLPixelFormat, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> colorFormats{};
    MTLPixelFormat depthFormat{MTLPixelFormatInvalid};
    size_t colorFormatCount{};
    uint32_t sampleCount{1};
    PreparedLayout *layout{};
    uint32_t workgroup[3]{1, 1, 1};
    uint32_t pushConstantSize{};
    MTLCullMode cullMode{MTLCullModeNone};
    MTLWinding frontFace{MTLWindingCounterClockwise};
    float depthBiasConstant{};
    float depthBiasSlope{};
    bool depthBiasEnabled{};
    bool graphics{};

    ~PreparedPipeline() {
        if (owner)
            owner->livePreparedPipelines.fetch_sub(1, std::memory_order_relaxed);
    }
};

struct PreparedBindingSet {
    struct Slot {
        VernonRuntimeProviderBindingLayoutEntry layout{};
        VernonRuntimeProviderResourceReference resource{};
        std::vector<uint8_t> inlineStorage;
        id<MTLBuffer> inlineBuffer;
        id<MTLSamplerState> defaultSampler;
        uint32_t stride{};
    };
    struct ArgumentBuffer {
        uint32_t stage{};
        uint32_t index{};
        id<MTLArgumentEncoder> encoder;
        id<MTLBuffer> buffer;
    };

    std::atomic<uint32_t> references{1};
    PreparedLayout *layout{};
    std::vector<Slot> slots;
    std::vector<ArgumentBuffer> argumentBuffers;
    std::unordered_map<uint32_t, size_t> slotIndices;
    std::mutex mutex;
};

void setNativeError(VernonRuntimeRhiAdapter &adapter, const char *operation, NSError *error) {
    adapter.error = operation;
    if (error.localizedDescription.length != 0) {
        adapter.error += ": ";
        adapter.error += error.localizedDescription.UTF8String;
    }
}

VernonStatus failNativeException(VernonRuntimeRhiAdapter &adapter, const char *operation, NSException *exception) {
    adapter.error = operation;
    if (exception.reason.length != 0) {
        adapter.error += ": ";
        adapter.error += exception.reason.UTF8String;
    }
    return VERNON_STATUS_INVALID_ARGUMENT;
}

bool stringEquals(VernonStringView value, const char *expected) {
    const size_t size = std::strlen(expected);
    return value.data && value.size == size && std::memcmp(value.data, expected, size) == 0;
}

bool isArgumentResource(const VernonRuntimeProviderBindingLayoutEntry &entry) {
    return entry.kind != VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && entry.set != UINT32_MAX;
}

MTLResourceUsage resourceUsage(const VernonRuntimeProviderBindingLayoutEntry &entry) {
    MTLResourceUsage usage = MTLResourceUsageRead;
    if (entry.access & 2u)
        usage |= MTLResourceUsageWrite;
    return usage;
}

MTLRenderStages renderStages(uint32_t stageMask) {
    MTLRenderStages stages = 0;
    if (stageMask & VERNON_RUNTIME_PROVIDER_STAGE_VERTEX)
        stages |= MTLRenderStageVertex;
    if (stageMask & VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT)
        stages |= MTLRenderStageFragment;
    return stages;
}

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
    return {0x4d4554414cull, metalDevice(adapter).device.registryID, 0};
}

VernonStatus prepareShader(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        (descriptor->stage != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE &&
         descriptor->stage != VERNON_RUNTIME_PROVIDER_STAGE_VERTEX &&
         descriptor->stage != VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT) ||
        !stringEquals(descriptor->format, "msl") || !descriptor->data || descriptor->size == 0 ||
        !descriptor->entry.data || descriptor->entry.size == 0)
        return fail(adapter, "Metal adapter received an invalid shader descriptor");
    @autoreleasepool {
        NSString *source =
            [[NSString alloc] initWithBytes:descriptor->data length:descriptor->size encoding:NSUTF8StringEncoding];
        NSString *entry = [[NSString alloc] initWithBytes:descriptor->entry.data
                                                   length:descriptor->entry.size
                                                 encoding:NSUTF8StringEncoding];
        if (!source || !entry)
            return fail(adapter, "Metal shader source or entry point is not valid UTF-8");
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion2_4;
        NSError *error = nil;
        id<MTLLibrary> library = [metalDevice(adapter).device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            setNativeError(adapter, "Metal shader library compilation failed", error);
            return VERNON_STATUS_INVALID_ARGUMENT;
        }
        id<MTLFunction> function = [library newFunctionWithName:entry];
        if (!function)
            return fail(adapter, "Metal shader entry point was not found");
        auto shader = std::unique_ptr<PreparedShader>(new (std::nothrow) PreparedShader());
        if (!shader)
            return fail(adapter, "Metal shader preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
        shader->library = library;
        shader->function = function;
        shader->stage = descriptor->stage;
        *output = toHandle(shader.release());
        adapter.shaderPreparations.fetch_add(1, std::memory_order_relaxed);
        return VERNON_STATUS_OK;
    }
}

VernonStatus prepareLayout(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        (descriptor->binding_count && !descriptor->bindings) ||
        (descriptor->vertex_attribute_count && !descriptor->vertex_attributes))
        return fail(adapter, "Metal adapter received an invalid pipeline layout");
    try {
        auto layout = std::make_unique<PreparedLayout>();
        layout->entries.assign(descriptor->bindings, descriptor->bindings + descriptor->binding_count);
        layout->slotIndices.reserve(layout->entries.size());
        layout->pushConstantSize = descriptor->push_constant_size;
        uint64_t argumentBuffers = 0;
        uint64_t argumentTextures = 0;
        uint64_t argumentSamplers = 0;
        bool writableTexture = false;
        std::set<std::tuple<uint32_t, uint32_t, uint32_t>> argumentLocations;
        for (size_t index = 0; index < layout->entries.size(); ++index) {
            const auto &entry = layout->entries[index];
            const bool supportedKind = entry.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
                                       entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
                                       entry.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                                       entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE ||
                                       entry.kind == VERNON_RUNTIME_PROVIDER_SAMPLER ||
                                       entry.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
                                       entry.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER;
            const bool bufferKind = entry.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
                                    entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
                                    entry.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
                                    entry.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER;
            const bool argumentResource = isArgumentResource(entry);
            const bool directResource = entry.kind != VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && !argumentResource;
            const uint32_t supportedStages = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE |
                                             VERNON_RUNTIME_PROVIDER_STAGE_VERTEX |
                                             VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT;
            if (!supportedKind || !entry.stage_mask || (entry.stage_mask & ~supportedStages) ||
                entry.array_count != 1 ||
                (argumentResource &&
                 (entry.set >= 8 || entry.binding == UINT32_MAX || (entry.stage_mask & (entry.stage_mask - 1)) != 0)) ||
                (directResource && entry.kind != VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
                 entry.kind != VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER) ||
                (directResource &&
                 (entry.binding >= (entry.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE ? 31u : 15u) ||
                  entry.binding == 15)) ||
                (entry.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && entry.binding >= 31) ||
                ((entry.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
                  entry.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
                  entry.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER) &&
                 entry.element_size == 0) ||
                (entry.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER &&
                 entry.stage_mask != VERNON_RUNTIME_PROVIDER_STAGE_VERTEX) ||
                !layout->slotIndices.emplace(entry.slot, index).second ||
                (argumentResource && !argumentLocations.emplace(entry.stage_mask, entry.set, entry.binding).second))
                return fail(adapter, "Metal layout contains an unsupported or duplicate binding");
            if (argumentResource) {
                if (bufferKind)
                    argumentBuffers += entry.array_count;
                else if (entry.kind == VERNON_RUNTIME_PROVIDER_SAMPLER)
                    argumentSamplers += entry.array_count;
                else
                    argumentTextures += entry.array_count;
                writableTexture |= entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE && (entry.access & 2u);
            }
        }
        if ((argumentBuffers || argumentTextures || argumentSamplers) &&
            !metalDevice(adapter).argumentBufferEncodingSupported)
            return fail(adapter, "Metal argument-buffer encoding is unavailable on this device",
                        VERNON_STATUS_UNSUPPORTED_TARGET);
        if (metalDevice(adapter).device.argumentBuffersSupport < MTLArgumentBuffersTier2 &&
            (argumentBuffers > 31 || argumentTextures > 31 || argumentSamplers > 16 || writableTexture))
            return fail(adapter, "Metal layout requires Argument Buffers Tier 2");
        for (size_t index = 0; index < descriptor->vertex_attribute_count; ++index) {
            const auto &attribute = descriptor->vertex_attributes[index];
            const bool bindingExists =
                std::any_of(layout->entries.begin(), layout->entries.end(), [&](const auto &entry) {
                    return entry.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && entry.binding == attribute.binding;
                });
            if (!bindingExists || attribute.location >= 31 || attribute.component_count == 0 ||
                attribute.component_count > 4)
                return fail(adapter, "Metal vertex attribute is invalid or references an unknown binding");
            layout->vertexAttributes.push_back(attribute);
        }
        *output = toHandle(layout.release());
        adapter.layoutPreparations.fetch_add(1, std::memory_order_relaxed);
        return VERNON_STATUS_OK;
    } catch (const std::bad_alloc &) {
        return fail(adapter, "Metal layout preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
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
    constexpr MTLCompareFunction functions[]{
        MTLCompareFunctionNever,        MTLCompareFunctionLess,    MTLCompareFunctionEqual,
        MTLCompareFunctionLessEqual,    MTLCompareFunctionGreater, MTLCompareFunctionNotEqual,
        MTLCompareFunctionGreaterEqual, MTLCompareFunctionAlways};
    if (operation >= std::size(functions))
        return false;
    output = functions[operation];
    return true;
}

bool stencilOperation(uint32_t operation, MTLStencilOperation &output) {
    constexpr MTLStencilOperation operations[]{
        MTLStencilOperationKeep,           MTLStencilOperationZero,           MTLStencilOperationReplace,
        MTLStencilOperationIncrementClamp, MTLStencilOperationDecrementClamp, MTLStencilOperationInvert,
        MTLStencilOperationIncrementWrap,  MTLStencilOperationDecrementWrap};
    if (operation >= std::size(operations))
        return false;
    output = operations[operation];
    return true;
}

bool blendFactor(uint32_t factor, MTLBlendFactor &output) {
    constexpr MTLBlendFactor factors[]{
        MTLBlendFactorZero,
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

bool configureStencilFace(const VernonStencilFaceState &source,
                          MTLStencilDescriptor *destination) {
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

VernonStatus initializeArgumentGroups(VernonRuntimeRhiAdapter &adapter, PreparedLayout &layout,
                                      const std::vector<PreparedShader *> &shaders) {
    @try {
        std::lock_guard<std::mutex> guard(layout.mutex);
        if (!layout.argumentGroups.empty())
            return VERNON_STATUS_OK;
        std::vector<PreparedLayout::ArgumentGroup> argumentGroups;
        for (const auto &entry : layout.entries) {
            if (!isArgumentResource(entry))
                continue;
            const bool exists = std::any_of(argumentGroups.begin(), argumentGroups.end(), [&](const auto &group) {
                return group.stage == entry.stage_mask && group.index == entry.set;
            });
            if (exists)
                continue;
            const auto shader = std::find_if(shaders.begin(), shaders.end(), [&](const PreparedShader *candidate) {
                return candidate && candidate->stage == entry.stage_mask;
            });
            if (shader == shaders.end())
                return fail(adapter, "Metal argument-buffer layout references a missing shader stage");
            id<MTLArgumentEncoder> encoder = [(*shader)->function newArgumentEncoderWithBufferIndex:entry.set];
            if (!encoder || encoder.encodedLength == 0)
                return fail(adapter, "Metal shader has no argument buffer at the reflected buffer index");
            argumentGroups.push_back({entry.stage_mask, entry.set, (*shader)->function});
        }
        layout.argumentGroups = std::move(argumentGroups);
        return VERNON_STATUS_OK;
    } @catch (NSException *exception) {
        return failNativeException(adapter, "Metal argument-buffer layout encoding raised an exception", exception);
    }
}

VernonStatus preparePipeline(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                             VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    PreparedLayout *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !layout || !descriptor->shaders ||
        descriptor->shader_count == 0)
        return fail(adapter, "Metal adapter received an invalid pipeline");
    @autoreleasepool {
        auto pipeline = std::unique_ptr<PreparedPipeline>(new (std::nothrow) PreparedPipeline());
        if (!pipeline)
            return fail(adapter, "Metal pipeline preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
        pipeline->owner = &adapter;
        adapter.livePreparedPipelines.fetch_add(1, std::memory_order_relaxed);
        pipeline->pushConstantSize = layout->pushConstantSize;
        pipeline->layout = layout;
        NSError *error = nil;
        if (descriptor->kind == VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE) {
            PreparedShader *shader =
                descriptor->shader_count == 1 ? fromHandle<PreparedShader>(descriptor->shaders[0]) : nullptr;
            if (!shader || shader->stage != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE ||
                descriptor->workgroup_size[0] == 0 || descriptor->workgroup_size[1] == 0 ||
                descriptor->workgroup_size[2] == 0)
                return fail(adapter, "Metal adapter received an invalid compute pipeline");
            const VernonStatus groupsStatus = initializeArgumentGroups(adapter, *layout, {shader});
            if (groupsStatus != VERNON_STATUS_OK)
                return groupsStatus;
            pipeline->compute =
                [metalDevice(adapter).device newComputePipelineStateWithFunction:shader->function error:&error];
            if (!pipeline->compute) {
                setNativeError(adapter, "Metal compute pipeline creation failed", error);
                return VERNON_STATUS_INVALID_ARGUMENT;
            }
            const uint64_t total = static_cast<uint64_t>(descriptor->workgroup_size[0]) *
                                   descriptor->workgroup_size[1] * descriptor->workgroup_size[2];
            if (!total || total > pipeline->compute.maxTotalThreadsPerThreadgroup)
                return fail(adapter, "Metal compute workgroup exceeds pipeline limits");
            std::copy_n(descriptor->workgroup_size, 3, pipeline->workgroup);
        } else if (descriptor->kind == VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE) {
            PreparedShader *vertex = nullptr;
            PreparedShader *fragment = nullptr;
            for (size_t index = 0; index < descriptor->shader_count; ++index) {
                auto *shader = fromHandle<PreparedShader>(descriptor->shaders[index]);
                if (shader && shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX)
                    vertex = shader;
                else if (shader && shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT)
                    fragment = shader;
            }
            if (!vertex || !fragment)
                return fail(adapter, "Metal graphics pipeline is missing a vertex or fragment shader");
            const VernonStatus groupsStatus = initializeArgumentGroups(adapter, *layout, {vertex, fragment});
            if (groupsStatus != VERNON_STATUS_OK)
                return groupsStatus;
            if (descriptor->color_format_count == 0 && descriptor->depth_stencil_format == 0) {
                pipeline->graphics = true;
                *output = toHandle(pipeline.release());
                adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
                return VERNON_STATUS_OK;
            }
            if (descriptor->color_format_count > 8 || (descriptor->color_format_count && !descriptor->color_formats) ||
                descriptor->sample_count != 1 ||
                (descriptor->depth_stencil_format && descriptor->depth_stencil_format != MTLPixelFormatDepth32Float &&
                 descriptor->depth_stencil_format != MTLPixelFormatDepth32Float_Stencil8) ||
                descriptor->rasterization.cull_mode > VERNON_RHI_CULL_BACK ||
                descriptor->rasterization.front_face > VERNON_RHI_FRONT_FACE_CLOCKWISE ||
                descriptor->rasterization.depth_clamp || descriptor->rasterization.depth_bias_enabled > 1 ||
                !std::isfinite(descriptor->rasterization.depth_bias_constant) ||
                !std::isfinite(descriptor->rasterization.depth_bias_slope) ||
                descriptor->color_blend_count != descriptor->color_format_count ||
                (descriptor->color_blend_count && !descriptor->color_blends))
                return fail(adapter, "Metal graphics pipeline uses an unsupported attachment configuration");
            MTLRenderPipelineDescriptor *nativeDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
            nativeDescriptor.vertexFunction = vertex->function;
            nativeDescriptor.fragmentFunction = fragment->function;
            nativeDescriptor.rasterSampleCount = std::max(1u, descriptor->sample_count);
            if (!layout->vertexAttributes.empty()) {
                if (!descriptor->vertex_strides || descriptor->vertex_stride_count == 0)
                    return fail(adapter, "Metal graphics pipeline has no vertex strides");
                MTLVertexDescriptor *vertexDescriptor = [MTLVertexDescriptor vertexDescriptor];
                for (const auto &entry : layout->entries) {
                    if (entry.kind != VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER)
                        continue;
                    if (entry.binding >= descriptor->vertex_stride_count ||
                        descriptor->vertex_strides[entry.binding] == 0)
                        return fail(adapter, "Metal graphics pipeline has an invalid vertex stride");
                    vertexDescriptor.layouts[entry.binding].stride = descriptor->vertex_strides[entry.binding];
                    vertexDescriptor.layouts[entry.binding].stepFunction =
                        entry.divisor ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
                    vertexDescriptor.layouts[entry.binding].stepRate = entry.divisor ? entry.divisor : 1;
                }
                for (const auto &attribute : layout->vertexAttributes) {
                    const MTLVertexFormat format = vertexFormat(attribute.dtype, attribute.component_count);
                    if (format == MTLVertexFormatInvalid)
                        return fail(adapter, "Metal graphics pipeline has an unsupported vertex format");
                    vertexDescriptor.attributes[attribute.location].format = format;
                    vertexDescriptor.attributes[attribute.location].offset = attribute.relative_offset;
                    vertexDescriptor.attributes[attribute.location].bufferIndex = attribute.binding;
                }
                nativeDescriptor.vertexDescriptor = vertexDescriptor;
            }
            for (size_t index = 0; index < descriptor->color_format_count; ++index) {
                const MTLPixelFormat format = static_cast<MTLPixelFormat>(descriptor->color_formats[index]);
                if (format == MTLPixelFormatInvalid)
                    return fail(adapter, "Metal graphics pipeline contains an invalid color format");
                auto *attachment = nativeDescriptor.colorAttachments[index];
                attachment.pixelFormat = format;
                const auto &blend = descriptor->color_blends[index];
                MTLBlendFactor sourceColor{}, destinationColor{}, sourceAlpha{}, destinationAlpha{};
                MTLBlendOperation colorOperation{}, alphaOperation{};
                if (blend.blend_enabled > 1 || (blend.write_mask & ~VERNON_RHI_COLOR_WRITE_ALL) ||
                    !blendFactor(blend.source_color_factor, sourceColor) ||
                    !blendFactor(blend.destination_color_factor, destinationColor) ||
                    !blendFactor(blend.source_alpha_factor, sourceAlpha) ||
                    !blendFactor(blend.destination_alpha_factor, destinationAlpha) ||
                    !blendOperation(blend.color_operation, colorOperation) ||
                    !blendOperation(blend.alpha_operation, alphaOperation))
                    return fail(adapter, "Metal graphics pipeline contains an invalid blend state");
                attachment.blendingEnabled = blend.blend_enabled;
                attachment.sourceRGBBlendFactor = sourceColor;
                attachment.destinationRGBBlendFactor = destinationColor;
                attachment.rgbBlendOperation = colorOperation;
                attachment.sourceAlphaBlendFactor = sourceAlpha;
                attachment.destinationAlphaBlendFactor = destinationAlpha;
                attachment.alphaBlendOperation = alphaOperation;
                attachment.writeMask = colorWriteMask(blend.write_mask);
            }
            nativeDescriptor.depthAttachmentPixelFormat = static_cast<MTLPixelFormat>(descriptor->depth_stencil_format);
            if (descriptor->depth_stencil_format == MTLPixelFormatDepth32Float_Stencil8)
                nativeDescriptor.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
            pipeline->render =
                [metalDevice(adapter).device newRenderPipelineStateWithDescriptor:nativeDescriptor error:&error];
            if (!pipeline->render) {
                setNativeError(adapter, "Metal render pipeline creation failed", error);
                return VERNON_STATUS_INVALID_ARGUMENT;
            }
            if (descriptor->depth_stencil_format) {
                MTLDepthStencilDescriptor *depthDescriptor = [[MTLDepthStencilDescriptor alloc] init];
                auto state = descriptor->depth_stencil;
                MTLCompareFunction depthCompare{};
                if (state.depth_test > 1 || state.depth_write > 1 || state.stencil_test > 1 ||
                    !compareFunction(state.depth_test ? state.depth_compare : VERNON_RHI_COMPARE_ALWAYS,
                                     depthCompare) ||
                    (state.stencil_test &&
                     descriptor->depth_stencil_format != MTLPixelFormatDepth32Float_Stencil8))
                    return fail(adapter, "Metal graphics pipeline contains an invalid depth/stencil state");
                depthDescriptor.depthCompareFunction = depthCompare;
                depthDescriptor.depthWriteEnabled = state.depth_write;
                if (state.stencil_test) {
                    MTLStencilDescriptor *front = [[MTLStencilDescriptor alloc] init];
                    MTLStencilDescriptor *back = [[MTLStencilDescriptor alloc] init];
                    if (!configureStencilFace(state.front, front) || !configureStencilFace(state.back, back) ||
                        state.stencil_read_mask > UINT8_MAX || state.stencil_write_mask > UINT8_MAX)
                        return fail(adapter, "Metal graphics pipeline contains an unsupported stencil state");
                    front.readMask = state.stencil_read_mask;
                    front.writeMask = state.stencil_write_mask;
                    back.readMask = state.stencil_read_mask;
                    back.writeMask = state.stencil_write_mask;
                    depthDescriptor.frontFaceStencil = front;
                    depthDescriptor.backFaceStencil = back;
                }
                pipeline->depthStencil =
                    [metalDevice(adapter).device newDepthStencilStateWithDescriptor:depthDescriptor];
                if (!pipeline->depthStencil)
                    return fail(adapter, "Metal depth state creation failed", VERNON_STATUS_INTERNAL_ERROR);
            }
            pipeline->colorFormatCount = descriptor->color_format_count;
            for (size_t index = 0; index < descriptor->color_format_count; ++index)
                pipeline->colorFormats[index] = static_cast<MTLPixelFormat>(descriptor->color_formats[index]);
            pipeline->depthFormat = static_cast<MTLPixelFormat>(descriptor->depth_stencil_format);
            pipeline->sampleCount = std::max(1u, descriptor->sample_count);
            pipeline->cullMode = static_cast<MTLCullMode>(descriptor->rasterization.cull_mode);
            pipeline->frontFace = descriptor->rasterization.front_face == VERNON_RHI_FRONT_FACE_CLOCKWISE
                                      ? MTLWindingClockwise
                                      : MTLWindingCounterClockwise;
            pipeline->depthBiasEnabled = descriptor->rasterization.depth_bias_enabled != 0;
            pipeline->depthBiasConstant = descriptor->rasterization.depth_bias_constant;
            pipeline->depthBiasSlope = descriptor->rasterization.depth_bias_slope;
            pipeline->graphics = true;
        } else {
            return fail(adapter, "Metal adapter received an unknown pipeline kind");
        }
        *output = toHandle(pipeline.release());
        adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
        return VERNON_STATUS_OK;
    }
}

VernonStatus retainResource(void *data, VernonRuntimeProviderResourceReference resource) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return retainRhiResource(adapter, resource) ? VERNON_STATUS_OK
                                                : fail(adapter, "Metal adapter received a stale resource");
}

void releaseResource(void *data, VernonRuntimeProviderResourceReference resource) {
    releaseRhiResource(*static_cast<VernonRuntimeRhiAdapter *>(data), resource);
}

VernonStatus createArgumentBuffers(VernonRuntimeRhiAdapter &adapter, PreparedLayout &layout,
                                   std::vector<PreparedBindingSet::ArgumentBuffer> &output, bool required) {
    @try {
        std::lock_guard<std::mutex> guard(layout.mutex);
        if (layout.argumentGroups.empty()) {
            if (required && std::any_of(layout.entries.begin(), layout.entries.end(), isArgumentResource))
                return fail(adapter, "Metal argument-buffer layout has no prepared pipeline");
            return VERNON_STATUS_OK;
        }
        std::vector<PreparedBindingSet::ArgumentBuffer> argumentBuffers;
        argumentBuffers.reserve(layout.argumentGroups.size());
        for (const auto &group : layout.argumentGroups) {
            id<MTLArgumentEncoder> encoder = [group.function newArgumentEncoderWithBufferIndex:group.index];
            if (!encoder || encoder.encodedLength == 0)
                return fail(adapter, "Metal could not create an argument encoder for the binding set");
            argumentBuffers.push_back({group.stage, group.index, encoder, nil});
        }
        output = std::move(argumentBuffers);
        return VERNON_STATUS_OK;
    } @catch (NSException *exception) {
        return failNativeException(adapter, "Metal argument-buffer creation raised an exception", exception);
    }
}

VernonStatus encodeArgumentBuffers(VernonRuntimeRhiAdapter &adapter, const std::vector<PreparedBindingSet::Slot> &slots,
                                   std::vector<PreparedBindingSet::ArgumentBuffer> &argumentBuffers) {
    @try {
        auto encodedArgumentBuffers = argumentBuffers;
        for (auto &argumentBuffer : encodedArgumentBuffers) {
            argumentBuffer.buffer =
                [metalDevice(adapter).device newBufferWithLength:argumentBuffer.encoder.encodedLength
                                                        options:MTLResourceStorageModeShared];
            if (!argumentBuffer.buffer)
                return fail(adapter, "Metal argument-buffer snapshot allocation failed", VERNON_STATUS_INTERNAL_ERROR);
            [argumentBuffer.encoder setArgumentBuffer:argumentBuffer.buffer offset:0];
            for (const auto &slot : slots) {
                if (!isArgumentResource(slot.layout) || slot.layout.stage_mask != argumentBuffer.stage ||
                    slot.layout.set != argumentBuffer.index)
                    continue;
                const uint32_t member = slot.layout.binding;
                if (packedUniformBytes(slot.layout.kind, slot.layout.interface_kind)) {
                    [argumentBuffer.encoder setBuffer:slot.inlineBuffer offset:0 atIndex:member];
                    continue;
                }
                if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER && slot.defaultSampler) {
                    [argumentBuffer.encoder setSamplerState:slot.defaultSampler atIndex:member];
                    continue;
                }
                const uint64_t resolved = resolveRhiResource(adapter, slot.resource);
                if (!resolved)
                    return fail(adapter, "Metal argument-buffer resource became stale during update");
                if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
                    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(reinterpret_cast<void *>(resolved));
                    [argumentBuffer.encoder setBuffer:buffer offset:slot.resource.offset atIndex:member];
                } else if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                           slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE) {
                    id<MTLTexture> texture = (__bridge id<MTLTexture>)(reinterpret_cast<void *>(resolved));
                    [argumentBuffer.encoder setTexture:texture atIndex:member];
                } else {
                    id<MTLSamplerState> sampler = (__bridge id<MTLSamplerState>)(reinterpret_cast<void *>(resolved));
                    [argumentBuffer.encoder setSamplerState:sampler atIndex:member];
                }
            }
        }
        argumentBuffers = std::move(encodedArgumentBuffers);
        return VERNON_STATUS_OK;
    } @catch (NSException *exception) {
        return failNativeException(adapter, "Metal argument-buffer resource encoding raised an exception", exception);
    }
}

VernonStatus updateBindingsImpl(VernonRuntimeRhiAdapter &adapter, PreparedBindingSet &bindings,
                                const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    if (valueCount != bindings.slots.size() || (valueCount && !values))
        return fail(adapter, "Metal binding values do not match the prepared layout");
    std::vector<size_t> valueIndices(bindings.slots.size());
    std::vector<uint8_t> seen(bindings.slots.size());
    for (size_t index = 0; index < valueCount; ++index) {
        const auto found = bindings.slotIndices.find(values[index].slot);
        if (found == bindings.slotIndices.end() || seen[found->second])
            return fail(adapter, "Metal binding slot is invalid or duplicated");
        seen[found->second] = 1;
        valueIndices[found->second] = index;
    }
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        const auto &slot = bindings.slots[index];
        const auto &value = values[valueIndices[index]];
        if (value.kind != slot.layout.kind)
            return fail(adapter, "Metal binding kind does not match the prepared layout");
        if (packedUniformBytes(slot.layout.kind, slot.layout.interface_kind)) {
            if (!value.payload.inline_value.data || value.payload.inline_value.size != slot.layout.element_size)
                return fail(adapter, "Metal inline binding has an invalid physical size");
        } else if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER &&
                   (value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE)) {
            if (value.payload.sampler.resource.resource.value)
                return fail(adapter, "Metal default sampler binding also supplied a resource");
        } else {
            const auto *resource = providerBindingResource(value);
            const bool image = slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                               slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
            const uint64_t expectedKind =
                slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER ? kRhiSamplerResource : kRhiBufferResource;
            if (!resource || (image ? !isRhiImageReference(*resource)
                                    : (resource->identity & kRhiResourceKindMask) != expectedKind) ||
                !resolveRhiResource(adapter, *resource))
                return fail(adapter,
                            "Metal resource binding has the wrong type, is stale, or belongs to another device");
            if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && value.payload.buffer.stride == 0)
                return fail(adapter, "Metal vertex binding has no stride");
        }
    }
    auto updatedSlots = bindings.slots;
    for (size_t index = 0; index < updatedSlots.size(); ++index) {
        auto &slot = updatedSlots[index];
        const auto &value = values[valueIndices[index]];
        if (packedUniformBytes(slot.layout.kind, slot.layout.interface_kind)) {
            slot.resource = {};
            slot.inlineStorage.assign(static_cast<const uint8_t *>(value.payload.inline_value.data),
                                      static_cast<const uint8_t *>(value.payload.inline_value.data) +
                                          value.payload.inline_value.size);
            if (isArgumentResource(slot.layout)) {
                slot.inlineBuffer = [metalDevice(adapter).device newBufferWithLength:value.payload.inline_value.size
                                                                             options:MTLResourceStorageModeShared];
                if (!slot.inlineBuffer)
                    return fail(adapter, "Metal inline argument-buffer resource allocation failed",
                                VERNON_STATUS_INTERNAL_ERROR);
                std::memcpy(slot.inlineBuffer.contents, value.payload.inline_value.data,
                            value.payload.inline_value.size);
            } else {
                slot.inlineBuffer = nil;
            }
            slot.defaultSampler = nil;
        } else {
            const auto *resource = providerBindingResource(value);
            slot.resource = resource ? *resource : VernonRuntimeProviderResourceReference{};
            slot.inlineStorage.clear();
            slot.inlineBuffer = nil;
            slot.stride = value.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER ? value.payload.buffer.stride : 0;
            if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER &&
                (value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE)) {
                if (!slot.defaultSampler) {
                    MTLSamplerDescriptor *descriptor = [[MTLSamplerDescriptor alloc] init];
                    descriptor.minFilter = MTLSamplerMinMagFilterLinear;
                    descriptor.magFilter = MTLSamplerMinMagFilterLinear;
                    descriptor.mipFilter = MTLSamplerMipFilterLinear;
                    descriptor.supportArgumentBuffers = YES;
                    slot.defaultSampler = [metalDevice(adapter).device newSamplerStateWithDescriptor:descriptor];
                    if (!slot.defaultSampler)
                        return fail(adapter, "Metal default sampler creation failed", VERNON_STATUS_INTERNAL_ERROR);
                }
            } else {
                slot.defaultSampler = nil;
            }
        }
    }
    auto encodedArgumentBuffers = bindings.argumentBuffers;
    const VernonStatus encodeStatus = encodeArgumentBuffers(adapter, updatedSlots, encodedArgumentBuffers);
    if (encodeStatus != VERNON_STATUS_OK)
        return encodeStatus;
    bindings.slots = std::move(updatedSlots);
    bindings.argumentBuffers = std::move(encodedArgumentBuffers);
    return VERNON_STATUS_OK;
}

VernonStatus createBindingSet(void *data, const VernonRuntimeProviderBindingSetDescriptor *descriptor,
                              VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    PreparedLayout *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !layout)
        return fail(adapter, "Metal adapter received an invalid binding-set descriptor");
    try {
        auto bindings = std::make_unique<PreparedBindingSet>();
        bindings->layout = layout;
        bindings->slots.resize(layout->entries.size());
        bindings->slotIndices = layout->slotIndices;
        for (size_t index = 0; index < layout->entries.size(); ++index)
            bindings->slots[index].layout = layout->entries[index];
        const VernonStatus argumentStatus = createArgumentBuffers(adapter, *layout, bindings->argumentBuffers, false);
        if (argumentStatus != VERNON_STATUS_OK)
            return argumentStatus;
        const VernonStatus status = updateBindingsImpl(adapter, *bindings, descriptor->values, descriptor->value_count);
        if (status != VERNON_STATUS_OK)
            return status;
        *output = toHandle(bindings.release());
        adapter.bindingCreations.fetch_add(1, std::memory_order_relaxed);
        return VERNON_STATUS_OK;
    } catch (const std::bad_alloc &) {
        return fail(adapter, "Metal binding-set preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    }
}

VernonStatus updateBindingSet(void *data, VernonRuntimeProviderObject handle,
                              const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    PreparedBindingSet *bindings = fromHandle<PreparedBindingSet>(handle);
    if (!bindings)
        return fail(adapter, "Metal adapter received an invalid binding set");
    try {
        std::lock_guard<std::mutex> guard(bindings->mutex);
        return updateBindingsImpl(adapter, *bindings, values, valueCount);
    } catch (const std::bad_alloc &) {
        return fail(adapter, "Metal binding update ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    }
}

VernonStatus encodeDispatch(void *data, VernonRuntimeProviderObject commandEncoder,
                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    PreparedPipeline *pipeline = descriptor ? fromHandle<PreparedPipeline>(descriptor->pipeline) : nullptr;
    PreparedBindingSet *bindings = descriptor ? fromHandle<PreparedBindingSet>(descriptor->bindings) : nullptr;
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !pipeline || pipeline->graphics ||
        !pipeline->compute || (!bindings && descriptor->bindings.value) || descriptor->group_count[0] == 0 ||
        (bindings && bindings->layout != pipeline->layout) ||
        (!bindings && pipeline->layout && !pipeline->layout->entries.empty()) || descriptor->group_count[1] == 0 ||
        descriptor->group_count[2] == 0 || descriptor->push_constant_size != pipeline->pushConstantSize ||
        (descriptor->push_constant_size && !descriptor->push_constants))
        return fail(adapter, "Metal adapter received an invalid dispatch");
    const uint64_t native = nativeCommandEncoder(adapter, commandEncoder);
    if (!native || commandEncoderRendering(adapter, commandEncoder))
        return fail(adapter, "Metal dispatch command encoder is invalid");
    if (!retainCommandObjects(adapter, commandEncoder, *pipeline, bindings))
        return fail(adapter, "Metal dispatch could not retain provider objects", VERNON_STATUS_INTERNAL_ERROR);
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)(reinterpret_cast<void *>(static_cast<uintptr_t>(native)));
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder)
        return fail(adapter, "Metal compute command encoder creation failed", VERNON_STATUS_INTERNAL_ERROR);
    [encoder setComputePipelineState:pipeline->compute];
    std::unique_lock<std::mutex> guard;
    if (bindings)
        guard = std::unique_lock<std::mutex>(bindings->mutex);
    if (bindings) {
        if (bindings->argumentBuffers.empty()) {
            const VernonStatus argumentStatus =
                createArgumentBuffers(adapter, *bindings->layout, bindings->argumentBuffers, true);
            if (argumentStatus != VERNON_STATUS_OK) {
                [encoder endEncoding];
                return argumentStatus;
            }
            const VernonStatus encodeStatus =
                encodeArgumentBuffers(adapter, bindings->slots, bindings->argumentBuffers);
            if (encodeStatus != VERNON_STATUS_OK) {
                [encoder endEncoding];
                return encodeStatus;
            }
        }
        for (const auto &slot : bindings->slots) {
            if ((slot.layout.stage_mask & VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE) == 0)
                continue;
            if (!isArgumentResource(slot.layout)) {
                if (packedUniformBytes(slot.layout.kind, slot.layout.interface_kind))
                    [encoder setBytes:slot.inlineStorage.data()
                               length:slot.inlineStorage.size()
                              atIndex:slot.layout.binding];
                continue;
            }
            if (packedUniformBytes(slot.layout.kind, slot.layout.interface_kind)) {
                [encoder useResource:slot.inlineBuffer usage:MTLResourceUsageRead];
                continue;
            }
            if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER && slot.defaultSampler)
                continue;
            if (!retainCommandResource(adapter, commandEncoder, slot.resource)) {
                [encoder endEncoding];
                return fail(adapter, "Metal dispatch could not retain a bound resource", VERNON_STATUS_INTERNAL_ERROR);
            }
            const uint64_t resolved = resolveRhiResource(adapter, slot.resource);
            if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
                id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(reinterpret_cast<void *>(resolved));
                [encoder useResource:buffer usage:resourceUsage(slot.layout)];
            } else if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                       slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE) {
                id<MTLTexture> texture = (__bridge id<MTLTexture>)(reinterpret_cast<void *>(resolved));
                [encoder useResource:texture usage:resourceUsage(slot.layout)];
            }
            if ((slot.layout.access & 2u) && !recordCommandWriteResource(adapter, commandEncoder, slot.resource)) {
                [encoder endEncoding];
                return fail(adapter, "Metal dispatch could not track a writable resource",
                            VERNON_STATUS_INTERNAL_ERROR);
            }
        }
        for (const auto &argumentBuffer : bindings->argumentBuffers)
            if (argumentBuffer.stage == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE)
                [encoder setBuffer:argumentBuffer.buffer offset:0 atIndex:argumentBuffer.index];
    }
    if (descriptor->push_constant_size)
        [encoder setBytes:descriptor->push_constants length:descriptor->push_constant_size atIndex:15];
    [encoder dispatchThreadgroups:MTLSizeMake(descriptor->group_count[0], descriptor->group_count[1],
                                              descriptor->group_count[2])
            threadsPerThreadgroup:MTLSizeMake(pipeline->workgroup[0], pipeline->workgroup[1], pipeline->workgroup[2])];
    [encoder endEncoding];
    if (!recordProviderCommand(adapter, commandEncoder, false))
        return fail(adapter, "Metal dispatch command encoder state changed", VERNON_STATUS_INTERNAL_ERROR);
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
}

MTLLoadAction loadAction(VernonRhiLoadOperation operation) {
    switch (operation) {
    case VERNON_RHI_LOAD_CLEAR:
        return MTLLoadActionClear;
    case VERNON_RHI_LOAD_PRESERVE:
        return MTLLoadActionLoad;
    case VERNON_RHI_LOAD_DISCARD:
        return MTLLoadActionDontCare;
    }
    return MTLLoadActionDontCare;
}

MTLStoreAction storeAction(VernonRhiStoreOperation operation) {
    return operation == VERNON_RHI_STORE_PRESERVE ? MTLStoreActionStore : MTLStoreActionDontCare;
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

bool sameRenderScope(const rhi::metal::RenderingState &left, const rhi::metal::RenderingState &right) {
    if (left.colorCount != right.colorCount || left.hasDepth != right.hasDepth)
        return false;
    for (size_t index = 0; index < left.colors.size(); ++index)
        if (!(left.colors[index] == right.colors[index]))
            return false;
    return !left.hasDepth || left.depth == right.depth;
}

VernonStatus encodeDraw(void *data, VernonRuntimeProviderObject commandEncoder,
                        const VernonRuntimeProviderDrawDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    PreparedPipeline *pipeline = descriptor ? fromHandle<PreparedPipeline>(descriptor->pipeline) : nullptr;
    PreparedBindingSet *bindings = descriptor ? fromHandle<PreparedBindingSet>(descriptor->bindings) : nullptr;
    MTLPrimitiveType primitive{};
    if (!validCommonDrawDescriptor(descriptor) || !pipeline || !pipeline->graphics ||
        !pipeline->render || (!bindings && descriptor->bindings.value) ||
        (bindings && bindings->layout != pipeline->layout) ||
        (!bindings && pipeline->layout && !pipeline->layout->entries.empty()) ||
        !primitiveType(descriptor->topology, primitive) ||
        descriptor->color_attachment_count != pipeline->colorFormatCount ||
        (descriptor->depth_stencil_view.resource.value && !pipeline->depthStencil) ||
        (!descriptor->depth_stencil_view.resource.value && pipeline->depthStencil) ||
        descriptor->instance_count == 0 || (!descriptor->index_count && descriptor->vertex_count == 0))
        return fail(adapter, "Metal adapter received an invalid or unsupported draw");
    const uint64_t native = nativeCommandEncoder(adapter, commandEncoder);
    const int renderingClaim = claimCommandRendering(adapter, commandEncoder, vernon::rhi::CommandRenderingDynamic);
    if (!native || renderingClaim < 0)
        return fail(adapter, "Metal draw command encoder is invalid");
    rhi::metal::RenderingState requested;
    requested.colorCount = descriptor->color_attachment_count;
    requested.hasDepth = descriptor->depth_stencil_view.resource.value != 0;
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    std::array<uint64_t, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> renderTargets{};
    std::array<uint64_t, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> renderResources{};
    std::array<bool, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> seenLocations{};
    for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
        const auto &source = descriptor->color_attachments[index];
        if (source.location >= pipeline->colorFormatCount || seenLocations[source.location] ||
            !retainCommandResource(adapter, commandEncoder, source.view))
            return fail(adapter, "Metal draw contains an invalid color attachment");
        seenLocations[source.location] = true;
        const uint64_t resolved = resolveRhiResource(adapter, source.view);
        id<MTLTexture> texture = (__bridge id<MTLTexture>)(reinterpret_cast<void *>(resolved));
        if (!texture || texture.pixelFormat != pipeline->colorFormats[source.location] ||
            texture.sampleCount != pipeline->sampleCount)
            return fail(adapter, "Metal draw contains an incompatible color attachment");
        VernonRhiLoadOperation load = static_cast<VernonRhiLoadOperation>(source.load_operation);
        VernonRhiStoreOperation store = static_cast<VernonRhiStoreOperation>(source.store_operation);
        float clear[4];
        std::copy(std::begin(source.clear_color), std::end(source.clear_color), clear);
        if (commandEncoderHasRenderingDescriptor(adapter, commandEncoder) &&
            !commandColorOperations(adapter, commandEncoder, index, load, store, clear))
            return fail(adapter, "Metal draw color operations do not match the active render scope");
        auto *attachment = pass.colorAttachments[source.location];
        attachment.texture = texture;
        attachment.loadAction = loadAction(load);
        attachment.storeAction = storeAction(store);
        attachment.clearColor = MTLClearColorMake(clear[0], clear[1], clear[2], clear[3]);
        requested.colors[source.location] = {source.view.identity,
                                             source.view.resource.value,
                                             source.location,
                                             static_cast<uint32_t>(texture.pixelFormat),
                                             static_cast<uint32_t>(load),
                                             static_cast<uint32_t>(store),
                                             static_cast<uint32_t>(texture.sampleCount)};
        requested.colorTextures[source.location] = texture;
        renderTargets[source.location] = resolved;
        renderResources[source.location] = resolved;
    }
    uint64_t depthTarget = 0;
    uint64_t depthResource = 0;
    if (requested.hasDepth) {
        if (!retainCommandResource(adapter, commandEncoder, descriptor->depth_stencil_view))
            return fail(adapter, "Metal draw contains an invalid depth attachment");
        const uint64_t resolved = resolveRhiResource(adapter, descriptor->depth_stencil_view);
        id<MTLTexture> texture = (__bridge id<MTLTexture>)(reinterpret_cast<void *>(resolved));
        if (!texture || texture.pixelFormat != pipeline->depthFormat || texture.sampleCount != pipeline->sampleCount)
            return fail(adapter, "Metal draw contains an incompatible depth attachment");
        VernonRhiLoadOperation depthLoad = static_cast<VernonRhiLoadOperation>(descriptor->depth_load_operation);
        VernonRhiStoreOperation depthStore = static_cast<VernonRhiStoreOperation>(descriptor->depth_store_operation);
        VernonRhiLoadOperation stencilLoad =
            static_cast<VernonRhiLoadOperation>(descriptor->stencil_load_operation);
        VernonRhiStoreOperation stencilStore =
            static_cast<VernonRhiStoreOperation>(descriptor->stencil_store_operation);
        float clearDepth = descriptor->clear_depth;
        uint32_t clearStencil = descriptor->clear_stencil;
        if (commandEncoderHasRenderingDescriptor(adapter, commandEncoder) &&
            !commandDepthOperations(adapter, commandEncoder, depthLoad, depthStore, stencilLoad, stencilStore,
                                    clearDepth, clearStencil))
            return fail(adapter, "Metal draw depth operations do not match the active render scope");
        pass.depthAttachment.texture = texture;
        pass.depthAttachment.loadAction = loadAction(depthLoad);
        pass.depthAttachment.storeAction = storeAction(depthStore);
        pass.depthAttachment.clearDepth = clearDepth;
        if (texture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8) {
            pass.stencilAttachment.texture = texture;
            pass.stencilAttachment.loadAction = loadAction(stencilLoad);
            pass.stencilAttachment.storeAction = storeAction(stencilStore);
            pass.stencilAttachment.clearStencil = clearStencil;
        } else if (stencilLoad != VERNON_RHI_LOAD_DISCARD || stencilStore != VERNON_RHI_STORE_DISCARD ||
                   clearStencil != 0) {
            return fail(adapter, "Metal draw requests stencil operations for a depth-only attachment");
        }
        requested.depth = {descriptor->depth_stencil_view.identity,
                           descriptor->depth_stencil_view.resource.value,
                           0,
                           static_cast<uint32_t>(texture.pixelFormat),
                           static_cast<uint32_t>(depthLoad),
                           static_cast<uint32_t>(depthStore),
                           static_cast<uint32_t>(texture.sampleCount)};
        requested.depthStencilTexture = texture;
        depthTarget = depthResource = resolved;
    }
    if (!retainCommandObjects(adapter, commandEncoder, *pipeline, bindings))
        return fail(adapter, "Metal draw could not retain provider objects", VERNON_STATUS_INTERNAL_ERROR);
    id<MTLRenderCommandEncoder> encoder = nil;
    if (renderingClaim != 0) {
        id<MTLCommandBuffer> commandBuffer =
            (__bridge id<MTLCommandBuffer>)(reinterpret_cast<void *>(static_cast<uintptr_t>(native)));
        encoder = [commandBuffer renderCommandEncoderWithDescriptor:pass];
        if (!encoder)
            return fail(adapter, "Metal render command encoder creation failed", VERNON_STATUS_INTERNAL_ERROR);
        auto rendering = std::make_unique<rhi::metal::RenderingState>(requested);
        rendering->encoder = encoder;
        const uint64_t renderingValue = reinterpret_cast<uintptr_t>(rendering.get());
        if (!setCommandRenderingTargets(adapter, commandEncoder, renderTargets.data(), renderResources.data(),
                                        renderTargets.size(), depthTarget, depthResource) ||
            commandRenderingObject(adapter, commandEncoder, renderingValue) != renderingValue) {
            [encoder endEncoding];
            return fail(adapter, "Metal command render-target registration failed", VERNON_STATUS_INTERNAL_ERROR);
        }
        rhi::metal::registerRenderingState(native, rendering.get());
        [[maybe_unused]] auto *ownedRendering = rendering.release();
    } else {
        const uint64_t renderingValue = commandRenderingObject(adapter, commandEncoder, 1);
        auto *rendering = reinterpret_cast<rhi::metal::RenderingState *>(static_cast<uintptr_t>(renderingValue));
        if (!rendering || !sameRenderScope(*rendering, requested))
            return fail(adapter, "Metal draw attachments do not match the active render scope");
        encoder = rendering->encoder;
        if (!encoder)
            return fail(adapter, "Metal render command encoder is unavailable", VERNON_STATUS_INTERNAL_ERROR);
    }
    [encoder setRenderPipelineState:pipeline->render];
    if (pipeline->depthStencil)
        [encoder setDepthStencilState:pipeline->depthStencil];
    [encoder setCullMode:pipeline->cullMode];
    [encoder setFrontFacingWinding:pipeline->frontFace];
    [encoder setStencilReferenceValue:descriptor->stencil_reference];
    adapter.lastStencilReference.store(descriptor->stencil_reference, std::memory_order_relaxed);
    if (pipeline->depthBiasEnabled)
        [encoder setDepthBias:pipeline->depthBiasConstant slopeScale:pipeline->depthBiasSlope clamp:0.0f];
    std::unique_lock<std::mutex> bindingsGuard;
    if (bindings)
        bindingsGuard = std::unique_lock<std::mutex>(bindings->mutex);
    if (bindings) {
        if (bindings->argumentBuffers.empty()) {
            const VernonStatus argumentStatus =
                createArgumentBuffers(adapter, *bindings->layout, bindings->argumentBuffers, true);
            if (argumentStatus != VERNON_STATUS_OK)
                return argumentStatus;
            const VernonStatus encodeStatus =
                encodeArgumentBuffers(adapter, bindings->slots, bindings->argumentBuffers);
            if (encodeStatus != VERNON_STATUS_OK)
                return encodeStatus;
        }
        for (const auto &slot : bindings->slots) {
            const uint32_t index = slot.layout.binding;
            const bool vertexStage = (slot.layout.stage_mask & VERNON_RUNTIME_PROVIDER_STAGE_VERTEX) != 0;
            const bool fragmentStage = (slot.layout.stage_mask & VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT) != 0;
            if (!isArgumentResource(slot.layout)) {
                if (packedUniformBytes(slot.layout.kind, slot.layout.interface_kind)) {
                    if (vertexStage)
                        [encoder setVertexBytes:slot.inlineStorage.data()
                                         length:slot.inlineStorage.size()
                                        atIndex:index];
                    if (fragmentStage)
                        [encoder setFragmentBytes:slot.inlineStorage.data()
                                           length:slot.inlineStorage.size()
                                          atIndex:index];
                    continue;
                }
                if (slot.layout.kind != VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER)
                    continue;
                if (!retainCommandResource(adapter, commandEncoder, slot.resource))
                    return fail(adapter, "Metal draw could not retain a vertex buffer", VERNON_STATUS_INTERNAL_ERROR);
                const uint64_t resolved = resolveRhiResource(adapter, slot.resource);
                id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(reinterpret_cast<void *>(resolved));
                [encoder setVertexBuffer:buffer offset:slot.resource.offset atIndex:index];
                continue;
            }
            if (packedUniformBytes(slot.layout.kind, slot.layout.interface_kind)) {
                [encoder useResource:slot.inlineBuffer
                               usage:MTLResourceUsageRead
                              stages:renderStages(slot.layout.stage_mask)];
                continue;
            }
            if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER && slot.defaultSampler)
                continue;
            if (!retainCommandResource(adapter, commandEncoder, slot.resource))
                return fail(adapter, "Metal draw could not retain a bound resource", VERNON_STATUS_INTERNAL_ERROR);
            const uint64_t resolved = resolveRhiResource(adapter, slot.resource);
            if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
                id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(reinterpret_cast<void *>(resolved));
                [encoder useResource:buffer
                               usage:resourceUsage(slot.layout)
                              stages:renderStages(slot.layout.stage_mask)];
            } else if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                       slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE) {
                id<MTLTexture> texture = (__bridge id<MTLTexture>)(reinterpret_cast<void *>(resolved));
                [encoder useResource:texture
                               usage:resourceUsage(slot.layout)
                              stages:renderStages(slot.layout.stage_mask)];
            }
            if ((slot.layout.access & 2u) && !recordCommandWriteResource(adapter, commandEncoder, slot.resource))
                return fail(adapter, "Metal draw could not track a writable resource", VERNON_STATUS_INTERNAL_ERROR);
        }
        for (const auto &argumentBuffer : bindings->argumentBuffers) {
            if (argumentBuffer.stage == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX)
                [encoder setVertexBuffer:argumentBuffer.buffer offset:0 atIndex:argumentBuffer.index];
            else if (argumentBuffer.stage == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT)
                [encoder setFragmentBuffer:argumentBuffer.buffer offset:0 atIndex:argumentBuffer.index];
        }
    }
    [encoder setViewport:{static_cast<double>(descriptor->viewport[0]), static_cast<double>(descriptor->viewport[1]),
                          static_cast<double>(descriptor->viewport[2]), static_cast<double>(descriptor->viewport[3]),
                          0.0, 1.0}];
    [encoder setScissorRect:{descriptor->scissor[0], descriptor->scissor[1], descriptor->scissor[2],
                             descriptor->scissor[3]}];
    adapter.lastDrawIndexed.store(descriptor->index_count != 0, std::memory_order_relaxed);
    if (descriptor->index_count) {
        if (descriptor->index_type != 0 || !retainCommandResource(adapter, commandEncoder, descriptor->index_buffer))
            return fail(adapter, "Metal draw contains an unsupported index buffer");
        const uint64_t resolved = resolveRhiResource(adapter, descriptor->index_buffer);
        id<MTLBuffer> indexBuffer = (__bridge id<MTLBuffer>)(reinterpret_cast<void *>(resolved));
        if (!indexBuffer)
            return fail(adapter, "Metal draw contains a stale index buffer");
        [encoder drawIndexedPrimitives:primitive
                            indexCount:descriptor->index_count
                             indexType:MTLIndexTypeUInt32
                           indexBuffer:indexBuffer
                     indexBufferOffset:descriptor->index_buffer.offset
                         instanceCount:descriptor->instance_count
                            baseVertex:descriptor->first_vertex
                          baseInstance:descriptor->first_instance];
    } else {
        [encoder drawPrimitives:primitive
                    vertexStart:descriptor->first_vertex
                    vertexCount:descriptor->vertex_count
                  instanceCount:descriptor->instance_count
                   baseInstance:descriptor->first_instance];
    }
    if (!recordProviderCommand(adapter, commandEncoder, true))
        return fail(adapter, "Metal draw command encoder state changed", VERNON_STATUS_INTERNAL_ERROR);
    return VERNON_STATUS_OK;
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
    if (pipeline && pipeline->references.fetch_sub(1, std::memory_order_acq_rel) == 1)
        delete pipeline;
}
void destroyPipeline(void *, VernonRuntimeProviderObject handle) {
    releaseCommandPipeline(fromHandle<PreparedPipeline>(handle), 0);
}

} // namespace

void initializeMetalProvider(VernonRuntimeRhiAdapter &adapter) {
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
    auto *deviceState = static_cast<rhi::metal::DeviceState *>(rhi::deviceState(device, backend));
    if (!deviceState)
        return nullptr;
    auto state = std::unique_ptr<rhi_adapter::MetalAdapterState>(new (std::nothrow) rhi_adapter::MetalAdapterState());
    auto adapter = std::unique_ptr<VernonRuntimeRhiAdapter>(new (std::nothrow) VernonRuntimeRhiAdapter());
    if (!state || !adapter)
        return nullptr;
    state->device = deviceState;
    adapter->rhiBackend = backend;
    if (!adapter->backend.adopt(state.get(), &rhi_adapter::backendOps))
        return nullptr;
    (void)state.release();
    rhi_adapter::initializeMetalProvider(*adapter);
    return adapter.release();
}

#endif
