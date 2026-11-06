#include "hal/metal/metal_state.h"
#include "hal/metal/metal_native.h"

#import <Foundation/Foundation.h>
#import <TargetConditionals.h>

#include <algorithm>
#include <string>

namespace vernon::hal::metal {
namespace {

// The device's own kernels. A draw record has the layout of MTLDrawIndexedPrimitivesIndirectArguments.
constexpr const char *kBuiltins = R"(
#include <metal_stdlib>
using namespace metal;

struct Record {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int baseVertex;
    uint firstInstance;
};

kernel void vernon_expand_stream(device const Record *records [[buffer(0)]], device const uint *count [[buffer(1)]],
                                 device Record *arguments [[buffer(2)]], constant uint &total [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
    if (index >= total)
        return;
    Record record = records[index];
    if (index >= count[0])
        record.instanceCount = 0;
    arguments[index] = record;
}

struct Copied {
    float4 position [[position]];
};

vertex Copied vernon_present_vertex(uint vertex_id [[vertex_id]]) {
    const float2 corner = float2((vertex_id << 1) & 2, vertex_id & 2);
    Copied result;
    result.position = float4(corner * 2.0 - 1.0, 0.0, 1.0);
    return result;
}

fragment float4 vernon_present_fragment(Copied copied [[stage_in]], texture2d<float> image [[texture(0)]]) {
    const uint2 last = uint2(image.get_width() - 1, image.get_height() - 1);
    return image.read(min(uint2(copied.position.xy), last));
}
)";

void retire(DeviceState &state) {
    Native &native = *state.native;
    while (!native.inFlight.empty()) {
        id<MTLCommandBuffer> commands = native.inFlight.front().second;
        const MTLCommandBufferStatus status = commands.status;
        if (status != MTLCommandBufferStatusCompleted && status != MTLCommandBufferStatusError)
            break;
        if (status == MTLCommandBufferStatusError && !state.lost) {
            state.lost = true;
            state.lostReason = "a Metal command buffer failed: " + describe(commands.error);
        }
        state.completedValue = native.inFlight.front().first;
        native.inFlight.pop_front();
    }
    if (native.inFlight.empty())
        state.completedValue = state.submittedValue;
    while (!native.retired.empty() && native.retired.front().first <= state.completedValue)
        native.retired.pop_front();
}

bool createBuiltins(Native &native, std::string &error) {
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    options.languageVersion = MTLLanguageVersion2_4;
    NSError *failure = nil;
    id<MTLLibrary> library = [native.device newLibraryWithSource:@(kBuiltins) options:options error:&failure];
    if (!library) {
        error = "cannot compile the device's kernels: " + describe(failure);
        return false;
    }
    id<MTLFunction> expand = [library newFunctionWithName:@"vernon_expand_stream"];
    native.streams = expand ? [native.device newComputePipelineStateWithFunction:expand error:&failure] : nil;
    if (!native.streams) {
        error = "cannot create the draw stream expansion: " + describe(failure);
        return false;
    }
    MTLRenderPipelineDescriptor *present = [[MTLRenderPipelineDescriptor alloc] init];
    present.vertexFunction = [library newFunctionWithName:@"vernon_present_vertex"];
    present.fragmentFunction = [library newFunctionWithName:@"vernon_present_fragment"];
    present.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    native.present = [native.device newRenderPipelineStateWithDescriptor:present error:&failure];
    if (!native.present) {
        error = "cannot create the surface copy: " + describe(failure);
        return false;
    }
    MTLSamplerDescriptor *sampler = [[MTLSamplerDescriptor alloc] init];
    sampler.minFilter = MTLSamplerMinMagFilterLinear;
    sampler.magFilter = MTLSamplerMinMagFilterLinear;
    sampler.mipFilter = MTLSamplerMipFilterNotMipmapped;
    sampler.sAddressMode = MTLSamplerAddressModeRepeat;
    sampler.tAddressMode = MTLSamplerAddressModeRepeat;
    sampler.rAddressMode = MTLSamplerAddressModeRepeat;
    sampler.lodMinClamp = 0;
    sampler.lodMaxClamp = 0;
    sampler.supportArgumentBuffers = YES;
    native.defaultSampler = [native.device newSamplerStateWithDescriptor:sampler];
    if (!native.defaultSampler) {
        error = "cannot create the default sampler";
        return false;
    }
    return true;
}

} // namespace

std::string describe(NSError *error) {
    if (!error)
        return "no reason given";
    NSString *description = error.localizedDescription;
    return description.length ? std::string(description.UTF8String) : std::string("no reason given");
}

MTLPixelFormat pixelFormat(VernonTextureFormat format) {
    switch (format) {
#define VERNON_FORMAT(name, NAME, texture, cls, bytes, dataFormat, dataType, caps, glInternal, glFormat, glType,       \
                      vulkan, dxgi, metal, ...)                                                                        \
    case VERNON_TEXTURE_##NAME:                                                                                        \
        return metal;
#include "VernonFormats.def"
    default:
        break;
    }
    return MTLPixelFormatInvalid;
}

uint64_t commit(DeviceState &state, id<MTLCommandBuffer> commands) {
    [commands commit];
    const uint64_t value = ++state.submittedValue;
    state.native->inFlight.emplace_back(value, commands);
    return value;
}

void retain(DeviceState &state, id object) {
    if (!object)
        return;
    retire(state);
    if (state.completedValue < state.submittedValue)
        state.native->retired.emplace_back(state.submittedValue, object);
}

DeviceState::DeviceState() : native(std::make_unique<Native>()) {}

DeviceState::~DeviceState() {
    if (!native)
        return;
    for (auto &[value, commands] : native->inFlight)
        [commands waitUntilCompleted];
    native->inFlight.clear();
    native->retired.clear();
}

bool DeviceState::initialize(uint32_t deviceIndex, uint32_t flags, std::string &error) {
    @autoreleasepool {
        if (flags & VERNON_HAL_DEVICE_FORCE_SOFTWARE) {
            error = "Metal has no software device";
            return false;
        }
        if (@available(macOS 12.0, iOS 15.0, *)) {
        } else {
            error = "Metal devices need macOS 12 or iOS 15, whose Metal compiles MSL 2.4";
            return false;
        }
        id<MTLDevice> device = nil;
#if TARGET_OS_OSX
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        if (deviceIndex >= devices.count) {
            error = "the Metal device index is out of range";
            return false;
        }
        device = devices[deviceIndex];
#else
        if (deviceIndex != 0) {
            error = "iOS has one Metal device, at index 0";
            return false;
        }
        device = MTLCreateSystemDefaultDevice();
#endif
        if (!device) {
            error = "the system has no Metal device";
            return false;
        }
        // The compiler declares descriptor sets as Tier 2 argument buffers.
        if (device.argumentBuffersSupport < MTLArgumentBuffersTier2) {
            error = "the Metal device lacks Tier 2 argument buffers";
            return false;
        }
        native->device = device;
        native->queue = [device newCommandQueue];
        if (!native->queue) {
            error = "cannot create a Metal command queue";
            return false;
        }
        if (@available(macOS 11.0, iOS 14.0, *))
            native->filters32BitFloat = device.supports32BitFloatFiltering;
        if (!createBuiltins(*native, error))
            return false;
        labels = (flags & (VERNON_HAL_DEVICE_LABELS | VERNON_HAL_DEVICE_VALIDATION)) != 0;
        return true;
    }
}

bool DeviceState::waitForValue(uint64_t value, std::string &error) {
    if (value > submittedValue) {
        error = "the value was not submitted yet";
        return false;
    }
    for (auto &[committed, commands] : native->inFlight) {
        if (committed > value)
            break;
        [commands waitUntilCompleted];
    }
    retire(*this);
    if (lost) {
        error = lostReason;
        return false;
    }
    return true;
}

bool DeviceState::completed(uint64_t value, bool &done, std::string &error) {
    retire(*this);
    done = value <= completedValue;
    if (lost && done) {
        error = lostReason;
        return false;
    }
    return true;
}

bool DeviceState::synchronize(std::string &error) { return waitForValue(submittedValue, error); }

} // namespace vernon::hal::metal
