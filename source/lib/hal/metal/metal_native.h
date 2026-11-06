#ifndef VERNON_HAL_METAL_METAL_NATIVE_H
#define VERNON_HAL_METAL_METAL_NATIVE_H

// The native objects of a Metal device, which only Objective-C++ translation units, compiled with ARC, see.

#if !defined(__OBJC__) || !__has_feature(objc_arc)
#error "hal/metal/metal_native.h is Objective-C++ compiled with -fobjc-arc only"
#endif

#include "VernonHal.h"
#include "VernonTextureTypes.h"
#include "hal/metal/metal_state.h"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace vernon::hal::metal {

struct Native {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    // Committed and not yet known complete, oldest first.
    std::deque<std::pair<uint64_t, id<MTLCommandBuffer>>> inFlight;
    // Objects released once the command buffer of the value committed when they were retired completes.
    std::deque<std::pair<uint64_t, id>> retired;
    // Copies a surface's image into its layer's drawable, whose format differs from every surface format.
    id<MTLRenderPipelineState> present;
    // Expands a counted device draw stream into indirect arguments, zeroing the instance count past its count.
    id<MTLComputePipelineState> streams;
    // Bound where a pass binds no sampler, as on every backend: linear, repeating and sampling the base level.
    id<MTLSamplerState> defaultSampler;
    bool filters32BitFloat{};
};

// Commits `commands` and returns its value. Runs under the state's mutex.
uint64_t commit(DeviceState &state, id<MTLCommandBuffer> commands);
// Keeps `object` until the work committed so far completes. Runs under the state's mutex.
void retain(DeviceState &state, id object);

std::string describe(NSError *error);

// MTLPixelFormatInvalid where Metal cannot create the format.
MTLPixelFormat pixelFormat(VernonTextureFormat format);

struct Buffer {
    id<MTLBuffer> buffer;
    uint64_t size{};
    VernonHalMemory memory{VERNON_HAL_MEMORY_DEVICE};
};

struct Image {
    id<MTLTexture> texture;
    VernonTextureFormat format{};
    VernonTextureDimension dimension{VERNON_TEXTURE_2D};
    uint32_t mipLevels{};
    uint32_t layers{};
};

// What passes bind of an image view: a texture view of `owner` over the view's subresources.
struct View {
    Image *owner{};
    id<MTLTexture> texture;
    VernonTextureFormat format{};
    VernonTextureDimension dimension{VERNON_TEXTURE_2D};
    uint32_t usage{};
    uint32_t aspects{};
};

struct Sampler {
    id<MTLSamplerState> sampler;
};

} // namespace vernon::hal::metal

#endif
