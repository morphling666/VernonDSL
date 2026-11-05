#ifndef VERNON_HAL_NATIVE_IMAGES_H
#define VERNON_HAL_NATIVE_IMAGES_H

// The image views and samplers a HAL device creates itself, outside the RHI's registry, and the image rules every
// backend checks.

#include "VernonHal.h"
#include "VernonResult.hpp"
#include "VernonRuntimeProvider.h"
#include "hal/native_buffers.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vernon::runtime {
struct GraphicsRenderScope;
}

namespace vernon::hal {

inline constexpr size_t kPackedDepthStencilBytes = 8;

// What a view reference names. `native` is the backend's view object, which lives as long as the entry.
struct NativeView {
    VernonImageDescriptor image{};
    VernonImageViewDescriptor view{};
    // The image's serial, which no later image reuses.
    uint64_t image_serial{};
    void *native{};
};

Serials<NativeView> &nativeViews();
// Each entry is the backend's sampler object.
Serials<void> &nativeSamplers();
uint64_t nextImageSerial();

// Describes a view reference of any HAL device; false when `reference` names no live HAL view.
bool describeNativeView(const VernonRuntimeProviderResourceReference &reference,
                        VernonRuntimeProviderImageDescription &description);

// Surface images whose contents are undefined, from their acquire until a submitted pass writes them.
void markUndefined(uint64_t image);
void forgetUndefined(const uint64_t *images, size_t count);
bool anyUndefined(const uint64_t *images, size_t count);

bool validImage(const VernonImageDescriptor &image);
bool validView(const VernonImageDescriptor &image, const VernonImageViewDescriptor &view);
// The bytes of `region`'s host data: depth and stencil together pack 8 bytes a texel. Empty when the region's data
// layout does not match the aspects it transfers.
std::optional<size_t> regionBytes(const VernonImageDescriptor &image, const VernonHalImageRegion &region);

// The images, by serial, whose contents a render scope reads and the attachments it writes, after the checks the
// planner leaves to the device: live views of `identity`'s device, single-mip attachments covering their format's
// aspects, and sampled reads.
struct ScopeImages {
    std::vector<const NativeView *> colors;
    const NativeView *depth{};
    std::vector<const NativeView *> reads;
    std::vector<uint64_t> readImages;
    std::vector<uint64_t> attachmentImages;
    bool stencil{};
};
Result<ScopeImages, std::string> checkRenderScope(const runtime::GraphicsRenderScope &scope, uint64_t identity);

} // namespace vernon::hal

#endif
