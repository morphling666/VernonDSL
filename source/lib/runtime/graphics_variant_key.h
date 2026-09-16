#ifndef VERNON_RUNTIME_GRAPHICS_VARIANT_KEY_H
#define VERNON_RUNTIME_GRAPHICS_VARIANT_KEY_H

#include "VernonGraphicsState.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vernon::runtime {

struct GraphicsVariantKey {
    uint32_t topology{};
    std::vector<uint32_t> colorFormats;
    uint32_t depthStencilFormat{};
    uint32_t sampleCount{1};
    std::vector<uint32_t> vertexStrides;
    VernonRasterizationState rasterization{};
    VernonDepthStencilState depthStencil{};
    std::vector<VernonColorBlendState> colorBlends;
};

bool graphicsVariantKeysEqual(const GraphicsVariantKey &left, const GraphicsVariantKey &right);
size_t graphicsVariantKeyHash(const GraphicsVariantKey &key);

struct GraphicsVariantKeyHash {
    size_t operator()(const GraphicsVariantKey &key) const { return graphicsVariantKeyHash(key); }
};

struct GraphicsVariantKeyEqual {
    bool operator()(const GraphicsVariantKey &left, const GraphicsVariantKey &right) const {
        return graphicsVariantKeysEqual(left, right);
    }
};

} // namespace vernon::runtime

#endif
