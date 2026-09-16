#include "graphics_variant_key.h"

#include <algorithm>
#include <functional>

namespace vernon::runtime {
namespace {

bool rasterizationEqual(const VernonRasterizationState &left, const VernonRasterizationState &right) {
    return left.cull_mode == right.cull_mode && left.front_face == right.front_face &&
           left.depth_clamp == right.depth_clamp && left.depth_bias_enabled == right.depth_bias_enabled &&
           left.depth_bias_constant == right.depth_bias_constant && left.depth_bias_slope == right.depth_bias_slope;
}

bool stencilFaceEqual(const VernonStencilFaceState &left, const VernonStencilFaceState &right) {
    return left.stencil_fail == right.stencil_fail && left.depth_fail == right.depth_fail && left.pass == right.pass &&
           left.compare == right.compare;
}

bool depthStencilEqual(const VernonDepthStencilState &left, const VernonDepthStencilState &right) {
    return left.depth_test == right.depth_test && left.depth_write == right.depth_write &&
           left.depth_compare == right.depth_compare && left.stencil_test == right.stencil_test &&
           stencilFaceEqual(left.front, right.front) && stencilFaceEqual(left.back, right.back) &&
           left.stencil_read_mask == right.stencil_read_mask && left.stencil_write_mask == right.stencil_write_mask;
}

bool blendEqual(const VernonColorBlendState &left, const VernonColorBlendState &right) {
    return left.blend_enabled == right.blend_enabled && left.source_color_factor == right.source_color_factor &&
           left.destination_color_factor == right.destination_color_factor &&
           left.color_operation == right.color_operation && left.source_alpha_factor == right.source_alpha_factor &&
           left.destination_alpha_factor == right.destination_alpha_factor &&
           left.alpha_operation == right.alpha_operation && left.write_mask == right.write_mask;
}

} // namespace

bool graphicsVariantKeysEqual(const GraphicsVariantKey &left, const GraphicsVariantKey &right) {
    return left.topology == right.topology && left.colorFormats == right.colorFormats &&
           left.depthStencilFormat == right.depthStencilFormat && left.sampleCount == right.sampleCount &&
           left.vertexStrides == right.vertexStrides && rasterizationEqual(left.rasterization, right.rasterization) &&
           depthStencilEqual(left.depthStencil, right.depthStencil) &&
           left.colorBlends.size() == right.colorBlends.size() &&
           std::equal(left.colorBlends.begin(), left.colorBlends.end(), right.colorBlends.begin(), blendEqual);
}

size_t graphicsVariantKeyHash(const GraphicsVariantKey &key) {
    size_t result = 0xcbf29ce484222325ull;
    const auto combine = [&](auto value) {
        result ^= std::hash<decltype(value)>{}(value) + 0x9e3779b97f4a7c15ull + (result << 6) + (result >> 2);
    };
    const auto combineFace = [&](const VernonStencilFaceState &face) {
        combine(face.stencil_fail);
        combine(face.depth_fail);
        combine(face.pass);
        combine(face.compare);
    };
    combine(key.topology);
    for (uint32_t format : key.colorFormats)
        combine(format);
    combine(key.colorFormats.size());
    combine(key.depthStencilFormat);
    combine(key.sampleCount);
    for (uint32_t stride : key.vertexStrides)
        combine(stride);
    combine(key.vertexStrides.size());
    combine(key.rasterization.cull_mode);
    combine(key.rasterization.front_face);
    combine(key.rasterization.depth_clamp);
    combine(key.rasterization.depth_bias_enabled);
    combine(key.rasterization.depth_bias_constant);
    combine(key.rasterization.depth_bias_slope);
    combine(key.depthStencil.depth_test);
    combine(key.depthStencil.depth_write);
    combine(key.depthStencil.depth_compare);
    combine(key.depthStencil.stencil_test);
    combineFace(key.depthStencil.front);
    combineFace(key.depthStencil.back);
    combine(key.depthStencil.stencil_read_mask);
    combine(key.depthStencil.stencil_write_mask);
    for (const auto &blend : key.colorBlends) {
        combine(blend.blend_enabled);
        combine(blend.source_color_factor);
        combine(blend.destination_color_factor);
        combine(blend.color_operation);
        combine(blend.source_alpha_factor);
        combine(blend.destination_alpha_factor);
        combine(blend.alpha_operation);
        combine(blend.write_mask);
    }
    combine(key.colorBlends.size());
    return result;
}

} // namespace vernon::runtime
