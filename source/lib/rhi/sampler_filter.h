#ifndef VERNON_RHI_SAMPLER_FILTER_H
#define VERNON_RHI_SAMPLER_FILTER_H

#include "VernonRHI.h"

#include <cmath>

namespace vernon::rhi {

struct SamplerFilter {
    bool minLinear{};
    bool magLinear{};
    bool mipLinear{};
    float maxAnisotropy{1.0f};
};

inline bool decodeSamplerFilter(const VernonRhiSamplerDescriptor &descriptor, SamplerFilter &filter) {
    if (descriptor.min_filter > VERNON_RHI_FILTER_NEAREST_MIPMAP_LINEAR ||
        descriptor.mag_filter > VERNON_RHI_FILTER_LINEAR || descriptor.mip_filter > VERNON_RHI_FILTER_LINEAR ||
        descriptor.address_u > VERNON_RHI_ADDRESS_MIRRORED_REPEAT ||
        descriptor.address_v > VERNON_RHI_ADDRESS_MIRRORED_REPEAT ||
        descriptor.address_w > VERNON_RHI_ADDRESS_MIRRORED_REPEAT || !std::isfinite(descriptor.max_anisotropy) ||
        descriptor.max_anisotropy < 0.0f || descriptor.max_anisotropy > 16.0f)
        return false;

    filter.magLinear = descriptor.mag_filter == VERNON_RHI_FILTER_LINEAR;
    switch (descriptor.min_filter) {
    case VERNON_RHI_FILTER_NEAREST:
        filter.minLinear = false;
        filter.mipLinear = descriptor.mip_filter == VERNON_RHI_FILTER_LINEAR;
        break;
    case VERNON_RHI_FILTER_LINEAR:
        filter.minLinear = true;
        filter.mipLinear = descriptor.mip_filter == VERNON_RHI_FILTER_LINEAR;
        break;
    case VERNON_RHI_FILTER_NEAREST_MIPMAP_NEAREST:
        filter.minLinear = false;
        filter.mipLinear = false;
        break;
    case VERNON_RHI_FILTER_LINEAR_MIPMAP_LINEAR:
        filter.minLinear = true;
        filter.mipLinear = true;
        break;
    case VERNON_RHI_FILTER_LINEAR_MIPMAP_NEAREST:
        filter.minLinear = true;
        filter.mipLinear = false;
        break;
    case VERNON_RHI_FILTER_NEAREST_MIPMAP_LINEAR:
        filter.minLinear = false;
        filter.mipLinear = true;
        break;
    default:
        return false;
    }
    filter.maxAnisotropy = descriptor.max_anisotropy > 1.0f ? descriptor.max_anisotropy : 1.0f;
    return true;
}

} // namespace vernon::rhi

#endif
