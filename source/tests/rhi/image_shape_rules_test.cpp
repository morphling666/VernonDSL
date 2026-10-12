#include "rhi/image_descriptor_validation.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace {

using vernon::rhi::compatibleViewDimension;
using vernon::rhi::validImageShape;

constexpr VernonRhiImageDescriptor shape(VernonRhiImageDimension dimension, uint32_t width, uint32_t height,
                                         uint32_t depth, uint32_t layers) {
    VernonRhiImageDescriptor descriptor{};
    descriptor.dimension = dimension;
    descriptor.width = width;
    descriptor.height = height;
    descriptor.depth = depth;
    descriptor.array_layers = layers;
    return descriptor;
}

static_assert(validImageShape(shape(VERNON_RHI_IMAGE_2D, 4, 2, 1, 1)));
static_assert(!validImageShape(shape(VERNON_RHI_IMAGE_2D, 4, 2, 1, 2)));
static_assert(validImageShape(shape(VERNON_RHI_IMAGE_2D_ARRAY, 4, 2, 1, 1)));
static_assert(validImageShape(shape(VERNON_RHI_IMAGE_2D_ARRAY, 4, 2, 1, 7)));
static_assert(!validImageShape(shape(VERNON_RHI_IMAGE_2D_ARRAY, 4, 2, 2, 7)));
static_assert(validImageShape(shape(VERNON_RHI_IMAGE_3D, 4, 2, 3, 1)));
static_assert(!validImageShape(shape(VERNON_RHI_IMAGE_3D, 4, 2, 3, 2)));
static_assert(validImageShape(shape(VERNON_RHI_IMAGE_CUBE, 4, 4, 1, 6)));
static_assert(!validImageShape(shape(VERNON_RHI_IMAGE_CUBE, 4, 4, 1, 12)));
static_assert(!validImageShape(shape(VERNON_RHI_IMAGE_CUBE, 4, 2, 1, 6)));
static_assert(validImageShape(shape(VERNON_RHI_IMAGE_CUBE_ARRAY, 4, 4, 1, 12)));
static_assert(!validImageShape(shape(VERNON_RHI_IMAGE_CUBE_ARRAY, 4, 4, 1, 9)));
static_assert(!validImageShape(shape(VERNON_RHI_IMAGE_CUBE_ARRAY, 4, 2, 1, 6)));
static_assert(!validImageShape(shape(VERNON_RHI_IMAGE_2D_ARRAY, 4, 2, 1, 0)));
static_assert(!validImageShape(shape(static_cast<VernonRhiImageDimension>(99), 4, 4, 1, 1)));

static_assert(compatibleViewDimension(VERNON_RHI_IMAGE_2D_ARRAY, VERNON_RHI_IMAGE_2D, 3, 1));
static_assert(!compatibleViewDimension(VERNON_RHI_IMAGE_2D_ARRAY, VERNON_RHI_IMAGE_2D, 0, 2));
static_assert(compatibleViewDimension(VERNON_RHI_IMAGE_CUBE, VERNON_RHI_IMAGE_2D_ARRAY, 0, 6));
static_assert(compatibleViewDimension(VERNON_RHI_IMAGE_CUBE_ARRAY, VERNON_RHI_IMAGE_CUBE, 6, 6));
static_assert(!compatibleViewDimension(VERNON_RHI_IMAGE_CUBE_ARRAY, VERNON_RHI_IMAGE_CUBE, 3, 6));
static_assert(compatibleViewDimension(VERNON_RHI_IMAGE_CUBE, VERNON_RHI_IMAGE_CUBE_ARRAY, 0, 6));
static_assert(!compatibleViewDimension(VERNON_RHI_IMAGE_CUBE_ARRAY, VERNON_RHI_IMAGE_CUBE_ARRAY, 0, 9));
static_assert(!compatibleViewDimension(VERNON_RHI_IMAGE_2D_ARRAY, VERNON_RHI_IMAGE_CUBE, 0, 6));
static_assert(!compatibleViewDimension(VERNON_RHI_IMAGE_3D, VERNON_RHI_IMAGE_2D, 0, 1));
static_assert(!compatibleViewDimension(VERNON_RHI_IMAGE_3D, VERNON_RHI_IMAGE_2D_ARRAY, 0, 1));
static_assert(compatibleViewDimension(VERNON_RHI_IMAGE_3D, VERNON_RHI_IMAGE_3D, 0, 1));
static_assert(!compatibleViewDimension(VERNON_RHI_IMAGE_2D, VERNON_RHI_IMAGE_3D, 0, 1));

TEST(RhiImageShapeRules, LayeredDimensionsKeepCubeFacesWhole) {
    for (uint32_t layers = 1; layers <= 24; ++layers) {
        EXPECT_EQ(validImageShape(shape(VERNON_RHI_IMAGE_CUBE_ARRAY, 8, 8, 1, layers)), layers % 6 == 0) << layers;
        for (uint32_t base = 0; base < layers; ++base)
            EXPECT_EQ(compatibleViewDimension(VERNON_RHI_IMAGE_CUBE_ARRAY, VERNON_RHI_IMAGE_CUBE, base, 6),
                      base % 6 == 0)
                << base;
    }
}

} // namespace
