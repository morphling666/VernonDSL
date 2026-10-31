#include "runtime/program_execution/program_tensor_copy.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace vernon::runtime::program_execution {
namespace {

struct View {
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;
    VernonTensorView tensor{};

    View(std::vector<uint64_t> extents, std::vector<int64_t> byteStrides, size_t byteSize, uint32_t elementBytes,
         size_t byteOffset = 0)
        : shape(std::move(extents)), strides(std::move(byteStrides)) {
        tensor.struct_size = sizeof(VernonTensorView);
        tensor.storage = VERNON_TENSOR_HOST;
        tensor.rank = static_cast<uint32_t>(shape.size());
        tensor.shape = shape.empty() ? nullptr : shape.data();
        tensor.byte_strides = strides.empty() ? nullptr : strides.data();
        tensor.byte_offset = byteOffset;
        tensor.byte_size = byteSize;
        tensor.element_layout.byte_size = elementBytes;
        tensor.element_layout.alignment = elementBytes;
    }
};

std::vector<ProgramTensorCopyRegion> plan(const View &source, const View &destination) {
    std::vector<ProgramTensorCopyRegion> regions;
    std::string error;
    EXPECT_TRUE(planProgramTensorCopy(source.tensor, destination.tensor, regions, error)) << error;
    return regions;
}

void expectRegions(const std::vector<ProgramTensorCopyRegion> &regions,
                   const std::vector<ProgramTensorCopyRegion> &expected) {
    ASSERT_EQ(regions.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(regions[i].sourceOffset, expected[i].sourceOffset) << i;
        EXPECT_EQ(regions[i].destinationOffset, expected[i].destinationOffset) << i;
        EXPECT_EQ(regions[i].size, expected[i].size) << i;
    }
}

TEST(ProgramTensorCopy, DenseViewsCopyAsOneRegion) {
    const View source({3, 1000000}, {8000000, 8}, 24000000, 8);
    const View destination({3, 1000000}, {8000000, 8}, 24000016, 8, 16);
    expectRegions(plan(source, destination), {{0, 16, 24000000}});
}

TEST(ProgramTensorCopy, PaddedRowsCopyRowByRow) {
    const View source({3, 4}, {16, 4}, 48, 4);
    const View destination({3, 4}, {32, 4}, 96, 4);
    expectRegions(plan(source, destination), {{0, 0, 16}, {16, 32, 16}, {32, 64, 16}});
}

TEST(ProgramTensorCopy, StridedInnerAxisCopiesElementByElement) {
    const View source({2, 3}, {24, 8}, 48, 4);
    const View destination({2, 3}, {12, 4}, 24, 4);
    expectRegions(plan(source, destination), {{0, 0, 4}, {8, 4, 4}, {16, 8, 4}, {24, 12, 4}, {32, 16, 4}, {40, 20, 4}});
}

TEST(ProgramTensorCopy, ReversedRowsKeepTheirOrder) {
    const View source({3, 2}, {-8, 4}, 24, 4, 16);
    const View destination({3, 2}, {8, 4}, 24, 4);
    expectRegions(plan(source, destination), {{16, 0, 8}, {8, 8, 8}, {0, 16, 8}});
}

TEST(ProgramTensorCopy, ScalarCopiesOneElement) {
    const View source({}, {}, 8, 8);
    const View destination({}, {}, 8, 8);
    expectRegions(plan(source, destination), {{0, 0, 8}});
}

TEST(ProgramTensorCopy, RejectsBlocksPastTheBacking) {
    const View source({2, 4}, {16, 4}, 32, 4);
    const View destination({2, 4}, {16, 4}, 28, 4);
    std::vector<ProgramTensorCopyRegion> regions;
    std::string error;
    EXPECT_FALSE(planProgramTensorCopy(source.tensor, destination.tensor, regions, error));
    EXPECT_EQ(error, "Program tensor copy exceeds Storage backing");
}

} // namespace
} // namespace vernon::runtime::program_execution
