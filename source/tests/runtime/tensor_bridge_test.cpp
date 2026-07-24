#include "runtime/tensor_bridge.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <limits>

namespace {

TEST(TensorBridgeTest, KeepsContiguousHostTensorAsSingleLogicalRange) {
    const std::array<float, 6> values{1, 2, 3, 4, 5, 6};
    const std::array<uint64_t, 2> shape{2, 3};
    const std::array<int64_t, 2> strides{3 * sizeof(float), sizeof(float)};
    VernonTensorView tensor{sizeof(VernonTensorView),
                            VERNON_TENSOR_HOST,
                            {values.data()},
                            VERNON_DATA_F32,
                            VERNON_ACCESS_READ,
                            2,
                            shape.data(),
                            strides.data(),
                            0,
                            sizeof(values)};

    EXPECT_TRUE(vernon::runtime::isRowMajorContiguous(tensor));
    const auto packed = vernon::runtime::packTensorRowMajor(tensor);
    ASSERT_TRUE(packed);
    ASSERT_EQ(packed->size(), sizeof(values));
    EXPECT_EQ(std::memcmp(packed->data(), values.data(), sizeof(values)), 0);
}

TEST(TensorBridgeTest, PacksMaximalContiguousInnerRows) {
    const std::array<float, 8> padded{1, 2, 3, -1, 4, 5, 6, -1};
    const std::array<float, 6> expected{1, 2, 3, 4, 5, 6};
    const std::array<uint64_t, 2> shape{2, 3};
    const std::array<int64_t, 2> strides{4 * sizeof(float), sizeof(float)};
    VernonTensorView tensor{sizeof(VernonTensorView),
                            VERNON_TENSOR_HOST,
                            {padded.data()},
                            VERNON_DATA_F32,
                            VERNON_ACCESS_READ,
                            2,
                            shape.data(),
                            strides.data(),
                            0,
                            sizeof(padded)};

    EXPECT_FALSE(vernon::runtime::isRowMajorContiguous(tensor));
    const auto packed = vernon::runtime::packTensorRowMajor(tensor);
    ASSERT_TRUE(packed);
    ASSERT_EQ(packed->size(), sizeof(expected));
    EXPECT_EQ(std::memcmp(packed->data(), expected.data(), sizeof(expected)), 0);
}

TEST(TensorBridgeTest, PacksFullyStridedTensorByLogicalIndex) {
    const std::array<float, 9> source{1, -1, 2, -1, -1, -1, 3, -1, 4};
    const std::array<float, 4> expected{1, 2, 3, 4};
    const std::array<uint64_t, 2> shape{2, 2};
    const std::array<int64_t, 2> strides{6 * sizeof(float), 2 * sizeof(float)};
    VernonTensorView tensor{sizeof(VernonTensorView),
                            VERNON_TENSOR_HOST,
                            {source.data()},
                            VERNON_DATA_F32,
                            VERNON_ACCESS_READ,
                            2,
                            shape.data(),
                            strides.data(),
                            0,
                            sizeof(source)};

    const auto packed = vernon::runtime::packTensorRowMajor(tensor);
    ASSERT_TRUE(packed);
    ASSERT_EQ(packed->size(), sizeof(expected));
    EXPECT_EQ(std::memcmp(packed->data(), expected.data(), sizeof(expected)), 0);
}

TEST(TensorBridgeTest, RejectsLogicalByteSizeOverflow) {
    const std::array<uint64_t, 1> shape{static_cast<uint64_t>(std::numeric_limits<size_t>::max())};
    VernonTensorView tensor{};
    tensor.dtype = VERNON_DATA_F64;
    tensor.rank = 1;
    tensor.shape = shape.data();

    EXPECT_FALSE(vernon::runtime::tensorLogicalByteSize(tensor));
}

TEST(TensorBridgeTest, PacksNegativeStrideFromLogicalFirstElement) {
    const std::array<float, 4> source{1, 2, 3, 4};
    const std::array<float, 4> expected{4, 3, 2, 1};
    const std::array<uint64_t, 1> shape{4};
    const std::array<int64_t, 1> strides{-static_cast<int64_t>(sizeof(float))};
    VernonTensorView tensor{sizeof(VernonTensorView),
                            VERNON_TENSOR_HOST,
                            {source.data()},
                            VERNON_DATA_F32,
                            VERNON_ACCESS_READ,
                            1,
                            shape.data(),
                            strides.data(),
                            3 * sizeof(float),
                            sizeof(source)};

    EXPECT_TRUE(vernon::runtime::tensorFitsAllocation(tensor));
    EXPECT_FALSE(vernon::runtime::isRowMajorContiguous(tensor));
    const auto packed = vernon::runtime::packTensorRowMajor(tensor);
    ASSERT_TRUE(packed);
    EXPECT_EQ(std::memcmp(packed->data(), expected.data(), sizeof(expected)), 0);
}

} // namespace
