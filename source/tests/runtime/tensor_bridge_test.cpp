#include "runtime/pipeline_manifest.h"
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
                            vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32),
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
                            vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32),
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
                            vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32),
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
    tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F64);
    tensor.rank = 1;
    tensor.shape = shape.data();

    EXPECT_FALSE(vernon::runtime::tensorLogicalByteSize(tensor));
}

TEST(TensorBridgeTest, RejectsBitIncompatibleTransportRepresentation) {
    vernon::runtime::TransportNode scalar{vernon::runtime::TransportNodeKind::Scalar, "i32", 0, 4, 4};
    vernon::runtime::TransportNode array{
        vernon::runtime::TransportNodeKind::Array, "", 0, 4, 4, {1}, {4}, {std::move(scalar)}};
    EXPECT_FALSE(
        vernon::runtime::compileElementStreamCopyPlan(vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32), {1}, array));
}

TEST(TensorBridgeTest, PacksNegativeStrideFromLogicalFirstElement) {
    const std::array<float, 4> source{1, 2, 3, 4};
    const std::array<float, 4> expected{4, 3, 2, 1};
    const std::array<uint64_t, 1> shape{4};
    const std::array<int64_t, 1> strides{-static_cast<int64_t>(sizeof(float))};
    VernonTensorView tensor{sizeof(VernonTensorView),
                            VERNON_TENSOR_HOST,
                            {source.data()},
                            vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32),
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

TEST(TensorBridgeTest, AcceptsInjectiveNegativeContiguousStride) {
    const std::array<float, 4> values{};
    const std::array<uint64_t, 1> shape{4};
    const std::array<int64_t, 1> strides{-static_cast<int64_t>(sizeof(float))};
    VernonTensorView tensor{sizeof(VernonTensorView),
                            VERNON_TENSOR_HOST,
                            {values.data()},
                            vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32),
                            VERNON_ACCESS_WRITE,
                            1,
                            shape.data(),
                            strides.data(),
                            3 * sizeof(float),
                            sizeof(values)};

    EXPECT_TRUE(vernon::runtime::tensorFitsAllocation(tensor));
    EXPECT_TRUE(vernon::runtime::tensorByteLayoutInjective(tensor));
}

TEST(TensorBridgeTest, RejectsSubElementWritableStride) {
    const std::array<uint8_t, 8> allocation{};
    const std::array<uint64_t, 1> shape{2};
    const std::array<int64_t, 1> strides{1};
    VernonTensorView tensor{sizeof(VernonTensorView),
                            VERNON_TENSOR_HOST,
                            {allocation.data()},
                            vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32),
                            VERNON_ACCESS_WRITE,
                            1,
                            shape.data(),
                            strides.data(),
                            0,
                            allocation.size()};

    EXPECT_TRUE(vernon::runtime::tensorFitsAllocation(tensor));
    EXPECT_FALSE(vernon::runtime::tensorByteLayoutInjective(tensor));
}

TEST(TensorBridgeTest, RejectsSignedStrideBoundsOverflowAndOutOfBounds) {
    const std::array<uint8_t, 16> allocation{};
    const std::array<uint64_t, 2> emptyShape{0, std::numeric_limits<uint64_t>::max()};
    const std::array<int64_t, 2> emptyStrides{std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
    VernonTensorView empty{sizeof(VernonTensorView), VERNON_TENSOR_HOST,
                           {allocation.data()},      vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32),
                           VERNON_ACCESS_WRITE,      2,
                           emptyShape.data(),        emptyStrides.data(),
                           allocation.size(),        allocation.size()};
    EXPECT_EQ(vernon::runtime::tensorElementCount(empty), 0);
    EXPECT_TRUE(vernon::runtime::tensorFitsAllocation(empty));

    const std::array<uint64_t, 1> overflowingShape{std::numeric_limits<uint64_t>::max()};
    const std::array<int64_t, 1> minimumStride{std::numeric_limits<int64_t>::min()};
    VernonTensorView overflowing{sizeof(VernonTensorView),
                                 VERNON_TENSOR_HOST,
                                 {allocation.data()},
                                 vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32),
                                 VERNON_ACCESS_WRITE,
                                 1,
                                 overflowingShape.data(),
                                 minimumStride.data(),
                                 0,
                                 allocation.size()};
    size_t before = 0;
    size_t after = 0;
    EXPECT_FALSE(vernon::runtime::tensorRelativeByteBounds(overflowing, before, after));

    const std::array<uint64_t, 1> shape{2};
    const std::array<int64_t, 1> negativeStride{-static_cast<int64_t>(sizeof(float))};
    VernonTensorView outOfBounds{sizeof(VernonTensorView),
                                 VERNON_TENSOR_HOST,
                                 {allocation.data()},
                                 vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32),
                                 VERNON_ACCESS_WRITE,
                                 1,
                                 shape.data(),
                                 negativeStride.data(),
                                 0,
                                 allocation.size()};
    EXPECT_FALSE(vernon::runtime::tensorFitsAllocation(outOfBounds));
    EXPECT_FALSE(vernon::runtime::tensorByteLayoutInjective(outOfBounds));
}

TEST(TensorBridgeTest, AllowsReadAliasesAndRejectsWritableByteOverlap) {
    const std::array<uint8_t, 24> allocation{};
    const std::array<uint64_t, 1> shape{2};
    const std::array<int64_t, 1> strides{8};
    VernonTensorView left{sizeof(VernonTensorView),
                          VERNON_TENSOR_HOST,
                          {allocation.data()},
                          vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32),
                          VERNON_ACCESS_READ,
                          1,
                          shape.data(),
                          strides.data(),
                          0,
                          allocation.size()};
    VernonTensorView right = left;

    EXPECT_FALSE(vernon::runtime::tensorViewsHaveWritableOverlap(left, right));

    right.access = VERNON_ACCESS_WRITE;
    EXPECT_TRUE(vernon::runtime::tensorViewsHaveWritableOverlap(left, right));

    right.byte_offset = sizeof(float);
    EXPECT_FALSE(vernon::runtime::tensorViewsHaveWritableOverlap(left, right));

    right.byte_offset = 2;
    EXPECT_TRUE(vernon::runtime::tensorViewsHaveWritableOverlap(left, right));
}

TEST(TensorBridgeTest, ProvesLargeInterleavedViewsDisjointWithoutEnumeration) {
    constexpr uint64_t count = 2'000'000;
    const std::array<uint64_t, 1> shape{count};
    const std::array<int64_t, 1> strides{2 * static_cast<int64_t>(sizeof(float))};
    std::array<uint8_t, 1> identity{};
    VernonTensorView even{};
    even.struct_size = sizeof(VernonTensorView);
    even.storage = VERNON_TENSOR_RHI_RESOURCE;
    even.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    even.access = VERNON_ACCESS_WRITE;
    even.rank = 1;
    even.shape = shape.data();
    even.byte_strides = strides.data();
    even.byte_size = static_cast<size_t>(count * 2 * sizeof(float));
    even.resource = {reinterpret_cast<uintptr_t>(identity.data()), {1}, 0, count * 2 * sizeof(float)};
    VernonTensorView odd = even;
    odd.resource.offset = sizeof(float);

    EXPECT_EQ(vernon::runtime::tensorViewsPhysicalOverlap(even, odd), vernon::runtime::TensorPhysicalOverlap::Disjoint);
}

TEST(TensorBridgeTest, ResolvesHostAndRhiPhysicalStorageRanges) {
    const std::array<uint8_t, 32> allocation{};
    const std::array<uint64_t, 1> shape{2};
    const std::array<int64_t, 1> strides{8};
    VernonTensorView left{sizeof(VernonTensorView),
                          VERNON_TENSOR_HOST,
                          {allocation.data()},
                          vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32),
                          VERNON_ACCESS_WRITE,
                          1,
                          shape.data(),
                          strides.data(),
                          0,
                          allocation.size()};
    VernonTensorView right = left;
    right.host_data = allocation.data() + 4;
    right.byte_size = allocation.size() - 4;
    EXPECT_EQ(vernon::runtime::tensorViewsPhysicalOverlap(left, right),
              vernon::runtime::TensorPhysicalOverlap::Disjoint);

    right.host_data = allocation.data() + 2;
    right.byte_size = allocation.size() - 2;
    EXPECT_EQ(vernon::runtime::tensorViewsPhysicalOverlap(left, right),
              vernon::runtime::TensorPhysicalOverlap::Overlapping);

    left.storage = VERNON_TENSOR_RHI_RESOURCE;
    left.resource = {7, {11}, 100, allocation.size()};
    right = left;
    right.resource.offset = 104;
    EXPECT_EQ(vernon::runtime::tensorViewsPhysicalOverlap(left, right),
              vernon::runtime::TensorPhysicalOverlap::Disjoint);
    right.resource.offset = 102;
    EXPECT_EQ(vernon::runtime::tensorViewsPhysicalOverlap(left, right),
              vernon::runtime::TensorPhysicalOverlap::Overlapping);
    right.resource.identity = 8;
    EXPECT_EQ(vernon::runtime::tensorViewsPhysicalOverlap(left, right),
              vernon::runtime::TensorPhysicalOverlap::Disjoint);

    right = left;
    right.resource.identity = 0;
    EXPECT_EQ(vernon::runtime::tensorViewsPhysicalOverlap(left, right),
              vernon::runtime::TensorPhysicalOverlap::Unknown);
    EXPECT_TRUE(vernon::runtime::tensorViewsHaveWritableOverlap(left, right));
    right = left;
    right.resource.offset = std::numeric_limits<uint64_t>::max() - right.resource.size + 1;
    EXPECT_EQ(vernon::runtime::tensorViewsPhysicalOverlap(left, right),
              vernon::runtime::TensorPhysicalOverlap::Unknown);
    left.access = VERNON_ACCESS_READ;
    right.access = VERNON_ACCESS_READ;
    EXPECT_FALSE(vernon::runtime::tensorViewsHaveWritableOverlap(left, right));
}

TEST(TensorBridgeTest, PacksNonSquareMatrixIntoReflectedColumnMajorLayout) {
    const std::array<float, 6> source{1, 2, 3, 4, 5, 6};
    const std::array<uint64_t, 2> shape{2, 3};
    const std::array<int64_t, 2> strides{3 * sizeof(float), sizeof(float)};
    VernonTensorView tensor{sizeof(VernonTensorView),
                            VERNON_TENSOR_HOST,
                            {source.data()},
                            vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32),
                            VERNON_ACCESS_READ,
                            2,
                            shape.data(),
                            strides.data(),
                            0,
                            sizeof(source)};
    vernon::runtime::TensorCopyPlan layout{sizeof(float), {2, 3}, {sizeof(float), 16}, 48, {{0, 0, sizeof(float)}}};

    const auto packed = vernon::runtime::packTensor(tensor, layout);
    ASSERT_TRUE(packed);
    ASSERT_EQ(packed->size(), 48u);
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < 3; ++column) {
            float value = 0;
            std::memcpy(&value, packed->data() + row * sizeof(float) + column * 16, sizeof(value));
            EXPECT_EQ(value, source[row * 3 + column]);
        }
    for (size_t offset : {size_t{8}, size_t{12}, size_t{24}, size_t{28}, size_t{40}, size_t{44}})
        EXPECT_EQ((*packed)[offset], 0);
}

TEST(TensorBridgeTest, PacksRankThreeTensorWithReflectedPadding) {
    const std::array<float, 8> source{1, 2, 3, 4, 5, 6, 7, 8};
    const std::array<uint64_t, 3> shape{2, 2, 2};
    const std::array<int64_t, 3> strides{4 * sizeof(float), 2 * sizeof(float), sizeof(float)};
    VernonTensorView tensor{sizeof(VernonTensorView),
                            VERNON_TENSOR_HOST,
                            {source.data()},
                            vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32),
                            VERNON_ACCESS_READ,
                            3,
                            shape.data(),
                            strides.data(),
                            0,
                            sizeof(source)};
    vernon::runtime::TensorCopyPlan layout{sizeof(float), {2, 2, 2}, {32, 16, 4}, 56, {{0, 0, sizeof(float)}}};

    const auto packed = vernon::runtime::packTensor(tensor, layout);
    ASSERT_TRUE(packed);
    for (size_t linear = 0; linear < source.size(); ++linear) {
        const size_t outer = linear / 4;
        const size_t middle = (linear / 2) % 2;
        const size_t inner = linear % 2;
        float value = 0;
        std::memcpy(&value, packed->data() + outer * 32 + middle * 16 + inner * 4, sizeof(value));
        EXPECT_EQ(value, source[linear]);
    }
}

TEST(TensorBridgeTest, PacksCompleteAggregateElementsFromPaddedRecords) {
    struct Record {
        float position;
        uint32_t padding;
        int32_t id;
        uint32_t recordPadding;
    };
    const std::array<Record, 2> source{{{1.5f, 0xaaaaaaaa, 7, 0xbbbbbbbb}, {2.5f, 0xcccccccc, 9, 0xdddddddd}}};
    static constexpr VernonValueLeafView leaves[] = {
        {VERNON_DATA_F32, 1, 0},
        {VERNON_DATA_I32, 1, 8},
    };
    static constexpr char hash[] = "mixed-test-layout";
    const VernonValueLayoutView layout{sizeof(VernonValueLayoutView), 12,     4,
                                       {hash, sizeof(hash) - 1},      leaves, std::size(leaves)};
    const std::array<uint64_t, 1> shape{2};
    const std::array<int64_t, 1> strides{sizeof(Record)};
    VernonTensorView tensor{sizeof(VernonTensorView),
                            VERNON_TENSOR_HOST,
                            {source.data()},
                            layout,
                            VERNON_ACCESS_READ,
                            1,
                            shape.data(),
                            strides.data(),
                            0,
                            sizeof(source)};

    EXPECT_TRUE(vernon::runtime::tensorFitsAllocation(tensor));
    EXPECT_FALSE(vernon::runtime::isRowMajorContiguous(tensor));
    const auto packed = vernon::runtime::packTensorRowMajor(tensor);
    ASSERT_TRUE(packed);
    ASSERT_EQ(packed->size(), 24u);
    EXPECT_EQ(std::memcmp(packed->data(), source.data(), 12), 0);
    EXPECT_EQ(std::memcmp(packed->data() + 12, source.data() + 1, 12), 0);
}

} // namespace
