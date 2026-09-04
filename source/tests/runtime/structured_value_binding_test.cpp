#include "VernonRuntime.hpp"
#include "runtime/pipeline_manifest.h"
#include "runtime/runtime_state.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using namespace vernon::runtime;

VernonProgramExecutable structuredPipeline(std::vector<uint64_t> outerShape = {}) {
    VernonProgramExecutable pipeline;
    Parameter parameter;
    parameter.slot = 3;
    parameter.name = "value";
    parameter.kind = "tensor";
    parameter.access = "read";
    parameter.shape = std::move(outerShape);
    parameter.elementLayout.logicalType = "!vernon.struct<\"Payload\">";
    parameter.elementLayout.structName = "Payload";
    parameter.elementLayout.layoutHash = "structured-binding-test-layout";
    parameter.elementLayout.byteSize = 32;
    parameter.elementLayout.alignment = 8;

    ValueLeaf id("i32", 1, 0);
    id.path = {{{"id"}, 0}};
    ValueLeaf weight("f64", 1, 8);
    weight.path = {{{"weight"}, 0}};
    ValueLeaf matrix("f32", 4, 16);
    matrix.path = {{{"matrix"}, 0}};
    matrix.shape = {2, 2};
    parameter.elementLayout.leaves = {std::move(id), std::move(weight), std::move(matrix)};
    parameter.elementLayout.abiLeaves = {
        {VERNON_DATA_I32, 1, 0},
        {VERNON_DATA_F64, 1, 8},
        {VERNON_DATA_F32, 4, 16},
    };
    rebuildValueLayoutPathViews(parameter.elementLayout);
    pipeline.variant.parameters.push_back(std::move(parameter));
    return pipeline;
}

template <typename T> T load(const uint8_t *data, size_t offset) {
    T result;
    std::memcpy(&result, data + offset, sizeof(T));
    return result;
}

TEST(StructuredValueBinding, PacksFieldTreeUsingReflectedOffsetsAndShape) {
    VernonProgramExecutable pipeline = structuredPipeline();
    const std::array<float, 4> matrix{1.0f, 2.0f, 3.0f, 4.0f};
    ProgramInvocationBuilder builder(&pipeline);
    builder.bindValue("value",
                      fields(field("id", int32_t{7}), field("weight", 2.5), field("matrix", shaped({2, 2}, matrix))));

    const VernonProgramSubmitDescriptor invocation = builder.invocation();
    ASSERT_EQ(invocation.argument_count, 1u);
    const VernonTensorView &tensor = invocation.arguments[0].tensor;
    ASSERT_EQ(tensor.byte_size, 32u);
    const auto *bytes = static_cast<const uint8_t *>(tensor.host_data);
    EXPECT_EQ(load<int32_t>(bytes, 0), 7);
    EXPECT_EQ(load<double>(bytes, 8), 2.5);
    EXPECT_EQ(load<float>(bytes, 16), 1.0f);
    EXPECT_EQ(load<float>(bytes, 28), 4.0f);
    EXPECT_EQ(bytes[4], 0u);
}

TEST(StructuredValueBinding, PacksNestedFieldAndTupleElementPaths) {
    VernonProgramExecutable pipeline = structuredPipeline();
    auto &layout = pipeline.variant.parameters[0].elementLayout;
    layout.byteSize = 8;
    layout.alignment = 4;
    ValueLeaf first("i32", 1, 0);
    first.path = {{{"meta"}, 0}, {std::nullopt, 0}};
    ValueLeaf second("f32", 1, 4);
    second.path = {{{"meta"}, 0}, {std::nullopt, 1}};
    layout.leaves = {std::move(first), std::move(second)};
    layout.abiLeaves = {{VERNON_DATA_I32, 1, 0}, {VERNON_DATA_F32, 1, 4}};
    rebuildValueLayoutPathViews(layout);

    ProgramInvocationBuilder builder(&pipeline);
    builder.bindValue("value", fields(field("meta", elements(int32_t{9}, 1.25f))));
    const VernonProgramSubmitDescriptor invocation = builder.invocation();
    const auto *bytes = static_cast<const uint8_t *>(invocation.arguments[0].tensor.host_data);
    EXPECT_EQ(load<int32_t>(bytes, 0), 9);
    EXPECT_EQ(load<float>(bytes, 4), 1.25f);
}

TEST(StructuredValueBinding, PacksTensorFromStructureOfArrays) {
    VernonProgramExecutable pipeline = structuredPipeline({2, 2});
    const std::array<int32_t, 4> ids{1, 2, 3, 4};
    const std::array<double, 4> weights{0.5, 1.5, 2.5, 3.5};
    const std::array<float, 16> matrices{
        0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33,
    };
    ProgramInvocationBuilder builder(&pipeline);
    builder.bindTensor(
        "value", {2, 2},
        fields(field("id", ids), field("weight", weights), field("matrix", shaped({2, 2, 2, 2}, matrices))));

    const VernonProgramSubmitDescriptor invocation = builder.invocation();
    const VernonTensorView &tensor = invocation.arguments[0].tensor;
    ASSERT_EQ(tensor.byte_size, 128u);
    EXPECT_EQ(tensor.byte_strides[0], 64);
    EXPECT_EQ(tensor.byte_strides[1], 32);
    const auto *bytes = static_cast<const uint8_t *>(tensor.host_data);
    EXPECT_EQ(load<int32_t>(bytes, 64), 3);
    EXPECT_EQ(load<double>(bytes, 96 + 8), 3.5);
    EXPECT_EQ(load<float>(bytes, 96 + 28), 33.0f);
}

TEST(StructuredValueBinding, PacksTensorFromLogicalElementCallback) {
    VernonProgramExecutable pipeline = structuredPipeline({2, 2});
    ProgramInvocationBuilder builder(&pipeline);
    builder.bindTensor(
        "value", {2, 2}, [](ProgramInvocationBuilder::ElementWriter &element, const std::vector<uint64_t> &index) {
            const int32_t linear = static_cast<int32_t>(index[0] * 2 + index[1]);
            const std::array<float, 4> matrix{static_cast<float>(linear), static_cast<float>(linear + 1),
                                              static_cast<float>(linear + 2), static_cast<float>(linear + 3)};
            element.field("id", linear)
                .field("weight", static_cast<double>(linear) + 0.25)
                .field("matrix", shaped({2, 2}, matrix));
        });

    const VernonProgramSubmitDescriptor invocation = builder.invocation();
    const auto *bytes = static_cast<const uint8_t *>(invocation.arguments[0].tensor.host_data);
    EXPECT_EQ(load<int32_t>(bytes, 3 * 32), 3);
    EXPECT_EQ(load<double>(bytes, 3 * 32 + 8), 3.25);
    EXPECT_EQ(load<float>(bytes, 3 * 32 + 28), 6.0f);
}

TEST(StructuredValueBinding, DiagnosesFieldDtypeShapeCompletenessAndDuplicateBinding) {
    VernonProgramExecutable pipeline = structuredPipeline({2, 2});
    const std::array<int32_t, 4> ids{};
    const std::array<double, 4> weights{};
    const std::array<float, 16> matrices{};

    ProgramInvocationBuilder missing(&pipeline);
    EXPECT_THROW(missing.bindTensor("value", {2, 2}, fields(field("id", ids), field("weight", weights))),
                 std::invalid_argument);

    ProgramInvocationBuilder dtype(&pipeline);
    EXPECT_THROW(dtype.bindTensor("value", {2, 2},
                                  fields(field("id", weights), field("weight", weights),
                                         field("matrix", shaped({2, 2, 2, 2}, matrices)))),
                 std::invalid_argument);

    ProgramInvocationBuilder shape(&pipeline);
    EXPECT_THROW(
        shape.bindTensor("value", {2, 2},
                         fields(field("id", ids), field("weight", weights), field("matrix", shaped({4, 4}, matrices)))),
        std::invalid_argument);

    ProgramInvocationBuilder duplicate(&pipeline);
    const StructuredValue valid =
        fields(field("id", ids), field("weight", weights), field("matrix", shaped({2, 2, 2, 2}, matrices)));
    duplicate.bindTensor("value", {2, 2}, valid);
    EXPECT_THROW(duplicate.bindTensor("value", {2, 2}, valid), std::invalid_argument);
}

} // namespace
