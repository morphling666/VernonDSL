#include "runtime/autodiff/program_shape_resolver.h"
#include "runtime/autodiff/program_value_arena.h"
#include "runtime/program_manifest.h"
#include "runtime/shape_layout.h"

#include <gtest/gtest.h>

namespace vernon::runtime::shape {
namespace {

TEST(ShapeLayout, SeparatesDeclaredAndConcreteExtents) {
    const std::optional<DeclaredShape> declared = decodeReflectedShape({4, -1, 0});
    ASSERT_TRUE(declared);
    ASSERT_EQ(declared->size(), 3u);
    EXPECT_EQ((*declared)[0], Extent::fixed(4));
    EXPECT_TRUE((*declared)[1].isDynamic());
    EXPECT_EQ((*declared)[2], Extent::fixed(0));
    EXPECT_FALSE(isConcrete(*declared));
    EXPECT_FALSE(concrete(*declared));
    EXPECT_TRUE(matches(*declared, {4, 7, 0}));
}

TEST(ShapeLayout, ResolvesOneDynamicExtentFromStorageBytes) {
    const DeclaredShape declared{Extent::fixed(2), Extent::dynamic(), Extent::fixed(4)};
    ConcreteShape resolved;
    ASSERT_TRUE(resolveSingleDynamicExtent(declared, sizeof(float), 2 * 3 * 4 * sizeof(float), resolved));
    EXPECT_EQ(resolved, (ConcreteShape{2, 3, 4}));
    EXPECT_FALSE(resolveSingleDynamicExtent({Extent::dynamic(), Extent::dynamic()}, sizeof(float), 16 * sizeof(float),
                                            resolved));
}

TEST(ShapeLayout, ComputesCheckedRowMajorByteStrides) {
    ByteStrides strides;
    ASSERT_TRUE(rowMajorByteStrides({2, 3, 4}, sizeof(float), strides));
    EXPECT_EQ(strides, (ByteStrides{48, 16, 4}));
    size_t elements = 0;
    ASSERT_TRUE(checkedElementCount({2, 3, 4}, elements));
    EXPECT_EQ(elements, 24u);
}

TEST(ShapeLayout, LeafProjectionPreservesConcreteOwnerStrides) {
    ConcreteShape projectedShape;
    ByteStrides projectedStrides;
    ASSERT_TRUE(materializeLeafProjection({4, 4}, {16, 4}, {Extent::dynamic(), Extent::dynamic()}, sizeof(float),
                                          projectedShape, projectedStrides));
    EXPECT_EQ(projectedShape, (ConcreteShape{4, 4}));
    EXPECT_EQ(projectedStrides, (ByteStrides{16, 4}));

    ASSERT_TRUE(materializeLeafProjection({4, 4}, {32, 8}, {Extent::dynamic(), Extent::dynamic(), Extent::fixed(2)},
                                          sizeof(float), projectedShape, projectedStrides));
    EXPECT_EQ(projectedShape, (ConcreteShape{4, 4, 2}));
    EXPECT_EQ(projectedStrides, (ByteStrides{32, 8, 4}));

    ASSERT_TRUE(materializeCompactProjection({3}, {Extent::dynamic()}, 8, projectedShape, projectedStrides));
    EXPECT_EQ(projectedShape, (ConcreteShape{3}));
    EXPECT_EQ(projectedStrides, (ByteStrides{8}));
}

TEST(ProgramShapeResolver, PropagatesOnlyAcrossExplicitStorageAliases) {
    ExecutableProgram execution;
    execution.values = {
        ProgramValueSlot{0, "source", "", "", {2, 0}, {}, 0},
        ProgramValueSlot{1, "alias", "", "", {0, 3}, {}, 0},
        ProgramValueSlot{2, "unrelated", "", "", {2, 3}, {}, std::nullopt},
    };
    execution.storages = {ProgramStorageSlot{0, 0, 0}};
    std::vector<ad::ProgramHostValue> values(execution.values.size());
    values[0].concreteShape = ConcreteShape{2, 3};

    std::string error;
    ASSERT_TRUE(ad::resolveProgramShapes(execution, nullptr, values, error)) << error;
    EXPECT_EQ(values[1].concreteShape, values[0].concreteShape);
    EXPECT_EQ(values[2].concreteShape, std::optional<ConcreteShape>(ConcreteShape{2, 3}));
}

TEST(ProgramShapeResolver, RejectsConflictingConcreteBindings) {
    ExecutableProgram execution;
    execution.values = {ProgramValueSlot{0, "value", "", "", {2, 0}}};
    std::vector<ad::ProgramHostValue> values(1);
    values[0].concreteShape = ConcreteShape{3, 4};

    std::string error;
    EXPECT_FALSE(ad::resolveProgramShapes(execution, nullptr, values, error));
    EXPECT_NE(error.find("conflicts"), std::string::npos);
}

} // namespace
} // namespace vernon::runtime::shape
