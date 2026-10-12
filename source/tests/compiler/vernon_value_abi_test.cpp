#include "compiler_frontend.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>
#include <memory>
#include <vector>

TEST(VernonValueAbi, SignlessI32RequiresExplicitLanguageDtype) {
    std::unique_ptr<vernon::compiler::CompilerFrontend, decltype(&vernon::compiler::destroyCompilerFrontend)> frontend(
        vernon::compiler::createCompilerFrontend(), vernon::compiler::destroyCompilerFrontend);
    ASSERT_NE(frontend, nullptr);
    mlir::MLIRContext &context = vernon::compiler::compilerMlirContext(*frontend);
    auto module = mlir::parseSourceString<mlir::ModuleOp>("module {}", &context);
    ASSERT_TRUE(module);
    mlir::Type i32 = mlir::IntegerType::get(&context, 32);
    EXPECT_TRUE(mlir::failed(mlir::vernon::getValueAbiLayout(i32, *module)));

    mlir::FailureOr<mlir::vernon::ValueAbiLayout> asUnsigned =
        mlir::vernon::getValueAbiLayout(i32, *module, {llvm::StringRef("u32")});
    mlir::FailureOr<mlir::vernon::ValueAbiLayout> asSigned =
        mlir::vernon::getValueAbiLayout(i32, *module, {llvm::StringRef("i32")});
    ASSERT_TRUE(mlir::succeeded(asUnsigned));
    ASSERT_TRUE(mlir::succeeded(asSigned));
    EXPECT_EQ(asUnsigned->leaves.front().dtype, "u32");
    EXPECT_EQ(asSigned->leaves.front().dtype, "i32");
    EXPECT_NE(asUnsigned->layoutHash, asSigned->layoutHash);

    mlir::FailureOr<mlir::vernon::ValueAbiLayout> storage = mlir::vernon::getValueStorageLayout(i32, *module);
    ASSERT_TRUE(mlir::succeeded(storage));
    EXPECT_EQ(storage->leaves.front().dtype, "i32");
    EXPECT_EQ(storage->layoutHash, asSigned->layoutHash);

    EXPECT_TRUE(mlir::failed(mlir::vernon::getCpuCallPlan(i32, *module)));
    EXPECT_TRUE(mlir::failed(
        mlir::vernon::getBackendInterfaceAbiPlan(i32, *module, mlir::vernon::PhysicalAbiProfile::HostValue)));
    mlir::FailureOr<mlir::vernon::CpuCallPlan> signedCall =
        mlir::vernon::getCpuCallPlan(i32, *module, {llvm::StringRef("i32")});
    mlir::FailureOr<mlir::vernon::BackendInterfaceAbiPlan> unsignedHost = mlir::vernon::getBackendInterfaceAbiPlan(
        i32, *module, mlir::vernon::PhysicalAbiProfile::HostValue, {llvm::StringRef("u32")});
    ASSERT_TRUE(mlir::succeeded(signedCall));
    ASSERT_TRUE(mlir::succeeded(unsignedHost));
    EXPECT_EQ(signedCall->layout.leaves.front().dtype, "i32");
    const auto *bytes = std::get_if<mlir::vernon::ByteTransportPlan>(&*unsignedHost);
    ASSERT_NE(bytes, nullptr);
    EXPECT_EQ(bytes->canonicalLayoutHash, asUnsigned->layoutHash);

    mlir::FailureOr<mlir::vernon::ByteTransportPlan> transport =
        mlir::vernon::getByteTransportPlan(i32, *module, mlir::vernon::PhysicalAbiProfile::VulkanPushConstant);
    ASSERT_TRUE(mlir::succeeded(transport));
    EXPECT_EQ(transport->canonicalLayoutHash, storage->layoutHash);
}

TEST(VernonValueAbi, FloatStorageIsUnambiguousLanguageDtype) {
    std::unique_ptr<vernon::compiler::CompilerFrontend, decltype(&vernon::compiler::destroyCompilerFrontend)> frontend(
        vernon::compiler::createCompilerFrontend(), vernon::compiler::destroyCompilerFrontend);
    ASSERT_NE(frontend, nullptr);
    mlir::MLIRContext &context = vernon::compiler::compilerMlirContext(*frontend);
    auto module = mlir::parseSourceString<mlir::ModuleOp>("module {}", &context);
    ASSERT_TRUE(module);
    mlir::Type f32 = mlir::Float32Type::get(&context);
    mlir::FailureOr<mlir::vernon::ValueAbiLayout> layout = mlir::vernon::getValueAbiLayout(f32, *module);
    ASSERT_TRUE(mlir::succeeded(layout));
    EXPECT_EQ(layout->leaves.front().dtype, "f32");
}

TEST(VernonValueAbi, PushConstantMatrixColumnsStartRegisters) {
    std::unique_ptr<vernon::compiler::CompilerFrontend, decltype(&vernon::compiler::destroyCompilerFrontend)> frontend(
        vernon::compiler::createCompilerFrontend(), vernon::compiler::destroyCompilerFrontend);
    ASSERT_NE(frontend, nullptr);
    mlir::MLIRContext &context = vernon::compiler::compilerMlirContext(*frontend);
    auto module = mlir::parseSourceString<mlir::ModuleOp>("module {}", &context);
    ASSERT_TRUE(module);
    const auto plan = [&](llvm::ArrayRef<int64_t> shape, mlir::vernon::PhysicalAbiProfile profile) {
        return mlir::vernon::getByteTransportPlan(mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&context)),
                                                  *module, profile);
    };
    for (int64_t columns = 2; columns <= 4; ++columns) {
        mlir::FailureOr<mlir::vernon::ByteTransportPlan> push =
            plan({2, columns}, mlir::vernon::PhysicalAbiProfile::VulkanPushConstant);
        mlir::FailureOr<mlir::vernon::ByteTransportPlan> storage =
            plan({2, columns}, mlir::vernon::PhysicalAbiProfile::VulkanStd430StorageBuffer);
        ASSERT_TRUE(mlir::succeeded(push));
        ASSERT_TRUE(mlir::succeeded(storage));
        EXPECT_EQ(push->root->alignment, 16u);
        EXPECT_EQ(push->root->size, 16u * static_cast<uint64_t>(columns));
        EXPECT_EQ(push->root->byteStrides, (llvm::SmallVector<uint64_t>{4, 16}));
        EXPECT_EQ(storage->root->byteStrides, (llvm::SmallVector<uint64_t>{4, 8}));
    }
    const std::vector<std::vector<int64_t>> unpaddedShapes{{}, {3}};
    for (const std::vector<int64_t> &shape : unpaddedShapes) {
        mlir::FailureOr<mlir::vernon::ByteTransportPlan> push =
            plan(shape, mlir::vernon::PhysicalAbiProfile::VulkanPushConstant);
        mlir::FailureOr<mlir::vernon::ByteTransportPlan> storage =
            plan(shape, mlir::vernon::PhysicalAbiProfile::VulkanStd430StorageBuffer);
        ASSERT_TRUE(mlir::succeeded(push));
        ASSERT_TRUE(mlir::succeeded(storage));
        EXPECT_EQ(push->root->size, storage->root->size);
        EXPECT_EQ(push->root->alignment, storage->root->alignment);
    }
}
