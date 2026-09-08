#include "compiler_frontend.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>
#include <memory>

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
