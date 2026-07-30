#pragma once

#include "VernonCommon.h"
#include "compiler_internal.h"

#include "mlir/IR/BuiltinOps.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace mlir {
class MLIRContext;
}

namespace vernon::compiler {

class CompilerFrontend;

class PreparedModule {
public:
    explicit PreparedModule(mlir::OwningOpRef<mlir::ModuleOp> module);

    mlir::MLIRContext &context();
    mlir::OwningOpRef<mlir::ModuleOp> clone();

private:
    mlir::OwningOpRef<mlir::ModuleOp> module_;
};

using PreparedModulePtr = std::unique_ptr<PreparedModule>;

CompilerFrontend *createCompilerFrontend();
void destroyCompilerFrontend(CompilerFrontend *frontend);
mlir::MLIRContext &compilerMlirContext(CompilerFrontend &frontend);

VernonStatus prepareMlir(CompilerFrontend &frontend, const char *source, size_t sourceSize, PreparedModulePtr &prepared,
                         std::vector<Artifact> &artifacts, std::string &reflection, std::string &diagnostics);

} // namespace vernon::compiler
