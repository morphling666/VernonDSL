#pragma once

#include "VernonCommon.h"
#include "compiler_internal.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
class MLIRContext;
}

namespace vernon::compiler {

class CompilerFrontend;

bool moduleUsesF16(mlir::ModuleOp module);

struct LogicalStructLayout {
    std::string name;
    uint64_t size{};
    uint64_t alignment{};
    std::vector<uint64_t> fieldOffsets;
    std::vector<std::string> fields;
};

struct LogicalDependency {
    std::string path;
    std::string sha256;
};

struct LogicalValueModel {
    unsigned index{};
    std::string sourcePath;
    bool sourcePathExplicit{};
    std::string dtype;
    bool dtypeExplicit{};
    std::vector<std::string> leafDtypes;
    std::vector<int64_t> shape;
    bool shapeExplicit{};
    std::string autodiffRole;
    std::string autodiffSource;
    std::vector<std::string> autodiffGradientPaths;
};

struct LogicalEntryModel {
    std::string name;
    std::string stage;
    bool isEntry{};
    std::vector<LogicalValueModel> arguments;
    std::vector<LogicalValueModel> results;
};

class LogicalReflectionModel {
public:
    int64_t compilerContractVersion{};
    std::vector<LogicalEntryModel> entries;
    std::vector<LogicalStructLayout> structLayouts;
    std::vector<LogicalDependency> dependencies;
    std::vector<std::string> requiredFeatures;
    std::string moduleHash;
};

class PreparedModule {
public:
    PreparedModule(mlir::OwningOpRef<mlir::ModuleOp> module, LogicalReflectionModel logicalReflection);

    mlir::MLIRContext &context();
    mlir::OwningOpRef<mlir::ModuleOp> clone();
    mlir::ModuleOp logicalModule() const;
    const LogicalReflectionModel &logicalReflection() const;

private:
    mlir::OwningOpRef<mlir::ModuleOp> module_;
    LogicalReflectionModel logicalReflection_;
};

struct PhysicalArgumentModel {
    unsigned index{};
    std::string type;
    std::string builtin;
    std::optional<unsigned> logicalIndex;
    std::string logicalPath;
    std::optional<unsigned> descriptorOwner;
    std::string descriptorComponent;
    std::optional<unsigned> descriptorDimension;
};

struct PhysicalResultModel {
    unsigned index{};
    std::string type;
    std::optional<unsigned> logicalIndex;
    std::string logicalPath;
};

struct PhysicalEntryModel {
    std::string name;
    std::string stage;
    std::vector<PhysicalArgumentModel> arguments;
    std::vector<PhysicalResultModel> results;
};

struct LogicalValueOrigin {
    unsigned index{};
    std::string path;
};

struct PhysicalEntryProvenance {
    std::string name;
    std::vector<std::optional<LogicalValueOrigin>> arguments;
    std::vector<std::optional<LogicalValueOrigin>> results;
};

class TargetPreparationProvenance {
public:
    TargetPreparationProvenance(mlir::ModuleOp module, const LogicalReflectionModel &logical);

    mlir::LogicalResult mapArgument(mlir::func::FuncOp function, unsigned physicalIndex, unsigned logicalIndex);
    mlir::LogicalResult mapResult(mlir::func::FuncOp function, unsigned physicalIndex, unsigned logicalIndex);
    mlir::FailureOr<std::vector<PhysicalEntryProvenance>> resolve(mlir::ModuleOp module);

private:
    const LogicalReflectionModel *logical_{};
};

struct TargetPreparationResult {
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::vector<PhysicalEntryModel> entries;
    std::vector<PhysicalEntryProvenance> provenance;
};

using TargetPreparer = std::function<mlir::LogicalResult(mlir::ModuleOp, TargetPreparationProvenance &)>;

using PreparedModulePtr = std::unique_ptr<PreparedModule>;

CompilerFrontend *createCompilerFrontend();
void destroyCompilerFrontend(CompilerFrontend *frontend);
mlir::MLIRContext &compilerMlirContext(CompilerFrontend &frontend);
mlir::LogicalResult preparePortableTargetModule(mlir::ModuleOp module, TargetPreparationProvenance &provenance);
mlir::FailureOr<TargetPreparationResult> prepareTargetModule(PreparedModule &prepared, const TargetPreparer &preparer);

VernonStatus prepareMlir(CompilerFrontend &frontend, const char *source, size_t sourceSize, PreparedModulePtr &prepared,
                         std::vector<Artifact> &artifacts, std::string &reflection, std::string &diagnostics);

VernonStatus prepareProgramModule(CompilerFrontend &frontend, mlir::OwningOpRef<mlir::ModuleOp> module,
                                  PreparedModulePtr &prepared, std::vector<Artifact> &artifacts,
                                  std::string &reflection, std::string &diagnostics);

} // namespace vernon::compiler
