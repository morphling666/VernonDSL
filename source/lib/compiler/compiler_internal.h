#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace mlir {
class Diagnostic;
}

namespace vernon::compiler {

struct Artifact {
    std::string name;
    std::string data;
};

struct TargetResourceSlot {
    std::string entryPoint;
    std::string stage;
    std::string kind;
    std::string name;
    uint32_t descriptorSet{};
    uint32_t binding{};
    uint32_t index{};
    uint32_t count{1};
};

struct CpuCodegenOptions {
    std::string targetTriple;
    std::string cpu;
    std::string features;
};

class CpuExecutionState;

struct CpuExecutionStateDeleter {
    void operator()(CpuExecutionState *state) const;
};

using CpuExecutionStatePtr = std::unique_ptr<CpuExecutionState, CpuExecutionStateDeleter>;

void appendDiagnostic(std::string &output, mlir::Diagnostic &diagnostic);

} // namespace vernon::compiler
