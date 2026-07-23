#pragma once

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
