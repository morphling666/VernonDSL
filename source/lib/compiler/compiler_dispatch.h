#pragma once

#include "VernonCompiler.h"
#include "compiler_internal.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace vernon::compiler {

class PreparedModule;

struct CompileOptions {
    uint32_t glslVersion{};
    uint32_t hlslShaderModel{60};
    VernonMetalPlatform metalPlatform{VERNON_METAL_PLATFORM_MACOS};
    CpuCodegenOptions cpu;
};

VernonTargetCapabilities targetCapabilities(VernonTarget target);

VernonStatus parseCompileOptions(const VernonCompileOptions *source, VernonTarget target, CompileOptions &options,
                                 std::string &diagnostics);

VernonStatus compileTarget(PreparedModule &module, VernonTarget target, const CompileOptions &options,
                           std::vector<Artifact> &artifacts, std::string &reflection, std::string &diagnostics,
                           CpuExecutionStatePtr &cpuExecution);

bool linkCpuHostObject(const void *object, size_t objectSize, Artifact &artifact, std::string &diagnostics);
VernonCpuEntryPoint findCompiledCpuEntry(const CpuExecutionState *execution, std::string_view entry);

} // namespace vernon::compiler
