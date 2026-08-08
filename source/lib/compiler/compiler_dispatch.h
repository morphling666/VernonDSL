#pragma once

#include "VernonCompiler.h"
#include "compiler_internal.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace vernon::compiler {

class PreparedModule;

struct OpenGLCompileOptions {
    VernonTarget target{VERNON_TARGET_OPENGL};
    uint32_t version{};
};

struct VulkanCompileOptions {};

struct MetalCompileOptions {
    VernonMetalPlatform platform{VERNON_METAL_PLATFORM_MACOS};
};

struct DirectXCompileOptions {
    uint32_t shaderModel{60};
};

struct CudaCompileOptions {};

using CompileOptions = std::variant<CpuCodegenOptions, OpenGLCompileOptions, VulkanCompileOptions, MetalCompileOptions,
                                    DirectXCompileOptions, CudaCompileOptions>;

VernonTargetCapabilities targetCapabilities(VernonTarget target);

VernonStatus parseCompileOptions(const VernonCompileOptions &source, CompileOptions &options, std::string &diagnostics);
CompileOptions defaultCompileOptions(VernonTarget target);
VernonTarget compileTargetKind(const CompileOptions &options);

VernonStatus compileTarget(PreparedModule &module, const CompileOptions &options, std::vector<Artifact> &artifacts,
                           std::string &reflection, std::string &diagnostics, CpuExecutionStatePtr &cpuExecution);

VernonCpuEntryPoint findCompiledCpuEntry(const CpuExecutionState *execution, std::string_view entry);

} // namespace vernon::compiler
