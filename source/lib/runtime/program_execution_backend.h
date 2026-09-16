#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_BACKEND_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_BACKEND_H

#include "VernonResult.hpp"
#include "VernonRuntime.h"

#include <filesystem>
#include <memory>
#include <string>

namespace vernon::runtime::program {

struct ArtifactSystem;
struct Program;

struct ProgramLoadError {
    std::string code;
    std::string path;
    std::string message;
};

using ProgramLoadResult = vernon::Result<std::unique_ptr<VernonProgramExecutable>, ProgramLoadError>;

ProgramLoadResult loadBackendProgramPipeline(VernonRuntimeContext &context, const Program &program,
                                             const ArtifactSystem &artifacts, const std::filesystem::path &bundleRoot);
std::string renderProgramLoadError(const ProgramLoadError &error);

} // namespace vernon::runtime::program

#endif
