#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_BACKEND_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_BACKEND_H

#include "VernonRuntime.h"

#include <filesystem>
#include <string>

namespace vernon::runtime::program {

struct ArtifactSystem;
struct Program;

VernonProgramExecutable *loadBackendProgramPipeline(VernonRuntimeContext &context, const Program &program,
                                                    const ArtifactSystem &artifacts,
                                                    const std::filesystem::path &bundleRoot, std::string &error);

} // namespace vernon::runtime::program

#endif
