#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_BACKEND_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_BACKEND_H

#include "VernonRuntime.h"

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>

namespace vernon::runtime::program {

VernonLoadedPipeline *loadBackendProgramPipeline(VernonRuntimeContext &context, const char *programJson,
                                                 size_t programJsonSize, const char *artifactSystemJson,
                                                 size_t artifactSystemJsonSize,
                                                 const std::map<std::string, std::string> &stageBindings,
                                                 const std::filesystem::path &bundleRoot, std::string &error);

} // namespace vernon::runtime::program

#endif
