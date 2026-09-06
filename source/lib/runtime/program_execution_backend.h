#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_BACKEND_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_BACKEND_H

#include "VernonRuntime.h"

#include <cstddef>
#include <filesystem>
#include <string>

namespace vernon::runtime::program {

VernonProgramExecutable *loadBackendProgramPipeline(VernonRuntimeContext &context, const char *programJson,
                                                    size_t programJsonSize, const char *targetJson,
                                                    size_t targetJsonSize, const char *blobsJson, size_t blobsJsonSize,
                                                    const char *artifactSystemJson, size_t artifactSystemJsonSize,
                                                    const std::filesystem::path &bundleRoot, std::string &error);

} // namespace vernon::runtime::program

#endif
