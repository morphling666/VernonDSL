#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_BACKEND_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_BACKEND_H

#include "VernonResult.hpp"
#include "VernonRuntime.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace vernon::formats {
struct Format;
}

namespace vernon::runtime::program {

struct ArtifactSystem;
struct Diagnostic;
struct Program;

// The VernonRhiFormatCapabilityBits a device grants a table format.
using FormatCapabilityQuery = std::function<uint32_t(const vernon::formats::Format &)>;

// Program load rejects image storages and graphics attachments whose formats the device cannot use as declared, so
// an unsupported target fails before any invocation.
bool validateTargetFormats(const Program &program, const FormatCapabilityQuery &capabilities, Diagnostic &diagnostic);

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
