#ifndef VERNON_COMPILE_PACKAGING_H
#define VERNON_COMPILE_PACKAGING_H

#include "VernonCompiler.h"

#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>

namespace vernon::tools {

struct PackagingOptions {
    std::optional<std::filesystem::path> outputDirectory;
    std::optional<std::filesystem::path> reflectionPath;
    std::optional<std::string> targetTriple;
};

VernonStatus packageCompileResult(const VernonCompileResult *result, VernonTarget target,
                                  const PackagingOptions &options, std::ostream &standardOutput,
                                  std::ostream &standardError);

} // namespace vernon::tools

#endif
