#ifndef VERNON_COMPILE_PACKAGING_H
#define VERNON_COMPILE_PACKAGING_H

#include "VernonCompiler.h"

#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>

namespace vernon::tools {

struct LegacyPackagingOptions {
  std::optional<std::filesystem::path> outputDirectory;
  std::optional<std::filesystem::path> reflectionPath;
  std::optional<std::filesystem::path> shaderBundlePath;
  std::optional<std::filesystem::path> computeBundlePath;
  std::optional<std::string> assetId;
  std::optional<std::string> targetTriple;
  bool hostRuntimeBundle{};
};

// Writes the legacy CLI formats without owning either compiler object.
VernonStatus packageCompileResult(VernonCompilerContext *context,
                                  const VernonCompileResult *result,
                                  VernonTarget target,
                                  const LegacyPackagingOptions &options,
                                  std::ostream &standardOutput,
                                  std::ostream &standardError);

} // namespace vernon::tools

#endif
