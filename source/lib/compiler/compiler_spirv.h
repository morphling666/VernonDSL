#pragma once

#include "VernonCompiler.h"
#include "compiler_internal.h"

#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vernon::compiler_detail {

VERNON_DSL_CAPI bool materializeImageQuerySizeLod(llvm::SmallVectorImpl<uint32_t> &words,
                                                  size_t expectedReplacementCount, std::string &diagnostics);

} // namespace vernon::compiler_detail

namespace vernon::compiler {

class PreparedModule;

bool compileSpirv(PreparedModule &prepared, VernonTarget target, std::vector<Artifact> &artifacts,
                  std::string &diagnostics);

} // namespace vernon::compiler
