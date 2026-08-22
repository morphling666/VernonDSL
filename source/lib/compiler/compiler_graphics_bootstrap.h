#ifndef VERNON_COMPILER_GRAPHICS_BOOTSTRAP_H
#define VERNON_COMPILER_GRAPHICS_BOOTSTRAP_H

#include "VernonCommon.h"
#include "compiler_frontend.h"

#include <cstddef>
#include <string>
#include <vector>

namespace vernon::compiler {

struct GraphicsStageSource {
    const char *source{};
    size_t sourceSize{};
};

struct GraphicsPlanOperand {
    std::string name;
    std::string type;
};

VernonStatus planGraphicsProgram(CompilerFrontend &frontend, const std::vector<GraphicsStageSource> &stages,
                                 const std::string &topology, const std::vector<std::string> &features,
                                 const std::vector<std::string> &attachmentTypes, uint32_t colorCount,
                                 const std::vector<GraphicsPlanOperand> &operands, std::vector<Artifact> &artifacts,
                                 std::string &reflection, std::string &diagnostics);

} // namespace vernon::compiler

#endif
