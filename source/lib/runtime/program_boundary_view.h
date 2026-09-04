#ifndef VERNON_RUNTIME_PROGRAM_BOUNDARY_VIEW_H
#define VERNON_RUNTIME_PROGRAM_BOUNDARY_VIEW_H

#include "VernonRuntime.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vernon::runtime {

struct ProgramBoundaryView {
    uint32_t slot{};
    uint32_t value{};
    std::string path;
    std::string role;
    std::string category;
};

std::vector<ProgramBoundaryView> programBoundaryViews(const VernonProgramExecutable &pipeline);

} // namespace vernon::runtime

#endif
