#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_SHAPE_RESOLVER_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_SHAPE_RESOLVER_H

#include <string>
#include <vector>

struct VernonProgramTopology;

namespace vernon::runtime::program {
struct Program;
}

namespace vernon::runtime::ad {

struct LogicalProgramValue;

bool resolveProgramShapes(const program::Program &program, const VernonProgramTopology *topology,
                          std::vector<LogicalProgramValue> &values, std::string &error);

} // namespace vernon::runtime::ad

#endif
