#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_SHAPE_RESOLVER_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_SHAPE_RESOLVER_H

#include <string>
#include <vector>

struct VernonPipelineTopology;

namespace vernon::runtime {
struct ExecutableProgram;
}

namespace vernon::runtime::ad {

struct ProgramHostValue;

bool resolveProgramShapes(const ExecutableProgram &execution, const VernonPipelineTopology *topology,
                          std::vector<ProgramHostValue> &values, std::string &error);

} // namespace vernon::runtime::ad

#endif
