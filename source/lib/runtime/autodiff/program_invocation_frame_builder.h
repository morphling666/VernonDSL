#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_INVOCATION_FRAME_BUILDER_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_INVOCATION_FRAME_BUILDER_H

#include "runtime/autodiff/program_invocation_spec.h"
#include "runtime/autodiff/program_value_arena.h"

struct VernonProgramTopology;
struct VernonRuntimeContext;

namespace vernon::runtime::ad {

bool matchesProgramValueAbi(const VernonAdValue &value, const ValueAbi &abi);

bool buildLogicalValueFrame(VernonRuntimeContext &context, const program::Program &execution,
                            const VernonProgramTopology *topology, std::vector<LogicalProgramValue> &storage,
                            const ForwardInvocationSpec &spec, std::shared_ptr<AutodiffMemoryPolicy> tapePolicy,
                            std::string &error);

bool buildLogicalValueFrame(VernonRuntimeContext &context, const program::Program &execution,
                            const VernonProgramTopology *topology, std::vector<LogicalProgramValue> &storage,
                            const PullbackInvocationSpec &spec, std::shared_ptr<AutodiffMemoryPolicy> tapePolicy,
                            std::string &error);

} // namespace vernon::runtime::ad

#endif
