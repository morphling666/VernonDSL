#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_INVOCATION_VALUES_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_INVOCATION_VALUES_H

#include "runtime/autodiff/program_boundary_binder.h"
#include "runtime/program_execution/program_invocation_state.h"

#include <map>
#include <optional>

namespace vernon::runtime::ad {

class ProgramTapeScratch;
class HostStaticTapeBatch;
class AutodiffMemoryPolicy;

std::optional<ValueLayout> resolvedProgramValueLayout(const program::Value &value);
std::optional<size_t> programValueByteSize(const program::Value &value, const Parameter *parameter = nullptr);
VernonValueLayoutView programValueLayoutView(const ValueLayout &layout);
inline bool programValueHasDynamicShape(const program::Value &value) {
    return !shape::isConcrete(shape::decodeRuntimeContractShape(value.shape));
}
bool fillProgramTapeHostValue(program_execution::ProgramValueState &value, std::shared_ptr<HostStaticTapeBatch> batch,
                              ProgramTapeScratch &tapeScratch, uint32_t valueId, std::string &error);

bool materializeProgramOwnedStorages(const program::Program &execution, const program::ResolvedExecutionPlan *plan,
                                     std::vector<program_execution::ProgramValueState> &values,
                                     const std::vector<char> &liveStorage,
                                     const std::vector<std::optional<ValueLayout>> &layouts,
                                     std::map<uint32_t, program_execution::ProgramStorageBacking> &backings,
                                     std::string &error);

bool materializeProgramValues(const program::Program &execution, const program::ResolvedExecutionPlan *plan,
                              std::vector<program_execution::ProgramValueState> &values, const std::vector<char> &live,
                              const std::vector<std::optional<ValueLayout>> &layouts,
                              const std::map<uint32_t, program_execution::ProgramStorageBacking> &backings,
                              const std::map<uint32_t, VernonProgramArgument> &externalValues,
                              const std::vector<std::shared_ptr<HostStaticTapeBatch>> *tapeCaptures,
                              const std::shared_ptr<AutodiffMemoryPolicy> &tapePolicy, ProgramTapeScratch &tapeScratch,
                              std::string &error);

} // namespace vernon::runtime::ad

#endif
