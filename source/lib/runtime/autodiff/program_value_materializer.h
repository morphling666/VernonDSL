#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_VALUE_MATERIALIZER_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_VALUE_MATERIALIZER_H

#include "runtime/autodiff/program_boundary_binder.h"
#include "runtime/autodiff/program_value_arena.h"

#include <map>
#include <optional>

struct VernonPipelineTopology;

namespace vernon::runtime::ad {

std::optional<ValueLayout> resolvedProgramValueLayout(const program::Value &value);
std::optional<size_t> programValueByteSize(const program::Value &value, const Parameter *parameter = nullptr);
VernonValueLayoutView programValueLayoutView(const ValueLayout &layout);
inline bool programValueHasDynamicShape(const program::Value &value) {
    return !shape::isConcrete(shape::decodeRuntimeContractShape(value.shape));
}
bool fillProgramTapeHostValue(ProgramHostValue &value, std::shared_ptr<HostStaticTapeBatch> batch, std::string &error);

bool materializeProgramOwnedStorages(const program::Program &execution, const VernonPipelineTopology *topology,
                                     std::vector<ProgramHostValue> &storage, const std::vector<char> &liveStorage,
                                     const std::vector<std::optional<ValueLayout>> &layouts,
                                     std::map<uint32_t, ProgramStorageBacking> &backings, std::string &error);

bool materializeProgramValues(const program::Program &execution, const VernonPipelineTopology *topology,
                              std::vector<ProgramHostValue> &storage, const std::vector<char> &live,
                              const std::vector<std::optional<ValueLayout>> &layouts,
                              const std::map<uint32_t, ProgramStorageBacking> &backings,
                              const std::map<uint32_t, VernonPipelineArgument> &externalValues,
                              const std::vector<std::shared_ptr<HostStaticTapeBatch>> *tapeCaptures,
                              const std::shared_ptr<AutodiffMemoryPolicy> &tapePolicy, std::string &error);

} // namespace vernon::runtime::ad

#endif
