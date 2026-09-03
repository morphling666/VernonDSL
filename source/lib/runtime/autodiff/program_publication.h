#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_PUBLICATION_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_PUBLICATION_H

#include "runtime/autodiff/program_value_arena.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace vernon::runtime::ad {

class ProgramOwnerBindings {
public:
    bool bind(program::ProgramOwnerId owner, const VernonPipelineArgument &argument, std::string &error);

private:
    std::map<std::pair<program::ProgramOwnerKind, uint32_t>, VernonPipelineArgument> bindings_;
};

struct PendingProgramPublication {
    const program::PublicationTarget *target{};
    VernonPipelineArgument destination{};
    std::optional<VernonPipelineArgument> initialValue;
    std::optional<VernonRhiBuffer> destinationBuffer;
    std::optional<VernonRhiBuffer> initialBuffer;
};

bool initializeProgramPublications(std::vector<ProgramHostValue> &storage,
                                   const std::vector<PendingProgramPublication> &publications, std::string &error);
bool applyProgramPublicationShapes(const program::Program &program,
                                   const std::vector<PendingProgramPublication> &publications,
                                   std::vector<ProgramHostValue> &storage, std::string &error);
bool commitProgramPublications(const std::vector<ProgramHostValue> &storage,
                               const std::vector<PendingProgramPublication> &publications, std::string &error);
VernonStatus initializeDeviceProgramPublications(VernonRuntimeContext &context, const ProgramInvocationFrame &frame,
                                                 const std::vector<PendingProgramPublication> &publications,
                                                 std::string &error);
VernonStatus commitDeviceProgramPublications(VernonRuntimeContext &context, const ProgramInvocationFrame &frame,
                                             const std::vector<PendingProgramPublication> &publications,
                                             std::string &error);

} // namespace vernon::runtime::ad

#endif
