#ifndef VERNON_COMPILER_PROGRAM_PUBLICATION_H
#define VERNON_COMPILER_PROGRAM_PUBLICATION_H

#include "compiler_program_boundary.h"

#include "llvm/Support/JSON.h"

#include <cstdint>
#include <vector>

namespace vernon::compiler {

enum class ProgramPublicationDisposition {
    CommitAfterSuccess,
};

struct ProgramPublicationTargetPlan {
    int64_t slot{};
    int64_t value{};
    ProgramBoundaryOwnerId owner;
    ProgramPublicationDisposition disposition{ProgramPublicationDisposition::CommitAfterSuccess};
};

struct ProgramPublicationPlan {
    std::vector<ProgramPublicationTargetPlan> targets;
};

ProgramPublicationPlan planProgramPublications(const ProgramBoundaryPlan &boundaries);
llvm::json::Array serializeProgramBoundarySlots(const ProgramBoundaryPlan &boundaries,
                                                const ProgramPublicationPlan &publication);

} // namespace vernon::compiler

#endif
