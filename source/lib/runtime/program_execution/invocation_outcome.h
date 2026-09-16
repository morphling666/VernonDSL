#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_INVOCATION_OUTCOME_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_INVOCATION_OUTCOME_H

#include "VernonRuntime.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <vector>

namespace vernon::runtime::program_execution {

enum class SubmissionState : uint8_t { NotSubmitted, Completed, Indeterminate };

struct BoundaryMutation {
    VernonMutationBoundaryKind kind{VERNON_MUTATION_PROGRAM_BOUNDARY};
    uint32_t slot{};
    VernonBoundaryMutationState state{VERNON_BOUNDARY_MUTATION_UNCHANGED};
};

struct InvocationMutationOutcome {
    SubmissionState submission{SubmissionState::NotSubmitted};
    std::vector<BoundaryMutation> boundaries;

    void set(VernonMutationBoundaryKind kind, uint32_t slot, VernonBoundaryMutationState state) {
        const auto found = std::find_if(boundaries.begin(), boundaries.end(), [&](const BoundaryMutation &entry) {
            return entry.kind == kind && entry.slot == slot;
        });
        if (found == boundaries.end())
            boundaries.push_back({kind, slot, state});
        else
            found->state = state;
    }

    void set(uint32_t slot, VernonBoundaryMutationState state) { set(VERNON_MUTATION_PROGRAM_BOUNDARY, slot, state); }

    void merge(InvocationMutationOutcome source) {
        if (submission != SubmissionState::Indeterminate) {
            if (source.submission == SubmissionState::Indeterminate)
                submission = SubmissionState::Indeterminate;
            else if (source.submission == SubmissionState::Completed && submission == SubmissionState::NotSubmitted)
                submission = SubmissionState::Completed;
        }
        for (const BoundaryMutation &boundary : source.boundaries)
            set(boundary.kind, boundary.slot, boundary.state);
    }
};

inline VernonInvocationSubmissionState toPublicSubmissionState(SubmissionState state) {
    switch (state) {
    case SubmissionState::NotSubmitted:
        return VERNON_INVOCATION_NOT_SUBMITTED;
    case SubmissionState::Completed:
        return VERNON_INVOCATION_COMPLETED;
    case SubmissionState::Indeterminate:
        return VERNON_INVOCATION_INDETERMINATE;
    }
    return VERNON_INVOCATION_INDETERMINATE;
}

inline bool preparePublicOutcome(VernonInvocationMutationOutcome *outcome, size_t requiredCapacity) {
    if (!outcome)
        return true;
    if (outcome->struct_size != sizeof(*outcome) ||
        std::any_of(std::begin(outcome->reserved), std::end(outcome->reserved),
                    [](uint32_t value) { return value != 0; }) ||
        requiredCapacity > outcome->mutation_capacity || (requiredCapacity && !outcome->mutations))
        return false;
    outcome->submission = VERNON_INVOCATION_NOT_SUBMITTED;
    outcome->mutation_count = 0;
    return true;
}

inline void publishPublicOutcome(const InvocationMutationOutcome &source, VernonInvocationMutationOutcome *outcome) {
    if (!outcome)
        return;
    outcome->submission = toPublicSubmissionState(source.submission);
    outcome->mutation_count = source.boundaries.size();
    for (size_t index = 0; index < source.boundaries.size(); ++index)
        outcome->mutations[index] = {
            source.boundaries[index].kind,
            source.boundaries[index].slot,
            source.boundaries[index].state,
        };
}

} // namespace vernon::runtime::program_execution

#endif
