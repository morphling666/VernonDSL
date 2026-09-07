#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_PUBLICATION_TRANSACTION_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_PUBLICATION_TRANSACTION_H

#include "program_invocation_state.h"

namespace vernon::runtime::program_execution {

class PublicationTransaction {
public:
    enum class Status { Open, Committed, RolledBack, Poisoned };

    explicit PublicationTransaction(const program::ResolvedPublicationPlan &plan) : plan_(&plan) {}
    ~PublicationTransaction() { rollback(); }

    bool stage(uint32_t slot, const program::PublicationTarget &target, const VernonProgramArgument &destination,
               std::optional<VernonRhiBuffer> destinationBuffer, std::string &error);
    bool stageHostBytes(void *destination, const void *source, size_t size, std::string &error);
    bool applyConcreteShapes(const program::Program &program, std::vector<ProgramValueState> &values,
                             std::string &error) const;
    std::vector<char> hostReadbackValues(size_t valueCount) const;
    VernonStatus commit(VernonRuntimeContext &context, const ProgramInvocationState &state, std::string &error);
    void rollback();
    void poison();

    Status status() const { return status_; }
    bool empty() const { return staged_.empty() && stagedHostCopies_.empty(); }

private:
    struct StagedPublication {
        const program::ResolvedPublicationTransaction *transaction{};
        const program::PublicationTarget *target{};
        VernonProgramArgument destination{};
        std::optional<VernonRhiBuffer> destinationBuffer;
    };
    struct StagedHostCopy {
        void *destination{};
        std::vector<uint8_t> bytes;
    };

    const program::ResolvedPublicationPlan *plan_;
    std::vector<StagedPublication> staged_;
    std::vector<StagedHostCopy> stagedHostCopies_;
    Status status_{Status::Open};
};

} // namespace vernon::runtime::program_execution

#endif
