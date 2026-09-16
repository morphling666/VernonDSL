#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_PUBLICATION_TRANSACTION_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_PUBLICATION_TRANSACTION_H

#include "device_commands.h"
#include "invocation_outcome.h"
#include "program_invocation_state.h"

#include <functional>
#include <variant>

namespace vernon::runtime::program_execution {

enum class PublicationError : uint8_t {
    TransactionNotOpen,
    TransactionNotPrepared,
    PlanningFailed,
    UnauthorizedBinding,
    EndpointMismatch,
    InvalidHostRegion,
    IncompleteShape,
    InvalidStagedValue,
    AliasedDeviceBacking,
    TensorCopyInvalid,
    InvalidImageDestination,
    InvalidImageParent,
    ImageAllocationFailed,
    ImageViewAllocationFailed,
    ImageReferenceFailed,
    ReadbackFailed,
    CommitFailed,
    ExecutorRequired,
};

template <typename T> using PublicationResult = vernon::Result<T, PublicationError>;

const char *publicationErrorMessage(PublicationError error) noexcept;
VernonStatus publicationErrorStatus(PublicationError error) noexcept;

class PublicationTransaction {
public:
    enum class Status { Open, Prepared, Committed, RolledBack, Poisoned };

    explicit PublicationTransaction(const program::ResolvedPublicationPlan &plan) : plan_(&plan) {}
    ~PublicationTransaction() noexcept { rollback(); }

    void reset();
    PublicationResult<void> bindHostCommit(uint32_t slot, const program::PublicationTarget &target,
                                           const VernonProgramArgument &destination);
    PublicationResult<void> bindDeviceCommit(uint32_t slot, const program::PublicationTarget &target,
                                             const VernonProgramArgument &destination,
                                             VernonRhiBuffer destinationBuffer);
    PublicationResult<void> bindImageCommit(VernonRuntimeContext &context, uint32_t slot,
                                            const program::PublicationTarget &target,
                                            const VernonProgramArgument &destination, VernonProgramArgument &staging);
    PublicationResult<void> bindInPlace(uint32_t slot, const VernonProgramArgument &destination);
    PublicationResult<void> stageHostRegion(uint32_t slot, void *destination, const void *source, size_t size);
    PublicationResult<void> applyConcreteShapes(const program::Program &program,
                                                std::vector<ProgramValueState> &values) const;
    std::vector<char> hostReadbackValues(size_t valueCount) const;
    std::vector<DeviceImageCopy> initializationImageCopies() const;
    PublicationResult<void> prepareCommit(const ProgramInvocationState &state,
                                          std::vector<DeviceBufferCopy> &bufferCopies,
                                          std::vector<DeviceImageCopy> &imageCopies);
    PublicationResult<void> completeCommit();
    PublicationResult<void> commit(VernonRuntimeContext &context, const ProgramInvocationState &state);
    void rollback() noexcept;
    void poison() noexcept;
    void noteSubmission(SubmissionState state);
    void noteInPlaceSubmission(SubmissionState state);
    InvocationMutationOutcome mutationOutcome() const;

    Status status() const { return status_; }
    bool empty() const { return entries_.empty() && hostRegions_.empty(); }
    size_t mutationCount() const { return boundMutations_.size(); }

private:
    struct HostCommitEntry {
        const program::ResolvedPublicationTransaction *transaction{};
        const program::PublicationTarget *target{};
        VernonProgramArgument destination{};
    };
    struct DeviceCommitEntry {
        const program::ResolvedPublicationTransaction *transaction{};
        const program::PublicationTarget *target{};
        VernonProgramArgument destination{};
        VernonRhiBuffer destinationBuffer{};
    };
    struct InPlaceEntry {
        const program::ResolvedPublicationTransaction *transaction{};
        VernonProgramArgument destination{};
    };
    struct ImageCommitEntry {
        const program::ResolvedPublicationTransaction *transaction{};
        const program::PublicationTarget *target{};
        VernonProgramArgument destination{};
        VernonRhiDevice device{};
        VernonRhiImage destinationImage{};
        VernonRhiImage stagingImage{};
        VernonRhiImageView stagingView{};
        VernonRhiImageDescriptor imageDescriptor{};
        VernonRhiImageViewDescriptor viewDescriptor{};
    };
    using Entry = std::variant<HostCommitEntry, DeviceCommitEntry, ImageCommitEntry, InPlaceEntry>;
    struct HostRegion {
        const program::ResolvedPublicationTransaction *transaction{};
        void *destination{};
        std::vector<uint8_t> bytes;
    };
    struct PreparedHostCopy {
        void *destination{};
        std::vector<uint8_t> bytes;
    };
    struct BoundMutation {
        uint32_t slot{};
        bool inPlace{};
    };

    PublicationResult<std::reference_wrapper<const program::ResolvedPublicationTransaction>>
    resolve(uint32_t slot, program::PublicationCommitMode mode) const;
    bool slotIsBound(uint32_t slot) const;
    void releaseDeviceImages() noexcept;

    const program::ResolvedPublicationPlan *plan_;
    std::vector<Entry> entries_;
    std::vector<BoundMutation> boundMutations_;
    std::vector<HostRegion> hostRegions_;
    std::vector<PreparedHostCopy> preparedHostCopies_;
    Status status_{Status::Open};
    SubmissionState overallSubmission_{SubmissionState::NotSubmitted};
    SubmissionState inPlaceSubmission_{SubmissionState::NotSubmitted};
};

} // namespace vernon::runtime::program_execution

#endif
