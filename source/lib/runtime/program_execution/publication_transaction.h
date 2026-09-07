#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_PUBLICATION_TRANSACTION_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_PUBLICATION_TRANSACTION_H

#include "device_commands.h"
#include "program_invocation_state.h"

#include <variant>

namespace vernon::runtime::program_execution {

class PublicationTransaction {
public:
    enum class Status { Open, Prepared, Committed, RolledBack, Poisoned };

    explicit PublicationTransaction(const program::ResolvedPublicationPlan &plan) : plan_(&plan) {}
    ~PublicationTransaction() { rollback(); }

    bool bindHostCommit(uint32_t slot, const program::PublicationTarget &target,
                        const VernonProgramArgument &destination, std::string &error);
    bool bindDeviceCommit(uint32_t slot, const program::PublicationTarget &target,
                          const VernonProgramArgument &destination, VernonRhiBuffer destinationBuffer,
                          std::string &error);
    bool bindImageCommit(VernonRuntimeContext &context, uint32_t slot, const program::PublicationTarget &target,
                         const VernonProgramArgument &destination, VernonProgramArgument &staging, std::string &error);
    bool bindInPlace(uint32_t slot, const VernonProgramArgument &destination, std::string &error);
    bool stageHostRegion(uint32_t slot, void *destination, const void *source, size_t size, std::string &error);
    bool applyConcreteShapes(const program::Program &program, std::vector<ProgramValueState> &values,
                             std::string &error) const;
    std::vector<char> hostReadbackValues(size_t valueCount) const;
    std::vector<DeviceImageCopy> initializationImageCopies() const;
    VernonStatus prepareCommit(const ProgramInvocationState &state, std::vector<DeviceBufferCopy> &bufferCopies,
                               std::vector<DeviceImageCopy> &imageCopies, std::string &error);
    VernonStatus completeCommit(std::string &error);
    VernonStatus commit(VernonRuntimeContext &context, const ProgramInvocationState &state, std::string &error);
    void rollback();
    void poison();

    Status status() const { return status_; }
    bool empty() const { return entries_.empty() && hostRegions_.empty(); }

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

    const program::ResolvedPublicationTransaction *resolve(uint32_t slot, program::PublicationCommitMode mode,
                                                           std::string &error) const;
    bool slotIsBound(uint32_t slot) const;
    void releaseDeviceImages();

    const program::ResolvedPublicationPlan *plan_;
    std::vector<Entry> entries_;
    std::vector<HostRegion> hostRegions_;
    std::vector<PreparedHostCopy> preparedHostCopies_;
    Status status_{Status::Open};
};

} // namespace vernon::runtime::program_execution

#endif
