#include "publication_transaction.h"

#include "device_commands.h"
#include "program_tensor_copy.h"
#include "runtime/runtime_state.h"

#include <algorithm>
#include <cstring>

namespace vernon::runtime::program_execution {

bool PublicationTransaction::stage(uint32_t slot, const program::PublicationTarget &target,
                                   const VernonProgramArgument &destination,
                                   std::optional<VernonRhiBuffer> destinationBuffer, std::string &error) {
    if (status_ != Status::Open)
        return error = "publication transaction is no longer open", false;
    const auto resolved =
        std::find_if(plan_->transactions.begin(), plan_->transactions.end(),
                     [&](const program::ResolvedPublicationTransaction &candidate) { return candidate.slot == slot; });
    if (resolved == plan_->transactions.end() || resolved->mode != program::PublicationCommitMode::CommitAfterSuccess ||
        resolved->value != target.value)
        return error = "publication staging is not authorized by resolved plan", false;
    if (std::any_of(staged_.begin(), staged_.end(),
                    [&](const StagedPublication &candidate) { return candidate.transaction->slot == slot; }))
        return error = "publication slot was staged more than once", false;
    staged_.push_back({&*resolved, &target, destination, destinationBuffer});
    return true;
}

bool PublicationTransaction::stageHostBytes(void *destination, const void *source, size_t size, std::string &error) {
    if (status_ != Status::Open || (!destination && size) || (!source && size))
        return error = "host publication has invalid transaction or endpoint", false;
    StagedHostCopy copy;
    copy.destination = destination;
    if (size) {
        const auto *bytes = static_cast<const uint8_t *>(source);
        copy.bytes.assign(bytes, bytes + size);
    }
    stagedHostCopies_.push_back(std::move(copy));
    return true;
}

bool PublicationTransaction::applyConcreteShapes(const program::Program &program,
                                                 std::vector<ProgramValueState> &values, std::string &error) const {
    for (const StagedPublication &publication : staged_) {
        if (publication.transaction->stagingOwner.kind != program::ProgramOwnerKind::Storage)
            continue;
        const VernonTensorView &tensor = publication.destination.tensor;
        if (tensor.rank && (!tensor.shape || !tensor.byte_strides))
            return error = "publication output has incomplete shape metadata", false;
        for (const program::Value &value : program.values) {
            if (!value.storage || *value.storage != publication.transaction->stagingOwner.id ||
                value.id >= values.size())
                continue;
            values[value.id].concreteShape = shape::ConcreteShape();
            if (tensor.rank)
                values[value.id].concreteShape->assign(tensor.shape, tensor.shape + tensor.rank);
        }
    }
    return true;
}

std::vector<char> PublicationTransaction::hostReadbackValues(size_t valueCount) const {
    std::vector<char> result(valueCount);
    for (const StagedPublication &publication : staged_)
        if (!publication.destinationBuffer && publication.transaction->value < result.size())
            result[publication.transaction->value] = 1;
    return result;
}

VernonStatus PublicationTransaction::commit(VernonRuntimeContext &context, const ProgramInvocationState &state,
                                            std::string &error) {
    if (status_ != Status::Open)
        return error = "publication transaction is not open", VERNON_STATUS_INVALID_ARGUMENT;
    struct HostCopy {
        void *destination{};
        std::vector<uint8_t> bytes;
    };
    std::vector<HostCopy> hostCopies;
    for (const StagedHostCopy &copy : stagedHostCopies_)
        hostCopies.push_back({copy.destination, copy.bytes});
    std::vector<DeviceBufferCopy> deviceCopies;
    for (const StagedPublication &publication : staged_) {
        const uint32_t value = publication.transaction->value;
        const VernonProgramArgument *staged = publication.destinationBuffer   ? state.argument(value)
                                              : value < state.values().size() ? &state.values()[value].argument
                                                                              : nullptr;
        if (!staged || staged->kind != VERNON_PROGRAM_TENSOR) {
            poison();
            return error = "publication staged Value has no Tensor", VERNON_STATUS_INVALID_ARGUMENT;
        }
        VernonTensorView source = staged->tensor;
        const VernonTensorView &destination = publication.destination.tensor;
        std::vector<ProgramTensorCopyRegion> regions;
        if (publication.destinationBuffer) {
            const VernonRhiBuffer sourceBuffer = state.buffer(value);
            if (sourceBuffer.index == VERNON_RHI_INVALID_HANDLE_INDEX) {
                poison();
                return error = "publication staged Value has no device buffer", VERNON_STATUS_INVALID_ARGUMENT;
            }
            source.byte_offset = 0;
            if (!planProgramTensorCopy(source, destination, regions, error)) {
                poison();
                return VERNON_STATUS_INVALID_ARGUMENT;
            }
            for (const ProgramTensorCopyRegion &region : regions)
                deviceCopies.push_back({sourceBuffer, *publication.destinationBuffer, region.sourceOffset,
                                        static_cast<size_t>(destination.resource.offset) + region.destinationOffset,
                                        region.size});
        } else {
            if (source.storage != VERNON_TENSOR_HOST || !source.host_data ||
                destination.storage != VERNON_TENSOR_HOST || !destination.host_data ||
                !planProgramTensorCopy(source, destination, regions, error)) {
                poison();
                return VERNON_STATUS_INVALID_ARGUMENT;
            }
            for (const ProgramTensorCopyRegion &region : regions) {
                const auto *bytes = static_cast<const uint8_t *>(source.host_data) + region.sourceOffset;
                auto *output =
                    static_cast<uint8_t *>(const_cast<void *>(destination.host_data)) + region.destinationOffset;
                hostCopies.push_back({output, std::vector<uint8_t>(bytes, bytes + region.size)});
            }
        }
    }
    if (!deviceCopies.empty()) {
        const VernonStatus status = executeBufferCopiesAndWait(context, deviceCopies);
        if (status != VERNON_STATUS_OK) {
            poison();
            error = "publication device commit failed: " + vernon::runtime::invocationDiagnostic(context);
            return status;
        }
    }
    for (const HostCopy &copy : hostCopies)
        std::memcpy(copy.destination, copy.bytes.data(), copy.bytes.size());
    status_ = Status::Committed;
    staged_.clear();
    stagedHostCopies_.clear();
    return VERNON_STATUS_OK;
}

void PublicationTransaction::rollback() {
    if (status_ == Status::Open) {
        staged_.clear();
        stagedHostCopies_.clear();
        status_ = Status::RolledBack;
    }
}

void PublicationTransaction::poison() {
    staged_.clear();
    stagedHostCopies_.clear();
    status_ = Status::Poisoned;
}

} // namespace vernon::runtime::program_execution
