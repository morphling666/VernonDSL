#include "publication_transaction.h"

#include "device_commands.h"
#include "failure_injection.h"
#include "program_tensor_copy.h"
#include "rhi/rhi_internal.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_state.h"

#include <algorithm>
#include <cstring>

namespace vernon::runtime::program_execution {
namespace {

vernon::Option<VernonRhiImage> decodeImage(uint64_t key) {
    const uint32_t encodedIndex = static_cast<uint32_t>(key);
    const uint32_t generation = static_cast<uint32_t>(key >> 32);
    if (!encodedIndex || !generation)
        return {};
    return vernon::Option<VernonRhiImage>{vernon::some(VernonRhiImage{encodedIndex - 1, generation})};
}

} // namespace

PublicationResult<std::reference_wrapper<const program::ResolvedPublicationTransaction>>
PublicationTransaction::resolve(uint32_t slot, program::PublicationCommitMode mode) const {
    const auto resolved =
        std::find_if(plan_->transactions.begin(), plan_->transactions.end(),
                     [&](const program::ResolvedPublicationTransaction &candidate) { return candidate.slot == slot; });
    if (resolved == plan_->transactions.end() || resolved->mode != mode)
        return PublicationResult<std::reference_wrapper<const program::ResolvedPublicationTransaction>>{
            vernon::err(PublicationError::UnauthorizedBinding)};
    return PublicationResult<std::reference_wrapper<const program::ResolvedPublicationTransaction>>{
        vernon::ok(std::cref(*resolved))};
}

bool PublicationTransaction::slotIsBound(uint32_t slot) const {
    return std::any_of(entries_.begin(), entries_.end(), [&](const Entry &entry) {
        return std::visit([&](const auto &value) { return value.transaction->slot == slot; }, entry);
    });
}

PublicationResult<void> PublicationTransaction::bindHostCommit(uint32_t slot, const program::PublicationTarget &target,
                                                               const VernonProgramArgument &destination) {
    if (status_ != Status::Open || injectFailure(FailureBoundary::Planning))
        return PublicationResult<void>{vernon::err(PublicationError::PlanningFailed)};
    auto resolved = resolve(slot, program::PublicationCommitMode::CommitAfterSuccess);
    if (resolved.isErr())
        return PublicationResult<void>{vernon::err(resolved.error())};
    if (resolved.value().get().value != target.value || destination.kind != VERNON_PROGRAM_TENSOR ||
        destination.tensor.storage != VERNON_TENSOR_HOST || !destination.tensor.host_data || slotIsBound(slot))
        return PublicationResult<void>{vernon::err(PublicationError::EndpointMismatch)};
    entries_.push_back(HostCommitEntry{&resolved.value().get(), &target, destination});
    boundMutations_.push_back({slot, false});
    return PublicationResult<void>{vernon::ok()};
}

PublicationResult<void> PublicationTransaction::bindDeviceCommit(uint32_t slot,
                                                                 const program::PublicationTarget &target,
                                                                 const VernonProgramArgument &destination,
                                                                 VernonRhiBuffer destinationBuffer) {
    if (status_ != Status::Open || injectFailure(FailureBoundary::Planning))
        return PublicationResult<void>{vernon::err(PublicationError::PlanningFailed)};
    auto resolved = resolve(slot, program::PublicationCommitMode::CommitAfterSuccess);
    if (resolved.isErr())
        return PublicationResult<void>{vernon::err(resolved.error())};
    if (resolved.value().get().value != target.value || destination.kind != VERNON_PROGRAM_TENSOR ||
        destination.tensor.storage != VERNON_TENSOR_RHI_RESOURCE ||
        destinationBuffer.index == VERNON_RHI_INVALID_HANDLE_INDEX || slotIsBound(slot))
        return PublicationResult<void>{vernon::err(PublicationError::EndpointMismatch)};
    entries_.push_back(DeviceCommitEntry{&resolved.value().get(), &target, destination, destinationBuffer});
    boundMutations_.push_back({slot, false});
    return PublicationResult<void>{vernon::ok()};
}

PublicationResult<void> PublicationTransaction::bindImageCommit(VernonRuntimeContext &context, uint32_t slot,
                                                                const program::PublicationTarget &target,
                                                                const VernonProgramArgument &destination,
                                                                VernonProgramArgument &staging) {
    if (status_ != Status::Open || injectFailure(FailureBoundary::Planning))
        return PublicationResult<void>{vernon::err(PublicationError::PlanningFailed)};
    auto resolved = resolve(slot, program::PublicationCommitMode::CommitAfterSuccess);
    if (resolved.isErr())
        return PublicationResult<void>{vernon::err(resolved.error())};
    if (resolved.value().get().value != target.value || destination.kind != VERNON_PROGRAM_IMAGE || slotIsBound(slot))
        return PublicationResult<void>{vernon::err(PublicationError::EndpointMismatch)};
    VernonRhiImageViewDescriptor view{};
    VernonRhiImageDescriptor image{};
    uint64_t parentKey = 0;
    if (!vernon::rhi::describeImageViewResource(context.rhiDevice, destination.image.view.resource.value, view, image,
                                                parentKey))
        return PublicationResult<void>{vernon::err(PublicationError::InvalidImageDestination)};
    auto destinationImage = decodeImage(parentKey);
    if (!destinationImage)
        return PublicationResult<void>{vernon::err(PublicationError::InvalidImageParent)};
    image.usage |= VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
    VernonRhiImage stagingImage{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    VernonRhiImageView stagingView{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    if (injectFailure(FailureBoundary::Allocation) ||
        vernonRhiDeviceCreateImage(context.rhiDevice, &image, &stagingImage) != VERNON_RHI_STATUS_OK)
        return PublicationResult<void>{vernon::err(PublicationError::ImageAllocationFailed)};
    view.image = stagingImage;
    if (vernonRhiDeviceCreateImageView(context.rhiDevice, &view, &stagingView) != VERNON_RHI_STATUS_OK) {
        (void)vernonRhiDeviceDestroyImage(context.rhiDevice, stagingImage);
        return PublicationResult<void>{vernon::err(PublicationError::ImageViewAllocationFailed)};
    }
    staging = destination;
    auto referenced = referenceBackendRhiImageView(context, stagingView);
    if (referenced.isErr()) {
        (void)vernonRhiDeviceDestroyImageView(context.rhiDevice, stagingView);
        (void)vernonRhiDeviceDestroyImage(context.rhiDevice, stagingImage);
        return PublicationResult<void>{vernon::err(PublicationError::ImageReferenceFailed)};
    }
    staging.image.view = referenced.value();
    entries_.push_back(ImageCommitEntry{&resolved.value().get(), &target, destination, context.rhiDevice,
                                        destinationImage.value(), stagingImage, stagingView, image, view});
    boundMutations_.push_back({slot, false});
    return PublicationResult<void>{vernon::ok()};
}

PublicationResult<void> PublicationTransaction::bindInPlace(uint32_t slot, const VernonProgramArgument &destination) {
    if (status_ != Status::Open || injectFailure(FailureBoundary::Planning))
        return PublicationResult<void>{vernon::err(PublicationError::PlanningFailed)};
    auto resolved = resolve(slot, program::PublicationCommitMode::InPlace);
    if (resolved.isErr())
        return PublicationResult<void>{vernon::err(resolved.error())};
    const bool writableTensor = destination.kind == VERNON_PROGRAM_TENSOR &&
                                (destination.tensor.storage == VERNON_TENSOR_HOST ||
                                 destination.tensor.storage == VERNON_TENSOR_RHI_RESOURCE) &&
                                destination.tensor.access != VERNON_ACCESS_READ && destination.tensor.byte_size;
    if ((!writableTensor && destination.kind != VERNON_PROGRAM_IMAGE) || slotIsBound(slot))
        return PublicationResult<void>{vernon::err(PublicationError::EndpointMismatch)};
    entries_.push_back(InPlaceEntry{&resolved.value().get(), destination});
    boundMutations_.push_back({slot, true});
    return PublicationResult<void>{vernon::ok()};
}

PublicationResult<void> PublicationTransaction::stageHostRegion(uint32_t slot, void *destination, const void *source,
                                                                size_t size) {
    if (status_ != Status::Open || (!destination && size) || (!source && size))
        return PublicationResult<void>{vernon::err(PublicationError::InvalidHostRegion)};
    if (std::none_of(hostRegions_.begin(), hostRegions_.end(),
                     [&](const HostRegion &region) { return region.transaction->slot == slot; }) &&
        injectFailure(FailureBoundary::Planning))
        return PublicationResult<void>{vernon::err(PublicationError::PlanningFailed)};
    auto resolved = resolve(slot, program::PublicationCommitMode::CommitAfterSuccess);
    if (resolved.isErr())
        return PublicationResult<void>{vernon::err(resolved.error())};
    HostRegion region;
    region.transaction = &resolved.value().get();
    region.destination = destination;
    if (size) {
        const auto *bytes = static_cast<const uint8_t *>(source);
        region.bytes.assign(bytes, bytes + size);
    }
    hostRegions_.push_back(std::move(region));
    return PublicationResult<void>{vernon::ok()};
}

PublicationResult<void> PublicationTransaction::applyConcreteShapes(const program::Program &program,
                                                                    std::vector<ProgramValueState> &values) const {
    for (const Entry &entry : entries_) {
        if (!std::holds_alternative<HostCommitEntry>(entry) && !std::holds_alternative<DeviceCommitEntry>(entry))
            continue;
        const auto &publication = std::holds_alternative<HostCommitEntry>(entry)
                                      ? std::get<HostCommitEntry>(entry).destination
                                      : std::get<DeviceCommitEntry>(entry).destination;
        const auto *transaction = std::visit([](const auto &value) { return value.transaction; }, entry);
        if (transaction->stagingOwner.kind != program::ProgramOwnerKind::Storage)
            continue;
        const VernonTensorView &tensor = publication.tensor;
        if (tensor.rank && (!tensor.shape || !tensor.byte_strides))
            return PublicationResult<void>{vernon::err(PublicationError::IncompleteShape)};
        for (const program::Value &value : program.values) {
            if (!value.storage || *value.storage != transaction->stagingOwner.id || value.id >= values.size())
                continue;
            values[value.id].concreteShape = shape::ConcreteShape();
            if (tensor.rank)
                values[value.id].concreteShape->assign(tensor.shape, tensor.shape + tensor.rank);
        }
    }
    return PublicationResult<void>{vernon::ok()};
}

std::vector<char> PublicationTransaction::hostReadbackValues(size_t valueCount) const {
    std::vector<char> result(valueCount);
    for (const Entry &entry : entries_)
        if (const auto *publication = std::get_if<HostCommitEntry>(&entry);
            publication && publication->transaction->value < result.size())
            result[publication->transaction->value] = 1;
    return result;
}

std::vector<DeviceImageCopy> PublicationTransaction::initializationImageCopies() const {
    std::vector<DeviceImageCopy> copies;
    for (const Entry &entry : entries_) {
        const auto *publication = std::get_if<ImageCommitEntry>(&entry);
        if (!publication)
            continue;
        DeviceImageCopy copy{publication->destinationImage, publication->stagingImage, {}};
        for (uint32_t layer = 0; layer < publication->viewDescriptor.array_layer_count; ++layer)
            for (uint32_t mip = 0; mip < publication->viewDescriptor.mip_level_count; ++mip) {
                const uint32_t mipLevel = publication->viewDescriptor.base_mip_level + mip;
                const uint32_t arrayLayer = publication->viewDescriptor.base_array_layer + layer;
                const uint32_t width = std::max(publication->imageDescriptor.width >> mipLevel, 1u);
                const uint32_t height = std::max(publication->imageDescriptor.height >> mipLevel, 1u);
                const uint32_t depth = publication->imageDescriptor.dimension == VERNON_RHI_IMAGE_3D
                                           ? std::max(publication->imageDescriptor.depth >> mipLevel, 1u)
                                           : 1u;
                copy.regions.push_back({sizeof(VernonRhiImageCopyRegion),
                                        mipLevel,
                                        arrayLayer,
                                        0,
                                        0,
                                        0,
                                        mipLevel,
                                        arrayLayer,
                                        0,
                                        0,
                                        0,
                                        width,
                                        height,
                                        depth,
                                        publication->viewDescriptor.aspects,
                                        {}});
            }
        copies.push_back(std::move(copy));
    }
    return copies;
}

PublicationResult<void> PublicationTransaction::prepareCommit(const ProgramInvocationState &state,
                                                              std::vector<DeviceBufferCopy> &deviceCopies,
                                                              std::vector<DeviceImageCopy> &imageCopies) {
    if (status_ != Status::Open)
        return PublicationResult<void>{vernon::err(PublicationError::TransactionNotOpen)};
    if (injectFailure(FailureBoundary::Readback))
        return rollback(), PublicationResult<void>{vernon::err(PublicationError::ReadbackFailed)};
    preparedHostCopies_.clear();
    deviceCopies.clear();
    imageCopies.clear();
    for (const HostRegion &copy : hostRegions_)
        preparedHostCopies_.push_back({copy.destination, copy.bytes});
    for (const Entry &entry : entries_) {
        if (std::holds_alternative<InPlaceEntry>(entry))
            continue;
        if (std::holds_alternative<ImageCommitEntry>(entry))
            continue;
        const auto *transaction = std::visit([](const auto &value) { return value.transaction; }, entry);
        const uint32_t value = transaction->value;
        const VernonProgramArgument *staged = value < state.values().size() ? &state.values()[value].argument : nullptr;
        if (!staged || staged->kind != VERNON_PROGRAM_TENSOR) {
            rollback();
            return PublicationResult<void>{vernon::err(PublicationError::InvalidStagedValue)};
        }
        VernonTensorView source = staged->tensor;
        std::vector<ProgramTensorCopyRegion> regions;
        if (const auto *publication = std::get_if<DeviceCommitEntry>(&entry)) {
            const VernonTensorView &destination = publication->destination.tensor;
            auto sourceBuffer = state.buffer(value);
            if (!sourceBuffer || (sourceBuffer.value().index == publication->destinationBuffer.index &&
                                  sourceBuffer.value().generation == publication->destinationBuffer.generation)) {
                rollback();
                return PublicationResult<void>{vernon::err(PublicationError::AliasedDeviceBacking)};
            }
            source.byte_offset = 0;
            std::string error;
            if (!planProgramTensorCopy(source, destination, regions, error)) {
                rollback();
                return PublicationResult<void>{vernon::err(PublicationError::TensorCopyInvalid)};
            }
            for (const ProgramTensorCopyRegion &region : regions)
                deviceCopies.push_back({sourceBuffer.value(), publication->destinationBuffer, region.sourceOffset,
                                        static_cast<size_t>(destination.resource.offset) + region.destinationOffset,
                                        region.size});
        } else {
            const auto &hostPublication = std::get<HostCommitEntry>(entry);
            const VernonTensorView &destination = hostPublication.destination.tensor;
            std::string error;
            if (source.storage != VERNON_TENSOR_HOST || !source.host_data ||
                destination.storage != VERNON_TENSOR_HOST || !destination.host_data ||
                !planProgramTensorCopy(source, destination, regions, error)) {
                rollback();
                return PublicationResult<void>{vernon::err(PublicationError::TensorCopyInvalid)};
            }
            for (const ProgramTensorCopyRegion &region : regions) {
                const auto *bytes = static_cast<const uint8_t *>(source.host_data) + region.sourceOffset;
                auto *output =
                    static_cast<uint8_t *>(const_cast<void *>(destination.host_data)) + region.destinationOffset;
                preparedHostCopies_.push_back({output, std::vector<uint8_t>(bytes, bytes + region.size)});
            }
        }
    }
    for (const Entry &entry : entries_) {
        const auto *publication = std::get_if<ImageCommitEntry>(&entry);
        if (!publication)
            continue;
        DeviceImageCopy copy{publication->stagingImage, publication->destinationImage, {}};
        for (uint32_t layer = 0; layer < publication->viewDescriptor.array_layer_count; ++layer)
            for (uint32_t mip = 0; mip < publication->viewDescriptor.mip_level_count; ++mip) {
                const uint32_t mipLevel = publication->viewDescriptor.base_mip_level + mip;
                const uint32_t arrayLayer = publication->viewDescriptor.base_array_layer + layer;
                const uint32_t width = std::max(publication->imageDescriptor.width >> mipLevel, 1u);
                const uint32_t height = std::max(publication->imageDescriptor.height >> mipLevel, 1u);
                const uint32_t depth = publication->imageDescriptor.dimension == VERNON_RHI_IMAGE_3D
                                           ? std::max(publication->imageDescriptor.depth >> mipLevel, 1u)
                                           : 1u;
                copy.regions.push_back({sizeof(VernonRhiImageCopyRegion),
                                        mipLevel,
                                        arrayLayer,
                                        0,
                                        0,
                                        0,
                                        mipLevel,
                                        arrayLayer,
                                        0,
                                        0,
                                        0,
                                        width,
                                        height,
                                        depth,
                                        publication->viewDescriptor.aspects,
                                        {}});
            }
        imageCopies.push_back(std::move(copy));
    }
    if (injectFailure(FailureBoundary::Commit))
        return rollback(), PublicationResult<void>{vernon::err(PublicationError::CommitFailed)};
    status_ = Status::Prepared;
    return PublicationResult<void>{vernon::ok()};
}

PublicationResult<void> PublicationTransaction::completeCommit() {
    if (status_ != Status::Prepared)
        return PublicationResult<void>{vernon::err(PublicationError::TransactionNotPrepared)};
    for (const PreparedHostCopy &copy : preparedHostCopies_)
        std::memcpy(copy.destination, copy.bytes.data(), copy.bytes.size());
    status_ = Status::Committed;
    releaseDeviceImages();
    entries_.clear();
    hostRegions_.clear();
    preparedHostCopies_.clear();
    return PublicationResult<void>{vernon::ok()};
}

PublicationResult<void> PublicationTransaction::commit(VernonRuntimeContext &, const ProgramInvocationState &state) {
    std::vector<DeviceBufferCopy> buffers;
    std::vector<DeviceImageCopy> images;
    auto prepared = prepareCommit(state, buffers, images);
    if (prepared.isErr())
        return PublicationResult<void>{vernon::err(prepared.error())};
    if (!buffers.empty() || !images.empty()) {
        rollback();
        return PublicationResult<void>{vernon::err(PublicationError::ExecutorRequired)};
    }
    return completeCommit();
}

const char *publicationErrorMessage(PublicationError error) noexcept {
    switch (error) {
    case PublicationError::TransactionNotOpen:
        return "publication transaction is not open";
    case PublicationError::TransactionNotPrepared:
        return "publication transaction is not prepared";
    case PublicationError::PlanningFailed:
        return "publication planning failed";
    case PublicationError::UnauthorizedBinding:
        return "publication binding is not authorized by resolved plan";
    case PublicationError::EndpointMismatch:
        return "publication endpoint does not match its exact resolved slot";
    case PublicationError::InvalidHostRegion:
        return "host publication region has an invalid transaction or endpoint";
    case PublicationError::IncompleteShape:
        return "publication output has incomplete shape metadata";
    case PublicationError::InvalidStagedValue:
        return "publication staged Value has no Tensor";
    case PublicationError::AliasedDeviceBacking:
        return "commit-after-success device execution backing aliases its destination";
    case PublicationError::TensorCopyInvalid:
        return "publication Tensor copy is invalid";
    case PublicationError::InvalidImageDestination:
        return "image publication destination is not a referenced RHI image view";
    case PublicationError::InvalidImageParent:
        return "image publication destination has an invalid parent image";
    case PublicationError::ImageAllocationFailed:
        return "image publication staging allocation failed";
    case PublicationError::ImageViewAllocationFailed:
        return "image publication staging view allocation failed";
    case PublicationError::ImageReferenceFailed:
        return "image publication cannot reference its staging view";
    case PublicationError::ReadbackFailed:
        return "publication readback boundary failed before destination mutation";
    case PublicationError::CommitFailed:
        return "publication commit failed before destination mutation";
    case PublicationError::ExecutorRequired:
        return "device publication requires the publication executor";
    }
    return "unknown publication error";
}

VernonStatus publicationErrorStatus(PublicationError error) noexcept {
    return error == PublicationError::PlanningFailed || error == PublicationError::ReadbackFailed ||
                   error == PublicationError::CommitFailed
               ? VERNON_STATUS_INTERNAL_ERROR
               : VERNON_STATUS_INVALID_ARGUMENT;
}

void PublicationTransaction::rollback() noexcept {
    if (status_ == Status::Open || status_ == Status::Prepared) {
        releaseDeviceImages();
        entries_.clear();
        hostRegions_.clear();
        preparedHostCopies_.clear();
        status_ = Status::RolledBack;
    }
}

void PublicationTransaction::poison() noexcept {
    releaseDeviceImages();
    entries_.clear();
    hostRegions_.clear();
    preparedHostCopies_.clear();
    status_ = Status::Poisoned;
}

void PublicationTransaction::noteSubmission(SubmissionState state) {
    if (overallSubmission_ == SubmissionState::Indeterminate)
        return;
    if (state == SubmissionState::Indeterminate || state == SubmissionState::Completed)
        overallSubmission_ = state;
}

void PublicationTransaction::noteInPlaceSubmission(SubmissionState state) {
    noteSubmission(state);
    if (inPlaceSubmission_ != SubmissionState::Indeterminate &&
        (state == SubmissionState::Indeterminate || state == SubmissionState::Completed))
        inPlaceSubmission_ = state;
}

InvocationMutationOutcome PublicationTransaction::mutationOutcome() const {
    InvocationMutationOutcome outcome;
    outcome.submission = status_ == Status::Poisoned || overallSubmission_ == SubmissionState::Indeterminate
                             ? SubmissionState::Indeterminate
                         : status_ == Status::Committed || overallSubmission_ == SubmissionState::Completed
                             ? SubmissionState::Completed
                             : SubmissionState::NotSubmitted;
    outcome.boundaries.reserve(boundMutations_.size());
    for (const BoundMutation &entry : boundMutations_) {
        VernonBoundaryMutationState state = VERNON_BOUNDARY_MUTATION_UNCHANGED;
        if (entry.inPlace) {
            if (inPlaceSubmission_ == SubmissionState::Completed)
                state = VERNON_BOUNDARY_MUTATION_IN_PLACE_COMMITTED;
            else if (inPlaceSubmission_ == SubmissionState::Indeterminate)
                state = VERNON_BOUNDARY_MUTATION_INDETERMINATE;
        } else if (status_ == Status::Committed) {
            state = VERNON_BOUNDARY_MUTATION_COMMITTED;
        } else if (status_ == Status::Poisoned) {
            state = VERNON_BOUNDARY_MUTATION_INDETERMINATE;
        }
        outcome.set(entry.slot, state);
    }
    return outcome;
}

void PublicationTransaction::releaseDeviceImages() noexcept {
    for (Entry &entry : entries_)
        if (auto *publication = std::get_if<ImageCommitEntry>(&entry)) {
            if (publication->stagingView.index != VERNON_RHI_INVALID_HANDLE_INDEX)
                (void)vernonRhiDeviceDestroyImageView(publication->device, publication->stagingView);
            if (publication->stagingImage.index != VERNON_RHI_INVALID_HANDLE_INDEX)
                (void)vernonRhiDeviceDestroyImage(publication->device, publication->stagingImage);
            publication->stagingView = {VERNON_RHI_INVALID_HANDLE_INDEX, 0};
            publication->stagingImage = {VERNON_RHI_INVALID_HANDLE_INDEX, 0};
        }
}

} // namespace vernon::runtime::program_execution
