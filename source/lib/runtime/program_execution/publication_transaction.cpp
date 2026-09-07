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

bool decodeImage(uint64_t key, VernonRhiImage &image) {
    const uint32_t encodedIndex = static_cast<uint32_t>(key);
    const uint32_t generation = static_cast<uint32_t>(key >> 32);
    if (!encodedIndex || !generation)
        return false;
    image = {encodedIndex - 1, generation};
    return true;
}

} // namespace

const program::ResolvedPublicationTransaction *
PublicationTransaction::resolve(uint32_t slot, program::PublicationCommitMode mode, std::string &error) const {
    const auto resolved =
        std::find_if(plan_->transactions.begin(), plan_->transactions.end(),
                     [&](const program::ResolvedPublicationTransaction &candidate) { return candidate.slot == slot; });
    if (resolved == plan_->transactions.end() || resolved->mode != mode) {
        error = "publication binding is not authorized by resolved plan";
        return nullptr;
    }
    return &*resolved;
}

bool PublicationTransaction::slotIsBound(uint32_t slot) const {
    return std::any_of(entries_.begin(), entries_.end(), [&](const Entry &entry) {
        return std::visit([&](const auto &value) { return value.transaction->slot == slot; }, entry);
    });
}

bool PublicationTransaction::bindHostCommit(uint32_t slot, const program::PublicationTarget &target,
                                            const VernonProgramArgument &destination, std::string &error) {
    if (status_ != Status::Open || injectFailure(FailureBoundary::Planning))
        return error = "host publication planning failed", false;
    const auto *resolved = resolve(slot, program::PublicationCommitMode::CommitAfterSuccess, error);
    if (!resolved || resolved->value != target.value || destination.kind != VERNON_PROGRAM_TENSOR ||
        destination.tensor.storage != VERNON_TENSOR_HOST || !destination.tensor.host_data || slotIsBound(slot))
        return error = "host publication endpoint does not match its exact resolved slot", false;
    entries_.push_back(HostCommitEntry{resolved, &target, destination});
    return true;
}

bool PublicationTransaction::bindDeviceCommit(uint32_t slot, const program::PublicationTarget &target,
                                              const VernonProgramArgument &destination,
                                              VernonRhiBuffer destinationBuffer, std::string &error) {
    if (status_ != Status::Open || injectFailure(FailureBoundary::Planning))
        return error = "device publication planning failed", false;
    const auto *resolved = resolve(slot, program::PublicationCommitMode::CommitAfterSuccess, error);
    if (!resolved || resolved->value != target.value || destination.kind != VERNON_PROGRAM_TENSOR ||
        destination.tensor.storage != VERNON_TENSOR_RHI_RESOURCE ||
        destinationBuffer.index == VERNON_RHI_INVALID_HANDLE_INDEX || slotIsBound(slot))
        return error = "device publication endpoint does not match its exact resolved slot", false;
    entries_.push_back(DeviceCommitEntry{resolved, &target, destination, destinationBuffer});
    return true;
}

bool PublicationTransaction::bindImageCommit(VernonRuntimeContext &context, uint32_t slot,
                                             const program::PublicationTarget &target,
                                             const VernonProgramArgument &destination, VernonProgramArgument &staging,
                                             std::string &error) {
    if (status_ != Status::Open || injectFailure(FailureBoundary::Planning))
        return error = "image publication planning failed", false;
    const auto *resolved = resolve(slot, program::PublicationCommitMode::CommitAfterSuccess, error);
    if (!resolved || resolved->value != target.value || destination.kind != VERNON_PROGRAM_IMAGE || slotIsBound(slot))
        return error = "image publication endpoint does not match its exact resolved slot", false;
    VernonRhiImageViewDescriptor view{};
    VernonRhiImageDescriptor image{};
    uint64_t parentKey = 0;
    if (!vernon::rhi::describeImageViewResource(context.rhiDevice, destination.image.view.resource.value, view, image,
                                                parentKey))
        return error = "image publication destination is not a referenced RHI image view", false;
    VernonRhiImage destinationImage{};
    if (!decodeImage(parentKey, destinationImage))
        return error = "image publication destination has an invalid parent image", false;
    image.usage |= VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
    VernonRhiImage stagingImage{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    VernonRhiImageView stagingView{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    if (injectFailure(FailureBoundary::Allocation) ||
        vernonRhiDeviceCreateImage(context.rhiDevice, &image, &stagingImage) != VERNON_RHI_STATUS_OK)
        return error = "image publication staging allocation failed", false;
    view.image = stagingImage;
    if (vernonRhiDeviceCreateImageView(context.rhiDevice, &view, &stagingView) != VERNON_RHI_STATUS_OK) {
        (void)vernonRhiDeviceDestroyImage(context.rhiDevice, stagingImage);
        return error = "image publication staging view allocation failed", false;
    }
    staging = destination;
    if (referenceBackendRhiImageView(context, stagingView, staging.image.view) != VERNON_STATUS_OK) {
        (void)vernonRhiDeviceDestroyImageView(context.rhiDevice, stagingView);
        (void)vernonRhiDeviceDestroyImage(context.rhiDevice, stagingImage);
        return error = "image publication cannot reference its staging view", false;
    }
    entries_.push_back(ImageCommitEntry{resolved, &target, destination, context.rhiDevice, destinationImage,
                                        stagingImage, stagingView, image, view});
    return true;
}

bool PublicationTransaction::bindInPlace(uint32_t slot, const VernonProgramArgument &destination, std::string &error) {
    if (status_ != Status::Open || injectFailure(FailureBoundary::Planning))
        return error = "in-place publication planning failed", false;
    const auto *resolved = resolve(slot, program::PublicationCommitMode::InPlace, error);
    const bool writableTensor = destination.kind == VERNON_PROGRAM_TENSOR &&
                                (destination.tensor.storage == VERNON_TENSOR_HOST ||
                                 destination.tensor.storage == VERNON_TENSOR_RHI_RESOURCE) &&
                                destination.tensor.access != VERNON_ACCESS_READ && destination.tensor.byte_size;
    if (!resolved || (!writableTensor && destination.kind != VERNON_PROGRAM_IMAGE) || slotIsBound(slot))
        return error = "in-place publication requires one writable resource bound at its exact resolved slot", false;
    entries_.push_back(InPlaceEntry{resolved, destination});
    return true;
}

bool PublicationTransaction::stageHostRegion(uint32_t slot, void *destination, const void *source, size_t size,
                                             std::string &error) {
    if (status_ != Status::Open || (!destination && size) || (!source && size))
        return error = "host publication region has invalid transaction or endpoint", false;
    if (std::none_of(hostRegions_.begin(), hostRegions_.end(),
                     [&](const HostRegion &region) { return region.transaction->slot == slot; }) &&
        injectFailure(FailureBoundary::Planning))
        return error = "host publication region planning failed", false;
    const auto *resolved = resolve(slot, program::PublicationCommitMode::CommitAfterSuccess, error);
    if (!resolved)
        return false;
    HostRegion region;
    region.transaction = resolved;
    region.destination = destination;
    if (size) {
        const auto *bytes = static_cast<const uint8_t *>(source);
        region.bytes.assign(bytes, bytes + size);
    }
    hostRegions_.push_back(std::move(region));
    return true;
}

bool PublicationTransaction::applyConcreteShapes(const program::Program &program,
                                                 std::vector<ProgramValueState> &values, std::string &error) const {
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
            return error = "publication output has incomplete shape metadata", false;
        for (const program::Value &value : program.values) {
            if (!value.storage || *value.storage != transaction->stagingOwner.id || value.id >= values.size())
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

VernonStatus PublicationTransaction::prepareCommit(const ProgramInvocationState &state,
                                                   std::vector<DeviceBufferCopy> &deviceCopies,
                                                   std::vector<DeviceImageCopy> &imageCopies, std::string &error) {
    if (status_ != Status::Open)
        return error = "publication transaction is not open", VERNON_STATUS_INVALID_ARGUMENT;
    if (injectFailure(FailureBoundary::Readback))
        return rollback(), error = "publication readback boundary failed before destination mutation",
                           VERNON_STATUS_INTERNAL_ERROR;
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
            poison();
            return error = "publication staged Value has no Tensor", VERNON_STATUS_INVALID_ARGUMENT;
        }
        VernonTensorView source = staged->tensor;
        std::vector<ProgramTensorCopyRegion> regions;
        if (const auto *publication = std::get_if<DeviceCommitEntry>(&entry)) {
            const VernonTensorView &destination = publication->destination.tensor;
            const VernonRhiBuffer sourceBuffer = state.buffer(value);
            if (sourceBuffer.index == VERNON_RHI_INVALID_HANDLE_INDEX ||
                (sourceBuffer.index == publication->destinationBuffer.index &&
                 sourceBuffer.generation == publication->destinationBuffer.generation)) {
                poison();
                return error = "commit-after-success device execution backing aliases its destination",
                       VERNON_STATUS_INVALID_ARGUMENT;
            }
            source.byte_offset = 0;
            if (!planProgramTensorCopy(source, destination, regions, error)) {
                poison();
                return VERNON_STATUS_INVALID_ARGUMENT;
            }
            for (const ProgramTensorCopyRegion &region : regions)
                deviceCopies.push_back({sourceBuffer, publication->destinationBuffer, region.sourceOffset,
                                        static_cast<size_t>(destination.resource.offset) + region.destinationOffset,
                                        region.size});
        } else {
            const auto &hostPublication = std::get<HostCommitEntry>(entry);
            const VernonTensorView &destination = hostPublication.destination.tensor;
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
        return rollback(), error = "publication commit failed before destination mutation",
                           VERNON_STATUS_INTERNAL_ERROR;
    status_ = Status::Prepared;
    return VERNON_STATUS_OK;
}

VernonStatus PublicationTransaction::completeCommit(std::string &error) {
    if (status_ != Status::Prepared)
        return error = "publication transaction is not prepared", VERNON_STATUS_INVALID_ARGUMENT;
    for (const PreparedHostCopy &copy : preparedHostCopies_)
        std::memcpy(copy.destination, copy.bytes.data(), copy.bytes.size());
    status_ = Status::Committed;
    releaseDeviceImages();
    entries_.clear();
    hostRegions_.clear();
    preparedHostCopies_.clear();
    return VERNON_STATUS_OK;
}

VernonStatus PublicationTransaction::commit(VernonRuntimeContext &, const ProgramInvocationState &state,
                                            std::string &error) {
    std::vector<DeviceBufferCopy> buffers;
    std::vector<DeviceImageCopy> images;
    const VernonStatus status = prepareCommit(state, buffers, images, error);
    if (status != VERNON_STATUS_OK)
        return status;
    if (!buffers.empty() || !images.empty()) {
        rollback();
        return error = "device publication requires the publication executor", VERNON_STATUS_INVALID_ARGUMENT;
    }
    return completeCommit(error);
}

void PublicationTransaction::rollback() {
    if (status_ == Status::Open || status_ == Status::Prepared) {
        releaseDeviceImages();
        entries_.clear();
        hostRegions_.clear();
        preparedHostCopies_.clear();
        status_ = Status::RolledBack;
    }
}

void PublicationTransaction::poison() {
    releaseDeviceImages();
    entries_.clear();
    hostRegions_.clear();
    preparedHostCopies_.clear();
    status_ = Status::Poisoned;
}

void PublicationTransaction::releaseDeviceImages() {
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
