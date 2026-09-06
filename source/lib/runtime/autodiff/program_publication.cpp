#include "program_publication.h"

#include "runtime_gpu_commands.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

namespace vernon::runtime::ad {
namespace {

bool sameResourceReference(const VernonRuntimeProviderResourceReference &lhs,
                           const VernonRuntimeProviderResourceReference &rhs) {
    return lhs.identity == rhs.identity && lhs.resource.value == rhs.resource.value && lhs.offset == rhs.offset &&
           lhs.size == rhs.size;
}

bool sameBoundaryBinding(const VernonProgramArgument &lhs, const VernonProgramArgument &rhs) {
    if (lhs.kind != rhs.kind)
        return false;
    switch (lhs.kind) {
    case VERNON_PROGRAM_TENSOR:
        if (lhs.tensor.storage != rhs.tensor.storage || lhs.tensor.byte_offset != rhs.tensor.byte_offset ||
            lhs.tensor.byte_size != rhs.tensor.byte_size)
            return false;
        return lhs.tensor.storage == VERNON_TENSOR_HOST
                   ? lhs.tensor.host_data == rhs.tensor.host_data
                   : sameResourceReference(lhs.tensor.resource, rhs.tensor.resource);
    case VERNON_PROGRAM_IMAGE:
        return sameResourceReference(lhs.image.view, rhs.image.view);
    case VERNON_PROGRAM_SAMPLER:
        return sameResourceReference(lhs.resource, rhs.resource);
    }
    return false;
}

struct TensorCopyRegion {
    size_t sourceOffset{};
    size_t destinationOffset{};
    size_t size{};
};

bool planTensorCopy(const VernonTensorView &source, const VernonTensorView &destination,
                    std::vector<TensorCopyRegion> &regions, std::string &error) {
    if (source.rank != destination.rank ||
        (source.rank && (!source.shape || !source.byte_strides || !destination.shape || !destination.byte_strides))) {
        error = "PublicationPlan source and destination tensor layouts disagree";
        return false;
    }
    const size_t elementBytes = source.element_layout.byte_size;
    if (!elementBytes || destination.element_layout.byte_size != elementBytes) {
        error = "PublicationPlan source and destination element layouts disagree";
        return false;
    }
    for (uint32_t axis = 0; axis < source.rank; ++axis)
        if (source.shape[axis] != destination.shape[axis]) {
            error = "PublicationPlan source and destination shapes disagree";
            return false;
        }
    if (source.rank &&
        std::any_of(source.shape, source.shape + source.rank, [](uint64_t extent) { return extent == 0; }))
        return true;

    std::vector<uint64_t> index(source.rank);
    for (;;) {
        if (source.byte_offset > std::numeric_limits<int64_t>::max() ||
            destination.byte_offset > std::numeric_limits<int64_t>::max()) {
            error = "PublicationPlan tensor byte offset exceeds the portable int64 range";
            return false;
        }
        int64_t sourceOffset = static_cast<int64_t>(source.byte_offset);
        int64_t destinationOffset = static_cast<int64_t>(destination.byte_offset);
        const auto advance = [](int64_t &offset, uint64_t coordinate, int64_t stride) {
            if (stride >= 0) {
                const uint64_t positive = static_cast<uint64_t>(stride);
                if (positive &&
                    coordinate > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - offset) / positive)
                    return false;
                offset += static_cast<int64_t>(coordinate * positive);
                return true;
            }
            const uint64_t magnitude = static_cast<uint64_t>(-(stride + 1)) + 1;
            if (magnitude && coordinate > static_cast<uint64_t>(offset) / magnitude)
                return false;
            offset -= static_cast<int64_t>(coordinate * magnitude);
            return true;
        };
        for (uint32_t axis = 0; axis < source.rank; ++axis) {
            if (!advance(sourceOffset, index[axis], source.byte_strides[axis]) ||
                !advance(destinationOffset, index[axis], destination.byte_strides[axis])) {
                error = "PublicationPlan tensor layout overflows the portable int64 range";
                return false;
            }
        }
        const auto validOffset = [&](const int64_t offset, const size_t capacity) {
            return offset >= 0 && static_cast<uint64_t>(offset) <= capacity &&
                   elementBytes <= capacity - static_cast<size_t>(offset);
        };
        if (!validOffset(sourceOffset, source.byte_size) || !validOffset(destinationOffset, destination.byte_size)) {
            error = "PublicationPlan tensor view exceeds its Storage backing";
            return false;
        }
        const size_t sourceByte = static_cast<size_t>(sourceOffset);
        const size_t destinationByte = static_cast<size_t>(destinationOffset);
        if (!regions.empty() && regions.back().sourceOffset + regions.back().size == sourceByte &&
            regions.back().destinationOffset + regions.back().size == destinationByte) {
            regions.back().size += elementBytes;
        } else {
            regions.push_back({sourceByte, destinationByte, elementBytes});
        }
        if (source.rank == 0)
            break;
        uint32_t axis = source.rank;
        while (axis) {
            --axis;
            if (++index[axis] < source.shape[axis])
                break;
            index[axis] = 0;
        }
        if (axis == 0 && index[0] == 0)
            break;
    }
    return true;
}

} // namespace

bool ProgramOwnerBindings::bind(program::ProgramOwnerId owner, const VernonProgramArgument &argument,
                                std::string &error) {
    const auto [binding, inserted] = bindings_.emplace(std::make_pair(owner.kind, owner.id), argument);
    if (!inserted && !sameBoundaryBinding(binding->second, argument)) {
        error = "Program invocation binds one owner to different resources";
        return false;
    }
    return true;
}

bool applyProgramPublicationShapes(const program::Program &program,
                                   const std::vector<PendingProgramPublication> &publications,
                                   std::vector<LogicalProgramValue> &storage, std::string &error) {
    for (const PendingProgramPublication &publication : publications) {
        if (!publication.target || publication.target->aliasOwner.kind != program::ProgramOwnerKind::Storage)
            continue;
        const VernonTensorView &tensor = publication.destination.tensor;
        if (tensor.rank && (!tensor.shape || !tensor.byte_strides)) {
            error = "PublicationPlan output has incomplete runtime shape metadata";
            return false;
        }
        for (const program::Value &value : program.values) {
            if (!value.storage || *value.storage != publication.target->aliasOwner.id || value.id >= storage.size())
                continue;
            storage[value.id].concreteShape = shape::ConcreteShape();
            if (tensor.rank)
                storage[value.id].concreteShape->assign(tensor.shape, tensor.shape + tensor.rank);
        }
    }
    return true;
}

bool commitProgramPublications(const std::vector<LogicalProgramValue> &storage,
                               const std::vector<PendingProgramPublication> &publications, std::string &error) {
    struct PendingCopy {
        void *destination{};
        std::vector<uint8_t> bytes;
    };
    std::vector<PendingCopy> copies;
    copies.reserve(publications.size());
    for (const PendingProgramPublication &publication : publications) {
        if (publication.destinationBuffer)
            continue;
        if (!publication.target || publication.target->value >= storage.size()) {
            error = "PublicationPlan target exceeds its invocation frame";
            return false;
        }
        const VernonProgramArgument &staged = storage[publication.target->value].argument;
        const VernonTensorView &destination = publication.destination.tensor;
        if (staged.kind != VERNON_PROGRAM_TENSOR || staged.tensor.storage != VERNON_TENSOR_HOST ||
            !staged.tensor.host_data || destination.storage != VERNON_TENSOR_HOST || !destination.host_data ||
            staged.tensor.rank != destination.rank) {
            error = "PublicationPlan cannot commit its staged host Storage";
            return false;
        }
        std::vector<TensorCopyRegion> regions;
        if (!planTensorCopy(staged.tensor, destination, regions, error))
            return false;
        for (const TensorCopyRegion &region : regions) {
            const auto *source = static_cast<const uint8_t *>(staged.tensor.host_data) + region.sourceOffset;
            auto *output = static_cast<uint8_t *>(const_cast<void *>(destination.host_data)) + region.destinationOffset;
            copies.push_back({output, std::vector<uint8_t>(source, source + region.size)});
        }
    }
    for (const PendingCopy &copy : copies)
        std::memcpy(copy.destination, copy.bytes.data(), copy.bytes.size());
    return true;
}

namespace {

bool devicePublicationCopies(const LogicalValueFrame &frame, const std::vector<PendingProgramPublication> &publications,
                             std::vector<gpu::DeviceBufferCopy> &copies, std::string &error) {
    for (const PendingProgramPublication &publication : publications) {
        if (!publication.destinationBuffer)
            continue;
        if (!publication.target) {
            error = "PublicationPlan device target is missing";
            return false;
        }
        const VernonProgramArgument *stagedArgument = frame.argument(publication.target->value);
        const VernonRhiBuffer staged = frame.buffer(publication.target->value);
        if (staged.index == VERNON_RHI_INVALID_HANDLE_INDEX) {
            error = "PublicationPlan device target has no staged buffer";
            return false;
        }
        const VernonTensorView &destination = publication.destination.tensor;
        if (!stagedArgument || stagedArgument->kind != VERNON_PROGRAM_TENSOR) {
            error = "PublicationPlan device target has no Tensor view";
            return false;
        }
        VernonTensorView canonicalStaging = stagedArgument->tensor;
        canonicalStaging.byte_offset = 0;
        std::vector<TensorCopyRegion> regions;
        if (!planTensorCopy(canonicalStaging, destination, regions, error))
            return false;
        for (const TensorCopyRegion &region : regions)
            copies.push_back({staged, *publication.destinationBuffer, region.sourceOffset,
                              static_cast<size_t>(destination.resource.offset) + region.destinationOffset,
                              region.size});
    }
    return true;
}

} // namespace

VernonStatus commitDeviceProgramPublications(VernonRuntimeContext &context, const LogicalValueFrame &frame,
                                             const std::vector<PendingProgramPublication> &publications,
                                             std::string &error) {
    std::vector<gpu::DeviceBufferCopy> copies;
    if (!devicePublicationCopies(frame, publications, copies, error))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const VernonStatus status = gpu::executeBufferCopiesAndWait(context, copies);
    if (status != VERNON_STATUS_OK)
        error = "PublicationPlan device commit failed: " + invocationDiagnostic(context);
    return status;
}

} // namespace vernon::runtime::ad
