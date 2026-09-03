#include "program_publication.h"

#include "runtime_gpu_commands.h"

#include <cstdint>
#include <cstring>

namespace vernon::runtime::ad {
namespace {

bool sameResourceReference(const VernonRuntimeProviderResourceReference &lhs,
                           const VernonRuntimeProviderResourceReference &rhs) {
    return lhs.identity == rhs.identity && lhs.resource.value == rhs.resource.value && lhs.offset == rhs.offset &&
           lhs.size == rhs.size;
}

bool sameBoundaryBinding(const VernonPipelineArgument &lhs, const VernonPipelineArgument &rhs) {
    if (lhs.kind != rhs.kind)
        return false;
    switch (lhs.kind) {
    case VERNON_PIPELINE_TENSOR:
        if (lhs.tensor.storage != rhs.tensor.storage || lhs.tensor.byte_offset != rhs.tensor.byte_offset ||
            lhs.tensor.byte_size != rhs.tensor.byte_size)
            return false;
        return lhs.tensor.storage == VERNON_TENSOR_HOST
                   ? lhs.tensor.host_data == rhs.tensor.host_data
                   : sameResourceReference(lhs.tensor.resource, rhs.tensor.resource);
    case VERNON_PIPELINE_IMAGE:
        return sameResourceReference(lhs.image.view, rhs.image.view);
    case VERNON_PIPELINE_SAMPLER:
        return sameResourceReference(lhs.resource, rhs.resource);
    }
    return false;
}

} // namespace

bool ProgramOwnerBindings::bind(program::ProgramOwnerId owner, const VernonPipelineArgument &argument,
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
                                   std::vector<ProgramHostValue> &storage, std::string &error) {
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
            storage[value.id].strides.clear();
            if (tensor.rank) {
                storage[value.id].concreteShape->assign(tensor.shape, tensor.shape + tensor.rank);
                storage[value.id].strides.assign(tensor.byte_strides, tensor.byte_strides + tensor.rank);
            }
        }
    }
    return true;
}

bool initializeProgramPublications(std::vector<ProgramHostValue> &storage,
                                   const std::vector<PendingProgramPublication> &publications, std::string &error) {
    for (const PendingProgramPublication &publication : publications) {
        if (publication.destinationBuffer)
            continue;
        if (!publication.initialValue)
            continue;
        if (!publication.target || publication.target->value >= storage.size()) {
            error = "PublicationPlan target exceeds its invocation frame";
            return false;
        }
        const VernonTensorView &source = publication.initialValue->tensor;
        const VernonPipelineArgument &staged = storage[publication.target->value].argument;
        if (staged.kind != VERNON_PIPELINE_TENSOR || staged.tensor.storage != VERNON_TENSOR_HOST ||
            !staged.tensor.host_data || source.storage != VERNON_TENSOR_HOST || !source.host_data ||
            source.byte_size != staged.tensor.byte_size) {
            error = "PublicationPlan cannot initialize its staged host Storage";
            return false;
        }
        auto *destination = static_cast<uint8_t *>(const_cast<void *>(staged.tensor.host_data));
        const auto *initial = static_cast<const uint8_t *>(source.host_data) + source.byte_offset;
        std::memcpy(destination, initial, source.byte_size);
    }
    return true;
}

bool commitProgramPublications(const std::vector<ProgramHostValue> &storage,
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
        const VernonPipelineArgument &staged = storage[publication.target->value].argument;
        const VernonTensorView &destination = publication.destination.tensor;
        if (staged.kind != VERNON_PIPELINE_TENSOR || staged.tensor.storage != VERNON_TENSOR_HOST ||
            !staged.tensor.host_data || destination.storage != VERNON_TENSOR_HOST || !destination.host_data ||
            destination.byte_size != staged.tensor.byte_size) {
            error = "PublicationPlan cannot commit its staged host Storage";
            return false;
        }
        const auto *source = static_cast<const uint8_t *>(staged.tensor.host_data);
        auto *output = static_cast<uint8_t *>(const_cast<void *>(destination.host_data)) + destination.byte_offset;
        copies.push_back({output, std::vector<uint8_t>(source, source + staged.tensor.byte_size)});
    }
    for (const PendingCopy &copy : copies)
        std::memcpy(copy.destination, copy.bytes.data(), copy.bytes.size());
    return true;
}

namespace {

bool devicePublicationCopies(const ProgramInvocationFrame &frame,
                             const std::vector<PendingProgramPublication> &publications, bool initialize,
                             std::vector<gpu::DeviceBufferCopy> &copies, std::string &error) {
    for (const PendingProgramPublication &publication : publications) {
        if (!publication.destinationBuffer || (initialize && !publication.initialValue))
            continue;
        if (!publication.target) {
            error = "PublicationPlan device target is missing";
            return false;
        }
        const VernonRhiBuffer staged = frame.buffer(publication.target->value);
        if (staged.index == VERNON_RHI_INVALID_HANDLE_INDEX) {
            error = "PublicationPlan device target has no staged buffer";
            return false;
        }
        const VernonTensorView &destination = publication.destination.tensor;
        const size_t destinationOffset = static_cast<size_t>(destination.resource.offset) + destination.byte_offset;
        if (initialize) {
            if (!publication.initialBuffer) {
                error = "PublicationPlan cannot resolve its initial device Storage";
                return false;
            }
            const VernonTensorView &source = publication.initialValue->tensor;
            copies.push_back({*publication.initialBuffer, staged,
                              static_cast<size_t>(source.resource.offset) + source.byte_offset, 0, source.byte_size});
        } else {
            copies.push_back({staged, *publication.destinationBuffer, 0, destinationOffset, destination.byte_size});
        }
    }
    return true;
}

} // namespace

VernonStatus initializeDeviceProgramPublications(VernonRuntimeContext &context, const ProgramInvocationFrame &frame,
                                                 const std::vector<PendingProgramPublication> &publications,
                                                 std::string &error) {
    std::vector<gpu::DeviceBufferCopy> copies;
    if (!devicePublicationCopies(frame, publications, true, copies, error))
        return VERNON_STATUS_INVALID_ARGUMENT;
    return gpu::executeBufferCopiesAndWait(context, copies);
}

VernonStatus commitDeviceProgramPublications(VernonRuntimeContext &context, const ProgramInvocationFrame &frame,
                                             const std::vector<PendingProgramPublication> &publications,
                                             std::string &error) {
    std::vector<gpu::DeviceBufferCopy> copies;
    if (!devicePublicationCopies(frame, publications, false, copies, error))
        return VERNON_STATUS_INVALID_ARGUMENT;
    return gpu::executeBufferCopiesAndWait(context, copies);
}

} // namespace vernon::runtime::ad
