#include "program_tape_scratch.h"

#include "runtime/autodiff/runtime_gpu_argument_binding.h"
#include "runtime/tensor_bridge.h"

#include <algorithm>

namespace vernon::runtime::ad {

ProgramTapeScratch::ProgramTapeScratch(size_t valueCount) : hostBatches_(valueCount), carriers_(valueCount) {}

void ProgramTapeScratch::setHostBatch(uint32_t value, std::shared_ptr<HostStaticTapeBatch> batch) {
    if (value < hostBatches_.size())
        hostBatches_[value] = std::move(batch);
}

std::shared_ptr<HostStaticTapeBatch> ProgramTapeScratch::hostBatch(uint32_t value) const {
    return value < hostBatches_.size() ? hostBatches_[value] : nullptr;
}

std::vector<std::shared_ptr<HostStaticTapeBatch>> ProgramTapeScratch::releaseHostBatches() {
    return std::move(hostBatches_);
}

void ProgramTapeScratch::importHostBatches(const std::vector<std::shared_ptr<HostStaticTapeBatch>> &batches) {
    hostBatches_ = batches;
    hostBatches_.resize(carriers_.size());
}

ProgramTapeSnapshot ProgramTapeScratch::releaseSnapshot() {
    ProgramTapeSnapshot snapshot;
    snapshot.hostBatches.reserve(hostBatches_.size());
    for (auto &batch : hostBatches_)
        snapshot.hostBatches.push_back(std::move(batch));
    hostBatches_.clear();
    snapshot.carriers.resize(carriers_.size());
    for (size_t value = 0; value < carriers_.size(); ++value)
        for (size_t index = 0; index < carriers_[value].size(); ++index) {
            Carrier &source = carriers_[value][index];
            auto &destination = snapshot.carriers[value][index];
            destination.owner = std::move(source.buffer);
            destination.argument = source.argument;
            destination.shape = std::move(source.shape);
            destination.strides = std::move(source.strides);
        }
    return snapshot;
}

void ProgramTapeScratch::importSnapshot(const ProgramTapeSnapshot &snapshot) {
    hostBatches_.assign(carriers_.size(), nullptr);
    const size_t valueCount = std::min(carriers_.size(), snapshot.carriers.size());
    for (size_t value = 0; value < valueCount; ++value)
        for (size_t index = 0; index < carriers_[value].size(); ++index) {
            Carrier &destination = carriers_[value][index];
            const auto &source = snapshot.carriers[value][index];
            destination.buffer.reset();
            destination.retainedBuffer = source.owner;
            destination.argument = source.argument;
            destination.shape = source.shape;
            destination.strides = source.strides;
            destination.readOnly = static_cast<bool>(source.owner);
        }
}

size_t ProgramTapeScratch::carrierIndex(program_plan::TapeCarrier carrier) {
    switch (carrier) {
    case program_plan::TapeCarrier::TapeData:
        return 0;
    case program_plan::TapeCarrier::ReplaySegment:
        return 1;
    case program_plan::TapeCarrier::ReplayStatus:
        return 2;
    case program_plan::TapeCarrier::LaunchMetadata:
        return 3;
    }
    return 4;
}

ProgramTapeScratch::Carrier *ProgramTapeScratch::carrier(uint32_t value, program_plan::TapeCarrier kind) {
    const size_t index = carrierIndex(kind);
    return value < carriers_.size() && index < 4 ? &carriers_[value][index] : nullptr;
}

const ProgramTapeScratch::Carrier *ProgramTapeScratch::carrier(uint32_t value, program_plan::TapeCarrier kind) const {
    const size_t index = carrierIndex(kind);
    return value < carriers_.size() && index < 4 ? &carriers_[value][index] : nullptr;
}

bool ProgramTapeScratch::allocateCarrier(VernonRuntimeContext &context, uint32_t value,
                                         const program::TargetBinding &binding, size_t byteSize,
                                         std::vector<uint64_t> shape, std::vector<int64_t> strides,
                                         std::string &error) {
    Carrier *slot = binding.tapeCarrier ? carrier(value, *binding.tapeCarrier) : nullptr;
    if (!slot || !byteSize || shape.size() != strides.size())
        return error = "invalid Program tape carrier allocation", false;
    slot->buffer = std::make_shared<program_execution::DeviceBuffer>(context, byteSize);
    slot->retainedBuffer.reset();
    slot->readOnly = false;
    if (!slot->buffer->valid())
        return error = "Program tape carrier allocation failed", false;
    slot->shape = std::move(shape);
    slot->strides = std::move(strides);
    slot->argument.kind = VERNON_PROGRAM_TENSOR;
    slot->argument.tensor.struct_size = sizeof(VernonTensorView);
    slot->argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    if (!slot->buffer->reference(slot->argument.tensor.resource))
        return error = "Program tape carrier reference failed", false;
    slot->argument.tensor.element_layout = pipelineValueLayout(binding.elementLayout);
    slot->argument.tensor.rank = static_cast<uint32_t>(slot->shape.size());
    slot->argument.tensor.shape = slot->shape.empty() ? nullptr : slot->shape.data();
    slot->argument.tensor.byte_strides = slot->strides.empty() ? nullptr : slot->strides.data();
    slot->argument.tensor.byte_size = byteSize;
    return true;
}

bool ProgramTapeScratch::uploadCarrier(uint32_t value, program_plan::TapeCarrier kind, const void *data,
                                       size_t byteSize, std::string &error) {
    Carrier *slot = carrier(value, kind);
    if (!slot || !slot->buffer || slot->readOnly || byteSize > slot->buffer->size() ||
        !slot->buffer->upload(data, byteSize))
        return error = "Program tape carrier upload failed", false;
    return true;
}

bool ProgramTapeScratch::downloadCarrier(uint32_t value, program_plan::TapeCarrier kind, void *data, size_t byteSize,
                                         std::string &error) const {
    const Carrier *slot = carrier(value, kind);
    const program_execution::DeviceBuffer *buffer = slot && slot->buffer           ? slot->buffer.get()
                                                    : slot && slot->retainedBuffer ? slot->retainedBuffer.get()
                                                                                   : nullptr;
    if (!buffer || byteSize > buffer->size() || !buffer->download(data, byteSize))
        return error = "Program tape carrier readback failed", false;
    return true;
}

const VernonProgramArgument *ProgramTapeScratch::argument(uint32_t value, const program::TargetBinding &binding) const {
    const Carrier *slot = binding.tapeCarrier ? carrier(value, *binding.tapeCarrier) : nullptr;
    return slot && (slot->buffer || slot->retainedBuffer) ? &slot->argument : nullptr;
}

} // namespace vernon::runtime::ad
