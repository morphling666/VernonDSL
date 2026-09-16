#include "retained_pullback_state.h"

#include <algorithm>

namespace vernon::runtime::ad {
namespace {

program_execution::CanonicalValueSnapshot ownRetainedStorage(program_execution::CanonicalValueSnapshot snapshot) {
    auto &logical = snapshot.logical;
    if (!snapshot.deviceOwner && logical.ownedHostBytes.empty() && logical.argument.kind == VERNON_PROGRAM_TENSOR &&
        logical.argument.tensor.storage == VERNON_TENSOR_HOST && logical.argument.tensor.host_data &&
        logical.argument.tensor.byte_size) {
        const auto *data = static_cast<const uint8_t *>(logical.argument.tensor.host_data);
        logical.ownedHostBytes.assign(data, data + logical.argument.tensor.byte_size);
        logical.argument.tensor.host_data = logical.ownedHostBytes.data();
        snapshot.residentArgument = logical.argument;
    }
    return snapshot;
}

std::vector<program_execution::CanonicalValueSnapshot>
captureValues(const ProgramResidualPlan &plan, const program_execution::ProgramInvocationState &invocation) {
    std::vector<program_execution::CanonicalValueSnapshot> result;
    result.reserve(plan.retainedValues.size());
    for (uint32_t value : plan.retainedValues) {
        auto snapshot = invocation.snapshotValue(value);
        if (snapshot)
            result.push_back(ownRetainedStorage(std::move(snapshot).value()));
    }
    return result;
}

std::vector<program_execution::CanonicalValueSnapshot>
ownRetainedStorage(std::vector<program_execution::CanonicalValueSnapshot> values) {
    for (auto &value : values)
        value = ownRetainedStorage(std::move(value));
    return values;
}

std::vector<std::vector<uint8_t>> captureBytes(size_t valueCount,
                                               const std::vector<program_execution::CanonicalValueSnapshot> &values) {
    std::vector<std::vector<uint8_t>> result(valueCount);
    for (const auto &snapshot : values)
        if (snapshot.value < result.size())
            result[snapshot.value] = snapshot.logical.ownedHostBytes;
    return result;
}

std::vector<std::vector<uint64_t>>
makeCaptureShapes(size_t valueCount, const std::vector<program_execution::CanonicalValueSnapshot> &values) {
    std::vector<std::vector<uint64_t>> result(valueCount);
    for (const auto &snapshot : values)
        if (snapshot.value < result.size() && snapshot.logical.concreteShape)
            result[snapshot.value] = *snapshot.logical.concreteShape;
    return result;
}

std::map<uint32_t, program_execution::ProgramStorageBacking>
sanitizeStorages(std::map<uint32_t, program_execution::ProgramStorageBacking> result) {
    for (auto &[storage, backing] : result) {
        (void)storage;
        backing.external.reset();
        backing.initial.reset();
    }
    return result;
}

std::map<uint32_t, program_execution::ProgramStorageBacking>
captureStorages(const program_execution::ProgramInvocationState &invocation) {
    return sanitizeStorages(invocation.storageBackings());
}

std::vector<char> captureTapeValues(const program::Program &program) {
    std::vector<char> result(program.values.size());
    for (const program::Value &value : program.values)
        result[value.id] = program::isTapeValue(value);
    return result;
}

std::vector<char> captureTapeValues(const program_execution::ProgramInvocationState &invocation) {
    return invocation.plan().resolvedProgram ? captureTapeValues(invocation.plan().resolvedProgram->program)
                                             : std::vector<char>{};
}

} // namespace

RetainedPullbackState::RetainedPullbackState(ProgramResidualPlan plan,
                                             const program_execution::ProgramInvocationState &invocation,
                                             ProgramTapeSnapshot tape)
    : residualPlan_(std::move(plan)), values_(captureValues(residualPlan_, invocation)),
      storages_(captureStorages(invocation)), residualCaptures_(captureBytes(invocation.values().size(), values_)),
      captureShapes_(makeCaptureShapes(invocation.values().size(), values_)),
      tapeValues_(captureTapeValues(invocation)), tape_(std::move(tape)) {}

RetainedPullbackState::RetainedPullbackState(ProgramResidualPlan plan, const program::Program &program,
                                             std::vector<program_execution::CanonicalValueSnapshot> values,
                                             std::map<uint32_t, program_execution::ProgramStorageBacking> storages,
                                             ProgramTapeSnapshot tape)
    : residualPlan_(std::move(plan)), values_(ownRetainedStorage(std::move(values))),
      storages_(sanitizeStorages(std::move(storages))), residualCaptures_(captureBytes(program.values.size(), values_)),
      captureShapes_(makeCaptureShapes(program.values.size(), values_)), tapeValues_(captureTapeValues(program)),
      tape_(std::move(tape)) {}

bool RetainedPullbackState::importInto(program_execution::ProgramInvocationState &invocation,
                                       ProgramTapeScratch &tapeScratch, bool importTape, std::string &error) const {
    if (importTape)
        tapeScratch.importSnapshot(tape_);
    auto importedStorages = invocation.importStorageSnapshots(storages_);
    if (importedStorages.isErr()) {
        error = program_execution::programInvocationErrorMessage(importedStorages.error());
        return false;
    }
    for (const program_execution::CanonicalValueSnapshot &snapshot : values_) {
        if (!importTape) {
            if (snapshot.value < tapeValues_.size() && tapeValues_[snapshot.value])
                continue;
            const bool hostTape = snapshot.value < tape_.hostBatches.size() && tape_.hostBatches[snapshot.value];
            const bool deviceTape =
                snapshot.value < tape_.carriers.size() &&
                std::any_of(
                    tape_.carriers[snapshot.value].begin(), tape_.carriers[snapshot.value].end(),
                    [](const ProgramTapeCarrierSnapshot &carrier) { return static_cast<bool>(carrier.owner); });
            if (hostTape || deviceTape ||
                snapshot.logical.ownership == program_execution::ProgramValueOwnership::TapeCarrier)
                continue;
        }
        auto imported = invocation.importSnapshot(snapshot);
        if (imported.isErr()) {
            error = program_execution::programInvocationErrorMessage(imported.error());
            return false;
        }
    }
    return true;
}

const void *RetainedPullbackState::snapshotHostIdentity(uint32_t value) const {
    const auto found =
        std::find_if(values_.begin(), values_.end(), [&](const auto &snapshot) { return snapshot.value == value; });
    return found != values_.end() && !found->logical.ownedHostBytes.empty() ? found->logical.ownedHostBytes.data()
                                                                            : nullptr;
}

size_t RetainedPullbackState::residentBytes() const {
    size_t result{};
    std::vector<const program_execution::DeviceBuffer *> deviceSnapshots;
    deviceSnapshots.reserve(values_.size());
    for (const auto &value : values_) {
        if (!value.deviceOwner ||
            std::find(deviceSnapshots.begin(), deviceSnapshots.end(), value.deviceOwner.get()) != deviceSnapshots.end())
            continue;
        deviceSnapshots.push_back(value.deviceOwner.get());
        result += value.deviceOwner->size();
    }
    for (const auto &capture : residualCaptures_)
        result += capture.size();
    for (const auto &batch : tape_.hostBatches)
        if (batch)
            result += batch->isCompacted() ? batch->logicalBytes() : batch->residentBytes();
    return result;
}

size_t RetainedPullbackState::allocatedTapeBytes() const {
    size_t result{};
    for (const auto &batch : tape_.hostBatches)
        if (batch)
            result += batch->allocatedBytes();
    return result;
}

} // namespace vernon::runtime::ad
