#include "retained_pullback_state.h"

#include <algorithm>

namespace vernon::runtime::ad {
namespace {

std::vector<program_execution::CanonicalValueSnapshot>
captureValues(const ProgramResidualPlan &plan, const program_execution::ProgramInvocationState &invocation) {
    std::vector<program_execution::CanonicalValueSnapshot> result;
    result.reserve(plan.retainedValues.size());
    for (uint32_t value : plan.retainedValues)
        result.push_back(invocation.snapshotValue(value));
    return result;
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
captureStorages(const program_execution::ProgramInvocationState &invocation) {
    auto result = invocation.storageBackings();
    for (auto &[storage, backing] : result) {
        (void)storage;
        backing.external.reset();
        backing.initial.reset();
    }
    return result;
}

std::vector<char> captureTapeValues(const program_execution::ProgramInvocationState &invocation) {
    if (!invocation.plan().resolvedProgram)
        return {};
    const program::Program &program = invocation.plan().resolvedProgram->program;
    std::vector<char> result(program.values.size());
    for (const program::Value &value : program.values)
        result[value.id] = program::isTapeValueType(value.type);
    return result;
}

} // namespace

RetainedPullbackState::RetainedPullbackState(ProgramResidualPlan plan,
                                             const program_execution::ProgramInvocationState &invocation,
                                             ProgramTapeScratch tapeScratch)
    : residualPlan_(std::move(plan)), values_(captureValues(residualPlan_, invocation)),
      storages_(captureStorages(invocation)), residualCaptures_(captureBytes(invocation.values().size(), values_)),
      captureShapes_(makeCaptureShapes(invocation.values().size(), values_)),
      tapeValues_(captureTapeValues(invocation)), tape_(tapeScratch.releaseSnapshot()) {}

bool RetainedPullbackState::importInto(program_execution::ProgramInvocationState &invocation,
                                       ProgramTapeScratch &tapeScratch, bool importTape, std::string &error) const {
    if (importTape)
        tapeScratch.importSnapshot(tape_);
    if (!invocation.importStorageSnapshots(storages_, error))
        return false;
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
        if (!invocation.importSnapshot(snapshot, error))
            return false;
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
