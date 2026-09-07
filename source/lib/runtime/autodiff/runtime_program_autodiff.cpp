#include "runtime_autodiff_internal.h"

#include "host_tape_allocator.h"
#include "program_invocation_builder.h"
#include "program_invocation_spec.h"
#include "program_invocation_values.h"
#include "program_residual_planner.h"
#include "program_tape_lifecycle.h"
#include "program_tape_scratch.h"
#include "retained_pullback_state.h"
#include "runtime/program_execution/failure_injection.h"
#include "runtime/program_execution/program_invocation_state.h"
#include "runtime/program_execution/publication_executor.h"
#include "runtime/program_execution/publication_transaction.h"
#include "runtime/program_execution/resolved_transfer_executor.h"
#include "runtime/program_execution_manifest.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_state.h"
#include "runtime_autodiff_memory_usage.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vernon::runtime::ad {
namespace {

using HostProgramValue = program_execution::ProgramValueState;
using program_execution::ProgramInvocationState;
using program_execution::ProgramStorageBacking;
using program_execution::ProgramValueState;
using program_execution::PublicationTransaction;
using program_execution::ResolvedTransferExecutor;

bool validateProgramTapeValues(const std::vector<HostProgramValue> &storage, const ProgramTapeScratch &tapeScratch,
                               std::string &error) {
    if (program_execution::injectFailure(program_execution::FailureBoundary::TapeValidation))
        return error = "injected Program tape validation failure", false;
    for (size_t value = 0; value < storage.size(); ++value) {
        const auto batch = tapeScratch.hostBatch(static_cast<uint32_t>(value));
        if (!batch)
            continue;
        for (size_t lane = 0; lane < batch->size(); ++lane) {
            const VernonAdTapeAllocator *allocator = batch->descriptor(lane);
            if (!allocator || allocator->status == VERNON_AD_TAPE_ALLOCATOR_OK)
                continue;
            const char *reason = allocator->status == VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED ? "capacity exhausted"
                                 : allocator->status == VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW
                                     ? "arithmetic overflow"
                                 : allocator->status == VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE
                                     ? "host allocation failed (context limit)"
                                 : allocator->status == VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI ? "invalid ABI"
                                                                                             : "invalid state";
            return error = std::string("autodiff tape allocator ") + reason, false;
        }
    }
    return true;
}

bool sealProgramTapeValues(std::vector<HostProgramValue> &storage, std::vector<VernonProgramArgument> &values,
                           ProgramTapeScratch &tapeScratch, std::string &error) {
    if (values.size() < storage.size())
        return error = "Program autodiff tape value arena is incomplete", false;
    if (!validateProgramTapeValues(storage, tapeScratch, error))
        return false;
    for (size_t value = 0; value < storage.size(); ++value) {
        HostProgramValue &slot = storage[value];
        const auto batch = tapeScratch.hostBatch(static_cast<uint32_t>(value));
        if (!batch)
            continue;
        if (!batch->compact(true))
            return error = "Program autodiff tape could not be compacted", false;
        if (!fillProgramTapeHostValue(slot, batch, tapeScratch, static_cast<uint32_t>(value), error))
            return false;
        values[value] = slot.argument;
    }
    return true;
}

bool transferLeaves(const VernonAdValueSet &supplied, const std::vector<ValueAbi> &signature,
                    const std::vector<ProgramLeafBinding> &bindings, std::vector<HostProgramValue> &storage,
                    PublicationTransaction *publication, std::string &error) {
    if (bindings.size() != signature.size() || supplied.value_count != signature.size())
        return error = "Program autodiff leaf set does not match its canonical boundary", false;
    const bool publish = publication != nullptr;
    for (size_t index = 0; index < bindings.size(); ++index) {
        const ValueAbi &abi = signature[index];
        const VernonAdValue *value = findValue(supplied, abi.path);
        const ProgramLeafBinding &binding = bindings[index];
        if (!value || !matchesProgramValueAbi(*value, abi) || binding.value >= storage.size())
            return error = "Program autodiff leaf does not match graph reflection", false;
        auto *leaf = static_cast<uint8_t *>(const_cast<void *>(value->data));
        size_t elementCount = binding.elementCount;
        if (!elementCount) {
            const size_t bytes = publish ? storage[binding.value].argument.tensor.byte_size : value->size;
            const size_t stride = publish ? binding.elementStride : binding.leafElementBytes;
            if (!stride || bytes % stride)
                return error = "Program autodiff leaf does not match its dynamic footprint", false;
            elementCount = bytes / stride;
        }
        HostProgramValue &host = storage[binding.value];
        if (!publish) {
            if (binding.elementStride && elementCount > std::numeric_limits<size_t>::max() / binding.elementStride)
                return error = "Program autodiff leaf footprint overflows", false;
            const size_t required = elementCount * binding.elementStride;
            if (host.argument.tensor.byte_size < required) {
                host.ownedHostBytes.resize(required);
                host.argument.tensor.host_data = host.ownedHostBytes.data();
                host.argument.tensor.byte_size = required;
                if (host.concreteShape &&
                    !shape::rowMajorByteStrides(*host.concreteShape, binding.elementStride, host.strides))
                    return error = "Program autodiff aggregate stride overflows", false;
                host.argument.tensor.byte_strides = host.strides.empty() ? nullptr : host.strides.data();
            }
        }
        auto *packed = static_cast<uint8_t *>(const_cast<void *>(host.argument.tensor.host_data));
        for (size_t element = 0; element < elementCount; ++element) {
            uint8_t *packedElement = packed + element * binding.elementStride + binding.byteOffset;
            uint8_t *leafElement = leaf + element * binding.leafElementBytes;
            if (publish) {
                if (!publication->stageHostRegion(binding.slot, leafElement, packedElement, binding.leafElementBytes,
                                                  error))
                    return false;
            } else
                std::memcpy(packedElement, leafElement, binding.leafElementBytes);
        }
    }
    return true;
}

VernonStatus fail(VernonRuntimeContext &context, const std::string &error) {
    invocationDiagnostic(context) = error;
    return VERNON_STATUS_INVALID_ARGUMENT;
}

class ProgramPullback final : public PullbackExecution {
public:
    ProgramPullback(VernonRuntimeContext &context, std::shared_ptr<const program::ResolvedExecutionPlan> topology,
                    Variant variant, Signature signature, std::vector<ProgramLeafBinding> cotangentBindings,
                    std::vector<ProgramLeafBinding> gradientBindings,
                    std::shared_ptr<const RetainedPullbackState> state,
                    std::vector<AutodiffPullbackPassTelemetry> passTelemetry)
        : context_(&context), topology_(std::move(topology)), variant_(std::move(variant)),
          signature_(std::move(signature)), cotangentBindings_(std::move(cotangentBindings)),
          gradientBindings_(std::move(gradientBindings)), state_(std::move(state)),
          passTelemetry_(std::move(passTelemetry)) {
        rebuildVariantLayoutViews(variant_);
    }

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients,
                       const PullbackApplyOptions &options) override {
        const std::lock_guard lock(applyMutex_);
        if (!topology_->resolvedProgram || !state_)
            return fail(*context_, "Program pullback has no retained canonical state");
        const program::Program &execution = topology_->resolvedProgram->program;
        const program::Graph *backward = program::findGraph(execution, "backward");
        const bool requiresCotangents =
            backward &&
            std::any_of(backward->inputs.begin(), backward->inputs.end(), [](const program::GraphInput &input) {
                return input.kind == program::GraphInputKind::UserInput;
            });
        if (!backward || (requiresCotangents && !cotangents))
            return fail(*context_, "Program pullback has no backward graph or required cotangents");
        std::vector<char> required(execution.values.size());
        program::markGraphValues(*backward, required);
        std::optional<program::Graph> replay;
        if (state_->residualPlan().replayEnd) {
            const program::Graph *forward = program::findGraph(execution, "forward");
            if (!forward)
                return fail(*context_, "Program pullback replay has no forward graph");
            replay = *forward;
            replay->nodes.resize(state_->residualPlan().replayEnd);
            program::markGraphValues(*replay, required);
        }
        for (uint32_t value : state_->residualPlan().retainedValues)
            if (value < required.size())
                required[value] = 1;

        const std::vector<std::vector<uint8_t>> &captures = state_->residualCaptures();
        const std::vector<std::vector<uint64_t>> &captureShapes = state_->captureShapes();
        std::vector<ProgramValueState> storage;
        PullbackInvocationSpec frame{
            required,
            {cotangents, cotangents ? &signature_.cotangents : nullptr, cotangents ? &cotangentBindings_ : nullptr},
            {&gradients, &signature_.gradients, &gradientBindings_},
            captures,
            captureShapes,
            nullptr,
            &state_->valueSnapshots(),
        };
        std::string error;
        std::map<uint32_t, ProgramStorageBacking> storageBackings;
        ProgramTapeScratch tapeScratch(execution.values.size());
        if (!buildProgramInvocationValues(*context_, execution, topology_.get(), storage, storageBackings, tapeScratch,
                                          frame, context_->autodiffMemoryPolicy, error) ||
            (cotangents &&
             !transferLeaves(*cotangents, signature_.cotangents, cotangentBindings_, storage, nullptr, error)))
            return fail(*context_, error);

        size_t temporaryBytes = 0;
        for (const ProgramValueState &value : storage) {
            if (value.ownedHostBytes.size() > std::numeric_limits<size_t>::max() - temporaryBytes)
                return fail(*context_, "Program pullback temporary memory accounting overflows");
            temporaryBytes += value.ownedHostBytes.size();
        }
        if (temporaryBytes > options.maximumTemporaryBytes)
            return fail(*context_, "Program pullback exceeds its temporary memory limit");
        if (program_execution::injectFailure(program_execution::FailureBoundary::Allocation))
            return fail(*context_, "injected Program pullback allocation failure");

        ProgramInvocationState values(*topology_, std::move(storage), std::move(storageBackings));
        if (!state_->importInto(values, tapeScratch, !replay, error) ||
            !values.allocateOwnedImageStorages(*context_, execution, error))
            return fail(*context_, error);
        if (program_execution::injectFailure(program_execution::FailureBoundary::TapeValidation))
            return fail(*context_, "injected Program pullback tape validation failure");
        const program_execution::ResolvePhysicalEndpoint resolvePhysicalEndpoint =
            [&](uint32_t value, const program::TargetBinding &binding) -> const VernonProgramArgument * {
            if (binding.semantic == program::CarrierSemantic::Tape)
                if (const VernonProgramArgument *argument = tapeScratch.argument(value, binding))
                    return argument;
            return values.argument(value);
        };
        const bool device = context_->backend != VERNON_RUNTIME_CPU;
        if (program_execution::injectFailure(program_execution::FailureBoundary::Submission))
            return fail(*context_, "injected Program pullback submission failure");
        VernonProgramExecutable proxy(*context_, topology_);
        if (replay) {
            if (device) {
                std::vector<ProgramTapeState> tapeStates;
                if (!prepareProgramTapeStates(values, tapeScratch, *context_, execution, *topology_, *replay,
                                              tapeStates, error))
                    return fail(*context_, error);
                for (;;) {
                    const VernonStatus replayStatus =
                        executePipelineProgramGraph(proxy, *replay, values, resolvePhysicalEndpoint);
                    if (replayStatus != VERNON_STATUS_OK)
                        return replayStatus;
                    bool retry = false;
                    if (!validateProgramTapeStates(tapeScratch, tapeStates, retry, error))
                        return fail(*context_, error);
                    if (!retry)
                        break;
                    ResolvedTransferExecutor transfers(*context_, values);
                    if (!transfers.restoreForRetry(program::GraphDirection::Forward, error))
                        return fail(*context_, error);
                    if (!state_->importInto(values, tapeScratch, false, error))
                        return fail(*context_, error);
                    for (ProgramTapeState &tape : tapeStates)
                        if (!allocateProgramTapeState(tapeScratch, *context_, tape, error))
                            return fail(*context_, error);
                }
            } else {
                const VernonStatus replayStatus =
                    executePipelineProgramGraph(proxy, *replay, values, resolvePhysicalEndpoint);
                if (replayStatus != VERNON_STATUS_OK)
                    return replayStatus;
                if (!sealProgramTapeValues(values.values(), values.arguments(), tapeScratch, error))
                    return fail(*context_, error);
            }
        }
        const VernonStatus status = executePipelineProgramGraph(proxy, *backward, values, resolvePhysicalEndpoint);
        if (status != VERNON_STATUS_OK)
            return status;

        if (device) {
            std::vector<char> downloads(execution.values.size());
            for (const ProgramLeafBinding &binding : gradientBindings_)
                if (binding.value < downloads.size())
                    downloads[binding.value] = 1;
            ResolvedTransferExecutor transfers(*context_, values);
            if (!transfers.readbackBoundaryValues(downloads, error))
                return fail(*context_, error);
            ++usage_.readbacks;
        }
        PublicationTransaction publication(topology_->publications);
        if (!transferLeaves(gradients, signature_.gradients, gradientBindings_, values.values(), &publication, error))
            return fail(*context_, error);
        if (const VernonStatus publicationStatus =
                program_execution::executePublicationCommit(*context_, publication, values, error);
            publicationStatus != VERNON_STATUS_OK)
            return error.empty() ? publicationStatus : fail(*context_, error);
        ++usage_.submissions;
        ++usage_.waits;
        ++usage_.atomicPublications;
        usage_.temporaryAllocationBytes += temporaryBytes;
        return VERNON_STATUS_OK;
    }

    PullbackMemoryUsage memoryUsage() const override {
        MemoryAccounting memory;
        memory.logicalPayloadBytes = state_->residualPlan().checkpoint.logicalResidualBytes;
        const auto add = [](size_t &total, size_t bytes) {
            if (bytes > std::numeric_limits<size_t>::max() - total)
                total = std::numeric_limits<size_t>::max();
            else
                total += bytes;
        };
        add(memory.residentBytes, state_->residentBytes());
        add(memory.retainedAllocationBytes, state_->residentBytes());
        add(memory.allocatedBytes, state_->allocatedTapeBytes());
        memory.retainedAllocationBytes =
            std::max(memory.retainedAllocationBytes,
                     static_cast<size_t>(state_->residualPlan().checkpoint.retainedAllocationBytes));
        add(memory.retainedAllocationBytes, memory.allocatedBytes);
        return pullbackMemoryUsage(memory);
    }

    program_execution::ExecutionControlPlaneUsage controlPlaneUsage() const override {
        const std::lock_guard lock(applyMutex_);
        return usage_;
    }

    AutodiffPullbackCheckpointPlan checkpointPlan() const override {
        AutodiffPullbackCheckpointPlan snapshot;
        snapshot.present = true;
        snapshot.peakBytes = state_->residualPlan().checkpoint.peakBytes;
        snapshot.memoryBudget = state_->residualPlan().checkpoint.memoryBudget;
        snapshot.logicalResidualBytes = state_->residualPlan().checkpoint.logicalResidualBytes;
        snapshot.retainedAllocationBytes = state_->residualPlan().checkpoint.retainedAllocationBytes;
        snapshot.initialStateBytes = state_->residualPlan().checkpoint.initialStateBytes;
        snapshot.restorationBytes = state_->residualPlan().checkpoint.restorationBytes;
        snapshot.transactionBytes = state_->residualPlan().checkpoint.transactionBytes;
        snapshot.persistentCheckpointBytes = state_->residualPlan().checkpoint.persistentCheckpointBytes;
        snapshot.backwardValueBytes = state_->residualPlan().checkpoint.backwardValueBytes;
        snapshot.replayCost = state_->residualPlan().checkpoint.replayCost;
        snapshot.recomputationCost = state_->residualPlan().checkpoint.recomputationCost;
        snapshot.selectedPolicy = state_->residualPlan().checkpoint.selectedPolicy;
        return snapshot;
    }

    std::vector<AutodiffPullbackPassTelemetry> passTelemetry() const override { return passTelemetry_; }

    uint64_t peakRuntimeManagedBytes() const override {
        const PullbackMemoryUsage usage = memoryUsage();
        const uint64_t held =
            std::max(std::max(usage.retainedAllocationBytes, usage.allocatedBytes), usage.peakTemporaryBytes);
        return std::max(held, state_->residualPlan().checkpoint.peakBytes);
    }

private:
    VernonRuntimeContext *context_;
    std::shared_ptr<const program::ResolvedExecutionPlan> topology_;
    Variant variant_;
    Signature signature_;
    std::vector<ProgramLeafBinding> cotangentBindings_;
    std::vector<ProgramLeafBinding> gradientBindings_;
    std::shared_ptr<const RetainedPullbackState> state_;
    std::vector<AutodiffPullbackPassTelemetry> passTelemetry_;
    mutable std::mutex applyMutex_;
    program_execution::ExecutionControlPlaneUsage usage_;
};

class ProgramExecutable final : public CanonicalProgramExecution {
public:
    ProgramExecutable(VernonRuntimeContext &context, std::weak_ptr<const program::ResolvedExecutionPlan> topology,
                      CanonicalProgramAutodiffState &programAutodiff, Variant variant)
        : context_(&context), topology_(std::move(topology)), programAutodiff_(&programAutodiff),
          variant_(std::move(variant)) {
        rebuildVariantLayoutViews(variant_);
        const std::shared_ptr<const program::ResolvedExecutionPlan> locked = topology_.lock();
        if (!locked || !locked->resolvedProgram) {
            signatureError_ = "Program executable has no resolved canonical owner";
            return;
        }
        const program::Program &canonicalProgram = locked->resolvedProgram->program;
        const bool hasBackward = program::findGraph(canonicalProgram, "backward") != nullptr;
        for (const program::TapePlan &plan : canonicalProgram.abi.tapePlans)
            if (!plan.forwardProducer || (hasBackward && !plan.backwardConsumer)) {
                signatureError_ = "compiler TapePlan does not connect its forward producer and backward consumer";
                return;
            }
        for (const program::BoundarySlot &slot : canonicalProgram.abi.boundarySlots) {
            if (slot.role == program::BoundaryRole::Input ||
                (slot.role == program::BoundaryRole::Output && slot.publication != program::BoundaryPublication::None))
                forwardBindings_.emplace_back(slot.id, slot.value);
        }
        // ProgramABI is the execution authority. Signature is validated against
        // it while parsing, but runtime binding and leaf materialization must
        // not join the two representations again.
        const auto append = [&](const program::BoundarySlot &publicSlot,
                                const std::vector<program::LayoutPathComponent> *projectionPath,
                                std::vector<ValueAbi> &values, std::vector<ProgramLeafBinding> &leafBindings) {
            if (publicSlot.category == program::BoundaryCategory::Texture ||
                publicSlot.category == program::BoundaryCategory::Sampler)
                return;
            std::string error;
            if (!publicSlot.layout) {
                signatureError_ = "ProgramABI tensor boundary has no canonical ValueLayout";
                return;
            }
            ValueLayout layout = program::materializeValueLayout(*publicSlot.layout, publicSlot.logicalType);
            rebuildValueLayoutPathViews(layout);
            std::vector<const ValueLeaf *> selectedLeaves;
            ValueLayout publicLayout = layout;
            if (projectionPath && !projectionPath->empty()) {
                publicLayout.leaves.clear();
                for (const ValueLeaf &leaf : layout.leaves) {
                    if (leaf.path.size() < projectionPath->size())
                        continue;
                    bool selected = true;
                    for (size_t component = 0; component < projectionPath->size(); ++component) {
                        const auto &expected = (*projectionPath)[component];
                        const auto &actual = leaf.path[component];
                        selected &= expected.index ? !actual.field && *expected.index == actual.index
                                                   : actual.field && expected.field == *actual.field;
                    }
                    if (!selected)
                        continue;
                    ValueLeaf projected = leaf;
                    projected.path.erase(projected.path.begin(),
                                         projected.path.begin() + static_cast<ptrdiff_t>(projectionPath->size()));
                    publicLayout.leaves.push_back(std::move(projected));
                    selectedLeaves.push_back(&leaf);
                }
                if (selectedLeaves.empty()) {
                    signatureError_ = "DerivativeProjection does not select a canonical Value leaf";
                    return;
                }
                rebuildValueLayoutPathViews(publicLayout);
            } else {
                for (const ValueLeaf &leaf : layout.leaves)
                    selectedLeaves.push_back(&leaf);
            }
            const size_t begin = values.size();
            if (!appendValueLayoutAbi(publicLayout, publicSlot.outerShape, publicSlot.path, values, error)) {
                signatureError_ = std::move(error);
                return;
            }
            size_t elementCount = 1;
            for (uint64_t extent : publicSlot.outerShape)
                elementCount *= static_cast<size_t>(extent);
            if (values.size() - begin != selectedLeaves.size()) {
                signatureError_ = "Program aggregate ABI contains duplicate leaf paths";
                return;
            }
            for (const ValueLeaf *selected : selectedLeaves) {
                const ValueLeaf &leaf = *selected;
                const std::optional<VernonDataType> dtype = pipelineDataType(leaf.dtype);
                leafBindings.push_back({publicSlot.value, publicSlot.id, leaf.byteOffset, layout.byteSize,
                                        dtype ? dtypeSize(*dtype) * static_cast<size_t>(leaf.scalarCount) : 0,
                                        elementCount});
            }
        };
        for (const program::BoundarySlot &slot : canonicalProgram.abi.boundarySlots) {
            if (slot.role == program::BoundaryRole::Input)
                append(slot, nullptr, signature_.inputs, inputBindings_);
            else if (slot.role == program::BoundaryRole::Output &&
                     slot.publication != program::BoundaryPublication::None)
                append(slot, nullptr, signature_.outputs, outputBindings_);
        }
        for (const program::DerivativeProjection &projection : canonicalProgram.abi.derivativeProjections) {
            const program::BoundarySlot &derivative = canonicalProgram.abi.boundarySlots[projection.derivative.slot];
            if (derivative.role == program::BoundaryRole::Cotangent)
                append(derivative, &projection.valuePath, signature_.cotangents, cotangentBindings_);
            else if (derivative.role == program::BoundaryRole::Gradient)
                append(derivative, &projection.valuePath, signature_.gradients, gradientBindings_);
            else {
                signatureError_ = "DerivativeProjection does not reference a derivative boundary";
                return;
            }
        }
    }

    const Signature &signature() const override { return signature_; }

    VernonStatus forward(const ForwardExecutionTarget &target, const VernonAdValueSet &inputs,
                         VernonAdValueSet *outputs, std::unique_ptr<PullbackExecution> &pullback) override {
        const std::shared_ptr<const program::ResolvedExecutionPlan> topology = topology_.lock();
        const program::Program *execution =
            topology && topology->resolvedProgram ? &topology->resolvedProgram->program : nullptr;
        const program::Graph *forward = execution ? program::findGraph(*execution, "forward") : nullptr;
        const bool hasBackward = execution && program::findGraph(*execution, "backward");
        const bool canonicalInvocation = target.invocation != nullptr;
        if (!signatureError_.empty())
            return fail(*context_, signatureError_);
        if (!topology || !forward || (!canonicalInvocation && !outputs) || target.encodedInvocation())
            return fail(*context_, "Program autodiff forward requires a live pipeline and host API values");
        std::vector<HostProgramValue> hostStorage;
        std::string error;
        std::vector<char> required(execution->values.size());
        program::markGraphValues(*forward, required);
        for (uint32_t value : program::residualCaptures(*execution))
            if (value < required.size())
                required[value] = 1;
        PublicationTransaction publication(topology->publications);
        ForwardInvocationSpec frame{
            required,
            canonicalInvocation ? std::variant<CanonicalForwardBindings, HostForwardBindings>(
                                      CanonicalForwardBindings{*target.invocation, forwardBindings_, publication})
                                : std::variant<CanonicalForwardBindings, HostForwardBindings>(HostForwardBindings{
                                      {&inputs, &signature_.inputs, &inputBindings_},
                                      {outputs, &signature_.outputs, &outputBindings_},
                                  }),
        };
        std::map<uint32_t, ProgramStorageBacking> storageBackings;
        ProgramTapeScratch tapeScratch(execution->values.size());
        if (!buildProgramInvocationValues(*context_, *execution, topology.get(), hostStorage, storageBackings,
                                          tapeScratch, frame, context_->autodiffMemoryPolicy, error) ||
            (!canonicalInvocation &&
             !transferLeaves(inputs, signature_.inputs, inputBindings_, hostStorage, nullptr, error)))
            return fail(*context_, error);
        ProgramInvocationState arena(*topology, std::move(hostStorage), std::move(storageBackings));
        if (!arena.allocateOwnedImageStorages(*context_, *execution, error))
            return fail(*context_, error);
        if (const VernonStatus initialization =
                program_execution::executePublicationInitialization(*context_, publication, error);
            initialization != VERNON_STATUS_OK)
            return error.empty() ? initialization : fail(*context_, error);
        const program_execution::ResolvePhysicalEndpoint resolvePhysicalEndpoint =
            [&](uint32_t value, const program::TargetBinding &binding) -> const VernonProgramArgument * {
            if (binding.semantic == program::CarrierSemantic::Tape)
                if (const VernonProgramArgument *argument = tapeScratch.argument(value, binding))
                    return argument;
            return arena.argument(value);
        };
        arena.setInvocationContext(target.programContext);
        VernonProgramExecutable proxy(*context_, topology);
        for (const program::Node &node : forward->nodes)
            if (program::executionKind(node) == program::ExecutionKind::Compute)
                for (const program::ControlComponent &control : program::computeOperation(node).workgroups) {
                    uint64_t value = 0;
                    if (!arena.resolveControl(*execution, control, value, error))
                        return fail(*context_, error);
                }
        const bool device = context_->backend != VERNON_RUNTIME_CPU;
        if (program_execution::injectFailure(program_execution::FailureBoundary::Submission))
            return fail(*context_, "injected Program forward submission failure");
        if (device) {
            std::vector<ProgramTapeState> tapeStates;
            if (!prepareProgramTapeStates(arena, tapeScratch, *context_, *execution, *topology, *forward, tapeStates,
                                          error))
                return fail(*context_, error);
            for (;;) {
                const VernonStatus status =
                    executePipelineProgramGraph(proxy, *forward, arena, resolvePhysicalEndpoint);
                if (status != VERNON_STATUS_OK)
                    return status;
                bool retry = false;
                if (!validateProgramTapeStates(tapeScratch, tapeStates, retry, error))
                    return fail(*context_, error);
                if (!retry)
                    break;
                ResolvedTransferExecutor transfers(*context_, arena);
                if (!transfers.restoreForRetry(program::GraphDirection::Forward, error))
                    return fail(*context_, error);
                for (ProgramTapeState &state : tapeStates)
                    if (!allocateProgramTapeState(tapeScratch, *context_, state, error))
                        return fail(*context_, error);
            }
        } else {
            const VernonStatus status = executePipelineProgramGraph(proxy, *forward, arena, resolvePhysicalEndpoint);
            if (status != VERNON_STATUS_OK)
                return status;
            if (!validateProgramTapeValues(arena.values(), tapeScratch, error))
                return fail(*context_, error);
        }

        if (device) {
            std::vector<char> downloads = publication.hostReadbackValues(execution->values.size());
            if (!canonicalInvocation)
                for (const ProgramLeafBinding &binding : outputBindings_)
                    if (binding.value < downloads.size())
                        downloads[binding.value] = 1;
            ResolvedTransferExecutor transfers(*context_, arena);
            if (!transfers.readbackBoundaryValues(downloads, error))
                return fail(*context_, error);
        }
        if (!canonicalInvocation &&
            !transferLeaves(*outputs, signature_.outputs, outputBindings_, arena.values(), &publication, error))
            return fail(*context_, error);
        if (const VernonStatus publicationStatus =
                program_execution::executePublicationCommit(*context_, publication, arena, error);
            publicationStatus != VERNON_STATUS_OK)
            return error.empty() ? publicationStatus : fail(*context_, error);
        if (!hasBackward)
            return VERNON_STATUS_OK;

        ProgramResidualPlan plan;
        uint64_t memoryBudget = context_->autodiffMemoryPolicy->invocationLimit();
        const bool rematerializeTapes = programAutodiff_->checkpointMemoryBudget.has_value();
        if (rematerializeTapes)
            memoryBudget = *programAutodiff_->checkpointMemoryBudget;
        if (!planProgramResiduals(*execution, topology.get(), variant_, arena.values(), tapeScratch, memoryBudget,
                                  programAutodiff_->checkpointPolicy, rematerializeTapes, plan, error))
            return fail(*context_, error);
        for (uint32_t value : plan.retainedValues) {
            if (value >= arena.values().size())
                return fail(*context_, "Program pullback residual plan references an unknown Value");
            ProgramValueState &slot = arena.values()[value];
            const auto batch = tapeScratch.hostBatch(value);
            if (batch) {
                if (!batch->compact(true) || !fillProgramTapeHostValue(slot, batch, tapeScratch, value, error))
                    return fail(*context_, error.empty() ? "Program autodiff tape could not be compacted" : error);
                arena.arguments()[value] = slot.argument;
            }
        }
        std::vector<AutodiffPullbackPassTelemetry> telemetry =
            collectProgramPassTelemetry(*forward, *execution, arena.values(), tapeScratch, plan);
        std::vector<char> retained(execution->values.size());
        for (uint32_t value : plan.retainedValues)
            if (value < retained.size())
                retained[value] = 1;
        arena.retainOnly(retained);
        auto state = std::make_shared<const RetainedPullbackState>(std::move(plan), arena, std::move(tapeScratch));
        pullback = std::make_unique<ProgramPullback>(*context_, topology, variant_, signature_, cotangentBindings_,
                                                     gradientBindings_, std::move(state), std::move(telemetry));
        return VERNON_STATUS_OK;
    }

private:
    VernonRuntimeContext *context_;
    std::weak_ptr<const program::ResolvedExecutionPlan> topology_;
    const CanonicalProgramAutodiffState *programAutodiff_{};
    Variant variant_;
    Signature signature_;
    std::vector<ProgramLeafBinding> inputBindings_;
    std::vector<ProgramLeafBinding> outputBindings_;
    std::vector<ProgramLeafBinding> cotangentBindings_;
    std::vector<ProgramLeafBinding> gradientBindings_;
    std::vector<std::pair<uint32_t, uint32_t>> forwardBindings_;
    std::string signatureError_;
};

} // namespace

bool resolveProgramAutodiff(VernonProgramExecutable &pipeline,
                            const std::vector<AutodiffDerivativeGroup> &derivativeGroups) {
    if (!pipeline.context)
        return false;
    CanonicalProgramAutodiffState &state = pipeline.programAutodiff;
    const program::Program &execution = pipeline.executionPlan->resolvedProgram->program;
    if (!program::findGraph(execution, "forward")) {
        invocationDiagnostic(*pipeline.context) = "Program execution topology requires a forward graph";
        return false;
    }
    if (!pipeline.context->autodiffMemoryPolicy)
        pipeline.context->autodiffMemoryPolicy = std::make_shared<AutodiffMemoryPolicy>();
    auto executable = std::make_shared<ProgramExecutable>(*pipeline.context, pipeline.executionPlan, state, Variant{});
    std::vector<AutodiffDerivativeGroup> groups = derivativeGroups;
    const auto appendGroups = [&](AutodiffDerivativeRole role, program::BoundaryRole boundaryRole,
                                  const std::vector<ValueAbi> &leaves, std::vector<AutodiffDerivativeGroup> &result) {
        for (const program::DerivativeProjection &projection : execution.abi.derivativeProjections) {
            const program::BoundarySlot &derivative = execution.abi.boundarySlots[projection.derivative.slot];
            if (derivative.role != boundaryRole)
                continue;
            if (!derivative.layout) {
                invocationDiagnostic(*pipeline.context) =
                    "ProgramABI derivative projection has no canonical ValueLayout";
                return false;
            }
            AutodiffDerivativeGroup group{role, projection.derivative.path, {}};
            ValueLayout layout = program::materializeValueLayout(*derivative.layout, derivative.logicalType);
            for (const ValueLeaf &layoutLeaf : layout.leaves) {
                if (layoutLeaf.path.size() < projection.valuePath.size())
                    continue;
                bool selected = true;
                for (size_t component = 0; component < projection.valuePath.size(); ++component) {
                    const auto &expected = projection.valuePath[component];
                    const auto &actual = layoutLeaf.path[component];
                    selected &= expected.index ? !actual.field && *expected.index == actual.index
                                               : actual.field && expected.field == *actual.field;
                }
                if (!selected)
                    continue;
                ValueLeaf publicLeaf = layoutLeaf;
                publicLeaf.path.erase(publicLeaf.path.begin(),
                                      publicLeaf.path.begin() + static_cast<ptrdiff_t>(projection.valuePath.size()));
                const std::string leafPath = canonicalValueLeafPath(derivative.path, publicLeaf);
                if (std::none_of(leaves.begin(), leaves.end(),
                                 [&](const ValueAbi &leaf) { return leaf.path == leafPath; })) {
                    invocationDiagnostic(*pipeline.context) =
                        "ProgramABI derivative projection does not resolve to its declared leaf ABI";
                    return false;
                }
                group.leafPaths.push_back(leafPath);
            }
            if (group.leafPaths.empty()) {
                invocationDiagnostic(*pipeline.context) =
                    "ProgramABI derivative projection selects no declared leaf ABI";
                return false;
            }
            std::sort(group.leafPaths.begin(), group.leafPaths.end());
            result.push_back(std::move(group));
        }
        return true;
    };
    if (groups.empty()) {
        if (!appendGroups(AutodiffDerivativeRole::Gradient, program::BoundaryRole::Gradient,
                          executable->signature().gradients, groups))
            return false;
        std::sort(groups.begin(), groups.end(),
                  [](const AutodiffDerivativeGroup &left, const AutodiffDerivativeGroup &right) {
                      return left.declaredPath < right.declaredPath;
                  });
        const size_t gradientCount = groups.size();
        if (!appendGroups(AutodiffDerivativeRole::Cotangent, program::BoundaryRole::Cotangent,
                          executable->signature().cotangents, groups))
            return false;
        std::sort(groups.begin() + static_cast<std::ptrdiff_t>(gradientCount), groups.end(),
                  [](const AutodiffDerivativeGroup &left, const AutodiffDerivativeGroup &right) {
                      return left.declaredPath < right.declaredPath;
                  });
    }
    std::string groupError;
    const bool completeDerivativeBoundary =
        !executable->signature().cotangents.empty() && !executable->signature().gradients.empty();
    const bool validGroups = !completeDerivativeBoundary || validateAutodiffDerivativeGroups(groups, groupError);
    const bool validSignature = !completeDerivativeBoundary || validateDerivativeGroupsAgainstSignature(
                                                                   *pipeline.context, groups, executable->signature());
    if (!validGroups || !validSignature) {
        if (!groupError.empty())
            invocationDiagnostic(*pipeline.context) = std::move(groupError);
        else if (invocationDiagnostic(*pipeline.context).empty())
            invocationDiagnostic(*pipeline.context) = "Program autodiff derivative groups do not match its signature";
        return false;
    }
    state.canonicalExecution = std::move(executable);
    state.derivativeGroups = std::move(groups);
    return true;
}

} // namespace vernon::runtime::ad
