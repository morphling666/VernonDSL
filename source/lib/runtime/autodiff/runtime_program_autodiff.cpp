#include "runtime_autodiff_internal.h"

#include "host_tape_allocator.h"
#include "program_invocation_frame_builder.h"
#include "program_invocation_spec.h"
#include "program_publication.h"
#include "program_residual_planner.h"
#include "program_tape_lifecycle.h"
#include "program_value_arena.h"
#include "program_value_materializer.h"
#include "runtime/program_execution_manifest.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_state.h"
#include "runtime_autodiff_memory_usage.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vernon::runtime::ad {
namespace {

using HostProgramValue = LogicalProgramValue;

bool validateProgramTapeValues(const std::vector<HostProgramValue> &storage, std::string &error) {
    for (const HostProgramValue &slot : storage) {
        if (!slot.tapeBatch)
            continue;
        for (size_t lane = 0; lane < slot.tapeBatch->size(); ++lane) {
            const VernonAdTapeAllocator *allocator = slot.tapeBatch->descriptor(lane);
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

void markDeviceResidentValues(const program::Program &program, const program::Graph &graph,
                              const VernonProgramTopology &topology, std::vector<char> &required) {
    const auto markAliasDomain = [&](uint32_t value) {
        if (value >= required.size() || value >= program.values.size())
            return;
        required[value] = 1;
        if (!program.values[value].storage)
            return;
        for (size_t candidate = 0; candidate < program.values.size(); ++candidate)
            if (program.values[candidate].storage == program.values[value].storage)
                required[candidate] = 1;
    };
    for (const program::Node &node : graph.nodes) {
        const auto nodePlan = topology.nodes.find({graph.direction, node.id});
        if (nodePlan == topology.nodes.end())
            continue;
        for (const VernonProgramStageBinding &binding : nodePlan->second.bindings) {
            if (!binding.target || binding.value >= required.size())
                continue;
            switch (binding.target->carrier) {
            case program::TargetCarrier::StorageBuffer:
            case program::TargetCarrier::VertexBuffer:
            case program::TargetCarrier::IndexBuffer:
                markAliasDomain(binding.value);
                break;
            default:
                break;
            }
        }
    }
}

bool sealProgramTapeValues(std::vector<HostProgramValue> &storage, std::vector<VernonProgramArgument> &values,
                           std::string &error) {
    if (values.size() < storage.size())
        return error = "Program autodiff tape value arena is incomplete", false;
    if (!validateProgramTapeValues(storage, error))
        return false;
    for (size_t value = 0; value < storage.size(); ++value) {
        HostProgramValue &slot = storage[value];
        if (!slot.tapeBatch)
            continue;
        if (!slot.tapeBatch->compact(true))
            return error = "Program autodiff tape could not be compacted", false;
        if (!fillProgramTapeHostValue(slot, slot.tapeBatch, error))
            return false;
        values[value] = slot.argument;
    }
    return true;
}

bool transferLeaves(const VernonAdValueSet &supplied, const std::vector<ValueAbi> &signature,
                    const std::vector<ProgramLeafBinding> &bindings, std::vector<HostProgramValue> &storage,
                    bool publish, std::string &error) {
    if (bindings.size() != signature.size() || supplied.value_count != signature.size())
        return error = "Program autodiff leaf set does not match its canonical boundary", false;
    struct PendingPublication {
        void *destination{};
        std::vector<uint8_t> bytes;
    };
    std::vector<PendingPublication> pending;
    for (size_t index = 0; index < bindings.size(); ++index) {
        const ValueAbi &abi = signature[index];
        const VernonAdValue *value = findValue(supplied, abi.path);
        const ProgramLeafBinding &binding = bindings[index];
        if (!value || !matchesProgramValueAbi(*value, abi) || binding.value >= storage.size())
            return error = "Program autodiff leaf does not match graph reflection", false;
        auto *packed = static_cast<uint8_t *>(const_cast<void *>(storage[binding.value].argument.tensor.host_data));
        auto *leaf = static_cast<uint8_t *>(const_cast<void *>(value->data));
        size_t elementCount = binding.elementCount;
        if (!elementCount) {
            const size_t bytes = publish ? storage[binding.value].argument.tensor.byte_size : value->size;
            if (!binding.elementStride || bytes % binding.elementStride)
                return error = "Program autodiff leaf does not match its dynamic footprint", false;
            elementCount = bytes / binding.elementStride;
        }
        for (size_t element = 0; element < elementCount; ++element) {
            uint8_t *packedElement = packed + element * binding.elementStride + binding.byteOffset;
            uint8_t *leafElement = leaf + element * binding.leafElementBytes;
            if (publish) {
                PendingPublication copy;
                copy.destination = leafElement;
                copy.bytes.assign(packedElement, packedElement + binding.leafElementBytes);
                pending.push_back(std::move(copy));
            } else
                std::memcpy(packedElement, leafElement, binding.leafElementBytes);
        }
    }
    // Publication is a commit-after-success boundary: validate and stage every
    // leaf before mutating any caller-owned destination.
    for (const PendingPublication &copy : pending)
        std::memcpy(copy.destination, copy.bytes.data(), copy.bytes.size());
    return true;
}

VernonStatus fail(VernonRuntimeContext &context, const std::string &error) {
    invocationDiagnostic(context) = error;
    return VERNON_STATUS_INVALID_ARGUMENT;
}

struct ProgramPullbackState {
    ProgramPullbackState(ProgramResidualPlan plan, LogicalValueFrame values)
        : plan(std::move(plan)), values(std::make_unique<const LogicalValueFrame>(std::move(values))) {}

    const ProgramResidualPlan plan;
    const std::unique_ptr<const LogicalValueFrame> values;
};

class ProgramPullback final : public PullbackExecution {
public:
    ProgramPullback(VernonRuntimeContext &context, std::shared_ptr<VernonProgramTopology> topology, Variant variant,
                    Signature signature, std::vector<ProgramLeafBinding> cotangentBindings,
                    std::vector<ProgramLeafBinding> gradientBindings, std::shared_ptr<const ProgramPullbackState> state,
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
        if (!topology_->resolvedProgram || !state_ || !state_->values)
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
        if (state_->plan.replayEnd) {
            const program::Graph *forward = program::findGraph(execution, "forward");
            if (!forward)
                return fail(*context_, "Program pullback replay has no forward graph");
            replay = *forward;
            replay->nodes.resize(state_->plan.replayEnd);
            program::markGraphValues(*replay, required);
        }
        for (uint32_t value : state_->plan.retainedValues)
            if (value < required.size())
                required[value] = 1;

        std::vector<std::vector<uint8_t>> captures(execution.values.size());
        std::vector<std::vector<uint64_t>> captureShapes(execution.values.size());
        for (uint32_t value : state_->plan.retainedValues)
            if (value < captureShapes.size() && value < state_->values->hostValues().size() &&
                state_->values->hostValues()[value].concreteShape)
                captureShapes[value] = *state_->values->hostValues()[value].concreteShape;
        std::vector<LogicalProgramValue> storage;
        PullbackInvocationSpec frame{
            required,
            {cotangents, cotangents ? &signature_.cotangents : nullptr, cotangents ? &cotangentBindings_ : nullptr},
            {&gradients, &signature_.gradients, &gradientBindings_},
            captures,
            captureShapes,
            nullptr,
            state_->values.get(),
        };
        std::string error;
        if (!buildLogicalValueFrame(*context_, execution, topology_.get(), storage, frame,
                                    context_->autodiffMemoryPolicy, error) ||
            (cotangents &&
             !transferLeaves(*cotangents, signature_.cotangents, cotangentBindings_, storage, false, error)))
            return fail(*context_, error);

        size_t temporaryBytes = 0;
        for (const LogicalProgramValue &value : storage) {
            if (value.owned.size() > std::numeric_limits<size_t>::max() - temporaryBytes)
                return fail(*context_, "Program pullback temporary memory accounting overflows");
            temporaryBytes += value.owned.size();
        }
        if (temporaryBytes > options.maximumTemporaryBytes)
            return fail(*context_, "Program pullback exceeds its temporary memory limit");

        LogicalValueFrame values(std::move(storage));
        const bool device = context_->backend != VERNON_RUNTIME_CPU;
        std::vector<char> deviceRequired(execution.values.size());
        markDeviceResidentValues(execution, *backward, *topology_, deviceRequired);
        if (replay)
            markDeviceResidentValues(execution, *replay, *topology_, deviceRequired);
        for (uint32_t value : state_->plan.retainedValues)
            if (!values.adoptRetainedValue(value, *state_->values, error))
                return fail(*context_, error);
        if (device && !values.materializeDevice(*context_, deviceRequired, error))
            return fail(*context_, error);
        VernonProgramExecutable proxy;
        proxy.context = context_;
        proxy.variant = variant_;
        proxy.topology = topology_;
        if (replay) {
            if (device) {
                std::vector<ProgramTapeState> tapeStates;
                if (!prepareProgramTapeStates(values, *context_, execution, *topology_, *replay, tapeStates, error))
                    return fail(*context_, error);
                for (;;) {
                    const VernonStatus replayStatus = executePipelineProgramGraph(proxy, *replay, values);
                    if (replayStatus != VERNON_STATUS_OK)
                        return replayStatus;
                    bool retry = false;
                    if (!validateProgramTapeStates(values, tapeStates, retry, error))
                        return fail(*context_, error);
                    if (!retry)
                        break;
                    if (!values.restoreDeviceValuesFromHost(required, error))
                        return fail(*context_, error);
                    for (ProgramTapeState &tape : tapeStates)
                        if (!allocateProgramTapeState(values, *context_, tape, error))
                            return fail(*context_, error);
                }
            } else {
                const VernonStatus replayStatus = executePipelineProgramGraph(proxy, *replay, values);
                if (replayStatus != VERNON_STATUS_OK)
                    return replayStatus;
                if (!sealProgramTapeValues(values.hostValues(), values.logicalArguments(), error))
                    return fail(*context_, error);
            }
        }
        const VernonStatus status = executePipelineProgramGraph(proxy, *backward, values);
        if (status != VERNON_STATUS_OK)
            return status;

        if (device) {
            std::vector<char> downloads(execution.values.size());
            for (const ProgramLeafBinding &binding : gradientBindings_)
                if (binding.value < downloads.size())
                    downloads[binding.value] = 1;
            if (!values.downloadLogicalToHost(downloads, error))
                return fail(*context_, error);
            ++usage_.readbacks;
        }
        if (!transferLeaves(gradients, signature_.gradients, gradientBindings_, values.hostValues(), true, error))
            return fail(*context_, error);
        ++usage_.submissions;
        ++usage_.waits;
        ++usage_.atomicPublications;
        usage_.temporaryAllocationBytes += temporaryBytes;
        return VERNON_STATUS_OK;
    }

    PullbackMemoryUsage memoryUsage() const override {
        MemoryAccounting memory;
        memory.logicalPayloadBytes = state_->plan.checkpoint.logicalResidualBytes;
        const auto add = [](size_t &total, size_t bytes) {
            if (bytes > std::numeric_limits<size_t>::max() - total)
                total = std::numeric_limits<size_t>::max();
            else
                total += bytes;
        };
        for (const LogicalProgramValue &value : state_->values->hostValues()) {
            const auto &batch = value.tapeBatch;
            if (!batch) {
                add(memory.residentBytes, value.owned.size());
                add(memory.retainedAllocationBytes, value.owned.size());
            } else {
                add(memory.residentBytes, batch->isCompacted() ? batch->logicalBytes() : batch->residentBytes());
                add(memory.allocatedBytes, batch->allocatedBytes());
            }
        }
        memory.retainedAllocationBytes = std::max(memory.retainedAllocationBytes,
                                                  static_cast<size_t>(state_->plan.checkpoint.retainedAllocationBytes));
        add(memory.retainedAllocationBytes, memory.allocatedBytes);
        return pullbackMemoryUsage(memory);
    }

    PullbackControlPlaneUsage controlPlaneUsage() const override {
        const std::lock_guard lock(applyMutex_);
        return usage_;
    }

    AutodiffPullbackCheckpointPlan checkpointPlan() const override {
        AutodiffPullbackCheckpointPlan snapshot;
        snapshot.present = true;
        snapshot.peakBytes = state_->plan.checkpoint.peakBytes;
        snapshot.memoryBudget = state_->plan.checkpoint.memoryBudget;
        snapshot.logicalResidualBytes = state_->plan.checkpoint.logicalResidualBytes;
        snapshot.retainedAllocationBytes = state_->plan.checkpoint.retainedAllocationBytes;
        snapshot.initialStateBytes = state_->plan.checkpoint.initialStateBytes;
        snapshot.restorationBytes = state_->plan.checkpoint.restorationBytes;
        snapshot.transactionBytes = state_->plan.checkpoint.transactionBytes;
        snapshot.persistentCheckpointBytes = state_->plan.checkpoint.persistentCheckpointBytes;
        snapshot.backwardValueBytes = state_->plan.checkpoint.backwardValueBytes;
        snapshot.replayCost = state_->plan.checkpoint.replayCost;
        snapshot.recomputationCost = state_->plan.checkpoint.recomputationCost;
        snapshot.selectedPolicy = state_->plan.checkpoint.selectedPolicy;
        return snapshot;
    }

    std::vector<AutodiffPullbackPassTelemetry> passTelemetry() const override { return passTelemetry_; }

    uint64_t peakRuntimeManagedBytes() const override {
        const PullbackMemoryUsage usage = memoryUsage();
        const uint64_t held =
            std::max(std::max(usage.retainedAllocationBytes, usage.allocatedBytes), usage.peakTemporaryBytes);
        return std::max(held, state_->plan.checkpoint.peakBytes);
    }

private:
    VernonRuntimeContext *context_;
    std::shared_ptr<VernonProgramTopology> topology_;
    Variant variant_;
    Signature signature_;
    std::vector<ProgramLeafBinding> cotangentBindings_;
    std::vector<ProgramLeafBinding> gradientBindings_;
    std::shared_ptr<const ProgramPullbackState> state_;
    std::vector<AutodiffPullbackPassTelemetry> passTelemetry_;
    mutable std::mutex applyMutex_;
    PullbackControlPlaneUsage usage_;
};

class ProgramExecutable final : public Executable {
public:
    ProgramExecutable(VernonRuntimeContext &context, std::weak_ptr<VernonProgramTopology> topology, Variant variant)
        : context_(&context), topology_(std::move(topology)), variant_(std::move(variant)) {
        rebuildVariantLayoutViews(variant_);
        const std::shared_ptr<VernonProgramTopology> locked = topology_.lock();
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
            const bool aliasesInput =
                std::any_of(canonicalProgram.abi.boundarySlots.begin(), canonicalProgram.abi.boundarySlots.end(),
                            [&](const program::BoundarySlot &candidate) {
                                return candidate.role == program::BoundaryRole::Input &&
                                       candidate.aliasOwner.kind == slot.aliasOwner.kind &&
                                       candidate.aliasOwner.id == slot.aliasOwner.id;
                            });
            if (slot.role == program::BoundaryRole::Input ||
                (slot.role == program::BoundaryRole::Output && !aliasesInput &&
                 program::findPublicationTarget(canonicalProgram.abi, slot.id)))
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
                leafBindings.push_back({publicSlot.value, leaf.byteOffset, layout.byteSize,
                                        dtype ? dtypeSize(*dtype) * static_cast<size_t>(leaf.scalarCount) : 0,
                                        elementCount});
            }
        };
        for (const program::BoundarySlot &slot : canonicalProgram.abi.boundarySlots) {
            if (slot.role == program::BoundaryRole::Input)
                append(slot, nullptr, signature_.inputs, inputBindings_);
            else if (slot.role == program::BoundaryRole::Output &&
                     program::findPublicationTarget(canonicalProgram.abi, slot.id))
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

    VernonStatus forward(const ForwardExecutionTarget &target, VernonLaunchSize, const VernonAdValueSet &inputs,
                         VernonAdValueSet *outputs, std::unique_ptr<PullbackExecution> &pullback) override {
        const std::shared_ptr<VernonProgramTopology> topology = topology_.lock();
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
        std::vector<PendingProgramPublication> publications;
        ForwardInvocationSpec frame{
            required,
            canonicalInvocation ? std::variant<CanonicalForwardBindings, HostForwardBindings>(
                                      CanonicalForwardBindings{*target.invocation, forwardBindings_, publications})
                                : std::variant<CanonicalForwardBindings, HostForwardBindings>(HostForwardBindings{
                                      {&inputs, &signature_.inputs, &inputBindings_},
                                      {outputs, &signature_.outputs, &outputBindings_},
                                  }),
        };
        if (!buildLogicalValueFrame(*context_, *execution, topology.get(), hostStorage, frame,
                                    context_->autodiffMemoryPolicy, error) ||
            (!canonicalInvocation &&
             !transferLeaves(inputs, signature_.inputs, inputBindings_, hostStorage, false, error)))
            return fail(*context_, error);
        LogicalValueFrame arena(std::move(hostStorage));
        arena.setInvocationContext(target.programContext);
        VernonProgramExecutable proxy;
        proxy.context = context_;
        proxy.variant = variant_;
        proxy.topology = topology;
        for (const program::Node &node : forward->nodes)
            if (program::executionKind(node) == program::ExecutionKind::Compute)
                for (const program::ControlComponent &control : program::computeOperation(node).workgroups) {
                    uint64_t value = 0;
                    if (!arena.resolveControl(*execution, control, value, error))
                        return fail(*context_, error);
                }
        const bool device = context_->backend != VERNON_RUNTIME_CPU;
        if (device) {
            std::vector<char> deviceRequired(execution->values.size());
            markDeviceResidentValues(*execution, *forward, *topology, deviceRequired);
            if (!arena.materializeDevice(*context_, deviceRequired, error))
                return fail(*context_, error);
            std::vector<ProgramTapeState> tapeStates;
            if (!prepareProgramTapeStates(arena, *context_, *execution, *topology, *forward, tapeStates, error))
                return fail(*context_, error);
            for (;;) {
                const VernonStatus status = executePipelineProgramGraph(proxy, *forward, arena);
                if (status != VERNON_STATUS_OK)
                    return status;
                bool retry = false;
                if (!validateProgramTapeStates(arena, tapeStates, retry, error))
                    return fail(*context_, error);
                if (!retry)
                    break;
                if (!arena.restoreDeviceValuesFromHost(required, error))
                    return fail(*context_, error);
                for (ProgramTapeState &state : tapeStates)
                    if (!allocateProgramTapeState(arena, *context_, state, error))
                        return fail(*context_, error);
            }
        } else {
            const VernonStatus status = executePipelineProgramGraph(proxy, *forward, arena);
            if (status != VERNON_STATUS_OK)
                return status;
            if (!validateProgramTapeValues(arena.hostValues(), error))
                return fail(*context_, error);
        }

        if (device) {
            if (const VernonStatus publicationStatus =
                    commitDeviceProgramPublications(*context_, arena, publications, error);
                publicationStatus != VERNON_STATUS_OK)
                return error.empty() ? publicationStatus : fail(*context_, error);
            std::vector<char> downloads(execution->values.size());
            for (const PendingProgramPublication &publication : publications)
                if (!publication.destinationBuffer && publication.target &&
                    publication.target->value < downloads.size())
                    downloads[publication.target->value] = 1;
            if (!canonicalInvocation)
                for (const ProgramLeafBinding &binding : outputBindings_)
                    if (binding.value < downloads.size())
                        downloads[binding.value] = 1;
            if (!arena.downloadLogicalToHost(downloads, error))
                return fail(*context_, error);
        }
        if (!commitProgramPublications(arena.hostValues(), publications, error))
            return fail(*context_, error);
        if (!canonicalInvocation &&
            !transferLeaves(*outputs, signature_.outputs, outputBindings_, arena.hostValues(), true, error))
            return fail(*context_, error);
        if (!hasBackward)
            return VERNON_STATUS_OK;

        ProgramResidualPlan plan;
        uint64_t memoryBudget = context_->autodiffMemoryPolicy->invocationLimit();
        const bool rematerializeTapes = topology->programCheckpointMemoryBudget.has_value();
        if (rematerializeTapes)
            memoryBudget = *topology->programCheckpointMemoryBudget;
        if (!planProgramResiduals(*execution, topology.get(), variant_, arena.hostValues(), memoryBudget,
                                  topology->programCheckpointPolicy, rematerializeTapes, plan, error))
            return fail(*context_, error);
        for (uint32_t value : plan.retainedValues) {
            if (value >= arena.hostValues().size())
                return fail(*context_, "Program pullback residual plan references an unknown Value");
            LogicalProgramValue &slot = arena.hostValues()[value];
            if (slot.tapeBatch) {
                if (!slot.tapeBatch->compact(true) || !fillProgramTapeHostValue(slot, slot.tapeBatch, error))
                    return fail(*context_, error.empty() ? "Program autodiff tape could not be compacted" : error);
                arena.logicalArguments()[value] = slot.argument;
            }
        }
        std::vector<AutodiffPullbackPassTelemetry> telemetry =
            collectProgramPassTelemetry(*forward, *execution, arena.hostValues(), plan);
        std::vector<char> retained(execution->values.size());
        for (uint32_t value : plan.retainedValues)
            if (value < retained.size())
                retained[value] = 1;
        arena.retainOnly(retained);
        auto state = std::make_shared<const ProgramPullbackState>(std::move(plan), std::move(arena));
        pullback = std::make_unique<ProgramPullback>(*context_, topology, variant_, signature_, cotangentBindings_,
                                                     gradientBindings_, std::move(state), std::move(telemetry));
        return VERNON_STATUS_OK;
    }

private:
    VernonRuntimeContext *context_;
    std::weak_ptr<VernonProgramTopology> topology_;
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
    if (!pipeline.context || !pipeline.topology || !pipeline.topology->resolvedProgram)
        return false;
    const program::Program &execution = pipeline.topology->resolvedProgram->program;
    if (!program::findGraph(execution, "forward")) {
        invocationDiagnostic(*pipeline.context) = "Program execution topology requires a forward graph";
        return false;
    }
    if (!pipeline.variant.parameters.empty()) {
        invocationDiagnostic(*pipeline.context) =
            "managed Program public parameters must come exclusively from compiler-emitted ProgramABI";
        return false;
    }
    if (!pipeline.context->autodiffMemoryPolicy)
        pipeline.context->autodiffMemoryPolicy = std::make_shared<AutodiffMemoryPolicy>();
    std::vector<ValueLayout> &layoutViews = pipeline.topology->boundaryLayoutViews;
    layoutViews.clear();
    layoutViews.resize(execution.abi.boundarySlots.size());
    for (size_t index = 0; index < execution.abi.boundarySlots.size(); ++index)
        if (const std::optional<program::ValueLayout> &layout = execution.abi.boundarySlots[index].layout) {
            layoutViews[index] =
                program::materializeValueLayout(*layout, execution.abi.boundarySlots[index].logicalType);
            rebuildValueLayoutPathViews(layoutViews[index]);
        }
    auto executable = std::make_shared<ProgramExecutable>(*pipeline.context, pipeline.topology, pipeline.variant);
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
    pipeline.differentiated = VernonDifferentiatedProgram{std::move(executable), std::move(groups)};
    return true;
}

} // namespace vernon::runtime::ad
