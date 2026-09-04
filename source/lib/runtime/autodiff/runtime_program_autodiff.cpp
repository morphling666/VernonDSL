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
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vernon::runtime::ad {
namespace {

using HostProgramValue = ProgramHostValue;

bool sealProgramTapeValues(std::vector<HostProgramValue> &storage, std::vector<VernonPipelineArgument> &values,
                           std::string &error) {
    if (values.size() < storage.size())
        return error = "Program autodiff tape value arena is incomplete", false;
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

class ProgramPullback final : public PullbackExecution {
public:
    ProgramPullback(VernonRuntimeContext &context, std::shared_ptr<VernonPipelineTopology> topology, Variant variant,
                    Signature signature, std::vector<ProgramLeafBinding> cotangentBindings,
                    std::vector<ProgramLeafBinding> gradientBindings, ProgramResidualPlan plan,
                    std::vector<std::vector<uint8_t>> residuals, std::vector<std::vector<uint64_t>> residualShapes,
                    std::vector<std::shared_ptr<HostStaticTapeBatch>> tapeResiduals,
                    std::vector<AutodiffPullbackPassTelemetry> passTelemetry,
                    std::unique_ptr<ProgramInvocationFrame> deviceState = nullptr)
        : context_(&context), topology_(std::move(topology)), variant_(std::move(variant)),
          signature_(std::move(signature)), cotangentBindings_(std::move(cotangentBindings)),
          gradientBindings_(std::move(gradientBindings)), plan_(std::move(plan)), residuals_(std::move(residuals)),
          residualShapes_(std::move(residualShapes)), tapeResiduals_(std::move(tapeResiduals)),
          passTelemetry_(std::move(passTelemetry)), deviceState_(std::move(deviceState)) {
        rebuildVariantLayoutViews(variant_);
    }

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients,
                       const PullbackApplyOptions &options) override {
        if (!topology_->resolvedProgram)
            return fail(*context_, "Program pullback has no resolved canonical owner");
        const program::Program &execution = topology_->resolvedProgram->program;
        const program::Graph *backward = program::findGraph(execution, "backward");
        const bool requiresCotangents =
            backward &&
            std::any_of(backward->inputs.begin(), backward->inputs.end(), [](const program::GraphInput &input) {
                return input.kind == program::GraphInputKind::UserInput;
            });
        if (!backward || (requiresCotangents && !cotangents))
            return fail(*context_, "Program pullback has no backward graph or required cotangents");
        if (deviceState_) {
            std::vector<HostProgramValue> storage;
            std::string error;
            std::vector<char> required(execution.values.size());
            program::markGraphValues(*backward, required);
            std::vector<std::vector<uint8_t>> captureBytes(execution.values.size());
            std::vector<std::vector<uint64_t>> captureShapes(execution.values.size());
            for (uint32_t value : program::residualCaptures(execution))
                if (value < captureShapes.size() && value < deviceState_->hostValues().size() &&
                    deviceState_->hostValues()[value].concreteShape)
                    captureShapes[value] = *deviceState_->hostValues()[value].concreteShape;
            PullbackInvocationSpec frame{
                required,
                {cotangents, cotangents ? &signature_.cotangents : nullptr, cotangents ? &cotangentBindings_ : nullptr},
                {&gradients, &signature_.gradients, &gradientBindings_},
                captureBytes,
                captureShapes,
            };
            if (!buildProgramInvocationFrame(*context_, execution, topology_.get(), storage, frame,
                                             context_->autodiffMemoryPolicy, error) ||
                (cotangents &&
                 !transferLeaves(*cotangents, signature_.cotangents, cotangentBindings_, storage, false, error)))
                return fail(*context_, error);
            ProgramInvocationFrame derivatives(std::move(storage));
            if (!derivatives.materializeDevice(*context_, required, error))
                return fail(*context_, error);
            for (uint32_t value : program::residualCaptures(execution))
                if (!derivatives.adoptRetainedValue(value, *deviceState_, error))
                    return fail(*context_, error);
            VernonLoadedPipeline proxy;
            proxy.context = context_;
            proxy.variant = variant_;
            proxy.topology = topology_;
            const VernonStatus status = executePipelineProgramGraph(proxy, *backward, derivatives);
            if (status != VERNON_STATUS_OK)
                return status;
            std::vector<char> downloads(execution.values.size());
            for (const ProgramLeafBinding &binding : gradientBindings_)
                if (binding.value < downloads.size())
                    downloads[binding.value] = 1;
            if (!derivatives.downloadLogicalToHost(downloads, error) ||
                !transferLeaves(gradients, signature_.gradients, gradientBindings_, derivatives.hostValues(), true,
                                error))
                return fail(*context_, error);
            ++usage_.submissions;
            ++usage_.waits;
            ++usage_.readbacks;
            ++usage_.atomicPublications;
            return VERNON_STATUS_OK;
        }
        std::vector<HostProgramValue> storage;
        std::string error;
        std::vector<char> required(execution.values.size());
        program::markGraphValues(*backward, required);
        if (plan_.replayEnd) {
            const program::Graph *forward = program::findGraph(execution, "forward");
            if (!forward)
                return fail(*context_, "Program pullback replay has no forward graph");
            program::Graph replay = *forward;
            replay.nodes.resize(plan_.replayEnd);
            program::markGraphValues(replay, required);
        }
        for (uint32_t value : program::residualCaptures(execution))
            if (value < required.size())
                required[value] = 1;
        for (uint32_t value : plan_.retainedValues)
            if (value < required.size())
                required[value] = 1;
        PullbackInvocationSpec frame{
            required,
            {cotangents, cotangents ? &signature_.cotangents : nullptr, cotangents ? &cotangentBindings_ : nullptr},
            {&gradients, &signature_.gradients, &gradientBindings_},
            residuals_,
            residualShapes_,
            &tapeResiduals_,
        };
        if (!buildProgramInvocationFrame(*context_, execution, topology_.get(), storage, frame,
                                         context_->autodiffMemoryPolicy, error))
            return fail(*context_, error);
        if (cotangents &&
            !transferLeaves(*cotangents, signature_.cotangents, cotangentBindings_, storage, false, error))
            return fail(*context_, error);
        size_t temporaryBytes = 0;
        for (const HostProgramValue &value : storage) {
            if (value.owned.size() > std::numeric_limits<size_t>::max() - temporaryBytes)
                return fail(*context_, "Program pullback temporary memory accounting overflows");
            temporaryBytes += value.owned.size();
        }
        if (temporaryBytes > options.maximumTemporaryBytes)
            return fail(*context_, "Program pullback exceeds its temporary memory limit");
        for (uint32_t value : plan_.retainedValues) {
            if (value >= residuals_.size() || residuals_[value].size() != storage[value].argument.tensor.byte_size)
                return fail(*context_, "Program pullback residual state is incomplete");
            if (value < tapeResiduals_.size() && tapeResiduals_[value])
                continue;
            std::memcpy(const_cast<void *>(storage[value].argument.tensor.host_data), residuals_[value].data(),
                        residuals_[value].size());
        }
        ProgramInvocationFrame arena(std::move(storage));
        VernonLoadedPipeline proxy;
        proxy.context = context_;
        proxy.variant = variant_;
        proxy.topology = topology_;
        if (plan_.replayEnd) {
            const program::Graph *forward = program::findGraph(execution, "forward");
            program::Graph replay = *forward;
            replay.nodes.resize(plan_.replayEnd);
            const VernonStatus replayStatus = executePipelineProgramGraph(proxy, replay, arena);
            if (replayStatus != VERNON_STATUS_OK)
                return replayStatus;
            if (!sealProgramTapeValues(arena.hostValues(), arena.logicalArguments(), error))
                return fail(*context_, error);
        }
        const VernonStatus status = executePipelineProgramGraph(proxy, *backward, arena);
        if (status == VERNON_STATUS_OK &&
            transferLeaves(gradients, signature_.gradients, gradientBindings_, arena.hostValues(), true, error)) {
            ++usage_.submissions;
            ++usage_.waits;
            ++usage_.atomicPublications;
            usage_.temporaryAllocationBytes += temporaryBytes;
            return status;
        }
        return status == VERNON_STATUS_OK ? fail(*context_, error) : status;
    }

    PullbackMemoryUsage memoryUsage() const override {
        MemoryAccounting memory;
        memory.logicalPayloadBytes = plan_.checkpoint.logicalResidualBytes;
        const auto add = [](size_t &total, size_t bytes) {
            if (bytes > std::numeric_limits<size_t>::max() - total)
                total = std::numeric_limits<size_t>::max();
            else
                total += bytes;
        };
        for (const auto &batch : tapeResiduals_) {
            if (!batch)
                continue;
            // compact(true) keeps write descriptors for the packed Program tape ABI.
            // Graph AD compact(false) drops that construction storage from resident.
            add(memory.residentBytes, batch->isCompacted() ? batch->logicalBytes() : batch->residentBytes());
            add(memory.allocatedBytes, batch->allocatedBytes());
        }
        for (size_t value = 0; value < residuals_.size(); ++value) {
            const std::vector<uint8_t> &residual = residuals_[value];
            if (residual.empty())
                continue;
            if (value >= tapeResiduals_.size() || !tapeResiduals_[value])
                add(memory.residentBytes, residual.size());
            add(memory.retainedAllocationBytes, residual.size());
        }
        add(memory.retainedAllocationBytes, memory.allocatedBytes);
        return pullbackMemoryUsage(memory);
    }

    PullbackControlPlaneUsage controlPlaneUsage() const override { return usage_; }

    AutodiffPullbackCheckpointPlan checkpointPlan() const override {
        AutodiffPullbackCheckpointPlan snapshot;
        snapshot.present = true;
        snapshot.peakBytes = plan_.checkpoint.peakBytes;
        snapshot.memoryBudget = plan_.checkpoint.memoryBudget;
        snapshot.logicalResidualBytes = plan_.checkpoint.logicalResidualBytes;
        snapshot.retainedAllocationBytes = plan_.checkpoint.retainedAllocationBytes;
        snapshot.initialStateBytes = plan_.checkpoint.initialStateBytes;
        snapshot.restorationBytes = plan_.checkpoint.restorationBytes;
        snapshot.transactionBytes = plan_.checkpoint.transactionBytes;
        snapshot.persistentCheckpointBytes = plan_.checkpoint.persistentCheckpointBytes;
        snapshot.backwardValueBytes = plan_.checkpoint.backwardValueBytes;
        snapshot.replayCost = plan_.checkpoint.replayCost;
        snapshot.recomputationCost = plan_.checkpoint.recomputationCost;
        snapshot.selectedPolicy = plan_.checkpoint.selectedPolicy;
        return snapshot;
    }

    std::vector<AutodiffPullbackPassTelemetry> passTelemetry() const override { return passTelemetry_; }

    uint64_t peakRuntimeManagedBytes() const override {
        const PullbackMemoryUsage usage = memoryUsage();
        const uint64_t held =
            std::max(std::max(usage.retainedAllocationBytes, usage.allocatedBytes), usage.peakTemporaryBytes);
        return std::max(held, plan_.checkpoint.peakBytes);
    }

private:
    VernonRuntimeContext *context_;
    std::shared_ptr<VernonPipelineTopology> topology_;
    Variant variant_;
    Signature signature_;
    std::vector<ProgramLeafBinding> cotangentBindings_;
    std::vector<ProgramLeafBinding> gradientBindings_;
    ProgramResidualPlan plan_;
    std::vector<std::vector<uint8_t>> residuals_;
    std::vector<std::vector<uint64_t>> residualShapes_;
    std::vector<std::shared_ptr<HostStaticTapeBatch>> tapeResiduals_;
    std::vector<AutodiffPullbackPassTelemetry> passTelemetry_;
    std::unique_ptr<ProgramInvocationFrame> deviceState_;
    PullbackControlPlaneUsage usage_;
};

class ProgramExecutable final : public Executable {
public:
    ProgramExecutable(VernonRuntimeContext &context, std::weak_ptr<VernonPipelineTopology> topology, Variant variant)
        : context_(&context), topology_(std::move(topology)), variant_(std::move(variant)) {
        rebuildVariantLayoutViews(variant_);
        const std::shared_ptr<VernonPipelineTopology> locked = topology_.lock();
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
        for (const program::BoundarySlot &slot : canonicalProgram.abi.boundarySlots)
            if (slot.role == program::BoundaryRole::Input ||
                (slot.role == program::BoundaryRole::Output &&
                 program::findPublicationTarget(canonicalProgram.abi, slot.id)))
                forwardBindings_.emplace_back(slot.id, slot.value);
        // ProgramABI is the execution authority. Signature is validated against
        // it while parsing, but runtime binding and leaf materialization must
        // not join the two representations again.
        const auto append = [&](const program::BoundarySlot &publicSlot, std::vector<ValueAbi> &values,
                                std::vector<ProgramLeafBinding> &leafBindings) {
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
            const size_t begin = values.size();
            if (!appendValueLayoutAbi(layout, publicSlot.outerShape, publicSlot.path, values, error)) {
                signatureError_ = std::move(error);
                return;
            }
            size_t elementCount = 1;
            for (uint64_t extent : publicSlot.outerShape)
                elementCount *= static_cast<size_t>(extent);
            if (values.size() - begin != layout.leaves.size()) {
                signatureError_ = "Program aggregate ABI contains duplicate leaf paths";
                return;
            }
            for (const ValueLeaf &leaf : layout.leaves) {
                const std::optional<VernonDataType> dtype = pipelineDataType(leaf.dtype);
                leafBindings.push_back({publicSlot.value, leaf.byteOffset, layout.byteSize,
                                        dtype ? dtypeSize(*dtype) * static_cast<size_t>(leaf.scalarCount) : 0,
                                        elementCount});
            }
        };
        for (const program::BoundarySlot &slot : canonicalProgram.abi.boundarySlots) {
            if (slot.role == program::BoundaryRole::Input)
                append(slot, signature_.inputs, inputBindings_);
            else if (slot.role == program::BoundaryRole::Output &&
                     program::findPublicationTarget(canonicalProgram.abi, slot.id))
                append(slot, signature_.outputs, outputBindings_);
        }
        for (const program::DerivativeProjection &projection : canonicalProgram.abi.derivativeProjections) {
            const program::BoundarySlot &derivative = canonicalProgram.abi.boundarySlots[projection.derivative.slot];
            if (derivative.role == program::BoundaryRole::Cotangent)
                append(derivative, signature_.cotangents, cotangentBindings_);
            else if (derivative.role == program::BoundaryRole::Gradient)
                append(derivative, signature_.gradients, gradientBindings_);
            else {
                signatureError_ = "DerivativeProjection does not reference a derivative boundary";
                return;
            }
        }
    }

    const Signature &signature() const override { return signature_; }

    VernonStatus forward(const ForwardExecutionTarget &target, VernonLaunchSize, const VernonAdValueSet &inputs,
                         VernonAdValueSet *outputs, std::unique_ptr<PullbackExecution> &pullback) override {
        const std::shared_ptr<VernonPipelineTopology> topology = topology_.lock();
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
        if (!buildProgramInvocationFrame(*context_, *execution, topology.get(), hostStorage, frame,
                                         context_->autodiffMemoryPolicy, error) ||
            (!canonicalInvocation &&
             !transferLeaves(inputs, signature_.inputs, inputBindings_, hostStorage, false, error)))
            return fail(*context_, error);
        ProgramInvocationFrame arena(std::move(hostStorage));
        arena.setInvocationContext(target.programContext);
        VernonLoadedPipeline proxy;
        proxy.context = context_;
        proxy.variant = variant_;
        proxy.topology = topology;
        if (context_->backend != VERNON_RUNTIME_CPU) {
            if (!arena.materializeDevice(*context_, required, error))
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
            if (const VernonStatus publicationStatus =
                    commitDeviceProgramPublications(*context_, arena, publications, error);
                publicationStatus != VERNON_STATUS_OK)
                return error.empty() ? publicationStatus : fail(*context_, error);
            if (!publications.empty()) {
                std::vector<char> downloads(execution->values.size());
                for (const PendingProgramPublication &publication : publications)
                    if (!publication.destinationBuffer && publication.target &&
                        publication.target->value < downloads.size())
                        downloads[publication.target->value] = 1;
                if (!arena.downloadLogicalToHost(downloads, error) ||
                    !commitProgramPublications(arena.hostValues(), publications, error))
                    return fail(*context_, error);
            }
            if (!canonicalInvocation) {
                std::vector<char> downloads(execution->values.size());
                for (const ProgramLeafBinding &binding : outputBindings_)
                    if (binding.value < downloads.size())
                        downloads[binding.value] = 1;
                if (!arena.downloadLogicalToHost(downloads, error) ||
                    !transferLeaves(*outputs, signature_.outputs, outputBindings_, arena.hostValues(), true, error))
                    return fail(*context_, error);
            }
            if (!hasBackward)
                return VERNON_STATUS_OK;
            ProgramResidualPlan plan;
            plan.retainedValues = program::residualCaptures(*execution);
            std::vector<AutodiffPullbackPassTelemetry> telemetry =
                collectProgramPassTelemetry(*forward, *execution, arena.hostValues(), plan);
            pullback = std::make_unique<ProgramPullback>(
                *context_, topology, variant_, signature_, cotangentBindings_, gradientBindings_, std::move(plan),
                std::vector<std::vector<uint8_t>>(execution->values.size()),
                std::vector<std::vector<uint64_t>>(execution->values.size()),
                std::vector<std::shared_ptr<HostStaticTapeBatch>>(execution->values.size()), std::move(telemetry),
                std::make_unique<ProgramInvocationFrame>(std::move(arena)));
            return VERNON_STATUS_OK;
        }
        const VernonStatus status = executePipelineProgramGraph(proxy, *forward, arena);
        if (status != VERNON_STATUS_OK)
            return status;
        std::vector<HostProgramValue> &storage = arena.hostValues();
        if (!commitProgramPublications(storage, publications, error))
            return fail(*context_, error);
        if (!canonicalInvocation &&
            !transferLeaves(*outputs, signature_.outputs, outputBindings_, storage, true, error))
            return fail(*context_, error);
        if (!hasBackward)
            return VERNON_STATUS_OK;
        ProgramResidualPlan plan;
        uint64_t memoryBudget = context_->autodiffMemoryPolicy->invocationLimit();
        const bool rematerializeTapes = topology->programCheckpointMemoryBudget.has_value();
        if (rematerializeTapes)
            memoryBudget = *topology->programCheckpointMemoryBudget;
        if (!planProgramResiduals(*execution, topology.get(), variant_, storage, memoryBudget,
                                  topology->programCheckpointPolicy, rematerializeTapes, plan, error))
            return fail(*context_, error);
        std::vector<std::vector<uint8_t>> residuals(execution->values.size());
        std::vector<std::vector<uint64_t>> residualShapes(execution->values.size());
        std::vector<std::shared_ptr<HostStaticTapeBatch>> tapeResiduals(execution->values.size());
        for (uint32_t value : program::residualCaptures(*execution))
            if (storage[value].concreteShape)
                residualShapes[value] = *storage[value].concreteShape;
        for (uint32_t value : plan.retainedValues) {
            HostProgramValue &slot = storage[value];
            if (slot.tapeBatch) {
                if (!slot.tapeBatch->compact(true))
                    return fail(*context_, "Program autodiff tape could not be compacted");
                if (!fillProgramTapeHostValue(slot, slot.tapeBatch, error))
                    return fail(*context_, error);
                tapeResiduals[value] = slot.tapeBatch;
            }
            const auto *data = static_cast<const uint8_t *>(slot.argument.tensor.host_data);
            if (!data || !slot.argument.tensor.byte_size)
                return fail(*context_, "Program autodiff residual state is incomplete");
            residuals[value].assign(data, data + slot.argument.tensor.byte_size);
            if (slot.concreteShape)
                residualShapes[value] = *slot.concreteShape;
        }
        std::vector<AutodiffPullbackPassTelemetry> telemetry =
            collectProgramPassTelemetry(*forward, *execution, storage, plan);
        pullback = std::make_unique<ProgramPullback>(
            *context_, topology, variant_, signature_, cotangentBindings_, gradientBindings_, std::move(plan),
            std::move(residuals), std::move(residualShapes), std::move(tapeResiduals), std::move(telemetry));
        return VERNON_STATUS_OK;
    }

private:
    VernonRuntimeContext *context_;
    std::weak_ptr<VernonPipelineTopology> topology_;
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

bool resolveProgramAutodiff(VernonLoadedPipeline &pipeline,
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
                const std::string leafPath = canonicalValueLeafPath(derivative.path, layoutLeaf);
                if (std::none_of(leaves.begin(), leaves.end(),
                                 [&](const ValueAbi &leaf) { return leaf.path == leafPath; })) {
                    invocationDiagnostic(*pipeline.context) =
                        "ProgramABI derivative projection does not resolve to its declared leaf ABI";
                    return false;
                }
                group.leafPaths.push_back(leafPath);
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
    pipeline.differentiated = VernonDifferentiatedPipeline{std::move(executable), std::move(groups)};
    return true;
}

} // namespace vernon::runtime::ad
