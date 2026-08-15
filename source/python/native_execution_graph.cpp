#include "native_execution_graph.h"

#include "VernonExecutionGraph.h"
#include "native_execution_graph_autodiff.h"
#include "runtime/autodiff/host_tape_allocator.h"

#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

struct PythonGraphResource {
    explicit PythonGraphResource(vernon::execution::GraphResource value) : resource(value) {}
    explicit PythonGraphResource(vernon::execution::GraphBuffer value) : resource(value), buffer(value) {}
    explicit PythonGraphResource(vernon::execution::GraphImage value) : resource(value), image(value), isImage(true) {}

    vernon::execution::GraphResource resource;
    vernon::execution::GraphBuffer buffer;
    vernon::execution::GraphImage image;
    bool isImage{};
};

struct PythonExecutionParameter {
    explicit PythonExecutionParameter(vernon::execution::ExecutionParameter value) : parameter(value) {}
    vernon::execution::ExecutionParameter parameter;
};

struct PythonExecutionBindingValue final : vernon::execution::ExecutionBindingValue {
    explicit PythonExecutionBindingValue(nb::object value) : value(std::move(value)) {}
    nb::object value;
};

struct PythonExecutionBindingsBuilder {
    explicit PythonExecutionBindingsBuilder(vernon::execution::ExecutionBindingsBuilder value)
        : builder(std::move(value)) {}

    void set(const PythonExecutionParameter &parameter, nb::object value) {
        builder.set(parameter.parameter, std::make_shared<PythonExecutionBindingValue>(std::move(value)));
    }

    vernon::execution::ExecutionBindingsBuilder builder;
};

struct PythonExecutionBindingsView {
    explicit PythonExecutionBindingsView(const vernon::execution::ExecutionResources &value) : resources(value) {}

    nb::object get(const PythonExecutionParameter &parameter) const {
        auto value =
            std::dynamic_pointer_cast<const PythonExecutionBindingValue>(resources.binding(parameter.parameter));
        if (!value)
            throw std::runtime_error("execution parameter binding has an incompatible native value type");
        return value->value;
    }

    uintptr_t token(const PythonExecutionParameter &parameter) const {
        return reinterpret_cast<uintptr_t>(resources.binding(parameter.parameter).get());
    }

    const vernon::execution::ExecutionResources &resources;
};

struct PythonExecutionGraph;

struct PythonRenderPass final : vernon::execution::RenderPass {
    PythonRenderPass(std::shared_ptr<PythonGraphCallbackState> callbackState, std::string name, PyObject *owner)
        : RenderPass(std::move(name)), callbackState(std::move(callbackState)), owner(owner) {}

    void declare() override;
    VernonRhiStatus execute(vernon::execution::GraphicsEncoder &encoder,
                            const vernon::execution::ExecutionResources &) override;

    void use(const PythonGraphResource &resource, vernon::execution::AccessMode access, VernonRhiResourceState state,
             uint32_t stageMask) {
        if (access == vernon::execution::AccessMode::Read) {
            if (resource.isImage)
                read(resource.image, state, stageMask);
            else
                read(resource.resource, state, stageMask);
        } else if (access == vernon::execution::AccessMode::Write) {
            if (resource.isImage)
                write(resource.image, state, stageMask);
            else
                write(resource.resource, state, stageMask);
        } else if (resource.isImage) {
            readWrite(resource.image, state, stageMask);
        } else {
            readWrite(resource.resource, state, stageMask);
        }
    }

    void addColor(uint32_t location, const PythonGraphResource &resource, VernonRhiLoadOperation load,
                  VernonRhiStoreOperation store, const std::array<float, 4> &clear) {
        if (!resource.isImage)
            throw std::invalid_argument("color attachment must be an image graph resource");
        vernon::execution::ColorAttachmentUse attachment{};
        attachment.image = resource.image;
        attachment.load = load;
        attachment.store = store;
        std::copy(clear.begin(), clear.end(), attachment.clear);
        color(location, attachment);
    }

    void setDepth(const PythonGraphResource &resource, VernonRhiLoadOperation depthLoad,
                  VernonRhiStoreOperation depthStore, float clearDepth, VernonRhiLoadOperation stencilLoad,
                  VernonRhiStoreOperation stencilStore, uint32_t clearStencil, bool readOnlyDepth,
                  bool readOnlyStencil) {
        if (!resource.isImage)
            throw std::invalid_argument("depth attachment must be an image graph resource");
        vernon::execution::DepthStencilAttachmentUse attachment{};
        attachment.image = resource.image;
        attachment.depthLoad = depthLoad;
        attachment.depthStore = depthStore;
        attachment.clearDepth = clearDepth;
        attachment.stencilLoad = stencilLoad;
        attachment.stencilStore = stencilStore;
        attachment.clearStencil = clearStencil;
        attachment.readOnlyDepth = readOnlyDepth;
        attachment.readOnlyStencil = readOnlyStencil;
        depth(attachment);
    }

    void setRenderArea(uint32_t x, uint32_t y, uint32_t width, uint32_t height) { renderArea(x, y, width, height); }

    std::shared_ptr<PythonGraphCallbackState> callbackState;
    PyObject *owner{};
};

struct PythonComputePass final : vernon::execution::ComputePass, vernon::execution::DifferentiablePass {
    PythonComputePass(std::shared_ptr<PythonGraphCallbackState> callbackState, std::string name, PyObject *owner)
        : ComputePass(std::move(name)), callbackState(std::move(callbackState)), owner(owner) {}

    void declare() override;
    VernonRhiStatus execute(vernon::execution::ComputeEncoder &encoder,
                            const vernon::execution::ExecutionResources &) override;
    vernon::execution::DifferentiablePass *differentiable() override {
        return gradientMappings_.empty() && cotangentMappings_.empty() ? nullptr : this;
    }
    const std::vector<vernon::execution::PassDerivativeMapping> &gradientMappings() const override {
        return gradientMappings_;
    }
    const std::vector<vernon::execution::PassDerivativeMapping> &cotangentMappings() const override {
        return cotangentMappings_;
    }
    const std::vector<vernon::execution::PassPrimalResourceMapping> &requiredPrimalResources() const override {
        return requiredPrimalResources_;
    }
    bool forward(vernon::execution::ComputeEncoder &encoder, const vernon::execution::ExecutionResources &resources,
                 std::unique_ptr<vernon::execution::PassPullback> &pullback, std::string &error) override;
    bool zeroCotangent(const std::string &path, const vernon::execution::ExecutionResources &resources,
                       std::shared_ptr<vernon::execution::GraphAutodiffValue> &value, std::string &error) override;
    bool implicitCotangent(const std::string &path, const vernon::execution::ExecutionResources &resources,
                           std::shared_ptr<vernon::execution::GraphAutodiffValue> &value, std::string &error) override;
    uint64_t estimatedResidualBytes() const override { return estimatedResidualBytes_; }
    uint64_t estimatedRetainedAllocationBytes() const override { return estimatedRetainedAllocationBytes_; }
    uint64_t estimatedForwardPeakBytes() const override { return estimatedForwardPeakBytes_; }
    uint64_t replayCost() const override { return replayCost_; }
    uint64_t resourceReloadCost() const override { return resourceReloadCost_; }
    uint64_t recomputationCost() const override { return recomputationCost_; }
    bool deterministicReductionLegal() const override { return deterministicReductionLegal_; }
    bool hasCheckpointPlanningMetadata() const override { return hasCheckpointPlanningMetadata_; }
    bool supportsReplay() const override { return true; }
    void setAutodiff(const nb::list &gradients, const nb::list &cotangents, const nb::list &requiredPrimals,
                     uint64_t invocationCount, uint64_t tapeStride, uint64_t replayCost, uint64_t resourceReloadCost,
                     uint64_t recomputationCost, uint64_t retainedPrimalBytes, bool deterministicReductionLegal,
                     bool hasCheckpointPlanningMetadata) {
        const auto convert = [](const nb::list &values) {
            std::vector<vernon::execution::PassDerivativeMapping> result;
            result.reserve(nb::len(values));
            for (nb::handle value : values) {
                nb::tuple entry = nb::cast<nb::tuple>(value);
                if (nb::len(entry) != 3)
                    throw std::invalid_argument("pass derivative mappings require path, endpoint kind, and id");
                result.push_back(
                    {nb::cast<std::string>(entry[0]),
                     {nb::cast<uint32_t>(entry[1]) == 0 ? vernon::execution::DerivativeEndpointKind::Resource
                                                        : vernon::execution::DerivativeEndpointKind::Parameter,
                      nb::cast<uint32_t>(entry[2])}});
            }
            return result;
        };
        gradientMappings_ = convert(gradients);
        cotangentMappings_ = convert(cotangents);
        requiredPrimalResources_.clear();
        requiredPrimalResources_.reserve(nb::len(requiredPrimals));
        for (nb::handle value : requiredPrimals) {
            nb::tuple entry = nb::cast<nb::tuple>(value);
            if (nb::len(entry) != 2)
                throw std::invalid_argument("required primal mappings require path and resource id");
            requiredPrimalResources_.push_back({nb::cast<std::string>(entry[0]), nb::cast<uint32_t>(entry[1])});
        }
        if (invocationCount > std::numeric_limits<size_t>::max() || tapeStride > std::numeric_limits<size_t>::max() ||
            (tapeStride && invocationCount > std::numeric_limits<uint64_t>::max() / tapeStride))
            throw std::overflow_error("autodiff checkpoint planning metadata exceeds the host representation");
        size_t forwardPeakBytes = 0;
        if (tapeStride && !vernon::runtime::ad::hostStaticTapeBatchPureStaticBytes(
                              static_cast<size_t>(invocationCount), static_cast<size_t>(tapeStride), forwardPeakBytes))
            throw std::overflow_error("autodiff checkpoint planning metadata exceeds the host representation");
        const uint64_t logicalTapeBytes = invocationCount * tapeStride;
        if (retainedPrimalBytes > std::numeric_limits<uint64_t>::max() - forwardPeakBytes)
            throw std::overflow_error("autodiff retained allocation estimate overflows");
        estimatedRetainedAllocationBytes_ = forwardPeakBytes + retainedPrimalBytes;
        estimatedForwardPeakBytes_ = forwardPeakBytes + retainedPrimalBytes;
        estimatedResidualBytes_ = logicalTapeBytes;
        replayCost_ = replayCost;
        resourceReloadCost_ = resourceReloadCost;
        recomputationCost_ = recomputationCost;
        deterministicReductionLegal_ = deterministicReductionLegal;
        hasCheckpointPlanningMetadata_ = hasCheckpointPlanningMetadata;
    }

    void use(const PythonGraphResource &resource, vernon::execution::AccessMode access, VernonRhiResourceState state,
             uint32_t stageMask) {
        if (access == vernon::execution::AccessMode::Read) {
            if (resource.isImage)
                read(resource.image, state, stageMask);
            else
                read(resource.resource, state, stageMask);
        } else if (access == vernon::execution::AccessMode::Write) {
            if (resource.isImage)
                write(resource.image, state, stageMask);
            else
                write(resource.resource, state, stageMask);
        } else if (resource.isImage) {
            readWrite(resource.image, state, stageMask);
        } else {
            readWrite(resource.resource, state, stageMask);
        }
    }

    std::shared_ptr<PythonGraphCallbackState> callbackState;
    PyObject *owner{};
    std::vector<vernon::execution::PassDerivativeMapping> gradientMappings_;
    std::vector<vernon::execution::PassDerivativeMapping> cotangentMappings_;
    std::vector<vernon::execution::PassPrimalResourceMapping> requiredPrimalResources_;
    uint64_t estimatedResidualBytes_{};
    uint64_t estimatedRetainedAllocationBytes_{};
    uint64_t estimatedForwardPeakBytes_{};
    uint64_t replayCost_{};
    uint64_t resourceReloadCost_{};
    uint64_t recomputationCost_{};
    bool deterministicReductionLegal_{true};
    bool hasCheckpointPlanningMetadata_{};
};

struct PythonCompiledBarrier {
    uint32_t sourceStageMask{};
    uint32_t destinationStageMask{};
    uint32_t sourceAccess{};
    uint32_t destinationAccess{};
    uint32_t oldState{};
    uint32_t newState{};
    bool isImage{};
    uint32_t baseMipLevel{};
    uint32_t mipLevelCount{};
    uint32_t baseArrayLayer{};
    uint32_t arrayLayerCount{};
    uint32_t aspects{};
};

struct PythonCompiledScope {
    bool rendering{};
    std::vector<uint32_t> passIndices;
    std::vector<PythonCompiledBarrier> barriers;
};

struct PythonExecutionSubmission {
    PythonExecutionSubmission(vernon::execution::ExecutionSubmission value,
                              std::shared_ptr<std::vector<nb::object>> retainedOwners)
        : submission(std::move(value)), owners(std::move(retainedOwners)) {}

    void wait() {
        const VernonRhiStatus status = submission.wait();
        if (status != VERNON_RHI_STATUS_OK)
            throw std::runtime_error("execution submission failed with provider status " +
                                     std::to_string(static_cast<uint32_t>(status)));
    }

    uint32_t state() const { return static_cast<uint32_t>(submission.state()); }
    vernon::execution::ExecutionSubmission submission;
    std::shared_ptr<std::vector<nb::object>> owners;
};

struct PythonCompiledExecutionGraph {
    PythonCompiledExecutionGraph(std::shared_ptr<vernon::execution::CompiledExecutionGraph> value,
                                 std::shared_ptr<PythonGraphCallbackState> state,
                                 std::shared_ptr<std::vector<nb::object>> retainedOwners)
        : plan(std::move(value)), callbackState(std::move(state)), owners(std::move(retainedOwners)) {}

    std::unique_ptr<PythonExecutionBindingsBuilder> createBindings(const nb::list &initial) {
        std::vector<vernon::execution::ExecutionBinding> bindings;
        bindings.reserve(nb::len(initial));
        for (nb::handle item : initial) {
            nb::tuple entry = nb::cast<nb::tuple>(item);
            if (nb::len(entry) != 2)
                throw std::invalid_argument("execution binding entries must contain a parameter and value");
            const auto &parameter = nb::cast<const PythonExecutionParameter &>(entry[0]);
            bindings.push_back(
                {parameter.parameter, std::make_shared<PythonExecutionBindingValue>(nb::borrow<nb::object>(entry[1]))});
        }
        return std::make_unique<PythonExecutionBindingsBuilder>(plan->createBindings(bindings));
    }

    std::unique_ptr<PythonExecutionSubmission> submit(PythonExecutionBindingsBuilder *bindings) {
        auto snapshot =
            bindings ? bindings->builder.snapshot() : std::shared_ptr<const vernon::execution::ExecutionBindings>{};
        callbackState->beginOperation(false);
        std::unique_ptr<PythonExecutionSubmission> submission;
        try {
            submission = std::make_unique<PythonExecutionSubmission>(plan->submit(std::move(snapshot)), owners);
        } catch (...) {
            (void)callbackState->endOperation();
            throw;
        }
        PythonGraphCallbackState::OperationFrame frame = callbackState->endOperation();
        if (frame.exception)
            std::rethrow_exception(std::move(frame.exception));
        return submission;
    }

    std::unique_ptr<PythonGraphPullback> vjp(PythonExecutionBindingsBuilder *bindings) {
        auto snapshot =
            bindings ? bindings->builder.snapshot() : std::shared_ptr<const vernon::execution::ExecutionBindings>{};
        std::string error;
        callbackState->beginOperation(false);
        std::shared_ptr<vernon::execution::GraphPullback> pullback;
        try {
            pullback = plan->vjp(std::move(snapshot), error);
        } catch (...) {
            (void)callbackState->endOperation();
            throw;
        }
        PythonGraphCallbackState::OperationFrame frame = callbackState->endOperation();
        if (!pullback) {
            if (frame.exception)
                std::rethrow_exception(std::move(frame.exception));
            throw std::runtime_error(error);
        }
        return std::make_unique<PythonGraphPullback>(std::move(pullback), callbackState, objectiveNames, owners);
    }

    std::vector<PythonCompiledScope> scopes() const {
        std::vector<PythonCompiledScope> result;
        result.reserve(plan->scopes().size());
        for (const auto &scope : plan->scopes()) {
            PythonCompiledScope compiled{scope.rendering, scope.passIndices, {}};
            compiled.barriers.reserve(scope.barriers.size());
            for (const VernonRhiBarrier &barrier : scope.barriers)
                compiled.barriers.push_back(
                    {barrier.source_stage_mask, barrier.destination_stage_mask, barrier.source_access,
                     barrier.destination_access, static_cast<uint32_t>(barrier.old_state),
                     static_cast<uint32_t>(barrier.new_state), barrier.is_image != 0,
                     barrier.image_subresources.base_mip_level, barrier.image_subresources.mip_level_count,
                     barrier.image_subresources.base_array_layer, barrier.image_subresources.array_layer_count,
                     barrier.image_subresources.aspects});
            result.push_back(std::move(compiled));
        }
        return result;
    }

    nb::object autodiffCheckpointPlan() const {
        const auto *checkpointPlan = plan->autodiffCheckpointPlan();
        if (!checkpointPlan)
            return nb::none();
        nb::dict result;
        result["persistent_checkpoint_bytes"] = checkpointPlan->persistentCheckpointBytes;
        result["initial_state_bytes"] = checkpointPlan->initialStateBytes;
        result["restoration_bytes"] = checkpointPlan->restorationBytes;
        result["transaction_bytes"] = checkpointPlan->transactionBytes;
        result["logical_residual_bytes"] = checkpointPlan->logicalResidualBytes;
        result["retained_allocation_bytes"] = checkpointPlan->retainedAllocationBytes;
        result["maximum_forward_peak_bytes"] = checkpointPlan->maximumForwardPeakBytes;
        result["backward_value_bytes"] = checkpointPlan->backwardValueBytes;
        result["memory_budget"] = checkpointPlan->memoryBudget;
        result["peak_bytes"] = checkpointPlan->peakBytes;
        result["replay_cost"] = checkpointPlan->replayCost;
        result["capture_store_bytes"] = checkpointPlan->captureStoreBytes;
        result["backward_load_bytes"] = checkpointPlan->backwardLoadBytes;
        result["checkpoint_copy_bytes"] = checkpointPlan->checkpointCopyBytes;
        result["resource_reload_cost"] = checkpointPlan->resourceReloadCost;
        result["recomputation_cost"] = checkpointPlan->recomputationCost;
        result["weighted_runtime_cost"] = checkpointPlan->weightedRuntimeCost;
        result["deterministic_reduction_legal"] = checkpointPlan->deterministicReductionLegal;
        result["selected_policy"] = checkpointPlan->selectedPolicy;
        nb::list resources;
        for (const auto &resource : checkpointPlan->checkpointResources) {
            nb::dict value;
            value["producer"] = resource.producer;
            value["resource"] = resource.version.resource;
            value["version"] = resource.version.epoch;
            value["offset"] = resource.offset;
            value["byte_size"] = resource.byteSize;
            value["alignment"] = resource.alignment;
            value["first_cut"] = resource.firstCut;
            value["last_cut"] = resource.lastCut;
            resources.append(std::move(value));
        }
        result["checkpoint_resources"] = std::move(resources);
        nb::list cuts;
        for (const auto &cut : checkpointPlan->cuts) {
            nb::dict value;
            value["schedule_offset"] = cut.scheduleOffset;
            value["checkpoint_resources"] = cut.checkpointResources;
            cuts.append(std::move(value));
        }
        result["cuts"] = std::move(cuts);
        nb::list segments;
        for (const auto &segment : checkpointPlan->replaySegments) {
            nb::dict value;
            value["begin_step"] = segment.beginStep;
            value["end_step"] = segment.endStep;
            value["checkpoint_index"] = segment.checkpointIndex;
            value["logical_residual_bytes"] = segment.logicalResidualBytes;
            value["retained_allocation_bytes"] = segment.retainedAllocationBytes;
            value["peak_bytes"] = segment.peakBytes;
            value["replay_cost"] = segment.replayCost;
            value["release_checkpoint_resources"] = segment.releaseCheckpointResources;
            segments.append(std::move(value));
        }
        result["replay_segments"] = std::move(segments);
        nb::list requiredVersions;
        for (const auto &required : checkpointPlan->requiredVersions) {
            nb::dict value;
            value["consumer"] = required.consumer;
            value["resource"] = required.version.resource;
            value["version"] = required.version.epoch;
            if (required.producer == UINT32_MAX)
                value["producer"] = nb::none();
            else
                value["producer"] = required.producer;
            switch (required.source) {
            case vernon::execution::AutodiffVersionSource::RetainedOwner:
                value["source"] = "retained_owner";
                break;
            case vernon::execution::AutodiffVersionSource::InitialState:
                value["source"] = "initial_state";
                break;
            case vernon::execution::AutodiffVersionSource::Checkpoint:
                value["source"] = "checkpoint";
                break;
            case vernon::execution::AutodiffVersionSource::Replay:
                value["source"] = "replay";
                break;
            }
            value["path"] = required.path;
            requiredVersions.append(std::move(value));
        }
        result["required_versions"] = std::move(requiredVersions);
        return std::move(result);
    }

    std::shared_ptr<vernon::execution::CompiledExecutionGraph> plan;
    std::shared_ptr<PythonGraphCallbackState> callbackState;
    std::shared_ptr<std::vector<nb::object>> owners;
    std::vector<std::string> objectiveNames;
};

struct PythonExecutionGraph {
    PythonExecutionGraph() = default;
    explicit PythonExecutionGraph(std::shared_ptr<RhiHostState> host)
        : host(std::move(host)), graph(this->host->device) {}

    PythonRenderPass *addRenderPass(const std::string &name, const nb::object &owner) {
        owners->push_back(owner);
        return &graph.emplacePass<PythonRenderPass>(callbackState, name, owner.ptr());
    }

    PythonComputePass *addComputePass(const std::string &name, const nb::object &owner) {
        owners->push_back(owner);
        return &graph.emplacePass<PythonComputePass>(callbackState, name, owner.ptr());
    }

    PythonGraphResource importBuffer(RhiBuffer &buffer, bool exported) {
        if (buffer.host != host)
            throw std::invalid_argument("buffer belongs to another execution graph device");
        return PythonGraphResource(graph.importBuffer(buffer.handle, exported));
    }

    PythonGraphResource importHostBuffer(uint64_t identity, const nb::object &checkpointBytes, bool exported) {
        std::shared_ptr<vernon::execution::GraphCheckpointResource> checkpoint;
        if (!checkpointBytes.is_none())
            checkpoint = makePythonCheckpointResource(nb::borrow<nb::object>(checkpointBytes));
        const vernon::execution::GraphBuffer resource =
            graph.importHostBuffer(identity, exported, std::move(checkpoint));
        if (resource.id == UINT32_MAX)
            throw std::invalid_argument("host execution graph resource identity must be non-zero");
        return PythonGraphResource(resource);
    }

    PythonGraphResource importImage(RhiImageView &view, bool exported) {
        if (view.host != host)
            throw std::invalid_argument("image belongs to another execution graph device");
        return PythonGraphResource(graph.importImage(view.image->handle, view.handle, exported));
    }

    PythonExecutionParameter parameter(const std::string &name) {
        return PythonExecutionParameter(graph.parameter(name));
    }

    void setAutodiffEndpoints(const nb::list &inputs, const nb::list &objectives) {
        const auto convert = [](const nb::list &values) {
            std::vector<vernon::execution::NamedDerivativeEndpoint> result;
            result.reserve(nb::len(values));
            for (nb::handle value : values) {
                nb::tuple entry = nb::cast<nb::tuple>(value);
                if (nb::len(entry) != 3)
                    throw std::invalid_argument("graph derivative endpoints require name, kind, and id");
                result.push_back(
                    {nb::cast<std::string>(entry[0]),
                     {nb::cast<uint32_t>(entry[1]) == 0 ? vernon::execution::DerivativeEndpointKind::Resource
                                                        : vernon::execution::DerivativeEndpointKind::Parameter,
                      nb::cast<uint32_t>(entry[2])}});
            }
            return result;
        };
        auto inputValues = convert(inputs);
        auto objectiveValues = convert(objectives);
        objectiveNames.clear();
        for (const auto &objective : objectiveValues)
            objectiveNames.push_back(objective.name);
        graph.setAutodiffEndpoints(std::move(inputValues), std::move(objectiveValues));
    }

    void planAutodiffCheckpoints(uint64_t memoryBudget) { graph.planAutodiffCheckpoints(memoryBudget); }

    std::unique_ptr<PythonCompiledExecutionGraph> compile() {
        std::string error;
        auto plan = graph.compile(error);
        if (!plan)
            throw std::invalid_argument(error);
        auto result = std::make_unique<PythonCompiledExecutionGraph>(std::move(plan), callbackState, owners);
        result->objectiveNames = objectiveNames;
        return result;
    }

    void validate() {
        std::string error;
        if (!graph.validate(error))
            throw std::invalid_argument(error);
    }

    std::shared_ptr<RhiHostState> host;
    std::shared_ptr<PythonGraphCallbackState> callbackState{std::make_shared<PythonGraphCallbackState>()};
    std::shared_ptr<std::vector<nb::object>> owners{std::make_shared<std::vector<nb::object>>()};
    std::vector<std::string> objectiveNames;
    vernon::execution::ExecutionGraph graph;
};

void PythonRenderPass::declare() { nb::borrow<nb::object>(owner).attr("_native_declare")(); }

VernonRhiStatus PythonRenderPass::execute(vernon::execution::GraphicsEncoder &encoder,
                                          const vernon::execution::ExecutionResources &resources) {
    try {
        if (resources.hasBindings()) {
            PythonExecutionBindingsView bindings(resources);
            nb::borrow<nb::object>(owner).attr("_native_execute")(nb::cast(encoder),
                                                                  nb::cast(&bindings, nb::rv_policy::reference));
        } else {
            nb::borrow<nb::object>(owner).attr("_native_execute")(nb::cast(encoder), nb::none());
        }
        return VERNON_RHI_STATUS_OK;
    } catch (...) {
        callbackState->captureException(std::current_exception());
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
}

void PythonComputePass::declare() { nb::borrow<nb::object>(owner).attr("_native_declare")(); }

VernonRhiStatus PythonComputePass::execute(vernon::execution::ComputeEncoder &encoder,
                                           const vernon::execution::ExecutionResources &resources) {
    try {
        if (resources.hasBindings()) {
            PythonExecutionBindingsView bindings(resources);
            nb::borrow<nb::object>(owner).attr("_native_execute")(nb::cast(encoder),
                                                                  nb::cast(&bindings, nb::rv_policy::reference));
        } else {
            nb::borrow<nb::object>(owner).attr("_native_execute")(nb::cast(encoder), nb::none());
        }
        return VERNON_RHI_STATUS_OK;
    } catch (...) {
        callbackState->captureException(std::current_exception());
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
}

bool PythonComputePass::forward(vernon::execution::ComputeEncoder &,
                                const vernon::execution::ExecutionResources &resources,
                                std::unique_ptr<vernon::execution::PassPullback> &pullback, std::string &error) {
    try {
        callbackState->recordReverseCallback();
        nb::object result;
        if (resources.hasBindings()) {
            PythonExecutionBindingsView bindings(resources);
            result = nb::borrow<nb::object>(owner).attr("_native_vjp_forward")(
                nb::cast(&bindings, nb::rv_policy::reference));
        } else {
            result = nb::borrow<nb::object>(owner).attr("_native_vjp_forward")(nb::none());
        }
        pullback = makePythonPassPullback(std::move(result), callbackState);
        return true;
    } catch (...) {
        callbackState->captureException(std::current_exception());
        error = "differentiable pass '" + name() + "' forward failed";
        return false;
    }
}

bool PythonComputePass::zeroCotangent(const std::string &path, const vernon::execution::ExecutionResources &resources,
                                      std::shared_ptr<vernon::execution::GraphAutodiffValue> &value,
                                      std::string &error) {
    try {
        callbackState->recordReverseCallback();
        nb::object result;
        if (resources.hasBindings()) {
            PythonExecutionBindingsView bindings(resources);
            result = nb::borrow<nb::object>(owner).attr("_native_graph_cotangent")(
                path, false, nb::cast(&bindings, nb::rv_policy::reference));
        } else {
            result = nb::borrow<nb::object>(owner).attr("_native_graph_cotangent")(path, false, nb::none());
        }
        value = makePythonGraphAutodiffValue(std::move(result), callbackState);
        return true;
    } catch (...) {
        callbackState->captureException(std::current_exception());
        error = "cannot create zero cotangent for differentiable pass '" + name() + "'";
        return false;
    }
}

bool PythonComputePass::implicitCotangent(const std::string &path,
                                          const vernon::execution::ExecutionResources &resources,
                                          std::shared_ptr<vernon::execution::GraphAutodiffValue> &value,
                                          std::string &error) {
    try {
        callbackState->recordReverseCallback();
        nb::object result;
        if (resources.hasBindings()) {
            PythonExecutionBindingsView bindings(resources);
            result = nb::borrow<nb::object>(owner).attr("_native_graph_cotangent")(
                path, true, nb::cast(&bindings, nb::rv_policy::reference));
        } else {
            result = nb::borrow<nb::object>(owner).attr("_native_graph_cotangent")(path, true, nb::none());
        }
        value = makePythonGraphAutodiffValue(std::move(result), callbackState);
        return true;
    } catch (...) {
        callbackState->captureException(std::current_exception());
        error = "cannot create implicit cotangent for differentiable pass '" + name() + "'";
        return false;
    }
}

} // namespace

nb::object createNativeExecutionGraph(std::shared_ptr<RhiHostState> host) {
    if (host)
        return nb::cast(std::make_unique<PythonExecutionGraph>(std::move(host)));
    return nb::cast(std::make_unique<PythonExecutionGraph>());
}

void bindNativeExecutionGraph(nb::module_ &module) {
    nb::class_<PythonGraphResource>(module, "_GraphResource")
        .def_prop_ro("id", [](const PythonGraphResource &value) { return value.resource.id; })
        .def_prop_ro("is_image", [](const PythonGraphResource &value) { return value.isImage; });
    nb::class_<vernon::execution::ExecutionPass>(module, "_ExecutionPass")
        .def("depends_on", &vernon::execution::ExecutionPass::dependsOn)
        .def("set_flags", &vernon::execution::ExecutionPass::setFlags)
        .def_prop_ro("flags", &vernon::execution::ExecutionPass::flags);
    nb::class_<PythonRenderPass, vernon::execution::ExecutionPass>(module, "_RenderPass")
        .def(
            "use",
            [](PythonRenderPass &pass, const PythonGraphResource &resource, uint32_t access, uint32_t state,
               uint32_t stageMask) {
                if (access > static_cast<uint32_t>(vernon::execution::AccessMode::ReadWrite) ||
                    state > static_cast<uint32_t>(VERNON_RHI_STATE_PRESENT) ||
                    (stageMask & ~(VERNON_RHI_STAGE_VERTEX | VERNON_RHI_STAGE_FRAGMENT)) != 0)
                    throw std::invalid_argument("invalid execution graph resource use");
                pass.use(resource, static_cast<vernon::execution::AccessMode>(access),
                         static_cast<VernonRhiResourceState>(state), stageMask);
            },
            nb::arg("resource"), nb::arg("access"), nb::arg("state"), nb::arg("stage_mask"))
        .def(
            "color",
            [](PythonRenderPass &pass, uint32_t location, const PythonGraphResource &resource, uint32_t load,
               uint32_t store, const std::array<float, 4> &clear) {
                if (load > static_cast<uint32_t>(VERNON_RHI_LOAD_DISCARD) ||
                    store > static_cast<uint32_t>(VERNON_RHI_STORE_DISCARD))
                    throw std::invalid_argument("invalid color attachment operation");
                pass.addColor(location, resource, static_cast<VernonRhiLoadOperation>(load),
                              static_cast<VernonRhiStoreOperation>(store), clear);
            },
            nb::arg("location"), nb::arg("resource"), nb::arg("load"), nb::arg("store"), nb::arg("clear"))
        .def(
            "depth",
            [](PythonRenderPass &pass, const PythonGraphResource &resource, uint32_t depthLoad, uint32_t depthStore,
               float clearDepth, uint32_t stencilLoad, uint32_t stencilStore, uint32_t clearStencil, bool readOnlyDepth,
               bool readOnlyStencil) {
                if (depthLoad > static_cast<uint32_t>(VERNON_RHI_LOAD_DISCARD) ||
                    stencilLoad > static_cast<uint32_t>(VERNON_RHI_LOAD_DISCARD) ||
                    depthStore > static_cast<uint32_t>(VERNON_RHI_STORE_DISCARD) ||
                    stencilStore > static_cast<uint32_t>(VERNON_RHI_STORE_DISCARD) || clearDepth < 0.0f ||
                    clearDepth > 1.0f)
                    throw std::invalid_argument("invalid depth attachment operation");
                pass.setDepth(resource, static_cast<VernonRhiLoadOperation>(depthLoad),
                              static_cast<VernonRhiStoreOperation>(depthStore), clearDepth,
                              static_cast<VernonRhiLoadOperation>(stencilLoad),
                              static_cast<VernonRhiStoreOperation>(stencilStore), clearStencil, readOnlyDepth,
                              readOnlyStencil);
            },
            nb::arg("resource"), nb::arg("depth_load"), nb::arg("depth_store"), nb::arg("clear_depth"),
            nb::arg("stencil_load"), nb::arg("stencil_store"), nb::arg("clear_stencil"), nb::arg("read_only_depth"),
            nb::arg("read_only_stencil"))
        .def("render_area", &PythonRenderPass::setRenderArea);
    nb::class_<PythonComputePass, vernon::execution::ExecutionPass>(module, "_ComputePass")
        .def(
            "use",
            [](PythonComputePass &pass, const PythonGraphResource &resource, uint32_t access, uint32_t state,
               uint32_t stageMask) {
                if (access > static_cast<uint32_t>(vernon::execution::AccessMode::ReadWrite) ||
                    state > static_cast<uint32_t>(VERNON_RHI_STATE_PRESENT) ||
                    (stageMask & ~VERNON_RHI_STAGE_COMPUTE) != 0)
                    throw std::invalid_argument("invalid execution graph resource use");
                pass.use(resource, static_cast<vernon::execution::AccessMode>(access),
                         static_cast<VernonRhiResourceState>(state), stageMask);
            },
            nb::arg("resource"), nb::arg("access"), nb::arg("state"), nb::arg("stage_mask"))
        .def("set_autodiff", &PythonComputePass::setAutodiff);
    nb::class_<PythonCompiledBarrier>(module, "_CompiledBarrier")
        .def_ro("source_stage_mask", &PythonCompiledBarrier::sourceStageMask)
        .def_ro("destination_stage_mask", &PythonCompiledBarrier::destinationStageMask)
        .def_ro("source_access", &PythonCompiledBarrier::sourceAccess)
        .def_ro("destination_access", &PythonCompiledBarrier::destinationAccess)
        .def_ro("old_state", &PythonCompiledBarrier::oldState)
        .def_ro("new_state", &PythonCompiledBarrier::newState)
        .def_ro("is_image", &PythonCompiledBarrier::isImage)
        .def_ro("base_mip_level", &PythonCompiledBarrier::baseMipLevel)
        .def_ro("mip_level_count", &PythonCompiledBarrier::mipLevelCount)
        .def_ro("base_array_layer", &PythonCompiledBarrier::baseArrayLayer)
        .def_ro("array_layer_count", &PythonCompiledBarrier::arrayLayerCount)
        .def_ro("aspects", &PythonCompiledBarrier::aspects);
    nb::class_<PythonCompiledScope>(module, "_CompiledScope")
        .def_ro("rendering", &PythonCompiledScope::rendering)
        .def_ro("pass_indices", &PythonCompiledScope::passIndices)
        .def_ro("barriers", &PythonCompiledScope::barriers);
    nb::class_<PythonExecutionParameter>(module, "_ExecutionParameter")
        .def_prop_ro("id", [](const PythonExecutionParameter &value) { return value.parameter.id; });
    nb::class_<PythonExecutionBindingsBuilder>(module, "_ExecutionBindings")
        .def("set", &PythonExecutionBindingsBuilder::set);
    nb::class_<PythonExecutionBindingsView>(module, "_ExecutionBindingsView")
        .def("get", &PythonExecutionBindingsView::get)
        .def("token", &PythonExecutionBindingsView::token);
    nb::class_<PythonExecutionGraph>(module, "_ExecutionGraph")
        .def("add_render_pass", &PythonExecutionGraph::addRenderPass, nb::rv_policy::reference)
        .def("add_compute_pass", &PythonExecutionGraph::addComputePass, nb::rv_policy::reference)
        .def("import_buffer", &PythonExecutionGraph::importBuffer, nb::arg("buffer"), nb::arg("exported") = false)
        .def("import_host_buffer", &PythonExecutionGraph::importHostBuffer, nb::arg("identity"),
             nb::arg("checkpoint_bytes") = nb::none(), nb::arg("exported") = false)
        .def("import_image", &PythonExecutionGraph::importImage, nb::arg("image"), nb::arg("exported") = false)
        .def("parameter", &PythonExecutionGraph::parameter)
        .def("set_autodiff_endpoints", &PythonExecutionGraph::setAutodiffEndpoints)
        .def("plan_autodiff_checkpoints", &PythonExecutionGraph::planAutodiffCheckpoints)
        .def("compile", &PythonExecutionGraph::compile)
        .def("validate", &PythonExecutionGraph::validate);
    nb::class_<PythonCompiledExecutionGraph>(module, "_CompiledExecutionGraph")
        .def("create_bindings", &PythonCompiledExecutionGraph::createBindings)
        .def("submit", &PythonCompiledExecutionGraph::submit, nb::arg("bindings") = nb::none())
        .def("vjp", &PythonCompiledExecutionGraph::vjp, nb::arg("bindings") = nb::none())
        .def_prop_ro(
            "schedule",
            [](const PythonCompiledExecutionGraph &value) -> const std::vector<uint32_t> & {
                return value.plan->schedule();
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro("scopes", &PythonCompiledExecutionGraph::scopes)
        .def_prop_ro("autodiff_checkpoint_plan", &PythonCompiledExecutionGraph::autodiffCheckpointPlan);
    bindExecutionGraphAutodiff(module);
    nb::class_<PythonExecutionSubmission>(module, "_ExecutionSubmission")
        .def("wait", &PythonExecutionSubmission::wait, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("state", &PythonExecutionSubmission::state);
    nb::class_<vernon::execution::GraphicsEncoder>(module, "_GraphicsEncoder");
    nb::class_<vernon::execution::ComputeEncoder>(module, "_ComputeEncoder");
}
