#include "runtime_autodiff_internal.h"

#include "execution_graph/execution_graph_checkpoint_planner_internal.h"
#include "host_tape_allocator.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_state.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vernon::runtime::ad {
namespace {

const ProgramGraph *findGraph(const ExecutableProgram &execution, const char *direction) {
    const auto found = std::find_if(execution.graphs.begin(), execution.graphs.end(),
                                    [&](const ProgramGraph &graph) { return graph.direction == direction; });
    return found == execution.graphs.end() ? nullptr : &*found;
}

const Parameter *findParameter(const Variant &variant, const std::string &name) {
    const auto external = std::find_if(variant.parameters.begin(), variant.parameters.end(),
                                       [&](const Parameter &parameter) { return parameter.name == name; });
    if (external != variant.parameters.end())
        return &*external;
    const auto internal = std::find_if(variant.internalParameters.begin(), variant.internalParameters.end(),
                                       [&](const Parameter &parameter) { return parameter.name == name; });
    return internal == variant.internalParameters.end() ? nullptr : &*internal;
}

std::optional<size_t> parameterByteSize(const Parameter &parameter, const ProgramValueSlot *slot = nullptr) {
    const ValueLayout &layout = slot && slot->valueLayout ? *slot->valueLayout
                                : parameter.valueLayout   ? *parameter.valueLayout
                                                          : parameter.elementLayout;
    if (!layout.byteSize)
        return std::nullopt;
    size_t result = layout.byteSize;
    const std::vector<uint64_t> &shape = slot ? slot->shape : parameter.shape;
    for (uint64_t extent : shape) {
        if (!extent || extent > std::numeric_limits<size_t>::max() / result)
            return std::nullopt;
        result *= static_cast<size_t>(extent);
    }
    return result;
}

VernonValueLayoutView layoutView(const ValueLayout &layout) {
    return {sizeof(VernonValueLayoutView),
            layout.byteSize,
            layout.alignment,
            {layout.layoutHash.data(), layout.layoutHash.size()},
            layout.abiLeaves.empty() ? nullptr : layout.abiLeaves.data(),
            layout.abiLeaves.size()};
}

void rebuildVariantLayouts(Variant &variant) {
    const auto rebuild = [](Parameter &parameter) {
        if (parameter.valueLayout)
            rebuildValueLayoutPathViews(*parameter.valueLayout);
        rebuildValueLayoutPathViews(parameter.elementLayout);
    };
    for (Parameter &parameter : variant.parameters)
        rebuild(parameter);
    for (Parameter &parameter : variant.internalParameters)
        rebuild(parameter);
}

struct HostProgramValue {
    std::vector<uint8_t> owned;
    std::vector<int64_t> strides;
    VernonPipelineArgument argument{};
};

struct ProgramResidualPlan {
    execution::AutodiffDagCheckpointPlan checkpoint;
    std::vector<uint32_t> retainedValues;
    uint32_t replayEnd{};
};

bool planProgramResiduals(const ExecutableProgram &execution, const Variant &variant, uint64_t memoryBudget,
                          ProgramResidualPlan &result, std::string &error) {
    const ProgramGraph *forward = findGraph(execution, "forward");
    if (!forward)
        return error = "Program autodiff topology has no forward graph", false;
    std::vector<std::optional<uint32_t>> producers(execution.values.size());
    std::vector<execution::detail::AutodiffDagNode> nodes(forward->nodes.size());
    for (const ProgramNode &node : forward->nodes) {
        execution::detail::AutodiffDagNode &planned = nodes[node.id];
        planned.predecessors = node.dependencies;
        planned.replayable = true;
        planned.replayCost = 1;
        planned.recomputationCost = 1;
        for (uint32_t value : node.results)
            producers[value] = node.id;
    }
    uint64_t initialStateBytes = 0;
    for (uint32_t value : forward->arguments) {
        const ProgramValueSlot &slot = execution.values[value];
        const Parameter *parameter = findParameter(variant, slot.name);
        const std::optional<size_t> bytes = parameter ? parameterByteSize(*parameter, &slot) : std::nullopt;
        if (!bytes || *bytes > std::numeric_limits<uint64_t>::max() - initialStateBytes)
            return error = "Program autodiff initial state size overflows", false;
        initialStateBytes += *bytes;
    }
    for (uint32_t value : execution.backwardCaptures()) {
        if (!producers[value])
            continue;
        const ProgramValueSlot &slot = execution.values[value];
        const Parameter *parameter = findParameter(variant, slot.name);
        const std::optional<size_t> bytes = parameter ? parameterByteSize(*parameter, &slot) : std::nullopt;
        execution::detail::AutodiffDagNode &node = nodes[*producers[value]];
        if (!bytes || *bytes > std::numeric_limits<uint64_t>::max() - node.residualBytes)
            return error = "Program autodiff residual size overflows", false;
        node.residualBytes += *bytes;
        node.retainedAllocationBytes += *bytes;
        node.forwardPeakBytes += *bytes;
    }
    if (!execution::detail::planDagAutodiffCheckpoints(nodes, memoryBudget, result.checkpoint, error, initialStateBytes,
                                                       true, 0, true, 0, true, 0,
                                                       execution::detail::AutodiffCheckpointPolicy::Balanced))
        return false;
    const uint32_t retainBegin =
        result.checkpoint.replaySegments.empty() ? 0 : result.checkpoint.replaySegments.back().beginStep;
    for (uint32_t value : execution.backwardCaptures()) {
        if (!producers[value] || *producers[value] >= retainBegin)
            result.retainedValues.push_back(value);
        else
            result.replayEnd = std::max(result.replayEnd, *producers[value] + 1);
    }
    if (result.replayEnd)
        for (uint32_t value : forward->arguments)
            if (std::find(result.retainedValues.begin(), result.retainedValues.end(), value) ==
                result.retainedValues.end())
                result.retainedValues.push_back(value);
    std::sort(result.retainedValues.begin(), result.retainedValues.end());
    return true;
}

bool materializeValues(const ExecutableProgram &execution, const Variant &variant,
                       std::vector<HostProgramValue> &storage, std::vector<VernonPipelineArgument> &arguments,
                       std::string &error) {
    storage.resize(execution.values.size());
    arguments.resize(execution.values.size());
    for (const ProgramValueSlot &slot : execution.values) {
        const Parameter *parameter = findParameter(variant, slot.name);
        const std::optional<size_t> bytes = parameter ? parameterByteSize(*parameter, &slot) : std::nullopt;
        if (!parameter || parameter->kind != "tensor" || !bytes) {
            error = "Program autodiff value has no materializable tensor parameter";
            return false;
        }
        const ValueLayout &layout = parameter->valueLayout ? *parameter->valueLayout : parameter->elementLayout;
        if (!layout.byteSize) {
            error = "Program autodiff tensor has an empty element layout";
            return false;
        }
        HostProgramValue &value = storage[slot.id];
        value.owned.resize(*bytes);
        value.strides.resize(slot.shape.size());
        size_t stride = layout.byteSize;
        for (size_t index = slot.shape.size(); index-- > 0;) {
            if (stride > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
                error = "Program autodiff tensor stride overflows";
                return false;
            }
            value.strides[index] = static_cast<int64_t>(stride);
            if (slot.shape[index] > std::numeric_limits<size_t>::max() / stride) {
                error = "Program autodiff tensor footprint overflows";
                return false;
            }
            stride *= static_cast<size_t>(slot.shape[index]);
        }
        value.argument.kind = VERNON_PIPELINE_TENSOR;
        value.argument.tensor.struct_size = sizeof(VernonTensorView);
        value.argument.tensor.storage = VERNON_TENSOR_HOST;
        value.argument.tensor.host_data = value.owned.data();
        value.argument.tensor.element_layout = layoutView(layout);
        value.argument.tensor.access = VERNON_ACCESS_READ_WRITE;
        value.argument.tensor.rank = static_cast<uint32_t>(slot.shape.size());
        value.argument.tensor.shape = slot.shape.empty() ? nullptr : slot.shape.data();
        value.argument.tensor.byte_strides = value.strides.empty() ? nullptr : value.strides.data();
        value.argument.tensor.byte_size = *bytes;
        arguments[slot.id] = value.argument;
    }
    return true;
}

struct ProgramLeafBinding {
    uint32_t value{UINT32_MAX};
    size_t byteOffset{};
    size_t elementStride{};
    size_t leafElementBytes{};
    size_t elementCount{};
};

bool transferLeaves(const VernonAdValueSet &supplied, const std::vector<ValueAbi> &signature,
                    const std::vector<ProgramLeafBinding> &bindings, std::vector<HostProgramValue> &storage,
                    bool publish, std::string &error) {
    if (bindings.size() != signature.size() || supplied.value_count != signature.size())
        return error = "Program autodiff leaf set does not match its canonical boundary", false;
    for (size_t index = 0; index < bindings.size(); ++index) {
        const ValueAbi &abi = signature[index];
        const VernonAdValue *value = findValue(supplied, abi.path);
        const ProgramLeafBinding &binding = bindings[index];
        if (!value || !valueMatches(*value, abi) || binding.value >= storage.size())
            return error = "Program autodiff leaf does not match graph reflection", false;
        auto *packed = storage[binding.value].owned.data();
        auto *leaf = static_cast<uint8_t *>(const_cast<void *>(value->data));
        for (size_t element = 0; element < binding.elementCount; ++element) {
            uint8_t *packedElement = packed + element * binding.elementStride + binding.byteOffset;
            uint8_t *leafElement = leaf + element * binding.leafElementBytes;
            if (publish)
                std::memcpy(leafElement, packedElement, binding.leafElementBytes);
            else
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
    ProgramPullback(VernonRuntimeContext &context, std::shared_ptr<VernonPipelineTopology> topology, Variant variant,
                    Signature signature, std::vector<ProgramLeafBinding> cotangentBindings,
                    std::vector<ProgramLeafBinding> gradientBindings, ProgramResidualPlan plan,
                    std::vector<std::vector<uint8_t>> residuals)
        : context_(&context), topology_(std::move(topology)), variant_(std::move(variant)),
          signature_(std::move(signature)), cotangentBindings_(std::move(cotangentBindings)),
          gradientBindings_(std::move(gradientBindings)), plan_(std::move(plan)), residuals_(std::move(residuals)) {
        rebuildVariantLayouts(variant_);
    }

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients,
                       const PullbackApplyOptions &options) override {
        const ProgramGraph *backward = findGraph(topology_->execution, "backward");
        if (!backward || (!backward->arguments.empty() && !cotangents))
            return fail(*context_, "Program pullback has no backward graph or required cotangents");
        std::vector<HostProgramValue> storage;
        std::vector<VernonPipelineArgument> values;
        std::string error;
        if (!materializeValues(topology_->execution, variant_, storage, values, error))
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
            if (value >= residuals_.size() || residuals_[value].size() != values[value].tensor.byte_size)
                return fail(*context_, "Program pullback residual state is incomplete");
            std::memcpy(storage[value].owned.data(), residuals_[value].data(), residuals_[value].size());
        }
        VernonLoadedPipeline proxy;
        proxy.context = context_;
        proxy.variant = variant_;
        proxy.topology = topology_;
        if (plan_.replayEnd) {
            const ProgramGraph *forward = findGraph(topology_->execution, "forward");
            ProgramGraph replay = *forward;
            replay.nodes.resize(plan_.replayEnd);
            const VernonStatus replayStatus = executePipelineProgramGraph(proxy, replay, values);
            if (replayStatus != VERNON_STATUS_OK)
                return replayStatus;
        }
        const VernonStatus status = executePipelineProgramGraph(proxy, *backward, values);
        if (status == VERNON_STATUS_OK &&
            transferLeaves(gradients, signature_.gradients, gradientBindings_, storage, true, error)) {
            ++usage_.submissions;
            ++usage_.waits;
            ++usage_.atomicPublications;
            usage_.temporaryAllocationBytes += temporaryBytes;
            return status;
        }
        return status == VERNON_STATUS_OK ? fail(*context_, error) : status;
    }

    PullbackMemoryUsage memoryUsage() const override {
        size_t bytes = 0;
        for (const std::vector<uint8_t> &residual : residuals_)
            bytes += residual.size();
        return {plan_.checkpoint.logicalResidualBytes, bytes, bytes, bytes, 0};
    }

    PullbackControlPlaneUsage controlPlaneUsage() const override { return usage_; }

private:
    VernonRuntimeContext *context_;
    std::shared_ptr<VernonPipelineTopology> topology_;
    Variant variant_;
    Signature signature_;
    std::vector<ProgramLeafBinding> cotangentBindings_;
    std::vector<ProgramLeafBinding> gradientBindings_;
    ProgramResidualPlan plan_;
    std::vector<std::vector<uint8_t>> residuals_;
    PullbackControlPlaneUsage usage_;
};

class ProgramExecutable final : public Executable {
public:
    ProgramExecutable(VernonRuntimeContext &context, std::weak_ptr<VernonPipelineTopology> topology, Variant variant)
        : context_(&context), topology_(std::move(topology)), variant_(std::move(variant)) {
        rebuildVariantLayouts(variant_);
        const std::shared_ptr<VernonPipelineTopology> locked = topology_.lock();
        const auto append = [&](const std::vector<ProgramAdSignatureBinding> &bindings, std::vector<ValueAbi> &values,
                                std::vector<ProgramLeafBinding> &leafBindings) {
            for (const ProgramAdSignatureBinding &binding : bindings) {
                const ProgramValueSlot &slot = locked->execution.values[binding.value];
                const Parameter *parameter = findParameter(variant_, slot.name);
                std::string error;
                if (parameter) {
                    Parameter linked = *parameter;
                    linked.shape = slot.shape;
                    if (slot.valueLayout)
                        linked.valueLayout = *slot.valueLayout;
                    const ValueLayout &layout = linked.valueLayout ? *linked.valueLayout : linked.elementLayout;
                    const size_t begin = values.size();
                    if (!appendParameterValueAbi(linked, binding.path, values, error)) {
                        signatureError_ = std::move(error);
                        continue;
                    }
                    size_t elementCount = 1;
                    for (uint64_t extent : linked.shape)
                        elementCount *= static_cast<size_t>(extent);
                    if (values.size() - begin != layout.leaves.size()) {
                        signatureError_ = "Program aggregate ABI contains duplicate leaf paths";
                        continue;
                    }
                    for (const ValueLeaf &leaf : layout.leaves) {
                        const std::optional<VernonDataType> dtype = pipelineDataType(leaf.dtype);
                        leafBindings.push_back({binding.value, leaf.byteOffset, layout.byteSize,
                                                dtype ? dtypeSize(*dtype) * static_cast<size_t>(leaf.scalarCount) : 0,
                                                elementCount});
                    }
                } else {
                    signatureError_ = "Program signature value has no linked canonical parameter";
                }
            }
        };
        append(locked->execution.adSignature.inputs, signature_.inputs, inputBindings_);
        append(locked->execution.adSignature.outputs, signature_.outputs, outputBindings_);
        append(locked->execution.adSignature.cotangents, signature_.cotangents, cotangentBindings_);
        append(locked->execution.adSignature.gradients, signature_.gradients, gradientBindings_);
        const auto sortBoundary = [](std::vector<ValueAbi> &values, std::vector<ProgramLeafBinding> &bindings) {
            std::vector<size_t> order(values.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](size_t left, size_t right) { return values[left].path < values[right].path; });
            std::vector<ValueAbi> sortedValues;
            std::vector<ProgramLeafBinding> sortedBindings;
            sortedValues.reserve(order.size());
            sortedBindings.reserve(order.size());
            for (size_t index : order) {
                sortedValues.push_back(std::move(values[index]));
                sortedBindings.push_back(bindings[index]);
            }
            values = std::move(sortedValues);
            bindings = std::move(sortedBindings);
        };
        sortBoundary(signature_.inputs, inputBindings_);
        sortBoundary(signature_.outputs, outputBindings_);
        sortBoundary(signature_.cotangents, cotangentBindings_);
        sortBoundary(signature_.gradients, gradientBindings_);
    }

    const Signature &signature() const override { return signature_; }

    VernonStatus forward(const ForwardExecutionTarget &target, VernonLaunchSize, const VernonAdValueSet &inputs,
                         VernonAdValueSet *outputs, std::unique_ptr<PullbackExecution> &pullback) override {
        const std::shared_ptr<VernonPipelineTopology> topology = topology_.lock();
        const ProgramGraph *forward = topology ? findGraph(topology->execution, "forward") : nullptr;
        if (!signatureError_.empty())
            return fail(*context_, signatureError_);
        if (!topology || !forward || !outputs || target.encodedInvocation())
            return fail(*context_, "Program autodiff forward requires a live pipeline and host API values");
        std::vector<HostProgramValue> storage;
        std::vector<VernonPipelineArgument> values;
        std::string error;
        if (!materializeValues(topology->execution, variant_, storage, values, error) ||
            !transferLeaves(inputs, signature_.inputs, inputBindings_, storage, false, error))
            return fail(*context_, error);
        VernonLoadedPipeline proxy;
        proxy.context = context_;
        proxy.variant = variant_;
        proxy.topology = topology;
        const VernonStatus status = executePipelineProgramGraph(proxy, *forward, values);
        if (status != VERNON_STATUS_OK)
            return status;
        if (!transferLeaves(*outputs, signature_.outputs, outputBindings_, storage, true, error))
            return fail(*context_, error);
        ProgramResidualPlan plan;
        if (!planProgramResiduals(topology->execution, variant_, context_->autodiffMemoryPolicy->invocationLimit(),
                                  plan, error))
            return fail(*context_, error);
        std::vector<std::vector<uint8_t>> residuals(topology->execution.values.size());
        for (uint32_t value : plan.retainedValues) {
            residuals[value] = storage[value].owned;
        }
        pullback = std::make_unique<ProgramPullback>(*context_, topology, variant_, signature_, cotangentBindings_,
                                                     gradientBindings_, std::move(plan), std::move(residuals));
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
    std::string signatureError_;
};

} // namespace

bool resolveProgramAutodiff(VernonLoadedPipeline &pipeline,
                            const std::vector<AutodiffDerivativeGroup> &derivativeGroups) {
    if (!pipeline.context || !pipeline.topology)
        return false;
    if (pipeline.context->backend != VERNON_RUNTIME_CPU) {
        invocationDiagnostic(*pipeline.context) = "Program autodiff currently requires the CPU backend";
        return false;
    }
    if (!findGraph(pipeline.topology->execution, "forward") || !findGraph(pipeline.topology->execution, "backward")) {
        invocationDiagnostic(*pipeline.context) = "Program autodiff topology requires forward and backward graphs";
        return false;
    }
    const auto physicallyEquivalent = [](const ValueLayout &left, const ValueLayout &right) {
        if (left.byteSize != right.byteSize || left.alignment != right.alignment ||
            left.leaves.size() != right.leaves.size())
            return false;
        for (size_t index = 0; index < left.leaves.size(); ++index) {
            const ValueLeaf &a = left.leaves[index];
            const ValueLeaf &b = right.leaves[index];
            if (a.dtype != b.dtype || a.scalarCount != b.scalarCount || a.byteOffset != b.byteOffset ||
                a.shape != b.shape)
                return false;
        }
        return true;
    };
    for (const ProgramValueSlot &value : pipeline.topology->execution.values) {
        const Parameter *parameter = findParameter(pipeline.variant, value.name);
        const ValueLayout *parameterLayout =
            parameter ? (parameter->valueLayout ? &*parameter->valueLayout : &parameter->elementLayout) : nullptr;
        if (!parameter || !parameterByteSize(*parameter, &value) ||
            (value.valueLayout && !value.valueLayout->layoutHash.empty() && parameterLayout &&
             value.valueLayout->layoutHash != parameterLayout->layoutHash &&
             !physicallyEquivalent(*value.valueLayout, *parameterLayout))) {
            invocationDiagnostic(*pipeline.context) = "Program autodiff topology has an unsupported value ABI";
            return false;
        }
    }
    if (!pipeline.context->autodiffMemoryPolicy)
        pipeline.context->autodiffMemoryPolicy = std::make_shared<AutodiffMemoryPolicy>();
    auto executable = std::make_shared<ProgramExecutable>(*pipeline.context, pipeline.topology, pipeline.variant);
    std::vector<AutodiffDerivativeGroup> groups = derivativeGroups;
    const auto appendGroups = [](AutodiffDerivativeRole role, const std::vector<ProgramAdSignatureBinding> &declared,
                                 const std::vector<ValueAbi> &leaves, std::vector<AutodiffDerivativeGroup> &result) {
        for (const ProgramAdSignatureBinding &binding : declared) {
            AutodiffDerivativeGroup group{role, binding.path, {}};
            for (const ValueAbi &leaf : leaves)
                if (leaf.path == binding.path || (leaf.path.size() > binding.path.size() &&
                                                  leaf.path.compare(0, binding.path.size(), binding.path) == 0 &&
                                                  leaf.path[binding.path.size()] == '.'))
                    group.leafPaths.push_back(leaf.path);
            std::sort(group.leafPaths.begin(), group.leafPaths.end());
            result.push_back(std::move(group));
        }
    };
    if (groups.empty()) {
        appendGroups(AutodiffDerivativeRole::Gradient, pipeline.topology->execution.adSignature.gradients,
                     executable->signature().gradients, groups);
        std::sort(groups.begin(), groups.end(),
                  [](const AutodiffDerivativeGroup &left, const AutodiffDerivativeGroup &right) {
                      return left.declaredPath < right.declaredPath;
                  });
        const size_t gradientCount = groups.size();
        appendGroups(AutodiffDerivativeRole::Cotangent, pipeline.topology->execution.adSignature.cotangents,
                     executable->signature().cotangents, groups);
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
    pipeline.topology->differentiated = VernonDifferentiatedPipeline{std::move(executable), std::move(groups)};
    return true;
}

} // namespace vernon::runtime::ad
