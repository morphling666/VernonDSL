#include "VernonAutodiffGraph.h"

#include "VernonExecutionGraph.h"
#include "runtime_autodiff_internal.h"
#include "runtime_state.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vernon::runtime {
namespace {

using ad::createGraphBuffer;
using ad::GpuBufferBinding;
using ad::GpuGraphExecutable;
using ad::GpuResourceRole;
using ad::ResourceAbi;
using ad::ValueAbi;
using execution::ExecutionGraph;
using execution::GraphBuffer;

constexpr const char *kLaunchResourcePath = "__vernon_launch";

VernonStatus fail(VernonRuntimeContext *context, std::string message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    if (context)
        invocationDiagnostic(*context) = std::move(message);
    return status;
}

const ResourceAbi *resource(const std::vector<ResourceAbi> &resources, const std::string &path) {
    const auto found = std::find_if(resources.begin(), resources.end(),
                                    [&](const ResourceAbi &value) { return value.value.path == path; });
    return found == resources.end() ? nullptr : &*found;
}

struct PassBinding {
    ResourceAbi spec;
    GraphBuffer buffer;
};

class AutodiffComputePass final : public execution::ComputePass {
public:
    AutodiffComputePass(std::string name, std::shared_ptr<GpuGraphExecutable> executable,
                        std::vector<PassBinding> bindings, VernonLaunchSize computeGrid, bool backward)
        : ComputePass(std::move(name)), executable_(std::move(executable)), bindings_(std::move(bindings)),
          computeGrid_(computeGrid), backward_(backward) {}

    void declare() override {
        for (const PassBinding &binding : bindings_) {
            if (binding.spec.access == VERNON_ACCESS_READ)
                read(binding.buffer, VERNON_RHI_STATE_SHADER_READ, VERNON_RHI_STAGE_COMPUTE);
            else if (binding.spec.access == VERNON_ACCESS_WRITE)
                write(binding.buffer, VERNON_RHI_STATE_SHADER_WRITE, VERNON_RHI_STAGE_COMPUTE);
            else
                readWrite(binding.buffer, VERNON_RHI_STATE_SHADER_WRITE, VERNON_RHI_STAGE_COMPUTE);
        }
    }

    VernonRhiStatus execute(execution::ComputeEncoder &encoder,
                            const execution::ExecutionResources &resources) override {
        VernonRuntimeProviderObject providerEncoder{};
        if (vernonRuntimeReferenceRhiCommandEncoder(&executable_->context(), encoder.native(), &providerEncoder) !=
            VERNON_STATUS_OK)
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        std::vector<GpuBufferBinding> bindings;
        bindings.reserve(bindings_.size());
        for (const PassBinding &binding : bindings_) {
            VernonRhiBuffer buffer{};
            if (!resources.buffer(binding.buffer, buffer))
                return VERNON_RHI_STATUS_INVALID_ARGUMENT;
            bindings.push_back({buffer, binding.spec.byteSize});
        }
        const VernonStatus status = backward_ ? executable_->encodeBackward(providerEncoder, computeGrid_, bindings)
                                              : executable_->encodeForward(providerEncoder, computeGrid_, bindings);
        return status == VERNON_STATUS_OK ? VERNON_RHI_STATUS_OK : VERNON_RHI_STATUS_INTERNAL_ERROR;
    }

private:
    std::shared_ptr<GpuGraphExecutable> executable_;
    std::vector<PassBinding> bindings_;
    VernonLaunchSize computeGrid_{};
    bool backward_{};
};

bool materializeResource(const ResourceAbi &descriptor, VernonLaunchSize computeGrid, ResourceAbi &resource) {
    size_t byteSize = 0;
    if (!ad::resourceByteSize(descriptor, computeGrid, byteSize))
        return false;
    resource = descriptor;
    resource.byteSize = byteSize;
    resource.value.byteSize = byteSize;
    if (resource.runtimeCarrier)
        resource.physicalShape = {computeGrid.z, computeGrid.y, computeGrid.x};
    return true;
}

std::atomic<uint64_t> nextAutodiffGraphIdentity{1};

} // namespace

class AutodiffGraphPullback::Impl {
public:
    struct InputSource {
        std::optional<uint32_t> sourceNode;
        size_t external{std::numeric_limits<size_t>::max()};
    };

    struct Node {
        std::string name;
        std::shared_ptr<GpuGraphExecutable> executable;
        std::unordered_map<std::string, InputSource> inputs;
        std::unordered_map<std::string, GraphBuffer> forwardBuffers;
    };

    struct External {
        std::string gradientPath;
        ValueAbi value;
    };

    Impl(std::shared_ptr<ContextLease> contextLease, std::unique_ptr<ExecutionGraph> forwardGraph,
         std::vector<Node> nodes, std::vector<uint32_t> order, std::vector<External> external,
         VernonLaunchSize computeGrid, uint32_t outputNode)
        : contextLease(std::move(contextLease)), context(&this->contextLease->get()),
          forwardGraph(std::move(forwardGraph)), nodes(std::move(nodes)), order(std::move(order)),
          external(std::move(external)), computeGrid(computeGrid), outputNode(outputNode),
          forwardStats(this->forwardGraph->lastStats()) {}

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients) {
        RuntimeDiagnosticScope diagnostic(context);
        std::lock_guard<std::mutex> lock(mutex);
        execution::DeviceExecutionSession session(nodes[outputNode].executable->device());
        if (!ad::validSet(&gradients, true))
            return fail(context, "invalid graph pullback gradient set");
        std::unordered_map<std::string, ValueAbi> externalGradientAbis;
        for (const External &binding : external) {
            if (binding.gradientPath.empty())
                continue;
            ValueAbi abi = binding.value;
            abi.path = binding.gradientPath;
            const auto [found, inserted] = externalGradientAbis.emplace(binding.gradientPath, abi);
            if (!inserted && !ad::sameValueAbi(found->second, abi))
                return fail(context, "external graph gradients have incompatible ABIs");
        }
        if (gradients.value_count != externalGradientAbis.size())
            return fail(context, "graph pullback gradient set does not match external differentiable inputs");
        for (const auto &[path, abi] : externalGradientAbis) {
            VernonAdValue *value = ad::findValue(gradients, path);
            if (!value || !ad::valueMatches(*value, abi))
                return fail(context, "graph pullback gradient output has an invalid ABI");
        }

        ValueAbi sinkCotangent = nodes[outputNode].executable->signature().cotangent;
        if (!ad::materializeCarrierValue(sinkCotangent, computeGrid))
            return fail(context, "graph pullback launch size overflows", VERNON_STATUS_INVALID_ARGUMENT);
        std::vector<uint8_t> cotangentBytes;
        if (!ad::makeCotangentBytes(cotangents, sinkCotangent, cotangentBytes, invocationDiagnostic(*context)))
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto backwardGraph = std::make_unique<ExecutionGraph>(nodes[outputNode].executable->device());
        std::unordered_map<uint32_t, GraphBuffer> nodeCotangents;
        GraphBuffer outputCotangent;
        if (createGraphBuffer(*backwardGraph, sinkCotangent, outputCotangent) != VERNON_RHI_STATUS_OK)
            return fail(context, "failed to allocate the graph output cotangent", VERNON_STATUS_INTERNAL_ERROR);
        if (!ad::uploadGpuBuffer(*context, outputCotangent.handle, 0, cotangentBytes.data(), cotangentBytes.size(),
                                 "the graph output cotangent"))
            return VERNON_STATUS_INTERNAL_ERROR;
        nodeCotangents.emplace(outputNode, outputCotangent);
        std::unordered_map<std::string, GraphBuffer> externalGradientBuffers;

        for (auto current = order.rbegin(); current != order.rend(); ++current) {
            const uint32_t nodeIndex = *current;
            auto seed = nodeCotangents.find(nodeIndex);
            if (seed == nodeCotangents.end())
                continue;
            Node &node = nodes[nodeIndex];
            std::vector<PassBinding> passBindings;
            passBindings.reserve(node.executable->backwardResources().size());
            for (const ResourceAbi &descriptor : node.executable->backwardResources()) {
                ResourceAbi spec;
                if (!materializeResource(descriptor, computeGrid, spec))
                    return fail(context, "graph pullback resource size overflows", VERNON_STATUS_INVALID_ARGUMENT);
                GraphBuffer buffer;
                if (spec.value.path == kLaunchResourcePath) {
                    const auto launch = node.forwardBuffers.find(spec.value.path);
                    if (launch == node.forwardBuffers.end())
                        return fail(context, "graph pullback has no bound launch resource");
                    buffer = backwardGraph->importBuffer(launch->second.handle);
                } else if (spec.role == GpuResourceRole::Tape) {
                    const auto tape = node.forwardBuffers.find(spec.value.path);
                    if (tape == node.forwardBuffers.end())
                        return fail(context, "graph pullback has no matching tape resource");
                    buffer = backwardGraph->importBuffer(tape->second.handle);
                } else if (spec.role == GpuResourceRole::Cotangent) {
                    buffer = seed->second;
                } else if (spec.role == GpuResourceRole::Gradient) {
                    const auto source = node.inputs.find(spec.value.path);
                    if (source == node.inputs.end())
                        return fail(context, "graph pullback gradient has no primal input source");
                    bool created = false;
                    if (source->second.sourceNode) {
                        const uint32_t upstream = *source->second.sourceNode;
                        const ValueAbi &upstreamCotangent = nodes[upstream].executable->signature().cotangent;
                        if (!ad::sameValueAbi(descriptor.value, upstreamCotangent))
                            return fail(context, "connected graph gradient and cotangent ABIs do not match");
                        auto [entry, inserted] = nodeCotangents.emplace(upstream, GraphBuffer{});
                        if (inserted) {
                            ValueAbi carrier = upstreamCotangent;
                            if (!ad::materializeCarrierValue(carrier, computeGrid))
                                return fail(context, "upstream graph cotangent size overflows");
                            if (createGraphBuffer(*backwardGraph, carrier, entry->second) != VERNON_RHI_STATUS_OK)
                                return fail(context, "failed to allocate an upstream graph cotangent",
                                            VERNON_STATUS_INTERNAL_ERROR);
                            created = true;
                        }
                        buffer = entry->second;
                    } else if (source->second.external != std::numeric_limits<size_t>::max()) {
                        const External &binding = external[source->second.external];
                        if (binding.gradientPath.empty()) {
                            if (createGraphBuffer(*backwardGraph, spec.value, buffer) != VERNON_RHI_STATUS_OK)
                                return fail(context, "failed to allocate a discarded graph gradient",
                                            VERNON_STATUS_INTERNAL_ERROR);
                            created = true;
                        } else {
                            auto [entry, inserted] =
                                externalGradientBuffers.emplace(binding.gradientPath, GraphBuffer{});
                            if (inserted) {
                                if (createGraphBuffer(*backwardGraph, spec.value, entry->second, true) !=
                                    VERNON_RHI_STATUS_OK)
                                    return fail(context, "failed to allocate an external graph gradient",
                                                VERNON_STATUS_INTERNAL_ERROR);
                                created = true;
                            }
                            buffer = entry->second;
                        }
                    } else {
                        return fail(context, "graph pullback gradient source is invalid");
                    }
                    if (created && !ad::clearGpuBuffer(*context, buffer.handle, 0, spec.byteSize,
                                                       "a graph pullback accumulation resource"))
                        return VERNON_STATUS_INTERNAL_ERROR;
                } else {
                    return fail(context, "graph pullback contains an invalid backward resource role");
                }
                passBindings.push_back({spec, buffer});
            }
            backwardGraph->emplacePass<AutodiffComputePass>(node.name + ".backward", node.executable,
                                                            std::move(passBindings), computeGrid, true);
        }

        if (backwardGraph->execute() != VERNON_RHI_STATUS_OK)
            return fail(context, "failed to execute a graph pullback", VERNON_STATUS_INTERNAL_ERROR);
        backwardStats = backwardGraph->lastStats();
        for (const auto &[path, abi] : externalGradientAbis) {
            const auto buffer = externalGradientBuffers.find(path);
            if (buffer == externalGradientBuffers.end())
                return fail(context, "graph pullback has no matching external gradient resource");
            VernonAdValue *value = ad::findValue(gradients, path);
            if (!ad::downloadGpuBuffer(*context, buffer->second.handle, 0, value->data, abi.byteSize,
                                       "an external graph gradient"))
                return VERNON_STATUS_INTERNAL_ERROR;
        }
        return VERNON_STATUS_OK;
    }

    VernonRhiCommandEncoderStats getForwardStats() const {
        std::lock_guard<std::mutex> lock(mutex);
        return forwardStats;
    }

    VernonRhiCommandEncoderStats getLastBackwardStats() const {
        std::lock_guard<std::mutex> lock(mutex);
        return backwardStats;
    }

    std::shared_ptr<ContextLease> contextLease;
    VernonRuntimeContext *context{};
    std::unique_ptr<ExecutionGraph> forwardGraph;
    std::vector<Node> nodes;
    std::vector<uint32_t> order;
    std::vector<External> external;
    VernonLaunchSize computeGrid{};
    uint32_t outputNode{};
    VernonRhiCommandEncoderStats forwardStats{};
    VernonRhiCommandEncoderStats backwardStats{};
    mutable std::mutex mutex;
};

class CompiledAutodiffGraph::Impl {
public:
    using InputSource = AutodiffGraphPullback::Impl::InputSource;

    struct Node {
        std::string name;
        std::shared_ptr<GpuGraphExecutable> executable;
        std::unordered_map<std::string, InputSource> inputs;
    };

    struct External {
        uint32_t node{};
        std::string inputPath;
        std::string valuePath;
        std::string gradientPath;
        ValueAbi value;
        bool writable{};
    };

    Impl(std::shared_ptr<ContextLease> contextLease, std::vector<Node> nodes, std::vector<uint32_t> order,
         std::vector<External> external, uint32_t outputNode)
        : contextLease(std::move(contextLease)), context(&this->contextLease->get()), nodes(std::move(nodes)),
          order(std::move(order)), external(std::move(external)), outputNode(outputNode) {}

    VernonStatus forward(VernonLaunchSize computeGrid, VernonAdValueSet &inputs, VernonAdValueSet &outputs,
                         std::unique_ptr<AutodiffGraphPullback> &pullback) const {
        RuntimeDiagnosticScope diagnostic(context);
        if (!ad::validLaunchSize(computeGrid))
            return fail(context, "compiled autodiff graph requires a positive compute grid");
        execution::DeviceExecutionSession session(nodes[outputNode].executable->device());
        pullback.reset();
        if (!ad::validSet(&inputs, true) || !ad::validSet(&outputs, true) || inputs.value_count != external.size())
            return fail(context, "invalid compiled autodiff graph forward invocation");

        struct BoundExternal {
            const External *descriptor{};
            VernonAdValue *value{};
            GraphBuffer buffer;
            uintptr_t hostBegin{};
            uintptr_t hostEnd{};
        };
        std::vector<BoundExternal> bound;
        bound.reserve(external.size());
        for (const External &descriptor : external) {
            VernonAdValue *value = ad::findValue(inputs, descriptor.valuePath);
            if (!value || !ad::valueMatches(*value, descriptor.value))
                return fail(context, "compiled autodiff graph input does not match its external port");
            const uintptr_t hostBegin = reinterpret_cast<uintptr_t>(value->data);
            if (!hostBegin || value->size > UINTPTR_MAX - hostBegin)
                return fail(context, "compiled autodiff graph input address range is invalid");
            const uintptr_t hostEnd = hostBegin + value->size;
            for (const BoundExternal &existing : bound)
                if ((descriptor.writable || existing.descriptor->writable) && hostBegin < existing.hostEnd &&
                    existing.hostBegin < hostEnd)
                    return fail(context, "writable compiled autodiff graph Storage inputs overlap");
            bound.push_back({&descriptor, value, {}, hostBegin, hostEnd});
        }

        auto graph = std::make_unique<ExecutionGraph>(context->rhiDevice);
        for (BoundExternal &binding : bound) {
            if (createGraphBuffer(*graph, binding.descriptor->value, binding.buffer, binding.descriptor->writable) !=
                VERNON_RHI_STATUS_OK)
                return fail(context, "failed to allocate a compiled autodiff graph input",
                            VERNON_STATUS_INTERNAL_ERROR);
            if (!ad::uploadGpuBuffer(*context, binding.buffer.handle, 0, binding.value->data,
                                     binding.descriptor->value.byteSize, "a compiled autodiff graph input"))
                return VERNON_STATUS_INTERNAL_ERROR;
        }

        std::vector<AutodiffGraphPullback::Impl::Node> invocationNodes;
        invocationNodes.reserve(nodes.size());
        for (const Node &node : nodes)
            invocationNodes.push_back({node.name, node.executable, node.inputs, {}});
        for (uint32_t nodeIndex : order) {
            AutodiffGraphPullback::Impl::Node &node = invocationNodes[nodeIndex];
            std::vector<PassBinding> passBindings;
            passBindings.reserve(node.executable->forwardResources().size());
            for (const ResourceAbi &descriptor : node.executable->forwardResources()) {
                ResourceAbi spec;
                if (!materializeResource(descriptor, computeGrid, spec))
                    return fail(context, "compiled autodiff graph resource size overflows");
                GraphBuffer buffer;
                if (spec.value.path == kLaunchResourcePath) {
                    if (createGraphBuffer(*graph, spec.value, buffer, true) != VERNON_RHI_STATUS_OK)
                        return fail(context, "failed to allocate a compiled autodiff launch resource",
                                    VERNON_STATUS_INTERNAL_ERROR);
                    const uint32_t dimensions[] = {computeGrid.x, computeGrid.y, computeGrid.z};
                    if (!ad::uploadGpuBuffer(*context, buffer.handle, 0, dimensions, sizeof(dimensions),
                                             "the compiled autodiff launch resource"))
                        return VERNON_STATUS_INTERNAL_ERROR;
                } else if (spec.role == GpuResourceRole::Input || spec.role == GpuResourceRole::Storage) {
                    const auto source = node.inputs.find(spec.value.path);
                    if (source == node.inputs.end())
                        return fail(context, "compiled autodiff graph node has an unbound input");
                    if (source->second.sourceNode) {
                        const auto &upstream = invocationNodes[*source->second.sourceNode];
                        const auto output = upstream.forwardBuffers.find(upstream.executable->signature().output.path);
                        if (output == upstream.forwardBuffers.end())
                            return fail(context, "compiled autodiff graph source output is unavailable");
                        buffer = output->second;
                    } else {
                        buffer = bound[source->second.external].buffer;
                    }
                } else {
                    const bool exported = spec.role == GpuResourceRole::Tape ||
                                          (spec.role == GpuResourceRole::Output && nodeIndex == outputNode);
                    if (createGraphBuffer(*graph, spec.value, buffer, exported) != VERNON_RHI_STATUS_OK)
                        return fail(context, "failed to allocate a compiled autodiff graph resource",
                                    VERNON_STATUS_INTERNAL_ERROR);
                }
                node.forwardBuffers.emplace(spec.value.path, buffer);
                passBindings.push_back({spec, buffer});
            }
            graph->emplacePass<AutodiffComputePass>(node.name + ".forward", node.executable, std::move(passBindings),
                                                    computeGrid, false);
        }
        if (graph->execute() != VERNON_RHI_STATUS_OK)
            return fail(context, "failed to execute compiled autodiff forward graph", VERNON_STATUS_INTERNAL_ERROR);

        AutodiffGraphPullback::Impl::Node &sink = invocationNodes[outputNode];
        ValueAbi outputSpec = sink.executable->signature().output;
        if (!ad::materializeCarrierValue(outputSpec, computeGrid))
            return fail(context, "compiled autodiff output size overflows");
        VernonAdValue *output = ad::findValue(outputs, outputSpec.path);
        const auto outputBuffer = sink.forwardBuffers.find(outputSpec.path);
        if (!output || !ad::valueMatches(*output, outputSpec) || outputBuffer == sink.forwardBuffers.end())
            return fail(context, "compiled autodiff graph output does not match the sink signature");
        if (!ad::downloadGpuBuffer(*context, outputBuffer->second.handle, 0, output->data, output->size,
                                   "compiled autodiff graph output"))
            return VERNON_STATUS_INTERNAL_ERROR;
        for (BoundExternal &binding : bound)
            if (binding.descriptor->writable &&
                !ad::downloadGpuBuffer(*context, binding.buffer.handle, 0, binding.value->data,
                                       binding.descriptor->value.byteSize, "compiled autodiff graph Storage"))
                return VERNON_STATUS_INTERNAL_ERROR;

        std::vector<AutodiffGraphPullback::Impl::External> pullbackExternal;
        pullbackExternal.reserve(external.size());
        for (const External &binding : external)
            pullbackExternal.push_back({binding.gradientPath, binding.value});
        auto execution =
            std::make_unique<AutodiffGraphPullback::Impl>(contextLease, std::move(graph), std::move(invocationNodes),
                                                          order, std::move(pullbackExternal), computeGrid, outputNode);
        pullback = CompiledAutodiffGraph::makePullback(std::move(execution));
        return VERNON_STATUS_OK;
    }

    std::shared_ptr<ContextLease> contextLease;
    VernonRuntimeContext *context{};
    const std::vector<Node> nodes;
    const std::vector<uint32_t> order;
    const std::vector<External> external;
    const uint32_t outputNode{};
};

class AutodiffGraph::Impl {
public:
    using InputSource = AutodiffGraphPullback::Impl::InputSource;

    struct External {
        uint32_t node{};
        std::string inputPath;
        std::string valuePath;
        std::string gradientPath;
        ValueAbi value;
        bool writable{};
    };

    struct Node {
        std::string name;
        std::shared_ptr<GpuGraphExecutable> executable;
        std::unordered_map<std::string, InputSource> inputs;
    };

    explicit Impl(VernonRuntimeContext *context)
        : contextLease(context ? acquireContextLease(*context) : nullptr), context(context),
          identity(nextAutodiffGraphIdentity.fetch_add(1, std::memory_order_relaxed)) {}

    bool valid(AutodiffGraphNode node) const { return node.graphIdentity == identity && node.index < nodes.size(); }

    VernonStatus orderNodes(std::vector<uint32_t> &order) const {
        if (!outputNode)
            return fail(context, "autodiff graph has no output node");
        std::vector<std::vector<uint32_t>> consumers(nodes.size());
        std::vector<uint32_t> indegree(nodes.size());
        for (uint32_t destination = 0; destination < nodes.size(); ++destination)
            for (const auto &[_, source] : nodes[destination].inputs)
                if (source.sourceNode) {
                    consumers[*source.sourceNode].push_back(destination);
                    ++indegree[destination];
                }
        std::queue<uint32_t> ready;
        for (uint32_t index = 0; index < indegree.size(); ++index)
            if (!indegree[index])
                ready.push(index);
        while (!ready.empty()) {
            const uint32_t current = ready.front();
            ready.pop();
            order.push_back(current);
            for (uint32_t consumer : consumers[current])
                if (!--indegree[consumer])
                    ready.push(consumer);
        }
        if (order.size() != nodes.size())
            return fail(context, "autodiff graph contains a cycle");
        std::vector<bool> contributes(nodes.size());
        std::vector<uint32_t> work{*outputNode};
        contributes[*outputNode] = true;
        while (!work.empty()) {
            const uint32_t current = work.back();
            work.pop_back();
            for (const auto &[_, source] : nodes[current].inputs)
                if (source.sourceNode && !contributes[*source.sourceNode]) {
                    contributes[*source.sourceNode] = true;
                    work.push_back(*source.sourceNode);
                }
        }
        if (std::find(contributes.begin(), contributes.end(), false) != contributes.end())
            return fail(context, "autodiff graph contains a node that does not contribute to its output");
        return VERNON_STATUS_OK;
    }

    std::shared_ptr<ContextLease> contextLease;
    VernonRuntimeContext *context{};
    std::vector<Node> nodes;
    std::vector<External> external;
    std::optional<uint32_t> outputNode;
    uint64_t identity{};
};

AutodiffGraphPullback::AutodiffGraphPullback(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
AutodiffGraphPullback::~AutodiffGraphPullback() = default;
AutodiffGraphPullback::AutodiffGraphPullback(AutodiffGraphPullback &&) noexcept = default;
AutodiffGraphPullback &AutodiffGraphPullback::operator=(AutodiffGraphPullback &&) noexcept = default;

VernonStatus AutodiffGraphPullback::apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients) {
    return impl_ ? impl_->apply(cotangents, gradients) : VERNON_STATUS_INVALID_ARGUMENT;
}

VernonRhiCommandEncoderStats AutodiffGraphPullback::forwardStats() const {
    RuntimeDiagnosticScope diagnostic(impl_ ? impl_->context : nullptr);
    return impl_ ? impl_->getForwardStats() : VernonRhiCommandEncoderStats{};
}

VernonRhiCommandEncoderStats AutodiffGraphPullback::lastBackwardStats() const {
    RuntimeDiagnosticScope diagnostic(impl_ ? impl_->context : nullptr);
    return impl_ ? impl_->getLastBackwardStats() : VernonRhiCommandEncoderStats{};
}

AutodiffGraph::AutodiffGraph(VernonRuntimeContext *context) : impl_(std::make_unique<Impl>(context)) {}
AutodiffGraph::~AutodiffGraph() = default;
AutodiffGraph::AutodiffGraph(AutodiffGraph &&) noexcept = default;
AutodiffGraph &AutodiffGraph::operator=(AutodiffGraph &&) noexcept = default;

VernonStatus AutodiffGraph::addNode(std::string name, VernonLoadedPipeline *pipeline, AutodiffGraphNode &node) {
    RuntimeDiagnosticScope diagnostic(impl_ ? impl_->context : nullptr);
    node = {};
    if (!impl_ || !impl_->context || !pipeline || pipeline->context != impl_->context || !pipeline->autodiff ||
        name.empty())
        return fail(impl_ ? impl_->context : nullptr, "invalid autodiff graph node");
    auto executable = std::dynamic_pointer_cast<GpuGraphExecutable>(pipeline->autodiff->executable);
    if (!executable)
        return fail(impl_->context, "autodiff ExecutionGraph requires a GPU resource-backed pipeline");
    if (std::any_of(impl_->nodes.begin(), impl_->nodes.end(),
                    [&](const Impl::Node &existing) { return existing.name == name; }))
        return fail(impl_->context, "autodiff graph node names must be unique");
    node.index = static_cast<uint32_t>(impl_->nodes.size());
    node.graphIdentity = impl_->identity;
    impl_->nodes.push_back({std::move(name), std::move(executable), {}});
    return VERNON_STATUS_OK;
}

VernonStatus AutodiffGraph::declareInput(AutodiffGraphNode node, std::string inputPath, std::string valuePath,
                                         std::string gradientPath) {
    RuntimeDiagnosticScope diagnostic(impl_ ? impl_->context : nullptr);
    if (!impl_ || !impl_->valid(node) || inputPath.empty() || valuePath.empty())
        return fail(impl_ ? impl_->context : nullptr, "invalid autodiff graph input declaration");
    Impl::Node &target = impl_->nodes[node.index];
    const ResourceAbi *spec = resource(target.executable->forwardResources(), inputPath);
    if (!spec || (spec->role != GpuResourceRole::Input && spec->role != GpuResourceRole::Storage) ||
        target.inputs.count(inputPath) ||
        std::any_of(impl_->external.begin(), impl_->external.end(),
                    [&](const Impl::External &existing) { return existing.valuePath == valuePath; }))
        return fail(impl_->context, "autodiff graph input declaration does not match the node signature");
    Impl::External external{node.index,           inputPath,
                            std::move(valuePath), std::move(gradientPath),
                            spec->value,          spec->role == GpuResourceRole::Storage};
    const size_t index = impl_->external.size();
    impl_->external.push_back(std::move(external));
    target.inputs.emplace(std::move(inputPath), Impl::InputSource{std::nullopt, index});
    return VERNON_STATUS_OK;
}

VernonStatus AutodiffGraph::connect(AutodiffGraphNode source, AutodiffGraphNode destination,
                                    std::string destinationInputPath) {
    RuntimeDiagnosticScope diagnostic(impl_ ? impl_->context : nullptr);
    if (!impl_ || !impl_->valid(source) || !impl_->valid(destination) || source.index == destination.index ||
        destinationInputPath.empty())
        return fail(impl_ ? impl_->context : nullptr, "invalid autodiff graph connection");
    Impl::Node &target = impl_->nodes[destination.index];
    const ValueAbi &output = impl_->nodes[source.index].executable->signature().output;
    const ResourceAbi *input = resource(target.executable->forwardResources(), destinationInputPath);
    if (!input || input->role != GpuResourceRole::Input || !ad::sameValueAbi(output, input->value) ||
        target.inputs.count(destinationInputPath))
        return fail(impl_->context, "autodiff graph connection ABIs do not match");
    target.inputs.emplace(std::move(destinationInputPath), Impl::InputSource{source.index, {}});
    return VERNON_STATUS_OK;
}

VernonStatus AutodiffGraph::setOutput(AutodiffGraphNode node) {
    RuntimeDiagnosticScope diagnostic(impl_ ? impl_->context : nullptr);
    if (!impl_ || !impl_->valid(node))
        return fail(impl_ ? impl_->context : nullptr, "invalid autodiff graph output node");
    impl_->outputNode = node.index;
    return VERNON_STATUS_OK;
}

VernonStatus AutodiffGraph::compile(std::unique_ptr<CompiledAutodiffGraph> &compiled) {
    RuntimeDiagnosticScope diagnostic(impl_ ? impl_->context : nullptr);
    compiled.reset();
    if (!impl_ || !impl_->context || impl_->nodes.empty() || !impl_->outputNode)
        return fail(impl_ ? impl_->context : nullptr, "invalid autodiff graph compilation");
    std::vector<uint32_t> order;
    VernonStatus status = impl_->orderNodes(order);
    if (status != VERNON_STATUS_OK)
        return status;
    for (const Impl::Node &node : impl_->nodes)
        for (const ResourceAbi &spec : node.executable->forwardResources())
            if ((spec.role == GpuResourceRole::Input || spec.role == GpuResourceRole::Storage) &&
                spec.value.path != kLaunchResourcePath && !node.inputs.count(spec.value.path))
                return fail(impl_->context, "autodiff graph node has an unbound input");

    std::vector<CompiledAutodiffGraph::Impl::Node> nodes;
    nodes.reserve(impl_->nodes.size());
    for (const Impl::Node &node : impl_->nodes)
        nodes.push_back({node.name, node.executable, node.inputs});
    std::vector<CompiledAutodiffGraph::Impl::External> external;
    external.reserve(impl_->external.size());
    for (const Impl::External &binding : impl_->external)
        external.push_back({binding.node, binding.inputPath, binding.valuePath, binding.gradientPath, binding.value,
                            binding.writable});
    auto plan =
        std::make_unique<CompiledAutodiffGraph::Impl>(acquireContextLease(*impl_->context), std::move(nodes),
                                                      std::move(order), std::move(external), *impl_->outputNode);
    compiled = std::unique_ptr<CompiledAutodiffGraph>(new CompiledAutodiffGraph(std::move(plan)));
    return VERNON_STATUS_OK;
}

CompiledAutodiffGraph::CompiledAutodiffGraph(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CompiledAutodiffGraph::~CompiledAutodiffGraph() = default;
CompiledAutodiffGraph::CompiledAutodiffGraph(CompiledAutodiffGraph &&) noexcept = default;
CompiledAutodiffGraph &CompiledAutodiffGraph::operator=(CompiledAutodiffGraph &&) noexcept = default;

std::unique_ptr<AutodiffGraphPullback>
CompiledAutodiffGraph::makePullback(std::unique_ptr<AutodiffGraphPullback::Impl> impl) {
    return std::unique_ptr<AutodiffGraphPullback>(new AutodiffGraphPullback(std::move(impl)));
}

VernonStatus CompiledAutodiffGraph::forward(VernonLaunchSize computeGrid, VernonAdValueSet &inputs,
                                            VernonAdValueSet &outputs,
                                            std::unique_ptr<AutodiffGraphPullback> &pullback) const {
    return impl_ ? impl_->forward(computeGrid, inputs, outputs, pullback) : VERNON_STATUS_INVALID_ARGUMENT;
}

} // namespace vernon::runtime
