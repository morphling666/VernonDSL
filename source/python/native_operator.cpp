#include "native_operator.h"

#include "execution_graph/execution_graph_internal.h"
#include "native_pipeline.h"
#include "native_rhi.h"
#include "native_runtime.h"
#include "operator/elementwise_operator.h"
#include "operator/operator_execution.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct PreparedOperatorInvocation {
    explicit PreparedOperatorInvocation(PipelineInvocationBuilder &builder)
        : runtime(builder.runtime), pipeline(builder.pipeline) {
        std::vector<VernonPipelineArgument> ignored;
        invocation = builder.invocation(ignored);
        arguments.reserve(builder.arguments.size());
        values.reserve(builder.arguments.size());
        for (const PreparedPipelineArgument *argument : builder.arguments)
            arguments.push_back(std::make_unique<PreparedPipelineArgument>(*argument));
        for (const auto &argument : arguments)
            values.push_back(argument->value);
        invocation.arguments = values.empty() ? nullptr : values.data();
        invocation.argument_count = values.size();
        invocation.command_encoder = {};
        invocation.index_binding = nullptr;
        invocation.color_attachments = nullptr;
        invocation.color_attachment_count = 0;
        invocation.depth_attachment = nullptr;
    }

    PreparedPipelineArgument &argument(uint32_t slot) {
        const auto found = std::find_if(arguments.begin(), arguments.end(),
                                        [&](const auto &value) { return value->value.slot == slot; });
        if (found == arguments.end())
            throw std::invalid_argument("operator pipeline argument is not bound");
        return **found;
    }

    VernonRuntimeContext *runtime{};
    VernonLoadedPipeline *pipeline{};
    std::vector<std::unique_ptr<PreparedPipelineArgument>> arguments;
    std::vector<VernonPipelineArgument> values;
    VernonPipelineInvocation invocation{};
};

VernonRhiStatus encodeOperatorInvocation(void *opaque, VernonRhiCommandEncoder encoder) {
    auto &invocation = *static_cast<PreparedOperatorInvocation *>(opaque);
    VernonRuntimeProviderObject provider{};
    if (vernonRuntimeReferenceRhiCommandEncoder(invocation.runtime, encoder, &provider) != VERNON_STATUS_OK)
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    return vernonRuntimePipelineEncode(provider, invocation.pipeline, &invocation.invocation) == VERNON_STATUS_OK
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

struct TensorViewKey {
    uint64_t ownerIdentity{};
    uint64_t byteOffset{};
    VernonDataType dtype{};
    std::vector<uint64_t> shape;
    std::vector<int64_t> byteStrides;

    explicit TensorViewKey(const vernon::ops::TensorViewDescriptor &view)
        : ownerIdentity(view.ownerIdentity), byteOffset(view.byteOffset), dtype(view.dtype), shape(view.shape),
          byteStrides(view.byteStrides) {}

    bool operator==(const TensorViewKey &other) const {
        return ownerIdentity == other.ownerIdentity && byteOffset == other.byteOffset && dtype == other.dtype &&
               shape == other.shape && byteStrides == other.byteStrides;
    }
};

struct TensorViewKeyHash {
    size_t operator()(const TensorViewKey &key) const {
        size_t result = std::hash<uint64_t>{}(key.ownerIdentity);
        const auto append = [&](uint64_t value) {
            result ^= std::hash<uint64_t>{}(value) + 0x9e3779b97f4a7c15ull + (result << 6) + (result >> 2);
        };
        append(key.byteOffset);
        append(static_cast<uint32_t>(key.dtype));
        for (uint64_t extent : key.shape)
            append(extent);
        for (int64_t stride : key.byteStrides)
            append(static_cast<uint64_t>(stride));
        return result;
    }
};

class PythonCommandCompletion final : public vernon::execution::detail::RhiCommandCompletion {
public:
    explicit PythonCommandCompletion(nb::object transaction) : transaction_(std::move(transaction)) {}

    void complete(bool succeeded, const vernon::execution::detail::RhiCommandDagExecutionStats &) override {
        nb::gil_scoped_acquire acquire;
        if (transaction_.is_none())
            return;
        transaction_.attr(succeeded ? "_commit_planned_state" : "_rollback_planned_state")();
        transaction_ = nb::none();
    }

private:
    nb::object transaction_;
};

} // namespace

struct PythonOperatorDagBuilder::Impl {
    explicit Impl(PipelineInvocationBuilder &builder)
        : runtime(builder.runtime),
          device(builder.owner && builder.owner->rhiHost
                     ? builder.owner->rhiHost->device
                     : VernonRhiDevice{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0}) {
        if (!runtime || device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
            throw std::invalid_argument("operator DAG execution requires an initialized RHI runtime");
    }

    vernon::ops::TensorViewDescriptor descriptor(PreparedPipelineArgument &argument, bool writable) {
        if (argument.value.kind != VERNON_PIPELINE_TENSOR)
            throw std::invalid_argument("operator Add arguments must be RHI Tensors");
        const VernonTensorView &tensor = argument.value.tensor;
        const bool accessAllowed =
            writable ? tensor.access == VERNON_ACCESS_WRITE || tensor.access == VERNON_ACCESS_READ_WRITE
                     : tensor.access == VERNON_ACCESS_READ || tensor.access == VERNON_ACCESS_READ_WRITE;
        if (!accessAllowed)
            throw std::invalid_argument(writable ? "operator Add output does not permit writes"
                                                 : "operator Add input does not permit reads");
        RhiBuffer *buffer = nb::cast<RhiBuffer *>(argument.owner);
        if (tensor.storage != VERNON_TENSOR_RHI_RESOURCE || !buffer ||
            buffer->handle.index == VERNON_RHI_INVALID_HANDLE_INDEX || tensor.element_layout.leaf_count != 1 ||
            !tensor.element_layout.leaves || tensor.element_layout.leaves[0].scalar_count != 1)
            throw std::invalid_argument("operator Add requires a single-scalar-leaf RHI Tensor element layout");
        vernon::ops::TensorViewDescriptor result;
        result.dtype = static_cast<VernonDataType>(tensor.element_layout.leaves[0].dtype);
        result.shape = argument.shape;
        result.byteStrides = argument.strides;
        result.byteOffset = tensor.byte_offset;
        result.allocationBytes = tensor.byte_size;
        result.ownerIdentity = vernon::rhi::encodeResourceKey(buffer->handle);
        if (std::none_of(bindings.begin(), bindings.end(),
                         [&](const auto &binding) { return binding.aliasDomain == result.ownerIdentity; }))
            bindings.push_back({result.ownerIdentity, vernon::execution::ResourceKind::Buffer, buffer->handle, {}});
        return result;
    }

    uint32_t inputNode(const vernon::ops::TensorViewDescriptor &view) {
        const TensorViewKey key(view);
        const auto found = values.find(key);
        if (found != values.end())
            return found->second;
        const uint32_t node = operators.addLeaf(view);
        values.emplace(key, node);
        return node;
    }

    VernonRuntimeContext *runtime{};
    VernonRhiDevice device{};
    vernon::ops::OperatorDag operators;
    std::vector<vernon::execution::detail::RhiCommandNodeEncoder> encoders;
    std::vector<vernon::execution::detail::RhiCommandResourceBinding> bindings;
    std::vector<std::unique_ptr<PreparedOperatorInvocation>> invocations;
    std::unordered_map<TensorViewKey, uint32_t, TensorViewKeyHash> values;
    bool executed{};
};

PythonOperatorDagBuilder::PythonOperatorDagBuilder(PipelineInvocationBuilder &builder)
    : impl_(std::make_shared<Impl>(builder)) {}

PythonOperatorDagBuilder::~PythonOperatorDagBuilder() = default;

void PythonOperatorDagBuilder::addElementwiseAdd(PipelineInvocationBuilder &builder, const nb::object &output,
                                                 const nb::object &left, const nb::object &right) {
    if (!impl_ || impl_->executed || builder.runtime != impl_->runtime)
        throw std::invalid_argument("operator invocation belongs to another runtime or was already executed");
    auto invocation = std::make_unique<PreparedOperatorInvocation>(builder);
    PreparedPipelineArgument &outputArgument = invocation->argument(builder.resolveParameter(output).slot);
    PreparedPipelineArgument &leftArgument = invocation->argument(builder.resolveParameter(left).slot);
    PreparedPipelineArgument &rightArgument = invocation->argument(builder.resolveParameter(right).slot);
    const vernon::ops::TensorViewDescriptor outputView = impl_->descriptor(outputArgument, true);
    const vernon::ops::TensorViewDescriptor leftView = impl_->descriptor(leftArgument, false);
    const vernon::ops::TensorViewDescriptor rightView = impl_->descriptor(rightArgument, false);
    vernon::ops::ElementwiseAddPlan plan;
    std::string error;
    if (!vernon::ops::planElementwiseAdd(leftView, rightView, outputView, plan, error))
        throw std::invalid_argument(error);
    if (!plan.device)
        throw std::invalid_argument(plan.fallbackReason);
    const uint32_t leftNode = impl_->inputNode(plan.left);
    const uint32_t rightNode = impl_->inputNode(plan.right);
    const uint32_t node =
        impl_->operators.addElementwise(vernon::ops::ElementwiseOperatorKind::Add, {leftNode, rightNode}, plan.output);
    impl_->values[TensorViewKey(plan.output)] = node;
    impl_->invocations.push_back(std::move(invocation));
    impl_->encoders.push_back({encodeOperatorInvocation, impl_->invocations.back().get(), nullptr});
}

void PythonOperatorDagBuilder::execute(vernon::execution::detail::RhiCommandPlanSink *sink, nb::object retained) {
    if (!impl_ || impl_->executed || impl_->invocations.empty())
        throw std::invalid_argument("operator DAG is empty or was already executed");
    impl_->executed = true;
    vernon::ops::OperatorExecutionPlan plan;
    std::string error;
    if (!vernon::ops::buildOperatorExecutionPlan(impl_->operators, impl_->encoders, impl_->bindings, plan, error))
        throw std::runtime_error(error);
    if (sink) {
        plan.retainedContexts.push_back(impl_);
        if (sink->append(std::move(plan)) != VERNON_RHI_STATUS_OK)
            throw std::runtime_error("cannot append operator command DAG");
        if (!retained.is_none())
            sink->retain(std::make_shared<nb::object>(std::move(retained)));
        return;
    }
    VernonRhiStatus status = VERNON_RHI_STATUS_INTERNAL_ERROR;
    {
        nb::gil_scoped_release release;
        status = vernon::ops::executeOperatorExecutionPlanAndWait(impl_->device, plan);
    }
    if (status != VERNON_RHI_STATUS_OK) {
        const std::string diagnostic = nativeStringView(vernonRhiDeviceGetLastError(impl_->device));
        throw std::runtime_error("operator command DAG execution failed (RHI status " +
                                 std::to_string(static_cast<uint32_t>(status)) + ")" +
                                 (diagnostic.empty() ? std::string() : ": " + diagnostic));
    }
}

std::unique_ptr<PythonOperatorDagBuilder> createPythonOperatorDagBuilder(PipelineInvocationBuilder &builder) {
    return std::make_unique<PythonOperatorDagBuilder>(builder);
}

void retainPythonCommandCompletion(vernon::execution::detail::RhiCommandPlanSink &sink, nb::object transaction) {
    sink.onCompletion(std::make_shared<PythonCommandCompletion>(std::move(transaction)));
}

void retainPythonObject(vernon::execution::detail::RhiCommandPlanSink &sink, nb::object value) {
    sink.retain(std::make_shared<nb::object>(std::move(value)));
}
