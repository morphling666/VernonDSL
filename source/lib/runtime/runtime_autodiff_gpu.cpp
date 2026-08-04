#include "runtime_autodiff_internal.h"

#include "pipeline_metadata.h"
#include "runtime_dispatch.h"
#include "runtime_state.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vernon::runtime::ad {
namespace {

struct ResourceBinding {
    ResourceAbi abi;
    uint32_t slot{};
    ValueLayout elementLayout;
};

struct ResourceProfileLayout {
    std::vector<ResourceBinding> resources;
    std::unordered_map<std::string, size_t> byPath;
};

struct GpuProfile {
    std::shared_ptr<VernonLoadedPipeline> pipeline;
    ResourceProfileLayout layout;
    std::vector<ResourceAbi> resourceAbis;
    std::vector<VernonPipelineArgument> argumentTemplate;
    VernonLaunchSize workgroup{1, 1, 1};
    bool serialDispatch{};
};

struct GpuInvocation {
    std::vector<VernonPipelineArgument> arguments;
    std::vector<std::vector<uint64_t>> shapes;
    std::vector<std::vector<int64_t>> strides;
    VernonPipelineInvocation invocation{};
};

VernonStatus fail(VernonRuntimeContext &context, std::string message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    invocationDiagnostic(context) = std::move(message);
    return status;
}

GpuResourceRole graphRole(AutodiffResourceRole role) {
    switch (role) {
    case AutodiffResourceRole::Input:
        return GpuResourceRole::Input;
    case AutodiffResourceRole::Storage:
        return GpuResourceRole::Storage;
    case AutodiffResourceRole::Output:
        return GpuResourceRole::Output;
    case AutodiffResourceRole::Tape:
        return GpuResourceRole::Tape;
    case AutodiffResourceRole::Cotangent:
        return GpuResourceRole::Cotangent;
    case AutodiffResourceRole::Gradient:
        return GpuResourceRole::Gradient;
    case AutodiffResourceRole::None:
        break;
    }
    return GpuResourceRole::Input;
}

bool parseResourceProfile(const VernonLoadedPipeline &pipeline, ResourceProfileLayout &layout, std::string &error) {
    for (const Parameter &parameter : pipeline.variant.parameters) {
        if (parameter.kind != "tensor" || parameter.name.empty() || parameter.shape.empty() ||
            parameter.elementLayout.byteSize == 0 || parameter.elementLayout.alignment == 0 ||
            parameter.elementLayout.leaves.size() != 1 || parameter.uses.size() != 1 ||
            parameter.uses[0].interfaceKind != "storage") {
            error = "autodiff GPU profile has a non-canonical resource parameter table";
            return false;
        }
        const ValueLeaf &leaf = parameter.elementLayout.leaves.front();
        const auto dtype = pipelineDataType(leaf.dtype);
        const size_t scalarSize = dtype ? dtypeSize(*dtype) : 0;
        const GpuResourceRole role = graphRole(parameter.autodiffRole);
        const bool runtimeCarrier =
            role == GpuResourceRole::Output || role == GpuResourceRole::Tape || role == GpuResourceRole::Cotangent;
        const bool dynamicShape =
            std::any_of(parameter.shape.begin(), parameter.shape.end(), [](uint64_t extent) { return extent == 0; });
        if ((runtimeCarrier && parameter.shape != std::vector<uint64_t>{0, 0, 0}) ||
            (!runtimeCarrier && dynamicShape)) {
            error = "autodiff GPU profile has an invalid dynamic resource shape";
            return false;
        }
        size_t elementCount = 1;
        for (uint64_t extent : parameter.shape) {
            if (!extent)
                continue;
            if (extent > SIZE_MAX / elementCount) {
                error = "autodiff GPU resource shape overflows";
                return false;
            }
            elementCount *= static_cast<size_t>(extent);
        }
        if (!scalarSize || !leaf.scalarCount || leaf.scalarCount > SIZE_MAX / scalarSize || leaf.byteOffset != 0 ||
            parameter.elementLayout.byteSize != leaf.scalarCount * scalarSize) {
            error = "autodiff GPU resource has an invalid Value carrier layout";
            return false;
        }
        if (!runtimeCarrier && parameter.elementLayout.byteSize > SIZE_MAX / elementCount) {
            error = "autodiff GPU resource byte size overflows";
            return false;
        }
        VernonValueAccess access{};
        if (parameter.access == "read")
            access = VERNON_ACCESS_READ;
        else if (parameter.access == "write")
            access = VERNON_ACCESS_WRITE;
        else if (parameter.access == "read_write")
            access = VERNON_ACCESS_READ_WRITE;
        else {
            error = "autodiff GPU resource has an invalid access mode";
            return false;
        }
        if (parameter.autodiffRole == AutodiffResourceRole::None ||
            ((parameter.autodiffRole == AutodiffResourceRole::Input ||
              parameter.autodiffRole == AutodiffResourceRole::Cotangent) &&
             access != VERNON_ACCESS_READ) ||
            (parameter.autodiffRole == AutodiffResourceRole::Storage && access != VERNON_ACCESS_READ_WRITE &&
             access != VERNON_ACCESS_WRITE) ||
            (parameter.autodiffRole == AutodiffResourceRole::Output && access != VERNON_ACCESS_WRITE) ||
            (parameter.autodiffRole == AutodiffResourceRole::Gradient && access != VERNON_ACCESS_WRITE &&
             access != VERNON_ACCESS_READ_WRITE)) {
            error = "autodiff GPU resource has an invalid or missing typed role";
            return false;
        }
        if (!layout.byPath.emplace(parameter.name, layout.resources.size()).second) {
            error = "autodiff GPU profile contains duplicate resource paths";
            return false;
        }
        ValueLayout elementLayout = parameter.elementLayout;
        rebuildValueLayoutPathViews(elementLayout);
        std::vector<uint64_t> logicalShape = leaf.shape;
        if (!runtimeCarrier && logicalShape.empty() && parameter.shape != std::vector<uint64_t>{1})
            logicalShape = parameter.shape;
        const size_t byteSize = runtimeCarrier ? static_cast<size_t>(parameter.elementLayout.byteSize)
                                               : static_cast<size_t>(parameter.elementLayout.byteSize) * elementCount;
        layout.resources.push_back({{{parameter.name, *dtype, byteSize,
                                      static_cast<size_t>(parameter.elementLayout.alignment), std::move(logicalShape)},
                                     parameter.shape,
                                     byteSize,
                                     access,
                                     role,
                                     runtimeCarrier},
                                    parameter.slot,
                                    std::move(elementLayout)});
    }
    if (layout.resources.empty()) {
        error = "autodiff GPU profile has no resource parameters";
        return false;
    }
    return true;
}

const ResourceBinding *binding(const ResourceProfileLayout &layout, const std::string &path) {
    const auto found = layout.byPath.find(path);
    return found == layout.byPath.end() ? nullptr : &layout.resources[found->second];
}

const ResourceBinding *uniqueBinding(const ResourceProfileLayout &layout, GpuResourceRole role) {
    const ResourceBinding *result = nullptr;
    for (const ResourceBinding &resource : layout.resources) {
        if (resource.abi.role != role)
            continue;
        if (result)
            return nullptr;
        result = &resource;
    }
    return result;
}

void buildInvocationTemplate(GpuProfile &profile) {
    const std::vector<ResourceBinding> &resources = profile.layout.resources;
    profile.argumentTemplate.resize(resources.size());
    for (size_t index = 0; index < resources.size(); ++index) {
        const ResourceBinding &resource = resources[index];
        VernonPipelineArgument &argument = profile.argumentTemplate[index];
        argument.slot = resource.slot;
        argument.kind = VERNON_PIPELINE_TENSOR;
        argument.tensor.struct_size = sizeof(VernonTensorView);
        argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
        argument.tensor.element_layout = pipelineValueLayout(resource.elementLayout);
        argument.tensor.access = resource.abi.access;
        argument.tensor.rank = resource.abi.physicalShape.size();
        argument.tensor.byte_size = resource.abi.byteSize;
    }
}

bool prepareInvocation(GpuProfile &profile, VernonLaunchSize computeGrid, const std::vector<GpuBufferBinding> &buffers,
                       GpuInvocation &prepared, std::string &error) {
    if (buffers.size() != profile.layout.resources.size()) {
        error = "autodiff GPU invocation resource count does not match its immutable profile";
        return false;
    }
    prepared.arguments = profile.argumentTemplate;
    prepared.shapes.resize(profile.layout.resources.size());
    prepared.strides.resize(profile.layout.resources.size());
    for (size_t index = 0; index < profile.layout.resources.size(); ++index) {
        const ResourceBinding &resource = profile.layout.resources[index];
        const GpuBufferBinding &buffer = buffers[index];
        size_t requiredBytes = 0;
        if (!resourceByteSize(resource.abi, computeGrid, requiredBytes) || buffer.size != requiredBytes ||
            buffer.buffer.index == VERNON_RHI_INVALID_HANDLE_INDEX) {
            error = "autodiff GPU invocation has no matching resource buffer";
            return false;
        }
        prepared.shapes[index] = resource.abi.runtimeCarrier
                                     ? std::vector<uint64_t>{computeGrid.z, computeGrid.y, computeGrid.x}
                                     : resource.abi.physicalShape;
        prepared.strides[index].resize(prepared.shapes[index].size());
        size_t stride = resource.elementLayout.byteSize;
        for (size_t dimension = prepared.shapes[index].size(); dimension-- > 0;) {
            if (stride > static_cast<size_t>(INT64_MAX)) {
                error = "autodiff GPU resource stride overflows";
                return false;
            }
            prepared.strides[index][dimension] = static_cast<int64_t>(stride);
            if (dimension && prepared.shapes[index][dimension] > SIZE_MAX / stride) {
                error = "autodiff GPU resource stride overflows";
                return false;
            }
            stride *= static_cast<size_t>(prepared.shapes[index][dimension]);
        }
        VernonRuntimeProviderResourceReference reference{};
        const VernonStatus status =
            referenceBackendRhiBuffer(*profile.pipeline->context, buffer.buffer, 0, buffer.size, reference);
        if (status != VERNON_STATUS_OK) {
            error = "failed to reference an autodiff GPU buffer";
            return false;
        }
        VernonPipelineArgument &argument = prepared.arguments[index];
        argument.tensor.resource = reference;
        argument.tensor.rank = prepared.shapes[index].size();
        argument.tensor.shape = prepared.shapes[index].data();
        argument.tensor.byte_strides = prepared.strides[index].data();
        argument.tensor.byte_size = requiredBytes;
    }
    prepared.invocation.struct_size = sizeof(VernonPipelineInvocation);
    prepared.invocation.abi_version = VERNON_PIPELINE_VERSION;
    prepared.invocation.arguments = prepared.arguments.data();
    prepared.invocation.argument_count = prepared.arguments.size();
    prepared.invocation.compute_grid = {
        (computeGrid.x - 1) / profile.workgroup.x + 1,
        (computeGrid.y - 1) / profile.workgroup.y + 1,
        (computeGrid.z - 1) / profile.workgroup.z + 1,
    };
    return true;
}

void buildResourceAbis(GpuProfile &profile) {
    profile.resourceAbis.reserve(profile.layout.resources.size());
    for (const ResourceBinding &resource : profile.layout.resources)
        profile.resourceAbis.push_back(resource.abi);
}

VernonStatus dispatchProfileUnlocked(GpuProfile &profile, VernonRuntimeProviderObject encoder,
                                     VernonLaunchSize computeGrid, const std::vector<GpuBufferBinding> &bindings) {
    VernonRuntimeContext &context = *profile.pipeline->context;
    if (encoder.value == 0)
        return fail(context, "autodiff GPU graph encoding requires a command encoder");
    if (bindings.size() != profile.layout.resources.size())
        return fail(context, "autodiff GPU graph bindings do not match the resource profile");
    GpuInvocation invocation;
    if (!prepareInvocation(profile, computeGrid, bindings, invocation, invocationDiagnostic(context)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (profile.serialDispatch)
        invocation.invocation.compute_grid = {1, 1, 1};
    return vernonRuntimePipelineEncode(encoder, profile.pipeline.get(), &invocation.invocation);
}

bool resolveProfile(VernonPipelineBundle &bundle, const std::string &id,
                    std::shared_ptr<VernonLoadedPipeline> &resolved) {
    const Stage &stage = bundle.stages.at(id);
    auto profile = std::make_unique<VernonLoadedPipeline>();
    profile->context = bundle.context;
    if (!buildReflectedComputeVariant(stage, bundle.context->backend, profile->variant, bundle.context->error))
        return false;
    profile->variant.compute = id;
    profile->variant.program["compute"] = id;
    for (Parameter &parameter : profile->variant.parameters)
        rebuildValueLayoutPathViews(parameter.elementLayout);
    if (!resolveBackendPipeline(bundle, profile->variant, *profile))
        return false;
    resolved = std::shared_ptr<VernonLoadedPipeline>(profile.release(), [](VernonLoadedPipeline *value) {
        destroyBackendPipeline(*value);
        delete value;
    });
    return true;
}

class GpuExecutable final : public GpuGraphExecutable {
public:
    GpuExecutable(VernonRuntimeContext &context, std::shared_ptr<GpuProfile> forward,
                  std::shared_ptr<GpuProfile> backward, Signature signature)
        : context_(context), forward_(std::move(forward)), backward_(std::move(backward)),
          signature_(std::move(signature)) {}

    const Signature &signature() const override { return signature_; }
    VernonRuntimeContext &context() const override { return context_; }
    VernonRhiDevice device() const override { return context_.rhiDevice; }
    const std::vector<ResourceAbi> &forwardResources() const override { return forward_->resourceAbis; }
    const std::vector<ResourceAbi> &backwardResources() const override { return backward_->resourceAbis; }

    VernonStatus encodeForward(VernonRuntimeProviderObject encoder, VernonLaunchSize computeGrid,
                               const std::vector<GpuBufferBinding> &bindings) override {
        return dispatchProfileUnlocked(*forward_, encoder, computeGrid, bindings);
    }

    VernonStatus encodeBackward(VernonRuntimeProviderObject encoder, VernonLaunchSize computeGrid,
                                const std::vector<GpuBufferBinding> &bindings) override {
        return dispatchProfileUnlocked(*backward_, encoder, computeGrid, bindings);
    }

private:
    VernonRuntimeContext &context_;
    std::shared_ptr<GpuProfile> forward_;
    std::shared_ptr<GpuProfile> backward_;
    Signature signature_;
};

} // namespace

bool createGpuExecutable(VernonPipelineBundle &bundle, const std::string &forwardId, const std::string &backwardId,
                         const std::vector<std::string> &gradientPaths, const AutodiffLaunchPlan &launch,
                         std::shared_ptr<Executable> &executable) {
    auto forward = std::make_shared<GpuProfile>();
    auto backward = std::make_shared<GpuProfile>();
    if (!resolveProfile(bundle, forwardId, forward->pipeline) ||
        !resolveProfile(bundle, backwardId, backward->pipeline) ||
        !parseResourceProfile(*forward->pipeline, forward->layout, bundle.context->error) ||
        !parseResourceProfile(*backward->pipeline, backward->layout, bundle.context->error))
        return false;
    buildInvocationTemplate(*forward);
    buildInvocationTemplate(*backward);
    const auto workgroupMatches = [&](const Stage &stage) {
        return stage.workgroup[0] == launch.workgroupSize.x && stage.workgroup[1] == launch.workgroupSize.y &&
               stage.workgroup[2] == launch.workgroupSize.z;
    };
    const bool serialBackward = std::any_of(
        launch.accumulationPlans.begin(), launch.accumulationPlans.end(), [&](const AutodiffAccumulationPlan &plan) {
            if (std::find(plan.evidence.begin(), plan.evidence.end(), "disjoint_scatter") != plan.evidence.end())
                return false;
            if (bundle.context->backend != VERNON_RUNTIME_CUDA)
                return true;
            const ResourceBinding *gradient = binding(backward->layout, plan.path);
            return gradient && gradient->abi.role == GpuResourceRole::Gradient &&
                   gradient->abi.value.dtype != VERNON_DATA_F32;
        });
    const Stage &backwardStage = bundle.stages.at(backwardId);
    const bool backwardWorkgroupMatches =
        serialBackward
            ? backwardStage.workgroup[0] == 1 && backwardStage.workgroup[1] == 1 && backwardStage.workgroup[2] == 1
            : workgroupMatches(backwardStage);
    if (!workgroupMatches(bundle.stages.at(forwardId)) || !backwardWorkgroupMatches) {
        bundle.context->error = "autodiff GPU profile workgroup size does not match its launch plan";
        return false;
    }
    forward->workgroup = launch.workgroupSize;
    backward->workgroup = serialBackward ? VernonLaunchSize{1, 1, 1} : launch.workgroupSize;
    backward->serialDispatch = serialBackward;
    Signature signature;
    for (const ResourceBinding &resource : forward->layout.resources) {
        if (resource.abi.value.path == "__vernon_launch") {
            if (resource.abi.role != GpuResourceRole::Input || resource.abi.value.dtype != VERNON_DATA_U32 ||
                resource.abi.value.logicalShape != std::vector<uint64_t>{3}) {
                bundle.context->error = "autodiff forward GPU profile has an invalid launch resource";
                return false;
            }
        } else if (resource.abi.role == GpuResourceRole::Input || resource.abi.role == GpuResourceRole::Storage) {
            signature.inputs.push_back(resource.abi.value);
        } else if (resource.abi.role == GpuResourceRole::Output) {
            if (!signature.output.path.empty()) {
                bundle.context->error = "autodiff forward GPU profile contains multiple output resources";
                return false;
            }
            signature.output = resource.abi.value;
        } else if (resource.abi.role == GpuResourceRole::Tape) {
            signature.tape.push_back(resource.abi.value);
        } else {
            bundle.context->error = "autodiff forward GPU profile contains a resource with an invalid role";
            return false;
        }
    }
    if (signature.output.path.empty()) {
        bundle.context->error = "autodiff forward GPU profile has no output resource";
        return false;
    }
    if (!binding(forward->layout, "__vernon_launch")) {
        bundle.context->error = "autodiff forward GPU profile has no launch resource";
        return false;
    }
    const ResourceBinding *forwardOutput = uniqueBinding(forward->layout, GpuResourceRole::Output);
    if (!forwardOutput) {
        bundle.context->error = "autodiff forward GPU profile has an invalid output resource";
        return false;
    }
    const ResourceBinding *cotangent = uniqueBinding(backward->layout, GpuResourceRole::Cotangent);
    if (!cotangent || !sameValueAbi(signature.output, cotangent->abi.value)) {
        bundle.context->error = "autodiff output and cotangent GPU resource ABIs do not match";
        return false;
    }
    signature.cotangent = cotangent->abi.value;
    if (cotangent->abi.physicalShape != forwardOutput->abi.physicalShape) {
        bundle.context->error = "autodiff output and cotangent physical shapes do not match";
        return false;
    }
    for (const ValueAbi &tape : signature.tape) {
        const ResourceBinding *backwardTape = binding(backward->layout, tape.path);
        if (!backwardTape || backwardTape->abi.role != GpuResourceRole::Tape ||
            backwardTape->abi.access != VERNON_ACCESS_READ || !sameValueAbi(tape, backwardTape->abi.value)) {
            bundle.context->error = "autodiff tape GPU resource ABIs do not match";
            return false;
        }
    }
    const ResourceBinding *backwardLaunch = binding(backward->layout, "__vernon_launch");
    if (!backwardLaunch || backwardLaunch->abi.role != GpuResourceRole::Input ||
        backwardLaunch->abi.value.dtype != VERNON_DATA_U32 ||
        backwardLaunch->abi.value.logicalShape != std::vector<uint64_t>{3}) {
        bundle.context->error = "autodiff backward GPU profile has no matching launch resource";
        return false;
    }
    for (const std::string &path : gradientPaths) {
        ResourceBinding *gradient = nullptr;
        const auto found = backward->layout.byPath.find(path);
        if (found != backward->layout.byPath.end())
            gradient = &backward->layout.resources[found->second];
        if (!gradient || gradient->abi.role != GpuResourceRole::Gradient) {
            bundle.context->error = "autodiff gradient paths do not match the GPU resource profile";
            return false;
        }
        signature.gradients.push_back(gradient->abi.value);
    }
    if (backward->layout.resources.size() != signature.tape.size() + 2 + signature.gradients.size()) {
        bundle.context->error = "autodiff backward GPU profile contains unexpected resources";
        return false;
    }
    buildResourceAbis(*forward);
    buildResourceAbis(*backward);
    executable =
        std::make_shared<GpuExecutable>(*bundle.context, std::move(forward), std::move(backward), std::move(signature));
    return true;
}

} // namespace vernon::runtime::ad
