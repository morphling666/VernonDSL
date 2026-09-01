#include "runtime/autodiff/runtime_gpu_bindings.h"
#include "runtime/autodiff/runtime_autodiff_internal.h"

#include <algorithm>
#include <limits>

namespace vernon::runtime::ad::gpu {
namespace {

bool multiply(size_t left, size_t right, size_t &result) {
    if (left && right > std::numeric_limits<size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

size_t sourceSlot(std::vector<std::string> &keys, const std::string &key) {
    const auto found = std::find(keys.begin(), keys.end(), key);
    if (found != keys.end())
        return static_cast<size_t>(std::distance(keys.begin(), found));
    keys.push_back(key);
    return keys.size() - 1;
}

bool singleLeafParameter(const Parameter &parameter) {
    const ValueLayout &layout = parameter.valueLayout ? *parameter.valueLayout : parameter.elementLayout;
    return layout.leaves.size() == 1;
}

bool isReplaySegment(const Parameter &parameter) {
    return parameter.autodiffRole == AutodiffResourceRole::ReplaySegment;
}
bool isReplayStatus(const Parameter &parameter) { return parameter.autodiffRole == AutodiffResourceRole::ReplayStatus; }
bool isTape(const Parameter &parameter) { return parameter.autodiffRole == AutodiffResourceRole::Tape; }

bool buildLayoutRecipe(const Parameter &parameter, BindingSpec &spec) {
    const ValueLayout &layout = parameter.valueLayout ? *parameter.valueLayout : parameter.elementLayout;
    if (layout.leaves.empty() || !layout.byteSize)
        return false;
    size_t tensorViewRank = parameter.shape.size();
    bool descriptorRank = false;
    std::vector<uint64_t> tensorShape = parameter.shape;
    for (const ParameterUse &use : parameter.uses) {
        if (!use.tensorViewDescriptor)
            continue;
        if (descriptorRank && tensorViewRank != use.tensorViewDescriptor->rank)
            return false;
        tensorViewRank = use.tensorViewDescriptor->rank;
        descriptorRank = true;
        if (use.shape.size() != tensorViewRank)
            return false;
        if (tensorShape.size() != tensorViewRank)
            tensorShape.assign(tensorViewRank, 0);
        for (size_t dimension = 0; dimension < tensorViewRank; ++dimension) {
            if (tensorShape[dimension] && use.shape[dimension] && tensorShape[dimension] != use.shape[dimension])
                return false;
            if (!tensorShape[dimension])
                tensorShape[dimension] = use.shape[dimension];
        }
    }
    spec.tensorViewRank = tensorViewRank;
    spec.tensorShape = std::move(tensorShape);
    if (layout.leaves.size() == 1)
        spec.leafShape = layout.leaves.front().shape;
    spec.recordByteStride = layout.byteSize;
    return true;
}

bool materializePhysicalShape(const BindingSpec &spec, const std::vector<uint64_t> &logicalShape,
                              std::vector<uint64_t> &shape, std::vector<int64_t> &strides) {
    if (logicalShape.size() != spec.tensorViewRank + spec.leafShape.size())
        return false;
    shape.assign(logicalShape.begin(), logicalShape.begin() + spec.tensorViewRank);
    for (size_t dimension = 0; dimension < spec.tensorShape.size(); ++dimension)
        if (spec.tensorShape[dimension] && spec.tensorShape[dimension] != shape[dimension])
            return false;
    for (size_t dimension = 0; dimension < spec.leafShape.size(); ++dimension)
        if (spec.leafShape[dimension] && spec.leafShape[dimension] != logicalShape[spec.tensorViewRank + dimension])
            return false;
    strides.resize(shape.size());
    size_t stride = spec.recordByteStride;
    for (size_t dimension = shape.size(); dimension-- > 0;) {
        if (stride > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
            return false;
        strides[dimension] = static_cast<int64_t>(stride);
        if (!multiply(stride, static_cast<size_t>(shape[dimension]), stride))
            return false;
    }
    return true;
}

bool preservePhysicalShape(const BindingSpec &spec, const std::vector<uint64_t> &shape,
                           const std::vector<int64_t> &strides) {
    if (shape.size() != spec.tensorViewRank || strides.size() != shape.size())
        return false;
    for (size_t dimension = 0; dimension < spec.tensorShape.size(); ++dimension)
        if (spec.tensorShape[dimension] && spec.tensorShape[dimension] != shape[dimension])
            return false;
    size_t expected = spec.recordByteStride;
    for (size_t dimension = shape.size(); dimension-- > 0;) {
        if (expected > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
            return false;
        const int64_t expectedStride = static_cast<int64_t>(expected);
        if (strides[dimension] != expectedStride &&
            !(spec.carrierDimension && dimension == 0 && strides[dimension] == 0))
            return false;
        if (!multiply(expected, static_cast<size_t>(shape[dimension]), expected))
            return false;
    }
    return true;
}

bool compatiblePhysicalShape(const BindingSpec &spec, const std::vector<uint64_t> &shape,
                             const std::vector<int64_t> &strides) {
    if (shape.size() != spec.tensorViewRank || strides.size() != shape.size())
        return false;
    for (size_t dimension = 0; dimension < spec.tensorShape.size(); ++dimension)
        if (spec.tensorShape[dimension] && spec.tensorShape[dimension] != shape[dimension])
            return false;
    return true;
}

std::optional<size_t> derivativeAbiIndex(const Signature &signature, AutodiffResourceRole role,
                                         const std::string &path) {
    const std::vector<ValueAbi> &values =
        role == AutodiffResourceRole::Cotangent ? signature.cotangents : signature.gradients;
    const auto found =
        std::find_if(values.begin(), values.end(), [&](const ValueAbi &value) { return value.path == path; });
    return found == values.end() ? std::nullopt
                                 : std::optional<size_t>(static_cast<size_t>(std::distance(values.begin(), found)));
}

const ValueAbi *findDerivativeAbi(const Signature &signature, AutodiffResourceRole role, const std::string &path) {
    const std::optional<size_t> index = derivativeAbiIndex(signature, role, path);
    if (!index)
        return nullptr;
    const std::vector<ValueAbi> &values =
        role == AutodiffResourceRole::Cotangent ? signature.cotangents : signature.gradients;
    return &values[*index];
}

} // namespace

bool findDerivativeAbi(const Signature &signature, AutodiffResourceRole role, const std::string &path, size_t &index) {
    const std::optional<size_t> found = derivativeAbiIndex(signature, role, path);
    if (!found)
        return false;
    index = *found;
    return true;
}

bool buildBindingSpecPlan(const Variant &variant, const Signature &signature, BindingSpecPlan &plan,
                          std::string &error) {
    plan.clear();
    plan.reserve(variant.parameters.size());
    plan.derivativeSlotCount = signature.cotangents.size() + signature.gradients.size();
    plan.cotangentBindingBySignature.assign(signature.cotangents.size(), std::numeric_limits<size_t>::max());
    plan.gradientSlotBySignature.resize(signature.gradients.size());
    for (size_t index = 0; index < signature.gradients.size(); ++index)
        plan.gradientSlotBySignature[index] = signature.cotangents.size() + index;
    for (size_t index = 0; index < variant.parameters.size(); ++index) {
        const Parameter &parameter = variant.parameters[index];
        BindingSpec binding;
        binding.parameterIndex = index;
        if (isReplaySegment(parameter))
            binding.source = BindingSource::ReplaySegment;
        else if (isReplayStatus(parameter))
            binding.source = BindingSource::ReplayStatus;
        else if (isTape(parameter))
            binding.source = BindingSource::Tape;
        else if (parameter.autodiffRole == AutodiffResourceRole::LaunchMetadata)
            binding.source = BindingSource::Launch;
        else {
            binding.sourceName = parameter.autodiffSource.empty() ? parameter.name : parameter.autodiffSource;
            const bool derivative = parameter.autodiffRole == AutodiffResourceRole::Gradient ||
                                    parameter.autodiffRole == AutodiffResourceRole::Cotangent;
            binding.sourcePath = derivative ? parameter.name : binding.sourceName;
            if (singleLeafParameter(parameter)) {
                const ValueLayout &layout = parameter.valueLayout ? *parameter.valueLayout : parameter.elementLayout;
                binding.sourcePath = canonicalValueLeafPath(binding.sourcePath, layout.leaves.front());
            }
            if (derivative) {
                binding.source = BindingSource::Device;
                binding.valueClass = BindingSpec::ValueClass::Derivative;
                binding.derivativeRole = parameter.autodiffRole;
                if (!findDerivativeAbi(signature, binding.derivativeRole, binding.sourcePath, binding.signatureIndex)) {
                    error =
                        "GPU autodiff derivative binding '" + parameter.name + "' has no matching logical ABI value";
                    return false;
                }
                binding.sourceSlot = binding.signatureIndex;
                if (binding.derivativeRole == AutodiffResourceRole::Gradient)
                    binding.sourceSlot += signature.cotangents.size();
            } else if (parameter.autodiffRole == AutodiffResourceRole::RetainedPrimal) {
                binding.source = BindingSource::Host;
                binding.valueClass = BindingSpec::ValueClass::RetainedPrimal;
                binding.sourceSlot = sourceSlot(plan.retainedNames, binding.sourceName);
            } else if (parameter.autodiffRole == AutodiffResourceRole::Primal) {
                binding.source = BindingSource::Host;
                binding.valueClass = BindingSpec::ValueClass::Primal;
                binding.sourceSlot = sourceSlot(plan.primalNames, binding.sourceName);
            } else {
                error = "GPU autodiff binding '" + parameter.name + "' has no explicit role";
                return false;
            }
            if (!buildLayoutRecipe(parameter, binding)) {
                error =
                    "GPU autodiff binding '" + parameter.name + "' has an invalid reflected TensorView layout recipe";
                return false;
            }
            if (binding.valueClass == BindingSpec::ValueClass::Derivative) {
                const ValueAbi *abi = findDerivativeAbi(signature, parameter.autodiffRole, binding.sourcePath);
                if (!abi || abi->logicalShape.size() < binding.leafShape.size()) {
                    error =
                        "GPU autodiff derivative binding '" + parameter.name + "' has no matching logical ABI value";
                    return false;
                }
                const size_t logicalTensorRank = abi->logicalShape.size() - binding.leafShape.size();
                if (binding.tensorViewRank == logicalTensorRank + 1 &&
                    parameter.autodiffRole == AutodiffResourceRole::Cotangent)
                    binding.carrierDimension = true;
                else if (binding.tensorViewRank != logicalTensorRank) {
                    error =
                        "GPU autodiff derivative binding '" + parameter.name + "' has an invalid physical carrier rank";
                    return false;
                }
                for (const BindingSpec &existing : plan)
                    if (existing.valueClass == BindingSpec::ValueClass::Derivative &&
                        existing.derivativeRole == binding.derivativeRole &&
                        existing.sourcePath == binding.sourcePath &&
                        (existing.carrierDimension != binding.carrierDimension ||
                         existing.tensorViewRank != binding.tensorViewRank ||
                         existing.tensorShape != binding.tensorShape || existing.leafShape != binding.leafShape ||
                         existing.recordByteStride != binding.recordByteStride)) {
                        error = "GPU autodiff derivative binding '" + parameter.name +
                                "' conflicts with another physical carrier layout";
                        return false;
                    }
                if (binding.derivativeRole == AutodiffResourceRole::Cotangent) {
                    size_t &bindingIndex = plan.cotangentBindingBySignature[binding.signatureIndex];
                    if (bindingIndex == std::numeric_limits<size_t>::max())
                        bindingIndex = plan.bindings.size();
                }
            }
        }
        plan.push_back(std::move(binding));
    }
    return true;
}

bool materializeBindingPlan(const Variant &variant, const BindingSpecPlan &specs, DeviceValues &retainedDevices,
                            HostValues &retainedHosts, DeviceValues &working, DerivativeSlots &derivatives,
                            BindingPlan &plan, std::string &error) {
    plan.clear();
    plan.reserve(specs.size());
    std::vector<DeviceValue *> retainedDeviceSlots(specs.retainedNames.size());
    std::vector<HostValue *> retainedHostSlots(specs.retainedNames.size());
    for (size_t slot = 0; slot < specs.retainedNames.size(); ++slot) {
        if (auto value = retainedDevices.find(specs.retainedNames[slot]); value != retainedDevices.end())
            retainedDeviceSlots[slot] = &value->second;
        if (auto value = retainedHosts.find(specs.retainedNames[slot]); value != retainedHosts.end())
            retainedHostSlots[slot] = &value->second;
    }
    std::vector<DeviceValue *> primalDeviceSlots(specs.primalNames.size());
    std::vector<HostValue *> primalHostSlots(specs.primalNames.size());
    for (size_t slot = 0; slot < specs.primalNames.size(); ++slot) {
        if (auto value = working.find(specs.primalNames[slot]); value != working.end())
            primalDeviceSlots[slot] = &value->second;
        if (auto value = retainedHosts.find(specs.primalNames[slot]); value != retainedHosts.end())
            primalHostSlots[slot] = &value->second;
    }
    for (const BindingSpec &spec : specs) {
        if (spec.parameterIndex >= variant.parameters.size())
            return false;
        const Parameter &parameter = variant.parameters[spec.parameterIndex];
        Binding binding;
        binding.parameter = &parameter;
        binding.source = spec.source;
        if (spec.source != BindingSource::Device && spec.source != BindingSource::Host) {
            plan.push_back(std::move(binding));
            continue;
        }
        DeviceValue *device = nullptr;
        HostValue *host = nullptr;
        if (spec.valueClass == BindingSpec::ValueClass::Derivative) {
            if (spec.sourceSlot < derivatives.size() && derivatives[spec.sourceSlot])
                device = &*derivatives[spec.sourceSlot];
        } else if (spec.valueClass == BindingSpec::ValueClass::RetainedPrimal) {
            if (spec.sourceSlot < retainedDeviceSlots.size()) {
                device = retainedDeviceSlots[spec.sourceSlot];
                host = retainedHostSlots[spec.sourceSlot];
                if (device)
                    host = nullptr;
            }
        } else if (spec.valueClass == BindingSpec::ValueClass::Primal) {
            if (spec.sourceSlot < primalDeviceSlots.size()) {
                device = primalDeviceSlots[spec.sourceSlot];
                host = primalHostSlots[spec.sourceSlot];
                if (device)
                    host = nullptr;
            }
        }
        if ((device != nullptr) == (host != nullptr)) {
            error = "GPU autodiff binding '" + parameter.name + "' does not resolve to exactly one source";
            return false;
        }
        binding.source = device ? BindingSource::Device : BindingSource::Host;
        binding.device = device;
        binding.host = host;
        const std::vector<uint64_t> &logicalShape = device ? device->shape : host->shape;
        const std::vector<int64_t> &logicalStrides = device ? device->strides : host->strides;
        const bool preservedLayout = device && device->physicalLayout
                                         ? compatiblePhysicalShape(spec, logicalShape, logicalStrides)
                                         : preservePhysicalShape(spec, logicalShape, logicalStrides);
        if (!preservedLayout &&
            !materializePhysicalShape(spec, logicalShape, binding.materializedShape, binding.materializedStrides)) {
            error = "GPU autodiff binding '" + parameter.name +
                    "' has no valid physical TensorView layout (logical rank " + std::to_string(logicalShape.size()) +
                    ", reflected rank " + std::to_string(parameter.shape.size()) + ")";
            return false;
        }
        binding.materializedLayout = !preservedLayout;
        plan.push_back(std::move(binding));
    }
    return true;
}

} // namespace vernon::runtime::ad::gpu
