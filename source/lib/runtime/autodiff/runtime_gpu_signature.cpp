#include "runtime_gpu_signature.h"

#include "runtime/runtime_state.h"

#include <algorithm>
#include <limits>
#include <optional>

namespace vernon::runtime::ad::gpu {
namespace {

bool checkedMultiply(size_t left, size_t right, size_t &result) {
    if (left && right > std::numeric_limits<size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

std::optional<VernonDataType> leafDtype(const ValueLeaf &leaf) { return pipelineDataType(leaf.dtype); }

std::string leafPath(const std::string &root, const ValueLeaf &leaf) {
    std::string result = root;
    for (const ValuePathComponent &component : leaf.path) {
        result.push_back('.');
        result += component.field ? *component.field : std::to_string(component.index);
    }
    return result;
}

bool appendParameterValues(const Parameter &parameter, std::vector<ValueAbi> &values, std::string &error) {
    const ValueLayout *layout = parameter.valueLayout ? &*parameter.valueLayout : &parameter.elementLayout;
    if (!layout || layout->leaves.empty()) {
        error = "GPU autodiff parameter has no canonical Value layout";
        return false;
    }
    size_t elementCount = 1;
    bool dynamicShape = false;
    for (uint64_t extent : parameter.shape) {
        if (!extent) {
            dynamicShape = true;
            continue;
        }
        if (!checkedMultiply(elementCount, static_cast<size_t>(extent), elementCount)) {
            error = "GPU autodiff reflected Tensor shape overflows";
            return false;
        }
    }
    const std::string &source = parameter.autodiffSource.empty() ? parameter.name : parameter.autodiffSource;
    for (const ValueLeaf &leaf : layout->leaves) {
        const std::optional<VernonDataType> dtype = leafDtype(leaf);
        const size_t scalarSize = dtype ? dtypeSize(*dtype) : 0;
        size_t byteSize = 0;
        if (!scalarSize || !leaf.scalarCount ||
            !checkedMultiply(static_cast<size_t>(leaf.scalarCount), scalarSize, byteSize) ||
            (!dynamicShape && !checkedMultiply(byteSize, elementCount, byteSize))) {
            error = "GPU autodiff parameter leaf byte size overflows";
            return false;
        }
        std::vector<uint64_t> shape = parameter.shape;
        shape.insert(shape.end(), leaf.shape.begin(), leaf.shape.end());
        ValueAbi candidate{leafPath(source, leaf), *dtype, byteSize, std::max<size_t>(layout->alignment, 1),
                           std::move(shape)};
        const auto existing = std::find_if(values.begin(), values.end(),
                                           [&](const ValueAbi &value) { return value.path == candidate.path; });
        if (existing != values.end()) {
            if (!sameValueAbi(*existing, candidate)) {
                error = "GPU autodiff parameters define conflicting ABI values for '" + candidate.path + "'";
                return false;
            }
            continue;
        }
        values.push_back(std::move(candidate));
    }
    return true;
}

bool isAutodiffInternal(const Parameter &parameter) {
    return parameter.autodiffRole == AutodiffResourceRole::Tape ||
           parameter.autodiffRole == AutodiffResourceRole::ReplaySegment ||
           parameter.autodiffRole == AutodiffResourceRole::ReplayStatus ||
           parameter.autodiffRole == AutodiffResourceRole::LaunchMetadata;
}

bool makeLogicalDerivativeAbi(ValueAbi &derivative, const std::vector<ValueAbi> &sources) {
    const auto source = std::find_if(sources.begin(), sources.end(),
                                     [&](const ValueAbi &value) { return value.path == derivative.path; });
    if (source == sources.end())
        return false;
    const size_t sourceScalar = dtypeSize(source->dtype);
    const size_t derivativeScalar = dtypeSize(derivative.dtype);
    size_t byteSize = 0;
    if (!sourceScalar || !derivativeScalar || source->byteSize % sourceScalar ||
        !checkedMultiply(source->byteSize / sourceScalar, derivativeScalar, byteSize))
        return false;
    derivative.logicalShape = source->logicalShape;
    derivative.byteSize = byteSize;
    return true;
}

} // namespace

bool buildSignature(VernonRuntimeContext &context, const Variant &forward, const Variant &backward,
                    const std::vector<std::string> &gradientPaths, Signature &signature) {
    signature.storageObjectives = true;
    for (const Parameter &parameter : forward.parameters)
        if (!isAutodiffInternal(parameter) &&
            (parameter.autodiffRole == AutodiffResourceRole::None ||
             parameter.autodiffRole == AutodiffResourceRole::Primal) &&
            !appendParameterValues(parameter, signature.inputs, invocationDiagnostic(context)))
            return false;
    for (const Parameter &parameter : backward.parameters) {
        if (parameter.autodiffRole != AutodiffResourceRole::Gradient &&
            parameter.autodiffRole != AutodiffResourceRole::Cotangent)
            continue;
        std::vector<ValueAbi> &destination =
            parameter.autodiffRole == AutodiffResourceRole::Gradient ? signature.gradients : signature.cotangents;
        if (!appendParameterValues(parameter, destination, invocationDiagnostic(context)))
            return false;
    }
    for (const ValueAbi &cotangent : signature.cotangents) {
        const auto input = std::find_if(signature.inputs.begin(), signature.inputs.end(),
                                        [&](const ValueAbi &value) { return value.path == cotangent.path; });
        if (input == signature.inputs.end())
            return invocationDiagnostic(context) = "GPU autodiff objective is not a primal Storage input", false;
        signature.outputs.push_back(*input);
    }
    for (ValueAbi &cotangent : signature.cotangents)
        if (!makeLogicalDerivativeAbi(cotangent, signature.outputs))
            return invocationDiagnostic(context) = "GPU autodiff cotangent ABI is inconsistent", false;
    for (ValueAbi &gradient : signature.gradients)
        if (!makeLogicalDerivativeAbi(gradient, signature.inputs))
            return invocationDiagnostic(context) = "GPU autodiff gradient ABI is inconsistent", false;
    if (signature.inputs.empty() || signature.outputs.empty() || signature.cotangents.empty() ||
        signature.gradients.size() != gradientPaths.size())
        return invocationDiagnostic(context) = "GPU autodiff profile signature is incomplete", false;
    return true;
}

} // namespace vernon::runtime::ad::gpu
