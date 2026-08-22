#include "runtime_gpu_signature.h"

#include "runtime/runtime_state.h"

#include <algorithm>
#include <limits>
#include <optional>

namespace vernon::runtime::ad::gpu {
namespace {

bool appendParameterValues(const Parameter &parameter, std::vector<ValueAbi> &values, std::string &error) {
    const std::string &source = parameter.autodiffSource.empty() ? parameter.name : parameter.autodiffSource;
    return appendParameterValueAbi(parameter, source, values, error);
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
        (source->byteSize / sourceScalar &&
         derivativeScalar > std::numeric_limits<size_t>::max() / (source->byteSize / sourceScalar)))
        return false;
    byteSize = (source->byteSize / sourceScalar) * derivativeScalar;
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
