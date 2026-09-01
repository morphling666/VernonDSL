#include "runtime_gpu_signature.h"

#include "runtime/runtime_state.h"

#include <algorithm>

namespace vernon::runtime::ad::gpu {
namespace {

bool isAutodiffInternal(const Parameter &parameter) {
    return parameter.autodiffRole == AutodiffResourceRole::Tape ||
           parameter.autodiffRole == AutodiffResourceRole::ReplaySegment ||
           parameter.autodiffRole == AutodiffResourceRole::ReplayStatus ||
           parameter.autodiffRole == AutodiffResourceRole::LaunchMetadata;
}

} // namespace

bool buildSignature(VernonRuntimeContext &context, const Variant &forward, const Variant &backward,
                    const std::vector<std::string> &gradientPaths, Signature &signature) {
    signature.storageObjectives = true;
    for (const Parameter &parameter : forward.parameters)
        if (!isAutodiffInternal(parameter) &&
            (parameter.autodiffRole == AutodiffResourceRole::None ||
             parameter.autodiffRole == AutodiffResourceRole::Primal) &&
            !appendParameterValueAbi(parameter, parameter.name, signature.inputs, invocationDiagnostic(context)))
            return false;
    for (const Parameter &parameter : backward.parameters) {
        if (parameter.autodiffRole != AutodiffResourceRole::Gradient &&
            parameter.autodiffRole != AutodiffResourceRole::Cotangent)
            continue;
        std::vector<ValueAbi> &destination =
            parameter.autodiffRole == AutodiffResourceRole::Gradient ? signature.gradients : signature.cotangents;
        if (!appendParameterValueAbi(parameter, parameter.name, destination, invocationDiagnostic(context)))
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
        if (!materializeDerivativeValueAbi(cotangent, signature.outputs))
            return invocationDiagnostic(context) = "GPU autodiff cotangent ABI is inconsistent", false;
    for (ValueAbi &gradient : signature.gradients)
        if (!materializeDerivativeValueAbi(gradient, signature.inputs))
            return invocationDiagnostic(context) = "GPU autodiff gradient ABI is inconsistent", false;
    if (signature.inputs.empty() || signature.outputs.empty() || signature.cotangents.empty() ||
        signature.gradients.size() != gradientPaths.size())
        return invocationDiagnostic(context) = "GPU autodiff profile signature is incomplete", false;
    return true;
}

} // namespace vernon::runtime::ad::gpu
