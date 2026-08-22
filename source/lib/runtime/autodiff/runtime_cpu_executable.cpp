#include "runtime_cpu_executable.h"

#include "host_effect_transaction.h"
#include "host_tape_allocator.h"
#include "runtime/backend_cpu.h"
#include "runtime/cpu_workgroup_dispatch.h"
#include "runtime/pipeline_metadata.h"
#include "runtime/runtime_state.h"
#include "runtime/tensor_bridge.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace vernon::runtime::ad::cpu {

bool validateDifferentiationSignature(VernonRuntimeContext &context, const Signature &signature) {
    if (signature.outputs.empty() || signature.cotangents.empty()) {
        invocationDiagnostic(context) = "autodiff output or cotangent signature is empty";
        return false;
    }
    std::set<std::string> outputPaths;
    for (const ValueAbi &output : signature.outputs)
        if (output.path.empty() || !outputPaths.insert(output.path).second) {
            invocationDiagnostic(context) = "autodiff output leaf paths are empty or duplicated";
            return false;
        }
    std::set<std::string> cotangentPaths;
    for (const ValueAbi &cotangent : signature.cotangents) {
        const auto output = std::find_if(signature.outputs.begin(), signature.outputs.end(),
                                         [&](const ValueAbi &value) { return value.path == cotangent.path; });
        if (cotangent.path.empty() || !cotangentPaths.insert(cotangent.path).second ||
            output == signature.outputs.end() || !derivativeAbiMatches(*output, cotangent)) {
            invocationDiagnostic(context) = "autodiff output and cotangent profile ABIs do not match";
            return false;
        }
    }
    std::set<std::string> inputPaths;
    for (const ValueAbi &input : signature.inputs)
        if (input.path.empty() || !inputPaths.insert(input.path).second) {
            invocationDiagnostic(context) = "autodiff input leaf paths are empty or duplicated";
            return false;
        }
    std::set<std::string> gradientPaths;
    for (const ValueAbi &gradient : signature.gradients) {
        if (gradient.path.empty() || !gradientPaths.insert(gradient.path).second) {
            invocationDiagnostic(context) = "autodiff gradient leaf paths are empty or duplicated";
            return false;
        }
        const auto primal = std::find_if(signature.inputs.begin(), signature.inputs.end(),
                                         [&](const ValueAbi &input) { return input.path == gradient.path; });
        if (primal == signature.inputs.end() || !derivativeAbiMatches(*primal, gradient)) {
            invocationDiagnostic(context) = "autodiff gradient leaf ABI does not match its primal input";
            return false;
        }
    }
    return true;
}

bool finishStructuredCpuExecutable(VernonRuntimeContext &context, const Stage &primalStage, const Stage &forwardStage,
                                   const Stage &backwardStage, std::shared_ptr<CpuKernelState> primalKernel,
                                   std::shared_ptr<CpuKernelState> forwardKernel,
                                   std::shared_ptr<CpuKernelState> backwardKernel,
                                   const std::vector<std::string> &gradientPaths, uint64_t staticTapeBytesHint,
                                   const std::string &residualStorage, const std::string &selectedPolicy,
                                   bool wholeDispatchRetentionPermitted, std::shared_ptr<Executable> &executable) {
    HostProfileLayout primal;
    HostProfileLayout forward;
    HostProfileLayout backward;
    if (!parseProfile(primalStage, primal, invocationDiagnostic(context)) ||
        !parseProfile(forwardStage, forward, invocationDiagnostic(context)) ||
        !parseProfile(backwardStage, backward, invocationDiagnostic(context)))
        return false;
    if (!std::equal(std::begin(primal.workgroup), std::end(primal.workgroup), std::begin(forward.workgroup)) ||
        !std::equal(std::begin(forward.workgroup), std::end(forward.workgroup), std::begin(backward.workgroup))) {
        invocationDiagnostic(context) = "structured CPU autodiff profiles use different workgroup sizes";
        return false;
    }
    if (!primal.results.empty() || primal.tapeAllocatorOffset || primal.tapeRootRegionOffset) {
        invocationDiagnostic(context) = "structured CPU autodiff primal profile has an invalid Tape ABI";
        return false;
    }
    CpuResidualStorage storage{};
    if (residualStorage == "none")
        storage = CpuResidualStorage::None;
    else if (residualStorage == "static")
        storage = CpuResidualStorage::Static;
    else if (residualStorage == "dynamic")
        storage = CpuResidualStorage::Dynamic;
    else {
        invocationDiagnostic(context) = "structured CPU autodiff planning metadata is invalid";
        return false;
    }
    PlanningPolicy policy{};
    if (!parsePlanningPolicy(selectedPolicy, policy)) {
        invocationDiagnostic(context) = "structured CPU autodiff planning metadata is invalid";
        return false;
    }
    const bool noTape = storage == CpuResidualStorage::None;
    if ((noTape != (staticTapeBytesHint == 0)) || staticTapeBytesHint > std::numeric_limits<size_t>::max()) {
        invocationDiagnostic(context) = "structured CPU autodiff planning metadata is invalid";
        return false;
    }
    if (wholeDispatchRetentionPermitted &&
        (storage != CpuResidualStorage::Static || policy == PlanningPolicy::MinMemory)) {
        invocationDiagnostic(context) = "structured CPU autodiff whole-dispatch permission is inconsistent";
        return false;
    }
    const bool validNoTape = noTape && forward.results.empty() && !forward.tapeAllocatorOffset &&
                             !forward.tapeRootRegionOffset && !backward.tapeAllocatorOffset &&
                             !backward.tapeRootRegionOffset;
    const bool validTape = !noTape && forward.results.empty() && forward.tapeAllocatorOffset &&
                           !forward.tapeRootRegionOffset && backward.tapeAllocatorOffset &&
                           backward.tapeRootRegionOffset;
    if (!validNoTape && !validTape) {
        invocationDiagnostic(context) = "structured CPU autodiff profile Tape ABI disagrees with its static hint";
        return false;
    }
    Signature signature;
    signature.storageObjectives = true;
    for (const HostArgument &argument : forward.arguments) {
        if (!argument.builtin.empty())
            continue;
        if (argument.tensorView)
            signature.inputs.insert(signature.inputs.end(), argument.tensorView->leaves.begin(),
                                    argument.tensorView->leaves.end());
        else
            for (const HostFrameLeaf &leaf : argument.leaves)
                signature.inputs.push_back(leaf.value);
    }
    std::vector<const HostArgument *> cotangentArguments;
    for (const HostArgument &argument : backward.arguments)
        if (argument.builtin.empty() && !isShapeSource(argument) && !isPrimalSource(argument) &&
            std::find(gradientPaths.begin(), gradientPaths.end(), argument.name) == gradientPaths.end())
            cotangentArguments.push_back(&argument);
    for (const HostArgument *argument : cotangentArguments) {
        const size_t leafCount = argument->tensorView ? argument->tensorView->leaves.size() : argument->leaves.size();
        if (argument->name.empty() || leafCount != 1) {
            invocationDiagnostic(context) = "structured CPU autodiff cotangent is not one canonical leaf";
            return false;
        }
        ValueAbi cotangent =
            argument->tensorView ? argument->tensorView->leaves.front() : argument->leaves.front().value;
        cotangent.path = argument->name;
        signature.cotangents.push_back(std::move(cotangent));
    }
    for (const ValueAbi &cotangent : signature.cotangents) {
        const auto input = std::find_if(signature.inputs.begin(), signature.inputs.end(),
                                        [&](const ValueAbi &value) { return value.path == cotangent.path; });
        if (input == signature.inputs.end()) {
            invocationDiagnostic(context) = "structured CPU autodiff Storage objective is not a primal input";
            return false;
        }
        signature.outputs.push_back(*input);
    }
    std::vector<size_t> resultGradientIndices;
    size_t resultIndex = 0;
    for (const std::string &gradientPath : gradientPaths) {
        const auto storageArgument =
            std::find_if(backward.arguments.begin(), backward.arguments.end(), [&](const HostArgument &argument) {
                return argument.builtin.empty() && argument.tensorView && argument.name == gradientPath;
            });
        ValueAbi gradient;
        if (storageArgument != backward.arguments.end()) {
            if (storageArgument->tensorView->leaves.size() != 1 || storageArgument->accumulationOwnership.empty()) {
                invocationDiagnostic(context) =
                    "structured CPU autodiff storage gradient has no canonical leaf or ownership";
                return false;
            }
            gradient = storageArgument->tensorView->leaves.front();
        } else {
            if (resultIndex >= backward.results.size()) {
                invocationDiagnostic(context) = "structured CPU autodiff has too few value gradient results";
                return false;
            }
            gradient = backward.results[resultIndex++].value;
            resultGradientIndices.push_back(signature.gradients.size());
        }
        gradient.path = gradientPath;
        signature.gradients.push_back(std::move(gradient));
    }
    if (resultIndex != backward.results.size()) {
        invocationDiagnostic(context) = "structured CPU autodiff has unclaimed value gradient results";
        return false;
    }
    if (!validateDifferentiationSignature(context, signature))
        return false;
    auto program = std::make_shared<CpuAutodiffProgram>();
    program->primalLayout = std::move(primal);
    program->forwardLayout = std::move(forward);
    program->backwardLayout = std::move(backward);
    program->primal = std::move(primalKernel);
    program->forward = std::move(forwardKernel);
    program->backward = std::move(backwardKernel);
    program->signature = std::move(signature);
    program->resultGradientIndices = std::move(resultGradientIndices);
    program->requiredPrimalTensorOwners = requiredPrimalTensorOwners(program->backwardLayout);
    if (noTape) {
        executable = createNoTapeStructuredCpuExecutable(context, std::move(program));
    } else {
        if (!context.autodiffMemoryPolicy)
            context.autodiffMemoryPolicy = std::make_shared<AutodiffMemoryPolicy>();
        auto plan = std::make_shared<CpuResidualPlan>(CpuResidualPlan{
            storage, policy, static_cast<size_t>(staticTapeBytesHint), wholeDispatchRetentionPermitted});
        executable = createTapedStructuredCpuExecutable(context, std::move(program), std::move(plan));
    }
    return true;
}

bool loadStructuredCpuExecutable(VernonRuntimeContext &context, const Stage &primalStage, const Stage &forwardStage,
                                 const Stage &backwardStage, const std::vector<std::string> &gradientPaths,
                                 uint64_t staticTapeBytesHint, const std::string &residualStorage,
                                 const std::string &selectedPolicy, bool wholeDispatchRetentionPermitted,
                                 std::shared_ptr<Executable> &executable) {
    if (!primalStage.cpuArtifact || !forwardStage.cpuArtifact || !backwardStage.cpuArtifact) {
        invocationDiagnostic(context) = "CPU autodiff profiles have no loadable artifacts";
        return false;
    }
    auto primalKernel = std::make_shared<CpuKernelState>();
    auto forwardKernel = std::make_shared<CpuKernelState>();
    auto backwardKernel = std::make_shared<CpuKernelState>();
    ReflectedEntry reflection;
    if (!loadCpuNativeArtifact(context, *primalStage.cpuArtifact, *primalKernel, reflection,
                               invocationDiagnostic(context)) ||
        !loadCpuNativeArtifact(context, *forwardStage.cpuArtifact, *forwardKernel, reflection,
                               invocationDiagnostic(context)) ||
        !loadCpuNativeArtifact(context, *backwardStage.cpuArtifact, *backwardKernel, reflection,
                               invocationDiagnostic(context)))
        return false;
    return finishStructuredCpuExecutable(context, primalStage, forwardStage, backwardStage, std::move(primalKernel),
                                         std::move(forwardKernel), std::move(backwardKernel), gradientPaths,
                                         staticTapeBytesHint, residualStorage, selectedPolicy,
                                         wholeDispatchRetentionPermitted, executable);
}

bool loadStructuredCpuEntryExecutable(VernonRuntimeContext &context, const Stage &primalStage,
                                      VernonCpuEntryPoint primalEntry, const Stage &forwardStage,
                                      VernonCpuEntryPoint forwardEntry, const Stage &backwardStage,
                                      VernonCpuEntryPoint backwardEntry, const std::vector<std::string> &gradientPaths,
                                      uint64_t staticTapeBytesHint, const std::string &residualStorage,
                                      const std::string &selectedPolicy, bool wholeDispatchRetentionPermitted,
                                      std::shared_ptr<Executable> &executable) {
    auto primalKernel = std::make_shared<CpuKernelState>();
    auto forwardKernel = std::make_shared<CpuKernelState>();
    auto backwardKernel = std::make_shared<CpuKernelState>();
    ReflectedEntry reflection;
    if (!loadCpuEntry(primalEntry, primalStage.reflection.data(), primalStage.reflection.size(),
                      primalStage.entry.data(), primalStage.entry.size(), *primalKernel, reflection,
                      invocationDiagnostic(context)) ||
        !loadCpuEntry(forwardEntry, forwardStage.reflection.data(), forwardStage.reflection.size(),
                      forwardStage.entry.data(), forwardStage.entry.size(), *forwardKernel, reflection,
                      invocationDiagnostic(context)) ||
        !loadCpuEntry(backwardEntry, backwardStage.reflection.data(), backwardStage.reflection.size(),
                      backwardStage.entry.data(), backwardStage.entry.size(), *backwardKernel, reflection,
                      invocationDiagnostic(context)))
        return false;
    return finishStructuredCpuExecutable(context, primalStage, forwardStage, backwardStage, std::move(primalKernel),
                                         std::move(forwardKernel), std::move(backwardKernel), gradientPaths,
                                         staticTapeBytesHint, residualStorage, selectedPolicy,
                                         wholeDispatchRetentionPermitted, executable);
}

} // namespace vernon::runtime::ad::cpu
