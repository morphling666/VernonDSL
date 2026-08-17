#include "runtime_autodiff_internal.h"
#include "runtime_cpu_executable.h"

#include "runtime/runtime_state.h"

#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
#include "host_tape_test_hooks.h"
#endif

namespace vernon::runtime::ad {

bool createCpuExecutable(VernonRuntimeContext &context, const Stage &primalStage, const Stage &forwardStage,
                         const Stage &backwardStage, const std::vector<std::string> &gradientPaths,
                         uint64_t staticTapeBytesHint, const std::string &residualStorage,
                         const std::string &selectedPolicy, bool wholeDispatchRetentionPermitted,
                         std::shared_ptr<Executable> &executable) {
    return cpu::loadStructuredCpuExecutable(context, primalStage, forwardStage, backwardStage, gradientPaths,
                                            staticTapeBytesHint, residualStorage, selectedPolicy,
                                            wholeDispatchRetentionPermitted, executable);
}

bool createCpuEntryExecutable(VernonRuntimeContext &context, VernonCpuEntryPoint primalEntry,
                              VernonStringView primalReflection, VernonStringView primalName,
                              VernonCpuEntryPoint forwardEntry, VernonStringView forwardReflection,
                              VernonStringView forwardName, VernonCpuEntryPoint backwardEntry,
                              VernonStringView backwardReflection, VernonStringView backwardName,
                              const std::vector<std::string> &gradientPaths, uint64_t staticTapeBytesHint,
                              const std::string &residualStorage, const std::string &selectedPolicy,
                              bool wholeDispatchRetentionPermitted, std::shared_ptr<Executable> &executable) {
    if (!primalEntry || !forwardEntry || !backwardEntry || !primalReflection.data || !primalReflection.size ||
        !forwardReflection.data || !forwardReflection.size || !backwardReflection.data || !backwardReflection.size ||
        !primalName.data || !primalName.size || !forwardName.data || !forwardName.size || !backwardName.data ||
        !backwardName.size) {
        invocationDiagnostic(context) = "direct CPU autodiff profiles are invalid";
        return false;
    }
    Stage primalStage;
    primalStage.entry.assign(primalName.data, primalName.size);
    primalStage.reflection.assign(primalReflection.data, primalReflection.size);
    Stage forwardStage;
    forwardStage.entry.assign(forwardName.data, forwardName.size);
    forwardStage.reflection.assign(forwardReflection.data, forwardReflection.size);
    Stage backwardStage;
    backwardStage.entry.assign(backwardName.data, backwardName.size);
    backwardStage.reflection.assign(backwardReflection.data, backwardReflection.size);
    return cpu::loadStructuredCpuEntryExecutable(
        context, primalStage, primalEntry, forwardStage, forwardEntry, backwardStage, backwardEntry, gradientPaths,
        staticTapeBytesHint, residualStorage, selectedPolicy, wholeDispatchRetentionPermitted, executable);
}

#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
void setHostTapeMemoryPolicyForTesting(VernonRuntimeContext &context, std::shared_ptr<HostTapeMemoryPolicy> policy) {
    context.autodiffMemoryPolicy = std::move(policy);
}
#endif

} // namespace vernon::runtime::ad
