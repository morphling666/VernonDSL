#include "runtime_direct_autodiff.h"

#include "runtime/runtime_dispatch.h"
#include "runtime_autodiff_internal.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vernon::runtime {

VernonLoadedPipeline *loadBackendCpuAutodiffPipeline(VernonRuntimeContext &context, VernonCpuEntryPoint primalEntry,
                                                     VernonStringView primalReflection, VernonStringView primalName,
                                                     VernonCpuEntryPoint forwardEntry,
                                                     VernonStringView forwardReflection, VernonStringView forwardName,
                                                     VernonCpuEntryPoint backwardEntry,
                                                     VernonStringView backwardReflection, VernonStringView backwardName,
                                                     const VernonStringView *gradientPaths, size_t gradientPathCount) {
    if ((!gradientPaths && gradientPathCount) || !gradientPathCount) {
        invocationDiagnostic(context) = "direct CPU autodiff requires gradient paths";
        return nullptr;
    }
    std::vector<std::string> paths;
    paths.reserve(gradientPathCount);
    for (size_t index = 0; index < gradientPathCount; ++index) {
        if (!gradientPaths[index].data || !gradientPaths[index].size) {
            invocationDiagnostic(context) = "direct CPU autodiff gradient path is invalid";
            return nullptr;
        }
        paths.emplace_back(gradientPaths[index].data, gradientPaths[index].size);
    }
    VernonLoadedPipeline *pipeline = loadBackendCpuEntryPipeline(
        context, primalEntry, primalReflection.data, primalReflection.size, primalName.data, primalName.size);
    if (!pipeline)
        return nullptr;
    std::shared_ptr<ad::Executable> executable;
    if (!ad::createCpuEntryExecutable(context, forwardEntry, forwardReflection, forwardName, backwardEntry,
                                      backwardReflection, backwardName, paths, executable)) {
        std::string error = invocationDiagnostic(context);
        vernonRuntimeLoadedPipelineDestroy(pipeline);
        invocationDiagnostic(context) = std::move(error);
        return nullptr;
    }
    pipeline->autodiff = VernonLoadedAutodiff{std::move(executable)};
    return pipeline;
}

} // namespace vernon::runtime
