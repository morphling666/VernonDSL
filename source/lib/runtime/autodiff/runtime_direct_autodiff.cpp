#include "runtime_direct_autodiff.h"

#include "runtime/runtime_dispatch.h"
#include "runtime_autodiff_internal.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vernon::runtime {

VernonLoadedPipeline *loadBackendCpuAutodiffPipeline(
    VernonRuntimeContext &context, VernonCpuEntryPoint primalEntry, VernonStringView primalReflection,
    VernonStringView primalName, VernonCpuEntryPoint forwardEntry, VernonStringView forwardReflection,
    VernonStringView forwardName, VernonCpuEntryPoint backwardEntry, VernonStringView backwardReflection,
    VernonStringView backwardName, VernonStringView forwardProtocol, VernonStringView backwardProtocol,
    const VernonStringView *gradientPaths, size_t gradientPathCount) {
    try {
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
        using PipelinePtr = std::unique_ptr<VernonLoadedPipeline, void (*)(VernonLoadedPipeline *)>;
        PipelinePtr pipeline(loadBackendCpuEntryPipeline(context, primalEntry, primalReflection.data,
                                                         primalReflection.size, primalName.data, primalName.size),
                             vernonRuntimeLoadedPipelineDestroy);
        if (!pipeline)
            return nullptr;
        std::shared_ptr<ad::Executable> executable;
        if (!ad::createCpuEntryExecutable(context, forwardEntry, forwardReflection, forwardName, backwardEntry,
                                          backwardReflection, backwardName, forwardProtocol, backwardProtocol, paths,
                                          executable)) {
            std::string error = invocationDiagnostic(context);
            pipeline.reset();
            invocationDiagnostic(context) = std::move(error);
            return nullptr;
        }
        pipeline->autodiff = VernonLoadedAutodiff{std::move(executable)};
        return pipeline.release();
    } catch (...) {
        try {
            invocationDiagnostic(context) = "cannot allocate direct CPU autodiff pipeline";
        } catch (...) {
        }
        return nullptr;
    }
}

} // namespace vernon::runtime
