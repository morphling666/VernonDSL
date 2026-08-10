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
    const AutodiffDerivativeGroupView *groupViews, size_t derivativeGroupCount) {
    try {
        if (!groupViews || !derivativeGroupCount) {
            invocationDiagnostic(context) = "direct CPU autodiff requires derivative groups";
            return nullptr;
        }
        std::vector<AutodiffDerivativeGroup> derivativeGroups;
        derivativeGroups.reserve(derivativeGroupCount);
        for (size_t groupIndex = 0; groupIndex < derivativeGroupCount; ++groupIndex) {
            const AutodiffDerivativeGroupView &view = groupViews[groupIndex];
            if (!view.declaredPath.data || !view.declaredPath.size || !view.leafPaths || !view.leafCount) {
                invocationDiagnostic(context) = "direct CPU autodiff derivative group view is invalid";
                return nullptr;
            }
            AutodiffDerivativeGroup group;
            group.role = view.role;
            group.declaredPath.assign(view.declaredPath.data, view.declaredPath.size);
            group.leafPaths.reserve(view.leafCount);
            for (size_t leafIndex = 0; leafIndex < view.leafCount; ++leafIndex) {
                const VernonStringView leaf = view.leafPaths[leafIndex];
                if (!leaf.data || !leaf.size) {
                    invocationDiagnostic(context) = "direct CPU autodiff derivative leaf view is invalid";
                    return nullptr;
                }
                group.leafPaths.emplace_back(leaf.data, leaf.size);
            }
            derivativeGroups.push_back(std::move(group));
        }
        if (!validateAutodiffDerivativeGroups(derivativeGroups, invocationDiagnostic(context)))
            return nullptr;
        const std::vector<std::string> paths =
            autodiffDerivativeLeafPaths(derivativeGroups, AutodiffDerivativeRole::Gradient);
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
        if (!ad::validateDerivativeGroupsAgainstSignature(context, derivativeGroups, executable->signature()))
            return nullptr;
        pipeline->autodiff = VernonLoadedAutodiff{std::move(executable), derivativeGroups};
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
