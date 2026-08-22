#include "runtime_direct_autodiff.h"

#include "runtime/runtime_dispatch.h"
#include "runtime_autodiff_internal.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vernon::runtime {
namespace {

bool materializeDerivativeGroups(VernonRuntimeContext &context, const AutodiffDerivativeGroupView *views, size_t count,
                                 std::vector<AutodiffDerivativeGroup> &groups) {
    if (!views || !count) {
        invocationDiagnostic(context) = "direct autodiff requires derivative groups";
        return false;
    }
    groups.reserve(count);
    for (size_t groupIndex = 0; groupIndex < count; ++groupIndex) {
        const AutodiffDerivativeGroupView &view = views[groupIndex];
        if (!view.declaredPath.data || !view.declaredPath.size || !view.leafPaths || !view.leafCount) {
            invocationDiagnostic(context) = "direct autodiff derivative group view is invalid";
            return false;
        }
        AutodiffDerivativeGroup group;
        group.role = view.role;
        group.declaredPath.assign(view.declaredPath.data, view.declaredPath.size);
        group.leafPaths.reserve(view.leafCount);
        for (size_t leafIndex = 0; leafIndex < view.leafCount; ++leafIndex) {
            const VernonStringView leaf = view.leafPaths[leafIndex];
            if (!leaf.data || !leaf.size) {
                invocationDiagnostic(context) = "direct autodiff derivative leaf view is invalid";
                return false;
            }
            group.leafPaths.emplace_back(leaf.data, leaf.size);
        }
        groups.push_back(std::move(group));
    }
    return validateAutodiffDerivativeGroups(groups, invocationDiagnostic(context));
}

bool validStageView(const AutodiffGpuStageView &view) {
    return view.artifact && view.artifactSize && view.reflection.data && view.reflection.size && view.entry.data &&
           view.entry.size;
}

} // namespace

VernonLoadedPipeline *loadBackendCpuAutodiffPipeline(
    VernonRuntimeContext &context, VernonCpuEntryPoint primalEntry, VernonStringView primalReflection,
    VernonStringView primalName, VernonCpuEntryPoint forwardEntry, VernonStringView forwardReflection,
    VernonStringView forwardName, VernonCpuEntryPoint backwardEntry, VernonStringView backwardReflection,
    VernonStringView backwardName, const AutodiffDerivativeGroupView *groupViews, size_t derivativeGroupCount,
    uint64_t staticTapeBytesHint, VernonStringView residualStorage, VernonStringView selectedPolicy,
    bool wholeDispatchRetentionPermitted) {
    try {
        if (!residualStorage.data || !residualStorage.size || !selectedPolicy.data || !selectedPolicy.size) {
            invocationDiagnostic(context) = "direct CPU autodiff metadata is incomplete";
            return nullptr;
        }
        std::vector<AutodiffDerivativeGroup> derivativeGroups;
        if (!materializeDerivativeGroups(context, groupViews, derivativeGroupCount, derivativeGroups))
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
        if (!ad::createCpuEntryExecutable(
                context, primalEntry, primalReflection, primalName, forwardEntry, forwardReflection, forwardName,
                backwardEntry, backwardReflection, backwardName, paths, staticTapeBytesHint,
                std::string(residualStorage.data, residualStorage.size),
                std::string(selectedPolicy.data, selectedPolicy.size), wholeDispatchRetentionPermitted, executable)) {
            std::string error = invocationDiagnostic(context);
            pipeline.reset();
            invocationDiagnostic(context) = std::move(error);
            return nullptr;
        }
        if (!ad::validateDerivativeGroupsAgainstSignature(context, derivativeGroups, executable->signature()))
            return nullptr;
        if (!pipeline->topology) {
            invocationDiagnostic(context) = "direct CPU pipeline has no normalized execution topology";
            return nullptr;
        }
        pipeline->topology->differentiated = VernonDifferentiatedPipeline{std::move(executable), derivativeGroups};
        return pipeline.release();
    } catch (...) {
        try {
            invocationDiagnostic(context) = "cannot allocate direct CPU autodiff pipeline";
        } catch (...) {
        }
        return nullptr;
    }
}

VernonLoadedPipeline *loadBackendGpuAutodiffPipeline(
    VernonRuntimeContext &context, const AutodiffGpuStageView &primal, const AutodiffGpuStageView &forward,
    const AutodiffGpuStageView &backward, const AutodiffDerivativeGroupView *groupViews, size_t derivativeGroupCount,
    uint64_t staticTapeBytesHint, VernonStringView residualStorage, VernonStringView selectedPolicy) {
    try {
        if (context.backend == VERNON_RUNTIME_CPU || !validStageView(primal) || !validStageView(forward) ||
            !validStageView(backward) || !residualStorage.data || !residualStorage.size || !selectedPolicy.data ||
            !selectedPolicy.size) {
            invocationDiagnostic(context) = "direct GPU autodiff metadata is incomplete";
            return nullptr;
        }
        std::vector<AutodiffDerivativeGroup> derivativeGroups;
        if (!materializeDerivativeGroups(context, groupViews, derivativeGroupCount, derivativeGroups))
            return nullptr;
        const std::vector<std::string> paths =
            autodiffDerivativeLeafPaths(derivativeGroups, AutodiffDerivativeRole::Gradient);
        auto materializeStage = [&](const AutodiffGpuStageView &view, Stage &stage) {
            Variant variant;
            ReflectedEntry reflection;
            return buildDirectComputeStage(context, view.artifact, view.artifactSize, view.reflection.data,
                                           view.reflection.size, view.entry.data, view.entry.size, stage, variant,
                                           reflection);
        };
        Stage primalStage;
        Stage forwardStage;
        Stage backwardStage;
        if (!materializeStage(primal, primalStage) || !materializeStage(forward, forwardStage) ||
            !materializeStage(backward, backwardStage))
            return nullptr;

        using PipelinePtr = std::unique_ptr<VernonLoadedPipeline, void (*)(VernonLoadedPipeline *)>;
        PipelinePtr pipeline(loadBackendArtifactPipeline(context, primal.artifact, primal.artifactSize,
                                                         primal.reflection.data, primal.reflection.size,
                                                         primal.entry.data, primal.entry.size),
                             vernonRuntimeLoadedPipelineDestroy);
        if (!pipeline)
            return nullptr;
        std::shared_ptr<ad::Executable> executable;
        if (!ad::createGpuExecutable(context, primalStage, forwardStage, backwardStage, paths, staticTapeBytesHint,
                                     std::string(residualStorage.data, residualStorage.size),
                                     std::string(selectedPolicy.data, selectedPolicy.size), executable)) {
            std::string error = invocationDiagnostic(context);
            pipeline.reset();
            invocationDiagnostic(context) = std::move(error);
            return nullptr;
        }
        if (!ad::validateDerivativeGroupsAgainstSignature(context, derivativeGroups, executable->signature()))
            return nullptr;
        if (!pipeline->topology) {
            invocationDiagnostic(context) = "direct GPU pipeline has no normalized execution topology";
            return nullptr;
        }
        pipeline->topology->differentiated = VernonDifferentiatedPipeline{std::move(executable), derivativeGroups};
        return pipeline.release();
    } catch (...) {
        try {
            invocationDiagnostic(context) = "cannot allocate direct GPU autodiff pipeline";
        } catch (...) {
        }
        return nullptr;
    }
}

} // namespace vernon::runtime
