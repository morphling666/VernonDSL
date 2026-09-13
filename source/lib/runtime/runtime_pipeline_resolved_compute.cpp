#include "runtime_dispatch.h"

#include "backend_cpu.h"

#include <memory>
#include <utility>

namespace vernon::runtime {

RuntimeResult<void> registerBackendStaticCpuEntry(VernonStringView symbol, VernonCpuEntryPoint entryPoint) {
    const VernonStatus status = registerStaticCpuEntry(symbol, entryPoint);
    return status == VERNON_STATUS_OK
               ? RuntimeResult<void>{vernon::ok()}
               : RuntimeResult<void>{
                     vernon::err(vernon::runtimeErrorFromStatus(status, {"register_static_cpu_entry", 0, 0}))};
}

BackendStageLoadResult loadBackendTypedComputePipeline(VernonRuntimeContext &context, StageBindingPlan stagePlan,
                                                       ReflectedEntry reflection, const void *artifact,
                                                       size_t artifactSize, const std::string &entry,
                                                       VernonCpuEntryPoint cpuEntry,
                                                       const std::vector<NativeResourceSlot> &nativeSlots) {
    auto child = RuntimeChildLifecycle::reserve(context.owner);
    if (child.isErr())
        return BackendStageLoadResult{vernon::err(BackendPipelineError::LifecycleUnavailable)};
    auto pipeline = std::make_unique<VernonStageExecutable>();
    pipeline->lifecycle.emplace(std::move(child).value());
    pipeline->context = &context;
    pipeline->bindingProjection = std::move(stagePlan);
    pipeline->workgroupSize = {reflection.workgroup[0], reflection.workgroup[1], reflection.workgroup[2]};
    pipeline->dispatchContract = reflection.dispatchContract;
    pipeline->readFootprints = reflection.readFootprints;
    pipeline->writeFootprints = reflection.writeFootprints;
    if (context.backend == VERNON_RUNTIME_CPU) {
        CpuKernelState kernel;
        kernel.entry = cpuEntry;
        auto state = std::make_unique<CpuPipelineState>();
        if (!prepareCpuComputePipeline(context, std::move(kernel), std::move(reflection), *state))
            return BackendStageLoadResult{vernon::err(BackendPipelineError::CpuPreparationFailed)};
        installRuntimeBackendState(*pipeline, state.release());
        if (pipeline->lifecycle.value().publish().isErr())
            return BackendStageLoadResult{vernon::err(BackendPipelineError::LifecycleUnavailable)};
        return BackendStageLoadResult{vernon::ok(std::move(pipeline))};
    }
    if (!artifact || !artifactSize)
        return BackendStageLoadResult{vernon::err(BackendPipelineError::InvalidArtifact)};
    LoadedStageArtifact stage;
    stage.entry = entry;
    stage.reflected = std::move(reflection);
    stage.nativeSlots = nativeSlots;
    stage.dispatchContract = pipeline->dispatchContract;
    stage.workgroup[0] = pipeline->workgroupSize.x;
    stage.workgroup[1] = pipeline->workgroupSize.y;
    stage.workgroup[2] = pipeline->workgroupSize.z;
    stage.readFootprints = pipeline->readFootprints;
    stage.writeFootprints = pipeline->writeFootprints;
    if (context.backend == VERNON_RUNTIME_CUDA || context.backend == VERNON_RUNTIME_METAL ||
        isOpenGLBackend(context.backend))
        stage.source.assign(static_cast<const char *>(artifact), artifactSize);
    else
        stage.binary.assign(static_cast<const uint8_t *>(artifact),
                            static_cast<const uint8_t *>(artifact) + artifactSize);
    BackendStageBuildInputs inputs;
    inputs.context = &context;
    inputs.artifacts.emplace(stage.entry, std::move(stage));
    auto resolved = resolveBackendPipeline(inputs, pipeline->bindingProjection, *pipeline);
    if (resolved.isErr())
        return BackendStageLoadResult{vernon::err(resolved.error())};
    if (pipeline->lifecycle.value().publish().isErr())
        return BackendStageLoadResult{vernon::err(BackendPipelineError::LifecycleUnavailable)};
    return BackendStageLoadResult{vernon::ok(std::move(pipeline))};
}

} // namespace vernon::runtime
