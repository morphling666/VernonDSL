#include "runtime_dispatch.h"

#include "backend_cpu.h"

#include <memory>
#include <utility>

namespace vernon::runtime {

VernonStatus registerBackendStaticCpuEntry(VernonStringView symbol, VernonCpuEntryPoint entryPoint) {
    return registerStaticCpuEntry(symbol, entryPoint);
}

VernonStageExecutable *loadBackendTypedComputePipeline(VernonRuntimeContext &context, StageBindingPlan stagePlan,
                                                       ReflectedEntry reflection, const void *artifact,
                                                       size_t artifactSize, const std::string &entry,
                                                       VernonCpuEntryPoint cpuEntry,
                                                       const std::vector<NativeResourceSlot> &nativeSlots) {
    auto pipeline = std::make_unique<VernonStageExecutable>();
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
            return nullptr;
        installRuntimeBackendState(*pipeline, state.release());
        return pipeline.release();
    }
    if (!artifact || !artifactSize)
        return nullptr;
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
    if (!resolveBackendPipeline(inputs, pipeline->bindingProjection, *pipeline))
        return nullptr;
    return pipeline.release();
}

} // namespace vernon::runtime
