#ifndef VERNON_RUNTIME_RUNTIME_DISPATCH_H
#define VERNON_RUNTIME_RUNTIME_DISPATCH_H

#include "VernonRuntimeCore.h"
#include "backend_stage_pipeline.h"
#include "compute_launch_planner.h"
#include "graphics_invocation_planner.h"
#include "pipeline_groups.h"
#include "program_execution/invocation_outcome.h"
#include "program_execution/materialized_node_frame.h"
#include "program_execution/program_invocation_state.h"
#include "program_execution/submission_record.h"
#include "program_execution_manifest.h"
#include "resolved_stage_invocation.h"
#include "runtime_state.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

struct VernonProgramExecutable;
struct VernonProgramBundle;
struct VernonRuntimeContext;

namespace vernon::runtime {

template <typename T> using RuntimeResult = vernon::Result<T, vernon::RuntimeError>;

VernonStatus executePipelineProgramGraph(VernonRuntimeContext &context, const program::ResolvedExecutionPlan &execution,
                                         const program::Graph &graph, program_execution::ProgramInvocationState &frame,
                                         const program_execution::ResolvePhysicalEndpoint &resolvePhysicalEndpoint,
                                         program_execution::SubmissionRecord &submission,
                                         program_execution::SubmissionState &step);

bool isOpenGLBackend(VernonRuntimeBackend backend);
RuntimeResult<void> probeBackend(VernonRuntimeBackend backend, std::string &diagnostic);
RuntimeResult<void> initializeBackend(VernonRuntimeContext &context, uint32_t deviceIndex);
RuntimeResult<void> initializeBackendForRhiDevice(VernonRuntimeContext &context, VernonRhiDevice device);
void destroyBackend(VernonRuntimeContext &context);
void fillBackendCapabilities(const VernonRuntimeContext &context, VernonRuntimeCapabilities &capabilities);
RuntimeResult<void> validateRuntimeRequirements(VernonRuntimeContext &context, const RuntimeRequirements &requirements);

RuntimeResult<void> registerBackendStaticCpuEntries(const VernonStaticCpuEntry *entries, size_t count);
RuntimeResult<void> unregisterBackendStaticCpuEntries(const VernonStaticCpuEntry *entries, size_t count);
using BackendStageLoadResult = vernon::Result<std::unique_ptr<VernonStageExecutable>, BackendPipelineFailure>;
BackendStageLoadResult loadBackendTypedComputePipeline(VernonRuntimeContext &context, StageBindingPlan stagePlan,
                                                       ReflectedEntry reflection, const void *artifact,
                                                       size_t artifactSize, const std::string &entry,
                                                       VernonCpuEntryPoint cpuEntry,
                                                       const std::vector<NativeResourceSlot> &nativeSlots);
BackendPipelineResult resolveBackendPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                             VernonStageExecutable &pipeline);
void destroyBackendPipeline(VernonStageExecutable &pipeline);
// Records one draw of `stream` on `packets`, inside its rendering scope.
RuntimeResult<void> invokeBackendPipeline(VernonStageExecutable &pipeline, VernonRuntimeCorePackets *packets,
                                          const GraphicsCommand &command, GraphicsStream &stream);
class CommandStream;

// A packet batch with the payload words it was recorded with, which any command stream of its provider encodes again.
class StoredPacketBatch {
public:
    StoredPacketBatch() = default;
    StoredPacketBatch(const StoredPacketBatch &) = delete;
    StoredPacketBatch &operator=(const StoredPacketBatch &) = delete;
    ~StoredPacketBatch() { vernonRuntimeCorePacketBatchDestroy(batch_); }

    explicit operator bool() const { return batch_ != nullptr; }
    const std::vector<uint32_t> &payload() const { return payload_; }

private:
    friend class CommandStream;

    VernonRuntimeCorePacketBatch *batch_{};
    std::vector<uint32_t> payload_;
};

// Dispatches on `stream`, whose encoder is the launch's. `owners`, when given, holds the owner of each Stage
// parameter's argument; otherwise the launch owns them all.
RuntimeResult<void> invokeBackendComputePipeline(VernonStageExecutable &pipeline, CommandStream &stream,
                                                 const PlannedComputeLaunch &plan, const ArgumentOwner *owners);

RuntimeResult<VernonRuntimeProviderResourceReference>
referenceBackendRhiBuffer(VernonRuntimeContext &context, VernonRhiBuffer buffer, uint64_t offset, uint64_t size);
RuntimeResult<VernonRhiBuffer>
resolveBackendRhiBufferReference(VernonRuntimeContext &context,
                                 const VernonRuntimeProviderResourceReference &reference);
RuntimeResult<VernonRuntimeProviderResourceReference> referenceBackendRhiImageView(VernonRuntimeContext &context,
                                                                                   VernonRhiImageView view);
RuntimeResult<VernonRuntimeProviderResourceReference> referenceBackendRhiSampler(VernonRuntimeContext &context,
                                                                                 VernonRhiSampler sampler);
RuntimeResult<VernonRuntimeProviderImageDescription>
describeBackendImage(VernonRuntimeContext &context, VernonRuntimeProviderResourceReference resource);
RuntimeResult<VernonRuntimeProviderObject> referenceBackendCommandEncoder(VernonRuntimeContext &context,
                                                                          VernonRhiCommandEncoder encoder);
// The command stream of one provider command encoder: its RuntimeCore commands, the commands recorded since the last
// flush and the transient groups they bind, which live until the stream ends. Ended on destruction when end() was not
// called, without encoding what was recorded since the last flush.
class CommandStream {
public:
    CommandStream() = default;
    CommandStream(CommandStream &&other) noexcept;
    CommandStream &operator=(CommandStream &&other) noexcept;
    CommandStream(const CommandStream &) = delete;
    CommandStream &operator=(const CommandStream &) = delete;
    ~CommandStream();

    explicit operator bool() const { return commands_ != nullptr; }
    VernonRuntimeCorePackets *packets() const { return packets_; }
    VernonRuntimeProviderObject encoder() const { return encoder_; }
    GroupStore &transient() const { return *transient_; }
    RuntimeResult<void> beginRendering(const VernonRuntimeProviderRenderingDescriptor &descriptor);
    RuntimeResult<void> endRendering();
    // Encodes the commands recorded since the last flush as one packet batch. Called outside a rendering scope before
    // other commands are recorded on the encoder, such as at the end of each RHI command encode callback.
    RuntimeResult<void> flush();
    // Flushes into the empty stored, which keeps the batch. The batch must bind no transient group.
    RuntimeResult<void> flush(StoredPacketBatch &stored);
    // Encodes stored with `payload`, a patched copy of its payload, or its own, after flushing; called where flush()
    // could be.
    RuntimeResult<void> encode(const StoredPacketBatch &stored, const std::vector<uint32_t> &payload);
    // Called before the next command once other commands were recorded on the encoder, outside a rendering scope.
    RuntimeResult<void> resume();
    // Writes the inline values of updatable `group` for the batches encoded after the call, including the commands
    // recorded since the last flush; see vernonRuntimeCoreUpdateGroup.
    RuntimeResult<void> updateGroup(VernonRuntimeCoreGroup *group, const VernonRuntimeProviderBindingValue *values,
                                    size_t valueCount);
    // Flushes, then ends the commands.
    RuntimeResult<void> end();

private:
    friend RuntimeResult<CommandStream> beginCommandStream(VernonRuntimeContext &, VernonRuntimeProviderObject);

    VernonRuntimeContext *context_{};
    VernonRuntimeCoreCommands *commands_{};
    VernonRuntimeCorePackets *packets_{};
    VernonRuntimeProviderObject encoder_{};
    std::unique_ptr<GroupStore> transient_;
};

// On the provider of `context`: the CPU provider, or the RHI adapter's.
RuntimeResult<CommandStream> beginCommandStream(VernonRuntimeContext &context, VernonRuntimeProviderObject encoder);
// Whether the provider of `context` writes updatable groups.
bool backendUpdatesGroups(VernonRuntimeContext &context);

// The command stream that the nodes of one RHI command plan share on the plan's encoder. Each node acquires it: the
// first begins it, and later ones resume it, because the plan records barriers, copies and scope commands between
// nodes. The plan's finisher ends it.
class EncoderCommandStream {
public:
    explicit EncoderCommandStream(VernonRuntimeContext &context) : context_(context) {}

    RuntimeResult<CommandStream *> acquire(VernonRhiCommandEncoder encoder);
    // An RhiCommandListEncoder for the plan's finishers.
    static VernonRhiStatus finish(void *stream, VernonRhiCommandEncoder encoder);

private:
    VernonRuntimeContext &context_;
    VernonRhiCommandEncoder encoder_{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    CommandStream stream_;
};

} // namespace vernon::runtime

#endif
