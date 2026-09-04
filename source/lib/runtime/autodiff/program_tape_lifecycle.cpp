#include "program_tape_lifecycle.h"

#include "runtime/autodiff/runtime_gpu_argument_binding.h"
#include "runtime/autodiff/runtime_gpu_replay.h"

#include <algorithm>
#include <limits>
#include <map>

namespace vernon::runtime::ad {
namespace {

bool multiplySize(size_t left, size_t right, size_t &result) {
    if (left && right > std::numeric_limits<size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

bool carrierShape(const program::TargetBinding &binding, size_t bytes, std::vector<uint64_t> &shape,
                  std::vector<int64_t> &strides, std::string &error) {
    gpu::InternalBufferView view;
    if (!gpu::materializeInternalBufferView(binding.shape, binding.elementLayout, bytes, view)) {
        error = "Program tape carrier allocation does not match its element ABI";
        return false;
    }
    shape = std::move(view.shape);
    strides = std::move(view.strides);
    return true;
}

} // namespace

bool allocateProgramTapeState(ProgramInvocationFrame &frame, VernonRuntimeContext &context, ProgramTapeState &state,
                              std::string &error) {
    size_t groupCount = 1;
    size_t workgroupVolume = 1;
    for (uint32_t extent : {state.grid.x, state.grid.y, state.grid.z})
        if (!extent || !multiplySize(groupCount, extent, groupCount))
            return error = "Program tape dispatch group count overflows", false;
    for (uint32_t extent : {state.workgroup.x, state.workgroup.y, state.workgroup.z})
        if (!extent || !multiplySize(workgroupVolume, extent, workgroupVolume))
            return error = "Program tape workgroup volume overflows", false;
    size_t groupTapeBytes = 0;
    size_t tapeBytes = 0;
    size_t segmentBytes = 0;
    if (!gpu::normalizeTapeStride(state.stride, state.stride) ||
        !multiplySize(state.stride, workgroupVolume, groupTapeBytes) ||
        !multiplySize(groupTapeBytes, groupCount, tapeBytes) ||
        !multiplySize(sizeof(gpu::Segment), groupCount, segmentBytes))
        return error = "Program tape carrier size overflows", false;
    const auto allocate = [&](const program::TargetBinding *binding, size_t bytes) {
        if (!binding)
            return true;
        std::vector<uint64_t> shape;
        std::vector<int64_t> strides;
        return carrierShape(*binding, bytes, shape, strides, error) &&
               frame.allocateCarrier(context, state.value, *binding, bytes, std::move(shape), std::move(strides),
                                     error);
    };
    if (!allocate(state.tape, tapeBytes) || !allocate(state.segment, segmentBytes) ||
        !allocate(state.status, sizeof(gpu::BatchSummary)) || !allocate(state.launch, sizeof(uint32_t) * 3))
        return false;
    std::vector<gpu::Segment> segments(groupCount);
    for (size_t group = 0; group < groupCount; ++group)
        if (!gpu::initializeSegment(state.grid, state.workgroup, group, state.stride, workgroupVolume, segments[group]))
            return error = "Program replay segment initialization overflows", false;
    const gpu::BatchSummary clear{};
    const uint32_t launch[3]{state.grid.x, state.grid.y, state.grid.z};
    return (!state.segment || frame.uploadCarrier(state.value, program_plan::TapeCarrier::ReplaySegment,
                                                  segments.data(), segmentBytes, error)) &&
           (!state.status ||
            frame.uploadCarrier(state.value, program_plan::TapeCarrier::ReplayStatus, &clear, sizeof(clear), error)) &&
           (!state.launch ||
            frame.uploadCarrier(state.value, program_plan::TapeCarrier::LaunchMetadata, launch, sizeof(launch), error));
}

bool prepareProgramTapeStates(ProgramInvocationFrame &frame, VernonRuntimeContext &context,
                              const program::Program &execution, const VernonPipelineTopology &topology,
                              const program::Graph &forward, std::vector<ProgramTapeState> &states,
                              std::string &error) {
    std::map<uint32_t, ProgramTapeState> byValue;
    for (const program::Node &node : forward.nodes) {
        const auto stageIndex = topology.stageIndices.find(node.stage);
        if (stageIndex == topology.stageIndices.end())
            return error = "Program tape stage is not resolved", false;
        const VernonResolvedProgramStage &stage = topology.stages[stageIndex->second];
        for (const VernonProgramStageBinding &binding : stage.bindings) {
            if (!binding.target || !binding.target->tapeCarrier)
                continue;
            ProgramTapeState &state = byValue[binding.value];
            state.value = binding.value;
            if (program::executionKind(node) != program::ExecutionKind::Compute)
                return error = "graphics Program nodes cannot carry autodiff tape dispatch metadata", false;
            const program::ComputeOperation &compute = program::computeOperation(node);
            for (size_t axis = 0; axis < 3; ++axis)
                if (compute.workgroups[axis].kind != program::ControlKind::Static ||
                    compute.workgroups[axis].value > std::numeric_limits<uint32_t>::max())
                    return error = "Program tape dispatch requires resolved static workgroups", false;
            state.grid = {static_cast<uint32_t>(compute.workgroups[0].value),
                          static_cast<uint32_t>(compute.workgroups[1].value),
                          static_cast<uint32_t>(compute.workgroups[2].value)};
            state.workgroup = stage.pipeline->workgroupSize;
            switch (*binding.target->tapeCarrier) {
            case program_plan::TapeCarrier::TapeData:
                state.tape = &*binding.target;
                break;
            case program_plan::TapeCarrier::ReplaySegment:
                state.segment = &*binding.target;
                break;
            case program_plan::TapeCarrier::ReplayStatus:
                state.status = &*binding.target;
                break;
            case program_plan::TapeCarrier::LaunchMetadata:
                state.launch = &*binding.target;
                break;
            }
        }
    }
    states.clear();
    states.reserve(byValue.size());
    for (auto &[value, state] : byValue) {
        const uint32_t tapeValue = value;
        const auto plan =
            std::find_if(execution.abi.tapePlans.begin(), execution.abi.tapePlans.end(),
                         [tapeValue](const program::TapePlan &candidate) { return candidate.value == tapeValue; });
        if (plan == execution.abi.tapePlans.end() || !plan->forwardProducer)
            return error = "Program tape producer is absent from the compiler TapePlan", false;
        for (program_plan::TapeCarrier carrier : plan->requiredCarriers) {
            const bool missing = (carrier == program_plan::TapeCarrier::TapeData && !state.tape) ||
                                 (carrier == program_plan::TapeCarrier::ReplaySegment && !state.segment) ||
                                 (carrier == program_plan::TapeCarrier::ReplayStatus && !state.status) ||
                                 (carrier == program_plan::TapeCarrier::LaunchMetadata && !state.launch);
            if (missing)
                return error = "Program tape producer is missing a required typed carrier", false;
        }
        if (!allocateProgramTapeState(frame, context, state, error))
            return false;
        states.push_back(state);
    }
    return true;
}

bool validateProgramTapeStates(ProgramInvocationFrame &frame, std::vector<ProgramTapeState> &states, bool &retry,
                               std::string &error) {
    retry = false;
    for (ProgramTapeState &state : states) {
        if (!state.status)
            continue;
        gpu::BatchSummary summary{};
        if (!frame.downloadCarrier(state.value, program_plan::TapeCarrier::ReplayStatus, &summary, sizeof(summary),
                                   error))
            return false;
        if (summary.status == 0)
            continue;
        if (summary.status != 1 || summary.requiredBytes <= state.stride)
            return error = "Program tape carrier reported an invalid replay status", false;
        state.stride = summary.requiredBytes;
        retry = true;
    }
    return true;
}

} // namespace vernon::runtime::ad
