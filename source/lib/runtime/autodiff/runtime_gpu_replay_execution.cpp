#include "runtime_gpu_replay_execution.h"

#include <algorithm>
#include <limits>

namespace vernon::runtime::ad::gpu {
namespace {

bool checkedMultiply(size_t left, size_t right, size_t &result) {
    if (left && right > std::numeric_limits<size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

bool checkedAdd(size_t left, size_t right, size_t &result) {
    if (right > std::numeric_limits<size_t>::max() - left)
        return false;
    result = left + right;
    return true;
}

bool isAutodiffInternal(const Parameter &parameter) {
    return parameter.autodiffRole == AutodiffResourceRole::Tape ||
           parameter.autodiffRole == AutodiffResourceRole::ReplaySegment ||
           parameter.autodiffRole == AutodiffResourceRole::ReplayStatus ||
           parameter.autodiffRole == AutodiffResourceRole::LaunchMetadata;
}

} // namespace

bool planReplayRestoreCopies(const VernonLoadedPipeline &forward, DeviceValues &working, DeviceValues &retained,
                             std::vector<DeviceBufferCopy> &copies) {
    copies.clear();
    for (const Parameter &parameter : forward.variant.parameters) {
        if (parameter.access == "read" || isAutodiffInternal(parameter))
            continue;
        const auto current = working.find(parameter.name);
        const auto source = retained.find(parameter.name);
        if (current == working.end() || source == retained.end() ||
            current->second.buffer.size() != source->second.buffer.size())
            return false;
        bool wholeView = true;
        std::vector<size_t> linearIndices;
        bool declared = false;
        for (const TensorViewWriteFootprint &footprint : forward.writeFootprints) {
            if (footprint.owner != parameter.name)
                continue;
            declared = true;
            if (footprint.wholeView) {
                wholeView = true;
                linearIndices.clear();
                break;
            }
            wholeView = false;
            if (footprint.indices.size() != current->second.shape.size()) {
                wholeView = true;
                linearIndices.clear();
                break;
            }
            size_t linearIndex = 0;
            for (size_t dimension = 0; dimension < footprint.indices.size(); ++dimension) {
                const uint64_t index = footprint.indices[dimension];
                const uint64_t extent = current->second.shape[dimension];
                if (index >= extent || !checkedMultiply(linearIndex, static_cast<size_t>(extent), linearIndex) ||
                    !checkedAdd(linearIndex, static_cast<size_t>(index), linearIndex)) {
                    wholeView = true;
                    linearIndices.clear();
                    break;
                }
            }
            if (wholeView)
                break;
            linearIndices.push_back(linearIndex);
        }
        if (!declared || wholeView) {
            copies.push_back(
                {source->second.buffer.handle(), current->second.buffer.handle(), 0, 0, current->second.buffer.size()});
            continue;
        }
        size_t elements = 1;
        for (uint64_t extent : current->second.shape)
            if (!checkedMultiply(elements, static_cast<size_t>(extent), elements))
                return false;
        if (!elements || source->second.buffer.size() % elements)
            return false;
        const size_t elementBytes = source->second.buffer.size() / elements;
        std::sort(linearIndices.begin(), linearIndices.end());
        linearIndices.erase(std::unique(linearIndices.begin(), linearIndices.end()), linearIndices.end());
        for (size_t index : linearIndices) {
            if (index >= elements)
                return false;
            copies.push_back({source->second.buffer.handle(), current->second.buffer.handle(), index * elementBytes,
                              index * elementBytes, elementBytes});
        }
    }
    return true;
}

bool appendReplayArguments(VernonRuntimeContext &context, const BindingPlan &bindings, DeviceBuffer &tape,
                           size_t tapeBytes, DeviceBuffer &segment, size_t segmentBytes, DeviceBuffer &status,
                           size_t statusBytes, DeviceBuffer &launch, size_t launchBytes, ReplayArgumentViews &views,
                           std::vector<VernonPipelineArgument> &arguments, std::string &failedParameter) {
    arguments.clear();
    arguments.reserve(bindings.size());
    for (const Binding &binding : bindings) {
        const Parameter &parameter = *binding.parameter;
        failedParameter = parameter.name;
        if (binding.source == BindingSource::ReplaySegment) {
            if (!appendInternalBufferArgument(context, parameter, segment, segmentBytes, views.segment, arguments))
                return false;
            continue;
        }
        if (binding.source == BindingSource::ReplayStatus) {
            if (!appendInternalBufferArgument(context, parameter, status, statusBytes, views.status, arguments))
                return false;
            continue;
        }
        if (binding.source == BindingSource::Tape) {
            if (!appendInternalBufferArgument(context, parameter, tape, tapeBytes, views.tape, arguments))
                return false;
            continue;
        }
        if (binding.source == BindingSource::Launch) {
            if (!appendInternalBufferArgument(context, parameter, launch, launchBytes, views.launch, arguments))
                return false;
            continue;
        }
        if (!appendBindingArgument(binding, arguments))
            return false;
    }
    failedParameter.clear();
    return true;
}

} // namespace vernon::runtime::ad::gpu
