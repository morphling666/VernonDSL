#include "runtime_cpu_preparation.h"

#include "host_effect_transaction.h"
#include "host_tape_allocator.h"
#include "runtime/backend_cpu.h"
#include "runtime/cpu_workgroup_dispatch.h"
#include "runtime/pipeline_metadata.h"
#include "runtime/runtime_state.h"
#include "runtime/tensor_bridge.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
#include "host_tape_test_hooks.h"
#endif

namespace vernon::runtime::ad::cpu {

bool checkedAddBytes(size_t &total, size_t count, size_t bytes) {
    if (count && bytes > std::numeric_limits<size_t>::max() / count)
        return false;
    const size_t additional = count * bytes;
    if (additional > std::numeric_limits<size_t>::max() - total)
        return false;
    total += additional;
    return true;
}

bool materializeRuntimeSignature(Signature &signature, const VernonAdValueSet &inputs) {
    for (ValueAbi &input : signature.inputs) {
        const VernonAdValue *value = findValue(inputs, input.path);
        if (!value || value->dtype != input.dtype || (value->rank && !value->shape))
            return false;
        input.byteSize = value->size;
        input.logicalShape.assign(value->shape, value->shape + value->rank);
    }
    auto materializeDerivative = [&](ValueAbi &derivative) {
        const auto primal = std::find_if(signature.inputs.begin(), signature.inputs.end(),
                                         [&](const ValueAbi &value) { return value.path == derivative.path; });
        if (primal == signature.inputs.end())
            return false;
        const size_t primalScalarSize = dtypeSize(primal->dtype);
        const size_t derivativeScalarSize = dtypeSize(derivative.dtype);
        if (!primalScalarSize || !derivativeScalarSize || primal->byteSize % primalScalarSize ||
            primal->byteSize / primalScalarSize > SIZE_MAX / derivativeScalarSize)
            return false;
        derivative.byteSize = primal->byteSize / primalScalarSize * derivativeScalarSize;
        derivative.logicalShape = primal->logicalShape;
        return true;
    };
    for (ValueAbi &output : signature.outputs)
        if (!materializeDerivative(output))
            return false;
    for (ValueAbi &cotangent : signature.cotangents)
        if (!materializeDerivative(cotangent))
            return false;
    for (ValueAbi &gradient : signature.gradients)
        if (!materializeDerivative(gradient))
            return false;
    return true;
}

VernonStatus accumulateFloatingBytes(VernonDataType dtype, uint8_t *destination, const uint8_t *source,
                                     size_t byteSize) {
    auto accumulate = [&](auto scalar) {
        using Scalar = decltype(scalar);
        if (byteSize % sizeof(Scalar))
            return false;
        for (size_t offset = 0; offset < byteSize; offset += sizeof(Scalar)) {
            Scalar current;
            Scalar contribution;
            std::memcpy(&current, destination + offset, sizeof(Scalar));
            std::memcpy(&contribution, source + offset, sizeof(Scalar));
            current += contribution;
            std::memcpy(destination + offset, &current, sizeof(Scalar));
        }
        return true;
    };
    if ((dtype == VERNON_DATA_F32 && accumulate(float{})) || (dtype == VERNON_DATA_F64 && accumulate(double{})))
        return VERNON_STATUS_OK;
    return VERNON_STATUS_INVALID_ARGUMENT;
}

VernonStatus accumulateBackwardResults(const HostProfileLayout &layout, const Signature &signature,
                                       const uint8_t *results, const std::vector<size_t> &resultGradientIndices,
                                       uint8_t *privateGradients, const std::vector<size_t> &privateGradientOffsets) {
    if (layout.results.size() != resultGradientIndices.size())
        return VERNON_STATUS_INVALID_ARGUMENT;
    for (size_t index = 0; index < resultGradientIndices.size(); ++index) {
        const size_t gradientIndex = resultGradientIndices[index];
        const HostFrameLeaf &source = layout.results[index];
        if (gradientIndex >= privateGradientOffsets.size() ||
            privateGradientOffsets[gradientIndex] == std::numeric_limits<size_t>::max())
            return VERNON_STATUS_INVALID_ARGUMENT;
        uint8_t *destination = privateGradients + privateGradientOffsets[gradientIndex];
        const uint8_t *contribution = results + source.frameOffset;
        if (VernonStatus accumulateStatus =
                accumulateFloatingBytes(source.value.dtype, destination, contribution, source.value.byteSize);
            accumulateStatus != VERNON_STATUS_OK)
            return accumulateStatus;
    }
    return VERNON_STATUS_OK;
}

VernonStatus accumulateGradientBytes(VernonRuntimeContext &context, const ValueAbi &abi, const uint8_t *source,
                                     std::vector<uint8_t> &destination) {
    const size_t scalarSize = dtypeSize(abi.dtype);
    if (!scalarSize || destination.size() != abi.byteSize)
        return fail(context, "CPU pullback gradient accumulation ABI is inconsistent");
    const VernonStatus status = accumulateFloatingBytes(abi.dtype, destination.data(), source, abi.byteSize);
    return status == VERNON_STATUS_OK
               ? status
               : fail(context, "CPU pullback can only accumulate well-formed floating gradients", status);
}

void commitGradientDestinations(const std::vector<VernonAdValue *> &destinations,
                                const std::vector<std::vector<uint8_t>> &stagedGradients) {
    for (size_t index = 0; index < destinations.size(); ++index)
        if (!stagedGradients[index].empty())
            std::memcpy(destinations[index]->data, stagedGradients[index].data(), stagedGradients[index].size());
}

} // namespace vernon::runtime::ad::cpu
