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

bool materializeTensorViewShape(const HostArgument &argument, const VernonAdValueSet &inputs,
                                std::vector<uint64_t> &shape) {
    const HostTensorView &view = *argument.tensorView;
    for (size_t leafIndex = 0; leafIndex < view.leaves.size(); ++leafIndex) {
        const ValueAbi &leaf = view.leaves[leafIndex];
        const ValueLeaf &layoutLeaf = view.elementLayout.leaves[leafIndex];
        const VernonAdValue *value = findValue(inputs, leaf.path);
        if (!value || value->dtype != leaf.dtype || value->rank != view.shape.size() + layoutLeaf.shape.size() ||
            (value->rank && !value->shape))
            return false;
        std::vector<uint64_t> candidate(value->shape, value->shape + view.shape.size());
        for (size_t dimension = 0; dimension < view.shape.size(); ++dimension)
            if (view.shape[dimension] >= 0 && candidate[dimension] != static_cast<uint64_t>(view.shape[dimension]))
                return false;
        for (size_t dimension = 0; dimension < layoutLeaf.shape.size(); ++dimension)
            if (value->shape[view.shape.size() + dimension] != layoutLeaf.shape[dimension])
                return false;
        size_t scalarCount = 1;
        for (uint64_t extent : candidate) {
            if (scalarCount && extent > SIZE_MAX / scalarCount)
                return false;
            scalarCount *= static_cast<size_t>(extent);
        }
        const size_t scalarSize = dtypeSize(leaf.dtype);
        if (!scalarSize || (scalarCount && layoutLeaf.scalarCount > SIZE_MAX / scalarCount))
            return false;
        const size_t leafScalarCount = scalarCount * layoutLeaf.scalarCount;
        if ((leafScalarCount && scalarSize > SIZE_MAX / leafScalarCount) || value->size != leafScalarCount * scalarSize)
            return false;
        if (shape.empty())
            shape = std::move(candidate);
        else if (shape != candidate)
            return false;
    }
    return !shape.empty();
}

bool appendStorageRange(const void *data, size_t size, bool writable, std::vector<StorageRange> &ranges) {
    const uintptr_t begin = reinterpret_cast<uintptr_t>(data);
    if (size > std::numeric_limits<uintptr_t>::max() - begin)
        return false;
    const StorageRange range{begin, begin + size, writable};
    for (const StorageRange &existing : ranges)
        if (hostByteRangesHaveWritableOverlap(reinterpret_cast<const void *>(range.begin), range.end - range.begin,
                                              range.writable, reinterpret_cast<const void *>(existing.begin),
                                              existing.end - existing.begin, existing.writable))
            return false;
    ranges.push_back(range);
    return true;
}

VernonStatus stageForwardInputs(VernonRuntimeContext &context, const HostProfileLayout &layout,
                                const VernonAdValueSet &inputs, std::vector<uint8_t> &arguments,
                                HostEffectTransaction &transaction, std::vector<StorageRange> &storageRanges,
                                std::vector<StagedTensorView> &tensorViews, bool preserveWriteOnly) {
    tensorViews.reserve(layout.arguments.size());
    for (const HostArgument &argument : layout.arguments) {
        if (!argument.builtin.empty())
            continue;
        if (argument.tensorView) {
            const HostTensorView &view = *argument.tensorView;
            if (view.leaves.size() != view.elementLayout.leaves.size())
                return fail(context, "autodiff forward TensorView has no canonical element layout");
            const bool writable = view.access != "read";
            const bool preserve = preserveWriteOnly || view.access != "write";
            StagedTensorView staged;
            staged.argument = &argument;
            staged.elementCount = 1;
            if (!materializeTensorViewShape(argument, inputs, staged.shape))
                return fail(context, "autodiff forward TensorView inputs do not match profile reflection");
            for (uint64_t extent : staged.shape) {
                if (staged.elementCount && extent > std::numeric_limits<size_t>::max() / staged.elementCount)
                    return fail(context, "autodiff forward TensorView element count overflows");
                staged.elementCount *= static_cast<size_t>(extent);
            }
            if (!view.elementLayout.byteSize ||
                staged.elementCount > std::numeric_limits<size_t>::max() / view.elementLayout.byteSize)
                return fail(context, "autodiff forward TensorView byte size overflows");
            try {
                staged.packed.resize(staged.elementCount * view.elementLayout.byteSize);
                staged.leafShadows.reserve(view.leaves.size());
            } catch (const std::bad_alloc &) {
                return fail(context, "cannot allocate native CPU autodiff TensorView shadow",
                            VERNON_STATUS_INTERNAL_ERROR);
            } catch (const std::length_error &) {
                return fail(context, "cannot allocate native CPU autodiff TensorView shadow",
                            VERNON_STATUS_INTERNAL_ERROR);
            }
            for (size_t leafIndex = 0; leafIndex < view.leaves.size(); ++leafIndex) {
                const ValueAbi &leaf = view.leaves[leafIndex];
                const VernonAdValue *value = findValue(inputs, leaf.path);
                if (!value)
                    return fail(context, "autodiff forward TensorView input '" + leaf.path +
                                             "' does not match profile reflection");
                if (!appendStorageRange(value->data, value->size, writable, storageRanges))
                    return fail(context, "observable native CPU autodiff Storage/output ranges overlap or overflow");
                auto *source = static_cast<const uint8_t *>(value->data);
                auto *shadow =
                    writable
                        ? static_cast<uint8_t *>(transaction.stageStorage(value->data, value->size, preserve, true))
                        : const_cast<uint8_t *>(source);
                if (value->size && !shadow)
                    return fail(context, "cannot allocate native CPU autodiff Storage shadow",
                                VERNON_STATUS_INTERNAL_ERROR);
                staged.leafShadows.push_back(shadow);
                if (!preserve)
                    continue;
                const ValueLeaf &layoutLeaf = view.elementLayout.leaves[leafIndex];
                const size_t scalarBytes = dtypeSize(leaf.dtype);
                if (!scalarBytes || layoutLeaf.scalarCount > SIZE_MAX / scalarBytes)
                    return fail(context, "autodiff TensorView leaf byte size overflows");
                const size_t leafBytes = layoutLeaf.scalarCount * scalarBytes;
                if (layoutLeaf.byteOffset > view.elementLayout.byteSize ||
                    leafBytes > view.elementLayout.byteSize - layoutLeaf.byteOffset)
                    return fail(context, "autodiff TensorView leaf exceeds its canonical element layout");
                const uint8_t *packedSource = writable ? shadow : source;
                for (size_t element = 0; element < staged.elementCount; ++element)
                    std::memcpy(staged.packed.data() + element * view.elementLayout.byteSize + layoutLeaf.byteOffset,
                                packedSource + element * leafBytes, leafBytes);
            }
            tensorViews.push_back(std::move(staged));
            if (!writeTensorViewDescriptor(argument, tensorViews.back().shape, tensorViews.back().packed.data(),
                                           arguments.data()))
                return fail(context, "autodiff forward TensorView descriptor overflows");
            continue;
        }
        for (const HostFrameLeaf &leaf : argument.leaves) {
            const VernonAdValue *value = findValue(inputs, leaf.value.path);
            if (!value || !valueMatches(*value, leaf.value))
                return fail(context,
                            "autodiff forward input '" + leaf.value.path +
                                "' does not match profile reflection (expected " + std::to_string(leaf.value.byteSize) +
                                " bytes at rank " + std::to_string(leaf.value.logicalShape.size()) + ", received " +
                                (value ? std::to_string(value->size) + " bytes at rank " + std::to_string(value->rank)
                                       : std::string("no value")) +
                                ")");
            std::memcpy(arguments.data() + leaf.frameOffset, value->data, value->size);
        }
    }
    return VERNON_STATUS_OK;
}

void restoreReplayReadWriteShadows(std::vector<StagedTensorView> &tensorViews) {
    for (StagedTensorView &staged : tensorViews) {
        if (!staged.argument || !staged.argument->tensorView || staged.argument->tensorView->access != "read_write")
            continue;
        const HostTensorView &view = *staged.argument->tensorView;
        for (size_t leafIndex = 0; leafIndex < view.leaves.size(); ++leafIndex) {
            const ValueLeaf &layoutLeaf = view.elementLayout.leaves[leafIndex];
            const size_t leafBytes = layoutLeaf.scalarCount * dtypeSize(view.leaves[leafIndex].dtype);
            const uint8_t *source = staged.leafShadows[leafIndex];
            for (size_t element = 0; element < staged.elementCount; ++element)
                std::memcpy(staged.packed.data() + element * view.elementLayout.byteSize + layoutLeaf.byteOffset,
                            source + element * leafBytes, leafBytes);
        }
    }
}

VernonStatus flushStagedTensorViews(VernonRuntimeContext &context, const std::vector<StagedTensorView> &tensorViews) {
    for (const StagedTensorView &staged : tensorViews) {
        if (!staged.argument || !staged.argument->tensorView || staged.argument->tensorView->access == "read" ||
            staged.argument->tensorView->leaves.size() != staged.leafShadows.size())
            continue;
        const HostTensorView &view = *staged.argument->tensorView;
        const ValueLayout &layout = view.elementLayout;
        for (size_t leafIndex = 0; leafIndex < view.leaves.size(); ++leafIndex) {
            const ValueAbi &leaf = view.leaves[leafIndex];
            const ValueLeaf &layoutLeaf = layout.leaves[leafIndex];
            const size_t scalarBytes = dtypeSize(leaf.dtype);
            if (!scalarBytes || layoutLeaf.scalarCount > SIZE_MAX / scalarBytes)
                return fail(context, "autodiff TensorView leaf byte size overflows");
            const size_t leafBytes = layoutLeaf.scalarCount * scalarBytes;
            for (size_t element = 0; element < staged.elementCount; ++element)
                std::memcpy(staged.leafShadows[leafIndex] + element * leafBytes,
                            staged.packed.data() + element * layout.byteSize + layoutLeaf.byteOffset, leafBytes);
        }
    }
    return VERNON_STATUS_OK;
}

bool writeInvocationBuiltin(const HostArgument &argument, const CpuLaneCoordinates &coordinates, uint8_t *arguments) {
    if ((argument.builtin != "global_invocation_id" && argument.builtin != "local_invocation_id" &&
         argument.builtin != "workgroup_id") ||
        argument.leaves.size() != 1 || argument.leaves.front().value.dtype != VERNON_DATA_U32 ||
        argument.leaves.front().value.logicalShape != std::vector<uint64_t>{3})
        return false;
    const uint32_t *value = argument.builtin == "global_invocation_id"
                                ? coordinates.global
                                : (argument.builtin == "local_invocation_id" ? coordinates.local : coordinates.group);
    std::memcpy(arguments + argument.leaves.front().frameOffset, value, sizeof(uint32_t) * 3);
    return true;
}

VernonStatus fail(VernonRuntimeContext &context, std::string message, VernonStatus status) {
    invocationDiagnostic(context) = std::move(message);
    return status;
}

const char *allocatorFailure(VernonAdTapeAllocatorStatus status) {
    switch (status) {
    case VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED:
        return "capacity exhausted";
    case VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW:
        return "arithmetic overflow";
    case VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE:
        return "host allocation failed";
    case VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE:
        return "invalid state";
    case VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI:
        return "invalid ABI";
    case VERNON_AD_TAPE_ALLOCATOR_OK:
        return "no failure";
    }
    return "unknown failure";
}

VernonStatus prepareGradientDestinations(VernonRuntimeContext &context, const Signature &signature,
                                         VernonAdValueSet &gradients, std::vector<VernonAdValue *> &destinations,
                                         std::vector<std::vector<uint8_t>> &stagedGradients) {
    if (gradients.value_count != signature.gradients.size())
        return fail(context, "invalid pullback invocation");
    destinations.reserve(signature.gradients.size());
    stagedGradients.reserve(signature.gradients.size());
    for (const ValueAbi &expected : signature.gradients) {
        VernonAdValue *gradient = findValue(gradients, expected.path);
        if (!gradient || !valueMatches(*gradient, expected))
            return fail(context, "gradient output does not match backward reflection");
        destinations.push_back(gradient);
    }
    try {
        for (const ValueAbi &expected : signature.gradients)
            stagedGradients.emplace_back(expected.byteSize, uint8_t{0});
    } catch (const std::bad_alloc &) {
        return fail(context, "cannot allocate staged pullback gradients", VERNON_STATUS_INTERNAL_ERROR);
    } catch (const std::length_error &) {
        return fail(context, "cannot allocate staged pullback gradients", VERNON_STATUS_INTERNAL_ERROR);
    }
    return VERNON_STATUS_OK;
}

bool writeTensorViewDescriptor(const HostArgument &argument, const std::vector<uint64_t> &shape, void *data,
                               uint8_t *arguments) {
    if (!argument.tensorView || shape.size() != argument.tensorView->shape.size() ||
        argument.size != sizeof(uint64_t) * (2 + 2 * shape.size()))
        return false;
    uint8_t *descriptor = arguments + argument.offset;
    const uintptr_t pointer = reinterpret_cast<uintptr_t>(data);
    const uint64_t zero = 0;
    std::memcpy(descriptor, &pointer, sizeof(pointer));
    std::memcpy(descriptor + sizeof(uint64_t), &zero, sizeof(zero));
    uint64_t stride = 1;
    for (size_t dimension = shape.size(); dimension-- > 0;) {
        if (stride && shape[dimension] > UINT64_MAX / stride)
            return false;
        std::memcpy(descriptor + sizeof(uint64_t) * (2 + dimension), &shape[dimension], sizeof(uint64_t));
        std::memcpy(descriptor + sizeof(uint64_t) * (2 + shape.size() + dimension), &stride, sizeof(uint64_t));
        stride *= shape[dimension];
    }
    return true;
}

bool tensorViewDescriptorShape(const HostArgument &argument, const std::vector<uint64_t> &logicalShape,
                               std::vector<uint64_t> &descriptorShape) {
    if (!argument.tensorView)
        return false;
    const size_t rank = argument.tensorView->shape.size();
    if (logicalShape.size() < rank)
        return false;
    descriptorShape.assign(logicalShape.begin(), logicalShape.begin() + rank);
    for (size_t dimension = 0; dimension < rank; ++dimension)
        if (argument.tensorView->shape[dimension] >= 0 &&
            descriptorShape[dimension] != static_cast<uint64_t>(argument.tensorView->shape[dimension]))
            return false;
    return std::any_of(argument.tensorView->elementLayout.leaves.begin(),
                       argument.tensorView->elementLayout.leaves.end(), [&](const ValueLeaf &leaf) {
                           return leaf.shape.size() == logicalShape.size() - rank &&
                                  std::equal(leaf.shape.begin(), leaf.shape.end(), logicalShape.begin() + rank);
                       });
}

bool isShapeSource(const HostArgument &argument) { return argument.name.rfind("shape.", 0) == 0; }
bool isPrimalSource(const HostArgument &argument) { return argument.name.rfind("primal.", 0) == 0; }

std::string tensorOwnerName(const HostArgument &argument) {
    std::string name = isShapeSource(argument) ? argument.name.substr(6) : argument.name;
    const size_t separator = name.find('.');
    if (separator != std::string::npos)
        name.resize(separator);
    return name;
}

} // namespace vernon::runtime::ad::cpu
