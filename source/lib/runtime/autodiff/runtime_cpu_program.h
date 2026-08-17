#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_CPU_PROGRAM_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_CPU_PROGRAM_H

#include "runtime_cpu_values.h"

#include "host_tape_allocator.h"
#include "runtime/backend_cpu.h"
#include "runtime/pipeline_metadata.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vernon::runtime::ad::cpu {

struct HostFrameLeaf {
    ValueAbi value;
    size_t frameOffset{};
};

struct HostTensorView {
    std::string access;
    std::vector<int64_t> shape;
    ValueLayout elementLayout;
    std::vector<ValueAbi> leaves;
};

struct HostArgument {
    std::string name;
    std::string builtin;
    std::string accumulationOwnership;
    size_t offset{};
    size_t size{};
    std::optional<HostTensorView> tensorView;
    std::vector<HostFrameLeaf> leaves;
};

struct HostProfileLayout {
    size_t argumentsSize{};
    size_t resultsSize{};
    std::optional<size_t> tapeAllocatorOffset;
    std::optional<size_t> tapeRootRegionOffset;
    uint32_t workgroup[3]{1, 1, 1};
    DispatchContract dispatchContract;
    std::vector<HostArgument> arguments;
    std::vector<HostFrameLeaf> results;
};

struct StorageRange {
    uintptr_t begin{};
    uintptr_t end{};
    bool writable{};
};

struct StagedTensorView {
    const HostArgument *argument{};
    size_t elementCount{};
    std::vector<uint64_t> shape;
    std::vector<uint8_t> packed;
    std::vector<uint8_t *> leafShadows;
};

using RuntimeTensorShapes = std::unordered_map<std::string, std::vector<uint64_t>>;
using RetainedPrimalLeaves = std::unordered_map<std::string, std::vector<uint8_t>>;
struct RetainedPrimalTensorView {
    std::vector<uint64_t> shape;
    std::vector<uint8_t> packed;
};
using RetainedPrimalTensorViews = std::unordered_map<std::string, RetainedPrimalTensorView>;
struct RetainedPrimalTensorViewRef {
    const std::vector<uint64_t> *shape{};
    uint8_t *packed{};
};
using RetainedPrimalTensorViewRefs = std::unordered_map<std::string, RetainedPrimalTensorViewRef>;

enum class CpuResidualStorage {
    None,
    Static,
    Dynamic,
};

struct CpuResidualPlan {
    CpuResidualStorage storage{};
    PlanningPolicy policy{};
    size_t staticTapeStride{};
    bool permitsWholeDispatchRetention{};
};

struct CpuAutodiffProgram {
    HostProfileLayout primalLayout;
    HostProfileLayout forwardLayout;
    HostProfileLayout backwardLayout;
    std::shared_ptr<CpuKernelState> primal;
    std::shared_ptr<CpuKernelState> forward;
    std::shared_ptr<CpuKernelState> backward;
    Signature signature;
    std::vector<size_t> resultGradientIndices;
    std::unordered_set<std::string> requiredPrimalTensorOwners;
};

} // namespace vernon::runtime::ad::cpu

#endif
