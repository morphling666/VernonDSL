#ifndef VERNON_RUNTIME_RESOLVED_EXECUTION_PLAN_H
#define VERNON_RUNTIME_RESOLVED_EXECUTION_PLAN_H

#include "VernonResult.hpp"
#include "backend_stage_pipeline.h"
#include "program_execution_manifest.h"
#include "target_binding_plan.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace vernon::runtime::program {

enum class GraphDirection {
    Forward,
    Backward,
    Replay,
};

struct NodeKey {
    GraphDirection graph{GraphDirection::Forward};
    uint32_t node{};

    bool operator<(const NodeKey &other) const {
        return graph != other.graph ? graph < other.graph : node < other.node;
    }
    bool operator==(const NodeKey &other) const { return graph == other.graph && node == other.node; }
    bool operator!=(const NodeKey &other) const { return !(*this == other); }
};

struct NodeEndpointProjection {
    uint32_t value{};
    std::optional<size_t> logicalLeaf;
    TargetBinding target;
};

struct ResolvedComputeControls {
    std::array<ControlComponent, 3> workgroups;
    DispatchMapping dispatchMapping{DispatchMapping::StaticGrid};
};

struct ResolvedGraphicsAttachment {
    uint32_t access{};
    uint32_t storage{};
    uint32_t location{};
    uint32_t aspects{};
};

struct ResolvedGraphicsControls {
    uint32_t renderPassControl{};
    uint32_t drawCommandControl{};
    uint32_t dynamicStateControl{};
    std::vector<ResolvedGraphicsAttachment> colorAttachments;
    std::optional<ResolvedGraphicsAttachment> depthStencilAttachment;
};

using ResolvedOperationControls = std::variant<ResolvedComputeControls, ResolvedGraphicsControls>;

struct ResolvedNodePlan {
    NodeKey key;
    std::shared_ptr<VernonStageExecutable> stage;
    std::vector<NodeEndpointProjection> projections;
    ResolvedOperationControls controls;
};

// The loaded Stage of every Node of a Program executable, which schedule fragments are built from. Nodes that run the
// same physical Stage share it; a Node's projections say which Program Values its Stage parameters read.
struct ResolvedExecutionPlan {
    std::shared_ptr<const ResolvedProgram> resolvedProgram;
    std::vector<vernon::runtime::ValueLayout> boundaryLayoutViews;
    std::vector<shape::DeclaredShape> declaredValueShapes;
    std::vector<std::shared_ptr<VernonStageExecutable>> stageCache;
    std::map<NodeKey, ResolvedNodePlan> nodes;

    const ResolvedNodePlan *node(GraphDirection graph, uint32_t nodeId) const;
};

using ExecutionPlanResult = vernon::Result<void, Diagnostic>;

std::optional<GraphDirection> graphDirection(std::string_view direction);
// Decodes the declared Value shapes and checks every Node against its Stage: one projection per Stage parameter, onto
// an existing Value and leaf, with a static Stage shape that the Value admits and no other Stage contradicts; and a
// Node binds another Node's result only after that Node in its graph, which is the order a schedule runs them in, or
// through its graph's captures.
ExecutionPlanResult completeResolvedExecutionPlan(ResolvedExecutionPlan &plan);

} // namespace vernon::runtime::program

#endif
