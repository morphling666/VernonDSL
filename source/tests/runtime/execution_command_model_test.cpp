#include "execution_graph/execution_command_model.h"
#include "execution_graph/execution_graph_internal.h"

#include <gtest/gtest.h>

namespace {

using namespace vernon::execution;
using namespace vernon::execution::detail;

CommandResourceAccess bufferAccess(uint64_t aliasDomain, uint64_t offset, uint64_t byteSize, AccessMode access) {
    CommandResourceAccess result;
    result.aliasDomain = aliasDomain;
    result.access = access;
    result.kind = ResourceKind::Buffer;
    result.bufferRange = {offset, byteSize};
    return result;
}

CommandNode nodeWithAccess(CommandResourceAccess access, std::vector<uint32_t> predecessors = {}) {
    CommandNode result;
    result.scopeIndices = {0};
    result.predecessors = std::move(predecessors);
    result.accesses = {access};
    return result;
}

class CommandModelComputePass final : public ComputePass {
public:
    using ComputePass::ComputePass;
    void declare() override {}
    VernonRhiStatus execute(ComputeEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }
};

TEST(ExecutionCommandModel, RejectsEmptyAndOverflowingBufferRanges) {
    std::string error;
    CommandResourceAccess empty = bufferAccess(1, 0, 0, AccessMode::Read);
    EXPECT_FALSE(normalizeCommandAccess(empty, error));
    EXPECT_NE(error.find("empty"), std::string::npos);

    error.clear();
    CommandResourceAccess overflow = bufferAccess(1, UINT64_MAX - 3, 8, AccessMode::Read);
    EXPECT_FALSE(normalizeCommandAccess(overflow, error));
    EXPECT_NE(error.find("overflows"), std::string::npos);
}

TEST(ExecutionCommandModel, UsesAliasDomainsAndByteRangesForBufferHazards) {
    const CommandResourceAccess first = bufferAccess(7, 0, 16, AccessMode::Write);
    const CommandResourceAccess disjoint = bufferAccess(7, 16, 16, AccessMode::Read);
    const CommandResourceAccess overlap = bufferAccess(7, 8, 16, AccessMode::Read);
    const CommandResourceAccess otherAlias = bufferAccess(8, 8, 16, AccessMode::Read);

    EXPECT_FALSE(commandAccessesOverlap(first, disjoint));
    EXPECT_TRUE(commandAccessesConflict(first, overlap));
    EXPECT_FALSE(commandAccessesOverlap(first, otherAlias));
}

TEST(ExecutionCommandModel, RequiresDependenciesForConflictingAccesses) {
    CommandDag dag;
    dag.nodes.push_back(nodeWithAccess(bufferAccess(1, 0, 16, AccessMode::Write)));
    dag.nodes.push_back(nodeWithAccess(bufferAccess(1, 0, 16, AccessMode::Read)));
    std::string error;
    EXPECT_FALSE(validateCommandDag(dag, error));
    EXPECT_NE(error.find("no dependency"), std::string::npos);

    dag.nodes[1].predecessors = {0};
    error.clear();
    EXPECT_TRUE(validateCommandDag(dag, error)) << error;
}

TEST(ExecutionCommandModel, AcceptsTransitiveHazardDependencies) {
    CommandDag dag;
    dag.nodes.push_back(nodeWithAccess(bufferAccess(1, 0, 16, AccessMode::Write)));
    dag.nodes.push_back(nodeWithAccess(bufferAccess(2, 0, 16, AccessMode::Write), {0}));
    dag.nodes.push_back(nodeWithAccess(bufferAccess(1, 0, 16, AccessMode::Read), {1}));
    std::string error;
    EXPECT_TRUE(validateCommandDag(dag, error)) << error;
}

TEST(ExecutionCommandModel, RejectsForwardAndDuplicatePredecessors) {
    CommandDag forward;
    forward.nodes.push_back(nodeWithAccess(bufferAccess(1, 0, 16, AccessMode::Read), {0}));
    std::string error;
    EXPECT_FALSE(validateCommandDag(forward, error));
    EXPECT_NE(error.find("topologically ordered"), std::string::npos);

    CommandDag duplicate;
    duplicate.nodes.push_back(nodeWithAccess(bufferAccess(1, 0, 16, AccessMode::Read)));
    duplicate.nodes.push_back(nodeWithAccess(bufferAccess(2, 0, 16, AccessMode::Read), {0, 0}));
    error.clear();
    EXPECT_FALSE(validateCommandDag(duplicate, error));
    EXPECT_NE(error.find("unique"), std::string::npos);
}

TEST(ExecutionCommandModel, AllowsRuntimeNodesWithoutCompiledScopes) {
    CommandDag dag;
    CommandNode transfer;
    transfer.kind = CommandNodeKind::Transfer;
    transfer.accesses = {bufferAccess(1, 0, 16, AccessMode::Write)};
    dag.nodes.push_back(std::move(transfer));
    CommandNode replay;
    replay.kind = CommandNodeKind::Replay;
    replay.predecessors = {0};
    replay.accesses = {bufferAccess(1, 0, 16, AccessMode::Read)};
    dag.nodes.push_back(std::move(replay));
    CommandNode derivative;
    derivative.kind = CommandNodeKind::Derivative;
    derivative.predecessors = {1};
    dag.nodes.push_back(std::move(derivative));
    std::string error;
    EXPECT_TRUE(validateCommandDag(dag, error)) << error;

    CommandDag compute;
    compute.nodes.emplace_back();
    error.clear();
    EXPECT_FALSE(validateCommandDag(compute, error));
    EXPECT_NE(error.find("compiled scope"), std::string::npos);
}

TEST(ExecutionCommandModel, LowersDerivativePassToDerivativeNode) {
    std::vector<std::unique_ptr<ExecutionPass>> passes;
    auto pass = std::make_unique<CommandModelComputePass>("derivative");
    pass->setFlags(PassDerivative | PassNoMerge);
    passes.push_back(std::move(pass));
    CompiledScope scope;
    scope.passIndices = {0};
    CommandDag dag;
    std::string error;
    ASSERT_TRUE(buildCommandDag({}, passes, {scope}, dag, error)) << error;
    ASSERT_EQ(dag.nodes.size(), 1u);
    EXPECT_EQ(dag.nodes.front().kind, CommandNodeKind::Derivative);
}

VernonRhiStatus encodeCommand(void *, VernonRhiCommandEncoder) { return VERNON_RHI_STATUS_OK; }

TEST(ExecutionCommandModel, ComposesOwnedRhiCommandPlans) {
    const auto plan = [](uint64_t aliasDomain, VernonRhiBuffer buffer) {
        RhiCommandExecutionPlan result;
        CommandNode node;
        node.kind = CommandNodeKind::Transfer;
        node.accesses = {bufferAccess(aliasDomain, 0, 16, AccessMode::Write)};
        result.commands.nodes.push_back(std::move(node));
        result.encoders.push_back({encodeCommand, nullptr, nullptr});
        result.bindings.push_back({aliasDomain, ResourceKind::Buffer, buffer, {}});
        result.initialAccesses.push_back(bufferAccess(aliasDomain, 0, 16, AccessMode::ReadWrite));
        return result;
    };
    RhiCommandExecutionPlan combined = plan(1, {1, 1});
    std::string error;
    ASSERT_TRUE(appendRhiCommandExecutionPlan(combined, plan(2, {2, 1}), true, error)) << error;
    ASSERT_EQ(combined.commands.nodes.size(), 2u);
    EXPECT_EQ(combined.commands.nodes[1].predecessors, std::vector<uint32_t>({0}));
    EXPECT_EQ(combined.commands.nodes[1].accesses[0].resource, 1u);
    ASSERT_EQ(combined.initialAccesses.size(), 2u);
    EXPECT_EQ(combined.initialAccesses[0].aliasDomain, 1u);
    EXPECT_EQ(combined.initialAccesses[1].aliasDomain, 2u);
    EXPECT_TRUE(validateRhiCommandExecutionPlan(combined, error)) << error;
}

TEST(ExecutionCommandModel, PreservesDisjointInitialRangesAndRejectsAmbiguousOverlap) {
    const VernonRhiBuffer buffer{1, 1};
    const auto plan = [&](uint64_t accessOffset, uint64_t initialOffset, uint64_t initialBytes) {
        RhiCommandExecutionPlan result;
        CommandNode node;
        node.kind = CommandNodeKind::Transfer;
        node.accesses = {bufferAccess(1, accessOffset, 16, AccessMode::Write)};
        result.commands.nodes.push_back(std::move(node));
        result.encoders.push_back({encodeCommand, nullptr, nullptr});
        result.bindings.push_back({1, ResourceKind::Buffer, buffer, {}});
        result.initialAccesses.push_back(bufferAccess(1, initialOffset, initialBytes, AccessMode::ReadWrite));
        return result;
    };

    RhiCommandExecutionPlan combined = plan(0, 0, 16);
    std::string error;
    ASSERT_TRUE(appendRhiCommandExecutionPlan(combined, plan(32, 32, 16), true, error)) << error;
    ASSERT_EQ(combined.initialAccesses.size(), 2u);
    EXPECT_EQ(combined.initialAccesses[0].bufferRange.offset, 0u);
    EXPECT_EQ(combined.initialAccesses[1].bufferRange.offset, 32u);

    RhiCommandExecutionPlan overlapping = plan(32, 0, 16);
    error.clear();
    EXPECT_FALSE(appendRhiCommandExecutionPlan(overlapping, plan(64, 8, 16), true, error));
    EXPECT_NE(error.find("overlapping initial resource states"), std::string::npos);
}

} // namespace
