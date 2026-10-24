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

VernonRhiStatus encodeCommand(void *, VernonRhiCommandEncoder) { return VERNON_RHI_STATUS_OK; }

TEST(ExecutionCommandModel, ComposesOwnedRhiCommandPlans) {
    const auto plan = [](uint64_t aliasDomain, VernonRhiBuffer buffer) {
        RhiCommandExecutionPlan result;
        CommandNode node;
        node.kind = CommandNodeKind::Transfer;
        node.accesses = {bufferAccess(aliasDomain, 0, 16, AccessMode::Write)};
        result.commands.nodes.push_back(std::move(node));
        result.encoders.push_back({encodeCommand, nullptr});
        result.bindings.push_back({aliasDomain, ResourceKind::Buffer, buffer, {}});
        result.initialAccesses.push_back(bufferAccess(aliasDomain, 0, 16, AccessMode::ReadWrite));
        return result;
    };
    ValidatedRhiCommandPlan combined;
    std::string error;
    ASSERT_TRUE(combined.append(plan(1, {1, 1}), error)) << error;
    ASSERT_TRUE(combined.append(plan(2, {2, 1}), error)) << error;
    const RhiCommandExecutionPlan &composed = combined.plan();
    ASSERT_EQ(composed.commands.nodes.size(), 2u);
    EXPECT_EQ(composed.commands.nodes[1].predecessors, std::vector<uint32_t>({0}));
    EXPECT_EQ(composed.commands.nodes[1].accesses[0].resource, 1u);
    ASSERT_EQ(composed.initialAccesses.size(), 2u);
    EXPECT_EQ(composed.initialAccesses[0].aliasDomain, 1u);
    EXPECT_EQ(composed.initialAccesses[1].aliasDomain, 2u);
    const RhiCommandResourceBinding *second = combined.binding({2, ResourceKind::Buffer});
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second->buffer.index, 2u);
    EXPECT_EQ(combined.binding({2, ResourceKind::Image}), nullptr);
}

TEST(ExecutionCommandModel, RejectedFragmentLeavesTheValidatedPlanUnchanged) {
    const auto fragment = [](std::vector<CommandNode> nodes, std::vector<uint64_t> domains) {
        RhiCommandExecutionPlan result;
        for (CommandNode &node : nodes) {
            node.kind = CommandNodeKind::Transfer;
            node.scopeIndices.clear();
            result.commands.nodes.push_back(std::move(node));
            result.encoders.push_back({encodeCommand, nullptr});
        }
        for (uint64_t domain : domains)
            result.bindings.push_back({domain, ResourceKind::Buffer, {static_cast<uint32_t>(domain), 1}, {}});
        return result;
    };
    ValidatedRhiCommandPlan plan;
    std::string error;
    ASSERT_TRUE(plan.append(fragment({nodeWithAccess(bufferAccess(1, 0, 16, AccessMode::Write))}, {1}), error))
        << error;
    // Both writes of buffer 2 follow the plan's last node but not each other.
    EXPECT_FALSE(plan.append(fragment({nodeWithAccess(bufferAccess(2, 0, 16, AccessMode::Write)),
                                       nodeWithAccess(bufferAccess(2, 0, 16, AccessMode::Write))},
                                      {2}),
                             error));
    EXPECT_NE(error.find("no dependency"), std::string::npos);
    EXPECT_EQ(plan.plan().commands.nodes.size(), 1u);
    EXPECT_EQ(plan.binding({2, ResourceKind::Buffer}), nullptr);

    error.clear();
    ASSERT_TRUE(plan.append(fragment({nodeWithAccess(bufferAccess(2, 0, 16, AccessMode::Write))}, {2}), error))
        << error;
    EXPECT_EQ(plan.plan().commands.nodes.size(), 2u);
}

TEST(ExecutionCommandModel, RejectsAFragmentRebindingAnAliasDomain) {
    const auto fragment = [](VernonRhiBuffer buffer) {
        RhiCommandExecutionPlan result;
        CommandNode node;
        node.kind = CommandNodeKind::Transfer;
        node.accesses = {bufferAccess(1, 0, 16, AccessMode::Read)};
        result.commands.nodes.push_back(std::move(node));
        result.encoders.push_back({encodeCommand, nullptr});
        result.bindings.push_back({1, ResourceKind::Buffer, buffer, {}});
        return result;
    };
    ValidatedRhiCommandPlan plan;
    std::string error;
    ASSERT_TRUE(plan.append(fragment({1, 1}), error)) << error;
    ASSERT_TRUE(plan.append(fragment({1, 1}), error)) << error;
    EXPECT_FALSE(plan.append(fragment({1, 2}), error));
    EXPECT_NE(error.find("different physical resources"), std::string::npos);
}

TEST(ExecutionCommandModel, FollowsDependenciesAcrossManyNodes) {
    CommandDag dag;
    dag.nodes.push_back(nodeWithAccess(bufferAccess(2, 0, 16, AccessMode::Read)));
    for (uint32_t node = 1; node < 130; ++node)
        dag.nodes.push_back(nodeWithAccess(
            bufferAccess(node == 65 ? 1 : 2, 0, 16, node == 65 ? AccessMode::Write : AccessMode::Read), {node - 1}));
    dag.nodes.push_back(nodeWithAccess(bufferAccess(1, 0, 16, AccessMode::Read), {129}));
    std::string error;
    EXPECT_TRUE(validateCommandDag(dag, error)) << error;

    dag.nodes.back().predecessors = {};
    EXPECT_FALSE(validateCommandDag(dag, error));
}

TEST(ExecutionCommandModel, PlacesBarriersOnlyBetweenAccessesOfOneDomain) {
    const auto withState = [](CommandResourceAccess access, VernonRhiResourceState state) {
        access.state = state;
        return access;
    };
    CommandDag dag;
    dag.nodes.push_back(
        nodeWithAccess(withState(bufferAccess(1, 0, 16, AccessMode::Write), VERNON_RHI_STATE_SHADER_WRITE)));
    dag.nodes.push_back(
        nodeWithAccess(withState(bufferAccess(2, 0, 16, AccessMode::Read), VERNON_RHI_STATE_SHADER_READ), {0}));
    dag.nodes.push_back(
        nodeWithAccess(withState(bufferAccess(1, 8, 16, AccessMode::Read), VERNON_RHI_STATE_SHADER_READ), {1}));
    dag.nodes.push_back(
        nodeWithAccess(withState(bufferAccess(1, 32, 16, AccessMode::Read), VERNON_RHI_STATE_SHADER_READ), {2}));
    CommandResourceAccess image = withState(bufferAccess(1, 0, 16, AccessMode::Write), VERNON_RHI_STATE_SHADER_WRITE);
    image.kind = ResourceKind::Image;
    image.imageSubresources = {0, 1, 0, 1, VERNON_RHI_IMAGE_ASPECT_COLOR};
    dag.nodes.push_back(nodeWithAccess(image, {3}));
    dag.nodes.push_back(
        nodeWithAccess(withState(bufferAccess(1, 8, 16, AccessMode::Read), VERNON_RHI_STATE_SHADER_READ), {4}));
    const std::vector<CommandResourceAccess> initial{
        withState(bufferAccess(2, 0, 16, AccessMode::Read), VERNON_RHI_STATE_COMMON)};
    const auto target = [](const void *, const CommandResourceAccess &access, VernonRhiBarrier &barrier,
                           std::string &) {
        if (barrier.is_image)
            barrier.image = {static_cast<uint32_t>(access.aliasDomain), 7};
        else
            barrier.buffer = {static_cast<uint32_t>(access.aliasDomain), 7};
        return true;
    };
    std::vector<std::vector<VernonRhiBarrier>> barriers;
    std::string error;
    ASSERT_TRUE(buildCommandBarriers(dag, initial, target, nullptr, barriers, error)) << error;
    ASSERT_EQ(barriers.size(), 6u);
    EXPECT_TRUE(barriers[0].empty());
    ASSERT_EQ(barriers[1].size(), 1u);
    EXPECT_EQ(barriers[1][0].buffer.index, 2u);
    EXPECT_EQ(barriers[1][0].old_state, VERNON_RHI_STATE_COMMON);
    ASSERT_EQ(barriers[2].size(), 1u);
    EXPECT_EQ(barriers[2][0].buffer.index, 1u);
    EXPECT_EQ(barriers[2][0].old_state, VERNON_RHI_STATE_SHADER_WRITE);
    EXPECT_EQ(barriers[2][0].new_state, VERNON_RHI_STATE_SHADER_READ);
    EXPECT_TRUE(barriers[3].empty());
    EXPECT_TRUE(barriers[4].empty());
    EXPECT_TRUE(barriers[5].empty());
}

TEST(ExecutionCommandModel, PreservesDisjointInitialRangesAndRejectsAmbiguousOverlap) {
    const VernonRhiBuffer buffer{1, 1};
    const auto plan = [&](uint64_t accessOffset, uint64_t initialOffset, uint64_t initialBytes) {
        RhiCommandExecutionPlan result;
        CommandNode node;
        node.kind = CommandNodeKind::Transfer;
        node.accesses = {bufferAccess(1, accessOffset, 16, AccessMode::Write)};
        result.commands.nodes.push_back(std::move(node));
        result.encoders.push_back({encodeCommand, nullptr});
        result.bindings.push_back({1, ResourceKind::Buffer, buffer, {}});
        result.initialAccesses.push_back(bufferAccess(1, initialOffset, initialBytes, AccessMode::ReadWrite));
        return result;
    };

    ValidatedRhiCommandPlan combined;
    std::string error;
    ASSERT_TRUE(combined.append(plan(0, 0, 16), error)) << error;
    ASSERT_TRUE(combined.append(plan(32, 32, 16), error)) << error;
    ASSERT_EQ(combined.plan().initialAccesses.size(), 2u);
    EXPECT_EQ(combined.plan().initialAccesses[0].bufferRange.offset, 0u);
    EXPECT_EQ(combined.plan().initialAccesses[1].bufferRange.offset, 32u);

    ValidatedRhiCommandPlan overlapping;
    ASSERT_TRUE(overlapping.append(plan(32, 0, 16), error)) << error;
    EXPECT_FALSE(overlapping.append(plan(64, 8, 16), error));
    EXPECT_NE(error.find("overlapping initial resource states"), std::string::npos);
}

struct RecordedCommand {
    std::vector<int> *events{};
    int id{};
    VernonRhiStatus status{VERNON_RHI_STATUS_OK};
};

VernonRhiStatus recordCommand(void *opaque, VernonRhiCommandEncoder) {
    const auto &command = *static_cast<const RecordedCommand *>(opaque);
    command.events->push_back(command.id);
    return command.status;
}

RhiCommandExecutionPlan finishedFragment(RecordedCommand &node, std::vector<RecordedCommand *> finishers) {
    RhiCommandExecutionPlan result;
    CommandNode transfer;
    transfer.kind = CommandNodeKind::Transfer;
    transfer.queue = CommandQueueClass::Transfer;
    result.commands.nodes.push_back(std::move(transfer));
    result.encoders.push_back({recordCommand, &node});
    for (RecordedCommand *finisher : finishers)
        result.finishers.push_back({recordCommand, finisher});
    return result;
}

TEST(ExecutionCommandModel, ComposesEachFinisherOnceInFirstUseOrder) {
    std::vector<int> events;
    RecordedCommand node{&events, 1};
    RecordedCommand first{&events, 10};
    RecordedCommand second{&events, 20};
    ValidatedRhiCommandPlan plan;
    std::string error;
    ASSERT_TRUE(plan.append(finishedFragment(node, {&first}), error)) << error;
    ASSERT_TRUE(plan.append(finishedFragment(node, {&second, &first}), error)) << error;
    ASSERT_TRUE(plan.append(finishedFragment(node, {&second}), error)) << error;
    const std::vector<RhiCommandFinisher> &finishers = plan.plan().finishers;
    ASSERT_EQ(finishers.size(), 2u);
    EXPECT_EQ(finishers[0].context, &first);
    EXPECT_EQ(finishers[1].context, &second);

    RhiCommandExecutionPlan invalid = finishedFragment(node, {});
    invalid.finishers.push_back({nullptr, &first});
    EXPECT_FALSE(plan.append(std::move(invalid), error));
    EXPECT_NE(error.find("invalid finisher"), std::string::npos);
    EXPECT_EQ(plan.plan().commands.nodes.size(), 3u);
}

VernonRhiDevice createAnyRhiDevice() {
    for (const VernonRhiBackend backend :
         {VERNON_RHI_BACKEND_VULKAN, VERNON_RHI_BACKEND_DIRECTX12, VERNON_RHI_BACKEND_METAL, VERNON_RHI_BACKEND_CUDA}) {
        VernonRhiOwnedDeviceDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.backend = backend;
        const VernonRhiDevice device = vernonRhiCreateDevice(&descriptor);
        if (device.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            return device;
    }
    return {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
}

TEST(ExecutionCommandModel, RunsFinishersAfterTheLastNodeAlsoWhenANodeFails) {
    const VernonRhiDevice device = createAnyRhiDevice();
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        GTEST_SKIP() << "no RHI device is available";
    for (const bool failing : {false, true}) {
        std::vector<int> events;
        RecordedCommand first{&events, 1};
        RecordedCommand second{&events, 2, failing ? VERNON_RHI_STATUS_INTERNAL_ERROR : VERNON_RHI_STATUS_OK};
        RecordedCommand third{&events, 3};
        RecordedCommand firstFinisher{&events, 10};
        RecordedCommand secondFinisher{&events, 20};
        ValidatedRhiCommandPlan plan;
        std::string error;
        ASSERT_TRUE(plan.append(finishedFragment(first, {&firstFinisher}), error)) << error;
        ASSERT_TRUE(plan.append(finishedFragment(second, {&secondFinisher, &firstFinisher}), error)) << error;
        ASSERT_TRUE(plan.append(finishedFragment(third, {}), error)) << error;
        VernonRhiCompletion completion{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
        const VernonRhiStatus status = submitRhiCommandPlan(device, VERNON_RHI_QUEUE_TRANSFER, plan, completion);
        if (failing) {
            EXPECT_NE(status, VERNON_RHI_STATUS_OK);
            EXPECT_EQ(events, std::vector<int>({1, 2, 10, 20}));
        } else {
            ASSERT_EQ(status, VERNON_RHI_STATUS_OK);
            EXPECT_EQ(events, std::vector<int>({1, 2, 3, 10, 20}));
            EXPECT_EQ(vernonRhiCompletionWait(device, completion), VERNON_RHI_STATUS_OK);
            EXPECT_EQ(vernonRhiDeviceDestroyCompletion(device, completion), VERNON_RHI_STATUS_OK);
        }
    }
    vernonRhiDestroyDevice(device);
}

} // namespace
