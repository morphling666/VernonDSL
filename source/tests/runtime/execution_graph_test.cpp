#include "VernonExecutionGraph.h"

#include <gtest/gtest.h>

namespace {

using namespace vernon::execution;

class TestComputePass final : public ComputePass {
public:
    TestComputePass(std::string name, GraphResource readResource, GraphResource writeResource)
        : ComputePass(std::move(name)), readResource_(readResource), writeResource_(writeResource) {}

    void declare() override {
        if (readResource_.id != UINT32_MAX)
            read(readResource_);
        if (writeResource_.id != UINT32_MAX)
            write(writeResource_);
    }

    VernonRhiStatus execute(ComputeEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }

private:
    GraphResource readResource_;
    GraphResource writeResource_;
};

class TestRenderPass final : public RenderPass {
public:
    TestRenderPass(std::string name, GraphImage target, VernonRhiLoadOperation load = VERNON_RHI_LOAD_PRESERVE,
                   VernonRhiStoreOperation store = VERNON_RHI_STORE_PRESERVE)
        : RenderPass(std::move(name)), target_(target), load_(load), store_(store) {}

    void declare() override {
        ColorAttachmentUse attachment{};
        attachment.image = target_;
        attachment.load = load_;
        attachment.store = store_;
        color(0, attachment);
    }

    VernonRhiStatus execute(GraphicsEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }

private:
    GraphImage target_;
    VernonRhiLoadOperation load_;
    VernonRhiStoreOperation store_;
};

class TestRenderReadPass final : public RenderPass {
public:
    TestRenderReadPass(std::string name, GraphImage target, GraphResource input)
        : RenderPass(std::move(name)), target_(target), input_(input) {}

    void declare() override {
        read(input_, VERNON_RHI_STATE_SHADER_READ, VERNON_RHI_STAGE_FRAGMENT);
        ColorAttachmentUse attachment{};
        attachment.image = target_;
        color(0, attachment);
    }

    VernonRhiStatus execute(GraphicsEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }

private:
    GraphImage target_;
    GraphResource input_;
};

class TestRenderResourcePass final : public RenderPass {
public:
    TestRenderResourcePass(std::string name, GraphImage target, GraphResource resource, AccessMode access)
        : RenderPass(std::move(name)), target_(target), resource_(resource), access_(access) {}

    void declare() override {
        if (access_ == AccessMode::Read)
            read(resource_);
        else
            write(resource_);
        ColorAttachmentUse attachment{};
        attachment.image = target_;
        color(0, attachment);
    }

    VernonRhiStatus execute(GraphicsEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }

private:
    GraphImage target_;
    GraphResource resource_;
    AccessMode access_;
};

class TestMultiRenderPass final : public RenderPass {
public:
    TestMultiRenderPass(std::string name, GraphImage first, GraphImage second)
        : RenderPass(std::move(name)), first_(first), second_(second) {}

    void declare() override {
        ColorAttachmentUse first{};
        first.image = first_;
        color(0, first);
        ColorAttachmentUse second{};
        second.image = second_;
        color(1, second);
    }

    VernonRhiStatus execute(GraphicsEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }

private:
    GraphImage first_;
    GraphImage second_;
};

class TestDepthPass final : public RenderPass {
public:
    TestDepthPass(std::string name, GraphImage target, VernonRhiLoadOperation load, bool readOnly)
        : RenderPass(std::move(name)), target_(target), load_(load), readOnly_(readOnly) {}

    void declare() override {
        DepthStencilAttachmentUse attachment{};
        attachment.image = target_;
        attachment.depthLoad = load_;
        attachment.readOnlyDepth = readOnly_;
        depth(attachment);
    }

    VernonRhiStatus execute(GraphicsEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }

private:
    GraphImage target_;
    VernonRhiLoadOperation load_;
    bool readOnly_;
};

GraphResource none() { return {UINT32_MAX, ResourceKind::Buffer}; }

TEST(ExecutionGraph, InfersHazardsAndHonorsExplicitDependencies) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphBuffer source = graph.importBuffer({0, 1});
    const GraphBuffer intermediate = graph.importBuffer({1, 1});
    const GraphBuffer output = graph.importBuffer({2, 1}, true);
    auto &produce = graph.emplacePass<TestComputePass>("produce", source, intermediate);
    auto &consume = graph.emplacePass<TestComputePass>("consume", intermediate, output);
    consume.dependsOn(produce);

    std::string error;
    ASSERT_TRUE(graph.compile(error)) << error;
    EXPECT_EQ(graph.schedule(), (std::vector<uint32_t>{0, 1}));
    ASSERT_EQ(graph.scopes().size(), 2u);
    EXPECT_FALSE(graph.scopes()[0].rendering);
}

TEST(ExecutionGraph, FusesCompatibleRenderPassesAndSplitsCompute) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphImage color = graph.importImage({0, 1}, {0, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 64, 64, 1, 1, true);
    const GraphBuffer buffer = graph.importBuffer({0, 1}, true);
    graph.emplacePass<TestRenderPass>("first", color, VERNON_RHI_LOAD_CLEAR);
    graph.emplacePass<TestRenderPass>("second", color);
    graph.emplacePass<TestComputePass>("compute", none(), buffer);

    std::string error;
    ASSERT_TRUE(graph.compile(error)) << error;
    ASSERT_EQ(graph.scopes().size(), 2u);
    EXPECT_TRUE(graph.scopes()[0].rendering);
    EXPECT_EQ(graph.scopes()[0].passIndices, (std::vector<uint32_t>{0, 1}));
    EXPECT_FALSE(graph.scopes()[1].rendering);
}

TEST(ExecutionGraph, RejectsDependencyCycles) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    auto &first = graph.emplacePass<TestComputePass>("first", none(), none());
    auto &second = graph.emplacePass<TestComputePass>("second", none(), none());
    first.setFlags(PassSideEffect);
    second.setFlags(PassSideEffect);
    first.dependsOn(second);
    second.dependsOn(first);

    std::string error;
    EXPECT_FALSE(graph.compile(error));
    EXPECT_NE(error.find("cycle"), std::string::npos);
    EXPECT_TRUE(graph.schedule().empty());
    EXPECT_TRUE(graph.scopes().empty());
}

TEST(ExecutionGraph, CullsTransientPassesWithoutLiveConsumers) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphBuffer transient = graph.importBuffer({0, 1});
    graph.emplacePass<TestComputePass>("dead", none(), transient);

    std::string error;
    ASSERT_TRUE(graph.compile(error)) << error;
    EXPECT_TRUE(graph.schedule().empty());
    EXPECT_TRUE(graph.scopes().empty());
}

TEST(ExecutionGraph, DeduplicatesImportsAndPromotesExportedResources) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphBuffer first = graph.importBuffer({4, 9});
    const GraphBuffer second = graph.importBuffer({4, 9}, true);
    EXPECT_EQ(first.id, second.id);
    EXPECT_EQ(first.graphIdentity, second.graphIdentity);

    const GraphImage firstView = graph.importImage({7, 3}, {10, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16);
    const GraphImage secondView = graph.importImage({7, 3}, {11, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16);
    EXPECT_EQ(firstView.id, secondView.id);
    EXPECT_NE(firstView.view.index, secondView.view.index);

    graph.emplacePass<TestComputePass>("write", none(), second);
    std::string error;
    ASSERT_TRUE(graph.compile(error)) << error;
    EXPECT_EQ(graph.schedule(), (std::vector<uint32_t>{0}));
}

TEST(ExecutionGraph, AliasedImportsPreserveRawWarAndWawHazards) {
    const auto expectBothPassesLive = [](AccessMode firstAccess, AccessMode secondAccess) {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphBuffer firstAlias = graph.importBuffer({4, 9}, secondAccess != AccessMode::Read);
        const GraphBuffer secondAlias = graph.importBuffer({4, 9});
        const GraphBuffer output = graph.importBuffer({5, 1}, secondAccess == AccessMode::Read);
        if (firstAccess == AccessMode::Read)
            graph.emplacePass<TestComputePass>("first", firstAlias, output);
        else
            graph.emplacePass<TestComputePass>("first", none(), firstAlias);
        if (secondAccess == AccessMode::Read)
            graph.emplacePass<TestComputePass>("second", secondAlias, output);
        else
            graph.emplacePass<TestComputePass>("second", none(), secondAlias);
        std::string error;
        EXPECT_TRUE(graph.compile(error)) << error;
        EXPECT_EQ(graph.schedule(), (std::vector<uint32_t>{0, 1}));
    };
    expectBothPassesLive(AccessMode::Write, AccessMode::Read);
    expectBothPassesLive(AccessMode::Read, AccessMode::Write);
    expectBothPassesLive(AccessMode::Write, AccessMode::Write);
}

TEST(ExecutionGraph, RejectsResourceFromAnotherGraphWithMatchingNumericId) {
    ExecutionGraph first({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    ExecutionGraph second({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphBuffer foreign = first.importBuffer({0, 1});
    const GraphBuffer output = second.importBuffer({1, 1}, true);
    second.emplacePass<TestComputePass>("foreign", foreign, output);

    std::string error;
    EXPECT_FALSE(second.compile(error));
    EXPECT_NE(error.find("another execution graph"), std::string::npos);
    EXPECT_NE(error.find("foreign"), std::string::npos);
}

TEST(ExecutionGraph, DerivesBarrierStageAccessAndStateFromUses) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphBuffer intermediate = graph.importBuffer({0, 1});
    const GraphImage color = graph.importImage({1, 1}, {1, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16, 1, 1, true);
    graph.emplacePass<TestComputePass>("produce", none(), intermediate);
    graph.emplacePass<TestRenderReadPass>("consume", color, intermediate);

    std::string error;
    ASSERT_TRUE(graph.compile(error)) << error;
    ASSERT_EQ(graph.scopes().size(), 2u);
    ASSERT_EQ(graph.scopes()[1].barriers.size(), 1u);
    const VernonRhiBarrier &barrier = graph.scopes()[1].barriers.front();
    EXPECT_EQ(barrier.source_stage_mask, VERNON_RHI_STAGE_COMPUTE);
    EXPECT_EQ(barrier.destination_stage_mask, VERNON_RHI_STAGE_FRAGMENT);
    EXPECT_EQ(barrier.source_access, VERNON_RHI_ACCESS_SHADER_WRITE);
    EXPECT_EQ(barrier.destination_access, VERNON_RHI_ACCESS_SHADER_READ);
    EXPECT_EQ(barrier.old_state, VERNON_RHI_STATE_SHADER_WRITE);
    EXPECT_EQ(barrier.new_state, VERNON_RHI_STATE_SHADER_READ);
}

TEST(ExecutionGraph, RejectsInvalidAttachmentFormatAndExtent) {
    {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage depth = graph.importImage({0, 1}, {0, 1}, VERNON_RHI_FORMAT_D32_FLOAT, 16, 16, 1, 1, true);
        graph.emplacePass<TestRenderPass>("depth-as-color", depth);
        std::string error;
        EXPECT_FALSE(graph.compile(error));
        EXPECT_NE(error.find("depth-as-color"), std::string::npos);
    }
    {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage first = graph.importImage({0, 1}, {0, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16, 1, 1, true);
        const GraphImage second = graph.importImage({1, 1}, {1, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 32, 16, 1, 1, true);
        graph.emplacePass<TestMultiRenderPass>("mismatched", first, second);
        std::string error;
        EXPECT_FALSE(graph.compile(error));
        EXPECT_NE(error.find("incompatible extent"), std::string::npos);
    }
}

TEST(ExecutionGraph, RejectsClearOnReadOnlyDepthAndSplitsReadOnlyChanges) {
    {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage depth = graph.importImage({0, 1}, {0, 1}, VERNON_RHI_FORMAT_D32_FLOAT, 16, 16, 1, 1, true);
        graph.emplacePass<TestDepthPass>("read-only-clear", depth, VERNON_RHI_LOAD_CLEAR, true);
        std::string error;
        EXPECT_FALSE(graph.compile(error));
        EXPECT_NE(error.find("read-only-clear"), std::string::npos);
    }
    {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage depth = graph.importImage({0, 1}, {0, 1}, VERNON_RHI_FORMAT_D32_FLOAT, 16, 16, 1, 1, true);
        graph.emplacePass<TestDepthPass>("write", depth, VERNON_RHI_LOAD_PRESERVE, false);
        auto &read = graph.emplacePass<TestDepthPass>("read", depth, VERNON_RHI_LOAD_PRESERVE, true);
        read.setFlags(PassSideEffect);
        std::string error;
        ASSERT_TRUE(graph.compile(error)) << error;
        EXPECT_EQ(graph.scopes().size(), 2u);
    }
}

TEST(ExecutionGraph, SplitsScopesWhenIntermediateDiscardCannotBeRepresented) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphImage color = graph.importImage({0, 1}, {0, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16, 1, 1, true);
    graph.emplacePass<TestRenderPass>("discard-output", color, VERNON_RHI_LOAD_CLEAR, VERNON_RHI_STORE_DISCARD);
    graph.emplacePass<TestRenderPass>("preserve-input", color, VERNON_RHI_LOAD_PRESERVE);

    std::string error;
    ASSERT_TRUE(graph.compile(error)) << error;
    ASSERT_EQ(graph.scopes().size(), 2u);
    EXPECT_TRUE(graph.scopes()[0].rendering);
    EXPECT_TRUE(graph.scopes()[1].rendering);
}

TEST(ExecutionGraph, FusesAnIntermediateClearButSplitsAnIntermediateDiscardLoad) {
    const auto compileScopes = [](VernonRhiLoadOperation secondLoad) {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage color = graph.importImage({0, 1}, {0, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16, 1, 1, true);
        graph.emplacePass<TestRenderPass>("first", color);
        graph.emplacePass<TestRenderPass>("second", color, secondLoad);
        std::string error;
        EXPECT_TRUE(graph.compile(error)) << error;
        return graph.scopes().size();
    };
    EXPECT_EQ(compileScopes(VERNON_RHI_LOAD_CLEAR), 1u);
    EXPECT_EQ(compileScopes(VERNON_RHI_LOAD_DISCARD), 2u);
}

TEST(ExecutionGraph, SplitsRenderScopesForNonAttachmentHazards) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphImage color = graph.importImage({0, 1}, {0, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16, 1, 1, true);
    const GraphBuffer buffer = graph.importBuffer({1, 1});
    graph.emplacePass<TestRenderResourcePass>("write", color, buffer, AccessMode::Write);
    graph.emplacePass<TestRenderResourcePass>("read", color, buffer, AccessMode::Read);

    std::string error;
    ASSERT_TRUE(graph.compile(error)) << error;
    ASSERT_EQ(graph.scopes().size(), 2u);
    ASSERT_EQ(graph.scopes()[1].barriers.size(), 2u);
    EXPECT_EQ(graph.scopes()[1].barriers[0].source_stage_mask, 0u);
    EXPECT_EQ(graph.scopes()[1].barriers[0].source_access,
              VERNON_RHI_ACCESS_COLOR_READ | VERNON_RHI_ACCESS_COLOR_WRITE);
}

} // namespace
