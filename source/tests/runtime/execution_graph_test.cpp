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
    TestRenderPass(std::string name, GraphImage target, VernonRhiLoadOperation load = VERNON_RHI_LOAD_PRESERVE)
        : RenderPass(std::move(name)), target_(target), load_(load) {}

    void declare() override {
        ColorAttachmentUse attachment{};
        attachment.image = target_;
        attachment.load = load_;
        color(0, attachment);
    }

    VernonRhiStatus execute(GraphicsEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }

private:
    GraphImage target_;
    VernonRhiLoadOperation load_;
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

} // namespace
