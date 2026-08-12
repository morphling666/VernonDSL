#include "VernonExecutionGraph.h"

#include <gtest/gtest.h>

namespace vernon::execution::detail {

struct ExecutionGraphTestAccess {
    static GraphBuffer importBuffer(ExecutionGraph &graph, VernonRhiBuffer buffer, bool exported = false) {
        const uint64_t key = (static_cast<uint64_t>(buffer.generation) << 32) | (uint64_t{buffer.index} + 1);
        if (const auto found = graph.importedBuffers_.find(key); found != graph.importedBuffers_.end()) {
            graph.resourceRecords_[found->second].exported |= exported;
            GraphBuffer result;
            result.id = found->second;
            result.kind = ResourceKind::Buffer;
            result.graphIdentity = graph.graphIdentity_;
            result.handle = buffer;
            return result;
        }
        GraphBuffer result;
        result.id = static_cast<uint32_t>(graph.resources_.size());
        result.kind = ResourceKind::Buffer;
        result.graphIdentity = graph.graphIdentity_;
        result.handle = buffer;
        graph.resources_.push_back(result);
        graph.resourceRecords_.push_back({result, exported});
        graph.resourceRecords_.back().buffer = buffer;
        graph.resourceRecords_.back().resourceKey = key;
        graph.importedBuffers_.emplace(key, result.id);
        graph.dirty_ = true;
        return result;
    }

    static GraphImage importImage(ExecutionGraph &graph, VernonRhiImage image, VernonRhiImageView view,
                                  VernonRhiFormat format, uint32_t width, uint32_t height, uint32_t layers = 1,
                                  uint32_t samples = 1, bool exported = false,
                                  const VernonRhiImageSubresourceRange *subresources = nullptr) {
        const uint32_t aspects =
            subresources ? subresources->aspects
                         : static_cast<uint32_t>(format == VERNON_RHI_FORMAT_D32_FLOAT ? VERNON_RHI_IMAGE_ASPECT_DEPTH
                                                 : format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT
                                                     ? VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL
                                                     : VERNON_RHI_IMAGE_ASPECT_COLOR);
        const uint32_t baseMip = subresources ? subresources->base_mip_level : 0;
        const uint32_t mipCount =
            subresources && subresources->mip_level_count != UINT32_MAX ? subresources->mip_level_count : 1;
        const uint32_t baseLayer = subresources ? subresources->base_array_layer : 0;
        const uint32_t layerCount =
            subresources && subresources->array_layer_count != UINT32_MAX ? subresources->array_layer_count : layers;
        GraphImage result;
        result.kind = ResourceKind::Image;
        result.graphIdentity = graph.graphIdentity_;
        result.handle = image;
        result.view = view;
        result.format = format;
        result.width = width;
        result.height = height;
        result.layers = layerCount;
        result.samples = samples;
        result.subresources = {baseMip, mipCount, baseLayer, layerCount, aspects};
        const uint64_t key = (static_cast<uint64_t>(image.generation) << 32) | (uint64_t{image.index} + 1);
        if (const auto found = graph.importedImages_.find(key); found != graph.importedImages_.end()) {
            graph.resourceRecords_[found->second].exported |= exported;
            result.id = found->second;
            return result;
        }
        result.id = static_cast<uint32_t>(graph.resources_.size());
        graph.resources_.push_back(result);
        graph.resourceRecords_.push_back({result, exported});
        graph.resourceRecords_.back().image = image;
        graph.resourceRecords_.back().resourceKey = key;
        graph.importedImages_.emplace(key, result.id);
        graph.dirty_ = true;
        return result;
    }
};

} // namespace vernon::execution::detail

namespace {

using namespace vernon::execution;

GraphBuffer importBufferForTesting(ExecutionGraph &graph, VernonRhiBuffer buffer, bool exported = false) {
    return detail::ExecutionGraphTestAccess::importBuffer(graph, buffer, exported);
}

GraphImage importImageForTesting(ExecutionGraph &graph, VernonRhiImage image, VernonRhiImageView view,
                                 VernonRhiFormat format, uint32_t width, uint32_t height, uint32_t layers = 1,
                                 uint32_t samples = 1, bool exported = false,
                                 const VernonRhiImageSubresourceRange *subresources = nullptr) {
    return detail::ExecutionGraphTestAccess::importImage(graph, image, view, format, width, height, layers, samples,
                                                         exported, subresources);
}

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

class CpuComputePass final : public ComputePass {
public:
    CpuComputePass(std::string name, GraphResource resource, AccessMode access, std::vector<std::string> &events,
                   VernonRhiStatus status = VERNON_RHI_STATUS_OK)
        : ComputePass(std::move(name)), resource_(resource), access_(access), events_(events), status_(status) {}

    void declare() override {
        if (access_ == AccessMode::Read)
            read(resource_);
        else if (access_ == AccessMode::Write)
            write(resource_);
        else
            readWrite(resource_);
        setFlags(PassSideEffect);
    }

    VernonRhiStatus execute(ComputeEncoder &, const ExecutionResources &) override {
        events_.push_back(name());
        return status_;
    }

private:
    GraphResource resource_;
    AccessMode access_;
    std::vector<std::string> &events_;
    VernonRhiStatus status_;
};

struct IntegerBindingValue final : ExecutionBindingValue {
    explicit IntegerBindingValue(int value) : value(value) {}
    int value;
};

class ParameterComputePass final : public ComputePass {
public:
    ParameterComputePass(ExecutionParameter parameter, std::vector<std::shared_ptr<const IntegerBindingValue>> &values)
        : ComputePass("parameter"), parameter_(parameter), values_(values) {}

    void declare() override { setFlags(PassSideEffect); }

    VernonRhiStatus execute(ComputeEncoder &, const ExecutionResources &resources) override {
        values_.push_back(std::dynamic_pointer_cast<const IntegerBindingValue>(resources.binding(parameter_)));
        return values_.back() ? VERNON_RHI_STATUS_OK : VERNON_RHI_STATUS_INTERNAL_ERROR;
    }

private:
    ExecutionParameter parameter_;
    std::vector<std::shared_ptr<const IntegerBindingValue>> &values_;
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

class TestImageSubresourcePass final : public ComputePass {
public:
    TestImageSubresourcePass(std::string name, GraphImage image, AccessMode access)
        : ComputePass(std::move(name)), image_(image), access_(access) {}

    void declare() override {
        if (access_ == AccessMode::Read)
            read(image_);
        else
            write(image_);
        setFlags(PassSideEffect);
    }

    VernonRhiStatus execute(ComputeEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }

private:
    GraphImage image_;
    AccessMode access_;
};

GraphResource none() { return {UINT32_MAX, ResourceKind::Buffer}; }

TEST(ExecutionGraph, InfersHazardsAndHonorsExplicitDependencies) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphBuffer source = importBufferForTesting(graph, {0, 1});
    const GraphBuffer intermediate = importBufferForTesting(graph, {1, 1});
    const GraphBuffer output = importBufferForTesting(graph, {2, 1}, true);
    auto &produce = graph.emplacePass<TestComputePass>("produce", source, intermediate);
    auto &consume = graph.emplacePass<TestComputePass>("consume", intermediate, output);
    consume.dependsOn(produce);

    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    EXPECT_EQ(plan->schedule(), (std::vector<uint32_t>{0, 1}));
    ASSERT_EQ(plan->scopes().size(), 2u);
    EXPECT_FALSE(plan->scopes()[0].rendering);
}

TEST(ExecutionGraph, CpuProviderExecutesCompiledRawWarAndWawSchedules) {
    const auto executeHazard = [](AccessMode firstAccess, AccessMode secondAccess) {
        ExecutionGraph graph;
        const GraphBuffer shared = graph.importHostBuffer(1, true);
        std::vector<std::string> events;
        graph.emplacePass<CpuComputePass>("first", shared, firstAccess, events);
        graph.emplacePass<CpuComputePass>("second", shared, secondAccess, events);

        std::string error;
        auto plan = graph.compile(error);
        ASSERT_TRUE(plan) << error;
        EXPECT_EQ(plan->schedule(), (std::vector<uint32_t>{0, 1}));
        auto submission = plan->submit();
        EXPECT_EQ(submission.wait(), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(events, (std::vector<std::string>{"first", "second"}));
    };

    executeHazard(AccessMode::Write, AccessMode::Read);
    executeHazard(AccessMode::Read, AccessMode::Write);
    executeHazard(AccessMode::Write, AccessMode::Write);
}

TEST(ExecutionGraph, CpuProviderPropagatesFailureAndStopsSchedule) {
    ExecutionGraph graph;
    const GraphBuffer shared = graph.importHostBuffer(1, true);
    std::vector<std::string> events;
    graph.emplacePass<CpuComputePass>("first", shared, AccessMode::Write, events);
    graph.emplacePass<CpuComputePass>("failure", shared, AccessMode::ReadWrite, events,
                                      VERNON_RHI_STATUS_INTERNAL_ERROR);
    graph.emplacePass<CpuComputePass>("not-run", shared, AccessMode::Read, events);

    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    auto submission = plan->submit();
    EXPECT_EQ(submission.wait(), VERNON_RHI_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(events, (std::vector<std::string>{"first", "failure"}));
}

TEST(ExecutionGraph, CompiledPlanOwnsPassesAndSupportsRepeatedSubmissions) {
    std::vector<std::string> events;
    std::shared_ptr<CompiledExecutionGraph> plan;
    {
        ExecutionGraph graph;
        const GraphBuffer shared = graph.importHostBuffer(1, true);
        graph.emplacePass<CpuComputePass>("run", shared, AccessMode::Write, events);
        std::string error;
        plan = graph.compile(error);
        ASSERT_TRUE(plan) << error;
        EXPECT_THROW(graph.emplacePass<CpuComputePass>("late", shared, AccessMode::Read, events), std::logic_error);
    }

    auto first = plan->submit();
    EXPECT_EQ(first.wait(), VERNON_RHI_STATUS_OK);
    auto second = plan->submit();
    EXPECT_EQ(second.wait(), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(events, (std::vector<std::string>{"run", "run"}));
}

TEST(ExecutionGraph, SubmissionRetainsCompiledPlan) {
    std::vector<std::string> events;
    ExecutionGraph graph;
    const GraphBuffer shared = graph.importHostBuffer(1, true);
    graph.emplacePass<CpuComputePass>("run", shared, AccessMode::Write, events);
    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;

    auto submission = plan->submit();
    plan.reset();
    EXPECT_EQ(submission.state(), ExecutionSubmission::State::Succeeded);
    EXPECT_EQ(submission.wait(), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(events, (std::vector<std::string>{"run"}));
}

TEST(ExecutionGraph, FusesCompatibleRenderPassesAndSplitsCompute) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphImage color =
        importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 64, 64, 1, 1, true);
    const GraphBuffer buffer = importBufferForTesting(graph, {0, 1}, true);
    graph.emplacePass<TestRenderPass>("first", color, VERNON_RHI_LOAD_CLEAR);
    graph.emplacePass<TestRenderPass>("second", color);
    graph.emplacePass<TestComputePass>("compute", none(), buffer);

    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    ASSERT_EQ(plan->scopes().size(), 2u);
    EXPECT_TRUE(plan->scopes()[0].rendering);
    EXPECT_EQ(plan->scopes()[0].passIndices, (std::vector<uint32_t>{0, 1}));
    EXPECT_FALSE(plan->scopes()[1].rendering);
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
    const GraphBuffer transient = importBufferForTesting(graph, {0, 1});
    graph.emplacePass<TestComputePass>("dead", none(), transient);

    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    EXPECT_TRUE(plan->schedule().empty());
    EXPECT_TRUE(plan->scopes().empty());
}

TEST(ExecutionGraph, DeduplicatesImportsAndPromotesExportedResources) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphBuffer first = importBufferForTesting(graph, {4, 9});
    const GraphBuffer second = importBufferForTesting(graph, {4, 9}, true);
    EXPECT_EQ(first.id, second.id);
    EXPECT_EQ(first.graphIdentity, second.graphIdentity);

    const GraphImage firstView = importImageForTesting(graph, {7, 3}, {10, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16);
    const GraphImage secondView = importImageForTesting(graph, {7, 3}, {11, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16);
    EXPECT_EQ(firstView.id, secondView.id);
    EXPECT_NE(firstView.view.index, secondView.view.index);

    graph.emplacePass<TestComputePass>("write", none(), second);
    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    EXPECT_EQ(plan->schedule(), (std::vector<uint32_t>{0}));
}

TEST(ExecutionGraph, AliasedImportsPreserveRawWarAndWawHazards) {
    const auto expectBothPassesLive = [](AccessMode firstAccess, AccessMode secondAccess) {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphBuffer firstAlias = importBufferForTesting(graph, {4, 9}, secondAccess != AccessMode::Read);
        const GraphBuffer secondAlias = importBufferForTesting(graph, {4, 9});
        const GraphBuffer output = importBufferForTesting(graph, {5, 1}, secondAccess == AccessMode::Read);
        if (firstAccess == AccessMode::Read)
            graph.emplacePass<TestComputePass>("first", firstAlias, output);
        else
            graph.emplacePass<TestComputePass>("first", none(), firstAlias);
        if (secondAccess == AccessMode::Read)
            graph.emplacePass<TestComputePass>("second", secondAlias, output);
        else
            graph.emplacePass<TestComputePass>("second", none(), secondAlias);
        std::string error;
        auto plan = graph.compile(error);
        ASSERT_TRUE(plan) << error;
        EXPECT_EQ(plan->schedule(), (std::vector<uint32_t>{0, 1}));
    };
    expectBothPassesLive(AccessMode::Write, AccessMode::Read);
    expectBothPassesLive(AccessMode::Read, AccessMode::Write);
    expectBothPassesLive(AccessMode::Write, AccessMode::Write);
}

TEST(ExecutionGraph, RejectsResourceFromAnotherGraphWithMatchingNumericId) {
    ExecutionGraph first({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    ExecutionGraph second({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphBuffer foreign = importBufferForTesting(first, {0, 1});
    const GraphBuffer output = importBufferForTesting(second, {1, 1}, true);
    second.emplacePass<TestComputePass>("foreign", foreign, output);

    std::string error;
    EXPECT_FALSE(second.compile(error));
    EXPECT_NE(error.find("another execution graph"), std::string::npos);
    EXPECT_NE(error.find("foreign"), std::string::npos);
}

TEST(ExecutionGraph, DerivesBarrierStageAccessAndStateFromUses) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphBuffer intermediate = importBufferForTesting(graph, {0, 1});
    const GraphImage color =
        importImageForTesting(graph, {1, 1}, {1, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16, 1, 1, true);
    graph.emplacePass<TestComputePass>("produce", none(), intermediate);
    graph.emplacePass<TestRenderReadPass>("consume", color, intermediate);

    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    ASSERT_EQ(plan->scopes().size(), 2u);
    ASSERT_EQ(plan->scopes()[1].barriers.size(), 1u);
    const VernonRhiBarrier &barrier = plan->scopes()[1].barriers.front();
    EXPECT_EQ(barrier.source_stage_mask, VERNON_RHI_STAGE_COMPUTE);
    EXPECT_EQ(barrier.destination_stage_mask, VERNON_RHI_STAGE_FRAGMENT);
    EXPECT_EQ(barrier.source_access, VERNON_RHI_ACCESS_SHADER_WRITE);
    EXPECT_EQ(barrier.destination_access, VERNON_RHI_ACCESS_SHADER_READ);
    EXPECT_EQ(barrier.old_state, VERNON_RHI_STATE_SHADER_WRITE);
    EXPECT_EQ(barrier.new_state, VERNON_RHI_STATE_SHADER_READ);
}

TEST(ExecutionGraph, TracksImageHazardsByParentAndSubresourceRange) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const VernonRhiImageSubresourceRange mip0{0, 1, 0, 1, VERNON_RHI_IMAGE_ASPECT_COLOR};
    const VernonRhiImageSubresourceRange mip1{1, 1, 0, 1, VERNON_RHI_IMAGE_ASPECT_COLOR};
    const GraphImage first =
        importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16, 1, 1, false, &mip0);
    const GraphImage second =
        importImageForTesting(graph, {0, 1}, {1, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 8, 8, 1, 1, false, &mip1);
    ASSERT_EQ(first.id, second.id);
    graph.emplacePass<TestImageSubresourcePass>("write-mip-0", first, AccessMode::Write);
    graph.emplacePass<TestImageSubresourcePass>("read-mip-1", second, AccessMode::Read);
    graph.emplacePass<TestImageSubresourcePass>("read-mip-0", first, AccessMode::Read);

    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    ASSERT_EQ(plan->scopes().size(), 3u);
    EXPECT_TRUE(plan->scopes()[1].barriers.empty());
    ASSERT_EQ(plan->scopes()[2].barriers.size(), 1u);
    const VernonRhiBarrier &barrier = plan->scopes()[2].barriers.front();
    EXPECT_EQ(barrier.image_subresources.base_mip_level, 0u);
    EXPECT_EQ(barrier.image_subresources.mip_level_count, 1u);
    EXPECT_EQ(barrier.image_subresources.base_array_layer, 0u);
    EXPECT_EQ(barrier.image_subresources.array_layer_count, 1u);
    EXPECT_EQ(barrier.image_subresources.aspects, VERNON_RHI_IMAGE_ASPECT_COLOR);
    EXPECT_EQ(barrier.old_state, VERNON_RHI_STATE_SHADER_WRITE);
    EXPECT_EQ(barrier.new_state, VERNON_RHI_STATE_SHADER_READ);
    EXPECT_EQ(barrier.source_access, VERNON_RHI_ACCESS_SHADER_WRITE);
    EXPECT_EQ(barrier.destination_access, VERNON_RHI_ACCESS_SHADER_READ);
}

TEST(ExecutionGraph, RejectsInvalidAttachmentFormatAndExtent) {
    {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage depth =
            importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_D32_FLOAT, 16, 16, 1, 1, true);
        graph.emplacePass<TestRenderPass>("depth-as-color", depth);
        std::string error;
        EXPECT_FALSE(graph.compile(error));
        EXPECT_NE(error.find("depth-as-color"), std::string::npos);
    }
    {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage first =
            importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16, 1, 1, true);
        const GraphImage second =
            importImageForTesting(graph, {1, 1}, {1, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 32, 16, 1, 1, true);
        graph.emplacePass<TestMultiRenderPass>("mismatched", first, second);
        std::string error;
        EXPECT_FALSE(graph.compile(error));
        EXPECT_NE(error.find("incompatible extent"), std::string::npos);
    }
}

TEST(ExecutionGraph, RejectsClearOnReadOnlyDepthAndSplitsReadOnlyChanges) {
    {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage depth =
            importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_D32_FLOAT, 16, 16, 1, 1, true);
        graph.emplacePass<TestDepthPass>("read-only-clear", depth, VERNON_RHI_LOAD_CLEAR, true);
        std::string error;
        EXPECT_FALSE(graph.compile(error));
        EXPECT_NE(error.find("read-only-clear"), std::string::npos);
    }
    {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage depth =
            importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_D32_FLOAT, 16, 16, 1, 1, true);
        graph.emplacePass<TestDepthPass>("write", depth, VERNON_RHI_LOAD_PRESERVE, false);
        auto &read = graph.emplacePass<TestDepthPass>("read", depth, VERNON_RHI_LOAD_PRESERVE, true);
        read.setFlags(PassSideEffect);
        std::string error;
        auto plan = graph.compile(error);
        ASSERT_TRUE(plan) << error;
        EXPECT_EQ(plan->scopes().size(), 2u);
    }
}

TEST(ExecutionGraph, SplitsScopesWhenIntermediateDiscardCannotBeRepresented) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphImage color =
        importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16, 1, 1, true);
    graph.emplacePass<TestRenderPass>("discard-output", color, VERNON_RHI_LOAD_CLEAR, VERNON_RHI_STORE_DISCARD);
    graph.emplacePass<TestRenderPass>("preserve-input", color, VERNON_RHI_LOAD_PRESERVE);

    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    ASSERT_EQ(plan->scopes().size(), 2u);
    EXPECT_TRUE(plan->scopes()[0].rendering);
    EXPECT_TRUE(plan->scopes()[1].rendering);
}

TEST(ExecutionGraph, FusesAnIntermediateClearButSplitsAnIntermediateDiscardLoad) {
    const auto compileScopes = [](VernonRhiLoadOperation secondLoad) {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage color =
            importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16, 1, 1, true);
        graph.emplacePass<TestRenderPass>("first", color);
        graph.emplacePass<TestRenderPass>("second", color, secondLoad);
        std::string error;
        auto plan = graph.compile(error);
        EXPECT_TRUE(plan) << error;
        return plan ? plan->scopes().size() : 0;
    };
    EXPECT_EQ(compileScopes(VERNON_RHI_LOAD_CLEAR), 1u);
    EXPECT_EQ(compileScopes(VERNON_RHI_LOAD_DISCARD), 2u);
}

TEST(ExecutionGraph, SplitsRenderScopesForNonAttachmentHazards) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphImage color =
        importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16, 1, 1, true);
    const GraphBuffer buffer = importBufferForTesting(graph, {1, 1});
    graph.emplacePass<TestRenderResourcePass>("write", color, buffer, AccessMode::Write);
    graph.emplacePass<TestRenderResourcePass>("read", color, buffer, AccessMode::Read);

    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    ASSERT_EQ(plan->scopes().size(), 2u);
    ASSERT_EQ(plan->scopes()[1].barriers.size(), 2u);
    EXPECT_EQ(plan->scopes()[1].barriers[0].source_stage_mask, 0u);
    EXPECT_EQ(plan->scopes()[1].barriers[0].source_access,
              VERNON_RHI_ACCESS_COLOR_READ | VERNON_RHI_ACCESS_COLOR_WRITE);
}

TEST(ExecutionGraph, ValidatesParameterSchemasAndInitialBindings) {
    ExecutionGraph graph;
    const ExecutionParameter parameter = graph.parameter("value");
    EXPECT_THROW(graph.parameter("value"), std::invalid_argument);
    std::vector<std::shared_ptr<const IntegerBindingValue>> observed;
    graph.emplacePass<ParameterComputePass>(parameter, observed);
    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;

    EXPECT_THROW(plan->createBindings({}), std::invalid_argument);
    const auto value = std::make_shared<IntegerBindingValue>(1);
    EXPECT_THROW(plan->createBindings({{parameter, value}, {parameter, value}}), std::invalid_argument);

    ExecutionGraph foreignGraph;
    const ExecutionParameter foreign = foreignGraph.parameter("foreign");
    EXPECT_THROW(plan->createBindings({{foreign, value}}), std::invalid_argument);
}

TEST(ExecutionGraph, SnapshotsSparseParameterUpdatesAndRetainsSubmissionValues) {
    ExecutionGraph graph;
    const ExecutionParameter dynamic = graph.parameter("dynamic");
    const ExecutionParameter constant = graph.parameter("constant");
    std::vector<std::shared_ptr<const IntegerBindingValue>> observed;
    graph.emplacePass<ParameterComputePass>(dynamic, observed);
    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;

    const auto firstDynamic = std::make_shared<IntegerBindingValue>(1);
    const auto constantValue = std::make_shared<IntegerBindingValue>(7);
    auto bindings = plan->createBindings({{dynamic, firstDynamic}, {constant, constantValue}});
    const auto firstSnapshot = bindings.snapshot();
    EXPECT_EQ(firstSnapshot, bindings.snapshot());
    EXPECT_EQ(firstSnapshot->at(constant), constantValue);

    ExecutionSubmission first = plan->submit(firstSnapshot);
    const auto secondDynamic = std::make_shared<IntegerBindingValue>(2);
    bindings.set(dynamic, secondDynamic);
    const auto secondSnapshot = bindings.snapshot();
    EXPECT_NE(firstSnapshot, secondSnapshot);
    EXPECT_EQ(firstSnapshot->at(dynamic), firstDynamic);
    EXPECT_EQ(secondSnapshot->at(dynamic), secondDynamic);
    EXPECT_EQ(secondSnapshot->at(constant), constantValue);
    ExecutionSubmission second = plan->submit(secondSnapshot);

    ASSERT_EQ(observed.size(), 2u);
    EXPECT_EQ(observed[0]->value, 1);
    EXPECT_EQ(observed[1]->value, 2);
    plan.reset();
    EXPECT_EQ(first.wait(), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(second.wait(), VERNON_RHI_STATUS_OK);
}

} // namespace
