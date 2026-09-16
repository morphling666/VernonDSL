#include "execution_graph/command_graph.h"
#include "execution_graph/execution_graph_checkpoint_planner_internal.h"
#include "execution_graph/execution_graph_internal.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace vernon::execution::detail {

struct CommandGraphTestAccess {
    static GraphBuffer importBuffer(CommandGraph &graph, VernonRhiBuffer buffer, bool exported = false) {
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

    static GraphImage importImage(CommandGraph &graph, VernonRhiImage image, VernonRhiImageView view,
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

GraphBuffer importBufferForTesting(CommandGraph &graph, VernonRhiBuffer buffer, bool exported = false) {
    return detail::CommandGraphTestAccess::importBuffer(graph, buffer, exported);
}

GraphImage importImageForTesting(CommandGraph &graph, VernonRhiImage image, VernonRhiImageView view,
                                 VernonRhiFormat format, uint32_t width, uint32_t height, uint32_t layers = 1,
                                 uint32_t samples = 1, bool exported = false,
                                 const VernonRhiImageSubresourceRange *subresources = nullptr) {
    return detail::CommandGraphTestAccess::importImage(graph, image, view, format, width, height, layers, samples,
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

    void addReadForTesting(GraphResource resource) { read(resource); }

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
    TestDepthPass(std::string name, GraphImage target, VernonRhiLoadOperation load, bool readOnlyDepth,
                  VernonRhiStoreOperation depthStore = VERNON_RHI_STORE_PRESERVE,
                  VernonRhiLoadOperation stencilLoad = VERNON_RHI_LOAD_DISCARD,
                  VernonRhiStoreOperation stencilStore = VERNON_RHI_STORE_DISCARD, bool readOnlyStencil = false)
        : RenderPass(std::move(name)), target_(target), load_(load), depthStore_(depthStore), stencilLoad_(stencilLoad),
          stencilStore_(stencilStore), readOnlyDepth_(readOnlyDepth), readOnlyStencil_(readOnlyStencil) {}

    void declare() override {
        DepthStencilAttachmentUse attachment{};
        attachment.image = target_;
        attachment.depthLoad = load_;
        attachment.depthStore = depthStore_;
        attachment.stencilLoad = stencilLoad_;
        attachment.stencilStore = stencilStore_;
        attachment.readOnlyDepth = readOnlyDepth_;
        attachment.readOnlyStencil = readOnlyStencil_;
        depth(attachment);
    }

    VernonRhiStatus execute(GraphicsEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }

private:
    GraphImage target_;
    VernonRhiLoadOperation load_;
    VernonRhiStoreOperation depthStore_;
    VernonRhiLoadOperation stencilLoad_;
    VernonRhiStoreOperation stencilStore_;
    bool readOnlyDepth_;
    bool readOnlyStencil_;
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

class ConditionalFlagPass final : public ComputePass {
public:
    ConditionalFlagPass(std::string name, bool &sideEffect) : ComputePass(std::move(name)), sideEffect_(sideEffect) {}

    void declare() override {
        if (sideEffect_)
            setFlags(PassSideEffect);
    }

    VernonRhiStatus execute(ComputeEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }

private:
    bool &sideEffect_;
};

GraphResource none() { return {UINT32_MAX, ResourceKind::Buffer}; }

TEST(CommandGraph, InfersHazardsAndHonorsExplicitDependencies) {
    CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
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

TEST(CommandGraph, CpuProviderExecutesCompiledRawWarAndWawSchedules) {
    const auto executeHazard = [](AccessMode firstAccess, AccessMode secondAccess) {
        CommandGraph graph;
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

TEST(CommandGraph, CpuProviderPropagatesFailureAndStopsSchedule) {
    CommandGraph graph;
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

TEST(CommandGraph, CompiledPlanOwnsPassesAndSupportsRepeatedSubmissions) {
    std::vector<std::string> events;
    std::shared_ptr<CompiledCommandGraph> plan;
    {
        CommandGraph graph;
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

TEST(CommandGraph, FreezesRetainedPassReferencesAfterCompilation) {
    CommandGraph graph;
    const GraphBuffer output = graph.importHostBuffer(1, true);
    auto &pass = graph.emplacePass<TestComputePass>("run", none(), output);
    std::string error;
    auto compiled = graph.compile(error);
    ASSERT_TRUE(compiled) << error;
    EXPECT_THROW(pass.setFlags(PassNeverCull), std::logic_error);
    EXPECT_THROW(pass.dependsOn(pass), std::logic_error);
    EXPECT_THROW(pass.addReadForTesting(output), std::logic_error);
}

TEST(CommandGraph, SubmissionRetainsCompiledPlan) {
    std::vector<std::string> events;
    CommandGraph graph;
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

TEST(CommandGraph, FusesCompatibleRenderPassesAndSplitsCompute) {
    CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
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

TEST(CommandGraph, RejectsDependencyCycles) {
    CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
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

TEST(CommandGraph, CullsTransientPassesWithoutLiveConsumers) {
    CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphBuffer transient = importBufferForTesting(graph, {0, 1});
    graph.emplacePass<TestComputePass>("dead", none(), transient);

    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    EXPECT_TRUE(plan->schedule().empty());
    EXPECT_TRUE(plan->scopes().empty());
}

TEST(CommandGraph, RebuildClearsDeclarationDerivedFlags) {
    CommandGraph graph;
    bool sideEffect = true;
    graph.emplacePass<ConditionalFlagPass>("conditional", sideEffect);
    std::string error;
    ASSERT_TRUE(graph.validate(error)) << error;
    sideEffect = false;
    auto compiled = graph.compile(error);
    ASSERT_TRUE(compiled) << error;
    EXPECT_TRUE(compiled->schedule().empty());
}

TEST(CommandGraph, DeduplicatesImportsAndPromotesExportedResources) {
    CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
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

TEST(CommandGraph, AliasedImportsPreserveRawWarAndWawHazards) {
    const auto expectBothPassesLive = [](AccessMode firstAccess, AccessMode secondAccess) {
        CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
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

TEST(CommandGraph, RejectsResourceFromAnotherGraphWithMatchingNumericId) {
    CommandGraph first({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    CommandGraph second({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphBuffer foreign = importBufferForTesting(first, {0, 1});
    const GraphBuffer output = importBufferForTesting(second, {1, 1}, true);
    second.emplacePass<TestComputePass>("foreign", foreign, output);

    std::string error;
    EXPECT_FALSE(second.validate(error));
    EXPECT_NE(error.find("another execution graph"), std::string::npos);
    error.clear();
    EXPECT_FALSE(second.compile(error));
    EXPECT_NE(error.find("another execution graph"), std::string::npos);
    EXPECT_NE(error.find("foreign"), std::string::npos);
}

TEST(CommandGraph, DerivesBarrierStageAccessAndStateFromUses) {
    CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
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

TEST(CommandGraph, TracksImageHazardsByParentAndSubresourceRange) {
    CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
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

TEST(CommandGraph, PreservesComputeImageViewMetadataAndRejectsSlicedImages) {
    {
        CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const VernonRhiImageSubresourceRange range{2, 1, 3, 2, VERNON_RHI_IMAGE_ASPECT_COLOR};
        const GraphImage image =
            importImageForTesting(graph, {4, 2}, {7, 3}, VERNON_RHI_FORMAT_RGBA8_UNORM, 8, 8, 2, 1, true, &range);
        auto &pass = graph.emplacePass<TestImageSubresourcePass>("image", image, AccessMode::Read);
        std::string error;
        ASSERT_TRUE(graph.validate(error)) << error;
        ASSERT_EQ(pass.uses().size(), 1u);
        ASSERT_TRUE(pass.uses().front().image.has_value());
        EXPECT_EQ(pass.uses().front().image->view.index, image.view.index);
        EXPECT_EQ(pass.uses().front().image->subresources.base_mip_level, 2u);
        EXPECT_EQ(pass.uses().front().image->subresources.base_array_layer, 3u);
    }
    {
        CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage image =
            importImageForTesting(graph, {4, 2}, {7, 3}, VERNON_RHI_FORMAT_RGBA8_UNORM, 8, 8, 1, 1, true);
        graph.emplacePass<TestComputePass>("sliced-image", none(), image);
        std::string error;
        EXPECT_FALSE(graph.validate(error));
    }
}

TEST(CommandGraph, RejectsInvalidAttachmentFormatAndExtent) {
    {
        CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage depth =
            importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_D32_FLOAT, 16, 16, 1, 1, true);
        graph.emplacePass<TestRenderPass>("depth-as-color", depth);
        std::string error;
        EXPECT_FALSE(graph.compile(error));
        EXPECT_NE(error.find("depth-as-color"), std::string::npos);
    }
    {
        CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
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

TEST(CommandGraph, RejectsClearOnReadOnlyDepthAndSplitsReadOnlyChanges) {
    {
        CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage depth =
            importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_D32_FLOAT, 16, 16, 1, 1, true);
        graph.emplacePass<TestDepthPass>("read-only-clear", depth, VERNON_RHI_LOAD_CLEAR, true);
        std::string error;
        EXPECT_FALSE(graph.compile(error));
        EXPECT_NE(error.find("read-only-clear"), std::string::npos);
    }
    {
        CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
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

TEST(CommandGraph, TracksWritableStencilAndRejectsReadOnlyStencilMutation) {
    {
        CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage depthStencil =
            importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT, 16, 16, 1, 1, true);
        graph.emplacePass<TestDepthPass>("stencil-write", depthStencil, VERNON_RHI_LOAD_PRESERVE, true,
                                         VERNON_RHI_STORE_PRESERVE, VERNON_RHI_LOAD_PRESERVE, VERNON_RHI_STORE_PRESERVE,
                                         false);
        std::string error;
        auto plan = graph.compile(error);
        ASSERT_TRUE(plan) << error;
        ASSERT_EQ(plan->schedule(), (std::vector<uint32_t>{0}));
    }
    {
        CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage depthStencil =
            importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT, 16, 16, 1, 1, true);
        graph.emplacePass<TestDepthPass>("read-only-stencil-clear", depthStencil, VERNON_RHI_LOAD_PRESERVE, false,
                                         VERNON_RHI_STORE_PRESERVE, VERNON_RHI_LOAD_CLEAR, VERNON_RHI_STORE_PRESERVE,
                                         true);
        std::string error;
        EXPECT_FALSE(graph.compile(error));
        EXPECT_NE(error.find("read-only-stencil-clear"), std::string::npos);
    }
}

TEST(CommandGraph, SplitsScopesWhenIntermediateDiscardCannotBeRepresented) {
    CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
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

TEST(CommandGraph, FusesAnIntermediateClearButSplitsAnIntermediateDiscardLoad) {
    const auto compileScopes = [](VernonRhiLoadOperation secondLoad) {
        CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
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

TEST(CommandGraph, SplitsRenderScopesAcrossDiscardedStencilBoundary) {
    CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphImage depthStencil =
        importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT, 16, 16, 1, 1, true);
    graph.emplacePass<TestDepthPass>("discard-stencil", depthStencil, VERNON_RHI_LOAD_PRESERVE, false,
                                     VERNON_RHI_STORE_PRESERVE, VERNON_RHI_LOAD_PRESERVE, VERNON_RHI_STORE_DISCARD);
    graph.emplacePass<TestDepthPass>("preserve-stencil", depthStencil, VERNON_RHI_LOAD_PRESERVE, false,
                                     VERNON_RHI_STORE_PRESERVE, VERNON_RHI_LOAD_PRESERVE, VERNON_RHI_STORE_PRESERVE);
    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    EXPECT_EQ(plan->scopes().size(), 2u);
}

TEST(CommandGraph, SplitsRenderScopesForNonAttachmentHazards) {
    CommandGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
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

TEST(CommandGraph, ValidatesParameterSchemasAndInitialBindings) {
    CommandGraph graph;
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

    CommandGraph foreignGraph;
    const ExecutionParameter foreign = foreignGraph.parameter("foreign");
    EXPECT_THROW(plan->createBindings({{foreign, value}}), std::invalid_argument);
}

TEST(CommandGraph, SnapshotsSparseParameterUpdatesAndRetainsSubmissionValues) {
    CommandGraph graph;
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

struct TestDagNode {
    std::vector<uint32_t> predecessors;
    uint64_t checkpointBytes;
    uint64_t residualBytes;
    uint64_t replayCost;
    bool replayable;
    uint64_t checkpointAlignment{1};
};

std::vector<detail::AutodiffDagNode> valueDag(std::initializer_list<TestDagNode> nodes) {
    std::vector<detail::AutodiffDagNode> result;
    result.reserve(nodes.size());
    for (const TestDagNode &source : nodes) {
        detail::AutodiffDagNode node;
        node.predecessors = source.predecessors;
        node.residualBytes = source.residualBytes;
        node.retainedAllocationBytes = source.residualBytes;
        node.forwardPeakBytes = source.residualBytes;
        node.replayCost = source.replayCost;
        node.replayable = source.replayable;
        if (source.checkpointBytes)
            node.outputs.push_back({{static_cast<uint32_t>(result.size()), 1},
                                    source.checkpointBytes,
                                    source.checkpointAlignment,
                                    true,
                                    {}});
        result.push_back(std::move(node));
    }
    for (uint32_t consumer = 0; consumer < result.size(); ++consumer)
        for (uint32_t producer : result[consumer].predecessors)
            if (producer < result.size() && !result[producer].outputs.empty() &&
                std::find(result[producer].outputs[0].consumers.begin(), result[producer].outputs[0].consumers.end(),
                          consumer) == result[producer].outputs[0].consumers.end())
                result[producer].outputs[0].consumers.push_back(consumer);
    return result;
}

TEST(AutodiffCheckpointPlannerTest, PlansDagCutsFromLiveResources) {
    std::vector<detail::AutodiffDagNode> nodes = valueDag({
        {{}, 4, 10, 1, true, 4},
        {{0}, 3, 10, 2, true},
        {{0}, 5, 10, 3, true},
        {{1, 2}, 0, 10, 4, true},
    });
    nodes[0].resourceReloadCost = 2;
    nodes[1].recomputationCost = 3;
    nodes[0].requiredVersions.push_back({"primal.initial", {7, 0}, UINT32_MAX});
    nodes[3].requiredVersions.push_back({"primal.join", nodes[0].outputs[0].version, 0});
    AutodiffDagCheckpointPlan plan;
    std::string error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 27, plan, error, 0, true, 0, true, 0, true, 0,
                                                   detail::AutodiffCheckpointPolicy::MinRuntime))
        << error;
    EXPECT_FALSE(plan.cuts.empty());
    for (const AutodiffLivenessCut &cut : plan.cuts)
        for (uint32_t resource : cut.checkpointResources)
            EXPECT_LT(resource, plan.checkpointResources.size());
    EXPECT_LE(plan.peakBytes, 27u);
    EXPECT_EQ(plan.selectedPolicy, "min_runtime");
    EXPECT_EQ(plan.captureStoreBytes, 80u);
    EXPECT_EQ(plan.backwardLoadBytes, 40u);
    EXPECT_EQ(plan.checkpointCopyBytes, 2 * plan.persistentCheckpointBytes);
    EXPECT_EQ(plan.resourceReloadCost, 2u);
    EXPECT_EQ(plan.recomputationCost, 3u);
    EXPECT_EQ(plan.logicalResidualBytes, 40u);
    EXPECT_EQ(plan.retainedAllocationBytes, 40u);
    EXPECT_EQ(plan.maximumForwardPeakBytes, 10u);
    ASSERT_EQ(plan.passVersions.size(), nodes.size());
    ASSERT_EQ(plan.requiredVersions.size(), 2u);
    EXPECT_EQ(plan.requiredVersions[0].source, AutodiffVersionSource::RetainedOwner);
    EXPECT_EQ(plan.requiredVersions[1].source, AutodiffVersionSource::Checkpoint);
    EXPECT_EQ(plan.requiredVersions[1].version.resource, nodes[0].outputs[0].version.resource);
    EXPECT_EQ(plan.requiredVersions[1].version.epoch, nodes[0].outputs[0].version.epoch);
    EXPECT_EQ(plan.weightedRuntimeCost, plan.captureStoreBytes + plan.backwardLoadBytes + plan.checkpointCopyBytes +
                                            4 * plan.resourceReloadCost + 8 * plan.recomputationCost +
                                            16 * plan.replayCost);
}

TEST(AutodiffCheckpointPlannerTest, RejectsReplayThatViolatesDeterministicReductionConstraints) {
    std::vector<detail::AutodiffDagNode> nodes = valueDag({
        {{}, 4, 10, 1, true, 4},
        {{0}, 3, 10, 2, true},
        {{0}, 5, 10, 3, true},
        {{1, 2}, 0, 10, 4, true},
    });
    nodes[1].deterministicReductionLegal = false;
    AutodiffDagCheckpointPlan plan;
    std::string error;
    EXPECT_FALSE(detail::planDagAutodiffCheckpoints(nodes, 27, plan, error));
    EXPECT_EQ(error, "autodiff DAG checkpoint schedule violates deterministic reduction constraints");
}

TEST(AutodiffCheckpointPlannerTest, SupportsInternalMemoryAndRuntimePolicies) {
    const std::vector<detail::AutodiffDagNode> nodes = valueDag({
        {{}, 1, 10, 1, true},
        {{0}, 1, 10, 1, true},
        {{1}, 0, 10, 1, true},
    });
    AutodiffDagCheckpointPlan runtimePlan;
    AutodiffDagCheckpointPlan memoryPlan;
    AutodiffDagCheckpointPlan balancedPlan;
    std::string error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 64, runtimePlan, error, 0, true, 0, true, 0, true, 0,
                                                   detail::AutodiffCheckpointPolicy::MinRuntime))
        << error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 64, memoryPlan, error, 0, true, 0, true, 0, true, 0,
                                                   detail::AutodiffCheckpointPolicy::MinMemory))
        << error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 64, balancedPlan, error, 0, true, 0, true, 0, true, 0,
                                                   detail::AutodiffCheckpointPolicy::Balanced))
        << error;
    EXPECT_EQ(runtimePlan.selectedPolicy, "min_runtime");
    EXPECT_TRUE(runtimePlan.cuts.empty());
    EXPECT_EQ(memoryPlan.selectedPolicy, "min_memory");
    EXPECT_LT(memoryPlan.peakBytes, runtimePlan.peakBytes);
    EXPECT_GT(memoryPlan.cuts.size(), runtimePlan.cuts.size());
    EXPECT_EQ(balancedPlan.selectedPolicy, "balanced");
    const auto score = [](const AutodiffDagCheckpointPlan &plan) { return plan.peakBytes + plan.weightedRuntimeCost; };
    EXPECT_LE(score(balancedPlan), score(runtimePlan));
    EXPECT_LE(score(balancedPlan), score(memoryPlan));
}

TEST(AutodiffCheckpointPlannerTest, DoesNotCheckpointSchedulingOnlyPredecessors) {
    std::vector<detail::AutodiffDagNode> nodes =
        valueDag({{{}, 10, 100, 1, true}, {{0}, 10, 100, 1, true}, {{1}, 0, 100, 1, true}});
    nodes[0].outputs[0].consumers.clear();
    AutodiffDagCheckpointPlan plan;
    std::string error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 200, plan, error, 0, true, 0, true, 0, true, 0,
                                                   detail::AutodiffCheckpointPolicy::MinRuntime))
        << error;
    ASSERT_EQ(plan.cuts.size(), 1u);
    EXPECT_EQ(plan.cuts[0].scheduleOffset, 1u);
    EXPECT_TRUE(plan.checkpointResources.empty());
}

TEST(AutodiffCheckpointPlannerTest, CheckpointsOnlyLiveOutputs) {
    std::vector<detail::AutodiffDagNode> nodes =
        valueDag({{{}, 1, 100, 1, true}, {{0}, 0, 100, 1, true}, {{1}, 0, 100, 1, true}});
    nodes[0].outputs.push_back({{99, 1}, 1000, 8, true, {}});
    AutodiffDagCheckpointPlan plan;
    std::string error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 101, plan, error)) << error;
    ASSERT_EQ(plan.checkpointResources.size(), 1u);
    EXPECT_EQ(plan.checkpointResources[0].version.resource, 0u);
    EXPECT_EQ(plan.persistentCheckpointBytes, 1u);
}

TEST(AutodiffCheckpointPlannerTest, NoCutPlanAllocatesOnlyTransactionSnapshot) {
    const std::vector<detail::AutodiffDagNode> nodes = valueDag({{{}, 4, 16, 1, true}, {{0}, 4, 16, 1, true}});
    auto noCheckpointNodes = nodes;
    for (auto &node : noCheckpointNodes)
        for (auto &output : node.outputs)
            output.checkpointable = false;
    AutodiffDagCheckpointPlan plan;
    std::string error;
    ASSERT_TRUE(
        detail::planDagAutodiffCheckpoints(noCheckpointNodes, 160, plan, error, 64, false, 128, true, 128, true))
        << error;
    EXPECT_TRUE(plan.cuts.empty());
    EXPECT_EQ(plan.initialStateBytes, 0u);
    EXPECT_EQ(plan.restorationBytes, 0u);
    EXPECT_EQ(plan.transactionBytes, 128u);
    EXPECT_EQ(plan.peakBytes, 160u);
}

TEST(AutodiffCheckpointPlannerTest, TracksSharedCheckpointCutLifetimes) {
    const std::vector<detail::AutodiffDagNode> nodes = valueDag({
        {{}, 2, 10, 1, true},
        {{}, 1, 10, 1, true},
        {{1}, 1, 10, 1, true},
        {{2}, 1, 10, 1, true},
        {{0, 3}, 0, 10, 1, true},
    });
    AutodiffDagCheckpointPlan plan;
    std::string error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 24, plan, error, 0, true, 0, true, 0, true, 0,
                                                   detail::AutodiffCheckpointPolicy::MinRuntime))
        << error;
    ASSERT_EQ(plan.cuts.size(), 2u);
    ASSERT_FALSE(plan.checkpointResources.empty());
    EXPECT_TRUE(
        std::any_of(plan.checkpointResources.begin(), plan.checkpointResources.end(),
                    [](const AutodiffCheckpointResource &resource) { return resource.firstCut < resource.lastCut; }));
    for (const AutodiffCheckpointResource &resource : plan.checkpointResources)
        EXPECT_LE(resource.firstCut, resource.lastCut);
    ASSERT_EQ(plan.replaySegments.size(), 3u);
    for (const AutodiffReplaySegment &segment : plan.replaySegments)
        for (uint32_t resource : segment.releaseCheckpointResources)
            EXPECT_LT(resource, plan.checkpointResources.size());
    EXPECT_LE(plan.peakBytes, 24u);
}

TEST(AutodiffCheckpointPlannerTest, SplitsTiedDagPeakSegmentsTogether) {
    const std::vector<detail::AutodiffDagNode> nodes = valueDag({
        {{}, 2, 10, 1, true},
        {{0}, 2, 10, 1, true},
        {{1}, 2, 10, 1, true},
        {{2}, 0, 10, 1, true},
    });
    AutodiffDagCheckpointPlan plan;
    std::string error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 16, plan, error)) << error;
    EXPECT_EQ(plan.cuts.size(), 3u);
    EXPECT_EQ(plan.replaySegments.size(), 4u);
    EXPECT_EQ(plan.persistentCheckpointBytes, 6u);
    EXPECT_EQ(plan.peakBytes, 16u);
}

TEST(AutodiffCheckpointPlannerTest, AcceptsTemporaryPeakIncreaseNeededForFeasibleCuts) {
    std::vector<detail::AutodiffDagNode> nodes(3);
    for (auto &node : nodes) {
        node.residualBytes = 10;
        node.retainedAllocationBytes = 10;
        node.forwardPeakBytes = 10;
        node.replayCost = 1;
        node.replayable = true;
    }
    nodes[0].outputs.push_back({{0, 1}, 15, 1, true, {2}});
    nodes[2].predecessors.push_back(0);
    AutodiffDagCheckpointPlan plan;
    std::string error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 25, plan, error)) << error;
    EXPECT_EQ(plan.cuts.size(), 2u);
    EXPECT_EQ(plan.peakBytes, 25u);
    EXPECT_EQ(plan.persistentCheckpointBytes, 15u);
}

TEST(AutodiffCheckpointPlannerTest, ReplacesEarlierCutsToFindFeasibleFrontier) {
    const std::vector<detail::AutodiffDagNode> nodes = valueDag({
        {{}, 7, 15, 1, true},
        {{0}, 19, 18, 1, true},
        {{1}, 10, 13, 1, true},
        {{2}, 0, 20, 1, true},
    });
    AutodiffDagCheckpointPlan plan;
    std::string error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 48, plan, error)) << error;
    ASSERT_EQ(plan.cuts.size(), 2u);
    EXPECT_EQ(plan.cuts[0].scheduleOffset, 1u);
    EXPECT_EQ(plan.cuts[1].scheduleOffset, 3u);
    EXPECT_EQ(plan.peakBytes, 48u);
}

TEST(AutodiffCheckpointPlannerTest, ChargesPhysicalRetainedAllocationInsteadOfLogicalPayload) {
    std::vector<detail::AutodiffDagNode> nodes = valueDag({
        {{}, 0, 1, 1, true},
        {{0}, 0, 1, 1, true},
    });
    nodes[0].retainedAllocationBytes = 24;
    nodes[0].forwardPeakBytes = 24;
    nodes[1].retainedAllocationBytes = 24;
    nodes[1].forwardPeakBytes = 24;
    AutodiffDagCheckpointPlan plan;
    std::string error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 48, plan, error)) << error;
    EXPECT_EQ(plan.peakBytes, 48u);
    EXPECT_EQ(plan.replaySegments.front().retainedAllocationBytes, 48u);
    EXPECT_EQ(plan.replaySegments.front().logicalResidualBytes, 2u);
    EXPECT_EQ(plan.logicalResidualBytes, 2u);
    EXPECT_EQ(plan.retainedAllocationBytes, 48u);
}

TEST(AutodiffCheckpointPlannerTest, RejectsInvalidOrUnbudgetableDags) {
    AutodiffDagCheckpointPlan plan;
    std::string error;
    EXPECT_FALSE(detail::planDagAutodiffCheckpoints(valueDag({{{1}, 1, 1, 1, true}}), 1, plan, error));
    EXPECT_EQ(error, "autodiff DAG nodes are not in topological order");
    EXPECT_FALSE(
        detail::planDagAutodiffCheckpoints(valueDag({{{}, 1, 1, 1, true}, {{0, 0}, 1, 1, 1, true}}), 2, plan, error));
    EXPECT_EQ(error, "autodiff DAG node contains a duplicate predecessor");
    auto invalidVersion = valueDag({{{}, 1, 1, 1, true}, {{0}, 0, 1, 1, true}});
    invalidVersion[1].requiredVersions.push_back({"primal.missing", {999, 1}, 0});
    EXPECT_FALSE(detail::planDagAutodiffCheckpoints(invalidVersion, 2, plan, error));
    EXPECT_EQ(error, "autodiff DAG contains an invalid required resource version");
    EXPECT_FALSE(detail::planDagAutodiffCheckpoints(valueDag({{{}, 1, 1, 1, false}}), 0, plan, error));
    EXPECT_EQ(error, "autodiff DAG checkpoint schedule cannot satisfy the memory budget");

    const auto nonReplayableTail = valueDag({{{}, 1, 10, 1, true}, {{0}, 1, 10, 1, true}, {{1}, 0, 1, 1, false}});
    EXPECT_FALSE(detail::planDagAutodiffCheckpoints(nonReplayableTail, 12, plan, error));
    EXPECT_EQ(error, "autodiff DAG checkpoint schedule requires replaying a non-replayable node");

    const std::vector<detail::AutodiffDagNode> diamond = valueDag({
        {{}, 4, 10, 1, true},
        {{0}, 3, 10, 1, true},
        {{0}, 5, 10, 1, true},
        {{1, 2}, 0, 10, 1, true},
    });
    EXPECT_FALSE(detail::planDagAutodiffCheckpoints(diamond, 21, plan, error));
    EXPECT_EQ(error, "autodiff DAG checkpoint schedule cannot satisfy the memory budget");
}

} // namespace
