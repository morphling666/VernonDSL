#include "VernonRuntimeCore.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct MockProvider {
    uint64_t nextObject{1};
    uint32_t capabilities{};
    uint32_t shaderPreparations{};
    uint32_t groupLayoutPreparations{};
    uint32_t destroyedGroupLayouts{};
    uint32_t layoutPreparations{};
    uint32_t pipelinePreparations{};
    uint32_t failedPipelinePreparation{};
    uint32_t destroyedPipelines{};
    uint32_t groupCreations{};
    uint32_t destroyedGroups{};
    uint32_t pipelineBinds{};
    uint32_t dispatches{};
    uint32_t commandBegins{};
    uint32_t commandEnds{};
    VernonRuntimeProviderObject lastEncoder{};
    VernonRuntimeProviderGroupLifetime lastGroupLifetime{};
    VernonRuntimeProviderDispatchDescriptor lastDispatch{};
    VernonRuntimeProviderDrawDescriptor lastDraw{};
    // The index and group of every encode_bind_group, in order.
    std::vector<std::pair<uint32_t, uint64_t>> groupBinds;
    uint32_t renderingBegins{};
    uint32_t renderingEnds{};
    size_t lastRenderingColorCount{};
    size_t lastRenderingSampledImageCount{};
    bool failNextGroupCreation{};
};

VernonRuntimeProviderObject next(MockProvider &mock) { return {mock.nextObject++}; }

MockProvider &state(void *data) { return *static_cast<MockProvider *>(data); }

VernonRuntimeDeviceProvider makeProvider(MockProvider &mock, uint32_t capabilities) {
    mock.capabilities = capabilities;
    VernonRuntimeDeviceProvider provider{};
    provider.struct_size = sizeof(provider);
    provider.abi_version = VERNON_PROGRAM_VERSION;
    provider.user_data = &mock;
    provider.get_capabilities = [](void *data) -> uint32_t { return state(data).capabilities; };
    provider.get_device_identity = [](void *) -> VernonRuntimeProviderDeviceIdentity { return {11, 22, 33}; };
    provider.prepare_shader = [](void *data, const VernonRuntimeProviderShaderDescriptor *,
                                 VernonRuntimeProviderObject *shader) {
        ++state(data).shaderPreparations;
        *shader = next(state(data));
        return VERNON_STATUS_OK;
    };
    provider.prepare_group_layout = [](void *data, const VernonRuntimeProviderGroupLayoutDescriptor *,
                                       VernonRuntimeProviderObject *layout) {
        ++state(data).groupLayoutPreparations;
        *layout = next(state(data));
        return VERNON_STATUS_OK;
    };
    provider.prepare_pipeline_layout = [](void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *,
                                          VernonRuntimeProviderObject *layout) {
        ++state(data).layoutPreparations;
        *layout = next(state(data));
        return VERNON_STATUS_OK;
    };
    provider.prepare_pipeline = [](void *data, const VernonRuntimeProviderPipelineDescriptor *,
                                   VernonRuntimeProviderObject *pipeline) {
        MockProvider &mock = state(data);
        ++mock.pipelinePreparations;
        *pipeline = next(mock);
        return mock.pipelinePreparations == mock.failedPipelinePreparation ? VERNON_STATUS_INTERNAL_ERROR
                                                                           : VERNON_STATUS_OK;
    };
    provider.describe_image = [](void *, VernonRuntimeProviderResourceReference resource,
                                 VernonRuntimeProviderImageDescription *description) {
        if (!description || description->struct_size < sizeof(*description))
            return VERNON_STATUS_INVALID_ARGUMENT;
        description->image = {VERNON_TEXTURE_2D,
                              {16, 16, 1},
                              VERNON_TEXTURE_RGBA8_UNORM,
                              1,
                              1,
                              1,
                              VERNON_IMAGE_SAMPLED | VERNON_IMAGE_STORAGE};
        description->view = {VERNON_TEXTURE_2D, VERNON_TEXTURE_RGBA8_UNORM, {0, 1, 0, 1, VERNON_IMAGE_ASPECT_COLOR}};
        description->parent_identity = 100;
        description->resource_kind = VERNON_RUNTIME_PROVIDER_IMAGE_VIEW;
        if (resource.identity == 101)
            description->view.dimension = VERNON_TEXTURE_3D;
        else if (resource.identity == 102)
            description->image.usage = VERNON_IMAGE_SAMPLED;
        else if (resource.identity == 103) {
            description->image.format = VERNON_TEXTURE_D32_FLOAT;
            description->image.usage = VERNON_IMAGE_SAMPLED;
            description->view.format = VERNON_TEXTURE_D32_FLOAT;
            description->view.subresources.aspects = VERNON_IMAGE_ASPECT_DEPTH;
        } else if (resource.identity == 104)
            description->image.usage = VERNON_IMAGE_COLOR_ATTACHMENT;
        else if (resource.identity == 105) {
            description->image.format = VERNON_TEXTURE_D32_FLOAT;
            description->image.usage = VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT;
            description->view.format = VERNON_TEXTURE_D32_FLOAT;
            description->view.subresources.aspects = VERNON_IMAGE_ASPECT_DEPTH;
        } else if (resource.identity == 106) {
            description->image.usage = VERNON_IMAGE_COLOR_ATTACHMENT;
            description->resource_kind = VERNON_RUNTIME_PROVIDER_IMAGE_OWNER;
        } else if (resource.identity == 107) {
            description->view.format = VERNON_TEXTURE_D32_FLOAT;
            description->view.subresources.aspects = VERNON_IMAGE_ASPECT_DEPTH;
        }
        if (resource.identity >= 104)
            description->parent_identity = resource.identity;
        if (resource.identity == 108) {
            description->image.usage = VERNON_IMAGE_COLOR_ATTACHMENT | VERNON_IMAGE_SAMPLED;
            description->parent_identity = 104;
        } else if (resource.identity == 109) {
            description->image.format = VERNON_TEXTURE_R32_UINT;
            description->image.usage = VERNON_IMAGE_COLOR_ATTACHMENT;
            description->view.format = VERNON_TEXTURE_R32_UINT;
        }
        return VERNON_STATUS_OK;
    };
    provider.create_group = [](void *data, const VernonRuntimeProviderGroupDescriptor *descriptor,
                               VernonRuntimeProviderObject *group) {
        MockProvider &mock = state(data);
        ++mock.groupCreations;
        mock.lastGroupLifetime = descriptor->lifetime;
        *group = next(mock);
        return std::exchange(mock.failNextGroupCreation, false) ? VERNON_STATUS_INTERNAL_ERROR : VERNON_STATUS_OK;
    };
    provider.begin_commands = [](void *data, VernonRuntimeProviderObject encoder,
                                 VernonRuntimeProviderObject *commands) {
        MockProvider &mock = state(data);
        ++mock.commandBegins;
        mock.lastEncoder = encoder;
        *commands = next(mock);
        return VERNON_STATUS_OK;
    };
    provider.end_commands = [](void *data, VernonRuntimeProviderObject) {
        ++state(data).commandEnds;
        return VERNON_STATUS_OK;
    };
    provider.encode_bind_pipeline = [](void *data, VernonRuntimeProviderObject, VernonRuntimeProviderObject) {
        ++state(data).pipelineBinds;
        return VERNON_STATUS_OK;
    };
    provider.encode_bind_group = [](void *data, VernonRuntimeProviderObject, uint32_t index,
                                    VernonRuntimeProviderObject group) {
        state(data).groupBinds.emplace_back(index, group.value);
        return VERNON_STATUS_OK;
    };
    provider.encode_dispatch = [](void *data, VernonRuntimeProviderObject,
                                  const VernonRuntimeProviderDispatchDescriptor *descriptor) {
        ++state(data).dispatches;
        state(data).lastDispatch = *descriptor;
        return VERNON_STATUS_OK;
    };
    provider.encode_draw = [](void *data, VernonRuntimeProviderObject,
                              const VernonRuntimeProviderDrawDescriptor *draw) {
        state(data).lastDraw = *draw;
        return VERNON_STATUS_OK;
    };
    provider.begin_rendering = [](void *data, VernonRuntimeProviderObject,
                                  const VernonRuntimeProviderRenderingDescriptor *rendering) {
        MockProvider &mock = state(data);
        ++mock.renderingBegins;
        mock.lastRenderingColorCount = rendering->color_attachment_count;
        mock.lastRenderingSampledImageCount = rendering->sampled_image_count;
        return VERNON_STATUS_OK;
    };
    provider.end_rendering = [](void *data, VernonRuntimeProviderObject) {
        ++state(data).renderingEnds;
        return VERNON_STATUS_OK;
    };
    provider.destroy_shader = [](void *, VernonRuntimeProviderObject) {};
    provider.destroy_group_layout = [](void *data, VernonRuntimeProviderObject) {
        ++state(data).destroyedGroupLayouts;
    };
    provider.destroy_pipeline_layout = [](void *, VernonRuntimeProviderObject) {};
    provider.destroy_pipeline = [](void *data, VernonRuntimeProviderObject) { ++state(data).destroyedPipelines; };
    provider.destroy_group = [](void *data, VernonRuntimeProviderObject) { ++state(data).destroyedGroups; };
    return provider;
}

const VernonRuntimeProviderBindingLayoutEntry kStorageEntry{
    0, 0, 0, VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER, VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE, 3, 1};

const std::array<VernonRuntimeProviderBindingLayoutEntry, 2> &imageEntries() {
    static const std::array<VernonRuntimeProviderBindingLayoutEntry, 2> entries = [] {
        std::array<VernonRuntimeProviderBindingLayoutEntry, 2> result{};
        result[0].slot = 0;
        result[0].kind = VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE;
        result[0].stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
        result[0].access = 1;
        result[0].array_count = 1;
        result[0].image_dimension = VERNON_TEXTURE_2D;
        result[0].sample_result_class = VERNON_IMAGE_SAMPLE_FLOAT;
        result[1].slot = 1;
        result[1].kind = VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
        result[1].stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
        result[1].access = 2;
        result[1].array_count = 1;
        result[1].image_dimension = VERNON_TEXTURE_2D;
        result[1].storage_image_format = VERNON_TEXTURE_RGBA8_UNORM;
        return result;
    }();
    return entries;
}

VernonRuntimeProviderBindingValue storageValue(uint64_t identity) {
    VernonRuntimeProviderBindingValue value{};
    value.slot = 0;
    value.kind = VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
    value.payload.buffer.resource = {identity, {identity + 1}, 0, 4096};
    return value;
}

std::array<VernonRuntimeProviderBindingValue, 2> imageValues() {
    std::array<VernonRuntimeProviderBindingValue, 2> values{};
    values[0].slot = 0;
    values[0].kind = VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE;
    values[0].payload.image.view = {100, {1}, 0, 0};
    values[1].slot = 1;
    values[1].kind = VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
    values[1].payload.image.view = {100, {2}, 0, 0};
    return values;
}

VernonRuntimeCorePipelineDescriptor computePipelineDescriptor(VernonRuntimeCoreGroupLayout *const *groupLayouts,
                                                              size_t groupLayoutCount) {
    static constexpr char source[] = ".version 8.0";
    static const VernonRuntimeProviderShaderDescriptor shader{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                              VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE,
                                                              {"ptx", 3},
                                                              source,
                                                              sizeof(source) - 1,
                                                              {"main", 4},
                                                              {"0123456789abcdef", 16},
                                                              {0, 0, 0, 0}};
    VernonRuntimeCorePipelineDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.kind = VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
    descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_COMPUTE;
    descriptor.shaders = &shader;
    descriptor.shader_count = 1;
    descriptor.group_layouts = groupLayouts;
    descriptor.group_layout_count = groupLayoutCount;
    descriptor.workgroup_size[0] = 8;
    descriptor.workgroup_size[1] = 1;
    descriptor.workgroup_size[2] = 1;
    return descriptor;
}

VernonRuntimeCorePipelineDescriptor graphicsPipelineDescriptor() {
    static constexpr char source[] = "shader";
    static const VernonRuntimeProviderShaderDescriptor shaders[]{{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                                  VERNON_RUNTIME_PROVIDER_STAGE_VERTEX,
                                                                  {"mock", 4},
                                                                  source,
                                                                  sizeof(source) - 1,
                                                                  {"vertex", 6},
                                                                  {},
                                                                  {0, 0, 0, 0}},
                                                                 {sizeof(VernonRuntimeProviderShaderDescriptor),
                                                                  VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT,
                                                                  {"mock", 4},
                                                                  source,
                                                                  sizeof(source) - 1,
                                                                  {"fragment", 8},
                                                                  {},
                                                                  {0, 0, 0, 0}}};
    VernonRuntimeCorePipelineDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.kind = VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE;
    descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_GRAPHICS;
    descriptor.shaders = shaders;
    descriptor.shader_count = 2;
    descriptor.sample_count = 1;
    return descriptor;
}

VernonRuntimeCoreGraphicsCompatibility singleColorCompatibility() {
    static const uint32_t formats[]{3};
    static const VernonColorBlendState blend = [] {
        VernonColorBlendState result{};
        result.write_mask = VERNON_RHI_COLOR_WRITE_ALL;
        return result;
    }();
    VernonRuntimeCoreGraphicsCompatibility compatibility{};
    compatibility.struct_size = sizeof(compatibility);
    compatibility.color_formats = formats;
    compatibility.color_format_count = std::size(formats);
    compatibility.sample_count = 1;
    compatibility.color_blends = &blend;
    compatibility.color_blend_count = 1;
    return compatibility;
}

VernonRuntimeProviderRenderingDescriptor colorRendering(const VernonRuntimeProviderColorAttachment &color) {
    VernonRuntimeProviderRenderingDescriptor rendering{};
    rendering.struct_size = sizeof(rendering);
    rendering.color_attachments = &color;
    rendering.color_attachment_count = 1;
    rendering.render_area[2] = 16;
    rendering.render_area[3] = 16;
    rendering.layers = 1;
    return rendering;
}

VernonRuntimeCoreGroupLayout *storageLayout(const VernonRuntimeDeviceProvider &provider) {
    VernonRuntimeCoreGroupLayout *layout = nullptr;
    EXPECT_EQ(vernonRuntimeCorePrepareGroupLayout(&provider, &kStorageEntry, 1, &layout), VERNON_STATUS_OK);
    return layout;
}

VernonRuntimeCoreGroup *persistentGroup(VernonRuntimeCoreGroupLayout *layout,
                                        const VernonRuntimeProviderBindingValue &value) {
    VernonRuntimeCoreGroup *group = nullptr;
    EXPECT_EQ(vernonRuntimeCoreCreateGroup(layout, &value, 1, VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT, {}, &group),
              VERNON_STATUS_OK);
    return group;
}

TEST(RuntimeForeignProvider, RejectsProvidersWithoutLifecycleCallbacks) {
    MockProvider mock;
    VernonRuntimeCoreGroupLayout *layout = nullptr;
    VernonRuntimeCorePipeline *pipeline = nullptr;
    const VernonRuntimeCorePipelineDescriptor descriptor = computePipelineDescriptor(nullptr, 0);

    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    provider.destroy_group = nullptr;
    EXPECT_EQ(vernonRuntimeCorePrepareGroupLayout(&provider, &kStorageEntry, 1, &layout),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_INVALID_ARGUMENT);
    provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    provider.end_commands = nullptr;
    EXPECT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(layout, nullptr);
    EXPECT_EQ(pipeline, nullptr);
    EXPECT_EQ(mock.groupLayoutPreparations, 0u);
}

TEST(RuntimeForeignProvider, ContainsProviderExceptionsAtTheCAbiBoundary) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    const VernonRuntimeCorePipelineDescriptor descriptor = computePipelineDescriptor(nullptr, 0);
    VernonRuntimeCorePipeline *pipeline = nullptr;

    provider.prepare_shader = [](void *, const VernonRuntimeProviderShaderDescriptor *,
                                 VernonRuntimeProviderObject *) -> VernonStatus {
        throw std::runtime_error("foreign provider failure");
    };
    EXPECT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(pipeline, nullptr);

    provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    provider.destroy_shader = [](void *, VernonRuntimeProviderObject) { throw std::runtime_error("cleanup failure"); };
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    EXPECT_NO_THROW(vernonRuntimeCorePipelineDestroy(pipeline));
}

TEST(RuntimeForeignProvider, ContainsEveryFallibleProviderCallbackClass) {
    MockProvider mock;
    const VernonRuntimeCorePipelineDescriptor descriptor = computePipelineDescriptor(nullptr, 0);
    VernonRuntimeCorePipeline *pipeline = nullptr;

    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    provider.get_capabilities = [](void *) -> uint32_t { throw std::runtime_error("capabilities"); };
    EXPECT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(pipeline, nullptr);

    provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    provider.get_device_identity = [](void *) -> VernonRuntimeProviderDeviceIdentity {
        throw std::runtime_error("identity");
    };
    EXPECT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(pipeline, nullptr);

    MockProvider layoutMock;
    provider = makeProvider(layoutMock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    provider.prepare_group_layout = [](void *, const VernonRuntimeProviderGroupLayoutDescriptor *,
                                       VernonRuntimeProviderObject *) -> VernonStatus {
        throw std::runtime_error("group layout");
    };
    VernonRuntimeCoreGroupLayout *layout = nullptr;
    EXPECT_EQ(vernonRuntimeCorePrepareGroupLayout(&provider, &kStorageEntry, 1, &layout), VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(layout, nullptr);

    MockProvider groupMock;
    provider = makeProvider(groupMock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    provider.create_group = [](void *, const VernonRuntimeProviderGroupDescriptor *,
                               VernonRuntimeProviderObject *) -> VernonStatus { throw std::runtime_error("group"); };
    layout = storageLayout(provider);
    ASSERT_NE(layout, nullptr);
    const VernonRuntimeProviderBindingValue value = storageValue(99);
    VernonRuntimeCoreGroup *group = nullptr;
    EXPECT_EQ(vernonRuntimeCoreCreateGroup(layout, &value, 1, VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT, {}, &group),
              VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(group, nullptr);
    vernonRuntimeCoreGroupLayoutDestroy(layout);
}

TEST(RuntimeForeignProvider, ContainsEncodeAndDescriptionCallbackExceptions) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    provider.encode_dispatch = [](void *, VernonRuntimeProviderObject,
                                  const VernonRuntimeProviderDispatchDescriptor *) -> VernonStatus {
        throw std::runtime_error("dispatch");
    };
    VernonRuntimeCoreGroupLayout *layout = storageLayout(provider);
    ASSERT_NE(layout, nullptr);
    const VernonRuntimeCorePipelineDescriptor descriptor = computePipelineDescriptor(&layout, 1);
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    VernonRuntimeCoreGroup *group = persistentGroup(layout, storageValue(99));
    ASSERT_NE(group, nullptr);

    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {200}, &commands), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCoreBindPipeline(commands, pipeline), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCoreBindGroup(commands, 0, group), VERNON_STATUS_OK);
    const uint32_t groups[3]{1, 1, 1};
    EXPECT_EQ(vernonRuntimeCoreDispatch(commands, groups, nullptr, 0), VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
    vernonRuntimeCoreGroupDestroy(group);
    vernonRuntimeCorePipelineDestroy(pipeline);
    vernonRuntimeCoreGroupLayoutDestroy(layout);

    MockProvider imageMock;
    provider = makeProvider(imageMock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    provider.describe_image = [](void *, VernonRuntimeProviderResourceReference,
                                 VernonRuntimeProviderImageDescription *) -> VernonStatus {
        throw std::runtime_error("describe");
    };
    ASSERT_EQ(vernonRuntimeCorePrepareGroupLayout(&provider, imageEntries().data(), imageEntries().size(), &layout),
              VERNON_STATUS_OK);
    const std::array<VernonRuntimeProviderBindingValue, 2> images = imageValues();
    EXPECT_EQ(vernonRuntimeCoreCreateGroup(layout, images.data(), images.size(),
                                           VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT, {}, &group),
              VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(group, nullptr);
    EXPECT_EQ(imageMock.groupCreations, 0u);
    vernonRuntimeCoreGroupLayoutDestroy(layout);
}

TEST(RuntimeForeignProvider, PreparesBindsAndDispatchesThroughCommandState) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    VernonRuntimeCoreGroupLayout *layout = storageLayout(provider);
    ASSERT_NE(layout, nullptr);
    const VernonRuntimeCorePipelineDescriptor descriptor = computePipelineDescriptor(&layout, 1);
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePipelineGetDeviceIdentity(pipeline).device_id, 22u);
    VernonRuntimeCoreGroup *group = persistentGroup(layout, storageValue(99));
    ASSERT_NE(group, nullptr);
    EXPECT_EQ(mock.lastGroupLifetime, VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT);

    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {200}, &commands), VERNON_STATUS_OK);
    EXPECT_EQ(mock.lastEncoder.value, 200u);
    const uint32_t groups[3]{4, 2, 1};
    EXPECT_EQ(vernonRuntimeCoreDispatch(commands, groups, nullptr, 0), VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeCoreBindPipeline(commands, pipeline), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreDispatch(commands, groups, nullptr, 0), VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeCoreBindGroup(commands, 0, group), VERNON_STATUS_OK);
    for (int repeat = 0; repeat < 3; ++repeat) {
        ASSERT_EQ(vernonRuntimeCoreBindPipeline(commands, pipeline), VERNON_STATUS_OK);
        ASSERT_EQ(vernonRuntimeCoreBindGroup(commands, 0, group), VERNON_STATUS_OK);
        EXPECT_EQ(vernonRuntimeCoreDispatch(commands, groups, nullptr, 0), VERNON_STATUS_OK);
    }
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);

    EXPECT_EQ(mock.shaderPreparations, 1u);
    EXPECT_EQ(mock.groupLayoutPreparations, 1u);
    EXPECT_EQ(mock.layoutPreparations, 1u);
    EXPECT_EQ(mock.pipelinePreparations, 1u);
    EXPECT_EQ(mock.groupCreations, 1u);
    EXPECT_EQ(mock.pipelineBinds, 1u);
    EXPECT_EQ(mock.groupBinds.size(), 1u);
    EXPECT_EQ(mock.dispatches, 3u);
    EXPECT_EQ(mock.lastDispatch.group_count[0], 4u);
    EXPECT_EQ(mock.commandBegins, 1u);
    EXPECT_EQ(mock.commandEnds, 1u);

    vernonRuntimeCorePipelineDestroy(pipeline);
    vernonRuntimeCoreGroupDestroy(group);
    vernonRuntimeCoreGroupLayoutDestroy(layout);
    EXPECT_EQ(mock.destroyedGroups, 1u);
    EXPECT_EQ(mock.destroyedGroupLayouts, 1u);
}

TEST(RuntimeForeignProvider, EqualEntriesShareOneGroupLayout) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    VernonRuntimeCoreGroupLayout *first = storageLayout(provider);
    VernonRuntimeCoreGroupLayout *second = storageLayout(provider);
    VernonRuntimeProviderBindingLayoutEntry writable = kStorageEntry;
    writable.access = 2;
    VernonRuntimeCoreGroupLayout *other = nullptr;
    ASSERT_EQ(vernonRuntimeCorePrepareGroupLayout(&provider, &writable, 1, &other), VERNON_STATUS_OK);
    EXPECT_EQ(first, second);
    EXPECT_NE(first, other);
    EXPECT_EQ(mock.groupLayoutPreparations, 2u);

    MockProvider otherMock;
    VernonRuntimeDeviceProvider otherProvider = makeProvider(otherMock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    VernonRuntimeCoreGroupLayout *foreign = storageLayout(otherProvider);
    EXPECT_NE(foreign, first);

    vernonRuntimeCoreGroupLayoutDestroy(first);
    EXPECT_EQ(mock.destroyedGroupLayouts, 0u);
    vernonRuntimeCoreGroupLayoutDestroy(second);
    vernonRuntimeCoreGroupLayoutDestroy(other);
    vernonRuntimeCoreGroupLayoutDestroy(foreign);
    EXPECT_EQ(mock.destroyedGroupLayouts, 2u);
    EXPECT_EQ(otherMock.destroyedGroupLayouts, 1u);
}

TEST(RuntimeForeignProvider, PipelineSwitchKeepsGroupsBoundOnlyAtSharedLayouts) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    VernonRuntimeCoreGroupLayout *shared = storageLayout(provider);
    VernonRuntimeProviderBindingLayoutEntry writable = kStorageEntry;
    writable.access = 2;
    VernonRuntimeCoreGroupLayout *distinct = nullptr;
    ASSERT_EQ(vernonRuntimeCorePrepareGroupLayout(&provider, &writable, 1, &distinct), VERNON_STATUS_OK);
    VernonRuntimeCoreGroupLayout *const firstLayouts[]{shared, shared};
    VernonRuntimeCoreGroupLayout *const secondLayouts[]{shared, distinct};
    const VernonRuntimeCorePipelineDescriptor firstDescriptor = computePipelineDescriptor(firstLayouts, 2);
    const VernonRuntimeCorePipelineDescriptor secondDescriptor = computePipelineDescriptor(secondLayouts, 2);
    VernonRuntimeCorePipeline *first = nullptr;
    VernonRuntimeCorePipeline *second = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &firstDescriptor, &first), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &secondDescriptor, &second), VERNON_STATUS_OK);
    VernonRuntimeCoreGroup *sharedGroup = persistentGroup(shared, storageValue(99));
    VernonRuntimeCoreGroup *distinctGroup = persistentGroup(distinct, storageValue(199));
    ASSERT_NE(sharedGroup, nullptr);
    ASSERT_NE(distinctGroup, nullptr);

    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {200}, &commands), VERNON_STATUS_OK);
    const uint32_t groups[3]{1, 1, 1};
    ASSERT_EQ(vernonRuntimeCoreBindPipeline(commands, first), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreBindGroup(commands, 1, distinctGroup), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRuntimeCoreBindGroup(commands, 2, sharedGroup), VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeCoreBindGroup(commands, 0, sharedGroup), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCoreBindGroup(commands, 1, sharedGroup), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreDispatch(commands, groups, nullptr, 0), VERNON_STATUS_OK);

    ASSERT_EQ(vernonRuntimeCoreBindPipeline(commands, second), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreDispatch(commands, groups, nullptr, 0), VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeCoreBindGroup(commands, 1, distinctGroup), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreDispatch(commands, groups, nullptr, 0), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);

    ASSERT_EQ(mock.groupBinds.size(), 3u);
    EXPECT_EQ(mock.groupBinds[0].first, 0u);
    EXPECT_EQ(mock.groupBinds[1].first, 1u);
    EXPECT_EQ(mock.groupBinds[2].first, 1u);
    EXPECT_EQ(mock.groupBinds[0].second, mock.groupBinds[1].second);
    EXPECT_NE(mock.groupBinds[1].second, mock.groupBinds[2].second);
    EXPECT_EQ(mock.dispatches, 2u);

    vernonRuntimeCoreGroupDestroy(distinctGroup);
    vernonRuntimeCoreGroupDestroy(sharedGroup);
    vernonRuntimeCorePipelineDestroy(second);
    vernonRuntimeCorePipelineDestroy(first);
    vernonRuntimeCoreGroupLayoutDestroy(distinct);
    vernonRuntimeCoreGroupLayoutDestroy(shared);
}

TEST(RuntimeForeignProvider, BoundGroupOutlivesItsDestroyUntilCommandsEnd) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    VernonRuntimeCoreGroupLayout *layout = storageLayout(provider);
    const VernonRuntimeCorePipelineDescriptor descriptor = computePipelineDescriptor(&layout, 1);
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    VernonRuntimeCoreGroup *group = persistentGroup(layout, storageValue(99));
    ASSERT_NE(group, nullptr);

    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {200}, &commands), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCoreBindPipeline(commands, pipeline), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCoreBindGroup(commands, 0, group), VERNON_STATUS_OK);
    vernonRuntimeCoreGroupDestroy(group);
    vernonRuntimeCorePipelineDestroy(pipeline);
    vernonRuntimeCoreGroupLayoutDestroy(layout);
    EXPECT_EQ(mock.destroyedGroups, 0u);
    EXPECT_EQ(mock.destroyedPipelines, 0u);
    const uint32_t groups[3]{1, 1, 1};
    EXPECT_EQ(vernonRuntimeCoreDispatch(commands, groups, nullptr, 0), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
    EXPECT_EQ(mock.destroyedGroups, 1u);
    EXPECT_EQ(mock.destroyedPipelines, 1u);
    EXPECT_EQ(mock.destroyedGroupLayouts, 1u);
}

TEST(RuntimeForeignProvider, TransientGroupsBindOnlyOnTheirCommandEncoder) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    VernonRuntimeCoreGroupLayout *layout = storageLayout(provider);
    const VernonRuntimeCorePipelineDescriptor descriptor = computePipelineDescriptor(&layout, 1);
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    const VernonRuntimeProviderBindingValue value = storageValue(99);
    VernonRuntimeCoreGroup *group = nullptr;
    EXPECT_EQ(vernonRuntimeCoreCreateGroup(layout, &value, 1, VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT, {7}, &group),
              VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeCoreCreateGroup(layout, &value, 1, VERNON_RUNTIME_PROVIDER_GROUP_TRANSIENT, {7}, &group),
              VERNON_STATUS_OK);
    EXPECT_EQ(mock.lastGroupLifetime, VERNON_RUNTIME_PROVIDER_GROUP_TRANSIENT);

    for (const uint64_t encoder : {8u, 7u}) {
        VernonRuntimeCoreCommands *commands = nullptr;
        ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {encoder}, &commands), VERNON_STATUS_OK);
        ASSERT_EQ(vernonRuntimeCoreBindPipeline(commands, pipeline), VERNON_STATUS_OK);
        EXPECT_EQ(vernonRuntimeCoreBindGroup(commands, 0, group),
                  encoder == 7 ? VERNON_STATUS_OK : VERNON_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
    }
    EXPECT_EQ(mock.groupBinds.size(), 1u);

    vernonRuntimeCoreGroupDestroy(group);
    vernonRuntimeCorePipelineDestroy(pipeline);
    vernonRuntimeCoreGroupLayoutDestroy(layout);
}

TEST(RuntimeForeignProvider, FailedGroupCreationPublishesNothing) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    VernonRuntimeCoreGroupLayout *layout = storageLayout(provider);
    ASSERT_NE(layout, nullptr);
    const VernonRuntimeProviderBindingValue value = storageValue(99);
    VernonRuntimeCoreGroup *group = nullptr;
    mock.failNextGroupCreation = true;
    EXPECT_EQ(vernonRuntimeCoreCreateGroup(layout, &value, 1, VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT, {}, &group),
              VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(group, nullptr);
    EXPECT_EQ(mock.destroyedGroups, 1u);

    VernonRuntimeProviderBindingValue mismatched = value;
    mismatched.kind = VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
    EXPECT_EQ(
        vernonRuntimeCoreCreateGroup(layout, &mismatched, 1, VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT, {}, &group),
        VERNON_STATUS_INVALID_ARGUMENT);
    VernonRuntimeProviderBindingValue unnamed = value;
    unnamed.payload.buffer.resource.identity = 0;
    EXPECT_EQ(vernonRuntimeCoreCreateGroup(layout, &unnamed, 1, VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT, {}, &group),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRuntimeCoreCreateGroup(layout, &value, 0, VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT, {}, &group),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(mock.groupCreations, 1u);
    vernonRuntimeCoreGroupLayoutDestroy(layout);
    EXPECT_EQ(mock.destroyedGroupLayouts, 1u);
}

TEST(RuntimeForeignProvider, RejectsCapabilitiesBeforePreparation) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_GRAPHICS);
    const VernonRuntimeCorePipelineDescriptor descriptor = computePipelineDescriptor(nullptr, 0);
    VernonRuntimeCorePipeline *pipeline = nullptr;
    EXPECT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_UNSUPPORTED_TARGET);
    EXPECT_EQ(pipeline, nullptr);
    EXPECT_EQ(mock.shaderPreparations, 0u);
    EXPECT_EQ(mock.layoutPreparations, 0u);
    EXPECT_EQ(mock.pipelinePreparations, 0u);
}

TEST(RuntimeForeignProvider, ValidatesSampledAndStorageImagesFromProviderDescriptors) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    VernonRuntimeCoreGroupLayout *layout = nullptr;
    ASSERT_EQ(vernonRuntimeCorePrepareGroupLayout(&provider, imageEntries().data(), imageEntries().size(), &layout),
              VERNON_STATUS_OK);
    std::array<VernonRuntimeProviderBindingValue, 2> values = imageValues();
    const auto create = [&] {
        VernonRuntimeCoreGroup *group = nullptr;
        const VernonStatus status = vernonRuntimeCoreCreateGroup(layout, values.data(), values.size(),
                                                                 VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT, {}, &group);
        vernonRuntimeCoreGroupDestroy(group);
        return status;
    };

    EXPECT_EQ(create(), VERNON_STATUS_OK);
    values[0].payload.image.view.identity = 101;
    EXPECT_EQ(create(), VERNON_STATUS_INVALID_ARGUMENT);
    // Ordinary sampling of a depth-format view still produces floating-point shader values.
    values[0].payload.image.view.identity = 103;
    EXPECT_EQ(create(), VERNON_STATUS_OK);
    values[0].payload.image.view.identity = 107;
    EXPECT_EQ(create(), VERNON_STATUS_INVALID_ARGUMENT);
    values[0].payload.image.view.identity = 100;
    values[1].payload.image.view.identity = 102;
    EXPECT_EQ(create(), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(mock.groupCreations, 2u);
    EXPECT_EQ(mock.destroyedGroups, 2u);
    vernonRuntimeCoreGroupLayoutDestroy(layout);
}

TEST(RuntimeForeignProvider, CachesGraphicsVariantsByCompatibility) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_GRAPHICS);
    const VernonRuntimeCorePipelineDescriptor descriptor = graphicsPipelineDescriptor();
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    ASSERT_NE(pipeline, nullptr);
    EXPECT_EQ(mock.pipelinePreparations, 1u);

    const uint32_t strides[]{12};
    VernonRuntimeCoreGraphicsCompatibility compatibility = singleColorCompatibility();
    compatibility.vertex_strides = strides;
    compatibility.vertex_stride_count = std::size(strides);
    VernonRuntimeCoreGraphicsVariant *first = nullptr;
    VernonRuntimeCoreGraphicsVariant *second = nullptr;
    ASSERT_EQ(vernonRuntimeCorePrepareGraphicsVariant(pipeline, &compatibility, &first), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePrepareGraphicsVariant(pipeline, &compatibility, &second), VERNON_STATUS_OK);
    EXPECT_EQ(mock.pipelinePreparations, 2u);

    VernonRuntimeProviderColorAttachment color{};
    color.view = {104, {1}, 0, 0};
    const VernonRuntimeProviderRenderingDescriptor rendering = colorRendering(color);
    VernonRuntimeCoreDrawInvocation draw{};
    draw.struct_size = sizeof(draw);
    draw.vertex_count = 3;
    draw.instance_count = 1;
    draw.stencil_reference = 17;
    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {7}, &commands), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCoreBindGraphicsVariant(commands, first), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreDraw(commands, &draw), VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeCoreBeginRendering(commands, &rendering), VERNON_STATUS_OK);
    draw.stencil_reference = 0x100;
    EXPECT_EQ(vernonRuntimeCoreDraw(commands, &draw), VERNON_STATUS_INVALID_ARGUMENT);
    draw.stencil_reference = 17;
    ASSERT_EQ(vernonRuntimeCoreDraw(commands, &draw), VERNON_STATUS_OK);
    EXPECT_EQ(mock.lastDraw.stencil_reference, 17u);
    ASSERT_EQ(vernonRuntimeCoreBindGraphicsVariant(commands, second), VERNON_STATUS_OK);
    EXPECT_EQ(mock.pipelineBinds, 1u);
    EXPECT_EQ(vernonRuntimeCoreEndRendering(commands), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);

    VernonRuntimeCoreGraphicsCompatibility changed = compatibility;
    const uint32_t changedStrides[]{16};
    changed.vertex_strides = changedStrides;
    VernonRuntimeCoreGraphicsVariant *third = nullptr;
    ASSERT_EQ(vernonRuntimeCorePrepareGraphicsVariant(pipeline, &changed, &third), VERNON_STATUS_OK);
    EXPECT_EQ(mock.pipelinePreparations, 3u);

    vernonRuntimeCoreGraphicsVariantDestroy(third);
    vernonRuntimeCoreGraphicsVariantDestroy(second);
    vernonRuntimeCoreGraphicsVariantDestroy(first);
    vernonRuntimeCorePipelineDestroy(pipeline);
}

TEST(RuntimeForeignProvider, DrawsRequireAGroupAtEveryIndexWithEntries) {
    static const VernonRuntimeProviderBindingLayoutEntry entry{
        0, 0, 0, VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER, VERNON_RUNTIME_PROVIDER_STAGE_VERTEX, 1, 1};
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_GRAPHICS);
    VernonRuntimeCoreGroupLayout *layout = nullptr;
    VernonRuntimeCoreGroupLayout *empty = nullptr;
    ASSERT_EQ(vernonRuntimeCorePrepareGroupLayout(&provider, &entry, 1, &layout), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePrepareGroupLayout(&provider, nullptr, 0, &empty), VERNON_STATUS_OK);
    VernonRuntimeCoreGroupLayout *const layouts[]{empty, layout, layout};
    VernonRuntimeCorePipelineDescriptor descriptor = graphicsPipelineDescriptor();
    descriptor.group_layouts = layouts;
    descriptor.group_layout_count = std::size(layouts);
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    const VernonRuntimeCoreGraphicsCompatibility compatibility = singleColorCompatibility();
    VernonRuntimeCoreGraphicsVariant *variant = nullptr;
    ASSERT_EQ(vernonRuntimeCorePrepareGraphicsVariant(pipeline, &compatibility, &variant), VERNON_STATUS_OK);
    VernonRuntimeCoreGroup *firstGroup = persistentGroup(layout, storageValue(99));
    VernonRuntimeCoreGroup *secondGroup = persistentGroup(layout, storageValue(199));

    VernonRuntimeProviderColorAttachment color{};
    color.view = {104, {1}, 0, 0};
    const VernonRuntimeProviderRenderingDescriptor rendering = colorRendering(color);
    VernonRuntimeCoreDrawInvocation draw{};
    draw.struct_size = sizeof(draw);
    draw.vertex_count = 3;
    draw.instance_count = 1;
    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {7}, &commands), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCoreBeginRendering(commands, &rendering), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCoreBindGraphicsVariant(commands, variant), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCoreBindGroup(commands, 1, firstGroup), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreDraw(commands, &draw), VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeCoreBindGroup(commands, 2, secondGroup), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreDraw(commands, &draw), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCoreBindGroup(commands, 1, secondGroup), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreDraw(commands, &draw), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(mock.renderingEnds, 1u);
    EXPECT_EQ(mock.groupBinds.size(), 3u);

    vernonRuntimeCoreGroupDestroy(secondGroup);
    vernonRuntimeCoreGroupDestroy(firstGroup);
    vernonRuntimeCoreGraphicsVariantDestroy(variant);
    vernonRuntimeCorePipelineDestroy(pipeline);
    vernonRuntimeCoreGroupLayoutDestroy(empty);
    vernonRuntimeCoreGroupLayoutDestroy(layout);
}

TEST(RuntimeForeignProvider, ValidatesRenderingAttachmentsBeforeBeginning) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_GRAPHICS);
    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {7}, &commands), VERNON_STATUS_OK);

    VernonRuntimeProviderColorAttachment color{};
    color.view = {104, {1}, 0, 0};
    VernonRuntimeProviderRenderingDescriptor rendering = colorRendering(color);
    EXPECT_EQ(vernonRuntimeCoreBeginRendering(commands, &rendering), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreBeginRendering(commands, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRuntimeCoreEndRendering(commands), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreEndRendering(commands), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(mock.renderingBegins, 1u);
    EXPECT_EQ(mock.renderingEnds, 1u);
    EXPECT_EQ(mock.lastRenderingColorCount, 1u);

    color.location = 1;
    EXPECT_EQ(vernonRuntimeCoreBeginRendering(commands, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    color.location = 0;
    color.view.identity = 100;
    EXPECT_EQ(vernonRuntimeCoreBeginRendering(commands, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    color.view.identity = 106;
    EXPECT_EQ(vernonRuntimeCoreBeginRendering(commands, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    color.view.identity = 104;
    rendering.render_area[0] = 1;
    EXPECT_EQ(vernonRuntimeCoreBeginRendering(commands, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    rendering.render_area[0] = 0;
    rendering.layers = 2;
    EXPECT_EQ(vernonRuntimeCoreBeginRendering(commands, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    rendering.layers = 1;

    VernonRuntimeProviderDepthStencilAttachment depth{};
    depth.view = {105, {2}, 0, 0};
    depth.depth_load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    depth.clear_depth = 1.0f;
    depth.stencil_load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    rendering.depth_stencil_attachment = &depth;
    EXPECT_EQ(vernonRuntimeCoreBeginRendering(commands, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    depth.stencil_load_operation = VERNON_RUNTIME_PROVIDER_LOAD_DISCARD;
    EXPECT_EQ(vernonRuntimeCoreBeginRendering(commands, &rendering), VERNON_STATUS_OK);
    EXPECT_EQ(mock.renderingBegins, 2u);
    EXPECT_EQ(vernonRuntimeCoreEndRendering(commands), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
}

// An integer clear carries raw bits, so a float NaN pattern is valid for an integer view and invalid for a float one.
TEST(RuntimeForeignProvider, ValidatesClearColorsByAttachmentComponentClass) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_GRAPHICS);
    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {7}, &commands), VERNON_STATUS_OK);

    VernonRuntimeProviderColorAttachment color{};
    color.view = {109, {1}, 0, 0};
    color.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    color.clear_color.u32[0] = 0xFFFFFFFFu;
    const VernonRuntimeProviderRenderingDescriptor rendering = colorRendering(color);
    EXPECT_EQ(vernonRuntimeCoreBeginRendering(commands, &rendering), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreEndRendering(commands), VERNON_STATUS_OK);

    color.view.identity = 104;
    EXPECT_EQ(vernonRuntimeCoreBeginRendering(commands, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    color.clear_color.f32[0] = 1.0f;
    EXPECT_EQ(vernonRuntimeCoreBeginRendering(commands, &rendering), VERNON_STATUS_OK);
    EXPECT_EQ(mock.renderingBegins, 2u);
    EXPECT_EQ(vernonRuntimeCoreEndRendering(commands), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
}

TEST(RuntimeForeignProvider, ValidatesRenderingSampledImagesBeforeBeginning) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_GRAPHICS);
    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {7}, &commands), VERNON_STATUS_OK);

    VernonRuntimeProviderColorAttachment color{};
    color.view = {104, {1}, 0, 0};
    VernonRuntimeProviderResourceReference sampled{102, {3}, 0, 0};
    VernonRuntimeProviderRenderingDescriptor rendering = colorRendering(color);
    rendering.sampled_images = &sampled;
    rendering.sampled_image_count = 1;
    EXPECT_EQ(vernonRuntimeCoreBeginRendering(commands, &rendering), VERNON_STATUS_OK);
    EXPECT_EQ(mock.lastRenderingSampledImageCount, 1u);
    EXPECT_EQ(vernonRuntimeCoreEndRendering(commands), VERNON_STATUS_OK);

    // A view of the attachment's image covering the rendered subresources.
    sampled.identity = 108;
    EXPECT_EQ(vernonRuntimeCoreBeginRendering(commands, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    sampled.identity = 104;
    EXPECT_EQ(vernonRuntimeCoreBeginRendering(commands, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    rendering.sampled_images = nullptr;
    EXPECT_EQ(vernonRuntimeCoreBeginRendering(commands, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(mock.renderingBegins, 1u);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
}

TEST(RuntimeForeignProvider, ReleasesPartiallyPreparedGraphicsVariant) {
    MockProvider mock;
    mock.failedPipelinePreparation = 2;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_GRAPHICS);
    const VernonRuntimeCorePipelineDescriptor descriptor = graphicsPipelineDescriptor();
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    const VernonRuntimeCoreGraphicsCompatibility compatibility = singleColorCompatibility();
    VernonRuntimeCoreGraphicsVariant *variant = nullptr;
    EXPECT_EQ(vernonRuntimeCorePrepareGraphicsVariant(pipeline, &compatibility, &variant),
              VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(variant, nullptr);
    EXPECT_EQ(mock.destroyedPipelines, 1u);
    vernonRuntimeCorePipelineDestroy(pipeline);
    EXPECT_EQ(mock.destroyedPipelines, 2u);
}

} // namespace
