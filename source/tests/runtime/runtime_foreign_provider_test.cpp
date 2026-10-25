#include "VernonRuntimeCore.h"
#include "runtime_core_packets.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
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
    uint32_t batchCreations{};
    uint32_t batchEncodes{};
    uint32_t destroyedBatches{};
    uint32_t pipelineBinds{};
    uint32_t dispatches{};
    uint32_t commandBegins{};
    uint32_t commandEnds{};
    uint32_t commandResumes{};
    uint32_t groupUpdates{};
    VernonRuntimeProviderObject lastUpdatedGroup{};
    VernonRuntimeProviderObject lastEncoder{};
    VernonRuntimeProviderGroupLifetime lastGroupLifetime{};
    std::array<uint32_t, 3> lastDispatch{};
    VernonRuntimeProviderDrawPayload lastDraw{};
    // The index and group of every encoded BIND_GROUP packet, in order.
    std::vector<std::pair<uint32_t, uint64_t>> groupBinds;
    // The first binding and the buffer objects of every encoded BIND_VERTEX_BUFFERS packet, in order.
    std::vector<std::pair<uint32_t, std::vector<uint64_t>>> vertexBufferBinds;
    uint32_t renderingBegins{};
    uint32_t renderingEnds{};
    size_t lastRenderingColorCount{};
    size_t lastRenderingSampledImageCount{};
    bool failNextGroupCreation{};
};

struct MockBatch {
    std::vector<uint32_t> packets;
    std::vector<uint64_t> objects;
    std::vector<uint64_t> resources;
    // The color attachment and sampled image counts of each rendering.
    std::vector<std::pair<size_t, size_t>> renderings;
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
    provider.resume_commands = [](void *data, VernonRuntimeProviderObject) {
        ++state(data).commandResumes;
        return VERNON_STATUS_OK;
    };
    provider.create_packet_batch = [](void *data, const VernonRuntimeProviderPacketBatchDescriptor *descriptor,
                                      VernonRuntimeProviderObject *handle) {
        auto *batch = new MockBatch{{descriptor->packets, descriptor->packets + descriptor->packet_word_count}};
        for (size_t index = 0; index < descriptor->object_count; ++index)
            batch->objects.push_back(descriptor->objects[index].value);
        for (size_t index = 0; index < descriptor->resource_count; ++index)
            batch->resources.push_back(descriptor->resources[index].resource.value);
        for (size_t index = 0; index < descriptor->rendering_count; ++index)
            batch->renderings.emplace_back(descriptor->renderings[index].color_attachment_count,
                                           descriptor->renderings[index].sampled_image_count);
        ++state(data).batchCreations;
        *handle = {reinterpret_cast<uint64_t>(batch)};
        return VERNON_STATUS_OK;
    };
    provider.encode_packets = [](void *data, VernonRuntimeProviderObject, VernonRuntimeProviderObject handle,
                                 const uint32_t *payload, size_t) {
        MockProvider &mock = state(data);
        const MockBatch &batch = *reinterpret_cast<const MockBatch *>(handle.value);
        ++mock.batchEncodes;
        for (size_t at = 0; at < batch.packets.size(); at += batch.packets[at] >> 16) {
            const uint32_t *operands = batch.packets.data() + at + 1;
            switch (batch.packets[at] & 0xffffu) {
            case VERNON_RUNTIME_PROVIDER_PACKET_BEGIN_RENDERING:
                ++mock.renderingBegins;
                mock.lastRenderingColorCount = batch.renderings[operands[0]].first;
                mock.lastRenderingSampledImageCount = batch.renderings[operands[0]].second;
                break;
            case VERNON_RUNTIME_PROVIDER_PACKET_END_RENDERING:
                ++mock.renderingEnds;
                break;
            case VERNON_RUNTIME_PROVIDER_PACKET_BIND_PIPELINE:
                ++mock.pipelineBinds;
                break;
            case VERNON_RUNTIME_PROVIDER_PACKET_BIND_GROUP:
                mock.groupBinds.emplace_back(operands[0], batch.objects[operands[1]]);
                break;
            case VERNON_RUNTIME_PROVIDER_PACKET_BIND_VERTEX_BUFFERS: {
                std::vector<uint64_t> buffers;
                for (uint32_t index = 1; index + 1 < (batch.packets[at] >> 16); ++index)
                    buffers.push_back(batch.resources[operands[index]]);
                mock.vertexBufferBinds.emplace_back(operands[0], std::move(buffers));
                break;
            }
            case VERNON_RUNTIME_PROVIDER_PACKET_DRAW:
                std::memcpy(&mock.lastDraw, payload + operands[1], sizeof(mock.lastDraw));
                break;
            case VERNON_RUNTIME_PROVIDER_PACKET_DRAW_INDEXED:
                std::memcpy(&mock.lastDraw, payload + operands[2], sizeof(mock.lastDraw));
                break;
            case VERNON_RUNTIME_PROVIDER_PACKET_DISPATCH:
                ++mock.dispatches;
                std::memcpy(mock.lastDispatch.data(), payload + operands[0], sizeof(mock.lastDispatch));
                break;
            default:
                break;
            }
        }
        return VERNON_STATUS_OK;
    };
    provider.destroy_shader = [](void *, VernonRuntimeProviderObject) {};
    provider.destroy_group_layout = [](void *data, VernonRuntimeProviderObject) {
        ++state(data).destroyedGroupLayouts;
    };
    provider.destroy_pipeline_layout = [](void *, VernonRuntimeProviderObject) {};
    provider.destroy_pipeline = [](void *data, VernonRuntimeProviderObject) { ++state(data).destroyedPipelines; };
    provider.destroy_group = [](void *data, VernonRuntimeProviderObject) { ++state(data).destroyedGroups; };
    provider.destroy_packet_batch = [](void *data, VernonRuntimeProviderObject handle) {
        delete reinterpret_cast<MockBatch *>(handle.value);
        ++state(data).destroyedBatches;
    };
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
    provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    provider.resume_commands = nullptr;
    EXPECT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_INVALID_ARGUMENT);
    provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    provider.destroy_packet_batch = nullptr;
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
    provider.encode_packets = [](void *, VernonRuntimeProviderObject, VernonRuntimeProviderObject, const uint32_t *,
                                 size_t) -> VernonStatus { throw std::runtime_error("encode"); };
    VernonRuntimeCoreGroupLayout *layout = storageLayout(provider);
    ASSERT_NE(layout, nullptr);
    const VernonRuntimeCorePipelineDescriptor descriptor = computePipelineDescriptor(&layout, 1);
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    VernonRuntimeCoreGroup *group = persistentGroup(layout, storageValue(99));
    ASSERT_NE(group, nullptr);

    VernonRuntimeCorePackets *packets = nullptr;
    ASSERT_EQ(vernonRuntimeCorePacketsCreate(&provider, &packets), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsBindPipeline(packets, pipeline), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsBindGroup(packets, 0, group), VERNON_STATUS_OK);
    const uint32_t groups[3]{1, 1, 1};
    ASSERT_EQ(vernonRuntimeCorePacketsDispatch(packets, groups), VERNON_STATUS_OK);
    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {200}, &commands), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::tests::encodeRecordedPackets(commands, packets), VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
    vernonRuntimeCorePacketsDestroy(packets);
    vernonRuntimeCoreGroupDestroy(group);
    vernonRuntimeCorePipelineDestroy(pipeline);
    vernonRuntimeCoreGroupLayoutDestroy(layout);
    EXPECT_EQ(mock.destroyedBatches, 1u);

    MockProvider batchMock;
    provider = makeProvider(batchMock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    provider.create_packet_batch = [](void *, const VernonRuntimeProviderPacketBatchDescriptor *,
                                      VernonRuntimeProviderObject *) -> VernonStatus {
        throw std::runtime_error("batch");
    };
    ASSERT_EQ(vernonRuntimeCorePacketsCreate(&provider, &packets), VERNON_STATUS_OK);
    VernonRuntimeCorePacketBatch *batch = nullptr;
    EXPECT_EQ(vernonRuntimeCorePacketBatchCreate(packets, &batch), VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(batch, nullptr);
    vernonRuntimeCorePacketsDestroy(packets);

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

TEST(RuntimeForeignProvider, PreparesBindsAndDispatchesThroughPacketBatches) {
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

    VernonRuntimeCorePackets *packets = nullptr;
    ASSERT_EQ(vernonRuntimeCorePacketsCreate(&provider, &packets), VERNON_STATUS_OK);
    const uint32_t groups[3]{4, 2, 1};
    EXPECT_EQ(vernonRuntimeCorePacketsDispatch(packets, groups), VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeCorePacketsBindPipeline(packets, pipeline), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsDispatch(packets, groups), VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeCorePacketsBindGroup(packets, 0, group), VERNON_STATUS_OK);
    for (int repeat = 0; repeat < 3; ++repeat) {
        ASSERT_EQ(vernonRuntimeCorePacketsBindPipeline(packets, pipeline), VERNON_STATUS_OK);
        ASSERT_EQ(vernonRuntimeCorePacketsBindGroup(packets, 0, group), VERNON_STATUS_OK);
        EXPECT_EQ(vernonRuntimeCorePacketsDispatch(packets, groups), VERNON_STATUS_OK);
    }
    EXPECT_EQ(mock.batchCreations, 0u);

    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {200}, &commands), VERNON_STATUS_OK);
    EXPECT_EQ(mock.lastEncoder.value, 200u);
    EXPECT_EQ(vernon::tests::encodeRecordedPackets(commands, packets), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
    vernonRuntimeCorePacketsDestroy(packets);

    EXPECT_EQ(mock.shaderPreparations, 1u);
    EXPECT_EQ(mock.groupLayoutPreparations, 1u);
    EXPECT_EQ(mock.layoutPreparations, 1u);
    EXPECT_EQ(mock.pipelinePreparations, 1u);
    EXPECT_EQ(mock.groupCreations, 1u);
    EXPECT_EQ(mock.batchCreations, 1u);
    EXPECT_EQ(mock.batchEncodes, 1u);
    EXPECT_EQ(mock.destroyedBatches, 1u);
    EXPECT_EQ(mock.pipelineBinds, 1u);
    EXPECT_EQ(mock.groupBinds.size(), 1u);
    EXPECT_EQ(mock.dispatches, 3u);
    EXPECT_EQ(mock.lastDispatch[0], 4u);
    EXPECT_EQ(mock.commandBegins, 1u);
    EXPECT_EQ(mock.commandEnds, 1u);

    vernonRuntimeCorePipelineDestroy(pipeline);
    vernonRuntimeCoreGroupDestroy(group);
    vernonRuntimeCoreGroupLayoutDestroy(layout);
    EXPECT_EQ(mock.destroyedGroups, 1u);
    EXPECT_EQ(mock.destroyedGroupLayouts, 1u);
}

TEST(RuntimeForeignProvider, EachBatchStartsWithNothingBoundAndEndsOutsideRendering) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider =
        makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE | VERNON_RUNTIME_PROVIDER_GRAPHICS);
    VernonRuntimeCoreGroupLayout *layout = storageLayout(provider);
    ASSERT_NE(layout, nullptr);
    const VernonRuntimeCorePipelineDescriptor descriptor = computePipelineDescriptor(&layout, 1);
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    VernonRuntimeCoreGroup *group = persistentGroup(layout, storageValue(99));
    ASSERT_NE(group, nullptr);

    EXPECT_EQ(vernonRuntimeCoreResumeCommands(nullptr), VERNON_STATUS_INVALID_ARGUMENT);
    VernonRuntimeCorePackets *packets = nullptr;
    ASSERT_EQ(vernonRuntimeCorePacketsCreate(&provider, &packets), VERNON_STATUS_OK);
    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {200}, &commands), VERNON_STATUS_OK);
    const uint32_t groups[3]{4, 2, 1};
    for (int segment = 0; segment < 3; ++segment) {
        if (segment != 0)
            ASSERT_EQ(vernonRuntimeCoreResumeCommands(commands), VERNON_STATUS_OK);
        const uint32_t dispatchesBefore = mock.dispatches;
        EXPECT_EQ(vernonRuntimeCorePacketsDispatch(packets, groups), VERNON_STATUS_INVALID_ARGUMENT);
        ASSERT_EQ(vernonRuntimeCorePacketsBindPipeline(packets, pipeline), VERNON_STATUS_OK);
        ASSERT_EQ(vernonRuntimeCorePacketsBindGroup(packets, 0, group), VERNON_STATUS_OK);
        ASSERT_EQ(vernonRuntimeCorePacketsDispatch(packets, groups), VERNON_STATUS_OK);
        EXPECT_EQ(vernon::tests::encodeRecordedPackets(commands, packets), VERNON_STATUS_OK);
        EXPECT_EQ(mock.dispatches, dispatchesBefore + 1);
    }
    EXPECT_EQ(mock.commandResumes, 2u);
    EXPECT_EQ(mock.pipelineBinds, 3u);
    EXPECT_EQ(mock.groupBinds.size(), 3u);
    EXPECT_EQ(mock.destroyedBatches, 3u);

    VernonRuntimeProviderColorAttachment color{};
    color.view = {104, {1}, 0, 0};
    const VernonRuntimeProviderRenderingDescriptor rendering = colorRendering(color);
    VernonRuntimeCorePacketBatch *batch = nullptr;
    ASSERT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketBatchCreate(packets, &batch), VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeCorePacketsEndRendering(packets), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsPushDebugGroup(packets, {"scope", 5}), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketBatchCreate(packets, &batch), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(batch, nullptr);
    ASSERT_EQ(vernonRuntimeCorePacketsPopDebugGroup(packets), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::tests::encodeRecordedPackets(commands, packets), VERNON_STATUS_OK);
    EXPECT_EQ(mock.renderingBegins, 1u);
    EXPECT_EQ(mock.renderingEnds, 1u);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
    EXPECT_EQ(mock.commandBegins, 1u);
    EXPECT_EQ(mock.commandEnds, 1u);
    vernonRuntimeCorePacketsDestroy(packets);

    vernonRuntimeCorePipelineDestroy(pipeline);
    vernonRuntimeCoreGroupDestroy(group);
    vernonRuntimeCoreGroupLayoutDestroy(layout);
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

    VernonRuntimeCorePackets *packets = nullptr;
    ASSERT_EQ(vernonRuntimeCorePacketsCreate(&provider, &packets), VERNON_STATUS_OK);
    const uint32_t groups[3]{1, 1, 1};
    ASSERT_EQ(vernonRuntimeCorePacketsBindPipeline(packets, first), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsBindGroup(packets, 1, distinctGroup), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRuntimeCorePacketsBindGroup(packets, 2, sharedGroup), VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeCorePacketsBindGroup(packets, 0, sharedGroup), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsBindGroup(packets, 1, sharedGroup), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsDispatch(packets, groups), VERNON_STATUS_OK);

    ASSERT_EQ(vernonRuntimeCorePacketsBindPipeline(packets, second), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsDispatch(packets, groups), VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeCorePacketsBindGroup(packets, 1, distinctGroup), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsDispatch(packets, groups), VERNON_STATUS_OK);
    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {200}, &commands), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::tests::encodeRecordedPackets(commands, packets), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
    vernonRuntimeCorePacketsDestroy(packets);

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

TEST(RuntimeForeignProvider, RecordingsAndBatchesKeepWhatTheyNameAlive) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    VernonRuntimeCoreGroupLayout *layout = storageLayout(provider);
    const VernonRuntimeCorePipelineDescriptor descriptor = computePipelineDescriptor(&layout, 1);
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    VernonRuntimeCoreGroup *group = persistentGroup(layout, storageValue(99));
    ASSERT_NE(group, nullptr);

    VernonRuntimeCorePackets *packets = nullptr;
    ASSERT_EQ(vernonRuntimeCorePacketsCreate(&provider, &packets), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsBindPipeline(packets, pipeline), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsBindGroup(packets, 0, group), VERNON_STATUS_OK);
    vernonRuntimeCoreGroupDestroy(group);
    vernonRuntimeCorePipelineDestroy(pipeline);
    vernonRuntimeCoreGroupLayoutDestroy(layout);
    EXPECT_EQ(mock.destroyedGroups, 0u);
    EXPECT_EQ(mock.destroyedPipelines, 0u);
    const uint32_t groups[3]{1, 1, 1};
    EXPECT_EQ(vernonRuntimeCorePacketsDispatch(packets, groups), VERNON_STATUS_OK);
    VernonRuntimeCorePacketBatch *batch = nullptr;
    ASSERT_EQ(vernonRuntimeCorePacketBatchCreate(packets, &batch), VERNON_STATUS_OK);
    size_t wordCount = 0;
    const uint32_t *words = vernonRuntimeCorePacketsGetPayload(packets, &wordCount);
    const std::vector<uint32_t> payload(words, words + wordCount);
    vernonRuntimeCorePacketsDestroy(packets);
    EXPECT_EQ(mock.destroyedGroups, 0u);
    EXPECT_EQ(mock.destroyedPipelines, 0u);

    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {200}, &commands), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreEncodePackets(commands, batch, payload.data(), payload.size()), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreEncodePackets(commands, batch, payload.data(), payload.size() - 1),
              VERNON_STATUS_INVALID_ARGUMENT);
    std::vector<uint32_t> emptyGrid = payload;
    emptyGrid[0] = 0;
    EXPECT_EQ(vernonRuntimeCoreEncodePackets(commands, batch, emptyGrid.data(), emptyGrid.size()),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
    EXPECT_EQ(mock.batchEncodes, 1u);
    EXPECT_EQ(mock.dispatches, 1u);
    EXPECT_EQ(mock.destroyedGroups, 0u);
    vernonRuntimeCorePacketBatchDestroy(batch);
    EXPECT_EQ(mock.destroyedBatches, 1u);
    EXPECT_EQ(mock.destroyedGroups, 1u);
    EXPECT_EQ(mock.destroyedPipelines, 1u);
    EXPECT_EQ(mock.destroyedGroupLayouts, 1u);
}

TEST(RuntimeForeignProvider, TransientGroupsEncodeOnlyOnTheirCommandEncoder) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    VernonRuntimeCoreGroupLayout *layout = storageLayout(provider);
    const VernonRuntimeCorePipelineDescriptor descriptor = computePipelineDescriptor(&layout, 1);
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    const VernonRuntimeProviderBindingValue value = storageValue(99);
    VernonRuntimeCoreGroup *group = nullptr;
    VernonRuntimeCoreGroup *other = nullptr;
    EXPECT_EQ(vernonRuntimeCoreCreateGroup(layout, &value, 1, VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT, {7}, &group),
              VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeCoreCreateGroup(layout, &value, 1, VERNON_RUNTIME_PROVIDER_GROUP_TRANSIENT, {7}, &group),
              VERNON_STATUS_OK);
    EXPECT_EQ(mock.lastGroupLifetime, VERNON_RUNTIME_PROVIDER_GROUP_TRANSIENT);
    ASSERT_EQ(vernonRuntimeCoreCreateGroup(layout, &value, 1, VERNON_RUNTIME_PROVIDER_GROUP_TRANSIENT, {8}, &other),
              VERNON_STATUS_OK);

    VernonRuntimeCorePackets *packets = nullptr;
    ASSERT_EQ(vernonRuntimeCorePacketsCreate(&provider, &packets), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsBindPipeline(packets, pipeline), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsBindGroup(packets, 0, group), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsBindGroup(packets, 0, other), VERNON_STATUS_INVALID_ARGUMENT);
    const uint32_t groups[3]{1, 1, 1};
    ASSERT_EQ(vernonRuntimeCorePacketsDispatch(packets, groups), VERNON_STATUS_OK);
    VernonRuntimeCorePacketBatch *batch = nullptr;
    ASSERT_EQ(vernonRuntimeCorePacketBatchCreate(packets, &batch), VERNON_STATUS_OK);
    size_t wordCount = 0;
    const uint32_t *payload = vernonRuntimeCorePacketsGetPayload(packets, &wordCount);
    for (const uint64_t encoder : {8u, 7u}) {
        VernonRuntimeCoreCommands *commands = nullptr;
        ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {encoder}, &commands), VERNON_STATUS_OK);
        EXPECT_EQ(vernonRuntimeCoreEncodePackets(commands, batch, payload, wordCount),
                  encoder == 7 ? VERNON_STATUS_OK : VERNON_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
    }
    EXPECT_EQ(mock.batchEncodes, 1u);
    EXPECT_EQ(mock.groupBinds.size(), 1u);

    vernonRuntimeCorePacketBatchDestroy(batch);
    vernonRuntimeCorePacketsDestroy(packets);
    vernonRuntimeCoreGroupDestroy(other);
    vernonRuntimeCoreGroupDestroy(group);
    vernonRuntimeCorePipelineDestroy(pipeline);
    vernonRuntimeCoreGroupLayoutDestroy(layout);
}
TEST(RuntimeForeignProvider, OnlyUpdatableGroupsUpdateAndOnlyOnProvidersThatWriteThem) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    const VernonRuntimeProviderBindingValue value = storageValue(99);
    VernonRuntimeCoreGroupLayout *unwritable = storageLayout(provider);
    VernonRuntimeCoreGroup *group = nullptr;
    EXPECT_EQ(vernonRuntimeCoreCreateGroup(unwritable, &value, 1, VERNON_RUNTIME_PROVIDER_GROUP_UPDATABLE, {}, &group),
              VERNON_STATUS_UNSUPPORTED_TARGET);
    EXPECT_EQ(group, nullptr);
    vernonRuntimeCoreGroupLayoutDestroy(unwritable);

    provider.update_group = [](void *data, VernonRuntimeProviderObject, VernonRuntimeProviderObject updated,
                               const VernonRuntimeProviderBindingValue *, size_t) {
        ++state(data).groupUpdates;
        state(data).lastUpdatedGroup = updated;
        return VERNON_STATUS_OK;
    };
    VernonRuntimeCoreGroupLayout *layout = storageLayout(provider);
    EXPECT_EQ(vernonRuntimeCoreCreateGroup(layout, &value, 1, VERNON_RUNTIME_PROVIDER_GROUP_UPDATABLE, {7}, &group),
              VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeCoreCreateGroup(layout, &value, 1, VERNON_RUNTIME_PROVIDER_GROUP_UPDATABLE, {}, &group),
              VERNON_STATUS_OK);
    EXPECT_EQ(mock.lastGroupLifetime, VERNON_RUNTIME_PROVIDER_GROUP_UPDATABLE);
    const VernonRuntimeProviderObject handle{mock.nextObject - 1};
    VernonRuntimeCoreGroup *persistent = persistentGroup(layout, value);

    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {7}, &commands), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreUpdateGroup(commands, persistent, &value, 1), VERNON_STATUS_INVALID_ARGUMENT);
    VernonRuntimeProviderBindingValue mismatched = value;
    mismatched.kind = VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
    EXPECT_EQ(vernonRuntimeCoreUpdateGroup(commands, group, &mismatched, 1), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRuntimeCoreUpdateGroup(commands, group, &value, 0), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(mock.groupUpdates, 0u);
    EXPECT_EQ(vernonRuntimeCoreUpdateGroup(commands, group, &value, 1), VERNON_STATUS_OK);
    EXPECT_EQ(mock.groupUpdates, 1u);
    EXPECT_EQ(mock.lastUpdatedGroup.value, handle.value);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);

    vernonRuntimeCoreGroupDestroy(persistent);
    vernonRuntimeCoreGroupDestroy(group);
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
    VernonRuntimeCorePackets *packets = nullptr;
    ASSERT_EQ(vernonRuntimeCorePacketsCreate(&provider, &packets), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsBindGraphicsVariant(packets, first), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsDraw(packets, &draw), VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_OK);
    draw.stencil_reference = 0x100;
    EXPECT_EQ(vernonRuntimeCorePacketsDraw(packets, &draw), VERNON_STATUS_INVALID_ARGUMENT);
    draw.stencil_reference = 17;
    ASSERT_EQ(vernonRuntimeCorePacketsDraw(packets, &draw), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsBindGraphicsVariant(packets, second), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsEndRendering(packets), VERNON_STATUS_OK);
    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {7}, &commands), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::tests::encodeRecordedPackets(commands, packets), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
    vernonRuntimeCorePacketsDestroy(packets);
    EXPECT_EQ(mock.lastDraw.stencil_reference, 17u);
    EXPECT_EQ(mock.pipelineBinds, 1u);

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

TEST(RuntimeForeignProvider, DrawPayloadIsWhatADrawWritesAndDrawsOfNoInstancesAreValid) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_GRAPHICS);
    const VernonRuntimeCorePipelineDescriptor descriptor = graphicsPipelineDescriptor();
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    const VernonRuntimeCoreGraphicsCompatibility compatibility = singleColorCompatibility();
    VernonRuntimeCoreGraphicsVariant *variant = nullptr;
    ASSERT_EQ(vernonRuntimeCorePrepareGraphicsVariant(pipeline, &compatibility, &variant), VERNON_STATUS_OK);

    VernonRuntimeCoreDrawInvocation draw{};
    draw.struct_size = sizeof(draw);
    draw.instance_count = 0;
    draw.first_instance = 2;
    std::copy_n(std::array<uint32_t, 4>{1, 2, 3, 4}.begin(), 4, draw.viewport);
    std::copy_n(std::array<uint32_t, 4>{5, 6, 7, 8}.begin(), 4, draw.scissor);
    draw.stencil_reference = 9;
    VernonRuntimeProviderDrawPayload payload{};
    EXPECT_EQ(vernonRuntimeCoreDrawPayload(&draw, &payload), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRuntimeCoreDrawPayload(nullptr, &payload), VERNON_STATUS_INVALID_ARGUMENT);
    draw.vertex_count = 3;
    ASSERT_EQ(vernonRuntimeCoreDrawPayload(&draw, &payload), VERNON_STATUS_OK);
    EXPECT_EQ(payload.vertex_count, 3u);
    EXPECT_EQ(payload.instance_count, 0u);
    EXPECT_EQ(payload.first_instance, 2u);
    EXPECT_EQ(payload.scissor[3], 8u);
    EXPECT_EQ(payload.stencil_reference, 9u);

    VernonRuntimeProviderColorAttachment color{};
    color.view = {104, {1}, 0, 0};
    const VernonRuntimeProviderRenderingDescriptor rendering = colorRendering(color);
    VernonRuntimeCorePackets *packets = nullptr;
    ASSERT_EQ(vernonRuntimeCorePacketsCreate(&provider, &packets), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsBindGraphicsVariant(packets, variant), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsDraw(packets, &draw), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsEndRendering(packets), VERNON_STATUS_OK);
    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {7}, &commands), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::tests::encodeRecordedPackets(commands, packets), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
    vernonRuntimeCorePacketsDestroy(packets);
    EXPECT_EQ(std::memcmp(&mock.lastDraw, &payload, sizeof(payload)), 0);

    vernonRuntimeCoreGraphicsVariantDestroy(variant);
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
    VernonRuntimeCorePackets *packets = nullptr;
    ASSERT_EQ(vernonRuntimeCorePacketsCreate(&provider, &packets), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsBindGraphicsVariant(packets, variant), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsBindGroup(packets, 1, firstGroup), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsDraw(packets, &draw), VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeCorePacketsBindGroup(packets, 2, secondGroup), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsDraw(packets, &draw), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsBindGroup(packets, 1, secondGroup), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsDraw(packets, &draw), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsEndRendering(packets), VERNON_STATUS_OK);
    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {7}, &commands), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::tests::encodeRecordedPackets(commands, packets), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
    vernonRuntimeCorePacketsDestroy(packets);
    EXPECT_EQ(mock.renderingEnds, 1u);
    EXPECT_EQ(mock.groupBinds.size(), 3u);

    vernonRuntimeCoreGroupDestroy(secondGroup);
    vernonRuntimeCoreGroupDestroy(firstGroup);
    vernonRuntimeCoreGraphicsVariantDestroy(variant);
    vernonRuntimeCorePipelineDestroy(pipeline);
    vernonRuntimeCoreGroupLayoutDestroy(empty);
    vernonRuntimeCoreGroupLayoutDestroy(layout);
}

TEST(RuntimeForeignProvider, ForwardsChangedVertexBuffersAndDrawsRequireEveryVertexBinding) {
    static const VernonRuntimeProviderVertexBinding bindings[]{{0, 0}, {1, 1}};
    static const VernonRuntimeProviderVertexAttribute attributes[]{{0, 0, VERNON_RUNTIME_PROVIDER_F32, 3, 0},
                                                                   {1, 1, VERNON_RUNTIME_PROVIDER_F32, 4, 0}};
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_GRAPHICS);
    VernonRuntimeCorePipelineDescriptor descriptor = graphicsPipelineDescriptor();
    descriptor.vertex_attributes = attributes;
    descriptor.vertex_attribute_count = std::size(attributes);
    descriptor.vertex_bindings = bindings;
    descriptor.vertex_binding_count = std::size(bindings);
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    const uint32_t strides[]{12, 16};
    VernonRuntimeCoreGraphicsCompatibility compatibility = singleColorCompatibility();
    compatibility.vertex_strides = strides;
    compatibility.vertex_stride_count = 1;
    VernonRuntimeCoreGraphicsVariant *variant = nullptr;
    EXPECT_EQ(vernonRuntimeCorePrepareGraphicsVariant(pipeline, &compatibility, &variant),
              VERNON_STATUS_INVALID_ARGUMENT);
    compatibility.vertex_stride_count = std::size(strides);
    ASSERT_EQ(vernonRuntimeCorePrepareGraphicsVariant(pipeline, &compatibility, &variant), VERNON_STATUS_OK);

    VernonRuntimeProviderColorAttachment color{};
    color.view = {104, {1}, 0, 0};
    const VernonRuntimeProviderRenderingDescriptor rendering = colorRendering(color);
    VernonRuntimeCoreDrawInvocation draw{};
    draw.struct_size = sizeof(draw);
    draw.vertex_count = 3;
    draw.instance_count = 1;
    const VernonRuntimeProviderResourceReference positions{201, {11}, 0, 36};
    const VernonRuntimeProviderResourceReference colors{202, {12}, 0, 16};
    const VernonRuntimeProviderResourceReference otherColors{203, {13}, 0, 16};
    VernonRuntimeCorePackets *packets = nullptr;
    ASSERT_EQ(vernonRuntimeCorePacketsCreate(&provider, &packets), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsBindGraphicsVariant(packets, variant), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsBindVertexBuffers(packets, 0, &positions, 1), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsDraw(packets, &draw), VERNON_STATUS_INVALID_ARGUMENT);
    const VernonRuntimeProviderResourceReference both[]{positions, colors};
    ASSERT_EQ(vernonRuntimeCorePacketsBindVertexBuffers(packets, 0, both, std::size(both)), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsDraw(packets, &draw), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePacketsBindVertexBuffers(packets, 0, both, std::size(both)), VERNON_STATUS_OK);
    const VernonRuntimeProviderResourceReference changed[]{positions, otherColors};
    ASSERT_EQ(vernonRuntimeCorePacketsBindVertexBuffers(packets, 0, changed, std::size(changed)), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsBindVertexBuffers(packets, VERNON_RUNTIME_PROVIDER_MAX_VERTEX_BINDINGS, both, 1),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRuntimeCorePacketsEndRendering(packets), VERNON_STATUS_OK);
    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {7}, &commands), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::tests::encodeRecordedPackets(commands, packets), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
    vernonRuntimeCorePacketsDestroy(packets);

    const std::vector<std::pair<uint32_t, std::vector<uint64_t>>> expected{{0, {11}}, {1, {12}}, {1, {13}}};
    EXPECT_EQ(mock.vertexBufferBinds, expected);
    vernonRuntimeCoreGraphicsVariantDestroy(variant);
    vernonRuntimeCorePipelineDestroy(pipeline);
}

TEST(RuntimeForeignProvider, RejectsVertexAttributesWithoutTheirBinding) {
    static const VernonRuntimeProviderVertexBinding binding{0, 0};
    static const VernonRuntimeProviderVertexAttribute attribute{1, 0, VERNON_RUNTIME_PROVIDER_F32, 3, 0};
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_GRAPHICS);
    VernonRuntimeCorePipelineDescriptor descriptor = graphicsPipelineDescriptor();
    descriptor.vertex_attributes = &attribute;
    descriptor.vertex_attribute_count = 1;
    descriptor.vertex_bindings = &binding;
    descriptor.vertex_binding_count = 1;
    VernonRuntimeCorePipeline *pipeline = nullptr;
    EXPECT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pipeline, nullptr);
    EXPECT_EQ(mock.pipelinePreparations, 0u);
}

TEST(RuntimeForeignProvider, ValidatesRenderingAttachmentsBeforeBeginning) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_GRAPHICS);
    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {7}, &commands), VERNON_STATUS_OK);
    VernonRuntimeCorePackets *packets = nullptr;
    ASSERT_EQ(vernonRuntimeCorePacketsCreate(&provider, &packets), VERNON_STATUS_OK);

    VernonRuntimeProviderColorAttachment color{};
    color.view = {104, {1}, 0, 0};
    VernonRuntimeProviderRenderingDescriptor rendering = colorRendering(color);
    EXPECT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRuntimeCorePacketsEndRendering(packets), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsEndRendering(packets), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernon::tests::encodeRecordedPackets(commands, packets), VERNON_STATUS_OK);
    EXPECT_EQ(mock.renderingBegins, 1u);
    EXPECT_EQ(mock.renderingEnds, 1u);
    EXPECT_EQ(mock.lastRenderingColorCount, 1u);

    color.location = 1;
    EXPECT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    color.location = 0;
    color.view.identity = 100;
    EXPECT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    color.view.identity = 106;
    EXPECT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    color.view.identity = 104;
    rendering.render_area[0] = 1;
    EXPECT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    rendering.render_area[0] = 0;
    rendering.layers = 2;
    EXPECT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    rendering.layers = 1;

    VernonRuntimeProviderDepthStencilAttachment depth{};
    depth.view = {105, {2}, 0, 0};
    depth.depth_load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    depth.clear_depth = 1.0f;
    depth.stencil_load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    rendering.depth_stencil_attachment = &depth;
    EXPECT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    depth.stencil_load_operation = VERNON_RUNTIME_PROVIDER_LOAD_DISCARD;
    EXPECT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsEndRendering(packets), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::tests::encodeRecordedPackets(commands, packets), VERNON_STATUS_OK);
    EXPECT_EQ(mock.renderingBegins, 2u);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
    vernonRuntimeCorePacketsDestroy(packets);
}

// An integer clear carries raw bits, so a float NaN pattern is valid for an integer view and invalid for a float one.
TEST(RuntimeForeignProvider, ValidatesClearColorsByAttachmentComponentClass) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_GRAPHICS);
    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {7}, &commands), VERNON_STATUS_OK);
    VernonRuntimeCorePackets *packets = nullptr;
    ASSERT_EQ(vernonRuntimeCorePacketsCreate(&provider, &packets), VERNON_STATUS_OK);

    VernonRuntimeProviderColorAttachment color{};
    color.view = {109, {1}, 0, 0};
    color.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    color.clear_color.u32[0] = 0xFFFFFFFFu;
    const VernonRuntimeProviderRenderingDescriptor rendering = colorRendering(color);
    EXPECT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsEndRendering(packets), VERNON_STATUS_OK);

    color.view.identity = 104;
    EXPECT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    color.clear_color.f32[0] = 1.0f;
    EXPECT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsEndRendering(packets), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::tests::encodeRecordedPackets(commands, packets), VERNON_STATUS_OK);
    EXPECT_EQ(mock.renderingBegins, 2u);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
    vernonRuntimeCorePacketsDestroy(packets);
}

TEST(RuntimeForeignProvider, ValidatesRenderingSampledImagesBeforeBeginning) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_GRAPHICS);
    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(&provider, {7}, &commands), VERNON_STATUS_OK);
    VernonRuntimeCorePackets *packets = nullptr;
    ASSERT_EQ(vernonRuntimeCorePacketsCreate(&provider, &packets), VERNON_STATUS_OK);

    VernonRuntimeProviderColorAttachment color{};
    color.view = {104, {1}, 0, 0};
    VernonRuntimeProviderResourceReference sampled{102, {3}, 0, 0};
    VernonRuntimeProviderRenderingDescriptor rendering = colorRendering(color);
    rendering.sampled_images = &sampled;
    rendering.sampled_image_count = 1;
    EXPECT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsEndRendering(packets), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::tests::encodeRecordedPackets(commands, packets), VERNON_STATUS_OK);
    EXPECT_EQ(mock.lastRenderingSampledImageCount, 1u);

    // A view of the attachment's image covering the rendered subresources.
    sampled.identity = 108;
    EXPECT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    sampled.identity = 104;
    EXPECT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    rendering.sampled_images = nullptr;
    EXPECT_EQ(vernonRuntimeCorePacketsBeginRendering(packets, &rendering), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernon::tests::encodeRecordedPackets(commands, packets), VERNON_STATUS_OK);
    EXPECT_EQ(mock.renderingBegins, 1u);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
    vernonRuntimeCorePacketsDestroy(packets);
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
