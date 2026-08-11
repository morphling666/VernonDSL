#include "VernonRuntimeCore.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace {

struct MockProvider {
    uint64_t nextObject{1};
    uint32_t capabilities{};
    uint32_t shaderPreparations{};
    uint32_t layoutPreparations{};
    uint32_t pipelinePreparations{};
    uint32_t failedPipelinePreparation{};
    uint32_t destroyedPipelines{};
    uint32_t bindingCreations{};
    uint32_t dispatches{};
    uint32_t retainedResources{};
    uint32_t releasedResources{};
    VernonRuntimeProviderObject lastEncoder{};
    VernonRuntimeProviderDispatchDescriptor lastDispatch{};
    VernonRuntimeProviderDrawDescriptor lastDraw{};
};

VernonRuntimeProviderObject next(MockProvider &mock) { return {mock.nextObject++}; }

VernonRuntimeDeviceProvider makeProvider(MockProvider &mock, uint32_t capabilities) {
    mock.capabilities = capabilities;
    VernonRuntimeDeviceProvider provider{};
    provider.struct_size = sizeof(provider);
    provider.abi_version = VERNON_PIPELINE_VERSION;
    provider.user_data = &mock;
    provider.get_capabilities = [](void *data) -> uint32_t { return static_cast<MockProvider *>(data)->capabilities; };
    provider.get_device_identity = [](void *) -> VernonRuntimeProviderDeviceIdentity { return {11, 22, 33}; };
    provider.prepare_shader = [](void *data, const VernonRuntimeProviderShaderDescriptor *,
                                 VernonRuntimeProviderObject *shader) {
        auto &state = *static_cast<MockProvider *>(data);
        ++state.shaderPreparations;
        *shader = next(state);
        return VERNON_STATUS_OK;
    };
    provider.prepare_pipeline_layout = [](void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *,
                                          VernonRuntimeProviderObject *layout) {
        auto &state = *static_cast<MockProvider *>(data);
        ++state.layoutPreparations;
        *layout = next(state);
        return VERNON_STATUS_OK;
    };
    provider.prepare_pipeline = [](void *data, const VernonRuntimeProviderPipelineDescriptor *,
                                   VernonRuntimeProviderObject *pipeline) {
        auto &state = *static_cast<MockProvider *>(data);
        ++state.pipelinePreparations;
        *pipeline = next(state);
        return state.pipelinePreparations == state.failedPipelinePreparation ? VERNON_STATUS_INTERNAL_ERROR
                                                                             : VERNON_STATUS_OK;
    };
    provider.retain_resource = [](void *data, VernonRuntimeProviderResourceReference) {
        ++static_cast<MockProvider *>(data)->retainedResources;
        return VERNON_STATUS_OK;
    };
    provider.release_resource = [](void *data, VernonRuntimeProviderResourceReference) {
        ++static_cast<MockProvider *>(data)->releasedResources;
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
        return VERNON_STATUS_OK;
    };
    provider.create_binding_set = [](void *data, const VernonRuntimeProviderBindingSetDescriptor *,
                                     VernonRuntimeProviderObject *bindings) {
        auto &state = *static_cast<MockProvider *>(data);
        ++state.bindingCreations;
        *bindings = next(state);
        return VERNON_STATUS_OK;
    };
    provider.update_binding_set = [](void *, VernonRuntimeProviderObject, const VernonRuntimeProviderBindingValue *,
                                     size_t) { return VERNON_STATUS_OK; };
    provider.encode_dispatch = [](void *data, VernonRuntimeProviderObject encoder,
                                  const VernonRuntimeProviderDispatchDescriptor *descriptor) {
        auto &state = *static_cast<MockProvider *>(data);
        ++state.dispatches;
        state.lastEncoder = encoder;
        state.lastDispatch = *descriptor;
        return VERNON_STATUS_OK;
    };
    provider.encode_draw = [](void *data, VernonRuntimeProviderObject,
                              const VernonRuntimeProviderDrawDescriptor *draw) {
        static_cast<MockProvider *>(data)->lastDraw = *draw;
        return VERNON_STATUS_OK;
    };
    provider.destroy_shader = [](void *, VernonRuntimeProviderObject) {};
    provider.destroy_pipeline_layout = [](void *, VernonRuntimeProviderObject) {};
    provider.destroy_pipeline = [](void *data, VernonRuntimeProviderObject) {
        ++static_cast<MockProvider *>(data)->destroyedPipelines;
    };
    provider.destroy_binding_set = [](void *, VernonRuntimeProviderObject) {};
    return provider;
}

VernonRuntimeCorePipelineDescriptor computePipelineDescriptor() {
    static constexpr char source[] = ".version 8.0";
    static const VernonRuntimeProviderShaderDescriptor shader{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                              VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE,
                                                              {"ptx", 3},
                                                              source,
                                                              sizeof(source) - 1,
                                                              {"main", 4},
                                                              {"0123456789abcdef", 16},
                                                              {0, 0, 0, 0}};
    static const VernonRuntimeProviderBindingLayoutEntry binding{
        0, 0, 0, VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER, VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE, 3, 1};
    VernonRuntimeCorePipelineDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.kind = VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
    descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_COMPUTE;
    descriptor.shaders = &shader;
    descriptor.shader_count = 1;
    descriptor.bindings = &binding;
    descriptor.binding_count = 1;
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

VernonRuntimeCorePipelineDescriptor imagePipelineDescriptor() {
    VernonRuntimeCorePipelineDescriptor descriptor = computePipelineDescriptor();
    static const std::array<VernonRuntimeProviderBindingLayoutEntry, 2> bindings = [] {
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
    descriptor.bindings = bindings.data();
    descriptor.binding_count = bindings.size();
    return descriptor;
}

TEST(RuntimeForeignProvider, RejectsProvidersWithoutLifecycleCallbacks) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    VernonRuntimeCorePipelineDescriptor descriptor = computePipelineDescriptor();
    VernonRuntimeCorePipeline *pipeline{};

    provider.retain_resource = nullptr;
    EXPECT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_INVALID_ARGUMENT);
    provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    provider.release_resource = nullptr;
    EXPECT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pipeline, nullptr);
}

TEST(RuntimeForeignProvider, PreparesBindsAndEncodesWithNumericSlots) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    VernonRuntimeCorePipelineDescriptor descriptor = computePipelineDescriptor();
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    ASSERT_NE(pipeline, nullptr);
    EXPECT_EQ(vernonRuntimeCorePipelineGetDeviceIdentity(pipeline).device_id, 22u);

    VernonRuntimeProviderBindingValue value{};
    value.slot = 0;
    value.kind = VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
    value.payload.buffer.resource = {99, {100}, 0, 4096};
    VernonRuntimeCoreBindings *bindings = nullptr;
    ASSERT_EQ(vernonRuntimeCoreCreateBindings(pipeline, &value, 1, &bindings), VERNON_STATUS_OK);
    const uint32_t groups[3]{4, 2, 1};
    EXPECT_EQ(vernonRuntimeCoreEncodeDispatch(pipeline, bindings, {200}, groups, nullptr, 0), VERNON_STATUS_OK);
    EXPECT_EQ(mock.shaderPreparations, 1u);
    EXPECT_EQ(mock.layoutPreparations, 1u);
    EXPECT_EQ(mock.pipelinePreparations, 1u);
    EXPECT_EQ(mock.bindingCreations, 1u);
    EXPECT_EQ(mock.dispatches, 1u);
    EXPECT_EQ(mock.lastEncoder.value, 200u);
    EXPECT_EQ(mock.lastDispatch.group_count[0], 4u);

    vernonRuntimeCorePipelineDestroy(pipeline);
    vernonRuntimeCoreBindingsDestroy(bindings);
    EXPECT_EQ(mock.retainedResources, mock.releasedResources);
}

TEST(RuntimeForeignProvider, RejectsCapabilitiesBeforePreparation) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_GRAPHICS);
    VernonRuntimeCorePipelineDescriptor descriptor = computePipelineDescriptor();
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
    VernonRuntimeCorePipelineDescriptor descriptor = imagePipelineDescriptor();
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);

    std::array<VernonRuntimeProviderBindingValue, 2> values{};
    values[0].slot = 0;
    values[0].kind = VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE;
    values[0].payload.image.view = {100, {1}, 0, 0};
    values[1].slot = 1;
    values[1].kind = VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
    values[1].payload.image.view = {100, {2}, 0, 0};
    VernonRuntimeCoreBindings *bindings = nullptr;
    ASSERT_EQ(vernonRuntimeCoreCreateBindings(pipeline, values.data(), values.size(), &bindings), VERNON_STATUS_OK);

    values[0].payload.image.view.identity = 101;
    EXPECT_EQ(vernonRuntimeCoreUpdateBindings(bindings, values.data(), values.size()), VERNON_STATUS_INVALID_ARGUMENT);
    // Ordinary sampling of a depth-format view still produces floating-point shader values.
    values[0].payload.image.view.identity = 103;
    EXPECT_EQ(vernonRuntimeCoreUpdateBindings(bindings, values.data(), values.size()), VERNON_STATUS_OK);
    values[0].payload.image.view.identity = 107;
    EXPECT_EQ(vernonRuntimeCoreUpdateBindings(bindings, values.data(), values.size()), VERNON_STATUS_INVALID_ARGUMENT);
    values[0].payload.image.view.identity = 100;
    values[1].payload.image.view.identity = 102;
    EXPECT_EQ(vernonRuntimeCoreUpdateBindings(bindings, values.data(), values.size()), VERNON_STATUS_INVALID_ARGUMENT);

    vernonRuntimeCoreBindingsDestroy(bindings);
    vernonRuntimeCorePipelineDestroy(pipeline);
    EXPECT_EQ(mock.retainedResources, mock.releasedResources);
}

TEST(RuntimeForeignProvider, CachesGraphicsVariantsByCompatibility) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_GRAPHICS);
    VernonRuntimeCorePipelineDescriptor descriptor = graphicsPipelineDescriptor();
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    ASSERT_NE(pipeline, nullptr);
    EXPECT_EQ(mock.pipelinePreparations, 1u);

    const uint32_t formats[]{3};
    const uint32_t strides[]{12};
    VernonRuntimeProviderColorBlendState blend{};
    blend.write_mask = VERNON_RHI_COLOR_WRITE_ALL;
    VernonRuntimeCoreGraphicsCompatibility compatibility{};
    compatibility.struct_size = sizeof(compatibility);
    compatibility.color_formats = formats;
    compatibility.color_format_count = std::size(formats);
    compatibility.sample_count = 1;
    compatibility.vertex_strides = strides;
    compatibility.vertex_stride_count = std::size(strides);
    compatibility.color_blends = &blend;
    compatibility.color_blend_count = 1;
    VernonRuntimeCoreGraphicsVariant *first = nullptr;
    VernonRuntimeCoreGraphicsVariant *second = nullptr;
    ASSERT_EQ(vernonRuntimeCorePrepareGraphicsVariant(pipeline, &compatibility, &first), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePrepareGraphicsVariant(pipeline, &compatibility, &second), VERNON_STATUS_OK);
    EXPECT_EQ(mock.pipelinePreparations, 2u);
    VernonRuntimeCoreDrawInvocation draw{};
    draw.struct_size = sizeof(draw);
    draw.vertex_count = 3;
    draw.instance_count = 1;
    draw.stencil_reference = 0x100;
    EXPECT_EQ(vernonRuntimeCoreEncodeGraphicsVariantDrawInvocation(first, nullptr, &draw),
              VERNON_STATUS_INVALID_ARGUMENT);
    draw.stencil_reference = 0;
    draw.clear_stencil = 0x100;
    EXPECT_EQ(vernonRuntimeCoreEncodeGraphicsVariantDrawInvocation(first, nullptr, &draw),
              VERNON_STATUS_INVALID_ARGUMENT);
    draw.clear_stencil = 0;
    VernonRuntimeProviderColorAttachment colorAttachment{};
    colorAttachment.location = 1;
    draw.color_attachments = &colorAttachment;
    draw.color_attachment_count = 1;
    EXPECT_EQ(vernonRuntimeCoreEncodeGraphicsVariantDrawInvocation(first, nullptr, &draw),
              VERNON_STATUS_INVALID_ARGUMENT);
    draw.color_attachments = nullptr;
    draw.color_attachment_count = 0;
    draw.stencil_reference = 17;
    ASSERT_EQ(vernonRuntimeCoreEncodeGraphicsVariantDrawInvocation(first, nullptr, &draw), VERNON_STATUS_OK);
    EXPECT_EQ(mock.lastDraw.stencil_reference, 17u);

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

TEST(RuntimeForeignProvider, ValidatesAttachmentViewsBeforeEncoding) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_GRAPHICS);
    VernonRuntimeCorePipelineDescriptor descriptor = graphicsPipelineDescriptor();
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);

    VernonRuntimeProviderColorAttachment color{};
    color.view = {104, {1}, 0, 0};
    VernonRuntimeCoreDrawInvocation draw{};
    draw.struct_size = sizeof(draw);
    draw.vertex_count = 3;
    draw.instance_count = 1;
    draw.color_attachments = &color;
    draw.color_attachment_count = 1;
    EXPECT_EQ(vernonRuntimeCoreEncodeDrawInvocation(pipeline, nullptr, &draw), VERNON_STATUS_OK);

    color.view.identity = 100;
    EXPECT_EQ(vernonRuntimeCoreEncodeDrawInvocation(pipeline, nullptr, &draw), VERNON_STATUS_INVALID_ARGUMENT);
    color.view.identity = 106;
    EXPECT_EQ(vernonRuntimeCoreEncodeDrawInvocation(pipeline, nullptr, &draw), VERNON_STATUS_INVALID_ARGUMENT);

    color.view.identity = 104;
    draw.depth_stencil_view = {105, {2}, 0, 0};
    EXPECT_EQ(vernonRuntimeCoreEncodeDrawInvocation(pipeline, nullptr, &draw), VERNON_STATUS_OK);
    vernonRuntimeCorePipelineDestroy(pipeline);
}

TEST(RuntimeForeignProvider, ReleasesPartiallyPreparedGraphicsVariant) {
    MockProvider mock;
    mock.failedPipelinePreparation = 2;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_GRAPHICS);
    VernonRuntimeCorePipelineDescriptor descriptor = graphicsPipelineDescriptor();
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    const uint32_t format[]{3};
    VernonRuntimeProviderColorBlendState blend{};
    blend.write_mask = VERNON_RHI_COLOR_WRITE_ALL;
    VernonRuntimeCoreGraphicsCompatibility compatibility{};
    compatibility.struct_size = sizeof(compatibility);
    compatibility.color_formats = format;
    compatibility.color_format_count = 1;
    compatibility.sample_count = 1;
    compatibility.color_blends = &blend;
    compatibility.color_blend_count = 1;
    VernonRuntimeCoreGraphicsVariant *variant = nullptr;
    EXPECT_EQ(vernonRuntimeCorePrepareGraphicsVariant(pipeline, &compatibility, &variant),
              VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(variant, nullptr);
    EXPECT_EQ(mock.destroyedPipelines, 1u);
    vernonRuntimeCorePipelineDestroy(pipeline);
    EXPECT_EQ(mock.destroyedPipelines, 2u);
}

} // namespace
