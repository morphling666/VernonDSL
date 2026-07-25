#include "VernonRuntimeCore.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace {

struct MockProvider {
    uint64_t nextObject{1};
    uint32_t capabilities{};
    uint32_t shaderPreparations{};
    uint32_t layoutPreparations{};
    uint32_t pipelinePreparations{};
    uint32_t bindingCreations{};
    uint32_t dispatches{};
    uint32_t retainedResources{};
    uint32_t releasedResources{};
    VernonRuntimeProviderDispatchDescriptor lastDispatch{};
};

VernonRuntimeProviderObject next(MockProvider &mock) { return {mock.nextObject++}; }

VernonRuntimeDeviceProvider makeProvider(MockProvider &mock, uint32_t capabilities) {
    mock.capabilities = capabilities;
    VernonRuntimeDeviceProvider provider{};
    provider.struct_size = sizeof(provider);
    provider.abi_version = VERNON_RUNTIME_DEVICE_PROVIDER_ABI_VERSION;
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
        return VERNON_STATUS_OK;
    };
    provider.retain_resource = [](void *data, VernonRuntimeProviderResourceReference) {
        ++static_cast<MockProvider *>(data)->retainedResources;
        return VERNON_STATUS_OK;
    };
    provider.release_resource = [](void *data, VernonRuntimeProviderResourceReference) {
        ++static_cast<MockProvider *>(data)->releasedResources;
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
    provider.encode_dispatch = [](void *data, VernonRuntimeProviderObject,
                                  const VernonRuntimeProviderDispatchDescriptor *descriptor) {
        auto &state = *static_cast<MockProvider *>(data);
        ++state.dispatches;
        state.lastDispatch = *descriptor;
        return VERNON_STATUS_OK;
    };
    provider.encode_draw = [](void *, VernonRuntimeProviderObject, const VernonRuntimeProviderDrawDescriptor *) {
        return VERNON_STATUS_OK;
    };
    provider.destroy_shader = [](void *, VernonRuntimeProviderObject) {};
    provider.destroy_pipeline_layout = [](void *, VernonRuntimeProviderObject) {};
    provider.destroy_pipeline = [](void *, VernonRuntimeProviderObject) {};
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

TEST(RuntimeForeignProvider, PreparesBindsAndEncodesWithNumericSlots) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_COMPUTE);
    VernonRuntimeCorePipelineDescriptor descriptor = computePipelineDescriptor();
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    ASSERT_NE(pipeline, nullptr);
    EXPECT_EQ(vernonRuntimeCorePipelineGetDeviceIdentity(pipeline).device_id, 22u);

    const VernonRuntimeProviderBindingValue value{0, VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER, {99, {100}, 0, 4096}};
    VernonRuntimeCoreBindings *bindings = nullptr;
    ASSERT_EQ(vernonRuntimeCoreCreateBindings(pipeline, &value, 1, &bindings), VERNON_STATUS_OK);
    const uint32_t groups[3]{4, 2, 1};
    EXPECT_EQ(vernonRuntimeCoreEncodeDispatch(pipeline, bindings, {200}, groups, nullptr, 0), VERNON_STATUS_OK);
    EXPECT_EQ(mock.shaderPreparations, 1u);
    EXPECT_EQ(mock.layoutPreparations, 1u);
    EXPECT_EQ(mock.pipelinePreparations, 1u);
    EXPECT_EQ(mock.bindingCreations, 1u);
    EXPECT_EQ(mock.dispatches, 1u);
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

TEST(RuntimeForeignProvider, CachesGraphicsVariantsByCompatibility) {
    MockProvider mock;
    VernonRuntimeDeviceProvider provider = makeProvider(mock, VERNON_RUNTIME_PROVIDER_GRAPHICS);
    VernonRuntimeCorePipelineDescriptor descriptor = graphicsPipelineDescriptor();
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    ASSERT_NE(pipeline, nullptr);
    EXPECT_EQ(mock.pipelinePreparations, 1u);

    const uint32_t formats[]{3};
    const VernonRuntimeCoreGraphicsCompatibility compatibility{
        sizeof(VernonRuntimeCoreGraphicsCompatibility), 0, formats, 1, 0, 1, nullptr, 0, 42, {0, 0, 0, 0}};
    VernonRuntimeCoreGraphicsVariant *first = nullptr;
    VernonRuntimeCoreGraphicsVariant *second = nullptr;
    ASSERT_EQ(vernonRuntimeCorePrepareGraphicsVariant(pipeline, &compatibility, &first), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCorePrepareGraphicsVariant(pipeline, &compatibility, &second), VERNON_STATUS_OK);
    EXPECT_EQ(mock.pipelinePreparations, 2u);

    VernonRuntimeCoreGraphicsCompatibility changed = compatibility;
    changed.vertex_layout_identity = 43;
    VernonRuntimeCoreGraphicsVariant *third = nullptr;
    ASSERT_EQ(vernonRuntimeCorePrepareGraphicsVariant(pipeline, &changed, &third), VERNON_STATUS_OK);
    EXPECT_EQ(mock.pipelinePreparations, 3u);

    vernonRuntimeCoreGraphicsVariantDestroy(third);
    vernonRuntimeCoreGraphicsVariantDestroy(second);
    vernonRuntimeCoreGraphicsVariantDestroy(first);
    vernonRuntimeCorePipelineDestroy(pipeline);
}

} // namespace
