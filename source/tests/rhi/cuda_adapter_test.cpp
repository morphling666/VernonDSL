#include "VernonRuntimeCore.h"
#include "VernonRuntimeRHIAdapter.h"
#include "runtime/rhi_adapter/adapter_test_hooks.h"

#include <gtest/gtest.h>

#include <vector>

TEST(CudaRhiAdapter, BorrowsOwnedRhiDeviceAndReferencesGenerationalBuffer) {
    VernonRhiOwnedDeviceDescriptor deviceDescriptor{};
    deviceDescriptor.struct_size = sizeof(deviceDescriptor);
    deviceDescriptor.backend = VERNON_RHI_BACKEND_CUDA;
    VernonRhiDevice device = vernonRhiCreateDevice(&deviceDescriptor);
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        GTEST_SKIP() << "CUDA RHI device is unavailable";

    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = sizeof(uint32_t) * 4;
    bufferDescriptor.usage =
        VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION | VERNON_RHI_BUFFER_STORAGE;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer buffer{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &buffer), VERNON_RHI_STATUS_OK);

    const uint32_t source[]{1, 2, 3, 4};
    uint32_t destination[4]{};
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(device, buffer, 0, source, sizeof(source)), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(device, buffer, 0, destination, sizeof(destination)), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(std::vector<uint32_t>(std::begin(destination), std::end(destination)),
              std::vector<uint32_t>(std::begin(source), std::end(source)));

    VernonRuntimeRhiAdapter *adapter = vernonRuntimeRhiAdapterCreateForDevice(device, VERNON_RHI_BACKEND_CUDA);
    ASSERT_NE(adapter, nullptr);
    VernonRuntimeProviderResourceReference reference{};
    EXPECT_EQ(vernonRuntimeRhiAdapterReferenceBuffer(adapter, buffer, 4, 12, &reference), VERNON_STATUS_OK);
    EXPECT_NE(reference.identity, 0u);
    EXPECT_NE(reference.resource.value, 0u);
    EXPECT_EQ(reference.offset, 4u);
    EXPECT_EQ(reference.size, 12u);

    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, buffer), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceIsBufferValid(device, buffer), 0u);
    EXPECT_EQ(vernonRuntimeRhiAdapterReferenceBuffer(adapter, buffer, 0, 16, &reference),
              VERNON_STATUS_INVALID_ARGUMENT);
    vernonRuntimeRhiAdapterDestroy(adapter);
    vernonRhiDestroyDevice(device);
}

TEST(CudaRhiAdapter, PreparesAndDispatchesThroughRuntimeCore) {
    VernonRhiOwnedDeviceDescriptor deviceDescriptor{};
    deviceDescriptor.struct_size = sizeof(deviceDescriptor);
    deviceDescriptor.backend = VERNON_RHI_BACKEND_CUDA;
    const VernonRhiDevice device = vernonRhiCreateDevice(&deviceDescriptor);
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        GTEST_SKIP() << "CUDA RHI adapter is unavailable";
    VernonRuntimeRhiAdapter *adapter = vernonRuntimeRhiAdapterCreateForDevice(device, VERNON_RHI_BACKEND_CUDA);
    ASSERT_NE(adapter, nullptr);

    static constexpr char ptx[] = R"(
.version 7.0
.target sm_52
.address_size 64
.visible .entry noop(
  .param .f32 factor
) {
  ret;
}
)";
    const VernonRuntimeProviderShaderDescriptor shader{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                       VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE,
                                                       {"ptx", 3},
                                                       ptx,
                                                       sizeof(ptx) - 1,
                                                       {"noop", 4},
                                                       {nullptr, 0},
                                                       {0, 0, 0, 0}};
    const VernonRuntimeProviderBindingLayoutEntry binding{
        0, 0, 0, VERNON_RUNTIME_PROVIDER_INLINE_VALUE, VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE, 1, 1, 0, sizeof(float)};
    VernonRuntimeCorePipelineDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.kind = VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
    descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_COMPUTE;
    descriptor.shaders = &shader;
    descriptor.shader_count = 1;
    descriptor.bindings = &binding;
    descriptor.binding_count = 1;
    descriptor.workgroup_size[0] = 1;
    descriptor.workgroup_size[1] = 1;
    descriptor.workgroup_size[2] = 1;

    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(vernonRuntimeRhiAdapterGetProvider(adapter), &descriptor, &pipeline),
              VERNON_STATUS_OK);
    float factor = 2.0f;
    VernonRuntimeProviderBindingValue value{};
    value.slot = 0;
    value.kind = VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
    value.payload.inline_value.data = &factor;
    value.payload.inline_value.size = sizeof(factor);
    VernonRuntimeCoreBindings *bindings = nullptr;
    ASSERT_EQ(vernonRuntimeCoreCreateBindings(pipeline, &value, 1, &bindings), VERNON_STATUS_OK);
    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderObject providerEncoder{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceCommandEncoder(adapter, encoder, &providerEncoder), VERNON_STATUS_OK);
    factor = 3.0f;
    ASSERT_EQ(vernonRuntimeCoreUpdateBindings(bindings, &value, 1), VERNON_STATUS_OK);
    const uint32_t groups[3]{1, 1, 1};
    EXPECT_EQ(vernonRuntimeCoreEncodeDispatch(pipeline, bindings, providerEncoder, groups, nullptr, 0),
              VERNON_STATUS_OK);
    factor = 4.0f;
    ASSERT_EQ(vernonRuntimeCoreUpdateBindings(bindings, &value, 1), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreEncodeDispatch(pipeline, bindings, providerEncoder, groups, nullptr, 0),
              VERNON_STATUS_OK);
    EXPECT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceSubmit(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyCommandEncoder(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRuntimeRhiAdapterSynchronize(adapter), VERNON_STATUS_OK);
    const vernon::runtime::RhiAdapterPreparationStats stats = vernon::runtime::getRhiAdapterPreparationStats(*adapter);
    EXPECT_EQ(stats.shaderPreparations, 1u);
    EXPECT_EQ(stats.layoutPreparations, 1u);
    EXPECT_EQ(stats.pipelinePreparations, 1u);
    EXPECT_EQ(stats.bindingCreations, 1u);
    EXPECT_EQ(stats.dispatches, 2u);

    vernonRuntimeCoreBindingsDestroy(bindings);
    vernonRuntimeCorePipelineDestroy(pipeline);
    vernonRuntimeRhiAdapterDestroy(adapter);
    vernonRhiDestroyDevice(device);
}
