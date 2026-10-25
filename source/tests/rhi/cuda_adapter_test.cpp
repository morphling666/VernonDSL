#include "VernonRuntimeCore.h"
#include "VernonRuntimeRHIAdapter.h"
#include "rhi/rhi_test_hooks.h"
#include "runtime/rhi_adapter/adapter_test_hooks.h"
#include "runtime_core_packets.h"
#include "runtime_rhi_test_utils.h"

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
    EXPECT_EQ(vernonRhiDestroyDevice(device), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDestroyDevice(device), VERNON_RHI_STATUS_INVALID_ARGUMENT);
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
    const VernonRuntimeDeviceProvider *provider = vernonRuntimeRhiAdapterGetProvider(adapter);
    VernonRuntimeCoreGroupLayout *groupLayout = nullptr;
    ASSERT_EQ(vernonRuntimeCorePrepareGroupLayout(provider, &binding, 1, &groupLayout), VERNON_STATUS_OK);
    VernonRuntimeCorePipelineDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.kind = VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
    descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_COMPUTE;
    descriptor.shaders = &shader;
    descriptor.shader_count = 1;
    descriptor.group_layouts = &groupLayout;
    descriptor.group_layout_count = 1;
    descriptor.workgroup_size[0] = 1;
    descriptor.workgroup_size[1] = 1;
    descriptor.workgroup_size[2] = 1;

    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    const auto createGroup = [&](float factor) {
        VernonRuntimeProviderBindingValue value{};
        value.slot = 0;
        value.kind = VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
        value.payload.inline_value.data = &factor;
        value.payload.inline_value.size = sizeof(factor);
        VernonRuntimeCoreGroup *group = nullptr;
        EXPECT_EQ(
            vernonRuntimeCoreCreateGroup(groupLayout, &value, 1, VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT, {}, &group),
            VERNON_STATUS_OK);
        return group;
    };
    VernonRuntimeCoreGroup *first = createGroup(3.0f);
    VernonRuntimeCoreGroup *second = createGroup(4.0f);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderObject providerEncoder{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceCommandEncoder(adapter, encoder, &providerEncoder), VERNON_STATUS_OK);
    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(provider, providerEncoder, &commands), VERNON_STATUS_OK);
    VernonRuntimeCorePackets *packets = nullptr;
    ASSERT_EQ(vernonRuntimeCorePacketsCreate(provider, &packets), VERNON_STATUS_OK);
    const uint32_t groups[3]{1, 1, 1};
    EXPECT_EQ(vernonRuntimeCorePacketsPushDebugGroup(packets, {"scale", 5}), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsBindPipeline(packets, pipeline), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsBindGroup(packets, 0, first), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsDispatch(packets, groups), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsInsertDebugMarker(packets, {"second", 6}), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsBindGroup(packets, 0, second), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsDispatch(packets, groups), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCorePacketsPopDebugGroup(packets), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::tests::encodeRecordedPackets(commands, packets), VERNON_STATUS_OK);
    vernonRuntimeCorePacketsDestroy(packets);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
    // The device has no labels, so the batch dropped its debug packets when it was created.
    const vernon::rhi::EncoderCallStats calls = vernon::rhi::getEncoderCallStats(device, encoder);
    EXPECT_EQ(calls.providerRecords, 1u);
    EXPECT_EQ(calls.debugCommands, 0u);
    EXPECT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernon::tests::completeSubmission(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRuntimeRhiAdapterSynchronize(adapter), VERNON_STATUS_OK);
    const vernon::runtime::RhiAdapterPreparationStats stats = vernon::runtime::getRhiAdapterPreparationStats(*adapter);
    EXPECT_EQ(stats.shaderPreparations, 1u);
    EXPECT_EQ(stats.layoutPreparations, 1u);
    EXPECT_EQ(stats.pipelinePreparations, 1u);
    EXPECT_EQ(stats.groupCreations, 2u);
    EXPECT_EQ(stats.dispatches, 2u);

    vernonRuntimeCoreGroupDestroy(second);
    vernonRuntimeCoreGroupDestroy(first);
    vernonRuntimeCorePipelineDestroy(pipeline);
    vernonRuntimeCoreGroupLayoutDestroy(groupLayout);
    vernonRuntimeRhiAdapterDestroy(adapter);
    vernonRhiDestroyDevice(device);
}

TEST(CudaRhiAdapter, HostStorageOfTransientGroupsReusesTheTransientMemoryOfRetiredRecordings) {
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
.visible .entry copy(
  .param .u64 source_allocated, .param .u64 source_aligned, .param .u64 source_offset, .param .u64 source_size,
  .param .u64 source_stride,
  .param .u64 target_allocated, .param .u64 target_aligned, .param .u64 target_offset, .param .u64 target_size,
  .param .u64 target_stride
) {
  .reg .u64 %rd<3>;
  .reg .u32 %r<2>;
  ld.param.u64 %rd1, [source_aligned];
  ld.param.u64 %rd2, [target_aligned];
  cvta.to.global.u64 %rd1, %rd1;
  cvta.to.global.u64 %rd2, %rd2;
  ld.global.u32 %r1, [%rd1];
  st.global.u32 [%rd2], %r1;
  ret;
}
)";
    const VernonRuntimeProviderShaderDescriptor shader{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                       VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE,
                                                       {"ptx", 3},
                                                       ptx,
                                                       sizeof(ptx) - 1,
                                                       {"copy", 4},
                                                       {nullptr, 0},
                                                       {0, 0, 0, 0}};
    const VernonRuntimeProviderBindingLayoutEntry bindings[2]{
        {0, 0, 0, VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER, VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE, 1, 1, 0,
         sizeof(uint32_t)},
        {1, 0, 1, VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER, VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE, 2, 1, 1,
         sizeof(uint32_t)}};
    const VernonRuntimeDeviceProvider *provider = vernonRuntimeRhiAdapterGetProvider(adapter);
    VernonRuntimeCoreGroupLayout *groupLayout = nullptr;
    ASSERT_EQ(vernonRuntimeCorePrepareGroupLayout(provider, bindings, 2, &groupLayout), VERNON_STATUS_OK);
    VernonRuntimeCorePipelineDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.kind = VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
    descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_COMPUTE;
    descriptor.shaders = &shader;
    descriptor.shader_count = 1;
    descriptor.group_layouts = &groupLayout;
    descriptor.group_layout_count = 1;
    descriptor.workgroup_size[0] = 1;
    descriptor.workgroup_size[1] = 1;
    descriptor.workgroup_size[2] = 1;
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(provider, &descriptor, &pipeline), VERNON_STATUS_OK);

    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = sizeof(uint32_t);
    bufferDescriptor.usage = VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_STORAGE;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer target{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &target), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference targetReference{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceBuffer(adapter, target, 0, sizeof(uint32_t), &targetReference),
              VERNON_STATUS_OK);

    const auto copyThroughTransientGroup = [&](uint32_t value) {
        VernonRhiCommandEncoderDescriptor encoderDescriptor{};
        encoderDescriptor.struct_size = sizeof(encoderDescriptor);
        encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
        VernonRhiCommandEncoder encoder{};
        ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
        VernonRuntimeProviderObject providerEncoder{};
        ASSERT_EQ(vernonRuntimeRhiAdapterReferenceCommandEncoder(adapter, encoder, &providerEncoder), VERNON_STATUS_OK);
        VernonRuntimeProviderBindingValue values[2]{};
        values[0].slot = 0;
        values[0].kind = VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
        values[0].flags = VERNON_RUNTIME_PROVIDER_BINDING_HOST_STORAGE;
        values[0].payload.inline_value.data = &value;
        values[0].payload.inline_value.size = sizeof(value);
        values[1].slot = 1;
        values[1].kind = VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
        values[1].payload.buffer.resource = targetReference;
        VernonRuntimeCoreGroup *group = nullptr;
        ASSERT_EQ(vernonRuntimeCoreCreateGroup(groupLayout, values, 2, VERNON_RUNTIME_PROVIDER_GROUP_TRANSIENT,
                                               providerEncoder, &group),
                  VERNON_STATUS_OK);
        VernonRuntimeCoreCommands *commands = nullptr;
        ASSERT_EQ(vernonRuntimeCoreBeginCommands(provider, providerEncoder, &commands), VERNON_STATUS_OK);
        VernonRuntimeCorePackets *packets = nullptr;
        ASSERT_EQ(vernonRuntimeCorePacketsCreate(provider, &packets), VERNON_STATUS_OK);
        const uint32_t groups[3]{1, 1, 1};
        EXPECT_EQ(vernonRuntimeCorePacketsBindPipeline(packets, pipeline), VERNON_STATUS_OK);
        EXPECT_EQ(vernonRuntimeCorePacketsBindGroup(packets, 0, group), VERNON_STATUS_OK);
        EXPECT_EQ(vernonRuntimeCorePacketsDispatch(packets, groups), VERNON_STATUS_OK);
        EXPECT_EQ(vernon::tests::encodeRecordedPackets(commands, packets), VERNON_STATUS_OK);
        vernonRuntimeCorePacketsDestroy(packets);
        EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
        vernonRuntimeCoreGroupDestroy(group);
        EXPECT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(vernon::tests::completeSubmission(device, encoder), VERNON_RHI_STATUS_OK);
        uint32_t copied{};
        EXPECT_EQ(vernonRhiDeviceDownloadBuffer(device, target, 0, &copied, sizeof(copied)), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(copied, value);
    };
    copyThroughTransientGroup(1);
    const size_t warmAllocations = vernon::rhi::getCudaAllocationCount(device);
    EXPECT_GT(warmAllocations, 0u);
    for (uint32_t value = 2; value < 10; ++value)
        copyThroughTransientGroup(value);
    EXPECT_EQ(vernon::rhi::getCudaAllocationCount(device), warmAllocations);

    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, target), VERNON_RHI_STATUS_OK);
    vernonRuntimeCorePipelineDestroy(pipeline);
    vernonRuntimeCoreGroupLayoutDestroy(groupLayout);
    vernonRuntimeRhiAdapterDestroy(adapter);
    vernonRhiDestroyDevice(device);
}
