#include "../../lib/rhi/rhi_test_hooks.h"
#include "VernonRuntimeRHIAdapter.h"

#include <gtest/gtest.h>

#if defined(_WIN32)

TEST(DirectXResourceRetention, RetainedResourcePreventsRhiSlotReuse) {
    VernonRhiOwnedDeviceDescriptor deviceDescriptor{};
    deviceDescriptor.struct_size = sizeof(deviceDescriptor);
    deviceDescriptor.backend = VERNON_RHI_BACKEND_DIRECTX12;
    const VernonRhiDevice device = vernonRhiCreateDevice(&deviceDescriptor);
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        GTEST_SKIP() << "DirectX 12 RHI is unavailable";
    VernonRuntimeRhiAdapter *adapter = vernonRuntimeRhiAdapterCreateForDevice(device, VERNON_RHI_BACKEND_DIRECTX12);
    ASSERT_NE(adapter, nullptr);
    const VernonRuntimeDeviceProvider *provider = vernonRuntimeRhiAdapterGetProvider(adapter);
    ASSERT_NE(provider, nullptr);

    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = 64;
    bufferDescriptor.usage = VERNON_RHI_BUFFER_STORAGE;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer first{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &first), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference retained{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceBuffer(adapter, first, 0, 64, &retained), VERNON_STATUS_OK);
    ASSERT_EQ(provider->retain_resource(provider->user_data, retained), VERNON_STATUS_OK);
    ASSERT_EQ(provider->retain_resource(provider->user_data, retained), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(device, first), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceIsBufferValid(device, first), 0u);
    EXPECT_EQ(provider->retain_resource(provider->user_data, retained), VERNON_STATUS_OK);

    VernonRhiBuffer replacement{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &replacement), VERNON_RHI_STATUS_OK);
    EXPECT_NE(replacement.index, first.index);
    provider->release_resource(provider->user_data, retained);
    provider->release_resource(provider->user_data, retained);
    provider->release_resource(provider->user_data, retained);

    VernonRhiBuffer recycled{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &recycled), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(recycled.index, first.index);
    EXPECT_NE(recycled.generation, first.generation);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, replacement), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, recycled), VERNON_RHI_STATUS_OK);

    vernonRuntimeRhiAdapterDestroy(adapter);
    vernonRhiDestroyDevice(device);
}

TEST(DirectXResourceRetention, AbandonedEncoderRollsBackTrackedResourceState) {
    VernonRhiOwnedDeviceDescriptor deviceDescriptor{};
    deviceDescriptor.struct_size = sizeof(deviceDescriptor);
    deviceDescriptor.backend = VERNON_RHI_BACKEND_DIRECTX12;
    const VernonRhiDevice device = vernonRhiCreateDevice(&deviceDescriptor);
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        GTEST_SKIP() << "DirectX 12 RHI is unavailable";

    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = 64;
    bufferDescriptor.usage = VERNON_RHI_BUFFER_STORAGE;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer buffer{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &buffer), VERNON_RHI_STATUS_OK);
    const uint64_t initialState = vernon::rhi::getTrackedBufferState(device, buffer);

    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
    VernonRhiBarrier barrier{};
    barrier.struct_size = sizeof(barrier);
    barrier.source_stage_mask = VERNON_RHI_STAGE_COMPUTE;
    barrier.destination_stage_mask = VERNON_RHI_STAGE_COMPUTE;
    barrier.source_access = VERNON_RHI_ACCESS_SHADER_READ;
    barrier.destination_access = VERNON_RHI_ACCESS_SHADER_WRITE;
    barrier.old_state = VERNON_RHI_STATE_COMMON;
    barrier.new_state = VERNON_RHI_STATE_SHADER_WRITE;
    barrier.buffer = buffer;
    ASSERT_EQ(vernonRhiCommandEncoderBarrier(device, encoder, &barrier, 1), VERNON_RHI_STATUS_OK);
    EXPECT_NE(vernon::rhi::getTrackedBufferState(device, buffer), initialState);
    ASSERT_EQ(vernonRhiDeviceDestroyCommandEncoder(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernon::rhi::getTrackedBufferState(device, buffer), initialState);

    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, buffer), VERNON_RHI_STATUS_OK);
    vernonRhiDestroyDevice(device);
}

#endif
