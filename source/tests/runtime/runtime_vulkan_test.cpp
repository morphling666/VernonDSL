#include "../../lib/rhi/rhi_internal.h"
#include "VernonRuntime.h"
#include "runtime_rhi_test_utils.h"

#include <gtest/gtest.h>
#include <vector>

TEST(RuntimeVulkan, RecordsIndependentCommandEncoders) {
    VernonRuntimeCapabilities capabilities = vernonRuntimeGetCapabilities(VERNON_RUNTIME_VULKAN);
    if (!capabilities.available)
        GTEST_SKIP() << "Vulkan runtime backend is unavailable";

    auto context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_VULKAN);
    ASSERT_TRUE(context.runtime);
    EXPECT_NE(vernon::rhi::deviceCommandCapabilities(context.device) & vernon::rhi::BackendCommandIndependentRecording,
              0u);
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder first{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    VernonRhiCommandEncoder second{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(context.device, &descriptor, &first), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(context.device, &descriptor, &second), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(context.device, second), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(context.device, first), VERNON_RHI_STATUS_OK);
    VernonRhiCompletion firstCompletion{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    VernonRhiCompletion secondCompletion{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceSubmit(context.device, second, &secondCompletion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceSubmit(context.device, first, &firstCompletion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiCompletionWait(context.device, firstCompletion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiCompletionWait(context.device, secondCompletion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyCompletion(context.device, firstCompletion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyCompletion(context.device, secondCompletion), VERNON_RHI_STATUS_OK);
    EXPECT_TRUE(vernonRuntimeDestroy(context.runtime) == VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
}

TEST(RuntimeVulkan, BoundsIndependentCommandRecordings) {
    VernonRuntimeCapabilities capabilities = vernonRuntimeGetCapabilities(VERNON_RUNTIME_VULKAN);
    if (!capabilities.available)
        GTEST_SKIP() << "Vulkan runtime backend is unavailable";

    auto context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_VULKAN);
    ASSERT_TRUE(context.runtime);
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    std::vector<VernonRhiCommandEncoder> encoders;
    VernonRhiStatus status = VERNON_RHI_STATUS_OK;
    for (size_t attempt = 0; attempt < 64 && status == VERNON_RHI_STATUS_OK; ++attempt) {
        VernonRhiCommandEncoder encoder{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
        status = vernonRhiDeviceCreateCommandEncoder(context.device, &descriptor, &encoder);
        if (status == VERNON_RHI_STATUS_OK)
            encoders.push_back(encoder);
    }
    ASSERT_EQ(status, VERNON_RHI_STATUS_RESOURCE_EXHAUSTED);
    ASSERT_GT(encoders.size(), 1u);
    ASSERT_LT(encoders.size(), 64u);
    ASSERT_EQ(vernonRhiDeviceDestroyCommandEncoder(context.device, encoders.back()), VERNON_RHI_STATUS_OK);
    encoders.pop_back();
    VernonRhiCommandEncoder replacement{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    EXPECT_EQ(vernonRhiDeviceCreateCommandEncoder(context.device, &descriptor, &replacement), VERNON_RHI_STATUS_OK);
    if (replacement.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        EXPECT_EQ(vernonRhiDeviceDestroyCommandEncoder(context.device, replacement), VERNON_RHI_STATUS_OK);
    for (VernonRhiCommandEncoder encoder : encoders)
        EXPECT_EQ(vernonRhiDeviceDestroyCommandEncoder(context.device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_TRUE(vernonRuntimeDestroy(context.runtime) == VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
}

TEST(RuntimeVulkan, HonorsConfiguredCommandLimits) {
    VernonRuntimeCapabilities capabilities = vernonRuntimeGetCapabilities(VERNON_RUNTIME_VULKAN);
    if (!capabilities.available)
        GTEST_SKIP() << "Vulkan runtime backend is unavailable";

    auto context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_VULKAN);
    ASSERT_TRUE(context.runtime);
    VernonRhiCommandLimits limits{};
    limits.struct_size = sizeof(limits);
    ASSERT_EQ(vernonRhiDeviceGetCommandLimits(context.device, &limits), VERNON_RHI_STATUS_OK);
    limits.max_active_recordings = 2;
    limits.max_in_flight_submissions = 2;
    limits.max_live_completions = 2;
    limits.max_upload_bytes = sizeof(uint32_t);
    ASSERT_EQ(vernonRhiDeviceSetCommandLimits(context.device, &limits), VERNON_RHI_STATUS_OK);

    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder first{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    VernonRhiCommandEncoder second{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    VernonRhiCommandEncoder exhausted{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(context.device, &descriptor, &first), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(context.device, &descriptor, &second), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceCreateCommandEncoder(context.device, &descriptor, &exhausted),
              VERNON_RHI_STATUS_RESOURCE_EXHAUSTED);
    EXPECT_EQ(vernonRhiDeviceSetCommandLimits(context.device, &limits), VERNON_RHI_STATUS_INVALID_ARGUMENT);

    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = sizeof(uint64_t);
    bufferDescriptor.alignment = alignof(uint64_t);
    bufferDescriptor.usage = VERNON_RHI_BUFFER_TRANSFER_DESTINATION;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer buffer{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(context.device, &bufferDescriptor, &buffer), VERNON_RHI_STATUS_OK);
    const uint64_t value = 1;
    EXPECT_EQ(vernonRhiCommandEncoderUploadBuffer(context.device, first, buffer, 0, &value, sizeof(value)),
              VERNON_RHI_STATUS_RESOURCE_EXHAUSTED);

    EXPECT_EQ(vernonRhiDeviceDestroyCommandEncoder(context.device, first), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyCommandEncoder(context.device, second), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(context.device, buffer), VERNON_RHI_STATUS_OK);
    EXPECT_TRUE(vernonRuntimeDestroy(context.runtime) == VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
}

TEST(RuntimeVulkan, BoundsLiveCompletionHandles) {
    VernonRuntimeCapabilities capabilities = vernonRuntimeGetCapabilities(VERNON_RUNTIME_VULKAN);
    if (!capabilities.available)
        GTEST_SKIP() << "Vulkan runtime backend is unavailable";

    auto context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_VULKAN);
    ASSERT_TRUE(context.runtime);
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    std::vector<VernonRhiCompletion> completions;
    VernonRhiCommandEncoder blocked{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    VernonRhiStatus status = VERNON_RHI_STATUS_OK;
    for (size_t attempt = 0; attempt < 128 && status == VERNON_RHI_STATUS_OK; ++attempt) {
        VernonRhiCommandEncoder encoder{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
        ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(context.device, &descriptor, &encoder), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiCommandEncoderFinish(context.device, encoder), VERNON_RHI_STATUS_OK);
        VernonRhiCompletion completion{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
        status = vernonRhiDeviceSubmit(context.device, encoder, &completion);
        if (status == VERNON_RHI_STATUS_OK)
            completions.push_back(completion);
        else
            blocked = encoder;
    }
    ASSERT_EQ(status, VERNON_RHI_STATUS_RESOURCE_EXHAUSTED);
    ASSERT_GT(completions.size(), 1u);
    ASSERT_LT(completions.size(), 128u);
    ASSERT_EQ(vernonRhiDeviceDestroyCompletion(context.device, completions.back()), VERNON_RHI_STATUS_OK);
    completions.pop_back();
    VernonRhiCompletion replacement{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    EXPECT_EQ(vernonRhiDeviceSubmit(context.device, blocked, &replacement), VERNON_RHI_STATUS_OK);
    if (replacement.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        completions.push_back(replacement);
    for (VernonRhiCompletion completion : completions)
        EXPECT_EQ(vernonRhiDeviceDestroyCompletion(context.device, completion), VERNON_RHI_STATUS_OK);
    EXPECT_TRUE(vernonRuntimeDestroy(context.runtime) == VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
}

TEST(RuntimeVulkan, RecordsUploadInActiveCommandEncoder) {
    VernonRuntimeCapabilities capabilities = vernonRuntimeGetCapabilities(VERNON_RUNTIME_VULKAN);
    if (!capabilities.available)
        GTEST_SKIP() << "Vulkan runtime backend is unavailable";

    auto context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_VULKAN);
    ASSERT_TRUE(context.runtime);
    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = sizeof(uint32_t);
    bufferDescriptor.alignment = alignof(uint32_t);
    bufferDescriptor.usage = VERNON_RHI_BUFFER_TRANSFER_DESTINATION | VERNON_RHI_BUFFER_TRANSFER_SOURCE;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer buffer{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(context.device, &bufferDescriptor, &buffer), VERNON_RHI_STATUS_OK);

    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_TRANSFER;
    VernonRhiCommandEncoder encoder{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(context.device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
    const uint32_t expected = 0x1234abcdu;
    ASSERT_EQ(vernonRhiCommandEncoderUploadBuffer(context.device, encoder, buffer, 0, &expected, sizeof(expected)),
              VERNON_RHI_STATUS_OK);
    VernonRhiBarrier barrier{};
    barrier.struct_size = sizeof(barrier);
    barrier.destination_stage_mask = VERNON_RHI_STAGE_COMPUTE;
    barrier.source_access = VERNON_RHI_ACCESS_TRANSFER_WRITE;
    barrier.destination_access = VERNON_RHI_ACCESS_SHADER_READ;
    barrier.old_state = VERNON_RHI_STATE_TRANSFER_DESTINATION;
    barrier.new_state = VERNON_RHI_STATE_SHADER_READ;
    barrier.buffer = buffer;
    ASSERT_EQ(vernonRhiCommandEncoderBarrier(context.device, encoder, &barrier, 1), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(context.device, encoder), VERNON_RHI_STATUS_OK);
    VernonRhiCompletion completion{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceSubmit(context.device, encoder, &completion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCompletionWait(context.device, completion), VERNON_RHI_STATUS_OK);

    uint32_t actual = 0;
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(context.device, buffer, 0, &actual, sizeof(actual)), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(vernonRhiDeviceDestroyCompletion(context.device, completion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(context.device, buffer), VERNON_RHI_STATUS_OK);
    EXPECT_TRUE(vernonRuntimeDestroy(context.runtime) == VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
}