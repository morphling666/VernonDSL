#include "VernonFormats.hpp"
#include "VernonRuntimeRHIAdapter.h"
#include "backend_runtime_owner.h"
#include "backend_test_matrix.h"
#include "execution_graph/command_graph.h"
#include "rhi/rhi_internal.h"
#include "vernon_test_support.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstring>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace {

class RhiResourceLifetime : public testing::TestWithParam<vernon::tests::BackendTestRow> {
protected:
    void SetUp() override {
        ASSERT_TRUE(GetParam().rhi.has_value());
        const vernon::tests::BackendProbeResult probe = runtime_.initialize(GetParam(), {});
        if (!probe.available()) {
            if (probe.skippable())
                GTEST_SKIP() << probe.reason;
            FAIL() << probe.reason;
        }
    }

    VernonRhiDevice device() { return runtime_.context().device; }
    VernonRuntimeContext *runtime() { return runtime_.context().runtime; }
    vernon::tests::BackendProbeResult graphicsProbe() {
        vernon::tests::BackendTestRequirements requirements;
        requirements.graphics = true;
        return vernon::tests::probeRuntimeBackend(GetParam(), requirements, runtime());
    }
    vernon::tests::BackendProbeResult imageSubresourceViewProbe() {
        vernon::tests::BackendTestRequirements requirements;
        requirements.graphics = true;
        if (GetParam().runtime == VERNON_RUNTIME_OPENGL) {
            requirements.minimumApiMajor = 4;
            requirements.minimumApiMinor = 3;
        } else if (GetParam().runtime == VERNON_RUNTIME_OPENGL_ES) {
            requirements.minimumApiMajor = 3;
            requirements.minimumApiMinor = 2;
        }
        return vernon::tests::probeRuntimeBackend(GetParam(), requirements, runtime());
    }

private:
    vernon::tests::BackendRuntimeOwner runtime_;
};

TEST_P(RhiResourceLifetime, BatchedBufferUploadsValidateBeforeMutation) {
    const VernonRhiDevice device = this->device();

    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = 32;
    bufferDescriptor.usage =
        VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION | VERNON_RHI_BUFFER_STORAGE;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer buffer{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &buffer), VERNON_RHI_STATUS_OK);

    std::array<uint8_t, 32> contents{};
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(device, buffer, 0, contents.data(), contents.size()), VERNON_RHI_STATUS_OK);
    const std::array<uint8_t, 3> first{1, 2, 3};
    const std::array<uint8_t, 4> second{4, 5, 6, 7};
    const std::array<VernonRhiBufferUploadRange, 2> valid{
        VernonRhiBufferUploadRange{4, first.data(), first.size()},
        VernonRhiBufferUploadRange{20, second.data(), second.size()},
    };
    ASSERT_EQ(vernonRhiDeviceUploadBufferRanges(device, buffer, valid.data(), valid.size()), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(device, buffer, 0, contents.data(), contents.size()), VERNON_RHI_STATUS_OK);
    const std::array<uint8_t, 3> actualFirst{contents[4], contents[5], contents[6]};
    const std::array<uint8_t, 4> actualSecond{contents[20], contents[21], contents[22], contents[23]};
    EXPECT_EQ(actualFirst, first);
    EXPECT_EQ(actualSecond, second);
    std::array<uint8_t, 3> partialDownload{};
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(device, buffer, 4, partialDownload.data(), partialDownload.size()),
              VERNON_RHI_STATUS_OK);
    EXPECT_EQ(partialDownload, first);
    std::array<uint8_t, 3> batchedFirst{};
    std::array<uint8_t, 4> batchedSecond{};
    const std::array<VernonRhiBufferDownloadRange, 2> downloads{
        VernonRhiBufferDownloadRange{4, batchedFirst.data(), batchedFirst.size()},
        VernonRhiBufferDownloadRange{20, batchedSecond.data(), batchedSecond.size()},
    };
    ASSERT_EQ(vernonRhiDeviceDownloadBufferRanges(device, buffer, downloads.data(), downloads.size()),
              VERNON_RHI_STATUS_OK);
    EXPECT_EQ(batchedFirst, first);
    EXPECT_EQ(batchedSecond, second);

    const std::array<uint8_t, 2> rejected{9, 9};
    const std::array<VernonRhiBufferUploadRange, 2> invalid{
        VernonRhiBufferUploadRange{0, rejected.data(), rejected.size()},
        VernonRhiBufferUploadRange{31, rejected.data(), rejected.size()},
    };
    EXPECT_EQ(vernonRhiDeviceUploadBufferRanges(device, buffer, invalid.data(), invalid.size()),
              VERNON_RHI_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(device, buffer, 0, contents.data(), contents.size()), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(contents[0], 0);
    EXPECT_EQ(contents[1], 0);
    batchedFirst.fill(0xaa);
    batchedSecond.fill(0xbb);
    const std::array<VernonRhiBufferDownloadRange, 2> invalidDownloads{
        VernonRhiBufferDownloadRange{4, batchedFirst.data(), batchedFirst.size()},
        VernonRhiBufferDownloadRange{31, batchedSecond.data(), batchedSecond.size()},
    };
    EXPECT_EQ(vernonRhiDeviceDownloadBufferRanges(device, buffer, invalidDownloads.data(), invalidDownloads.size()),
              VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(batchedFirst, (std::array<uint8_t, 3>{0xaa, 0xaa, 0xaa}));
    EXPECT_EQ(batchedSecond, (std::array<uint8_t, 4>{0xbb, 0xbb, 0xbb, 0xbb}));
    EXPECT_EQ(vernonRhiDeviceUploadBufferRanges(device, buffer, nullptr, 0), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiDeviceDownloadBufferRanges(device, buffer, nullptr, 0), VERNON_RHI_STATUS_INVALID_ARGUMENT);

    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, buffer), VERNON_RHI_STATUS_OK);
}

TEST_P(RhiResourceLifetime, DeviceUploadsAreOrderedBetweenUnwaitedSubmissions) {
    const VernonRhiDevice device = this->device();
    constexpr uint32_t frameCount = 128;
    VernonRhiCommandLimits limits{};
    limits.struct_size = sizeof(limits);
    ASSERT_EQ(vernonRhiDeviceGetCommandLimits(device, &limits), VERNON_RHI_STATUS_OK);
    limits.max_in_flight_submissions = frameCount;
    limits.max_live_completions = frameCount;
    ASSERT_EQ(vernonRhiDeviceSetCommandLimits(device, &limits), VERNON_RHI_STATUS_OK);
    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = sizeof(uint32_t);
    bufferDescriptor.usage = VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer source{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &source), VERNON_RHI_STATUS_OK);
    bufferDescriptor.size = frameCount * sizeof(uint32_t);
    VernonRhiBuffer destination{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &destination), VERNON_RHI_STATUS_OK);

    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_TRANSFER;
    std::vector<VernonRhiCompletion> completions;
    for (uint32_t frame = 0; frame < frameCount; ++frame) {
        const uint32_t value = frame * 7 + 1;
        ASSERT_EQ(vernonRhiDeviceUploadBuffer(device, source, 0, &value, sizeof(value)), VERNON_RHI_STATUS_OK);
        VernonRhiCommandEncoder encoder{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
        ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiCommandEncoderCopyBuffer(device, encoder, source, 0, destination, frame * sizeof(value),
                                                    sizeof(value)),
                  VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
        VernonRhiCompletion completion{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
        ASSERT_EQ(vernonRhiDeviceSubmit(device, encoder, &completion), VERNON_RHI_STATUS_OK);
        completions.push_back(completion);
    }
    for (VernonRhiCompletion completion : completions) {
        EXPECT_EQ(vernonRhiCompletionWait(device, completion), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(vernonRhiDeviceDestroyCompletion(device, completion), VERNON_RHI_STATUS_OK);
    }

    std::vector<uint32_t> copied(frameCount);
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(device, destination, 0, copied.data(), copied.size() * sizeof(uint32_t)),
              VERNON_RHI_STATUS_OK);
    for (uint32_t frame = 0; frame < frameCount; ++frame)
        EXPECT_EQ(copied[frame], frame * 7 + 1) << "frame " << frame;
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, source), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, destination), VERNON_RHI_STATUS_OK);
}

TEST_P(RhiResourceLifetime, CommandGraphDestroysOwnedBuffersButNotImportedBuffers) {
    const VernonRhiDevice device = this->device();

    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = 64;
    bufferDescriptor.usage = VERNON_RHI_BUFFER_STORAGE;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;

    VernonRhiBuffer imported{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &imported), VERNON_RHI_STATUS_OK);
    {
        vernon::execution::CommandGraph graph(device);
        const vernon::execution::GraphBuffer first = graph.importBuffer(imported);
        const vernon::execution::GraphBuffer second = graph.importBuffer(imported);
        EXPECT_EQ(first.id, second.id);
    }
    EXPECT_EQ(vernonRhiDeviceIsBufferValid(device, imported), 1u);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, imported), VERNON_RHI_STATUS_OK);

    VernonRhiBuffer owned{};
    {
        vernon::execution::CommandGraph graph(device);
        vernon::execution::GraphBuffer graphBuffer;
        ASSERT_EQ(graph.createBuffer(bufferDescriptor, graphBuffer), VERNON_RHI_STATUS_OK);
        owned = graphBuffer.handle;
        const vernon::execution::GraphBuffer duplicate = graph.importBuffer(owned);
        EXPECT_EQ(graphBuffer.id, duplicate.id);
        EXPECT_EQ(vernonRhiDeviceIsBufferValid(device, owned), 1u);
    }
    EXPECT_EQ(vernonRhiDeviceIsBufferValid(device, owned), 0u);

    VernonRhiBuffer recycled{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &recycled), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(recycled.index, owned.index);
    EXPECT_NE(recycled.generation, owned.generation);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, recycled), VERNON_RHI_STATUS_OK);
}

TEST_P(RhiResourceLifetime, RecordsIndependentCommandEncoders) {
    const VernonRhiDevice device = this->device();
    if ((vernon::rhi::deviceCommandCapabilities(device) & vernon::rhi::BackendCommandIndependentRecording) == 0)
        GTEST_SKIP() << GetParam().name << " does not support independent command recording";

    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder first{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    VernonRhiCommandEncoder second{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &first), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &second), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, second), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, first), VERNON_RHI_STATUS_OK);

    VernonRhiCompletion firstCompletion{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    VernonRhiCompletion secondCompletion{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceSubmit(device, second, &secondCompletion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceSubmit(device, first, &firstCompletion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiCompletionWait(device, firstCompletion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiCompletionWait(device, secondCompletion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyCompletion(device, firstCompletion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyCompletion(device, secondCompletion), VERNON_RHI_STATUS_OK);
}

TEST_P(RhiResourceLifetime, CommandDestroyRejectsConcurrentOperationPinAndCanRetry) {
    const VernonRhiDevice device = this->device();
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;

    for (unsigned iteration = 0; iteration != 64; ++iteration) {
        VernonRhiCommandEncoder encoder{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
        ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &encoder), VERNON_RHI_STATUS_OK);
        std::atomic<bool> start{};
        VernonRhiStatus finishStatus = VERNON_RHI_STATUS_INTERNAL_ERROR;
        VernonRhiStatus destroyStatus = VERNON_RHI_STATUS_INTERNAL_ERROR;
        std::thread finisher([&] {
            while (!start.load(std::memory_order_acquire)) {
            }
            finishStatus = vernonRhiCommandEncoderFinish(device, encoder);
        });
        std::thread destroyer([&] {
            while (!start.load(std::memory_order_acquire)) {
            }
            destroyStatus = vernonRhiDeviceDestroyCommandEncoder(device, encoder);
        });
        start.store(true, std::memory_order_release);
        finisher.join();
        destroyer.join();
        EXPECT_TRUE(finishStatus == VERNON_RHI_STATUS_OK || finishStatus == VERNON_RHI_STATUS_INVALID_ARGUMENT);
        EXPECT_TRUE(destroyStatus == VERNON_RHI_STATUS_OK || destroyStatus == VERNON_RHI_STATUS_INVALID_ARGUMENT);
        if (destroyStatus != VERNON_RHI_STATUS_OK)
            EXPECT_EQ(vernonRhiDeviceDestroyCommandEncoder(device, encoder), VERNON_RHI_STATUS_OK);
    }
}

TEST_P(RhiResourceLifetime, CommandChildrenKeepDevicePublishedUntilExplicitDestroy) {
    const VernonRhiDevice device = this->device();
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder encoder{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &encoder), VERNON_RHI_STATUS_OK);

    EXPECT_EQ(vernonRhiDestroyDevice(device), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_TRUE(vernon::rhi::deviceExists(device));
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    VernonRhiCompletion completion{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceSubmit(device, encoder, &completion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCompletionWait(device, completion), VERNON_RHI_STATUS_OK);

    EXPECT_EQ(vernonRhiDestroyDevice(device), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_TRUE(vernon::rhi::deviceExists(device));
    ASSERT_EQ(vernonRhiDeviceDestroyCompletion(device, completion), VERNON_RHI_STATUS_OK);
    EXPECT_TRUE(vernon::rhi::deviceExists(device));
}

TEST_P(RhiResourceLifetime, FailedCommandDestroyRollsBackAndCanRetry) {
    const VernonRhiDevice device = this->device();
    const vernon::tests::BackendProbeResult probe = graphicsProbe();
    if (!probe.available()) {
        if (probe.skippable())
            GTEST_SKIP() << probe.reason;
        FAIL() << probe.reason;
        return;
    }
    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.width = 4;
    imageDescriptor.height = 4;
    imageDescriptor.depth = 1;
    imageDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageDescriptor.mip_levels = 1;
    imageDescriptor.array_layers = 1;
    imageDescriptor.sample_count = 1;
    imageDescriptor.usage = VERNON_RHI_IMAGE_COLOR_ATTACHMENT;
    VernonRhiImage image{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &image), VERNON_RHI_STATUS_OK);
    VernonRhiImageViewDescriptor viewDescriptor{};
    viewDescriptor.struct_size = sizeof(viewDescriptor);
    viewDescriptor.image = image;
    viewDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    viewDescriptor.format = imageDescriptor.format;
    viewDescriptor.mip_level_count = 1;
    viewDescriptor.array_layer_count = 1;
    viewDescriptor.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
    VernonRhiImageView view{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateImageView(device, &viewDescriptor, &view), VERNON_RHI_STATUS_OK);

    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_GRAPHICS;
    VernonRhiCommandEncoder encoder{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &encoder), VERNON_RHI_STATUS_OK);
    VernonRhiColorAttachment color{};
    color.view = view;
    color.initial_state = VERNON_RHI_STATE_COLOR_ATTACHMENT;
    color.final_state = VERNON_RHI_STATE_COLOR_ATTACHMENT;
    color.load_operation = VERNON_RHI_LOAD_PRESERVE;
    color.store_operation = VERNON_RHI_STORE_PRESERVE;
    VernonRhiRenderingDescriptor rendering{};
    rendering.struct_size = sizeof(rendering);
    rendering.color_attachments = &color;
    rendering.color_attachment_count = 1;
    rendering.width = imageDescriptor.width;
    rendering.height = imageDescriptor.height;
    rendering.layers = 1;
    ASSERT_EQ(vernonRhiCommandEncoderBeginRendering(device, encoder, &rendering), VERNON_RHI_STATUS_OK);

    EXPECT_EQ(vernonRhiDeviceDestroyCommandEncoder(device, encoder), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiCommandEncoderEndRendering(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyCommandEncoder(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImageView(device, view), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
}

TEST_P(RhiResourceLifetime, RenderingWithoutDrawsAppliesItsClears) {
    const VernonRhiDevice device = this->device();
    const vernon::tests::BackendProbeResult probe = graphicsProbe();
    if (!probe.available()) {
        if (probe.skippable())
            GTEST_SKIP() << probe.reason;
        FAIL() << probe.reason;
        return;
    }
    VernonRuntimeRhiAdapter *adapter = vernonRuntimeRhiAdapterCreateForDevice(device, *GetParam().rhi);
    ASSERT_NE(adapter, nullptr);
    constexpr uint32_t extent = 4;
    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.width = extent;
    imageDescriptor.height = extent;
    imageDescriptor.depth = 1;
    imageDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageDescriptor.mip_levels = 1;
    imageDescriptor.array_layers = 1;
    imageDescriptor.sample_count = 1;
    imageDescriptor.usage = VERNON_RHI_IMAGE_COLOR_ATTACHMENT;
    VernonRhiImage image{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &image), VERNON_RHI_STATUS_OK);
    VernonRhiImageViewDescriptor viewDescriptor{};
    viewDescriptor.struct_size = sizeof(viewDescriptor);
    viewDescriptor.image = image;
    viewDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    viewDescriptor.format = imageDescriptor.format;
    viewDescriptor.mip_level_count = 1;
    viewDescriptor.array_layer_count = 1;
    viewDescriptor.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
    VernonRhiImageView view{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateImageView(device, &viewDescriptor, &view), VERNON_RHI_STATUS_OK);

    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_GRAPHICS;
    VernonRhiCommandEncoder encoder{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &encoder), VERNON_RHI_STATUS_OK);
    VernonRhiColorAttachment color{};
    color.view = view;
    color.initial_state = VERNON_RHI_STATE_COLOR_ATTACHMENT;
    color.final_state = VERNON_RHI_STATE_COLOR_ATTACHMENT;
    color.load_operation = VERNON_RHI_LOAD_CLEAR;
    color.store_operation = VERNON_RHI_STORE_PRESERVE;
    color.clear_color.f32[1] = 1.0f;
    color.clear_color.f32[3] = 1.0f;
    VernonRhiRenderingDescriptor rendering{};
    rendering.struct_size = sizeof(rendering);
    rendering.color_attachments = &color;
    rendering.color_attachment_count = 1;
    rendering.width = extent;
    rendering.height = extent;
    rendering.layers = 1;
    ASSERT_EQ(vernonRhiCommandEncoderBeginRendering(device, encoder, &rendering), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderEndRendering(device, encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    VernonRhiCompletion completion{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceSubmit(device, encoder, &completion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCompletionWait(device, completion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyCompletion(device, completion), VERNON_RHI_STATUS_OK);

    std::vector<uint8_t> pixels(extent * extent * 4);
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
    download.width = extent;
    download.height = extent;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(device, image, &download, pixels.data(), pixels.size()),
              VERNON_RHI_STATUS_OK);
    for (size_t pixel = 0; pixel < pixels.size(); pixel += 4)
        EXPECT_EQ((std::array<uint8_t, 4>{pixels[pixel], pixels[pixel + 1], pixels[pixel + 2], pixels[pixel + 3]}),
                  (std::array<uint8_t, 4>{0, 255, 0, 255}))
            << "pixel " << pixel / 4;
    EXPECT_EQ(vernonRhiDeviceDestroyImageView(device, view), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
    vernonRuntimeRhiAdapterDestroy(adapter);
}

TEST_P(RhiResourceLifetime, GuessedCompletionCannotObservePartiallyInitializedSubmission) {
    const VernonRhiDevice device = this->device();
    if ((vernon::rhi::deviceCommandCapabilities(device) & vernon::rhi::BackendCommandConcurrentSubmission) == 0)
        GTEST_SKIP() << GetParam().name << " is bound to the thread that owns its context";
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCompletion previous{VERNON_RHI_INVALID_HANDLE_INDEX, 0};

    for (unsigned iteration = 0; iteration != 32; ++iteration) {
        VernonRhiCommandEncoder encoder{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
        ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &encoder), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);

        VernonRhiCompletion guessed{};
        if (previous.index != VERNON_RHI_INVALID_HANDLE_INDEX) {
            guessed.index = previous.index;
            guessed.generation = previous.generation + 1;
            if (!guessed.generation)
                guessed.generation = 1;
        }
        VernonRhiCompletion submitted{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
        VernonRhiStatus submitStatus = VERNON_RHI_STATUS_INTERNAL_ERROR;
        std::atomic<bool> done{};
        std::thread submitter([&] {
            submitStatus = vernonRhiDeviceSubmit(device, encoder, &submitted);
            done.store(true, std::memory_order_release);
        });
        if (previous.index != VERNON_RHI_INVALID_HANDLE_INDEX) {
            while (!done.load(std::memory_order_acquire)) {
                VernonRhiCompletionState state{};
                const VernonRhiStatus status = vernonRhiCompletionGetState(device, guessed, &state);
                EXPECT_TRUE(status == VERNON_RHI_STATUS_INVALID_ARGUMENT || status == VERNON_RHI_STATUS_OK);
            }
        }
        submitter.join();
        ASSERT_EQ(submitStatus, VERNON_RHI_STATUS_OK);
        if (previous.index != VERNON_RHI_INVALID_HANDLE_INDEX) {
            EXPECT_EQ(submitted.index, guessed.index);
            EXPECT_EQ(submitted.generation, guessed.generation);
        }
        ASSERT_EQ(vernonRhiCompletionWait(device, submitted), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceDestroyCompletion(device, submitted), VERNON_RHI_STATUS_OK);
        previous = submitted;
    }
}

TEST_P(RhiResourceLifetime, SubmitMovesRetainedLeaseAndCompletionKeepsOnlyDeviceChild) {
    const VernonRhiDevice device = this->device();
    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = 64;
    bufferDescriptor.usage = VERNON_RHI_BUFFER_STORAGE;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer retained{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &retained), VERNON_RHI_STATUS_OK);

    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder encoder{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
    VernonRhiBarrier barrier{};
    barrier.struct_size = sizeof(barrier);
    barrier.destination_stage_mask = VERNON_RHI_STAGE_COMPUTE;
    barrier.destination_access = VERNON_RHI_ACCESS_SHADER_READ;
    barrier.old_state = VERNON_RHI_STATE_COMMON;
    barrier.new_state = VERNON_RHI_STATE_SHADER_READ;
    barrier.buffer = retained;
    ASSERT_EQ(vernonRhiCommandEncoderBarrier(device, encoder, &barrier, 1), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(device, retained), VERNON_RHI_STATUS_OK);
    auto encoderKeyResult = vernon::rhi::commandEncoderKey(device, encoder);
    ASSERT_TRUE(encoderKeyResult.isOk());
    const uint64_t encoderKey = encoderKeyResult.value();
    EXPECT_TRUE(vernon::rhi::bufferResource(device, retained).isErr());
    const uint64_t retainedKey =
        (static_cast<uint64_t>(retained.generation) << 32) | (static_cast<uint64_t>(retained.index) + 1);
    EXPECT_TRUE(
        vernon::rhi::resolveCommandResource(device, encoderKey, vernon::rhi::ResourceKind::Image, retainedKey).isErr());
    EXPECT_TRUE(
        vernon::rhi::resolveCommandResource(device, encoderKey, vernon::rhi::ResourceKind::Buffer, retainedKey).isOk());

    VernonRhiBuffer whileRetained{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &whileRetained), VERNON_RHI_STATUS_OK);
    EXPECT_NE(whileRetained.index, retained.index);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    VernonRhiCompletion completion{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceSubmit(device, encoder, &completion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCompletionWait(device, completion), VERNON_RHI_STATUS_OK);

    VernonRhiBuffer recycled{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &recycled), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(recycled.index, retained.index);
    EXPECT_NE(recycled.generation, retained.generation);
    EXPECT_EQ(vernonRhiDestroyDevice(device), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_TRUE(vernon::rhi::deviceExists(device));
    EXPECT_EQ(vernonRhiDeviceDestroyCompletion(device, completion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, whileRetained), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, recycled), VERNON_RHI_STATUS_OK);
}

TEST_P(RhiResourceLifetime, FailedMultiResourceRetainRollsBackEarlierCommandLease) {
    const VernonRhiDevice device = this->device();
    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = 64;
    bufferDescriptor.usage = VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer source{};
    VernonRhiBuffer destination{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &source), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &destination), VERNON_RHI_STATUS_OK);

    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_TRANSFER;
    VernonRhiCommandEncoder encoder{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
    const VernonRhiBuffer staleDestination{destination.index, destination.generation + 1};
    EXPECT_NE(vernonRhiCommandEncoderCopyBuffer(device, encoder, source, 0, staleDestination, 0, 16),
              VERNON_RHI_STATUS_OK);

    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(device, source), VERNON_RHI_STATUS_OK);
    VernonRhiBuffer recycled{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &recycled), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(recycled.index, source.index);
    EXPECT_NE(recycled.generation, source.generation);

    EXPECT_EQ(vernonRhiDeviceDestroyCommandEncoder(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, destination), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, recycled), VERNON_RHI_STATUS_OK);
}

TEST_P(RhiResourceLifetime, ImageAndSamplerSupportMatchesGraphicsCapability) {
    const VernonRhiDevice device = this->device();
    const bool supportsGraphics = vernonRuntimeGetContextCapabilities(runtime()).supports_graphics;

    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageDescriptor.width = 1;
    imageDescriptor.height = 1;
    imageDescriptor.depth = 1;
    imageDescriptor.mip_levels = 1;
    imageDescriptor.array_layers = 1;
    imageDescriptor.sample_count = 1;
    imageDescriptor.usage = VERNON_RHI_IMAGE_SAMPLED;
    VernonRhiImage image{};
    const VernonRhiStatus imageStatus = vernonRhiDeviceCreateImage(device, &imageDescriptor, &image);
    if (supportsGraphics) {
        ASSERT_EQ(imageStatus, VERNON_RHI_STATUS_OK);
        EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
    } else
        EXPECT_EQ(imageStatus, VERNON_RHI_STATUS_UNSUPPORTED);

    VernonRhiSamplerDescriptor samplerDescriptor{};
    samplerDescriptor.struct_size = sizeof(samplerDescriptor);
    VernonRhiSampler sampler{};
    const VernonRhiStatus samplerStatus = vernonRhiDeviceCreateSampler(device, &samplerDescriptor, &sampler);
    if (supportsGraphics) {
        ASSERT_EQ(samplerStatus, VERNON_RHI_STATUS_OK);
        EXPECT_EQ(vernonRhiDeviceDestroySampler(device, sampler), VERNON_RHI_STATUS_OK);
    } else
        EXPECT_EQ(samplerStatus, VERNON_RHI_STATUS_UNSUPPORTED);
}

TEST_P(RhiResourceLifetime, RetainedBufferDelaysSlotReuse) {
    const vernon::tests::BackendTestRow test = GetParam();
    const VernonRhiDevice device = this->device();

    VernonRuntimeRhiAdapter *adapter = vernonRuntimeRhiAdapterCreateForDevice(device, *test.rhi);
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
    VernonRuntimeProviderResourceReference reference{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceBuffer(adapter, first, 0, 64, &reference), VERNON_STATUS_OK);
    ASSERT_EQ(provider->retain_resource(provider->user_data, reference), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(device, first), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceIsBufferValid(device, first), 0u);

    VernonRhiBuffer replacement{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &replacement), VERNON_RHI_STATUS_OK);
    EXPECT_NE(replacement.index, first.index);
    provider->release_resource(provider->user_data, reference);

    VernonRhiBuffer recycled{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &recycled), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(recycled.index, first.index);
    EXPECT_NE(recycled.generation, first.generation);

    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, replacement), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, recycled), VERNON_RHI_STATUS_OK);
    vernonRuntimeRhiAdapterDestroy(adapter);
}

TEST_P(RhiResourceLifetime, ResolveRequiresExactRetainedLeaseAfterPublicDestroy) {
    const VernonRhiDevice device = this->device();
    VernonRhiBufferDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.size = 64;
    descriptor.usage = VERNON_RHI_BUFFER_STORAGE;
    descriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer buffer{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &descriptor, &buffer), VERNON_RHI_STATUS_OK);
    auto keyResult = vernon::rhi::bufferResource(device, buffer);
    ASSERT_TRUE(keyResult.isOk());
    const uint64_t key = keyResult.value();
    auto retained = vernon::rhi::retainResource(device, vernon::rhi::ResourceKind::Buffer, key);
    ASSERT_TRUE(retained.isOk());
    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(device, buffer), VERNON_RHI_STATUS_OK);

    auto wrongKey = vernon::rhi::resolveResource(device, vernon::rhi::ResourceKind::Buffer, key + 1, retained.value());
    EXPECT_TRUE(wrongKey.isErr());
    auto wrongKind = vernon::rhi::resolveResource(device, vernon::rhi::ResourceKind::Image, key, retained.value());
    EXPECT_TRUE(wrongKind.isErr());
    auto resolved = vernon::rhi::resolveResource(device, vernon::rhi::ResourceKind::Buffer, key, retained.value());
    EXPECT_TRUE(resolved.isOk());
    EXPECT_TRUE(retained.value().release().isOk());
}

TEST_P(RhiResourceLifetime, RetainedImageAndSamplerDelaySlotReuse) {
    const vernon::tests::BackendTestRow test = GetParam();
    const VernonRhiDevice device = this->device();
    const vernon::tests::BackendProbeResult probe = graphicsProbe();
    if (!probe.available()) {
        if (probe.skippable())
            GTEST_SKIP() << probe.reason;
        FAIL() << probe.reason;
        return;
    }

    VernonRuntimeRhiAdapter *adapter = vernonRuntimeRhiAdapterCreateForDevice(device, *test.rhi);
    ASSERT_NE(adapter, nullptr);
    const VernonRuntimeDeviceProvider *provider = vernonRuntimeRhiAdapterGetProvider(adapter);
    ASSERT_NE(provider, nullptr);

    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageDescriptor.width = 1;
    imageDescriptor.height = 1;
    imageDescriptor.depth = 1;
    imageDescriptor.mip_levels = 1;
    imageDescriptor.array_layers = 1;
    imageDescriptor.sample_count = 1;
    imageDescriptor.usage = VERNON_RHI_IMAGE_SAMPLED;
    VernonRhiImage firstImage{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &firstImage), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference imageReference{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceImage(adapter, firstImage, &imageReference), VERNON_STATUS_OK);
    ASSERT_EQ(provider->retain_resource(provider->user_data, imageReference), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(device, firstImage), VERNON_RHI_STATUS_OK);
    VernonRhiImage replacementImage{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &replacementImage), VERNON_RHI_STATUS_OK);
    EXPECT_NE(replacementImage.index, firstImage.index);
    provider->release_resource(provider->user_data, imageReference);
    VernonRhiImage recycledImage{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &recycledImage), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(recycledImage.index, firstImage.index);
    EXPECT_NE(recycledImage.generation, firstImage.generation);

    VernonRhiSamplerDescriptor samplerDescriptor{};
    samplerDescriptor.struct_size = sizeof(samplerDescriptor);
    VernonRhiSampler firstSampler{};
    ASSERT_EQ(vernonRhiDeviceCreateSampler(device, &samplerDescriptor, &firstSampler), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference samplerReference{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceSampler(adapter, firstSampler, &samplerReference), VERNON_STATUS_OK);
    ASSERT_EQ(provider->retain_resource(provider->user_data, samplerReference), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroySampler(device, firstSampler), VERNON_RHI_STATUS_OK);
    VernonRhiSampler replacementSampler{};
    ASSERT_EQ(vernonRhiDeviceCreateSampler(device, &samplerDescriptor, &replacementSampler), VERNON_RHI_STATUS_OK);
    EXPECT_NE(replacementSampler.index, firstSampler.index);
    provider->release_resource(provider->user_data, samplerReference);
    VernonRhiSampler recycledSampler{};
    ASSERT_EQ(vernonRhiDeviceCreateSampler(device, &samplerDescriptor, &recycledSampler), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(recycledSampler.index, firstSampler.index);
    EXPECT_NE(recycledSampler.generation, firstSampler.generation);

    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, replacementImage), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, recycledImage), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroySampler(device, replacementSampler), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroySampler(device, recycledSampler), VERNON_RHI_STATUS_OK);
    vernonRuntimeRhiAdapterDestroy(adapter);
}

TEST_P(RhiResourceLifetime, RetainedImageViewKeepsParentDescriptorAlive) {
    const vernon::tests::BackendTestRow test = GetParam();
    const VernonRhiDevice device = this->device();
    const vernon::tests::BackendProbeResult probe = imageSubresourceViewProbe();
    if (!probe.available()) {
        if (probe.skippable())
            GTEST_SKIP() << probe.reason;
        FAIL() << probe.reason;
        return;
    }
    VernonRuntimeRhiAdapter *adapter = vernonRuntimeRhiAdapterCreateForDevice(device, *test.rhi);
    ASSERT_NE(adapter, nullptr);
    const VernonRuntimeDeviceProvider *provider = vernonRuntimeRhiAdapterGetProvider(adapter);

    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageDescriptor.width = imageDescriptor.height = 4;
    imageDescriptor.depth = 1;
    imageDescriptor.mip_levels = 2;
    imageDescriptor.array_layers = 1;
    imageDescriptor.sample_count = 1;
    imageDescriptor.usage = VERNON_RHI_IMAGE_SAMPLED;
    VernonRhiImage image{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &image), VERNON_RHI_STATUS_OK);
    VernonRhiImageViewDescriptor viewDescriptor{};
    viewDescriptor.struct_size = sizeof(viewDescriptor);
    viewDescriptor.image = image;
    viewDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    viewDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    viewDescriptor.base_mip_level = 1;
    viewDescriptor.mip_level_count = 1;
    viewDescriptor.array_layer_count = 1;
    viewDescriptor.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
    VernonRhiImageView view{};
    ASSERT_EQ(vernonRhiDeviceCreateImageView(device, &viewDescriptor, &view), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference reference{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceImageView(adapter, view, &reference), VERNON_STATUS_OK);
    ASSERT_EQ(provider->retain_resource(provider->user_data, reference), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImageView(device, view), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);

    VernonRuntimeProviderImageDescription description{};
    description.struct_size = sizeof(description);
    ASSERT_EQ(provider->describe_image(provider->user_data, reference, &description), VERNON_STATUS_OK);
    EXPECT_NE(description.parent_identity, 0u);
    EXPECT_EQ(description.view.subresources.base_mip_level, 1u);
    EXPECT_EQ(description.view.subresources.mip_level_count, 1u);

    VernonRhiImage replacement{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &replacement), VERNON_RHI_STATUS_OK);
    EXPECT_NE(replacement.index, image.index);
    provider->release_resource(provider->user_data, reference);
    VernonRhiImage recycled{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &recycled), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(recycled.index, image.index);
    EXPECT_NE(recycled.generation, image.generation);

    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, replacement), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, recycled), VERNON_RHI_STATUS_OK);
    vernonRuntimeRhiAdapterDestroy(adapter);
}

TEST_P(RhiResourceLifetime, RuntimeContextAndAdapterRetainTheirRhiDevice) {
    const auto backend = GetParam();
    const VernonRhiDevice retainedDevice = device();
    VernonRuntimeRhiAdapter *adapter = vernonRuntimeRhiAdapterCreateForDevice(retainedDevice, *backend.rhi);
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(vernonRhiDestroyDevice(retainedDevice), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_TRUE(vernon::rhi::deviceExists(retainedDevice));
    vernonRuntimeRhiAdapterDestroy(adapter);

    VernonRuntimeContext *context = vernonRuntimeCreateForRhiDevice(backend.runtime, retainedDevice, nullptr);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(vernonRhiDestroyDevice(retainedDevice), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_TRUE(vernon::rhi::deviceExists(retainedDevice));
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
    EXPECT_TRUE(vernon::rhi::deviceExists(retainedDevice));
}

TEST_P(RhiResourceLifetime, CommandGraphRetainsImportedImageViewAndParent) {
    const VernonRhiDevice device = this->device();
    const vernon::tests::BackendProbeResult probe = graphicsProbe();
    if (!probe.available()) {
        if (probe.skippable())
            GTEST_SKIP() << probe.reason;
        FAIL() << probe.reason;
        return;
    }

    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageDescriptor.width = imageDescriptor.height = 4;
    imageDescriptor.depth = 1;
    imageDescriptor.mip_levels = 1;
    imageDescriptor.array_layers = 1;
    imageDescriptor.sample_count = 1;
    imageDescriptor.usage = VERNON_RHI_IMAGE_SAMPLED;
    VernonRhiImage image{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &image), VERNON_RHI_STATUS_OK);

    VernonRhiImageViewDescriptor viewDescriptor{};
    viewDescriptor.struct_size = sizeof(viewDescriptor);
    viewDescriptor.image = image;
    viewDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    viewDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    viewDescriptor.mip_level_count = 1;
    viewDescriptor.array_layer_count = 1;
    viewDescriptor.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
    VernonRhiImageView view{};
    ASSERT_EQ(vernonRhiDeviceCreateImageView(device, &viewDescriptor, &view), VERNON_RHI_STATUS_OK);

    VernonRhiImage replacement{};
    {
        vernon::execution::CommandGraph graph(device);
        const vernon::execution::GraphImage imported = graph.importImage(image, view);
        ASSERT_NE(imported.id, UINT32_MAX);
        ASSERT_EQ(vernonRhiDeviceDestroyImageView(device, view), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &replacement), VERNON_RHI_STATUS_OK);
        EXPECT_NE(replacement.index, image.index);
    }

    VernonRhiImage recycled{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &recycled), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(recycled.index, image.index);
    EXPECT_NE(recycled.generation, image.generation);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, replacement), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, recycled), VERNON_RHI_STATUS_OK);
}

TEST_P(RhiResourceLifetime, FormatCapabilitiesNarrowTheTableBaseline) {
    const VernonRhiDevice device = this->device();
    const bool images = GetParam().runtime != VERNON_RUNTIME_CUDA;
    for (const vernon::formats::Format &format : vernon::formats::table) {
        SCOPED_TRACE(std::string(format.name));
        uint32_t capabilities = ~0u;
        ASSERT_EQ(vernonRhiDeviceGetFormatCapabilities(device, format.rhi, &capabilities), VERNON_RHI_STATUS_OK)
            << vernon::test::text(vernonRhiDeviceGetLastError(device));
        EXPECT_EQ(capabilities & ~format.baseline, 0u);
        if (!images)
            EXPECT_EQ(capabilities, 0u);
    }
    if (images) {
        uint32_t rgba8 = 0;
        ASSERT_EQ(vernonRhiDeviceGetFormatCapabilities(device, VERNON_RHI_FORMAT_RGBA8_UNORM, &rgba8),
                  VERNON_RHI_STATUS_OK);
        constexpr uint32_t color = VERNON_RHI_FORMAT_RENDERABLE | VERNON_RHI_FORMAT_BLENDABLE |
                                   VERNON_RHI_FORMAT_FILTERABLE | VERNON_RHI_FORMAT_COPYABLE;
        EXPECT_EQ(rgba8 & color, color);
        for (const VernonRhiFormat format : {VERNON_RHI_FORMAT_R32_UINT, VERNON_RHI_FORMAT_RG32_UINT}) {
            uint32_t integer = 0;
            ASSERT_EQ(vernonRhiDeviceGetFormatCapabilities(device, format, &integer), VERNON_RHI_STATUS_OK);
            EXPECT_EQ(integer & (VERNON_RHI_FORMAT_RENDERABLE | VERNON_RHI_FORMAT_COPYABLE),
                      VERNON_RHI_FORMAT_RENDERABLE | VERNON_RHI_FORMAT_COPYABLE);
        }
    }
    uint32_t capabilities = 0;
    EXPECT_EQ(vernonRhiDeviceGetFormatCapabilities(device, VERNON_RHI_FORMAT_R32_UINT, nullptr),
              VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiDeviceGetFormatCapabilities(device, VERNON_RHI_FORMAT_UNDEFINED, &capabilities),
              VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiDeviceGetFormatCapabilities(device, static_cast<VernonRhiFormat>(0x7fff), &capabilities),
              VERNON_RHI_STATUS_INVALID_ARGUMENT);
}

// Values above 2^24 and with the top bit set survive only a transfer that never passes through float.
TEST_P(RhiResourceLifetime, IntegerImagesRoundTripEveryBit) {
    const vernon::tests::BackendProbeResult probe = graphicsProbe();
    if (!probe.available()) {
        if (probe.skippable())
            GTEST_SKIP() << probe.reason;
        FAIL() << probe.reason;
    }
    const VernonRhiDevice device = this->device();
    constexpr uint32_t extent = 4;
    for (const auto [format, dataFormat, components] :
         {std::tuple{VERNON_RHI_FORMAT_R32_UINT, VERNON_RHI_IMAGE_DATA_RED, 1u},
          std::tuple{VERNON_RHI_FORMAT_RG32_UINT, VERNON_RHI_IMAGE_DATA_RG, 2u}}) {
        SCOPED_TRACE(std::string(vernon::formats::find(format)->name));
        std::vector<uint32_t> texels(extent * extent * components);
        for (size_t index = 0; index < texels.size(); ++index)
            texels[index] = index == 0 ? 0xFFFFFFFFu : (1u << 24) + 1u + static_cast<uint32_t>(index) * 0x01010101u;
        VernonRhiImageDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.dimension = VERNON_RHI_IMAGE_2D;
        descriptor.width = extent;
        descriptor.height = extent;
        descriptor.depth = 1;
        descriptor.format = format;
        descriptor.mip_levels = 1;
        descriptor.array_layers = 1;
        descriptor.sample_count = 1;
        descriptor.usage = VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_TRANSFER_SOURCE |
                           VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
        std::array<VernonRhiImage, 2> images{};
        for (VernonRhiImage &image : images)
            ASSERT_EQ(vernonRhiDeviceCreateImage(device, &descriptor, &image), VERNON_RHI_STATUS_OK)
                << vernon::test::text(vernonRhiDeviceGetLastError(device));
        VernonRhiImageUploadDescriptor upload{};
        upload.struct_size = sizeof(upload);
        upload.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
        upload.width = extent;
        upload.height = extent;
        upload.depth = 1;
        upload.source_format = dataFormat;
        upload.source_type = VERNON_RHI_IMAGE_DATA_UINT32;
        upload.data = texels.data();
        ASSERT_EQ(vernonRhiDeviceUploadImage(device, images[0], &upload, 1), VERNON_RHI_STATUS_OK)
            << vernon::test::text(vernonRhiDeviceGetLastError(device));

        VernonRhiCommandEncoderDescriptor encoderDescriptor{};
        encoderDescriptor.struct_size = sizeof(encoderDescriptor);
        encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_TRANSFER;
        VernonRhiCommandEncoder encoder{};
        ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
        VernonRhiImageCopyRegion region{};
        region.struct_size = sizeof(region);
        region.width = extent;
        region.height = extent;
        region.depth = 1;
        region.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
        ASSERT_EQ(vernonRhiCommandEncoderCopyImage(device, encoder, images[0], images[1], &region, 1),
                  VERNON_RHI_STATUS_OK)
            << vernon::test::text(vernonRhiDeviceGetLastError(device));
        ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
        VernonRhiCompletion completion{};
        ASSERT_EQ(vernonRhiDeviceSubmit(device, encoder, &completion), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiCompletionWait(device, completion), VERNON_RHI_STATUS_OK)
            << vernon::test::text(vernonRhiDeviceGetLastError(device));
        EXPECT_EQ(vernonRhiDeviceDestroyCompletion(device, completion), VERNON_RHI_STATUS_OK);

        VernonRhiImageDownloadDescriptor download{};
        download.struct_size = sizeof(download);
        download.width = extent;
        download.height = extent;
        download.depth = 1;
        download.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
        download.destination_format = dataFormat;
        download.destination_type = VERNON_RHI_IMAGE_DATA_UINT32;
        for (const VernonRhiImage image : images) {
            std::vector<uint32_t> read(texels.size());
            ASSERT_EQ(vernonRhiDeviceDownloadImage(device, image, &download, read.data(), read.size() * 4),
                      VERNON_RHI_STATUS_OK)
                << vernon::test::text(vernonRhiDeviceGetLastError(device));
            EXPECT_EQ(read, texels);
        }
        for (const VernonRhiImage image : images)
            EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
    }
}

// Both the load-operation clear and the in-scope clear write integer values that a float clear cannot represent.
TEST_P(RhiResourceLifetime, IntegerColorClearsAreExact) {
    const vernon::tests::BackendProbeResult probe = graphicsProbe();
    if (!probe.available()) {
        if (probe.skippable())
            GTEST_SKIP() << probe.reason;
        FAIL() << probe.reason;
    }
    const VernonRhiDevice device = this->device();
    constexpr uint32_t extent = 4;
    for (const auto [format, dataFormat, components] :
         {std::tuple{VERNON_RHI_FORMAT_R32_UINT, VERNON_RHI_IMAGE_DATA_RED, 1u},
          std::tuple{VERNON_RHI_FORMAT_RG32_UINT, VERNON_RHI_IMAGE_DATA_RG, 2u}}) {
        SCOPED_TRACE(std::string(vernon::formats::find(format)->name));
        VernonRhiImageDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.dimension = VERNON_RHI_IMAGE_2D;
        descriptor.width = extent;
        descriptor.height = extent;
        descriptor.depth = 1;
        descriptor.format = format;
        descriptor.mip_levels = 1;
        descriptor.array_layers = 1;
        descriptor.sample_count = 1;
        descriptor.usage = VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_TRANSFER_SOURCE;
        VernonRhiImage image{};
        ASSERT_EQ(vernonRhiDeviceCreateImage(device, &descriptor, &image), VERNON_RHI_STATUS_OK)
            << vernon::test::text(vernonRhiDeviceGetLastError(device));
        VernonRhiImageViewDescriptor viewDescriptor{};
        viewDescriptor.struct_size = sizeof(viewDescriptor);
        viewDescriptor.image = image;
        viewDescriptor.dimension = VERNON_RHI_IMAGE_2D;
        viewDescriptor.format = format;
        viewDescriptor.mip_level_count = 1;
        viewDescriptor.array_layer_count = 1;
        viewDescriptor.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
        VernonRhiImageView view{};
        ASSERT_EQ(vernonRhiDeviceCreateImageView(device, &viewDescriptor, &view), VERNON_RHI_STATUS_OK);

        const auto render = [&](VernonRhiLoadOperation load, const VernonClearColor &loadClear,
                                const VernonClearColor *scopeClear) {
            VernonRhiCommandEncoderDescriptor encoderDescriptor{};
            encoderDescriptor.struct_size = sizeof(encoderDescriptor);
            encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_GRAPHICS;
            VernonRhiCommandEncoder encoder{};
            ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
            VernonRhiColorAttachment color{};
            color.view = view;
            color.initial_state = VERNON_RHI_STATE_COLOR_ATTACHMENT;
            color.final_state = VERNON_RHI_STATE_COLOR_ATTACHMENT;
            color.load_operation = load;
            color.store_operation = VERNON_RHI_STORE_PRESERVE;
            color.clear_color = loadClear;
            VernonRhiRenderingDescriptor rendering{};
            rendering.struct_size = sizeof(rendering);
            rendering.color_attachments = &color;
            rendering.color_attachment_count = 1;
            rendering.width = extent;
            rendering.height = extent;
            rendering.layers = 1;
            ASSERT_EQ(vernonRhiCommandEncoderBeginRendering(device, encoder, &rendering), VERNON_RHI_STATUS_OK)
                << vernon::test::text(vernonRhiDeviceGetLastError(device));
            if (scopeClear)
                ASSERT_EQ(vernonRhiCommandEncoderClearColorAttachment(device, encoder, 0, scopeClear),
                          VERNON_RHI_STATUS_OK)
                    << vernon::test::text(vernonRhiDeviceGetLastError(device));
            ASSERT_EQ(vernonRhiCommandEncoderEndRendering(device, encoder), VERNON_RHI_STATUS_OK);
            ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
            VernonRhiCompletion completion{};
            ASSERT_EQ(vernonRhiDeviceSubmit(device, encoder, &completion), VERNON_RHI_STATUS_OK);
            ASSERT_EQ(vernonRhiCompletionWait(device, completion), VERNON_RHI_STATUS_OK)
                << vernon::test::text(vernonRhiDeviceGetLastError(device));
            EXPECT_EQ(vernonRhiDeviceDestroyCompletion(device, completion), VERNON_RHI_STATUS_OK);
        };
        const auto expectTexels = [&](const VernonClearColor &clear) {
            VernonRhiImageDownloadDescriptor download{};
            download.struct_size = sizeof(download);
            download.width = extent;
            download.height = extent;
            download.depth = 1;
            download.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
            download.destination_format = dataFormat;
            download.destination_type = VERNON_RHI_IMAGE_DATA_UINT32;
            std::vector<uint32_t> read(extent * extent * components);
            ASSERT_EQ(vernonRhiDeviceDownloadImage(device, image, &download, read.data(), read.size() * 4),
                      VERNON_RHI_STATUS_OK)
                << vernon::test::text(vernonRhiDeviceGetLastError(device));
            for (size_t index = 0; index < read.size(); ++index)
                EXPECT_EQ(read[index], clear.u32[index % components]) << "component " << index;
        };

        VernonClearColor loadClear{};
        loadClear.u32[0] = 0xFFFFFFFFu;
        loadClear.u32[1] = (1u << 24) + 1u;
        render(VERNON_RHI_LOAD_CLEAR, loadClear, nullptr);
        expectTexels(loadClear);

        VernonClearColor scopeClear{};
        scopeClear.u32[0] = 0x80000001u;
        scopeClear.u32[1] = 0xFFFFFFFEu;
        render(VERNON_RHI_LOAD_PRESERVE, VernonClearColor{}, &scopeClear);
        expectTexels(scopeClear);

        EXPECT_EQ(vernonRhiDeviceDestroyImageView(device, view), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
    }
}

TEST_P(RhiResourceLifetime, CreateImageRejectsDescriptorsOfEveryBackendAlike) {
    const vernon::tests::BackendProbeResult probe = graphicsProbe();
    if (!probe.available()) {
        if (probe.skippable())
            GTEST_SKIP() << probe.reason;
        FAIL() << probe.reason;
    }
    const VernonRhiDevice device = this->device();
    VernonRhiImageDescriptor valid{};
    valid.struct_size = sizeof(valid);
    valid.dimension = VERNON_RHI_IMAGE_2D;
    valid.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    valid.width = 2;
    valid.height = 2;
    valid.depth = 1;
    valid.mip_levels = 2;
    valid.array_layers = 1;
    valid.sample_count = 1;
    valid.usage = VERNON_RHI_IMAGE_SAMPLED;
    VernonRhiImage image{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &valid, &image), VERNON_RHI_STATUS_OK)
        << vernon::test::text(vernonRhiDeviceGetLastError(device));
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);

    const std::array<std::pair<const char *, void (*)(VernonRhiImageDescriptor &)>, 8> invalid{{
        {"struct size", [](VernonRhiImageDescriptor &image) { image.struct_size = 0; }},
        {"shape", [](VernonRhiImageDescriptor &image) { image.depth = 2; }},
        {"no mips", [](VernonRhiImageDescriptor &image) { image.mip_levels = 0; }},
        {"longer than the mip chain", [](VernonRhiImageDescriptor &image) { image.mip_levels = 3; }},
        {"multisampled", [](VernonRhiImageDescriptor &image) { image.sample_count = 4; }},
        {"unknown usage", [](VernonRhiImageDescriptor &image) { image.usage |= uint32_t{1} << 31; }},
        {"color format as depth attachment",
         [](VernonRhiImageDescriptor &image) { image.usage |= VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT; }},
        {"depth format as color attachment",
         [](VernonRhiImageDescriptor &image) {
             image.format = VERNON_RHI_FORMAT_D32_FLOAT;
             image.mip_levels = 1;
             image.usage = VERNON_RHI_IMAGE_COLOR_ATTACHMENT;
         }},
    }};
    for (const auto &[name, mutate] : invalid) {
        SCOPED_TRACE(name);
        VernonRhiImageDescriptor descriptor = valid;
        mutate(descriptor);
        EXPECT_EQ(vernonRhiDeviceCreateImage(device, &descriptor, &image), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    }
}

// A scope that renders fewer layers than its attachment view has clears only those layers, both through its load
// operation and through an in-scope clear.
TEST_P(RhiResourceLifetime, RenderingClearsOnlyTheLayersOfItsScope) {
    const vernon::tests::BackendProbeResult probe = graphicsProbe();
    if (!probe.available()) {
        if (probe.skippable())
            GTEST_SKIP() << probe.reason;
        FAIL() << probe.reason;
    }
    const VernonRhiDevice device = this->device();
    constexpr uint32_t extent = 4;
    struct Case {
        VernonRhiFormat format;
        VernonRhiImageDataFormat dataFormat;
        VernonRhiImageDataType dataType;
        std::vector<uint8_t> seeded;
        std::array<VernonClearColor, 2> clears;
        std::array<std::vector<uint8_t>, 2> cleared;
    };
    const auto unorm = [](size_t channel) {
        VernonClearColor clear{};
        clear.f32[channel] = 1.0f;
        clear.f32[3] = 1.0f;
        return clear;
    };
    const auto integer = [](uint32_t value) {
        VernonClearColor clear{};
        clear.u32[0] = value;
        return clear;
    };
    const std::array<Case, 2> cases{Case{VERNON_RHI_FORMAT_RGBA8_UNORM,
                                         VERNON_RHI_IMAGE_DATA_RGBA,
                                         VERNON_RHI_IMAGE_DATA_UINT8,
                                         {255, 0, 0, 255},
                                         {unorm(1), unorm(2)},
                                         {std::vector<uint8_t>{0, 255, 0, 255}, std::vector<uint8_t>{0, 0, 255, 255}}},
                                    Case{VERNON_RHI_FORMAT_R32_UINT,
                                         VERNON_RHI_IMAGE_DATA_RED,
                                         VERNON_RHI_IMAGE_DATA_UINT32,
                                         std::vector<uint8_t>(4, 0xA1),
                                         {integer(0xB2B2B2B2u), integer(0xC3C3C3C3u)},
                                         {std::vector<uint8_t>(4, 0xB2), std::vector<uint8_t>(4, 0xC3)}}};
    for (const Case &tested : cases) {
        SCOPED_TRACE(std::string(vernon::formats::find(tested.format)->name));
        VernonRhiImageDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.dimension = VERNON_RHI_IMAGE_2D_ARRAY;
        descriptor.width = extent;
        descriptor.height = extent;
        descriptor.depth = 1;
        descriptor.format = tested.format;
        descriptor.mip_levels = 1;
        descriptor.array_layers = 2;
        descriptor.sample_count = 1;
        descriptor.usage = VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_TRANSFER_SOURCE |
                           VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
        VernonRhiImage image{};
        ASSERT_EQ(vernonRhiDeviceCreateImage(device, &descriptor, &image), VERNON_RHI_STATUS_OK)
            << vernon::test::text(vernonRhiDeviceGetLastError(device));
        VernonRhiImageViewDescriptor viewDescriptor{};
        viewDescriptor.struct_size = sizeof(viewDescriptor);
        viewDescriptor.image = image;
        viewDescriptor.dimension = VERNON_RHI_IMAGE_2D_ARRAY;
        viewDescriptor.format = tested.format;
        viewDescriptor.mip_level_count = 1;
        viewDescriptor.array_layer_count = 2;
        viewDescriptor.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
        VernonRhiImageView view{};
        ASSERT_EQ(vernonRhiDeviceCreateImageView(device, &viewDescriptor, &view), VERNON_RHI_STATUS_OK)
            << vernon::test::text(vernonRhiDeviceGetLastError(device));

        std::vector<uint8_t> seed;
        for (uint32_t texel = 0; texel < extent * extent; ++texel)
            seed.insert(seed.end(), tested.seeded.begin(), tested.seeded.end());
        VernonRhiImageUploadDescriptor upload{};
        upload.struct_size = sizeof(upload);
        upload.array_layer = 1;
        upload.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
        upload.width = extent;
        upload.height = extent;
        upload.depth = 1;
        upload.source_format = tested.dataFormat;
        upload.source_type = tested.dataType;
        upload.data = seed.data();
        ASSERT_EQ(vernonRhiDeviceUploadImage(device, image, &upload, 1), VERNON_RHI_STATUS_OK)
            << vernon::test::text(vernonRhiDeviceGetLastError(device));

        const auto render = [&](VernonRhiLoadOperation load, const VernonClearColor &loadClear,
                                const VernonClearColor *scopeClear) {
            VernonRhiCommandEncoderDescriptor encoderDescriptor{};
            encoderDescriptor.struct_size = sizeof(encoderDescriptor);
            encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_GRAPHICS;
            VernonRhiCommandEncoder encoder{};
            ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
            VernonRhiColorAttachment color{};
            color.view = view;
            color.initial_state = VERNON_RHI_STATE_COLOR_ATTACHMENT;
            color.final_state = VERNON_RHI_STATE_COLOR_ATTACHMENT;
            color.load_operation = load;
            color.store_operation = VERNON_RHI_STORE_PRESERVE;
            color.clear_color = loadClear;
            VernonRhiRenderingDescriptor rendering{};
            rendering.struct_size = sizeof(rendering);
            rendering.color_attachments = &color;
            rendering.color_attachment_count = 1;
            rendering.width = extent;
            rendering.height = extent;
            rendering.layers = 1;
            ASSERT_EQ(vernonRhiCommandEncoderBeginRendering(device, encoder, &rendering), VERNON_RHI_STATUS_OK)
                << vernon::test::text(vernonRhiDeviceGetLastError(device));
            if (scopeClear)
                ASSERT_EQ(vernonRhiCommandEncoderClearColorAttachment(device, encoder, 0, scopeClear),
                          VERNON_RHI_STATUS_OK)
                    << vernon::test::text(vernonRhiDeviceGetLastError(device));
            ASSERT_EQ(vernonRhiCommandEncoderEndRendering(device, encoder), VERNON_RHI_STATUS_OK);
            ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
            VernonRhiCompletion completion{};
            ASSERT_EQ(vernonRhiDeviceSubmit(device, encoder, &completion), VERNON_RHI_STATUS_OK);
            ASSERT_EQ(vernonRhiCompletionWait(device, completion), VERNON_RHI_STATUS_OK)
                << vernon::test::text(vernonRhiDeviceGetLastError(device));
            EXPECT_EQ(vernonRhiDeviceDestroyCompletion(device, completion), VERNON_RHI_STATUS_OK);
        };
        const auto expectLayer = [&](uint32_t layer, const std::vector<uint8_t> &texel) {
            VernonRhiImageDownloadDescriptor download{};
            download.struct_size = sizeof(download);
            download.array_layer = layer;
            download.width = extent;
            download.height = extent;
            download.depth = 1;
            download.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
            download.destination_format = tested.dataFormat;
            download.destination_type = tested.dataType;
            std::vector<uint8_t> read(extent * extent * texel.size());
            ASSERT_EQ(vernonRhiDeviceDownloadImage(device, image, &download, read.data(), read.size()),
                      VERNON_RHI_STATUS_OK)
                << vernon::test::text(vernonRhiDeviceGetLastError(device));
            for (size_t index = 0; index < read.size(); ++index)
                EXPECT_EQ(read[index], texel[index % texel.size()]) << "layer " << layer << " byte " << index;
        };

        render(VERNON_RHI_LOAD_CLEAR, tested.clears[0], nullptr);
        expectLayer(0, tested.cleared[0]);
        expectLayer(1, tested.seeded);
        render(VERNON_RHI_LOAD_PRESERVE, VernonClearColor{}, &tested.clears[1]);
        expectLayer(0, tested.cleared[1]);
        expectLayer(1, tested.seeded);

        EXPECT_EQ(vernonRhiDeviceDestroyImageView(device, view), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
    }
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiResourceLifetime, testing::ValuesIn(vernon::tests::rhiBackendCases()),
                         [](const testing::TestParamInfo<vernon::tests::BackendTestRow> &info) {
                             return std::string(info.param.name);
                         });

// OpenGL ES downloads depth and stencil through a compute pass (OpenGL ES 3.1), and cube-array images need OpenGL ES
// 3.2, so the context is created at that version instead of the default.
class RhiDepthStencilDownload : public testing::TestWithParam<vernon::tests::BackendTestRow> {
protected:
    void SetUp() override {
        ASSERT_TRUE(GetParam().rhi.has_value());
        vernon::tests::BackendTestRequirements requirements;
        requirements.graphics = true;
        if (GetParam().runtime == VERNON_RUNTIME_OPENGL_ES) {
            requirements.minimumApiMajor = 3;
            requirements.minimumApiMinor = 2;
        }
        const vernon::tests::BackendProbeResult probe = runtime_.initialize(GetParam(), requirements);
        if (!probe.available()) {
            if (probe.skippable())
                GTEST_SKIP() << probe.reason;
            FAIL() << probe.reason;
        }
    }

    VernonRhiDevice device() { return runtime_.context().device; }

private:
    vernon::tests::BackendRuntimeOwner runtime_;
};

// Every texel of every mip and layer holds its own depth and stencil value, so a download that reads the wrong layer,
// cube face, mip or region cannot match.
TEST_P(RhiDepthStencilDownload, ReadsTheRequestedAspectSubresourceAndRegion) {
    const VernonRhiDevice device = this->device();
    constexpr uint32_t extent = 8;
    constexpr uint32_t mipLevels = 2;
    const auto depthValue = [](uint32_t mip, uint32_t layer, uint32_t x, uint32_t y) {
        return static_cast<float>(1 + x + 8 * y + 64 * layer + 1024 * mip) / 65536.0f;
    };
    const auto stencilValue = [](uint32_t mip, uint32_t layer, uint32_t x, uint32_t y) {
        return static_cast<uint8_t>((x + 3 * y + 17 * layer + 101 * mip) & 0xff);
    };
    struct Shape {
        VernonRhiImageDimension dimension;
        uint32_t layers;
    };
    for (const Shape shape : {Shape{VERNON_RHI_IMAGE_2D, 1}, Shape{VERNON_RHI_IMAGE_2D_ARRAY, 3},
                              Shape{VERNON_RHI_IMAGE_CUBE, 6}, Shape{VERNON_RHI_IMAGE_CUBE_ARRAY, 12}}) {
        for (const VernonRhiFormat format : {VERNON_RHI_FORMAT_D32_FLOAT, VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT}) {
            const bool hasStencil = format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT;
            SCOPED_TRACE("dimension " + std::to_string(shape.dimension) + (hasStencil ? " with stencil" : ""));
            VernonRhiImageDescriptor descriptor{};
            descriptor.struct_size = sizeof(descriptor);
            descriptor.dimension = shape.dimension;
            descriptor.width = extent;
            descriptor.height = extent;
            descriptor.depth = 1;
            descriptor.format = format;
            descriptor.mip_levels = mipLevels;
            descriptor.array_layers = shape.layers;
            descriptor.sample_count = 1;
            descriptor.usage = VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT | VERNON_RHI_IMAGE_TRANSFER_SOURCE |
                               VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
            VernonRhiImage image{};
            ASSERT_EQ(vernonRhiDeviceCreateImage(device, &descriptor, &image), VERNON_RHI_STATUS_OK);
            for (uint32_t mip = 0; mip < mipLevels; ++mip) {
                const uint32_t size = extent >> mip;
                for (uint32_t layer = 0; layer < shape.layers; ++layer) {
                    std::vector<uint8_t> texels(size * size * (hasStencil ? 8 : 4));
                    for (uint32_t y = 0; y < size; ++y)
                        for (uint32_t x = 0; x < size; ++x) {
                            const size_t index = static_cast<size_t>(y) * size + x;
                            const float depth = depthValue(mip, layer, x, y);
                            std::memcpy(texels.data() + index * (hasStencil ? 8 : 4), &depth, sizeof(depth));
                            if (hasStencil)
                                texels[index * 8 + 4] = stencilValue(mip, layer, x, y);
                        }
                    VernonRhiImageUploadDescriptor upload{};
                    upload.struct_size = sizeof(upload);
                    upload.mip_level = mip;
                    upload.array_layer = layer;
                    upload.aspect = hasStencil ? VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL
                                               : VERNON_RHI_IMAGE_ASPECT_DEPTH;
                    upload.width = size;
                    upload.height = size;
                    upload.depth = 1;
                    upload.source_format =
                        hasStencil ? VERNON_RHI_IMAGE_DATA_DEPTH_STENCIL : VERNON_RHI_IMAGE_DATA_DEPTH;
                    upload.source_type = VERNON_RHI_IMAGE_DATA_FLOAT32;
                    upload.data = texels.data();
                    ASSERT_EQ(vernonRhiDeviceUploadImage(device, image, &upload, 1), VERNON_RHI_STATUS_OK)
                        << vernon::test::text(vernonRhiDeviceGetLastError(device));
                }
            }

            constexpr uint32_t mip = 1;
            constexpr uint32_t offsetX = 1;
            constexpr uint32_t offsetY = 2;
            constexpr uint32_t width = 2;
            constexpr uint32_t height = 1;
            const uint32_t layer = shape.layers - 1;
            VernonRhiImageDownloadDescriptor download{};
            download.struct_size = sizeof(download);
            download.mip_level = mip;
            download.array_layer = layer;
            download.offset_x = offsetX;
            download.offset_y = offsetY;
            download.width = width;
            download.height = height;
            download.depth = 1;
            download.aspect = VERNON_RHI_IMAGE_ASPECT_DEPTH;
            download.destination_format = VERNON_RHI_IMAGE_DATA_DEPTH;
            download.destination_type = VERNON_RHI_IMAGE_DATA_FLOAT32;
            std::array<float, width * height> depth{};
            ASSERT_EQ(vernonRhiDeviceDownloadImage(device, image, &download, depth.data(), sizeof(depth)),
                      VERNON_RHI_STATUS_OK)
                << vernon::test::text(vernonRhiDeviceGetLastError(device));
            std::array<uint8_t, width * height> stencil{};
            std::array<uint8_t, width * height * 8> packed{};
            if (hasStencil) {
                download.aspect = VERNON_RHI_IMAGE_ASPECT_STENCIL;
                download.destination_format = VERNON_RHI_IMAGE_DATA_STENCIL;
                download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
                ASSERT_EQ(vernonRhiDeviceDownloadImage(device, image, &download, stencil.data(), stencil.size()),
                          VERNON_RHI_STATUS_OK)
                    << vernon::test::text(vernonRhiDeviceGetLastError(device));
                download.aspect = VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL;
                download.destination_format = VERNON_RHI_IMAGE_DATA_DEPTH_STENCIL;
                download.destination_type = VERNON_RHI_IMAGE_DATA_FLOAT32;
                ASSERT_EQ(vernonRhiDeviceDownloadImage(device, image, &download, packed.data(), packed.size()),
                          VERNON_RHI_STATUS_OK)
                    << vernon::test::text(vernonRhiDeviceGetLastError(device));
            }
            for (uint32_t y = 0; y < height; ++y)
                for (uint32_t x = 0; x < width; ++x) {
                    SCOPED_TRACE("texel " + std::to_string(x) + "," + std::to_string(y));
                    const size_t index = static_cast<size_t>(y) * width + x;
                    const float expectedDepth = depthValue(mip, layer, offsetX + x, offsetY + y);
                    EXPECT_EQ(depth[index], expectedDepth);
                    if (!hasStencil)
                        continue;
                    const uint8_t expectedStencil = stencilValue(mip, layer, offsetX + x, offsetY + y);
                    EXPECT_EQ(stencil[index], expectedStencil);
                    float packedDepth{};
                    std::memcpy(&packedDepth, packed.data() + index * 8, sizeof(packedDepth));
                    EXPECT_EQ(packedDepth, expectedDepth);
                    EXPECT_EQ(packed[index * 8 + 4], expectedStencil);
                }
            if (hasStencil) {
                const float depthOnly = 0.5f;
                VernonRhiImageUploadDescriptor upload{};
                upload.struct_size = sizeof(upload);
                upload.aspect = VERNON_RHI_IMAGE_ASPECT_DEPTH;
                upload.width = 1;
                upload.height = 1;
                upload.depth = 1;
                upload.source_format = VERNON_RHI_IMAGE_DATA_DEPTH;
                upload.source_type = VERNON_RHI_IMAGE_DATA_FLOAT32;
                upload.data = &depthOnly;
                const bool opengl =
                    GetParam().runtime == VERNON_RUNTIME_OPENGL || GetParam().runtime == VERNON_RUNTIME_OPENGL_ES;
                EXPECT_EQ(vernonRhiDeviceUploadImage(device, image, &upload, 1),
                          opengl ? VERNON_RHI_STATUS_UNSUPPORTED : VERNON_RHI_STATUS_OK)
                    << vernon::test::text(vernonRhiDeviceGetLastError(device));
            }
            EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiDepthStencilDownload, testing::ValuesIn(vernon::tests::rhiBackendCases()),
                         [](const testing::TestParamInfo<vernon::tests::BackendTestRow> &info) {
                             return std::string(info.param.name);
                         });

} // namespace
