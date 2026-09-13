#include "VernonRuntimeRHIAdapter.h"
#include "backend_runtime_owner.h"
#include "backend_test_matrix.h"
#include "execution_graph/command_graph.h"
#include "rhi/rhi_internal.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <thread>
#include <vector>

namespace {

std::vector<vernon::tests::BackendTestRow> rhiBackendCases() {
    std::vector<vernon::tests::BackendTestRow> result;
    for (const vernon::tests::BackendTestRow &backend : vernon::tests::backendTestMatrix)
        if (backend.rhi)
            result.push_back(backend);
    return result;
}

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

private:
    vernon::tests::BackendRuntimeOwner runtime_;
};

VernonRhiStatus probeImageSupport(VernonRhiDevice device) {
    VernonRhiImageDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.dimension = VERNON_RHI_IMAGE_2D;
    descriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    descriptor.width = descriptor.height = descriptor.depth = 1;
    descriptor.mip_levels = descriptor.array_layers = descriptor.sample_count = 1;
    descriptor.usage = VERNON_RHI_IMAGE_SAMPLED;
    VernonRhiImage image{};
    const VernonRhiStatus status = vernonRhiDeviceCreateImage(device, &descriptor, &image);
    if (status != VERNON_RHI_STATUS_OK)
        return status;
    return vernonRhiDeviceDestroyImage(device, image);
}

VernonRhiStatus probeSamplerSupport(VernonRhiDevice device) {
    VernonRhiSamplerDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    VernonRhiSampler sampler{};
    const VernonRhiStatus status = vernonRhiDeviceCreateSampler(device, &descriptor, &sampler);
    if (status != VERNON_RHI_STATUS_OK)
        return status;
    return vernonRhiDeviceDestroySampler(device, sampler);
}

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

    vernonRhiDestroyDevice(device);
    EXPECT_TRUE(vernon::rhi::deviceExists(device));
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    VernonRhiCompletion completion{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceSubmit(device, encoder, &completion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCompletionWait(device, completion), VERNON_RHI_STATUS_OK);

    vernonRhiDestroyDevice(device);
    EXPECT_TRUE(vernon::rhi::deviceExists(device));
    ASSERT_EQ(vernonRhiDeviceDestroyCompletion(device, completion), VERNON_RHI_STATUS_OK);
    EXPECT_TRUE(vernon::rhi::deviceExists(device));
}

TEST_P(RhiResourceLifetime, FailedCommandDestroyRollsBackAndCanRetry) {
    const VernonRhiDevice device = this->device();
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_GRAPHICS;
    VernonRhiCommandEncoder encoder{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &encoder), VERNON_RHI_STATUS_OK);
    ASSERT_TRUE(vernon::rhi::beginProviderRendering(device, encoder).isOk());

    EXPECT_EQ(vernonRhiDeviceDestroyCommandEncoder(device, encoder), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiCommandEncoderEndRendering(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyCommandEncoder(device, encoder), VERNON_RHI_STATUS_OK);
}

TEST_P(RhiResourceLifetime, GuessedCompletionCannotObservePartiallyInitializedSubmission) {
    const VernonRhiDevice device = this->device();
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
    vernonRhiDestroyDevice(device);
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

TEST_P(RhiResourceLifetime, UnsupportedResourceKindsAreRejected) {
    const vernon::tests::BackendTestRow test = GetParam();
    const VernonRhiDevice device = this->device();

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
    if (imageStatus == VERNON_RHI_STATUS_OK) {
        EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
        GTEST_SKIP() << test.name << " supports images and samplers";
    }
    EXPECT_EQ(imageStatus, VERNON_RHI_STATUS_UNSUPPORTED);

    VernonRhiSamplerDescriptor samplerDescriptor{};
    samplerDescriptor.struct_size = sizeof(samplerDescriptor);
    VernonRhiSampler sampler{};
    EXPECT_EQ(vernonRhiDeviceCreateSampler(device, &samplerDescriptor, &sampler), VERNON_RHI_STATUS_UNSUPPORTED);
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
    const VernonRhiStatus imageSupport = probeImageSupport(device);
    if (imageSupport == VERNON_RHI_STATUS_UNSUPPORTED) {
        GTEST_SKIP() << test.name << " does not support images";
    }
    ASSERT_EQ(imageSupport, VERNON_RHI_STATUS_OK);
    const VernonRhiStatus samplerSupport = probeSamplerSupport(device);
    if (samplerSupport == VERNON_RHI_STATUS_UNSUPPORTED) {
        GTEST_SKIP() << test.name << " does not support samplers";
    }
    ASSERT_EQ(samplerSupport, VERNON_RHI_STATUS_OK);

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
    const VernonRhiStatus imageSupport = probeImageSupport(device);
    if (imageSupport == VERNON_RHI_STATUS_UNSUPPORTED) {
        GTEST_SKIP() << test.name << " does not support images";
    }
    ASSERT_EQ(imageSupport, VERNON_RHI_STATUS_OK);
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
    const VernonRhiStatus viewStatus = vernonRhiDeviceCreateImageView(device, &viewDescriptor, &view);
    if (viewStatus == VERNON_RHI_STATUS_UNSUPPORTED) {
        EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
        vernonRuntimeRhiAdapterDestroy(adapter);
        GTEST_SKIP() << test.name << " does not support non-identity image views";
    }
    ASSERT_EQ(viewStatus, VERNON_RHI_STATUS_OK);
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
    vernonRhiDestroyDevice(retainedDevice);
    EXPECT_TRUE(vernon::rhi::deviceExists(retainedDevice));
    vernonRuntimeRhiAdapterDestroy(adapter);

    VernonRuntimeContext *context = vernonRuntimeCreateForRhiDevice(backend.runtime, retainedDevice);
    ASSERT_NE(context, nullptr);

    vernonRhiDestroyDevice(retainedDevice);
    EXPECT_TRUE(vernon::rhi::deviceExists(retainedDevice));
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
    EXPECT_TRUE(vernon::rhi::deviceExists(retainedDevice));
}

TEST_P(RhiResourceLifetime, CommandGraphRetainsImportedImageViewAndParent) {
    const vernon::tests::BackendTestRow test = GetParam();
    const VernonRhiDevice device = this->device();
    const VernonRhiStatus imageSupport = probeImageSupport(device);
    if (imageSupport == VERNON_RHI_STATUS_UNSUPPORTED) {
        GTEST_SKIP() << test.name << " does not support images";
    }
    ASSERT_EQ(imageSupport, VERNON_RHI_STATUS_OK);

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

INSTANTIATE_TEST_SUITE_P(Backends, RhiResourceLifetime, testing::ValuesIn(rhiBackendCases()),
                         [](const testing::TestParamInfo<vernon::tests::BackendTestRow> &info) {
                             return std::string(info.param.name);
                         });

} // namespace
