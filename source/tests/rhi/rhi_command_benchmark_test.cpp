#include "VernonRHI.h"
#include "backend_runtime_owner.h"
#include "backend_test_matrix.h"
#include "rhi/rhi_internal.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <string>
#include <vector>

namespace {

thread_local bool trackAllocations{};
std::atomic<std::size_t> trackedAllocations{};

constexpr std::size_t recordIterationCount = 4096;
constexpr std::size_t submissionIterationCount = 128;

template <typename Function>
VernonRhiStatus sampleOperation(Function &&function, std::vector<std::uint64_t> &samples,
                                std::size_t &allocationCount) {
    const std::size_t before = trackedAllocations.load(std::memory_order_relaxed);
    trackAllocations = true;
    const auto begin = std::chrono::steady_clock::now();
    const VernonRhiStatus status = function();
    const auto end = std::chrono::steady_clock::now();
    trackAllocations = false;
    allocationCount += trackedAllocations.load(std::memory_order_relaxed) - before;
    samples.push_back(
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()));
    return status;
}

template <typename Function>
bool sampleBookkeeping(Function &&function, std::vector<std::uint64_t> &samples, std::size_t &allocationCount) {
    const std::size_t before = trackedAllocations.load(std::memory_order_relaxed);
    trackAllocations = true;
    const auto begin = std::chrono::steady_clock::now();
    const bool succeeded = function();
    const auto end = std::chrono::steady_clock::now();
    trackAllocations = false;
    allocationCount += trackedAllocations.load(std::memory_order_relaxed) - before;
    samples.push_back(
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()));
    return succeeded;
}

void recordSamples(const char *name, std::vector<std::uint64_t> &samples, std::size_t allocations) {
    ASSERT_FALSE(samples.empty());
    std::sort(samples.begin(), samples.end());
    const std::string prefix{name};
    const std::size_t p99Index = (samples.size() * 99 + 99) / 100 - 1;
    testing::Test::RecordProperty((prefix + "_median_ns").c_str(), samples[samples.size() / 2]);
    testing::Test::RecordProperty((prefix + "_p99_ns").c_str(), samples[p99Index]);
    testing::Test::RecordProperty((prefix + "_maximum_ns").c_str(), samples.back());
    testing::Test::RecordProperty((prefix + "_allocation_count").c_str(), allocations);
}

class RhiCommandBenchmark : public testing::TestWithParam<vernon::tests::BackendTestRow> {
protected:
    void SetUp() override {
        ASSERT_TRUE(GetParam().rhi.has_value());
        const auto probe = runtime_.initialize(GetParam(), {});
        if (!probe.available()) {
            if (probe.skippable())
                GTEST_SKIP() << probe.reason;
            FAIL() << probe.reason;
        }
    }

    VernonRhiDevice device() { return runtime_.context().device; }
    VernonRuntimeContext *runtime() { return runtime_.context().runtime; }

private:
    vernon::tests::BackendRuntimeOwner runtime_;
};

VernonRhiBuffer createTransferBuffer(VernonRhiDevice device) {
    VernonRhiBufferDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.size = 4096;
    descriptor.usage =
        VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION | VERNON_RHI_BUFFER_STORAGE;
    descriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer buffer{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    EXPECT_EQ(vernonRhiDeviceCreateBuffer(device, &descriptor, &buffer), VERNON_RHI_STATUS_OK);
    return buffer;
}

VernonRhiCommandEncoder createEncoder(VernonRhiDevice device, std::uint32_t capabilities) {
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = capabilities;
    VernonRhiCommandEncoder encoder{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    EXPECT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &encoder), VERNON_RHI_STATUS_OK);
    return encoder;
}

TEST_P(RhiCommandBenchmark, BarrierCopyAndDispatchBookkeepingHotPath) {
    RecordProperty("backend", std::string(GetParam().name));
    const VernonRhiDevice testDevice = device();
    const VernonRhiBuffer source = createTransferBuffer(testDevice);
    const VernonRhiBuffer destination = createTransferBuffer(testDevice);
    ASSERT_NE(source.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(destination.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    const VernonRhiCommandEncoder encoder =
        createEncoder(testDevice, VERNON_RHI_QUEUE_TRANSFER | VERNON_RHI_QUEUE_COMPUTE);
    ASSERT_NE(encoder.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    VernonRhiBarrier barrier{};
    barrier.struct_size = sizeof(barrier);
    barrier.source_stage_mask = VERNON_RHI_STAGE_COMPUTE;
    barrier.destination_stage_mask = VERNON_RHI_STAGE_COMPUTE;
    barrier.source_access = VERNON_RHI_ACCESS_SHADER_READ;
    barrier.destination_access = VERNON_RHI_ACCESS_SHADER_READ;
    barrier.old_state = VERNON_RHI_STATE_COMMON;
    barrier.new_state = VERNON_RHI_STATE_COMMON;
    barrier.buffer = destination;

    ASSERT_EQ(vernonRhiCommandEncoderBarrier(testDevice, encoder, &barrier, 1), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderCopyBuffer(testDevice, encoder, source, 0, destination, 0, 256),
              VERNON_RHI_STATUS_OK);
    auto encoderKeyResult = vernon::rhi::commandEncoderKey(testDevice, encoder);
    ASSERT_TRUE(encoderKeyResult.isOk());
    const std::uint64_t encoderKey = encoderKeyResult.value();
    ASSERT_TRUE(vernon::rhi::recordProviderCommand(testDevice, encoderKey, false).isOk());

    std::vector<std::uint64_t> barrierSamples;
    std::vector<std::uint64_t> copySamples;
    std::vector<std::uint64_t> dispatchSamples;
    barrierSamples.reserve(recordIterationCount);
    copySamples.reserve(recordIterationCount);
    dispatchSamples.reserve(recordIterationCount);
    std::size_t barrierAllocations{};
    std::size_t copyAllocations{};
    std::size_t dispatchAllocations{};
    for (std::size_t iteration = 0; iteration < recordIterationCount; ++iteration) {
        ASSERT_EQ(sampleOperation([&] { return vernonRhiCommandEncoderBarrier(testDevice, encoder, &barrier, 1); },
                                  barrierSamples, barrierAllocations),
                  VERNON_RHI_STATUS_OK);
        ASSERT_EQ(
            sampleOperation(
                [&] { return vernonRhiCommandEncoderCopyBuffer(testDevice, encoder, source, 0, destination, 0, 256); },
                copySamples, copyAllocations),
            VERNON_RHI_STATUS_OK);
        ASSERT_TRUE(
            sampleBookkeeping([&] { return vernon::rhi::recordProviderCommand(testDevice, encoderKey, false).isOk(); },
                              dispatchSamples, dispatchAllocations));
    }

    recordSamples("barrier", barrierSamples, barrierAllocations);
    recordSamples("copy_buffer", copySamples, copyAllocations);
    recordSamples("dispatch_bookkeeping", dispatchSamples, dispatchAllocations);
    RecordProperty("global_lifecycle_lock_measurement", "forbidden-path source audit");

    ASSERT_EQ(vernonRhiCommandEncoderFinish(testDevice, encoder), VERNON_RHI_STATUS_OK);
    VernonRhiCompletion completion{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceSubmit(testDevice, encoder, &completion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCompletionWait(testDevice, completion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyCompletion(testDevice, completion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(testDevice, source), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(testDevice, destination), VERNON_RHI_STATUS_OK);
}

TEST_P(RhiCommandBenchmark, SubmissionAndCompletionPollHotPath) {
    RecordProperty("backend", std::string(GetParam().name));
    const VernonRhiDevice testDevice = device();

    // Populate and recycle command slots before allocation accounting begins.
    {
        const VernonRhiCommandEncoder warmup = createEncoder(testDevice, VERNON_RHI_QUEUE_COMPUTE);
        ASSERT_EQ(vernonRhiCommandEncoderFinish(testDevice, warmup), VERNON_RHI_STATUS_OK);
        VernonRhiCompletion completion{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
        ASSERT_EQ(vernonRhiDeviceSubmit(testDevice, warmup, &completion), VERNON_RHI_STATUS_OK);
        VernonRhiCompletionState state{};
        ASSERT_EQ(vernonRhiCompletionGetState(testDevice, completion, &state), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiCompletionWait(testDevice, completion), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceDestroyCompletion(testDevice, completion), VERNON_RHI_STATUS_OK);
    }

    std::vector<std::uint64_t> submissionSamples;
    std::vector<std::uint64_t> pollSamples;
    submissionSamples.reserve(submissionIterationCount);
    pollSamples.reserve(submissionIterationCount);
    std::size_t submissionAllocations{};
    std::size_t pollAllocations{};
    for (std::size_t iteration = 0; iteration < submissionIterationCount; ++iteration) {
        const VernonRhiCommandEncoder encoder = createEncoder(testDevice, VERNON_RHI_QUEUE_COMPUTE);
        ASSERT_EQ(vernonRhiCommandEncoderFinish(testDevice, encoder), VERNON_RHI_STATUS_OK);
        VernonRhiCompletion completion{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
        ASSERT_EQ(sampleOperation([&] { return vernonRhiDeviceSubmit(testDevice, encoder, &completion); },
                                  submissionSamples, submissionAllocations),
                  VERNON_RHI_STATUS_OK);
        VernonRhiCompletionState state{};
        ASSERT_EQ(sampleOperation([&] { return vernonRhiCompletionGetState(testDevice, completion, &state); },
                                  pollSamples, pollAllocations),
                  VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiCompletionWait(testDevice, completion), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceDestroyCompletion(testDevice, completion), VERNON_RHI_STATUS_OK);
    }

    recordSamples("submission", submissionSamples, submissionAllocations);
    recordSamples("completion_poll", pollSamples, pollAllocations);
    RecordProperty("global_lifecycle_lock_measurement", "forbidden-path source audit");
}

TEST_P(RhiCommandBenchmark, GraphicsBeginDrawEndBookkeepingHotPath) {
    RecordProperty("backend", std::string(GetParam().name));
    vernon::tests::BackendTestRequirements requirements;
    requirements.graphics = true;
    const vernon::tests::BackendProbeResult probe =
        vernon::tests::probeRuntimeBackend(GetParam(), requirements, runtime());
    if (!probe.available()) {
        if (probe.skippable())
            GTEST_SKIP() << probe.reason;
        FAIL() << probe.reason;
        return;
    }

    const VernonRhiDevice testDevice = device();
    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageDescriptor.width = imageDescriptor.height = imageDescriptor.depth = 1;
    imageDescriptor.mip_levels = imageDescriptor.array_layers = imageDescriptor.sample_count = 1;
    imageDescriptor.usage = VERNON_RHI_IMAGE_COLOR_ATTACHMENT;
    VernonRhiImage image{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    const VernonRhiStatus imageStatus = vernonRhiDeviceCreateImage(testDevice, &imageDescriptor, &image);
    ASSERT_EQ(imageStatus, VERNON_RHI_STATUS_OK);
    VernonRhiImageViewDescriptor viewDescriptor{};
    viewDescriptor.struct_size = sizeof(viewDescriptor);
    viewDescriptor.image = image;
    viewDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    viewDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    viewDescriptor.mip_level_count = viewDescriptor.array_layer_count = 1;
    viewDescriptor.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
    VernonRhiImageView view{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    const VernonRhiStatus viewStatus = vernonRhiDeviceCreateImageView(testDevice, &viewDescriptor, &view);
    ASSERT_EQ(viewStatus, VERNON_RHI_STATUS_OK);
    const VernonRhiCommandEncoder encoder = createEncoder(testDevice, VERNON_RHI_QUEUE_GRAPHICS);
    ASSERT_NE(encoder.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    auto encoderKeyResult = vernon::rhi::commandEncoderKey(testDevice, encoder);
    ASSERT_TRUE(encoderKeyResult.isOk());
    const std::uint64_t encoderKey = encoderKeyResult.value();

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
    rendering.width = rendering.height = rendering.layers = 1;

    ASSERT_EQ(vernonRhiCommandEncoderBeginRendering(testDevice, encoder, &rendering), VERNON_RHI_STATUS_OK);
    ASSERT_TRUE(vernon::rhi::recordProviderCommand(testDevice, encoderKey, true).isOk());
    ASSERT_EQ(vernonRhiCommandEncoderEndRendering(testDevice, encoder), VERNON_RHI_STATUS_OK);

    std::vector<std::uint64_t> beginSamples;
    std::vector<std::uint64_t> drawSamples;
    std::vector<std::uint64_t> endSamples;
    beginSamples.reserve(recordIterationCount);
    drawSamples.reserve(recordIterationCount);
    endSamples.reserve(recordIterationCount);
    std::size_t beginAllocations{};
    std::size_t drawAllocations{};
    std::size_t endAllocations{};
    for (std::size_t iteration = 0; iteration < recordIterationCount; ++iteration) {
        ASSERT_EQ(
            sampleOperation([&] { return vernonRhiCommandEncoderBeginRendering(testDevice, encoder, &rendering); },
                            beginSamples, beginAllocations),
            VERNON_RHI_STATUS_OK);
        ASSERT_TRUE(
            sampleBookkeeping([&] { return vernon::rhi::recordProviderCommand(testDevice, encoderKey, true).isOk(); },
                              drawSamples, drawAllocations));
        ASSERT_EQ(sampleOperation([&] { return vernonRhiCommandEncoderEndRendering(testDevice, encoder); }, endSamples,
                                  endAllocations),
                  VERNON_RHI_STATUS_OK);
    }

    recordSamples("graphics_begin", beginSamples, beginAllocations);
    recordSamples("draw_bookkeeping", drawSamples, drawAllocations);
    recordSamples("graphics_end", endSamples, endAllocations);
    RecordProperty("global_lifecycle_lock_measurement", "forbidden-path source audit");

    ASSERT_EQ(vernonRhiCommandEncoderFinish(testDevice, encoder), VERNON_RHI_STATUS_OK);
    VernonRhiCompletion completion{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceSubmit(testDevice, encoder, &completion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCompletionWait(testDevice, completion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyCompletion(testDevice, completion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImageView(testDevice, view), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(testDevice, image), VERNON_RHI_STATUS_OK);
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiCommandBenchmark, testing::ValuesIn(vernon::tests::rhiBackendCases()),
                         [](const testing::TestParamInfo<vernon::tests::BackendTestRow> &info) {
                             return std::string(info.param.name);
                         });

} // namespace

void *operator new(std::size_t size) {
    if (void *memory = std::malloc(size)) {
        if (trackAllocations)
            trackedAllocations.fetch_add(1, std::memory_order_relaxed);
        return memory;
    }
    throw std::bad_alloc{};
}

void *operator new[](std::size_t size) { return ::operator new(size); }
void *operator new(std::size_t size, const std::nothrow_t &) noexcept {
    void *memory = std::malloc(size);
    if (memory && trackAllocations)
        trackedAllocations.fetch_add(1, std::memory_order_relaxed);
    return memory;
}
void *operator new[](std::size_t size, const std::nothrow_t &tag) noexcept { return ::operator new(size, tag); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete(void *memory, const std::nothrow_t &) noexcept { std::free(memory); }
void operator delete[](void *memory, const std::nothrow_t &) noexcept { std::free(memory); }
