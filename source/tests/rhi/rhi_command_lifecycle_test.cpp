#include "VernonRHI.h"
#include "backend_runtime_owner.h"
#include "backend_test_matrix.h"
#include "rhi/rhi_internal.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

class RhiCommandLifecycleTest : public testing::TestWithParam<vernon::tests::BackendTestRow> {
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
    vernon::tests::RhiRuntime &rhi() { return runtime_.context(); }

private:
    vernon::tests::BackendRuntimeOwner runtime_;
};

struct RetryableCleanup {
    size_t calls{};
    bool fail{true};
};

void retryableCleanup(void *context, uint64_t) {
    auto &state = *static_cast<RetryableCleanup *>(context);
    ++state.calls;
    if (state.fail) {
        state.fail = false;
        throw std::runtime_error("injected completion cleanup failure");
    }
}

TEST_P(RhiCommandLifecycleTest, FailedCompletionCleanupRemainsRetryable) {
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder encoder{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device(), &descriptor, &encoder), VERNON_RHI_STATUS_OK);
    const uint64_t encoderKey = static_cast<uint64_t>(encoder.generation) << 32 | uint64_t{encoder.index} + 1;
    RetryableCleanup cleanup;
    ASSERT_TRUE(vernon::rhi::deferCommandCleanup(device(), encoderKey, &cleanup, 1, retryableCleanup).isOk());
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device(), encoder), VERNON_RHI_STATUS_OK);
    VernonRhiCompletion completion{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceSubmit(device(), encoder, &completion), VERNON_RHI_STATUS_OK);

    EXPECT_EQ(vernonRhiCompletionWait(device(), completion), VERNON_RHI_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(vernonRhiCompletionWait(device(), completion), VERNON_RHI_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(cleanup.calls, 2u);
    EXPECT_EQ(vernonRhiDeviceDestroyCompletion(device(), completion), VERNON_RHI_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(vernonRhiDeviceDestroyCompletion(device(), completion), VERNON_RHI_STATUS_INVALID_ARGUMENT);
}

TEST_P(RhiCommandLifecycleTest, TimersSpanLongerWorkLonger) {
    if (!(vernon::rhi::deviceCommandCapabilities(device()) & vernon::rhi::BackendCommandTimers)) {
        ASSERT_NE(GetParam().runtime, VERNON_RUNTIME_CUDA);
        ASSERT_NE(GetParam().runtime, VERNON_RUNTIME_DIRECTX12);
        GTEST_SKIP() << GetParam().name << " device records no timestamps";
    }
    // Large enough that the copies outlast the host recording them on backends that execute while recording, and that
    // the large one spills out of the GPU caches.
    constexpr uint64_t smallBytes = uint64_t{1} << 20;
    constexpr uint64_t largeBytes = 100 * smallBytes;
    const vernon::tests::RhiBuffer source = vernon::tests::createBuffer(rhi(), largeBytes, 16, 0);
    const vernon::tests::RhiBuffer destination = vernon::tests::createBuffer(rhi(), largeBytes, 16, 0);
    ASSERT_NE(source.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(destination.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder encoder{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    // Drivers may commit buffer memory on first use, which would charge the first timed copy for it.
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device(), &descriptor, &encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderCopyBuffer(device(), encoder, source.handle, 0, destination.handle, 0, largeBytes),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device(), encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernon::tests::completeSubmission(device(), encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device(), &descriptor, &encoder), VERNON_RHI_STATUS_OK);
    for (const uint64_t bytes : {smallBytes, largeBytes}) {
        uint32_t timer{};
        ASSERT_EQ(vernonRhiCommandEncoderBeginTimer(device(), encoder, &timer), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiCommandEncoderCopyBuffer(device(), encoder, source.handle, 0, destination.handle, 0, bytes),
                  VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiCommandEncoderEndTimer(device(), encoder, timer), VERNON_RHI_STATUS_OK);
    }
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device(), encoder), VERNON_RHI_STATUS_OK);
    VernonRhiCompletion completion{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceSubmit(device(), encoder, &completion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCompletionWait(device(), completion), VERNON_RHI_STATUS_OK);
    std::array<VernonRhiTimerSpan, 2> spans{};
    size_t count{};
    ASSERT_EQ(vernonRhiCompletionGetTimers(device(), completion, spans.data(), spans.size(), &count),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(count, spans.size());
    EXPECT_EQ(spans[0].begin_nanoseconds, 0U);
    for (const VernonRhiTimerSpan &span : spans)
        EXPECT_LE(span.begin_nanoseconds, span.end_nanoseconds);
    EXPECT_GT(spans[1].end_nanoseconds - spans[1].begin_nanoseconds,
              spans[0].end_nanoseconds - spans[0].begin_nanoseconds);
    EXPECT_EQ(vernonRhiDeviceDestroyCompletion(device(), completion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device(), destination.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device(), source.handle), VERNON_RHI_STATUS_OK);
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiCommandLifecycleTest, testing::ValuesIn(vernon::tests::rhiBackendCases()),
                         [](const testing::TestParamInfo<vernon::tests::BackendTestRow> &info) {
                             return std::string(info.param.name);
                         });

} // namespace
