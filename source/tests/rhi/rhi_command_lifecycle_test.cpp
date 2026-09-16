#include "VernonRHI.h"
#include "backend_runtime_owner.h"
#include "backend_test_matrix.h"
#include "rhi/rhi_internal.h"

#include <gtest/gtest.h>

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

INSTANTIATE_TEST_SUITE_P(Backends, RhiCommandLifecycleTest, testing::ValuesIn(vernon::tests::rhiBackendCases()),
                         [](const testing::TestParamInfo<vernon::tests::BackendTestRow> &info) {
                             return std::string(info.param.name);
                         });

} // namespace
