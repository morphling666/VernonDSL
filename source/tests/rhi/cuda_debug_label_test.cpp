#include "VernonRHI.h"
#include "nvtx_call_recorder.h"
#include "runtime_rhi_test_utils.h"

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <vector>

namespace {

class CudaDebugLabels : public testing::Test {
protected:
    void SetUp() override {
        VernonRhiOwnedDeviceDescriptor deviceDescriptor{};
        deviceDescriptor.struct_size = sizeof(deviceDescriptor);
        deviceDescriptor.backend = VERNON_RHI_BACKEND_CUDA;
        device = vernonRhiCreateDevice(&deviceDescriptor);
        if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
            GTEST_SKIP() << "CUDA RHI device is unavailable";
        VernonRhiCommandEncoderDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
        ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &encoder), VERNON_RHI_STATUS_OK);
        vernon::tests::takeNvtxCalls();
    }

    void TearDown() override {
        if (device.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            EXPECT_EQ(vernonRhiDestroyDevice(device), VERNON_RHI_STATUS_OK);
    }

    void push(std::string_view label) const {
        ASSERT_EQ(vernonRhiCommandEncoderPushDebugGroup(device, encoder, {label.data(), label.size()}),
                  VERNON_RHI_STATUS_OK);
    }

    VernonRhiDevice device{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiCommandEncoder encoder{};
};

TEST_F(CudaDebugLabels, RecordsGroupsAndMarkersAsNvtxRangesAndMarks) {
    push("a");
    push("b");
    constexpr std::string_view marker = "m";
    ASSERT_EQ(vernonRhiCommandEncoderInsertDebugMarker(device, encoder, {marker.data(), marker.size()}),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderPopDebugGroup(device, encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderPopDebugGroup(device, encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernon::tests::completeSubmission(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernon::tests::takeNvtxCalls(),
              (std::vector<std::string>{"start a", "start b", "mark m", "end b", "end a"}));
}

TEST_F(CudaDebugLabels, EndsTheRangesOfADestroyedEncoder) {
    push("a");
    push("b");
    ASSERT_EQ(vernonRhiDeviceDestroyCommandEncoder(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernon::tests::takeNvtxCalls(), (std::vector<std::string>{"start a", "start b", "end b", "end a"}));
}

} // namespace
