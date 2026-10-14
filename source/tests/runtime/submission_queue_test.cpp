#include "backend_runtime_owner.h"
#include "backend_test_matrix.h"
#include "rhi/image_data_layout.h"
#include "runtime/program_execution/device_buffer.h"
#include "runtime/program_execution/submission_queue.h"
#include "runtime/program_execution/submission_record.h"
#include "runtime/runtime_state.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

using vernon::runtime::program_execution::DeviceBuffer;
using vernon::runtime::program_execution::SubmissionQueue;
using vernon::runtime::program_execution::SubmissionRecord;
using vernon::runtime::program_execution::SubmissionState;
using vernon::runtime::program_execution::TransientAllocationMeter;

constexpr size_t kCopyBytes = 4;

class RuntimeSubmissionQueue : public testing::TestWithParam<vernon::tests::BackendTestRow> {
protected:
    void SetUp() override {
        ASSERT_TRUE(GetParam().rhi.has_value());
        const vernon::tests::BackendProbeResult probe = runtime_.initialize(GetParam(), {});
        if (!probe.available()) {
            if (probe.skippable())
                GTEST_SKIP() << probe.reason;
            FAIL() << probe.reason;
        }
        source_ = std::make_unique<DeviceBuffer>(context(), kCopyBytes);
        destination_ = std::make_unique<DeviceBuffer>(context(), kCopyBytes);
        ASSERT_TRUE(source_->valid());
        ASSERT_TRUE(destination_->valid());
    }

    void TearDown() override {
        destination_.reset();
        source_.reset();
    }

    VernonRuntimeContext &context() { return *runtime_.context().runtime; }

    // A record holding one submitted copy, retaining `retained` until it is retired.
    std::shared_ptr<SubmissionRecord> submitCopy(std::shared_ptr<const void> retained = nullptr) {
        const VernonRhiDevice device = context().rhiDevice;
        VernonRhiCommandEncoderDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.required_capabilities = VERNON_RHI_QUEUE_TRANSFER;
        VernonRhiCommandEncoder encoder{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
        EXPECT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &encoder), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(vernonRhiCommandEncoderCopyBuffer(device, encoder, source_->handle(), 0, destination_->handle(), 0,
                                                    kCopyBytes),
                  VERNON_RHI_STATUS_OK);
        EXPECT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
        auto record = std::make_shared<SubmissionRecord>(context());
        EXPECT_EQ(record->submit(encoder), VERNON_RHI_STATUS_OK);
        record->setOutcome({SubmissionState::Submitted, {}});
        if (retained)
            record->retain(std::move(retained));
        return record;
    }

    static VernonSubmissionState state(SubmissionRecord &record) {
        VernonSubmissionState result{VERNON_SUBMISSION_PENDING};
        EXPECT_EQ(record.poll(result), VERNON_STATUS_OK);
        return result;
    }

private:
    vernon::tests::BackendRuntimeOwner runtime_;
    std::unique_ptr<DeviceBuffer> source_;
    std::unique_ptr<DeviceBuffer> destination_;
};

TEST_P(RuntimeSubmissionQueue, TransientBudgetRetiresTheOldestSubmissionFirst) {
    SubmissionQueue queue;
    queue.configure({100, 16}, context().rhiDevice);
    auto token = std::make_shared<int>(0);
    const std::weak_ptr<int> observed = token;
    const std::shared_ptr<SubmissionRecord> oldest = submitCopy(std::move(token));
    queue.admit({64, 1});
    queue.push(oldest, 64);
    EXPECT_EQ(queue.telemetry().pending_transient_bytes, 64u);
    EXPECT_LE(queue.telemetry().pending_rhi_submissions, 1u);

    queue.admit({64, 1});
    EXPECT_TRUE(observed.expired());
    EXPECT_EQ(state(*oldest), VERNON_SUBMISSION_SUCCEEDED);
    VernonSubmissionTelemetry telemetry = queue.telemetry();
    EXPECT_EQ(telemetry.pending_submissions, 0u);
    EXPECT_EQ(telemetry.pending_transient_bytes, 0u);
    EXPECT_EQ(telemetry.pending_limit_waits, 0u);
    EXPECT_LE(telemetry.transient_budget_waits, 1u);
    const uint64_t budgetWaits = telemetry.transient_budget_waits;

    const std::shared_ptr<SubmissionRecord> fitting = submitCopy();
    queue.push(fitting, 64);
    queue.admit({36, 1});
    telemetry = queue.telemetry();
    EXPECT_EQ(telemetry.transient_budget_waits, budgetWaits);
    EXPECT_LE(telemetry.pending_submissions, 1u);
    EXPECT_EQ(fitting->wait(), VERNON_STATUS_OK);
}

TEST_P(RuntimeSubmissionQueue, PendingLimitRetiresTheOldestSubmissionFirst) {
    SubmissionQueue queue;
    queue.configure({uint64_t{1} << 40, 2}, context().rhiDevice);
    std::vector<std::shared_ptr<SubmissionRecord>> records;
    for (int index = 0; index < 6; ++index) {
        queue.admit({0, 1});
        if (records.size() >= 2)
            EXPECT_EQ(state(*records[records.size() - 2]), VERNON_SUBMISSION_SUCCEEDED) << "submission " << index - 2;
        records.push_back(submitCopy());
        queue.push(records.back(), 0);
        const VernonSubmissionTelemetry telemetry = queue.telemetry();
        EXPECT_LE(telemetry.pending_submissions, 2u);
        EXPECT_EQ(telemetry.transient_budget_waits, 0u);
    }
    EXPECT_LE(queue.telemetry().pending_limit_waits, 4u);
    for (const std::shared_ptr<SubmissionRecord> &record : records)
        EXPECT_EQ(record->wait(), VERNON_STATUS_OK);
}

TEST_P(RuntimeSubmissionQueue, DestroyedRecordsAreDroppedWithoutWaiting) {
    SubmissionQueue queue;
    queue.configure({1, 1}, context().rhiDevice);
    {
        const std::shared_ptr<SubmissionRecord> record = submitCopy();
        queue.push(record, 64);
        EXPECT_EQ(record->wait(), VERNON_STATUS_OK);
    }
    queue.admit({64, 1});
    const VernonSubmissionTelemetry telemetry = queue.telemetry();
    EXPECT_EQ(telemetry.pending_submissions, 0u);
    EXPECT_EQ(telemetry.pending_transient_bytes, 0u);
    EXPECT_EQ(telemetry.pending_rhi_submissions, 0u);
    EXPECT_EQ(telemetry.pending_limit_waits + telemetry.transient_budget_waits + telemetry.device_quota_waits, 0u);
}

TEST_P(RuntimeSubmissionQueue, MeterChargesTheRuntimeStorageOfItsContextOnItsThread) {
    TransientAllocationMeter meter(context());
    {
        const DeviceBuffer charged(context(), 256);
        ASSERT_TRUE(charged.valid());
    }
    EXPECT_EQ(meter.bytes(), 256u);
    const bool threadAffine =
        GetParam().runtime == VERNON_RUNTIME_OPENGL || GetParam().runtime == VERNON_RUNTIME_OPENGL_ES;
    if (!threadAffine)
        std::thread([&] {
            const DeviceBuffer elsewhere(context(), 512);
            EXPECT_TRUE(elsewhere.valid());
        }).join();
    VernonRuntimeContext *other = vernonRuntimeCreateForRhiDevice(GetParam().runtime, context().rhiDevice, nullptr);
    ASSERT_NE(other, nullptr);
    {
        const DeviceBuffer foreign(*other, 1024);
        EXPECT_TRUE(foreign.valid());
    }
    EXPECT_EQ(vernonRuntimeDestroy(other), VERNON_STATUS_OK);
    EXPECT_EQ(meter.bytes(), 256u);
    {
        TransientAllocationMeter nested(context());
        const DeviceBuffer inner(context(), 64);
        EXPECT_EQ(nested.bytes(), 64u);
    }
    EXPECT_EQ(meter.bytes(), 256u);
}

TEST(RuntimeTransientImageBytes, SumEveryMipOfEveryLayer) {
    VernonRhiImageDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.dimension = VERNON_RHI_IMAGE_2D_ARRAY;
    descriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    descriptor.width = 4;
    descriptor.height = 4;
    descriptor.depth = 1;
    descriptor.mip_levels = 2;
    descriptor.array_layers = 3;
    EXPECT_EQ(vernon::rhi::imageByteSize(descriptor), size_t{(64 + 16) * 3});
    descriptor.format = VERNON_RHI_FORMAT_UNDEFINED;
    EXPECT_FALSE(vernon::rhi::imageByteSize(descriptor).has_value());
}

INSTANTIATE_TEST_SUITE_P(Backends, RuntimeSubmissionQueue, testing::ValuesIn(vernon::tests::rhiBackendCases()),
                         [](const testing::TestParamInfo<vernon::tests::BackendTestRow> &info) {
                             return std::string(info.param.name);
                         });

} // namespace
