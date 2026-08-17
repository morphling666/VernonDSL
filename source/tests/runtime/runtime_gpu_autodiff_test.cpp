#include "VernonRuntime.h"
#include "runtime/autodiff/runtime_direct_autodiff.h"
#include "runtime/autodiff/runtime_gpu_failure_injection.h"
#include "runtime/autodiff/runtime_gpu_replay.h"

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using vernon::runtime::ad::gpu::BatchSummary;
using vernon::runtime::ad::gpu::FailureBoundary;
using vernon::runtime::ad::gpu::Segment;

VernonRhiBackend rhiBackend(VernonRuntimeBackend backend) {
    switch (backend) {
    case VERNON_RUNTIME_CUDA:
        return VERNON_RHI_BACKEND_CUDA;
    case VERNON_RUNTIME_VULKAN:
        return VERNON_RHI_BACKEND_VULKAN;
    case VERNON_RUNTIME_DIRECTX12:
        return VERNON_RHI_BACKEND_DIRECTX12;
    case VERNON_RUNTIME_METAL:
        return VERNON_RHI_BACKEND_METAL;
    default:
        return VERNON_RHI_BACKEND_OPENGL;
    }
}

class OwnedGpuRuntime {
public:
    explicit OwnedGpuRuntime(VernonRuntimeBackend backend) {
        if (backend == VERNON_RUNTIME_OPENGL || backend == VERNON_RUNTIME_OPENGL_ES) {
            device_ = vernonRhiCreateOpenGLDevice(nullptr, backend == VERNON_RUNTIME_OPENGL_ES);
        } else {
            VernonRhiOwnedDeviceDescriptor descriptor{};
            descriptor.struct_size = sizeof(descriptor);
            descriptor.backend = rhiBackend(backend);
            device_ = vernonRhiCreateDevice(&descriptor);
        }
        if (device_.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            runtime_ = vernonRuntimeCreateForRhiDevice(backend, device_);
    }

    ~OwnedGpuRuntime() {
        if (runtime_)
            (void)vernonRuntimeDestroy(runtime_);
        if (device_.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            (void)vernonRhiDestroyDevice(device_);
    }

    OwnedGpuRuntime(const OwnedGpuRuntime &) = delete;
    OwnedGpuRuntime &operator=(const OwnedGpuRuntime &) = delete;

    VernonRuntimeContext *get() const { return runtime_; }
    VernonRhiDevice device() const { return device_; }

private:
    VernonRhiDevice device_{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRuntimeContext *runtime_{};
};

TEST(RuntimeGpuAutodiff, FailureInjectionSelectsBoundaryAndOccurrence) {
    using namespace vernon::runtime::ad::gpu;
    setFailureInjectionForTesting(FailureBoundary::Upload, 2);
    EXPECT_FALSE(injectFailure(FailureBoundary::Allocation));
    EXPECT_FALSE(injectFailure(FailureBoundary::Upload));
    EXPECT_TRUE(injectFailure(FailureBoundary::Upload));
    EXPECT_FALSE(injectFailure(FailureBoundary::Upload));
}

TEST(RuntimeGpuAutodiff, MemoryAccountingSeparatesLogicalObservationalAndBudgetedBytes) {
    vernon::runtime::ad::MemoryAccounting usage;
    usage.logicalPayloadBytes = 10;
    usage.retainedAllocationBytes = 20;
    usage.residentBytes = 30;
    usage.allocatedBytes = 40;
    usage.checkpointBytes = 50;
    usage.observeTemporary(60);
    usage.observeTemporary(5);

    size_t budgeted = 0;
    ASSERT_TRUE(usage.budgetedBytes(budgeted));
    EXPECT_EQ(budgeted, 130u);
    EXPECT_EQ(usage.logicalPayloadBytes, 10u);
    EXPECT_EQ(usage.residentBytes, 30u);
    EXPECT_EQ(usage.allocatedBytes, 40u);
    EXPECT_EQ(usage.peakTemporaryBytes, 60u);
}

TEST(RuntimeGpuAutodiff, ReplayBudgetSelectsLargestBoundedBatch) {
    vernon::runtime::ad::gpu::BatchBudget budget;
    ASSERT_TRUE(vernon::runtime::ad::gpu::planBatchBudget(vernon::runtime::ad::PlanningPolicy::MinRuntime, 100, 1000,
                                                          10, 100, 20, budget));
    EXPECT_EQ(budget.capacity, 3u);
    EXPECT_EQ(budget.tapeBytes, 300u);
    EXPECT_EQ(budget.segmentBytes, 3 * sizeof(Segment));
    EXPECT_EQ(budget.statusBytes, 20u);
    EXPECT_EQ(budget.memory.peakTemporaryBytes, 420 + 6 * sizeof(Segment));
    EXPECT_FALSE(vernon::runtime::ad::gpu::planBatchBudget(vernon::runtime::ad::PlanningPolicy::MinRuntime, 100, 299, 1,
                                                           100, 20, budget));
}

TEST(RuntimeGpuAutodiff, ReplayBudgetShrinksBatchAfterDynamicStrideGrowth) {
    vernon::runtime::ad::gpu::BatchBudget initial;
    vernon::runtime::ad::gpu::BatchBudget grown;
    ASSERT_TRUE(vernon::runtime::ad::gpu::planBatchBudget(vernon::runtime::ad::PlanningPolicy::MinRuntime, 100, 1000,
                                                          10, 100, 20, initial));
    ASSERT_TRUE(vernon::runtime::ad::gpu::planBatchBudget(vernon::runtime::ad::PlanningPolicy::MinRuntime, 100, 1000,
                                                          10, 300, 20, grown));
    EXPECT_EQ(initial.capacity, 3u);
    EXPECT_EQ(grown.capacity, 1u);
    EXPECT_EQ(grown.memory.peakTemporaryBytes, 420 + 2 * sizeof(Segment));
    EXPECT_LE(grown.memory.peakTemporaryBytes, 1000u);
}

TEST(RuntimeGpuAutodiff, ReplayBudgetUsesSharedPlanningPolicy) {
    using vernon::runtime::ad::PlanningPolicy;
    vernon::runtime::ad::gpu::BatchBudget minMemory;
    vernon::runtime::ad::gpu::BatchBudget balanced;
    vernon::runtime::ad::gpu::BatchBudget minRuntime;
    ASSERT_TRUE(
        vernon::runtime::ad::gpu::planBatchBudget(PlanningPolicy::MinMemory, 100, 1000, 10, 100, 20, minMemory));
    ASSERT_TRUE(vernon::runtime::ad::gpu::planBatchBudget(PlanningPolicy::Balanced, 100, 1000, 10, 100, 20, balanced));
    ASSERT_TRUE(
        vernon::runtime::ad::gpu::planBatchBudget(PlanningPolicy::MinRuntime, 100, 1000, 10, 100, 20, minRuntime));
    EXPECT_EQ(minMemory.capacity, 1u);
    EXPECT_EQ(balanced.capacity, 2u);
    EXPECT_EQ(minRuntime.capacity, 3u);
    EXPECT_LT(minMemory.memory.peakTemporaryBytes, balanced.memory.peakTemporaryBytes);
    EXPECT_LT(balanced.memory.peakTemporaryBytes, minRuntime.memory.peakTemporaryBytes);
}

TEST(RuntimeGpuAutodiff, MinMemoryReplayBoundsControlPlaneBatchCount) {
    vernon::runtime::ad::gpu::BatchBudget budget;
    constexpr size_t groupCount = 4096;
    ASSERT_TRUE(vernon::runtime::ad::gpu::planBatchBudget(vernon::runtime::ad::PlanningPolicy::MinMemory, 0,
                                                          1024 * 1024, groupCount, 64, sizeof(BatchSummary), budget));
    EXPECT_EQ(budget.capacity, 64u);
    EXPECT_LE((groupCount + budget.capacity - 1) / budget.capacity, 64u);
}

TEST(RuntimeGpuAutodiff, ReplaySegmentPreservesVirtualCoordinates) {
    Segment segment{};
    ASSERT_TRUE(vernon::runtime::ad::gpu::initializeSegment({3, 2, 2}, {4, 8, 1}, 10, 256, 32, segment));
    EXPECT_EQ(segment.virtualWorkgroup[0], 1u);
    EXPECT_EQ(segment.virtualWorkgroup[1], 1u);
    EXPECT_EQ(segment.virtualWorkgroup[2], 1u);
    EXPECT_EQ(segment.virtualGlobalBase[0], 4u);
    EXPECT_EQ(segment.virtualGlobalBase[1], 8u);
    EXPECT_EQ(segment.virtualGlobalBase[2], 1u);
    EXPECT_EQ(segment.tapeStride, 256u);
    EXPECT_EQ(segment.capacityBytes, 8192u);
}

TEST(RuntimeGpuAutodiff, ReplayBatchSegmentsUseDisjointTapeRanges) {
    Segment first{};
    Segment second{};
    ASSERT_TRUE(vernon::runtime::ad::gpu::initializeBatchSegment({8, 1, 1}, {4, 1, 1}, 6, 0, 256, 4, 2048, first));
    ASSERT_TRUE(vernon::runtime::ad::gpu::initializeBatchSegment({8, 1, 1}, {4, 1, 1}, 7, 1, 256, 4, 2048, second));
    EXPECT_EQ(first.virtualWorkgroup[0], 6u);
    EXPECT_EQ(second.virtualWorkgroup[0], 7u);
    EXPECT_EQ(first.tapeBase, 0u);
    EXPECT_EQ(second.tapeBase, 1024u);
    EXPECT_EQ(first.capacityBytes, 2048u);
    EXPECT_EQ(second.capacityBytes, 2048u);
}

TEST(RuntimeGpuAutodiff, ReplaySchedulerOwnsReverseBatchProgressAndResize) {
    vernon::runtime::ad::gpu::ReverseBatchScheduler scheduler(10, 4);
    ASSERT_TRUE(scheduler.valid());
    EXPECT_EQ(scheduler.current().begin, 6u);
    EXPECT_EQ(scheduler.current().count, 4u);
    scheduler.commit();
    EXPECT_EQ(scheduler.current().begin, 2u);
    EXPECT_EQ(scheduler.current().count, 4u);
    ASSERT_TRUE(scheduler.setCapacity(3));
    EXPECT_EQ(scheduler.current().begin, 3u);
    EXPECT_EQ(scheduler.current().count, 3u);
    scheduler.commit();
    EXPECT_EQ(scheduler.current().begin, 0u);
    EXPECT_EQ(scheduler.current().count, 3u);
    scheduler.commit();
    EXPECT_TRUE(scheduler.empty());
}

TEST(RuntimeGpuAutodiff, DynamicTapeRetryUsesLargestLaneRequirement) {
    const BatchSummary summary{257, 1};
    size_t stride = 0;
    size_t bytes = 0;
    uint32_t status = 0;
    ASSERT_TRUE(vernon::runtime::ad::gpu::requiredTapeBytes(summary, 4, 128, stride, bytes, status));
    EXPECT_EQ(stride, 260u);
    EXPECT_EQ(bytes, 1040u);
    EXPECT_EQ(status, 1u);
}

TEST(RuntimeGpuAutodiff, TapeStrideKeepsAdjacentLaneWordsDisjoint) {
    size_t stride = 0;
    ASSERT_TRUE(vernon::runtime::ad::gpu::normalizeTapeStride(17, stride));
    EXPECT_EQ(stride, 20u);
    ASSERT_TRUE(vernon::runtime::ad::gpu::normalizeTapeStride(0, stride));
    EXPECT_EQ(stride, 16u);
    EXPECT_FALSE(vernon::runtime::ad::gpu::normalizeTapeStride(std::numeric_limits<size_t>::max(), stride));
}

TEST(RuntimeGpuAutodiff, TapeReadbackIsOneFixedBatchSummary) {
    vernon::runtime::ad::gpu::BatchBudget oneGroup;
    vernon::runtime::ad::gpu::BatchBudget manyGroups;
    ASSERT_TRUE(vernon::runtime::ad::gpu::planBatchBudget(vernon::runtime::ad::PlanningPolicy::MinRuntime, 100, 1000, 1,
                                                          100, sizeof(BatchSummary), oneGroup));
    ASSERT_TRUE(vernon::runtime::ad::gpu::planBatchBudget(vernon::runtime::ad::PlanningPolicy::MinRuntime, 100, 1000,
                                                          100, 100, sizeof(BatchSummary), manyGroups));
    EXPECT_EQ(oneGroup.statusBytes, sizeof(BatchSummary));
    EXPECT_EQ(manyGroups.statusBytes, sizeof(BatchSummary));
}

std::string lastError(VernonRuntimeContext *context) {
    const VernonStringView error = vernonRuntimeGetLastError(context);
    return error.data ? std::string(error.data, error.size) : std::string();
}

void runNoTapeVjp(VernonRuntimeBackend backend, const std::filesystem::path &manifestPath) {
    OwnedGpuRuntime owned(backend);
    VernonRuntimeContext *context = owned.get();
    if (!context)
        GTEST_SKIP() << "GPU backend is unavailable";

    std::ifstream input(manifestPath, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const std::string bundleDirectory = manifestPath.parent_path().string();
    VernonPipelineBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    VernonPipelineBundle *bundle =
        vernonRuntimeLoadPipelineBundleWithOptions(context, manifest.data(), manifest.size(), &options);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << lastError(context);

    constexpr uint64_t shape[]{2, 2};
    std::array<float, 4> values{2.0f, 3.0f, 5.0f, 7.0f};
    std::array<float, 4> loss{};
    VernonAdValue inputValues[]{
        {sizeof(VernonAdValue), {"values", 6}, VERNON_DATA_F32, values.data(), sizeof(values), 2, shape},
        {sizeof(VernonAdValue), {"loss", 4}, VERNON_DATA_F32, loss.data(), sizeof(loss), 2, shape},
    };
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, std::size(inputValues), {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), nullptr, 0, {}};
    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(owned.device(), &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderObject providerEncoder{};
    ASSERT_EQ(vernonRuntimeReferenceRhiCommandEncoder(context, encoder, &providerEncoder), VERNON_STATUS_OK);
    VernonPipelineInvocation invalidInvocation{};
    invalidInvocation.struct_size = sizeof(invalidInvocation);
    invalidInvocation.abi_version = VERNON_PIPELINE_VERSION;
    invalidInvocation.argument_count = 1;
    invalidInvocation.compute_grid = {1, 1, 1};
    VernonPullback *invalidPullback = nullptr;
    EXPECT_EQ(vernonAdPipelineEncodeForward(providerEncoder, pipeline, &invalidInvocation, &inputs, &invalidPullback),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(invalidPullback, nullptr);
    EXPECT_EQ(vernonRhiDeviceDestroyCommandEncoder(owned.device(), encoder), VERNON_RHI_STATUS_OK);

    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    EXPECT_EQ(loss, (std::array<float, 4>{4.0f, 9.0f, 25.0f, 49.0f}));

    values.fill(100.0f);
    std::array<float, 4> cotangent{1.0f, 1.0f, 1.0f, 1.0f};
    std::array<float, 4> gradient{};
    VernonAdValue cotangentValue{
        sizeof(VernonAdValue), {"loss", 4}, VERNON_DATA_F32, cotangent.data(), sizeof(cotangent), 2, shape};
    VernonAdValueSet cotangents{sizeof(VernonAdValueSet), &cotangentValue, 1, {}};
    VernonAdValue gradientValue{
        sizeof(VernonAdValue), {"values", 6}, VERNON_DATA_F32, gradient.data(), sizeof(gradient), 2, shape};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradientValue, 1, {}};

    constexpr uint64_t exactTemporaryBytes = sizeof(cotangent) + 2 * sizeof(gradient) + 3 * sizeof(uint32_t);
    VernonPullbackApplyOptions applyOptions{
        sizeof(VernonPullbackApplyOptions), VERNON_PULLBACK_APPLY_OPTIONS_VERSION, exactTemporaryBytes - 1, 0, {}};
    gradient.fill(-23.0f);
    vernon::runtime::ad::gpu::setFailureInjectionForTesting(vernon::runtime::ad::gpu::FailureBoundary::Allocation);
    EXPECT_NE(vernonPullbackApplyWithOptions(pullback, &cotangents, &gradients, &applyOptions), VERNON_STATUS_OK);
    EXPECT_EQ(gradient, (std::array<float, 4>{-23.0f, -23.0f, -23.0f, -23.0f}));
    applyOptions.maximum_temporary_bytes = exactTemporaryBytes;
    EXPECT_NE(vernonPullbackApplyWithOptions(pullback, &cotangents, &gradients, &applyOptions), VERNON_STATUS_OK)
        << "the rejected under-budget apply must not consume the pending allocation failure";
    vernon::runtime::ad::gpu::clearFailureInjectionForTesting();
    ASSERT_EQ(vernonPullbackApplyWithOptions(pullback, &cotangents, &gradients, &applyOptions), VERNON_STATUS_OK)
        << lastError(context);
    EXPECT_EQ(gradient, (std::array<float, 4>{4.0f, 6.0f, 10.0f, 14.0f}));

    gradient.fill(0.0f);
    ASSERT_EQ(vernonPullbackApply(pullback, &cotangents, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_EQ(gradient, (std::array<float, 4>{4.0f, 6.0f, 10.0f, 14.0f}));
    const vernon::runtime::AutodiffPullbackControlPlaneUsage control =
        vernon::runtime::autodiffPullbackControlPlaneUsage(pullback);
    EXPECT_EQ(control.submissions, 1u);
    EXPECT_EQ(control.waits, 1u);
    EXPECT_EQ(control.readbacks, 1u);
    EXPECT_EQ(control.temporaryAllocationBytes, exactTemporaryBytes);
    EXPECT_GT(control.deviceWaitNanoseconds, 0u);

    gradient.fill(0.0f);
    ASSERT_EQ(vernonPullbackApply(pullback, &cotangents, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_EQ(gradient, (std::array<float, 4>{4.0f, 6.0f, 10.0f, 14.0f}));

    vernonPullbackDestroy(pullback);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(bundle);
}

void runNoTapeFailureInjection(VernonRuntimeBackend backend, const std::filesystem::path &manifestPath) {
    using namespace vernon::runtime::ad::gpu;
    OwnedGpuRuntime owned(backend);
    VernonRuntimeContext *context = owned.get();
    if (!context)
        GTEST_SKIP() << "GPU backend is unavailable";

    std::ifstream input(manifestPath, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const std::string bundleDirectory = manifestPath.parent_path().string();
    VernonPipelineBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    VernonPipelineBundle *bundle =
        vernonRuntimeLoadPipelineBundleWithOptions(context, manifest.data(), manifest.size(), &options);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << lastError(context);

    constexpr uint64_t shape[]{2, 2};
    const std::array<float, 4> original{2.0f, 3.0f, 5.0f, 7.0f};
    std::array<float, 4> values = original;
    std::array<float, 4> loss{};
    VernonAdValue inputValues[]{
        {sizeof(VernonAdValue), {"values", 6}, VERNON_DATA_F32, values.data(), sizeof(values), 2, shape},
        {sizeof(VernonAdValue), {"loss", 4}, VERNON_DATA_F32, loss.data(), sizeof(loss), 2, shape},
    };
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, std::size(inputValues), {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), nullptr, 0, {}};

    constexpr std::array forwardBoundaries{
        FailureBoundary::Allocation, FailureBoundary::Upload,   FailureBoundary::Submit,
        FailureBoundary::Wait,       FailureBoundary::Download, FailureBoundary::Publication,
    };
    for (FailureBoundary boundary : forwardBoundaries) {
        values = original;
        loss.fill(-91.0f);
        VernonPullback *failedPullback = nullptr;
        setFailureInjectionForTesting(boundary);
        EXPECT_NE(vernonAdPipelineForward(pipeline, {1, 1, 1}, &inputs, &outputs, &failedPullback), VERNON_STATUS_OK)
            << static_cast<int>(boundary);
        clearFailureInjectionForTesting();
        EXPECT_EQ(failedPullback, nullptr);
        EXPECT_EQ(values, original);
        EXPECT_EQ(loss, (std::array<float, 4>{-91.0f, -91.0f, -91.0f, -91.0f}));
    }

    values = original;
    loss.fill(0.0f);
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);

    std::array<float, 4> cotangent{1.0f, 1.0f, 1.0f, 1.0f};
    std::array<float, 4> gradient{};
    VernonAdValue cotangentValue{
        sizeof(VernonAdValue), {"loss", 4}, VERNON_DATA_F32, cotangent.data(), sizeof(cotangent), 2, shape};
    VernonAdValueSet cotangents{sizeof(VernonAdValueSet), &cotangentValue, 1, {}};
    VernonAdValue gradientValue{
        sizeof(VernonAdValue), {"values", 6}, VERNON_DATA_F32, gradient.data(), sizeof(gradient), 2, shape};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradientValue, 1, {}};
    for (FailureBoundary boundary : forwardBoundaries) {
        gradient.fill(-37.0f);
        setFailureInjectionForTesting(boundary);
        EXPECT_NE(vernonPullbackApply(pullback, &cotangents, &gradients), VERNON_STATUS_OK)
            << static_cast<int>(boundary);
        clearFailureInjectionForTesting();
        EXPECT_EQ(gradient, (std::array<float, 4>{-37.0f, -37.0f, -37.0f, -37.0f}));
    }
    gradient.fill(0.0f);
    ASSERT_EQ(vernonPullbackApply(pullback, &cotangents, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_EQ(gradient, (std::array<float, 4>{4.0f, 6.0f, 10.0f, 14.0f}));

    vernonPullbackDestroy(pullback);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(bundle);
}

void runCapturedTapeVjp(VernonRuntimeBackend backend, const std::filesystem::path &manifestPath, bool dynamic) {
    OwnedGpuRuntime owned(backend);
    VernonRuntimeContext *context = owned.get();
    if (!context)
        GTEST_SKIP() << "GPU backend is unavailable";

    std::ifstream input(manifestPath, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const std::string bundleDirectory = manifestPath.parent_path().string();
    VernonPipelineBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    VernonPipelineBundle *bundle =
        vernonRuntimeLoadPipelineBundleWithOptions(context, manifest.data(), manifest.size(), &options);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << lastError(context);

    constexpr size_t laneCount = 8;
    constexpr uint64_t outputShape[]{laneCount};
    float x = dynamic ? 2.0f : 3.0f;
    float y = 2.0f;
    int32_t count = 2;
    std::array<float, laneCount> output{};
    VernonAdValue inputValues[3]{
        {sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &x, sizeof(x), 0, nullptr},
        {sizeof(VernonAdValue),
         {dynamic ? "count" : "y", dynamic ? 5u : 1u},
         dynamic ? VERNON_DATA_I32 : VERNON_DATA_F32,
         dynamic ? static_cast<void *>(&count) : static_cast<void *>(&y),
         dynamic ? sizeof(count) : sizeof(y),
         0,
         nullptr},
        {sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, output.data(), sizeof(output), 1, outputShape},
    };
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, std::size(inputValues), {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), nullptr, 0, {}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(pipeline, {2, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    for (size_t lane = 0; lane < laneCount; ++lane)
        EXPECT_FLOAT_EQ(output[lane], dynamic ? 11.0f : 2.5f);

    x = 100.0f;
    std::array<float, laneCount * laneCount> seeds{};
    for (size_t lane = 0; lane < laneCount; ++lane)
        seeds[lane * laneCount + lane] = 1.0f;
    constexpr uint64_t seedShape[]{1, 1, laneCount, laneCount};
    VernonAdValue seed{sizeof(VernonAdValue),
                       {"output", 6},
                       VERNON_DATA_F32,
                       seeds.data(),
                       sizeof(seeds),
                       static_cast<uint32_t>(std::size(seedShape)),
                       seedShape};
    VernonAdValueSet cotangents{sizeof(VernonAdValueSet), &seed, 1, {}};
    std::array<float, 2> gradientStorage{};
    VernonAdValue gradientValues[2]{
        {sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &gradientStorage[0], sizeof(float), 0, nullptr},
        {sizeof(VernonAdValue), {"y", 1}, VERNON_DATA_F32, &gradientStorage[1], sizeof(float), 0, nullptr},
    };
    if (!dynamic)
        std::swap(gradientValues[0], gradientValues[1]);
    VernonAdValueSet gradients{
        sizeof(VernonAdValueSet), gradientValues, dynamic ? size_t{1} : std::size(gradientValues), {}};
    if (dynamic) {
        gradientStorage.fill(-37.0f);
        vernon::runtime::ad::gpu::setFailureInjectionForTesting(FailureBoundary::Resize);
        EXPECT_NE(vernonPullbackApply(pullback, &cotangents, &gradients), VERNON_STATUS_OK);
        vernon::runtime::ad::gpu::clearFailureInjectionForTesting();
        EXPECT_EQ(gradientStorage, (std::array<float, 2>{-37.0f, -37.0f}));
    }
    ASSERT_EQ(vernonPullbackApply(pullback, &cotangents, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_NEAR(gradientStorage[0], dynamic ? 104.0f : 24.0f, 1e-4f);
    if (!dynamic)
        EXPECT_NEAR(gradientStorage[1], -26.0f, 1e-4f);
    std::array<float, laneCount> sharedSeeds{};
    sharedSeeds.fill(1.0f);
    VernonAdValue sharedSeed{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, sharedSeeds.data(), sizeof(sharedSeeds), 1, outputShape};
    VernonAdValueSet sharedCotangents{sizeof(VernonAdValueSet), &sharedSeed, 1, {}};
    gradientStorage.fill(-19.0f);
    ASSERT_EQ(vernonPullbackApply(pullback, &sharedCotangents, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_NEAR(gradientStorage[0], dynamic ? 104.0f : 24.0f, 1e-4f);
    if (!dynamic)
        EXPECT_NEAR(gradientStorage[1], -26.0f, 1e-4f);
    const vernon::runtime::AutodiffPullbackMemoryUsage memory = vernon::runtime::autodiffPullbackMemoryUsage(pullback);
    EXPECT_EQ(memory.logicalResidualBytes, 0u);
    EXPECT_EQ(memory.residentBytes, 0u);
    EXPECT_EQ(memory.allocatedBytes, 0u);
    EXPECT_GT(memory.peakTemporaryBytes, 0u);

    constexpr std::array transactionalBoundaries{
        FailureBoundary::Allocation, FailureBoundary::Upload, FailureBoundary::Copy,     FailureBoundary::Encode,
        FailureBoundary::Submit,     FailureBoundary::Wait,   FailureBoundary::Download, FailureBoundary::Publication,
    };
    for (FailureBoundary boundary : transactionalBoundaries) {
        gradientStorage.fill(-41.0f);
        vernon::runtime::ad::gpu::setFailureInjectionForTesting(boundary);
        EXPECT_NE(vernonPullbackApply(pullback, &cotangents, &gradients), VERNON_STATUS_OK)
            << static_cast<int>(boundary);
        vernon::runtime::ad::gpu::clearFailureInjectionForTesting();
        EXPECT_EQ(gradientStorage, (std::array<float, 2>{-41.0f, -41.0f}));
    }
    gradientStorage.fill(0.0f);
    ASSERT_EQ(vernonPullbackApply(pullback, &cotangents, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_NEAR(gradientStorage[0], dynamic ? 104.0f : 24.0f, 1e-4f);

    vernonPullbackDestroy(pullback);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(bundle);
}

void runNonPowerOfTwoReductionVjp(VernonRuntimeBackend backend, const std::filesystem::path &manifestPath) {
    OwnedGpuRuntime owned(backend);
    VernonRuntimeContext *context = owned.get();
    if (!context)
        GTEST_SKIP() << "GPU backend is unavailable";

    std::ifstream input(manifestPath, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const std::string bundleDirectory = manifestPath.parent_path().string();
    VernonPipelineBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    VernonPipelineBundle *bundle =
        vernonRuntimeLoadPipelineBundleWithOptions(context, manifest.data(), manifest.size(), &options);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << lastError(context);

    constexpr size_t laneCount = 192;
    constexpr uint64_t shape[]{laneCount};
    float scale = 2.0f;
    std::array<float, laneCount> values{};
    std::array<float, laneCount> carriedLoss{};
    std::array<float, laneCount> sharedLoss{};
    for (size_t lane = 0; lane < laneCount; ++lane)
        values[lane] = static_cast<float>(lane + 1);
    VernonAdValue inputValues[]{
        {sizeof(VernonAdValue), {"scale", 5}, VERNON_DATA_F32, &scale, sizeof(scale), 0, nullptr},
        {sizeof(VernonAdValue), {"values", 6}, VERNON_DATA_F32, values.data(), sizeof(values), 1, shape},
        {sizeof(VernonAdValue),
         {"carried_loss", 12},
         VERNON_DATA_F32,
         carriedLoss.data(),
         sizeof(carriedLoss),
         1,
         shape},
        {sizeof(VernonAdValue), {"shared_loss", 11}, VERNON_DATA_F32, sharedLoss.data(), sizeof(sharedLoss), 1, shape},
    };
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, std::size(inputValues), {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), nullptr, 0, {}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(pipeline, {2, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    for (size_t lane = 0; lane < laneCount; ++lane) {
        EXPECT_FLOAT_EQ(carriedLoss[lane], 2.0f * values[lane]);
        EXPECT_FLOAT_EQ(sharedLoss[lane], 2.0f * values[lane] * values[lane]);
    }

    constexpr uint64_t carriedShape[]{1, 1, laneCount, laneCount};
    std::vector<float> carriedSeeds(laneCount * laneCount);
    for (size_t lane = 0; lane < laneCount; ++lane)
        carriedSeeds[lane * laneCount + lane] = 1.0f;
    std::array<float, laneCount> sharedSeeds{};
    sharedSeeds.fill(1.0f);
    VernonAdValue seeds[]{
        {sizeof(VernonAdValue),
         {"carried_loss", 12},
         VERNON_DATA_F32,
         carriedSeeds.data(),
         carriedSeeds.size() * sizeof(float),
         static_cast<uint32_t>(std::size(carriedShape)),
         carriedShape},
        {sizeof(VernonAdValue),
         {"shared_loss", 11},
         VERNON_DATA_F32,
         sharedSeeds.data(),
         sizeof(sharedSeeds),
         1,
         shape},
    };
    VernonAdValueSet cotangents{sizeof(VernonAdValueSet), seeds, std::size(seeds), {}};
    float gradient = 0.0f;
    VernonAdValue gradientValue{
        sizeof(VernonAdValue), {"scale", 5}, VERNON_DATA_F32, &gradient, sizeof(gradient), 0, nullptr};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradientValue, 1, {}};
    ASSERT_EQ(vernonPullbackApply(pullback, &cotangents, &gradients), VERNON_STATUS_OK) << lastError(context);
    float expected = 0.0f;
    for (float value : values)
        expected += value + value * value;
    EXPECT_NEAR(gradient, expected, 0.5f);

    vernonPullbackDestroy(pullback);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(bundle);
}

#if defined(VERNON_GPU_AUTODIFF_VULKAN_MANIFEST)
TEST(RuntimeGpuAutodiff, VulkanNoTapePullbackStaysOnDevice) {
    runNoTapeVjp(VERNON_RUNTIME_VULKAN, VERNON_GPU_AUTODIFF_VULKAN_MANIFEST);
}
TEST(RuntimeGpuAutodiff, VulkanNoTapeFailuresAreTransactional) {
    runNoTapeFailureInjection(VERNON_RUNTIME_VULKAN, VERNON_GPU_AUTODIFF_VULKAN_MANIFEST);
}
#if defined(VERNON_GPU_AUTODIFF_VULKAN_NON_POWER_OF_TWO_MANIFEST)
TEST(RuntimeGpuAutodiff, VulkanNonPowerOfTwoWorkgroupReductionIsNumericallyCorrect) {
    runNonPowerOfTwoReductionVjp(VERNON_RUNTIME_VULKAN, VERNON_GPU_AUTODIFF_VULKAN_NON_POWER_OF_TWO_MANIFEST);
}
#endif
#if defined(VERNON_GPU_AUTODIFF_VULKAN_STATIC_MANIFEST)
TEST(RuntimeGpuAutodiff, VulkanStaticTapeUsesBoundedReplay) {
    runCapturedTapeVjp(VERNON_RUNTIME_VULKAN, VERNON_GPU_AUTODIFF_VULKAN_STATIC_MANIFEST, false);
}
TEST(RuntimeGpuAutodiff, VulkanDynamicTapeUsesBoundedReplay) {
    runCapturedTapeVjp(VERNON_RUNTIME_VULKAN, VERNON_GPU_AUTODIFF_VULKAN_DYNAMIC_MANIFEST, true);
}
#endif
#endif

#if defined(VERNON_GPU_AUTODIFF_CUDA_MANIFEST)
TEST(RuntimeGpuAutodiff, CudaNoTapePullbackStaysOnDevice) {
    runNoTapeVjp(VERNON_RUNTIME_CUDA, VERNON_GPU_AUTODIFF_CUDA_MANIFEST);
}
#if defined(VERNON_GPU_AUTODIFF_CUDA_NON_POWER_OF_TWO_MANIFEST)
TEST(RuntimeGpuAutodiff, CudaNonPowerOfTwoWorkgroupReductionIsNumericallyCorrect) {
    runNonPowerOfTwoReductionVjp(VERNON_RUNTIME_CUDA, VERNON_GPU_AUTODIFF_CUDA_NON_POWER_OF_TWO_MANIFEST);
}
#endif
#if defined(VERNON_GPU_AUTODIFF_CUDA_STATIC_MANIFEST)
TEST(RuntimeGpuAutodiff, CudaStaticTapeUsesBoundedReplay) {
    runCapturedTapeVjp(VERNON_RUNTIME_CUDA, VERNON_GPU_AUTODIFF_CUDA_STATIC_MANIFEST, false);
}
TEST(RuntimeGpuAutodiff, CudaDynamicTapeUsesBoundedReplay) {
    runCapturedTapeVjp(VERNON_RUNTIME_CUDA, VERNON_GPU_AUTODIFF_CUDA_DYNAMIC_MANIFEST, true);
}
#endif
#endif

#if defined(VERNON_GPU_AUTODIFF_DIRECTX_MANIFEST)
TEST(RuntimeGpuAutodiff, DirectXNoTapePullbackStaysOnDevice) {
    runNoTapeVjp(VERNON_RUNTIME_DIRECTX12, VERNON_GPU_AUTODIFF_DIRECTX_MANIFEST);
}
#if defined(VERNON_GPU_AUTODIFF_DIRECTX_NON_POWER_OF_TWO_MANIFEST)
TEST(RuntimeGpuAutodiff, DirectXNonPowerOfTwoWorkgroupReductionIsNumericallyCorrect) {
    runNonPowerOfTwoReductionVjp(VERNON_RUNTIME_DIRECTX12, VERNON_GPU_AUTODIFF_DIRECTX_NON_POWER_OF_TWO_MANIFEST);
}
#endif
#if defined(VERNON_GPU_AUTODIFF_DIRECTX_STATIC_MANIFEST)
TEST(RuntimeGpuAutodiff, DirectXStaticTapeUsesBoundedReplay) {
    runCapturedTapeVjp(VERNON_RUNTIME_DIRECTX12, VERNON_GPU_AUTODIFF_DIRECTX_STATIC_MANIFEST, false);
}
TEST(RuntimeGpuAutodiff, DirectXDynamicTapeUsesBoundedReplay) {
    runCapturedTapeVjp(VERNON_RUNTIME_DIRECTX12, VERNON_GPU_AUTODIFF_DIRECTX_DYNAMIC_MANIFEST, true);
}
#endif
#endif

#if defined(VERNON_GPU_AUTODIFF_METAL_MANIFEST)
TEST(RuntimeGpuAutodiff, MetalNoTapePullbackStaysOnDevice) {
    runNoTapeVjp(VERNON_RUNTIME_METAL, VERNON_GPU_AUTODIFF_METAL_MANIFEST);
}
TEST(RuntimeGpuAutodiff, MetalNoTapeFailuresAreTransactional) {
    runNoTapeFailureInjection(VERNON_RUNTIME_METAL, VERNON_GPU_AUTODIFF_METAL_MANIFEST);
}
#if defined(VERNON_GPU_AUTODIFF_METAL_NON_POWER_OF_TWO_MANIFEST)
TEST(RuntimeGpuAutodiff, MetalNonPowerOfTwoWorkgroupReductionIsNumericallyCorrect) {
    runNonPowerOfTwoReductionVjp(VERNON_RUNTIME_METAL, VERNON_GPU_AUTODIFF_METAL_NON_POWER_OF_TWO_MANIFEST);
}
#endif
#if defined(VERNON_GPU_AUTODIFF_METAL_STATIC_MANIFEST)
TEST(RuntimeGpuAutodiff, MetalStaticTapeUsesBoundedReplay) {
    runCapturedTapeVjp(VERNON_RUNTIME_METAL, VERNON_GPU_AUTODIFF_METAL_STATIC_MANIFEST, false);
}
TEST(RuntimeGpuAutodiff, MetalDynamicTapeUsesBoundedReplay) {
    runCapturedTapeVjp(VERNON_RUNTIME_METAL, VERNON_GPU_AUTODIFF_METAL_DYNAMIC_MANIFEST, true);
}
#endif
#endif

#if defined(VERNON_GPU_AUTODIFF_OPENGL_MANIFEST)
TEST(RuntimeGpuAutodiff, OpenGLNoTapePullbackStaysOnDevice) {
    runNoTapeVjp(VERNON_RUNTIME_OPENGL, VERNON_GPU_AUTODIFF_OPENGL_MANIFEST);
}
TEST(RuntimeGpuAutodiff, OpenGLNoTapeFailuresAreTransactional) {
    runNoTapeFailureInjection(VERNON_RUNTIME_OPENGL, VERNON_GPU_AUTODIFF_OPENGL_MANIFEST);
}
#if defined(VERNON_GPU_AUTODIFF_OPENGL_NON_POWER_OF_TWO_MANIFEST)
TEST(RuntimeGpuAutodiff, OpenGLNonPowerOfTwoWorkgroupReductionIsNumericallyCorrect) {
    runNonPowerOfTwoReductionVjp(VERNON_RUNTIME_OPENGL, VERNON_GPU_AUTODIFF_OPENGL_NON_POWER_OF_TWO_MANIFEST);
}
#endif
#if defined(VERNON_GPU_AUTODIFF_OPENGL_STATIC_MANIFEST)
TEST(RuntimeGpuAutodiff, OpenGLStaticTapeUsesBoundedReplay) {
    runCapturedTapeVjp(VERNON_RUNTIME_OPENGL, VERNON_GPU_AUTODIFF_OPENGL_STATIC_MANIFEST, false);
}
TEST(RuntimeGpuAutodiff, OpenGLDynamicTapeUsesBoundedReplay) {
    runCapturedTapeVjp(VERNON_RUNTIME_OPENGL, VERNON_GPU_AUTODIFF_OPENGL_DYNAMIC_MANIFEST, true);
}
#endif
#endif

} // namespace
