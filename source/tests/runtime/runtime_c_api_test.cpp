#include "VernonCpuWorkgroupABI.h"
#include "runtime_rhi_test_utils.h"
#include "vernon-c/Runtime.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static VernonStatus fill_grid(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1)
        return VERNON_STATUS_INVALID_ARGUMENT;
    VernonCpuRangeV1 *range = (VernonCpuRangeV1 *)(uintptr_t)invocation->arguments;
    if (!range || range->struct_size != sizeof(*range))
        return VERNON_STATUS_INVALID_ARGUMENT;
    uintptr_t address = 0;
    memcpy(&address, range->arguments, sizeof(address));
    float *values = (float *)address;
    for (size_t lane = range->lane_begin; lane < range->lane_end; ++lane) {
        const uint32_t local_x = (uint32_t)(lane % range->workgroup[0]);
        const uint32_t local_y = (uint32_t)((lane / range->workgroup[0]) % range->workgroup[1]);
        const uint32_t local_z = (uint32_t)(lane / ((size_t)range->workgroup[0] * range->workgroup[1]));
        const uint32_t x = range->group[0] * range->workgroup[0] + local_x;
        const uint32_t y = range->group[1] * range->workgroup[1] + local_y;
        const uint32_t z = range->group[2] * range->workgroup[2] + local_z;
        values[z * 8 + y * 4 + x] = (float)(x + 10 * y + 100 * z);
    }
    return VERNON_STATUS_OK;
}

TEST(RuntimeCApi, PullbackOpaqueHandleRejectsInvalidCalls) {
    EXPECT_EQ(vernonProgramPullbackApply(nullptr, nullptr, 0), VERNON_STATUS_INVALID_ARGUMENT);
    vernonProgramPullbackDestroy(nullptr);
}

TEST(RuntimeCApi, CpuComputePipelineAndBundleBehavior) {
    static const char static_symbol[] = "__vernon_cpu_test_fill";
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntry({static_symbol, sizeof(static_symbol) - 1}, fill_grid) ==
                VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntry({static_symbol, sizeof(static_symbol) - 1}, fill_grid) ==
                VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntry({nullptr, 0}, fill_grid) == VERNON_STATUS_INVALID_ARGUMENT);
    static const char reflection[] =
        "{" VERNON_JSON_VERSION_FIELDS ",\"entries\":[{\"name\":\"fill\","
        "\"physical_layouts\":{\"host_value\":{\"profile\":\"host_value\","
        "\"packed_arguments_size\":20}},"
        "\"workgroup_size\":[2,2,1],"
        "\"dispatch_contract\":{\"unit_grid_axes\":[],\"requires_unit_workgroup\":false},\"arguments\":["
        "{\"kind\":\"tensor\",\"dtype\":\"f32\",\"access\":\"read_write\",\"shape\":[16],"
        "\"element_layout\":{\"logical_type\":\"f32\",\"byte_size\":4,"
        "\"alignment\":4,\"layout_hash\":"
        "\"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b\","
        "\"leaves\":[{\"path\":[],\"dtype\":\"f32\",\"byte_offset\":0,"
        "\"scalar_count\":1}]},"
        "\"physical_layouts\":{\"host_value\":{\"profile\":\"host_value\","
        "\"kind\":\"resource_binding\",\"resource_kind\":\"tensor_view_descriptor\","
        "\"size\":8,\"alignment\":8,\"frame_offset\":0}}},"
        "{\"kind\":\"builtin\",\"builtin\":\"global_invocation_id\","
        "\"physical_layouts\":{\"host_value\":{\"profile\":\"host_value\",\"kind\":\"cpu_call\","
        "\"frame_offset\":8,\"root\":{\"kind\":\"array\",\"offset\":0,\"size\":12,\"alignment\":4,"
        "\"shape\":[3],\"byte_strides\":[4],\"children\":[{\"kind\":\"scalar\","
        "\"representation\":\"i32\",\"offset\":0,\"size\":4,\"alignment\":4}]}}},"
        "\"index\":1}]}]}";
    VernonRuntimeCapabilities capabilities = vernonRuntimeGetCapabilities(VERNON_RUNTIME_CPU);
    ASSERT_TRUE(capabilities.available && capabilities.supports_compute);
    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_TRUE(runtime);
    static const char bad_reflection[] = "{\"compiler_contract_version\":" VERNON_COMPILER_CONTRACT_VERSION_STRING
                                         ",\"program_version\":0,\"entries\":[]}";
    ASSERT_TRUE(!vernonRuntimeLoadCpuEntry(runtime, fill_grid, bad_reflection, sizeof(bad_reflection) - 1, "fill", 4));
    VernonStageExecutable *pipeline =
        vernonRuntimeLoadCpuEntry(runtime, fill_grid, reflection, sizeof(reflection) - 1, "fill", 4);
    ASSERT_TRUE(pipeline);
    uint64_t shape[] = {16};
    int64_t strides[] = {4};
    float values[16] = {0};
    VernonProgramArgument argument = {};
    argument.slot = 0;
    argument.kind = VERNON_PROGRAM_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_HOST;
    argument.tensor.host_data = values;
    argument.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    argument.tensor.access = VERNON_ACCESS_READ_WRITE;
    argument.tensor.rank = 1;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = 16 * sizeof(float);
    VernonLaunchSize grid = {2, 1, 2};
    VernonStageInvocationDescriptor invocation = {};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PROGRAM_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    invocation.compute_grid = grid;
    VernonSubmission *submission{};
    ASSERT_EQ(vernonRuntimeStageSubmit(pipeline, &invocation, &submission), VERNON_STATUS_OK);
    ASSERT_NE(submission, nullptr);
    VernonSubmissionState submissionState{};
    ASSERT_EQ(vernonSubmissionGetState(submission, &submissionState), VERNON_STATUS_OK);
    EXPECT_EQ(submissionState, VERNON_SUBMISSION_SUCCEEDED);
    ASSERT_EQ(vernonSubmissionWait(submission), VERNON_STATUS_OK);
    vernonSubmissionDestroy(submission);
    ASSERT_TRUE(values[0] == 0.0f && values[3] == 3.0f);
    ASSERT_TRUE(values[4] == 10.0f && values[15] == 113.0f);

    nlohmann::json constrainedJson = nlohmann::json::parse(reflection);
    constrainedJson["entries"][0]["workgroup_size"] = {2, 1, 1};
    constrainedJson["entries"][0]["dispatch_contract"] = {{"unit_grid_axes", {1, 2}},
                                                          {"requires_unit_workgroup", false}};
    const std::string constrainedReflection = constrainedJson.dump();
    VernonStageExecutable *constrainedPipeline = vernonRuntimeLoadCpuEntry(
        runtime, fill_grid, constrainedReflection.data(), constrainedReflection.size(), "fill", 4);
    ASSERT_NE(constrainedPipeline, nullptr);
    EXPECT_EQ(vernon::tests::completeSubmission(constrainedPipeline, &invocation), VERNON_STATUS_INVALID_ARGUMENT);
    invocation.compute_grid = {2, 1, 1};
    EXPECT_EQ(vernon::tests::completeSubmission(constrainedPipeline, &invocation), VERNON_STATUS_OK);
    vernonRuntimeStageExecutableDestroy(constrainedPipeline);

    constrainedJson["entries"][0]["workgroup_size"] = {1, 1, 1};
    constrainedJson["entries"][0]["dispatch_contract"] = {{"unit_grid_axes", {0, 1, 2}},
                                                          {"requires_unit_workgroup", true}};
    const std::string singleInvocationReflection = constrainedJson.dump();
    VernonStageExecutable *singleInvocationPipeline = vernonRuntimeLoadCpuEntry(
        runtime, fill_grid, singleInvocationReflection.data(), singleInvocationReflection.size(), "fill", 4);
    ASSERT_NE(singleInvocationPipeline, nullptr);
    EXPECT_EQ(vernon::tests::completeSubmission(singleInvocationPipeline, &invocation), VERNON_STATUS_INVALID_ARGUMENT);
    invocation.compute_grid = {1, 1, 1};
    EXPECT_EQ(vernon::tests::completeSubmission(singleInvocationPipeline, &invocation), VERNON_STATUS_OK);
    vernonRuntimeStageExecutableDestroy(singleInvocationPipeline);

    ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_INVALID_ARGUMENT);
    vernonRuntimeStageExecutableDestroy(pipeline);

    ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
}
