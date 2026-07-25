#include "vernon-c/Runtime.h"

#include <gtest/gtest.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef VERNON_CPU_BUNDLE_PATH
#error VERNON_CPU_BUNDLE_PATH must name the CPU bundle test fixture
#endif

static VernonStatus fill_grid(const VernonCpuInvocation *invocation) {
    uintptr_t address = 0;
    uint32_t gid[3] = {0, 0, 0};
    memcpy(&address, invocation->arguments, sizeof(address));
    memcpy(gid, (const unsigned char *)invocation->arguments + 8, sizeof(gid));
    float *values = (float *)address;
    values[gid[2] * 6 + gid[1] * 3 + gid[0]] = (float)(gid[0] + 10 * gid[1] + 100 * gid[2]);
    return VERNON_STATUS_OK;
}

TEST(RuntimeCApi, CpuComputePipelineAndBundleBehavior) {
    static const char static_symbol[] = "__vernon_cpu_test_fill";
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntry({static_symbol, sizeof(static_symbol) - 1}, fill_grid) ==
                VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntry({static_symbol, sizeof(static_symbol) - 1}, fill_grid) ==
                VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntry({nullptr, 0}, fill_grid) == VERNON_STATUS_INVALID_ARGUMENT);
    static const char reflection[] = "{\"gpu_launch_abi_version\":1,\"entries\":[{\"name\":\"fill\","
                                     "\"cpu_arguments_size\":20,\"workgroup_size\":[2,2,1],\"arguments\":["
                                     "{\"kind\":\"tensor\",\"dtype\":\"f32\",\"shape\":[12],"
                                     "\"tensor_bytes\":48,\"tensor_element_size\":4,"
                                     "\"alignment\":4,\"cpu_offset\":0,\"cpu_size\":8},"
                                     "{\"kind\":\"builtin\",\"builtin\":\"global_invocation_id\","
                                     "\"cpu_offset\":8,\"cpu_size\":12}]}]}";
    VernonRuntimeCapabilities capabilities = vernonRuntimeGetCapabilities(VERNON_RUNTIME_CPU);
    ASSERT_TRUE(capabilities.available && capabilities.supports_compute);
    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_TRUE(runtime);
    VernonDeviceBuffer *buffer = vernonRuntimeBufferAllocate(runtime, 12 * sizeof(float), 4);
    ASSERT_TRUE(buffer);
    static const char bad_reflection[] = "{\"gpu_launch_abi_version\":2,\"entries\":[]}";
    ASSERT_TRUE(!vernonRuntimeLoadCpuEntry(runtime, fill_grid, bad_reflection, sizeof(bad_reflection) - 1, "fill", 4));
    VernonLoadedPipeline *pipeline =
        vernonRuntimeLoadCpuEntry(runtime, fill_grid, reflection, sizeof(reflection) - 1, "fill", 4);
    ASSERT_TRUE(pipeline);
    uint64_t shape[] = {12};
    int64_t strides[] = {4};
    VernonPipelineArgument argument = {};
    argument.slot = 0;
    argument.kind = VERNON_PIPELINE_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_DEVICE;
    argument.tensor.buffer = buffer;
    argument.tensor.dtype = VERNON_DATA_F32;
    argument.tensor.access = VERNON_ACCESS_READ_WRITE;
    argument.tensor.rank = 1;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = 12 * sizeof(float);
    VernonLaunchSize grid = {3, 2, 2};
    VernonPipelineInvocation invocation = {};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    invocation.compute_grid = grid;
    ASSERT_TRUE(vernonRuntimePipelineInvoke(pipeline, &invocation) == VERNON_STATUS_OK);
    float values[12] = {0};
    ASSERT_TRUE(vernonRuntimeCopyToHost(buffer, 0, values, sizeof(values)) == VERNON_STATUS_OK);
    ASSERT_TRUE(values[0] == 0.0f && values[2] == 2.0f);
    ASSERT_TRUE(values[3] == 10.0f && values[11] == 112.0f);
    ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_INVALID_ARGUMENT);
    vernonRuntimeLoadedPipelineDestroy(pipeline);

    memset(values, 0, sizeof(values));
    ASSERT_TRUE(vernonRuntimeCopyFromHost(buffer, 0, values, sizeof(values)) == VERNON_STATUS_OK);
    pipeline = vernonRuntimeLoadComputeBundle(runtime, VERNON_CPU_BUNDLE_PATH);
    if (!pipeline) {
        VernonStringView error = vernonRuntimeGetLastError(runtime);
        fprintf(stderr, "%.*s\n", (int)error.size, error.data);
    }
    ASSERT_TRUE(pipeline);
    ASSERT_TRUE(vernonRuntimePipelineInvoke(pipeline, &invocation) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeCopyToHost(buffer, 0, values, sizeof(values)) == VERNON_STATUS_OK);
    ASSERT_TRUE(values[0] == 0.0f && values[2] == 2.0f);
    ASSERT_TRUE(values[3] == 10.0f && values[11] == 112.0f);
    vernonRuntimeLoadedPipelineDestroy(pipeline);

    ASSERT_TRUE(vernonRuntimeBufferFree(buffer) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
}
