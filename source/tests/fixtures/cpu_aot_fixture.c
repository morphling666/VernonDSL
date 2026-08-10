#include "VernonCommon.h"
#include "VernonCpuWorkgroupABI.h"

#include <stdint.h>
#include <string.h>

#if defined(_WIN32)
#define VERNON_TEST_EXPORT __declspec(dllexport)
#else
#define VERNON_TEST_EXPORT __attribute__((visibility("default")))
#endif

VERNON_TEST_EXPORT VernonStatus vernon_test_fill(const VernonCpuInvocation *invocation) {
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
