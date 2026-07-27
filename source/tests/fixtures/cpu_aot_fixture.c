#include "VernonCommon.h"

#include <stdint.h>
#include <string.h>

#if defined(_WIN32)
#define VERNON_TEST_EXPORT __declspec(dllexport)
#else
#define VERNON_TEST_EXPORT __attribute__((visibility("default")))
#endif

VERNON_TEST_EXPORT VernonStatus vernon_test_fill(const VernonCpuInvocation *invocation) {
    uintptr_t address = 0;
    uint32_t gid[3] = {0, 0, 0};
    if (!invocation || invocation->arguments_size < 20)
        return VERNON_STATUS_INVALID_ARGUMENT;
    memcpy(&address, invocation->arguments, sizeof(address));
    memcpy(gid, (const unsigned char *)invocation->arguments + 8, sizeof(gid));
    float *values = (float *)address;
    values[gid[2] * 6 + gid[1] * 3 + gid[0]] = (float)(gid[0] + 10 * gid[1] + 100 * gid[2]);
    return VERNON_STATUS_OK;
}
