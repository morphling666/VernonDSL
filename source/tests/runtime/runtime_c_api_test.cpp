#include "vernon-c/Runtime.h"

#include <gtest/gtest.h>

static VernonStatus registered_stub(const VernonCpuInvocation *) { return VERNON_STATUS_OK; }

TEST(RuntimeCApi, PullbackOpaqueHandleRejectsInvalidCalls) {
    EXPECT_EQ(vernonProgramPullbackApply(nullptr, nullptr, 0), VERNON_STATUS_INVALID_ARGUMENT);
    vernonProgramPullbackDestroy(nullptr);
}

TEST(RuntimeCApi, StaticCpuRegistrationValidatesIdentity) {
    static const char static_symbol[] = "__vernon_cpu_test_fill";
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntry({static_symbol, sizeof(static_symbol) - 1}, registered_stub) ==
                VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntry({static_symbol, sizeof(static_symbol) - 1}, registered_stub) ==
                VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntry({nullptr, 0}, registered_stub) == VERNON_STATUS_INVALID_ARGUMENT);
}
