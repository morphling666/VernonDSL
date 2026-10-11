#include "runtime/backend_cpu.h"
#include "vernon-c/Runtime.h"

#include <gtest/gtest.h>

static VernonStatus registered_stub(const VernonCpuInvocation *) { return VERNON_STATUS_OK; }
static VernonStatus other_image_stub(const VernonCpuInvocation *) { return VERNON_STATUS_INTERNAL_ERROR; }

TEST(RuntimeCApi, PullbackOpaqueHandleRejectsInvalidCalls) {
    EXPECT_EQ(vernonProgramPullbackApply(nullptr, nullptr, 0, nullptr), VERNON_STATUS_INVALID_ARGUMENT);
    vernonProgramPullbackDestroy(nullptr);
}

TEST(RuntimeCApi, StaticCpuRegistrationValidatesIdentity) {
    static const char static_symbol[] = "__vernon_cpu_test_fill";
    const VernonStaticCpuEntry entry{{static_symbol, sizeof(static_symbol) - 1}, registered_stub};
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntries(&entry, 1) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntries(&entry, 1) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntries(nullptr, 0) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntries(nullptr, 1) == VERNON_STATUS_INVALID_ARGUMENT);
    const VernonStaticCpuEntry unnamed[]{entry, {{nullptr, 0}, registered_stub}};
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntries(unnamed, 2) == VERNON_STATUS_INVALID_ARGUMENT);
    const VernonStaticCpuEntry missingEntry{{static_symbol, sizeof(static_symbol) - 1}, nullptr};
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntries(&missingEntry, 1) == VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_TRUE(vernonRuntimeUnregisterStaticCpuEntries(&missingEntry, 1) == VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_TRUE(vernonRuntimeUnregisterStaticCpuEntries(&entry, 1) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeUnregisterStaticCpuEntries(&entry, 1) == VERNON_STATUS_OK);
}

TEST(RuntimeCApi, StaticCpuRegistrationTracksEveryLoadedImage) {
    static const char symbol[] = "__vernon_cpu_test_images";
    const VernonStaticCpuEntry first{{symbol, sizeof(symbol) - 1}, registered_stub};
    const VernonStaticCpuEntry second{{symbol, sizeof(symbol) - 1}, other_image_stub};
    ASSERT_EQ(vernon::runtime::findStaticCpuEntry(symbol), nullptr);

    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntries(&first, 1), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntries(&first, 1), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntries(&second, 1), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::runtime::findStaticCpuEntry(symbol), &registered_stub);

    ASSERT_EQ(vernonRuntimeUnregisterStaticCpuEntries(&first, 1), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::runtime::findStaticCpuEntry(symbol), &registered_stub);
    ASSERT_EQ(vernonRuntimeUnregisterStaticCpuEntries(&first, 1), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::runtime::findStaticCpuEntry(symbol), &other_image_stub);
    ASSERT_EQ(vernonRuntimeUnregisterStaticCpuEntries(&second, 1), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::runtime::findStaticCpuEntry(symbol), nullptr);
}
