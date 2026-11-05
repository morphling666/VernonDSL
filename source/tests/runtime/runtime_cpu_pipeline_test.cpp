#include "program_fixture_manifest_table.h"
#include "runtime/runtime_state.h"
#include "schedule_test_support.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>

#if !defined(VERNON_RUNTIME_PROFILE_WEB)
#ifndef VERNON_CPU_CANONICAL_PROGRAM_MANIFEST
#define VERNON_CPU_CANONICAL_PROGRAM_MANIFEST ""
#endif

#endif

namespace {

std::string readFile(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string lastError(VernonRuntimeContext *context) {
    const VernonStringView error = vernonRuntimeGetLastError(context);
    return std::string(error.data ? error.data : "", error.size);
}

} // namespace

#if !defined(VERNON_RUNTIME_PROFILE_WEB)
TEST(RuntimeCpuPipeline, LoadsValidatesAndCallsBundles) {
    const std::string manifest = readFile(VERNON_CPU_CANONICAL_PROGRAM_MANIFEST);
    ASSERT_FALSE(manifest.empty());
    VernonRuntimeBackend target = VERNON_RUNTIME_CUDA;
    ASSERT_EQ(vernonRuntimeProgramBundleInspectTarget(manifest.data(), manifest.size(), &target), VERNON_STATUS_OK);
    EXPECT_EQ(target, VERNON_RUNTIME_CPU);

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), nullptr);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    EXPECT_EQ(context->owner.childCount(), 1u);
    VernonProgramExecutable *first = vernonRuntimeResolveProgram(bundle, nullptr);
    VernonProgramExecutable *second = vernonRuntimeResolveProgram(bundle, nullptr);
    ASSERT_NE(first, nullptr) << lastError(context);
    ASSERT_NE(second, nullptr) << lastError(context);
    EXPECT_NE(first, second);
    EXPECT_GT(context->owner.childCount(), 1u);

    {
        vernon::tests::LinkedProgram linked(context, first);
        ASSERT_TRUE(linked) << linked.error();
        vernon::tests::HostTensor<float> source({3.0f});
        vernon::tests::HostTensor<float> output({0.0f});
        linked.record().bind("source", source.view());
        linked.record().bind("output", output.view());
        linked.record().bindGrid(1);
        ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();
        EXPECT_FLOAT_EQ(output.values[0], 9.0f);
    }

    vernonRuntimeProgramExecutableDestroy(second);
    vernonRuntimeProgramExecutableDestroy(first);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeCpuPipeline, RejectsLegacyExecutableTopology) {
    std::string manifest = readFile(VERNON_CPU_CANONICAL_PROGRAM_MANIFEST);
    ASSERT_FALSE(manifest.empty());
    manifest.insert(manifest.rfind('}'), R"(,"stage_artifacts":{"legacy":{"artifact":{"path":"../does-not-exist"}}})");
    const size_t typeField = manifest.find("\"type\"");
    const size_t type = manifest.find("\"program\"", typeField);
    ASSERT_NE(type, std::string::npos);
    manifest.replace(type, std::strlen("\"program\""), "\"pipeline\"");

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), nullptr), nullptr);
    EXPECT_EQ(lastError(context), "unsupported or invalid Program bundle");
    EXPECT_EQ(context->owner.childCount(), 0u);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

// An executable keeps its context alive after its bundle is destroyed.
TEST(RuntimeCpuPipeline, KeepsTheContextAliveThroughExecutableOwnership) {
    const std::string manifest = readFile(VERNON_CPU_CANONICAL_PROGRAM_MANIFEST);
    ASSERT_FALSE(manifest.empty());
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), nullptr);
    ASSERT_NE(bundle, nullptr);
    VernonProgramExecutable *executable = vernonRuntimeResolveProgram(bundle, nullptr);
    ASSERT_NE(executable, nullptr);

    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_NE(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
    vernonRuntimeProgramExecutableDestroy(executable);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}
#endif
