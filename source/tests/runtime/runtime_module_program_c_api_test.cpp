#include "VernonRuntime.hpp"
#include "program_fixture_manifest_table.h"
#include "runtime/runtime_state.h"
#include "runtime/runtime_test_hooks.h"
#include "schedule_test_support.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

std::atomic<size_t> runtimeAllocationFailures{};

void *operator new(size_t size) {
    size_t remaining = runtimeAllocationFailures.load(std::memory_order_relaxed);
    while (remaining &&
           !runtimeAllocationFailures.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
    }
    if (remaining)
        throw std::bad_alloc{};
    if (void *memory = std::malloc(size))
        return memory;
    throw std::bad_alloc{};
}

void *operator new[](size_t size) { return ::operator new(size); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, size_t) noexcept { std::free(memory); }

namespace {

using vernon::tests::LinkedProgram;
using vernon::tests::LinkedVjp;
using vernon::tests::ResolvedProgram;

std::string lastError(VernonRuntimeContext *context) {
    const VernonStringView error = vernonRuntimeGetLastError(context);
    return std::string(error.data ? error.data : "", error.size);
}

VernonProgramParameterView parameter(VernonProgramExecutable *pipeline, const char *name) {
    VernonProgramParameterView result{};
    EXPECT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline, {name, std::strlen(name)}, &result),
              VERNON_STATUS_OK);
    return result;
}

// Dense host floats a CPU schedule argument views; the view points at this object, which must outlive the call.
struct HostFloats {
    explicit HostFloats(std::vector<float> initial) : values(std::move(initial)), shape(values.size()) {}
    HostFloats(const HostFloats &) = delete;
    HostFloats &operator=(const HostFloats &) = delete;

    VernonTensorView view() {
        return vernon::tests::hostView(values.data(), values.size() * sizeof(float), 1, &shape, &stride);
    }

    std::vector<float> values;
    uint64_t shape;
    int64_t stride{sizeof(float)};
};

// The name of the executable's first output boundary, which republishes the output it writes.
std::string publishedOutput(VernonProgramExecutable *executable) {
    VernonProgramParameterView published{};
    EXPECT_EQ(
        vernonRuntimeProgramExecutableGetBoundaryByIndex(executable, VERNON_PROGRAM_BOUNDARY_OUTPUT, 0, &published),
        VERNON_STATUS_OK);
    return std::string(published.name.data, published.name.size);
}

const vernon::tests::ProgramFixtureManifest &cpuFixture(std::string_view fixtureId) {
    const auto *fixture = vernon::tests::findProgramFixtureManifest(fixtureId, VERNON_RUNTIME_CPU);
    if (!fixture)
        throw std::logic_error("missing CPU fixture '" + std::string(fixtureId) + "'");
    return *fixture;
}

std::string fixtureManifest(std::string_view fixtureId) {
    const auto &fixture = cpuFixture(fixtureId);
    std::ifstream input(std::string(fixture.manifestPath), std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read CPU fixture manifest '" + std::string(fixture.manifestPath) + "'");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::string readFile(const char *path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

} // namespace

TEST(RuntimeModuleProgramCApi, ConcurrentFirstResolvePublishesOneContextMemoryPolicySafely) {
    const std::string manifest = fixtureManifest("module_program");
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), nullptr);
    ASSERT_NE(bundle, nullptr) << lastError(context);

    std::atomic<uint32_t> ready{};
    std::atomic<bool> start{};
    std::array<VernonProgramExecutable *, 2> executables{};
    std::array<std::thread, 2> resolvers;
    for (size_t index = 0; index < resolvers.size(); ++index)
        resolvers[index] = std::thread([&, index] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire))
                std::this_thread::yield();
            executables[index] = vernonRuntimeResolveProgram(bundle, nullptr);
        });
    while (ready.load(std::memory_order_acquire) != resolvers.size())
        std::this_thread::yield();
    start.store(true, std::memory_order_release);
    for (std::thread &resolver : resolvers)
        resolver.join();

    ASSERT_NE(executables[0], nullptr);
    ASSERT_NE(executables[1], nullptr);
    vernonRuntimeProgramExecutableDestroy(executables[0]);
    vernonRuntimeProgramExecutableDestroy(executables[1]);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCApi, ComputeModuleForward9AndVjpGradient6WithCheckpointPolicies) {
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    {
        const ResolvedProgram program(context, VERNON_MODULE_PROGRAM_MANIFEST);
        ASSERT_TRUE(program) << lastError(context);
        ASSERT_EQ(vernonRuntimeProgramExecutableGetParameterCount(program.get()), 2u);
        ASSERT_EQ(vernonRuntimeProgramExecutableHasProgramAutodiff(program.get()), 1u);
        ASSERT_EQ(parameter(program.get(), "source").kind, VERNON_PROGRAM_TENSOR);
        ASSERT_EQ(parameter(program.get(), "output").kind, VERNON_PROGRAM_TENSOR);

        const auto differentiate = [&](LinkedVjp &vjp) {
            HostFloats source({3.0f});
            HostFloats output({0.0f});
            HostFloats seed({1.0f});
            HostFloats gradient({0.0f});
            vjp.forwardRecord().bind("source", source.view());
            vjp.forwardRecord().bind("output", output.view());
            vjp.forwardRecord().bindGrid(1);
            ASSERT_EQ(vjp.forward(), VERNON_STATUS_OK) << vjp.error();
            EXPECT_FLOAT_EQ(output.values[0], 9.0f);
            vjp.backwardRecord().bind("output", seed.view(), VERNON_PROGRAM_BOUNDARY_COTANGENT);
            vjp.backwardRecord().bind("source", gradient.view(), VERNON_PROGRAM_BOUNDARY_GRADIENT);
            ASSERT_EQ(vjp.backward(), VERNON_STATUS_OK) << vjp.error();
            EXPECT_FLOAT_EQ(gradient.values[0], 6.0f);
        };
        LinkedVjp plain(context, program.get());
        ASSERT_TRUE(plain) << plain.error();
        differentiate(plain);

        for (const auto &[policy, budget] : {std::pair{VERNON_SCHEDULE_CHECKPOINT_MIN_MEMORY, uint64_t{1} << 20},
                                             std::pair{VERNON_SCHEDULE_CHECKPOINT_MIN_RUNTIME, uint64_t{2} << 20}}) {
            SCOPED_TRACE(policy);
            const VernonScheduleCheckpointOptions options{sizeof(VernonScheduleCheckpointOptions), budget, policy};
            LinkedVjp checkpointed(context, program.get(), &options);
            ASSERT_TRUE(checkpointed) << checkpointed.error();
            differentiate(checkpointed);
        }
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCApi, ComposedProgramGraphChainsItsNodesPullbacksWithoutCompositeAutodiff) {
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    {
        const ResolvedProgram program(context, VERNON_MODULE_PROGRAM_MANIFEST);
        ASSERT_TRUE(program) << lastError(context);
        VernonProgramGraph *graph = vernonRuntimeProgramGraphCreate(context);
        ASSERT_NE(graph, nullptr);
        VernonProgramNodeId firstNode = UINT32_MAX;
        VernonProgramNodeId secondNode = UINT32_MAX;
        ASSERT_EQ(vernonRuntimeProgramGraphAddProgram(graph, program.bundle(), nullptr, {}, &firstNode),
                  VERNON_STATUS_OK);
        ASSERT_EQ(vernonRuntimeProgramGraphAddProgram(graph, program.bundle(), nullptr, {}, &secondNode),
                  VERNON_STATUS_OK);
        const auto boundary = [&](VernonProgramNodeId node, VernonProgramBoundaryRole role, const char *name) {
            VernonProgramNodeBindingToken token{sizeof(VernonProgramNodeBindingToken)};
            EXPECT_EQ(vernonRuntimeProgramGraphFindBoundary(graph, node, role, {name, std::strlen(name)}, &token),
                      VERNON_STATUS_OK);
            return token;
        };
        const VernonProgramNodeBindingToken firstSource = boundary(firstNode, VERNON_PROGRAM_BOUNDARY_INPUT, "source");
        const VernonProgramNodeBindingToken firstOutput = boundary(firstNode, VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
        const VernonProgramNodeBindingToken secondSource =
            boundary(secondNode, VERNON_PROGRAM_BOUNDARY_INPUT, "source");
        const VernonProgramNodeBindingToken secondOutput =
            boundary(secondNode, VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
        VernonProgramGraphValue intermediate{sizeof(VernonProgramGraphValue)};
        ASSERT_EQ(vernonRuntimeProgramGraphCreateValue(graph, &firstOutput, &intermediate), VERNON_STATUS_OK);
        ASSERT_EQ(vernonRuntimeProgramGraphConnectValue(graph, &intermediate, &secondSource), VERNON_STATUS_OK);
        ASSERT_EQ(vernonRuntimeProgramGraphExportBoundary(graph, &firstSource, {"source", std::strlen("source")}),
                  VERNON_STATUS_OK);
        ASSERT_EQ(vernonRuntimeProgramGraphExportBoundary(graph, &secondOutput, {"output", std::strlen("output")}),
                  VERNON_STATUS_OK);
        VernonProgramExecutable *composite = vernonRuntimeResolveProgramGraph(graph);
        ASSERT_NE(composite, nullptr) << lastError(context);
        EXPECT_EQ(vernonRuntimeProgramExecutableHasProgramAutodiff(composite), 0u);
        EXPECT_EQ(vernonRuntimeProgramExecutableGetBoundaryCount(composite, VERNON_PROGRAM_BOUNDARY_COTANGENT), 0u);
        EXPECT_EQ(vernonRuntimeProgramExecutableGetBoundaryCount(composite, VERNON_PROGRAM_BOUNDARY_GRADIENT), 0u);
        {
            LinkedVjp vjp(context, composite);
            ASSERT_TRUE(vjp) << vjp.error();
            HostFloats source({3.0f});
            HostFloats output({0.0f});
            HostFloats seed({1.0f});
            HostFloats gradient({0.0f});
            vjp.forwardRecord().bind("source", source.view());
            vjp.forwardRecord().bind("output", output.view());
            vjp.forwardRecord().bindGrid(1);
            ASSERT_EQ(vjp.forward(), VERNON_STATUS_OK) << vjp.error();
            EXPECT_FLOAT_EQ(output.values[0], 81.0f);
            vjp.backwardRecord().bind("output", seed.view(), VERNON_PROGRAM_BOUNDARY_COTANGENT);
            vjp.backwardRecord().bind("source", gradient.view(), VERNON_PROGRAM_BOUNDARY_GRADIENT);
            ASSERT_EQ(vjp.backward(), VERNON_STATUS_OK) << vjp.error();
            EXPECT_FLOAT_EQ(gradient.values[0], 108.0f);
        }
        vernonRuntimeProgramExecutableDestroy(composite);
        vernonRuntimeProgramGraphDestroy(graph);
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCApi, LoadsCanonicalBundleThroughBundleThenResolve) {
    const std::string manifest = readFile(VERNON_MODULE_PROGRAM_MANIFEST);
    ASSERT_FALSE(manifest.empty());

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), nullptr);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    const VernonStringView id = vernonRuntimeProgramBundleGetId(bundle);
    EXPECT_GT(id.size, 0u);

    const VernonProgramVariantSelector invalidSelector{};
    VernonProgramExecutable *rejected = reinterpret_cast<VernonProgramExecutable *>(uintptr_t{1});
    EXPECT_EQ(vernonRuntimeResolveProgramResult(bundle, &invalidSelector, &rejected),
              VERNON_RUNTIME_OPERATION_INVALID_ARGUMENT);
    EXPECT_EQ(rejected, nullptr);
    EXPECT_EQ(lastError(context), "Program variant selector is invalid");
    const std::string missingName = "NO_SUCH_SPECIALIZATION";
    VernonProgramSpecialization missing{};
    missing.struct_size = sizeof(missing);
    missing.name = {missingName.data(), missingName.size()};
    missing.kind = VERNON_PROGRAM_SPECIALIZATION_BOOL;
    missing.value.boolean_value = 1;
    const VernonProgramVariantSelector missingSelector{sizeof(VernonProgramVariantSelector), &missing, 1, {}};
    EXPECT_EQ(vernonRuntimeResolveProgramResult(bundle, &missingSelector, &rejected),
              VERNON_RUNTIME_OPERATION_PARSE_FAILURE);
    EXPECT_EQ(rejected, nullptr);
    EXPECT_EQ(lastError(context), "Program bundle has no matching variant");

    VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(bundle, nullptr);
    ASSERT_NE(pipeline, nullptr) << lastError(context);
    VernonProgramExecutable *second = vernonRuntimeResolveProgram(bundle, nullptr);
    ASSERT_NE(second, nullptr) << lastError(context);
    EXPECT_NE(second, pipeline);
    vernonRuntimeProgramExecutableDestroy(second);
    EXPECT_EQ(vernonRuntimeProgramExecutableGetParameterCount(pipeline), 2u);

    {
        LinkedProgram linked(context, pipeline);
        ASSERT_TRUE(linked) << linked.error();
        HostFloats source({5.0f});
        HostFloats output({0.0f});
        linked.record().bind("source", source.view());
        linked.record().bind("output", output.view());
        linked.record().bindGrid(1);
        ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();
        EXPECT_FLOAT_EQ(output.values[0], 25.0f);
    }

    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCApi, RejectsLegacyEnvelopeBeforeArtifactIo) {
    std::string manifest = readFile(VERNON_MODULE_PROGRAM_MANIFEST);
    ASSERT_FALSE(manifest.empty());
    const size_t typeField = manifest.find("\"type\"");
    ASSERT_NE(typeField, std::string::npos);
    const size_t type = manifest.find("\"program\"", typeField);
    ASSERT_NE(type, std::string::npos);
    manifest.replace(type, std::strlen("\"program\""), "\"pipeline\"");
    manifest.insert(manifest.rfind('}'), R"(,"stage_artifacts":{"legacy":{"artifact":{"path":"../does-not-exist"}}})");

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), nullptr), nullptr);
    EXPECT_EQ(lastError(context), "unsupported or invalid Program bundle");
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

class RuntimeProgramEnvelopeRejection : public testing::TestWithParam<const char *> {};

TEST_P(RuntimeProgramEnvelopeRejection, RejectsEveryNonProgramTypeBeforeArtifactIo) {
    std::string manifest = readFile(VERNON_MODULE_PROGRAM_MANIFEST);
    ASSERT_FALSE(manifest.empty());
    const size_t typeField = manifest.find("\"type\"");
    ASSERT_NE(typeField, std::string::npos);
    const size_t type = manifest.find("\"program\"", typeField);
    ASSERT_NE(type, std::string::npos);
    manifest.replace(type, std::strlen("\"program\""), std::string{"\""} + GetParam() + "\"");
    manifest.insert(manifest.rfind('}'), R"(,"stage_artifacts":{"legacy":{"artifact":{"path":"../does-not-exist"}}})");

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), nullptr), nullptr);
    EXPECT_EQ(lastError(context), "unsupported or invalid Program bundle");
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

INSTANTIATE_TEST_SUITE_P(RetiredAndUnknownTypes, RuntimeProgramEnvelopeRejection,
                         testing::Values("pipeline", "program_bundle", "", "Program", "program-v1", "program_v1",
                                         "bundle", "asset", "shader", "kernel", "module", "graph", "executable",
                                         "deployment", "stage", "compute", "graphics", "autodiff", "vjp",
                                         "pipeline_bundle", "cooked_pipeline", "cooked_program", "program_asset",
                                         "program_manifest", "vernon_program", "native_program", "runtime_program",
                                         "programs", "PROGRAM", " program", "program ", "null", "true", "unknown"));

TEST(RuntimeModuleProgramCppApi, PreservesLoadAndResolveFailureIdentity) {
    static_assert(noexcept(vernon::runtime::ProgramAsset::load(nullptr, nullptr, 0)));
    static_assert(noexcept(std::declval<const vernon::runtime::ProgramAsset &>().resolve()));
    static_assert(noexcept(vernon::runtime::ProgramGraph::create(nullptr)));
    static_assert(noexcept(std::declval<vernon::runtime::ProgramVariant &>().set(std::string{}, uint32_t{})));

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);

    constexpr std::string_view malformed = "{";
    auto malformedResult = vernon::runtime::ProgramAsset::load(context, malformed.data(), malformed.size());
    ASSERT_TRUE(malformedResult.isErr());
    EXPECT_EQ(malformedResult.error().code, vernon::RuntimeErrorCode::ParseFailure);
    EXPECT_STREQ(malformedResult.error().context.operation, "ProgramAsset.load");
    EXPECT_EQ(lastError(context), "invalid Program bundle: malformed JSON");

    {
        const std::string manifest = fixtureManifest("module_program");
        std::string corrupted = manifest;
        const size_t hashMember = corrupted.find("\"content_hash\"");
        ASSERT_NE(hashMember, std::string::npos);
        const size_t hashValue = corrupted.find('"', corrupted.find(':', hashMember) + 1);
        ASSERT_NE(hashValue, std::string::npos);
        ASSERT_LT(hashValue + 1, corrupted.size());
        corrupted[hashValue + 1] = corrupted[hashValue + 1] == '0' ? '1' : '0';
        auto verificationResult = vernon::runtime::ProgramAsset::load(context, corrupted.data(), corrupted.size());
        ASSERT_TRUE(verificationResult.isErr());
        EXPECT_EQ(verificationResult.error().code, vernon::RuntimeErrorCode::VerificationFailure);
        EXPECT_STREQ(verificationResult.error().context.operation, "ProgramAsset.load");
        EXPECT_EQ(lastError(context), "Program bundle content_hash does not match canonical content");

        std::string unsupported = manifest;
        const size_t targetMember = unsupported.find("\"target\"");
        ASSERT_NE(targetMember, std::string::npos);
        const size_t cpuTarget = unsupported.find("\"cpu\"", targetMember);
        ASSERT_NE(cpuTarget, std::string::npos);
        unsupported.replace(cpuTarget, std::strlen("\"cpu\""), "\"cuda\"");
        auto unsupportedResult = vernon::runtime::ProgramAsset::load(context, unsupported.data(), unsupported.size());
        ASSERT_TRUE(unsupportedResult.isErr());
        EXPECT_EQ(unsupportedResult.error().code, vernon::RuntimeErrorCode::Unsupported);
        EXPECT_STREQ(unsupportedResult.error().context.operation, "ProgramAsset.load");
        EXPECT_EQ(lastError(context), "Program bundle target does not match the Runtime backend");

        auto assetResult = vernon::runtime::ProgramAsset::load(context, manifest.data(), manifest.size());
        ASSERT_TRUE(assetResult) << lastError(context);
        EXPECT_TRUE(lastError(context).empty());
        auto asset = std::move(assetResult).value();

        vernon::runtime::ProgramVariant missing;
        ASSERT_TRUE(missing.set("missing", uint32_t{1}));
        auto resolveResult = asset.resolve(missing);
        ASSERT_TRUE(resolveResult.isErr());
        EXPECT_EQ(resolveResult.error().code, vernon::RuntimeErrorCode::ParseFailure);
        EXPECT_STREQ(resolveResult.error().context.operation, "ProgramAsset.resolve");
        EXPECT_EQ(lastError(context), "Program bundle has no matching variant");

        auto graphResult = vernon::runtime::ProgramGraph::create(context);
        ASSERT_TRUE(graphResult);
        auto emptyGraph = std::move(graphResult).value();
        auto compileResult = emptyGraph.compile();
        ASSERT_TRUE(compileResult.isErr());
        EXPECT_EQ(compileResult.error().code, vernon::RuntimeErrorCode::InvalidArgument);
        EXPECT_STREQ(compileResult.error().context.operation, "ProgramGraph.compile");
        EXPECT_EQ(lastError(context), "invalid ProgramGraph resolve invocation");

        EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_INTERNAL_ERROR);
        EXPECT_EQ(lastError(context), "runtime context still owns live handles");
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCApi, DiagnosticsAreThreadLocalAndRejectReusedAddressGeneration) {
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    constexpr std::string_view malformed = "{";
    VernonProgramBundle *output = reinterpret_cast<VernonProgramBundle *>(uintptr_t{1});
    EXPECT_EQ(
        vernonRuntimeLoadProgramBundleWithOptionsResult(context, malformed.data(), malformed.size(), nullptr, &output),
        VERNON_RUNTIME_OPERATION_PARSE_FAILURE);
    EXPECT_EQ(output, nullptr);
    EXPECT_EQ(lastError(context), "invalid Program bundle: malformed JSON");

    ++context->diagnosticGeneration;
    EXPECT_TRUE(lastError(context).empty());
    EXPECT_EQ(
        vernonRuntimeLoadProgramBundleWithOptionsResult(context, malformed.data(), malformed.size(), nullptr, &output),
        VERNON_RUNTIME_OPERATION_PARSE_FAILURE);
    EXPECT_EQ(lastError(context), "invalid Program bundle: malformed JSON");

    std::string otherThreadDiagnostic = "not set";
    std::thread other([&] { otherThreadDiagnostic = lastError(context); });
    other.join();
    EXPECT_TRUE(otherThreadDiagnostic.empty());

    const std::string manifest = fixtureManifest("module_program");
    EXPECT_EQ(
        vernonRuntimeLoadProgramBundleWithOptionsResult(context, manifest.data(), manifest.size(), nullptr, &output),
        VERNON_RUNTIME_OPERATION_OK);
    ASSERT_NE(output, nullptr);
    EXPECT_TRUE(lastError(context).empty());
    VernonProgramBundle *rejected = nullptr;
    EXPECT_EQ(vernonRuntimeLoadProgramBundleWithOptionsResult(context, malformed.data(), malformed.size(), nullptr,
                                                              &rejected),
              VERNON_RUNTIME_OPERATION_PARSE_FAILURE);
    EXPECT_EQ(rejected, nullptr);
    EXPECT_EQ(lastError(context), "invalid Program bundle: malformed JSON");
    vernonRuntimeProgramBundleDestroy(output);
    EXPECT_EQ(lastError(context), "invalid Program bundle: malformed JSON");
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);

    VernonRuntimeContext *replacement = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(replacement, nullptr);
    EXPECT_TRUE(lastError(replacement).empty());
    EXPECT_EQ(vernonRuntimeDestroy(replacement), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCApi, DiagnosticsPreserveNestedContextsAcrossOverflow) {
    constexpr size_t contextCount = 24;
    std::array<VernonRuntimeContext *, contextCount> contexts{};
    for (VernonRuntimeContext *&context : contexts) {
        context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
        ASSERT_NE(context, nullptr);
    }

    const uint64_t savedClock = vernon::runtime::diagnosticClockForTesting();
    vernon::runtime::setDiagnosticClockForTesting(std::numeric_limits<uint64_t>::max() - 1);
    vernon::runtime::failNextDiagnosticOverflowAllocationForTesting();
    const auto nest = [&](auto &&self, size_t index) -> void {
        if (index == contexts.size())
            return;
        vernon::runtime::RuntimeDiagnosticScope scope(contexts[index]);
        const std::string expected = "nested diagnostic " + std::to_string(index);
        vernon::runtime::invocationDiagnostic(*contexts[index]) = expected;
        self(self, index + 1);
        const std::string *current = vernon::runtime::currentInvocationDiagnostic(*contexts[index]);
        ASSERT_NE(current, nullptr);
        EXPECT_EQ(*current, expected);
    };
    nest(nest, 0);
    EXPECT_EQ(vernon::runtime::diagnosticClockForTesting(), std::numeric_limits<uint64_t>::max());

    for (size_t index = 0; index < contexts.size(); ++index) {
        const std::string *published = vernon::runtime::currentInvocationDiagnostic(*contexts[index]);
        ASSERT_NE(published, nullptr);
        EXPECT_EQ(*published, "nested diagnostic " + std::to_string(index));
        EXPECT_EQ(vernonRuntimeDestroy(contexts[index]), VERNON_STATUS_OK);
    }
    vernon::runtime::setDiagnosticClockForTesting(savedClock);
}

TEST(RuntimeModuleProgramCApi, DiagnosticGenerationCounterSaturatesWithoutReuse) {
    const uint64_t savedGeneration = vernon::runtime::diagnosticGenerationCounterForTesting();
    vernon::runtime::setDiagnosticGenerationCounterForTesting(std::numeric_limits<uint64_t>::max() - 1);

    VernonRuntimeContext *last = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    EXPECT_NE(last, nullptr);
    if (last)
        EXPECT_EQ(last->diagnosticGeneration, std::numeric_limits<uint64_t>::max() - 1);
    EXPECT_EQ(vernon::runtime::diagnosticGenerationCounterForTesting(), std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr), nullptr);
    EXPECT_EQ(vernon::runtime::diagnosticGenerationCounterForTesting(), std::numeric_limits<uint64_t>::max());

    if (last)
        EXPECT_EQ(vernonRuntimeDestroy(last), VERNON_STATUS_OK);
    vernon::runtime::setDiagnosticGenerationCounterForTesting(savedGeneration);
}

TEST(RuntimeModuleProgramCApi, AllocationFailurePublishesExactEmergencyDiagnosticWithoutAllocating) {
    vernon::runtime::ProgramVariant variant;
    runtimeAllocationFailures.store(1, std::memory_order_relaxed);
    auto setResult = variant.set("extent", uint32_t{1});
    ASSERT_TRUE(setResult.isErr());
    EXPECT_EQ(setResult.error().code, vernon::RuntimeErrorCode::ResourceExhausted);
    EXPECT_STREQ(setResult.error().context.operation, "ProgramVariant.set");
    EXPECT_EQ(runtimeAllocationFailures.load(std::memory_order_relaxed), 0u);

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    const std::string manifest = fixtureManifest("module_program");
    VernonProgramBundle *output = nullptr;
    runtimeAllocationFailures.store(2, std::memory_order_relaxed);
    EXPECT_EQ(
        vernonRuntimeLoadProgramBundleWithOptionsResult(context, manifest.data(), manifest.size(), nullptr, &output),
        VERNON_RUNTIME_OPERATION_RESOURCE_EXHAUSTED);
    EXPECT_EQ(output, nullptr);
    EXPECT_EQ(lastError(context), "domain=3 code=5 operation=create_operation_control detail=0 value=0");
    EXPECT_EQ(runtimeAllocationFailures.load(std::memory_order_relaxed), 0u);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCppApi, ProgramGraphDifferentiatesThroughCanonicalExportedBoundaries) {
    const std::string manifest = readFile(VERNON_MODULE_PROGRAM_MANIFEST);
    ASSERT_FALSE(manifest.empty());
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    {
        auto assetResult = vernon::runtime::ProgramAsset::load(context, manifest.data(), manifest.size());
        ASSERT_TRUE(assetResult);
        auto asset = std::move(assetResult).value();
        auto graphResult = vernon::runtime::ProgramGraph::create(context);
        ASSERT_TRUE(graphResult);
        auto graph = std::move(graphResult).value();
        auto firstResult = graph.add(asset);
        ASSERT_TRUE(firstResult);
        const auto first = std::move(firstResult).value();
        auto secondResult = graph.add(asset);
        ASSERT_TRUE(secondResult);
        const auto second = std::move(secondResult).value();
        for (const auto &[node, prefix] : {std::pair{&first, "first"}, std::pair{&second, "second"}}) {
            auto source = node->boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "source");
            ASSERT_TRUE(source);
            auto output = node->boundary(VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
            ASSERT_TRUE(output);
            ASSERT_TRUE(graph.exportBoundary(source.value(), std::string(prefix) + "_source"));
            ASSERT_TRUE(graph.exportBoundary(output.value(), std::string(prefix) + "_output"));
        }
        auto executableResult = graph.compile();
        ASSERT_TRUE(executableResult);
        auto executable = std::move(executableResult).value();

        LinkedVjp vjp(context, executable.get());
        ASSERT_TRUE(vjp) << vjp.error();
        HostFloats firstSource({2.0f});
        HostFloats firstOutput({0.0f});
        HostFloats secondSource({3.0f});
        HostFloats secondOutput({0.0f});
        vjp.forwardRecord().bind("first_source", firstSource.view());
        vjp.forwardRecord().bind("first_output", firstOutput.view());
        vjp.forwardRecord().bind("second_source", secondSource.view());
        vjp.forwardRecord().bind("second_output", secondOutput.view());
        vjp.forwardRecord().bindGrid(1);
        ASSERT_TRUE(vjp.forwardRecord().unbound().empty());
        ASSERT_EQ(vjp.forward(), VERNON_STATUS_OK) << vjp.error();
        EXPECT_FLOAT_EQ(firstOutput.values[0], 4.0f);
        EXPECT_FLOAT_EQ(secondOutput.values[0], 9.0f);

        HostFloats firstSeed({1.0f});
        HostFloats secondSeed({1.0f});
        HostFloats firstGradient({0.0f});
        HostFloats secondGradient({0.0f});
        vjp.backwardRecord().bind("first_output", firstSeed.view(), VERNON_PROGRAM_BOUNDARY_COTANGENT);
        vjp.backwardRecord().bind("second_output", secondSeed.view(), VERNON_PROGRAM_BOUNDARY_COTANGENT);
        vjp.backwardRecord().bind("first_source", firstGradient.view(), VERNON_PROGRAM_BOUNDARY_GRADIENT);
        vjp.backwardRecord().bind("second_source", secondGradient.view(), VERNON_PROGRAM_BOUNDARY_GRADIENT);
        ASSERT_TRUE(vjp.backwardRecord().unbound().empty());
        ASSERT_EQ(vjp.backward(), VERNON_STATUS_OK) << vjp.error();
        EXPECT_FLOAT_EQ(firstGradient.values[0], 4.0f);
        EXPECT_FLOAT_EQ(secondGradient.values[0], 6.0f);
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCppApi, TypedVariantsSelectIndependentProgramGraphNodes) {
    const std::string manifest = readFile(VERNON_TYPED_SPECIALIZATION_MANIFEST);
    ASSERT_FALSE(manifest.empty());
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    {
        auto assetResult = vernon::runtime::ProgramAsset::load(context, manifest.data(), manifest.size());
        ASSERT_TRUE(assetResult);
        auto asset = std::move(assetResult).value();
        vernon::runtime::ProgramVariant one;
        ASSERT_TRUE(one.set("extent", uint32_t{1}));
        vernon::runtime::ProgramVariant four;
        ASSERT_TRUE(four.set("extent", uint32_t{4}));
        auto oneExecutableResult = asset.resolve(one);
        ASSERT_TRUE(oneExecutableResult);
        const auto oneExecutable = std::move(oneExecutableResult).value();
        auto fourExecutableResult = asset.resolve(four);
        ASSERT_TRUE(fourExecutableResult);
        const auto fourExecutable = std::move(fourExecutableResult).value();
        const VernonProgramParameterView oneOutput = parameter(oneExecutable.get(), "output");
        const VernonProgramParameterView fourOutput = parameter(fourExecutable.get(), "output");
        ASSERT_EQ(oneOutput.rank, 1u);
        ASSERT_EQ(fourOutput.rank, 1u);
        EXPECT_EQ(oneOutput.static_shape[0], 1);
        EXPECT_EQ(fourOutput.static_shape[0], 4);

        auto graphResult = vernon::runtime::ProgramGraph::create(context);
        ASSERT_TRUE(graphResult);
        auto graph = std::move(graphResult).value();
        auto oneNodeResult = graph.add(asset, one);
        ASSERT_TRUE(oneNodeResult);
        const auto oneNode = std::move(oneNodeResult).value();
        auto fourNodeResult = graph.add(asset, four);
        ASSERT_TRUE(fourNodeResult);
        const auto fourNode = std::move(fourNodeResult).value();
        for (const auto &[node, prefix] : {std::pair{&oneNode, "one"}, std::pair{&fourNode, "four"}}) {
            auto output = node->boundary(VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
            ASSERT_TRUE(output);
            ASSERT_TRUE(graph.exportBoundary(output.value(), std::string(prefix) + "_output"));
            for (const char *axis : {"x", "y", "z"}) {
                auto grid = node->boundary(VERNON_PROGRAM_BOUNDARY_INPUT, std::string("__grid_") + axis);
                ASSERT_TRUE(grid);
                ASSERT_TRUE(graph.exportBoundary(grid.value(), std::string(prefix) + "_grid_" + axis));
            }
        }
        auto graphExecutableResult = graph.compile();
        ASSERT_TRUE(graphExecutableResult);
        auto graphExecutable = std::move(graphExecutableResult).value();

        LinkedProgram linked(context, graphExecutable.get());
        ASSERT_TRUE(linked) << linked.error();
        HostFloats oneValues{std::vector<float>(1)};
        HostFloats fourValues{std::vector<float>(4)};
        linked.record().bind("one_output", oneValues.view());
        linked.record().bind("four_output", fourValues.view());
        for (const char *name : {"one_grid_x", "one_grid_y", "one_grid_z", "four_grid_x", "four_grid_y", "four_grid_z"})
            linked.record().bindScalar<uint32_t>(name, 1);
        ASSERT_TRUE(linked.record().unbound().empty());
        ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();
        EXPECT_FLOAT_EQ(oneValues.values[0], 1.0f);
        EXPECT_FLOAT_EQ(fourValues.values[0], 4.0f);
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCppApi, ParamsLeavesBindAcrossTypedVariants) {
    const std::string manifest = readFile(VERNON_PARAMS_MANIFEST);
    ASSERT_FALSE(manifest.empty());
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    {
        auto assetResult = vernon::runtime::ProgramAsset::load(context, manifest.data(), manifest.size());
        ASSERT_TRUE(assetResult);
        auto asset = std::move(assetResult).value();
        for (const uint32_t extent : {2u, 4u}) {
            SCOPED_TRACE(extent);
            vernon::runtime::ProgramVariant variant;
            ASSERT_TRUE(variant.set("extent", extent));
            auto executableResult = asset.resolve(variant);
            ASSERT_TRUE(executableResult);
            auto executable = std::move(executableResult).value();
            VernonProgramParameterView group{};
            EXPECT_NE(vernonRuntimeProgramExecutableFindParameter(executable.get(), {"terms", 5}, &group),
                      VERNON_STATUS_OK);
            const VernonProgramParameterView outputParameter = parameter(executable.get(), "output");
            ASSERT_EQ(outputParameter.rank, 1u);
            EXPECT_EQ(outputParameter.static_shape[0], static_cast<int64_t>(extent));

            LinkedProgram linked(context, executable.get());
            ASSERT_TRUE(linked) << linked.error();
            std::vector<float> weightValues(extent);
            for (uint32_t index = 0; index < extent; ++index)
                weightValues[index] = static_cast<float>(index + 1);
            HostFloats weights(weightValues);
            HostFloats output{std::vector<float>(extent)};
            constexpr float gain = 3.0f;
            linked.record().bindScalar("terms.gain", gain);
            linked.record().bind("terms.lookup.weights", weights.view());
            linked.record().bind("output", output.view());
            if (const std::string published = publishedOutput(executable.get()); published != "output")
                linked.record().bind(published, output.view());
            linked.record().bindGrid(extent);
            ASSERT_TRUE(linked.record().unbound().empty());
            ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();
            for (uint32_t index = 0; index < extent; ++index)
                EXPECT_FLOAT_EQ(output.values[index], weightValues[index] * gain + static_cast<float>(extent));
        }
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCppApi, TypeVariantsSelectParamsValuesByIdentity) {
    const std::string manifest = readFile(VERNON_TYPE_SPECIALIZATION_MANIFEST);
    ASSERT_FALSE(manifest.empty());
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    {
        auto assetResult = vernon::runtime::ProgramAsset::load(context, manifest.data(), manifest.size());
        ASSERT_TRUE(assetResult);
        auto asset = std::move(assetResult).value();

        vernon::runtime::ProgramVariant empty;
        EXPECT_TRUE(empty.setType("MATERIAL", "").isErr());
        vernon::runtime::ProgramVariant unknown;
        ASSERT_TRUE(unknown.setType("MATERIAL", "source.tests.fixtures.type_specialization_fixture_materials:Lookup"));
        auto unknownResult = asset.resolve(unknown);
        ASSERT_TRUE(unknownResult.isErr());
        EXPECT_EQ(unknownResult.error().code, vernon::RuntimeErrorCode::ParseFailure);

        struct Material {
            const char *identity;
            bool glossy;
        };
        constexpr uint32_t extent = 3;
        for (const Material material :
             {Material{"source.tests.fixtures.type_specialization_fixture_materials:Matte", false},
              Material{"source.tests.fixtures.type_specialization_fixture_materials:Glossy", true}}) {
            SCOPED_TRACE(material.identity);
            vernon::runtime::ProgramVariant variant;
            ASSERT_TRUE(variant.setType("MATERIAL", material.identity));
            auto executableResult = asset.resolve(variant);
            ASSERT_TRUE(executableResult);
            auto executable = std::move(executableResult).value();
            VernonProgramParameterView offsetParameter{};
            const bool hasOffset = vernonRuntimeProgramExecutableFindParameter(
                                       executable.get(), {"material.offset", 15}, &offsetParameter) == VERNON_STATUS_OK;
            ASSERT_EQ(hasOffset, material.glossy);

            LinkedProgram linked(context, executable.get());
            ASSERT_TRUE(linked) << linked.error();
            constexpr float gain = 3.0f;
            constexpr float offset = 0.5f;
            HostFloats weights({1.0f, 2.0f, 4.0f});
            HostFloats output{std::vector<float>(extent)};
            linked.record().bindScalar("material.gain", gain);
            if (material.glossy)
                linked.record().bindScalar("material.offset", offset);
            linked.record().bind("material.lookup.weights", weights.view());
            linked.record().bind("output", output.view());
            if (const std::string published = publishedOutput(executable.get()); published != "output")
                linked.record().bind(published, output.view());
            linked.record().bindGrid(extent);
            ASSERT_TRUE(linked.record().unbound().empty());
            ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();
            for (uint32_t index = 0; index < extent; ++index)
                EXPECT_FLOAT_EQ(output.values[index], weights.values[index] * gain + (material.glossy ? offset : 0.0f));
        }
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCApi, TypeSpecializationSelectorRequiresAnIdentity) {
    const std::string manifest = readFile(VERNON_TYPE_SPECIALIZATION_MANIFEST);
    ASSERT_FALSE(manifest.empty());
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), nullptr);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    const std::string name = "MATERIAL";
    VernonProgramSpecialization specialization{};
    specialization.struct_size = sizeof(specialization);
    specialization.name = {name.data(), name.size()};
    specialization.kind = VERNON_PROGRAM_SPECIALIZATION_TYPE;
    const VernonProgramVariantSelector selector{sizeof(VernonProgramVariantSelector), &specialization, 1, {}};
    VernonProgramExecutable *executable = nullptr;
    EXPECT_EQ(vernonRuntimeResolveProgramResult(bundle, &selector, &executable),
              VERNON_RUNTIME_OPERATION_INVALID_ARGUMENT);
    EXPECT_EQ(executable, nullptr);
    EXPECT_EQ(lastError(context), "Program type specialization requires a type identity");

    const std::string identity = "source.tests.fixtures.type_specialization_fixture_materials:Matte";
    specialization.value.type_value = {identity.data(), identity.size()};
    ASSERT_EQ(vernonRuntimeResolveProgramResult(bundle, &selector, &executable), VERNON_RUNTIME_OPERATION_OK)
        << lastError(context);
    ASSERT_NE(executable, nullptr);
    vernonRuntimeProgramExecutableDestroy(executable);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCppApi, ProgramGraphValueConnectsProducerToConsumer) {
    const std::string manifest = readFile(VERNON_MODULE_PROGRAM_MANIFEST);
    ASSERT_FALSE(manifest.empty());
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    {
        auto assetResult = vernon::runtime::ProgramAsset::load(context, manifest.data(), manifest.size());
        ASSERT_TRUE(assetResult);
        auto asset = std::move(assetResult).value();
        auto graphResult = vernon::runtime::ProgramGraph::create(context);
        ASSERT_TRUE(graphResult);
        auto graph = std::move(graphResult).value();
        auto producerResult = graph.add(asset);
        ASSERT_TRUE(producerResult);
        const auto producer = std::move(producerResult).value();
        auto consumerResult = graph.add(asset);
        ASSERT_TRUE(consumerResult);
        const auto consumer = std::move(consumerResult).value();
        auto producerSource = producer.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "source");
        ASSERT_TRUE(producerSource);
        auto producerOutput = producer.boundary(VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
        ASSERT_TRUE(producerOutput);
        auto consumerSource = consumer.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "source");
        ASSERT_TRUE(consumerSource);
        auto consumerOutput = consumer.boundary(VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
        ASSERT_TRUE(consumerOutput);
        auto valueResult = graph.createValue(producerOutput.value());
        ASSERT_TRUE(valueResult);
        ASSERT_TRUE(graph.connect(std::move(valueResult).value(), consumerSource.value()));
        ASSERT_TRUE(graph.exportBoundary(producerSource.value(), "source"));
        ASSERT_TRUE(graph.exportBoundary(consumerOutput.value(), "output"));
        auto executableResult = graph.compile();
        ASSERT_TRUE(executableResult);
        auto executable = std::move(executableResult).value();

        LinkedProgram linked(context, executable.get());
        ASSERT_TRUE(linked) << linked.error();
        HostFloats source({2.0f});
        HostFloats output({0.0f});
        linked.record().bind("source", source.view());
        linked.record().bind("output", output.view());
        linked.record().bindGrid(1);
        ASSERT_TRUE(linked.record().unbound().empty());
        ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();
        EXPECT_FLOAT_EQ(output.values[0], 16.0f);
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}
