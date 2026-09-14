#include "VernonRuntime.hpp"
#include "program_fixture_manifest_table.h"
#include "runtime/autodiff/runtime_autodiff_telemetry.h"
#include "runtime/runtime_state.h"
#include "runtime/runtime_test_hooks.h"

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

extern "C" VernonStatus vernonRegisterTypedSpecializationFixture(void);

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

VernonProgramArgument tensorArgument(const VernonProgramParameterView &parameter, float &value) {
    static const uint64_t shape[]{1};
    static const int64_t strides[]{sizeof(float)};
    VernonProgramArgument result{};
    result.slot = parameter.slot;
    result.kind = VERNON_PROGRAM_TENSOR;
    result.tensor.struct_size = sizeof(VernonTensorView);
    result.tensor.storage = VERNON_TENSOR_HOST;
    result.tensor.host_data = &value;
    result.tensor.element_layout = parameter.element_layout;
    result.tensor.access = parameter.access;
    result.tensor.rank = 1;
    result.tensor.shape = shape;
    result.tensor.byte_strides = strides;
    result.tensor.byte_size = sizeof(value);
    return result;
}

VernonProgramArgument tensorArrayArgument(const VernonProgramParameterView &parameter, float *values,
                                          const uint64_t *shape) {
    VernonProgramArgument result{};
    result.slot = parameter.slot;
    result.kind = VERNON_PROGRAM_TENSOR;
    result.tensor.struct_size = sizeof(VernonTensorView);
    result.tensor.storage = VERNON_TENSOR_HOST;
    result.tensor.host_data = values;
    result.tensor.element_layout = parameter.element_layout;
    result.tensor.access = parameter.access;
    result.tensor.rank = 1;
    result.tensor.shape = shape;
    static const int64_t stride = sizeof(float);
    result.tensor.byte_strides = &stride;
    result.tensor.byte_size = *shape * sizeof(float);
    return result;
}

VernonProgramArgument scalarArgument(const VernonProgramParameterView &parameter, uint32_t &value) {
    VernonProgramArgument result{};
    result.slot = parameter.slot;
    result.kind = VERNON_PROGRAM_TENSOR;
    result.tensor.struct_size = sizeof(VernonTensorView);
    result.tensor.storage = VERNON_TENSOR_HOST;
    result.tensor.host_data = &value;
    result.tensor.element_layout = parameter.element_layout;
    result.tensor.access = parameter.access;
    result.tensor.byte_size = sizeof(value);
    return result;
}

VernonProgramArgument scalarArgument(const VernonProgramParameterView &parameter, float &value) {
    VernonProgramArgument result{};
    result.slot = parameter.slot;
    result.kind = VERNON_PROGRAM_TENSOR;
    result.tensor.struct_size = sizeof(VernonTensorView);
    result.tensor.storage = VERNON_TENSOR_HOST;
    result.tensor.host_data = &value;
    result.tensor.element_layout = parameter.element_layout;
    result.tensor.access = parameter.access;
    result.tensor.byte_size = sizeof(value);
    return result;
}

VernonProgramBindingToken bindingToken(const char *value) {
    return {sizeof(VernonProgramBindingToken), value, std::strlen(value)};
}

const vernon::tests::ProgramFixtureManifest &cpuFixture(std::string_view fixtureId) {
    const auto *fixture = vernon::tests::findProgramFixtureManifest(fixtureId, VERNON_RUNTIME_CPU);
    if (!fixture)
        throw std::logic_error("missing CPU fixture '" + std::string(fixtureId) + "'");
    return *fixture;
}

std::string fixtureManifest(std::string_view fixtureId) {
    const auto &fixture = cpuFixture(fixtureId);
    if (fixture.prepare() != VERNON_STATUS_OK)
        throw std::runtime_error("cannot register CPU fixture '" + std::string(fixtureId) + "'");
    std::ifstream input(std::string(fixture.manifestPath), std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read CPU fixture manifest '" + std::string(fixture.manifestPath) + "'");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::atomic<bool> throwOnLeaseRelease{};

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

TEST(RuntimeModuleProgramCApi, ForeignLeaseCallbacksHaveImmediateFailureBoundaries) {
    const std::string manifest = fixtureManifest("module_program");
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), nullptr);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(bundle, nullptr);
    ASSERT_NE(pipeline, nullptr) << lastError(context);
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(pipeline);
    ASSERT_NE(instance, nullptr);
    const VernonProgramParameterView sourceParameter = parameter(pipeline, "source");
    float source = 3.0f;
    VernonProgramArgument sourceArgument = tensorArgument(sourceParameter, source);
    const VernonProgramBindingToken token = bindingToken("throwing-lease");

    struct LeaseState {
        uint32_t retains{};
        uint32_t releases{};
    } failed;
    const VernonProgramResourceLease throwingRetain{
        sizeof(VernonProgramResourceLease),
        &failed,
        [](void *object) {
            ++static_cast<LeaseState *>(object)->retains;
            throw std::runtime_error("retain failed");
        },
        [](void *object) { ++static_cast<LeaseState *>(object)->releases; },
    };
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    EXPECT_EQ(vernonRuntimeProgramInvocationBind(invocation, &token, &sourceArgument, &throwingRetain, 0, 0),
              VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(lastError(context), "Program resource retain callback failed");
    EXPECT_EQ(failed.retains, 1u);
    EXPECT_EQ(failed.releases, 0u);
    vernonRuntimeProgramInvocationRollback(invocation);
    vernonRuntimeProgramInvocationDestroy(invocation);
    EXPECT_EQ(failed.releases, 0u);

    LeaseState retained;
    const VernonProgramResourceLease throwingRelease{
        sizeof(VernonProgramResourceLease),
        &retained,
        [](void *object) { ++static_cast<LeaseState *>(object)->retains; },
        [](void *object) {
            ++static_cast<LeaseState *>(object)->releases;
            if (throwOnLeaseRelease.load(std::memory_order_relaxed))
                throw std::runtime_error("release failed");
        },
    };
    invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &token, &sourceArgument, &throwingRelease, 0, 0),
              VERNON_STATUS_OK);
    EXPECT_EQ(retained.retains, 1u);
    throwOnLeaseRelease.store(true, std::memory_order_relaxed);
    EXPECT_DEATH_IF_SUPPORTED(vernonRuntimeProgramInvocationRollback(invocation), "");
    throwOnLeaseRelease.store(false, std::memory_order_relaxed);
    vernonRuntimeProgramInvocationRollback(invocation);
    vernonRuntimeProgramInvocationDestroy(invocation);
    EXPECT_EQ(retained.releases, 1u);

    vernonRuntimeProgramInstanceDestroy(instance);
    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCApi, ComputeModuleForward9AndVjpGradient6ThroughPublicLifecycle) {
    ASSERT_EQ(cpuFixture("module_program").prepare(), VERNON_STATUS_OK);
    const std::filesystem::path manifestPath = VERNON_MODULE_PROGRAM_MANIFEST;
    std::ifstream input(manifestPath, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), nullptr);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(bundle, nullptr);
    ASSERT_NE(pipeline, nullptr) << lastError(context);
    ASSERT_EQ(vernonRuntimeProgramExecutableGetParameterCount(pipeline), 2u);
    ASSERT_EQ(vernonRuntimeProgramExecutableHasProgramAutodiff(pipeline), 1u);

    const VernonProgramParameterView sourceParameter = parameter(pipeline, "source");
    const VernonProgramParameterView outputParameter = parameter(pipeline, "output");
    ASSERT_EQ(sourceParameter.kind, VERNON_PROGRAM_TENSOR);
    ASSERT_EQ(outputParameter.kind, VERNON_PROGRAM_TENSOR);

    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(pipeline);
    ASSERT_NE(instance, nullptr);
    float source = 3.0f;
    float output = 0.0f;
    VernonProgramArgument sourceArgument = tensorArgument(sourceParameter, source);
    VernonProgramArgument outputArgument = tensorArgument(outputParameter, output);
    const VernonProgramBindingToken sourceToken = bindingToken("source-v1");
    const VernonProgramBindingToken outputToken = bindingToken("output-v1");
    const VernonProgramBindingToken renderPassToken = bindingToken("render-pass-v1");
    const VernonProgramBindingToken dynamicStateToken = bindingToken("dynamic-state-v1");
    int controlLeaseCount = 0;
    VernonProgramResourceLease controlLease{
        sizeof(VernonProgramResourceLease),
        &controlLeaseCount,
        [](void *value) { ++*static_cast<int *>(value); },
        [](void *value) { --*static_cast<int *>(value); },
    };
    VernonRenderPass renderPass{};
    renderPass.struct_size = sizeof(renderPass);
    VernonDynamicState dynamicState{};
    dynamicState.struct_size = sizeof(dynamicState);
    dynamicState.viewport[2] = 1;
    dynamicState.viewport[3] = 1;

    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &sourceToken, &sourceArgument, nullptr, sizeof(source), 1),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &outputToken, &outputArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    ASSERT_EQ(
        vernonRuntimeProgramInvocationBindRenderPass(invocation, 0, &renderPassToken, &renderPass, &controlLease, 1),
        VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeProgramInvocationBindDynamicState(invocation, 0, &dynamicStateToken, &dynamicState),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(controlLeaseCount, 0);
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonRuntimeProgramInvocationExecute(invocation, 1, nullptr), VERNON_STATUS_OK) << lastError(context);
    EXPECT_EQ(pullback, nullptr);
    VernonProgramBindingTelemetry stagedTelemetry{};
    stagedTelemetry.struct_size = sizeof(stagedTelemetry);
    ASSERT_EQ(vernonRuntimeProgramInstanceGetTelemetry(instance, &stagedTelemetry), VERNON_STATUS_OK);
    EXPECT_EQ(stagedTelemetry.prepare_count, 0u);
    EXPECT_EQ(vernonRuntimeProgramInvocationBind(invocation, &sourceToken, &sourceArgument, nullptr, 0, 0),
              VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeProgramInvocationCommit(invocation, &pullback), VERNON_STATUS_OK) << lastError(context);
    vernonRuntimeProgramInvocationDestroy(invocation);
    ASSERT_NE(pullback, nullptr);
    EXPECT_FLOAT_EQ(output, 9.0f);

    float seedValue = 1.0f;
    float gradientValue = 0.0f;
    VernonProgramParameterView cotangent{};
    VernonProgramParameterView gradient{};
    ASSERT_EQ(
        vernonRuntimeProgramExecutableGetBoundaryByIndex(pipeline, VERNON_PROGRAM_BOUNDARY_COTANGENT, 0, &cotangent),
        VERNON_STATUS_OK);
    ASSERT_EQ(
        vernonRuntimeProgramExecutableGetBoundaryByIndex(pipeline, VERNON_PROGRAM_BOUNDARY_GRADIENT, 0, &gradient),
        VERNON_STATUS_OK);
    VernonProgramInvocation *privateBoundaryInvocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(privateBoundaryInvocation, nullptr);
    const VernonProgramArgument privateBoundaryArgument = tensorArgument(cotangent, seedValue);
    uint8_t reused = 1;
    EXPECT_EQ(
        vernonRuntimeProgramInvocationTryReuse(privateBoundaryInvocation, cotangent.slot, &sourceToken, 0, 0, &reused),
        VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(reused, 0);
    EXPECT_EQ(vernonRuntimeProgramInvocationBind(privateBoundaryInvocation, &sourceToken, &privateBoundaryArgument,
                                                 nullptr, 0, 0),
              VERNON_STATUS_INVALID_ARGUMENT);
    vernonRuntimeProgramInvocationRollback(privateBoundaryInvocation);
    vernonRuntimeProgramInvocationDestroy(privateBoundaryInvocation);
    VernonProgramArgument derivativeArguments[]{
        tensorArgument(cotangent, seedValue),
        tensorArgument(gradient, gradientValue),
    };
    ASSERT_EQ(vernonProgramPullbackApply(pullback, derivativeArguments, std::size(derivativeArguments), nullptr),
              VERNON_STATUS_OK)
        << lastError(context);
    EXPECT_FLOAT_EQ(gradientValue, 6.0f);
    vernonProgramPullbackDestroy(pullback);

    VernonProgramInvocation *minimumMemory = vernonRuntimeProgramInstanceBeginInvocation(instance);
    VernonProgramInvocation *minimumRuntime = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(minimumMemory, nullptr);
    ASSERT_NE(minimumRuntime, nullptr);
    const auto configureCheckpointing = [](VernonProgramInvocation *configured, uint64_t budget,
                                           std::string_view policy) {
        VernonProgramAutodiffInvocationOptions options{};
        options.struct_size = sizeof(options);
        options.abi_version = VERNON_PROGRAM_AUTODIFF_INVOCATION_OPTIONS_VERSION;
        options.has_checkpoint_memory_budget = 1;
        options.checkpoint_memory_budget = budget;
        options.checkpoint_policy = {policy.data(), policy.size()};
        return vernonRuntimeProgramInvocationSetAutodiffOptions(configured, &options);
    };
    VernonProgramAutodiffInvocationOptions invalidOptions{};
    invalidOptions.struct_size = sizeof(invalidOptions);
    invalidOptions.abi_version = VERNON_PROGRAM_AUTODIFF_INVOCATION_OPTIONS_VERSION;
    invalidOptions.reserved_bytes[0] = 1;
    EXPECT_EQ(vernonRuntimeProgramInvocationSetAutodiffOptions(minimumMemory, &invalidOptions),
              VERNON_STATUS_INVALID_ARGUMENT);
    invalidOptions.reserved_bytes[0] = 0;
    invalidOptions.reserved[0] = 1;
    EXPECT_EQ(vernonRuntimeProgramInvocationSetAutodiffOptions(minimumMemory, &invalidOptions),
              VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(configureCheckpointing(minimumMemory, 1u << 20, "min_memory"), VERNON_STATUS_OK);
    ASSERT_EQ(configureCheckpointing(minimumRuntime, 2u << 20, "min_runtime"), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationExecute(minimumMemory, 1, nullptr), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationExecute(minimumRuntime, 1, nullptr), VERNON_STATUS_OK);
    VernonPullback *minimumMemoryPullback = nullptr;
    VernonPullback *minimumRuntimePullback = nullptr;
    ASSERT_EQ(vernonRuntimeProgramInvocationCommit(minimumMemory, &minimumMemoryPullback), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationCommit(minimumRuntime, &minimumRuntimePullback), VERNON_STATUS_OK);
    const auto minimumMemoryPlan = vernon::runtime::autodiffPullbackCheckpointPlan(minimumMemoryPullback);
    const auto minimumRuntimePlan = vernon::runtime::autodiffPullbackCheckpointPlan(minimumRuntimePullback);
    EXPECT_EQ(minimumMemoryPlan.memoryBudget, 1u << 20);
    EXPECT_EQ(minimumMemoryPlan.selectedPolicy, "min_memory");
    EXPECT_EQ(minimumRuntimePlan.memoryBudget, 2u << 20);
    EXPECT_EQ(minimumRuntimePlan.selectedPolicy, "min_runtime");
    vernonProgramPullbackDestroy(minimumMemoryPullback);
    vernonProgramPullbackDestroy(minimumRuntimePullback);
    vernonRuntimeProgramInvocationDestroy(minimumMemory);
    vernonRuntimeProgramInvocationDestroy(minimumRuntime);

    invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &sourceToken, &sourceArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &outputToken, &outputArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationExecute(invocation, 0, nullptr), VERNON_STATUS_OK) << lastError(context);
    ASSERT_EQ(vernonRuntimeProgramInvocationCommit(invocation, nullptr), VERNON_STATUS_OK) << lastError(context);
    vernonRuntimeProgramInvocationDestroy(invocation);
    EXPECT_FLOAT_EQ(output, 9.0f);

    VernonProgramBindingTelemetry telemetry{};
    telemetry.struct_size = sizeof(telemetry);
    ASSERT_EQ(vernonRuntimeProgramInstanceGetTelemetry(instance, &telemetry), VERNON_STATUS_OK);
    EXPECT_EQ(telemetry.prepare_count, 2u);
    EXPECT_EQ(telemetry.reuse_count, 2u);
    EXPECT_EQ(telemetry.upload_bytes, sizeof(source));
    EXPECT_EQ(telemetry.upload_ranges, 1u);
    EXPECT_EQ(controlLeaseCount, 0);

    vernonRuntimeProgramInstanceDestroy(instance);
    EXPECT_EQ(controlLeaseCount, 0);
    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCApi, ProgramGraphRetainsNodeLocalPullbackWithoutCompositeAutodiff) {
    ASSERT_EQ(cpuFixture("module_program").prepare(), VERNON_STATUS_OK);
    std::ifstream input(VERNON_MODULE_PROGRAM_MANIFEST, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), nullptr);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonProgramExecutable *standalone = vernonRuntimeResolveProgram(bundle, nullptr);
    ASSERT_NE(standalone, nullptr) << lastError(context);

    VernonProgramGraph *graph = vernonRuntimeProgramGraphCreate(context);
    ASSERT_NE(graph, nullptr);
    VernonProgramNodeId firstNode = UINT32_MAX;
    VernonProgramNodeId secondNode = UINT32_MAX;
    ASSERT_EQ(vernonRuntimeProgramGraphAddProgram(graph, bundle, nullptr, &firstNode), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramGraphAddProgram(graph, bundle, nullptr, &secondNode), VERNON_STATUS_OK);
    VernonProgramNodeBindingToken firstSource{};
    VernonProgramNodeBindingToken firstOutput{};
    VernonProgramNodeBindingToken secondSource{};
    VernonProgramNodeBindingToken secondOutput{};
    firstSource.struct_size = sizeof(firstSource);
    firstOutput.struct_size = sizeof(firstOutput);
    secondSource.struct_size = sizeof(secondSource);
    secondOutput.struct_size = sizeof(secondOutput);
    ASSERT_EQ(vernonRuntimeProgramGraphFindBoundary(graph, firstNode, VERNON_PROGRAM_BOUNDARY_INPUT,
                                                    {"source", std::strlen("source")}, &firstSource),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramGraphFindBoundary(graph, firstNode, VERNON_PROGRAM_BOUNDARY_OUTPUT,
                                                    {"output", std::strlen("output")}, &firstOutput),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramGraphFindBoundary(graph, secondNode, VERNON_PROGRAM_BOUNDARY_INPUT,
                                                    {"source", std::strlen("source")}, &secondSource),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramGraphFindBoundary(graph, secondNode, VERNON_PROGRAM_BOUNDARY_OUTPUT,
                                                    {"output", std::strlen("output")}, &secondOutput),
              VERNON_STATUS_OK);
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

    float source = 3.0f;
    float output = 0.0f;
    VernonProgramArgument sourceArgument = tensorArgument(parameter(composite, "source"), source);
    VernonProgramArgument outputArgument = tensorArgument(parameter(composite, "output"), output);
    const VernonProgramBindingToken sourceToken = bindingToken("composite-source");
    const VernonProgramBindingToken outputToken = bindingToken("composite-output");
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(composite);
    ASSERT_NE(instance, nullptr);
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &sourceToken, &sourceArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &outputToken, &outputArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    VernonPullback *compositePullback = reinterpret_cast<VernonPullback *>(uintptr_t{1});
    ASSERT_EQ(vernonRuntimeProgramInvocationExecute(invocation, 1, nullptr), VERNON_STATUS_OK) << lastError(context);
    EXPECT_EQ(vernonRuntimeProgramInvocationGetNodePullback(invocation, firstNode, &compositePullback),
              VERNON_STATUS_INVALID_ARGUMENT);
    compositePullback = reinterpret_cast<VernonPullback *>(uintptr_t{1});
    ASSERT_EQ(vernonRuntimeProgramInvocationCommit(invocation, &compositePullback), VERNON_STATUS_OK)
        << lastError(context);
    EXPECT_EQ(compositePullback, nullptr);
    EXPECT_FLOAT_EQ(output, 81.0f);

    VernonPullback *firstPullback = nullptr;
    VernonPullback *secondPullback = nullptr;
    ASSERT_EQ(vernonRuntimeProgramInvocationGetNodePullback(invocation, firstNode, &firstPullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_EQ(vernonRuntimeProgramInvocationGetNodePullback(invocation, secondNode, &secondPullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(firstPullback, nullptr);
    ASSERT_NE(secondPullback, nullptr);
    vernonRuntimeProgramInvocationDestroy(invocation);

    VernonProgramParameterView cotangent{};
    VernonProgramParameterView gradient{};
    ASSERT_EQ(
        vernonRuntimeProgramExecutableGetBoundaryByIndex(standalone, VERNON_PROGRAM_BOUNDARY_COTANGENT, 0, &cotangent),
        VERNON_STATUS_OK);
    ASSERT_EQ(
        vernonRuntimeProgramExecutableGetBoundaryByIndex(standalone, VERNON_PROGRAM_BOUNDARY_GRADIENT, 0, &gradient),
        VERNON_STATUS_OK);
    float seed = 1.0f;
    float intermediateCotangent = 0.0f;
    VernonProgramArgument secondDerivatives[]{
        tensorArgument(cotangent, seed),
        tensorArgument(gradient, intermediateCotangent),
    };
    ASSERT_EQ(vernonProgramPullbackApply(secondPullback, secondDerivatives, std::size(secondDerivatives), nullptr),
              VERNON_STATUS_OK)
        << lastError(context);
    EXPECT_FLOAT_EQ(intermediateCotangent, 18.0f);
    float result = 0.0f;
    VernonProgramArgument firstDerivatives[]{
        tensorArgument(cotangent, intermediateCotangent),
        tensorArgument(gradient, result),
    };
    ASSERT_EQ(vernonProgramPullbackApply(firstPullback, firstDerivatives, std::size(firstDerivatives), nullptr),
              VERNON_STATUS_OK)
        << lastError(context);
    EXPECT_FLOAT_EQ(result, 108.0f);

    vernonProgramPullbackDestroy(secondPullback);
    vernonProgramPullbackDestroy(firstPullback);
    vernonRuntimeProgramInstanceDestroy(instance);
    vernonRuntimeProgramExecutableDestroy(composite);
    vernonRuntimeProgramGraphDestroy(graph);
    vernonRuntimeProgramExecutableDestroy(standalone);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCApi, LoadsCanonicalBundleThroughBundleThenResolve) {
    ASSERT_EQ(cpuFixture("module_program").prepare(), VERNON_STATUS_OK);
    std::ifstream input(VERNON_MODULE_PROGRAM_MANIFEST, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};

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

    const VernonProgramParameterView sourceParameter = parameter(pipeline, "source");
    const VernonProgramParameterView outputParameter = parameter(pipeline, "output");
    float source = 5.0f;
    float output = 0.0f;
    VernonProgramArgument sourceArgument = tensorArgument(sourceParameter, source);
    VernonProgramArgument outputArgument = tensorArgument(outputParameter, output);
    const VernonProgramBindingToken sourceToken = bindingToken("resolve-source-v1");
    const VernonProgramBindingToken outputToken = bindingToken("resolve-output-v1");

    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(pipeline);
    ASSERT_NE(instance, nullptr);
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &sourceToken, &sourceArgument, nullptr, sizeof(source), 1),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &outputToken, &outputArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationExecute(invocation, 0, nullptr), VERNON_STATUS_OK) << lastError(context);
    ASSERT_EQ(vernonRuntimeProgramInvocationCommit(invocation, nullptr), VERNON_STATUS_OK) << lastError(context);
    vernonRuntimeProgramInvocationDestroy(invocation);
    EXPECT_FLOAT_EQ(output, 25.0f);

    vernonRuntimeProgramInstanceDestroy(instance);
    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCApi, RejectsLegacyEnvelopeBeforeArtifactIo) {
    std::ifstream input(VERNON_MODULE_PROGRAM_MANIFEST, std::ios::binary);
    ASSERT_TRUE(input);
    std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
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
    std::ifstream input(VERNON_MODULE_PROGRAM_MANIFEST, std::ios::binary);
    ASSERT_TRUE(input);
    std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
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

TEST(RuntimeModuleProgramCppApi, RetainsExecutableForPersistentInstance) {
    ASSERT_EQ(cpuFixture("module_program").prepare(), VERNON_STATUS_OK);
    std::ifstream input(VERNON_MODULE_PROGRAM_MANIFEST, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    {
        auto assetResult = vernon::runtime::ProgramAsset::load(context, manifest.data(), manifest.size());
        ASSERT_TRUE(assetResult);
        auto asset = std::move(assetResult).value();
        auto executableResult = asset.resolve();
        ASSERT_TRUE(executableResult);
        auto executable = std::move(executableResult).value();
        const VernonProgramParameterView sourceParameter = parameter(executable.get(), "source");
        const VernonProgramParameterView outputParameter = parameter(executable.get(), "output");
        auto instanceResult = vernon::runtime::ProgramInstance::create(executable);
        ASSERT_TRUE(instanceResult);
        auto instance = std::move(instanceResult).value();
        executable = {};

        float source = 4.0f;
        float output = 0.0f;
        const VernonProgramArgument sourceArgument = tensorArgument(sourceParameter, source);
        const VernonProgramArgument outputArgument = tensorArgument(outputParameter, output);
        const VernonProgramBindingToken sourceToken = bindingToken("cpp-source-v1");
        const VernonProgramBindingToken outputToken = bindingToken("cpp-output-v1");
        auto invocationResult = instance.begin();
        ASSERT_TRUE(invocationResult);
        auto invocation = std::move(invocationResult).value();
        ASSERT_TRUE(invocation.bind(sourceToken, sourceArgument, nullptr, sizeof(source), 1));
        ASSERT_TRUE(invocation.bind(outputToken, outputArgument));
        ASSERT_TRUE(invocation.execute(false));
        auto commitResult = invocation.commit();
        ASSERT_TRUE(commitResult);
        EXPECT_FALSE(commitResult.value());
        EXPECT_FLOAT_EQ(output, 16.0f);
        auto telemetryResult = instance.telemetry();
        ASSERT_TRUE(telemetryResult);
        const VernonProgramBindingTelemetry telemetry = telemetryResult.value();
        EXPECT_EQ(telemetry.prepare_count, 2u);
        EXPECT_EQ(telemetry.upload_bytes, sizeof(source));
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCppApi, PreservesLoadAndResolveFailureIdentity) {
    static_assert(noexcept(vernon::runtime::ProgramAsset::load(nullptr, nullptr, 0)));
    static_assert(noexcept(std::declval<const vernon::runtime::ProgramAsset &>().resolve()));
    static_assert(noexcept(vernon::runtime::ProgramGraph::create(nullptr)));
    static_assert(
        noexcept(vernon::runtime::ProgramInstance::create(std::declval<const vernon::runtime::ProgramExecutable &>())));
    static_assert(noexcept(std::declval<vernon::runtime::ProgramInstance &>().begin()));
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

TEST(RuntimeModuleProgramCppApi, ProgramGraphExecutesThroughCanonicalExportedBoundaries) {
    ASSERT_EQ(cpuFixture("module_program").prepare(), VERNON_STATUS_OK);
    std::ifstream input(VERNON_MODULE_PROGRAM_MANIFEST, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    {
        auto assetResult = vernon::runtime::ProgramAsset::load(context, manifest.data(), manifest.size());
        ASSERT_TRUE(assetResult);
        auto asset = std::move(assetResult).value();
        auto standaloneResult = asset.resolve();
        ASSERT_TRUE(standaloneResult);
        auto standalone = std::move(standaloneResult).value();
        VernonProgramParameterView cotangent{};
        VernonProgramParameterView gradient{};
        ASSERT_EQ(vernonRuntimeProgramExecutableGetBoundaryByIndex(standalone.get(), VERNON_PROGRAM_BOUNDARY_COTANGENT,
                                                                   0, &cotangent),
                  VERNON_STATUS_OK);
        ASSERT_EQ(vernonRuntimeProgramExecutableGetBoundaryByIndex(standalone.get(), VERNON_PROGRAM_BOUNDARY_GRADIENT,
                                                                   0, &gradient),
                  VERNON_STATUS_OK);
        auto graphResult = vernon::runtime::ProgramGraph::create(context);
        ASSERT_TRUE(graphResult);
        auto graph = std::move(graphResult).value();
        auto firstResult = graph.add(asset);
        ASSERT_TRUE(firstResult);
        const auto first = std::move(firstResult).value();
        auto secondResult = graph.add(asset);
        ASSERT_TRUE(secondResult);
        const auto second = std::move(secondResult).value();
        auto firstSourceResult = first.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "source");
        ASSERT_TRUE(firstSourceResult);
        const auto firstSource = std::move(firstSourceResult).value();
        auto firstOutputResult = first.boundary(VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
        ASSERT_TRUE(firstOutputResult);
        const auto firstOutput = std::move(firstOutputResult).value();
        auto secondSourceResult = second.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "source");
        ASSERT_TRUE(secondSourceResult);
        const auto secondSource = std::move(secondSourceResult).value();
        auto secondOutputResult = second.boundary(VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
        ASSERT_TRUE(secondOutputResult);
        const auto secondOutput = std::move(secondOutputResult).value();
        ASSERT_TRUE(graph.exportBoundary(firstSource, "first_source"));
        ASSERT_TRUE(graph.exportBoundary(firstOutput, "first_output"));
        ASSERT_TRUE(graph.exportBoundary(secondSource, "second_source"));
        ASSERT_TRUE(graph.exportBoundary(secondOutput, "second_output"));
        auto executableResult = graph.compile();
        ASSERT_TRUE(executableResult);
        auto executable = std::move(executableResult).value();
        auto instanceResult = vernon::runtime::ProgramInstance::create(executable);
        ASSERT_TRUE(instanceResult);
        auto instance = std::move(instanceResult).value();
        float firstSourceValue = 2.0f;
        float firstOutputValue = 0.0f;
        float secondSourceValue = 3.0f;
        float secondOutputValue = 0.0f;
        auto invocationResult = instance.begin();
        ASSERT_TRUE(invocationResult);
        auto invocation = std::move(invocationResult).value();
        ASSERT_TRUE(invocation.bind(bindingToken("first-source"),
                                    tensorArgument(parameter(executable.get(), "first_source"), firstSourceValue)));
        ASSERT_TRUE(invocation.bind(bindingToken("first-output"),
                                    tensorArgument(parameter(executable.get(), "first_output"), firstOutputValue)));
        ASSERT_TRUE(invocation.bind(bindingToken("second-source"),
                                    tensorArgument(parameter(executable.get(), "second_source"), secondSourceValue)));
        ASSERT_TRUE(invocation.bind(bindingToken("second-output"),
                                    tensorArgument(parameter(executable.get(), "second_output"), secondOutputValue)));
        ASSERT_TRUE(invocation.execute(true));
        auto earlyPullback = invocation.pullback(first);
        ASSERT_TRUE(earlyPullback.isErr());
        EXPECT_EQ(earlyPullback.error().code, vernon::RuntimeErrorCode::InvalidArgument);
        EXPECT_STREQ(earlyPullback.error().context.operation, "ProgramInvocation.pullback");
        EXPECT_EQ(earlyPullback.error().context.value, 0u);
        EXPECT_EQ(earlyPullback.error().context.detail, 0u);
        auto commitResult = invocation.commit();
        ASSERT_TRUE(commitResult);
        EXPECT_FALSE(commitResult.value());
        EXPECT_FLOAT_EQ(firstOutputValue, 4.0f);
        EXPECT_FLOAT_EQ(secondOutputValue, 9.0f);
        auto otherGraphResult = vernon::runtime::ProgramGraph::create(context);
        ASSERT_TRUE(otherGraphResult);
        auto otherGraph = std::move(otherGraphResult).value();
        auto foreignNodeResult = otherGraph.add(asset);
        ASSERT_TRUE(foreignNodeResult);
        const auto foreignNode = std::move(foreignNodeResult).value();
        auto foreignPullback = invocation.pullback(foreignNode);
        ASSERT_TRUE(foreignPullback.isErr());
        EXPECT_EQ(foreignPullback.error().code, vernon::RuntimeErrorCode::InvalidArgument);
        EXPECT_STREQ(foreignPullback.error().context.operation, "ProgramInvocation.pullback");
        EXPECT_EQ(foreignPullback.error().context.value, 0u);
        EXPECT_EQ(foreignPullback.error().context.detail, 0u);
        auto firstPullbackResult = invocation.pullback(first);
        ASSERT_TRUE(firstPullbackResult);
        auto firstPullback = std::move(firstPullbackResult).value();
        auto secondPullbackResult = invocation.pullback(second);
        ASSERT_TRUE(secondPullbackResult);
        auto secondPullback = std::move(secondPullbackResult).value();
        float seed = 1.0f;
        float firstGradient = 0.0f;
        float secondGradient = 0.0f;
        VernonProgramArgument firstDerivatives[]{
            tensorArgument(cotangent, seed),
            tensorArgument(gradient, firstGradient),
        };
        VernonProgramArgument secondDerivatives[]{
            tensorArgument(cotangent, seed),
            tensorArgument(gradient, secondGradient),
        };
        ASSERT_TRUE(firstPullback.apply(firstDerivatives, std::size(firstDerivatives)));
        EXPECT_FLOAT_EQ(firstGradient, 4.0f);
        EXPECT_FLOAT_EQ(secondGradient, 0.0f);
        ASSERT_TRUE(secondPullback.apply(secondDerivatives, std::size(secondDerivatives)));
        EXPECT_FLOAT_EQ(firstGradient, 4.0f);
        EXPECT_FLOAT_EQ(secondGradient, 6.0f);
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCppApi, TypedVariantsSelectIndependentProgramGraphNodes) {
    ASSERT_EQ(vernonRegisterTypedSpecializationFixture(), VERNON_STATUS_OK);
    std::ifstream input(VERNON_TYPED_SPECIALIZATION_MANIFEST, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
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
        auto oneBoundaryResult = oneNode.boundary(VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
        ASSERT_TRUE(oneBoundaryResult);
        const auto oneBoundary = std::move(oneBoundaryResult).value();
        auto fourBoundaryResult = fourNode.boundary(VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
        ASSERT_TRUE(fourBoundaryResult);
        const auto fourBoundary = std::move(fourBoundaryResult).value();
        auto oneGridXResult = oneNode.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "__grid_x");
        ASSERT_TRUE(oneGridXResult);
        const auto oneGridX = std::move(oneGridXResult).value();
        auto oneGridYResult = oneNode.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "__grid_y");
        ASSERT_TRUE(oneGridYResult);
        const auto oneGridY = std::move(oneGridYResult).value();
        auto oneGridZResult = oneNode.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "__grid_z");
        ASSERT_TRUE(oneGridZResult);
        const auto oneGridZ = std::move(oneGridZResult).value();
        auto fourGridXResult = fourNode.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "__grid_x");
        ASSERT_TRUE(fourGridXResult);
        const auto fourGridX = std::move(fourGridXResult).value();
        auto fourGridYResult = fourNode.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "__grid_y");
        ASSERT_TRUE(fourGridYResult);
        const auto fourGridY = std::move(fourGridYResult).value();
        auto fourGridZResult = fourNode.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "__grid_z");
        ASSERT_TRUE(fourGridZResult);
        const auto fourGridZ = std::move(fourGridZResult).value();
        ASSERT_TRUE(graph.exportBoundary(oneBoundary, "one_output"));
        ASSERT_TRUE(graph.exportBoundary(fourBoundary, "four_output"));
        ASSERT_TRUE(graph.exportBoundary(oneGridX, "one_grid_x"));
        ASSERT_TRUE(graph.exportBoundary(oneGridY, "one_grid_y"));
        ASSERT_TRUE(graph.exportBoundary(oneGridZ, "one_grid_z"));
        ASSERT_TRUE(graph.exportBoundary(fourGridX, "four_grid_x"));
        ASSERT_TRUE(graph.exportBoundary(fourGridY, "four_grid_y"));
        ASSERT_TRUE(graph.exportBoundary(fourGridZ, "four_grid_z"));
        auto graphExecutableResult = graph.compile();
        ASSERT_TRUE(graphExecutableResult);
        auto graphExecutable = std::move(graphExecutableResult).value();
        auto instanceResult = vernon::runtime::ProgramInstance::create(graphExecutable);
        ASSERT_TRUE(instanceResult);
        auto instance = std::move(instanceResult).value();
        float oneValues[1]{};
        float fourValues[4]{};
        const uint64_t oneShape[]{1};
        const uint64_t fourShape[]{4};
        uint32_t gridExtent = 1;
        auto invocationResult = instance.begin();
        ASSERT_TRUE(invocationResult);
        auto invocation = std::move(invocationResult).value();
        const size_t graphParameterCount = vernonRuntimeProgramExecutableGetParameterCount(graphExecutable.get());
        const auto bindExport = [&](std::string_view exportName, const auto &makeArgument) {
            size_t matches = 0;
            for (size_t index = 0; index < graphParameterCount; ++index) {
                VernonProgramParameterView reflected{};
                ASSERT_EQ(vernonRuntimeProgramExecutableGetParameterByIndex(graphExecutable.get(), index, &reflected),
                          VERNON_STATUS_OK);
                if (std::string_view(reflected.name.data, reflected.name.size) != exportName)
                    continue;
                const std::string tokenText = std::string(exportName) + "-" + std::to_string(reflected.slot);
                ASSERT_TRUE(invocation.bind({sizeof(VernonProgramBindingToken), tokenText.data(), tokenText.size()},
                                            makeArgument(reflected)));
                ++matches;
            }
            ASSERT_GT(matches, 0u);
        };
        bindExport("one_output", [&](const VernonProgramParameterView &reflected) {
            return tensorArrayArgument(reflected, oneValues, oneShape);
        });
        bindExport("four_output", [&](const VernonProgramParameterView &reflected) {
            return tensorArrayArgument(reflected, fourValues, fourShape);
        });
        for (const char *name : {"one_grid_x", "one_grid_y", "one_grid_z", "four_grid_x", "four_grid_y", "four_grid_z"})
            bindExport(name, [&](const VernonProgramParameterView &reflected) {
                return scalarArgument(reflected, gridExtent);
            });
        ASSERT_TRUE(invocation.execute(false));
        auto commitResult = invocation.commit();
        ASSERT_TRUE(commitResult);
        EXPECT_FALSE(commitResult.value());
        EXPECT_FLOAT_EQ(oneValues[0], 1.0f);
        EXPECT_FLOAT_EQ(fourValues[0], 4.0f);
        auto pullbackResult = invocation.pullback(oneNode);
        ASSERT_TRUE(pullbackResult.isErr());
        EXPECT_EQ(pullbackResult.error().code, vernon::RuntimeErrorCode::InvalidArgument);
        EXPECT_STREQ(pullbackResult.error().context.operation, "ProgramInvocation.pullback");
        EXPECT_EQ(pullbackResult.error().context.value, 0u);
        EXPECT_EQ(pullbackResult.error().context.detail, 0u);
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCppApi, ProgramGraphValueConnectsProducerToConsumer) {
    ASSERT_EQ(cpuFixture("module_program").prepare(), VERNON_STATUS_OK);
    std::ifstream input(VERNON_MODULE_PROGRAM_MANIFEST, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
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
        auto producerSourceResult = producer.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "source");
        ASSERT_TRUE(producerSourceResult);
        const auto producerSource = std::move(producerSourceResult).value();
        auto producerOutputResult = producer.boundary(VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
        ASSERT_TRUE(producerOutputResult);
        const auto producerOutput = std::move(producerOutputResult).value();
        auto consumerSourceResult = consumer.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "source");
        ASSERT_TRUE(consumerSourceResult);
        const auto consumerSource = std::move(consumerSourceResult).value();
        auto consumerOutputResult = consumer.boundary(VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
        ASSERT_TRUE(consumerOutputResult);
        const auto consumerOutput = std::move(consumerOutputResult).value();
        auto valueResult = graph.createValue(producerOutput);
        ASSERT_TRUE(valueResult);
        ASSERT_TRUE(graph.connect(std::move(valueResult).value(), consumerSource));
        ASSERT_TRUE(graph.exportBoundary(producerSource, "source"));
        ASSERT_TRUE(graph.exportBoundary(consumerOutput, "output"));
        auto executableResult = graph.compile();
        ASSERT_TRUE(executableResult);
        auto executable = std::move(executableResult).value();
        auto instanceResult = vernon::runtime::ProgramInstance::create(executable);
        ASSERT_TRUE(instanceResult);
        auto instance = std::move(instanceResult).value();
        float source = 2.0f;
        float output = 0.0f;
        auto invocationResult = instance.begin();
        ASSERT_TRUE(invocationResult);
        auto invocation = std::move(invocationResult).value();
        ASSERT_TRUE(
            invocation.bind(bindingToken("source"), tensorArgument(parameter(executable.get(), "source"), source)));
        ASSERT_TRUE(
            invocation.bind(bindingToken("output"), tensorArgument(parameter(executable.get(), "output"), output)));
        ASSERT_TRUE(invocation.execute(false));
        auto commitResult = invocation.commit();
        ASSERT_TRUE(commitResult);
        EXPECT_FALSE(commitResult.value());
        EXPECT_FLOAT_EQ(output, 16.0f);
        auto pullbackResult = invocation.pullback(producer);
        ASSERT_TRUE(pullbackResult.isErr());
        EXPECT_EQ(pullbackResult.error().code, vernon::RuntimeErrorCode::InvalidArgument);
        EXPECT_STREQ(pullbackResult.error().context.operation, "ProgramInvocation.pullback");
        EXPECT_EQ(pullbackResult.error().context.value, 0u);
        EXPECT_EQ(pullbackResult.error().context.detail, 0u);
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCpuMatrix, ReusesOneStageAcrossDifferentNodeProjections) {
    const std::string manifest = fixtureManifest("reused_stage");
    ASSERT_FALSE(manifest.empty());
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    {
        auto assetResult = vernon::runtime::ProgramAsset::load(context, manifest.data(), manifest.size());
        ASSERT_TRUE(assetResult);
        auto asset = std::move(assetResult).value();
        auto executableResult = asset.resolve();
        ASSERT_TRUE(executableResult);
        auto executable = std::move(executableResult).value();
        std::array<float, 4> source{1, 2, 3, 4};
        std::array<float, 4> output{};
        const uint64_t shape[]{source.size()};
        uint32_t gridX = 4;
        uint32_t gridY = 1;
        uint32_t gridZ = 1;
        std::array<VernonProgramArgument, 5> arguments{
            tensorArrayArgument(parameter(executable.get(), "source"), source.data(), shape),
            tensorArrayArgument(parameter(executable.get(), "output"), output.data(), shape),
            scalarArgument(parameter(executable.get(), "grid_x"), gridX),
            scalarArgument(parameter(executable.get(), "grid_y"), gridY),
            scalarArgument(parameter(executable.get(), "grid_z"), gridZ),
        };
        auto instanceResult = vernon::runtime::ProgramInstance::create(executable);
        ASSERT_TRUE(instanceResult);
        auto instance = std::move(instanceResult).value();
        auto invocationResult = instance.begin();
        ASSERT_TRUE(invocationResult);
        auto invocation = std::move(invocationResult).value();
        for (size_t index = 0; index < arguments.size(); ++index) {
            const std::string text = "cpu-reused-stage-" + std::to_string(index);
            ASSERT_TRUE(
                invocation.bind({sizeof(VernonProgramBindingToken), text.data(), text.size()}, arguments[index]));
        }
        ASSERT_TRUE(invocation.execute(false));
        auto commitResult = invocation.commit();
        ASSERT_TRUE(commitResult);
        EXPECT_FALSE(commitResult.value());
        EXPECT_EQ(output, (std::array<float, 4>{3, 4, 5, 6}));
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCpuMatrix, ChainsDistinctComputeKernelsThroughTensorViews) {
    const std::string manifest = fixtureManifest("tensor_view_chain");
    ASSERT_FALSE(manifest.empty());
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    {
        auto assetResult = vernon::runtime::ProgramAsset::load(context, manifest.data(), manifest.size());
        ASSERT_TRUE(assetResult);
        auto asset = std::move(assetResult).value();
        auto executableResult = asset.resolve();
        ASSERT_TRUE(executableResult);
        auto executable = std::move(executableResult).value();
        float source = 3.0f;
        float output = 0.0f;
        auto instanceResult = vernon::runtime::ProgramInstance::create(executable);
        ASSERT_TRUE(instanceResult);
        auto instance = std::move(instanceResult).value();
        auto invocationResult = instance.begin();
        ASSERT_TRUE(invocationResult);
        auto invocation = std::move(invocationResult).value();
        ASSERT_TRUE(invocation.bind(bindingToken("cpu-tensor-chain-source"),
                                    tensorArgument(parameter(executable.get(), "source"), source)));
        ASSERT_TRUE(invocation.bind(bindingToken("cpu-tensor-chain-output"),
                                    tensorArgument(parameter(executable.get(), "output"), output)));
        ASSERT_TRUE(invocation.execute(false));
        auto commitResult = invocation.commit();
        ASSERT_TRUE(commitResult);
        EXPECT_FALSE(commitResult.value());
        EXPECT_FLOAT_EQ(output, 8.0f);
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCpuMatrix, ReusesLoadedProgramAcrossDynamicShapesAndGrids) {
    const std::string manifest = fixtureManifest("dynamic_shape_grid");
    ASSERT_FALSE(manifest.empty());
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    {
        auto assetResult = vernon::runtime::ProgramAsset::load(context, manifest.data(), manifest.size());
        ASSERT_TRUE(assetResult);
        auto asset = std::move(assetResult).value();
        auto executableResult = asset.resolve();
        ASSERT_TRUE(executableResult);
        auto executable = std::move(executableResult).value();
        const VernonProgramParameterView sourceParameter = parameter(executable.get(), "source");
        const VernonProgramParameterView outputParameter = parameter(executable.get(), "output");
        VernonProgramParameterView publishedOutputParameter{};
        ASSERT_EQ(vernonRuntimeProgramExecutableGetBoundaryByIndex(executable.get(), VERNON_PROGRAM_BOUNDARY_OUTPUT, 0,
                                                                   &publishedOutputParameter),
                  VERNON_STATUS_OK);
        const VernonProgramParameterView factorParameter = parameter(executable.get(), "factor");
        std::array<VernonProgramParameterView, 3> gridParameters{
            parameter(executable.get(), "__grid_x"),
            parameter(executable.get(), "__grid_y"),
            parameter(executable.get(), "__grid_z"),
        };
        auto instanceResult = vernon::runtime::ProgramInstance::create(executable);
        ASSERT_TRUE(instanceResult);
        auto instance = std::move(instanceResult).value();
        constexpr std::array<size_t, 2> counts{3, 5};
        constexpr std::array<float, 2> factors{2, 3};
        uint32_t one = 1;
        for (size_t iteration = 0; iteration < counts.size(); ++iteration) {
            const size_t count = counts[iteration];
            std::vector<float> source(count);
            for (size_t index = 0; index < count; ++index)
                source[index] = static_cast<float>(index + 1);
            std::vector<float> output(count);
            const uint64_t shape[]{count};
            uint32_t gridX = static_cast<uint32_t>(count);
            float factor = factors[iteration];
            std::array<VernonProgramArgument, 7> arguments{
                tensorArrayArgument(sourceParameter, source.data(), shape),
                tensorArrayArgument(outputParameter, output.data(), shape),
                tensorArrayArgument(publishedOutputParameter, output.data(), shape),
                scalarArgument(factorParameter, factor),
                scalarArgument(gridParameters[0], gridX),
                scalarArgument(gridParameters[1], one),
                scalarArgument(gridParameters[2], one),
            };
            auto invocationResult = instance.begin();
            ASSERT_TRUE(invocationResult);
            auto invocation = std::move(invocationResult).value();
            for (size_t index = 0; index < arguments.size(); ++index) {
                const std::string text = index >= 5
                                             ? "cpu-dynamic-stable-" + std::to_string(index)
                                             : "cpu-dynamic-" + std::to_string(iteration) + "-" + std::to_string(index);
                ASSERT_TRUE(
                    invocation.bind({sizeof(VernonProgramBindingToken), text.data(), text.size()}, arguments[index]));
            }
            ASSERT_TRUE(invocation.execute(false));
            auto commitResult = invocation.commit();
            ASSERT_TRUE(commitResult);
            EXPECT_FALSE(commitResult.value());
            for (size_t index = 0; index < count; ++index)
                EXPECT_FLOAT_EQ(output[index], source[index] * factor);
        }
        auto telemetryResult = instance.telemetry();
        ASSERT_TRUE(telemetryResult);
        const VernonProgramBindingTelemetry telemetry = telemetryResult.value();
        EXPECT_EQ(telemetry.prepare_count, 12u);
        EXPECT_EQ(telemetry.reuse_count, 2u);
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}
