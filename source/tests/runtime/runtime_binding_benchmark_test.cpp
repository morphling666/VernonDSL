#include "VernonRuntimeCore.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

std::size_t vernonTestAllocationCount() noexcept;

namespace {

struct BenchmarkProvider {
    uint64_t nextObject{1};
};

VernonRuntimeProviderObject next(BenchmarkProvider &provider) { return {provider.nextObject++}; }

VernonRuntimeDeviceProvider makeProvider(BenchmarkProvider &state) {
    VernonRuntimeDeviceProvider provider{};
    provider.struct_size = sizeof(provider);
    provider.abi_version = VERNON_PROGRAM_VERSION;
    provider.user_data = &state;
    provider.get_capabilities = [](void *) { return uint32_t{VERNON_RUNTIME_PROVIDER_COMPUTE}; };
    provider.get_device_identity = [](void *) { return VernonRuntimeProviderDeviceIdentity{1, 1, 1}; };
    provider.prepare_shader = [](void *data, const VernonRuntimeProviderShaderDescriptor *,
                                 VernonRuntimeProviderObject *output) {
        *output = next(*static_cast<BenchmarkProvider *>(data));
        return VERNON_STATUS_OK;
    };
    provider.prepare_pipeline_layout = [](void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *,
                                          VernonRuntimeProviderObject *output) {
        *output = next(*static_cast<BenchmarkProvider *>(data));
        return VERNON_STATUS_OK;
    };
    provider.prepare_pipeline = [](void *data, const VernonRuntimeProviderPipelineDescriptor *,
                                   VernonRuntimeProviderObject *output) {
        *output = next(*static_cast<BenchmarkProvider *>(data));
        return VERNON_STATUS_OK;
    };
    provider.retain_resource = [](void *, VernonRuntimeProviderResourceReference) { return VERNON_STATUS_OK; };
    provider.release_resource = [](void *, VernonRuntimeProviderResourceReference) {};
    provider.describe_image = [](void *, VernonRuntimeProviderResourceReference,
                                 VernonRuntimeProviderImageDescription *) { return VERNON_STATUS_INVALID_ARGUMENT; };
    provider.create_binding_set = [](void *data, const VernonRuntimeProviderBindingSetDescriptor *,
                                     VernonRuntimeProviderObject *output) {
        *output = next(*static_cast<BenchmarkProvider *>(data));
        return VERNON_STATUS_OK;
    };
    provider.encode_dispatch = [](void *, VernonRuntimeProviderObject,
                                  const VernonRuntimeProviderDispatchDescriptor *) { return VERNON_STATUS_OK; };
    provider.encode_draw = [](void *, VernonRuntimeProviderObject, const VernonRuntimeProviderDrawDescriptor *) {
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    };
    provider.destroy_shader = [](void *, VernonRuntimeProviderObject) {};
    provider.destroy_pipeline_layout = [](void *, VernonRuntimeProviderObject) {};
    provider.destroy_pipeline = [](void *, VernonRuntimeProviderObject) {};
    provider.destroy_binding_set = [](void *, VernonRuntimeProviderObject) {};
    return provider;
}

VernonRuntimeCorePipelineDescriptor pipelineDescriptor() {
    static constexpr char source[] = "benchmark";
    static const VernonRuntimeProviderShaderDescriptor shader{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                              VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE,
                                                              {"mock", 4},
                                                              source,
                                                              sizeof(source) - 1,
                                                              {"main", 4},
                                                              {},
                                                              {0, 0, 0, 0}};
    static const VernonRuntimeProviderBindingLayoutEntry binding{
        0, 0, 0, VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER, VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE, 4, 1};
    VernonRuntimeCorePipelineDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.kind = VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
    descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_COMPUTE;
    descriptor.shaders = &shader;
    descriptor.shader_count = 1;
    descriptor.bindings = &binding;
    descriptor.binding_count = 1;
    descriptor.workgroup_size[0] = descriptor.workgroup_size[1] = descriptor.workgroup_size[2] = 1;
    return descriptor;
}

void recordSamples(const char *name, std::vector<uint64_t> &samples, size_t allocations) {
    std::sort(samples.begin(), samples.end());
    const std::string prefix{name};
    testing::Test::RecordProperty((prefix + "_median_ns").c_str(), samples[samples.size() / 2]);
    testing::Test::RecordProperty((prefix + "_p95_ns").c_str(), samples[samples.size() * 95 / 100]);
    testing::Test::RecordProperty((prefix + "_p99_ns").c_str(), samples[samples.size() * 99 / 100]);
    testing::Test::RecordProperty((prefix + "_maximum_ns").c_str(), samples.back());
    testing::Test::RecordProperty((prefix + "_allocation_count").c_str(), allocations);
}

TEST(RuntimeBindingBenchmark, UnchangedReuseAndEncodeAreAllocationFree) {
    constexpr size_t iterationCount = 50000;
    BenchmarkProvider state;
    VernonRuntimeDeviceProvider provider = makeProvider(state);
    VernonRuntimeCorePipelineDescriptor descriptor = pipelineDescriptor();
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(&provider, &descriptor, &pipeline), VERNON_STATUS_OK);

    VernonRuntimeProviderBindingValue value{};
    value.slot = 0;
    value.kind = VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
    value.payload.buffer.resource = {1, {1}, 0, 4096};
    VernonRuntimeCoreBindings *bindings = nullptr;
    ASSERT_EQ(vernonRuntimeCoreCreateBindings(pipeline, &value, 1, &bindings), VERNON_STATUS_OK);
    const uint32_t groups[3]{1, 1, 1};
    ASSERT_EQ(vernonRuntimeCoreEncodeDispatch(pipeline, bindings, {1}, groups, nullptr, 0), VERNON_STATUS_OK);

    std::vector<uint64_t> reuseSamples(iterationCount);
    std::vector<uint64_t> encodeSamples(iterationCount);
    const size_t allocationsBefore = vernonTestAllocationCount();
    for (size_t iteration = 0; iteration < iterationCount; ++iteration) {
        const auto reuseBegin = std::chrono::steady_clock::now();
        ASSERT_EQ(vernonRuntimeCoreUpdateBindings(bindings, &value, 1), VERNON_STATUS_OK);
        const auto reuseEnd = std::chrono::steady_clock::now();
        ASSERT_EQ(vernonRuntimeCoreEncodeDispatch(pipeline, bindings, {1}, groups, nullptr, 0), VERNON_STATUS_OK);
        const auto encodeEnd = std::chrono::steady_clock::now();
        reuseSamples[iteration] =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(reuseEnd - reuseBegin).count());
        encodeSamples[iteration] =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(encodeEnd - reuseEnd).count());
    }
    const size_t allocations = vernonTestAllocationCount() - allocationsBefore;
    EXPECT_EQ(allocations, 0u);
    recordSamples("unchanged_binding_reuse", reuseSamples, allocations);
    recordSamples("compute_encode", encodeSamples, allocations);

    vernonRuntimeCoreBindingsDestroy(bindings);
    vernonRuntimeCorePipelineDestroy(pipeline);
}

} // namespace
