#include "VernonRHI.h"
#include "VernonRuntime.h"
#include "backend_runtime_owner.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "benchmark_registration_declarations.inc"

namespace {

using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;

struct Options {
    std::string fixture;
    std::string backend;
    Json parameters;
    size_t warmup{};
    size_t iterations{};
};

struct Skip : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(std::string message) { throw std::runtime_error(std::move(message)); }

Options parseOptions(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (index + 1 >= argc)
            fail("missing value for " + std::string(argument));
        const std::string value = argv[++index];
        if (argument == "--fixture")
            options.fixture = value;
        else if (argument == "--backend")
            options.backend = value;
        else if (argument == "--parameters-json")
            options.parameters = Json::parse(value);
        else if (argument == "--warmup")
            options.warmup = std::stoull(value);
        else if (argument == "--iterations")
            options.iterations = std::stoull(value);
        else
            fail("unknown argument " + std::string(argument));
    }
    if (options.fixture.empty() || options.backend.empty() || !options.parameters.is_object() || !options.iterations)
        fail("incomplete benchmark driver arguments");
    return options;
}

std::string diagnostic(VernonStringView value) {
    return value.data ? std::string(value.data, value.size) : std::string{};
}

std::string lastError(VernonRuntimeContext *context) {
    return context ? diagnostic(vernonRuntimeGetLastError(context)) : std::string{};
}

class Runtime {
public:
    Runtime(std::string_view name, const vernon::tests::BackendTestRequirements &requirements)
        : backend_(findBackend(name)) {
        if (backend_.runtime == VERNON_RUNTIME_CPU) {
            context_ = vernonRuntimeCreateWithOptions(backend_.runtime, nullptr);
            const vernon::tests::BackendProbeResult probe =
                vernon::tests::probeRuntimeBackend(backend_, requirements, context_);
            handleProbe(probe);
        } else {
            const vernon::tests::BackendProbeResult probe = owner_.initialize(backend_, requirements);
            handleProbe(probe);
            context_ = owner_.owned().runtime();
        }
    }

    ~Runtime() {
        if (backend_.runtime == VERNON_RUNTIME_CPU && context_ && vernonRuntimeDestroy(context_) != VERNON_STATUS_OK)
            std::terminate();
    }

    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    VernonRuntimeContext *get() const { return context_; }
    vernon::tests::RhiRuntime &rhi() {
        if (backend_.runtime == VERNON_RUNTIME_CPU)
            fail("CPU has no RHI benchmark context");
        return owner_.context();
    }

private:
    static const vernon::tests::BackendTestRow &findBackend(std::string_view name) {
        std::string requested(name);
        requested.erase(std::remove(requested.begin(), requested.end(), '_'), requested.end());
        const auto found = std::find_if(
            vernon::tests::backendTestMatrix.begin(), vernon::tests::backendTestMatrix.end(), [&](const auto &row) {
                std::string normalized(row.name);
                std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                               [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
                normalized.erase(std::remove(normalized.begin(), normalized.end(), '_'), normalized.end());
                return normalized == requested;
            });
        if (found == vernon::tests::backendTestMatrix.end())
            fail("unsupported backend " + std::string(name));
        return *found;
    }

    static void handleProbe(const vernon::tests::BackendProbeResult &probe) {
        if (probe.available())
            return;
        if (probe.skippable())
            throw Skip(probe.reason);
        fail(probe.reason);
    }

    const vernon::tests::BackendTestRow &backend_;
    vernon::tests::BackendRuntimeOwner owner_;
    VernonRuntimeContext *context_{};
};

struct Manifest {
    std::string_view asset;
    std::string_view backend;
    std::string_view path;
    VernonStatus (*prepare)();
};

constexpr Manifest manifests[]{
#include "benchmark_manifest_rows.inc"
};

const Manifest &manifest(std::string_view asset, std::string_view backend) {
    const auto found = std::find_if(std::begin(manifests), std::end(manifests), [&](const Manifest &entry) {
        return entry.asset == asset && entry.backend == backend;
    });
    if (found == std::end(manifests))
        throw Skip(std::string(backend) + " compiler/runtime target is not built");
    if (found->prepare && found->prepare() != VERNON_STATUS_OK)
        fail("register CPU fixture failed");
    return *found;
}

class LoadedProgram {
public:
    LoadedProgram(VernonRuntimeContext *context, const std::filesystem::path &path) : context_(context) {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            fail("cannot read Program manifest " + path.string());
        const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        const std::string directory = path.parent_path().string();
        VernonProgramBundleLoadOptions options{};
        options.struct_size = sizeof(options);
        options.bundle_directory = directory.c_str();
        bundle_ = vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), &options);
        if (!bundle_)
            fail("cannot load Program bundle: " + lastError(context));
        executable_ = vernonRuntimeResolveProgram(bundle_, nullptr);
        if (!executable_)
            fail("cannot resolve Program: " + lastError(context));
    }

    ~LoadedProgram() {
        vernonRuntimeProgramExecutableDestroy(executable_);
        vernonRuntimeProgramBundleDestroy(bundle_);
    }

    VernonProgramBundle *bundle() const { return bundle_; }
    VernonProgramExecutable *executable() const { return executable_; }

private:
    VernonRuntimeContext *context_{};
    VernonProgramBundle *bundle_{};
    VernonProgramExecutable *executable_{};
};

VernonProgramParameterView parameter(VernonProgramExecutable *executable, std::string_view name) {
    VernonProgramParameterView result{};
    if (vernonRuntimeProgramExecutableFindParameter(executable, {name.data(), name.size()}, &result) !=
        VERNON_STATUS_OK)
        fail("Program parameter not found: " + std::string(name));
    return result;
}

VernonProgramBindingToken token(std::string_view value) {
    return {sizeof(VernonProgramBindingToken), value.data(), value.size()};
}

VernonProgramArgument tensorArgument(const VernonProgramParameterView &parameter, float *data, const uint64_t *shape,
                                     size_t count) {
    static const int64_t stride = sizeof(float);
    VernonProgramArgument result{};
    result.slot = parameter.slot;
    result.kind = VERNON_PROGRAM_TENSOR;
    result.tensor.struct_size = sizeof(result.tensor);
    result.tensor.storage = VERNON_TENSOR_HOST;
    result.tensor.host_data = data;
    result.tensor.element_layout = parameter.element_layout;
    result.tensor.access = parameter.access;
    result.tensor.rank = 1;
    result.tensor.shape = shape;
    result.tensor.byte_strides = &stride;
    result.tensor.byte_size = count * sizeof(float);
    return result;
}

VernonProgramArgument scalarArgument(const VernonProgramParameterView &parameter, const void *data) {
    VernonProgramArgument result{};
    result.slot = parameter.slot;
    result.kind = VERNON_PROGRAM_TENSOR;
    result.tensor.struct_size = sizeof(result.tensor);
    result.tensor.storage = VERNON_TENSOR_HOST;
    result.tensor.host_data = data;
    result.tensor.element_layout = parameter.element_layout;
    result.tensor.access = parameter.access;
    result.tensor.byte_size = parameter.element_layout.byte_size;
    return result;
}

double elapsedNs(Clock::time_point begin) {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
}

Json measurement(std::string name, std::string unit, std::vector<double> samples) {
    if (samples.empty())
        fail("measurement has no samples");
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&](double value) {
        const double position = value * static_cast<double>(samples.size() - 1);
        const size_t lower = static_cast<size_t>(position);
        const size_t upper = std::min(lower + 1, samples.size() - 1);
        const double fraction = position - static_cast<double>(lower);
        return samples[lower] + (samples[upper] - samples[lower]) * fraction;
    };
    return {
        {"name", std::move(name)},    {"unit", std::move(unit)},   {"sample_count", samples.size()},
        {"median", percentile(0.5)},  {"p95", percentile(0.95)},   {"p99", percentile(0.99)},
        {"minimum", samples.front()}, {"maximum", samples.back()},
    };
}

void requireStatus(VernonStatus status, VernonRuntimeContext *context, std::string_view operation) {
    if (status != VERNON_STATUS_OK)
        fail(std::string(operation) + " failed: " + lastError(context));
}

struct TensorStorage {
    std::vector<float> source;
    std::vector<float> output;
    uint64_t shape{};
    float factor{2.0f};
    uint32_t gridX{};
    uint32_t one{1};
};

std::vector<VernonProgramArgument> argumentsFor(VernonProgramExecutable *executable, TensorStorage &storage) {
    std::vector<VernonProgramArgument> arguments;
    const size_t count = vernonRuntimeProgramExecutableGetParameterCount(executable);
    arguments.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        VernonProgramParameterView current{};
        requireStatus(vernonRuntimeProgramExecutableGetParameterByIndex(executable, index, &current), nullptr,
                      "reflect parameter");
        const std::string_view name(current.name.data, current.name.size);
        if (name == "source")
            arguments.push_back(tensorArgument(current, storage.source.data(), &storage.shape, storage.shape));
        else if (name == "output")
            arguments.push_back(tensorArgument(current, storage.output.data(), &storage.shape, storage.shape));
        else if (name == "factor" || name == "amount")
            arguments.push_back(scalarArgument(current, &storage.factor));
        else if (name == "__grid_x" || name == "grid_x" || name == "groups")
            arguments.push_back(scalarArgument(current, &storage.gridX));
        else if (name == "__grid_y" || name == "__grid_z" || name == "grid_y" || name == "grid_z")
            arguments.push_back(scalarArgument(current, &storage.one));
        else
            fail("unsupported benchmark parameter " + std::string(name));
    }
    return arguments;
}

void executeProgram(VernonRuntimeContext *context, VernonProgramExecutable *executable, VernonProgramInstance *instance,
                    TensorStorage &storage) {
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    if (!invocation)
        fail("Program begin failed");
    const auto arguments = argumentsFor(executable, storage);
    for (size_t index = 0; index < arguments.size(); ++index) {
        const std::string tokenText = "argument-" + std::to_string(index);
        const VernonProgramBindingToken bindingToken = token(tokenText);
        requireStatus(vernonRuntimeProgramInvocationBind(invocation, &bindingToken, &arguments[index], nullptr, 0, 0),
                      context, "Program bind");
    }
    requireStatus(vernonRuntimeProgramInvocationExecute(invocation, 0, nullptr), context, "Program execute");
    requireStatus(vernonRuntimeProgramInvocationCommit(invocation, nullptr), context, "Program commit");
    vernonRuntimeProgramInvocationDestroy(invocation);
}

Json runBackend(const Options &options, Runtime &runtime) {
    std::string asset;
    if (options.fixture == "backend.empty_kernel")
        asset = "empty_kernel";
    else if (options.fixture == "backend.elementwise")
        asset = "elementwise";
    else
        asset = "reduction";
    LoadedProgram program(runtime.get(), manifest(asset, options.backend).path);
    const size_t sourceCount = options.fixture == "backend.empty_kernel" ? 0 : 1048576;
    const size_t outputCount = options.fixture == "backend.reduction" ? 1 : sourceCount;
    std::vector<float> source(sourceCount, 1.0f);
    std::vector<float> output(outputCount, 0.0f);
    uint64_t sourceShape = sourceCount;
    uint64_t outputShape = outputCount;
    uint32_t gridX = options.fixture == "backend.elementwise" ? 16384 : 1;
    uint32_t one = 1;
    std::vector<VernonProgramArgument> arguments;
    for (size_t index = 0; index < vernonRuntimeProgramExecutableGetParameterCount(program.executable()); ++index) {
        VernonProgramParameterView current{};
        requireStatus(vernonRuntimeProgramExecutableGetParameterByIndex(program.executable(), index, &current),
                      runtime.get(), "reflect backend parameter");
        const std::string_view name(current.name.data, current.name.size);
        if (name == "source")
            arguments.push_back(tensorArgument(current, source.data(), &sourceShape, sourceCount));
        else if (name == "output")
            arguments.push_back(tensorArgument(current, output.data(), &outputShape, outputCount));
        else if (name == "__grid_x" || name == "grid_x")
            arguments.push_back(scalarArgument(current, &gridX));
        else if (name == "__grid_y" || name == "__grid_z" || name == "grid_y" || name == "grid_z")
            arguments.push_back(scalarArgument(current, &one));
        else
            fail("unsupported backend benchmark parameter " + std::string(name));
    }
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(program.executable());
    if (!instance)
        fail("backend benchmark instance creation failed");
    const auto invoke = [&] {
        VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
        if (!invocation)
            fail("backend benchmark begin failed");
        for (size_t index = 0; index < arguments.size(); ++index) {
            const std::string tokenText = "argument-" + std::to_string(index);
            const VernonProgramBindingToken bindingToken = token(tokenText);
            requireStatus(
                vernonRuntimeProgramInvocationBind(invocation, &bindingToken, &arguments[index], nullptr, 0, 0),
                runtime.get(), "bind backend argument");
        }
        requireStatus(vernonRuntimeProgramInvocationExecute(invocation, 0, nullptr), runtime.get(),
                      "execute backend benchmark");
        requireStatus(vernonRuntimeProgramInvocationCommit(invocation, nullptr), runtime.get(),
                      "commit backend benchmark");
        vernonRuntimeProgramInvocationDestroy(invocation);
    };
    for (size_t index = 0; index < options.warmup; ++index)
        invoke();
    std::vector<double> samples;
    samples.reserve(options.iterations);
    for (size_t index = 0; index < options.iterations; ++index) {
        const auto begin = Clock::now();
        invoke();
        samples.push_back(elapsedNs(begin));
    }
    vernonRuntimeProgramInstanceDestroy(instance);
    if (options.fixture == "backend.elementwise" &&
        std::any_of(output.begin(), output.end(), [](float value) { return std::fabs(value - 3.0f) > 1e-5f; }))
        fail("elementwise benchmark validation failed");
    if (options.fixture == "backend.reduction" && (output.empty() || std::fabs(output.front() - 1048576.0f) > 1e-3f))
        fail("reduction benchmark validation failed");
    return {
        {"status", "completed"},
        {"measurements", Json::array({measurement("call_ns", "ns", std::move(samples))})},
    };
}

Json runTensorBinding(const Options &options, Runtime &runtime) {
    const bool dynamic = options.fixture == "binding.dynamic_shape";
    const Manifest &entry = manifest(dynamic ? "dynamic_binding" : "binding", options.backend);
    LoadedProgram program(runtime.get(), entry.path);
    const bool changed = options.parameters.value("update", std::string{}) == "changed";
    TensorStorage storage;
    storage.shape = 1024;
    storage.gridX = static_cast<uint32_t>((storage.shape + 63) / 64);
    storage.source.assign(2048, 1.0f);
    storage.output.assign(2048, 0.0f);
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(program.executable());
    if (!instance)
        fail("binding instance creation failed");
    for (size_t index = 0; index < options.warmup; ++index) {
        if (dynamic)
            storage.shape = changed && index % 2 ? 2048 : 1024;
        else
            storage.factor = changed && index % 2 ? 3.0f : 2.0f;
        storage.gridX = static_cast<uint32_t>((storage.shape + 63) / 64);
        executeProgram(runtime.get(), program.executable(), instance, storage);
    }
    std::vector<double> samples;
    samples.reserve(options.iterations);
    for (size_t index = 0; index < options.iterations; ++index) {
        const size_t sequence = options.warmup + index;
        if (dynamic)
            storage.shape = changed && sequence % 2 ? 2048 : 1024;
        else
            storage.factor = changed && sequence % 2 ? 3.0f : 2.0f;
        storage.gridX = static_cast<uint32_t>((storage.shape + 63) / 64);
        const auto begin = Clock::now();
        executeProgram(runtime.get(), program.executable(), instance, storage);
        samples.push_back(elapsedNs(begin));
    }
    const float expected = dynamic ? 2.0f : 1.0f + storage.factor;
    if (std::fabs(storage.output[0] - expected) > 1e-5f)
        fail("binding validation produced an incorrect output");
    VernonProgramBindingTelemetry telemetry{};
    telemetry.struct_size = sizeof(telemetry);
    requireStatus(vernonRuntimeProgramInstanceGetTelemetry(instance, &telemetry), runtime.get(),
                  "read binding telemetry");
    vernonRuntimeProgramInstanceDestroy(instance);
    return {
        {"status", "completed"},
        {"measurements", Json::array({measurement("call_ns", "ns", std::move(samples))})},
        {"counters",
         {{"prepare_count", telemetry.prepare_count},
          {"reuse_count", telemetry.reuse_count},
          {"upload_bytes", telemetry.upload_bytes},
          {"upload_ranges", telemetry.upload_ranges}}},
    };
}

Json runControlBinding(const Options &options, Runtime &runtime) {
    const bool graphicsBenchmark = options.fixture == "graphics.draw";
    const size_t drawCount = static_cast<size_t>(options.parameters.value("draw_count", 1));
    const std::string asset = graphicsBenchmark ? "draw_" + std::to_string(drawCount) : "control";
    LoadedProgram program(runtime.get(), manifest(asset, options.backend).path);
    const size_t graphicsNodeCount = vernonRuntimeProgramExecutableGetGraphicsNodeCount(program.executable());
    std::vector<VernonProgramGraphicsControlsView> slots(graphicsNodeCount);
    for (size_t index = 0; index < slots.size(); ++index) {
        slots[index].struct_size = sizeof(slots[index]);
        requireStatus(
            vernonRuntimeProgramExecutableGetGraphicsControlsByIndex(program.executable(), index, &slots[index]),
            runtime.get(), "reflect graphics controls");
    }
    const uint32_t extent = graphicsBenchmark ? 64 : 16;
    constexpr std::array<float, 6> positions{-0.75f, -0.75f, 0.75f, -0.75f, 0.0f, 0.75f};
    vernon::tests::RhiRuntime &rhi = runtime.rhi();
    vernon::tests::RhiBuffer vertices =
        vernon::tests::createBuffer(rhi, sizeof(positions), alignof(float), VERNON_RHI_BUFFER_VERTEX, positions.data());
    if (vertices.handle.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        fail("cannot create control benchmark vertex buffer");
    vernon::tests::RhiImage target = vernon::tests::createImage(rhi, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM,
                                                                extent, extent, 1, VERNON_RHI_IMAGE_COLOR_ATTACHMENT);
    vernon::tests::RhiImageView targetView =
        vernon::tests::createImageView(rhi, target, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);
    if (target.handle.index == VERNON_RHI_INVALID_HANDLE_INDEX ||
        targetView.handle.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        fail("cannot create control benchmark target");
    constexpr uint64_t shape[]{3, 2};
    constexpr int64_t strides[]{2 * sizeof(float), sizeof(float)};
    std::vector<VernonProgramArgument> arguments;
    for (size_t index = 0; index < vernonRuntimeProgramExecutableGetParameterCount(program.executable()); ++index) {
        VernonProgramParameterView current{};
        requireStatus(vernonRuntimeProgramExecutableGetParameterByIndex(program.executable(), index, &current),
                      runtime.get(), "reflect graphics parameter");
        const std::string_view name(current.name.data, current.name.size);
        VernonProgramArgument argument{};
        argument.slot = current.slot;
        if (name == "vertices") {
            argument.kind = VERNON_PROGRAM_TENSOR;
            argument.tensor.struct_size = sizeof(argument.tensor);
            argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
            argument.tensor.resource = vertices.reference;
            argument.tensor.element_layout = current.element_layout;
            argument.tensor.access = current.access;
            argument.tensor.rank = 2;
            argument.tensor.shape = shape;
            argument.tensor.byte_strides = strides;
            argument.tensor.byte_size = sizeof(positions);
        } else if (name == "output") {
            argument.kind = VERNON_PROGRAM_IMAGE;
            argument.image = {targetView.reference};
        } else {
            fail("unsupported graphics benchmark parameter " + std::string(name));
        }
        arguments.push_back(argument);
    }
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(program.executable());
    if (!instance)
        fail("control binding instance creation failed");
    const bool changed = graphicsBenchmark ? options.parameters.value("state_changes", 0) != 0
                                           : options.parameters.value("update", std::string{}) == "changed";
    VernonColorAttachment attachment{};
    attachment.location = 0;
    attachment.view = targetView.reference;
    attachment.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    attachment.store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
    attachment.clear_color[3] = 1.0f;
    VernonRenderPass renderPass{};
    renderPass.struct_size = sizeof(renderPass);
    renderPass.color_attachments = &attachment;
    renderPass.color_attachment_count = 1;
    renderPass.render_area[2] = extent;
    renderPass.render_area[3] = extent;
    VernonDrawCommand draw{};
    draw.struct_size = sizeof(draw);
    draw.vertex_count = 3;
    draw.instance_count = 1;
    VernonDynamicState dynamic{};
    dynamic.struct_size = sizeof(dynamic);
    dynamic.viewport[2] = extent;
    dynamic.viewport[3] = extent;
    dynamic.scissor[2] = extent;
    dynamic.scissor[3] = extent;
    const auto invoke = [&](size_t sequence) {
        dynamic.viewport[2] = changed && sequence % 2 ? extent - 1 : extent;
        VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
        if (!invocation)
            fail("control binding begin failed");
        for (size_t index = 0; index < arguments.size(); ++index) {
            const std::string tokenText = "argument-" + std::to_string(index);
            const VernonProgramBindingToken bindingToken = token(tokenText);
            requireStatus(
                vernonRuntimeProgramInvocationBind(invocation, &bindingToken, &arguments[index], nullptr, 0, 0),
                runtime.get(), "bind graphics argument");
        }
        const VernonProgramBindingToken renderToken = token("render-pass");
        const VernonProgramBindingToken drawToken = token("draw");
        const std::string dynamicText = changed ? "dynamic-" + std::to_string(sequence) : "dynamic";
        const VernonProgramBindingToken dynamicToken = token(dynamicText);
        for (const VernonProgramGraphicsControlsView &nodeSlots : slots) {
            requireStatus(vernonRuntimeProgramInvocationBindRenderPass(invocation, nodeSlots.render_pass_control,
                                                                       &renderToken, &renderPass, nullptr, 0),
                          runtime.get(), "bind render pass");
            requireStatus(vernonRuntimeProgramInvocationBindDrawCommand(invocation, nodeSlots.draw_command_control,
                                                                        &drawToken, &draw, nullptr),
                          runtime.get(), "bind draw");
            requireStatus(vernonRuntimeProgramInvocationBindDynamicState(invocation, nodeSlots.dynamic_state_control,
                                                                         &dynamicToken, &dynamic),
                          runtime.get(), "bind dynamic state");
        }
        requireStatus(vernonRuntimeProgramInvocationExecute(invocation, 0, nullptr), runtime.get(), "execute graphics");
        requireStatus(vernonRuntimeProgramInvocationCommit(invocation, nullptr), runtime.get(), "commit graphics");
        vernonRuntimeProgramInvocationDestroy(invocation);
    };
    for (size_t index = 0; index < options.warmup; ++index)
        invoke(index);
    std::vector<double> samples;
    samples.reserve(options.iterations);
    for (size_t index = 0; index < options.iterations; ++index) {
        const auto begin = Clock::now();
        invoke(options.warmup + index);
        samples.push_back(elapsedNs(begin));
    }
    vernonRuntimeProgramInstanceDestroy(instance);
    std::vector<uint8_t> pixels(extent * extent * 4);
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
    download.width = extent;
    download.height = extent;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    if (vernonRhiDeviceDownloadImage(rhi.device, target.handle, &download, pixels.data(), pixels.size()) !=
            VERNON_RHI_STATUS_OK ||
        std::none_of(pixels.begin(), pixels.end(), [](uint8_t value) { return value != 0; }))
        fail("graphics control benchmark validation failed");
    if (vernonRhiDeviceDestroyImageView(rhi.device, targetView.handle) != VERNON_RHI_STATUS_OK ||
        vernonRhiDeviceDestroyImage(rhi.device, target.handle) != VERNON_RHI_STATUS_OK ||
        vernonRhiDeviceDestroyBuffer(rhi.device, vertices.handle) != VERNON_RHI_STATUS_OK)
        fail("graphics control benchmark resource destruction failed");
    Json result{
        {"status", "completed"},
        {"measurements", Json::array({measurement("call_ns", "ns", std::move(samples))})},
    };
    if (graphicsBenchmark)
        result["counters"] = {{"draw_count", drawCount}};
    return result;
}

Json runProgram(const Options &options, Runtime &runtime) {
    const size_t nodeCount = static_cast<size_t>(options.parameters.value("node_count", 1));
    const std::string asset = options.fixture == "program.square" ? "square" : "chain_" + std::to_string(nodeCount);
    LoadedProgram loaded(runtime.get(), manifest(asset, options.backend).path);
    VernonProgramExecutable *executable = loaded.executable();
    TensorStorage storage;
    storage.shape = options.fixture == "program.square" ? 1 : 1024;
    storage.gridX = static_cast<uint32_t>((storage.shape + 63) / 64);
    storage.source.assign(storage.shape, 1.0f);
    storage.output.assign(storage.shape, 0.0f);
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(executable);
    if (!instance)
        fail("Program benchmark instance creation failed");
    for (size_t index = 0; index < options.warmup; ++index)
        executeProgram(runtime.get(), executable, instance, storage);

    std::vector<double> samples;
    samples.reserve(options.iterations);
    for (size_t index = 0; index < options.iterations; ++index) {
        const auto begin = Clock::now();
        executeProgram(runtime.get(), executable, instance, storage);
        samples.push_back(elapsedNs(begin));
    }
    vernonRuntimeProgramInstanceDestroy(instance);
    const float expected = options.fixture == "program.square" ? 1.0f : 1.0f + static_cast<float>(nodeCount);
    if (std::fabs(storage.output[0] - expected) > 1e-5f)
        fail("Program benchmark validation produced an incorrect output");
    return {
        {"status", "completed"},
        {"measurements", Json::array({measurement("call_ns", "ns", std::move(samples))})},
        {"counters", {{"node_count", nodeCount}}},
    };
}

Json run(const Options &options) {
    vernon::tests::BackendTestRequirements requirements{};
    requirements.storageBuffers = options.fixture != "backend.empty_kernel";
    if (options.fixture == "binding.control" || options.fixture == "graphics.draw")
        requirements.graphics = true;
    else
        requirements.compute = true;
    Runtime runtime(options.backend, requirements);
    if (options.fixture == "binding.tensor" || options.fixture == "binding.dynamic_shape")
        return runTensorBinding(options, runtime);
    if (options.fixture == "binding.control" || options.fixture == "graphics.draw")
        return runControlBinding(options, runtime);
    if (options.fixture == "program.square" || options.fixture == "program.chain")
        return runProgram(options, runtime);
    if (options.fixture.compare(0, 8, "backend.") == 0)
        return runBackend(options, runtime);
    fail("unsupported fixture " + options.fixture);
}

} // namespace

int main(int argc, char **argv) {
    try {
        std::cout << run(parseOptions(argc, argv)).dump() << '\n';
        return 0;
    } catch (const Skip &error) {
        std::cout << Json{{"status", "skipped"}, {"skip_reason", error.what()}, {"measurements", Json::array()}}.dump()
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
