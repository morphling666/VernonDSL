#include "VernonRuntime.h"
#include "program_fixture_runtime_test.h"
#include "runtime/program_execution/failure_injection.h"
#include "runtime/runtime_test_hooks.h"
#include "runtime_rhi_test_utils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using vernon::runtime::program_execution::FailureBoundary;
using vernon::tests::RhiBuffer;
using vernon::tests::RhiImage;
using vernon::tests::RhiImageView;
using vernon::tests::RhiRuntime;

constexpr uint32_t kExtent = 32;
constexpr std::array<float, 6> kLeftTriangle{-0.9f, -0.9f, -0.1f, -0.9f, -0.5f, 0.9f};
constexpr std::array<float, 6> kRightTriangle{0.1f, -0.9f, 0.9f, -0.9f, 0.5f, 0.9f};

std::string lastError(VernonRuntimeContext *context) {
    const VernonStringView error = vernonRuntimeGetLastError(context);
    return std::string(error.data ? error.data : "", error.size);
}

VernonProgramBundle *loadBundle(RhiRuntime &runtime, const std::filesystem::path &manifestPath) {
    std::ifstream input(manifestPath, std::ios::binary);
    if (!input)
        return nullptr;
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const std::string directory = manifestPath.parent_path().string();
    VernonProgramBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = directory.c_str();
    return vernonRuntimeLoadProgramBundleWithOptions(runtime.runtime, manifest.data(), manifest.size(), &options);
}

VernonProgramParameterView findParameter(VernonProgramExecutable *executable, std::string_view name) {
    VernonProgramParameterView result{};
    EXPECT_EQ(vernonRuntimeProgramExecutableFindParameter(executable, {name.data(), name.size()}, &result),
              VERNON_STATUS_OK);
    return result;
}

// `shape` and `strides` must outlive the argument.
template <size_t Rank>
VernonProgramArgument tensorArgument(const VernonProgramParameterView &parameter,
                                     VernonRuntimeProviderResourceReference resource, const uint64_t (&shape)[Rank],
                                     const int64_t (&strides)[Rank], size_t byteSize, size_t byteOffset) {
    VernonProgramArgument argument{};
    argument.slot = parameter.slot;
    argument.kind = VERNON_PROGRAM_TENSOR;
    argument.tensor.struct_size = sizeof(argument.tensor);
    argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    argument.tensor.resource = resource;
    argument.tensor.element_layout = parameter.element_layout;
    argument.tensor.access = parameter.access;
    argument.tensor.rank = Rank;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_offset = byteOffset;
    argument.tensor.byte_size = byteSize;
    return argument;
}

VernonProgramArgument verticesArgument(const VernonProgramParameterView &parameter,
                                       VernonRuntimeProviderResourceReference resource, size_t byteOffset = 0) {
    static constexpr uint64_t shape[]{3, 2};
    static constexpr int64_t strides[]{2 * sizeof(float), sizeof(float)};
    return tensorArgument(parameter, resource, shape, strides, 3 * 2 * sizeof(float), byteOffset);
}

// `value` must outlive the argument.
template <typename T>
VernonProgramArgument hostArgument(const VernonProgramParameterView &parameter, const std::array<T, 4> &value) {
    VernonProgramArgument argument{};
    argument.slot = parameter.slot;
    argument.kind = VERNON_PROGRAM_TENSOR;
    argument.tensor.struct_size = sizeof(argument.tensor);
    argument.tensor.storage = VERNON_TENSOR_HOST;
    argument.tensor.host_data = value.data();
    argument.tensor.element_layout = parameter.element_layout;
    argument.tensor.access = parameter.access;
    argument.tensor.byte_size = sizeof(value);
    return argument;
}

VernonProgramBindingToken token(std::string_view value) {
    return {sizeof(VernonProgramBindingToken), value.data(), value.size()};
}

std::string bytesToken(std::string name, const void *data, size_t size) {
    const auto *bytes = static_cast<const unsigned char *>(data);
    for (size_t index = 0; index < size; ++index)
        name += std::to_string(bytes[index]) + ".";
    return name;
}

// Equal tokens let the instance reuse its prepared binding, so a token must name the bound resource or value.
std::string resourceToken(std::string_view name, const VernonRuntimeProviderResourceReference &resource,
                          size_t byteOffset = 0) {
    return bytesToken(std::string(name), &resource, sizeof(resource)) + std::to_string(byteOffset);
}

std::string argumentToken(const VernonProgramArgument &argument) {
    const std::string name = "argument-" + std::to_string(argument.slot) + "-";
    if (argument.kind == VERNON_PROGRAM_IMAGE)
        return resourceToken(name, argument.image.view);
    if (argument.kind != VERNON_PROGRAM_TENSOR)
        return resourceToken(name, argument.resource);
    return argument.tensor.storage == VERNON_TENSOR_HOST
               ? bytesToken(name, argument.tensor.host_data, argument.tensor.byte_size)
               : resourceToken(name, argument.tensor.resource, argument.tensor.byte_offset);
}

struct Target {
    RhiRuntime &runtime;
    uint32_t extent;
    RhiImage image;
    RhiImageView view;

    explicit Target(RhiRuntime &value, uint32_t size = kExtent, uint32_t usage = VERNON_RHI_IMAGE_COLOR_ATTACHMENT)
        : runtime(value), extent(size),
          image(vernon::tests::createImage(value, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM, size, size, 1,
                                           usage)),
          view(vernon::tests::createImageView(value, image, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM)) {}
    Target(const Target &) = delete;
    Target &operator=(const Target &) = delete;
    ~Target() {
        if (view.handle.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            EXPECT_EQ(vernonRhiDeviceDestroyImageView(runtime.device, view.handle), VERNON_RHI_STATUS_OK);
        if (image.handle.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            EXPECT_EQ(vernonRhiDeviceDestroyImage(runtime.device, image.handle), VERNON_RHI_STATUS_OK);
    }

    std::vector<uint8_t> pixels() const {
        std::vector<uint8_t> result(static_cast<size_t>(extent) * extent * 4);
        VernonRhiImageDownloadDescriptor download{};
        download.struct_size = sizeof(download);
        download.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
        download.width = extent;
        download.height = extent;
        download.depth = 1;
        download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
        download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
        EXPECT_EQ(vernonRhiDeviceDownloadImage(runtime.device, image.handle, &download, result.data(), result.size()),
                  VERNON_RHI_STATUS_OK);
        return result;
    }
};

size_t renderedPixels(const std::vector<uint8_t> &pixels) {
    size_t rendered = 0;
    for (size_t pixel = 0; pixel < pixels.size(); pixel += 4)
        rendered += pixels[pixel] == 255 && pixels[pixel + 1] == 64 && pixels[pixel + 2] == 0;
    return rendered;
}

// One invocation of the pass Node: binds the resource parameters, `output`, the RenderPass, and the dynamic state,
// plus a DrawCommand of `instances` instances unless the caller records the draws.
class PassInvocation {
public:
    PassInvocation(VernonProgramExecutable *executable, VernonProgramInstance *instance, const Target &target,
                   std::initializer_list<VernonProgramArgument> arguments, VernonRuntimeProviderLoadOperation load,
                   bool recorded, uint32_t instances = 1)
        : invocation_(vernonRuntimeProgramInstanceBeginInvocation(instance)) {
        EXPECT_NE(invocation_, nullptr);
        EXPECT_EQ(vernonRuntimeProgramExecutableGetGraphicsControlsByIndex(executable, 0, &controls_),
                  VERNON_STATUS_OK);
        for (const VernonProgramArgument &argument : arguments) {
            const std::string text = argumentToken(argument);
            const VernonProgramBindingToken bindingToken = token(text);
            EXPECT_EQ(vernonRuntimeProgramInvocationBind(invocation_, &bindingToken, &argument, nullptr, 0, 0),
                      VERNON_STATUS_OK);
        }
        VernonProgramArgument output{};
        output.slot = findParameter(executable, "output").slot;
        output.kind = VERNON_PROGRAM_IMAGE;
        output.image = {target.view.reference};
        const std::string targetText = resourceToken("output-", target.view.reference);
        const std::string renderText = "render-" + std::to_string(load) + targetText;
        const VernonProgramBindingToken outputToken = token(targetText);
        EXPECT_EQ(vernonRuntimeProgramInvocationBind(invocation_, &outputToken, &output, nullptr, 0, 0),
                  VERNON_STATUS_OK);
        attachment_.location = 0;
        attachment_.view = target.view.reference;
        attachment_.load_operation = load;
        attachment_.store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
        attachment_.clear_color.f32[3] = 1.0f;
        vernon::tests::CanonicalGraphicsControls graphics(&attachment_, 1, 3, instances);
        graphics.renderPass.render_area[2] = target.extent;
        graphics.renderPass.render_area[3] = target.extent;
        graphics.dynamic.viewport[2] = static_cast<float>(target.extent);
        graphics.dynamic.viewport[3] = static_cast<float>(target.extent);
        graphics.dynamic.scissor[2] = target.extent;
        graphics.dynamic.scissor[3] = target.extent;
        const std::string drawText = "pass-draw-" + std::to_string(instances);
        const std::string dynamicText = "pass-dynamic-" + std::to_string(target.extent);
        const VernonProgramBindingToken renderToken = token(renderText);
        const VernonProgramBindingToken drawToken = token(drawText);
        const VernonProgramBindingToken dynamicToken = token(dynamicText);
        EXPECT_EQ(vernonRuntimeProgramInvocationBindRenderPass(invocation_, controls_.render_pass_control, &renderToken,
                                                               &graphics.renderPass, nullptr, 0),
                  VERNON_STATUS_OK);
        EXPECT_EQ(vernonRuntimeProgramInvocationBindDynamicState(invocation_, controls_.dynamic_state_control,
                                                                 &dynamicToken, &graphics.dynamic),
                  VERNON_STATUS_OK);
        if (!recorded)
            EXPECT_EQ(vernonRuntimeProgramInvocationBindDrawCommand(invocation_, controls_.draw_command_control,
                                                                    &drawToken, &graphics.draw, nullptr),
                      VERNON_STATUS_OK);
    }
    PassInvocation(const PassInvocation &) = delete;
    PassInvocation &operator=(const PassInvocation &) = delete;
    ~PassInvocation() { vernonRuntimeProgramInvocationDestroy(invocation_); }

    VernonProgramInvocation *handle() const { return invocation_; }
    VernonProgramNodeId node() const { return controls_.node; }
    uint32_t renderPassControl() const { return controls_.render_pass_control; }

    VernonStatus record(const std::vector<VernonGraphicsCommand> &commands) {
        return vernonRuntimeProgramInvocationRecordGraphics(invocation_, controls_.node, commands.data(),
                                                            commands.size(), nullptr, 0);
    }

    VernonStatus setCompletionMode(VernonCompletionMode mode) {
        const VernonProgramInvocationOptions options{
            sizeof(options), VERNON_PROGRAM_INVOCATION_OPTIONS_VERSION, mode, {}};
        return vernonRuntimeProgramInvocationSetOptions(invocation_, &options);
    }

    VernonStatus run(VernonInvocationMutationOutcome *outcome = nullptr) {
        const VernonStatus status = vernonRuntimeProgramInvocationExecute(invocation_, 0, outcome);
        return status == VERNON_STATUS_OK ? vernonRuntimeProgramInvocationCommit(invocation_, nullptr) : status;
    }

    VernonStatus takeSubmission(VernonSubmission *&submission) {
        return vernonRuntimeProgramInvocationTakeSubmission(invocation_, &submission);
    }

    VernonProgramGraphicsNodeTelemetry telemetry() const {
        VernonProgramGraphicsNodeTelemetry result{};
        result.struct_size = sizeof(result);
        EXPECT_EQ(vernonRuntimeProgramInvocationGetGraphicsTelemetry(invocation_, controls_.node, &result),
                  VERNON_STATUS_OK);
        return result;
    }

private:
    VernonProgramInvocation *invocation_;
    VernonProgramGraphicsControlsView controls_{};
    VernonColorAttachment attachment_{};
};

class ScopedFailureInjection {
public:
    explicit ScopedFailureInjection(FailureBoundary boundary) {
        vernon::runtime::program_execution::setFailureInjectionForTesting(boundary);
    }
    ScopedFailureInjection(const ScopedFailureInjection &) = delete;
    ScopedFailureInjection &operator=(const ScopedFailureInjection &) = delete;
    ~ScopedFailureInjection() { vernon::runtime::program_execution::clearFailureInjectionForTesting(); }
};

struct MutationOutcome {
    std::array<VernonBoundaryMutation, 8> records{};
    VernonInvocationMutationOutcome outcome{sizeof(VernonInvocationMutationOutcome),
                                            VERNON_INVOCATION_NOT_SUBMITTED,
                                            records.data(),
                                            records.size(),
                                            0,
                                            {}};

    MutationOutcome() = default;
    MutationOutcome(const MutationOutcome &) = delete;
    MutationOutcome &operator=(const MutationOutcome &) = delete;

    bool all(VernonBoundaryMutationState state) const {
        return outcome.mutation_count > 0 &&
               std::all_of(records.begin(), records.begin() + static_cast<ptrdiff_t>(outcome.mutation_count),
                           [&](const VernonBoundaryMutation &mutation) { return mutation.state == state; });
    }
};

VernonGraphicsCommand bindPipeline(const VernonGraphicsPipeline *pipeline) {
    VernonGraphicsCommand command{};
    command.kind = VERNON_GRAPHICS_COMMAND_BIND_PIPELINE;
    command.pipeline = pipeline;
    return command;
}

VernonGraphicsCommand bindSet(const VernonBindingSet *set) {
    VernonGraphicsCommand command{};
    command.kind = VERNON_GRAPHICS_COMMAND_BIND_SET;
    command.binding_set = set;
    return command;
}

VernonGraphicsCommand draw(uint32_t vertexCount, uint32_t firstInstance = 0) {
    VernonGraphicsCommand command{};
    command.kind = VERNON_GRAPHICS_COMMAND_DRAW;
    command.draw = {sizeof(VernonDrawCommand), nullptr, vertexCount, 1, firstInstance};
    return command;
}

VernonIndexBinding indexBinding(VernonRuntimeProviderResourceReference indices, size_t offset, int32_t baseVertex) {
    return {VERNON_INDEX_U32, offset, 3, indices, baseVertex};
}

// `index` must stay alive until the command is recorded.
VernonGraphicsCommand drawIndexed(const VernonIndexBinding &index) {
    VernonGraphicsCommand command{};
    command.kind = VERNON_GRAPHICS_COMMAND_DRAW;
    command.draw = {sizeof(VernonDrawCommand), &index, 0, 1, 0};
    return command;
}

VernonBindingSet *sealedSet(RhiRuntime &runtime, VernonGraphicsPipeline *pipeline,
                            std::initializer_list<VernonProgramArgument> arguments) {
    VernonBindingSet *set = nullptr;
    EXPECT_EQ(vernonRuntimeBindingSetCreate(pipeline, &set), VERNON_RUNTIME_OPERATION_OK);
    for (const VernonProgramArgument &argument : arguments)
        EXPECT_EQ(vernonRuntimeBindingSetBind(set, &argument, nullptr), VERNON_STATUS_OK) << lastError(runtime.runtime);
    EXPECT_EQ(vernonRuntimeBindingSetSeal(set), VERNON_STATUS_OK);
    return set;
}

VernonBindingSet *verticesSet(RhiRuntime &runtime, VernonGraphicsPipeline *pipeline, VernonProgramExecutable *program,
                              VernonRuntimeProviderResourceReference vertices) {
    return sealedSet(runtime, pipeline, {verticesArgument(findParameter(program, "vertices"), vertices)});
}

class RuntimeRecordedGraphicsMatrix : public vernon::tests::ProgramFixtureRuntimeTest {
protected:
    vernon::tests::BackendTestRequirements requirements() const override {
        return vernon::tests::graphicsImageCopyFixtureRequirements(GetParam().runtime);
    }

    void SetUp() override {
        ProgramFixtureRuntimeTest::SetUp();
        if (IsSkipped() || HasFatalFailure())
            return;
        bundle_ = loadBundle(runtime(), GetParam().manifestPath);
        ASSERT_NE(bundle_, nullptr) << lastError(runtime().runtime);
        executable_ = vernonRuntimeResolveProgram(bundle_, nullptr);
        ASSERT_NE(executable_, nullptr) << lastError(runtime().runtime);
        instance_ = vernonRuntimeProgramInstanceCreate(executable_);
        ASSERT_NE(instance_, nullptr);
        left_ = vernon::tests::createBuffer(runtime(), sizeof(kLeftTriangle), alignof(float), VERNON_RHI_BUFFER_VERTEX,
                                            kLeftTriangle.data());
        right_ = vernon::tests::createBuffer(runtime(), sizeof(kRightTriangle), alignof(float),
                                             VERNON_RHI_BUFFER_VERTEX, kRightTriangle.data());
        ASSERT_NE(left_.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
        ASSERT_NE(right_.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    }

    void TearDown() override {
        if (right_.handle.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            EXPECT_EQ(vernonRhiDeviceDestroyBuffer(runtime().device, right_.handle), VERNON_RHI_STATUS_OK);
        if (left_.handle.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            EXPECT_EQ(vernonRhiDeviceDestroyBuffer(runtime().device, left_.handle), VERNON_RHI_STATUS_OK);
        if (instance_)
            vernonRuntimeProgramInstanceDestroy(instance_);
        if (executable_)
            vernonRuntimeProgramExecutableDestroy(executable_);
        if (bundle_)
            vernonRuntimeProgramBundleDestroy(bundle_);
        ProgramFixtureRuntimeTest::TearDown();
    }

    VernonProgramArgument passVertices(const RhiBuffer &buffer) const {
        return verticesArgument(findParameter(executable_, "vertices"), buffer.reference);
    }

    // Both triangles drawn by two ordinary invocations in sequence.
    std::vector<uint8_t> sequentialReference() {
        Target target(runtime());
        {
            PassInvocation first(executable_, instance_, target, {passVertices(left_)},
                                 VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, false);
            EXPECT_EQ(first.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
        }
        {
            PassInvocation second(executable_, instance_, target, {passVertices(right_)},
                                  VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE, false);
            EXPECT_EQ(second.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
        }
        return target.pixels();
    }

    std::vector<uint8_t> leftReference() {
        Target target(runtime());
        PassInvocation pass(executable_, instance_, target, {passVertices(left_)}, VERNON_RUNTIME_PROVIDER_LOAD_CLEAR,
                            false);
        EXPECT_EQ(pass.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
        EXPECT_GT(renderedPixels(target.pixels()), 0u);
        return target.pixels();
    }

    VernonGraphicsPipeline *passPipeline(VernonProgramNodeId node) {
        VernonGraphicsPipeline *pipeline = nullptr;
        EXPECT_EQ(vernonRuntimeGraphicsPipelineCreate(executable_, node, nullptr, &pipeline),
                  VERNON_RUNTIME_OPERATION_OK)
            << lastError(runtime().runtime);
        return pipeline;
    }

    bool direct3d() const { return std::string_view(GetParam().target) == "directx"; }
    // Until the compiler emits the base constants, only targets whose native instance index includes the first
    // instance draw with a nonzero one.
    bool nativeFirstInstance() const {
        return GetParam().runtime == VERNON_RUNTIME_VULKAN || GetParam().runtime == VERNON_RUNTIME_METAL;
    }

    VernonProgramBundle *bundle_{};
    VernonProgramExecutable *executable_{};
    VernonProgramInstance *instance_{};
    RhiBuffer left_;
    RhiBuffer right_;
};

TEST_P(RuntimeRecordedGraphicsMatrix, RecordedDrawsMatchSequentialInvocations) {
    const std::vector<uint8_t> reference = sequentialReference();
    const auto rendered = [&](uint32_t x, uint32_t y) { return reference[(y * kExtent + x) * 4] == 255; };
    ASSERT_TRUE(rendered(kExtent / 4, kExtent / 2));
    ASSERT_TRUE(rendered(3 * kExtent / 4, kExtent / 2));
    ASSERT_FALSE(rendered(kExtent / 2, kExtent / 2));

    Target target(runtime());
    PassInvocation pass(executable_, instance_, target, {passVertices(left_)}, VERNON_RUNTIME_PROVIDER_LOAD_CLEAR,
                        true);
    VernonGraphicsPipeline *pipeline = passPipeline(pass.node());
    ASSERT_NE(pipeline, nullptr);
    VernonBindingSet *right = verticesSet(runtime(), pipeline, executable_, right_.reference);
    ASSERT_EQ(pass.record({bindPipeline(nullptr), draw(3), bindSet(right), draw(3)}), VERNON_STATUS_OK)
        << lastError(runtime().runtime);
    ASSERT_EQ(pass.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
    const VernonProgramGraphicsNodeTelemetry telemetry = pass.telemetry();
    EXPECT_EQ(telemetry.render_scopes, 1u);
    EXPECT_EQ(telemetry.pipeline_binds, 1u);
    EXPECT_EQ(telemetry.set_binds, 1u);
    EXPECT_EQ(telemetry.draws, 2u);
    EXPECT_EQ(target.pixels(), reference);

    vernonRuntimeBindingSetDestroy(right);
    vernonRuntimeGraphicsPipelineDestroy(pipeline);
}

TEST_P(RuntimeRecordedGraphicsMatrix, RecordingSupersedesTheBoundDrawCommand) {
    Target reference(runtime());
    {
        PassInvocation single(executable_, instance_, reference, {passVertices(right_)},
                              VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, false);
        ASSERT_EQ(single.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
    }
    Target target(runtime());
    PassInvocation pass(executable_, instance_, target, {passVertices(left_)}, VERNON_RUNTIME_PROVIDER_LOAD_CLEAR,
                        false);
    VernonGraphicsPipeline *pipeline = passPipeline(pass.node());
    VernonBindingSet *right = verticesSet(runtime(), pipeline, executable_, right_.reference);
    ASSERT_EQ(pass.record({bindPipeline(nullptr), bindSet(right), draw(3)}), VERNON_STATUS_OK);
    ASSERT_EQ(pass.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
    EXPECT_EQ(pass.telemetry().draws, 1u);
    EXPECT_EQ(target.pixels(), reference.pixels());
    vernonRuntimeBindingSetDestroy(right);
    vernonRuntimeGraphicsPipelineDestroy(pipeline);
}

TEST_P(RuntimeRecordedGraphicsMatrix, AsynchronousExecuteCompletesThroughItsSubmission) {
    const std::vector<uint8_t> reference = leftReference();
    Target target(runtime());
    MutationOutcome submitted;
    VernonSubmission *submission = nullptr;
    {
        PassInvocation pass(executable_, instance_, target, {passVertices(left_)}, VERNON_RUNTIME_PROVIDER_LOAD_CLEAR,
                            false);
        ASSERT_EQ(pass.setCompletionMode(VERNON_COMPLETION_ASYNCHRONOUS), VERNON_STATUS_OK);
        ASSERT_EQ(pass.run(&submitted.outcome), VERNON_STATUS_OK) << lastError(runtime().runtime);
        EXPECT_EQ(submitted.outcome.submission, VERNON_INVOCATION_SUBMITTED);
        EXPECT_TRUE(submitted.all(VERNON_BOUNDARY_MUTATION_SUBMITTED));
        ASSERT_EQ(pass.takeSubmission(submission), VERNON_STATUS_OK) << lastError(runtime().runtime);
        ASSERT_NE(submission, nullptr);
        VernonSubmission *again = nullptr;
        EXPECT_NE(pass.takeSubmission(again), VERNON_STATUS_OK);
        EXPECT_EQ(again, nullptr);
    }
    MutationOutcome completed;
    VernonSubmissionState state{};
    ASSERT_EQ(vernonSubmissionGetState(submission, &state), VERNON_STATUS_OK);
    // The frame may finish between the two queries, so only a completed state fixes the early outcome.
    const VernonStatus early = vernonSubmissionGetMutationOutcome(submission, &completed.outcome);
    if (state == VERNON_SUBMISSION_PENDING)
        EXPECT_TRUE(early == VERNON_STATUS_NOT_READY || early == VERNON_STATUS_OK) << early;
    else
        EXPECT_EQ(early, VERNON_STATUS_OK);
    ASSERT_EQ(vernonSubmissionWait(submission), VERNON_STATUS_OK) << lastError(runtime().runtime);
    ASSERT_EQ(vernonSubmissionGetState(submission, &state), VERNON_STATUS_OK);
    EXPECT_EQ(state, VERNON_SUBMISSION_SUCCEEDED);
    ASSERT_EQ(vernonSubmissionGetMutationOutcome(submission, &completed.outcome), VERNON_STATUS_OK);
    EXPECT_EQ(completed.outcome.submission, VERNON_INVOCATION_COMPLETED);
    EXPECT_EQ(completed.outcome.mutation_count, submitted.outcome.mutation_count);
    EXPECT_TRUE(completed.all(VERNON_BOUNDARY_MUTATION_IN_PLACE_COMMITTED));
    vernonSubmissionDestroy(submission);
    EXPECT_EQ(target.pixels(), reference);
}

TEST_P(RuntimeRecordedGraphicsMatrix, WaitModeSubmissionIsTakenComplete) {
    Target target(runtime());
    PassInvocation pass(executable_, instance_, target, {passVertices(left_)}, VERNON_RUNTIME_PROVIDER_LOAD_CLEAR,
                        false);
    MutationOutcome executed;
    ASSERT_EQ(pass.run(&executed.outcome), VERNON_STATUS_OK) << lastError(runtime().runtime);
    EXPECT_EQ(executed.outcome.submission, VERNON_INVOCATION_COMPLETED);
    EXPECT_TRUE(executed.all(VERNON_BOUNDARY_MUTATION_IN_PLACE_COMMITTED));
    VernonSubmission *submission = nullptr;
    ASSERT_EQ(pass.takeSubmission(submission), VERNON_STATUS_OK) << lastError(runtime().runtime);
    VernonSubmissionState state{};
    ASSERT_EQ(vernonSubmissionGetState(submission, &state), VERNON_STATUS_OK);
    EXPECT_EQ(state, VERNON_SUBMISSION_SUCCEEDED);
    MutationOutcome completed;
    ASSERT_EQ(vernonSubmissionGetMutationOutcome(submission, &completed.outcome), VERNON_STATUS_OK);
    EXPECT_EQ(completed.outcome.submission, VERNON_INVOCATION_COMPLETED);
    EXPECT_EQ(completed.outcome.mutation_count, executed.outcome.mutation_count);
    EXPECT_TRUE(completed.all(VERNON_BOUNDARY_MUTATION_IN_PLACE_COMMITTED));
    vernonSubmissionDestroy(submission);
}

TEST_P(RuntimeRecordedGraphicsMatrix, PendingSubmissionsAreWaitedForOnDestroy) {
    const std::vector<uint8_t> reference = leftReference();
    {
        Target target(runtime());
        {
            PassInvocation pass(executable_, instance_, target, {passVertices(left_)},
                                VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, false);
            ASSERT_EQ(pass.setCompletionMode(VERNON_COMPLETION_ASYNCHRONOUS), VERNON_STATUS_OK);
            ASSERT_EQ(pass.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
        }
        EXPECT_EQ(target.pixels(), reference);
    }
    Target target(runtime());
    VernonSubmission *submission = nullptr;
    {
        PassInvocation pass(executable_, instance_, target, {passVertices(left_)}, VERNON_RUNTIME_PROVIDER_LOAD_CLEAR,
                            false);
        ASSERT_EQ(pass.setCompletionMode(VERNON_COMPLETION_ASYNCHRONOUS), VERNON_STATUS_OK);
        ASSERT_EQ(pass.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
        ASSERT_EQ(pass.takeSubmission(submission), VERNON_STATUS_OK) << lastError(runtime().runtime);
    }
    vernonSubmissionDestroy(submission);
    EXPECT_EQ(target.pixels(), reference);
}

TEST_P(RuntimeRecordedGraphicsMatrix, InvalidInvocationOptionsAndSubmissionRetrievalAreRejected) {
    Target target(runtime());
    PassInvocation pass(executable_, instance_, target, {passVertices(left_)}, VERNON_RUNTIME_PROVIDER_LOAD_CLEAR,
                        false);
    VernonSubmission *submission = nullptr;
    EXPECT_NE(pass.takeSubmission(submission), VERNON_STATUS_OK);
    EXPECT_EQ(submission, nullptr);
    const auto rejected = [&](auto edit) {
        VernonProgramInvocationOptions options{
            sizeof(options), VERNON_PROGRAM_INVOCATION_OPTIONS_VERSION, VERNON_COMPLETION_ASYNCHRONOUS, {}};
        edit(options);
        return vernonRuntimeProgramInvocationSetOptions(pass.handle(), &options) == VERNON_STATUS_INVALID_ARGUMENT;
    };
    EXPECT_TRUE(rejected([](VernonProgramInvocationOptions &options) { options.struct_size -= 1; }));
    EXPECT_TRUE(rejected([](VernonProgramInvocationOptions &options) { options.abi_version += 1; }));
    EXPECT_TRUE(rejected([](VernonProgramInvocationOptions &options) { options.reserved[3] = 1; }));
    EXPECT_TRUE(rejected([](VernonProgramInvocationOptions &options) {
        options.completion_mode = static_cast<VernonCompletionMode>(2);
    }));
    EXPECT_EQ(vernonRuntimeProgramInvocationSetOptions(pass.handle(), nullptr), VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(pass.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
    EXPECT_EQ(pass.setCompletionMode(VERNON_COMPLETION_ASYNCHRONOUS), VERNON_STATUS_INVALID_ARGUMENT);
}

TEST_P(RuntimeRecordedGraphicsMatrix, PipelineFromLaterExecutableDrawsInPassAndStandalone) {
    const std::vector<uint8_t> reference = sequentialReference();

    VernonProgramExecutable *source = vernonRuntimeResolveProgram(bundle_, nullptr);
    ASSERT_NE(source, nullptr) << lastError(runtime().runtime);
    Target target(runtime());
    {
        PassInvocation pass(executable_, instance_, target, {passVertices(left_)}, VERNON_RUNTIME_PROVIDER_LOAD_CLEAR,
                            true);
        VernonGraphicsPipeline *pipeline = nullptr;
        ASSERT_EQ(vernonRuntimeGraphicsPipelineCreate(executable_, pass.node(), source, &pipeline),
                  VERNON_RUNTIME_OPERATION_OK)
            << lastError(runtime().runtime);
        VernonBindingSet *right = verticesSet(runtime(), pipeline, source, right_.reference);

        vernonRuntimeProgramExecutableDestroy(source);
        EXPECT_NE(lastError(runtime().runtime).find("still owns live instances"), std::string::npos)
            << lastError(runtime().runtime);

        ASSERT_EQ(pass.record({bindPipeline(nullptr), draw(3), bindPipeline(pipeline), bindSet(right), draw(3)}),
                  VERNON_STATUS_OK);
        ASSERT_EQ(pass.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
        EXPECT_EQ(pass.telemetry().pipeline_binds, 2u);
        EXPECT_EQ(target.pixels(), reference);
        vernonRuntimeBindingSetDestroy(right);
        vernonRuntimeGraphicsPipelineDestroy(pipeline);
    }

    VernonProgramInstance *standaloneInstance = vernonRuntimeProgramInstanceCreate(source);
    ASSERT_NE(standaloneInstance, nullptr) << lastError(runtime().runtime);
    {
        Target standalone(runtime());
        PassInvocation single(source, standaloneInstance, standalone,
                              {verticesArgument(findParameter(source, "vertices"), right_.reference)},
                              VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, false);
        ASSERT_EQ(single.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
        EXPECT_GT(renderedPixels(standalone.pixels()), 0u);
    }
    vernonRuntimeProgramInstanceDestroy(standaloneInstance);
    vernonRuntimeProgramExecutableDestroy(source);
}

TEST_P(RuntimeRecordedGraphicsMatrix, IndexedDrawsHonourIndexOffsetAndBaseVertex) {
    Target reference(runtime());
    {
        PassInvocation single(executable_, instance_, reference, {passVertices(left_)},
                              VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, false);
        ASSERT_EQ(single.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
    }
    const std::vector<uint8_t> expected = reference.pixels();
    ASSERT_GT(renderedPixels(expected), 0u);

    constexpr std::array<uint32_t, 4> skipFirst{7, 0, 1, 2};
    RhiBuffer offsetIndices = vernon::tests::createBuffer(runtime(), sizeof(skipFirst), alignof(uint32_t),
                                                          VERNON_RHI_BUFFER_INDEX, skipFirst.data());
    {
        Target target(runtime());
        PassInvocation pass(executable_, instance_, target, {passVertices(left_)}, VERNON_RUNTIME_PROVIDER_LOAD_CLEAR,
                            true);
        const VernonIndexBinding index = indexBinding(offsetIndices.reference, sizeof(uint32_t), 0);
        ASSERT_EQ(pass.record({bindPipeline(nullptr), drawIndexed(index)}), VERNON_STATUS_OK);
        ASSERT_EQ(pass.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
        EXPECT_EQ(target.pixels(), expected);
    }

    std::array<float, 8> shifted{};
    std::copy(kLeftTriangle.begin(), kLeftTriangle.end(), shifted.begin() + 2);
    RhiBuffer shiftedVertices = vernon::tests::createBuffer(runtime(), sizeof(shifted), alignof(float),
                                                            VERNON_RHI_BUFFER_VERTEX, shifted.data());
    constexpr std::array<uint32_t, 3> shiftedIndices{1, 2, 3};
    RhiBuffer baseIndices = vernon::tests::createBuffer(runtime(), sizeof(shiftedIndices), alignof(uint32_t),
                                                        VERNON_RHI_BUFFER_INDEX, shiftedIndices.data());
    {
        Target target(runtime());
        const VernonProgramArgument vertices =
            verticesArgument(findParameter(executable_, "vertices"), shiftedVertices.reference, 2 * sizeof(float));
        PassInvocation pass(executable_, instance_, target, {vertices}, VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, true);
        const VernonIndexBinding index = indexBinding(baseIndices.reference, 0, -1);
        ASSERT_EQ(pass.record({bindPipeline(nullptr), drawIndexed(index)}), VERNON_STATUS_OK);
        if (direct3d()) {
            EXPECT_NE(pass.run(), VERNON_STATUS_OK);
        } else {
            ASSERT_EQ(pass.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
            EXPECT_EQ(target.pixels(), expected);
        }
    }
    {
        Target target(runtime());
        PassInvocation pass(executable_, instance_, target, {passVertices(left_)}, VERNON_RUNTIME_PROVIDER_LOAD_CLEAR,
                            true);
        ASSERT_EQ(pass.record({bindPipeline(nullptr), draw(3, 2)}), VERNON_STATUS_OK);
        if (!nativeFirstInstance()) {
            EXPECT_NE(pass.run(), VERNON_STATUS_OK);
        } else {
            ASSERT_EQ(pass.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
            EXPECT_EQ(target.pixels(), expected);
        }
    }
    for (RhiBuffer *buffer : {&offsetIndices, &shiftedVertices, &baseIndices})
        EXPECT_EQ(vernonRhiDeviceDestroyBuffer(runtime().device, buffer->handle), VERNON_RHI_STATUS_OK);
}

TEST_P(RuntimeRecordedGraphicsMatrix, RecordingWithoutDrawsAppliesTheScopeOperations) {
    Target target(runtime());
    {
        PassInvocation single(executable_, instance_, target, {passVertices(left_)}, VERNON_RUNTIME_PROVIDER_LOAD_CLEAR,
                              false);
        ASSERT_EQ(single.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
    }
    const std::vector<uint8_t> drawn = target.pixels();
    ASSERT_GT(renderedPixels(drawn), 0u);
    {
        PassInvocation preserve(executable_, instance_, target, {passVertices(left_)},
                                VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE, true);
        ASSERT_EQ(preserve.record({bindPipeline(nullptr)}), VERNON_STATUS_OK);
        ASSERT_EQ(preserve.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
        EXPECT_EQ(preserve.telemetry().draws, 0u);
        EXPECT_EQ(target.pixels(), drawn);
    }
    {
        PassInvocation clear(executable_, instance_, target, {passVertices(left_)}, VERNON_RUNTIME_PROVIDER_LOAD_CLEAR,
                             true);
        ASSERT_EQ(clear.record({bindPipeline(nullptr)}), VERNON_STATUS_OK);
        ASSERT_EQ(clear.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
        const VernonProgramGraphicsNodeTelemetry telemetry = clear.telemetry();
        EXPECT_EQ(telemetry.render_scopes, 1u);
        EXPECT_EQ(telemetry.draws, 0u);
    }
    std::vector<uint8_t> cleared(kExtent * kExtent * 4, 0);
    for (size_t pixel = 3; pixel < cleared.size(); pixel += 4)
        cleared[pixel] = 255;
    EXPECT_EQ(target.pixels(), cleared);
}

TEST_P(RuntimeRecordedGraphicsMatrix, InvalidRecordingsReportDiagnostics) {
    VernonGraphicsPipeline *unknown = nullptr;
    EXPECT_NE(vernonRuntimeGraphicsPipelineCreate(executable_, 9999, nullptr, &unknown), VERNON_RUNTIME_OPERATION_OK);
    EXPECT_EQ(unknown, nullptr);

    Target target(runtime());
    {
        PassInvocation pass(executable_, instance_, target, {passVertices(left_)}, VERNON_RUNTIME_PROVIDER_LOAD_CLEAR,
                            true);
        ASSERT_EQ(pass.record({draw(3)}), VERNON_STATUS_OK);
        EXPECT_EQ(pass.run(), VERNON_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(lastError(runtime().runtime), "PROGRAM_RECORDING_INVALID at node " + std::to_string(pass.node()) +
                                                    ".commands[0]: draw has no bound pipeline");
    }
    {
        PassInvocation pass(executable_, instance_, target, {passVertices(left_)}, VERNON_RUNTIME_PROVIDER_LOAD_CLEAR,
                            true);
        VernonGraphicsPipeline *pipeline = passPipeline(pass.node());
        VernonBindingSet *unsealed = nullptr;
        ASSERT_EQ(vernonRuntimeBindingSetCreate(pipeline, &unsealed), VERNON_RUNTIME_OPERATION_OK);
        EXPECT_EQ(pass.record({bindPipeline(nullptr), bindSet(unsealed), draw(3)}), VERNON_STATUS_INVALID_ARGUMENT);
        VernonProgramArgument output{};
        output.slot = findParameter(executable_, "output").slot;
        output.kind = VERNON_PROGRAM_IMAGE;
        output.image = {target.view.reference};
        EXPECT_EQ(vernonRuntimeBindingSetBind(unsealed, &output, nullptr), VERNON_STATUS_INVALID_ARGUMENT);
        vernonRuntimeBindingSetDestroy(unsealed);
        vernonRuntimeGraphicsPipelineDestroy(pipeline);
    }
}

const std::vector<vernon::tests::ProgramFixtureManifest> &moduleGraphicsBackendCases() {
    static const auto cases = vernon::tests::programFixtureCases("module_graphics");
    return cases;
}

INSTANTIATE_TEST_SUITE_P(EnabledTargets, RuntimeRecordedGraphicsMatrix, testing::ValuesIn(moduleGraphicsBackendCases()),
                         [](const testing::TestParamInfo<vernon::tests::ProgramFixtureManifest> &info) {
                             return std::string(info.param.target);
                         });

constexpr std::array<float, 4> kRed{1.0f, 0.0f, 0.0f, 1.0f};
constexpr std::array<float, 4> kTranslucentGreen{0.0f, 1.0f, 0.0f, 0.5f};
constexpr std::array<float, 4> kBlue{0.0f, 0.0f, 1.0f, 1.0f};
constexpr std::array<uint32_t, 4> kPackedMagenta{255, 0, 255, 255};
constexpr std::array<float, 6> kFullscreenTriangle{-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};
constexpr uint32_t kHeavyExtent = 1024;
constexpr uint32_t kLeftX = kExtent / 4;
constexpr uint32_t kRightX = 3 * kExtent / 4;
constexpr uint32_t kMiddleY = kExtent / 2;

using Pixel = std::array<int, 4>;

Pixel pixelAt(const std::vector<uint8_t> &pixels, uint32_t x, uint32_t y) {
    const size_t offset = (static_cast<size_t>(y) * kExtent + x) * 4;
    return {pixels[offset], pixels[offset + 1], pixels[offset + 2], pixels[offset + 3]};
}

// Blended 8-bit channels may round either way.
testing::AssertionResult nearPixel(const Pixel &actual, const Pixel &expected) {
    for (size_t channel = 0; channel < actual.size(); ++channel)
        if (std::abs(actual[channel] - expected[channel]) > 1)
            return testing::AssertionFailure() << "pixel (" << actual[0] << ", " << actual[1] << ", " << actual[2]
                                               << ", " << actual[3] << ") is not (" << expected[0] << ", "
                                               << expected[1] << ", " << expected[2] << ", " << expected[3] << ")";
    return testing::AssertionSuccess();
}

// Parameterised by the opaque pass fixture; the surface fixtures cooked for the same backend are its siblings.
class RuntimeRecordedSurfaceMatrix : public vernon::tests::ProgramFixtureRuntimeTest {
protected:
    vernon::tests::BackendTestRequirements requirements() const override {
        return vernon::tests::computeGraphicsFixtureRequirements(GetParam().runtime);
    }

    void SetUp() override {
        ProgramFixtureRuntimeTest::SetUp();
        if (IsSkipped() || HasFatalFailure())
            return;
        left_ = buffer(kLeftTriangle, VERNON_RHI_BUFFER_VERTEX | VERNON_RHI_BUFFER_STORAGE);
        right_ = buffer(kRightTriangle, VERNON_RHI_BUFFER_VERTEX | VERNON_RHI_BUFFER_STORAGE);
        fullscreen_ = buffer(kFullscreenTriangle, VERNON_RHI_BUFFER_VERTEX);
    }

    void TearDown() override {
        for (const vernon::tests::RhiSampler &owned : samplers_)
            EXPECT_EQ(vernonRhiDeviceDestroySampler(runtime().device, owned.handle), VERNON_RHI_STATUS_OK);
        for (VernonBindingSet *set : sets_)
            vernonRuntimeBindingSetDestroy(set);
        for (VernonGraphicsPipeline *created : pipelines_)
            vernonRuntimeGraphicsPipelineDestroy(created);
        for (VernonProgramInstance *instance : instances_)
            vernonRuntimeProgramInstanceDestroy(instance);
        for (auto executable = executables_.rbegin(); executable != executables_.rend(); ++executable)
            vernonRuntimeProgramExecutableDestroy(*executable);
        for (VernonProgramBundle *bundle : bundles_)
            vernonRuntimeProgramBundleDestroy(bundle);
        for (const RhiBuffer &owned : buffers_)
            EXPECT_EQ(vernonRhiDeviceDestroyBuffer(runtime().device, owned.handle), VERNON_RHI_STATUS_OK);
        ProgramFixtureRuntimeTest::TearDown();
    }

    template <typename T, size_t N> RhiBuffer buffer(const std::array<T, N> &data, uint32_t usage) {
        RhiBuffer result = vernon::tests::createBuffer(runtime(), sizeof(data), alignof(T), usage, data.data());
        EXPECT_NE(result.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
        buffers_.push_back(result);
        return result;
    }

    VernonProgramExecutable *resolve(std::string_view fixtureId) {
        const vernon::tests::ProgramFixtureManifest *manifest =
            vernon::tests::findProgramFixtureManifest(fixtureId, GetParam().target);
        EXPECT_NE(manifest, nullptr) << fixtureId;
        if (!manifest)
            return nullptr;
        VernonProgramBundle *bundle = loadBundle(runtime(), manifest->manifestPath);
        EXPECT_NE(bundle, nullptr) << lastError(runtime().runtime);
        if (!bundle)
            return nullptr;
        bundles_.push_back(bundle);
        VernonProgramExecutable *executable = vernonRuntimeResolveProgram(bundle, nullptr);
        EXPECT_NE(executable, nullptr) << lastError(runtime().runtime);
        if (executable)
            executables_.push_back(executable);
        return executable;
    }

    VernonProgramInstance *instance(VernonProgramExecutable *executable) {
        VernonProgramInstance *result = vernonRuntimeProgramInstanceCreate(executable);
        EXPECT_NE(result, nullptr) << lastError(runtime().runtime);
        if (result)
            instances_.push_back(result);
        return result;
    }

    static VernonProgramNodeId graphicsNode(VernonProgramExecutable *executable) {
        VernonProgramGraphicsControlsView controls{};
        EXPECT_EQ(vernonRuntimeProgramExecutableGetGraphicsControlsByIndex(executable, 0, &controls), VERNON_STATUS_OK);
        return controls.node;
    }

    VernonGraphicsPipeline *pipeline(VernonProgramExecutable *pass, VernonProgramExecutable *source) {
        VernonGraphicsPipeline *result = nullptr;
        EXPECT_EQ(vernonRuntimeGraphicsPipelineCreate(pass, graphicsNode(pass), source, &result),
                  VERNON_RUNTIME_OPERATION_OK)
            << lastError(runtime().runtime);
        if (result)
            pipelines_.push_back(result);
        return result;
    }

    VernonBindingSet *set(VernonGraphicsPipeline *created, std::initializer_list<VernonProgramArgument> arguments) {
        VernonBindingSet *result = sealedSet(runtime(), created, arguments);
        if (result)
            sets_.push_back(result);
        return result;
    }

    VernonProgramArgument vertices(VernonProgramExecutable *program, const RhiBuffer &resource) const {
        return verticesArgument(findParameter(program, "vertices"), resource.reference);
    }

    template <typename T>
    static VernonProgramArgument tint(VernonProgramExecutable *program, const std::array<T, 4> &value) {
        return hostArgument(findParameter(program, "tint"), value);
    }

    static VernonProgramArgument image(VernonProgramExecutable *program, std::string_view name,
                                       VernonRuntimeProviderResourceReference view) {
        VernonProgramArgument argument{};
        argument.slot = findParameter(program, name).slot;
        argument.kind = VERNON_PROGRAM_IMAGE;
        argument.image = {view};
        return argument;
    }

    static VernonProgramArgument samplerArgument(VernonProgramExecutable *program, std::string_view name,
                                                 VernonRuntimeProviderResourceReference resource) {
        VernonProgramArgument argument{};
        argument.slot = findParameter(program, name).slot;
        argument.kind = VERNON_PROGRAM_SAMPLER;
        argument.resource = resource;
        return argument;
    }

    VernonRuntimeProviderResourceReference sampler() {
        const vernon::tests::RhiSampler created = vernon::tests::createSampler(runtime());
        EXPECT_NE(created.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
        samplers_.push_back(created);
        return created.reference;
    }

    // Instances of the opaque pass's full-target triangle that keep the GPU busy for at least 100 ms per frame, so
    // that a test observes a frame while it is pending.
    uint32_t heavyInstances(VernonProgramExecutable *pass, VernonProgramInstance *passInstance) {
        Target target(runtime(), kHeavyExtent);
        for (uint32_t instances = 64; instances <= (1u << 20); instances *= 2) {
            PassInvocation invocation(pass, passInstance, target, {vertices(pass, fullscreen_), tint(pass, kRed)},
                                      VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, false, instances);
            const auto start = std::chrono::steady_clock::now();
            EXPECT_EQ(invocation.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
            if (std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(100))
                return instances;
        }
        ADD_FAILURE() << "no instance count keeps a frame on the GPU for 100 ms";
        return 0;
    }

    std::vector<VernonProgramBundle *> bundles_;
    std::vector<VernonProgramExecutable *> executables_;
    std::vector<VernonProgramInstance *> instances_;
    std::vector<VernonGraphicsPipeline *> pipelines_;
    std::vector<VernonBindingSet *> sets_;
    std::vector<RhiBuffer> buffers_;
    std::vector<vernon::tests::RhiSampler> samplers_;
    RhiBuffer left_;
    RhiBuffer right_;
    RhiBuffer fullscreen_;
};

TEST_P(RuntimeRecordedSurfaceMatrix, InterleavedPipelinesFollowRecordingOrder) {
    VernonProgramExecutable *pass = resolve("recorded_graphics_opaque_pass");
    VernonProgramExecutable *blendedSource = resolve("recorded_graphics_blended_surface");
    ASSERT_NE(pass, nullptr);
    ASSERT_NE(blendedSource, nullptr);
    VernonProgramInstance *passInstance = instance(pass);
    ASSERT_NE(passInstance, nullptr);

    VernonGraphicsPipeline *opaque = pipeline(pass, nullptr);
    VernonGraphicsPipeline *blended = pipeline(pass, blendedSource);
    ASSERT_NE(opaque, nullptr);
    ASSERT_NE(blended, nullptr);
    VernonBindingSet *geometry = set(opaque, {vertices(pass, left_)});
    VernonBindingSet *red = set(opaque, {tint(pass, kRed)});
    VernonBindingSet *green = set(blended, {tint(blendedSource, kTranslucentGreen)});
    const std::array<VernonGraphicsCommand, 3> opaqueDraw{bindPipeline(opaque), bindSet(red), draw(3)};
    const std::array<VernonGraphicsCommand, 3> blendedDraw{bindPipeline(blended), bindSet(green), draw(3)};

    // The pass binding would draw the right triangle in blue; the geometry set moves every draw to the left one and
    // stays bound across the switch to the blended pipeline of another executable.
    const auto render = [&](const auto &first, const auto &second) {
        Target target(runtime());
        PassInvocation invocation(pass, passInstance, target, {vertices(pass, right_), tint(pass, kBlue)},
                                  VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, true);
        std::vector<VernonGraphicsCommand> commands{bindPipeline(opaque), bindSet(geometry)};
        commands.insert(commands.end(), first.begin(), first.end());
        commands.insert(commands.end(), second.begin(), second.end());
        EXPECT_EQ(invocation.record(commands), VERNON_STATUS_OK) << lastError(runtime().runtime);
        EXPECT_EQ(invocation.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
        const VernonProgramGraphicsNodeTelemetry telemetry = invocation.telemetry();
        EXPECT_EQ(telemetry.render_scopes, 1u);
        EXPECT_EQ(telemetry.pipeline_binds, 3u);
        EXPECT_EQ(telemetry.draws, 2u);
        return target.pixels();
    };

    const std::vector<uint8_t> greenOverRed = render(opaqueDraw, blendedDraw);
    EXPECT_TRUE(nearPixel(pixelAt(greenOverRed, kLeftX, kMiddleY), {128, 128, 0, 255}));
    EXPECT_TRUE(nearPixel(pixelAt(greenOverRed, kRightX, kMiddleY), {0, 0, 0, 255}));
    const std::vector<uint8_t> redOverGreen = render(blendedDraw, opaqueDraw);
    EXPECT_TRUE(nearPixel(pixelAt(redOverGreen, kLeftX, kMiddleY), {255, 0, 0, 255}));
    EXPECT_TRUE(nearPixel(pixelAt(redOverGreen, kRightX, kMiddleY), {0, 0, 0, 255}));
}

TEST_P(RuntimeRecordedSurfaceMatrix, SurfaceTypesSharingParameterNamesBindTheirOwnSets) {
    VernonProgramExecutable *pass = resolve("recorded_graphics_opaque_pass");
    VernonProgramExecutable *packedSource = resolve("recorded_graphics_packed_tint_surface");
    VernonProgramExecutable *blendedSource = resolve("recorded_graphics_blended_surface");
    ASSERT_NE(pass, nullptr);
    ASSERT_NE(packedSource, nullptr);
    ASSERT_NE(blendedSource, nullptr);
    VernonProgramInstance *passInstance = instance(pass);
    ASSERT_NE(passInstance, nullptr);

    VernonGraphicsPipeline *opaque = pipeline(pass, nullptr);
    VernonGraphicsPipeline *packed = pipeline(pass, packedSource);
    VernonGraphicsPipeline *blended = pipeline(pass, blendedSource);
    ASSERT_NE(opaque, nullptr);
    ASSERT_NE(packed, nullptr);
    ASSERT_NE(blended, nullptr);
    VernonBindingSet *magenta = set(packed, {tint(packedSource, kPackedMagenta)});
    VernonBindingSet *rightGeometry = set(packed, {vertices(packedSource, right_)});

    Target target(runtime());
    {
        PassInvocation invocation(pass, passInstance, target, {vertices(pass, left_), tint(pass, kBlue)},
                                  VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, true);
        // `vertices` of the packed pipeline starts from the pass binding; its u32 `tint` does not, so the recording
        // binds it. Switching back to pipeline 0 unbinds that `tint`, and the pass binding (blue) applies again.
        ASSERT_EQ(invocation.record({bindPipeline(packed), bindSet(magenta), draw(3), bindPipeline(opaque),
                                     bindSet(rightGeometry), draw(3)}),
                  VERNON_STATUS_OK);
        ASSERT_EQ(invocation.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
        const std::vector<uint8_t> pixels = target.pixels();
        EXPECT_TRUE(nearPixel(pixelAt(pixels, kLeftX, kMiddleY), {255, 0, 255, 255}));
        EXPECT_TRUE(nearPixel(pixelAt(pixels, kRightX, kMiddleY), {0, 0, 255, 255}));
    }

    const std::string node = "node " + std::to_string(graphicsNode(pass));
    {
        PassInvocation mismatch(pass, passInstance, target, {vertices(pass, left_), tint(pass, kBlue)},
                                VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, true);
        ASSERT_EQ(mismatch.record({bindPipeline(blended), bindSet(magenta), draw(3)}), VERNON_STATUS_OK);
        EXPECT_EQ(mismatch.run(), VERNON_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(lastError(runtime().runtime),
                  "PROGRAM_BINDING_SET_MISMATCH at " + node +
                      ".commands[1].tint: parameter 'tint' has no counterpart with equal reflection in the pipeline");
    }
    {
        PassInvocation unbound(pass, passInstance, target, {vertices(pass, left_), tint(pass, kBlue)},
                               VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, true);
        ASSERT_EQ(unbound.record({bindPipeline(packed), draw(3)}), VERNON_STATUS_OK);
        EXPECT_EQ(unbound.run(), VERNON_STATUS_INVALID_ARGUMENT);
        const std::string error = lastError(runtime().runtime);
        EXPECT_EQ(error.rfind("PROGRAM_RECORDING_INVALID at " + node + ".commands[1]", 0), 0u) << error;
        EXPECT_NE(error.find("tint"), std::string::npos) << error;
    }
}

TEST_P(RuntimeRecordedSurfaceMatrix, ComputeWriteIsOrderedBeforeRecordedDraws) {
    VernonProgramExecutable *pass = resolve("recorded_graphics_mirrored_vertices_pass");
    ASSERT_NE(pass, nullptr);
    VernonProgramInstance *passInstance = instance(pass);
    ASSERT_NE(passInstance, nullptr);
    const RhiBuffer mirrored = buffer(kLeftTriangle, VERNON_RHI_BUFFER_VERTEX | VERNON_RHI_BUFFER_STORAGE);

    Target target(runtime());
    PassInvocation invocation(pass, passInstance, target, {vertices(pass, mirrored), tint(pass, kRed)},
                              VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, true);
    VernonGraphicsPipeline *passPipeline = pipeline(pass, nullptr);
    ASSERT_NE(passPipeline, nullptr);
    VernonBindingSet *blue = set(passPipeline, {tint(pass, kBlue)});
    ASSERT_EQ(invocation.record({bindPipeline(passPipeline), bindSet(blue), draw(3)}), VERNON_STATUS_OK);
    ASSERT_EQ(invocation.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
    EXPECT_EQ(invocation.telemetry().render_scopes, 1u);
    const std::vector<uint8_t> pixels = target.pixels();
    EXPECT_TRUE(nearPixel(pixelAt(pixels, kLeftX, kMiddleY), {0, 0, 0, 255}));
    EXPECT_TRUE(nearPixel(pixelAt(pixels, kRightX, kMiddleY), {0, 0, 255, 255}));
}

TEST_P(RuntimeRecordedSurfaceMatrix, FixedEndpointsAndForeignAttachmentsAreRejected) {
    VernonProgramExecutable *pass = resolve("recorded_graphics_mirrored_vertices_pass");
    VernonProgramExecutable *opaquePass = resolve("recorded_graphics_opaque_pass");
    VernonProgramExecutable *floatSource = resolve("recorded_graphics_float_target_surface");
    ASSERT_NE(pass, nullptr);
    ASSERT_NE(opaquePass, nullptr);
    ASSERT_NE(floatSource, nullptr);
    VernonProgramInstance *passInstance = instance(pass);
    ASSERT_NE(passInstance, nullptr);
    const RhiBuffer written = buffer(kLeftTriangle, VERNON_RHI_BUFFER_VERTEX | VERNON_RHI_BUFFER_STORAGE);

    VernonGraphicsPipeline *rejected = nullptr;
    EXPECT_NE(vernonRuntimeGraphicsPipelineCreate(pass, graphicsNode(pass), floatSource, &rejected),
              VERNON_RUNTIME_OPERATION_OK);
    EXPECT_EQ(rejected, nullptr);
    EXPECT_EQ(lastError(runtime().runtime),
              "PROGRAM_PIPELINE_CONFORMANCE at attachments: pipeline attachment signature differs from the pass Node");

    VernonGraphicsPipeline *passPipeline = pipeline(pass, nullptr);
    VernonGraphicsPipeline *opaquePipeline = pipeline(opaquePass, nullptr);
    ASSERT_NE(passPipeline, nullptr);
    ASSERT_NE(opaquePipeline, nullptr);
    VernonBindingSet *fixed = nullptr;
    ASSERT_EQ(vernonRuntimeBindingSetCreate(passPipeline, &fixed), VERNON_RUNTIME_OPERATION_OK);
    sets_.push_back(fixed);
    const VernonProgramArgument fixedVertices = vertices(pass, left_);
    EXPECT_EQ(vernonRuntimeBindingSetBind(fixed, &fixedVertices, nullptr), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(lastError(runtime().runtime),
              "PROGRAM_RECORDING_INVALID at vertices: parameter 'vertices' binds an endpoint the pass Node fixes");

    // The opaque pass's `vertices` is read-only and per-draw, so it has no counterpart in the mirrored pass.
    VernonBindingSet *readOnly = set(opaquePipeline, {vertices(opaquePass, left_)});
    Target target(runtime());
    PassInvocation invocation(pass, passInstance, target, {vertices(pass, written), tint(pass, kRed)},
                              VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, true);
    ASSERT_EQ(invocation.record({bindPipeline(passPipeline), bindSet(readOnly), draw(3)}), VERNON_STATUS_OK);
    EXPECT_EQ(invocation.run(), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(lastError(runtime().runtime),
              "PROGRAM_BINDING_SET_MISMATCH at node " + std::to_string(invocation.node()) +
                  ".commands[1].vertices: parameter 'vertices' has no counterpart with equal reflection in the "
                  "pipeline");
}

TEST_P(RuntimeRecordedSurfaceMatrix, ResourcesWrittenByTheInvocationAreHazards) {
    VernonProgramExecutable *pass = resolve("recorded_graphics_scratch_writing_pass");
    ASSERT_NE(pass, nullptr);
    VernonProgramInstance *passInstance = instance(pass);
    ASSERT_NE(passInstance, nullptr);
    const RhiBuffer scratch =
        buffer(kLeftTriangle, VERNON_RHI_BUFFER_VERTEX | VERNON_RHI_BUFFER_STORAGE | VERNON_RHI_BUFFER_INDEX);
    const VernonProgramArgument scratchArgument = verticesArgument(findParameter(pass, "scratch"), scratch.reference);
    VernonGraphicsPipeline *passPipeline = pipeline(pass, nullptr);
    ASSERT_NE(passPipeline, nullptr);
    VernonBindingSet *aliased = set(passPipeline, {vertices(pass, scratch)});
    const std::string node = "node " + std::to_string(graphicsNode(pass));

    Target target(runtime());
    {
        PassInvocation invocation(pass, passInstance, target,
                                  {vertices(pass, left_), scratchArgument, tint(pass, kRed)},
                                  VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, true);
        ASSERT_EQ(invocation.record({bindPipeline(passPipeline), bindSet(aliased), draw(3)}), VERNON_STATUS_OK);
        EXPECT_EQ(invocation.run(), VERNON_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(lastError(runtime().runtime),
                  "PROGRAM_BINDING_SET_HAZARD at " + node +
                      ".commands[1].vertices: resource is written by a Node of the same invocation");
    }
    {
        PassInvocation invocation(pass, passInstance, target,
                                  {vertices(pass, left_), scratchArgument, tint(pass, kRed)},
                                  VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, true);
        const VernonIndexBinding index = indexBinding(scratch.reference, 0, 0);
        ASSERT_EQ(invocation.record({bindPipeline(passPipeline), drawIndexed(index)}), VERNON_STATUS_OK);
        EXPECT_EQ(invocation.run(), VERNON_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(lastError(runtime().runtime),
                  "PROGRAM_BINDING_SET_HAZARD at " + node +
                      ".commands[1].index: index buffer is written by a Node of the same invocation");
    }
    {
        PassInvocation invocation(pass, passInstance, target,
                                  {vertices(pass, left_), scratchArgument, tint(pass, kRed)},
                                  VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, true);
        ASSERT_EQ(invocation.record({bindPipeline(passPipeline), draw(3)}), VERNON_STATUS_OK);
        ASSERT_EQ(invocation.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
        EXPECT_TRUE(nearPixel(pixelAt(target.pixels(), kLeftX, kMiddleY), {255, 0, 0, 255}));
    }
}

VernonSubmission *submitAsynchronously(PassInvocation &invocation, VernonRuntimeContext *context) {
    VernonSubmission *submission = nullptr;
    EXPECT_EQ(invocation.setCompletionMode(VERNON_COMPLETION_ASYNCHRONOUS), VERNON_STATUS_OK);
    EXPECT_EQ(invocation.run(), VERNON_STATUS_OK) << lastError(context);
    EXPECT_EQ(invocation.takeSubmission(submission), VERNON_STATUS_OK) << lastError(context);
    return submission;
}

VernonSubmissionTelemetry submissionTelemetry(VernonRuntimeContext *context) {
    VernonSubmissionTelemetry telemetry{};
    telemetry.struct_size = sizeof(telemetry);
    EXPECT_EQ(vernonRuntimeContextGetSubmissionTelemetry(context, &telemetry), VERNON_STATUS_OK);
    return telemetry;
}

TEST_P(RuntimeRecordedSurfaceMatrix, PendingSubmissionsKeepTheirExecutionWorkspace) {
    VernonProgramExecutable *pass = resolve("recorded_graphics_opaque_pass");
    ASSERT_NE(pass, nullptr);
    VernonProgramInstance *passInstance = instance(pass);
    ASSERT_NE(passInstance, nullptr);
    Target red(runtime());
    Target blue(runtime());
    // Each invocation is destroyed while its frame is pending, so the next one on the instance needs a workspace
    // other than the one the pending frame reads its uniform from.
    std::array<VernonSubmission *, 2> submissions{};
    {
        PassInvocation invocation(pass, passInstance, red, {vertices(pass, left_), tint(pass, kRed)},
                                  VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, false);
        submissions[0] = submitAsynchronously(invocation, runtime().runtime);
    }
    {
        PassInvocation invocation(pass, passInstance, blue, {vertices(pass, left_), tint(pass, kBlue)},
                                  VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, false);
        submissions[1] = submitAsynchronously(invocation, runtime().runtime);
    }
    for (VernonSubmission *submission : submissions) {
        ASSERT_NE(submission, nullptr);
        EXPECT_EQ(vernonSubmissionWait(submission), VERNON_STATUS_OK) << lastError(runtime().runtime);
        vernonSubmissionDestroy(submission);
    }
    EXPECT_TRUE(nearPixel(pixelAt(red.pixels(), kLeftX, kMiddleY), {255, 0, 0, 255}));
    EXPECT_TRUE(nearPixel(pixelAt(blue.pixels(), kLeftX, kMiddleY), {0, 0, 255, 255}));
}

TEST_P(RuntimeRecordedSurfaceMatrix, CreateOptionsAndTelemetryQueriesAreValidated) {
    VernonRuntimeCreateOptions options{};
    options.struct_size = sizeof(options);
    options.device_index = 1;
    EXPECT_EQ(vernonRuntimeCreateForRhiDevice(GetParam().runtime, runtime().device, &options), nullptr);
    options.device_index = 0;
    options.struct_size = sizeof(options) - 1;
    EXPECT_EQ(vernonRuntimeCreateForRhiDevice(GetParam().runtime, runtime().device, &options), nullptr);

    const VernonSubmissionTelemetry fresh = submissionTelemetry(runtime().runtime);
    EXPECT_EQ(fresh.struct_size, sizeof(fresh));
    EXPECT_EQ(fresh.pending_submissions, 0u);
    EXPECT_EQ(fresh.pending_limit_waits + fresh.transient_budget_waits, 0u);
    VernonSubmissionTelemetry truncated{};
    truncated.struct_size = sizeof(truncated) - 1;
    EXPECT_EQ(vernonRuntimeContextGetSubmissionTelemetry(runtime().runtime, &truncated),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRuntimeContextGetSubmissionTelemetry(runtime().runtime, nullptr), VERNON_STATUS_INVALID_ARGUMENT);
}

// The triangle as host memory, which an Execute copies into device storage that its submission owns.
VernonProgramArgument hostVerticesArgument(const VernonProgramParameterView &parameter,
                                           const std::array<float, 6> &triangle) {
    VernonProgramArgument argument = verticesArgument(parameter, {});
    argument.tensor.storage = VERNON_TENSOR_HOST;
    argument.tensor.host_data = triangle.data();
    return argument;
}

VernonSubmissionState submissionState(VernonSubmission *submission) {
    VernonSubmissionState state{};
    EXPECT_EQ(vernonSubmissionGetState(submission, &state), VERNON_STATUS_OK);
    return state;
}

// A second context over the fixture's device, created with `options`, that submits asynchronous frames of the opaque
// pass covering their whole target.
// Lowers the device's in-flight submission quota, which the device shares with every other context, for one scope.
class ScopedInFlightSubmissionLimit {
public:
    ScopedInFlightSubmissionLimit(VernonRhiDevice device, uint32_t limit) : device_(device) {
        shared_.struct_size = sizeof(shared_);
        if (vernonRhiDeviceGetCommandLimits(device_, &shared_) != VERNON_RHI_STATUS_OK)
            return;
        VernonRhiCommandLimits lowered = shared_;
        lowered.max_in_flight_submissions = limit;
        applied_ = vernonRhiDeviceSetCommandLimits(device_, &lowered) == VERNON_RHI_STATUS_OK;
    }
    ScopedInFlightSubmissionLimit(const ScopedInFlightSubmissionLimit &) = delete;
    ScopedInFlightSubmissionLimit &operator=(const ScopedInFlightSubmissionLimit &) = delete;
    ~ScopedInFlightSubmissionLimit() {
        if (applied_)
            EXPECT_EQ(vernonRhiDeviceSetCommandLimits(device_, &shared_), VERNON_RHI_STATUS_OK);
    }

    bool applied() const { return applied_; }

private:
    VernonRhiDevice device_;
    VernonRhiCommandLimits shared_{};
    bool applied_{};
};

class BoundedContext {
public:
    BoundedContext(RhiRuntime &shared, VernonRuntimeBackend backend, const std::filesystem::path &manifestPath,
                   const VernonRuntimeCreateOptions &options)
        : runtime_{shared.device, vernonRuntimeCreateForRhiDevice(backend, shared.device, &options)} {
        EXPECT_NE(runtime_.runtime, nullptr);
        if (!runtime_.runtime)
            return;
        bundle_ = loadBundle(runtime_, manifestPath);
        EXPECT_NE(bundle_, nullptr) << lastError(runtime_.runtime);
        executable_ = bundle_ ? vernonRuntimeResolveProgram(bundle_, nullptr) : nullptr;
        EXPECT_NE(executable_, nullptr) << lastError(runtime_.runtime);
        instance_ = executable_ ? vernonRuntimeProgramInstanceCreate(executable_) : nullptr;
        EXPECT_NE(instance_, nullptr) << lastError(runtime_.runtime);
        fullscreen_ = vernon::tests::createBuffer(runtime_, sizeof(kFullscreenTriangle), alignof(float),
                                                  VERNON_RHI_BUFFER_VERTEX, kFullscreenTriangle.data());
    }
    BoundedContext(const BoundedContext &) = delete;
    BoundedContext &operator=(const BoundedContext &) = delete;
    ~BoundedContext() {
        if (fullscreen_.handle.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            EXPECT_EQ(vernonRhiDeviceDestroyBuffer(runtime_.device, fullscreen_.handle), VERNON_RHI_STATUS_OK);
        if (instance_)
            vernonRuntimeProgramInstanceDestroy(instance_);
        if (executable_)
            vernonRuntimeProgramExecutableDestroy(executable_);
        if (bundle_)
            vernonRuntimeProgramBundleDestroy(bundle_);
        if (runtime_.runtime)
            EXPECT_EQ(vernonRuntimeDestroy(runtime_.runtime), VERNON_STATUS_OK) << lastError(runtime_.runtime);
    }

    bool ready() const { return instance_ && fullscreen_.handle.index != VERNON_RHI_INVALID_HANDLE_INDEX; }
    RhiRuntime &runtime() { return runtime_; }
    VernonSubmissionTelemetry telemetry() const { return submissionTelemetry(runtime_.runtime); }

    VernonSubmission *submit(const Target &target, uint32_t instances, bool hostVertices) {
        const VernonProgramParameterView vertices = findParameter(executable_, "vertices");
        PassInvocation invocation(executable_, instance_, target,
                                  {hostVertices ? hostVerticesArgument(vertices, kFullscreenTriangle)
                                                : verticesArgument(vertices, fullscreen_.reference),
                                   hostArgument(findParameter(executable_, "tint"), kRed)},
                                  VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, false, instances);
        return submitAsynchronously(invocation, runtime_.runtime);
    }

private:
    RhiRuntime runtime_;
    VernonProgramBundle *bundle_{};
    VernonProgramExecutable *executable_{};
    VernonProgramInstance *instance_{};
    RhiBuffer fullscreen_;
};

TEST_P(RuntimeRecordedSurfaceMatrix, ExecuteReturnsWhileAGpuHeavyFrameIsPending) {
    VernonProgramExecutable *pass = resolve("recorded_graphics_opaque_pass");
    ASSERT_NE(pass, nullptr);
    VernonProgramInstance *passInstance = instance(pass);
    ASSERT_NE(passInstance, nullptr);
    const uint32_t instances = heavyInstances(pass, passInstance);
    ASSERT_GT(instances, 0u);

    Target target(runtime(), kHeavyExtent);
    PassInvocation invocation(pass, passInstance, target, {vertices(pass, fullscreen_), tint(pass, kRed)},
                              VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, false, instances);
    VernonSubmission *submission = submitAsynchronously(invocation, runtime().runtime);
    ASSERT_NE(submission, nullptr);
    EXPECT_EQ(submissionState(submission), VERNON_SUBMISSION_PENDING);
    EXPECT_EQ(vernonSubmissionWait(submission), VERNON_STATUS_OK) << lastError(runtime().runtime);
    EXPECT_EQ(submissionState(submission), VERNON_SUBMISSION_SUCCEEDED);
    vernonSubmissionDestroy(submission);
}

TEST_P(RuntimeRecordedSurfaceMatrix, PendingSubmissionOutlivesItsExecutable) {
    const size_t idlePipelines = vernon::runtime::getRhiAdapterLivePreparedPipelineCount(runtime().runtime);
    VernonProgramExecutable *pass = resolve("recorded_graphics_opaque_pass");
    ASSERT_NE(pass, nullptr);
    VernonProgramInstance *passInstance = instance(pass);
    ASSERT_NE(passInstance, nullptr);
    const uint32_t instances = heavyInstances(pass, passInstance);
    ASSERT_GT(instances, 0u);

    Target target(runtime(), kHeavyExtent);
    VernonSubmission *submission = nullptr;
    {
        PassInvocation invocation(pass, passInstance, target, {vertices(pass, fullscreen_), tint(pass, kRed)},
                                  VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, false, instances);
        submission = submitAsynchronously(invocation, runtime().runtime);
    }
    ASSERT_NE(submission, nullptr);
    vernonRuntimeProgramInstanceDestroy(passInstance);
    instances_.clear();
    vernonRuntimeProgramExecutableDestroy(pass);
    executables_.clear();
    for (VernonProgramBundle *bundle : bundles_)
        vernonRuntimeProgramBundleDestroy(bundle);
    bundles_.clear();
    EXPECT_EQ(submissionState(submission), VERNON_SUBMISSION_PENDING);
    EXPECT_GT(vernon::runtime::getRhiAdapterLivePreparedPipelineCount(runtime().runtime), idlePipelines);
    EXPECT_EQ(vernonSubmissionWait(submission), VERNON_STATUS_OK) << lastError(runtime().runtime);
    EXPECT_EQ(submissionState(submission), VERNON_SUBMISSION_SUCCEEDED);
    vernonSubmissionDestroy(submission);
    EXPECT_EQ(vernon::runtime::getRhiAdapterLivePreparedPipelineCount(runtime().runtime), idlePipelines);
    EXPECT_TRUE(nearPixel(pixelAt(target.pixels(), kLeftX, kMiddleY), {255, 0, 0, 255}));
}

TEST_P(RuntimeRecordedSurfaceMatrix, ChainedAsynchronousFramesMatchWaitedFrames) {
    VernonProgramExecutable *pass = resolve("recorded_graphics_opaque_pass");
    VernonProgramExecutable *copy = resolve("recorded_graphics_sampled_pass");
    ASSERT_NE(pass, nullptr);
    ASSERT_NE(copy, nullptr);
    VernonProgramInstance *passInstance = instance(pass);
    VernonProgramInstance *copyInstance = instance(copy);
    ASSERT_NE(passInstance, nullptr);
    ASSERT_NE(copyInstance, nullptr);
    const VernonRuntimeProviderResourceReference nearest = sampler();

    // The second frame samples the image the first renders; queue order alone separates them.
    const auto render = [&](VernonCompletionMode mode) {
        Target rendered(runtime(), kExtent, VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_SAMPLED);
        Target copied(runtime());
        PassInvocation first(pass, passInstance, rendered, {vertices(pass, left_), tint(pass, kRed)},
                             VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, false);
        PassInvocation second(copy, copyInstance, copied,
                              {vertices(copy, fullscreen_), image(copy, "source", rendered.view.reference),
                               samplerArgument(copy, "source_sampler", nearest)},
                              VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, false);
        if (mode == VERNON_COMPLETION_WAIT) {
            EXPECT_EQ(first.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
            EXPECT_EQ(second.run(), VERNON_STATUS_OK) << lastError(runtime().runtime);
        } else {
            const std::array<VernonSubmission *, 2> submissions{submitAsynchronously(first, runtime().runtime),
                                                                submitAsynchronously(second, runtime().runtime)};
            for (VernonSubmission *submission : submissions) {
                EXPECT_EQ(vernonSubmissionWait(submission), VERNON_STATUS_OK) << lastError(runtime().runtime);
                vernonSubmissionDestroy(submission);
            }
        }
        return copied.pixels();
    };

    const std::vector<uint8_t> waited = render(VERNON_COMPLETION_WAIT);
    EXPECT_TRUE(nearPixel(pixelAt(waited, kLeftX, kMiddleY), {255, 0, 0, 255}));
    EXPECT_TRUE(nearPixel(pixelAt(waited, kRightX, kMiddleY), {0, 0, 0, 255}));
    EXPECT_EQ(render(VERNON_COMPLETION_ASYNCHRONOUS), waited);
}

TEST_P(RuntimeRecordedSurfaceMatrix, EveryAsynchronousFrameReadsItsOwnUniformAndVertices) {
    VernonProgramExecutable *pass = resolve("recorded_graphics_opaque_pass");
    ASSERT_NE(pass, nullptr);
    VernonProgramInstance *passInstance = instance(pass);
    ASSERT_NE(passInstance, nullptr);
    const RhiBuffer geometry = buffer(kLeftTriangle, VERNON_RHI_BUFFER_VERTEX);

    // Frame f rewrites the device-memory vertex buffer to the left or right triangle and binds its own tint; no frame
    // is waited for until all of them are submitted.
    constexpr uint32_t frames = 64;
    std::vector<std::array<float, 4>> tints(frames);
    std::vector<std::unique_ptr<Target>> targets;
    std::vector<VernonSubmission *> submissions;
    for (uint32_t frame = 0; frame < frames; ++frame) {
        const std::array<float, 6> &triangle = frame % 2 ? kRightTriangle : kLeftTriangle;
        ASSERT_EQ(vernonRhiDeviceUploadBuffer(runtime().device, geometry.handle, 0, triangle.data(), sizeof(triangle)),
                  VERNON_RHI_STATUS_OK);
        tints[frame] = {static_cast<float>(4 * frame) / 255.0f, static_cast<float>(255 - 4 * frame) / 255.0f, 0.0f,
                        1.0f};
        targets.push_back(std::make_unique<Target>(runtime()));
        PassInvocation invocation(pass, passInstance, *targets.back(),
                                  {vertices(pass, geometry), tint(pass, tints[frame])},
                                  VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, false);
        submissions.push_back(submitAsynchronously(invocation, runtime().runtime));
    }
    for (VernonSubmission *submission : submissions) {
        ASSERT_NE(submission, nullptr);
        EXPECT_EQ(vernonSubmissionWait(submission), VERNON_STATUS_OK) << lastError(runtime().runtime);
        vernonSubmissionDestroy(submission);
    }
    for (uint32_t frame = 0; frame < frames; ++frame) {
        const std::vector<uint8_t> pixels = targets[frame]->pixels();
        const int red = static_cast<int>(4 * frame);
        EXPECT_TRUE(nearPixel(pixelAt(pixels, frame % 2 ? kRightX : kLeftX, kMiddleY), {red, 255 - red, 0, 255}))
            << "frame " << frame;
        EXPECT_TRUE(nearPixel(pixelAt(pixels, frame % 2 ? kLeftX : kRightX, kMiddleY), {0, 0, 0, 255}))
            << "frame " << frame;
    }
}

TEST_P(RuntimeRecordedSurfaceMatrix, PendingLimitWaitsForTheOldestSubmission) {
    VernonProgramExecutable *pass = resolve("recorded_graphics_opaque_pass");
    ASSERT_NE(pass, nullptr);
    VernonProgramInstance *passInstance = instance(pass);
    ASSERT_NE(passInstance, nullptr);
    const uint32_t instances = heavyInstances(pass, passInstance);
    ASSERT_GT(instances, 0u);

    VernonRuntimeCreateOptions options{};
    options.struct_size = sizeof(options);
    options.max_pending_submissions = 1;
    BoundedContext bounded(runtime(), GetParam().runtime, GetParam().manifestPath, options);
    ASSERT_TRUE(bounded.ready());
    Target heavy(bounded.runtime(), kHeavyExtent);
    Target light(bounded.runtime());
    VernonSubmission *oldest = bounded.submit(heavy, instances, false);
    ASSERT_NE(oldest, nullptr);
    ASSERT_EQ(submissionState(oldest), VERNON_SUBMISSION_PENDING);
    VernonSubmission *newest = bounded.submit(light, 1, false);
    ASSERT_NE(newest, nullptr);
    EXPECT_EQ(submissionState(oldest), VERNON_SUBMISSION_SUCCEEDED);
    const VernonSubmissionTelemetry telemetry = bounded.telemetry();
    EXPECT_EQ(telemetry.pending_submissions, 1u);
    EXPECT_EQ(telemetry.pending_limit_waits, 1u);
    EXPECT_GT(telemetry.pending_limit_wait_nanoseconds, 0u);
    EXPECT_EQ(telemetry.transient_budget_waits, 0u);
    EXPECT_EQ(vernonSubmissionWait(newest), VERNON_STATUS_OK) << lastError(bounded.runtime().runtime);
    vernonSubmissionDestroy(newest);
    vernonSubmissionDestroy(oldest);
}

TEST_P(RuntimeRecordedSurfaceMatrix, DeviceQuotaWaitsForTheOldestSubmission) {
    VernonProgramExecutable *pass = resolve("recorded_graphics_opaque_pass");
    ASSERT_NE(pass, nullptr);
    VernonProgramInstance *passInstance = instance(pass);
    ASSERT_NE(passInstance, nullptr);
    const uint32_t instances = heavyInstances(pass, passInstance);
    ASSERT_GT(instances, 0u);

    // The context's own bounds admit both frames; only the device's in-flight quota of one holds the second back.
    VernonRuntimeCreateOptions options{};
    options.struct_size = sizeof(options);
    BoundedContext bounded(runtime(), GetParam().runtime, GetParam().manifestPath, options);
    ASSERT_TRUE(bounded.ready());
    Target heavy(bounded.runtime(), kHeavyExtent);
    Target light(bounded.runtime());
    const ScopedInFlightSubmissionLimit limit(bounded.runtime().device, 1);
    ASSERT_TRUE(limit.applied());

    VernonSubmission *oldest = bounded.submit(heavy, instances, false);
    ASSERT_NE(oldest, nullptr);
    ASSERT_EQ(submissionState(oldest), VERNON_SUBMISSION_PENDING);
    EXPECT_EQ(bounded.telemetry().pending_rhi_submissions, 1u);
    VernonSubmission *newest = bounded.submit(light, 1, false);
    ASSERT_NE(newest, nullptr) << lastError(bounded.runtime().runtime);
    EXPECT_EQ(submissionState(oldest), VERNON_SUBMISSION_SUCCEEDED);
    const VernonSubmissionTelemetry telemetry = bounded.telemetry();
    EXPECT_EQ(telemetry.pending_submissions, 1u);
    EXPECT_EQ(telemetry.pending_rhi_submissions, 1u);
    EXPECT_EQ(telemetry.device_quota_waits, 1u);
    EXPECT_GT(telemetry.device_quota_wait_nanoseconds, 0u);
    EXPECT_EQ(telemetry.pending_limit_waits + telemetry.transient_budget_waits, 0u);
    EXPECT_EQ(vernonSubmissionWait(newest), VERNON_STATUS_OK) << lastError(bounded.runtime().runtime);
    vernonSubmissionDestroy(newest);
    vernonSubmissionDestroy(oldest);
}

TEST_P(RuntimeRecordedSurfaceMatrix, TransientBudgetWaitsForTheOldestSubmission) {
    VernonProgramExecutable *pass = resolve("recorded_graphics_opaque_pass");
    ASSERT_NE(pass, nullptr);
    VernonProgramInstance *passInstance = instance(pass);
    ASSERT_NE(passInstance, nullptr);
    const uint32_t instances = heavyInstances(pass, passInstance);
    ASSERT_GT(instances, 0u);

    // Each frame owns the device copy of its host vertices, so with a one-byte budget the second frame's estimate,
    // the first frame's bytes, cannot be admitted while the first frame is pending.
    VernonRuntimeCreateOptions options{};
    options.struct_size = sizeof(options);
    options.transient_budget_bytes = 1;
    BoundedContext bounded(runtime(), GetParam().runtime, GetParam().manifestPath, options);
    ASSERT_TRUE(bounded.ready());
    Target heavy(bounded.runtime(), kHeavyExtent);
    Target light(bounded.runtime());
    VernonSubmission *oldest = bounded.submit(heavy, instances, true);
    ASSERT_NE(oldest, nullptr);
    ASSERT_EQ(submissionState(oldest), VERNON_SUBMISSION_PENDING);
    EXPECT_EQ(bounded.telemetry().pending_transient_bytes, sizeof(kFullscreenTriangle));
    VernonSubmission *newest = bounded.submit(light, 1, true);
    ASSERT_NE(newest, nullptr);
    EXPECT_EQ(submissionState(oldest), VERNON_SUBMISSION_SUCCEEDED);
    const VernonSubmissionTelemetry telemetry = bounded.telemetry();
    EXPECT_EQ(telemetry.pending_submissions, 1u);
    EXPECT_EQ(telemetry.pending_transient_bytes, sizeof(kFullscreenTriangle));
    EXPECT_EQ(telemetry.transient_budget_waits, 1u);
    EXPECT_GT(telemetry.transient_budget_wait_nanoseconds, 0u);
    EXPECT_EQ(telemetry.pending_limit_waits, 0u);
    EXPECT_EQ(vernonSubmissionWait(newest), VERNON_STATUS_OK) << lastError(bounded.runtime().runtime);
    vernonSubmissionDestroy(newest);
    vernonSubmissionDestroy(oldest);
}

TEST_P(RuntimeRecordedSurfaceMatrix, FailedSubmissionLeavesExactlyItsWrittenBoundariesIndeterminate) {
    VernonProgramExecutable *pass = resolve("recorded_graphics_opaque_pass");
    ASSERT_NE(pass, nullptr);
    VernonProgramInstance *passInstance = instance(pass);
    ASSERT_NE(passInstance, nullptr);
    Target failedTarget(runtime());
    Target completedTarget(runtime());
    PassInvocation failing(pass, passInstance, failedTarget, {vertices(pass, left_), tint(pass, kRed)},
                           VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, false);
    PassInvocation completing(pass, passInstance, completedTarget, {vertices(pass, right_), tint(pass, kBlue)},
                              VERNON_RUNTIME_PROVIDER_LOAD_CLEAR, false);
    MutationOutcome submitted;
    std::array<VernonSubmission *, 2> submissions{};
    {
        // The first completion observed is the failing frame's: both frames are pending before either is waited for.
        const ScopedFailureInjection failure(FailureBoundary::Completion);
        ASSERT_EQ(failing.setCompletionMode(VERNON_COMPLETION_ASYNCHRONOUS), VERNON_STATUS_OK);
        ASSERT_EQ(failing.run(&submitted.outcome), VERNON_STATUS_OK) << lastError(runtime().runtime);
        ASSERT_EQ(failing.takeSubmission(submissions[0]), VERNON_STATUS_OK) << lastError(runtime().runtime);
        submissions[1] = submitAsynchronously(completing, runtime().runtime);
        ASSERT_NE(submissions[1], nullptr);
        EXPECT_EQ(vernonSubmissionWait(submissions[0]), VERNON_STATUS_INTERNAL_ERROR);
        EXPECT_EQ(lastError(runtime().runtime), "injected GPU command DAG wait failure");
        EXPECT_EQ(vernonSubmissionWait(submissions[1]), VERNON_STATUS_OK) << lastError(runtime().runtime);
    }
    EXPECT_EQ(submissionState(submissions[0]), VERNON_SUBMISSION_FAILED);
    EXPECT_EQ(submissionState(submissions[1]), VERNON_SUBMISSION_SUCCEEDED);

    // Each frame writes its `output` boundary and its RenderPass attachments, and reads `vertices` and `tint`.
    using Written = std::vector<std::pair<VernonMutationBoundaryKind, uint32_t>>;
    const auto written = [](const MutationOutcome &result) {
        Written boundaries;
        for (size_t index = 0; index < result.outcome.mutation_count; ++index)
            boundaries.emplace_back(result.records[index].kind, result.records[index].slot);
        std::sort(boundaries.begin(), boundaries.end());
        return boundaries;
    };
    const Written expected{{VERNON_MUTATION_PROGRAM_BOUNDARY, findParameter(pass, "output").slot},
                           {VERNON_MUTATION_RENDER_PASS_CONTROL, failing.renderPassControl()}};
    MutationOutcome failed;
    ASSERT_EQ(vernonSubmissionGetMutationOutcome(submissions[0], &failed.outcome), VERNON_STATUS_OK);
    EXPECT_EQ(failed.outcome.submission, VERNON_INVOCATION_INDETERMINATE);
    EXPECT_EQ(written(failed), expected);
    EXPECT_EQ(written(submitted), expected);
    EXPECT_TRUE(failed.all(VERNON_BOUNDARY_MUTATION_INDETERMINATE));
    MutationOutcome completed;
    ASSERT_EQ(vernonSubmissionGetMutationOutcome(submissions[1], &completed.outcome), VERNON_STATUS_OK);
    EXPECT_EQ(completed.outcome.submission, VERNON_INVOCATION_COMPLETED);
    EXPECT_EQ(written(completed), expected);
    EXPECT_TRUE(completed.all(VERNON_BOUNDARY_MUTATION_IN_PLACE_COMMITTED));
    for (VernonSubmission *submission : submissions)
        vernonSubmissionDestroy(submission);
    EXPECT_TRUE(nearPixel(pixelAt(completedTarget.pixels(), kRightX, kMiddleY), {0, 0, 255, 255}));
}

const std::vector<vernon::tests::ProgramFixtureManifest> &recordedSurfaceBackendCases() {
    static const auto cases = vernon::tests::programFixtureCases("recorded_graphics_opaque_pass");
    return cases;
}

INSTANTIATE_TEST_SUITE_P(EnabledTargets, RuntimeRecordedSurfaceMatrix, testing::ValuesIn(recordedSurfaceBackendCases()),
                         [](const testing::TestParamInfo<vernon::tests::ProgramFixtureManifest> &info) {
                             return std::string(info.param.target);
                         });

} // namespace
