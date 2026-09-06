#include "engine_demo.h"
#include "graphics_host.h"

#include "VernonExecutionGraph.h"
#include "VernonRuntime.h"
#include "embedded_bundles.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

namespace {

constexpr uint64_t kVectorShape[] = {3};
constexpr int64_t kVectorStride[] = {sizeof(float)};
constexpr uint64_t kPositionShape[] = {3, 2};
constexpr int64_t kPositionStrides[] = {2 * sizeof(float), sizeof(float)};

std::string runtimeError(VernonRuntimeContext *runtime) {
    const VernonStringView error = vernonRuntimeGetLastError(runtime);
    return error.data ? std::string(error.data, error.size) : "unknown Runtime error";
}

bool findParameter(VernonProgramExecutable *pipeline, const char *name, size_t size,
                   VernonProgramParameterView &parameter) {
    return vernonRuntimeProgramExecutableFindParameter(pipeline, {name, size}, &parameter) == VERNON_STATUS_OK;
}

VernonProgramArgument hostArgument(const VernonProgramParameterView &parameter, const void *data, size_t size,
                                   uint32_t rank = 0, const uint64_t *shape = nullptr,
                                   const int64_t *strides = nullptr) {
    VernonProgramArgument argument{};
    argument.slot = parameter.slot;
    argument.kind = VERNON_PROGRAM_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_HOST;
    argument.tensor.host_data = data;
    argument.tensor.element_layout = parameter.element_layout;
    argument.tensor.access = parameter.access;
    argument.tensor.rank = rank;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = size;
    return argument;
}

class MandelbulbRenderPass final : public vernon::execution::RenderPass {
public:
    MandelbulbRenderPass(vernon::execution::GraphImage target, VernonRuntimeContext *runtime,
                         VernonProgramExecutable *pipeline, VernonProgramSubmitDescriptor *invocation)
        : RenderPass("mandelbulb-raymarch"), target_(target), runtime_(runtime), pipeline_(pipeline),
          invocation_(invocation) {}

    void declare() override {
        vernon::execution::ColorAttachmentUse attachment{};
        attachment.image = target_;
        attachment.load = VERNON_RHI_LOAD_CLEAR;
        attachment.store = VERNON_RHI_STORE_PRESERVE;
        attachment.clear[3] = 1.0F;
        color(0, attachment);
        renderArea(0, 0, target_.width, target_.height);
    }

    VernonRhiStatus execute(vernon::execution::GraphicsEncoder &encoder,
                            const vernon::execution::ExecutionResources &) override {
        VernonRuntimeProviderObject providerEncoder{};
        if (vernonRuntimeReferenceRhiCommandEncoder(runtime_, encoder.native(), &providerEncoder) != VERNON_STATUS_OK)
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        if (vernonRuntimeProgramEncode(providerEncoder, pipeline_, invocation_) == VERNON_STATUS_OK)
            return VERNON_RHI_STATUS_OK;
        std::cerr << "Mandelbulb encode failed: " << runtimeError(runtime_) << '\n';
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }

private:
    vernon::execution::GraphImage target_;
    VernonRuntimeContext *runtime_{};
    VernonProgramExecutable *pipeline_{};
    VernonProgramSubmitDescriptor *invocation_{};
};

class MandelbulbPanel final : public ExternalEnginePanel {
public:
    ~MandelbulbPanel() override { shutdown(); }

    bool initialize(GraphicsHost *graphics, bool headless) override {
        if (headless) {
            std::cerr << "the Mandelbulb graphics demo requires a browser or desktop window\n";
            return false;
        }
        graphics_ = graphics;
        if (!graphics_)
            return false;
#if defined(__EMSCRIPTEN__)
        runtime_ = vernonRuntimeCreateForRhiDevice(VERNON_RUNTIME_OPENGL_ES, graphics_->device());
#else
        runtime_ = vernonRuntimeCreateForRhiDevice(VERNON_RUNTIME_OPENGL, graphics_->device());
#endif
        if (!runtime_)
            return false;
        bundle_ =
            vernonRuntimeLoadProgramBundleWithOptions(runtime_, vernon_external_engine::graphics_bundle::kManifest,
                                                      vernon_external_engine::graphics_bundle::kManifestSize, nullptr);
        pipeline_ = bundle_ ? vernonRuntimeResolveProgram(bundle_, {nullptr, 0}) : nullptr;
        if (!pipeline_) {
            std::cerr << "failed to load the Mandelbulb pipeline: " << runtimeError(runtime_) << '\n';
            return false;
        }

        constexpr std::array<float, 6> positions{-1.0F, -1.0F, 3.0F, -1.0F, -1.0F, 3.0F};
        VernonRhiBufferDescriptor vertexDescriptor{};
        vertexDescriptor.struct_size = sizeof(vertexDescriptor);
        vertexDescriptor.size = sizeof(positions);
        vertexDescriptor.alignment = alignof(float);
        vertexDescriptor.usage = VERNON_RHI_BUFFER_VERTEX;
        vertexDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
        if (vernonRhiDeviceCreateBuffer(graphics_->device(), &vertexDescriptor, &vertexBuffer_) !=
                VERNON_RHI_STATUS_OK ||
            vernonRhiDeviceUploadBuffer(graphics_->device(), vertexBuffer_, 0, positions.data(), sizeof(positions)) !=
                VERNON_RHI_STATUS_OK ||
            vernonRuntimeReferenceRhiBuffer(runtime_, vertexBuffer_, 0, sizeof(positions), &vertexReference_) !=
                VERNON_STATUS_OK)
            return false;

        const char *names[] = {"position", "camera_position", "camera_target", "time",
                               "power",    "max_iterations",  "max_steps",     "shadow_steps"};
        std::array<VernonProgramParameterView, 8> parameters{};
        for (size_t index = 0; index < parameters.size(); ++index)
            if (!findParameter(pipeline_, names[index], std::char_traits<char>::length(names[index]),
                               parameters[index]))
                return false;
        arguments_[0] = hostArgument(parameters[0], nullptr, sizeof(positions), 2, kPositionShape, kPositionStrides);
        arguments_[0].tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
        arguments_[0].tensor.resource = vertexReference_;
        arguments_[1] = hostArgument(parameters[1], cameraPosition_.data(), sizeof(cameraPosition_), 1, kVectorShape,
                                     kVectorStride);
        arguments_[2] =
            hostArgument(parameters[2], cameraTarget_.data(), sizeof(cameraTarget_), 1, kVectorShape, kVectorStride);
        arguments_[3] = hostArgument(parameters[3], &time_, sizeof(time_));
        arguments_[4] = hostArgument(parameters[4], &power_, sizeof(power_));
        arguments_[5] = hostArgument(parameters[5], &maxIterations_, sizeof(maxIterations_));
        arguments_[6] = hostArgument(parameters[6], &maxSteps_, sizeof(maxSteps_));
        arguments_[7] = hostArgument(parameters[7], &shadowSteps_, sizeof(shadowSteps_));
        invocation_.struct_size = sizeof(invocation_);
        invocation_.abi_version = VERNON_PROGRAM_VERSION;
        invocation_.arguments = arguments_.data();
        invocation_.argument_count = arguments_.size();
        attachment_.location = 0;
        attachment_.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
        attachment_.store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
        attachment_.clear_color[3] = 1.0F;
        invocation_.color_attachments = &attachment_;
        invocation_.color_attachment_count = 1;
        invocation_.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
        invocation_.vertex_count = 3;
        invocation_.instance_count = 1;
        return true;
    }

    bool renderFrame(double elapsedSeconds, uint32_t panelWidth, uint32_t panelHeight) override {
        if (!panelWidth || !panelHeight)
            return true;
        if ((panelWidth != renderWidth_ || panelHeight != renderHeight_) &&
            !recreateRenderTarget(panelWidth, panelHeight))
            return false;
        const float phase = static_cast<float>(elapsedSeconds);
        const float angle = phase * 0.22F + 0.55F;
        cameraPosition_ = {3.15F * std::cos(angle), 0.48F + std::sin(phase * 0.17F) * 0.12F, 3.15F * std::sin(angle)};
        time_ = phase;
        power_ = 8.0F + std::sin(phase * 0.21F) * 0.18F;
        if (!graph_ || graph_->submit().wait() != VERNON_RHI_STATUS_OK)
            return false;
        if (frame_++ == 0)
            std::cout << "Vernon Mandelbulb Execution Graph animation started\n";
        return true;
    }

    VernonRhiImage image() const override { return image_; }
    uint32_t imageWidth() const override { return renderWidth_; }
    uint32_t imageHeight() const override { return renderHeight_; }

    void shutdown() override {
        graph_.reset();
        if (view_.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            vernonRhiDeviceDestroyImageView(graphics_->device(), view_);
        if (image_.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            vernonRhiDeviceDestroyImage(graphics_->device(), image_);
        if (vertexBuffer_.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            vernonRhiDeviceDestroyBuffer(graphics_->device(), vertexBuffer_);
        view_ = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
        image_ = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
        vertexBuffer_ = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
        if (pipeline_)
            vernonRuntimeProgramExecutableDestroy(pipeline_);
        if (bundle_)
            vernonRuntimeProgramBundleDestroy(bundle_);
        if (runtime_)
            vernonRuntimeDestroy(runtime_);
        pipeline_ = nullptr;
        bundle_ = nullptr;
        runtime_ = nullptr;
    }

private:
    bool recreateRenderTarget(uint32_t width, uint32_t height) {
        graph_.reset();
        if (view_.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            vernonRhiDeviceDestroyImageView(graphics_->device(), view_);
        if (image_.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            vernonRhiDeviceDestroyImage(graphics_->device(), image_);
        view_ = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
        image_ = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};

        VernonRhiImageDescriptor imageDescriptor{};
        imageDescriptor.struct_size = sizeof(imageDescriptor);
        imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
        imageDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
        imageDescriptor.width = width;
        imageDescriptor.height = height;
        imageDescriptor.depth = 1;
        imageDescriptor.mip_levels = 1;
        imageDescriptor.array_layers = 1;
        imageDescriptor.sample_count = 1;
        imageDescriptor.usage = VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_TRANSFER_SOURCE;
        if (vernonRhiDeviceCreateImage(graphics_->device(), &imageDescriptor, &image_) != VERNON_RHI_STATUS_OK)
            return false;
        VernonRhiImageViewDescriptor viewDescriptor{};
        viewDescriptor.struct_size = sizeof(viewDescriptor);
        viewDescriptor.image = image_;
        viewDescriptor.dimension = VERNON_RHI_IMAGE_2D;
        viewDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
        viewDescriptor.mip_level_count = 1;
        viewDescriptor.array_layer_count = 1;
        viewDescriptor.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
        if (vernonRhiDeviceCreateImageView(graphics_->device(), &viewDescriptor, &view_) != VERNON_RHI_STATUS_OK ||
            vernonRuntimeReferenceRhiImageView(runtime_, view_, &viewReference_) != VERNON_STATUS_OK)
            return false;

        renderWidth_ = width;
        renderHeight_ = height;
        attachment_.view = viewReference_;
        invocation_.viewport[2] = width;
        invocation_.viewport[3] = height;
        invocation_.scissor[2] = width;
        invocation_.scissor[3] = height;
        vernon::execution::ExecutionGraph graph(graphics_->device());
        const auto target = graph.importImage(image_, view_, true);
        graph.emplacePass<MandelbulbRenderPass>(target, runtime_, pipeline_, &invocation_);
        std::string error;
        graph_ = graph.compile(error);
        if (graph_)
            return true;
        std::cerr << "failed to compile Mandelbulb graph: " << error << '\n';
        return false;
    }

    GraphicsHost *graphics_{};
    VernonRuntimeContext *runtime_{};
    VernonProgramBundle *bundle_{};
    VernonProgramExecutable *pipeline_{};
    VernonRhiBuffer vertexBuffer_{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRuntimeProviderResourceReference vertexReference_{};
    VernonRhiImage image_{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiImageView view_{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRuntimeProviderResourceReference viewReference_{};
    std::array<float, 3> cameraPosition_{};
    std::array<float, 3> cameraTarget_{};
    float time_{};
    float power_{8.0F};
    int32_t maxIterations_{18};
    int32_t maxSteps_{112};
    int32_t shadowSteps_{32};
    std::array<VernonProgramArgument, 8> arguments_{};
    VernonColorAttachment attachment_{};
    VernonProgramSubmitDescriptor invocation_{};
    std::shared_ptr<vernon::execution::CompiledExecutionGraph> graph_;
    uint32_t renderWidth_{};
    uint32_t renderHeight_{};
    uint64_t frame_{};
};

} // namespace

std::unique_ptr<ExternalEnginePanel> createMandelbulbPanel() { return std::make_unique<MandelbulbPanel>(); }
