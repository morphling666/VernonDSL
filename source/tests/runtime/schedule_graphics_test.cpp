#include "VernonHal.h"
#include "VernonRuntime.h"
#include "VernonSchedule.h"
#include "program_fixture_manifest_table.h"

#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE)
#include "glfw_context_owner.h"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using vernon::tests::ProgramFixtureManifest;

constexpr uint32_t kExtent = 32;
// Windows widens a titled window to about 120 pixels, so the OpenGL window, and with it the surface, is larger than the
// ordinary targets.
constexpr uint32_t kSurfaceExtent = 128;
constexpr std::array<float, 6> kLeftTriangle{-0.9f, -0.9f, -0.1f, -0.9f, -0.5f, 0.9f};
constexpr std::array<float, 6> kRightTriangle{0.1f, -0.9f, 0.9f, -0.9f, 0.5f, 0.9f};
constexpr std::array<float, 4> kRed{1.0f, 0.0f, 0.0f, 1.0f};
constexpr uint32_t kLeftX = kExtent / 4;
constexpr uint32_t kRightX = 3 * kExtent / 4;
constexpr uint32_t kMiddleY = kExtent / 2;

std::string halError() {
    const VernonStringView error = vernonHalGetLastError();
    return error.data ? std::string(error.data, error.size) : std::string();
}

std::string lastError(VernonRuntimeContext *context) {
    const VernonStringView error = vernonRuntimeGetLastError(context);
    return std::string(error.data ? error.data : "", error.size);
}

VernonProgramParameterView findParameter(VernonProgramExecutable *executable, std::string_view name) {
    VernonProgramParameterView result{};
    EXPECT_EQ(vernonRuntimeProgramExecutableFindParameter(executable, {name.data(), name.size()}, &result),
              VERNON_STATUS_OK);
    return result;
}

VernonProgramArgument verticesArgument(const VernonProgramParameterView &parameter,
                                       VernonRuntimeProviderResourceReference resource) {
    static constexpr uint64_t shape[]{3, 2};
    static constexpr int64_t strides[]{2 * sizeof(float), sizeof(float)};
    VernonProgramArgument argument{};
    argument.slot = parameter.slot;
    argument.kind = VERNON_PROGRAM_TENSOR;
    argument.tensor.struct_size = sizeof(argument.tensor);
    argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    argument.tensor.resource = resource;
    argument.tensor.element_layout = parameter.element_layout;
    argument.tensor.access = parameter.access;
    argument.tensor.rank = 2;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = 3 * 2 * sizeof(float);
    return argument;
}

// `value` must outlive the argument.
template <typename T, size_t N>
VernonProgramArgument hostArgument(const VernonProgramParameterView &parameter, const std::array<T, N> &value) {
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

VernonIndexBinding indexBinding(VernonRuntimeProviderResourceReference indices) {
    return {VERNON_INDEX_U32, 0, 3, indices, 0};
}

using Pixel = std::array<int, 4>;

Pixel pixelAt(const std::vector<uint8_t> &pixels, uint32_t x, uint32_t y) {
    const size_t offset = (static_cast<size_t>(y) * kExtent + x) * 4;
    return {pixels[offset], pixels[offset + 1], pixels[offset + 2], pixels[offset + 3]};
}

// The module_graphics pass draws (255, 64, 0) over the opaque black clear.
bool drawn(const std::vector<uint8_t> &pixels, uint32_t x, uint32_t y) {
    const Pixel pixel = pixelAt(pixels, x, y);
    return pixel[0] == 255 && pixel[1] == 64 && pixel[2] == 0;
}

testing::AssertionResult drewTriangles(const std::vector<uint8_t> &pixels, bool left, bool right) {
    if (drawn(pixels, kExtent / 2, kMiddleY))
        return testing::AssertionFailure() << "the gap between the triangles is drawn";
    if (drawn(pixels, kLeftX, kMiddleY) != left)
        return testing::AssertionFailure() << "the left triangle is " << (left ? "missing" : "drawn");
    if (drawn(pixels, kRightX, kMiddleY) != right)
        return testing::AssertionFailure() << "the right triangle is " << (right ? "missing" : "drawn");
    return testing::AssertionSuccess();
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

std::vector<ProgramFixtureManifest> graphicsScheduleCases() {
    std::vector<ProgramFixtureManifest> result;
    for (const ProgramFixtureManifest &entry : vernon::tests::programFixtureManifestTable)
        if (entry.fixtureId == "module_graphics" &&
            (entry.runtime == VERNON_RUNTIME_VULKAN || entry.runtime == VERNON_RUNTIME_OPENGL ||
             entry.runtime == VERNON_RUNTIME_OPENGL_ES))
            result.push_back(entry);
    return result;
}

// The device's RGBA8 color target and a view of all of it.
class Target {
public:
    explicit Target(VernonHalDevice *device) {
        VernonImageDescriptor descriptor{};
        descriptor.dimension = VERNON_TEXTURE_2D;
        descriptor.extent = {kExtent, kExtent, 1};
        descriptor.format = VERNON_TEXTURE_RGBA8_UNORM;
        descriptor.mip_level_count = 1;
        descriptor.array_layer_count = 1;
        descriptor.sample_count = 1;
        descriptor.usage = VERNON_IMAGE_COLOR_ATTACHMENT | VERNON_IMAGE_TRANSFER_SOURCE;
        EXPECT_EQ(vernonHalCreateImage(device, &descriptor, &image_), VERNON_HAL_OK) << halError();
        if (!image_)
            return;
        VernonImageViewDescriptor view{};
        view.dimension = VERNON_TEXTURE_2D;
        view.format = VERNON_TEXTURE_RGBA8_UNORM;
        view.subresources = {0, 1, 0, 1, VERNON_IMAGE_ASPECT_COLOR};
        EXPECT_EQ(vernonHalCreateImageView(image_, &view, &view_), VERNON_HAL_OK) << halError();
        if (view_)
            EXPECT_EQ(vernonHalReferenceImageView(view_, &reference_), VERNON_HAL_OK) << halError();
    }
    Target(const Target &) = delete;
    Target &operator=(const Target &) = delete;
    ~Target() {
        if (view_)
            EXPECT_EQ(vernonHalDestroyImageView(view_), VERNON_HAL_OK) << halError();
        if (image_)
            EXPECT_EQ(vernonHalDestroyImage(image_), VERNON_HAL_OK) << halError();
    }

    VernonRuntimeProviderResourceReference reference() const { return reference_; }

    std::vector<uint8_t> pixels() const {
        std::vector<uint8_t> result(static_cast<size_t>(kExtent) * kExtent * 4);
        VernonHalImageRead read{};
        read.region.aspect = VERNON_IMAGE_ASPECT_COLOR;
        read.region.width = kExtent;
        read.region.height = kExtent;
        read.region.depth = 1;
        read.region.data_format = VERNON_HAL_IMAGE_DATA_RGBA;
        read.region.data_type = VERNON_HAL_IMAGE_DATA_UINT8;
        read.destination = result.data();
        read.size = result.size();
        EXPECT_EQ(vernonHalImageRead(image_, &read, 1), VERNON_HAL_OK) << halError();
        return result;
    }

private:
    VernonHalImage *image_{};
    VernonHalImageView *view_{};
    VernonRuntimeProviderResourceReference reference_{};
};

// A clearing, preserving color attachment and the render pass and dynamic state of a full kExtent target.
struct PassControls {
    VernonColorAttachment attachment{};
    VernonRenderPass renderPass{};
    VernonDynamicState dynamic{};
    const VernonRenderPass *pointer{&renderPass};

    explicit PassControls(VernonRuntimeProviderResourceReference view, uint32_t extent = kExtent) {
        attachment.view = view;
        attachment.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
        attachment.store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
        attachment.clear_color.f32[3] = 1.0f;
        renderPass.struct_size = sizeof(renderPass);
        renderPass.color_attachments = &attachment;
        renderPass.color_attachment_count = 1;
        renderPass.render_area[2] = extent;
        renderPass.render_area[3] = extent;
        dynamic.struct_size = sizeof(dynamic);
        dynamic.viewport[2] = static_cast<float>(extent);
        dynamic.viewport[3] = static_cast<float>(extent);
        dynamic.scissor[2] = extent;
        dynamic.scissor[3] = extent;
    }
    PassControls(const PassControls &) = delete;
    PassControls &operator=(const PassControls &) = delete;
};

// A linked schedule and the record it is called with.
class Linked {
public:
    Linked(VernonRuntimeContext *runtime, VernonLinkedSchedule *schedule) : runtime_(runtime), schedule_(schedule) {
        if (schedule_)
            record_.assign(vernonScheduleRecordTemplate(schedule_),
                           vernonScheduleRecordTemplate(schedule_) + vernonScheduleRecordSize(schedule_));
    }
    Linked(const Linked &) = delete;
    Linked &operator=(const Linked &) = delete;
    ~Linked() { vernonScheduleDestroy(schedule_); }

    explicit operator bool() const { return schedule_ != nullptr; }
    VernonLinkedSchedule *get() const { return schedule_; }
    size_t slotCount() const { return vernonScheduleSlotCount(schedule_); }
    const VernonScheduleSlot &slot(size_t index) const { return vernonScheduleSlots(schedule_)[index]; }

    template <typename T> void write(size_t index, const T &value) {
        ASSERT_LT(index, slotCount());
        ASSERT_EQ(slot(index).size, sizeof(value));
        std::memcpy(record_.data() + slot(index).offset, &value, sizeof(value));
    }

    template <typename T> void writeNode(size_t node, size_t index, const T &value) {
        const uint32_t linked = vernonScheduleNodeSlot(schedule_, node, index);
        ASSERT_NE(linked, UINT32_MAX);
        write(linked, value);
    }

    // Calls and, when the call succeeds, waits for it.
    VernonStatus call(uint64_t *point = nullptr) {
        uint64_t submitted = 0;
        const VernonStatus status = vernonScheduleCall(schedule_, record_.data(), record_.size(), &submitted);
        if (status == VERNON_STATUS_OK)
            EXPECT_EQ(vernonScheduleWait(runtime_, submitted), VERNON_STATUS_OK) << lastError(runtime_);
        if (point)
            *point = submitted;
        return status;
    }

    VernonScheduleGraphicsTelemetry telemetry() const {
        VernonScheduleGraphicsTelemetry result{};
        result.struct_size = sizeof(result);
        EXPECT_EQ(vernonScheduleGetGraphicsTelemetry(schedule_, &result), VERNON_STATUS_OK);
        return result;
    }

private:
    VernonRuntimeContext *runtime_;
    VernonLinkedSchedule *schedule_;
    std::vector<uint8_t> record_;
};

// The module_graphics fixture's targets on a VernonHal.h device: an OpenGL device adopts a hidden window's context,
// which is also the window its surface presents. A device the machine lacks skips the test.
class HalGraphicsSchedule : public testing::TestWithParam<ProgramFixtureManifest> {
protected:
    void SetUp() override {
        const VernonRuntimeBackend backend = GetParam().runtime;
        VernonHalDeviceDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.backend = backend;
        if (backend != VERNON_RUNTIME_VULKAN) {
#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE)
            const bool desktop = backend == VERNON_RUNTIME_OPENGL;
            try {
                context_ = std::make_unique<vernon::host::GlfwContextOwner>(
                    desktop ? vernon::host::GlfwContextApi::OpenGL : vernon::host::GlfwContextApi::OpenGLES,
                    desktop ? 4 : 3, desktop ? 3 : 2, kSurfaceExtent, kSurfaceExtent);
            } catch (const std::runtime_error &error) {
                GTEST_SKIP() << GetParam().target << " context is unavailable: " << error.what();
            }
            callbacks_ = context_->callbacks();
            descriptor.opengl = &callbacks_;
#else
            GTEST_SKIP() << "the GLFW context owner is not built";
#endif
        }
        if (vernonHalCreateDevice(&descriptor, &device_) != VERNON_HAL_OK)
            GTEST_SKIP() << GetParam().target << " device is unavailable: " << halError();
        runtime_ = vernonHalDeviceGetRuntime(device_);
        ASSERT_NE(runtime_, nullptr);
        executable_ = resolve(GetParam().manifestPath);
        ASSERT_NE(executable_, nullptr) << lastError(runtime_);
        left_ = vertexBuffer(kLeftTriangle);
        right_ = vertexBuffer(kRightTriangle);
    }

    void TearDown() override {
        for (VernonHalBuffer *buffer : buffers_)
            vernonHalDestroyBuffer(buffer);
        for (auto executable = executables_.rbegin(); executable != executables_.rend(); ++executable)
            vernonRuntimeProgramExecutableDestroy(*executable);
        for (VernonProgramBundle *bundle : bundles_)
            vernonRuntimeProgramBundleDestroy(bundle);
        if (device_)
            EXPECT_EQ(vernonHalDestroyDevice(device_), VERNON_HAL_OK) << halError();
    }

    // Resolves the cooked bundle whose blobs sit next to `manifestPath`; the fixture owns both.
    VernonProgramExecutable *resolve(const std::filesystem::path &manifestPath) {
        std::ifstream input(manifestPath, std::ios::binary);
        const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        EXPECT_FALSE(manifest.empty()) << manifestPath;
        const std::string directory = manifestPath.parent_path().u8string();
        VernonProgramBundleLoadOptions options{};
        options.struct_size = sizeof(options);
        options.bundle_directory = directory.c_str();
        VernonProgramBundle *bundle =
            vernonRuntimeLoadProgramBundleWithOptions(runtime_, manifest.data(), manifest.size(), &options);
        if (!bundle)
            return nullptr;
        bundles_.push_back(bundle);
        VernonProgramExecutable *executable = vernonRuntimeResolveProgram(bundle, nullptr);
        if (executable)
            executables_.push_back(executable);
        return executable;
    }

    VernonProgramExecutable *resolveFixture(std::string_view fixtureId) {
        const ProgramFixtureManifest *manifest =
            vernon::tests::findProgramFixtureManifest(fixtureId, GetParam().target);
        EXPECT_NE(manifest, nullptr) << fixtureId;
        return manifest ? resolve(manifest->manifestPath) : nullptr;
    }

    struct Resource {
        VernonHalBuffer *buffer{};
        VernonRuntimeProviderResourceReference reference{};
    };

    Resource buffer(const void *data, uint64_t size, uint32_t usage) {
        VernonHalBufferDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.size = size;
        descriptor.usage = usage;
        descriptor.memory = VERNON_HAL_MEMORY_DEVICE;
        Resource result;
        EXPECT_EQ(vernonHalCreateBuffer(device_, &descriptor, &result.buffer), VERNON_HAL_OK) << halError();
        if (!result.buffer)
            return result;
        buffers_.push_back(result.buffer);
        const VernonHalBufferWrite write{0, data, size};
        EXPECT_EQ(vernonHalBufferWrite(result.buffer, &write, 1), VERNON_HAL_OK) << halError();
        EXPECT_EQ(vernonHalReferenceBuffer(result.buffer, 0, size, &result.reference), VERNON_HAL_OK) << halError();
        return result;
    }

    template <size_t N> Resource vertexBuffer(const std::array<float, N> &vertices) {
        return buffer(vertices.data(), sizeof(vertices), VERNON_HAL_BUFFER_VERTEX | VERNON_HAL_BUFFER_STORAGE);
    }

    VernonProgramArgument passVertices(const Resource &vertices) const {
        return verticesArgument(findParameter(executable_, "vertices"), vertices.reference);
    }

    VernonLinkedSchedule *linkProgram(VernonProgramExecutable *executable) const {
        VernonScheduleProgramDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.graph = {"forward", 7};
        descriptor.executable = executable;
        return vernonScheduleLinkProgram(runtime_, &descriptor);
    }

    // A pipeline of the pass Node drawing with `source`'s Stage, null for the Node's own.
    VernonGraphicsPipeline *pipeline(VernonProgramExecutable *source) const {
        VernonProgramGraphicsControlsView controls{};
        EXPECT_EQ(vernonRuntimeProgramExecutableGetGraphicsControlsByIndex(executable_, 0, &controls),
                  VERNON_STATUS_OK);
        VernonGraphicsPipeline *result = nullptr;
        EXPECT_EQ(vernonRuntimeGraphicsPipelineCreate(executable_, controls.node, source, &result),
                  VERNON_RUNTIME_OPERATION_OK)
            << lastError(runtime_);
        return result;
    }

    VernonHalDevice *device_{};
    VernonRuntimeContext *runtime_{};
    VernonProgramExecutable *executable_{};
    Resource left_;
    Resource right_;
#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE)
    std::unique_ptr<vernon::host::GlfwContextOwner> context_;
    VernonOpenGLContextCallbacks callbacks_{};
#endif

private:
    std::vector<VernonProgramBundle *> bundles_;
    std::vector<VernonProgramExecutable *> executables_;
    std::vector<VernonHalBuffer *> buffers_;
};

// The graph links as one pass whose record holds the vertices argument, the RenderPass, the draw and the dynamic state.
TEST_P(HalGraphicsSchedule, AProgramLinksAsOnePass) {
    Linked schedule(runtime_, linkProgram(executable_));
    ASSERT_TRUE(schedule) << lastError(runtime_);
    ASSERT_EQ(schedule.slotCount(), 6u);
    EXPECT_EQ(schedule.slot(0).kind, VERNON_SCHEDULE_SLOT_ARGUMENT);
    EXPECT_EQ(schedule.slot(1).kind, VERNON_SCHEDULE_SLOT_PASS);
    EXPECT_EQ(schedule.slot(2).kind, VERNON_SCHEDULE_SLOT_DRAW);
    EXPECT_EQ(schedule.slot(3).kind, VERNON_SCHEDULE_SLOT_DYNAMIC);
    EXPECT_EQ(schedule.slot(4).kind, VERNON_SCHEDULE_SLOT_STREAM);
    EXPECT_EQ(schedule.slot(5).kind, VERNON_SCHEDULE_SLOT_TABLE);

    Target target(device_);
    PassControls controls(target.reference());
    schedule.write(0, passVertices(left_));
    schedule.write(1, controls.pointer);
    schedule.write(3, controls.dynamic);
    ASSERT_EQ(schedule.call(), VERNON_STATUS_OK) << lastError(runtime_);
    EXPECT_TRUE(drewTriangles(target.pixels(), true, false));

    // A second call with other vertices preserves the first triangle.
    controls.attachment.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE;
    schedule.write(0, passVertices(right_));
    ASSERT_EQ(schedule.call(), VERNON_STATUS_OK) << lastError(runtime_);
    EXPECT_TRUE(drewTriangles(target.pixels(), true, true));
}

// Two graph nodes into one target: the second preserves what the first cleared and drew, so their passes share a
// scope; a third node clears another target and begins its own.
TEST_P(HalGraphicsSchedule, GraphPassesDrawInOrder) {
    VernonScheduleProgramDescriptor nodes[3]{};
    for (VernonScheduleProgramDescriptor &node : nodes) {
        node.struct_size = sizeof(node);
        node.graph = {"forward", 7};
        node.executable = executable_;
    }
    VernonScheduleGraphDescriptor graph{};
    graph.struct_size = sizeof(graph);
    graph.nodes = nodes;
    graph.node_count = 3;
    Linked schedule(runtime_, vernonScheduleLinkGraph(runtime_, &graph));
    ASSERT_TRUE(schedule) << lastError(runtime_);

    Target shared(device_);
    Target other(device_);
    PassControls first(shared.reference());
    PassControls second(shared.reference());
    PassControls third(other.reference());
    second.attachment.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE;
    const PassControls *controls[3]{&first, &second, &third};
    for (size_t node = 0; node < 3; ++node) {
        schedule.writeNode(node, 0, passVertices(node == 1 ? right_ : left_));
        schedule.writeNode(node, 1, controls[node]->pointer);
    }
    ASSERT_EQ(schedule.call(), VERNON_STATUS_OK) << lastError(runtime_);
    EXPECT_TRUE(drewTriangles(shared.pixels(), true, true));
    EXPECT_TRUE(drewTriangles(other.pixels(), true, false));
    const VernonScheduleGraphicsTelemetry telemetry = schedule.telemetry();
    EXPECT_EQ(telemetry.render_scopes, 2u);
    EXPECT_EQ(telemetry.draws, 3u);
}

// A stream pass draws its counted records: the second record reaches the right triangle through its base vertex and
// starts at instance 1. A host stream is read during the call; a device stream reads the same records from buffers.
TEST_P(HalGraphicsSchedule, StreamsDrawTheirCountedRecords) {
    const bool vulkan = GetParam().runtime == VERNON_RUNTIME_VULKAN;
    const bool desktopGl = GetParam().runtime == VERNON_RUNTIME_OPENGL;
    std::array<float, 12> triangles{};
    std::copy(kLeftTriangle.begin(), kLeftTriangle.end(), triangles.begin());
    std::copy(kRightTriangle.begin(), kRightTriangle.end(), triangles.begin() + 6);
    const Resource vertices = vertexBuffer(triangles);
    constexpr std::array<uint32_t, 3> indexValues{0, 1, 2};
    const Resource indices = buffer(indexValues.data(), sizeof(indexValues), VERNON_HAL_BUFFER_INDEX);
    const std::array<VernonRuntimeProviderDrawRecord, 3> records{{{3, 1, 0, 0, 0}, {3, 1, 0, 3, 1}, {0, 0, 0, 0, 0}}};
    const Resource recordBuffer =
        buffer(records.data(), sizeof(records), VERNON_HAL_BUFFER_INDIRECT | VERNON_HAL_BUFFER_STORAGE);
    uint32_t count = 2;
    const Resource countBuffer = buffer(&count, sizeof(count), VERNON_HAL_BUFFER_INDIRECT | VERNON_HAL_BUFFER_STORAGE);

    Linked schedule(runtime_, linkProgram(executable_));
    ASSERT_TRUE(schedule) << lastError(runtime_);
    Target target(device_);
    PassControls controls(target.reference());
    const VernonIndexBinding index = indexBinding(indices.reference);
    VernonDrawCommand draw{};
    draw.struct_size = sizeof(draw);
    draw.index_binding = &index;
    draw.instance_count = 1;
    schedule.write(0, passVertices(vertices));
    schedule.write(1, controls.pointer);
    schedule.write(2, draw);

    VernonScheduleDrawStream stream{};
    stream.struct_size = sizeof(stream);
    stream.max_records = static_cast<uint32_t>(records.size());
    stream.host_records = records.data();
    stream.host_counts = &count;
    const auto call = [&]() {
        schedule.write(4, stream);
        return schedule.call();
    };
    ASSERT_EQ(call(), VERNON_STATUS_OK) << lastError(runtime_);
    EXPECT_TRUE(drewTriangles(target.pixels(), true, true));
    EXPECT_EQ(schedule.telemetry().draws, 1u);

    count = 1;
    ASSERT_EQ(call(), VERNON_STATUS_OK) << lastError(runtime_);
    EXPECT_TRUE(drewTriangles(target.pixels(), true, false));

    // A host record whose indices leave the index buffer is rejected before it is drawn.
    const std::array<VernonRuntimeProviderDrawRecord, 1> outside{{{3, 1, 1, 0, 0}}};
    stream.host_records = outside.data();
    stream.max_records = 1;
    EXPECT_NE(call(), VERNON_STATUS_OK);

    stream.host_records = nullptr;
    stream.host_counts = nullptr;
    stream.max_records = static_cast<uint32_t>(records.size());
    stream.records = recordBuffer.reference;
    stream.counts = countBuffer.reference;
    if (vulkan) {
        ASSERT_EQ(call(), VERNON_STATUS_OK) << lastError(runtime_);
        EXPECT_TRUE(drewTriangles(target.pixels(), true, true));
    } else if (!desktopGl) {
        EXPECT_NE(call(), VERNON_STATUS_OK);
    }

    // Without counts a device stream draws all its records, the zeroed third drawing nothing: one multi-draw on
    // OpenGL 4.3.
    stream.counts = {};
    if (vulkan || desktopGl) {
        ASSERT_EQ(call(), VERNON_STATUS_OK) << lastError(runtime_);
        EXPECT_TRUE(drewTriangles(target.pixels(), true, true));
        stream.max_records = 1;
        ASSERT_EQ(call(), VERNON_STATUS_OK) << lastError(runtime_);
        EXPECT_TRUE(drewTriangles(target.pixels(), true, false));
    } else {
        EXPECT_NE(call(), VERNON_STATUS_OK);
    }
}

// A pass table draws its entries in order, each with its own pipeline and arguments: the node's Stage draws the left
// triangle, a pipeline of another resolution of the Program the right one, into the node's attachments.
TEST_P(HalGraphicsSchedule, PassTablesDrawEachEntryWithItsPipeline) {
    VernonProgramGraphicsControlsView graphics{};
    ASSERT_EQ(vernonRuntimeProgramExecutableGetGraphicsControlsByIndex(executable_, 0, &graphics), VERNON_STATUS_OK);
    VernonProgramExecutable *source = resolve(GetParam().manifestPath);
    ASSERT_NE(source, nullptr) << lastError(runtime_);
    VernonGraphicsPipeline *other = pipeline(source);
    ASSERT_NE(other, nullptr);

    constexpr std::array<uint32_t, 3> indexValues{0, 1, 2};
    const Resource indices = buffer(indexValues.data(), sizeof(indexValues), VERNON_HAL_BUFFER_INDEX);
    const VernonIndexBinding index = indexBinding(indices.reference);
    const VernonRuntimeProviderDrawRecord drawRecord{3, 1, 0, 0, 0};
    const uint32_t count = 1;
    VernonScheduleDrawStream stream{};
    stream.struct_size = sizeof(stream);
    stream.max_records = 1;
    stream.host_records = &drawRecord;
    stream.host_counts = &count;
    const VernonProgramArgument leftVertices = passVertices(left_);
    const VernonProgramArgument rightVertices = verticesArgument(findParameter(source, "vertices"), right_.reference);
    VernonSchedulePassEntry entries[2]{};
    for (VernonSchedulePassEntry &entry : entries) {
        entry.struct_size = sizeof(entry);
        entry.argument_count = 1;
        entry.draw.struct_size = sizeof(entry.draw);
    }
    // Entry 0 draws a stream of the indices; entry 1 draws its vertices directly, inside the right half only.
    entries[0].arguments = &leftVertices;
    entries[0].draw.index_binding = &index;
    entries[0].stream = stream;
    entries[1].pipeline = other;
    entries[1].arguments = &rightVertices;
    entries[1].draw.vertex_count = 3;
    entries[1].draw.instance_count = 1;
    entries[1].dynamic.struct_size = sizeof(entries[1].dynamic);
    entries[1].dynamic.viewport[2] = kExtent;
    entries[1].dynamic.viewport[3] = kExtent;
    entries[1].dynamic.scissor[0] = kExtent / 2;
    entries[1].dynamic.scissor[2] = kExtent - kExtent / 2;
    entries[1].dynamic.scissor[3] = kExtent;
    VernonSchedulePassTable table{};
    table.struct_size = sizeof(table);
    table.entries = entries;
    table.entry_count = 2;
    const VernonSchedulePassTable *tablePointer = &table;

    Linked schedule(runtime_, linkProgram(executable_));
    ASSERT_TRUE(schedule) << lastError(runtime_);
    ASSERT_EQ(schedule.slotCount(), 6u);
    ASSERT_EQ(schedule.slot(5).kind, VERNON_SCHEDULE_SLOT_TABLE);
    Target target(device_);
    PassControls controls(target.reference());
    schedule.write(1, controls.pointer);
    schedule.write(5, tablePointer);
    ASSERT_EQ(schedule.call(), VERNON_STATUS_OK) << lastError(runtime_);
    EXPECT_TRUE(drewTriangles(target.pixels(), true, true));
    const VernonScheduleGraphicsTelemetry telemetry = schedule.telemetry();
    EXPECT_EQ(telemetry.render_scopes, 1u);
    EXPECT_EQ(telemetry.draws, 2u);

    // A per-draw parameter the entry leaves unbound draws with the node's own binding, as a recording's does.
    entries[1].argument_count = 0;
    schedule.write(0, passVertices(left_));
    ASSERT_EQ(schedule.call(), VERNON_STATUS_OK) << lastError(runtime_);
    EXPECT_TRUE(drewTriangles(target.pixels(), true, false));

    // Debug commands leave the pixels alone, and a timed call measures its one render scope.
    const VernonScheduleDebugCommand groups[]{{VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP, {"left", 4}},
                                              {VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP, {}},
                                              {VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP, {"right", 5}},
                                              {VERNON_GRAPHICS_COMMAND_INSERT_DEBUG_MARKER, {"marker", 6}},
                                              {VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP, {}}};
    entries[0].debug_commands = &groups[0];
    entries[0].debug_command_count = 1;
    entries[1].debug_commands = &groups[1];
    entries[1].debug_command_count = 3;
    table.debug_commands = &groups[4];
    table.debug_command_count = 1;
    ASSERT_EQ(vernonScheduleSetTimings(schedule.get(), 1), VERNON_STATUS_OK);
    uint64_t point = 0;
    ASSERT_EQ(schedule.call(&point), VERNON_STATUS_OK) << lastError(runtime_);
    EXPECT_TRUE(drewTriangles(target.pixels(), true, false));
    size_t timed = 0;
    const VernonStatus timing = vernonScheduleGetScopeTimings(schedule.get(), point, nullptr, 0, &timed);
    if (timing != VERNON_STATUS_UNSUPPORTED_TARGET) {
        ASSERT_EQ(timing, VERNON_STATUS_OK) << lastError(runtime_);
        ASSERT_EQ(timed, 1u);
        VernonScopeTiming scope{};
        ASSERT_EQ(vernonScheduleGetScopeTimings(schedule.get(), point, &scope, 1, &timed), VERNON_STATUS_OK);
        EXPECT_EQ(scope.first_node, graphics.node);
        EXPECT_EQ(scope.node_count, 1u);
        EXPECT_LE(scope.begin_nanoseconds, scope.end_nanoseconds);
    }
    ASSERT_EQ(vernonScheduleSetTimings(schedule.get(), 0), VERNON_STATUS_OK);
    ASSERT_EQ(schedule.call(&point), VERNON_STATUS_OK) << lastError(runtime_);
    EXPECT_EQ(vernonScheduleGetScopeTimings(schedule.get(), point, nullptr, 0, &timed),
              VERNON_STATUS_UNSUPPORTED_TARGET);

    // A table must pop every group it pushes.
    table.debug_command_count = 0;
    EXPECT_NE(schedule.call(), VERNON_STATUS_OK);
    vernonRuntimeGraphicsPipelineDestroy(other);
}

// A table kept under one revision is planned once and its scope replayed, while the counts of its host stream still
// change what it draws: each call draws what a table planned anew (revision zero) draws.
TEST_P(HalGraphicsSchedule, PassTableRevisionsReplayWithTheirCurrentRecords) {
    VernonProgramExecutable *source = resolve(GetParam().manifestPath);
    ASSERT_NE(source, nullptr) << lastError(runtime_);
    VernonGraphicsPipeline *other = pipeline(source);
    ASSERT_NE(other, nullptr);
    constexpr std::array<uint32_t, 3> indexValues{0, 1, 2};
    const Resource indices = buffer(indexValues.data(), sizeof(indexValues), VERNON_HAL_BUFFER_INDEX);
    const VernonIndexBinding index = indexBinding(indices.reference);
    const VernonRuntimeProviderDrawRecord drawRecord{3, 1, 0, 0, 0};
    uint32_t count = 1;
    const VernonProgramArgument leftVertices = passVertices(left_);
    const VernonProgramArgument rightVertices = verticesArgument(findParameter(source, "vertices"), right_.reference);
    VernonSchedulePassEntry entries[2]{};
    for (VernonSchedulePassEntry &entry : entries) {
        entry.struct_size = sizeof(entry);
        entry.argument_count = 1;
        entry.draw.struct_size = sizeof(entry.draw);
    }
    entries[0].arguments = &leftVertices;
    entries[0].draw.index_binding = &index;
    entries[0].stream.struct_size = sizeof(entries[0].stream);
    entries[0].stream.max_records = 1;
    entries[0].stream.host_records = &drawRecord;
    entries[0].stream.host_counts = &count;
    entries[1].pipeline = other;
    entries[1].arguments = &rightVertices;
    entries[1].draw.vertex_count = 3;
    entries[1].draw.instance_count = 1;
    VernonSchedulePassTable table{};
    table.struct_size = sizeof(table);
    table.entries = entries;
    table.entry_count = 2;
    const VernonSchedulePassTable *tablePointer = &table;

    Linked schedule(runtime_, linkProgram(executable_));
    ASSERT_TRUE(schedule) << lastError(runtime_);
    Target target(device_);
    PassControls controls(target.reference());
    schedule.write(1, controls.pointer);
    schedule.write(5, tablePointer);
    const auto draw = [&](uint64_t revision) {
        table.revision = revision;
        EXPECT_EQ(schedule.call(), VERNON_STATUS_OK) << lastError(runtime_);
        return target.pixels();
    };

    const std::vector<uint8_t> both = draw(1);
    EXPECT_TRUE(drewTriangles(both, true, true));
    EXPECT_EQ(draw(1), both);
    count = 0;
    const std::vector<uint8_t> right = draw(1);
    EXPECT_TRUE(drewTriangles(right, false, true));
    EXPECT_EQ(draw(0), right);
    count = 1;
    EXPECT_EQ(draw(1), both);
    // Another revision plans the table again: the right entry now draws with the node's own binding.
    entries[1].argument_count = 0;
    schedule.write(0, leftVertices);
    EXPECT_TRUE(drewTriangles(draw(2), true, false));
    vernonRuntimeGraphicsPipelineDestroy(other);
}

// A compute node and a graphics node link as one schedule: the dispatch mirrors the vertices before the pass draws
// them, and every call mirrors them again.
TEST_P(HalGraphicsSchedule, ComputeNodesDispatchBeforeThePassDraws) {
    VernonProgramExecutable *pass = resolveFixture("recorded_graphics_mirrored_vertices_pass");
    ASSERT_NE(pass, nullptr);
    const Resource mirrored = vertexBuffer(kLeftTriangle);
    Linked schedule(runtime_, linkProgram(pass));
    ASSERT_TRUE(schedule) << lastError(runtime_);
    std::vector<size_t> arguments;
    size_t renderPassSlot = SIZE_MAX;
    size_t dynamicSlot = SIZE_MAX;
    for (size_t slot = 0; slot < schedule.slotCount(); ++slot) {
        if (schedule.slot(slot).kind == VERNON_SCHEDULE_SLOT_ARGUMENT)
            arguments.push_back(slot);
        else if (schedule.slot(slot).kind == VERNON_SCHEDULE_SLOT_PASS)
            renderPassSlot = slot;
        else if (schedule.slot(slot).kind == VERNON_SCHEDULE_SLOT_DYNAMIC)
            dynamicSlot = slot;
    }
    ASSERT_EQ(arguments.size(), 2u);
    ASSERT_NE(renderPassSlot, SIZE_MAX);
    ASSERT_NE(dynamicSlot, SIZE_MAX);

    Target target(device_);
    PassControls controls(target.reference());
    schedule.write(arguments[0], verticesArgument(findParameter(pass, "vertices"), mirrored.reference));
    schedule.write(arguments[1], hostArgument(findParameter(pass, "tint"), kRed));
    schedule.write(renderPassSlot, controls.pointer);
    schedule.write(dynamicSlot, controls.dynamic);
    ASSERT_EQ(schedule.call(), VERNON_STATUS_OK) << lastError(runtime_);
    const VernonScheduleGraphicsTelemetry telemetry = schedule.telemetry();
    EXPECT_EQ(telemetry.render_scopes, 1u);
    EXPECT_EQ(telemetry.draws, 1u);
    std::vector<uint8_t> pixels = target.pixels();
    EXPECT_TRUE(nearPixel(pixelAt(pixels, kLeftX, kMiddleY), {0, 0, 0, 255}));
    EXPECT_TRUE(nearPixel(pixelAt(pixels, kRightX, kMiddleY), {255, 0, 0, 255}));

    ASSERT_EQ(schedule.call(), VERNON_STATUS_OK) << lastError(runtime_);
    pixels = target.pixels();
    EXPECT_TRUE(nearPixel(pixelAt(pixels, kLeftX, kMiddleY), {255, 0, 0, 255}));
    EXPECT_TRUE(nearPixel(pixelAt(pixels, kRightX, kMiddleY), {0, 0, 0, 255}));
}

// An acquired surface image is a schedule's attachment like any other image; its contents are undefined until a call
// writes them, so a pass preserving them fails until one clears them, and again after each present.
TEST_P(HalGraphicsSchedule, PassesDrawIntoAcquiredSurfaceImages) {
    VernonHalSurfaceDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE)
    VernonOpenGLSurfaceCallbacks window{};
    std::unique_ptr<vernon::host::GlfwWindowOwner> native;
    if (context_) {
        window = context_->surfaceCallbacks();
        descriptor.native.opengl = &window;
    }
#if defined(_WIN32)
    if (GetParam().runtime == VERNON_RUNTIME_VULKAN) {
        native = std::make_unique<vernon::host::GlfwWindowOwner>(kSurfaceExtent, kSurfaceExtent);
        descriptor.native.win32_window = native->win32Window();
    }
#endif
#endif
    VernonHalSurface *created = nullptr;
    if (vernonHalCreateSurface(device_, &descriptor, &created) != VERNON_HAL_OK)
        GTEST_SKIP() << "no surface presents on " << GetParam().target << ": " << halError();
    const auto destroy = [](VernonHalSurface *owned) {
        EXPECT_EQ(vernonHalDestroySurface(owned), VERNON_HAL_OK) << halError();
    };
    const std::unique_ptr<VernonHalSurface, decltype(destroy)> owner(created, destroy);
    VernonHalSurface *surface = owner.get();
    VernonHalSurfaceCapabilities capabilities{};
    capabilities.struct_size = sizeof(capabilities);
    ASSERT_EQ(vernonHalSurfaceGetCapabilities(surface, &capabilities), VERNON_HAL_OK) << halError();
    const auto format =
        std::find(capabilities.formats, capabilities.formats + capabilities.format_count, VERNON_TEXTURE_RGBA8_UNORM);
    ASSERT_NE(format, capabilities.formats + capabilities.format_count);
    VernonHalSurfaceConfiguration configuration{};
    configuration.struct_size = sizeof(configuration);
    configuration.format = VERNON_TEXTURE_RGBA8_UNORM;
    configuration.usage = VERNON_IMAGE_COLOR_ATTACHMENT;
    configuration.width = capabilities.current_width == UINT32_MAX ? kSurfaceExtent : capabilities.current_width;
    configuration.height = capabilities.current_height == UINT32_MAX ? kSurfaceExtent : capabilities.current_height;
    configuration.present_mode = VERNON_HAL_PRESENT_FIFO;
    configuration.image_count = std::max(capabilities.min_image_count, 2u);
    if (capabilities.max_image_count)
        configuration.image_count = std::min(configuration.image_count, capabilities.max_image_count);
    ASSERT_EQ(vernonHalSurfaceConfigure(surface, &configuration), VERNON_HAL_OK) << halError();
    const uint32_t extent = std::min(configuration.width, configuration.height);

    Linked schedule(runtime_, linkProgram(executable_));
    ASSERT_TRUE(schedule) << lastError(runtime_);
    schedule.write(0, passVertices(left_));
    for (int frame = 0; frame < 3; ++frame) {
        VernonHalSurfaceImage image{};
        ASSERT_EQ(vernonHalSurfaceAcquire(surface, &image), VERNON_HAL_OK) << halError();
        VernonRuntimeProviderResourceReference view{};
        ASSERT_EQ(vernonHalReferenceImageView(image.view, &view), VERNON_HAL_OK) << halError();
        PassControls controls(view, extent);
        schedule.write(1, controls.pointer);
        schedule.write(3, controls.dynamic);
        controls.attachment.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE;
        EXPECT_NE(schedule.call(), VERNON_STATUS_OK);
        controls.attachment.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
        ASSERT_EQ(schedule.call(), VERNON_STATUS_OK) << lastError(runtime_);
        controls.attachment.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE;
        schedule.write(0, passVertices(right_));
        ASSERT_EQ(schedule.call(), VERNON_STATUS_OK) << lastError(runtime_);
        schedule.write(0, passVertices(left_));
        ASSERT_EQ(vernonHalSurfacePresent(surface), VERNON_HAL_OK) << halError();
    }
}

INSTANTIATE_TEST_SUITE_P(Devices, HalGraphicsSchedule, testing::ValuesIn(graphicsScheduleCases()),
                         [](const testing::TestParamInfo<ProgramFixtureManifest> &info) {
                             return std::string(info.param.target);
                         });

} // namespace
