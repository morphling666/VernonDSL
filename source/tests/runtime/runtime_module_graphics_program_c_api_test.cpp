#include "VernonRuntime.h"
#include "runtime_rhi_test_utils.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace {

using vernon::tests::RhiImage;
using vernon::tests::RhiImageView;
using vernon::tests::RhiRuntime;

std::string lastError(VernonRuntimeContext *context) {
    const VernonStringView error = vernonRuntimeGetLastError(context);
    return std::string(error.data ? error.data : "", error.size);
}

std::string_view name(const VernonProgramParameterView &parameter) {
    return {parameter.name.data, parameter.name.size};
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

VernonProgramArgument tensorArgument(const VernonProgramParameterView &parameter,
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
    argument.tensor.rank = std::size(shape);
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = 3 * 2 * sizeof(float);
    return argument;
}

VernonProgramArgument scalarArgument(const VernonProgramParameterView &parameter, const uint32_t *value) {
    VernonProgramArgument argument{};
    argument.slot = parameter.slot;
    argument.kind = VERNON_PROGRAM_TENSOR;
    argument.tensor.struct_size = sizeof(argument.tensor);
    argument.tensor.storage = VERNON_TENSOR_HOST;
    argument.tensor.host_data = value;
    argument.tensor.element_layout = parameter.element_layout;
    argument.tensor.access = parameter.access;
    argument.tensor.byte_size = sizeof(*value);
    return argument;
}

void destroyTarget(RhiRuntime &runtime, RhiImageView &view, RhiImage &image) {
    if (view.handle.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        EXPECT_EQ(vernonRhiDeviceDestroyImageView(runtime.device, view.handle), VERNON_RHI_STATUS_OK);
    if (image.handle.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        EXPECT_EQ(vernonRhiDeviceDestroyImage(runtime.device, image.handle), VERNON_RHI_STATUS_OK);
}

void invokeAndExpectTriangle(VernonRuntimeBackend backend, const std::filesystem::path &manifestPath,
                             bool externalVertices) {
    RhiRuntime runtime = vernon::tests::createRhiRuntime(backend);
    if (!runtime.runtime) {
        vernon::tests::destroyRhiRuntime(runtime);
        GTEST_SKIP() << "GPU backend is unavailable";
    }

    VernonProgramBundle *bundle = loadBundle(runtime, manifestPath);
    ASSERT_NE(bundle, nullptr) << lastError(runtime.runtime);
    VernonProgramExecutable *executable = vernonRuntimeResolveProgram(bundle, {nullptr, 0});
    ASSERT_NE(executable, nullptr) << lastError(runtime.runtime);
    ASSERT_EQ(vernonRuntimeProgramExecutableGetGraphicsNodeCount(executable), 1u);

    constexpr uint32_t extent = 32;
    RhiImage target = vernon::tests::createImage(runtime, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM, extent,
                                                 extent, 1, VERNON_RHI_IMAGE_COLOR_ATTACHMENT);
    ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    RhiImageView targetView =
        vernon::tests::createImageView(runtime, target, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);
    ASSERT_NE(targetView.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    constexpr std::array<float, 6> positions{-0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f};
    vernon::tests::RhiBuffer vertices;
    if (externalVertices) {
        vertices = vernon::tests::createBuffer(runtime, sizeof(positions), alignof(float), VERNON_RHI_BUFFER_VERTEX,
                                               positions.data());
        ASSERT_NE(vertices.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    }

    constexpr std::array<uint32_t, 3> grid{3, 1, 1};
    std::vector<VernonProgramArgument> arguments;
    const size_t parameterCount = vernonRuntimeProgramExecutableGetParameterCount(executable);
    arguments.reserve(parameterCount);
    for (size_t index = 0; index < parameterCount; ++index) {
        VernonProgramParameterView parameter{};
        ASSERT_EQ(vernonRuntimeProgramExecutableGetParameterByIndex(executable, index, &parameter), VERNON_STATUS_OK);
        if (name(parameter) == "vertices") {
            ASSERT_TRUE(externalVertices);
            arguments.push_back(tensorArgument(parameter, vertices.reference));
            continue;
        }
        if (name(parameter) == "output") {
            VernonProgramArgument output{};
            output.slot = parameter.slot;
            output.kind = VERNON_PROGRAM_IMAGE;
            output.image = {targetView.reference};
            arguments.push_back(output);
            continue;
        }
        bool matchedGrid = false;
        for (size_t axis = 0; axis < grid.size(); ++axis) {
            const std::string expected = "grid_" + std::string(1, "xyz"[axis]);
            if (name(parameter) == expected) {
                arguments.push_back(scalarArgument(parameter, &grid[axis]));
                matchedGrid = true;
                break;
            }
        }
        ASSERT_TRUE(matchedGrid) << "unbound reflected boundary " << name(parameter);
    }
    if (externalVertices)
        ASSERT_EQ(arguments.size(), 2u);
    else
        ASSERT_EQ(arguments.size(), 4u);

    VernonColorAttachment attachment{};
    attachment.location = 0;
    attachment.view = targetView.reference;
    attachment.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    attachment.store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
    attachment.clear_color[3] = 1.0f;
    vernon::tests::CanonicalGraphicsControls graphics(&attachment, 1, 3);
    graphics.renderPass.render_area[2] = extent;
    graphics.renderPass.render_area[3] = extent;
    graphics.dynamic.viewport[2] = extent;
    graphics.dynamic.viewport[3] = extent;
    graphics.dynamic.scissor[2] = extent;
    graphics.dynamic.scissor[3] = extent;

    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(executable);
    ASSERT_NE(instance, nullptr);
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    for (size_t index = 0; index < arguments.size(); ++index) {
        const std::string tokenText = "module-argument-" + std::to_string(index);
        const VernonProgramBindingToken token{sizeof(token), tokenText.data(), tokenText.size()};
        ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &token, &arguments[index], nullptr, 0, 0),
                  VERNON_STATUS_OK);
    }
    VernonProgramGraphicsControlsView controlSlots{};
    ASSERT_EQ(vernonRuntimeProgramExecutableGetGraphicsControlsByIndex(executable, 0, &controlSlots), VERNON_STATUS_OK);
    ASSERT_EQ(graphics.bind(invocation, controlSlots), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationForward(invocation, nullptr), VERNON_STATUS_OK)
        << lastError(runtime.runtime);
    vernonRuntimeProgramInvocationDestroy(invocation);
    vernonRuntimeProgramInstanceDestroy(instance);

    std::vector<uint8_t> pixels(extent * extent * 4);
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.width = extent;
    download.height = extent;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(runtime.device, target.handle, &download, pixels.data(), pixels.size()),
              VERNON_RHI_STATUS_OK);
    const size_t center = (extent / 2 * extent + extent / 2) * 4;
    const size_t corner = 4;
    EXPECT_GT(pixels[center], 240);
    EXPECT_GT(pixels[center + 1], 40);
    EXPECT_LT(pixels[center + 2], 10);
    EXPECT_EQ(pixels[corner], 0);
    EXPECT_EQ(pixels[corner + 1], 0);
    EXPECT_EQ(pixels[corner + 2], 0);

    if (vertices.handle.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        EXPECT_EQ(vernonRhiDeviceDestroyBuffer(runtime.device, vertices.handle), VERNON_RHI_STATUS_OK);
    destroyTarget(runtime, targetView, target);
    vernonRuntimeProgramExecutableDestroy(executable);
    vernonRuntimeProgramBundleDestroy(bundle);
    vernon::tests::destroyRhiRuntime(runtime);
}

void reuseGraphicsProgramAcrossExtentsAndDynamicStates(VernonRuntimeBackend backend,
                                                       const std::filesystem::path &manifestPath) {
    RhiRuntime runtime = vernon::tests::createRhiRuntime(backend);
    if (!runtime.runtime) {
        vernon::tests::destroyRhiRuntime(runtime);
        GTEST_SKIP() << "GPU backend is unavailable";
    }
    VernonProgramBundle *bundle = loadBundle(runtime, manifestPath);
    ASSERT_NE(bundle, nullptr) << lastError(runtime.runtime);
    VernonProgramExecutable *executable = vernonRuntimeResolveProgram(bundle, {nullptr, 0});
    ASSERT_NE(executable, nullptr) << lastError(runtime.runtime);
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(executable);
    ASSERT_NE(instance, nullptr);

    constexpr std::array<float, 6> positions{-0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f};
    vernon::tests::RhiBuffer vertices = vernon::tests::createBuffer(runtime, sizeof(positions), alignof(float),
                                                                    VERNON_RHI_BUFFER_VERTEX, positions.data());
    ASSERT_NE(vertices.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonProgramParameterView verticesParameter{};
    VernonProgramParameterView outputParameter{};
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(executable, {"vertices", 8}, &verticesParameter),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(executable, {"output", 6}, &outputParameter),
              VERNON_STATUS_OK);
    const VernonProgramArgument verticesArgument = tensorArgument(verticesParameter, vertices.reference);
    VernonProgramGraphicsControlsView slots{};
    ASSERT_EQ(vernonRuntimeProgramExecutableGetGraphicsControlsByIndex(executable, 0, &slots), VERNON_STATUS_OK);

    constexpr std::array<std::array<uint32_t, 2>, 2> extents{{{24, 24}, {40, 20}}};
    constexpr std::array<uint32_t, 2> viewportWidths{24, 20};
    for (size_t iteration = 0; iteration < extents.size(); ++iteration) {
        const uint32_t width = extents[iteration][0];
        const uint32_t height = extents[iteration][1];
        RhiImage target = vernon::tests::createImage(runtime, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM, width,
                                                     height, 1, VERNON_RHI_IMAGE_COLOR_ATTACHMENT);
        ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
        RhiImageView view =
            vernon::tests::createImageView(runtime, target, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);
        ASSERT_NE(view.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
        VernonProgramArgument output{};
        output.slot = outputParameter.slot;
        output.kind = VERNON_PROGRAM_IMAGE;
        output.image = {view.reference};
        VernonColorAttachment attachment{};
        attachment.location = 0;
        attachment.view = view.reference;
        attachment.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
        attachment.store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
        attachment.clear_color[3] = 1.0f;
        vernon::tests::CanonicalGraphicsControls graphics(&attachment, 1, 3);
        graphics.renderPass.render_area[2] = width;
        graphics.renderPass.render_area[3] = height;
        graphics.dynamic.viewport[2] = viewportWidths[iteration];
        graphics.dynamic.viewport[3] = height;
        graphics.dynamic.scissor[2] = width;
        graphics.dynamic.scissor[3] = height;
        graphics.dynamic.stencil_reference = static_cast<uint32_t>(iteration + 3);

        VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
        ASSERT_NE(invocation, nullptr);
        const VernonProgramBindingToken verticesToken{sizeof(verticesToken), "reuse-vertices", 14};
        const std::string outputText = "reuse-output-" + std::to_string(iteration);
        const VernonProgramBindingToken outputToken{sizeof(outputToken), outputText.data(), outputText.size()};
        ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &verticesToken, &verticesArgument, nullptr, 0, 0),
                  VERNON_STATUS_OK);
        ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &outputToken, &output, nullptr, 0, 0),
                  VERNON_STATUS_OK);
        const std::string renderText = "reuse-render-" + std::to_string(iteration);
        const std::string dynamicText = "reuse-dynamic-" + std::to_string(iteration);
        const VernonProgramBindingToken renderToken{sizeof(renderToken), renderText.data(), renderText.size()};
        const VernonProgramBindingToken drawToken{sizeof(drawToken), "reuse-draw", 10};
        const VernonProgramBindingToken dynamicToken{sizeof(dynamicToken), dynamicText.data(), dynamicText.size()};
        ASSERT_EQ(vernonRuntimeProgramInvocationBindRenderPass(invocation, slots.render_pass_control, &renderToken,
                                                               &graphics.renderPass, nullptr, 0),
                  VERNON_STATUS_OK);
        ASSERT_EQ(vernonRuntimeProgramInvocationBindDrawCommand(invocation, slots.draw_command_control, &drawToken,
                                                                &graphics.draw, nullptr),
                  VERNON_STATUS_OK);
        ASSERT_EQ(vernonRuntimeProgramInvocationBindDynamicState(invocation, slots.dynamic_state_control, &dynamicToken,
                                                                 &graphics.dynamic),
                  VERNON_STATUS_OK);
        ASSERT_EQ(vernonRuntimeProgramInvocationForward(invocation, nullptr), VERNON_STATUS_OK)
            << lastError(runtime.runtime);
        vernonRuntimeProgramInvocationDestroy(invocation);

        std::vector<uint8_t> pixels(width * height * 4);
        VernonRhiImageDownloadDescriptor download{};
        download.struct_size = sizeof(download);
        download.width = width;
        download.height = height;
        download.depth = 1;
        download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
        download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
        ASSERT_EQ(vernonRhiDeviceDownloadImage(runtime.device, target.handle, &download, pixels.data(), pixels.size()),
                  VERNON_RHI_STATUS_OK);
        const size_t colored = (height / 2 * width + viewportWidths[iteration] / 2) * 4;
        EXPECT_GT(pixels[colored], 240);
        EXPECT_GT(pixels[colored + 1], 40);
        if (iteration == 1) {
            const size_t outsideViewport = (height / 2 * width + 3 * width / 4) * 4;
            EXPECT_EQ(pixels[outsideViewport], 0);
            EXPECT_EQ(pixels[outsideViewport + 1], 0);
            EXPECT_EQ(pixels[outsideViewport + 2], 0);
        }
        destroyTarget(runtime, view, target);
    }

    VernonProgramBindingTelemetry telemetry{};
    telemetry.struct_size = sizeof(telemetry);
    ASSERT_EQ(vernonRuntimeProgramInstanceGetTelemetry(instance, &telemetry), VERNON_STATUS_OK);
    EXPECT_EQ(telemetry.prepare_count, 3u);
    EXPECT_EQ(telemetry.reuse_count, 1u);
    vernonRuntimeProgramInstanceDestroy(instance);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(runtime.device, vertices.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeProgramExecutableDestroy(executable);
    vernonRuntimeProgramBundleDestroy(bundle);
    vernon::tests::destroyRhiRuntime(runtime);
}

} // namespace

#if defined(VERNON_MODULE_GRAPHICS_METAL_MANIFEST)
TEST(RuntimeModuleGraphicsProgramCApi, MetalReusesLoadedGraphicsModuleAcrossExtentsAndDynamicStates) {
    reuseGraphicsProgramAcrossExtentsAndDynamicStates(VERNON_RUNTIME_METAL, VERNON_MODULE_GRAPHICS_METAL_MANIFEST);
}

TEST(RuntimeModuleGraphicsProgramCApi, MetalComputeGeneratedVerticesReachGraphicsModule) {
    invokeAndExpectTriangle(VERNON_RUNTIME_METAL, VERNON_MODULE_MIXED_METAL_MANIFEST, false);
}
#endif

#if defined(VERNON_MODULE_GRAPHICS_VULKAN_MANIFEST)
TEST(RuntimeModuleGraphicsProgramCApi, VulkanExternalVerticesGraphicsModuleRendersPixels) {
    invokeAndExpectTriangle(VERNON_RUNTIME_VULKAN, VERNON_MODULE_GRAPHICS_VULKAN_MANIFEST, true);
}

TEST(RuntimeModuleGraphicsProgramCApi, VulkanComputeGeneratedVerticesReachGraphicsModule) {
    invokeAndExpectTriangle(VERNON_RUNTIME_VULKAN, VERNON_MODULE_MIXED_VULKAN_MANIFEST, false);
}
#endif

#if defined(VERNON_MODULE_GRAPHICS_DIRECTX_MANIFEST)
TEST(RuntimeModuleGraphicsProgramCApi, DirectX12ExternalVerticesGraphicsModuleRendersPixels) {
    invokeAndExpectTriangle(VERNON_RUNTIME_DIRECTX12, VERNON_MODULE_GRAPHICS_DIRECTX_MANIFEST, true);
}

TEST(RuntimeModuleGraphicsProgramCApi, DirectX12ComputeGeneratedVerticesReachGraphicsModule) {
    invokeAndExpectTriangle(VERNON_RUNTIME_DIRECTX12, VERNON_MODULE_MIXED_DIRECTX_MANIFEST, false);
}
#endif

TEST(RuntimeModuleGraphicsProgramCApi, OpenGLExternalVerticesGraphicsModuleRendersPixels) {
    invokeAndExpectTriangle(VERNON_RUNTIME_OPENGL, VERNON_MODULE_GRAPHICS_OPENGL_MANIFEST, true);
}

TEST(RuntimeModuleGraphicsProgramCApi, OpenGLComputeGeneratedVerticesReachGraphicsModule) {
    invokeAndExpectTriangle(VERNON_RUNTIME_OPENGL, VERNON_MODULE_MIXED_OPENGL_MANIFEST, false);
}
