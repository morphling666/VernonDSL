#include "runtime/resolved_execution_plan.h"
#include "runtime/runtime_state.h"
#include "schedule_test_support.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using vernon::tests::DeviceTensor;
using vernon::tests::HalImage;
using vernon::tests::LinkedProgram;
using vernon::tests::PassControls;
using vernon::tests::ProgramFixtureManifest;
using vernon::tests::ResolvedProgram;

const std::vector<float> kTriangle{-0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f};

std::string testName(const testing::TestParamInfo<ProgramFixtureManifest> &info) {
    return std::string(info.param.target);
}

DeviceTensor vertices(const vernon::tests::HalTestDevice &device) {
    return DeviceTensor(device, kTriangle.data(), kTriangle.size() * sizeof(float), {3, 2}, sizeof(float), {}, 0,
                        VERNON_HAL_BUFFER_VERTEX | VERNON_HAL_BUFFER_STORAGE);
}

// The module_graphics pass draws (255, 64, 0) over the opaque black clear; nothing else may appear.
void expectTriangleCoverage(const std::vector<uint8_t> &pixels, size_t expectedRendered) {
    size_t rendered = 0;
    size_t cleared = 0;
    size_t unexpected = 0;
    for (size_t pixel = 0; pixel < pixels.size(); pixel += 4) {
        if (pixels[pixel] == 255 && pixels[pixel + 1] == 64 && pixels[pixel + 2] == 0 && pixels[pixel + 3] == 255)
            ++rendered;
        else if (pixels[pixel] == 0 && pixels[pixel + 1] == 0 && pixels[pixel + 2] == 0 && pixels[pixel + 3] == 255)
            ++cleared;
        else
            ++unexpected;
    }
    EXPECT_EQ(rendered, expectedRendered);
    EXPECT_EQ(cleared, pixels.size() / 4 - expectedRendered);
    EXPECT_EQ(unexpected, 0u);
}

void expectColorAttachmentOutput(VernonProgramExecutable *executable) {
    for (size_t index = 0; index < vernonRuntimeProgramExecutableGetParameterCount(executable); ++index) {
        VernonProgramParameterView parameter{};
        ASSERT_EQ(vernonRuntimeProgramExecutableGetParameterByIndex(executable, index, &parameter), VERNON_STATUS_OK);
        if (std::string_view(parameter.name.data, parameter.name.size) != "output")
            continue;
        VernonProgramImageConstraintView constraint{};
        constraint.struct_size = sizeof(constraint);
        ASSERT_EQ(vernonRuntimeProgramExecutableGetImageConstraintByParameterIndex(executable, index, &constraint),
                  VERNON_STATUS_OK);
        EXPECT_EQ(constraint.binding_role, VERNON_IMAGE_BINDING_COLOR_ATTACHMENT);
        EXPECT_EQ(constraint.sample_result_class, VERNON_IMAGE_SAMPLE_NONE);
        return;
    }
    ADD_FAILURE() << "the Program has no output parameter";
}

class ModuleGraphicsProgram : public vernon::tests::HalProgramTest {
protected:
    vernon::tests::BackendTestRequirements requirements() const override {
        return vernon::tests::graphicsImageCopyFixtureRequirements(GetParam().runtime);
    }
};

TEST_P(ModuleGraphicsProgram, ExternalVerticesGraphicsModuleRendersPixels) {
    const ResolvedProgram program(runtime(), GetParam().manifestPath);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(runtime());
    ASSERT_EQ(vernonRuntimeProgramExecutableGetGraphicsNodeCount(program.get()), 1u);
    expectColorAttachmentOutput(program.get());

    constexpr uint32_t extent = 32;
    HalImage target(device_.get(), VERNON_TEXTURE_RGBA8_UNORM, extent, extent, VERNON_IMAGE_COLOR_ATTACHMENT);
    ASSERT_TRUE(target);
    const DeviceTensor triangle = vertices(device_);
    LinkedProgram linked(runtime(), program.get());
    ASSERT_TRUE(linked) << linked.error();
    linked.record().bind("vertices", triangle);
    ASSERT_TRUE(linked.record().unbound().empty());
    const PassControls controls(target.reference(), extent);
    controls.writeTo(linked.record());
    ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();
    expectTriangleCoverage(target.rgba8(), 338);
}

TEST_P(ModuleGraphicsProgram, ReusesOneScheduleAcrossExtentsAndDynamicStates) {
    const ResolvedProgram program(runtime(), GetParam().manifestPath);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(runtime());
    const DeviceTensor triangle = vertices(device_);
    LinkedProgram linked(runtime(), program.get());
    ASSERT_TRUE(linked) << linked.error();
    linked.record().bind("vertices", triangle);

    constexpr std::array<std::array<uint32_t, 2>, 2> extents{{{24, 24}, {40, 20}}};
    constexpr std::array<uint32_t, 2> viewportWidths{24, 20};
    for (size_t iteration = 0; iteration < extents.size(); ++iteration) {
        const uint32_t width = extents[iteration][0];
        const uint32_t height = extents[iteration][1];
        HalImage target(device_.get(), VERNON_TEXTURE_RGBA8_UNORM, width, height, VERNON_IMAGE_COLOR_ATTACHMENT);
        ASSERT_TRUE(target);
        PassControls controls(target.reference(), width);
        controls.renderPass.render_area[3] = height;
        controls.dynamic.viewport[2] = static_cast<float>(viewportWidths[iteration]);
        controls.dynamic.viewport[3] = static_cast<float>(height);
        controls.dynamic.scissor[3] = height;
        controls.dynamic.stencil_reference = static_cast<uint32_t>(iteration + 3);
        controls.writeTo(linked.record());
        ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();

        const std::vector<uint8_t> pixels = target.rgba8();
        const size_t colored = (height / 2 * width + viewportWidths[iteration] / 2) * 4;
        EXPECT_GT(pixels[colored], 240);
        EXPECT_GT(pixels[colored + 1], 40);
        if (iteration == 1) {
            const size_t outsideViewport = (height / 2 * width + 3 * width / 4) * 4;
            EXPECT_EQ(pixels[outsideViewport], 0);
            EXPECT_EQ(pixels[outsideViewport + 1], 0);
            EXPECT_EQ(pixels[outsideViewport + 2], 0);
        }
    }
}

TEST_P(ModuleGraphicsProgram, ProgramGraphFusesCookedGraphicsProgramsOnSharedFramebuffer) {
    const ResolvedProgram program(runtime(), GetParam().manifestPath);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(runtime());
    VernonProgramGraph *graph = vernonRuntimeProgramGraphCreate(runtime());
    ASSERT_NE(graph, nullptr);
    VernonProgramNodeId scene{};
    VernonProgramNodeId overlay{};
    ASSERT_EQ(vernonRuntimeProgramGraphAddProgram(graph, program.bundle(), nullptr, {}, &scene), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramGraphAddProgram(graph, program.bundle(), nullptr, {}, &overlay), VERNON_STATUS_OK);
    const auto boundary = [&](VernonProgramNodeId node, VernonProgramBoundaryRole role, const char *name) {
        VernonProgramNodeBindingToken token{sizeof(VernonProgramNodeBindingToken)};
        EXPECT_EQ(vernonRuntimeProgramGraphFindBoundary(graph, node, role, {name, std::strlen(name)}, &token),
                  VERNON_STATUS_OK);
        return token;
    };
    const VernonProgramNodeBindingToken sceneOutput = boundary(scene, VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
    const VernonProgramNodeBindingToken overlayOutput = boundary(overlay, VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
    const VernonProgramNodeBindingToken sceneVertices = boundary(scene, VERNON_PROGRAM_BOUNDARY_INPUT, "vertices");
    const VernonProgramNodeBindingToken overlayVertices = boundary(overlay, VERNON_PROGRAM_BOUNDARY_INPUT, "vertices");
    VernonProgramGraphStorage framebuffer{sizeof(VernonProgramGraphStorage)};
    ASSERT_EQ(vernonRuntimeProgramGraphCreateStorage(graph, &sceneOutput, &framebuffer), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramGraphAppendStorage(graph, &framebuffer, &overlayOutput), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramGraphExportStorage(graph, &framebuffer, {"output", 6}), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramGraphExportBoundary(graph, &sceneVertices, {"scene_vertices", 14}), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramGraphExportBoundary(graph, &overlayVertices, {"overlay_vertices", 16}),
              VERNON_STATUS_OK);
    const std::unique_ptr<VernonProgramExecutable, decltype(&vernonRuntimeProgramExecutableDestroy)> executable(
        vernonRuntimeResolveProgramGraph(graph), vernonRuntimeProgramExecutableDestroy);
    vernonRuntimeProgramGraphDestroy(graph);
    ASSERT_NE(executable, nullptr) << vernon::tests::runtimeError(runtime());

    constexpr uint32_t extent = 8;
    HalImage target(device_.get(), VERNON_TEXTURE_RGBA8_UNORM, extent, extent, VERNON_IMAGE_COLOR_ATTACHMENT);
    ASSERT_TRUE(target);
    const DeviceTensor triangle = vertices(device_);
    LinkedProgram linked(runtime(), executable.get());
    ASSERT_TRUE(linked) << linked.error();
    linked.record().bind("scene_vertices", triangle);
    linked.record().bind("overlay_vertices", triangle);
    ASSERT_TRUE(linked.record().unbound().empty());
    // The overlay pass shares the scene's render scope while it preserves the framebuffer, and begins its own when it
    // clears it.
    for (const bool preserve : {true, false}) {
        SCOPED_TRACE(preserve);
        PassControls sceneControls(target.reference(), extent);
        PassControls overlayControls(target.reference(), extent);
        if (preserve)
            overlayControls.attachment.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE;
        sceneControls.writeTo(linked.record(), 0);
        overlayControls.writeTo(linked.record(), 1);
        ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();
        const VernonScheduleGraphicsTelemetry telemetry = linked.telemetry();
        EXPECT_EQ(telemetry.render_scopes, preserve ? 1u : 2u);
        EXPECT_EQ(telemetry.draws, 2u);
    }
}

INSTANTIATE_TEST_SUITE_P(EnabledTargets, ModuleGraphicsProgram,
                         testing::ValuesIn(vernon::tests::programFixtureCases("module_graphics")), testName);

class ModuleMixedProgram : public vernon::tests::HalProgramTest {
protected:
    vernon::tests::BackendTestRequirements requirements() const override {
        return vernon::tests::computeGraphicsFixtureRequirements(GetParam().runtime);
    }
};

TEST_P(ModuleMixedProgram, ComputeGeneratedVerticesReachGraphicsModule) {
    const ResolvedProgram program(runtime(), GetParam().manifestPath);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(runtime());
    ASSERT_EQ(vernonRuntimeProgramExecutableGetGraphicsNodeCount(program.get()), 1u);
    expectColorAttachmentOutput(program.get());

    constexpr uint32_t extent = 32;
    HalImage target(device_.get(), VERNON_TEXTURE_RGBA8_UNORM, extent, extent, VERNON_IMAGE_COLOR_ATTACHMENT);
    ASSERT_TRUE(target);
    LinkedProgram linked(runtime(), program.get());
    ASSERT_TRUE(linked) << linked.error();
    linked.record().bindScalar<uint32_t>("grid_x", 3);
    linked.record().bindScalar<uint32_t>("grid_y", 1);
    linked.record().bindScalar<uint32_t>("grid_z", 1);
    ASSERT_TRUE(linked.record().unbound().empty());
    const PassControls controls(target.reference(), extent);
    controls.writeTo(linked.record());
    ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();
    expectTriangleCoverage(target.rgba8(), 192);
}

INSTANTIATE_TEST_SUITE_P(EnabledTargets, ModuleMixedProgram,
                         testing::ValuesIn(vernon::tests::programFixtureCases("module_mixed")), testName);

} // namespace
