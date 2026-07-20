#include "VernonCompiler.h"
#include "VernonRuntime.h"

#include <cassert>
#include <cstddef>
#include <cstring>

namespace {

constexpr char module[] = R"(
module {
  func.func @increment(
      %values: !vernon.buffer<f32, "read_write"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      },
      %id: index {
        vernon.interface = "input",
        vernon.builtin = "global_invocation_id"
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 8, 1, 1>
      } {
    %value = "vernon.intrinsic"(%values, %id) {
      name = "buffer_load"
    } : (!vernon.buffer<f32, "read_write">, index) -> f32
    %one = arith.constant 1.0 : f32
    %sum = arith.addf %value, %one : f32
    "vernon.intrinsic"(%values, %id, %sum) {
      name = "buffer_store"
    } : (!vernon.buffer<f32, "read_write">, index, f32) -> ()
    return
  }
}
)";

void runBackend(VernonCompilerContext *compiler, VernonRuntimeBackend backend) {
  if (!vernonRuntimeGetCapabilities(backend).available)
    return;

  // OpenGL ES runtime execution intentionally uses the desktop OpenGL
  // compatibility shader path when no explicit EGL provider is configured.
  VernonCompileResult *compiled = vernonCompilerCompileMlir(
      compiler, module, std::strlen(module), VERNON_TARGET_OPENGL);
  assert(compiled);
  assert(vernonCompileResultGetStatus(compiled) == VERNON_STATUS_OK);

  VernonRuntimeContext *runtime = vernonRuntimeCreate(backend, 0);
  assert(runtime);
  VernonStringView artifact = vernonCompileResultGetArtifactData(compiled, 0);
  VernonStringView reflection = vernonCompileResultGetReflection(compiled);
  VernonLoadedKernel *kernel = vernonRuntimeLoadArtifact(
      runtime, artifact.data, artifact.size, reflection.data, reflection.size,
      "increment", std::strlen("increment"));
  assert(kernel);

  float input[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  float output[8] = {};
  VernonDeviceBuffer *buffer =
      vernonRuntimeBufferAllocate(runtime, sizeof(input), alignof(float));
  assert(buffer);
  assert(vernonRuntimeCopyFromHost(buffer, 0, input, sizeof(input)) ==
         VERNON_STATUS_OK);
  VernonLaunchArgument argument{VERNON_LAUNCH_TENSOR, buffer, nullptr, 0};
  assert(vernonRuntimeLaunch(kernel, {8, 1, 1}, &argument, 1) ==
         VERNON_STATUS_OK);
  assert(vernonRuntimeCopyToHost(buffer, 0, output, sizeof(output)) ==
         VERNON_STATUS_OK);
  for (int index = 0; index < 8; ++index)
    assert(output[index] == input[index] + 1.0f);

  assert(vernonRuntimeBufferFree(buffer) == VERNON_STATUS_OK);
  assert(vernonRuntimeKernelUnload(kernel) == VERNON_STATUS_OK);
  assert(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
  vernonCompileResultDestroy(compiled);
}

void runGraphics() {
  if (!vernonRuntimeGetCapabilities(VERNON_RUNTIME_OPENGL).available)
    return;
  VernonRuntimeCreateOptions options{};
  options.struct_size = sizeof(options);
  options.api_version_major = 3;
  options.api_version_minor = 3;
  VernonRuntimeContext *runtime =
      vernonRuntimeCreateWithOptions(VERNON_RUNTIME_OPENGL, &options);
  assert(runtime);
  VernonRuntimeCapabilities capabilities =
      vernonRuntimeGetContextCapabilities(runtime);
  assert(capabilities.available && capabilities.supports_graphics);
  assert(capabilities.graphics_draw_abi_version == 2);
  assert(capabilities.api_version_major > 3 ||
         (capabilities.api_version_major == 3 &&
          capabilities.api_version_minor >= 3));

  constexpr char vertex[] = R"(#version 330
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 offset;
void main() { gl_Position = vec4(position + offset, 0.0, 1.0); }
)";
  constexpr char fragment[] = R"(#version 330
layout(location = 0) out vec4 color;
layout(location = 1) out vec4 object_id;
void main() {
  color = vec4(1.0, 0.25, 0.0, 1.0);
  object_id = vec4(0.0, 1.0, 0.25, 1.0);
}
)";
  VernonLoadedProgram *program = vernonRuntimeProgramLoadGraphics(
      runtime, {vertex, sizeof(vertex) - 1}, {fragment, sizeof(fragment) - 1});
  assert(program);
  constexpr float positions[] = {
      -0.75f, -0.75f, 0.75f, -0.75f, 0.0f, 0.75f,
  };
  VernonDeviceBuffer *vertices =
      vernonRuntimeBufferAllocate(runtime, sizeof(positions), alignof(float));
  assert(vertices);
  assert(vernonRuntimeCopyFromHost(vertices, 0, positions, sizeof(positions)) ==
         VERNON_STATUS_OK);
  VernonDeviceTexture *target =
      vernonRuntimeTextureCreate2D(runtime, 64, 64, VERNON_TEXTURE_RGBA8_UNORM);
  assert(target);
  VernonDrawBinding binding{0, vertices, 2, 2 * sizeof(float), 0, 0};
  VernonDrawDescription draw{};
  draw.struct_size = offsetof(VernonDrawDescription, index_binding);
  draw.target = target;
  draw.bindings = &binding;
  draw.binding_count = 1;
  draw.vertex_count = 3;
  draw.instance_count = 1;
  assert(vernonRuntimeDraw(program, &draw) == VERNON_STATUS_OK);
  assert(vernonRuntimeSynchronize(runtime) == VERNON_STATUS_OK);
  unsigned char pixels[64 * 64 * 4]{};
  assert(vernonRuntimeTextureCopyToHost(target, pixels, sizeof(pixels)) ==
         VERNON_STATUS_OK);
  const size_t center = (32 * 64 + 32) * 4;
  assert(pixels[0] == 0 && pixels[1] == 0 && pixels[2] == 0 && pixels[3] == 0);
  assert(pixels[center] > 240 && pixels[center + 1] > 40 &&
         pixels[center + 2] == 0 && pixels[center + 3] > 240);

  constexpr uint32_t indexValues[] = {0, 1, 2};
  VernonDeviceBuffer *indices = vernonRuntimeBufferAllocate(
      runtime, sizeof(indexValues), alignof(uint32_t));
  assert(indices);
  assert(vernonRuntimeCopyFromHost(indices, 0, indexValues,
                                   sizeof(indexValues)) == VERNON_STATUS_OK);
  constexpr float offsetValue[] = {0.0f, 0.0f};
  VernonDeviceBuffer *offset =
      vernonRuntimeBufferAllocate(runtime, sizeof(offsetValue), alignof(float));
  assert(offset);
  assert(vernonRuntimeCopyFromHost(offset, 0, offsetValue,
                                   sizeof(offsetValue)) == VERNON_STATUS_OK);
  VernonDeviceTexture *objectId =
      vernonRuntimeTextureCreate2D(runtime, 64, 64, VERNON_TEXTURE_RGBA8_UNORM);
  assert(objectId);
  VernonDrawBinding advancedBindings[] = {
      binding,
      {1, offset, 2, 2 * sizeof(float), 0, 1},
  };
  VernonIndexBinding indexBinding{indices, VERNON_INDEX_U32, 0, 3};
  VernonColorAttachment attachments[] = {
      {1, objectId},
      {0, target},
  };
  draw.struct_size = sizeof(draw);
  draw.bindings = advancedBindings;
  draw.binding_count = 2;
  draw.index_binding = &indexBinding;
  draw.color_attachments = attachments;
  draw.color_attachment_count = 2;
  draw.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
  assert(vernonRuntimeDraw(program, &draw) == VERNON_STATUS_OK);
  assert(vernonRuntimeSynchronize(runtime) == VERNON_STATUS_OK);
  assert(vernonRuntimeTextureCopyToHost(objectId, pixels, sizeof(pixels)) ==
         VERNON_STATUS_OK);
  assert(pixels[center] == 0 && pixels[center + 1] > 240 &&
         pixels[center + 2] > 40 && pixels[center + 3] > 240);
  draw.struct_size = offsetof(VernonDrawDescription, index_binding);
  draw.target = target;
  draw.bindings = &binding;
  draw.binding_count = 1;
  draw.vertex_count = 3;
  draw.instance_count = 1;
  assert(vernonRuntimeDraw(program, &draw) == VERNON_STATUS_OK);

  assert(vernonRuntimeTextureFree(objectId) == VERNON_STATUS_OK);
  assert(vernonRuntimeBufferFree(offset) == VERNON_STATUS_OK);
  assert(vernonRuntimeBufferFree(indices) == VERNON_STATUS_OK);
  assert(vernonRuntimeTextureFree(target) == VERNON_STATUS_OK);
  assert(vernonRuntimeBufferFree(vertices) == VERNON_STATUS_OK);
  assert(vernonRuntimeProgramUnload(program) == VERNON_STATUS_OK);
  assert(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
}

} // namespace

int main() {
  VernonCompilerContext *compiler = vernonCompilerCreate();
  assert(compiler);
  runBackend(compiler, VERNON_RUNTIME_OPENGL);
  runBackend(compiler, VERNON_RUNTIME_OPENGL_ES);
  runGraphics();
  vernonCompilerDestroy(compiler);
  return 0;
}
