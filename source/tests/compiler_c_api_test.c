#include "VernonCompiler.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static int view_contains(VernonStringView value, const char *needle) {
  const size_t needle_size = strlen(needle);
  if (needle_size == 0 || needle_size > value.size)
    return 0;
  for (size_t index = 0; index + needle_size <= value.size; ++index) {
    if (memcmp(value.data + index, needle, needle_size) == 0)
      return 1;
  }
  return 0;
}

static void sample_texture(void *user_data, uintptr_t texture, float u, float v,
                           float out_rgba[4]) {
  const float bias = *(const float *)user_data;
  out_rgba[0] = u;
  out_rgba[1] = v;
  out_rgba[2] = (float)texture;
  out_rgba[3] = bias;
}

int main(void) {
  static const char module[] =
      "module {\n"
      "  func.func @vertex_main("
      "%position: vector<3xf32> {vernon.interface = \"input\", "
      "vernon.location = 0 : i64}) attributes {vernon.entry, "
      "vernon.stage = \"vertex\"} {\n"
      "    return\n"
      "  }\n"
      "}\n";
  static const char invalid_module[] =
      "module {\n"
      "  func.func @compute_main() attributes {vernon.entry, "
      "vernon.stage = \"compute\"} {\n"
      "    return\n"
      "  }\n"
      "}\n";
  static const char cpu_module[] =
      "module {\n"
      "  func.func @add_vectors("
      "%left: tensor<4xf32> {vernon.interface = \"input\", "
      "vernon.location = 0 : i64}, "
      "%right: tensor<4xf32> {vernon.interface = \"input\", "
      "vernon.location = 1 : i64}) -> "
      "(tensor<4xf32> {vernon.interface = \"output\", "
      "vernon.location = 0 : i64}) attributes {vernon.entry, "
      "vernon.stage = \"fragment\"} {\n"
      "    %sum = arith.addf %left, %right : tensor<4xf32>\n"
      "    return %sum : tensor<4xf32>\n"
      "  }\n"
      "}\n";
  static const char cpu_intrinsic_module[] =
      "module {\n"
      "  func.func @normal_score("
      "%normal: tensor<3xf32> {vernon.interface = \"input\", "
      "vernon.location = 0 : i64}, "
      "%light: tensor<3xf32> {vernon.interface = \"input\", "
      "vernon.location = 1 : i64}) -> "
      "(f32 {vernon.interface = \"output\", vernon.location = 0 : i64}) "
      "attributes {vernon.entry, vernon.stage = \"fragment\"} {\n"
      "    %unit = \"vernon.intrinsic\"(%normal) "
      "{name = \"normalize\"} : (tensor<3xf32>) -> tensor<3xf32>\n"
      "    %score = \"vernon.intrinsic\"(%unit, %light) "
      "{name = \"dot\"} : (tensor<3xf32>, tensor<3xf32>) -> f32\n"
      "    return %score : f32\n"
      "  }\n"
      "}\n";
  static const char cpu_compute_module[] =
      "module {\n"
      "  func.func @increment("
      "%values: !vernon.buffer<f32, \"read_write\"> "
      "{vernon.interface = \"resource\", vernon.set = 0 : i64, "
      "vernon.binding = 0 : i64}, "
      "%id: index {vernon.interface = \"input\", "
      "vernon.builtin = \"global_invocation_id\"}) attributes {vernon.entry, "
      "vernon.stage = \"compute\", "
      "vernon.workgroup_size = array<i32: 8, 1, 1>} {\n"
      "    %value = \"vernon.intrinsic\"(%values, %id) "
      "{name = \"buffer_load\"} : "
      "(!vernon.buffer<f32, \"read_write\">, index) -> f32\n"
      "    %one = arith.constant 1.0 : f32\n"
      "    %sum = arith.addf %value, %one : f32\n"
      "    \"vernon.intrinsic\"(%values, %id, %sum) "
      "{name = \"buffer_store\"} : "
      "(!vernon.buffer<f32, \"read_write\">, index, f32) -> ()\n"
      "    return\n"
      "  }\n"
      "}\n";
  static const char cuda_while_module[] =
      "module {\n"
      "  func.func @loop(%values: !vernon.buffer<f32, \"read_write\"> "
      "{vernon.interface = \"resource\", vernon.set = 0 : i64, "
      "vernon.binding = 0 : i64}, %phase: f32 "
      "{vernon.interface = \"input\", vernon.location = 1 : i64}) "
      "attributes {vernon.entry, "
      "vernon.stage = \"compute\", "
      "vernon.workgroup_size = array<i32: 1, 1, 1>} {\n"
      "    %true = arith.constant true\n"
      "    scf.if %true {\n"
      "      %minus_point_eight = arith.constant -8.000000e-01 : f32\n"
      "      %cosine = math.cos %phase : f32\n"
      "      %point_two = arith.constant 2.000000e-01 : f32\n"
      "      %imaginary = arith.mulf %cosine, %point_two : f32\n"
      "      %c = \"vernon.intrinsic\"(%minus_point_eight, %imaginary) "
      "{name = \"construct\"} : (f32, f32) -> tensor<2xf32>\n"
      "      %point_one = arith.constant 1.000000e-01 : f32\n"
      "      %z = \"vernon.intrinsic\"(%point_one, %point_two) "
      "{name = \"construct\"} : (f32, f32) -> tensor<2xf32>\n"
      "      %initial = arith.constant 0 : i32\n"
      "      %result, %final = scf.while "
      "(%iteration = %initial, %value = %z) : "
      "(i32, tensor<2xf32>) -> (i32, tensor<2xf32>) {\n"
      "        %squared = \"vernon.intrinsic\"(%value, %value) "
      "{name = \"dot\"} : (tensor<2xf32>, tensor<2xf32>) -> f32\n"
      "        %length = math.sqrt %squared : f32\n"
      "        %radius = arith.constant 2.000000e+01 : f32\n"
      "        %inside = arith.cmpf olt, %length, %radius : f32\n"
      "        %limit = arith.constant 4 : i32\n"
      "        %below_limit = arith.cmpi slt, %iteration, %limit : i32\n"
      "        %condition = arith.andi %inside, %below_limit : i1\n"
      "        scf.condition(%condition) %iteration, %value "
      ": i32, tensor<2xf32>\n"
      "      } do {\n"
      "        ^bb0(%iteration: i32, %value: tensor<2xf32>):\n"
      "        %one = arith.constant 1 : i32\n"
      "        %next = arith.addi %iteration, %one : i32\n"
      "        scf.yield %next, %c : i32, tensor<2xf32>\n"
      "      }\n"
      "      %index = arith.constant 0 : index\n"
      "      %value = arith.sitofp %result : i32 to f32\n"
      "      \"vernon.intrinsic\"(%values, %index, %value) "
      "{name = \"buffer_store\"} : "
      "(!vernon.buffer<f32, \"read_write\">, index, f32) -> ()\n"
      "      scf.yield\n"
      "    } else {\n"
      "      scf.yield\n"
      "    }\n"
      "    return\n"
      "  }\n"
      "}\n";
  static const char cuda_rank_three_tensor_module[] =
      "module {\n"
      "  func.func @tensor3(%values: !vernon.buffer<f32, \"read_write\"> "
      "{vernon.interface = \"resource\", vernon.set = 0 : i64, "
      "vernon.binding = 0 : i64}) attributes {vernon.entry, "
      "vernon.stage = \"compute\", "
      "vernon.workgroup_size = array<i32: 1, 1, 1>} {\n"
      "    %ones = arith.constant dense<1.0> : tensor<2x3x4xf32>\n"
      "    %two = arith.constant 2.0 : f32\n"
      "    %twos = tensor.splat %two : tensor<2x3x4xf32>\n"
      "    %sum = arith.addf %ones, %twos : tensor<2x3x4xf32>\n"
      "    %i = arith.constant 1 : index\n"
      "    %j = arith.constant 2 : index\n"
      "    %k = arith.constant 3 : index\n"
      "    %value = tensor.extract %sum[%i, %j, %k] "
      ": tensor<2x3x4xf32>\n"
      "    %index = arith.constant 0 : index\n"
      "    \"vernon.intrinsic\"(%values, %index, %value) "
      "{name = \"buffer_store\"} : "
      "(!vernon.buffer<f32, \"read_write\">, index, f32) -> ()\n"
      "    return\n"
      "  }\n"
      "}\n";
  static const char cuda_dynamic_local_tensor_module[] =
      "module {\n"
      "  func.func @dynamic_local("
      "%values: !vernon.buffer<f32, \"read_write\"> "
      "{vernon.interface = \"resource\", vernon.set = 0 : i64, "
      "vernon.binding = 0 : i64}) attributes {vernon.entry, "
      "vernon.stage = \"compute\", "
      "vernon.workgroup_size = array<i32: 1, 1, 1>} {\n"
      "    %size = arith.constant 4 : index\n"
      "    %value = tensor.empty(%size) : tensor<?xf32>\n"
      "    %index = arith.constant 0 : index\n"
      "    %element = tensor.extract %value[%index] : tensor<?xf32>\n"
      "    \"vernon.intrinsic\"(%values, %index, %element) "
      "{name = \"buffer_store\"} : "
      "(!vernon.buffer<f32, \"read_write\">, index, f32) -> ()\n"
      "    return\n"
      "  }\n"
      "}\n";
  static const char cpu_texture_module[] =
      "module {\n"
      "  func.func @sample_color("
      "%texture: !vernon.texture<\"2d\", f32> "
      "{vernon.interface = \"resource\", vernon.set = 0 : i64, "
      "vernon.binding = 0 : i64}, "
      "%sampler: !vernon.sampler {vernon.interface = \"resource\", "
      "vernon.set = 0 : i64, vernon.binding = 1 : i64}, "
      "%uv: tensor<2xf32> {vernon.interface = \"input\", "
      "vernon.location = 0 : i64}) -> "
      "(tensor<4xf32> {vernon.interface = \"output\", "
      "vernon.location = 0 : i64}) attributes {vernon.entry, "
      "vernon.stage = \"fragment\"} {\n"
      "    %color = \"vernon.intrinsic\"(%texture, %sampler, %uv) "
      "{name = \"texture_sample\"} : "
      "(!vernon.texture<\"2d\", f32>, !vernon.sampler, tensor<2xf32>) "
      "-> tensor<4xf32>\n"
      "    return %color : tensor<4xf32>\n"
      "  }\n"
      "}\n";

  VernonCompilerContext *context = vernonCompilerCreate();
  assert(context != NULL);
  VernonTargetCapabilities vulkan =
      vernonCompilerGetTargetCapabilities(context, VERNON_TARGET_VULKAN);
  assert(vulkan.available && vulkan.supports_graphics);
  VernonTargetCapabilities cuda =
      vernonCompilerGetTargetCapabilities(context, VERNON_TARGET_CUDA);
  assert(cuda.available && cuda.supports_compute && !cuda.supports_graphics);
  VernonTargetCapabilities cpu =
      vernonCompilerGetTargetCapabilities(context, VERNON_TARGET_CPU);
  assert(cpu.available && cpu.supports_graphics && cpu.supports_compute);
  VernonTargetCapabilities opengl =
      vernonCompilerGetTargetCapabilities(context, VERNON_TARGET_OPENGL);
  VernonTargetCapabilities opengles =
      vernonCompilerGetTargetCapabilities(context, VERNON_TARGET_OPENGL_ES);
  VernonTargetCapabilities metal =
      vernonCompilerGetTargetCapabilities(context, VERNON_TARGET_METAL);

  VernonCompileResult *validation =
      vernonCompilerValidateMlir(context, module, strlen(module));
  assert(validation != NULL);
  assert(vernonCompileResultGetStatus(validation) == VERNON_STATUS_OK);

  VernonStringView reflection = vernonCompileResultGetReflection(validation);
  assert(reflection.data != NULL);
  assert(reflection.size != 0);
  assert(reflection.data[0] == '{');
  assert(vernonCompileResultGetCpuEntry(validation, "vertex_main", 11) == NULL);
  vernonCompileResultDestroy(validation);

  VernonCompileResult *invalid = vernonCompilerValidateMlir(
      context, invalid_module, strlen(invalid_module));
  assert(invalid != NULL);
  assert(vernonCompileResultGetStatus(invalid) ==
         VERNON_STATUS_VERIFICATION_ERROR);
  vernonCompileResultDestroy(invalid);

  VernonCompileResult *parse_error =
      vernonCompilerValidateMlir(context, "not mlir", 8);
  assert(parse_error != NULL);
  assert(vernonCompileResultGetStatus(parse_error) ==
         VERNON_STATUS_PARSE_ERROR);
  assert(vernonCompileResultGetDiagnostics(parse_error).size != 0);
  vernonCompileResultDestroy(parse_error);

  VernonCompileResult *compile = vernonCompilerCompileMlir(
      context, module, strlen(module), VERNON_TARGET_DIRECTX);
  assert(compile != NULL);
  assert(vernonCompileResultGetStatus(compile) ==
         VERNON_STATUS_UNSUPPORTED_TARGET);
  vernonCompileResultDestroy(compile);

  VernonCompileResult *vulkan_compile = vernonCompilerCompileMlir(
      context, module, strlen(module), VERNON_TARGET_VULKAN);
  assert(vulkan_compile != NULL);
  assert(vernonCompileResultGetStatus(vulkan_compile) == VERNON_STATUS_OK);
  assert(vernonCompileResultGetArtifactCount(vulkan_compile) == 1);
  VernonStringView name = vernonCompileResultGetArtifactName(vulkan_compile, 0);
  assert(name.size == strlen("module.spv"));
  assert(memcmp(name.data, "module.spv", name.size) == 0);
  VernonStringView spirv =
      vernonCompileResultGetArtifactData(vulkan_compile, 0);
  uint32_t magic = 0;
  assert(spirv.size >= sizeof(magic));
  memcpy(&magic, spirv.data, sizeof(magic));
  assert(magic == 0x07230203u);
  VernonStringView vulkan_reflection =
      vernonCompileResultGetReflection(vulkan_compile);
  assert(view_contains(vulkan_reflection, "\"schema_version\":2"));
  assert(view_contains(vulkan_reflection, "\"target\":\"vulkan\""));
  assert(view_contains(vulkan_reflection, "\"entry_point\":\"vertex_main\""));
  assert(view_contains(vulkan_reflection, "\"filename\":\"module.spv\""));
  vernonCompileResultDestroy(vulkan_compile);

  if (opengl.available) {
    VernonCompileOptions options = {sizeof(VernonCompileOptions), 450, {0}};
    VernonCompileResult *opengl_compile = vernonCompilerCompileMlirWithOptions(
        context, cpu_module, strlen(cpu_module), VERNON_TARGET_OPENGL,
        &options);
    assert(opengl_compile != NULL);
    assert(vernonCompileResultGetStatus(opengl_compile) == VERNON_STATUS_OK);
    VernonStringView glsl =
        vernonCompileResultGetArtifactData(opengl_compile, 0);
    assert(glsl.size >= strlen("#version 450"));
    assert(memcmp(glsl.data, "#version 450", strlen("#version 450")) == 0);
    VernonStringView opengl_reflection =
        vernonCompileResultGetReflection(opengl_compile);
    assert(view_contains(opengl_reflection, "\"target\":\"opengl\""));
    assert(view_contains(opengl_reflection, "\"glsl_version\":450"));
    assert(view_contains(opengl_reflection, "\"format\":\"glsl\""));
    vernonCompileResultDestroy(opengl_compile);

    options.glsl_version = 330;
    VernonCompileResult *invalid_options = vernonCompilerCompileMlirWithOptions(
        context, module, strlen(module), VERNON_TARGET_VULKAN, &options);
    assert(invalid_options != NULL);
    assert(vernonCompileResultGetStatus(invalid_options) ==
           VERNON_STATUS_INVALID_ARGUMENT);
    vernonCompileResultDestroy(invalid_options);
  }

  if (opengles.available) {
    VernonCompileOptions options = {sizeof(VernonCompileOptions), 310, {0}};
    VernonCompileResult *opengles_compile =
        vernonCompilerCompileMlirWithOptions(context, cpu_module,
                                             strlen(cpu_module),
                                             VERNON_TARGET_OPENGL_ES, &options);
    assert(opengles_compile != NULL);
    assert(vernonCompileResultGetStatus(opengles_compile) == VERNON_STATUS_OK);
    VernonStringView glsl =
        vernonCompileResultGetArtifactData(opengles_compile, 0);
    assert(glsl.size >= strlen("#version 310 es"));
    assert(memcmp(glsl.data, "#version 310 es", strlen("#version 310 es")) ==
           0);
    vernonCompileResultDestroy(opengles_compile);
  }

  VernonCompileResult *vulkan_compute_compile = vernonCompilerCompileMlir(
      context, cpu_compute_module, strlen(cpu_compute_module),
      VERNON_TARGET_VULKAN);
  assert(vulkan_compute_compile != NULL);
  assert(vernonCompileResultGetStatus(vulkan_compute_compile) ==
         VERNON_STATUS_OK);
  assert(vernonCompileResultGetArtifactCount(vulkan_compute_compile) == 1);
  vernonCompileResultDestroy(vulkan_compute_compile);

  VernonCompileResult *cuda_compile =
      vernonCompilerCompileMlir(context, cpu_compute_module,
                                strlen(cpu_compute_module), VERNON_TARGET_CUDA);
  assert(cuda_compile != NULL);
  assert(vernonCompileResultGetStatus(cuda_compile) == VERNON_STATUS_OK);
  VernonStringView ptx = vernonCompileResultGetArtifactData(cuda_compile, 0);
  assert(ptx.size != 0);
  assert(strstr(ptx.data, ".version") != NULL);
  vernonCompileResultDestroy(cuda_compile);

  VernonCompileResult *cuda_while_compile =
      vernonCompilerCompileMlir(context, cuda_while_module,
                                strlen(cuda_while_module), VERNON_TARGET_CUDA);
  assert(cuda_while_compile != NULL);
  assert(vernonCompileResultGetStatus(cuda_while_compile) == VERNON_STATUS_OK);
  VernonStringView while_ptx =
      vernonCompileResultGetArtifactData(cuda_while_compile, 0);
  assert(while_ptx.size != 0);
  assert(view_contains(while_ptx, ".version"));
  assert(view_contains(while_ptx, "cos.approx"));
  assert(view_contains(while_ptx, "sqrt.rn"));
  assert(!view_contains(while_ptx, "__nv_"));
  vernonCompileResultDestroy(cuda_while_compile);

  if (metal.available) {
    VernonCompileResult *metal_compute = vernonCompilerCompileMlir(
        context, cuda_while_module, strlen(cuda_while_module),
        VERNON_TARGET_METAL);
    assert(metal_compute != NULL);
    assert(vernonCompileResultGetStatus(metal_compute) == VERNON_STATUS_OK);
    VernonStringView metal_source =
        vernonCompileResultGetArtifactData(metal_compute, 0);
    assert(view_contains(metal_source, "kernel void loop"));
    assert(view_contains(vernonCompileResultGetReflection(metal_compute),
                         "\"target\":\"metal\""));
    vernonCompileResultDestroy(metal_compute);
  }

  VernonCompileResult *cuda_rank_three_compile = vernonCompilerCompileMlir(
      context, cuda_rank_three_tensor_module,
      strlen(cuda_rank_three_tensor_module), VERNON_TARGET_CUDA);
  assert(cuda_rank_three_compile != NULL);
  if (vernonCompileResultGetStatus(cuda_rank_three_compile) !=
      VERNON_STATUS_OK) {
    VernonStringView diagnostics =
        vernonCompileResultGetDiagnostics(cuda_rank_three_compile);
    fprintf(stderr, "%.*s\n", (int)diagnostics.size, diagnostics.data);
  }
  assert(vernonCompileResultGetStatus(cuda_rank_three_compile) ==
         VERNON_STATUS_OK);
  assert(vernonCompileResultGetArtifactCount(cuda_rank_three_compile) == 1);
  vernonCompileResultDestroy(cuda_rank_three_compile);

  VernonCompileResult *cuda_dynamic_local_compile = vernonCompilerCompileMlir(
      context, cuda_dynamic_local_tensor_module,
      strlen(cuda_dynamic_local_tensor_module), VERNON_TARGET_CUDA);
  assert(cuda_dynamic_local_compile != NULL);
  assert(vernonCompileResultGetStatus(cuda_dynamic_local_compile) !=
         VERNON_STATUS_OK);
  assert(view_contains(
      vernonCompileResultGetDiagnostics(cuda_dynamic_local_compile),
      "dynamic local value Tensor"));
  vernonCompileResultDestroy(cuda_dynamic_local_compile);

  VernonCompileResult *cpu_compile = vernonCompilerCompileMlir(
      context, cpu_module, strlen(cpu_module), VERNON_TARGET_CPU);
  assert(cpu_compile != NULL);
  assert(vernonCompileResultGetStatus(cpu_compile) == VERNON_STATUS_OK);
  VernonCpuEntryPoint add_vectors =
      vernonCompileResultGetCpuEntry(cpu_compile, "add_vectors", 11);
  assert(add_vectors != NULL);
  float cpu_arguments[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float cpu_results[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  VernonCpuInvocation invocation = {cpu_arguments, sizeof(cpu_arguments),
                                    cpu_results, sizeof(cpu_results), NULL};
  assert(add_vectors(&invocation) == VERNON_STATUS_OK);
  assert(cpu_results[0] == 6.0f && cpu_results[1] == 8.0f);
  assert(cpu_results[2] == 10.0f && cpu_results[3] == 12.0f);
  invocation.arguments_size = 0;
  assert(add_vectors(&invocation) == VERNON_STATUS_INVALID_ARGUMENT);
  vernonCompileResultDestroy(cpu_compile);

  VernonCompileResult *cpu_intrinsic_compile = vernonCompilerCompileMlir(
      context, cpu_intrinsic_module, strlen(cpu_intrinsic_module),
      VERNON_TARGET_CPU);
  assert(cpu_intrinsic_compile != NULL);
  assert(vernonCompileResultGetStatus(cpu_intrinsic_compile) ==
         VERNON_STATUS_OK);
  VernonCpuEntryPoint normal_score =
      vernonCompileResultGetCpuEntry(cpu_intrinsic_compile, "normal_score", 12);
  assert(normal_score != NULL);
  float intrinsic_arguments[7] = {0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  float intrinsic_result = 0.0f;
  VernonCpuInvocation intrinsic_invocation = {
      intrinsic_arguments, sizeof(intrinsic_arguments), &intrinsic_result,
      sizeof(intrinsic_result), NULL};
  assert(normal_score(&intrinsic_invocation) == VERNON_STATUS_OK);
  assert(intrinsic_result == 1.0f);
  vernonCompileResultDestroy(cpu_intrinsic_compile);

  VernonCompileResult *cpu_compute_compile =
      vernonCompilerCompileMlir(context, cpu_compute_module,
                                strlen(cpu_compute_module), VERNON_TARGET_CPU);
  assert(cpu_compute_compile != NULL);
  assert(vernonCompileResultGetStatus(cpu_compute_compile) == VERNON_STATUS_OK);
  VernonCpuEntryPoint increment =
      vernonCompileResultGetCpuEntry(cpu_compute_compile, "increment", 9);
  assert(increment != NULL);
  float compute_values[3] = {2.0f, 4.0f, 6.0f};
  struct {
    float *values;
    size_t id;
  } compute_arguments = {compute_values, 1};
  VernonCpuInvocation compute_invocation = {
      &compute_arguments, sizeof(compute_arguments), NULL, 0, NULL};
  assert(increment(&compute_invocation) == VERNON_STATUS_OK);
  assert(compute_values[0] == 2.0f && compute_values[1] == 5.0f &&
         compute_values[2] == 6.0f);
  vernonCompileResultDestroy(cpu_compute_compile);

  VernonCompileResult *cpu_texture_compile =
      vernonCompilerCompileMlir(context, cpu_texture_module,
                                strlen(cpu_texture_module), VERNON_TARGET_CPU);
  assert(cpu_texture_compile != NULL);
  assert(vernonCompileResultGetStatus(cpu_texture_compile) == VERNON_STATUS_OK);
  VernonCpuEntryPoint sample_color =
      vernonCompileResultGetCpuEntry(cpu_texture_compile, "sample_color", 12);
  assert(sample_color != NULL);
  struct {
    uintptr_t texture;
    uintptr_t sampler;
    float uv[2];
  } texture_arguments = {3, 0, {0.25f, 0.75f}};
  float texture_result[4] = {0};
  float texture_bias = 0.5f;
  VernonCpuTextureCallbacks texture_callbacks = {&texture_bias, sample_texture,
                                                 NULL};
  VernonCpuInvocation texture_invocation = {
      &texture_arguments, sizeof(texture_arguments), texture_result,
      sizeof(texture_result), &texture_callbacks};
  assert(sample_color(&texture_invocation) == VERNON_STATUS_OK);
  assert(texture_result[0] == 0.25f && texture_result[1] == 0.75f);
  assert(texture_result[2] == 3.0f && texture_result[3] == 0.5f);
  texture_invocation.textures = NULL;
  assert(sample_color(&texture_invocation) == VERNON_STATUS_INVALID_ARGUMENT);
  vernonCompileResultDestroy(cpu_texture_compile);

  vernonCompilerDestroy(context);
  return 0;
}
