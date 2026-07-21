#include "VernonCompiler.h"

#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <string_view>

namespace {

constexpr std::string_view module = R"mlir(
module {
  func.func @cube_map_vertex(
      %aPos: tensor<3xf32> {
        vernon.interface = "input",
        vernon.location = 0 : i64,
        vernon.source_name = "aPos"
      },
      %projection: tensor<4x4xf32> {
        vernon.interface = "uniform",
        vernon.source_name = "projection"
      },
      %view: tensor<4x4xf32> {
        vernon.interface = "uniform",
        vernon.source_name = "view"
      },
      %model: tensor<4x4xf32> {
        vernon.interface = "uniform",
        vernon.source_name = "model"
      }) -> (
        tensor<4xf32> {
          vernon.interface = "output",
          vernon.builtin = "position",
          vernon.source_name = "position"
        },
        tensor<3xf32> {
          vernon.interface = "output",
          vernon.location = 0 : i64,
          vernon.source_name = "tex_coord"
        }) attributes {vernon.entry, vernon.stage = "vertex"} {
    %one = arith.constant 1.0 : f32
    %position = "vernon.intrinsic"(%aPos, %one) {
      name = "construct"
    } : (tensor<3xf32>, f32) -> tensor<4xf32>
    return %position, %aPos : tensor<4xf32>, tensor<3xf32>
  }

  func.func @cube_map_fragment(
      %texCoord: tensor<3xf32> {
        vernon.interface = "input",
        vernon.location = 0 : i64,
        vernon.source_name = "tex_coord"
      },
      %cubeMap: !vernon.texture<"cube", f32> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64,
        vernon.source_name = "cubeMap"
      },
      %sampler: !vernon.sampler {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 1 : i64,
        vernon.source_name = "cubeMapSampler"
      }) -> (
        tensor<4xf32> {
          vernon.interface = "output",
          vernon.location = 0 : i64,
          vernon.source_name = "color"
        },
        tensor<4xf32> {
          vernon.interface = "output",
          vernon.location = 1 : i64,
          vernon.source_name = "bloom_color"
        }) attributes {vernon.entry, vernon.stage = "fragment"} {
    %color = "vernon.intrinsic"(%cubeMap, %sampler, %texCoord) {
      name = "texture_sample"
    } : (!vernon.texture<"cube", f32>, !vernon.sampler, tensor<3xf32>)
        -> tensor<4xf32>
    %rgb = "vernon.swizzle"(%color) {
      mask = "rgb"
    } : (tensor<4xf32>) -> tensor<3xf32>
    %rgba = "vernon.swizzle"(%color) {
      mask = "rgba"
    } : (tensor<4xf32>) -> tensor<4xf32>
    %alpha = "vernon.swizzle"(%color) {
      mask = "a"
    } : (tensor<4xf32>) -> f32
    %redWeight = arith.constant 0.2126 : f32
    %greenWeight = arith.constant 0.7152 : f32
    %blueWeight = arith.constant 0.0722 : f32
    %weights = "vernon.intrinsic"(
        %redWeight, %greenWeight, %blueWeight) {
      name = "construct"
    } : (f32, f32, f32) -> tensor<3xf32>
    %brightness = "vernon.intrinsic"(%rgb, %weights) {
      name = "dot"
    } : (tensor<3xf32>, tensor<3xf32>) -> f32
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %black = "vernon.intrinsic"(%zero, %zero, %zero, %alpha) {
      name = "construct"
    } : (f32, f32, f32, f32) -> tensor<4xf32>
    %threshold = arith.cmpf ogt, %brightness, %one : f32
    %selectedBrightness = arith.select %threshold, %brightness, %zero : f32
    %bloom = scf.if %threshold -> (tensor<4xf32>) {
      %nestedThreshold = arith.cmpf oge, %selectedBrightness, %brightness : f32
      %nested = scf.if %nestedThreshold -> (tensor<4xf32>) {
        scf.yield %color : tensor<4xf32>
      } else {
        scf.yield %black : tensor<4xf32>
      }
      scf.yield %nested : tensor<4xf32>
    } else {
      scf.yield %black : tensor<4xf32>
    }
    return %rgba, %bloom : tensor<4xf32>, tensor<4xf32>
  }
}
)mlir";

std::string artifacts(const VernonCompileResult *result) {
  std::string output;
  for (size_t index = 0; index < vernonCompileResultGetArtifactCount(result);
       ++index) {
    VernonStringView artifact =
        vernonCompileResultGetArtifactData(result, index);
    output.append(artifact.data, artifact.size);
  }
  return output;
}

} // namespace

TEST(CompilerGraphicsOutput, PreservesInterfacesTexturesAndSwizzles) {
  VernonCompilerContext *compiler = vernonCompilerCreate();
  ASSERT_TRUE(compiler);
  for (VernonTarget target :
       {VERNON_TARGET_VULKAN, VERNON_TARGET_OPENGL, VERNON_TARGET_METAL}) {
    VernonCompileResult *result = vernonCompilerCompileMlir(
        compiler, module.data(), module.size(), target);
    ASSERT_TRUE(result);
    if (vernonCompileResultGetStatus(result) != VERNON_STATUS_OK) {
      const VernonStringView diagnostics =
          vernonCompileResultGetDiagnostics(result);
      std::fprintf(stderr, "target %d compile failed: %.*s\n",
                   static_cast<int>(target), static_cast<int>(diagnostics.size),
                   diagnostics.data);
    }
    ASSERT_TRUE(vernonCompileResultGetStatus(result) == VERNON_STATUS_OK);
    const std::string output = artifacts(result);
    for (std::string_view name :
         {"projection", "view", "model", "aPos", "TexCoord", "cubeMap", "color",
          "bloom_color"}) {
      if (output.find(name) == std::string::npos) {
        std::fprintf(stderr, "target %d missing %.*s\n",
                     static_cast<int>(target), static_cast<int>(name.size()),
                     name.data());
        if (target != VERNON_TARGET_VULKAN)
          std::fwrite(output.data(), 1, output.size(), stderr);
      }
      ASSERT_TRUE(output.find(name) != std::string::npos);
    }
    if (target != VERNON_TARGET_VULKAN) {
      ASSERT_TRUE(output.find("if (") != std::string::npos);
      ASSERT_TRUE(output.find(">= 1.0") != std::string::npos ||
                  output.find("> 1.0") != std::string::npos);
      ASSERT_TRUE(output.find("dot(") != std::string::npos);
      ASSERT_TRUE(output.find(".xyz") != std::string::npos);
      ASSERT_TRUE(output.find("dot(vec3(0.0") == std::string::npos);
      ASSERT_TRUE(output.find("dot(float3(0.0") == std::string::npos);
    }
    vernonCompileResultDestroy(result);
  }
  vernonCompilerDestroy(compiler);
}
