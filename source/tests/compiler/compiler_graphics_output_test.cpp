#include "VernonCompiler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
    %worldPosition = "vernon.intrinsic"(%model, %position) {
      name = "matmul"
    } : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %viewPosition = "vernon.intrinsic"(%view, %worldPosition) {
      name = "matmul"
    } : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %clipPosition = "vernon.intrinsic"(%projection, %viewPosition) {
      name = "matmul"
    } : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %clipPosition, %aPos : tensor<4xf32>, tensor<3xf32>
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
    for (size_t index = 0; index < vernonCompileResultGetArtifactCount(result); ++index) {
        VernonStringView artifact = vernonCompileResultGetArtifactData(result, index);
        output.append(artifact.data, artifact.size);
    }
    return output;
}

void expectPackedPushConstants(std::string_view artifact) {
    ASSERT_EQ(artifact.size() % sizeof(uint32_t), 0u);
    std::vector<uint32_t> words(artifact.size() / sizeof(uint32_t));
    std::memcpy(words.data(), artifact.data(), artifact.size());
    ASSERT_GE(words.size(), 5u);
    ASSERT_EQ(words[0], 0x07230203u);

    constexpr uint16_t opEntryPoint = 15;
    constexpr uint16_t opTypeStruct = 30;
    constexpr uint16_t opTypePointer = 32;
    constexpr uint16_t opVariable = 59;
    constexpr uint16_t opMemberDecorate = 72;
    constexpr uint32_t storageClassPushConstant = 9;
    constexpr uint32_t decorationBlock = 2;
    constexpr uint32_t decorationColMajor = 5;
    constexpr uint32_t decorationMatrixStride = 7;
    constexpr uint32_t decorationOffset = 35;

    uint32_t pushPointer = 0;
    uint32_t pushStruct = 0;
    uint32_t pushVariable = 0;
    std::vector<std::pair<uint32_t, uint32_t>> pushPointers;
    size_t pushVariableCount = 0;
    size_t pushInterfaceCount = 0;
    std::array<uint32_t, 3> offsets{UINT32_MAX, UINT32_MAX, UINT32_MAX};
    std::array<uint32_t, 3> strides{};
    std::array<bool, 3> columnMajor{};
    bool blockDecorated = false;
    size_t structMemberCount = 0;

    for (size_t cursor = 5; cursor < words.size();) {
        const uint16_t wordCount = static_cast<uint16_t>(words[cursor] >> 16);
        const uint16_t opcode = static_cast<uint16_t>(words[cursor]);
        ASSERT_GT(wordCount, 0u);
        ASSERT_LE(cursor + wordCount, words.size());
        if (opcode == opTypePointer && wordCount == 4 && words[cursor + 2] == storageClassPushConstant) {
            pushPointers.emplace_back(words[cursor + 1], words[cursor + 3]);
        } else if (opcode == opVariable && wordCount >= 4 && words[cursor + 3] == storageClassPushConstant) {
            ++pushVariableCount;
            pushPointer = words[cursor + 1];
            pushVariable = words[cursor + 2];
        } else if (opcode == opTypeStruct && wordCount >= 2 && words[cursor + 1] == pushStruct) {
            structMemberCount = wordCount - 2;
        } else if (opcode == opMemberDecorate && wordCount >= 4 && words[cursor + 1] == pushStruct &&
                   words[cursor + 2] < offsets.size()) {
            const size_t member = words[cursor + 2];
            const uint32_t decoration = words[cursor + 3];
            if (decoration == decorationOffset && wordCount == 5)
                offsets[member] = words[cursor + 4];
            else if (decoration == decorationMatrixStride && wordCount == 5)
                strides[member] = words[cursor + 4];
            else if (decoration == decorationColMajor && wordCount == 4)
                columnMajor[member] = true;
        }
        cursor += wordCount;
    }
    const auto pointer = std::find_if(pushPointers.begin(), pushPointers.end(),
                                      [&](const auto &value) { return value.first == pushPointer; });
    ASSERT_NE(pointer, pushPointers.end());
    pushStruct = pointer->second;

    for (size_t cursor = 5; cursor < words.size();) {
        const uint16_t wordCount = static_cast<uint16_t>(words[cursor] >> 16);
        const uint16_t opcode = static_cast<uint16_t>(words[cursor]);
        if (opcode == opTypeStruct && wordCount >= 2 && words[cursor + 1] == pushStruct) {
            structMemberCount = wordCount - 2;
        } else if (opcode == opMemberDecorate && wordCount >= 4 && words[cursor + 1] == pushStruct &&
                   words[cursor + 2] < offsets.size()) {
            const size_t member = words[cursor + 2];
            const uint32_t decoration = words[cursor + 3];
            if (decoration == decorationOffset && wordCount == 5)
                offsets[member] = words[cursor + 4];
            else if (decoration == decorationMatrixStride && wordCount == 5)
                strides[member] = words[cursor + 4];
            else if (decoration == decorationColMajor && wordCount == 4)
                columnMajor[member] = true;
        } else if (opcode == 71 && wordCount == 3 && words[cursor + 1] == pushStruct &&
                   words[cursor + 2] == decorationBlock) {
            blockDecorated = true;
        } else if (opcode == opEntryPoint && wordCount >= 4) {
            size_t interface = cursor + 3;
            while (interface < cursor + wordCount) {
                const uint32_t word = words[interface++];
                if ((word & 0xff000000u) == 0 || (word & 0x00ff0000u) == 0 || (word & 0x0000ff00u) == 0 ||
                    (word & 0x000000ffu) == 0)
                    break;
            }
            for (; interface < cursor + wordCount; ++interface)
                if (words[interface] == pushVariable)
                    ++pushInterfaceCount;
        }
        cursor += wordCount;
    }

    EXPECT_EQ(pushVariableCount, 1u);
    EXPECT_EQ(structMemberCount, 2u);
    EXPECT_EQ(offsets, (std::array<uint32_t, 3>{0, 64, UINT32_MAX}));
    EXPECT_EQ(strides, (std::array<uint32_t, 3>{16, 16, 0}));
    EXPECT_EQ(columnMajor, (std::array<bool, 3>{true, true, false}));
    EXPECT_TRUE(blockDecorated);
    EXPECT_EQ(pushInterfaceCount, 1u);
}

} // namespace

TEST(CompilerGraphicsOutput, PreservesInterfacesTexturesAndSwizzles) {
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    for (VernonTarget target : {VERNON_TARGET_VULKAN, VERNON_TARGET_OPENGL, VERNON_TARGET_OPENGL_ES,
                                VERNON_TARGET_METAL, VERNON_TARGET_DIRECTX}) {
        VernonCompileResult *result = vernonCompilerCompileMlir(compiler, module.data(), module.size(), target);
        ASSERT_TRUE(result);
        if (vernonCompileResultGetStatus(result) != VERNON_STATUS_OK) {
            const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
            std::fprintf(stderr, "target %d compile failed: %.*s\n", static_cast<int>(target),
                         static_cast<int>(diagnostics.size), diagnostics.data);
        }
        ASSERT_TRUE(vernonCompileResultGetStatus(result) == VERNON_STATUS_OK);
        std::string output;
        if (target == VERNON_TARGET_DIRECTX) {
            for (size_t index = 0; index < vernonCompileResultGetArtifactCount(result); ++index) {
                const VernonStringView artifact = vernonCompileResultGetArtifactData(result, index);
                ASSERT_GE(artifact.size, 4u);
                ASSERT_EQ(std::memcmp(artifact.data, "DXBC", 4), 0);
            }
            const VernonStringView reflection = vernonCompileResultGetReflection(result);
            output.assign(reflection.data, reflection.size);
            for (std::string_view name : {"aPos", "cubeMap", "color", "bloom_color", "projection"})
                ASSERT_TRUE(output.find(name) != std::string::npos);
            vernonCompileResultDestroy(result);
            continue;
        } else {
            output = artifacts(result);
        }
        for (std::string_view name : {"aPos", "TexCoord", "cubeMap", "color", "bloom_color"}) {
            if (output.find(name) == std::string::npos) {
                std::fprintf(stderr, "target %d missing %.*s\n", static_cast<int>(target),
                             static_cast<int>(name.size()), name.data());
                if (target != VERNON_TARGET_VULKAN)
                    std::fwrite(output.data(), 1, output.size(), stderr);
            }
            ASSERT_TRUE(output.find(name) != std::string::npos);
        }
        if (target == VERNON_TARGET_VULKAN) {
            expectPackedPushConstants(output);
        } else {
            for (std::string_view name : {"projection", "view", "model"})
                ASSERT_TRUE(output.find(name) != std::string::npos);
            EXPECT_EQ(output.find("cube_map_vertex_push_constants"), std::string::npos);
            if (target == VERNON_TARGET_OPENGL) {
                for (std::string_view declaration : {"uniform mat4 projection;", "uniform mat4 view;"})
                    EXPECT_NE(output.find(declaration), std::string::npos);
                EXPECT_NE(output.find("model._m0"), std::string::npos);
            }
        }
        if (target != VERNON_TARGET_VULKAN) {
            ASSERT_TRUE(output.find("if (") != std::string::npos);
            ASSERT_TRUE(output.find(">= 1.0") != std::string::npos || output.find("> 1.0") != std::string::npos);
            ASSERT_TRUE(output.find("dot(") != std::string::npos);
            ASSERT_TRUE(output.find(".xyz") != std::string::npos);
            ASSERT_TRUE(output.find("dot(vec3(0.0") == std::string::npos);
            ASSERT_TRUE(output.find("dot(float3(0.0") == std::string::npos);
        }
        vernonCompileResultDestroy(result);
    }
    vernonCompilerDestroy(compiler);
}

TEST(CompilerGraphicsOutput, LowersSamplingBuiltinsTextureSizeAndMath) {
    constexpr std::string_view samplingModule = R"mlir(
module {
  func.func @sampling_fragment(
      %texture: !vernon.texture<"2d", f32> {
        vernon.interface = "resource", vernon.set = 0 : i64,
        vernon.binding = 0 : i64, vernon.source_name = "texture"
      },
      %texture3d: !vernon.texture<"3d", f32> {
        vernon.interface = "resource", vernon.set = 0 : i64,
        vernon.binding = 2 : i64, vernon.source_name = "texture3d"
      },
      %textureCube: !vernon.texture<"cube", f32> {
        vernon.interface = "resource", vernon.set = 0 : i64,
        vernon.binding = 3 : i64, vernon.source_name = "textureCube"
      },
      %sampler: !vernon.sampler {
        vernon.interface = "resource", vernon.implicit = "sampler",
        vernon.implicit_texture = "texture",
        vernon.source_name = "__vernon_implicit_sampler_texture",
        vernon.set = 0 : i64, vernon.binding = 1 : i64
      },
      %uv: tensor<2xf32> {
        vernon.interface = "uniform"
      },
      %fragCoord: tensor<4xf32> {
        vernon.interface = "input", vernon.builtin = "frag_coord"
      },
      %frontFacing: i1 {
        vernon.interface = "input", vernon.builtin = "front_facing"
      }) -> (
        tensor<4xf32> {
          vernon.interface = "output", vernon.location = 0 : i64
        },
        tensor<2xi32> {
          vernon.interface = "output", vernon.location = 1 : i64
        },
        tensor<2xi32> {
          vernon.interface = "output", vernon.location = 2 : i64
        },
        tensor<3xi32> {
          vernon.interface = "output", vernon.location = 3 : i64
        },
        tensor<3xi32> {
          vernon.interface = "output", vernon.location = 4 : i64
        },
        tensor<2xi32> {
          vernon.interface = "output", vernon.location = 5 : i64
        },
        tensor<2xi32> {
          vernon.interface = "output", vernon.location = 6 : i64
        }) attributes {vernon.entry, vernon.stage = "fragment"} {
    %lod = arith.constant 1.0 : f32
    %level = arith.constant 1 : i32
    %implicit = "vernon.intrinsic"(%texture, %sampler, %uv)
        {name = "texture_sample"} :
        (!vernon.texture<"2d", f32>, !vernon.sampler, tensor<2xf32>)
        -> tensor<4xf32>
    %explicit = "vernon.intrinsic"(%texture, %sampler, %uv, %lod)
        {name = "texture_sample"} :
        (!vernon.texture<"2d", f32>, !vernon.sampler, tensor<2xf32>, f32)
        -> tensor<4xf32>
    %size2d0 = "vernon.intrinsic"(%texture) {name = "texture_size"} :
        (!vernon.texture<"2d", f32>) -> tensor<2xi32>
    %size2d1 = "vernon.intrinsic"(%texture, %level) {name = "texture_size"} :
        (!vernon.texture<"2d", f32>, i32) -> tensor<2xi32>
    %size3d0 = "vernon.intrinsic"(%texture3d) {name = "texture_size"} :
        (!vernon.texture<"3d", f32>) -> tensor<3xi32>
    %size3d1 = "vernon.intrinsic"(%texture3d, %level)
        {name = "texture_size"} :
        (!vernon.texture<"3d", f32>, i32) -> tensor<3xi32>
    %sizeCube0 = "vernon.intrinsic"(%textureCube) {name = "texture_size"} :
        (!vernon.texture<"cube", f32>) -> tensor<2xi32>
    %sizeCube1 = "vernon.intrinsic"(%textureCube, %level)
        {name = "texture_size"} :
        (!vernon.texture<"cube", f32>, i32) -> tensor<2xi32>
    %x = "vernon.swizzle"(%fragCoord) {mask = "x"} :
        (tensor<4xf32>) -> f32
    %sin = math.sin %x : f32
    %cos = math.cos %sin : f32
    %exp = math.exp %cos : f32
    %log = math.log %exp : f32
    %sqrt = math.sqrt %log : f32
    %abs = math.absf %sqrt : f32
    return %explicit, %size2d0, %size2d1, %size3d0, %size3d1,
        %sizeCube0, %sizeCube1 :
        tensor<4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<3xi32>,
        tensor<3xi32>, tensor<2xi32>, tensor<2xi32>
  }

  func.func @builtin_vertex(
      %vertexId: i32 {
        vernon.interface = "input", vernon.builtin = "vertex_index"
      },
      %instanceId: i32 {
        vernon.interface = "input", vernon.builtin = "instance_index"
      },
      %position: tensor<4xf32> {
        vernon.interface = "input", vernon.location = 0 : i64
      }) -> (tensor<4xf32> {
        vernon.interface = "output", vernon.builtin = "position"
      }) attributes {vernon.entry, vernon.stage = "vertex"} {
    return %position : tensor<4xf32>
  }
}
)mlir";

    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    for (VernonTarget target : {VERNON_TARGET_VULKAN, VERNON_TARGET_OPENGL}) {
        VernonCompileResult *result =
            vernonCompilerCompileMlir(compiler, samplingModule.data(), samplingModule.size(), target);
        ASSERT_TRUE(result);
        if (vernonCompileResultGetStatus(result) != VERNON_STATUS_OK) {
            const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
            std::fprintf(stderr, "sampling compile failed: %.*s\n", static_cast<int>(diagnostics.size),
                         diagnostics.data);
        }
        ASSERT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_OK);
        const std::string output = artifacts(result);
        if (target == VERNON_TARGET_VULKAN) {
            ASSERT_EQ(output.size() % sizeof(uint32_t), 0u);
            size_t queryCount = 0;
            const size_t wordCount = output.size() / sizeof(uint32_t);
            std::vector<uint32_t> words(wordCount);
            std::memcpy(words.data(), output.data(), output.size());
            for (size_t cursor = 5; cursor < wordCount;) {
                const uint16_t instructionWords = static_cast<uint16_t>(words[cursor] >> 16);
                ASSERT_GT(instructionWords, 0u);
                ASSERT_LE(cursor + instructionWords, wordCount);
                if (static_cast<uint16_t>(words[cursor]) == 103)
                    ++queryCount;
                cursor += instructionWords;
            }
            EXPECT_EQ(queryCount, 6u);
            const VernonStringView reflection = vernonCompileResultGetReflection(result);
            const std::string_view reflectionView(reflection.data, reflection.size);
            EXPECT_NE(reflectionView.find("\"vernon.implicit\":\"sampler\""), std::string_view::npos);
            EXPECT_EQ(reflectionView.find("\"vernon.implicit\":\"texture_size\""), std::string_view::npos);
            EXPECT_NE(reflectionView.find("\"sampled_texture_bindings\""), std::string_view::npos);
            EXPECT_EQ(reflectionView.find("\"sampled_texture_binding\":"), std::string_view::npos);
        } else {
            size_t queryCount = 0;
            for (size_t offset = 0; (offset = output.find("textureSize(", offset)) != std::string::npos; offset += 12)
                ++queryCount;
            EXPECT_EQ(queryCount, 6u);
            for (std::string_view texture : {"texture", "texture3d", "textureCube"})
                EXPECT_NE(output.find(texture), std::string::npos);
        }
        vernonCompileResultDestroy(result);
    }
    vernonCompilerDestroy(compiler);
}

TEST(CompilerGraphicsOutput, LowersRecursiveStaticTensorValuesAndReflectsPhysicalLayout) {
    constexpr std::string_view tensorModule = R"mlir(
module {
  func.func @static_tensor_fragment(
      %value: tensor<2x3x5xf32> {
        vernon.interface = "uniform",
        vernon.source_name = "value",
        vernon.dtype = "f32"
      },
      %matrix: tensor<3x4xf32> {
        vernon.interface = "uniform",
        vernon.source_name = "matrix",
        vernon.dtype = "f32"
      },
      %matrix2: tensor<4x4xf32> {
        vernon.interface = "uniform",
        vernon.source_name = "matrix2",
        vernon.dtype = "f32"
      },
      %matrix3: tensor<4x4xf32> {
        vernon.interface = "uniform",
        vernon.source_name = "matrix3",
        vernon.dtype = "f32"
      }) -> (f32 {
        vernon.interface = "output",
        vernon.location = 0 : i64
      }) attributes {vernon.entry, vernon.stage = "fragment"} {
    %ones = arith.constant dense<1.0> : tensor<2x3x5xf32>
    %sum = arith.addf %value, %ones : tensor<2x3x5xf32>
    %one = arith.constant 1.0 : f32
    %two = arith.constant 2.0 : f32
    %constructed = tensor.from_elements %one, %two : tensor<1x1x2xf32>
    %matrix_constructed = tensor.from_elements
        %one, %two, %one, %two, %one, %two,
        %one, %two, %one, %two, %one, %two : tensor<3x4xf32>
    %zero_index = arith.constant 0 : index
    %one_index = arith.constant 1 : index
    %two_index = arith.constant 2 : index
    %four_index = arith.constant 4 : index
    %selected = tensor.extract %sum[%one_index, %two_index, %four_index]
        : tensor<2x3x5xf32>
    %constructed_value = tensor.extract %constructed[
        %zero_index, %zero_index, %one_index] : tensor<1x1x2xf32>
    %matrix_value = tensor.extract %matrix_constructed[%one_index, %two_index]
        : tensor<3x4xf32>
    %partial = arith.addf %selected, %constructed_value : f32
    %result = arith.addf %partial, %matrix_value : f32
    return %result : f32
  }
}
)mlir";

    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    VernonCompileResult *result =
        vernonCompilerCompileMlir(compiler, tensorModule.data(), tensorModule.size(), VERNON_TARGET_VULKAN);
    ASSERT_TRUE(result);
    if (vernonCompileResultGetStatus(result) != VERNON_STATUS_OK) {
        const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
        std::fprintf(stderr, "static Tensor compile failed: %.*s\n", static_cast<int>(diagnostics.size),
                     diagnostics.data);
    }
    ASSERT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_OK);
    ASSERT_EQ(vernonCompileResultGetArtifactCount(result), 1u);
    const std::string spirv = artifacts(result);
    ASSERT_EQ(spirv.size() % sizeof(uint32_t), 0u);
    std::vector<uint32_t> words(spirv.size() / sizeof(uint32_t));
    std::memcpy(words.data(), spirv.data(), spirv.size());
    std::vector<uint32_t> arrayStrides;
    for (size_t cursor = 5; cursor < words.size();) {
        const uint16_t wordCount = static_cast<uint16_t>(words[cursor] >> 16);
        ASSERT_GT(wordCount, 0u);
        ASSERT_LE(cursor + wordCount, words.size());
        if (static_cast<uint16_t>(words[cursor]) == 71 && wordCount == 4 && words[cursor + 2] == 6)
            arrayStrides.push_back(words[cursor + 3]);
        cursor += wordCount;
    }
    for (uint32_t stride : {16u, 80u, 240u})
        EXPECT_NE(std::find(arrayStrides.begin(), arrayStrides.end(), stride), arrayStrides.end());

    const VernonStringView reflection = vernonCompileResultGetReflection(result);
    const std::string_view reflected(reflection.data, reflection.size);
    EXPECT_NE(reflected.find("\"kind\":\"tensor_value\""), std::string_view::npos);
    EXPECT_NE(reflected.find("\"shape\":[2,3,5]"), std::string_view::npos);
    EXPECT_NE(reflected.find("\"array_strides\":[240,80,16]"), std::string_view::npos);
    EXPECT_NE(reflected.find("\"physical_size\":480"), std::string_view::npos);
    EXPECT_NE(reflected.find("\"physical_alignment\":16"), std::string_view::npos);
    EXPECT_NE(reflected.find("\"proposed_storage_class\":\"Uniform\""), std::string_view::npos);
    EXPECT_NE(reflected.find("\"proposed_storage_class\":\"PushConstant\""), std::string_view::npos);
    EXPECT_NE(reflected.find("\"vernon.binding\":0"), std::string_view::npos);
    EXPECT_NE(reflected.find("\"vernon.binding\":1"), std::string_view::npos);
    EXPECT_NE(reflected.find("\"shape\":[3,4]"), std::string_view::npos);
    EXPECT_NE(reflected.find("\"matrix_stride\":16"), std::string_view::npos);
    EXPECT_NE(reflected.find("\"matrix_order\":\"column_major\""), std::string_view::npos);

    vernonCompileResultDestroy(result);
    for (VernonTarget target : {VERNON_TARGET_OPENGL, VERNON_TARGET_DIRECTX}) {
        result = vernonCompilerCompileMlir(compiler, tensorModule.data(), tensorModule.size(), target);
        ASSERT_TRUE(result);
        if (vernonCompileResultGetStatus(result) != VERNON_STATUS_OK) {
            const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
            std::fprintf(stderr, "target %d graphics Tensor spill compile failed: %.*s\n", static_cast<int>(target),
                         static_cast<int>(diagnostics.size), diagnostics.data);
        }
        ASSERT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_OK);
        ASSERT_EQ(vernonCompileResultGetArtifactCount(result), 1u);
        const VernonStringView targetReflection = vernonCompileResultGetReflection(result);
        const std::string_view targetReflected(targetReflection.data, targetReflection.size);
        EXPECT_NE(targetReflected.find("\"vernon.binding\":0"), std::string_view::npos);
        EXPECT_NE(targetReflected.find("\"vernon.binding\":1"), std::string_view::npos);
        if (target == VERNON_TARGET_OPENGL) {
            const std::string output = artifacts(result);
            EXPECT_NE(output.find("uniform"), std::string::npos);
            EXPECT_NE(output.find("value"), std::string::npos);
            EXPECT_NE(output.find("matrix3"), std::string::npos);
        }
        vernonCompileResultDestroy(result);
    }
    vernonCompilerDestroy(compiler);
}

TEST(CompilerGraphicsOutput, LowersStaticTensorByValueComputeAbi) {
    constexpr std::string_view tensorModule = R"mlir(
module {
  func.func @static_tensor_compute(
      %value: tensor<2x2x2xf32> {
        vernon.interface = "input",
        vernon.location = 0 : i64,
        vernon.dtype = "f32"
      },
      %output: !vernon.tensor_view<f32, 1, "write"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64,
        vernon.tensor_shape = array<i64: 1>
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    %one = arith.constant 1 : index
    %zero = arith.constant 0 : index
    %ones = arith.constant dense<1.0> : tensor<2x2x2xf32>
    %sum = arith.addf %value, %ones : tensor<2x2x2xf32>
    %value_at_index = tensor.extract %sum[%one, %zero, %one] : tensor<2x2x2xf32>
    "vernon.intrinsic"(%output, %zero, %value_at_index) {
      name = "tensor_view_store"
    } : (!vernon.tensor_view<f32, 1, "write">, index, f32) -> ()
    return
  }
}
)mlir";

    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    for (VernonTarget target : {VERNON_TARGET_VULKAN, VERNON_TARGET_OPENGL, VERNON_TARGET_DIRECTX}) {
        VernonCompileResult *result =
            vernonCompilerCompileMlir(compiler, tensorModule.data(), tensorModule.size(), target);
        ASSERT_TRUE(result);
        if (vernonCompileResultGetStatus(result) != VERNON_STATUS_OK) {
            const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
            std::fprintf(stderr, "target %d static compute Tensor compile failed: %.*s\n", static_cast<int>(target),
                         static_cast<int>(diagnostics.size), diagnostics.data);
        }
        ASSERT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_OK);
        ASSERT_EQ(vernonCompileResultGetArtifactCount(result), 1u);
        const std::string artifact = artifacts(result);
        if (target == VERNON_TARGET_VULKAN) {
            ASSERT_EQ(artifact.size() % sizeof(uint32_t), 0u);
            std::vector<uint32_t> words(artifact.size() / sizeof(uint32_t));
            std::memcpy(words.data(), artifact.data(), artifact.size());
            for (size_t cursor = 5; cursor < words.size();) {
                const uint16_t wordCount = static_cast<uint16_t>(words[cursor] >> 16);
                ASSERT_GT(wordCount, 0u);
                ASSERT_LE(cursor + wordCount, words.size());
                if (static_cast<uint16_t>(words[cursor]) == 23) {
                    ASSERT_EQ(wordCount, 4u);
                    EXPECT_GE(words[cursor + 3], 2u);
                    EXPECT_LE(words[cursor + 3], 4u);
                }
                cursor += wordCount;
            }
        }
        const VernonStringView reflection = vernonCompileResultGetReflection(result);
        const std::string_view reflected(reflection.data, reflection.size);
        EXPECT_NE(reflected.find("\"kind\":\"tensor_value\""), std::string_view::npos);
        EXPECT_NE(reflected.find("\"shape\":[2,2,2]"), std::string_view::npos);
        EXPECT_NE(reflected.find("\"array_strides\":[64,32,16]"), std::string_view::npos);
        EXPECT_NE(reflected.find("\"physical_size\":128"), std::string_view::npos);
        EXPECT_NE(reflected.find("\"proposed_storage_class\":\"StorageBuffer\""), std::string_view::npos);
        EXPECT_NE(reflected.find("\"vernon.binding\":0"), std::string_view::npos);
        EXPECT_NE(reflected.find("\"binding\":1"), std::string_view::npos);
        vernonCompileResultDestroy(result);
    }

    VernonCompileResult *cudaResult =
        vernonCompilerCompileMlir(compiler, tensorModule.data(), tensorModule.size(), VERNON_TARGET_CUDA);
    ASSERT_TRUE(cudaResult);
    EXPECT_NE(vernonCompileResultGetStatus(cudaResult), VERNON_STATUS_OK);
    const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(cudaResult);
    EXPECT_NE(std::string_view(diagnostics.data, diagnostics.size).find("descriptor-backed SPIR-V compute targets"),
              std::string_view::npos);
    vernonCompileResultDestroy(cudaResult);
    vernonCompilerDestroy(compiler);
}

TEST(CompilerGraphicsOutput, ReflectsAndValidatesCanonicalTensorAttributeLeaves) {
    constexpr std::string_view validModule = R"mlir(
module {
  func.func @tensor_attribute(
      %value: tensor<2x3xf32> {
        vernon.interface = "input",
        vernon.location = 0 : i64,
        vernon.dtype = "f32",
        vernon.source_name = "value"
      }) -> (tensor<4xf32> {
        vernon.interface = "output",
        vernon.builtin = "position"
      }) attributes {vernon.entry, vernon.stage = "vertex"} {
    %position = arith.constant dense<[0.0, 0.0, 0.0, 1.0]> : tensor<4xf32>
    return %position : tensor<4xf32>
  }
}
)mlir";
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    VernonCompileResult *result =
        vernonCompilerCompileMlir(compiler, validModule.data(), validModule.size(), VERNON_TARGET_VULKAN);
    ASSERT_TRUE(result);
    if (vernonCompileResultGetStatus(result) != VERNON_STATUS_OK) {
        const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
        std::fprintf(stderr, "Tensor attribute compile failed: %.*s\n", static_cast<int>(diagnostics.size),
                     diagnostics.data);
    }
    ASSERT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_OK);
    const VernonStringView reflection = vernonCompileResultGetReflection(result);
    const std::string_view reflected(reflection.data, reflection.size);
    EXPECT_NE(reflected.find("\"location_span\":2"), std::string_view::npos);
    EXPECT_NE(reflected.find("\"component_count\":4"), std::string_view::npos);
    EXPECT_NE(reflected.find("\"component_count\":2"), std::string_view::npos);
    EXPECT_NE(reflected.find("\"byte_offset\":16"), std::string_view::npos);
    vernonCompileResultDestroy(result);

    constexpr std::string_view overlapModule = R"mlir(
module {
  func.func @overlap(
      %value: tensor<2x3xf32> {
        vernon.interface = "input", vernon.location = 0 : i64, vernon.dtype = "f32"
      },
      %other: tensor<4xf32> {
        vernon.interface = "input", vernon.location = 1 : i64, vernon.dtype = "f32"
      }) -> (tensor<4xf32> {
        vernon.interface = "output", vernon.builtin = "position"
      }) attributes {vernon.entry, vernon.stage = "vertex"} {
    return %other : tensor<4xf32>
  }
}
)mlir";
    result = vernonCompilerCompileMlir(compiler, overlapModule.data(), overlapModule.size(), VERNON_TARGET_VULKAN);
    ASSERT_TRUE(result);
    EXPECT_NE(vernonCompileResultGetStatus(result), VERNON_STATUS_OK);
    const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
    EXPECT_NE(std::string_view(diagnostics.data, diagnostics.size).find("location:1"), std::string_view::npos);
    vernonCompileResultDestroy(result);
    vernonCompilerDestroy(compiler);
}
