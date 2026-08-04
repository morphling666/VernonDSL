#include "VernonCompiler.h"
#include "VernonVersions.h"
#include "compiler_target_test_utils.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::string_view module = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
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
        if (vernon::tests::unavailableDirectXTarget(compiler, target))
            continue;
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
        if (target == VERNON_TARGET_METAL) {
            const VernonStringView reflection = vernonCompileResultGetReflection(result);
            const auto reflected = nlohmann::json::parse(reflection.data, reflection.data + reflection.size);
            ASSERT_TRUE(reflected.contains("metal_resource_slots"));
            const auto &slots = reflected["metal_resource_slots"];
            auto hasSlot = [&](std::string_view entry, std::string_view kind, uint32_t set, uint32_t binding,
                               uint32_t argumentBuffer, uint32_t memberId) {
                return std::any_of(slots.begin(), slots.end(), [&](const nlohmann::json &slot) {
                    return slot.value("entry_point", "") == entry && slot.value("kind", "") == kind &&
                           slot.value("set", UINT32_MAX) == set && slot.value("binding", UINT32_MAX) == binding &&
                           slot.value("argument_buffer_index", UINT32_MAX) == argumentBuffer &&
                           slot.value("member_id", UINT32_MAX) == memberId &&
                           slot.value("direct_buffer_index", 0u) == UINT32_MAX && slot.value("count", 0u) == 1;
                });
            };
            EXPECT_TRUE(hasSlot("cube_map_fragment", "sampled_image", 0, 0, 0, 0)) << slots.dump();
            EXPECT_TRUE(hasSlot("cube_map_fragment", "sampler", 0, 0, 0, 1)) << slots.dump();
            EXPECT_NE(output.find("[[id(0)]]"), std::string::npos);
            EXPECT_NE(output.find("[[id(1)]]"), std::string::npos);
            EXPECT_NE(output.find("spvDescriptorSetBuffer0"), std::string::npos);
            EXPECT_NE(output.find("[[buffer(0)]]"), std::string::npos);
            EXPECT_EQ(output.find("[[texture(0)]]"), std::string::npos);
            EXPECT_EQ(output.find("[[sampler(0)]]"), std::string::npos);
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

TEST(CompilerGraphicsOutput, LowersEachMetalDescriptorSetToOneArgumentBuffer) {
    static constexpr char multiSetModule[] = R"(
module attributes {)" VERNON_MLIR_VERSION_ATTRIBUTES R"(} {
  func.func @multi_set_compute(
      %left: !vernon.tensor_view<f32, [1], "read_write", "device"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      },
      %right: !vernon.tensor_view<f32, [1], "read", "device"> {
        vernon.interface = "resource",
        vernon.set = 1 : i64,
        vernon.binding = 0 : i64
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    %zero = arith.constant 0 : index
    %left_value = "vernon.load"(%left, %zero) :
      (!vernon.tensor_view<f32, [1], "read_write", "device">, index) -> f32
    %right_value = "vernon.load"(%right, %zero) :
      (!vernon.tensor_view<f32, [1], "read", "device">, index) -> f32
    %sum = arith.addf %left_value, %right_value : f32
    "vernon.store"(%sum, %left, %zero) :
      (f32, !vernon.tensor_view<f32, [1], "read_write", "device">, index) -> ()
    return
  }
}
)";
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_NE(compiler, nullptr);
    VernonCompileResult *result =
        vernonCompilerCompileMlir(compiler, multiSetModule, sizeof(multiSetModule) - 1, VERNON_TARGET_METAL);
    ASSERT_NE(result, nullptr);
    if (vernonCompileResultGetStatus(result) != VERNON_STATUS_OK) {
        const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
        std::fprintf(stderr, "multi-set Metal compile failed: %.*s\n", static_cast<int>(diagnostics.size),
                     diagnostics.data);
    }
    ASSERT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_OK);
    const std::string output = artifacts(result);
    EXPECT_NE(output.find("spvDescriptorSetBuffer0"), std::string::npos);
    EXPECT_NE(output.find("spvDescriptorSetBuffer1"), std::string::npos);
    EXPECT_NE(output.find("[[buffer(0)]]"), std::string::npos);
    EXPECT_NE(output.find("[[buffer(1)]]"), std::string::npos);
    const VernonStringView reflection = vernonCompileResultGetReflection(result);
    const auto reflected = nlohmann::json::parse(reflection.data, reflection.data + reflection.size);
    const auto &slots = reflected.at("metal_resource_slots");
    for (uint32_t set = 0; set < 2; ++set) {
        EXPECT_TRUE(std::any_of(slots.begin(), slots.end(), [&](const nlohmann::json &slot) {
            return slot.value("entry_point", "") == "multi_set_compute" && slot.value("kind", "") == "storage_buffer" &&
                   slot.value("set", UINT32_MAX) == set && slot.value("argument_buffer_index", UINT32_MAX) == set &&
                   slot.value("member_id", UINT32_MAX) == 0;
        })) << slots.dump();
    }
    vernonCompileResultDestroy(result);
    vernonCompilerDestroy(compiler);
}

TEST(CompilerGraphicsOutput, LowersSamplingBuiltinsTextureSizeAndMath) {
    constexpr std::string_view samplingModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
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
    %floor = math.floor %abs : f32
    %acos = math.acos %x : f32
    %atan2 = math.atan2 %floor, %acos : f32
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
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
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
    EXPECT_NE(reflected.find("\"profile\":\"vulkan_std140_uniform_buffer\""), std::string_view::npos);
    EXPECT_NE(reflected.find("\"byte_strides\":[240,80,16]"), std::string_view::npos);
    EXPECT_NE(reflected.find("\"size\":480"), std::string_view::npos);
    EXPECT_NE(reflected.find("\"alignment\":16"), std::string_view::npos);
    EXPECT_NE(reflected.find("\"value_transport\":\"uniform_buffer\""), std::string_view::npos);
    EXPECT_NE(reflected.find("\"value_transport\":\"push_constant\""), std::string_view::npos);
    EXPECT_NE(reflected.find("\"vernon.binding\":0"), std::string_view::npos);
    EXPECT_NE(reflected.find("\"vernon.binding\":1"), std::string_view::npos);
    EXPECT_NE(reflected.find("\"shape\":[3,4]"), std::string_view::npos);

    vernonCompileResultDestroy(result);
    for (VernonTarget target : {VERNON_TARGET_OPENGL, VERNON_TARGET_DIRECTX}) {
        if (vernon::tests::unavailableDirectXTarget(compiler, target))
            continue;
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

TEST(CompilerGraphicsOutput, PreservesStd140ScalarArrayStrideInSpirvInterfaces) {
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    for (uint32_t tailWidth : {1u, 2u, 3u}) {
        const std::string tensorModule = std::string(R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @std140_tail_width(
      %tail: tensor<2x2x)mlir") + std::to_string(tailWidth) +
                                         R"mlir(xf32> {
        vernon.interface = "uniform",
        vernon.source_name = "tail",
        vernon.dtype = "f32"
      }) -> (f32 {
        vernon.interface = "output",
        vernon.location = 0 : i64
      }) attributes {vernon.entry, vernon.stage = "fragment"} {
    %one = arith.constant 1 : index
    %tail_index = arith.constant )mlir" + std::to_string(tailWidth - 1) +
                                         R"mlir( : index
    %value = tensor.extract %tail[%one, %one, %tail_index] : tensor<2x2x)mlir" +
                                         std::to_string(tailWidth) + R"mlir(xf32>
    return %value : f32
  }
}
)mlir";

        SCOPED_TRACE("tail width " + std::to_string(tailWidth));
        VernonCompileResult *result =
            vernonCompilerCompileMlir(compiler, tensorModule.data(), tensorModule.size(), VERNON_TARGET_VULKAN);
        ASSERT_TRUE(result);
        if (vernonCompileResultGetStatus(result) != VERNON_STATUS_OK) {
            const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
            std::fprintf(stderr, "std140 tail-width Vulkan compile failed: %.*s\n", static_cast<int>(diagnostics.size),
                         diagnostics.data);
        }
        ASSERT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_OK);
        const std::string spirv = artifacts(result);
        ASSERT_EQ(spirv.size() % sizeof(uint32_t), 0u);
        std::vector<uint32_t> words(spirv.size() / sizeof(uint32_t));
        std::memcpy(words.data(), spirv.data(), spirv.size());
        bool hasFourComponentCarrier = false;
        std::vector<uint32_t> arrayStrides;
        for (size_t cursor = 5; cursor < words.size();) {
            const uint16_t wordCount = static_cast<uint16_t>(words[cursor] >> 16);
            ASSERT_GT(wordCount, 0u);
            ASSERT_LE(cursor + wordCount, words.size());
            const uint16_t opcode = static_cast<uint16_t>(words[cursor]);
            if (opcode == 23 && wordCount == 4 && words[cursor + 3] == 4)
                hasFourComponentCarrier = true;
            if (opcode == 71 && wordCount == 4 && words[cursor + 2] == 6)
                arrayStrides.push_back(words[cursor + 3]);
            cursor += wordCount;
        }
        EXPECT_TRUE(hasFourComponentCarrier);
        for (uint32_t stride : {16u, tailWidth * 16u, tailWidth * 32u})
            EXPECT_NE(std::find(arrayStrides.begin(), arrayStrides.end(), stride), arrayStrides.end());

        const VernonStringView reflected = vernonCompileResultGetReflection(result);
        const nlohmann::json root = nlohmann::json::parse(reflected.data, reflected.data + reflected.size);
        const nlohmann::json &argument = root.at("entries").at(0).at("arguments").at(0);
        const nlohmann::json &layout = argument.at("physical_layouts").at("vulkan_std140_uniform_buffer");
        EXPECT_EQ(argument.at("shape"), nlohmann::json::array({2, 2, tailWidth}));
        EXPECT_EQ(layout.at("root").at("byte_strides"), nlohmann::json::array({tailWidth * 32u, tailWidth * 16u, 16u}));
        EXPECT_EQ(layout.at("root").at("size"), tailWidth * 64u);
        vernonCompileResultDestroy(result);
    }
    vernonCompilerDestroy(compiler);
}

TEST(CompilerGraphicsOutput, LowersStaticTensorByValueComputeAbi) {
    constexpr std::string_view tensorModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @static_tensor_compute(
      %value: tensor<2x2x2xf32> {
        vernon.interface = "input",
        vernon.location = 0 : i64,
        vernon.dtype = "f32"
      },
      %output: !vernon.tensor_view<f32, [1], "write", "device"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
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
    "vernon.store"(%value_at_index, %output, %zero) :
      (f32, !vernon.tensor_view<f32, [1], "write", "device">, index) -> ()
    return
  }
}
)mlir";

    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    for (VernonTarget target :
         {VERNON_TARGET_VULKAN, VERNON_TARGET_OPENGL, VERNON_TARGET_DIRECTX, VERNON_TARGET_CUDA}) {
        if (vernon::tests::unavailableDirectXTarget(compiler, target))
            continue;
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
        EXPECT_EQ(reflected.find("\"profile\":\"vulkan_std430_storage_buffer\"") != std::string_view::npos,
                  target != VERNON_TARGET_CUDA);
        EXPECT_EQ(reflected.find("\"profile\":\"cuda_kernel_parameter\"") != std::string_view::npos,
                  target == VERNON_TARGET_CUDA);
        EXPECT_EQ(reflected.find("\"profile\":\"host_value\""), std::string_view::npos);
        EXPECT_EQ(reflected.find("\"profile\":\"metal_constant_buffer\""), std::string_view::npos);
        EXPECT_EQ(reflected.find("\"profile\":\"directx_constant_buffer\""), std::string_view::npos);
        EXPECT_EQ(reflected.find("\"profile\":\"opengl_native_uniform\""), std::string_view::npos);
        EXPECT_NE(reflected.find("\"byte_strides\":[16,8,4]"), std::string_view::npos);
        EXPECT_NE(reflected.find("\"size\":32"), std::string_view::npos);
        EXPECT_NE(reflected.find("\"value_transport\":\"storage_buffer\""), std::string_view::npos);
        EXPECT_NE(reflected.find("\"vernon.binding\":0"), std::string_view::npos);
        EXPECT_NE(reflected.find("\"binding\":1"), std::string_view::npos);
        EXPECT_NE(reflected.find("\"tensor_view_descriptor\":"), std::string_view::npos);
        EXPECT_NE(reflected.find("\"offset_binding\":2"), std::string_view::npos);
        EXPECT_NE(reflected.find("\"extent_bindings\":[3]"), std::string_view::npos);
        EXPECT_NE(reflected.find("\"stride_bindings\":[4]"), std::string_view::npos);
        vernonCompileResultDestroy(result);
    }
    vernonCompilerDestroy(compiler);
}

TEST(CompilerGraphicsOutput, ReflectsAndValidatesCanonicalTensorAttributeLeaves) {
    constexpr std::string_view validModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
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
    EXPECT_NE(reflected.find("\"byte_strides\":[4,16]"), std::string_view::npos);
    EXPECT_NE(reflected.find("\"size\":48"), std::string_view::npos);
    vernonCompileResultDestroy(result);

    constexpr std::string_view overlapModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
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

TEST(CompilerGraphicsOutput, LowersAndReflectsMixedDtypeStructAttributes) {
    constexpr std::string_view aggregateModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  "vernon.struct"() {
    sym_name = "Vertex",
    fields = ["position:tensor<3xf32>", "object_id:i32", "uv:tensor<2xf32>"],
    abi_leaf_dtypes = ["f32", "u32", "f32"]
  } : () -> ()
  func.func @aggregate_vertex(
      %value: !vernon.struct<"Vertex"> {
        vernon.interface = "input",
        vernon.location = 0 : i64,
        vernon.source_name = "value"
      }) -> (tensor<4xf32> {
        vernon.interface = "output",
        vernon.builtin = "position"
      }) attributes {vernon.entry, vernon.stage = "vertex"} {
    %position = "vernon.struct_get"(%value) {
      field = "position", index = 0 : i64
    } : (!vernon.struct<"Vertex">) -> tensor<3xf32>
    %one = arith.constant 1.0 : f32
    %result = "vernon.intrinsic"(%position, %one) {
      name = "construct"
    } : (tensor<3xf32>, f32) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
)mlir";

    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    VernonCompileResult *result =
        vernonCompilerCompileMlir(compiler, aggregateModule.data(), aggregateModule.size(), VERNON_TARGET_VULKAN);
    ASSERT_TRUE(result);
    if (vernonCompileResultGetStatus(result) != VERNON_STATUS_OK) {
        const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
        std::fprintf(stderr, "aggregate attribute compile failed: %.*s\n", static_cast<int>(diagnostics.size),
                     diagnostics.data);
    }
    ASSERT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_OK);
    const VernonStringView reflection = vernonCompileResultGetReflection(result);
    const std::string_view reflected(reflection.data, reflection.size);
    EXPECT_NE(reflected.find("\"compiler_contract_version\":" + std::to_string(VERNON_COMPILER_CONTRACT_VERSION)),
              std::string_view::npos);
    EXPECT_NE(reflected.find("\"pipeline_version\":" + std::to_string(VERNON_PIPELINE_VERSION)),
              std::string_view::npos);
    EXPECT_NE(reflected.find("\"struct_name\":\"Vertex\""), std::string_view::npos);
    EXPECT_NE(reflected.find("\"location_span\":3"), std::string_view::npos);
    EXPECT_NE(reflected.find("\"dtype\":\"u32\""), std::string_view::npos);
    EXPECT_NE(reflected.find("\"path\":[\"object_id\"]"), std::string_view::npos);
    EXPECT_NE(reflected.find("\"shape\":[3]"), std::string_view::npos);
    vernonCompileResultDestroy(result);
    vernonCompilerDestroy(compiler);
}

TEST(CompilerGraphicsOutput, LowersStructCarryingStructuredLoopToSpirv) {
    constexpr std::string_view aggregateLoopModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  "vernon.struct"() {
    sym_name = "State",
    fields = ["value:f32"],
    abi_leaf_dtypes = ["f32"]
  } : () -> ()
  func.func @struct_loop_fragment() -> (f32 {
      vernon.interface = "output", vernon.location = 0 : i64
    }) attributes {vernon.entry, vernon.stage = "fragment"} {
    %zero = arith.constant 0.0 : f32
    %initial = "vernon.struct_create"(%zero) {
      type_name = "State"
    } : (f32) -> !vernon.struct<"State">
    %zero_index = arith.constant 0 : i32
    %result, %iterations = scf.while (%state = %initial, %index = %zero_index)
        : (!vernon.struct<"State">, i32) -> (!vernon.struct<"State">, i32) {
      %limit = arith.constant 2 : i32
      %continue = arith.cmpi slt, %index, %limit : i32
      scf.condition(%continue) %state, %index : !vernon.struct<"State">, i32
    } do {
    ^bb0(%state: !vernon.struct<"State">, %index: i32):
      %value = "vernon.struct_get"(%state) {
        field = "value", index = 0 : i64
      } : (!vernon.struct<"State">) -> f32
      %one = arith.constant 1.0 : f32
      %next_value = arith.addf %value, %one : f32
      %next_state = "vernon.struct_create"(%next_value) {
        type_name = "State"
      } : (f32) -> !vernon.struct<"State">
      %one_index = arith.constant 1 : i32
      %next_index = arith.addi %index, %one_index : i32
      scf.yield %next_state, %next_index : !vernon.struct<"State">, i32
    }
    %value = "vernon.struct_get"(%result) {
      field = "value", index = 0 : i64
    } : (!vernon.struct<"State">) -> f32
    return %value : f32
  }
}
)mlir";
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    VernonCompileResult *result = vernonCompilerCompileMlir(compiler, aggregateLoopModule.data(),
                                                            aggregateLoopModule.size(), VERNON_TARGET_VULKAN);
    ASSERT_TRUE(result);
    if (vernonCompileResultGetStatus(result) != VERNON_STATUS_OK) {
        const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
        std::fprintf(stderr, "struct loop compile failed: %.*s\n", static_cast<int>(diagnostics.size),
                     diagnostics.data);
    }
    EXPECT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_OK);
    vernonCompileResultDestroy(result);
    vernonCompilerDestroy(compiler);
}

TEST(CompilerGraphicsOutput, ReflectsNestedLogicalAndPhysicalAbiRoutes) {
    constexpr std::string_view nestedModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  "vernon.struct"() {
    sym_name = "Payload",
    fields = ["id:i32", "weights:tensor<2xf32>"],
    abi_leaf_dtypes = ["i32", "f32"]
  } : () -> ()
  "vernon.struct"() {
    sym_name = "Nested",
    fields = ["payload:!vernon.struct<\"Payload\">", "pair:tuple<f32, i32>"],
    abi_leaf_dtypes = ["i32", "f32", "f32", "u32"]
  } : () -> ()
  func.func @nested(
      %values: !vernon.tensor_view<!vernon.struct<"Nested">, [2], "read", "device"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    return
  }
}
)mlir";

    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    VernonCompileResult *result = vernonCompilerValidateMlir(compiler, nestedModule.data(), nestedModule.size());
    ASSERT_TRUE(result);
    ASSERT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_OK);
    const VernonStringView reflected = vernonCompileResultGetReflection(result);
    const nlohmann::json root = nlohmann::json::parse(reflected.data, reflected.data + reflected.size);
    const nlohmann::json &argument = root.at("entries").at(0).at("arguments").at(0);
    EXPECT_FALSE(root.contains("backend_abi_routes"));
    EXPECT_EQ(argument.at("physical_layouts").at("host_value").at("kind"), "resource_binding");
    EXPECT_EQ(argument.at("physical_layouts").at("host_value").at("resource_kind"), "tensor_view_descriptor");
    EXPECT_EQ(argument.at("physical_layouts").at("host_value").at("size"), 32);
    EXPECT_EQ(argument.at("physical_layouts").at("cuda_kernel_parameter").at("resource_kind"),
              "strided_memref_storage_leaves");
    EXPECT_EQ(argument.at("physical_layouts").at("vulkan_std430_storage_buffer").at("resource_kind"),
              "descriptor_storage_leaves");
    EXPECT_EQ(argument.at("physical_layouts").at("directx_constant_buffer").at("resource_kind"),
              "descriptor_storage_leaves");
    EXPECT_FALSE(argument.at("physical_layouts").at("cuda_kernel_parameter").contains("element_layout_hash"));
    EXPECT_EQ(argument.at("element_layout").at("byte_size"), 20);
    EXPECT_EQ(argument.at("element_layout").at("alignment"), 4);
    EXPECT_EQ(argument.at("element_layout").at("leaves").at(3).at("dtype"), "u32");
    EXPECT_EQ(argument.at("element_layout").at("leaves").at(3).at("byte_offset"), 16);
    EXPECT_EQ(argument.at("storage_leaves").at(0).at("byte_offset"), 0);
    EXPECT_EQ(argument.at("storage_leaves").at(1).at("byte_offset"), 4);
    EXPECT_EQ(argument.at("storage_leaves").at(2).at("byte_offset"), 12);
    EXPECT_EQ(argument.at("storage_leaves").at(3).at("byte_offset"), 16);
    vernonCompileResultDestroy(result);
    vernonCompilerDestroy(compiler);
}
