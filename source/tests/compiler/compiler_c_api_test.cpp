#include "VernonCompiler.h"
#include "VernonCpuWorkgroupABI.h"
#include "VernonVersions.h"
#include "compiler_artifacts.h"
#include "compiler_target_test_utils.h"

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <stdio.h>
#include <string.h>
#include <string>
#include <string_view>
#include <vector>

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

static VernonStatus invoke_cpu_range(VernonCpuEntryPoint entry, const VernonCpuInvocation &lane,
                                     size_t localLinear = 0) {
    VernonCpuRangeV1 range{};
    range.struct_size = sizeof(range);
    range.arguments = lane.arguments;
    range.arguments_size = lane.arguments_size;
    range.results = lane.results;
    range.results_size = lane.results_size;
    range.textures = lane.textures;
    range.grid[0] = range.grid[1] = range.grid[2] = 1;
    range.workgroup[0] = static_cast<uint32_t>(localLinear + 1);
    range.workgroup[1] = range.workgroup[2] = 1;
    range.lane_begin = localLinear;
    range.lane_end = localLinear + 1;
    const VernonCpuInvocation invocation{&range, VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1, nullptr, 0, nullptr};
    return entry(&invocation);
}

TEST(CompilerArtifacts, RejectsInvalidReflectionBeforePublishingArtifacts) {
    std::string reflection = "not-json";
    std::string diagnostics;
    const std::vector<vernon::compiler::Artifact> artifacts;
    EXPECT_FALSE(vernon::compiler::addArtifactTable(reflection, diagnostics, artifacts,
                                                    vernon::compiler::MetalCompileOptions{}));
    EXPECT_NE(diagnostics.find("not valid JSON"), std::string::npos);
}

static int views_equal(VernonStringView left, VernonStringView right) {
    return left.size == right.size && (left.size == 0 || memcmp(left.data, right.data, left.size) == 0);
}

static void sample_texture(void *user_data, uintptr_t texture, float u, float v, float out_rgba[4]) {
    const float bias = *(const float *)user_data;
    out_rgba[0] = u;
    out_rgba[1] = v;
    out_rgba[2] = (float)texture;
    out_rgba[3] = bias;
}

TEST(CompilerCApi, ReflectsHiddenCpuTapeAllocatorBuiltin) {
    static const char module[] = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @forward(
      %allocator: index {
        vernon.interface = "input",
        vernon.builtin = "ad_tape_allocator"
      }) -> (
      f32 {
        vernon.interface = "output",
        vernon.location = 0 : i64
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    %zero = arith.constant 0.0 : f32
    return %zero : f32
  }
}
)mlir";
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_NE(compiler, nullptr);
    VernonCompileResult *compiled = vernonCompilerCompileMlir(compiler, module, strlen(module), VERNON_TARGET_CPU);
    ASSERT_NE(compiled, nullptr);
    ASSERT_EQ(vernonCompileResultGetStatus(compiled), VERNON_STATUS_OK) << std::string(
        vernonCompileResultGetDiagnostics(compiled).data, vernonCompileResultGetDiagnostics(compiled).size);
    const VernonStringView reflected = vernonCompileResultGetReflection(compiled);
    const nlohmann::json reflection = nlohmann::json::parse(reflected.data, reflected.data + reflected.size);
    const nlohmann::json &entry = reflection.at("entries").at(0);
    const nlohmann::json &allocator = entry.at("arguments").at(0);
    EXPECT_EQ(allocator.at("kind"), "builtin");
    EXPECT_EQ(allocator.at("builtin"), "ad_tape_allocator");
    EXPECT_EQ(allocator.at("physical_layouts").at("host_value").at("root").at("size"), sizeof(void *));
    EXPECT_EQ(entry.at("physical_layouts").at("host_value").at("packed_arguments_size"), sizeof(void *));
    EXPECT_EQ(entry.at("results").size(), 1u);
    EXPECT_EQ(entry.at("results").at(0).at("physical_layouts").at("host_value").at("root").at("size"), sizeof(float));
    EXPECT_EQ(entry.at("physical_layouts").at("host_value").at("packed_results_size"), sizeof(float));

    vernonCompileResultDestroy(compiled);
    vernonCompilerDestroy(compiler);
}

TEST(CompilerCApi, RejectsInvalidHiddenCpuTapeAllocatorAbi) {
    static const char module[] = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @forward(
      %allocator: f32 {
        vernon.interface = "input",
        vernon.builtin = "ad_tape_allocator"
      }) -> (
      f32 {
        vernon.interface = "output",
        vernon.location = 0 : i64
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    return %allocator : f32
  }
}
)mlir";
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_NE(compiler, nullptr);
    VernonCompileResult *compiled = vernonCompilerCompileMlir(compiler, module, strlen(module), VERNON_TARGET_CPU);
    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(vernonCompileResultGetStatus(compiled), VERNON_STATUS_VERIFICATION_ERROR);
    const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(compiled);
    EXPECT_NE(std::string(diagnostics.data, diagnostics.size).find("ad_tape_allocator must have index type"),
              std::string::npos);

    vernonCompileResultDestroy(compiled);
    vernonCompilerDestroy(compiler);
}

TEST(CompilerCApi, CpuCanonicalAbiRoundTripsNestedVectorThreeAggregateTensor) {
    static const char module[] = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  "vernon.struct"() {
    abi_leaf_dtypes = ["f32", "i32"],
    fields = ["direction:tensor<3xf32>", "id:i32"],
    sym_name = "Payload"
  } : () -> ()
  func.func @roundtrip(
      %value: !vernon.tensor<!vernon.struct<"Payload">, [2]> {
        vernon.abi_leaf_dtypes = ["f32", "i32", "f32", "i32"],
        vernon.element_abi_leaf_dtypes = ["f32", "i32"],
        vernon.interface = "input",
        vernon.location = 0 : i64
      }) -> (!vernon.tensor<!vernon.struct<"Payload">, [2]> {
        vernon.abi_leaf_dtypes = ["f32", "i32", "f32", "i32"],
        vernon.element_abi_leaf_dtypes = ["f32", "i32"],
        vernon.interface = "output",
        vernon.location = 0 : i64
      }) attributes {vernon.entry, vernon.stage = "fragment"} {
    return %value : !vernon.tensor<!vernon.struct<"Payload">, [2]>
  }
}
)mlir";
    struct Payload {
        float direction[3];
        int32_t id;
    };
    static_assert(sizeof(Payload) == 16);

    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    VernonCompileResult *compiled = vernonCompilerCompileMlir(compiler, module, strlen(module), VERNON_TARGET_CPU);
    ASSERT_TRUE(compiled);
    if (vernonCompileResultGetStatus(compiled) != VERNON_STATUS_OK) {
        VernonStringView diagnostics = vernonCompileResultGetDiagnostics(compiled);
        fprintf(stderr, "%.*s\n", static_cast<int>(diagnostics.size), diagnostics.data);
    }
    ASSERT_EQ(vernonCompileResultGetStatus(compiled), VERNON_STATUS_OK);
    const VernonStringView reflected = vernonCompileResultGetReflection(compiled);
    const nlohmann::json reflection = nlohmann::json::parse(reflected.data, reflected.data + reflected.size);
    const nlohmann::json &entry = reflection.at("entries").at(0);
    const nlohmann::json &entryHost = entry.at("physical_layouts").at("host_value");
    const nlohmann::json &argumentHost = entry.at("arguments").at(0).at("physical_layouts").at("host_value");
    const nlohmann::json &resultHost = entry.at("results").at(0).at("physical_layouts").at("host_value");
    EXPECT_EQ(entryHost.at("packed_arguments_size"), 32);
    EXPECT_EQ(entryHost.at("packed_results_size"), 32);
    EXPECT_EQ(argumentHost.at("kind"), "cpu_call");
    EXPECT_EQ(resultHost.at("kind"), "cpu_call");
    EXPECT_TRUE(argumentHost.contains("canonical_layout_hash"));
    EXPECT_TRUE(resultHost.contains("canonical_layout_hash"));
    EXPECT_TRUE(entry.at("arguments").at(0).contains("value_layout"));
    EXPECT_TRUE(entry.at("arguments").at(0).contains("element_layout"));

    VernonCpuEntryPoint roundtrip = vernonCompileResultGetCpuEntry(compiled, "roundtrip", 9);
    ASSERT_TRUE(roundtrip);
    const Payload input[2] = {{{1.0f, 2.0f, 3.0f}, 7}, {{4.0f, 5.0f, 6.0f}, 11}};
    Payload output[2]{};
    VernonCpuInvocation invocation = {input, sizeof(input), output, sizeof(output), nullptr};
    ASSERT_EQ(invoke_cpu_range(roundtrip, invocation), VERNON_STATUS_OK);
    EXPECT_EQ(memcmp(input, output, sizeof(input)), 0);

    for (std::string_view triple : {"x86_64-pc-windows-msvc", "arm64-apple-ios17.0"}) {
        VernonCompileOptions options{};
        options.struct_size = sizeof(options);
        options.target = VERNON_TARGET_CPU;
        options.as.cpu.triple = {triple.data(), triple.size()};
        VernonCompileResult *cross = vernonCompilerCompileMlirWithOptions(compiler, module, strlen(module), &options);
        ASSERT_TRUE(cross);
        EXPECT_EQ(vernonCompileResultGetStatus(cross), VERNON_STATUS_OK);
        vernonCompileResultDestroy(cross);
    }

    vernonCompileResultDestroy(compiled);
    vernonCompilerDestroy(compiler);
}

TEST(CompilerCApi, CpuHalfConversionsAreSelfContainedAndIeeeCompliant) {
    static const char module[] = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @extend_f32(
      %value: f16 {vernon.interface = "input", vernon.location = 0 : i64}) ->
      (f32 {vernon.interface = "output", vernon.location = 0 : i64})
      attributes {vernon.entry, vernon.stage = "compute",
                  vernon.workgroup_size = array<i32: 1, 1, 1>} {
    %result = arith.extf %value : f16 to f32
    return %result : f32
  }
  func.func @truncate_f32(
      %value: f32 {vernon.interface = "input", vernon.location = 0 : i64}) ->
      (f16 {vernon.interface = "output", vernon.location = 0 : i64})
      attributes {vernon.entry, vernon.stage = "compute",
                  vernon.workgroup_size = array<i32: 1, 1, 1>} {
    %result = arith.truncf %value : f32 to f16
    return %result : f16
  }
  func.func @extend_f64(
      %value: f16 {vernon.interface = "input", vernon.location = 0 : i64}) ->
      (f64 {vernon.interface = "output", vernon.location = 0 : i64})
      attributes {vernon.entry, vernon.stage = "compute",
                  vernon.workgroup_size = array<i32: 1, 1, 1>} {
    %result = arith.extf %value : f16 to f64
    return %result : f64
  }
  func.func @truncate_f64(
      %value: f64 {vernon.interface = "input", vernon.location = 0 : i64}) ->
      (f16 {vernon.interface = "output", vernon.location = 0 : i64})
      attributes {vernon.entry, vernon.stage = "compute",
                  vernon.workgroup_size = array<i32: 1, 1, 1>} {
    %result = arith.truncf %value : f64 to f16
    return %result : f16
  }
  func.func private @extend_v2(%value: vector<2xf16>) -> vector<2xf32> {
    %result = arith.extf %value : vector<2xf16> to vector<2xf32>
    return %result : vector<2xf32>
  }
  func.func @square_f16(
      %value: f16 {vernon.interface = "input", vernon.location = 0 : i64}) ->
      (f16 {vernon.interface = "output", vernon.location = 0 : i64})
      attributes {vernon.entry, vernon.stage = "compute",
                  vernon.workgroup_size = array<i32: 1, 1, 1>} {
    %result = arith.mulf %value, %value : f16
    return %result : f16
  }
}
)mlir";
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    VernonCompileResult *compiled = vernonCompilerCompileMlir(compiler, module, strlen(module), VERNON_TARGET_CPU);
    ASSERT_TRUE(compiled);
    ASSERT_EQ(vernonCompileResultGetStatus(compiled), VERNON_STATUS_OK) << std::string(
        vernonCompileResultGetDiagnostics(compiled).data, vernonCompileResultGetDiagnostics(compiled).size);

    VernonCpuEntryPoint extendF32 = vernonCompileResultGetCpuEntry(compiled, "extend_f32", 10);
    VernonCpuEntryPoint truncateF32 = vernonCompileResultGetCpuEntry(compiled, "truncate_f32", 12);
    VernonCpuEntryPoint extendF64 = vernonCompileResultGetCpuEntry(compiled, "extend_f64", 10);
    VernonCpuEntryPoint truncateF64 = vernonCompileResultGetCpuEntry(compiled, "truncate_f64", 12);
    VernonCpuEntryPoint squareF16 = vernonCompileResultGetCpuEntry(compiled, "square_f16", 10);
    ASSERT_TRUE(extendF32);
    ASSERT_TRUE(truncateF32);
    ASSERT_TRUE(extendF64);
    ASSERT_TRUE(truncateF64);
    ASSERT_TRUE(squareF16);

    const auto extend32 = [&](uint16_t input) {
        uint32_t output = 0;
        VernonCpuInvocation invocation{&input, sizeof(input), &output, sizeof(output), nullptr};
        EXPECT_EQ(invoke_cpu_range(extendF32, invocation), VERNON_STATUS_OK);
        return output;
    };
    EXPECT_EQ(extend32(0x0000), 0x00000000u);
    EXPECT_EQ(extend32(0x8000), 0x80000000u);
    EXPECT_EQ(extend32(0x0001), 0x33800000u);
    EXPECT_EQ(extend32(0x03ff), 0x387fc000u);
    EXPECT_EQ(extend32(0x83ff), 0xb87fc000u);
    EXPECT_EQ(extend32(0x0400), 0x38800000u);
    EXPECT_EQ(extend32(0x3c00), 0x3f800000u);
    EXPECT_EQ(extend32(0x7c00), 0x7f800000u);
    EXPECT_EQ(extend32(0xfc00), 0xff800000u);
    EXPECT_EQ(extend32(0x7e00), 0x7fc00000u);

    const auto truncate32 = [&](uint32_t input) {
        uint16_t output = 0;
        VernonCpuInvocation invocation{&input, sizeof(input), &output, sizeof(output), nullptr};
        EXPECT_EQ(invoke_cpu_range(truncateF32, invocation), VERNON_STATUS_OK);
        return output;
    };
    EXPECT_EQ(truncate32(0x00000000), 0x0000);
    EXPECT_EQ(truncate32(0x80000000), 0x8000);
    EXPECT_EQ(truncate32(0x33000000), 0x0000);
    EXPECT_EQ(truncate32(0x33000001), 0x0001);
    EXPECT_EQ(truncate32(0x33800000), 0x0001);
    EXPECT_EQ(truncate32(0x387fc000), 0x03ff);
    EXPECT_EQ(truncate32(0x387fe000), 0x0400);
    EXPECT_EQ(truncate32(0x3f800000), 0x3c00);
    EXPECT_EQ(truncate32(0x477fe000), 0x7bff);
    EXPECT_EQ(truncate32(0x477ff000), 0x7c00);
    EXPECT_EQ(truncate32(0x7f800000), 0x7c00);
    EXPECT_EQ(truncate32(0xff800000), 0xfc00);
    EXPECT_EQ(truncate32(0x7fc00000), 0x7e00);
    EXPECT_EQ(truncate32(0x3f801000), 0x3c00);
    EXPECT_EQ(truncate32(0x3f803000), 0x3c02);

    uint16_t halfOne = 0x3c00;
    uint64_t doubleOne = 0;
    VernonCpuInvocation extendDoubleInvocation{&halfOne, sizeof(halfOne), &doubleOne, sizeof(doubleOne), nullptr};
    EXPECT_EQ(invoke_cpu_range(extendF64, extendDoubleInvocation), VERNON_STATUS_OK);
    EXPECT_EQ(doubleOne, 0x3ff0000000000000ull);
    uint64_t doubleSmallestHalf = 0x3e70000000000000ull;
    uint16_t smallestHalf = 0;
    VernonCpuInvocation truncateDoubleInvocation{&doubleSmallestHalf, sizeof(doubleSmallestHalf), &smallestHalf,
                                                 sizeof(smallestHalf), nullptr};
    EXPECT_EQ(invoke_cpu_range(truncateF64, truncateDoubleInvocation), VERNON_STATUS_OK);
    EXPECT_EQ(smallestHalf, 0x0001);
    uint16_t halfOnePointFive = 0x3e00;
    uint16_t halfSquare = 0;
    VernonCpuInvocation squareInvocation{&halfOnePointFive, sizeof(halfOnePointFive), &halfSquare, sizeof(halfSquare),
                                         nullptr};
    EXPECT_EQ(invoke_cpu_range(squareF16, squareInvocation), VERNON_STATUS_OK);
    EXPECT_EQ(halfSquare, 0x4080);

    VernonCompileOptions options{};
    options.struct_size = sizeof(options);
    options.target = VERNON_TARGET_CPU;
    constexpr std::string_view linuxTriple = "x86_64-unknown-linux-gnu";
    options.as.cpu.triple = {linuxTriple.data(), linuxTriple.size()};
    VernonCompileResult *linux = vernonCompilerCompileMlirWithOptions(compiler, module, strlen(module), &options);
    ASSERT_TRUE(linux);
    ASSERT_EQ(vernonCompileResultGetStatus(linux), VERNON_STATUS_OK)
        << std::string(vernonCompileResultGetDiagnostics(linux).data, vernonCompileResultGetDiagnostics(linux).size);
    VernonStringView object = vernonCompileResultGetArtifactData(linux, 0);
    EXPECT_FALSE(view_contains(object, "__extendhfsf2"));
    EXPECT_FALSE(view_contains(object, "__extendhfdf2"));
    EXPECT_FALSE(view_contains(object, "__truncsfhf2"));
    EXPECT_FALSE(view_contains(object, "__truncdfhf2"));
    EXPECT_FALSE(view_contains(object, "__mulhf3"));
    EXPECT_TRUE(view_contains(object, "__vernon_cpu_f16_to_f32_bits"));

    constexpr std::string_view windowsTriple = "x86_64-pc-windows-msvc";
    options.as.cpu.triple = {windowsTriple.data(), windowsTriple.size()};
    VernonCompileResult *windows = vernonCompilerCompileMlirWithOptions(compiler, module, strlen(module), &options);
    ASSERT_TRUE(windows);
    ASSERT_EQ(vernonCompileResultGetStatus(windows), VERNON_STATUS_OK) << std::string(
        vernonCompileResultGetDiagnostics(windows).data, vernonCompileResultGetDiagnostics(windows).size);
    VernonStringView windowsObject = vernonCompileResultGetArtifactData(windows, 0);
    EXPECT_FALSE(view_contains(windowsObject, "__extendhfsf2"));
    EXPECT_FALSE(view_contains(windowsObject, "__extendhfdf2"));
    EXPECT_FALSE(view_contains(windowsObject, "__truncsfhf2"));
    EXPECT_FALSE(view_contains(windowsObject, "__truncdfhf2"));
    EXPECT_FALSE(view_contains(windowsObject, "__mulhf3"));

    constexpr std::string_view appleTriple = "arm64-apple-macosx14.0";
    options.as.cpu.triple = {appleTriple.data(), appleTriple.size()};
    VernonCompileResult *apple = vernonCompilerCompileMlirWithOptions(compiler, module, strlen(module), &options);
    ASSERT_TRUE(apple);
    ASSERT_EQ(vernonCompileResultGetStatus(apple), VERNON_STATUS_OK)
        << std::string(vernonCompileResultGetDiagnostics(apple).data, vernonCompileResultGetDiagnostics(apple).size);
    VernonStringView appleObject = vernonCompileResultGetArtifactData(apple, 0);
    EXPECT_FALSE(view_contains(appleObject, "__vernon_cpu_f16_to_f32_bits"));
    EXPECT_FALSE(view_contains(appleObject, "__extendhfsf2"));
    EXPECT_FALSE(view_contains(appleObject, "__extendhfdf2"));
    EXPECT_FALSE(view_contains(appleObject, "__truncsfhf2"));
    EXPECT_FALSE(view_contains(appleObject, "__truncdfhf2"));
    EXPECT_FALSE(view_contains(appleObject, "__mulhf3"));

    constexpr std::string_view avx512Fp16 = "+avx512fp16";
    options.as.cpu.triple = {linuxTriple.data(), linuxTriple.size()};
    options.as.cpu.features = {avx512Fp16.data(), avx512Fp16.size()};
    VernonCompileResult *nativeX86 = vernonCompilerCompileMlirWithOptions(compiler, module, strlen(module), &options);
    ASSERT_TRUE(nativeX86);
    ASSERT_EQ(vernonCompileResultGetStatus(nativeX86), VERNON_STATUS_OK) << std::string(
        vernonCompileResultGetDiagnostics(nativeX86).data, vernonCompileResultGetDiagnostics(nativeX86).size);
    VernonStringView nativeX86Object = vernonCompileResultGetArtifactData(nativeX86, 0);
    EXPECT_FALSE(view_contains(nativeX86Object, "__vernon_cpu_f16_to_f32_bits"));
    EXPECT_FALSE(view_contains(nativeX86Object, "__extendhfsf2"));
    EXPECT_FALSE(view_contains(nativeX86Object, "__extendhfdf2"));
    EXPECT_FALSE(view_contains(nativeX86Object, "__truncsfhf2"));
    EXPECT_FALSE(view_contains(nativeX86Object, "__truncdfhf2"));
    EXPECT_FALSE(view_contains(nativeX86Object, "__mulhf3"));

    static const char unsupportedModule[] = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @integer_to_half(
      %value: i32 {vernon.interface = "input", vernon.location = 0 : i64}) ->
      (f16 {vernon.interface = "output", vernon.location = 0 : i64})
      attributes {vernon.entry, vernon.stage = "compute",
                  vernon.workgroup_size = array<i32: 1, 1, 1>} {
    %result = arith.sitofp %value : i32 to f16
    return %result : f16
  }
}
)mlir";
    options.as.cpu.features = {};
    VernonCompileResult *unsupported =
        vernonCompilerCompileMlirWithOptions(compiler, unsupportedModule, strlen(unsupportedModule), &options);
    ASSERT_TRUE(unsupported);
    EXPECT_NE(vernonCompileResultGetStatus(unsupported), VERNON_STATUS_OK);
    EXPECT_TRUE(view_contains(vernonCompileResultGetDiagnostics(unsupported), "unsupported by software legalization"));

    static const char collisionModule[] = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func private @__vernon_cpu_f16_to_f32_bits(%value: f32) -> f32 {
    return %value : f32
  }
  func.func @extend_with_collision(
      %value: f16 {vernon.interface = "input", vernon.location = 0 : i64}) ->
      (f32 {vernon.interface = "output", vernon.location = 0 : i64})
      attributes {vernon.entry, vernon.stage = "compute",
                  vernon.workgroup_size = array<i32: 1, 1, 1>} {
    %result = arith.extf %value : f16 to f32
    return %result : f32
  }
}
)mlir";
    VernonCompileResult *collision =
        vernonCompilerCompileMlirWithOptions(compiler, collisionModule, strlen(collisionModule), &options);
    ASSERT_TRUE(collision);
    EXPECT_EQ(vernonCompileResultGetStatus(collision), VERNON_STATUS_OK) << std::string(
        vernonCompileResultGetDiagnostics(collision).data, vernonCompileResultGetDiagnostics(collision).size);

    vernonCompileResultDestroy(collision);
    vernonCompileResultDestroy(unsupported);
    vernonCompileResultDestroy(nativeX86);
    vernonCompileResultDestroy(apple);
    vernonCompileResultDestroy(windows);
    vernonCompileResultDestroy(linux);
    vernonCompileResultDestroy(compiled);
    vernonCompilerDestroy(compiler);
}

TEST(CompilerCApi, ValidatesAndCompilesAllTargets) {
    static const char module[] = "module attributes {" VERNON_MLIR_VERSION_ATTRIBUTES "} {\n"
                                 "  func.func @vertex_main("
                                 "%position: vector<3xf32> {vernon.interface = \"input\", "
                                 "vernon.location = 0 : i64}) attributes {vernon.entry, "
                                 "vernon.stage = \"vertex\"} {\n"
                                 "    return\n"
                                 "  }\n"
                                 "}\n";
    static const char invalid_module[] = "module attributes {" VERNON_MLIR_VERSION_ATTRIBUTES "} {\n"
                                         "  func.func @compute_main() attributes {vernon.entry, "
                                         "vernon.stage = \"compute\"} {\n"
                                         "    return\n"
                                         "  }\n"
                                         "}\n";
    static const char cpu_module[] = "module attributes {" VERNON_MLIR_VERSION_ATTRIBUTES "} {\n"
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
    static const char cpu_intrinsic_module[] = "module attributes {" VERNON_MLIR_VERSION_ATTRIBUTES "} {\n"
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
    static const char cpu_large_vector_module[] = "module attributes {" VERNON_MLIR_VERSION_ATTRIBUTES "} {\n"
                                                  "  func.func @add_large(%left: tensor<20xf32> "
                                                  "{vernon.interface = \"input\", vernon.location = 0 : i64}, "
                                                  "%right: tensor<20xf32> "
                                                  "{vernon.interface = \"input\", vernon.location = 1 : i64}) -> "
                                                  "(tensor<20xf32> {vernon.interface = \"output\", "
                                                  "vernon.location = 0 : i64}) attributes "
                                                  "{vernon.entry, vernon.stage = \"fragment\"} {\n"
                                                  "    %sum = arith.addf %left, %right : tensor<20xf32>\n"
                                                  "    return %sum : tensor<20xf32>\n"
                                                  "  }\n"
                                                  "}\n";
    static const char cpu_unknown_intrinsic_module[] = "module attributes {" VERNON_MLIR_VERSION_ATTRIBUTES "} {\n"
                                                       "  func.func @unknown_cpu(%value: f32 "
                                                       "{vernon.interface = \"input\", vernon.location = 0 : i64}) -> "
                                                       "(f32 {vernon.interface = \"output\", "
                                                       "vernon.location = 0 : i64}) attributes "
                                                       "{vernon.entry, vernon.stage = \"fragment\"} {\n"
                                                       "    %result = \"vernon.intrinsic\"(%value) "
                                                       "{name = \"not_a_cpu_intrinsic\"} : (f32) -> f32\n"
                                                       "    return %result : f32\n"
                                                       "  }\n"
                                                       "}\n";
    static const char cpu_compute_module[] =
        "module attributes {" VERNON_MLIR_VERSION_ATTRIBUTES "} {\n"
        "  func.func @increment("
        "%values: !vernon.tensor_view<f32, [3], \"read_write\", \"device\"> "
        "{vernon.interface = \"resource\", vernon.set = 0 : i64, "
        "vernon.binding = 0 : i64}, "
        "%id: tensor<3xi32> {vernon.interface = \"input\", "
        "vernon.builtin = \"global_invocation_id\"}) attributes {vernon.entry, "
        "vernon.stage = \"compute\", "
        "vernon.workgroup_size = array<i32: 8, 1, 1>} {\n"
        "    %zero = arith.constant 0 : index\n"
        "    %id_i32 = tensor.extract %id[%zero] : tensor<3xi32>\n"
        "    %id_x = arith.index_castui %id_i32 : i32 to index\n"
        "    %value = \"vernon.load\"(%values, %id_x) : "
        "(!vernon.tensor_view<f32, [3], \"read_write\", \"device\">, index) -> f32\n"
        "    %one = arith.constant 1.0 : f32\n"
        "    %sum = arith.addf %value, %one : f32\n"
        "    \"vernon.store\"(%sum, %values, %id_x) : "
        "(f32, !vernon.tensor_view<f32, [3], \"read_write\", \"device\">, index) -> ()\n"
        "    return\n"
        "  }\n"
        "}\n";
    static const char cuda_while_module[] =
        "module attributes {" VERNON_MLIR_VERSION_ATTRIBUTES "} {\n"
        "  func.func @loop(%values: !vernon.tensor_view<f32, [1], \"read_write\", \"device\"> "
        "{vernon.interface = \"resource\", vernon.set = 0 : i64, "
        "vernon.binding = 0 : i64}, "
        "%phase: f32 "
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
        "      \"vernon.store\"(%value, %values, %index) : "
        "(f32, !vernon.tensor_view<f32, [1], \"read_write\", \"device\">, index) -> ()\n"
        "      scf.yield\n"
        "    } else {\n"
        "      scf.yield\n"
        "    }\n"
        "    return\n"
        "  }\n"
        "}\n";
    static const char cuda_rank_three_tensor_module[] =
        "module attributes {" VERNON_MLIR_VERSION_ATTRIBUTES "} {\n"
        "  func.func @tensor3(%values: !vernon.tensor_view<f32, [1], \"read_write\", \"device\"> "
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
        "    \"vernon.store\"(%value, %values, %index) : "
        "(f32, !vernon.tensor_view<f32, [1], \"read_write\", \"device\">, index) -> ()\n"
        "    return\n"
        "  }\n"
        "}\n";
    static const char cuda_dynamic_local_tensor_module[] =
        "module attributes {" VERNON_MLIR_VERSION_ATTRIBUTES "} {\n"
        "  func.func @dynamic_local("
        "%values: !vernon.tensor_view<f32, [1], \"read_write\", \"device\"> "
        "{vernon.interface = \"resource\", vernon.set = 0 : i64, "
        "vernon.binding = 0 : i64}) attributes {vernon.entry, "
        "vernon.stage = \"compute\", "
        "vernon.workgroup_size = array<i32: 1, 1, 1>} {\n"
        "    %size = arith.constant 4 : index\n"
        "    %value = tensor.empty(%size) : tensor<?xf32>\n"
        "    %index = arith.constant 0 : index\n"
        "    %element = tensor.extract %value[%index] : tensor<?xf32>\n"
        "    \"vernon.store\"(%element, %values, %index) : "
        "(f32, !vernon.tensor_view<f32, [1], \"read_write\", \"device\">, index) -> ()\n"
        "    return\n"
        "  }\n"
        "}\n";
    static const char cpu_texture_module[] = "module attributes {" VERNON_MLIR_VERSION_ATTRIBUTES "} {\n"
                                             "  func.func @sample_color("
                                             "%texture: !vernon.texture<\"2d\", f32, \"unknown\", \"sampled\"> "
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
                                             "(!vernon.texture<\"2d\", f32, \"unknown\", \"sampled\">, "
                                             "!vernon.sampler, tensor<2xf32>) "
                                             "-> tensor<4xf32>\n"
                                             "    return %color : tensor<4xf32>\n"
                                             "  }\n"
                                             "}\n";

    VernonCompilerContext *context = vernonCompilerCreate();
    ASSERT_TRUE(context != NULL);
    VernonTargetCapabilities vulkan = vernonCompilerGetTargetCapabilities(context, VERNON_TARGET_VULKAN);
    ASSERT_TRUE(vulkan.available && vulkan.supports_graphics);
    ASSERT_TRUE(vulkan.supports_device_storage_atomics && !vulkan.supports_f32_device_atomic_add);
    VernonTargetCapabilities cuda = vernonCompilerGetTargetCapabilities(context, VERNON_TARGET_CUDA);
    ASSERT_TRUE(cuda.available && cuda.supports_compute && !cuda.supports_graphics);
    ASSERT_TRUE(cuda.supports_device_storage_atomics && cuda.supports_f32_device_atomic_add);
    VernonTargetCapabilities cpu = vernonCompilerGetTargetCapabilities(context, VERNON_TARGET_CPU);
    ASSERT_TRUE(cpu.available && cpu.supports_graphics && cpu.supports_compute);
    ASSERT_TRUE(cpu.supports_device_storage_atomics && cpu.supports_f32_device_atomic_add);
    VernonTargetCapabilities opengl = vernonCompilerGetTargetCapabilities(context, VERNON_TARGET_OPENGL);
    VernonTargetCapabilities opengles = vernonCompilerGetTargetCapabilities(context, VERNON_TARGET_OPENGL_ES);
    VernonTargetCapabilities metal = vernonCompilerGetTargetCapabilities(context, VERNON_TARGET_METAL);
    ASSERT_TRUE(!opengl.supports_device_storage_atomics && !opengles.supports_device_storage_atomics &&
                !metal.supports_device_storage_atomics);
    VernonTargetCapabilities directx = vernonCompilerGetTargetCapabilities(context, VERNON_TARGET_DIRECTX);
    if (directx.available)
        ASSERT_TRUE(directx.supports_graphics && directx.supports_compute);

    VernonCompileResult *validation = vernonCompilerValidateMlir(context, module, strlen(module));
    ASSERT_TRUE(validation != NULL);
    ASSERT_TRUE(vernonCompileResultGetStatus(validation) == VERNON_STATUS_OK);

    VernonStringView reflection = vernonCompileResultGetReflection(validation);
    ASSERT_TRUE(reflection.data != NULL);
    ASSERT_TRUE(reflection.size != 0);
    ASSERT_TRUE(reflection.data[0] == '{');
    ASSERT_TRUE(vernonCompileResultGetCpuEntry(validation, "vertex_main", 11) == NULL);
    vernonCompileResultDestroy(validation);

    VernonCompileResult *invalid = vernonCompilerValidateMlir(context, invalid_module, strlen(invalid_module));
    ASSERT_TRUE(invalid != NULL);
    ASSERT_TRUE(vernonCompileResultGetStatus(invalid) == VERNON_STATUS_VERIFICATION_ERROR);
    vernonCompileResultDestroy(invalid);

    VernonCompileResult *parse_error = vernonCompilerValidateMlir(context, "not mlir", 8);
    ASSERT_TRUE(parse_error != NULL);
    ASSERT_TRUE(vernonCompileResultGetStatus(parse_error) == VERNON_STATUS_PARSE_ERROR);
    ASSERT_TRUE(vernonCompileResultGetDiagnostics(parse_error).size != 0);
    vernonCompileResultDestroy(parse_error);

    VernonCompileOptions directx_options{};
    directx_options.struct_size = sizeof(directx_options);
    directx_options.target = VERNON_TARGET_DIRECTX;
    if (!vernon::tests::unavailableDirectXTarget(context, VERNON_TARGET_DIRECTX)) {
        VernonCompileResult *compile =
            vernonCompilerCompileMlir(context, module, strlen(module), VERNON_TARGET_DIRECTX);
        ASSERT_TRUE(compile != NULL);
        ASSERT_TRUE(vernonCompileResultGetStatus(compile) == VERNON_STATUS_OK);
        ASSERT_TRUE(vernonCompileResultGetArtifactCount(compile) == 1);
        VernonStringView directx_name = vernonCompileResultGetArtifactName(compile, 0);
        ASSERT_TRUE(view_contains(directx_name, "vertex_main.vert.dxil"));
        VernonStringView dxil = vernonCompileResultGetArtifactData(compile, 0);
        ASSERT_TRUE(dxil.size >= 4);
        ASSERT_TRUE(std::memcmp(dxil.data, "DXBC", 4) == 0);
        VernonStringView directx_reflection = vernonCompileResultGetReflection(compile);
        ASSERT_TRUE(view_contains(directx_reflection, "\"kind\":\"directx\""));
        ASSERT_TRUE(view_contains(directx_reflection, "\"format\":\"dxil\""));
        ASSERT_TRUE(view_contains(directx_reflection, "\"shader_model\":60"));
        vernonCompileResultDestroy(compile);

        directx_options.as.directx.shader_model = 60;
        VernonCompileResult *directx_sm60 =
            vernonCompilerCompileMlirWithOptions(context, module, strlen(module), &directx_options);
        ASSERT_TRUE(directx_sm60 != NULL);
        ASSERT_TRUE(vernonCompileResultGetStatus(directx_sm60) == VERNON_STATUS_OK);
        ASSERT_TRUE(view_contains(vernonCompileResultGetReflection(directx_sm60), "\"shader_model\":60"));
        vernonCompileResultDestroy(directx_sm60);
    }
    directx_options.as.directx.shader_model = 55;
    VernonCompileResult *invalid_directx_options =
        vernonCompilerCompileMlirWithOptions(context, module, strlen(module), &directx_options);
    ASSERT_TRUE(invalid_directx_options != NULL);
    ASSERT_TRUE(vernonCompileResultGetStatus(invalid_directx_options) == VERNON_STATUS_INVALID_ARGUMENT);
    vernonCompileResultDestroy(invalid_directx_options);

    VernonCompileResult *vulkan_compile =
        vernonCompilerCompileMlir(context, module, strlen(module), VERNON_TARGET_VULKAN);
    ASSERT_TRUE(vulkan_compile != NULL);
    ASSERT_TRUE(vernonCompileResultGetStatus(vulkan_compile) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonCompileResultGetArtifactCount(vulkan_compile) == 1);
    VernonStringView name = vernonCompileResultGetArtifactName(vulkan_compile, 0);
    ASSERT_TRUE(name.size == strlen("module.spv"));
    ASSERT_TRUE(memcmp(name.data, "module.spv", name.size) == 0);
    VernonStringView spirv = vernonCompileResultGetArtifactData(vulkan_compile, 0);
    uint32_t magic = 0;
    ASSERT_TRUE(spirv.size >= sizeof(magic));
    memcpy(&magic, spirv.data, sizeof(magic));
    ASSERT_TRUE(magic == 0x07230203u);
    VernonStringView vulkan_reflection = vernonCompileResultGetReflection(vulkan_compile);
    const std::string compilerVersion =
        "\"compiler_contract_version\":" + std::to_string(VERNON_COMPILER_CONTRACT_VERSION);
    const std::string pipelineVersion = "\"pipeline_version\":" + std::to_string(VERNON_PIPELINE_VERSION);
    ASSERT_TRUE(view_contains(vulkan_reflection, compilerVersion.c_str()));
    ASSERT_TRUE(view_contains(vulkan_reflection, pipelineVersion.c_str()));
    ASSERT_TRUE(view_contains(vulkan_reflection, "\"kind\":\"vulkan\""));
    ASSERT_TRUE(view_contains(vulkan_reflection, "\"entry_point\":\"vertex_main\""));
    ASSERT_TRUE(view_contains(vulkan_reflection, "\"filename\":\"module.spv\""));

    VernonCompileResult *repeated_vulkan_compile =
        vernonCompilerCompileMlir(context, module, strlen(module), VERNON_TARGET_VULKAN);
    ASSERT_TRUE(repeated_vulkan_compile != NULL);
    ASSERT_TRUE(vernonCompileResultGetStatus(repeated_vulkan_compile) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonCompileResultGetArtifactCount(repeated_vulkan_compile) == 1);
    ASSERT_TRUE(views_equal(name, vernonCompileResultGetArtifactName(repeated_vulkan_compile, 0)));
    ASSERT_TRUE(views_equal(spirv, vernonCompileResultGetArtifactData(repeated_vulkan_compile, 0)));
    ASSERT_TRUE(views_equal(vulkan_reflection, vernonCompileResultGetReflection(repeated_vulkan_compile)));
    vernonCompileResultDestroy(repeated_vulkan_compile);
    vernonCompileResultDestroy(vulkan_compile);

    if (opengl.available) {
        VernonCompileOptions options{};
        options.struct_size = sizeof(options);
        options.target = VERNON_TARGET_OPENGL;
        options.as.opengl.version = 450;
        VernonCompileResult *opengl_compile =
            vernonCompilerCompileMlirWithOptions(context, cpu_module, strlen(cpu_module), &options);
        ASSERT_TRUE(opengl_compile != NULL);
        ASSERT_TRUE(vernonCompileResultGetStatus(opengl_compile) == VERNON_STATUS_OK);
        VernonStringView glsl = vernonCompileResultGetArtifactData(opengl_compile, 0);
        ASSERT_TRUE(glsl.size >= strlen("#version 450"));
        ASSERT_TRUE(memcmp(glsl.data, "#version 450", strlen("#version 450")) == 0);
        VernonStringView opengl_reflection = vernonCompileResultGetReflection(opengl_compile);
        ASSERT_TRUE(view_contains(opengl_reflection, "\"kind\":\"opengl\""));
        ASSERT_TRUE(view_contains(opengl_reflection, "\"version\":450"));
        ASSERT_TRUE(view_contains(opengl_reflection, "\"format\":\"glsl\""));
        vernonCompileResultDestroy(opengl_compile);

        options.target = static_cast<VernonTarget>(UINT32_MAX);
        VernonCompileResult *invalid_options =
            vernonCompilerCompileMlirWithOptions(context, module, strlen(module), &options);
        ASSERT_TRUE(invalid_options != NULL);
        ASSERT_TRUE(vernonCompileResultGetStatus(invalid_options) == VERNON_STATUS_INVALID_ARGUMENT);
        vernonCompileResultDestroy(invalid_options);
    }

    if (opengles.available) {
        VernonCompileOptions options{};
        options.struct_size = sizeof(options);
        options.target = VERNON_TARGET_OPENGL_ES;
        options.as.opengl.version = 310;
        VernonCompileResult *opengles_compile =
            vernonCompilerCompileMlirWithOptions(context, cpu_module, strlen(cpu_module), &options);
        ASSERT_TRUE(opengles_compile != NULL);
        ASSERT_TRUE(vernonCompileResultGetStatus(opengles_compile) == VERNON_STATUS_OK);
        VernonStringView glsl = vernonCompileResultGetArtifactData(opengles_compile, 0);
        ASSERT_TRUE(glsl.size >= strlen("#version 310 es"));
        ASSERT_TRUE(memcmp(glsl.data, "#version 310 es", strlen("#version 310 es")) == 0);
        vernonCompileResultDestroy(opengles_compile);
    }

    VernonCompileResult *vulkan_compute_compile =
        vernonCompilerCompileMlir(context, cpu_compute_module, strlen(cpu_compute_module), VERNON_TARGET_VULKAN);
    ASSERT_TRUE(vulkan_compute_compile != NULL);
    ASSERT_TRUE(vernonCompileResultGetStatus(vulkan_compute_compile) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonCompileResultGetArtifactCount(vulkan_compute_compile) == 1);
    vernonCompileResultDestroy(vulkan_compute_compile);

    std::string vulkan_f16_module(cpu_compute_module);
    for (size_t position = 0; (position = vulkan_f16_module.find("f32", position)) != std::string::npos; position += 3)
        vulkan_f16_module.replace(position, 3, "f16");
    VernonCompileResult *vulkan_f16_compile =
        vernonCompilerCompileMlir(context, vulkan_f16_module.data(), vulkan_f16_module.size(), VERNON_TARGET_VULKAN);
    ASSERT_TRUE(vulkan_f16_compile != NULL);
    ASSERT_TRUE(vernonCompileResultGetStatus(vulkan_f16_compile) == VERNON_STATUS_INTERNAL_ERROR);
    ASSERT_TRUE(view_contains(vernonCompileResultGetDiagnostics(vulkan_f16_compile), "shaderFloat16"));
    vernonCompileResultDestroy(vulkan_f16_compile);

    VernonCompileResult *cuda_compile =
        vernonCompilerCompileMlir(context, cpu_compute_module, strlen(cpu_compute_module), VERNON_TARGET_CUDA);
    ASSERT_TRUE(cuda_compile != NULL);
    ASSERT_TRUE(vernonCompileResultGetStatus(cuda_compile) == VERNON_STATUS_OK);
    VernonStringView ptx = vernonCompileResultGetArtifactData(cuda_compile, 0);
    ASSERT_TRUE(ptx.size != 0);
    ASSERT_TRUE(strstr(ptx.data, ".version") != NULL);
    vernonCompileResultDestroy(cuda_compile);

    VernonCompileResult *cuda_while_compile =
        vernonCompilerCompileMlir(context, cuda_while_module, strlen(cuda_while_module), VERNON_TARGET_CUDA);
    ASSERT_TRUE(cuda_while_compile != NULL);
    ASSERT_TRUE(vernonCompileResultGetStatus(cuda_while_compile) == VERNON_STATUS_OK);
    VernonStringView while_ptx = vernonCompileResultGetArtifactData(cuda_while_compile, 0);
    ASSERT_TRUE(while_ptx.size != 0);
    ASSERT_TRUE(view_contains(while_ptx, ".version"));
    ASSERT_TRUE(view_contains(while_ptx, "cos.approx"));
    ASSERT_TRUE(view_contains(while_ptx, "sqrt.rn"));
    ASSERT_TRUE(!view_contains(while_ptx, "__nv_"));
    vernonCompileResultDestroy(cuda_while_compile);

    if (metal.available) {
        VernonCompileResult *metal_compute =
            vernonCompilerCompileMlir(context, cuda_while_module, strlen(cuda_while_module), VERNON_TARGET_METAL);
        ASSERT_TRUE(metal_compute != NULL);
        ASSERT_TRUE(vernonCompileResultGetStatus(metal_compute) == VERNON_STATUS_OK);
        VernonStringView metal_source = vernonCompileResultGetArtifactData(metal_compute, 0);
        ASSERT_TRUE(view_contains(metal_source, "kernel void loop"));
        ASSERT_TRUE(view_contains(vernonCompileResultGetReflection(metal_compute), "\"kind\":\"metal\""));
        ASSERT_TRUE(view_contains(vernonCompileResultGetReflection(metal_compute), "\"platform\":\"macos\""));
        ASSERT_TRUE(view_contains(vernonCompileResultGetReflection(metal_compute), "\"version\":[2,4]"));
        ASSERT_TRUE(view_contains(vernonCompileResultGetReflection(metal_compute), "\"minimum_os_version\":[11,0]"));
        vernonCompileResultDestroy(metal_compute);

        VernonCompileOptions ios_options{};
        ios_options.struct_size = sizeof(ios_options);
        ios_options.target = VERNON_TARGET_METAL;
        ios_options.as.metal.platform = VERNON_METAL_PLATFORM_IOS;
        VernonCompileResult *ios_metal_compute =
            vernonCompilerCompileMlirWithOptions(context, cuda_while_module, strlen(cuda_while_module), &ios_options);
        ASSERT_TRUE(ios_metal_compute != NULL);
        ASSERT_TRUE(vernonCompileResultGetStatus(ios_metal_compute) == VERNON_STATUS_OK);
        ASSERT_TRUE(view_contains(vernonCompileResultGetReflection(ios_metal_compute), "\"platform\":\"ios\""));
        ASSERT_TRUE(view_contains(vernonCompileResultGetReflection(ios_metal_compute), "\"version\":[2,4]"));
        ASSERT_TRUE(
            view_contains(vernonCompileResultGetReflection(ios_metal_compute), "\"minimum_os_version\":[15,0]"));
        vernonCompileResultDestroy(ios_metal_compute);
    }

    VernonCompileResult *cuda_rank_three_compile = vernonCompilerCompileMlir(
        context, cuda_rank_three_tensor_module, strlen(cuda_rank_three_tensor_module), VERNON_TARGET_CUDA);
    ASSERT_TRUE(cuda_rank_three_compile != NULL);
    if (vernonCompileResultGetStatus(cuda_rank_three_compile) != VERNON_STATUS_OK) {
        VernonStringView diagnostics = vernonCompileResultGetDiagnostics(cuda_rank_three_compile);
        fprintf(stderr, "%.*s\n", (int)diagnostics.size, diagnostics.data);
    }
    ASSERT_TRUE(vernonCompileResultGetStatus(cuda_rank_three_compile) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonCompileResultGetArtifactCount(cuda_rank_three_compile) == 1);
    vernonCompileResultDestroy(cuda_rank_three_compile);

    VernonCompileResult *cuda_dynamic_local_compile = vernonCompilerCompileMlir(
        context, cuda_dynamic_local_tensor_module, strlen(cuda_dynamic_local_tensor_module), VERNON_TARGET_CUDA);
    ASSERT_TRUE(cuda_dynamic_local_compile != NULL);
    ASSERT_TRUE(vernonCompileResultGetStatus(cuda_dynamic_local_compile) != VERNON_STATUS_OK);
    ASSERT_TRUE(
        view_contains(vernonCompileResultGetDiagnostics(cuda_dynamic_local_compile), "dynamic local value Tensor"));
    vernonCompileResultDestroy(cuda_dynamic_local_compile);

    VernonCompileResult *cpu_compile =
        vernonCompilerCompileMlir(context, cpu_module, strlen(cpu_module), VERNON_TARGET_CPU);
    ASSERT_TRUE(cpu_compile != NULL);
    ASSERT_TRUE(vernonCompileResultGetStatus(cpu_compile) == VERNON_STATUS_OK);
#if defined(_WIN32)
    ASSERT_TRUE(view_contains(vernonCompileResultGetArtifactName(cpu_compile, 0), "module.obj"));
#else
    ASSERT_TRUE(view_contains(vernonCompileResultGetArtifactName(cpu_compile, 0), "module.o"));
#endif
    ASSERT_TRUE(view_contains(vernonCompileResultGetReflection(cpu_compile), "__vernon_cpu_"));
    ASSERT_TRUE(view_contains(vernonCompileResultGetReflection(cpu_compile), "_add_vectors"));
    VernonCpuEntryPoint add_vectors = vernonCompileResultGetCpuEntry(cpu_compile, "add_vectors", 11);
    ASSERT_TRUE(add_vectors != NULL);
    float cpu_arguments[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float cpu_results[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    VernonCpuInvocation invocation = {cpu_arguments, sizeof(cpu_arguments), cpu_results, sizeof(cpu_results), NULL};
    ASSERT_TRUE(invoke_cpu_range(add_vectors, invocation) == VERNON_STATUS_OK);
    ASSERT_TRUE(cpu_results[0] == 6.0f && cpu_results[1] == 8.0f);
    ASSERT_TRUE(cpu_results[2] == 10.0f && cpu_results[3] == 12.0f);
    invocation.arguments_size = 0;
    ASSERT_TRUE(invoke_cpu_range(add_vectors, invocation) == VERNON_STATUS_INVALID_ARGUMENT);
    invocation.arguments_size = sizeof(cpu_arguments);
    invocation.results_size = 0;
    ASSERT_TRUE(invoke_cpu_range(add_vectors, invocation) == VERNON_STATUS_INVALID_ARGUMENT);
    invocation.results_size = sizeof(cpu_results);
    invocation.results = NULL;
    ASSERT_TRUE(invoke_cpu_range(add_vectors, invocation) == VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_TRUE(add_vectors(NULL) == VERNON_STATUS_INVALID_ARGUMENT);
    vernonCompileResultDestroy(cpu_compile);

    const char *ios_triple = "arm64-apple-ios17.0";
    VernonCompileOptions ios_options = {0};
    ios_options.struct_size = sizeof(ios_options);
    ios_options.target = VERNON_TARGET_CPU;
    ios_options.as.cpu.triple.data = ios_triple;
    ios_options.as.cpu.triple.size = strlen(ios_triple);
    VernonCompileResult *ios_cpu_compile =
        vernonCompilerCompileMlirWithOptions(context, cpu_module, strlen(cpu_module), &ios_options);
    ASSERT_TRUE(ios_cpu_compile != NULL);
    ASSERT_TRUE(vernonCompileResultGetStatus(ios_cpu_compile) == VERNON_STATUS_OK);
    VernonStringView ios_object = vernonCompileResultGetArtifactData(ios_cpu_compile, 0);
    ASSERT_TRUE(ios_object.size > 4);
    ASSERT_TRUE((unsigned char)ios_object.data[0] == 0xcf);
    ASSERT_TRUE((unsigned char)ios_object.data[1] == 0xfa);
    ASSERT_TRUE((unsigned char)ios_object.data[2] == 0xed);
    ASSERT_TRUE((unsigned char)ios_object.data[3] == 0xfe);
    ASSERT_TRUE(vernonCompileResultGetCpuEntry(ios_cpu_compile, "add_vectors", 11) == NULL);
    vernonCompileResultDestroy(ios_cpu_compile);

    VernonCompileResult *cpu_large_compile =
        vernonCompilerCompileMlir(context, cpu_large_vector_module, strlen(cpu_large_vector_module), VERNON_TARGET_CPU);
    ASSERT_TRUE(cpu_large_compile != NULL);
    if (vernonCompileResultGetStatus(cpu_large_compile) != VERNON_STATUS_OK) {
        VernonStringView diagnostics = vernonCompileResultGetDiagnostics(cpu_large_compile);
        fprintf(stderr, "%.*s\n", (int)diagnostics.size, diagnostics.data);
    }
    ASSERT_TRUE(vernonCompileResultGetStatus(cpu_large_compile) == VERNON_STATUS_OK);
    VernonCpuEntryPoint add_large = vernonCompileResultGetCpuEntry(cpu_large_compile, "add_large", 9);
    ASSERT_TRUE(add_large != NULL);
    float large_arguments[40];
    float large_results[20] = {0};
    for (size_t index = 0; index < 20; ++index) {
        large_arguments[index] = (float)index;
        large_arguments[index + 20] = 2.0f;
    }
    VernonCpuInvocation large_invocation = {large_arguments, sizeof(large_arguments), large_results,
                                            sizeof(large_results), NULL};
    ASSERT_TRUE(invoke_cpu_range(add_large, large_invocation) == VERNON_STATUS_OK);
    ASSERT_TRUE(large_results[0] == 2.0f && large_results[19] == 21.0f);
    vernonCompileResultDestroy(cpu_large_compile);

    VernonCompileResult *cpu_unknown_compile = vernonCompilerCompileMlir(
        context, cpu_unknown_intrinsic_module, strlen(cpu_unknown_intrinsic_module), VERNON_TARGET_CPU);
    ASSERT_TRUE(cpu_unknown_compile != NULL);
    ASSERT_TRUE(vernonCompileResultGetStatus(cpu_unknown_compile) != VERNON_STATUS_OK);
    ASSERT_TRUE(view_contains(vernonCompileResultGetDiagnostics(cpu_unknown_compile), "not_a_cpu_intrinsic"));
    ASSERT_TRUE(view_contains(vernonCompileResultGetDiagnostics(cpu_unknown_compile), "unknown_cpu"));
    vernonCompileResultDestroy(cpu_unknown_compile);

    VernonCompileResult *cpu_intrinsic_compile =
        vernonCompilerCompileMlir(context, cpu_intrinsic_module, strlen(cpu_intrinsic_module), VERNON_TARGET_CPU);
    ASSERT_TRUE(cpu_intrinsic_compile != NULL);
    ASSERT_TRUE(vernonCompileResultGetStatus(cpu_intrinsic_compile) == VERNON_STATUS_OK);
    ASSERT_TRUE(view_contains(vernonCompileResultGetReflection(cpu_intrinsic_compile), "\"packed_arguments_size\":24"));
    VernonCpuEntryPoint normal_score = vernonCompileResultGetCpuEntry(cpu_intrinsic_compile, "normal_score", 12);
    ASSERT_TRUE(normal_score != NULL);
    float intrinsic_arguments[6] = {0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 1.0f};
    float intrinsic_result = 0.0f;
    VernonCpuInvocation intrinsic_invocation = {intrinsic_arguments, sizeof(intrinsic_arguments), &intrinsic_result,
                                                sizeof(intrinsic_result), NULL};
    ASSERT_TRUE(invoke_cpu_range(normal_score, intrinsic_invocation) == VERNON_STATUS_OK);
    ASSERT_TRUE(intrinsic_result == 1.0f);
    vernonCompileResultDestroy(cpu_intrinsic_compile);

    VernonCompileResult *cpu_compute_compile =
        vernonCompilerCompileMlir(context, cpu_compute_module, strlen(cpu_compute_module), VERNON_TARGET_CPU);
    ASSERT_TRUE(cpu_compute_compile != NULL);
    ASSERT_TRUE(vernonCompileResultGetStatus(cpu_compute_compile) == VERNON_STATUS_OK);
    VernonCpuEntryPoint increment = vernonCompileResultGetCpuEntry(cpu_compute_compile, "increment", 9);
    ASSERT_TRUE(increment != NULL);
    float compute_values[3] = {2.0f, 4.0f, 6.0f};
    struct RankOneTensorViewDescriptor {
        float *values;
        int64_t offset;
        uint64_t extent;
        int64_t stride;
    };
    struct {
        RankOneTensorViewDescriptor values;
        uint32_t id[3];
    } compute_arguments = {{compute_values, 0, 3, 1}, {1, 0, 0}};
    VernonCpuInvocation compute_invocation = {&compute_arguments, sizeof(compute_arguments), NULL, 0, NULL};
    ASSERT_TRUE(invoke_cpu_range(increment, compute_invocation, 1) == VERNON_STATUS_OK);
    ASSERT_TRUE(compute_values[0] == 2.0f && compute_values[1] == 5.0f && compute_values[2] == 6.0f);
    vernonCompileResultDestroy(cpu_compute_compile);

    VernonCompileResult *cpu_scf_compile =
        vernonCompilerCompileMlir(context, cuda_while_module, strlen(cuda_while_module), VERNON_TARGET_CPU);
    ASSERT_TRUE(cpu_scf_compile != NULL);
    ASSERT_TRUE(vernonCompileResultGetStatus(cpu_scf_compile) == VERNON_STATUS_OK);
    VernonCpuEntryPoint loop = vernonCompileResultGetCpuEntry(cpu_scf_compile, "loop", 4);
    ASSERT_TRUE(loop != NULL);
    float loop_values[1] = {0.0f};
    struct {
        RankOneTensorViewDescriptor values;
        float phase;
    } loop_arguments = {{loop_values, 0, 1, 1}, 0.0f};
    VernonCpuInvocation loop_invocation = {&loop_arguments, sizeof(loop_arguments), NULL, 0, NULL};
    ASSERT_TRUE(invoke_cpu_range(loop, loop_invocation) == VERNON_STATUS_OK);
    ASSERT_TRUE(loop_values[0] == 4.0f);
    vernonCompileResultDestroy(cpu_scf_compile);

    VernonCompileResult *cpu_texture_compile =
        vernonCompilerCompileMlir(context, cpu_texture_module, strlen(cpu_texture_module), VERNON_TARGET_CPU);
    ASSERT_TRUE(cpu_texture_compile != NULL);
    ASSERT_TRUE(vernonCompileResultGetStatus(cpu_texture_compile) == VERNON_STATUS_OK);
    VernonCpuEntryPoint sample_color = vernonCompileResultGetCpuEntry(cpu_texture_compile, "sample_color", 12);
    ASSERT_TRUE(sample_color != NULL);
    struct {
        uintptr_t texture;
        uintptr_t sampler;
        float uv[2];
    } texture_arguments = {3, 0, {0.25f, 0.75f}};
    float texture_result[4] = {0};
    float texture_bias = 0.5f;
    VernonCpuTextureCallbacks texture_callbacks = {&texture_bias, sample_texture, NULL};
    VernonCpuInvocation texture_invocation = {&texture_arguments, sizeof(texture_arguments), texture_result,
                                              sizeof(texture_result), &texture_callbacks};
    ASSERT_TRUE(invoke_cpu_range(sample_color, texture_invocation) == VERNON_STATUS_OK);
    ASSERT_TRUE(texture_result[0] == 0.25f && texture_result[1] == 0.75f);
    ASSERT_TRUE(texture_result[2] == 3.0f && texture_result[3] == 0.5f);
    texture_invocation.textures = NULL;
    ASSERT_TRUE(invoke_cpu_range(sample_color, texture_invocation) == VERNON_STATUS_INVALID_ARGUMENT);
    vernonCompileResultDestroy(cpu_texture_compile);

    vernonCompilerDestroy(context);
}

TEST(CompilerCApi, RejectsMissingOrUnsupportedContractVersions) {
    VernonCompilerContext *context = vernonCompilerCreate();
    ASSERT_TRUE(context);
    const char missing[] = "module { func.func @empty() { return } }";
    VernonCompileResult *result = vernonCompilerValidateMlir(context, missing, strlen(missing));
    ASSERT_TRUE(result);
    EXPECT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_VERIFICATION_ERROR);
    EXPECT_TRUE(view_contains(vernonCompileResultGetDiagnostics(result), "vernon.compiler_contract_version"));
    EXPECT_TRUE(view_contains(vernonCompileResultGetDiagnostics(result), "vernon.pipeline_version"));
    vernonCompileResultDestroy(result);

    const std::string unsupported = "module attributes {vernon.compiler_contract_version = " +
                                    std::to_string(VERNON_COMPILER_CONTRACT_VERSION + 1) +
                                    " : i64, vernon.pipeline_version = " + std::to_string(VERNON_PIPELINE_VERSION + 1) +
                                    " : i64} { func.func @empty() { return } }";
    result = vernonCompilerValidateMlir(context, unsupported.data(), unsupported.size());
    ASSERT_TRUE(result);
    EXPECT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_VERIFICATION_ERROR);
    EXPECT_TRUE(view_contains(vernonCompileResultGetDiagnostics(result), "requires vernon.compiler_contract_version"));
    EXPECT_TRUE(view_contains(vernonCompileResultGetDiagnostics(result), "requires vernon.pipeline_version"));
    vernonCompileResultDestroy(result);

    const char obsoleteAbi[] =
        "module attributes {" VERNON_MLIR_VERSION_ATTRIBUTES "} { "
        "\"vernon.struct\"() {sym_name = \"Old\", fields = [\"value:i32\"], abi_size = 4 : i64} : () -> () "
        "}";
    result = vernonCompilerValidateMlir(context, obsoleteAbi, strlen(obsoleteAbi));
    ASSERT_TRUE(result);
    EXPECT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_VERIFICATION_ERROR);
    EXPECT_TRUE(view_contains(vernonCompileResultGetDiagnostics(result), "obsolete frontend-authored Value ABI"));
    vernonCompileResultDestroy(result);

    const char obsoleteTensorViewAbi[] =
        "module attributes {" VERNON_MLIR_VERSION_ATTRIBUTES "} { "
        "func.func @main(%value: !vernon.tensor_view<f32, [1], \"read\", \"device\"> "
        "{vernon.abi_size = 4 : i64, vernon.interface = \"resource\", vernon.set = 0 : i64, "
        "vernon.binding = 0 : i64}) attributes {vernon.entry, vernon.stage = \"compute\", "
        "vernon.workgroup_size = array<i32: 1, 1, 1>} { return } "
        "}";
    result = vernonCompilerValidateMlir(context, obsoleteTensorViewAbi, strlen(obsoleteTensorViewAbi));
    ASSERT_TRUE(result);
    EXPECT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_VERIFICATION_ERROR);
    EXPECT_TRUE(view_contains(vernonCompileResultGetDiagnostics(result), "retired duplicated Value ABI metadata"));
    vernonCompileResultDestroy(result);
    vernonCompilerDestroy(context);
}
