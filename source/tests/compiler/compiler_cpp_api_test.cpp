#include "VernonVersions.h"
#include "compiler_frontend.h"
#include "compiler_reflection.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Parser/Parser.h"
#include "vernon-c/Compiler.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <gtest/gtest.h>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace {

nlohmann::json validateReflection(VernonCompilerContext *context, std::string_view module) {
    VernonCompileResult *result = vernonCompilerValidateMlir(context, module.data(), module.size());
    EXPECT_NE(result, nullptr);
    if (!result)
        return {};
    if (vernonCompileResultGetStatus(result) != VERNON_STATUS_OK) {
        const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
        std::fwrite(diagnostics.data, 1, diagnostics.size, stderr);
        std::fputc('\n', stderr);
    }
    EXPECT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_OK);
    if (vernonCompileResultGetStatus(result) != VERNON_STATUS_OK) {
        vernonCompileResultDestroy(result);
        return {};
    }
    const VernonStringView reflection = vernonCompileResultGetReflection(result);
    nlohmann::json parsed = nlohmann::json::parse(reflection.data, reflection.data + reflection.size);
    vernonCompileResultDestroy(result);
    return parsed;
}

const nlohmann::json &entry(const nlohmann::json &reflection, std::string_view name) {
    for (const nlohmann::json &candidate : reflection.at("entries"))
        if (candidate.at("name").get<std::string>() == name)
            return candidate;
    ADD_FAILURE() << "missing reflected entry " << name;
    return reflection;
}

const nlohmann::json &argument(const nlohmann::json &reflectedEntry, unsigned index) {
    for (const nlohmann::json &candidate : reflectedEntry.at("arguments"))
        if (candidate.at("index") == index)
            return candidate;
    ADD_FAILURE() << "missing reflected argument " << index;
    return reflectedEntry;
}

void expectDiagnostic(VernonCompilerContext *context, std::string_view module, std::string_view expected) {
    VernonCompileResult *result = vernonCompilerValidateMlir(context, module.data(), module.size());
    ASSERT_TRUE(result);
    ASSERT_TRUE(vernonCompileResultGetStatus(result) != VERNON_STATUS_OK);
    const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
    const std::string_view diagnosticView(diagnostics.data, diagnostics.size);
    if (diagnosticView.find(expected) == std::string_view::npos)
        std::fprintf(stderr, "missing diagnostic '%.*s' in:\n%.*s\n", static_cast<int>(expected.size()),
                     expected.data(), static_cast<int>(diagnosticView.size()), diagnosticView.data());
    ASSERT_TRUE(diagnosticView.find(expected) != std::string_view::npos);
    vernonCompileResultDestroy(result);
}

} // namespace

TEST(CompilerPreparation, MockGpuPreparerMaterializesResourceSignatureFromLogicalProfile) {
    std::unique_ptr<vernon::compiler::CompilerFrontend, decltype(&vernon::compiler::destroyCompilerFrontend)> frontend(
        vernon::compiler::createCompilerFrontend(), vernon::compiler::destroyCompilerFrontend);
    ASSERT_NE(frontend, nullptr);
    mlir::MLIRContext &context = vernon::compiler::compilerMlirContext(*frontend);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @compute(
      %values: tensor<4xf32> {
        vernon.source_name = "values",
        vernon.dtype = "f32",
        vernon.abi_leaf_dtypes = ["f32"]
      }) attributes {vernon.entry, vernon.stage = "compute"} {
    func.return
  }
}
)mlir",
                                                          &context);
    ASSERT_TRUE(module);
    mlir::FailureOr<vernon::compiler::LogicalReflectionModel> logical =
        vernon::compiler::buildLogicalReflectionModel(*module);
    ASSERT_TRUE(mlir::succeeded(logical));
    ASSERT_EQ(logical->entries.size(), 1u);
    ASSERT_EQ(logical->entries[0].arguments.size(), 1u);
    EXPECT_EQ(logical->entries[0].arguments[0].sourcePath, "values");
    EXPECT_EQ(logical->entries[0].arguments[0].dtype, "f32");
    EXPECT_EQ(logical->entries[0].arguments[0].shape, (std::vector<int64_t>{4}));
    vernon::compiler::PreparedModule prepared(std::move(module), std::move(*logical));
    mlir::FailureOr<vernon::compiler::TargetPreparationResult> result = vernon::compiler::prepareTargetModule(
        prepared, [](mlir::ModuleOp target, vernon::compiler::TargetPreparationProvenance &provenance) {
            mlir::func::FuncOp function = *target.getOps<mlir::func::FuncOp>().begin();
            mlir::Builder builder(target.getContext());
            auto logicalTensor = mlir::cast<mlir::RankedTensorType>(function.getArgumentTypes()[0]);
            auto resource = mlir::vernon::TensorViewType::get(target.getContext(), logicalTensor.getElementType(),
                                                              logicalTensor.getShape(), "read_write", "device");
            function.getArgument(0).setType(resource);
            function.setType(builder.getFunctionType({resource}, {}));
            function->setAttr("vernon.dispatch_contract",
                              builder.getDictionaryAttr({
                                  builder.getNamedAttr("unit_grid_axes", builder.getDenseI32ArrayAttr({1, 2})),
                                  builder.getNamedAttr("requires_unit_workgroup", builder.getBoolAttr(false)),
                              }));
            function.setArgAttrs(
                0, builder.getDictionaryAttr({
                       builder.getNamedAttr("vernon.interface", builder.getStringAttr("resource")),
                       builder.getNamedAttr("vernon.source_name", builder.getStringAttr("physical_resource")),
                       builder.getNamedAttr("vernon.dtype", builder.getStringAttr("f32")),
                       builder.getNamedAttr("vernon.element_abi_leaf_dtypes",
                                            builder.getArrayAttr({builder.getStringAttr("f32")})),
                       builder.getNamedAttr("vernon.set", builder.getI64IntegerAttr(0)),
                       builder.getNamedAttr("vernon.binding", builder.getI64IntegerAttr(0)),
                   }));
            if (mlir::failed(provenance.mapArgument(function, 0, 0)))
                return mlir::failure();
            mlir::NamedAttribute builtin(builder.getStringAttr("vernon.builtin"),
                                         builder.getStringAttr("mock_gpu_dispatch"));
            if (mlir::failed(function.insertArgument(function.getNumArguments(), builder.getIndexType(),
                                                     builder.getDictionaryAttr(builtin), function.getLoc())))
                return mlir::failure();
            return mlir::success();
        });
    ASSERT_TRUE(mlir::succeeded(result));
    ASSERT_EQ(result->entries.size(), 1u);
    EXPECT_EQ(result->entries[0].name, "compute");
    EXPECT_EQ(result->entries[0].stage, "compute");
    ASSERT_EQ(result->entries[0].arguments.size(), 2u);
    EXPECT_EQ(result->entries[0].arguments[0].index, 0u);
    EXPECT_EQ(result->entries[0].arguments[0].type, "!vernon.tensor_view<f32, [4], \"read_write\", \"device\">");
    ASSERT_TRUE(result->entries[0].arguments[0].logicalIndex.has_value());
    EXPECT_EQ(*result->entries[0].arguments[0].logicalIndex, 0u);
    EXPECT_EQ(result->entries[0].arguments[0].logicalPath, "values");
    EXPECT_EQ(result->entries[0].arguments[1].index, 1u);
    EXPECT_EQ(result->entries[0].arguments[1].builtin, "mock_gpu_dispatch");
    EXPECT_EQ(result->entries[0].arguments[1].type, "index");
    EXPECT_FALSE(result->entries[0].arguments[1].logicalIndex.has_value());
    mlir::FailureOr<std::string> reflected = vernon::compiler::buildReflection(
        *result->module, prepared.logicalReflection(), result->entries, result->provenance);
    ASSERT_TRUE(mlir::succeeded(reflected));
    const nlohmann::json reflectedJson = nlohmann::json::parse(*reflected);
    EXPECT_EQ(argument(entry(reflectedJson, "compute"), 0).at("vernon.source_name"), "values");
    EXPECT_EQ(entry(reflectedJson, "compute").at("dispatch_contract"),
              (nlohmann::json{{"unit_grid_axes", {1, 2}}, {"requires_unit_workgroup", false}}));
    std::vector<vernon::compiler::PhysicalEntryModel> staleEntries = result->entries;
    staleEntries[0].arguments[0].type = "f32";
    EXPECT_TRUE(mlir::failed(vernon::compiler::buildReflection(*result->module, prepared.logicalReflection(),
                                                               staleEntries, result->provenance)));
    staleEntries = result->entries;
    staleEntries[0].arguments[0].logicalIndex.reset();
    EXPECT_TRUE(mlir::failed(vernon::compiler::buildReflection(*result->module, prepared.logicalReflection(),
                                                               staleEntries, result->provenance)));
    mlir::OwningOpRef<mlir::ModuleOp> original = prepared.clone();
    mlir::func::FuncOp originalFunction = *original->getOps<mlir::func::FuncOp>().begin();
    EXPECT_EQ(originalFunction.getNumArguments(), 1u);
}

TEST(CompilerReflection, ValidationKeepsCanonicalProfilesAndTargetCompileSelectsOne) {
    constexpr std::string_view module = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @compute(
      %value: f32 {
        vernon.source_name = "value",
        vernon.dtype = "f32",
        vernon.interface = "input",
        vernon.location = 0 : i64
      }
    ) -> (f32 {
      vernon.source_name = "output",
      vernon.dtype = "f32",
      vernon.interface = "output",
      vernon.location = 0 : i64
    }) attributes {
      vernon.entry,
      vernon.stage = "compute",
      vernon.workgroup_size = array<i32: 1, 1, 1>
    } {
    return %value : f32
  }
}
)mlir";
    VernonCompilerContext *context = vernonCompilerCreate();
    ASSERT_NE(context, nullptr);
    VernonCompileResult *validation = vernonCompilerValidateMlir(context, module.data(), module.size());
    ASSERT_NE(validation, nullptr);
    if (vernonCompileResultGetStatus(validation) != VERNON_STATUS_OK) {
        const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(validation);
        std::fwrite(diagnostics.data, 1, diagnostics.size, stderr);
        std::fputc('\n', stderr);
    }
    ASSERT_EQ(vernonCompileResultGetStatus(validation), VERNON_STATUS_OK);
    const VernonStringView validationView = vernonCompileResultGetReflection(validation);
    const nlohmann::json validationReflection =
        nlohmann::json::parse(validationView.data, validationView.data + validationView.size);

    VernonCompileResult *compiled = vernonCompilerCompileMlir(context, module.data(), module.size(), VERNON_TARGET_CPU);
    ASSERT_NE(compiled, nullptr);
    if (vernonCompileResultGetStatus(compiled) != VERNON_STATUS_OK) {
        const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(compiled);
        std::fwrite(diagnostics.data, 1, diagnostics.size, stderr);
        std::fputc('\n', stderr);
    }
    ASSERT_EQ(vernonCompileResultGetStatus(compiled), VERNON_STATUS_OK);
    const VernonStringView compiledView = vernonCompileResultGetReflection(compiled);
    const nlohmann::json compiledReflection =
        nlohmann::json::parse(compiledView.data, compiledView.data + compiledView.size);

    for (std::string_view field :
         {"compiler_contract_version", "pipeline_version", "struct_layouts", "dependencies", "module_hash"})
        EXPECT_EQ(validationReflection.at(field), compiledReflection.at(field));
    const nlohmann::json &validationEntry = entry(validationReflection, "compute");
    const nlohmann::json &compiledEntry = entry(compiledReflection, "compute");
    EXPECT_GT(validationEntry.at("physical_layouts").size(), 1u);
    EXPECT_EQ(compiledEntry.at("physical_layouts").size(), 1u);
    EXPECT_TRUE(compiledEntry.at("physical_layouts").contains("host_value"));
    for (std::string_view field : {"vernon.source_name", "vernon.dtype"}) {
        EXPECT_EQ(argument(validationEntry, 0).at(field), argument(compiledEntry, 0).at(field));
        EXPECT_EQ(validationEntry.at("results").at(0).at(field), compiledEntry.at("results").at(0).at(field));
    }
    EXPECT_GT(argument(validationEntry, 0).at("physical_layouts").size(), 1u);
    EXPECT_EQ(argument(compiledEntry, 0).at("physical_layouts").size(), 1u);

    vernonCompileResultDestroy(compiled);
    vernonCompileResultDestroy(validation);
    vernonCompilerDestroy(context);
}

TEST(CompilerReflection, TracksTextureSamplerAndSwizzleSemantics) {
    constexpr std::string_view module = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @fragment_main(
      %color: tensor<4xf32> {
        vernon.interface = "input",
        vernon.location = 0 : i64
      },
      %environment: !vernon.texture<"cube", f32> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 1 : i64
      },
      %sampler: !vernon.sampler {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 2 : i64
      },
      %direction: tensor<3xf32> {
        vernon.interface = "input",
        vernon.location = 1 : i64
      }) -> (tensor<4xf32> {
        vernon.interface = "output",
        vernon.location = 0 : i64
      }) attributes {
        vernon.entry,
        vernon.stage = "fragment"
      } {
    %sample = "vernon.intrinsic"(%environment, %sampler, %direction) {
      name = "texture_sample"
    } : (!vernon.texture<"cube", f32>, !vernon.sampler, tensor<3xf32>)
        -> tensor<4xf32>
    return %sample : tensor<4xf32>
  }
}
)mlir";

    VernonCompilerContext *context = vernonCompilerCreate();
    ASSERT_TRUE(context);

    VernonCompileResult *result = vernonCompilerValidateMlir(context, module.data(), module.size());
    ASSERT_TRUE(result);
    ASSERT_TRUE(vernonCompileResultGetStatus(result) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonCompileResultGetArtifactCount(result) == 1);
    VernonStringView artifactName = vernonCompileResultGetArtifactName(result, 0);
    ASSERT_TRUE(std::string_view(artifactName.data, artifactName.size) == "module.mlir");

    VernonStringView reflection = vernonCompileResultGetReflection(result);
    std::string_view reflectionView(reflection.data, reflection.size);
    ASSERT_TRUE(reflectionView.find("\"fragment_main\"") != std::string_view::npos);
    ASSERT_TRUE(reflectionView.find("\"results\"") != std::string_view::npos);
    ASSERT_TRUE(reflectionView.find("\"vernon.location\":0") != std::string_view::npos);
    ASSERT_TRUE(reflectionView.find("\"kind\":\"texture\"") != std::string_view::npos);
    ASSERT_TRUE(reflectionView.find("\"dimension\":\"cube\"") != std::string_view::npos);
    ASSERT_TRUE(reflectionView.find("\"kind\":\"sampler\"") != std::string_view::npos);
    ASSERT_TRUE(reflectionView.find("\"sampled_texture_bindings\"") != std::string_view::npos);
    ASSERT_TRUE(reflectionView.find("\"sampled_texture_binding\":") == std::string_view::npos);
    ASSERT_TRUE(reflectionView.find("\"sampled_texture_set\":") == std::string_view::npos);

    vernonCompileResultDestroy(result);

    std::fprintf(stderr, "relation\n");
    constexpr std::string_view relationModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @relation(
      %uv: tensor<2xf32> {
        vernon.interface = "input", vernon.location = 0 : i64
      },
      %texture_a: !vernon.texture<"2d", f32> {
        vernon.interface = "resource", vernon.set = 0 : i64,
        vernon.binding = 1 : i64
      },
      %shared_sampler: !vernon.sampler {
        vernon.interface = "resource", vernon.set = 0 : i64,
        vernon.binding = 7 : i64
      },
      %texture_b: !vernon.texture<"2d", f32> {
        vernon.interface = "resource", vernon.set = 0 : i64,
        vernon.binding = 2 : i64
      },
      %texture_c: !vernon.texture<"2d", f32> {
        vernon.interface = "resource", vernon.set = 0 : i64,
        vernon.binding = 3 : i64
      },
      %sampler_c: !vernon.sampler {
        vernon.interface = "resource", vernon.set = 0 : i64,
        vernon.binding = 8 : i64
      }) -> (tensor<4xf32> {
        vernon.interface = "output", vernon.location = 0 : i64
      }) attributes {vernon.entry, vernon.stage = "fragment"} {
    %a0 = "vernon.intrinsic"(%texture_a, %shared_sampler, %uv)
        {name = "texture_sample"} :
        (!vernon.texture<"2d", f32>, !vernon.sampler, tensor<2xf32>)
        -> tensor<4xf32>
    %a1 = "vernon.intrinsic"(%texture_a, %shared_sampler, %uv)
        {name = "texture_sample"} :
        (!vernon.texture<"2d", f32>, !vernon.sampler, tensor<2xf32>)
        -> tensor<4xf32>
    %b = "vernon.intrinsic"(%texture_b, %shared_sampler, %uv)
        {name = "texture_sample"} :
        (!vernon.texture<"2d", f32>, !vernon.sampler, tensor<2xf32>)
        -> tensor<4xf32>
    %c = "vernon.intrinsic"(%texture_c, %sampler_c, %uv)
        {name = "texture_sample"} :
        (!vernon.texture<"2d", f32>, !vernon.sampler, tensor<2xf32>)
        -> tensor<4xf32>
    return %c : tensor<4xf32>
  }
}
)mlir";
    nlohmann::json relationReflection = validateReflection(context, relationModule);
    std::fprintf(stderr, "%s\n", relationReflection.dump(2).c_str());
    const nlohmann::json &relationEntry = entry(relationReflection, "relation");
    ASSERT_TRUE(argument(relationEntry, 1).at("kind") == "texture");
    ASSERT_TRUE(argument(relationEntry, 2).at("kind") == "sampler");
    ASSERT_TRUE(argument(relationEntry, 2).at("sampled_texture_bindings") ==
                nlohmann::json::array({{{"set", 0}, {"binding", 1}}, {{"set", 0}, {"binding", 2}}}));
    ASSERT_TRUE(!argument(relationEntry, 2).contains("sampled_texture_set"));
    ASSERT_TRUE(!argument(relationEntry, 2).contains("sampled_texture_binding"));
    ASSERT_TRUE(argument(relationEntry, 5).at("sampled_texture_bindings") ==
                nlohmann::json::array({{{"set", 0}, {"binding", 3}}}));
    ASSERT_TRUE(!argument(relationEntry, 5).contains("sampled_texture_set"));
    ASSERT_TRUE(!argument(relationEntry, 5).contains("sampled_texture_binding"));

    std::fprintf(stderr, "forwarding\n");
    constexpr std::string_view forwardingModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @forwarding(
      %condition: i1 {
        vernon.interface = "input", vernon.location = 0 : i64
      },
      %texture_a: !vernon.texture<"2d", f32> {
        vernon.interface = "resource", vernon.set = 0 : i64,
        vernon.binding = 1 : i64
      },
      %uv: tensor<2xf32> {
        vernon.interface = "input", vernon.location = 1 : i64
      },
      %sampler: !vernon.sampler {
        vernon.interface = "resource", vernon.set = 0 : i64,
        vernon.binding = 7 : i64
      },
      %texture_b: !vernon.texture<"2d", f32> {
        vernon.interface = "resource", vernon.set = 0 : i64,
        vernon.binding = 2 : i64
      }) -> (tensor<4xf32> {
        vernon.interface = "output", vernon.location = 0 : i64
      }) attributes {vernon.entry, vernon.stage = "fragment"} {
    scf.if %condition {
      %nested = "vernon.intrinsic"(%texture_a, %sampler, %uv)
          {name = "texture_sample"} :
          (!vernon.texture<"2d", f32>, !vernon.sampler, tensor<2xf32>)
          -> tensor<4xf32>
      scf.yield
    } else {
      scf.yield
    }
    %selected = arith.select %condition, %texture_a, %texture_b :
        !vernon.texture<"2d", f32>
    %conditional = scf.if %condition -> (!vernon.texture<"2d", f32>) {
      scf.yield %selected : !vernon.texture<"2d", f32>
    } else {
      scf.yield %texture_b : !vernon.texture<"2d", f32>
    }
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %looped = scf.for %i = %c0 to %c1 step %c1
        iter_args(%carried = %conditional)
        -> (!vernon.texture<"2d", f32>) {
      %loop_sample = "vernon.intrinsic"(%carried, %sampler, %uv)
          {name = "texture_sample"} :
          (!vernon.texture<"2d", f32>, !vernon.sampler, tensor<2xf32>)
          -> tensor<4xf32>
      scf.yield %carried : !vernon.texture<"2d", f32>
    }
    %while_result = scf.while (%before = %looped) :
        (!vernon.texture<"2d", f32>) -> (!vernon.texture<"2d", f32>) {
      %while_sample = "vernon.intrinsic"(%before, %sampler, %uv)
          {name = "texture_sample"} :
          (!vernon.texture<"2d", f32>, !vernon.sampler, tensor<2xf32>)
          -> tensor<4xf32>
      scf.condition(%condition) %before : !vernon.texture<"2d", f32>
    } do {
    ^bb0(%after: !vernon.texture<"2d", f32>):
      scf.yield %after : !vernon.texture<"2d", f32>
    }
    %result = "vernon.intrinsic"(%while_result, %sampler, %uv)
        {name = "texture_sample"} :
        (!vernon.texture<"2d", f32>, !vernon.sampler, tensor<2xf32>)
        -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
)mlir";
    nlohmann::json forwardingReflection = validateReflection(context, forwardingModule);
    const nlohmann::json &forwardingEntry = entry(forwardingReflection, "forwarding");
    ASSERT_TRUE(argument(forwardingEntry, 3).at("sampled_texture_bindings") ==
                nlohmann::json::array({{{"set", 0}, {"binding", 1}}, {{"set", 0}, {"binding", 2}}}));

    std::fprintf(stderr, "helper\n");
    constexpr std::string_view helperModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func private @sample_helper(
      %texture: !vernon.texture<"2d", f32>, %sampler: !vernon.sampler,
      %uv: tensor<2xf32>) -> tensor<4xf32> {
    %sample = "vernon.intrinsic"(%texture, %sampler, %uv)
        {name = "texture_sample"} :
        (!vernon.texture<"2d", f32>, !vernon.sampler, tensor<2xf32>)
        -> tensor<4xf32>
    return %sample : tensor<4xf32>
  }
  func.func @helper_entry(
      %uv: tensor<2xf32> {
        vernon.interface = "input", vernon.location = 0 : i64
      },
      %sampler: !vernon.sampler {
        vernon.interface = "resource", vernon.set = 0 : i64,
        vernon.binding = 7 : i64
      },
      %texture: !vernon.texture<"2d", f32> {
        vernon.interface = "resource", vernon.set = 0 : i64,
        vernon.binding = 4 : i64
      }) -> (tensor<4xf32> {
        vernon.interface = "output", vernon.location = 0 : i64
      }) attributes {vernon.entry, vernon.stage = "fragment"} {
    %sample = call @sample_helper(%texture, %sampler, %uv) :
        (!vernon.texture<"2d", f32>, !vernon.sampler, tensor<2xf32>)
        -> tensor<4xf32>
    return %sample : tensor<4xf32>
  }
}
)mlir";
    nlohmann::json helperReflection = validateReflection(context, helperModule);
    const nlohmann::json &helperEntry = entry(helperReflection, "helper_entry");
    ASSERT_TRUE(helperReflection.at("entries").size() == 1);
    ASSERT_TRUE(argument(helperEntry, 1).at("sampled_texture_bindings") ==
                nlohmann::json::array({{{"set", 0}, {"binding", 4}}}));

    std::fprintf(stderr, "ambiguity\n");
    constexpr std::string_view ambiguousSamplerModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @ambiguous(
      %condition: i1 {
        vernon.interface = "input", vernon.location = 0 : i64
      },
      %texture: !vernon.texture<"2d", f32> {
        vernon.interface = "resource", vernon.set = 0 : i64,
        vernon.binding = 1 : i64
      },
      %sampler_a: !vernon.sampler {
        vernon.interface = "resource", vernon.set = 0 : i64,
        vernon.binding = 7 : i64
      },
      %uv: tensor<2xf32> {
        vernon.interface = "input", vernon.location = 1 : i64
      },
      %sampler_b: !vernon.sampler {
        vernon.interface = "resource", vernon.set = 0 : i64,
        vernon.binding = 8 : i64
      }) -> (tensor<4xf32> {
        vernon.interface = "output", vernon.location = 0 : i64
      }) attributes {vernon.entry, vernon.stage = "fragment"} {
    %selected = arith.select %condition, %sampler_a, %sampler_b :
        !vernon.sampler
    %sample = "vernon.intrinsic"(%texture, %selected, %uv)
        {name = "texture_sample"} :
        (!vernon.texture<"2d", f32>, !vernon.sampler, tensor<2xf32>)
        -> tensor<4xf32>
    return %sample : tensor<4xf32>
  }
}
)mlir";
    expectDiagnostic(context, ambiguousSamplerModule, "may use multiple sampler entry arguments (#2, #4)");

    std::fprintf(stderr, "unknown\n");
    constexpr std::string_view unknownProvenanceModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @unknown(
      %raw: i64 {
        vernon.interface = "uniform"
      },
      %sampler: !vernon.sampler {
        vernon.interface = "resource", vernon.set = 0 : i64,
        vernon.binding = 7 : i64
      },
      %uv: tensor<2xf32> {
        vernon.interface = "input", vernon.location = 0 : i64
      }) -> (tensor<4xf32> {
        vernon.interface = "output", vernon.location = 0 : i64
      }) attributes {vernon.entry, vernon.stage = "fragment"} {
    %unknown_texture = builtin.unrealized_conversion_cast %raw :
        i64 to !vernon.texture<"2d", f32>
    %sample = "vernon.intrinsic"(%unknown_texture, %sampler, %uv)
        {name = "texture_sample"} :
        (!vernon.texture<"2d", f32>, !vernon.sampler, tensor<2xf32>)
        -> tensor<4xf32>
    return %sample : tensor<4xf32>
  }
}
)mlir";
    expectDiagnostic(context, unknownProvenanceModule, "argument #0 has no finite canonical Value ABI layout");

    std::fprintf(stderr, "intrinsic\n");
    expectDiagnostic(context, R"mlir(module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
        func.func @bad(%texture: !vernon.texture<"2d", f32>,
                       %coordinates: tensor<2xf32>) -> tensor<4xf32> {
          %sample = "vernon.intrinsic"(%texture, %coordinates)
              {name = "texture_sample"} :
              (!vernon.texture<"2d", f32>, tensor<2xf32>) -> tensor<4xf32>
          return %sample : tensor<4xf32>
        }
      })mlir",
                     "texture_sample requires texture, sampler, coordinates, "
                     "and optional lod");

    struct InvalidSwizzleCase {
        std::string_view module;
        std::string_view diagnostic;
    };
    for (const InvalidSwizzleCase &test : {
             InvalidSwizzleCase{R"mlir(module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
                 func.func @empty(%input: tensor<4xf32>) -> f32 {
                   %result = "vernon.swizzle"(%input) {mask = ""} :
                       (tensor<4xf32>) -> f32
                   return %result : f32
                 }
               })mlir",
                                "requires a non-empty component mask"},
             InvalidSwizzleCase{R"mlir(module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
                 func.func @invalid(%input: tensor<4xf32>) -> f32 {
                   %result = "vernon.swizzle"(%input) {mask = "q"} :
                       (tensor<4xf32>) -> f32
                   return %result : f32
                 }
               })mlir",
                                "contains invalid component 'q'"},
             InvalidSwizzleCase{R"mlir(module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
                 func.func @bounds(%input: tensor<3xf32>) -> f32 {
                   %result = "vernon.swizzle"(%input) {mask = "a"} :
                       (tensor<3xf32>) -> f32
                   return %result : f32
                 }
               })mlir",
                                "component 'a' is out of bounds for input width 3"},
             InvalidSwizzleCase{R"mlir(module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
                 func.func @rank(%input: tensor<2x2xf32>) -> f32 {
                   %result = "vernon.swizzle"(%input) {mask = "x"} :
                       (tensor<2x2xf32>) -> f32
                   return %result : f32
                 }
               })mlir",
                                "requires a rank-one tensor or vector input"},
             InvalidSwizzleCase{R"mlir(module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
                 func.func @scalar_result_type(%input: tensor<4xf32>) -> i32 {
                   %result = "vernon.swizzle"(%input) {mask = "r"} :
                       (tensor<4xf32>) -> i32
                   return %result : i32
                 }
               })mlir",
                                "single-component result must have input element type 'f32'"},
             InvalidSwizzleCase{R"mlir(module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
                 func.func @result_shape(%input: tensor<4xf32>)
                     -> tensor<2xf32> {
                   %result = "vernon.swizzle"(%input) {mask = "rgb"} :
                       (tensor<4xf32>) -> tensor<2xf32>
                   return %result : tensor<2xf32>
                 }
               })mlir",
                                "result width 2 does not match mask length 3"},
             InvalidSwizzleCase{R"mlir(module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
                 func.func @result_type(%input: tensor<4xf32>)
                     -> tensor<3xi32> {
                   %result = "vernon.swizzle"(%input) {mask = "rgb"} :
                       (tensor<4xf32>) -> tensor<3xi32>
                   return %result : tensor<3xi32>
                 }
               })mlir",
                                "result element type 'i32' does not match input element type "
                                "'f32'"},
         }) {
        result = vernonCompilerValidateMlir(context, test.module.data(), test.module.size());
        ASSERT_TRUE(result);
        ASSERT_TRUE(vernonCompileResultGetStatus(result) != VERNON_STATUS_OK);
        const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
        const std::string_view diagnosticView(diagnostics.data, diagnostics.size);
        if (diagnosticView.find(test.diagnostic) == std::string_view::npos)
            std::fprintf(stderr, "missing diagnostic '%.*s' in:\n%.*s\n", static_cast<int>(test.diagnostic.size()),
                         test.diagnostic.data(), static_cast<int>(diagnosticView.size()), diagnosticView.data());
        ASSERT_TRUE(diagnosticView.find(test.diagnostic) != std::string_view::npos);
        vernonCompileResultDestroy(result);
    }

    vernonCompilerDestroy(context);
}
