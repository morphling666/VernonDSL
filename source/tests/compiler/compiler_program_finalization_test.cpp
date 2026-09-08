#include "VernonProgramCapabilities.h"
#include "compiler_program_boundary.h"
#include "compiler_program_derivative.h"
#include "compiler_program_graph.h"
#include "compiler_program_implementation.h"
#include "compiler_program_publication.h"
#include "compiler_program_serializer.h"
#include "compiler_program_stage.h"
#include "compiler_program_storage.h"
#include "compiler_program_tape.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"

#include <gtest/gtest.h>
#include <string>

namespace {

TEST(ProgramCapabilities, CurrentReleaseUsesStableUnsupportedDecisions) {
    using namespace vernon::program_capabilities;
    EXPECT_TRUE(get(Id::ComputeVjp).supported);
    EXPECT_TRUE(get(Id::GraphicsTextureSampling).supported);
    EXPECT_FALSE(get(Id::ComputeSamplerBinding).supported);
    EXPECT_EQ(get(Id::ComputeSamplerBinding).diagnosticCode, "PROGRAM_COMPUTE_SAMPLER_UNSUPPORTED");
    EXPECT_FALSE(get(Id::GraphicsVjp).supported);
    EXPECT_FALSE(get(Id::OpaqueResourceVjp).supported);
    EXPECT_TRUE(get(Id::CpuF16).supported);
    EXPECT_FALSE(get(Id::GpuF16).supported);
}

llvm::json::Value parse(llvm::StringRef text) {
    auto parsed = llvm::json::parse(text);
    EXPECT_TRUE(bool(parsed)) << toString(parsed.takeError());
    return parsed ? std::move(*parsed) : llvm::json::Object{};
}

llvm::json::Value executionWithBindings(llvm::StringRef bindings) {
    return parse(llvm::Twine(R"({
  "graphs": [{
    "name": "backward",
    "direction": "backward",
    "nodes": [{
      "id": 0,
      "stage": "backward:0",
      "bindings": )")
                     .concat(bindings)
                     .concat(R"(,
      "operands": [0, 1, 2],
      "results": [3]
    }]
  }]
})")
                     .str());
}

llvm::json::Value requestWithBindings(llvm::StringRef bindings) {
    return parse(llvm::Twine(R"({
  "id": "backward:0",
  "graph": "backward",
  "bindings": )")
                     .concat(bindings)
                     .concat("}")
                     .str());
}

const char *kBindings = R"([
  {
    "parameter": "cotangent.divergence",
    "value": 1,
    "autodiff_role": "cotangent",
    "autodiff_source": "divergence"
  },
  {
    "parameter": "cotangent.pressure_a",
    "value": 2,
    "autodiff_role": "cotangent",
    "autodiff_source": "pressure_a"
  },
  {
    "parameter": "gradient.advected_velocity",
    "value": 3,
    "autodiff_role": "gradient",
    "autodiff_source": "advected_velocity"
  }
])";

llvm::json::Value activeKernelAbi() {
    return parse(R"({
  "arguments": [
    {
      "vernon.source_name": "cotangent.divergence",
      "vernon.autodiff_role": "cotangent",
      "vernon.autodiff_source": "divergence"
    },
    {
      "vernon.source_name": "advected_velocity",
      "vernon.autodiff_role": "gradient",
      "vernon.autodiff_source": "advected_velocity"
    }
  ]
})");
}

bool hasParameter(const llvm::json::Array *bindings, llvm::StringRef name) {
    if (!bindings)
        return false;
    for (const llvm::json::Value &value : *bindings)
        if (const llvm::json::Object *binding = value.getAsObject())
            if (binding->getString("parameter") == name)
                return true;
    return false;
}

bool hasEndpoint(const llvm::json::Array *bindings, llvm::StringRef name) {
    if (!bindings)
        return false;
    for (const llvm::json::Value &value : *bindings)
        if (const llvm::json::Object *binding = value.getAsObject())
            if (binding->getString("endpoint") == name)
                return true;
    return false;
}

} // namespace

TEST(CompilerProgramDerivativePlanner, SelectsLongestPrimalAndTypedValuePath) {
    using namespace vernon::compiler;
    const std::vector<ProgramBoundaryIdentity> boundaries{
        {0, "result", ProgramBoundaryRole::Output},
        {1, "result.inner", ProgramBoundaryRole::Output},
        {2, "result.inner.2", ProgramBoundaryRole::Cotangent},
    };
    std::vector<ProgramDerivativeProjectionPlan> projections;
    std::string error;
    ASSERT_TRUE(planProgramDerivativeProjections(boundaries, projections, error)) << error;
    ASSERT_EQ(projections.size(), 1u);
    EXPECT_EQ(projections[0].primal.slot, 1);
    ASSERT_EQ(projections[0].valuePath.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<uint32_t>(projections[0].valuePath[0]));
    EXPECT_EQ(std::get<uint32_t>(projections[0].valuePath[0]), 2u);
}

TEST(CompilerProgramDerivativePlanner, RejectsUnprojectableDerivative) {
    using namespace vernon::compiler;
    const std::vector<ProgramBoundaryIdentity> boundaries{
        {0, "source", ProgramBoundaryRole::Input},
        {1, "other", ProgramBoundaryRole::Gradient},
    };
    std::vector<ProgramDerivativeProjectionPlan> projections;
    std::string error;
    EXPECT_FALSE(planProgramDerivativeProjections(boundaries, projections, error));
    EXPECT_NE(error.find("cannot project derivative boundary"), std::string::npos);
}

TEST(CompilerProgramBoundaryPlanner, SerializesExactStorageBoundaryContract) {
    using namespace vernon::compiler;
    llvm::json::Value document = parse(R"({
      "signature": {
        "inputs": [{"path": "state", "value": 0}],
        "outputs": [{"path": "result", "value": 1}],
        "cotangents": [],
        "gradients": []
      },
      "values": [
        {"id": 0, "type": "tensor<1xf32>", "shape": [1], "storage": 0},
        {"id": 1, "type": "tensor<1xf32>", "shape": [1], "storage": 0}
      ],
      "storages": [{
        "id": 0,
        "descriptor": {"tag": "buffer", "byte_length": 4, "alignment": 4, "memory": "device", "usage": ["storage"]}
      }],
      "graphs": [{
        "direction": "forward",
        "nodes": [{"accesses": [{"storage": 0, "access": "read_write"}]}]
      }]
    })");
    llvm::json::Object *root = document.getAsObject();
    ASSERT_NE(root, nullptr);
    ProgramBoundaryPlan boundaries;
    std::string error;
    ASSERT_TRUE(planProgramBoundaries(*root->getObject("signature"), *root->getArray("values"),
                                      *root->getArray("storages"), *root->getArray("graphs"), boundaries, error))
        << error;
    ASSERT_EQ(boundaries.slots.size(), 2u);
    EXPECT_EQ(boundaries.slots[0].access, ProgramBoundaryAccess::ReadWrite);
    EXPECT_EQ(boundaries.slots[1].direction, ProgramBoundaryDirection::Output);
    llvm::json::Array serialized = serializeProgramBoundarySlots(boundaries);
    EXPECT_EQ(serialized, *parse(R"([
                {
                  "id": 0,
                  "path": "state",
                  "value": 0,
                  "role": "input",
                  "direction": "input",
                  "category": "storage_view",
                  "access": "read_write",
                  "logical_type": "tensor<1xf32>",
                  "outer_shape": [1],
                  "alias_owner": "storage:0",
                  "storage_id": 0,
                  "storage_descriptor": {
                    "tag": "buffer",
                    "byte_length": 4,
                    "alignment": 4,
                    "memory": "device",
                    "usage": ["storage"]
                  }
                },
                {
                  "id": 1,
                  "path": "result",
                  "value": 1,
                  "role": "output",
                  "direction": "output",
                  "category": "storage_view",
                  "access": "write",
                  "logical_type": "tensor<1xf32>",
                  "outer_shape": [1],
                  "alias_owner": "storage:0",
                  "storage_id": 0,
                  "storage_descriptor": {
                    "tag": "buffer",
                    "byte_length": 4,
                    "alignment": 4,
                    "memory": "device",
                    "usage": ["storage"]
                  },
                  "publication": "commit_after_success"
                }
              ])")
                               .getAsArray());
}

TEST(CompilerProgramTapePlanner, ProducerAndConsumerComeFromCanonicalGraphs) {
    using namespace vernon::compiler;
    llvm::json::Value values = parse(R"([
      {
        "id": 4,
        "type": "!vernon.ad_tape<16>",
        "origin": {"tag": "node_result", "graph": "forward", "node": 0}
      }
    ])");
    llvm::json::Value graphs = parse(R"([
      {"direction": "forward"},
      {"direction": "backward"}
    ])");
    const std::vector<ProgramTapePlan> plans = planProgramTapes(*values.getAsArray(), *graphs.getAsArray());
    ASSERT_EQ(plans.size(), 1u);
    EXPECT_TRUE(plans[0].forwardProducer);
    EXPECT_TRUE(plans[0].backwardConsumer);
    EXPECT_EQ(plans[0].requiredCarriers,
              std::vector<ProgramTapeCarrier>({ProgramTapeCarrier::TapeData, ProgramTapeCarrier::ReplaySegment}));
}

TEST(CompilerProgramSerializer, SerializesPlanWithoutDerivingFields) {
    using namespace vernon::compiler;
    CanonicalProgramSerializationPlan plan;
    plan.stages["stage"] = llvm::json::Object{{"operation", "compute"}};
    plan.abi["boundary_slots"] = llvm::json::Array();
    llvm::json::Object serialized = serializeCanonicalProgram(std::move(plan));
    ASSERT_NE(serialized.getObject("stages"), nullptr);
    EXPECT_EQ(serialized.get("signature"), nullptr);
    ASSERT_NE(serialized.getObject("abi"), nullptr);
    EXPECT_EQ(serialized.get("residual_contract"), nullptr);
}

TEST(CompilerProgramLinker, ProducesTypedBoundaryIdsAndChecksStageCoverage) {
    using namespace vernon::compiler;
    llvm::json::Value graphs = parse(R"([
      {
        "name": "forward",
        "direction": "forward",
        "arguments": [3],
        "results": [4],
        "nodes": [{"id": 0}]
      }
    ])");
    CanonicalProgramLinkPlan plan;
    std::string error;
    ASSERT_TRUE(linkCanonicalProgram(*graphs.getAsArray(), 1, plan, error)) << error;
    ASSERT_EQ(plan.graphs.size(), 1u);
    EXPECT_EQ(plan.graphs[0].arguments, std::vector<int64_t>({3}));
    EXPECT_EQ(plan.graphs[0].results, std::vector<int64_t>({4}));
    EXPECT_FALSE(linkCanonicalProgram(*graphs.getAsArray(), 0, plan, error));
    EXPECT_NE(error.find("do not exactly cover"), std::string::npos);
}

TEST(CompilerProgramStorageAliasPlanner, ProducesTypedOwnerGroupsAndRejectsCycles) {
    using namespace vernon::compiler;
    ProgramStorageAliasPlan plan;
    std::string error;
    ASSERT_TRUE(planProgramStorageAliases({{0, 0}, {1, 0}, {2, 2}}, plan, error)) << error;
    EXPECT_EQ(plan.ownerByValue.at(1), ProgramStorageOwnerId{0});
    EXPECT_EQ(plan.versionsByOwner.at(ProgramStorageOwnerId{0}), std::vector<int64_t>({0, 1}));
    EXPECT_FALSE(planProgramStorageAliases({{0, 1}, {1, 0}}, plan, error));
    EXPECT_NE(error.find("cycle"), std::string::npos);
}

TEST(CompilerProgramStoragePlanner, CapturedDynamicExtentUsesCanonicalValueIdentity) {
    using namespace vernon::compiler;
    llvm::json::Value values = parse(R"([
      {"id": 6, "shape": [-1]},
      {"id": 9, "shape": [-1], "like": 6}
    ])");
    llvm::json::Value shape = parse(R"([-1])");
    llvm::json::Array extents;
    std::string error;
    ASSERT_TRUE(planOwnedDynamicExtents(*values.getAsArray(), 9, "backward", true, *shape.getAsArray(), {}, {6}, {}, {},
                                        extents, error))
        << error;
    EXPECT_EQ(extents, *parse(R"([
                {"dimension": {"control": {"value": 6}, "axis": 0}}
              ])")
                            .getAsArray());
}

TEST(CompilerProgramFinalization, OmitsInactiveNestedVjpCotangents) {
    llvm::json::Value execution = executionWithBindings(kBindings);
    llvm::json::Value request = requestWithBindings(kBindings);
    llvm::json::Value compiled = activeKernelAbi();
    std::string error;
    ASSERT_TRUE(vernon::compiler::normalizeProgramImplementationAbi(*execution.getAsObject(), *request.getAsObject(),
                                                                    *compiled.getAsObject(), error))
        << error;
    const llvm::json::Array *requestBindings = request.getAsObject()->getArray("bindings");
    EXPECT_TRUE(hasParameter(requestBindings, "cotangent.divergence"));
    EXPECT_FALSE(hasParameter(requestBindings, "cotangent.pressure_a"));
    EXPECT_TRUE(hasEndpoint(requestBindings, "advected_velocity"));
    const llvm::json::Array *nodeBindings = (*execution.getAsObject()->getArray("graphs"))[0]
                                                .getAsObject()
                                                ->getArray("nodes")
                                                ->front()
                                                .getAsObject()
                                                ->getArray("bindings");
    EXPECT_TRUE(hasParameter(nodeBindings, "cotangent.divergence"));
    EXPECT_FALSE(hasParameter(nodeBindings, "cotangent.pressure_a"));
}

TEST(CompilerProgramFinalization, RejectsUnmappedKernelGradient) {
    llvm::json::Value execution = executionWithBindings(kBindings);
    llvm::json::Value request = requestWithBindings(kBindings);
    llvm::json::Value compiled = parse(R"({
  "arguments": [
    {
      "vernon.source_name": "cotangent.divergence",
      "vernon.autodiff_role": "cotangent",
      "vernon.autodiff_source": "divergence"
    }
  ]
})");
    std::string error;
    EXPECT_FALSE(vernon::compiler::normalizeProgramImplementationAbi(*execution.getAsObject(), *request.getAsObject(),
                                                                     *compiled.getAsObject(), error));
    EXPECT_EQ(error, "compiled kernel ABI does not provide Program gradient 'advected_velocity'");
}

TEST(CompilerProgramFinalization, RejectsKernelCotangentProgramDidNotBind) {
    llvm::json::Value execution = executionWithBindings(R"([
  {
    "parameter": "gradient.advected_velocity",
    "value": 3,
    "autodiff_role": "gradient",
    "autodiff_source": "advected_velocity"
  }
])");
    llvm::json::Value request = requestWithBindings(R"([
  {
    "parameter": "gradient.advected_velocity",
    "value": 3,
    "autodiff_role": "gradient",
    "autodiff_source": "advected_velocity"
  }
])");
    llvm::json::Value compiled = activeKernelAbi();
    std::string error;
    EXPECT_FALSE(vernon::compiler::normalizeProgramImplementationAbi(*execution.getAsObject(), *request.getAsObject(),
                                                                     *compiled.getAsObject(), error));
    EXPECT_EQ(error, "compiled kernel ABI requires unmapped Program value 'cotangent.divergence'");
}

TEST(CompilerProgramFinalization, ExpandsLogicalTapeBindingIntoPhysicalCarrierBundle) {
    const char *bindings = R"([
  {
    "parameter": "tape",
    "value": 0,
    "autodiff_role": "tape",
    "autodiff_source": "tape"
  }
])";
    llvm::json::Value execution = executionWithBindings(bindings);
    llvm::json::Value request = requestWithBindings(bindings);
    llvm::json::Value compiled = parse(R"({
  "arguments": [
    {
      "vernon.source_name": "__vernon_ad_tape",
      "vernon.autodiff_role": "tape"
    },
    {
      "vernon.source_name": "__vernon_ad_segment",
      "vernon.autodiff_role": "replay_segment"
    },
    {
      "vernon.source_name": "__vernon_ad_status",
      "vernon.autodiff_role": "replay_status"
    }
  ]
})");
    std::string error;
    ASSERT_TRUE(vernon::compiler::normalizeProgramImplementationAbi(*execution.getAsObject(), *request.getAsObject(),
                                                                    *compiled.getAsObject(), error))
        << error;
    const llvm::json::Array *requestBindings = request.getAsObject()->getArray("bindings");
    EXPECT_TRUE(hasParameter(requestBindings, "__vernon_ad_tape"));
    EXPECT_TRUE(hasParameter(requestBindings, "__vernon_ad_segment"));
    EXPECT_TRUE(hasParameter(requestBindings, "__vernon_ad_status"));
}

TEST(CompilerProgramFinalization, AcceptsOnlyDeclaredCotangentInvocationCarrierShape) {
    llvm::json::Value logical = parse(R"([-1, -1])");
    llvm::json::Value direct = parse(R"([-1, -1])");
    llvm::json::Value broadcast = parse(R"([-1, -1, -1])");
    EXPECT_TRUE(
        vernon::compiler::compatibleProgramBindingShape("cotangent", "", logical.getAsArray(), direct.getAsArray()));
    EXPECT_FALSE(
        vernon::compiler::compatibleProgramBindingShape("cotangent", "", logical.getAsArray(), broadcast.getAsArray()));
    EXPECT_TRUE(vernon::compiler::compatibleProgramBindingShape("cotangent", "invocation_linear", logical.getAsArray(),
                                                                broadcast.getAsArray()));
    EXPECT_FALSE(vernon::compiler::compatibleProgramBindingShape("gradient", "invocation_linear", logical.getAsArray(),
                                                                 broadcast.getAsArray()));
    EXPECT_FALSE(vernon::compiler::compatibleProgramBindingShape("primal", "invocation_linear", logical.getAsArray(),
                                                                 broadcast.getAsArray()));
}

TEST(CompilerProgramFinalization, ResolvesRootValueLeafWithoutSyntheticPath) {
    llvm::json::Value layout = parse(R"({
  "leaves": [
    {
      "path": [],
      "dtype": "f32",
      "byte_offset": 0,
      "scalar_count": 1,
      "shape": []
    }
  ]
})");
    ASSERT_NE(layout.getAsObject(), nullptr);
    EXPECT_EQ(vernon::compiler::resolveProgramValueLeafIndex(*layout.getAsObject(), "output_loss", "output_loss"), 0u);
    EXPECT_FALSE(
        vernon::compiler::resolveProgramValueLeafIndex(*layout.getAsObject(), "output_loss", "other").has_value());
}

TEST(CompilerProgramFinalization, BindsGradientDestByRoleNotPrimalName) {
    const char *bindings = R"([
  {
    "parameter": "primal.projected_velocity",
    "value": 1,
    "autodiff_role": "retained_primal",
    "autodiff_source": "projected_velocity"
  },
  {
    "parameter": "gradient.projected_velocity",
    "value": 2,
    "autodiff_role": "gradient",
    "autodiff_source": "projected_velocity"
  }
])";
    llvm::json::Value execution = executionWithBindings(bindings);
    llvm::json::Value request = requestWithBindings(bindings);
    llvm::json::Value compiled = parse(R"({
  "arguments": [
    {
      "vernon.source_name": "projected_velocity"
    },
    {
      "vernon.source_name": "primal.projected_velocity",
      "vernon.autodiff_role": "retained_primal",
      "vernon.autodiff_source": "projected_velocity"
    },
    {
      "vernon.source_name": "projected_velocity.x",
      "vernon.autodiff_role": "gradient",
      "vernon.autodiff_source": "projected_velocity"
    }
  ]
})");
    std::string error;
    ASSERT_TRUE(vernon::compiler::normalizeProgramImplementationAbi(*execution.getAsObject(), *request.getAsObject(),
                                                                    *compiled.getAsObject(), error))
        << error;
    const llvm::json::Array *requestBindings = request.getAsObject()->getArray("bindings");
    EXPECT_TRUE(hasParameter(requestBindings, "primal.projected_velocity"));
    EXPECT_TRUE(hasEndpoint(requestBindings, "projected_velocity.x"));
    EXPECT_FALSE(hasParameter(requestBindings, "projected_velocity"));
    EXPECT_TRUE(hasParameter(requestBindings, "gradient.projected_velocity"));
    const llvm::json::Array *nodeBindings = (*execution.getAsObject()->getArray("graphs"))[0]
                                                .getAsObject()
                                                ->getArray("nodes")
                                                ->front()
                                                .getAsObject()
                                                ->getArray("bindings");
    EXPECT_TRUE(hasEndpoint(nodeBindings, "projected_velocity.x"));
    EXPECT_FALSE(hasParameter(nodeBindings, "projected_velocity"));
}

TEST(CompilerProgramFinalization, ExpandsAggregateGradientBindingToCanonicalLeafNames) {
    const char *bindings = R"([
  {
    "parameter": "gradient.particles",
    "value": 2,
    "autodiff_role": "gradient",
    "autodiff_source": "particles"
  }
])";
    llvm::json::Value execution = executionWithBindings(bindings);
    llvm::json::Value values = parse(R"([
  {"id": 0},
  {"id": 1},
  {
    "id": 2,
    "value_layout": {
      "leaves": [
        {"path": ["mass"], "dtype": "f32"},
        {"path": ["velocity"], "dtype": "f32"}
      ]
    }
  }
])");
    (*execution.getAsObject())["values"] = std::move(*values.getAsArray());
    llvm::json::Value request = requestWithBindings(bindings);
    llvm::json::Value compiled = parse(R"({
  "arguments": [
    {
      "vernon.source_name": "particles.mass",
      "vernon.autodiff_role": "gradient",
      "vernon.autodiff_source": "particles"
    },
    {
      "vernon.source_name": "particles.velocity",
      "vernon.autodiff_role": "gradient",
      "vernon.autodiff_source": "particles"
    }
  ]
})");
    std::string error;
    ASSERT_TRUE(vernon::compiler::normalizeProgramImplementationAbi(*execution.getAsObject(), *request.getAsObject(),
                                                                    *compiled.getAsObject(), error))
        << error;
    const llvm::json::Array *requestBindings = request.getAsObject()->getArray("bindings");
    ASSERT_EQ(requestBindings->size(), 2u);
    EXPECT_TRUE(hasEndpoint(requestBindings, "particles.mass"));
    EXPECT_TRUE(hasEndpoint(requestBindings, "particles.velocity"));
    for (const llvm::json::Value &binding : *requestBindings) {
        EXPECT_EQ(binding.getAsObject()->getInteger("value"), 2);
        const std::optional<llvm::StringRef> endpoint = binding.getAsObject()->getString("endpoint");
        ASSERT_TRUE(endpoint);
        EXPECT_EQ(binding.getAsObject()->getInteger("leaf"), *endpoint == "particles.mass" ? 0 : 1);
    }

    const llvm::json::Array *nodeBindings = (*execution.getAsObject()->getArray("graphs"))[0]
                                                .getAsObject()
                                                ->getArray("nodes")
                                                ->front()
                                                .getAsObject()
                                                ->getArray("bindings");
    ASSERT_EQ(nodeBindings->size(), 2u);
    EXPECT_TRUE(hasEndpoint(nodeBindings, "particles.mass"));
    EXPECT_TRUE(hasEndpoint(nodeBindings, "particles.velocity"));
    for (const llvm::json::Value &binding : *nodeBindings) {
        EXPECT_EQ(binding.getAsObject()->getInteger("value"), 2);
        const std::optional<llvm::StringRef> endpoint = binding.getAsObject()->getString("endpoint");
        ASSERT_TRUE(endpoint);
        EXPECT_EQ(binding.getAsObject()->getInteger("leaf"), *endpoint == "particles.mass" ? 0 : 1);
    }
}

TEST(CompilerProgramFinalization, ResolvesAggregateLeafByCanonicalPath) {
    llvm::json::Value layout = parse(R"({
  "leaves": [
    {"path": ["mass"], "dtype": "f32"},
    {"path": ["nested", "mass"], "dtype": "f32"}
  ]
})");
    const llvm::json::Object &object = *layout.getAsObject();
    EXPECT_EQ(vernon::compiler::resolveProgramValueLeafIndex(object, "particles", "particles.mass"), 0);
    EXPECT_EQ(vernon::compiler::resolveProgramValueLeafIndex(object, "particles", "particles.nested.mass"), 1);
    EXPECT_FALSE(vernon::compiler::resolveProgramValueLeafIndex(object, "particles", "other.mass"));
}
