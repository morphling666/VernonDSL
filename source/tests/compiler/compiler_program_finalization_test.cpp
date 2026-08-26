#include "compiler_program_finalization.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"

#include <gtest/gtest.h>
#include <string>

namespace {

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

} // namespace

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
    EXPECT_TRUE(hasParameter(requestBindings, "advected_velocity"));
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
    EXPECT_TRUE(hasParameter(requestBindings, "projected_velocity.x"));
    EXPECT_FALSE(hasParameter(requestBindings, "projected_velocity"));
    EXPECT_FALSE(hasParameter(requestBindings, "gradient.projected_velocity"));
    const llvm::json::Array *nodeBindings = (*execution.getAsObject()->getArray("graphs"))[0]
                                                .getAsObject()
                                                ->getArray("nodes")
                                                ->front()
                                                .getAsObject()
                                                ->getArray("bindings");
    EXPECT_TRUE(hasParameter(nodeBindings, "projected_velocity.x"));
    EXPECT_FALSE(hasParameter(nodeBindings, "projected_velocity"));
}
