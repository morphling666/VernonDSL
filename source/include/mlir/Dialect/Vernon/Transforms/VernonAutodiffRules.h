#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"

#include <functional>
#include <optional>
#include <string>

namespace mlir::vernon {

class VernonAutodiffAnalysisResult;

enum class AutodiffPrimalKind {
    Operand,
    Result,
};

struct AutodiffPrimalRequirement {
    AutodiffPrimalKind kind{AutodiffPrimalKind::Operand};
    unsigned index{};

    static AutodiffPrimalRequirement operand(unsigned index) {
        return AutodiffPrimalRequirement{AutodiffPrimalKind::Operand, index};
    }
    static AutodiffPrimalRequirement result(unsigned index) {
        return AutodiffPrimalRequirement{AutodiffPrimalKind::Result, index};
    }

    bool operator==(const AutodiffPrimalRequirement &other) const { return kind == other.kind && index == other.index; }
};

/// The local use-def edges through which differentiable activity propagates.
struct AutodiffRuleActivity {
    SmallVector<unsigned> operandIndices;
    SmallVector<unsigned> resultIndices;
};

/// Inputs supplied by a semantic reverse transform. Primal operands/results may
/// already be promoted to the derivative type.
struct AutodiffVjpBuildContext {
    OpBuilder &builder;
    Location location;
    ValueRange primalOperands;
    ValueRange primalResults;
    ValueRange resultCotangents;
};

/// Inputs reserved for the future forward transform. Keeping this callback in
/// the rule object avoids introducing a second registry when JVP is enabled.
struct AutodiffJvpBuildContext {
    OpBuilder &builder;
    Location location;
    ValueRange primalOperands;
    ValueRange primalResults;
    ValueRange tangentOperands;
};

using AutodiffVjpBuilder =
    std::function<LogicalResult(Operation *, const AutodiffVjpBuildContext &, SmallVectorImpl<Value> &)>;
using AutodiffJvpBuilder =
    std::function<LogicalResult(Operation *, const AutodiffJvpBuildContext &, SmallVectorImpl<Value> &)>;
using AutodiffRuleVerifier = std::function<LogicalResult(Operation *)>;

class DifferentiationRule {
public:
    DifferentiationRule(std::string registryKey, std::optional<unsigned> operandCount, unsigned resultCount,
                        SmallVector<AutodiffPrimalRequirement> vjpPrimalRequirements, AutodiffVjpBuilder vjpBuilder,
                        AutodiffJvpBuilder jvpBuilder = {}, AutodiffRuleVerifier verifier = {});

    StringRef getRegistryKey() const { return registryKey; }
    ArrayRef<AutodiffPrimalRequirement> getVjpPrimalRequirements() const { return vjpPrimalRequirements; }
    bool hasJvpBuilder() const { return static_cast<bool>(jvpBuilder); }

    FailureOr<AutodiffRuleActivity> classifyActivity(Operation *operation) const;
    LogicalResult verifyCompatibility(Operation *operation) const;
    FailureOr<SmallVector<Value>> buildVjp(Operation *operation, const AutodiffVjpBuildContext &context) const;
    FailureOr<SmallVector<Value>> buildJvp(Operation *operation, const AutodiffJvpBuildContext &context) const;

private:
    std::string registryKey;
    std::optional<unsigned> operandCount;
    unsigned resultCount{};
    SmallVector<AutodiffPrimalRequirement> vjpPrimalRequirements;
    AutodiffVjpBuilder vjpBuilder;
    AutodiffJvpBuilder jvpBuilder;
    AutodiffRuleVerifier verifier;
};

/// A transform-owned registry. It deliberately has no global mutable state, so
/// tests and future passes may add external rules without changing transforms.
class VernonAutodiffRuleRegistry {
public:
    LogicalResult registerRule(DifferentiationRule rule);
    const DifferentiationRule *lookup(StringRef registryKey) const;
    const DifferentiationRule *lookup(Operation *operation) const;
    SmallVector<StringRef> getRegisteredKeys() const;

private:
    llvm::StringMap<DifferentiationRule> rules;
};

VernonAutodiffRuleRegistry createDefaultAutodiffRuleRegistry();

/// Checks every active differentiable operation after shared activity
/// analysis. Structural operations and constants are owned by their transforms,
/// not by local differentiation rules.
LogicalResult verifyAutodiffRuleCoverage(const VernonAutodiffAnalysisResult &analysis,
                                         const VernonAutodiffRuleRegistry &registry);

} // namespace mlir::vernon
