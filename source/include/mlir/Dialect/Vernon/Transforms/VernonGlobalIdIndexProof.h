#pragma once

#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cstdint>
#include <optional>

namespace mlir::vernon {

enum class IndexOwnershipDomain {
    None,
    Invocation,
    Workgroup,
};

struct GlobalIdAffineIndex {
    std::array<int64_t, 3> coefficients{};
    int64_t constant{};
    unsigned dimension{};
    bool scalarGlobalId{};
    IndexOwnershipDomain domain{IndexOwnershipDomain::Invocation};

    bool operator==(const GlobalIdAffineIndex &other) const;
};

using GlobalIdAffineIndexTuple = SmallVector<GlobalIdAffineIndex>;

struct ConditionalIndexProof {
    GlobalIdAffineIndexTuple normalizedIndices;
    std::array<bool, 3> coveredAxes{};
    SmallVector<unsigned> unitGridAxes;
    IndexOwnershipDomain ownership{IndexOwnershipDomain::None};
};

std::optional<GlobalIdAffineIndex> normalizeGlobalIdAffineIndex(Value value);

bool usesScalarGlobalInvocationId(ValueRange values);

std::optional<ConditionalIndexProof> proveInvocationOwnedIndex(ValueRange indices);
std::optional<ConditionalIndexProof> proveStrictInvocationOwnedIndex(ValueRange indices);
std::optional<ConditionalIndexProof> proveLeaderGuardedWorkgroupOwnedIndex(Operation *operation, ValueRange indices);

} // namespace mlir::vernon
