#include "mlir/Dialect/Vernon/Transforms/VernonGlobalIdIndexProof.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

namespace mlir::vernon {
namespace {

struct NormalizationResult {
    std::optional<GlobalIdAffineIndex> index;
    bool invalidArithmetic{};
};

Value transparentIntegerValue(Value value) {
    while (auto bitcast = value.getDefiningOp<arith::BitcastOp>()) {
        auto sourceType = dyn_cast<IntegerType>(bitcast.getIn().getType());
        auto resultType = dyn_cast<IntegerType>(bitcast.getOut().getType());
        if (!sourceType || !resultType || sourceType.getWidth() != resultType.getWidth())
            break;
        value = bitcast.getIn();
    }
    return value;
}

std::optional<int64_t> constantInteger(Value value) {
    value = transparentIntegerValue(value);
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
        return constantInteger(cast.getIn());
    if (auto cast = value.getDefiningOp<arith::IndexCastUIOp>())
        return constantInteger(cast.getIn());
    auto constant = value.getDefiningOp<arith::ConstantOp>();
    auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : nullptr;
    return integer ? std::optional<int64_t>(integer.getValue().getSExtValue()) : std::nullopt;
}

struct BuiltinAxis {
    unsigned axis{};
    bool scalar{};
};

std::optional<BuiltinAxis> builtinAxis(Value value, StringRef expectedBuiltin) {
    value = transparentIntegerValue(value);
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
        return builtinAxis(cast.getIn(), expectedBuiltin);
    if (auto cast = value.getDefiningOp<arith::IndexCastUIOp>())
        return builtinAxis(cast.getIn(), expectedBuiltin);
    if (auto argument = dyn_cast<BlockArgument>(value)) {
        auto function = dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp());
        auto builtin =
            function ? function.getArgAttrOfType<StringAttr>(argument.getArgNumber(), kBuiltinAttrName) : nullptr;
        return builtin && builtin.getValue() == expectedBuiltin && !isa<ShapedType>(value.getType())
                   ? std::optional<BuiltinAxis>(BuiltinAxis{0, true})
                   : std::nullopt;
    }
    auto extract = value.getDefiningOp<tensor::ExtractOp>();
    if (!extract || extract.getIndices().size() != 1)
        return std::nullopt;
    auto source = dyn_cast<BlockArgument>(extract.getTensor());
    std::optional<int64_t> axis = constantInteger(extract.getIndices().front());
    if (!source || !axis || *axis < 0 || *axis >= 3)
        return std::nullopt;
    auto function = dyn_cast_or_null<func::FuncOp>(source.getOwner()->getParentOp());
    auto builtin = function ? function.getArgAttrOfType<StringAttr>(source.getArgNumber(), kBuiltinAttrName) : nullptr;
    return builtin && builtin.getValue() == expectedBuiltin
               ? std::optional<BuiltinAxis>(BuiltinAxis{static_cast<unsigned>(*axis), false})
               : std::nullopt;
}

func::FuncOp enclosingFunction(Value value) {
    if (auto argument = dyn_cast<BlockArgument>(value))
        return dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp());
    Operation *definition = value.getDefiningOp();
    return definition ? definition->getParentOfType<func::FuncOp>() : func::FuncOp();
}

std::optional<GlobalIdAffineIndex> reconstructedGlobalInvocation(Value value) {
    value = transparentIntegerValue(value);
    auto add = value.getDefiningOp<arith::AddIOp>();
    if (!add)
        return std::nullopt;

    auto match = [&](Value base, Value local) -> std::optional<GlobalIdAffineIndex> {
        auto localAxis = builtinAxis(local, "local_invocation_id");
        auto multiply = transparentIntegerValue(base).getDefiningOp<arith::MulIOp>();
        if (!localAxis || !multiply)
            return std::nullopt;
        Value group = multiply.getLhs();
        std::optional<int64_t> scale = constantInteger(multiply.getRhs());
        if (!scale) {
            group = multiply.getRhs();
            scale = constantInteger(multiply.getLhs());
        }
        auto groupAxis = builtinAxis(group, "workgroup_id");
        if (!groupAxis || groupAxis->axis != localAxis->axis || groupAxis->scalar != localAxis->scalar)
            return std::nullopt;
        func::FuncOp function = enclosingFunction(value);
        auto workgroup = function ? function->getAttrOfType<DenseI32ArrayAttr>(kWorkgroupSizeAttrName) : nullptr;
        if (!workgroup || workgroup.size() != 3 || *scale != workgroup[localAxis->axis])
            return std::nullopt;
        GlobalIdAffineIndex result;
        result.coefficients[localAxis->axis] = 1;
        return result;
    };
    if (auto result = match(add.getLhs(), add.getRhs()))
        return result;
    return match(add.getRhs(), add.getLhs());
}

NormalizationResult normalize(Value value, DenseSet<Value> &visiting) {
    if (!visiting.insert(value).second)
        return {};
    const auto remove = llvm::make_scope_exit([&] { visiting.erase(value); });
    Value transparent = transparentIntegerValue(value);
    if (transparent != value)
        return normalize(transparent, visiting);
    if (std::optional<int64_t> constant = constantInteger(value))
        return {GlobalIdAffineIndex{{}, *constant}, false};
    if (auto reconstructed = reconstructedGlobalInvocation(value))
        return {*reconstructed, false};
    if (std::optional<BuiltinAxis> builtin = builtinAxis(value, "global_invocation_id")) {
        GlobalIdAffineIndex result;
        result.coefficients[builtin->axis] = 1;
        result.scalarGlobalId = builtin->scalar;
        return {result, false};
    }
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
        auto sourceType = dyn_cast<IntegerType>(cast.getIn().getType());
        if (!isa<IndexType>(cast.getOut().getType()) || !sourceType || sourceType.getWidth() > 32)
            return {{}, true};
        return normalize(cast.getIn(), visiting);
    }
    if (auto cast = value.getDefiningOp<arith::IndexCastUIOp>()) {
        auto sourceType = dyn_cast<IntegerType>(cast.getIn().getType());
        if (!isa<IndexType>(cast.getOut().getType()) || !sourceType || sourceType.getWidth() > 32)
            return {{}, true};
        return normalize(cast.getIn(), visiting);
    }
    auto combine = [&](Value lhs, Value rhs, int64_t sign) -> NormalizationResult {
        NormalizationResult left = normalize(lhs, visiting);
        NormalizationResult right = normalize(rhs, visiting);
        if (left.invalidArithmetic || right.invalidArithmetic)
            return {{}, true};
        if (!left.index || !right.index)
            return {};
        GlobalIdAffineIndex result = *left.index;
        for (unsigned axis = 0; axis < result.coefficients.size(); ++axis)
            result.coefficients[axis] += sign * right.index->coefficients[axis];
        result.constant += sign * right.index->constant;
        result.scalarGlobalId |= right.index->scalarGlobalId;
        return {result, false};
    };
    if (auto add = value.getDefiningOp<arith::AddIOp>())
        return combine(add.getLhs(), add.getRhs(), 1);
    if (auto subtract = value.getDefiningOp<arith::SubIOp>())
        return combine(subtract.getLhs(), subtract.getRhs(), -1);
    if (auto multiply = value.getDefiningOp<arith::MulIOp>()) {
        Value expression = multiply.getLhs();
        std::optional<int64_t> scale = constantInteger(multiply.getRhs());
        if (!scale) {
            expression = multiply.getRhs();
            scale = constantInteger(multiply.getLhs());
        }
        if (!scale)
            return {{}, true};
        NormalizationResult result = normalize(expression, visiting);
        if (!result.index)
            return result;
        for (int64_t &coefficient : result.index->coefficients)
            coefficient *= *scale;
        result.index->constant *= *scale;
        return result;
    }
    if (value.getDefiningOp<arith::DivSIOp>() || value.getDefiningOp<arith::DivUIOp>() ||
        value.getDefiningOp<arith::RemSIOp>() || value.getDefiningOp<arith::RemUIOp>() ||
        value.getDefiningOp<arith::TruncIOp>())
        return {{}, true};
    return {};
}

std::optional<unsigned> uniqueGlobalIdAxis(const GlobalIdAffineIndex &index) {
    std::optional<unsigned> uniqueAxis;
    for (auto [axis, coefficient] : llvm::enumerate(index.coefficients)) {
        if (!coefficient)
            continue;
        if (uniqueAxis)
            return std::nullopt;
        uniqueAxis = static_cast<unsigned>(axis);
    }
    return uniqueAxis;
}

NormalizationResult normalizeWorkgroup(Value value, DenseSet<Value> &visiting) {
    if (!visiting.insert(value).second)
        return {};
    const auto remove = llvm::make_scope_exit([&] { visiting.erase(value); });
    Value transparent = transparentIntegerValue(value);
    if (transparent != value)
        return normalizeWorkgroup(transparent, visiting);
    if (std::optional<int64_t> constant = constantInteger(value)) {
        GlobalIdAffineIndex result{{}, *constant};
        result.domain = IndexOwnershipDomain::Workgroup;
        return {result, false};
    }
    if (std::optional<BuiltinAxis> builtin = builtinAxis(value, "workgroup_id")) {
        GlobalIdAffineIndex result;
        result.coefficients[builtin->axis] = 1;
        result.scalarGlobalId = builtin->scalar;
        result.domain = IndexOwnershipDomain::Workgroup;
        return {result, false};
    }
    auto combine = [&](Value lhs, Value rhs, int64_t sign) -> NormalizationResult {
        NormalizationResult left = normalizeWorkgroup(lhs, visiting);
        NormalizationResult right = normalizeWorkgroup(rhs, visiting);
        if (left.invalidArithmetic || right.invalidArithmetic)
            return {{}, true};
        if (!left.index || !right.index)
            return {};
        GlobalIdAffineIndex result = *left.index;
        for (unsigned axis = 0; axis < result.coefficients.size(); ++axis)
            result.coefficients[axis] += sign * right.index->coefficients[axis];
        result.constant += sign * right.index->constant;
        result.scalarGlobalId |= right.index->scalarGlobalId;
        return {result, false};
    };
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
        return normalizeWorkgroup(cast.getIn(), visiting);
    if (auto cast = value.getDefiningOp<arith::IndexCastUIOp>())
        return normalizeWorkgroup(cast.getIn(), visiting);
    if (auto add = value.getDefiningOp<arith::AddIOp>())
        return combine(add.getLhs(), add.getRhs(), 1);
    if (auto subtract = value.getDefiningOp<arith::SubIOp>())
        return combine(subtract.getLhs(), subtract.getRhs(), -1);
    if (auto multiply = value.getDefiningOp<arith::MulIOp>()) {
        Value expression = multiply.getLhs();
        std::optional<int64_t> scale = constantInteger(multiply.getRhs());
        if (!scale) {
            expression = multiply.getRhs();
            scale = constantInteger(multiply.getLhs());
        }
        if (!scale)
            return {{}, true};
        NormalizationResult result = normalizeWorkgroup(expression, visiting);
        if (!result.index)
            return result;
        for (int64_t &coefficient : result.index->coefficients)
            coefficient *= *scale;
        result.index->constant *= *scale;
        return result;
    }
    if (value.getDefiningOp<arith::DivSIOp>() || value.getDefiningOp<arith::DivUIOp>() ||
        value.getDefiningOp<arith::RemSIOp>() || value.getDefiningOp<arith::RemUIOp>() ||
        value.getDefiningOp<arith::TruncIOp>())
        return {{}, true};
    return {};
}

std::optional<ConditionalIndexProof> proveWorkgroupOwnedIndex(ValueRange indices) {
    if (indices.empty())
        return std::nullopt;
    GlobalIdAffineIndexTuple tuple;
    DenseSet<unsigned> axes;
    bool sawUnknownSuffix = false;
    for (auto [dimension, value] : llvm::enumerate(indices)) {
        DenseSet<Value> visiting;
        NormalizationResult normalized = normalizeWorkgroup(value, visiting);
        if (normalized.invalidArithmetic)
            return std::nullopt;
        if (!normalized.index) {
            sawUnknownSuffix = true;
            continue;
        }
        std::optional<unsigned> axis = uniqueGlobalIdAxis(*normalized.index);
        if (!axis) {
            sawUnknownSuffix = true;
            continue;
        }
        if (sawUnknownSuffix || !axes.insert(*axis).second)
            return std::nullopt;
        normalized.index->dimension = dimension;
        tuple.push_back(*normalized.index);
    }
    ConditionalIndexProof proof;
    proof.normalizedIndices = std::move(tuple);
    proof.ownership = IndexOwnershipDomain::Workgroup;
    for (unsigned axis : axes)
        proof.coveredAxes[axis] = true;
    if (axes.size() == 3)
        return proof;
    if (axes.empty())
        return std::nullopt;
    for (unsigned axis = 0; axis < 3; ++axis)
        if (!axes.contains(axis))
            proof.unitGridAxes.push_back(axis);
    return proof;
}

bool isInsideThenRegion(Operation *operation, scf::IfOp ifOp) {
    for (Operation *current = operation; current && current != ifOp; current = current->getParentOp())
        if (current->getParentRegion() == &ifOp.getThenRegion())
            return true;
    return false;
}

bool collectLeaderGuardAxes(Operation *operation, ArrayRef<int32_t> workgroupSize, DenseSet<unsigned> &guardedAxes) {
    bool sawLeaderGuard = false;
    for (Operation *parent = operation->getParentOp(); parent; parent = parent->getParentOp()) {
        if (isa<func::FuncOp>(parent))
            break;
        auto ifOp = dyn_cast<scf::IfOp>(parent);
        if (!ifOp || !isInsideThenRegion(operation, ifOp))
            continue;
        Value condition = transparentIntegerValue(ifOp.getCondition());
        auto compare = condition.getDefiningOp<arith::CmpIOp>();
        if (!compare || compare.getPredicate() != arith::CmpIPredicate::eq)
            continue;
        Value local = compare.getLhs();
        std::optional<int64_t> constant = constantInteger(compare.getRhs());
        if (!constant) {
            local = compare.getRhs();
            constant = constantInteger(compare.getLhs());
        }
        auto axis = builtinAxis(local, "local_invocation_id");
        if (!axis || !constant || *constant < 0)
            continue;
        if (axis->scalar) {
            int64_t laneCount = 1;
            for (int32_t extent : workgroupSize)
                laneCount *= extent;
            if (*constant >= laneCount)
                continue;
            sawLeaderGuard = true;
            for (auto [guardedAxis, extent] : llvm::enumerate(workgroupSize))
                if (extent > 1)
                    guardedAxes.insert(static_cast<unsigned>(guardedAxis));
            continue;
        }
        if (*constant >= workgroupSize[axis->axis])
            continue;
        sawLeaderGuard = true;
        guardedAxes.insert(axis->axis);
    }
    if (!sawLeaderGuard)
        return false;
    for (auto [axis, extent] : llvm::enumerate(workgroupSize))
        if (extent > 1 && !guardedAxes.contains(static_cast<unsigned>(axis)))
            return false;
    return true;
}

bool usesScalarGlobalInvocationId(Value value, DenseSet<Value> &visited) {
    if (!visited.insert(value).second)
        return false;
    if (std::optional<BuiltinAxis> builtin = builtinAxis(value, "global_invocation_id"))
        return builtin->scalar;
    Operation *definition = value.getDefiningOp();
    return definition && llvm::any_of(definition->getOperands(),
                                      [&](Value operand) { return usesScalarGlobalInvocationId(operand, visited); });
}

} // namespace

bool GlobalIdAffineIndex::operator==(const GlobalIdAffineIndex &other) const {
    return coefficients == other.coefficients && constant == other.constant && dimension == other.dimension &&
           scalarGlobalId == other.scalarGlobalId && domain == other.domain;
}

std::optional<GlobalIdAffineIndex> normalizeGlobalIdAffineIndex(Value value) {
    DenseSet<Value> visiting;
    return normalize(value, visiting).index;
}

bool usesScalarGlobalInvocationId(ValueRange values) {
    DenseSet<Value> visited;
    return llvm::any_of(values, [&](Value value) { return usesScalarGlobalInvocationId(value, visited); });
}

std::optional<ConditionalIndexProof> proveStrictInvocationOwnedIndex(ValueRange indices) {
    if (indices.empty())
        return std::nullopt;
    GlobalIdAffineIndexTuple tuple;
    DenseSet<unsigned> axes;
    for (auto [dimension, value] : llvm::enumerate(indices)) {
        DenseSet<Value> visiting;
        NormalizationResult normalized = normalize(value, visiting);
        if (!normalized.index)
            return std::nullopt;
        std::optional<unsigned> axis = uniqueGlobalIdAxis(*normalized.index);
        if (!axis || normalized.index->scalarGlobalId || !axes.insert(*axis).second)
            return std::nullopt;
        normalized.index->dimension = dimension;
        tuple.push_back(*normalized.index);
    }
    if (axes.size() != 3)
        return std::nullopt;
    ConditionalIndexProof proof;
    proof.normalizedIndices = std::move(tuple);
    proof.coveredAxes = {true, true, true};
    proof.ownership = IndexOwnershipDomain::Invocation;
    return proof;
}

std::optional<ConditionalIndexProof> proveInvocationOwnedIndex(ValueRange indices) {
    if (indices.empty())
        return std::nullopt;
    GlobalIdAffineIndexTuple tuple;
    DenseSet<unsigned> axes;
    bool sawUnknownSuffix = false;
    func::FuncOp function = enclosingFunction(indices.front());
    auto workgroup = function ? function->getAttrOfType<DenseI32ArrayAttr>(kWorkgroupSizeAttrName) : nullptr;
    for (auto [dimension, value] : llvm::enumerate(indices)) {
        DenseSet<Value> visiting;
        NormalizationResult normalized = normalize(value, visiting);
        if (normalized.invalidArithmetic)
            return std::nullopt;
        if (!normalized.index) {
            sawUnknownSuffix = true;
            continue;
        }
        std::optional<unsigned> axis = uniqueGlobalIdAxis(*normalized.index);
        if (!axis) {
            sawUnknownSuffix = true;
            continue;
        }
        if (sawUnknownSuffix || !axes.insert(*axis).second)
            return std::nullopt;
        normalized.index->dimension = dimension;
        if (normalized.index->scalarGlobalId)
            return std::nullopt;
        tuple.push_back(*normalized.index);
    }
    ConditionalIndexProof proof;
    proof.normalizedIndices = std::move(tuple);
    proof.ownership = IndexOwnershipDomain::Invocation;
    for (unsigned axis : axes)
        proof.coveredAxes[axis] = true;
    if (axes.size() == 3)
        return proof;
    if (axes.empty() || !workgroup || workgroup.size() != 3)
        return std::nullopt;
    for (unsigned axis = 0; axis < 3; ++axis) {
        if (axes.contains(axis))
            continue;
        if (workgroup[axis] != 1)
            return std::nullopt;
        proof.unitGridAxes.push_back(axis);
    }
    return proof;
}

std::optional<ConditionalIndexProof> proveLeaderGuardedWorkgroupOwnedIndex(Operation *operation, ValueRange indices) {
    func::FuncOp function = operation ? operation->getParentOfType<func::FuncOp>() : func::FuncOp();
    auto workgroup = function ? function->getAttrOfType<DenseI32ArrayAttr>(kWorkgroupSizeAttrName) : nullptr;
    if (!workgroup || workgroup.size() != 3)
        return std::nullopt;
    DenseSet<unsigned> guardedAxes;
    if (!collectLeaderGuardAxes(operation, workgroup.asArrayRef(), guardedAxes))
        return std::nullopt;
    return proveWorkgroupOwnedIndex(indices);
}

} // namespace mlir::vernon
