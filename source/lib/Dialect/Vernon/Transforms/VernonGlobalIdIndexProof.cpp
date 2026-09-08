#include "mlir/Dialect/Vernon/Transforms/VernonGlobalIdIndexProof.h"

// Lemmas: specs/compiler/invocation_index_ownership.md

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

#include <limits>
#include <memory>
#include <optional>
#include <utility>

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
    // specs/compiler/invocation_index_ownership.md §5
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

struct IndexTerm {
    enum class Kind { Const, Gid, ShapeDim, Add, Sub, Mul, URem, UDiv };
    Kind kind{};
    int64_t constant{};
    unsigned axis{};
    bool scalarGid{};
    Value shapeSource;
    std::shared_ptr<IndexTerm> lhs;
    std::shared_ptr<IndexTerm> rhs;
};

using IndexTermPtr = std::shared_ptr<IndexTerm>;

bool termEqual(const IndexTerm *left, const IndexTerm *right) {
    if (left == right)
        return true;
    if (!left || !right || left->kind != right->kind)
        return false;
    switch (left->kind) {
    case IndexTerm::Kind::Const:
        return left->constant == right->constant;
    case IndexTerm::Kind::Gid:
        return left->axis == right->axis && left->scalarGid == right->scalarGid;
    case IndexTerm::Kind::ShapeDim:
        return left->shapeSource == right->shapeSource && left->axis == right->axis;
    case IndexTerm::Kind::Mul:
        return left->constant == right->constant && termEqual(left->lhs.get(), right->lhs.get());
    case IndexTerm::Kind::Add:
    case IndexTerm::Kind::Sub:
    case IndexTerm::Kind::URem:
    case IndexTerm::Kind::UDiv:
        return termEqual(left->lhs.get(), right->lhs.get()) && termEqual(left->rhs.get(), right->rhs.get());
    }
    return false;
}

IndexTermPtr makeTerm(IndexTerm term) { return std::make_shared<IndexTerm>(std::move(term)); }

bool isPositiveExtent(const IndexTerm &term) {
    if (term.kind == IndexTerm::Kind::Const)
        return term.constant >= 1;
    return term.kind == IndexTerm::Kind::ShapeDim;
}

MixedRadixExtent encodeExtent(const IndexTerm &term) {
    if (term.kind == IndexTerm::Kind::Const)
        return MixedRadixExtent{{}, term.constant};
    return MixedRadixExtent{term.shapeSource, static_cast<int64_t>(term.axis)};
}

IndexTermPtr normalizeIndexTerm(Value value, DenseSet<Value> &visiting);

std::optional<GlobalIdAffineIndex> affineFromTerm(const IndexTerm &term) {
    auto combine = [&](const IndexTerm &lhs, const IndexTerm &rhs, int64_t sign) -> std::optional<GlobalIdAffineIndex> {
        std::optional<GlobalIdAffineIndex> left = affineFromTerm(lhs);
        std::optional<GlobalIdAffineIndex> right = affineFromTerm(rhs);
        if (!left || !right)
            return std::nullopt;
        GlobalIdAffineIndex result = *left;
        for (unsigned axis = 0; axis < result.coefficients.size(); ++axis)
            result.coefficients[axis] += sign * right->coefficients[axis];
        result.constant += sign * right->constant;
        result.scalarGlobalId |= right->scalarGlobalId;
        return result;
    };
    switch (term.kind) {
    case IndexTerm::Kind::Const:
        return GlobalIdAffineIndex{{}, term.constant};
    case IndexTerm::Kind::Gid: {
        GlobalIdAffineIndex result;
        result.coefficients[term.axis] = 1;
        result.scalarGlobalId = term.scalarGid;
        return result;
    }
    case IndexTerm::Kind::Add:
        return term.lhs && term.rhs ? combine(*term.lhs, *term.rhs, 1) : std::nullopt;
    case IndexTerm::Kind::Sub:
        return term.lhs && term.rhs ? combine(*term.lhs, *term.rhs, -1) : std::nullopt;
    case IndexTerm::Kind::Mul: {
        if (!term.lhs)
            return std::nullopt;
        std::optional<GlobalIdAffineIndex> inner = affineFromTerm(*term.lhs);
        if (!inner)
            return std::nullopt;
        for (int64_t &coefficient : inner->coefficients)
            coefficient *= term.constant;
        inner->constant *= term.constant;
        return inner;
    }
    case IndexTerm::Kind::ShapeDim:
    case IndexTerm::Kind::URem:
    case IndexTerm::Kind::UDiv:
        return std::nullopt;
    }
    return std::nullopt;
}

IndexTermPtr normalizeIndexTerm(Value value, DenseSet<Value> &visiting) {
    if (!visiting.insert(value).second)
        return {};
    const auto remove = llvm::make_scope_exit([&] { visiting.erase(value); });
    Value transparent = transparentIntegerValue(value);
    if (transparent != value)
        return normalizeIndexTerm(transparent, visiting);
    if (std::optional<int64_t> constant = constantInteger(value)) {
        IndexTerm term;
        term.kind = IndexTerm::Kind::Const;
        term.constant = *constant;
        return makeTerm(std::move(term));
    }
    if (auto reconstructed = reconstructedGlobalInvocation(value)) {
        std::optional<unsigned> axis = uniqueGlobalIdAxis(*reconstructed);
        if (!axis || reconstructed->scalarGlobalId)
            return {};
        IndexTerm term;
        term.kind = IndexTerm::Kind::Gid;
        term.axis = *axis;
        return makeTerm(std::move(term));
    }
    if (std::optional<BuiltinAxis> builtin = builtinAxis(value, "global_invocation_id")) {
        IndexTerm term;
        term.kind = IndexTerm::Kind::Gid;
        term.axis = builtin->axis;
        term.scalarGid = builtin->scalar;
        return makeTerm(std::move(term));
    }
    if (auto extract = value.getDefiningOp<tensor::ExtractOp>()) {
        if (extract.getIndices().size() == 1) {
            auto getShape = extract.getTensor().getDefiningOp<GetShapeOp>();
            std::optional<int64_t> axis = constantInteger(extract.getIndices().front());
            if (getShape && axis && *axis >= 0 && *axis <= static_cast<int64_t>(std::numeric_limits<unsigned>::max())) {
                IndexTerm term;
                term.kind = IndexTerm::Kind::ShapeDim;
                term.axis = static_cast<unsigned>(*axis);
                term.shapeSource = getShape.getSource();
                return makeTerm(std::move(term));
            }
        }
    }
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
        auto sourceType = dyn_cast<IntegerType>(cast.getIn().getType());
        if (!isa<IndexType>(cast.getOut().getType()) || !sourceType || sourceType.getWidth() > 32)
            return {};
        return normalizeIndexTerm(cast.getIn(), visiting);
    }
    if (auto cast = value.getDefiningOp<arith::IndexCastUIOp>()) {
        auto sourceType = dyn_cast<IntegerType>(cast.getIn().getType());
        if (!isa<IndexType>(cast.getOut().getType()) || !sourceType || sourceType.getWidth() > 32)
            return {};
        return normalizeIndexTerm(cast.getIn(), visiting);
    }
    if (auto add = value.getDefiningOp<arith::AddIOp>()) {
        IndexTermPtr lhs = normalizeIndexTerm(add.getLhs(), visiting);
        IndexTermPtr rhs = normalizeIndexTerm(add.getRhs(), visiting);
        if (!lhs || !rhs)
            return {};
        return makeTerm(IndexTerm{IndexTerm::Kind::Add, 0, 0, false, {}, std::move(lhs), std::move(rhs)});
    }
    if (auto subtract = value.getDefiningOp<arith::SubIOp>()) {
        IndexTermPtr lhs = normalizeIndexTerm(subtract.getLhs(), visiting);
        IndexTermPtr rhs = normalizeIndexTerm(subtract.getRhs(), visiting);
        if (!lhs || !rhs)
            return {};
        return makeTerm(IndexTerm{IndexTerm::Kind::Sub, 0, 0, false, {}, std::move(lhs), std::move(rhs)});
    }
    if (auto multiply = value.getDefiningOp<arith::MulIOp>()) {
        Value expression = multiply.getLhs();
        std::optional<int64_t> scale = constantInteger(multiply.getRhs());
        if (!scale) {
            expression = multiply.getRhs();
            scale = constantInteger(multiply.getLhs());
        }
        if (!scale)
            return {};
        IndexTermPtr inner = normalizeIndexTerm(expression, visiting);
        if (!inner)
            return {};
        IndexTerm term;
        term.kind = IndexTerm::Kind::Mul;
        term.constant = *scale;
        term.lhs = std::move(inner);
        return makeTerm(std::move(term));
    }
    auto makeEuclidean = [&](Value lhsValue, Value rhsValue, IndexTerm::Kind kind) -> IndexTermPtr {
        IndexTermPtr lhs = normalizeIndexTerm(lhsValue, visiting);
        IndexTermPtr rhs = normalizeIndexTerm(rhsValue, visiting);
        if (!lhs || !rhs || !isPositiveExtent(*rhs))
            return {};
        return makeTerm(IndexTerm{kind, 0, 0, false, {}, std::move(lhs), std::move(rhs)});
    };
    if (auto rem = value.getDefiningOp<arith::RemUIOp>())
        return makeEuclidean(rem.getLhs(), rem.getRhs(), IndexTerm::Kind::URem);
    if (auto div = value.getDefiningOp<arith::DivUIOp>())
        return makeEuclidean(div.getLhs(), div.getRhs(), IndexTerm::Kind::UDiv);
    return {};
}

std::optional<ConditionalIndexProof> constrainUnusedGridAxes(ConditionalIndexProof proof,
                                                             const DenseSet<unsigned> &axes, func::FuncOp function) {
    // Residual G_a = 1 and W_a = 1 on uncovered axes:
    // specs/compiler/invocation_index_ownership.md §2, §3.3
    auto workgroup = function ? function->getAttrOfType<DenseI32ArrayAttr>(kWorkgroupSizeAttrName) : nullptr;
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

std::optional<ConditionalIndexProof> proveAffineInvocationOwnedIndex(ValueRange indices) {
    // Unique-axis affine matching: specs/compiler/invocation_index_ownership.md §3
    if (indices.empty())
        return std::nullopt;
    GlobalIdAffineIndexTuple tuple;
    DenseSet<unsigned> axes;
    bool sawUnknownSuffix = false;
    func::FuncOp function = enclosingFunction(indices.front());
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
    return constrainUnusedGridAxes(std::move(proof), axes, function);
}

std::optional<ConditionalIndexProof> proveMixedRadixInvocationOwnedIndex(ValueRange indices) {
    // Unwrapped unsigned mixed radix of a unique-axis linear form:
    // specs/compiler/invocation_index_ownership.md §4
    if (indices.size() < 2)
        return std::nullopt;
    SmallVector<IndexTermPtr> digits;
    digits.reserve(indices.size());
    for (Value value : indices) {
        DenseSet<Value> visiting;
        IndexTermPtr term = normalizeIndexTerm(value, visiting);
        if (!term)
            return std::nullopt;
        digits.push_back(std::move(term));
    }

    IndexTermPtr linear;
    SmallVector<MixedRadixExtent> extents(digits.size() - 1);
    for (unsigned dimension = digits.size() - 1; dimension > 0; --dimension) {
        const IndexTerm &digit = *digits[dimension];
        if (digit.kind != IndexTerm::Kind::URem || !digit.lhs || !digit.rhs || !isPositiveExtent(*digit.rhs))
            return std::nullopt;
        if (linear && !termEqual(digit.lhs.get(), linear.get()))
            return std::nullopt;
        extents[dimension - 1] = encodeExtent(*digit.rhs);
        IndexTerm next;
        next.kind = IndexTerm::Kind::UDiv;
        next.lhs = digit.lhs;
        next.rhs = digit.rhs;
        linear = makeTerm(std::move(next));
    }
    if (!linear || !termEqual(digits.front().get(), linear.get()))
        return std::nullopt;

    const IndexTerm &source = *digits.back()->lhs;
    std::optional<GlobalIdAffineIndex> affine = affineFromTerm(source);
    std::optional<unsigned> axis = affine ? uniqueGlobalIdAxis(*affine) : std::nullopt;
    if (!affine || !axis || affine->scalarGlobalId)
        return std::nullopt;

    ConditionalIndexProof proof;
    proof.ownership = IndexOwnershipDomain::Invocation;
    proof.mixedRadixLinear = *affine;
    proof.mixedRadixExtents = std::move(extents);
    DenseSet<unsigned> axes;
    axes.insert(*axis);
    return constrainUnusedGridAxes(std::move(proof), axes, enclosingFunction(indices.front()));
}

} // namespace

bool GlobalIdAffineIndex::operator==(const GlobalIdAffineIndex &other) const {
    return coefficients == other.coefficients && constant == other.constant && dimension == other.dimension &&
           scalarGlobalId == other.scalarGlobalId && domain == other.domain;
}

bool MixedRadixExtent::operator==(const MixedRadixExtent &other) const {
    return shapeSource == other.shapeSource && value == other.value;
}

bool ConditionalIndexProof::operator==(const ConditionalIndexProof &other) const {
    return normalizedIndices == other.normalizedIndices && coveredAxes == other.coveredAxes &&
           unitGridAxes == other.unitGridAxes && ownership == other.ownership &&
           mixedRadixLinear == other.mixedRadixLinear && mixedRadixExtents == other.mixedRadixExtents;
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
    // Full three-axis affine matching, no residual:
    // specs/compiler/invocation_index_ownership.md §3.4
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
    // Affine §3, else mixed-radix §4:
    // specs/compiler/invocation_index_ownership.md
    if (auto affine = proveAffineInvocationOwnedIndex(indices))
        return affine;
    return proveMixedRadixInvocationOwnedIndex(indices);
}

std::optional<ConditionalIndexProof> proveLeaderGuardedWorkgroupOwnedIndex(Operation *operation, ValueRange indices) {
    // specs/compiler/invocation_index_ownership.md §5
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
