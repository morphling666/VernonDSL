#include "mlir/Dialect/VernonProgram/Transforms/VernonProgramVjp.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffRules.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffUtils.h"
#include "mlir/Dialect/VernonProgram/IR/VernonProgram.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringMap.h"

#include <algorithm>
#include <limits>

using namespace mlir;

namespace mlir::vernon::program {
namespace {

FailureOr<Type> derivativeType(Type type, Operation *scope) {
    ModuleOp module = scope->getParentOfType<ModuleOp>();
    if (!module)
        return failure();
    return getAutodiffDerivativeType(type, module);
}

FailureOr<Type> cotangentType(Type type, Operation *scope) {
    ModuleOp module = scope->getParentOfType<ModuleOp>();
    if (!module)
        return failure();
    return getAutodiffDerivativeType(type, module, "read");
}

FailureOr<Type> gradientDestType(Type type, Operation *scope) {
    ModuleOp module = scope->getParentOfType<ModuleOp>();
    if (!module)
        return failure();
    return getAutodiffDerivativeType(type, module, "write");
}

Value createProgramIntrinsic(OpBuilder &builder, Location location, StringRef name, ValueRange operands,
                             Type resultType, ArrayRef<NamedAttribute> attributes = {}) {
    OperationState state(location, IntrinsicOp::getOperationName());
    state.addOperands(operands);
    state.addTypes(resultType);
    state.addAttribute("name", builder.getStringAttr(name));
    state.addAttributes(attributes);
    return builder.create(state)->getResult(0);
}

SmallVector<int64_t, 3> linearizedLaunchGrid(Type type) {
    auto view = dyn_cast<TensorViewType>(type);
    if (!view)
        return {1, 1, 1};
    ArrayRef<int64_t> shape = view.getShape();
    if (!llvm::all_of(shape, [](int64_t extent) { return extent >= 0; }))
        return {1, 1, 1};
    int64_t count = 1;
    for (int64_t extent : shape) {
        const int64_t factor = std::max<int64_t>(extent, 1);
        if (count > std::numeric_limits<int64_t>::max() / factor)
            return {1, 1, 1};
        count *= factor;
    }
    return {count, 1, 1};
}

bool hasDynamicExtent(Type type) {
    auto view = dyn_cast<TensorViewType>(type);
    return view && llvm::any_of(view.getShape(), [](int64_t extent) { return extent < 0; });
}

Value createZero(OpBuilder &builder, Location location, Type type, Value like = {}) {
    if (auto scalar = dyn_cast<FloatType>(type))
        return arith::ConstantOp::create(builder, location, builder.getFloatAttr(scalar, 0.0));
    if (isa<TensorViewType>(type)) {
        Value zero = like && hasDynamicExtent(type)
                         ? createProgramIntrinsic(builder, location, "zeros_like", like, type)
                         : createProgramIntrinsic(builder, location, "zeros", {}, type);
        if (like)
            applyProgramLanguageAbi(zero.getDefiningOp(), getProgramValueLanguageAbi(like));
        return zero;
    }
    if (isa<TensorType>(type)) {
        Value zero = createProgramIntrinsic(builder, location, "zeros", {}, type);
        if (like)
            applyProgramLanguageAbi(zero.getDefiningOp(), getProgramValueLanguageAbi(like));
        return zero;
    }
    auto tensor = dyn_cast<RankedTensorType>(type);
    auto element = tensor ? dyn_cast<FloatType>(tensor.getElementType()) : FloatType{};
    if (!tensor || !element || !tensor.hasStaticShape()) {
        if (auto tuple = dyn_cast<TupleType>(type)) {
            SmallVector<Value> elements;
            for (Type elementType : tuple.getTypes()) {
                Value elementValue = createZero(builder, location, elementType);
                if (!elementValue)
                    return {};
                elements.push_back(elementValue);
            }
            return TupleCreateOp::create(builder, location, tuple, elements);
        }
        return {};
    }
    return arith::ConstantOp::create(builder, location,
                                     DenseElementsAttr::get(tensor, builder.getFloatAttr(element, 0.0)));
}

Value createDestinationPassingCompute(OpBuilder &builder, Location location, StringRef callee, ValueRange operands,
                                      ArrayRef<StringRef> operandNames, ArrayRef<StringRef> operandAccesses,
                                      Type resultType, int64_t destOperand) {
    OperationState state(location, ComputeOp::getOperationName());
    state.addOperands(operands);
    state.addTypes(resultType);
    state.addAttribute("callee", builder.getStringAttr(callee));
    state.addAttribute("grid", builder.getDenseI64ArrayAttr(linearizedLaunchGrid(resultType)));
    state.addAttribute("features", builder.getArrayAttr({}));
    state.addAttribute("operand_names",
                       builder.getArrayAttr(llvm::map_to_vector(
                           operandNames, [&](StringRef name) -> Attribute { return builder.getStringAttr(name); })));
    state.addAttribute("result_names", builder.getArrayAttr({builder.getStringAttr("output")}));
    state.addAttribute(kOperandAccessesAttrName,
                       builder.getArrayAttr(llvm::map_to_vector(operandAccesses, [&](StringRef access) -> Attribute {
                           return builder.getStringAttr(access);
                       })));
    state.addAttribute("vernon_program.result_resource_sources",
                       builder.getDenseI64ArrayAttr(SmallVector<int64_t, 1>{destOperand}));
    Value result = builder.create(state)->getResult(0);
    applyProgramLanguageAbi(result.getDefiningOp(), getProgramValueLanguageAbi(operands[destOperand]));
    return result;
}

Value copyValue(OpBuilder &builder, Location location, Value source, Type destType, Value like = {}) {
    if (source.getType() == destType)
        return source;
    if (!isa<TensorViewType>(source.getType()) || !isa<TensorViewType>(destType))
        return {};
    Value dest = createZero(builder, location, destType, like ? like : source);
    if (!dest)
        return {};
    return createDestinationPassingCompute(builder, location, "vernon.builtin.copy", {source, dest},
                                           {"source", "output"}, {"read", "write"}, destType, 1);
}

Value addValues(OpBuilder &builder, Location location, Value left, Value right) {
    if (left.getType() != right.getType())
        return {};
    if (isa<TensorViewType>(left.getType())) {
        Value dest = createZero(builder, location, left.getType(), left);
        if (!dest)
            return {};
        return createDestinationPassingCompute(builder, location, "vernon.builtin.add", {left, right, dest},
                                               {"left", "right", "output"}, {"read", "read", "write"}, left.getType(),
                                               2);
    }
    if (isa<TensorType>(left.getType()))
        return createProgramIntrinsic(builder, location, "add", {left, right}, left.getType());
    auto tuple = dyn_cast<TupleType>(left.getType());
    if (!tuple)
        return arith::AddFOp::create(builder, location, left, right);
    SmallVector<Value> elements;
    for (auto [index, elementType] : llvm::enumerate(tuple.getTypes())) {
        IntegerAttr indexAttr = builder.getI64IntegerAttr(index);
        Value leftElement = TupleGetOp::create(builder, location, elementType, left, indexAttr);
        Value rightElement = TupleGetOp::create(builder, location, elementType, right, indexAttr);
        Value sum = addValues(builder, location, leftElement, rightElement);
        if (!sum)
            return {};
        elements.push_back(sum);
    }
    return TupleCreateOp::create(builder, location, tuple, elements);
}

StringRef sourceName(func::FuncOp function, unsigned argument) {
    if (auto name = function.getArgAttrOfType<StringAttr>(argument, "vernon.source_name"))
        return name.getValue();
    if (auto names = function->getAttrOfType<ArrayAttr>("vernon_program.argument_names");
        names && argument < names.size())
        if (auto name = dyn_cast<StringAttr>(names[argument]))
            return name.getValue();
    return {};
}

StringRef resultSourceName(func::FuncOp function, unsigned result) {
    if (auto name = function.getResultAttrOfType<StringAttr>(result, "vernon.source_name"))
        return name.getValue();
    if (auto names = function->getAttrOfType<ArrayAttr>("vernon_program.result_names"); names && result < names.size())
        if (auto name = dyn_cast<StringAttr>(names[result]))
            return name.getValue();
    return {};
}

FailureOr<SmallVector<Value>> buildComputeVjp(ComputeOp operation, const AutodiffVjpBuildContext &context) {
    OpBuilder &builder = context.builder;
    SmallVector<Value> operands(context.primalOperands);
    llvm::append_range(operands, context.primalResults);
    llvm::append_range(operands, context.resultCotangents);
    SmallVector<Attribute> operandNames;
    SmallVector<Attribute> operandRoles;
    SmallVector<Attribute> operandSources;
    SmallVector<Attribute> operandAccesses;
    for (auto [index, name] : llvm::enumerate(operation.getOperandNames())) {
        operandNames.push_back(builder.getStringAttr(("primal." + cast<StringAttr>(name).getValue()).str()));
        operandRoles.push_back(builder.getStringAttr("retained_primal"));
        operandSources.push_back(name);
        StringRef access = getProgramOperandAccess(operation, static_cast<unsigned>(index));
        operandAccesses.push_back(builder.getStringAttr(access.empty() ? "read" : access));
    }
    for (Attribute name : operation.getResultNames()) {
        operandNames.push_back(builder.getStringAttr(("result." + cast<StringAttr>(name).getValue()).str()));
        operandRoles.push_back(builder.getStringAttr("retained_primal"));
        operandSources.push_back(name);
        operandAccesses.push_back(builder.getStringAttr("read"));
    }
    for (Attribute name : operation.getResultNames()) {
        operandNames.push_back(builder.getStringAttr(("cotangent." + cast<StringAttr>(name).getValue()).str()));
        operandRoles.push_back(builder.getStringAttr("cotangent"));
        operandSources.push_back(name);
        operandAccesses.push_back(builder.getStringAttr("read"));
    }

    SmallVector<Type> resultTypes;
    SmallVector<Attribute> resultNames;
    SmallVector<Attribute> resultRoles;
    SmallVector<Attribute> resultDtypeRows;
    SmallVector<int64_t> resultResourceSources;
    SmallVector<Value> contributions(operation.getNumOperands());
    for (unsigned index : context.activeOperandIndices) {
        if (isWriteOnlyProgramOperand(operation, index))
            continue;
        FailureOr<Type> type = derivativeType(operation.getOperand(index).getType(), operation);
        if (failed(type))
            continue;
        Attribute name = index < operation.getOperandNames().size() ? operation.getOperandNames()[index]
                                                                    : builder.getStringAttr(std::to_string(index));
        ProgramLanguageAbi abi;
        if (isa<TensorViewType>(*type)) {
            Value dest = createZero(builder, context.location, *type, context.getPrimalOperand(index));
            if (!dest)
                return operation.emitOpError("Program VJP cannot allocate a nested gradient dest");
            resultResourceSources.push_back(static_cast<int64_t>(operands.size()));
            operands.push_back(dest);
            operandNames.push_back(name);
            operandRoles.push_back(builder.getStringAttr("gradient"));
            operandSources.push_back(name);
            operandAccesses.push_back(builder.getStringAttr("write"));
            abi = getProgramValueLanguageAbi(dest);
        } else {
            resultResourceSources.push_back(-1);
            abi = getProgramValueLanguageAbi(context.getPrimalOperand(index));
        }
        resultTypes.push_back(*type);
        resultNames.push_back(name);
        resultRoles.push_back(builder.getStringAttr("gradient"));
        resultDtypeRows.push_back(programLanguageAbiLeafArray(builder.getContext(), abi));
    }
    if (resultTypes.empty())
        return contributions;

    OperationState state(context.location, ComputeOp::getOperationName());
    state.addOperands(operands);
    state.addTypes(resultTypes);
    state.addAttribute("callee", builder.getStringAttr((Twine(operation.getCallee()) + ".vjp").str()));
    state.addAttribute("grid", operation.getGridAttr());
    state.addAttribute("features", operation.getFeaturesAttr());
    state.addAttribute("operand_names", builder.getArrayAttr(operandNames));
    state.addAttribute(kOperandAccessesAttrName, builder.getArrayAttr(operandAccesses));
    state.addAttribute("vernon_program.operand_autodiff_roles", builder.getArrayAttr(operandRoles));
    state.addAttribute("vernon_program.operand_autodiff_sources", builder.getArrayAttr(operandSources));
    state.addAttribute("result_names", builder.getArrayAttr(resultNames));
    state.addAttribute("vernon_program.result_autodiff_roles", builder.getArrayAttr(resultRoles));
    state.addAttribute("vernon_program.result_autodiff_sources", builder.getArrayAttr(resultNames));
    state.addAttribute("vernon_program.result_abi_leaf_dtypes", builder.getArrayAttr(resultDtypeRows));
    if (llvm::any_of(resultResourceSources, [](int64_t source) { return source >= 0; }))
        state.addAttribute("vernon_program.result_resource_sources",
                           builder.getDenseI64ArrayAttr(resultResourceSources));
    for (StringRef name : {"constant_names", "constant_values"})
        if (Attribute attribute = operation->getAttr(name))
            state.addAttribute(name, attribute);
    Operation *vjp = builder.create(state);
    unsigned result = 0;
    for (unsigned index : context.activeOperandIndices) {
        if (isWriteOnlyProgramOperand(operation, index))
            continue;
        if (succeeded(derivativeType(operation.getOperand(index).getType(), operation)))
            contributions[index] = vjp->getResult(result++);
    }
    return contributions;
}

FailureOr<SmallVector<int64_t>> broadcastBatchShape(ArrayRef<int64_t> left, ArrayRef<int64_t> right) {
    SmallVector<int64_t> result(std::max(left.size(), right.size()), 1);
    for (unsigned offset = 0; offset < result.size(); ++offset) {
        const int64_t leftExtent = offset < left.size() ? left[left.size() - 1 - offset] : 1;
        const int64_t rightExtent = offset < right.size() ? right[right.size() - 1 - offset] : 1;
        if (leftExtent != rightExtent && leftExtent != 1 && rightExtent != 1)
            return failure();
        result[result.size() - 1 - offset] = std::max(leftExtent, rightExtent);
    }
    return result;
}

Value transposeLastTwo(OpBuilder &builder, Location location, Value input, RankedTensorType type) {
    SmallVector<int64_t> permutation;
    for (int64_t index = 0; index < type.getRank(); ++index)
        permutation.push_back(index);
    std::swap(permutation[permutation.size() - 2], permutation[permutation.size() - 1]);
    SmallVector<int64_t> shape(type.getShape());
    std::swap(shape[shape.size() - 2], shape[shape.size() - 1]);
    NamedAttribute permutationAttribute(builder.getStringAttr("permutation"),
                                        builder.getDenseI64ArrayAttr(permutation));
    return createProgramIntrinsic(builder, location, "transpose", input,
                                  RankedTensorType::get(shape, type.getElementType()), {permutationAttribute});
}

Value reduceSumToType(OpBuilder &builder, Location location, Value input, RankedTensorType type) {
    return input.getType() == type ? input
                                   : createProgramIntrinsic(builder, location, "reduce_sum_to_shape", input, type);
}

FailureOr<SmallVector<Value>> buildProgramMatmulVjp(IntrinsicOp operation, const AutodiffVjpBuildContext &context) {
    if (operation.getNumOperands() != 2 || operation.getNumResults() != 1)
        return operation.emitOpError("Program matmul VJP requires two operands and one result");
    FailureOr<Type> leftDerivativeType = derivativeType(operation.getOperand(0).getType(), operation);
    FailureOr<Type> rightDerivativeType = derivativeType(operation.getOperand(1).getType(), operation);
    if (failed(leftDerivativeType) || failed(rightDerivativeType))
        return operation.emitOpError("Program matmul VJP requires differentiable operands");
    auto leftType = dyn_cast<RankedTensorType>(*leftDerivativeType);
    auto rightType = dyn_cast<RankedTensorType>(*rightDerivativeType);
    if (!leftType || !rightType || !leftType.hasStaticShape() || !rightType.hasStaticShape() ||
        leftType.getRank() < 2 || rightType.getRank() < 2 || leftType.getElementType() != rightType.getElementType())
        return operation.emitOpError("Program matmul VJP requires static floating-point matrix operands");
    const int64_t rows = leftType.getDimSize(leftType.getRank() - 2);
    const int64_t reduction = leftType.getDimSize(leftType.getRank() - 1);
    const int64_t rightReduction = rightType.getDimSize(rightType.getRank() - 2);
    const int64_t columns = rightType.getDimSize(rightType.getRank() - 1);
    FailureOr<SmallVector<int64_t>> batchShape =
        broadcastBatchShape(leftType.getShape().drop_back(2), rightType.getShape().drop_back(2));
    if (failed(batchShape) || reduction != rightReduction)
        return operation.emitOpError("Program matmul VJP received incompatible operand shapes");

    Value left = context.getPrimalOperand(0);
    Value right = context.getPrimalOperand(1);
    Value seed = context.resultCotangents[0];
    if (left.getType() != leftType)
        left = createProgramIntrinsic(context.builder, context.location, "cast", left, leftType);
    if (right.getType() != rightType)
        right = createProgramIntrinsic(context.builder, context.location, "cast", right, rightType);
    Value transposedRight = transposeLastTwo(context.builder, context.location, right, rightType);
    Value transposedLeft = transposeLastTwo(context.builder, context.location, left, leftType);
    SmallVector<int64_t> leftGradientShape(*batchShape);
    leftGradientShape.append({rows, reduction});
    SmallVector<int64_t> rightGradientShape(*batchShape);
    rightGradientShape.append({reduction, columns});
    auto leftGradientType = RankedTensorType::get(leftGradientShape, leftType.getElementType());
    auto rightGradientType = RankedTensorType::get(rightGradientShape, rightType.getElementType());
    Value leftGradient =
        createProgramIntrinsic(context.builder, context.location, "matmul", {seed, transposedRight}, leftGradientType);
    Value rightGradient =
        createProgramIntrinsic(context.builder, context.location, "matmul", {transposedLeft, seed}, rightGradientType);
    return SmallVector<Value>{reduceSumToType(context.builder, context.location, leftGradient, leftType),
                              reduceSumToType(context.builder, context.location, rightGradient, rightType)};
}

} // namespace

FailureOr<SmallVector<unsigned>> resolveProgramWrtBoundaryIndices(func::FuncOp primal,
                                                                  ArrayRef<StringRef> publicPaths) {
    llvm::StringMap<unsigned> boundaries;
    for (unsigned index = 0; index < primal.getNumArguments(); ++index) {
        StringRef name = sourceName(primal, index);
        if (name.empty() || !boundaries.try_emplace(name, index).second) {
            primal.emitError("Program VJP primal boundaries require unique source names");
            return failure();
        }
    }
    llvm::BitVector selected(primal.getNumArguments());
    SmallVector<unsigned> indices;
    for (StringRef path : publicPaths) {
        auto boundary = boundaries.find(path);
        if (path.empty() || boundary == boundaries.end() || selected.test(boundary->second)) {
            primal.emitError("Program VJP wrt path does not identify a unique primal boundary");
            return failure();
        }
        selected.set(boundary->second);
        indices.push_back(boundary->second);
    }
    return indices;
}

LogicalResult buildProgramVjp(func::FuncOp primal, const ProgramVjpOptions &options) {
    if (!llvm::hasSingleElement(primal.getBody()))
        return primal.emitError("Program VJP requires one straight-line entry block");
    auto returnOp = dyn_cast<func::ReturnOp>(primal.getBody().front().getTerminator());
    if (!returnOp || returnOp.getNumOperands() == 0)
        return primal.emitError("Program VJP requires differentiable function results");
    if (options.wrtBoundaryIndices.empty())
        return primal.emitError("Program VJP requires at least one wrt input");
    if (options.forwardSymbol.empty() || options.backwardSymbol.empty() ||
        options.forwardSymbol == options.backwardSymbol)
        return primal.emitError("Program VJP requires distinct forward and backward symbols");

    ModuleOp module = primal->getParentOfType<ModuleOp>();
    if (module.lookupSymbol(options.forwardSymbol) || module.lookupSymbol(options.backwardSymbol))
        return primal.emitError("Program VJP output symbol already exists");
    if (primal.getBody()
            .walk([](GraphicsOp operation) {
                operation.emitOpError("is not differentiable");
                return WalkResult::interrupt();
            })
            .wasInterrupted())
        return failure();
    SmallVector<Operation *> primalOperations;
    SmallVector<Value> retainedValues(primal.getArguments());
    for (Operation &operation : primal.getBody().front().without_terminator()) {
        if (operation.getNumRegions() != 0)
            return operation.emitError("Program VJP currently requires region-free semantic operators");
        primalOperations.push_back(&operation);
        retainedValues.append(operation.getResults().begin(), operation.getResults().end());
    }

    llvm::BitVector selected(primal.getNumArguments());
    SmallVector<unsigned> wrtIndices(options.wrtBoundaryIndices);
    for (unsigned index : wrtIndices) {
        if (index >= primal.getNumArguments() || selected.test(index))
            return primal.emitError("Program VJP requires unique valid wrt boundary indices");
        selected.set(index);
    }
    for (unsigned index : wrtIndices)
        if (failed(derivativeType(primal.getArgument(index).getType(), primal)))
            return primal.emitError("Program VJP wrt argument is not differentiable");

    OpBuilder moduleBuilder(primal);
    auto forward = cast<func::FuncOp>(primal.clone());
    forward.setSymName(options.forwardSymbol);
    forward->setAttr("vernon_program.graph", moduleBuilder.getStringAttr("forward"));
    forward->removeAttr("vernon.entry");
    forward->removeAttr("vernon.stage");
    moduleBuilder.insert(forward);

    SmallVector<Type> backwardArguments;
    backwardArguments.reserve(retainedValues.size() + primal.getNumResults());
    for (Value retained : retainedValues)
        backwardArguments.push_back(retained.getType());
    for (Type result : primal.getResultTypes()) {
        FailureOr<Type> derivative = cotangentType(result, primal);
        if (failed(derivative)) {
            forward.erase();
            return primal.emitError("Program VJP result is not differentiable");
        }
        backwardArguments.push_back(*derivative);
    }
    SmallVector<Type> backwardResults;
    for (unsigned index : wrtIndices)
        backwardResults.push_back(*gradientDestType(primal.getArgument(index).getType(), primal));
    auto backward = func::FuncOp::create(moduleBuilder, primal.getLoc(), options.backwardSymbol,
                                         FunctionType::get(primal.getContext(), backwardArguments, backwardResults));
    backward->setAttr("vernon_program.graph", moduleBuilder.getStringAttr("backward"));
    bool committed = false;
    auto cleanup = llvm::make_scope_exit([&] {
        if (!committed) {
            forward.erase();
            backward.erase();
        }
    });

    Block *entry = backward.addEntryBlock();
    OpBuilder builder = OpBuilder::atBlockEnd(entry);
    for (unsigned index = 0; index < retainedValues.size(); ++index) {
        backward.setArgAttr(index, kCaptureForwardValueAttr, builder.getI32IntegerAttr(index));
        applyProgramLanguageAbi(backward, index, getProgramValueLanguageAbi(retainedValues[index]), false);
    }
    for (unsigned index = 0; index < primal.getNumResults(); ++index) {
        const unsigned argument = retainedValues.size() + index;
        if (StringRef name = resultSourceName(primal, index); !name.empty())
            backward.setArgAttr(argument, "vernon.source_name", builder.getStringAttr(name));
        backward.setArgAttr(argument, "vernon.autodiff_role", builder.getStringAttr("cotangent"));
        applyProgramLanguageAbi(
            backward, argument,
            programLanguageAbiFromAttrs(primal.getResultAttrDict(index), backward.getArgument(argument).getType()),
            false);
    }
    for (auto [result, primalArgument] : llvm::enumerate(wrtIndices)) {
        if (StringRef name = sourceName(primal, primalArgument); !name.empty())
            backward.setResultAttr(result, "vernon.source_name", builder.getStringAttr(name));
        backward.setResultAttr(result, "vernon.autodiff_role", builder.getStringAttr("gradient"));
        applyProgramLanguageAbi(
            backward, static_cast<unsigned>(result),
            programLanguageAbiFromAttrs(primal.getArgAttrDict(primalArgument), backward.getResultTypes()[result]),
            true);
    }
    IRMapping primals;
    for (auto [source, target] :
         llvm::zip_equal(retainedValues, entry->getArguments().take_front(retainedValues.size())))
        primals.map(source, target);

    DenseMap<Value, Value> adjoints;
    const unsigned cotangentBase = retainedValues.size();
    for (auto [index, result] : llvm::enumerate(returnOp.getOperands())) {
        Value publicCotangent = entry->getArgument(cotangentBase + index);
        FailureOr<Type> interior = derivativeType(result.getType(), primal);
        if (failed(interior))
            return primal.emitError("Program VJP result is not differentiable");
        Value seed = copyValue(builder, primal.getLoc(), publicCotangent, *interior, primals.lookup(result));
        if (!seed)
            return primal.emitError("Program VJP cannot copy a public cotangent into an interior adjoint");
        adjoints[result] = seed;
    }
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    for (Operation *operation : llvm::reverse(primalOperations)) {
        bool active = llvm::any_of(operation->getResults(), [&](Value result) { return adjoints.contains(result); });
        if (!active)
            continue;
        SmallVector<Value> primalOperands;
        SmallVector<Value> primalResults;
        SmallVector<Value> resultCotangents;
        SmallVector<unsigned> activeOperands;
        for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
            primalOperands.push_back(primals.lookup(operand));
            // Local VJP wrt is the callee's read inputs, not every involved
            // buffer. Write-only dests are kernel outputs / cotangents.
            if (isWriteOnlyProgramOperand(operation, index))
                continue;
            if (succeeded(derivativeType(operand.getType(), operation)))
                activeOperands.push_back(index);
        }
        for (Value result : operation->getResults()) {
            primalResults.push_back(primals.lookup(result));
            Value cotangent = adjoints.lookup(result);
            if (!cotangent) {
                FailureOr<Type> type = derivativeType(result.getType(), operation);
                if (failed(type) ||
                    !(cotangent = createZero(builder, operation->getLoc(), *type, primals.lookup(result))))
                    return operation->emitError("Program VJP cannot create a zero result cotangent");
            }
            resultCotangents.push_back(cotangent);
        }
        AutodiffVjpBuildContext context{builder,       operation->getLoc(), primalOperands,
                                        primalResults, resultCotangents,    activeOperands};
        FailureOr<SmallVector<Value>> contributions = failure();
        if (auto compute = dyn_cast<ComputeOp>(operation)) {
            contributions = buildComputeVjp(compute, context);
        } else if (auto intrinsic = dyn_cast<IntrinsicOp>(operation); intrinsic && intrinsic.getName() == "matmul") {
            contributions = buildProgramMatmulVjp(intrinsic, context);
        } else if (isa<arith::ConstantOp>(operation) ||
                   (isa<IntrinsicOp>(operation) &&
                    isProgramAllocIntrinsicName(cast<IntrinsicOp>(operation).getName()))) {
            // Allocations and constants introduce a new owner or a literal. They
            // do not depend on operand values, so result cotangents stop here.
            continue;
        } else {
            const DifferentiationRule *rule = registry.lookup(operation);
            if (!rule)
                return operation->emitError("Program VJP has no registered differentiation rule");
            contributions = rule->buildVjp(operation, context);
        }
        if (failed(contributions))
            return failure();
        for (auto [operand, contribution] : llvm::zip_equal(operation->getOperands(), *contributions)) {
            if (!contribution)
                continue;
            Value previous = adjoints.lookup(operand);
            Value accumulated =
                previous ? addValues(builder, operation->getLoc(), previous, contribution) : contribution;
            if (!accumulated)
                return operation->emitError("Program VJP cannot accumulate derivative values");
            adjoints[operand] = accumulated;
        }
    }

    SmallVector<Value> gradients;
    for (unsigned index : wrtIndices) {
        Value primalArgument = primal.getArgument(index);
        Value gradient = adjoints.lookup(primalArgument);
        Type publicType = backwardResults[gradients.size()];
        if (!gradient)
            gradient = createZero(builder, primal.getLoc(), publicType, primals.lookup(primalArgument));
        else if (gradient.getType() != publicType)
            gradient = copyValue(builder, primal.getLoc(), gradient, publicType, primals.lookup(primalArgument));
        if (!gradient)
            return primal.emitError("Program VJP cannot create a public input gradient");
        gradients.push_back(gradient);
    }
    func::ReturnOp::create(builder, primal.getLoc(), gradients);
    llvm::BitVector unusedCaptures(backward.getNumArguments());
    for (unsigned index = 0; index < retainedValues.size(); ++index)
        if (backward.getArgument(index).use_empty())
            unusedCaptures.set(index);
    if (unusedCaptures.any() && failed(backward.eraseArguments(unusedCaptures)))
        return backward.emitError("cannot remove unused Program residual captures");
    if (failed(verify(forward)) || failed(verify(backward)))
        return primal.emitError("generated Program VJP graph failed verification");
    primal.erase();
    committed = true;
    return success();
}

namespace {

struct VernonProgramVjpPass final : PassWrapper<VernonProgramVjpPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonProgramVjpPass)

    VernonProgramVjpPass() = default;
    VernonProgramVjpPass(const VernonProgramVjpPass &other) : PassWrapper(other), options(other.options) {}
    explicit VernonProgramVjpPass(ProgramVjpOptions options) : options(std::move(options)) {}

    StringRef getArgument() const final { return "vernon-program-vjp"; }
    StringRef getDescription() const final { return "Derive a backward Program graph from a semantic forward graph"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, func::FuncDialect, tensor::TensorDialect, VernonDialect,
                        VernonProgramDialect>();
    }

    void runOnOperation() override {
        ProgramVjpOptions selected = options;
        if (selected.forwardSymbol == "forward" && !forward.empty())
            selected.forwardSymbol = forward;
        if (selected.backwardSymbol == "backward" && !backward.empty())
            selected.backwardSymbol = backward;
        SmallVector<func::FuncOp> primals;
        getOperation().walk([&](func::FuncOp function) {
            auto kind = function->getAttrOfType<StringAttr>("vernon_program.graph");
            if (kind && kind.getValue() == "primal")
                primals.push_back(function);
        });
        if (primals.size() != 1) {
            getOperation().emitError(
                "Program VJP requires exactly one function marked vernon_program.graph = \"primal\"");
            return signalPassFailure();
        }
        if (selected.wrtBoundaryIndices.empty()) {
            SmallVector<StringRef> publicPaths;
            publicPaths.reserve(wrt.size());
            for (const std::string &path : wrt)
                publicPaths.push_back(path);
            FailureOr<SmallVector<unsigned>> indices = resolveProgramWrtBoundaryIndices(primals.front(), publicPaths);
            if (failed(indices))
                return signalPassFailure();
            selected.wrtBoundaryIndices = std::move(*indices);
        }
        if (failed(buildProgramVjp(primals.front(), selected)))
            signalPassFailure();
    }

    ProgramVjpOptions options;
    ListOption<std::string> wrt{*this, "wrt", llvm::cl::desc("Program input names to differentiate"),
                                llvm::cl::ZeroOrMore};
    Option<std::string> forward{*this, "forward", llvm::cl::desc("Generated forward symbol"), llvm::cl::init("")};
    Option<std::string> backward{*this, "backward", llvm::cl::desc("Generated backward symbol"), llvm::cl::init("")};
};

} // namespace

std::unique_ptr<Pass> createVernonProgramVjpPass() { return std::make_unique<VernonProgramVjpPass>(); }

std::unique_ptr<Pass> createVernonProgramVjpPass(ProgramVjpOptions options) {
    return std::make_unique<VernonProgramVjpPass>(std::move(options));
}

void registerVernonProgramVjpPass() { PassRegistration<VernonProgramVjpPass>(); }

} // namespace mlir::vernon::program
