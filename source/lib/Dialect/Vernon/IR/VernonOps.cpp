//===- VernonOps.cpp - Vernon Operation Implementation --------*- C++ -*-===//
//
// Part of the Vernon DSL Project
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <limits>

using namespace mlir;
using namespace mlir::vernon;

#define GET_OP_CLASSES
#include "mlir/Dialect/Vernon/IR/VernonOps.cpp.inc"

std::optional<unsigned> mlir::vernon::decodeSwizzleComponent(char component) {
    switch (component) {
    case 'x':
    case 'r':
        return 0;
    case 'y':
    case 'g':
        return 1;
    case 'z':
    case 'b':
        return 2;
    case 'w':
    case 'a':
        return 3;
    default:
        return std::nullopt;
    }
}

LogicalResult SwizzleOp::verify() {
    StringRef mask = getMask();
    if (mask.empty())
        return emitOpError("requires a non-empty component mask");

    Type inputElementType;
    int64_t inputWidth = 0;
    if (auto tensor = dyn_cast<RankedTensorType>(getInput().getType())) {
        if (tensor.getRank() != 1)
            return emitOpError("requires a rank-one tensor or vector input");
        if (tensor.isDynamicDim(0))
            return emitOpError("requires a statically sized component axis");
        inputElementType = tensor.getElementType();
        inputWidth = tensor.getDimSize(0);
    } else if (auto vector = dyn_cast<VectorType>(getInput().getType())) {
        if (vector.getRank() != 1)
            return emitOpError("requires a rank-one tensor or vector input");
        inputElementType = vector.getElementType();
        inputWidth = vector.getDimSize(0);
    } else {
        return emitOpError("requires a rank-one tensor or vector input");
    }

    for (char component : mask) {
        std::optional<unsigned> index = decodeSwizzleComponent(component);
        if (!index)
            return emitOpError() << "contains invalid component '" << component << "'; expected an XYZW or RGBA alias";
        if (*index >= static_cast<uint64_t>(inputWidth))
            return emitOpError() << "component '" << component << "' is out of bounds for input width " << inputWidth;
    }

    Type resultType = getResult().getType();
    if (mask.size() == 1) {
        if (resultType != inputElementType)
            return emitOpError() << "single-component result must have input element type " << inputElementType;
        return success();
    }

    Type resultElementType;
    int64_t resultWidth = 0;
    if (auto tensor = dyn_cast<RankedTensorType>(resultType)) {
        if (tensor.getRank() != 1 || tensor.isDynamicDim(0))
            return emitOpError("multi-component result must be a statically sized rank-one tensor "
                               "or vector");
        resultElementType = tensor.getElementType();
        resultWidth = tensor.getDimSize(0);
    } else if (auto vector = dyn_cast<VectorType>(resultType)) {
        if (vector.getRank() != 1)
            return emitOpError("multi-component result must be a statically sized rank-one tensor "
                               "or vector");
        resultElementType = vector.getElementType();
        resultWidth = vector.getDimSize(0);
    } else {
        return emitOpError("multi-component result must be a statically sized rank-one tensor "
                           "or vector");
    }
    if (resultWidth != static_cast<int64_t>(mask.size()))
        return emitOpError() << "result width " << resultWidth << " does not match mask length " << mask.size();
    if (resultElementType != inputElementType)
        return emitOpError() << "result element type " << resultElementType << " does not match input element type "
                             << inputElementType;
    return success();
}

LogicalResult IntrinsicOp::verify() {
    if (getNameAttr().getValue().empty())
        return emitOpError("requires a non-empty intrinsic name");
    if (getName() != "texture_sample" && getName() != "texture_size" && getName() != "texture_load" &&
        getName() != "texture_store")
        return success();

    if (getNumOperands() == 0)
        return emitOpError() << getName() << " requires a texture operand";

    auto shapedWidthAndElement = [](Type type) -> std::optional<std::pair<int64_t, Type>> {
        if (auto tensor = dyn_cast<RankedTensorType>(type)) {
            if (tensor.getRank() != 1 || tensor.isDynamicDim(0))
                return std::nullopt;
            return std::pair<int64_t, Type>{tensor.getDimSize(0), tensor.getElementType()};
        }
        if (auto vector = dyn_cast<VectorType>(type)) {
            if (vector.getRank() != 1)
                return std::nullopt;
            return std::pair<int64_t, Type>{vector.getDimSize(0), vector.getElementType()};
        }
        return std::nullopt;
    };

    if (getName() == "texture_load" || getName() == "texture_store") {
        auto texture = dyn_cast<TextureType>(getOperand(0).getType());
        if (!texture || texture.getAccess() == "sampled")
            return emitOpError() << getName() << " operand #0 must be a Vernon storage Texture";
        const bool load = getName() == "texture_load";
        if ((load && texture.getAccess() == "write") || (!load && texture.getAccess() == "read"))
            return emitOpError() << getName() << " is incompatible with " << texture.getAccess() << " access";
        if (getNumOperands() != (load ? 2u : 3u) || getNumResults() != (load ? 1u : 0u))
            return emitOpError() << getName() << " has an invalid operand or result count";
        const int64_t rank = texture.getDimension() == "3d" ? 3 : 2;
        auto coordinates = shapedWidthAndElement(getOperand(1).getType());
        if (!coordinates || coordinates->first != rank || !coordinates->second.isSignlessInteger(32))
            return emitOpError() << getName() << " coordinates must be a " << rank << "-component i32 vector";
        Type texelType = load ? getResult().getType() : getOperand(2).getType();
        auto texel = shapedWidthAndElement(texelType);
        if (!texel || texel->first != 4 || texel->second != texture.getElementType())
            return emitOpError() << getName() << " texel must be a 4-component " << texture.getElementType()
                                 << " vector";
        return success();
    }

    if (getNumResults() != 1)
        return emitOpError() << getName() << " requires exactly one result";
    auto texture = dyn_cast<TextureType>(getOperand(0).getType());
    if (!texture)
        return emitOpError() << getName() << " operand #0 must be a Vernon texture";
    if (texture.getAccess() != "sampled")
        return emitOpError() << getName() << " requires a sampled Texture";

    if (getName() == "texture_size") {
        if (getNumOperands() < 1 || getNumOperands() > 2)
            return emitOpError("texture_size requires a texture and an optional integer lod");
        if (getNumOperands() == 2 && !getOperand(1).getType().isSignlessInteger(32))
            return emitOpError("texture_size lod must be a 32-bit integer scalar");
        const int64_t expectedResultWidth = texture.getDimension() == "3d" ? 3 : 2;
        auto result = shapedWidthAndElement(getResult().getType());
        if (!result || result->first != expectedResultWidth || !result->second.isSignlessInteger(32))
            return emitOpError() << "texture_size result must be a statically sized rank-one " << expectedResultWidth
                                 << "-component i32 tensor or vector";
        return success();
    }

    if (getNumOperands() < 3 || getNumOperands() > 4)
        return emitOpError("texture_sample requires texture, sampler, coordinates, and optional "
                           "lod");
    if (!isa<SamplerType>(getOperand(1).getType()))
        return emitOpError("texture_sample operand #1 must be a Vernon sampler");

    std::optional<std::pair<int64_t, Type>> coordinates = shapedWidthAndElement(getOperand(2).getType());
    const int64_t expectedCoordinateWidth = texture.getDimension() == "2d" ? 2 : 3;
    if (!coordinates || coordinates->first != expectedCoordinateWidth ||
        coordinates->second != texture.getElementType() || !isa<FloatType>(coordinates->second))
        return emitOpError() << "texture_sample operand #2 must be a statically sized rank-one "
                             << expectedCoordinateWidth << "-component " << texture.getElementType()
                             << " tensor or vector";
    if (getNumOperands() == 4 && !isa<FloatType>(getOperand(3).getType()))
        return emitOpError("texture_sample operand #3 lod must be a floating-point scalar");

    std::optional<std::pair<int64_t, Type>> result = shapedWidthAndElement(getResult().getType());
    if (!result || result->first != 4 || result->second != texture.getElementType())
        return emitOpError() << "texture_sample result must be a statically sized rank-one "
                                "4-component "
                             << texture.getElementType() << " tensor or vector";
    return success();
}

LogicalResult WorkgroupAllocOp::verify() {
    TensorViewType type = getResult().getType();
    if (type.getAddressSpace() != "workgroup")
        return emitOpError("result must use the workgroup address space");
    if (type.getAccess() != "read_write")
        return emitOpError("result must be a read_write TensorView");
    if (llvm::any_of(type.getShape(), [](int64_t extent) { return extent <= 0; }))
        return emitOpError("requires positive static dimensions");
    ModuleOp module = (*this)->getParentOfType<ModuleOp>();
    if (!module)
        return emitOpError("must be nested in a module");
    FailureOr<WorkgroupPhysicalStoragePlan> plan = getWorkgroupPhysicalStoragePlan(type, module);
    if (failed(plan))
        return emitOpError("cannot derive a finite canonical physical storage plan");
    if (plan->totalPhysicalBytes > kPortableWorkgroupStorageLimit)
        return emitOpError("exceeds the portable 16 KiB workgroup storage limit");
    return success();
}

LogicalResult LoadOp::verify() {
    auto type = dyn_cast<TensorViewType>(getStorage().getType());
    if (!type)
        return emitOpError("storage must be a TensorView");
    if (type.getAccess() == "write")
        return emitOpError("cannot load through a write-only TensorView");
    if (getIndices().size() != type.getShape().size())
        return emitOpError("requires one index per TensorView dimension");
    return getResult().getType() == type.getElementType() ? success()
                                                          : emitOpError("result type must match the element type");
}

LogicalResult StoreOp::verify() {
    auto type = dyn_cast<TensorViewType>(getStorage().getType());
    if (!type)
        return emitOpError("storage must be a TensorView");
    if (type.getAccess() == "read")
        return emitOpError("cannot store through a read-only TensorView");
    if (getIndices().size() != type.getShape().size())
        return emitOpError("requires one index per TensorView dimension");
    return getValue().getType() == type.getElementType() ? success()
                                                         : emitOpError("value type must match the element type");
}

LogicalResult AtomicOp::verify() {
    auto view = dyn_cast<TensorViewType>(getStorage().getType());
    if (!view)
        return emitOpError("storage must be a TensorView");
    if (view.getAccess() == "read")
        return emitOpError("requires writable TensorView storage");
    if (view.getAddressSpace() != "device" && view.getAddressSpace() != "workgroup")
        return emitOpError("requires device or workgroup TensorView storage");
    if (getIndices().size() != view.getShape().size())
        return emitOpError("requires one index per TensorView dimension");
    Type elementType = view.getElementType();
    const bool supportedElement =
        elementType.isSignlessInteger(32) || ((elementType.isF32() || elementType.isF64()) && getAtomicKind() == "add");
    if (!supportedElement || getValue().getType() != elementType || getResult().getType() != elementType)
        return emitOpError("requires matching i32 types or f32/f64 types for atomic add");
    if (getAtomicKind() != "add" && getAtomicKind() != "min" && getAtomicKind() != "max" && getAtomicKind() != "umin" &&
        getAtomicKind() != "umax" && getAtomicKind() != "exchange")
        return emitOpError("operation must be add, min, max, umin, umax, or exchange");
    if (getOrdering() != "relaxed")
        return emitOpError("currently supports only relaxed memory ordering");
    return success();
}

static LogicalResult verifyAccumulationContribution(Operation *operation, Value value, Value storage,
                                                    ValueRange indices) {
    auto view = dyn_cast<TensorViewType>(storage.getType());
    if (!view)
        return operation->emitOpError("storage must be a TensorView");
    if (view.getAccess() == "read")
        return operation->emitOpError("requires writable TensorView storage");
    if (view.getAddressSpace() != "device")
        return operation->emitOpError("requires device TensorView storage");
    if (indices.size() != view.getShape().size())
        return operation->emitOpError("requires one index per TensorView dimension");
    Type elementType = view.getElementType();
    Type scalarType = elementType;
    if (auto shaped = dyn_cast<ShapedType>(elementType)) {
        if (!shaped.hasStaticShape())
            return operation->emitOpError("requires statically shaped gradient elements");
        scalarType = shaped.getElementType();
    }
    if (!scalarType.isF16() && !scalarType.isF32() && !scalarType.isF64())
        return operation->emitOpError("requires floating scalar or tensor gradient storage");
    if (value.getType() != elementType)
        return operation->emitOpError("contribution type must match the storage element type");
    return success();
}

LogicalResult ReduceSumOp::verify() {
    return verifyAccumulationContribution(getOperation(), getValue(), getStorage(), getIndices());
}

LogicalResult ScatterAddOp::verify() {
    return verifyAccumulationContribution(getOperation(), getValue(), getStorage(), getIndices());
}

LogicalResult PhysicalLoadOp::verify() {
    auto type = dyn_cast<TensorViewType>(getStorage().getType());
    if (!type)
        return emitOpError("storage must be a TensorView");
    if (type.getAccess() == "write")
        return emitOpError("cannot load through a write-only TensorView");
    return getResult().getType() == type.getElementType() ? success()
                                                          : emitOpError("result type must match the element type");
}

LogicalResult PhysicalStoreOp::verify() {
    auto type = dyn_cast<TensorViewType>(getStorage().getType());
    if (!type)
        return emitOpError("storage must be a TensorView");
    if (type.getAccess() == "read")
        return emitOpError("cannot store through a read-only TensorView");
    return getValue().getType() == type.getElementType() ? success()
                                                         : emitOpError("value type must match the element type");
}

LogicalResult PhysicalAtomicOp::verify() {
    auto view = dyn_cast<TensorViewType>(getStorage().getType());
    if (!view)
        return emitOpError("storage must be a TensorView");
    if (view.getAccess() == "read")
        return emitOpError("requires writable TensorView storage");
    if (view.getAddressSpace() != "device" && view.getAddressSpace() != "workgroup")
        return emitOpError("requires device or workgroup TensorView storage");
    Type elementType = view.getElementType();
    const bool supportedElement =
        elementType.isSignlessInteger(32) || ((elementType.isF32() || elementType.isF64()) && getAtomicKind() == "add");
    if (!supportedElement || getValue().getType() != elementType || getResult().getType() != elementType)
        return emitOpError("requires matching i32 types or f32/f64 types for atomic add");
    if (getAtomicKind() != "add" && getAtomicKind() != "min" && getAtomicKind() != "max" && getAtomicKind() != "umin" &&
        getAtomicKind() != "umax" && getAtomicKind() != "exchange")
        return emitOpError("operation must be add, min, max, umin, umax, or exchange");
    if (getOrdering() != "relaxed")
        return emitOpError("currently supports only relaxed memory ordering");
    return success();
}

LogicalResult BarrierOp::verify() {
    if (getOrdering() != "acquire" && getOrdering() != "release" && getOrdering() != "acquire_release" &&
        getOrdering() != "sequential")
        return emitOpError("barrier ordering must be acquire, release, acquire_release, or sequential");
    if (getScope() != "workgroup" && getScope() != "device")
        return emitOpError("barrier scope must be workgroup or device");
    return success();
}

namespace {

AdCaptureOp enclosingCapture(Operation *operation) { return operation->getParentOfType<AdCaptureOp>(); }

bool isInsideCommit(Operation *operation) { return static_cast<bool>(operation->getParentOfType<AdCommitOp>()); }

LogicalResult verifyCaptureMutation(Operation *operation) {
    if (!enclosingCapture(operation))
        return operation->emitOpError("is only legal inside vernon.ad.capture");
    if (isInsideCommit(operation))
        return operation->emitOpError("cannot be nested in vernon.ad.commit");
    return success();
}

LogicalResult verifyReverseRead(Operation *operation, Value region) {
    if (enclosingCapture(operation) || isInsideCommit(operation))
        return operation->emitOpError("is only legal in the reverse read phase outside capture and commit");
    if (Operation *definition = region.getDefiningOp()) {
        if (!isa<AdCaptureOp, AdReadNestedRegionOp>(definition))
            return operation->emitOpError("requires a finalized capture or nested-region handle");
    } else if (!isa<BlockArgument>(region)) {
        return operation->emitOpError("requires a region block argument or finalized region handle");
    }
    return success();
}

bool isPowerOfTwo(int64_t value) {
    return value > 0 && (static_cast<uint64_t>(value) & (static_cast<uint64_t>(value) - 1)) == 0;
}

struct CanonicalLeafLayout {
    int64_t size;
    int64_t alignment;
};

std::optional<CanonicalLeafLayout> getCanonicalLeafLayout(Operation *operation, Type type) {
    ModuleOp module = operation->getParentOfType<ModuleOp>();
    if (!module)
        return std::nullopt;
    FailureOr<ValueAbiLayout> layout = getValueAbiLayout(type, module);
    if (failed(layout) || !layout->tree.root || layout->tree.root->kind != CanonicalAbiNodeKind::Scalar ||
        layout->leaves.size() != 1 || layout->leaves.front().scalarCount != 1 ||
        layout->size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        layout->alignment > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return std::nullopt;
    return CanonicalLeafLayout{static_cast<int64_t>(layout->size), static_cast<int64_t>(layout->alignment)};
}

LogicalResult verifyRecordLayout(Operation *operation, int64_t recordSize, int64_t recordAlignment) {
    if (recordSize <= 0)
        return operation->emitOpError("requires a positive record_size");
    if (!isPowerOfTwo(recordAlignment))
        return operation->emitOpError("requires record_alignment to be a positive power of two");
    if (recordSize % recordAlignment != 0)
        return operation->emitOpError("requires record_size to be a multiple of record_alignment");
    return success();
}

LogicalResult verifyLeafLayout(Operation *operation, Type leafType, int64_t recordSize, int64_t recordAlignment,
                               int64_t leafOffset) {
    if (failed(verifyRecordLayout(operation, recordSize, recordAlignment)))
        return failure();
    std::optional<CanonicalLeafLayout> leaf = getCanonicalLeafLayout(operation, leafType);
    if (!leaf)
        return operation->emitOpError("requires a canonical scalar ABI leaf type (i1, i32, f16, f32, or f64)");
    if (recordAlignment < leaf->alignment)
        return operation->emitOpError("record_alignment is smaller than the canonical leaf alignment");
    if (leafOffset < 0 || leafOffset % leaf->alignment != 0)
        return operation->emitOpError("leaf_offset is negative or violates canonical leaf alignment");
    if (leafOffset > recordSize || leaf->size > recordSize - leafOffset)
        return operation->emitOpError("canonical leaf extends beyond the checked record layout");
    return success();
}

bool isLogicalAdCaptureOperation(Operation *operation) {
    return isa<AdCaptureYieldOp, AdBeginInvocationOp, AdBeginRegionOp, AdReserveRecordOp, AdCheckedIncrementOp,
               AdWriteLeafOp, AdEndRegionOp>(operation);
}

bool isAllowedCaptureOperation(Operation *operation) {
    if (isLogicalAdCaptureOperation(operation))
        return true;
    if (operation->getNumRegions() != 0) {
        if (!operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
            return false;
        auto effects = dyn_cast<MemoryEffectOpInterface>(operation);
        if (!effects)
            return true;
        SmallVector<MemoryEffects::EffectInstance> instances;
        effects.getEffects(instances);
        return llvm::all_of(instances, [](const MemoryEffects::EffectInstance &effect) {
            return isa<MemoryEffects::Read, MemoryEffects::Allocate, MemoryEffects::Free>(effect.getEffect());
        });
    }
    if (isMemoryEffectFree(operation))
        return true;
    if (isa<IntrinsicOp>(operation) && operation->getNumResults() == 1)
        return true;
    if (isa<LoadOp, PhysicalLoadOp, WorkgroupAllocOp>(operation))
        return true;
    if (isa<StoreOp, PhysicalStoreOp, ReduceSumOp, ScatterAddOp>(operation))
        return operation->hasAttrOfType<UnitAttr>("vernon.ad.functionalized");
    if (auto atomic = dyn_cast<AtomicOp>(operation))
        return atomic.getStorage().getType().getAddressSpace() != "device" ||
               operation->hasAttrOfType<UnitAttr>("vernon.ad.functionalized");
    if (auto atomic = dyn_cast<PhysicalAtomicOp>(operation))
        return atomic.getStorage().getType().getAddressSpace() != "device";
    if (auto barrier = dyn_cast<BarrierOp>(operation))
        return barrier.getScope() == "workgroup";

    auto effects = dyn_cast<MemoryEffectOpInterface>(operation);
    if (!effects)
        return false;
    SmallVector<MemoryEffects::EffectInstance> instances;
    effects.getEffects(instances);
    return llvm::all_of(instances, [](const MemoryEffects::EffectInstance &effect) {
        return isa<MemoryEffects::Read, MemoryEffects::Allocate, MemoryEffects::Free>(effect.getEffect());
    });
}

Operation *getAncestorInBlock(Operation *operation, Block *block) {
    while (operation && operation->getBlock() != block)
        operation = operation->getParentOp();
    return operation;
}

LogicalResult verifyRegionHandle(AdBeginRegionOp begin) {
    AdCaptureOp capture = enclosingCapture(begin);
    unsigned endCount = 0;
    AdEndRegionOp end;
    for (Operation *user : begin.getRegion().getUsers()) {
        if (auto candidate = dyn_cast<AdEndRegionOp>(user)) {
            ++endCount;
            end = candidate;
        } else if (!isa<AdReserveRecordOp, AdWriteLeafOp, AdBeginRegionOp, AdCaptureYieldOp, AdReadRecordOffsetOp,
                        AdReadExecutedCountOp, AdReadExitKindOp, AdReadNestedRegionOp, AdReadLeafOp>(user)) {
            return begin.emitOpError() << "region handle has illegal capture-phase user " << user->getName();
        }
        if (enclosingCapture(user) != capture)
            return begin.emitOpError("region handle escapes its owning capture");
    }
    if (endCount != 1)
        return begin.emitOpError() << "region handle must be finalized exactly once; found " << endCount
                                   << " vernon.ad.end_region users";
    if (begin->getBlock() != end->getBlock())
        return begin.emitOpError("region begin and end must be in the same structured block");

    for (Operation *user : begin.getRegion().getUsers()) {
        if (user == end.getOperation() || isa<AdCaptureYieldOp>(user))
            continue;
        Operation *ancestor = getAncestorInBlock(user, end->getBlock());
        if (!ancestor)
            return user->emitOpError("uses a region handle outside its structured lifetime block");
        if (end->isBeforeInBlock(ancestor))
            return user->emitOpError("uses a region handle after vernon.ad.end_region");
    }
    return success();
}

} // namespace

LogicalResult AdCaptureYieldOp::verify() {
    auto capture = cast<AdCaptureOp>((*this)->getParentOp());
    auto invocation = getTape().getDefiningOp<AdBeginInvocationOp>();
    auto root = getRootRegion().getDefiningOp<AdBeginRegionOp>();
    if (!invocation || enclosingCapture(invocation) != capture)
        return emitOpError("tape must be produced by the enclosing capture's begin_invocation");
    if (!root || enclosingCapture(root) != capture)
        return emitOpError("root_region must be produced by the enclosing capture's begin_region");
    if (root.getTape() != getTape() || !root.getParentLink().empty() || root.getChildOrdinalAttr())
        return emitOpError("must yield the parentless root region owned by the yielded tape");
    return success();
}

LogicalResult AdCaptureOp::verify() {
    if ((*this)->getParentOfType<AdCaptureOp>() || (*this)->getParentOfType<AdCommitOp>())
        return emitOpError("cannot be nested in another capture or commit transaction");
    if (!getBody().hasOneBlock())
        return emitOpError("requires exactly one body block");

    unsigned invocationCount = 0;
    unsigned rootCount = 0;
    WalkResult result = getBody().walk([&](Operation *operation) {
        if (operation == getOperation())
            return WalkResult::advance();
        if (isa<AdCaptureOp, AdCommitOp>(operation)) {
            operation->emitOpError("cannot nest a capture or commit transaction inside capture");
            return WalkResult::interrupt();
        }
        if (isa<AdBeginInvocationOp>(operation))
            ++invocationCount;
        if (auto begin = dyn_cast<AdBeginRegionOp>(operation); begin && begin.getParentLink().empty())
            ++rootCount;
        if (!isAllowedCaptureOperation(operation)) {
            operation->emitOpError(
                "has unclassified or externally visible effects and is illegal during autodiff capture");
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    if (result.wasInterrupted())
        return failure();
    if (invocationCount != 1)
        return emitOpError() << "requires exactly one ad.begin_invocation; found " << invocationCount;
    if (rootCount != 1)
        return emitOpError() << "requires exactly one parentless root region; found " << rootCount;
    return success();
}

LogicalResult AdCommitOp::verify() {
    if ((*this)->getParentOfType<AdCaptureOp>() || (*this)->getParentOfType<AdCommitOp>())
        return emitOpError("cannot be nested in another capture or commit transaction");
    if (!getBody().hasOneBlock())
        return emitOpError("requires exactly one body block");
    auto capture = getTape().getDefiningOp<AdCaptureOp>();
    if (!capture || getTape() != capture.getTape() || getCaptureSuccess() != capture.getSuccess())
        return emitOpError("tape and capture_success must be paired results of the same capture");
    return success();
}

LogicalResult AdBeginInvocationOp::verify() {
    if (failed(verifyCaptureMutation(getOperation())))
        return failure();
    AdCaptureOp capture = enclosingCapture(*this);
    unsigned count = 0;
    capture.getBody().walk([&](AdBeginInvocationOp) { ++count; });
    if (count != 1)
        return emitOpError("the enclosing capture must have exactly one invocation owner");
    for (Operation *user : getTape().getUsers()) {
        if (!isa<AdBeginRegionOp, AdCaptureYieldOp>(user) || enclosingCapture(user) != capture)
            return emitOpError() << "invocation tape has illegal or escaping user " << user->getName();
    }
    return success();
}

static bool areMutuallyExclusiveBranchOperations(Operation *left, Operation *right) {
    auto contains = [](Region &region, Operation *operation) {
        Region *owner = operation->getParentRegion();
        return owner == &region || region.isAncestor(owner);
    };
    for (Operation *ancestor = left->getParentOp(); ancestor; ancestor = ancestor->getParentOp()) {
        auto conditional = dyn_cast<scf::IfOp>(ancestor);
        if (!conditional || conditional.getElseRegion().empty())
            continue;
        const bool leftThen = contains(conditional.getThenRegion(), left);
        const bool leftElse = contains(conditional.getElseRegion(), left);
        const bool rightThen = contains(conditional.getThenRegion(), right);
        const bool rightElse = contains(conditional.getElseRegion(), right);
        if ((leftThen && rightElse) || (leftElse && rightThen))
            return true;
    }
    return false;
}

LogicalResult AdBeginRegionOp::verify() {
    if (failed(verifyCaptureMutation(getOperation())))
        return failure();
    auto invocation = getTape().getDefiningOp<AdBeginInvocationOp>();
    if (!invocation || enclosingCapture(invocation) != enclosingCapture(*this))
        return emitOpError("tape must come from the enclosing capture's begin_invocation");

    ValueRange link = getParentLink();
    if (link.empty()) {
        if (getChildOrdinalAttr())
            return emitOpError("root region must not define child_ordinal");
    } else {
        if (link.size() != 2 || !isa<AdRegionHeaderType>(link[0].getType()) || !link[1].getType().isIndex())
            return emitOpError("nested region parent_link must contain parent region and parent record offset");
        auto parent = link[0].getDefiningOp<AdBeginRegionOp>();
        auto reservation = link[1].getDefiningOp<AdReserveRecordOp>();
        if (!parent || !reservation || reservation.getRegion() != link[0])
            return emitOpError("nested region requires a checked parent record from its direct parent region");
        if (parent.getTape() != getTape() || enclosingCapture(parent) != enclosingCapture(*this))
            return emitOpError("nested region and parent must have the same invocation owner");
        if (!getChildOrdinalAttr() || getChildOrdinalAttr().getInt() < 0)
            return emitOpError("nested region requires a non-negative child_ordinal");
        for (Operation *user : parent.getRegion().getUsers()) {
            auto sibling = dyn_cast<AdBeginRegionOp>(user);
            if (!sibling || sibling == *this || sibling.getParentLink().size() != 2)
                continue;
            if (sibling.getParentLink()[1] == link[1] && sibling.getChildOrdinalAttr() &&
                sibling.getChildOrdinalAttr().getInt() == getChildOrdinalAttr().getInt() &&
                !areMutuallyExclusiveBranchOperations(*this, sibling))
                return emitOpError("duplicates a child_ordinal in the same parent record");
        }
    }
    return verifyRegionHandle(*this);
}

LogicalResult AdReserveRecordOp::verify() {
    if (failed(verifyCaptureMutation(getOperation())))
        return failure();
    auto begin = getRegion().getDefiningOp<AdBeginRegionOp>();
    if (!begin || enclosingCapture(begin) != enclosingCapture(*this))
        return emitOpError("region must be owned by the enclosing capture");
    return verifyRecordLayout(getOperation(), getRecordSizeAttr().getInt(), getRecordAlignmentAttr().getInt());
}

LogicalResult AdCheckedIncrementOp::verify() {
    if (failed(verifyCaptureMutation(getOperation())))
        return failure();
    APInt constant;
    if (matchPattern(getCounter(), m_ConstantInt(&constant))) {
        if (constant.isNegative())
            return emitOpError("counter must not be negative");
        if (constant.isMaxSignedValue())
            return emitOpError("constant counter increment overflows index representation");
    }
    return success();
}

LogicalResult AdWriteLeafOp::verify() {
    if (failed(verifyCaptureMutation(getOperation())))
        return failure();
    auto reservation = getRecordOffset().getDefiningOp<AdReserveRecordOp>();
    if (!reservation || reservation.getRegion() != getRegion())
        return emitOpError("record_offset must come directly from checked ad.reserve_record for this region");
    if (enclosingCapture(reservation) != enclosingCapture(*this))
        return emitOpError("record reservation belongs to a different capture invocation");
    return verifyLeafLayout(getOperation(), getValue().getType(), reservation.getRecordSizeAttr().getInt(),
                            reservation.getRecordAlignmentAttr().getInt(), getLeafOffsetAttr().getInt());
}

LogicalResult AdEndRegionOp::verify() {
    if (failed(verifyCaptureMutation(getOperation())))
        return failure();
    auto begin = getRegion().getDefiningOp<AdBeginRegionOp>();
    if (!begin || enclosingCapture(begin) != enclosingCapture(*this))
        return emitOpError("region must be owned by the enclosing capture");
    APInt constant;
    if (matchPattern(getExecutedCount(), m_ConstantInt(&constant)) && constant.isNegative())
        return emitOpError("executed_count must not be negative");
    if (matchPattern(getExitKind(), m_ConstantInt(&constant)) &&
        (constant.getSExtValue() < 0 || constant.getSExtValue() > 3))
        return emitOpError("exit_kind must be fallthrough(0), break(1), continue(2), or return(3)");

    if (!begin.getParentLink().empty()) {
        auto parent = begin.getParentLink()[0].getDefiningOp<AdBeginRegionOp>();
        AdEndRegionOp parentEnd;
        for (Operation *user : parent.getRegion().getUsers())
            if (auto candidate = dyn_cast<AdEndRegionOp>(user))
                parentEnd = candidate;
        if (parentEnd && parentEnd->getBlock() == getOperation()->getBlock() &&
            parentEnd->isBeforeInBlock(getOperation()))
            return emitOpError("nested region must be finalized before its parent region");
    }
    return success();
}

LogicalResult AdReadRecordOffsetOp::verify() { return verifyReverseRead(getOperation(), getRegion()); }

LogicalResult AdReadExecutedCountOp::verify() { return verifyReverseRead(getOperation(), getRegion()); }

LogicalResult AdReadExitKindOp::verify() { return verifyReverseRead(getOperation(), getRegion()); }

LogicalResult AdReadNestedRegionOp::verify() {
    if (failed(verifyReverseRead(getOperation(), getRegion())))
        return failure();
    if (getChildOrdinalAttr().getInt() < 0)
        return emitOpError("requires a non-negative child_ordinal");
    APInt constant;
    if (matchPattern(getRecordIndex(), m_ConstantInt(&constant)) && constant.isNegative())
        return emitOpError("record_index must not be negative");
    return success();
}

LogicalResult AdReadLeafOp::verify() {
    if (failed(verifyReverseRead(getOperation(), getRegion())))
        return failure();
    APInt constant;
    if (matchPattern(getRecordIndex(), m_ConstantInt(&constant)) && constant.isNegative())
        return emitOpError("record_index must not be negative");
    return verifyLeafLayout(getOperation(), getValue().getType(), getRecordSizeAttr().getInt(),
                            getRecordAlignmentAttr().getInt(), getLeafOffsetAttr().getInt());
}

static Type getAdjointBufferValueType(AdAdjointBufferType buffer) {
    ArrayRef<int64_t> trailing = buffer.getShape().drop_front(buffer.getIndexRank());
    return trailing.empty() ? buffer.getElementType()
                            : static_cast<Type>(RankedTensorType::get(trailing, buffer.getElementType()));
}

static LogicalResult verifyAdjointBufferIndexing(Operation *operation, AdAdjointBufferType buffer, ValueRange indices) {
    if (indices.size() != buffer.getIndexRank())
        return operation->emitOpError("index count must match the adjoint buffer index rank");
    return success();
}

LogicalResult AdAdjointBufferCreateOp::verify() {
    StringRef ownership = getOwnershipAttr() ? getOwnershipAttr().getValue() : "lane_private";
    if (ownership != "lane_private" && ownership != "workgroup_shared")
        return emitOpError("ownership must be 'lane_private' or 'workgroup_shared'");
    AdAdjointBufferType buffer = cast<AdAdjointBufferType>(getBuffer().getType());
    if (Value shapeSource = getShapeSource()) {
        TensorViewType source = cast<TensorViewType>(shapeSource.getType());
        if (source.getShape().size() != buffer.getIndexRank() ||
            !llvm::equal(source.getShape(), buffer.getShape().take_front(buffer.getIndexRank())))
            return emitOpError("shape source must match the indexed adjoint buffer shape");
    } else if (llvm::is_contained(buffer.getShape(), int64_t{-1})) {
        return emitOpError("dynamic adjoint buffer requires a shape source");
    }
    return success();
}

LogicalResult AdAdjointScatterAddOp::verify() {
    AdAdjointBufferType type = cast<AdAdjointBufferType>(getBuffer().getType());
    if (failed(verifyAdjointBufferIndexing(getOperation(), type, getIndices())))
        return failure();
    return getValue().getType() == getAdjointBufferValueType(type)
               ? success()
               : emitOpError("contribution type must match the indexed adjoint buffer element");
}

LogicalResult AdAdjointAccumulateDenseOp::verify() {
    AdAdjointBufferType type = cast<AdAdjointBufferType>(getBuffer().getType());
    if (auto view = dyn_cast<TensorViewType>(getValue().getType())) {
        if (view.getShape().size() != type.getIndexRank() ||
            !llvm::equal(view.getShape(), type.getShape().take_front(type.getIndexRank())) ||
            view.getElementType() != getAdjointBufferValueType(type))
            return emitOpError("TensorView contribution must match the adjoint buffer shape and element");
        return success();
    }
    Type expected = RankedTensorType::get(type.getShape(), type.getElementType());
    return getValue().getType() == expected ? success()
                                            : emitOpError("dense contribution must match the complete adjoint buffer");
}

LogicalResult AdAdjointTakeAndClearOp::verify() {
    AdAdjointBufferType type = cast<AdAdjointBufferType>(getBuffer().getType());
    if (failed(verifyAdjointBufferIndexing(getOperation(), type, getIndices())))
        return failure();
    return getValue().getType() == getAdjointBufferValueType(type)
               ? success()
               : emitOpError("result type must match the indexed adjoint buffer element");
}

LogicalResult AdAdjointPeekOp::verify() {
    AdAdjointBufferType type = cast<AdAdjointBufferType>(getBuffer().getType());
    if (failed(verifyAdjointBufferIndexing(getOperation(), type, getIndices())))
        return failure();
    return getValue().getType() == getAdjointBufferValueType(type)
               ? success()
               : emitOpError("result type must match the indexed adjoint buffer element");
}

LogicalResult AdAdjointStoreOp::verify() {
    AdAdjointBufferType buffer = cast<AdAdjointBufferType>(getBuffer().getType());
    TensorViewType destination = getDestination().getType();
    if (destination.getAccess() == "read")
        return emitOpError("destination must be writable");
    if (destination.getShape().size() != buffer.getIndexRank() ||
        !llvm::equal(destination.getShape(), buffer.getShape().take_front(buffer.getIndexRank())))
        return emitOpError("destination shape must match the adjoint buffer indexed shape");
    Type expectedElement = getAdjointBufferValueType(buffer);
    return destination.getElementType() == expectedElement
               ? success()
               : emitOpError("destination element type must match the adjoint buffer element");
}
