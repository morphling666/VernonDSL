//===- VernonOps.td - Vernon Operation Definitions ---------*- tablegen -*-===//
//
// Part of the Vernon DSL Project
//
//===----------------------------------------------------------------------===//

#ifndef VERNON_OPS
#define VERNON_OPS

include "mlir/Dialect/Vernon/IR/VernonDialect.td"
include "mlir/Dialect/Vernon/IR/VernonTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Vernon_Op<string mnemonic, list<Trait> traits = []>
    : Op<Vernon_Dialect, mnemonic, traits>;

def Vernon_StructDeclOp : Vernon_Op<"struct", [Pure]> {
  let summary = "Declare the fields of a named DSL struct";
  let arguments = (ins StrArrayAttr:$fields, StrAttr:$sym_name);
}

def Vernon_StructCreateOp : Vernon_Op<"struct_create", [Pure]> {
  let summary = "Construct a DSL struct value";
  let arguments = (ins Variadic<AnyType>:$fields, StrAttr:$type_name);
  let results = (outs AnyType:$result);
}

def Vernon_StructGetOp : Vernon_Op<"struct_get", [Pure]> {
  let summary = "Extract a named field from a DSL struct value";
  let arguments = (ins AnyType:$input, StrAttr:$field, I64Attr:$index);
  let results = (outs AnyType:$result);
}

def Vernon_TupleCreateOp : Vernon_Op<"tuple_create", [Pure]> {
  let summary = "Construct a structural Tuple value";
  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs AnyType:$result);
}

def Vernon_TupleGetOp : Vernon_Op<"tuple_get", [Pure]> {
  let summary = "Extract a constant-indexed Tuple element";
  let arguments = (ins AnyType:$input, I64Attr:$index);
  let results = (outs AnyType:$result);
}

def Vernon_TensorGetOp : Vernon_Op<"tensor_get", [Pure]> {
  let summary = "Extract an element from an aggregate-element Tensor Value";
  let arguments = (ins AnyType:$input, Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
}

def Vernon_SwizzleOp : Vernon_Op<"swizzle", [Pure]> {
  let summary = "Extract or reorder components of a rank-one tensor";
  let arguments = (ins AnyType:$input, StrAttr:$mask);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Vernon_IntrinsicOp : Vernon_Op<"intrinsic"> {
  let summary = "Target-independent shader or compute intrinsic";
  let description = [{
    Represents operations whose semantics are shared by graphics, compute, and
    CPU targets but do not have a suitable standard MLIR operation yet. The
    `name` is part of the stable Vernon IR contract. Target lowering must reject
    unknown names instead of silently dropping the operation.
  }];
  let arguments = (ins Variadic<AnyType>:$operands, StrAttr:$name);
  let results = (outs Optional<AnyType>:$result);
  let hasVerifier = 1;
}

def Vernon_WorkgroupAllocOp : Vernon_Op<"workgroup_alloc"> {
  let summary = "Allocate statically sized workgroup storage";
  let results = (outs Vernon_WorkgroupType:$result);
  let hasVerifier = 1;
}

def Vernon_WorkgroupLoadOp : Vernon_Op<"workgroup_load"> {
  let summary = "Load one workgroup storage element";
  let arguments = (ins Vernon_WorkgroupType:$storage, Index:$index);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Vernon_WorkgroupStoreOp : Vernon_Op<"workgroup_store"> {
  let summary = "Store one workgroup storage element";
  let arguments = (ins AnyType:$value, Vernon_WorkgroupType:$storage, Index:$index);
  let hasVerifier = 1;
}

def Vernon_AtomicOp : Vernon_Op<"atomic"> {
  let summary = "Atomic read-modify-write on workgroup or device storage";
  let arguments = (ins AnyType:$storage, Index:$index, AnyType:$value,
                       StrAttr:$atomic_kind, StrAttr:$ordering, StrAttr:$scope);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Vernon_BarrierOp : Vernon_Op<"barrier"> {
  let summary = "Memory and execution barrier";
  let arguments = (ins StrAttr:$ordering, StrAttr:$scope);
  let hasVerifier = 1;
}

#endif // VERNON_OPS
