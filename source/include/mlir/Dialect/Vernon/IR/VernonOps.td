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
  let summary = "Allocate a statically shaped workgroup TensorView";
  let results = (outs Vernon_TensorViewType:$result);
  let hasVerifier = 1;
}

def Vernon_LoadOp : Vernon_Op<"load"> {
  let summary = "Load one element through a ranked TensorView index";
  let arguments = (ins Vernon_TensorViewType:$storage,
      Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Vernon_StoreOp : Vernon_Op<"store"> {
  let summary = "Store one element through a ranked TensorView index";
  let arguments = (ins AnyType:$value, Vernon_TensorViewType:$storage,
      Variadic<Index>:$indices);
  let hasVerifier = 1;
}

def Vernon_AtomicOp : Vernon_Op<"atomic"> {
  let summary = "Atomic read-modify-write with scope inferred from TensorView";
  let arguments = (ins Vernon_TensorViewType:$storage, Variadic<Index>:$indices,
      AnyType:$value, StrAttr:$atomic_kind, StrAttr:$ordering);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Vernon_ReduceSumOp : Vernon_Op<"reduce_sum"> {
  let summary = "Grid-independent sum accumulation contribution";
  let description = [{
    Adds one logical invocation's contribution to a destination element.
    Target lowering selects an atomic add when available, or a canonical
    serial-grid fallback when deterministic execution is required.
  }];
  let arguments = (ins AnyType:$value, Vernon_TensorViewType:$storage,
      Variadic<Index>:$indices, BoolAttr:$deterministic);
  let hasVerifier = 1;
}

def Vernon_ScatterAddOp : Vernon_Op<"scatter_add"> {
  let summary = "Grid-independent indexed accumulation contribution";
  let description = [{
    Adds one logical invocation's contribution at the supplied destination
    indices. `disjoint` is an optimization hint backed by grid-independent
    injectivity evidence; when present, lowering may use a direct store instead
    of a read-modify-write.
  }];
  let arguments = (ins AnyType:$value, Vernon_TensorViewType:$storage,
      Variadic<Index>:$indices, BoolAttr:$deterministic,
      OptionalAttr<UnitAttr>:$disjoint);
  let hasVerifier = 1;
}

def Vernon_PhysicalLoadOp : Vernon_Op<"physical_load"> {
  let summary = "Internal load after TensorView index projection";
  let arguments = (ins Vernon_TensorViewType:$storage, Index:$index);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Vernon_PhysicalStoreOp : Vernon_Op<"physical_store"> {
  let summary = "Internal store after TensorView index projection";
  let arguments = (ins AnyType:$value, Vernon_TensorViewType:$storage,
      Index:$index);
  let hasVerifier = 1;
}

def Vernon_PhysicalAtomicOp : Vernon_Op<"physical_atomic"> {
  let summary = "Internal atomic after TensorView index projection";
  let arguments = (ins Vernon_TensorViewType:$storage, Index:$index,
      AnyType:$value, StrAttr:$atomic_kind, StrAttr:$ordering);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Vernon_BarrierOp : Vernon_Op<"barrier"> {
  let summary = "Memory and execution barrier";
  let arguments = (ins StrAttr:$ordering, StrAttr:$scope);
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Internal logical autodiff protocol
//===----------------------------------------------------------------------===//

def Vernon_AdCaptureYieldOp
    : Vernon_Op<"ad.capture_yield", [Terminator, HasParent<"AdCaptureOp">]> {
  let summary = "Yield the successfully captured invocation handles";
  let arguments = (ins Vernon_AdTapeType:$tape,
      Vernon_AdRegionHeaderType:$root_region);
  let hasVerifier = 1;
}

def Vernon_AdCaptureOp
    : Vernon_Op<
          "ad.capture", [SingleBlockImplicitTerminator<"AdCaptureYieldOp">,
                         RecursiveMemoryEffects]> {
  let summary = "Capture one side-effect-free autodiff invocation";
  let description = [{
    Executes an augmented-forward capture transaction. The body may reserve and
    write logical tape records but may not perform externally visible writes.
    The results return the invocation tape, its root region, and the logical
    capture outcome: success, exact required byte count, and arithmetic
    overflow. Physical capacity and backend status representation are not part
    of this operation.
  }];
  let results = (outs Vernon_AdTapeType:$tape,
      Vernon_AdRegionHeaderType:$root_region, I1:$success,
      Index:$required_bytes, I1:$overflow);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def Vernon_AdCommitYieldOp
    : Vernon_Op<"ad.commit_yield", [Terminator, HasParent<"AdCommitOp">]> {
  let summary = "Terminate an autodiff commit transaction";
}

def Vernon_AdCommitOp
    : Vernon_Op<"ad.commit", [SingleBlockImplicitTerminator<"AdCommitYieldOp">,
                              RecursiveMemoryEffects]> {
  let summary = "Conditionally commit visible effects after successful capture";
  let description = [{
    Executes the body only when capture_success is true. The tape and success
    value must be results of the same dominating ad.capture operation.
  }];
  let arguments = (ins Vernon_AdTapeType:$tape, I1:$capture_success);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def Vernon_AdBeginInvocationOp : Vernon_Op<"ad.begin_invocation"> {
  let summary = "Create the tape owned by the enclosing capture invocation";
  let results = (outs Vernon_AdTapeType:$tape);
  let hasVerifier = 1;
}

def Vernon_AdBeginRegionOp : Vernon_Op<"ad.begin_region"> {
  let summary = "Begin a checked dynamic region in an invocation tape";
  let description = [{
    A root region has an empty parent_link and no child_ordinal. A nested region
    has exactly two parent_link operands: its parent region and the checked
    parent record offset returned by ad.reserve_record, together with a
    non-negative child_ordinal. This keeps nested identity logical and
    backend-neutral.
  }];
  let arguments = (ins Vernon_AdTapeType:$tape, Variadic<AnyType>:$parent_link,
      OptionalAttr<I64Attr>:$child_ordinal);
  let results = (outs Vernon_AdRegionHeaderType:$region);
  let hasVerifier = 1;
}

def Vernon_AdReserveRecordOp : Vernon_Op<"ad.reserve_record"> {
  let summary =
      "Reserve one statically described record with checked arithmetic";
  let description = [{
    Reserves record_size bytes at record_alignment in the region. Required-byte
    accounting, alignment, and offset addition are checked and fail closed into
    the enclosing capture outcome.
  }];
  let arguments = (ins Vernon_AdRegionHeaderType:$region, I64Attr:$record_size,
      I64Attr:$record_alignment);
  let results = (outs Index:$record_offset);
  let hasVerifier = 1;
}

def Vernon_AdCheckedIncrementOp : Vernon_Op<"ad.checked_increment"> {
  let summary = "Increment a dynamic execution counter with checked arithmetic";
  let description = [{
    Adds one to a non-negative index counter. Overflow fails the enclosing
    capture transaction instead of wrapping. Backend lowerings own the physical
    failure representation.
  }];
  let arguments = (ins Index:$counter);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def Vernon_AdWriteLeafOp : Vernon_Op<"ad.write_leaf"> {
  let summary = "Write one canonical ABI leaf into a checked record";
  let arguments = (ins Vernon_AdRegionHeaderType:$region, Index:$record_offset,
      AnyType:$value, I64Attr:$leaf_offset);
  let hasVerifier = 1;
}

def Vernon_AdEndRegionOp : Vernon_Op<"ad.end_region"> {
  let summary = "Finalize a dynamic region header";
  let arguments = (ins Vernon_AdRegionHeaderType:$region, Index:$executed_count,
      I32:$exit_kind);
  let hasVerifier = 1;
}

def Vernon_AdReadRecordOffsetOp : Vernon_Op<"ad.read_record_offset", [Pure]> {
  let summary = "Read the checked record offset from a region header";
  let arguments = (ins Vernon_AdRegionHeaderType:$region);
  let results = (outs Index:$record_offset);
  let hasVerifier = 1;
}

def Vernon_AdReadExecutedCountOp : Vernon_Op<"ad.read_executed_count", [Pure]> {
  let summary = "Read the actual execution count from a region header";
  let arguments = (ins Vernon_AdRegionHeaderType:$region);
  let results = (outs Index:$executed_count);
  let hasVerifier = 1;
}

def Vernon_AdReadExitKindOp : Vernon_Op<"ad.read_exit_kind", [Pure]> {
  let summary = "Read the dynamic exit kind from a region header";
  let arguments = (ins Vernon_AdRegionHeaderType:$region);
  let results = (outs I32:$exit_kind);
  let hasVerifier = 1;
}

def Vernon_AdReadNestedRegionOp : Vernon_Op<"ad.read_nested_region", [Pure]> {
  let summary =
      "Read a nested-region handle by parent record and child ordinal";
  let arguments = (ins Vernon_AdRegionHeaderType:$region, Index:$record_index,
      I64Attr:$child_ordinal);
  let results = (outs Vernon_AdRegionHeaderType:$nested_region);
  let hasVerifier = 1;
}

def Vernon_AdReadLeafOp : Vernon_Op<"ad.read_leaf", [Pure]> {
  let summary =
      "Read one typed canonical ABI leaf from a checked record layout";
  let arguments = (ins Vernon_AdRegionHeaderType:$region, Index:$record_index,
      I64Attr:$record_size, I64Attr:$record_alignment, I64Attr:$leaf_offset);
  let results = (outs AnyType:$value);
  let hasVerifier = 1;
}

def Vernon_CpuAdCallbackOp : Vernon_Op<"cpu_ad.callback"> {
  let summary = "Invoke one typed semantic CPU autodiff tape callback";
  let arguments = (ins Index:$descriptor, I32Attr:$callback,
      Variadic<AnyType>:$arguments);
  let results = (outs Variadic<AnyType>:$results);
}

def Vernon_CpuAdRequiredBytesOp : Vernon_Op<"cpu_ad.required_bytes", [Pure]> {
  let summary = "Read the exact required-byte count from the CPU tape descriptor";
  let arguments = (ins Index:$descriptor);
  let results = (outs Index:$required_bytes);
}

#endif // VERNON_OPS
