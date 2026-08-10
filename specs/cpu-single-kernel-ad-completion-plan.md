# CPU Single-Kernel AD Completion Plan

## Status and Scope

This plan closes the remaining correctness, ownership, validation, and
scalability gaps in CPU single-kernel `dynamic_v2` autodiff.

The only supported autodiff execution path is CPU `dynamic_v2`. GPU/graphics
autodiff, graph VJP, checkpointing, graph-level Storage SSA, and higher-order AD
remain deferred.

Do not restore a legacy protocol, native GPU AD emitter, whole-kernel serial
fallback, or compatibility execution path. Do not update compiler contract 11
or pipeline contract 15 before the next release boundary.

## Current Findings

The current implementation is not yet complete:

1. Ordinary compute enforces `required_unit_grid_axes`, but CPU AD forward and
   pullback dispatch directly through `CpuWorkgroupScheduler` and do not consume
   that metadata.
2. External gradient accumulation is marked `invocation_private` using
   `static shape && element count > 1`. Ownership must come from memory and
   index proofs, not shape size.
3. Dispatch safety is split between `required_unit_grid_axes` and
   `hasConstantOrdinaryWrite`, with different parsers and enforcement points.
4. TensorView overlap may enumerate and sort up to one million element
   addresses. Large interleaved but disjoint views therefore do not have a
   scalable proof path.
5. Tape snapshots and private gradient staging scale with invocation count.
   This cost is real for general reverse mode, but the runtime must preflight a
   dispatch-wide budget before any observable effect.
6. `reduce_sum`, `scatter_add`, and atomic-add adjoints have compiler coverage
   but insufficient multi-invocation runtime finite-difference coverage.

## Dispatch Contract Decision

### Why a Constraint Is Necessary

Partial global-ID indexing is only conditionally injective:

- `output[gid.x]` is injective only when `grid.y == grid.z == 1`;
- `output[gid.x, gid.y]` is injective only when `grid.z == 1`;
- `output[gid.x, gid.y, gid.z]` needs no unit-axis constraint;
- a constant ordinary store is safe only for one total invocation;
- `gid.x % 2` remains non-injective and must be rejected;
- a leader-guarded `workgroup_id.x` write additionally requires unit Y/Z grid
  axes and a proven local-invocation leader guard.

The compiler is proving:

```text
under dispatch constraints C:
  index(p) == index(q) implies p == q
```

Some representation of `C` is therefore required. User-visible
`dispatch_rank`, `single_invocation`, or similar annotations are not required
and must not be introduced.

### Replace the Narrow Feature

Replace the standalone `required_unit_grid_axes` mechanism with one
compiler-derived `vernon.dispatch_contract`.

The contract is an internal executable property produced by validation and
consumed uniformly by runtime dispatch. The initial contract needs only:

```text
dispatch_contract:
  unit_grid_axes: subset of {0, 1, 2}
  requires_unit_workgroup: boolean
```

The representation may later gain checked limits, but it must not encode
backend strategy or trigger automatic serialization.

The compiler accumulates the residual constraints from every ordinary device
write. Constraints are intersected: if any write requires an axis to be unit,
that axis is unit for the entry. A write with no valid conditional injectivity
proof is rejected.

For a unit-workgroup constant store, the contract requires every grid axis to
be unit. A call with a larger grid fails before staging or dispatch. It is never
silently serialized. Multi-invocation shared loss writes must use an atomic or
formal accumulation operation.

## Required Architecture

### 1. Conditional Injectivity Analysis

Refactor
`source/lib/Dialect/Vernon/Transforms/VernonGlobalIdIndexProof.{h,cpp}` to
return a proof result containing:

- normalized affine index tuple;
- covered global or workgroup axes;
- residual dispatch constraints;
- ownership domain (`Invocation`, `Workgroup`, or `None`).

`VernonValidation.cpp`, `VernonAutodiffAnalysis.cpp`, and
`VernonLowerAccumulation.cpp` must consume the same proof API.

Strict `scatter_add(disjoint=true)` remains strict unless its lowering also
propagates and enforces the same dispatch contract. Do not infer safety from
mere dependence on a global ID.

### 2. One Dispatch Contract Parser and Validator

Create one runtime representation and one parser for the reflected dispatch
contract. Remove independent parsing from `VernonRuntime.cpp` and
`pipeline_metadata.cpp`.

Create one validator:

```text
validateDispatchContract(contract, grid, workgroup)
```

It must run before allocation, staging, mutation, or backend submission in:

- ordinary CPU and GPU compute invocation;
- ExecutionGraph compute nodes;
- direct, cooked, and CPU AOT pipelines;
- CPU AD forward;
- CPU AD pullback/replay.

The low-level provider/Core API may remain a trusted primitive, but all public
pipeline and AD APIs must pass through the validator.

Remove `hasConstantOrdinaryWrite` and its CPU-only dispatch branch after the
unified contract covers the same case.

### 3. AD Profile Preservation

Forward-with-tape and backward profiles must each reflect their dispatch
contract. CPU AD profile parsing must preserve it in `HostProfileLayout`.

The primal requested grid must satisfy both profile contracts. Forward and
pullback must validate before constructing `HostEffectTransaction`, allocating
tape, or staging gradients.

Missing or malformed dispatch-contract metadata for a writable compute profile
must fail closed. Do not add a compatibility default.

Changing the externally serialized schema is a release-gated contract change.
Implement and test the internal model now, but update compiler/pipeline version
constants only at the coordinated release requested by the project owner.

### 4. Proof-Derived Gradient Ownership

Delete shape-based ownership logic from `VernonStructuredVjp.cpp`, including
conditions based on static shape or element count.

For every external gradient destination, analysis must produce an explicit
ownership result:

- `InvocationPrivate`: all writes are proven invocation-owned under the
  dispatch contract;
- `WorkgroupShared`: the destination is formally workgroup-owned and uses
  workgroup staging;
- `AtomicShared`: updates use a target-supported atomic accumulation;
- `None`: no internal gradient destination is required.

Dynamic and static TensorViews use identical ownership rules. Missing ownership
for a multi-invocation writable gradient is a compiler error; runtime must not
guess or fall back to shared staging.

Reflection must carry the ownership as required backward ABI metadata.

### 5. Scalable TensorView Validation

Keep `tensor_bridge.{h,cpp}` as the only implementation of signed-stride bounds,
writable injectivity, physical identity, and overlap.

Replace production element-address enumeration with layered arithmetic proofs:

1. different allocation/resource identity proves disjointness;
2. non-overlapping physical byte spans prove disjointness;
3. contiguous intervals use interval overlap;
4. regular strided views use stride-lattice/GCD congruence to prove common
   interleaved cases disjoint;
5. any unproven case is `Unknown` and fails closed when either view is writable.

Do not retain an element-count threshold that changes semantic acceptance.
Python must delegate semantic validation to the shared native implementation
and retain only borrow/lifetime management.

### 6. Tape and Gradient Memory Policy

Per-invocation tape is required when control flow or saved values differ by
invocation. Per-invocation private gradient staging is required when direct
publication cannot be proven race-free. These are not compatibility paths.

Make the cost explicit and bounded:

- compute checked dispatch-wide tape and gradient upper bounds;
- reject overflow and policy-limit violations before observable effects;
- report required bytes and configured limit;
- use workgroup sharing only when a formal proof allows it;
- do not add serial execution as a memory fallback.

### 7. Accumulation Lowering

Analyze every `reduce_sum` and `scatter_add` once and cache the selected lowering
decision. Remove duplicate validation/lowering analysis walks.

Allowed lowering decisions are:

- proven disjoint ordinary store;
- target-supported atomic accumulation;
- invocation-private staging;
- dedicated reduction when implemented.

Otherwise fail closed.

## Implementation Order

1. Add the conditional-proof result and internal dispatch-contract model.
2. Reflect, parse, and centrally validate the contract on ordinary compute.
3. Integrate the same validator into CPU AD forward and pullback.
4. Delete `required_unit_grid_axes`, `hasConstantOrdinaryWrite`, and duplicate
   parsers/enforcement.
5. Replace shape-based gradient ownership with proof-derived mandatory metadata.
6. Replace element enumeration with scalable arithmetic TensorView proofs.
7. Add dispatch-wide tape/gradient budget preflight.
8. Cache accumulation lowering decisions.
9. Update `specs/autodiff.md` and `specs/language/contract.md`.
10. Perform the coordinated compiler/pipeline schema release only when
    explicitly authorized.

## Verification Matrix

### Dispatch Contract

- 1D `gid.x` write accepts `(N,1,1)` and rejects `(N,2,1)`;
- 2D `gid.x/gid.y` write accepts `(N,M,1)` and rejects `(N,M,2)`;
- full 3D tuple accepts unrestricted positive grid axes;
- constant store accepts only one total invocation;
- scalar X-only ID cannot prove a multidimensional domain;
- modulo, division, truncation, and unknown affine mappings fail closed;
- leader-guarded workgroup writes validate both guard and grid constraints;
- malformed or missing writable-entry contracts fail load.

Each case must cover direct, cooked, CPU AOT, ExecutionGraph, CPU AD forward,
and CPU AD pullback paths where applicable.

### Ownership

- dynamic-shape invocation-private gradients across multiple workgroups;
- static and dynamic shapes produce the same ownership result;
- non-leader lane contributions are retained;
- missing ownership fails compilation or profile load;
- workgroup-shared gradients publish exactly once per workgroup;
- atomic shared gradients require a declared backend capability.

### TensorView and Memory

- large contiguous disjoint views;
- large interleaved even/odd views;
- negative strides and nonzero offsets;
- writable non-injective views rejected;
- read/read overlap accepted;
- writable overlap rejected for host and RHI resource bindings;
- unknown overlap fails closed without enumerating elements.

### Numerical AD

- multi-invocation `reduce_sum`, `scatter_add`, and atomic-add VJP versus finite
  differences;
- scaled cotangents and multiple cotangent carriers;
- reusable and concurrent pullbacks;
- dynamic loops and branch-dependent tape;
- workgroup reverse barriers, lane failure, cancellation, and rollback;
- direct/cooked/AOT parity.

## Completion Criteria

CPU single-kernel AD may be called complete only when:

- every public AD dispatch enforces the same compiler-derived dispatch contract;
- no ownership decision depends on static shape size;
- no runtime path guesses missing ownership or silently serializes;
- TensorView validation has no element-count-dependent semantic behavior;
- dispatch-wide memory is checked before effects;
- accumulation primitives have runtime numerical coverage;
- full build, CTest, Python, and CPU direct/cooked/AOT parity pass;
- searches find no `required_unit_grid_axes`, `hasConstantOrdinaryWrite`,
  shape-based ownership branch, legacy AD protocol, or serial-dispatch fallback.
