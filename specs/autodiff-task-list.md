# Autodiff implementation task list

## Status

This is the single source of truth for autodiff implementation progress.
The architecture documents explain decisions and constraints; they do not
track completion:

- [`mlir-centered-autodiff-architecture.md`](mlir-centered-autodiff-architecture.md)
  defines the structured-MLIR transform and dynamic-tape model.
- [`scalable-autodiff-rules-architecture.md`](scalable-autodiff-rules-architecture.md)
  defines compiler component boundaries and reusable differentiation rules.
- [`gpu-autodiff-responsibility-architecture.md`](gpu-autodiff-responsibility-architecture.md)
  defines the GPU physical ABI and Runtime ownership boundary.
- [`autodiff-implementation-status.md`](autodiff-implementation-status.md)
  records the current implemented-behavior snapshot.

The public transform remains VJP-only during this migration. JVP and
higher-order AD reuse the shared analysis and rule registry later, but do not
block the current P0 work.

Do not update compiler contract version 10 or pipeline contract version 13
until the next release.

## Checklist rules

- Check an implementation item only when its code and focused tests pass.
- Check a phase gate only when every preceding item in that phase is checked.
- Run formatting tools through `.venv`.
- Use one `*build/` directory at a time and run build/test stages sequentially.
- Keep the existing Python/native path until CPU, Metal, and Vulkan have
  numerical and failure-path parity.
- Never silently fall back from structured AD to legacy native AD.
- Treat all allocation, alignment, offset, counter, and size arithmetic as
  checked, fail-closed operations.

## Phase 0 — Verified baseline

Implemented foundations that the migration must preserve:

- [x] Keep the declarative `ProgramExpression` and `ProgramTransformSpec` VJP
  API and deterministic profile identities.
- [x] Keep canonical `VernonValueAbi` layouts for Scalar, Tensor, Tuple, and
  Struct leaves.
- [x] Keep semantic `vernon.reduce_sum` and `vernon.scatter_add` operations and
  compiler-owned accumulation lowering.
- [x] Keep current CPU, Metal, and Vulkan numerical VJP acceptance.
- [x] Keep typed input, Storage, output, tape, cotangent, and gradient resource
  roles in reflection and Runtime.
- [x] Keep generic Metal and Vulkan physical-resource binding free of
  structured-AD branches.
- [x] Keep compiler reflection free of predicted or synthesized GPU AD
  resources.
- [x] Keep structured-compilation failures explicit; do not substitute a
  primal-only or legacy artifact.
- [x] Preserve the current fixed-tape production path as the migration
  baseline.

Gate:

- [x] The existing Python, compiler, cooked Runtime, CPU, Metal, and Vulkan AD
  tests provide a passing checkpoint.

## Phase 1 — Logical AD IR

Primary files:

- `source/include/mlir/Dialect/Vernon/IR/VernonTypes.td`
- `source/include/mlir/Dialect/Vernon/IR/VernonOps.td`
- `source/lib/Dialect/Vernon/IR/VernonOps.cpp`
- `source/tests/mlir/Autodiff/`

Implementation:

- [x] Add the internal `!vernon.ad_tape` type.
- [x] Add the internal `!vernon.ad_region_header` type.
- [x] Add `vernon.ad.begin_invocation` and define invocation ownership.
- [x] Add `vernon.ad.begin_region` and `vernon.ad.end_region`.
- [x] Add checked `vernon.ad.reserve_record`.
- [x] Add typed/canonical `vernon.ad.write_leaf` and
  `vernon.ad.read_leaf`.
- [x] Add region-header reads for record offset, executed count, exit kind,
  and nested-region handles.
- [x] Add explicit `vernon.ad.capture` and `vernon.ad.commit` operations.
- [x] Verify that visible Storage and output writes are illegal in capture.
- [x] Verify tape/region handle nesting, ownership, and read/write phase
  legality.
- [x] Reject malformed or unchecked record layouts in the verifier.

Acceptance:

- [x] Add parser/printer round-trip tests for every AD type and operation.
- [x] Add negative verifier tests for invalid phase effects, handle lifetimes,
  and region nesting.
- [x] Add tests proving logical AD IR contains no CPU, Metal, Vulkan, binding,
  or status-word details.

Gate:

- [x] The logical AD dialect is independently verifiable and backend-neutral.

## Phase 2 — Stable CPU allocator ABI

Primary files:

- `source/lib/runtime/autodiff/tape_allocator_abi.h`
- `source/lib/runtime/autodiff/host_tape_allocator.{h,cpp}`
- `source/lib/runtime/autodiff/runtime_autodiff_internal.h`
- `source/lib/runtime/autodiff/runtime_autodiff_cpu.cpp`
- `source/lib/Dialect/Vernon/Transforms/VernonVerifyCPUAutodiffABI.cpp`

Implementation:

- [x] Define a hidden packed-argument `ad_tape_allocator` builtin without
  changing the user-visible function signature.
- [x] Define `VernonAdTapeAllocator` with `struct_size` and `abi_version`.
- [x] Specify callback behavior after allocation or arithmetic failure.
- [x] Distinguish capacity exhaustion, arithmetic overflow, and host allocation
  failure.
- [x] Require exact `required_bytes` accounting after capacity exhaustion.
- [x] Specify region-handle lifetime and read legality.
- [x] Specify reset/repeated-capture behavior.
- [x] Specify thread and invocation ownership.
- [x] Specify successful tape ownership transfer to the pullback.
- [x] Keep `VernonCpuInvocation.textures` unchanged and do not repurpose it.
- [x] Verify the hidden builtin in a dedicated CPU AD ABI pass without exposing
  the internal descriptor through the public Runtime headers.

Acceptance:

- [x] Add ABI size/version compatibility tests.
- [x] Add callback failure-state and exact-size accounting tests.
- [x] Add region-handle lifetime and repeated-capture tests.

Gate:

- [x] The allocator boundary is documented, versioned, checked, and frozen
  before generated code depends on it.

## Phase 3 — Shared AD analysis

Primary files:

- new `VernonAutodiffAnalysis.{h,cpp}`
- existing frontend structured MLIR from `control_flow_lowering.py` and
  `loop_lowering.py`

Implementation:

- [x] Classify differentiable Scalar and canonical aggregate leaf types.
- [x] Resolve canonical `wrt` and active result leaves.
- [x] Propagate activity through SSA use-def chains.
- [x] Discover `scf.if`, `scf.while`, and nested structured regions.
- [x] Classify pure, Storage, atomic, barrier, and externally visible effects.
- [x] Reuse `VernonValueAbi` for aggregate leaf projection.
- [x] Emit deterministic diagnostics for unsupported active operations and
  effects.
- [x] Keep analysis independent of JVP/VJP mode and CPU/GPU target.

Acceptance:

- [x] Add unit tests for inactive pruning and active result/`wrt` projection.
- [x] Add tests for nested region discovery and effect classification.
- [x] Add aggregate leaf activity tests.

Gate:

- [x] VJP and future JVP can consume one immutable shared analysis result.

## Phase 4 — Differentiation rule registry

Primary files:

- new `VernonAutodiffRules.{h,cpp}`
- `python/vernon_dsl/frontend/autodiff.py`
- `python/vernon_dsl/frontend/autodiff_native_math.py`

Implementation:

- [x] Define one rule abstraction for activity classification, VJP primal
  requirements, VJP construction, and future JVP construction.
- [x] Use a pass-local registry keyed by MLIR operation name or `TypeID`.
- [x] Register `arith.addf`, `arith.subf`, `arith.mulf`, `arith.divf`, and
  negation rules.
- [x] Register currently supported scalar `math` rules without changing
  numerical behavior.
- [x] Declare exact required primal operands/results in each VJP rule.
- [x] Keep rules free of tape allocation, backend resources, and CPU/GPU
  branches.
- [x] Diagnose active operations with missing or incompatible rules.
- [x] Add parity coverage against the current frontend derivative registry.
- [x] Leave a migration path to Vernon differentiation `OpInterface` external
  models without modifying upstream dialect definitions.

Acceptance:

- [x] Add isolated rule tests for primal requirements and emitted local
  derivatives.
- [x] Compare scalar rule results with finite differences and the CPU reference
  oracle.
- [x] Prove that adding an operation requires one rule registration rather
  than edits to multiple transforms.

Gate:

- [x] Existing scalar VJP rules are represented once in the shared registry.

## Phase 5 — Rule-driven tape planning

Primary files:

- new `VernonAutodiffTapePlanning.{h,cpp}`
- `source/include/mlir/Dialect/Vernon/IR/VernonValueAbi.h`

Implementation:

- [x] Collect only rule-required active primal values.
- [x] Deduplicate saved SSA values.
- [x] Decompose aggregate values through canonical ABI leaves.
- [x] Separate fixed invocation records from dynamic-region records.
- [x] Assign checked leaf offsets, record stride, and alignment.
- [x] Define invocation and region header schemas.
- [x] Assign parent-record identity and child-region ordinals.
- [x] Represent predicates, actual iteration counts, and exit kinds needed by
  reverse control flow.
- [x] Keep save-versus-recompute policy separate from rule semantics.
- [x] Ensure `tape_bytes` is a static layout/statistics hint, never an
  iteration or allocation cap.

Acceptance:

- [x] Add tests showing `addf` saves no primal values.
- [x] Add tests showing multiplication/division save only declared values.
- [x] Add deduplication and mixed-alignment aggregate tests.
- [x] Add checked record-size, offset, and alignment overflow tests.

Gate:

- [x] VJP tape contents are derived from rule requirements rather than
  save-every-scalar behavior.

## Phase 6 — Structured scalar VJP

Primary files:

- new `VernonStructuredVjp.{h,cpp}`
- compiler pipeline registration under
  `source/lib/Dialect/Vernon/Transforms/`
- `python/vernon_dsl/_shader_assets/cooking.py`
- `python/vernon_dsl/_runtime/autodiff.py`
- `python/vernon_dsl/frontend/autodiff_native_cpu.py`
- direct CPU autodiff loading under `source/lib/runtime/autodiff/`

Implementation:

- [x] Transform the normal structured primal MLIR instead of rebuilding an AST
  graph.
- [x] Generate an augmented forward function.
- [x] Generate a reverse function consuming output cotangents.
- [x] Use the shared analysis, rule registry, and tape plan.
- [x] Accumulate adjoints in promoted gradient types.
- [x] Keep scalar reverse accumulation in ordinary differentiable MLIR;
  `vernon.reduce_sum` and `vernon.scatter_add` remain tensor/effect boundaries
  for the later structured tensor and effect phases.
- [x] Keep generated derivative arithmetic in ordinary differentiable MLIR
  until semantic reverse construction is complete.
- [x] Emit the existing `primal`, `forward_with_tape`, and `backward` profile
  names and metadata without a contract version change.
- [x] Route eligible CPU scalar cooks through structured compilation by
  default, with explicit capability dispatch for paths not yet migrated.
- [x] Do not catch structured transform failure to invoke legacy emission.
- [x] Reuse one structured profile builder for cooked and direct execution.
- [x] Compile and load direct Python CPU VJP profiles in memory without cooking
  or loading a pipeline manifest.
- [x] Retain compiled profile ownership and invalidate loaded profiles across
  source dependency changes and Runtime recreation.
- [x] Derive cotangent and gradient ABI metadata from promoted derivative types.
- [x] Keep legacy CPU VJP cooking tractable during migration by merging
  adjoints before reverse-topological propagation, so shared semantic DAG nodes
  are emitted once instead of being recursively expanded per use path.

Acceptance:

- [x] Add straight-line scalar IR tests for each migrated rule.
- [x] Add forward/tape/backward symbol and signature tests.
- [x] Compare structured CPU VJP numerics with the existing CPU reference.
- [x] Keep all legacy VJP tests green.
- [x] Execute structured VJP directly from Python without a cooked manifest.
- [x] Cover all migrated scalar rules, reusable pullbacks, multi-invocation
  accumulation, and promoted `f16` gradients through the direct Runtime path.
- [x] Numerically validate the C++ rule builders for dot, cross, matmul,
  normalize, reflect, construct, splat, and broadcast at the compiler-rule
  layer, including the dot-plus-sqrt norm decomposition and promoted tensor
  gradients. Production tensor cooking remains on the legacy emitter until the
  structured tensor Runtime phases.
- [x] Add a shared-subgraph regression for legacy backward emission and reduce
  the complex numeric fixture from 38.5 MB / 679,070 lines of backward MLIR to
  312 KB / 5,473 lines while preserving cooked Runtime numerics, with an
  automated upper-bound regression on both bytes and lines.

Gate:

- [x] Structured scalar VJP cooks by default and executes on CPU with current
  numerical behavior.
- [x] Structured scalar VJP compiles and executes through the direct Python CPU
  Runtime path.

## Phase 7 — Structured control-flow VJP

Implementation:

- [x] Transform `scf.if` by recording/reusing the executed predicate and
  reversing only the selected branch.
- [x] Transform data-dependent `scf.while` without static unrolling.
- [x] Record one checked dynamic record per actually executed iteration.
- [x] Traverse loop records in descending execution order.
- [x] Support zero-, one-, and many-iteration loops.
- [x] Support nested `continue`.
- [x] Support `break` and distinguish normal loop completion.
- [x] Support arbitrary in-loop `return` and multiple return exits.
- [x] Support both normal and break paths of `for ... else`.
- [x] Store nested-region handles in parent iteration records.
- [x] Remove dependence on the 1024-iteration literal-range limit.
- [x] Remove dependence on the 256-iteration dynamic-leading-break limit.
- [x] Do not add another CFG or dynamic-region model to Python
  `AutodiffProgram`.

Acceptance:

- [x] Add IR tests for `if`, data-dependent `while`, `continue`, `break`,
  return, and `for ... else`.
- [x] Add nested-region and nested-aggregate record tests.
- [x] Test iteration counts beyond the old 1024/256 limits.
- [x] Test checked iteration counter and record offset overflow.
- [x] Compare compiler-rule arithmetic with the CPU reference oracle where no
  dynamic Runtime tape is required; end-to-end control-flow numerics are Phase
  8 acceptance work.

Gate:

- [x] Reverse structured control flow is runtime-bounded and has no
  compile-time trip-count cap.

## Phase 8 — CPU dynamic tape and effect transaction

Primary files:

- new CPU AD lowering pass
- `source/lib/runtime/autodiff/runtime_autodiff_cpu.cpp`
- `source/lib/runtime/autodiff/runtime_autodiff_internal.h`

Implementation:

- [ ] Implement Runtime-owned `HostDynamicTape` using checked
  `std::vector<std::byte>` growth.
- [ ] Lower logical tape operations through the hidden allocator descriptor.
- [ ] Grow tape during one CPU forward execution without replaying the primal.
- [ ] Read dynamic region headers and records during backward execution.
- [ ] Replace fixed forward-result-leaf to backward-argument tape copying for
  structured profiles.
- [ ] Defer or functionalize Storage and observable output effects during
  capture.
- [ ] Commit visible effects exactly once after successful capture.
- [ ] Transfer successful tape ownership to the pullback.
- [ ] Keep the tape immutable and reusable across pullback applications.
- [ ] Release tape ownership when the pullback is destroyed.

Acceptance:

- [ ] Test zero, one, long, and nested dynamic loops end to end on CPU.
- [ ] Compare every structured control-flow path, including break, return, and
  loop-else exits, with the CPU reference oracle.
- [ ] Test allocator rejection and process memory-policy rejection.
- [ ] Test no visible Storage/output change after failed capture.
- [ ] Test successful execution commits once.
- [ ] Test repeated pullback application against one stable tape snapshot.

Gate:

- [ ] Structured CPU VJP has dynamic control-flow and failure-path parity with
  the architecture contract.

## Phase 9 — Aggregate and Storage parity

Implementation:

- [ ] Move Scalar, Tensor, Tuple, and Struct input projection to the structured
  path.
- [ ] Move aggregate output cotangent projection to the structured path.
- [ ] Move aggregate tape records to canonical ABI leaf layout.
- [ ] Move per-leaf f16/f32/f64 gradient promotion to the structured path.
- [ ] Move static and dynamic Tensor indexing VJPs.
- [ ] Move Tensor scalar broadcasting and reduction-to-source-shape VJPs.
- [ ] Move functionalized TensorView load/store VJPs.
- [ ] Preserve alias rejection for overlapping writable Storage.
- [ ] Preserve disjoint-scatter evidence and target-owned accumulation choice.

Acceptance:

- [ ] Add native and cooked CPU numerical tests for nested Tuple/Struct values.
- [ ] Add dynamic indexing, broadcasting, and Storage mutation tests.
- [ ] Verify reflection paths, shapes, offsets, and promoted dtypes leaf by
  leaf.

Gate:

- [ ] Structured CPU VJP covers the current production CPU capability set.

## Phase 10 — GPU physical-entry ABI

Primary files:

- new `VernonLowerGPUAutodiff.{h,cpp}`
- compiler target-preparation and reflection flow
- `source/lib/compiler/compiler_reflection.cpp`
- `source/lib/Dialect/Vernon/Transforms/VernonToGPU.cpp`

Implementation:

- [ ] Make target preparation produce one target-prepared module consumed by
  both reflection and code generation.
- [ ] Make `VernonLowerGPUAutodiff` the sole owner of GPU AD physical-boundary
  materialization.
- [ ] Convert logical tape handles to ordinary status and dynamic-tape
  `TensorView` parameters.
- [ ] Materialize cotangent, output, and gradient physical resources when
  required by the GPU entry ABI.
- [ ] Assign the actual set/binding values consumed by code generation.
- [ ] Attach typed `vernon.autodiff_role` metadata to physical parameters.
- [ ] Attach `vernon.autodiff_protocol = capture_status | dynamic_tape`.
- [ ] Reflect only arguments that exist in the prepared physical signature.
- [ ] Keep `VernonToGPU.cpp` generic and free of AD conditionals.
- [ ] Keep Metal, Vulkan, CUDA, and generic backend binding free of AD
  conditionals.

Acceptance:

- [ ] Prove logical AD IR contains no backend resources.
- [ ] Prove physical GPU entries contain no `!vernon.ad_tape`.
- [ ] Prove every physical resource has a unique actual binding.
- [ ] Prove reflection matches physical arguments, roles, access, shapes, and
  bindings one-to-one.
- [ ] Prove reflection contains no synthesized arguments.

Gate:

- [ ] Reflection and code generation consume the same physical-entry ABI.

## Phase 11 — GPU tape wire protocol

Primary files:

- new internal compiler/Runtime shared protocol header
- GPU AD physical lowering

Implementation:

- [ ] Define capture and commit phase values.
- [ ] Define status header word indices and protocol error codes.
- [ ] Define checked 32-bit byte-offset and capacity limits.
- [ ] Define one invocation-root entry per logical grid invocation.
- [ ] Define region-header layout with last-record offset, executed count, and
  exit kind.
- [ ] Define record-prefix layout with previous-record offset and statically
  planned child handles.
- [ ] Keep canonical payload layout in `VernonValueAbi`, outside the protocol
  header.
- [ ] Use one protocol-defined allocation alignment.
- [ ] Implement checked atomic record reservation.
- [ ] Count every attempted reservation toward exact required bytes after
  capacity exhaustion.
- [ ] Suppress all out-of-capacity payload writes.
- [ ] Reverse dynamic records through links rather than physical contiguity.
- [ ] Lower capture and commit phase guards.

Acceptance:

- [ ] Add protocol IR tests for single and multiple invocations.
- [ ] Add linked-record tests for parallel interleaving.
- [ ] Add nested-region, continue, return, and `for ... else` protocol tests.
- [ ] Add representability, counter, offset, and malformed-header failures.
- [ ] Verify exact required-size reporting after overflow.

Gate:

- [ ] The compiler and Runtime share one backend-independent, fail-closed GPU
  tape protocol.

## Phase 12 — Typed GPU tape session

Primary files:

- `source/lib/runtime/pipeline_manifest.h`
- `source/lib/runtime/runtime_pipeline_direct.cpp`
- new `GpuAutodiffTapeSession.{h,cpp}`
- `source/lib/runtime/autodiff/runtime_autodiff_gpu.cpp`
- `source/lib/runtime/autodiff/runtime_autodiff_graph.cpp`

Implementation:

- [ ] Parse serialized protocol metadata once into a typed enum.
- [ ] Remove downstream protocol decisions based on `__vernon_*` resource
  names.
- [ ] Allocate provisional status and tape resources.
- [ ] Build and submit provisional capture.
- [ ] Read back and validate compact status.
- [ ] Check required capacity against representation and memory policy.
- [ ] Allocate exact tape capacity when provisional capacity is insufficient.
- [ ] Permit at most one exact capture retry.
- [ ] Reject a second overflow or malformed exact-capture status.
- [ ] Forbid commit unless the latest capture is fully valid.
- [ ] Submit commit exactly once.
- [ ] Transfer successful tape ownership to the pullback.
- [ ] Make cleanup and error reporting deterministic.
- [ ] Keep `runtime_autodiff_graph.cpp` responsible only for topology, value
  wiring, cotangent/gradient wiring, and pullback lifetime.

Acceptance:

- [ ] Test provisional success without retry.
- [ ] Test forced exact allocation and successful retry.
- [ ] Test exact-retry overflow, malformed status, offset overflow, and memory
  policy rejection.
- [ ] Test no commit after any failed capture.
- [ ] Test commit-once and stable pullback tape ownership.

Gate:

- [ ] A backend-independent Runtime session owns the complete GPU
  capture/retry/commit state machine.

## Phase 13 — Metal structured VJP

Implementation:

- [ ] Enable the structured compiler capability for Metal.
- [ ] Use normal reflected resource binding without Metal AD branches.
- [ ] Execute scalar structured VJP numerically.
- [ ] Execute canonical aggregate structured VJP numerically.
- [ ] Exercise forced retry and no-commit-on-failure behavior.
- [ ] Keep all existing cooked Metal pullback tests green.

Gate:

- [ ] Metal has structured numerical and failure-path parity with CPU for the
  migrated capability set.

## Phase 14 — Vulkan and multi-node sessions

Implementation:

- [ ] Enable Vulkan with the same physical ABI, protocol, and Runtime tape
  session.
- [ ] Keep backend-specific changes limited to fixtures and capability
  acceptance.
- [ ] Execute scalar and canonical aggregate structured VJP numerically.
- [ ] Exercise forced retry and no-commit-on-failure behavior.
- [ ] Give each forward graph node an independent tape session.
- [ ] Commit each node before exposing its outputs to downstream nodes.
- [ ] Preserve reverse-topological cotangent routing and reusable pullbacks.
- [ ] Do not merge per-node protocol buffers without measured justification.

Gate:

- [ ] Vulkan matches Metal/CPU, and multi-stage forward graphs use independent
  checked sessions.

## Phase 15 — Legacy removal

Prerequisite: CPU, Metal, and Vulkan numerical and failure-path parity.

Implementation:

- [ ] Route production cooking through structured MLIR AD.
- [ ] Reduce Python `AutodiffProgram` to transform metadata, canonical profile
  identity, and launch/accumulation policy.
- [ ] Keep the Python CPU implementation only as a test oracle.
- [ ] Remove Python native forward/backward graph evaluators.
- [ ] Remove duplicated CPU/GPU derivative rule bodies.
- [ ] Remove `_GpuForwardEmitter` as the owner of effect transactions.
- [ ] Remove flat-graph control-flow reconstruction.
- [ ] Remove static loop unrolling and its 1024/256 limits.
- [ ] Remove migration-only opt-ins and legacy profile capability switches.
- [ ] Remove tests that assert unsupported legacy limits, while retaining
  numerical regression coverage.

Gate:

- [ ] One structured MLIR program, one rule registry, and one tape/effect
  protocol define production VJP semantics.

## Phase 16 — Post-P0 work

These tasks are intentionally non-blocking for the current VJP migration:

- [ ] Add minimal JVP for `addf`, `subf`, `mulf`, and negation using the shared
  analysis and rule registry.
- [ ] Add structured `scf.if` and `scf.while` JVP.
- [ ] Verify JVP emits no tape or capture/commit operations.
- [ ] Add JVP profiles only with an intentional contract release.
- [ ] Add JVP-of-VJP and VJP-of-JVP foundations without differentiating
  allocator or backend tape operations.
- [ ] Replace serial conflicting accumulation with scalable contribution
  buffers, deterministic sorting, and segmented reduction.
- [ ] Add an RHI device buffer-fill operation for GPU gradient initialization.
- [ ] Add ExecutionGraph Storage mutation edges and multiple sink outputs.
- [ ] Expose ExecutionGraph AD composition through public C and Python APIs.
- [ ] Support keyword arguments and recursive helpers in differentiated code.
- [ ] Add graphics-stage backward lowering and custom differentiable rendering
  rules.
- [ ] Add persistent `accumulate_into` gradient-buffer semantics.
- [ ] Resolve forward and backward backend pipelines lazily.
- [ ] Complete CUDA, DirectX, OpenGL, and OpenGL ES hardware acceptance.
