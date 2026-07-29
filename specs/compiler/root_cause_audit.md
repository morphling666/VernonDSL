# Compiler root-cause audit

Date: 2026-07-29
Status: repaired, with explicitly recorded follow-up work

## Scope and review standard

This audit covers TensorView projection, canonical Value ABI planning,
aggregate workgroup storage, schema-v5 parsing, runtime binding validation,
and the associated compiler/runtime tests.

The final review used the following rules:

- production behavior must follow one documented contract, not a condition
  introduced only to satisfy a fixture;
- obsolete schema fields and frontend-authored canonical layout values are
  rejected rather than silently accepted;
- dead helpers, unused backend branches, duplicate projection entry points,
  and superseded ABI implementations are removed;
- a passing test is not considered evidence when it exercises only
  compilation or only an identity layout;
- intentional current limitations are recorded below rather than hidden by
  compatibility fallbacks.

No production branch keyed on a test name, fixture value, backend test target,
or magic test shape remains in the reviewed changes.

## Resolution summary

### 1. TensorView layout now has one enforceable specialization contract

Status: resolved for the supported ahead-of-time contract.

`vernon.tensor_strides` and `vernon.tensor_offset` are the only projection
metadata. They must appear together and match the TensorView rank.
`vernon.tensor_shape` is optional and, when present, is a compile-time shape
specialization that must agree with every static type extent.

The common projection pass materializes the logical index before backend
lowering. Runtime compute and graphics planners both compare the supplied
TensorView byte offset and byte strides against reflected specialization
metadata. A layout mismatch is rejected before binding.

An unspecialized rank-one view may use the canonical identity projection
without a shape specialization. This is not a general runtime-stride ABI:
non-unit stride and non-zero offset are rejected by runtime specialization
matching. Unspecialized dynamic rank greater than one remains unsupported and
is listed under remaining limitations.

Coverage includes incompatible runtime strides, offsets, rank/shape
specialization, and compute/graphics use sites.

### 2. Aggregate workgroup storage uses one physical plan

Status: resolved.

`getWorkgroupPhysicalStoragePlan` computes compact SoA leaf counts and the
exact physical byte total. Verification and CPU/GPU lowering consume that same
plan. Aggregate leaves no longer allocate a complete padded record stride per
leaf.

Validation sums all workgroup allocations in an entry and rejects totals above
the portable 16 KiB limit. Tests cover both a single expanded aggregate and
multiple individually legal allocations whose combined physical total is too
large.

### 3. Nested aggregate workgroup storage is lowered recursively

Status: resolved.

GPU aggregate workgroup allocations are discovered through a recursive kernel
walk. Every load/store user is rewritten through the same expansion, including
operations nested under structured control flow. Unsupported users cause a
diagnostic instead of leaving partially lowered Vernon operations.

### 4. Source IR can no longer forge `physical_index`

Status: mitigated; representation cleanup remains.

Native validation rejects `physical_index` on source load/store/atomic
operations. Only the common projection stage may create it, and backends no
longer perform independent projection fallback.

The marker still exists as an internal pass-state attribute because logical
storage operations retain their original rank after projection. Replacing it
with dedicated single-index internal operations is remaining architectural
work.

### 5. Native Value ABI planning is authoritative

Status: resolved.

Module validation unconditionally derives a finite canonical layout for every
ABI-bearing argument, result, TensorView element, and declared struct.
Reflection derives struct size, alignment, field offsets, leaves, and layout
hashes from the native planner.

Frontend-authored struct `abi_size`, `abi_alignment`, `abi_field_offsets`, and
`abi_element_stride` are obsolete and now rejected. Only semantic metadata
that native signless MLIR cannot recover, such as logical signedness in leaf
dtypes, remains frontend-authored.

The superseded transform-local `VernonValueAbi.h/.cpp` implementation and its
dead wrapper helper were removed. The canonical implementation lives with the
Vernon IR and is shared by validation, reflection, and lowering.

### 6. Projection has one schema and one stage

Status: resolved.

`projection_strides` and `projection_offset` were deleted. CPU and GPU
lowerings no longer invoke projection independently. The explicit common
materialization stage runs after validation and before backend-specific
storage lowering.

The standalone projection pass remains registered because it is a real
compiler stage used by the CPU lowering integration surface; it is not a
test-only alternate implementation.

### 7. Static TensorView extents are checked against specialization

Status: resolved.

Every non-dynamic extent in the TensorView type must equal the corresponding
`vernon.tensor_shape` value. Rank, non-negative shape, paired stride/offset,
and non-negative offset checks are performed in native validation.

### 8. IR, Value ABI, and manifest versions are enforced

Status: resolved.

Native module validation requires the current frontend and Value ABI versions.
Missing and unsupported versions are rejected.

The schema-v5 runtime parser now:

- rejects unknown keys and all retired compiler-generated markers;
- requires canonical variant tables and canonical parameter records;
- validates integer ranges instead of coercing malformed values to zero;
- validates logical type, parameter kind, address space, texture dimension,
  and texture format coherence;
- accepts an omitted `internal_parameters` table only as the canonical empty
  representation emitted by `PipelineVariant.to_json`;
- does not accept a legacy field spelling or legacy internal-parameter form.

### 9. Runtime TensorView specialization is shared

Status: resolved.

`tensorMatchesSpecialization` is the single runtime helper for offset, stride,
rank, optional shape, element-size, and overflow checks. Compute and graphics
invocation planners call the same helper.

### 10. Aggregate construction and storage indexing are shared

Status: substantially resolved.

CPU and GPU storage lowering use shared aggregate construction,
decomposition, compact workgroup indexing, and physical leaf helpers. The
review removed an unused public byte-count wrapper, an exposed helper that had
no external caller, obsolete backend naming, and unreachable configurable
branches in the CPU-only aggregate conversion pattern.

Logical ABI planning is still independently implemented across the native and
Python language boundary; that remaining duplication is listed below.

### 11. Optimization-dependent runtime assertions were removed

Status: resolved for runtime/session and user-input boundaries.

Runtime handle, session state, dispatch argument, and user-derived checks use
explicit exceptions and remain active under `python -O`. Optimization-mode
regression coverage verifies those production checks.

Compiler-internal assertions that follow prior type/inference validation
remain as local invariants. They are not used to validate runtime input or
native resource state.

### 12. Roadmap and language contract were synchronized

Status: resolved.

TensorView, synchronization, address-space, frontend-version, compiler design,
and completion-roadmap documents now describe the implemented contract. They
do not claim end-to-end runtime coverage where only compilation is currently
tested.

## Final cleanup review

The following suspicious categories were reviewed explicitly:

- No test-specific production branches were found. Test fixture changes add
  required contract metadata or remove obsolete metadata; they do not weaken
  production validation.
- The old projection schema, backend projection fallbacks, duplicate runtime
  specialization implementation, dead aggregate atomic branch, unused
  workgroup byte wrapper, and superseded Value ABI files were removed.
- Schema-v5 parsing does not migrate old keys or infer missing required
  parameter/output fields. Unknown and retired records fail closed.
- CPU `compute.json` schema 3 remains intentionally separate from pipeline
  bundle schema 5. It is a native artifact manifest, not a compatibility
  parser for schema 5.
- C API `struct_size` checks and graphics pipeline compatibility records are
  current ABI-extension and cache-key mechanisms. They are not legacy
  compiler compatibility paths.
- OpenGL's `compatibility` profile string is a current graphics API profile,
  not a Vernon schema compatibility mode.

## Remaining issues

### A. Replace pass-state `physical_index` with internal physical operations

Priority: high.

Introduce internal physical load/store/atomic operations with exactly one
index. The projection pass should replace logical storage operations with
those operations, and all backend lowerings should consume them. Then remove
`physical_index` from public ODS definitions and delete rank-preserving zero
index padding.

This is the only remaining forgeable-looking representation. Source
validation currently blocks forgery, so it is not an accepted source-IR bypass.

### B. Replace ToGPU's bridge and manual clone driver

Priority: high.

`VernonToGPU.cpp` still maps one logical TensorView to multiple physical kernel
arguments with `UnrealizedConversionCastOp`, manually clones the entry body,
walks the cloned kernel to rewrite storage operations, and erases the bridge.

Move this to MLIR one-to-many type conversion with recursive conversion
patterns. An unrealized cast at the final backend boundary should be a hard
failure, not an expected temporary bridge.

### C. Define a general runtime layout ABI for dynamic multi-rank TensorViews

Priority: medium.

The implemented AOT contract specializes strides and offset in compiled
variants. It does not pass runtime shape/stride/offset values to shaders.
Consequently:

- dynamic rank-one identity layout is supported and non-identity runtime
  layouts are rejected;
- fully specialized higher-rank layouts are supported;
- an unspecialized dynamic higher-rank TensorView is not supported;
- selecting one compiled variant across arbitrary runtime subviews is not
  supported.

If general runtime-strided views are required, add hidden layout arguments and
reflect their ABI. Do not add another metadata fallback.

### D. Remove cross-language Value ABI algorithm duplication

Priority: medium.

Canonical layout logic still exists in native planning,
`python/vernon_dsl/frontend/abi.py`, and host-value packing. Native planning is
authoritative for accepted IR and reflected metadata, but Python must still
predict the same layout before compilation.

Create versioned golden vectors generated from one declarative ABI
specification, or expose the native planner to frontend/host code. Keep
cross-language tests, but avoid maintaining independent layout rules manually.

### E. Add executable aggregate workgroup coverage

Priority: medium.

Compiler tests cover compact planning, total limits, every backend lowering,
and structured control flow, but aggregate workgroup values are not executed
on every runtime backend.

Add runtime tests that store/load non-zero aggregate indices, cover nested
Tensor/Struct leaves and padding, and execute conditional/loop accesses.
Unavailable devices should be explicit skips.

### F. Finish interface ABI metadata minimization

Priority: low.

Struct declarations no longer carry frontend-computed canonical sizes and
offsets. Entry argument/result dictionaries still carry complete
`vernon.abi_*` metadata for boundary validation and reflection. Most values
can be derived natively; logical leaf dtype is the principal semantic
exception.

Refactor reflection to consume native plans plus semantic leaf dtypes, then
remove any remaining derivable interface size/alignment/offset/count/path
fields in a deliberate IR version bump.

## Verification

The final Release build passes 130/130 native tests and 328/328 Python tests.
The four production-boundary checks also pass under `python -O`.
`git diff --check` and linter inspection report no source errors. Environment
or toolchain warnings from generated MLIR headers are not treated as source
regressions.
