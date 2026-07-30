# Language v4 roadmap

## Status

The released Python frontend remains language version 3. There is no numeric
`FRONTEND_VERSION` constant. Version 4 is the target defined by
[`contract.md`](contract.md) and will be declared only after every required gate
below passes.

Implemented foundations include:

- Value, Storage, and Resource semantic categories;
- Tensor, Tuple, Struct, TensorStorage, TensorView, Texture, and Sampler;
- call-site specialization of `@func` helpers;
- structured effects and expression/statement control flow;
- dynamic range bounds, early return, break, and continue;
- workgroup storage, relaxed i32/u32 atomics, and barriers in source and IR;
- deterministic ABI, reflection, and target-specific lowering.

These foundations are current behavior, not proof that all v4 runtime and
correctness guarantees are complete.

## Required v4 gates

### First-order pure-function autodiff

- Add typed APIs for `jvp`, `vjp`, `grad`, `value_and_grad`, and
  `stop_gradient`.
- Transform only specialized, validated, pure, non-recursive `@func` IR.
- Derive tangent and adjoint Values recursively for floating Scalar, Tensor,
  Tuple, and Struct leaves.
- Reject integer, Boolean, Storage, Resource, sampler, and opaque
  differentiation unless an explicit custom rule exists.
- Define versioned derivative rules for arithmetic, casts, Tensor
  construction, `matmul`, and supported math intrinsics.
- Allocate gradient Storage separately from primal element types.
- Compare analytical derivatives with finite differences on CPU and available
  CUDA/Vulkan runtimes.
- Reject nested transforms, Hessians, and HVPs explicitly.

Whether all autodiff items block the v4 declaration must be decided before
release; unchecked behavior cannot be implied by the version number.

### Workgroup memory and synchronization

- Execute aggregate workgroup storage on every advertised GPU runtime where
  hardware is available.
- Cover nested Tensor/Struct leaves, non-zero indices, padding, branches, and
  loops.
- Prove independent allocation between workgroups.
- Prove barrier-visible writes and relaxed atomic semantics end to end.
- Validate backend limits, ordering, scope, races, and unsupported element
  types before artifact or Runtime mutation.

Unavailable devices remain explicit skips, not silent passes.

### TensorView runtime layout

- Decide whether v4 requires arbitrary runtime-strided multi-rank TensorViews.
- If required, add one reflected hidden-layout ABI and implement it across
  supported backends.
- Otherwise retain the current AOT-specialization rule and make the limitation
  explicit in diagnostics and the language contract.
- Preserve bounds, injectivity, overlap, alias, access, and owner-lifetime
  validation.

### Cross-backend acceptance

- Compare representative CPU, CUDA, Vulkan, OpenGL, and DirectX programs where
  each backend advertises support.
- Keep target capability failures deterministic and target-independent where
  possible.
- Include compiler-contract and pipeline versions in cache identity.
- Preserve deterministic MLIR, artifacts, reflection, and symbols.

## Post-v4 compiler work

### ProgramGraph for autodiff

`ProgramGraph` is private compiler IR for one specialized program. It is not a
host graph of PipelineAssets, dispatches, render passes, or backend
transitions.

Future work may:

- represent value flow, control flow, Storage effects, alias constraints, and
  differentiability boundaries;
- support JVP/VJP transformation, mutation functionalization, tape planning,
  checkpointing, and reverse traversal;
- lower transformed graphs through existing target pipelines.

`VernonExecutionGraph` remains the separate public host-orchestration API.

### Stateful kernel autodiff

Stateful differentiation is post-v4 unless the language contract changes. It
requires:

- functionalization of local mutation and TensorView writes;
- gather/scatter adjoints and deterministic accumulation;
- explicit tape bounds, checkpointing, and recomputation;
- effect-preserving reverse traversal;
- separate texture-sampling rules;
- explicit treatment of rasterization, visibility, depth, and blending.

Do not claim general differentiable rendering without finite-difference
evidence and selected custom primitives.

## Optional language expansion

The following features require independent proposals and versioned acceptance:

- constrained and const generics;
- type aliases and compile-time assertions;
- enums and deterministic underlying representations;
- tagged unions, `Option[T]`, and exhaustive `match`;
- deterministic constant evaluation and compile-time data structures;
- fixed-capacity containers built over TensorStorage.

Sparse and hash structures remain library work over explicit dense Storage and
atomic policies; they do not introduce another semantic category.

## Acceptance policy

For every language addition:

- update the normative contract before implementation;
- represent the feature in typed semantic IR before lowering;
- add CPU numeric and diagnostic tests;
- compare available GPU runtimes where support is advertised;
- fail unsupported target combinations explicitly;
- preserve deterministic cache, reflection, and artifact identity;
- document non-differentiable behavior, memory bounds, and effects.

Git history records completed phases. This roadmap contains only remaining
acceptance work and concise context needed to understand it.
