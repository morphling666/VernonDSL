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

### First-order VJP autodiff

The normative design is [`../autodiff.md`](../autodiff.md).

- Add one public reverse-mode transform, `vd.ad.vjp`, for a Kernel, graphics
  stage tuple, or existing ExecutionGraph. Keep authored entry signatures
  unchanged.
- Extend `program_asset(program=...)` with a declarative VJP
  `ProgramExpression`; cook independent primal, forward-with-tape, and
  backward profiles under one asset ID.
- Derive adjoint Values recursively for floating Scalar, Tensor, Tuple, and
  Struct leaves. Return new owned Storage for differentiated TensorView or
  mutable Storage inputs.
- Use `wrt` as the only differentiated-input declaration. Do not introduce
  `grad_or_not`, `requires_grad`, implicit `.grad`, global gradient clearing,
  or hidden backward execution.
- Accept Tensor/aggregate outputs through explicit cotangents. Permit omitted
  cotangent only for one floating Scalar output, where the seed is `1`.
- Define versioned derivative rules for arithmetic, casts, Tensor
  construction, `matmul`, and supported math intrinsics.
- Build compiler-internal ProgramGraph Value/control/effect dependencies,
  functionalize legal Storage mutation, prove alias/scatter behavior, and
  reject unbounded tape or unsupported accumulation.
- Define required custom VJPs for rasterization, visibility, depth, blend, and
  texture sampling before accepting cross-stage graphics differentiation.
- Add deterministic transform, rule-set, tape, cotangent, gradient, reflection,
  and manifest identity across Python, C, and C++.
- Compare analytical derivatives with finite differences on CPU and every
  backend that advertises the corresponding AD capability.

JVP, full-Jacobian materialization, convenience `grad` aliases, nested
transforms, Hessians, and HVPs are outside the initial public surface.
Whether all VJP items block the v4 declaration must be decided before release;
unchecked behavior cannot be implied by the version number.

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

## Post-v4 autodiff expansion

- Forward-mode JVP and batched JVP/VJP.
- Explicit full-Jacobian materialization for statically bounded small Values.
- Convenience aliases such as `grad` only when they are exact sugar over VJP.
- Reusable explicit gradient-accumulation buffers.
- Higher-order transforms, Hessians, HVPs, checkpoint optimization, and
  measured recomputation policies.

Do not claim a derivative for a graphics discontinuity without a versioned
custom rule and finite-difference evidence for its declared domain.

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
