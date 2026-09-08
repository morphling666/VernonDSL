# Program acceptance

Status: current regression requirements for Kernel, graphics pipeline, Module,
Program execution, and Program VJP.

The semantic model is defined in
[`../program/architecture.md`](../program/architecture.md). This document owns
only permanent regression requirements. Cross-backend test infrastructure work
is tracked in
[`cross_backend_language_testing_plan.md`](cross_backend_language_testing_plan.md).

## 1. Public model

- A Kernel, `vd.pipeline(...)`, initialized Module, and supported transform all
  lower to Program.
- Standalone executables are one-node Programs; Module differs only in node
  count.
- C++ uses bundle → executable → instance → invocation → bind → forward.
- VJP uses the same forward lifecycle and returns a reusable pullback.
- Public boundaries use stable slots and `VernonProgramArgument`.
- Python pass descriptors and public ExecutionGraph authoring are not Program
  APIs.

## 2. Input and output parity

Tests must cover equivalent direct Kernel and one-kernel Module behavior for:

- floating and integer Scalar Values;
- Tensor Values, including static aggregate Values;
- TensorView inputs, outputs, and read-write Storage;
- Tuple and Struct Values;
- dynamic shapes and rank-zero TensorViews;
- positive, negative, and non-contiguous strides where supported;
- Texture and Sampler rejection/acceptance according to stage capability.

Program outputs must verify:

- exact numerical content;
- shape, dtype, layout, and ownership;
- publication mode;
- failure rollback;
- no aliasing with temporary or retained pullback state unless explicitly
  declared in-place.

## 3. Module composition

Module tests must include:

- multiple distinct compute Stages;
- one Stage reused by multiple Nodes with distinct projections;
- TensorView-backed intermediate Storage;
- aggregate fan-in and fan-out;
- multiple public outputs;
- dynamic shape and grid reuse without recompilation;
- compute-to-graphics data flow;
- graphics-only and mixed compute/graphics Programs;
- resource hazard and command-order verification.

Internal Values are connected by canonical Value IDs and Storage versions.
Tests must not infer graph correctness from names or incidental endpoint order.

## 4. VJP

Direct Kernel and equivalent one-node Module VJP must agree for:

- primal outputs;
- selected `wrt` gradients;
- explicit and implicit scalar cotangents;
- Tensor and aggregate cotangents;
- mutable TensorView inputs;
- dynamic and signed-stride layouts;
- repeated pullback application;
- failure transaction and retained-state isolation.

Multi-node Program VJP must cover:

- reverse Node order;
- explicit cotangent fan-in;
- internal primal capture;
- tape and no-Tape plans;
- checkpoint/replay equivalence;
- dynamic control Values and launch geometry;
- scalar reduction across multiple invocations;
- device-resident residual and gradient paths.

Graphics derivative paths and unsupported opaque Resources must fail before
execution.

## 5. Binding and lifetime

Persistent Program instances must verify:

- all required slots are bound before invocation;
- rebinding changes only the selected slot;
- one conceptual read-write argument may satisfy reflected input and output
  boundaries through explicit same-name/slot mapping performed by canonical
  helpers;
- in-flight resources outlive caller handles;
- Runtime generation changes invalidate stale native bindings;
- invocations do not share mutable scratch;
- pullbacks retain immutable state and remain reusable.

## 6. Layout and aggregate behavior

Tests must prove:

- canonical Value layout is independent of Stage-local carrier packing;
- logical and physical aggregate leaves project explicitly;
- padding and alignment are preserved;
- static Tensor Values do not use TensorView Storage transport accidentally;
- aggregate TensorView owners scatter gradients into the correct tangent
  leaves;
- zero-sized tensors and rank-zero tensors remain distinct;
- malformed or ambiguous leaf projections fail closed.

## 7. Backend coverage

Backend-independent semantics run on every applicable backend:

- CPU;
- CUDA;
- Vulkan;
- DirectX 12;
- Metal;
- OpenGL;
- OpenGL ES.

Compile capability and runtime availability are evaluated independently.
Unavailable platforms, devices, contexts, or API versions are explicit skips.
Compile, load, resolve, bind, execute, synchronize, and result failures are
never converted into skips.

OpenGL compute requires OpenGL 4.3+. OpenGL ES compute requires OpenGL ES 3.1+.
Backend-native interop and ABI tests remain backend-specific.

## 8. Required gates

Before a Program-contract release:

1. build all enabled native targets;
2. run all CTest tests;
3. run the complete Python suite;
4. run MLIR lit tests;
5. run platform runtime matrices and audit skips;
6. run the independent WASM build and runtime gate;
7. verify that legacy schema/API source guards remain clean;
8. verify manifests, reflection, symbols, and cache identity are deterministic.

Removing or weakening an established regression requires evidence that the
corresponding public capability was removed from the normative contract.
