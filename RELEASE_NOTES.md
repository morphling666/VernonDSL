# VernonDSL 0.1.2 release notes

Status: development release notes. Final publication requires every gate in
[`RELEASE_READINESS.md`](RELEASE_READINESS.md).

VernonDSL 0.1.2 uses Compiler Contract 1 and Program Version 1. The latest
released version is 0.1.1. These contract numbers remain fixed throughout the
0.1.2 release line.

## Unified Program assets

- Kernel, `vd.pipeline(...)`, initialized Module, and explicit VJP transforms
  compile to one canonical Program model.
- Standalone compute and graphics executables are one-node Programs.
- Cooked assets contain one Program plus target-specific ArtifactSystem
  variants and content-addressed Stage artifacts.
- The public C/C++ lifecycle is bundle → executable → instance → invocation →
  bind → forward.
- Runtime owns command recording and submission internally.
- Legacy pipeline manifests, profile executors, Stage binding tables, direct
  asset submit paths, and compatibility normalization were removed.

## Program execution

- `ResolvedExecutionPlan` is the immutable physical execution authority.
- Nodes retain explicit Program-to-Stage endpoint and aggregate-leaf
  projections even when multiple Nodes reuse one Stage.
- Runtime plans residency, uploads, device copies, readbacks, hazards, backend
  barriers, graphics scopes, and publication transactions before invocation.
- Dynamic TensorView shape, stride, offset, extent, and compute grid remain
  invocation data and do not trigger Stage recompilation.
- Zero-grid and zero-extent inference sentinels were removed.
- Buffer and image `commit_after_success` publication is transactional.

## Module and graphics

- Cooked compute Modules, graphics Modules, and mixed compute-to-graphics
  Modules execute through the public Program API.
- Graphics assets use `vd.pipeline(...)` with declared target formats.
- Render pass, draw command, and dynamic state are canonical Program controls.
- Attachments use explicit image subresource versions and load/store/clear/
  resolve semantics.
- Native render-scope fusion remains a private Command DAG optimization.
- Graphics differentiation remains unsupported and fails closed.

## Autodiff

- Direct Kernel VJP and Module VJP share the canonical differentiated Program
  path.
- Forward and backward graphs use explicit residual, cotangent, and gradient
  boundaries.
- Pullbacks retain immutable Value, Storage, tape, replay, and checkpoint
  state; each application creates fresh invocation scratch.
- CPU and supported GPU backends execute covered compute Program VJP,
  including aggregate Storage, dynamic/signed-stride TensorViews, scalar
  reduction, tape, and no-Tape rematerialization cases.
- Boundary binding uses `VernonProgramArgument`; parallel AD-specific execution
  and reflection ABIs were removed.

## Runtime and platform

- CPU AOT emits relocatable objects and generated static-registration sources.
- wasm32 uses target-sized pointer ABI layouts and statically linked CPU
  Program objects.
- CUDA, Vulkan, DirectX 12, Metal, OpenGL, and OpenGL ES remain
  capability-gated at compile and runtime boundaries.
- OpenGL/OpenGL ES use host-owned contexts; compute requires OpenGL 4.3+ or
  OpenGL ES 3.1+.
- External-engine desktop and browser examples use the canonical Program
  loader and invocation lifecycle.

## Distribution

The release target is wheel-only:

- CPython 3.11 through 3.14;
- Windows x64;
- Linux x64 with the wheel's manylinux tag;
- Apple Silicon macOS 15 or newer.

GPU drivers, Vulkan loaders/ICDs, and window-system contexts are not bundled.
Runtime availability depends on the selected backend and host capabilities.

The Runtime provides offscreen graphics and host readback. Swapchain/window
presentation, CPU rasterization, CUDA image/sampler support, GPU f16, graphics
VJP, and guaranteed concurrent multi-frame execution are outside the current
contract.

## Compatibility

Artifacts from older compiler or Program contracts are rejected rather than
translated. Runtime embedders rebuild the bundled Runtime source together with
each VernonDSL release.

Public API, compatibility, support, and security policies are documented in:

- [`PUBLIC_API.md`](PUBLIC_API.md)
- [`COMPATIBILITY.md`](COMPATIBILITY.md)
- [`SUPPORT.md`](SUPPORT.md)
- [`SECURITY.md`](SECURITY.md)

Current contracts and architecture are indexed by
[`specs/README.md`](specs/README.md).
