# Unified Shader Execution Plan

## Goal

Give compute and graphics the same Tensor-first Python model:

```python
vd.init(arch=vd.opengl, api_version=(4, 3))

compute_shader(tensor, time, grid=(width, height, 1))

render = vd.pipeline(vertex_shader, fragment_shader)
# Optional preprocessing:
# render = vd.pipeline(compute_shader, vertex_shader, fragment_shader)
render(vertices=vertices, transform=transform, target=target)
```

The standalone VernonDSL runtime owns this API. OpenGL 4.3 offscreen rendering
is the first graphics implementation; Vernon Engine integration, presentation,
Vulkan graphics, geometry/tessellation, bindless resources, transform feedback,
and native EGL/GLES execution are later work.

## Function kinds

- `@vd.kernel` / `@vd.compute` entries are directly callable.
- `@vd.vertex` and `@vd.fragment` are declarative stages. Calling either from
  host Python, or creating a pipeline with only a vertex stage, is an error.
- Reusable shader helpers require `@vd.func`. They may be imported and called
  transitively by any shader stage or another `@vd.func`, but not by host
  Python.
- `@vd.func` is stage-polymorphic. Each reachable entry validates its
  operations for that stage. Builtins and resources must be explicit typed
  parameters; recursion and calls to stage entries are rejected.

A future explicit `vd.transform(vertex_shader)` may provide vertex-only
transform feedback without making ordinary vertex calls ambiguous.

## Pipeline composition

`vd.pipeline(stage0, stage1, ...)` accepts these MVP sequences:

1. `(vertex, fragment)`
2. `(compute, vertex, fragment)`

The second sequence is one Vernon program but remains a native compute pipeline
followed by a native graphics pipeline. The runtime dispatches compute, inserts
the reflected resource barrier, then draws.

External pipeline parameters are the union of stage parameter names. Equal
names identify the same Tensor across stages. Their types and access semantics
must be compatible. Builtins and vertex-to-fragment varyings are internal and
do not appear in the callable signature.

The graphics stages are not ordinary sequential function calls. Their execution
contains fixed-function work:

```text
vertex shader
-> primitive assembly
-> rasterization and varying interpolation
-> fragment shader
-> render target
```

## Tensor and Texture model

`Tensor` remains the only buffer/value container. Vertices, instance data,
transforms, uniforms, and storage resources are all Tensors. Shader annotations
select their interface behavior:

- `location(...)`: per-vertex input;
- `instance(...)`: per-instance input;
- `uniform(...)`: draw-shared data;
- `resource(...)`: addressable storage.

`Texture` is separate because an image has format, sampling, render-target, and
copy semantics that a linear Tensor does not.

Pipeline execution infers:

- vertex count from the common leading dimension of `location(...)` Tensors;
- instance count from the common leading dimension of `instance(...)` Tensors;
- viewport and scissor from the target Texture;
- compute grid from writable Tensor shapes.

All writable compute-domain Tensors must have the same shape. Row-major Tensor
dimensions map to GPU coordinates in reverse order: `(height, width)` maps to
`(width, height, 1)`. Kernels without an inferable writable domain will require
a future explicit override.

The OpenGL MVP uses triangle-list topology, one RGBA8 target, clear-on-call, no
depth, and non-indexed drawing.

## Backend version and capabilities

Backend API version and shader language version are different settings.
Initialization accepts an explicit context version:

```python
vd.init(arch=vd.opengl, api_version=(4, 3))
```

`api_version` selects the runtime context. OpenGL defaults to `(4, 3)` for
compatibility with the existing compute backend; graphics-only programs may
request `(3, 3)`. MVP compute support does not rely on `ARB_compute_shader` in
an older context. Native OpenGL ES version selection remains deferred with the
EGL runtime; the current desktop compatibility path uses the desktop context
version.

`glsl_version` remains a compiler option and must not be interpreted as the
context version. Runtime capability reporting records the actual context
version and support for compute, graphics, storage buffers, image load/store,
and other reflected requirements. Program creation validates those requirements
before compilation or execution and reports the missing capability. A
`(compute, vertex, fragment)` program therefore requires OpenGL 4.3, while a
`(vertex, fragment)` program may run on a requested 3.3 context.

## Persistent residency

Direct call syntax remains the primary API. Performance does not require public
Device, Encoder, Draw, or manual binding objects.

Each Tensor caches a native buffer per runtime generation and tracks host/device
versions:

- a host mutation such as `copy_from()` marks the Tensor host-dirty;
- a shader reads from the existing native allocation;
- only a host-dirty readable Tensor uploads before execution;
- a writable shader argument becomes device-dirty;
- `to_numpy()` downloads a device-dirty Tensor lazily.

Passing the same Tensor objects every frame is therefore cheap. Unchanged
vertices and matrices are neither reallocated nor retransmitted.

CUDA still passes its kernel argument list on every `cuLaunchKernel`, but stable
Tensor arguments reuse the same device pointers. Small by-value scalar
parameters may be copied per launch.

## Compilation and caching

Interactive `vd.pipeline(...)` compiles decorated stage callables directly. It
does not require a cooked shader asset.

The implementation reuses the existing module graph, helper inlining, interface
validation, SPIR-V lowering, SPIRV-Cross, and shader asset cache rules. Native
program cache keys include:

- stage source and transitive dependency hashes;
- selected entries and feature set;
- target and target options;
- merged external interface;
- generated stage artifact hashes.

The existing `shader-module` / `shader-pipeline` manifests and cooked
`shader.json` bundles remain the AOT deployment format.

## Native runtime additions

Extend `source/include/VernonRuntime.h` without changing the compute launch ABI:

- accept a requested backend API version and report the actual version;
- report `supports_graphics`;
- report stage and resource capabilities instead of assuming compute support;
- add `graphics_draw_abi_version`;
- add Texture creation, upload/readback, and lifetime APIs;
- add a multi-stage program handle;
- add reflected Tensor/Texture argument binding;
- add offscreen target and draw submission;
- add compute-to-graphics resource barriers.

The OpenGL implementation in `source/lib/VernonRuntime.cpp` will compile/link
vertex and fragment artifacts, create VAO/FBO/Texture objects, bind Tensor
buffers from reflection, execute optional compute, apply barriers, draw, and
read pixels. Reuse the proven patterns in Vernon Engine's
`render/shader.cpp` and `render/mesh_render_pass.cpp`.

## Implementation tasks

- [ ] Add and export mandatory `@vd.func`; validate local, imported, and
      transitive helper call graphs and migrate existing helpers.
- [ ] Add `vd.pipeline`, merged-name routing, legal stage-order checks, shape
      inference, and deterministic cache keys.
- [ ] Add automatic per-runtime Tensor residency, dirty version tracking,
      partial updates, and lazy readback while preserving direct kernel calls.
- [ ] Add Texture and graphics/multi-stage C runtime APIs without changing
      `vernonRuntimeLaunch`.
- [ ] Add `vd.init(..., api_version=...)`, exact OpenGL/OpenGL ES context
      requests, actual-version reporting, and reflected capability validation.
- [ ] Compile interactive stage callables and cache their native artifacts;
      preserve cooked assets as the AOT path.
- [ ] Implement OpenGL offscreen vertex/fragment execution and optional
      compute-to-graphics barriers.
- [ ] Expose only the focused Python surface: `vd.init`, `Tensor`, `Texture`,
      directly callable compute kernels, and directly callable pipelines.
- [ ] Record final ABI and non-obvious ordering constraints in runtime/compiler
      design documents.

## Verification

- Frontend: local/imported/transitive `@vd.func`, helper-to-helper calls,
  missing decorators, recursion, host calls, stage-entry calls, and
  stage-incompatible operations.
- Composition: direct vertex/fragment calls, lone vertex pipelines, invalid
  stage order, incompatible same-name parameters, unmatched stage interfaces,
  conflicting writable domains, and stable cache keys.
- Capabilities: OpenGL 3.3 accepts graphics-only composition and rejects
  compute composition; OpenGL 4.3 accepts both. Unsupported requested context
  versions and GLSL/context mismatches produce precise diagnostics.
- Residency: update only one of two matrices across repeated
  CUDA/Vulkan/OpenGL compute calls; verify stable allocations, no upload of the
  unchanged Tensor, correct results, and lazy download.
- Native OpenGL: render one triangle to a 64x64 RGBA8 Texture and verify clear
  pixels outside and shaded pixels inside.
- Python: execute both `(vertex, fragment)` and
  `(compute, vertex, fragment)` directly with named Tensors; verify inferred
  counts/grid, barrier visibility, Texture readback, diagnostics, and unchanged
  resource reuse.
- Run the complete native CTest and Python suites.
