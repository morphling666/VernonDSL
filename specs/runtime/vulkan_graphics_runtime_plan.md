# Vulkan Offscreen Graphics Plan

## Goal

Run the existing Tensor-first graphics API on Vulkan:

```python
vd.init(arch=vd.vulkan)
render = vd.pipeline(compute_shader, vertex_shader, fragment_shader)
render(..., target=texture)
```

The first implementation is offscreen RGBA8 rendering with synchronous
readback. Swapchains and native presentation remain out of scope.

## Native runtime

- Select a Vulkan queue family supporting both graphics and compute.
- Load the image, render-pass, graphics-pipeline, draw, copy, and barrier
  Vulkan entry points.
- Create buffers with both storage-buffer and vertex-buffer usage so compute
  output can feed vertex input without a host copy.
- Implement `VernonDeviceTexture` with an optimal-tiled RGBA8 image, image
  view, framebuffer, tracked layout, and host-visible staging buffer.
- Store vertex and fragment SPIR-V modules and entry points in
  `VernonLoadedProgram`.
- Create and cache graphics pipelines lazily from draw binding signatures,
  because Vulkan bakes vertex formats and strides into pipeline state.
- Implement triangle-list, no-depth, clear-on-call drawing through the existing
  Texture/program/draw C ABI.
- Map stage uniforms to stage-specific push constants.
- Record compute-to-vertex and image-layout barriers on the shared queue.

## Python runtime

- Allow `vd.pipeline(...)` when the active architecture is Vulkan.
- Compile each graphics stage with `Target.VULKAN`.
- Select the reflected SPIR-V artifact instead of an OpenGL GLSL artifact.
- Load both SPIR-V stages through the existing native graphics binding.
- Preserve Tensor residency, inferred vertex/instance counts, Texture
  readback, and deterministic pipeline caching.
- Add `"vulkan": vd.vulkan` to `examples/unified_pipeline.py`; no
  backend-specific shader code should be required.

## Capabilities

Report Vulkan graphics support only when:

- a combined graphics-and-compute queue is available;
- RGBA8 supports color-attachment and transfer usage;
- the required graphics pipeline and synchronization operations are present.

Keep the current compute-only capability available on devices that do not meet
these graphics requirements.

## Verification

- Native RGBA8 Texture upload/readback round trip.
- Native triangle draw into a 64×64 target with clear and shaded pixel checks.
- Compute writes a shared buffer, an explicit barrier runs, and graphics reads
  the same allocation as vertex input.
- Repeated draws reuse the Texture, buffers, shader modules, and cached
  pipeline.
- Python graphics-only and compute→graphics pipelines run on Vulkan when
  available.
- Runtime-updated uniform bindings change rendered output.
- `examples/unified_pipeline.py --arch vulkan --frames 2` completes.
- Full Release CTest and Python test suites pass.

## Design records

Document these constraints in `specs/runtime/design.md`:

- the combined queue-family requirement for the MVP;
- image-layout transition ordering;
- push-constant packing and stage visibility;
- Vulkan pipeline cache keys derived from vertex binding layouts.
