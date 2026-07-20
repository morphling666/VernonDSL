# Lightweight Multibackend Runtime Plan

## Goal

Make `VernonRuntime` the deployable execution library shared by Vernon Engine,
Python, and pure C/C++ applications. It must support:

- CPU pipelines compiled ahead of time to native libraries;
- externally owned OpenGL contexts and resources;
- runtime-owned headless Vulkan compute and graphics;
- dynamically loaded CUDA compute.

`VernonRuntime` must not link LLVM, MLIR, GLFW, the Vulkan loader, or the CUDA
Toolkit. Compiler and display-tool dependencies remain in VernonDSL tooling.

The first functional milestone is:

```powershell
uv run python examples/complete_pipeline.py --arch vulkan --headless --frames 3
```

This must execute compute, a compute-to-graphics barrier, indexed instanced
drawing, dynamic TensorView layouts, feature variants, MRT rendering, and
texture readback.

## Decisions

### Runtime ownership

- The first Vulkan implementation creates and owns a headless Vulkan instance,
  device, combined graphics/compute queue, and command resources.
- Vernon Engine remains on external OpenGL for this milestone.
- Engine-owned Vulkan and `vernonRuntimeCreateExternalVulkan` are deferred.
- CUDA remains compute-only. Vulkan-CUDA external-memory interoperability is
  deferred.

### Dependency boundary

`VernonRuntime` may depend on:

- `nlohmann_json`;
- Vulkan headers at compile time;
- operating-system dynamic-library APIs;
- graphics/compute drivers discovered at runtime.

It must dynamically load:

- `vulkan-1.dll`, `libvulkan.so.1`, or the platform equivalent;
- `nvcuda.dll` or `libcuda.so.1`.

It must not link:

- LLVM or MLIR;
- GLFW;
- a Vulkan loader import library;
- `cudart`, `nvrtc`, or other CUDA Toolkit libraries.

### CPU execution

CPU execution does not require an LLVM JIT in the runtime.

VernonDSLCompiler lowers CPU entries to native objects and links a
platform-specific `.dll`, `.so`, or `.dylib` during cooking. It generates one
stable exported C wrapper per entry. The wrapper accepts Vernon invocation
types and adapts them to the lowered memref/function ABI.

The bundle records:

- operating system and architecture;
- runtime invocation ABI version;
- exported entry symbol;
- native artifact content hash.

`VernonRuntime` uses `LoadLibrary/GetProcAddress` or `dlopen/dlsym` and invokes
the wrapper directly. Interactive Python CPU execution uses the AST
interpreter; deployable Python can use `CpuAotRuntime` to load the same native
bundle format. LLVM remains a compiler dependency only.

Static registration may be added later for platforms that prohibit dynamic
code loading.

## Implemented state

One `VernonRuntime` target and one implementation in
`source/lib/VernonRuntime.cpp` provide CPU AOT, dynamically loaded CUDA,
Vulkan compute/graphics, and external-context OpenGL/OpenGL ES. One `_native`
module exposes compiler and runtime services to Python.

Vulkan supports storage buffers, textures, staging transfers, compute and
graphics pipelines, render passes, MRT, indexed/instanced drawing, barriers,
and pipeline bundles. Python compiles interactive GPU stages into the same
versioned bundle representation used by cooked assets.

## Target Architecture

```text
VernonDSLCompiler (LLVM/MLIR)
  |-- CPU native library + C wrappers
  |-- Vulkan SPIR-V
  |-- CUDA PTX
  `-- OpenGL GLSL
               |
               v
      normalized pipeline.bundle
               |
               v
VernonRuntime (no LLVM/MLIR/GLFW)
  |-- CPU native-library loader
  |-- external-context OpenGL
  |-- runtime-owned Vulkan
  `-- dynamically loaded CUDA
               |
       stable pipeline C ABI
        /              \
   Python adapter    Vernon adapter
```

The deployable runtime owns bundle parsing, exact variant resolution, concrete
argument validation, resource binding, dispatch/draw orchestration, and
pipeline-local synchronization. Python and Vernon only map ergonomic names and
native resource wrappers to stable slots.

## Implementation Plan

### 1. Split the lightweight runtime

Refactor `source/lib/VernonExternalOpenGLRuntime.cpp` and reusable portions of
`source/lib/VernonRuntime.cpp` into focused runtime sources, for example:

```text
source/lib/runtime/
  platform_library.*
  runtime_context.*
  runtime_c_api.cpp
  pipeline_bundle.*
  backend_cpu_aot.*
  backend_opengl_external.*
  backend_vulkan_driver.*
  backend_vulkan.*
  backend_cuda_driver.*
  backend_cuda.*
```

Replace:

- `llvm::sys::DynamicLibrary` with a Win32/POSIX loader;
- `llvm::json` with `nlohmann_json`;
- LLVM collection/utility helpers with the C++ standard library.

Move shared `VernonStatus` and `VernonStringView` declarations to a common C
header so `VernonRuntime.h` does not depend on the compiler API.

Keep one stable public runtime C ABI and dispatch contexts, buffers, textures,
bundles, and loaded pipelines by backend.

### 2. Rework CMake targets

Update root and `source/CMakeLists.txt` so a standalone runtime configuration
can build:

- CPU AOT loading;
- Vulkan;
- CUDA;
- external OpenGL.

This configuration must return before LLVM/MLIR/compiler/GLFW discovery.

`Vulkan::Headers` is compile-only. CUDA uses locally declared Driver API types
and dynamically loaded symbols. External OpenGL receives function addresses
from the host.

The full VernonDSL build may build the compiler and Python module, but those
targets consume the same lightweight `VernonRuntime`; they must not introduce
LLVM or GLFW as transitive runtime dependencies.

### 3. Normalize target artifacts

Refactor `python/vernon_dsl/shader_assets.py` so AOT cooking and Python JIT use
one bundle-construction function.

The bundle stores:

- schema and invocation ABI versions;
- pipeline ID and content hash;
- exact sorted feature keys;
- stable parameter slots and constraints;
- outputs and host-only draw state;
- ordered `dispatch`, `barrier`, and `draw` steps;
- backend/capability requirements;
- target-aware artifacts.

GLSL remains UTF-8 text. SPIR-V is stored as binary data encoded for the
single-file JSON bundle, with explicit format and encoding metadata. CPU native
libraries and other large deployable artifacts may be referenced by
content-addressed relative paths with hashes.

The runtime must reject unsupported schema, ABI, target, capability, artifact
format, and content-hash combinations before creating GPU objects.

### 4. Implement Vulkan graphics

Extend the extracted Vulkan driver table and context initialization with:

- graphics and compute queue-family selection;
- device-level graphics, image, render-pass, synchronization, and copy
  functions;
- graphics capability and RGBA8 format checks.

Implement:

- buffers usable as storage, vertex, index, transfer source, and transfer
  destination;
- optimal-tiled RGBA8 images and image views;
- staging upload and readback;
- explicit image-layout transitions;
- shader modules for compute, vertex, and fragment SPIR-V;
- descriptor/push-constant layouts required by compiler reflection;
- render passes and framebuffers for multiple color attachments;
- indexed and non-indexed instanced drawing;
- triangle, line, and point topology;
- compute-to-vertex/graphics memory barriers;
- synchronous submission for the initial runtime contract.

Vulkan vertex stride and offset can change between invocations. Cache concrete
graphics pipelines using the loaded program plus vertex layout, topology,
attachment formats/count, and other fixed Vulkan state as the cache key.

Only report `supports_graphics` when the selected device satisfies the complete
offscreen graphics contract.

### 5. Move Python to unified invocation

Expose bundle loading, exact variant resolution, and pipeline invocation from
`source/python/native_module.cpp`.

Refactor `python/vernon_dsl/runtime.py::Pipeline` so it:

1. compiles stages for the selected target;
2. constructs the same normalized in-memory bundle used by AOT cooking;
3. loads and resolves the exact feature variant;
4. maps Python argument names to precomputed stable slots;
5. converts Tensor, TensorView, Texture, scalar, index, target, topology, and
   dynamic state values into borrowed invocation views;
6. calls `vernonRuntimePipelineInvoke`.

Remove Python ownership of backend binding rules, matrix expansion, generated
uniform names, MRT routing, and compute-to-graphics orchestration.

Use the same path for OpenGL and Vulkan to prevent the implementations from
diverging.

### 6. Enable the complete example

Add `vulkan` to `examples/complete_pipeline.py --arch`. OpenGL keeps its API
version argument; Vulkan initialization does not receive an OpenGL version.

Vulkan remains headless and offscreen. OpenCV displays or writes images after
`Texture.to_numpy()` readback.

### 7. Extract CUDA into the lightweight target

Move the existing CUDA Driver API loader, context, buffer, PTX module, launch,
copy, and synchronization code into the lightweight runtime. Replace LLVM
dynamic-library helpers with the shared platform loader.

CUDA remains a compute backend. A bundle containing a draw step must return an
explicit unsupported-target error.

## Verification

### Dependency checks

- Configure and build `VernonRuntime` without configuring LLVM, MLIR, or GLFW.
- Inspect the produced library and verify it does not import LLVM, GLFW,
  Vulkan loader, CUDA Runtime, or CUDA Toolkit libraries.
- Build Vernon Engine against this target and retain the existing external
  OpenGL pipeline test.

### CPU AOT tests

- Cook a CPU native library with an exported wrapper.
- Load and invoke it from a process linked only to `VernonRuntime`.
- Verify Tensor shape/stride/value behavior.
- Reject mismatched OS, architecture, ABI, symbol, and hash metadata.
- Verify Python compiler-cache execution uses the same AOT loader.

### Vulkan native tests

- Device/capability probing.
- Buffer upload, dispatch, and readback.
- RGBA8 texture upload and readback.
- Basic triangle rendering.
- MRT routing by output location.
- Indexed instanced drawing.
- Dynamic packed and interleaved TensorView layouts.
- Compute writes consumed by vertex input after a barrier.
- Exact feature variant selection and malformed bundle rejection.
- Graphics pipeline cache reuse.

Hardware tests may skip only when no usable Vulkan device is present. Bundle
parser and validation tests must remain hardware-independent.

### Python tests

- Cook and validate Vulkan bundles.
- Run unified OpenGL and Vulkan pipeline invocation tests.
- Verify argument errors come from the shared native planner.
- Verify dynamic resource rebinding does not trigger stage recompilation.

### Acceptance

Run:

```powershell
uv run python examples/complete_pipeline.py `
  --arch vulkan `
  --headless `
  --frames 3 `
  --output build/complete_pipeline_vulkan.png `
  --id-output build/complete_pipeline_vulkan_id.png
```

All three feature variants must execute. Packed and swizzled vertex layouts,
both index buffers, both tint values, compute animation, and both MRT outputs
must produce correct readback.

## Deferred Work

- Vernon Engine `GraphicsBackend::Vulkan`;
- swapchain and presentation;
- `vernonRuntimeCreateExternalVulkan`;
- engine-owned Vulkan buffers and textures;
- CUDA graphics;
- Vulkan-CUDA external memory and semaphore interoperability;
- static CPU pipeline registration for restricted platforms.
