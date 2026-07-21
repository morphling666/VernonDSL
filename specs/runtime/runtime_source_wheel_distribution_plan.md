# Runtime Source-in-Wheel Distribution Plan

Status: proposed; not started. This is a future packaging and Vernon Engine
integration design, not a description of the current setuptools package.

## Goal

Ship the version-matched `VernonRuntime` source tree inside the VernonDSL
Python wheel. `uv sync` installs the source but does not compile it. Vernon
locates that installed source during CMake configuration and compiles it with
Vernon's own desktop, mobile, or WebAssembly toolchain.

The Python wheel also contains a prebuilt runtime for standalone Python use.
When Python is embedded in Vernon, the Engine-built runtime is the process
runtime provider and `_native` must reuse it.

## Wheel layout

Install the minimum standalone source closure:

```text
vernon_dsl/
  _native.*
  VernonDSLCompiler.*
  VernonRuntime.*              # standalone Python fallback
  runtime_src/
    CMakeLists.txt
    include/
      VernonCommon.h
      VernonRuntime.h
      vernon-c/
    lib/
      VernonRuntime.cpp
      runtime/
    cmake/
      VernonRuntimeTarget.cmake
    VERSION
```

Do not include LLVM/MLIR sources, compiler implementation sources, tests,
examples, build outputs, or generated platform binaries in `runtime_src`.
Vulkan headers and `nlohmann_json` remain external CMake dependencies.

## Source discovery

Add a stable Python API and command:

```python
from vernon_dsl.runtime_source import cmake_source_dir, version
```

```powershell
uv run python -m vernon_dsl.runtime_source --cmake-dir
```

The returned directory is read-only package data. CMake must always use a
separate binary directory and must never generate files under `site-packages`.

Vernon configuration accepts:

```cmake
VERNON_RUNTIME_SOURCE_DIR=<explicit development checkout or wheel path>
VERNON_DSL_PYTHON=<python from the selected uv environment>
```

Resolution order:

1. explicit `VERNON_RUNTIME_SOURCE_DIR`;
2. query `VERNON_DSL_PYTHON -m vernon_dsl.runtime_source --cmake-dir`;
3. fail with a precise instruction to run `uv sync`.

After discovery, Vernon uses:

```cmake
add_subdirectory(
  "${VERNON_RUNTIME_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/VernonRuntime"
)
target_link_libraries(VCore PUBLIC Vernon::Runtime)
```

`uv sync` must not use install hooks to invoke CMake or mutate the Vernon
checkout. Cross compilation belongs to the Vernon build.

## Toolchain and platform profiles

The source package must accept the parent project's active toolchain and avoid
host-environment assumptions.

Add these standalone options:

```cmake
VERNON_RUNTIME_LIBRARY_TYPE=SHARED|STATIC
VERNON_RUNTIME_PROFILE=desktop|mobile|web
VERNON_ENABLE_CUDA_RUNTIME=ON|OFF
VERNON_ENABLE_VULKAN_RUNTIME=ON|OFF
```

Profiles:

- `desktop`: shared library by default; CPU AOT dynamic loading, CUDA, Vulkan,
  and external OpenGL/OpenGL ES may be enabled.
- `mobile`: static library by default; external OpenGL ES is enabled; Vulkan
  is platform-selectable; unsupported dynamic CPU AOT paths are disabled.
- `web`: static library; external WebGL/OpenGL ES only; disable CUDA, Vulkan
  loader discovery, and dynamic CPU AOT loading.

Backend capability queries must report disabled profile features accurately.
Platform-specific source files should be selected by CMake rather than hidden
behind unavailable system headers.

## Device/context ownership modes

Support two explicit ownership modes for graphics backends:

1. **External/adopted**: Vernon creates the OpenGL context or Vulkan device and
   passes borrowed handles plus procedure/synchronization callbacks to Runtime.
   This is the production Engine path because it shares Vernon's window,
   render graph, resources, and queue schedule.
2. **Runtime-owned**: Runtime creates and destroys the backend objects. This is
   the standalone/headless Python path.

Initial Engine integration decision:

- Vernon uses external OpenGL and external Vulkan.
- Do not add a runtime-owned OpenGL context provider in the first
  implementation.
- Keep the existing runtime-owned Vulkan path only for standalone/headless
  Python.
- Implement external Vulkan adoption and resource/synchronization import
  before adding a Vulkan backend to Vernon production rendering.

Vulkan can create an instance/device/queue without a window for compute and
offscreen graphics. Presentation additionally requires a host-created
`VkSurfaceKHR` or a surface-creation callback.

OpenGL has no API that creates an OpenGL device/context. A runtime-owned OpenGL
mode therefore requires a platform provider:

```c
typedef struct VernonOpenGLContextProvider {
  void *user_data;
  VernonStatus (*create_context)(void *user_data,
                                 const VernonOpenGLContextRequest *request,
                                 void **context);
  void (*destroy_context)(void *user_data, void *context);
  VernonStatus (*make_current)(void *user_data, void *context);
  void *(*get_proc_address)(void *user_data, const char *name);
} VernonOpenGLContextProvider;
```

SDL/GLFW/EGL/WGL remains on the provider side. Runtime invokes the callback
table but does not link those libraries. Passing only
`SDL_GL_GetProcAddress`/`gladLoadGL` is insufficient to create a context: an
SDL GL context must already exist and be current. The current
`VernonExternalOpenGLContext` is the external/adopted mode because it receives
`make_current` and `get_proc_address` for an existing context.

## Engine-owned Vulkan context

The current Vulkan runtime creates and owns its own instance, device, queue,
and command pool. It cannot directly reuse a Vulkan context initialized by
Vernon's `MainWindow`. Add an external Vulkan constructor before Vernon enables
its Vulkan renderer:

```c
VernonRuntimeContext *vernonRuntimeCreateExternalVulkan(
    const VernonExternalVulkanContext *external);
```

Keep Vulkan-specific declarations in `VernonRuntimeVulkan.h` so the core C
header remains independent of Vulkan headers. The versioned external descriptor
provides:

- `VkInstance`, `VkPhysicalDevice`, and `VkDevice`;
- graphics/compute `VkQueue`, family index, and queue index;
- instance/device procedure-address callbacks;
- enabled API version, extensions, features, and optional allocator callbacks;
- optional host submission/synchronization callbacks.

The runtime borrows instance/device/queue handles and never destroys them. It
may create and own command pools, descriptor pools, pipeline layouts, shader
modules, and other child objects on the borrowed device.

Vulkan queues require external synchronization. The first implementation must
submit only on Vernon's graphics thread and may use `vkQueueWaitIdle` for
correctness. The production path should return a fence/timeline-semaphore value
or submit through a Vernon callback so the render graph can order runtime work
without stalling the entire queue.

Add non-owning resource imports:

```c
vernonRuntimeImportVulkanBuffer(...)
vernonRuntimeImportVulkanImage(...)
```

Imported records include size/format/extent/usage, current image layout, queue
family ownership, and optional image view. Vernon retains memory ownership.
Runtime invocation must report final image layout and synchronization state
back to the host; it must not assume that an imported image starts in
`VK_IMAGE_LAYOUT_UNDEFINED`.

Owned Vulkan creation through `vernonRuntimeCreate(VERNON_RUNTIME_VULKAN, ...)`
remains available for standalone Python/headless execution. Engine builds use
the external constructor, ensuring Vernon rendering and pipeline bundles share
one device, queue schedule, pipeline cache policy, and resource universe.

## ABI and same-process ownership

Add runtime ABI-major, semantic version, source revision, and build-id queries
to `VernonRuntime.h`.

- ABI-major and source revision must match between `_native` and the
  Engine-built runtime.
- Build-id may differ because Vernon uses a different toolchain or static/shared
  configuration; it is diagnostic, not an equality requirement.
- The Engine loads its runtime before initializing embedded Python.
- The fallback runtime in the wheel uses the same ABI-major library basename.
  The platform loader must reuse the already-loaded Engine runtime.
- `_native` fails import with a precise version diagnostic when the loaded
  runtime is incompatible.

Runtime handles and imported graphics resources must never cross between two
different loaded runtime modules.

## Python build backend

Replace the pure-setuptools wheel build with `scikit-build-core` and nanobind.
CMake installs:

- `_native`;
- `VernonDSLCompiler` and required compiler libraries;
- the standalone fallback `VernonRuntime`;
- `runtime_src` package data.

Move build/development tools such as Conan, clang-format, Taichi, and web
services out of mandatory runtime dependencies into optional extras.

Use `cibuildwheel` for host platform wheels. Packaging repair tools must retain
the stable Runtime library identity instead of renaming it to a wheel-private
ABI name.

## Vernon integration

Replace the current checkout-specific `VERNON_DSL_SOURCE_DIR` contract in
Vernon's top-level CMake with runtime-source discovery. Keep an explicit source
override for simultaneous Vernon/VernonDSL development.

Vernon's Conan configuration continues to provide C++ dependencies such as
`nlohmann_json` and Vulkan Headers. It does not download a second copy of the
Runtime source.

On Windows, stage the Engine-built shared Runtime beside executables that
import it. Static mobile/web builds require no runtime staging.

## Implementation stages

1. **Standalone source closure**
   - Extract a self-contained `runtime_src/CMakeLists.txt`.
   - Add library type and platform profile options.
   - Verify native Windows/Linux/macOS and cross WebAssembly configuration.

2. **Wheel source packaging**
   - Adopt `scikit-build-core`.
   - Install `runtime_src` and expose the discovery API/CLI.
   - Add a test that compares packaged source revision with `_native`.

3. **Vernon discovery**
   - Query the selected uv Python during configure.
   - Preserve explicit checkout override.
   - Compile in a Vernon-owned binary directory with Vernon's toolchain.

4. **External Vulkan integration**
   - Add the external instance/device/queue constructor.
   - Add borrowed buffer/image imports and layout/queue ownership metadata.
   - Integrate submission with Vernon's graphics thread and synchronization.

5. **Runtime provider validation**
   - Add ABI/version query APIs.
   - Validate the runtime loaded by `_native`.
   - Test standalone Python and same-process Vernon embedded Python.

6. **Cross-platform verification**
   - Desktop shared builds and DLL/RPATH staging.
   - Android/iOS static mobile profile.
   - Emscripten static web profile.

## Verification

- Build and import a wheel without a Vernon checkout.
- Run `uv sync`, delete all VernonDSL source checkout paths, and configure
  Vernon using only the wheel-provided `runtime_src`.
- Confirm Vernon compile commands use Vernon's selected compiler, sysroot,
  architecture, sanitizer flags, and Emscripten/mobile toolchain.
- Build normal desktop tests and verify only one runtime module is loaded in
  the embedded-Python process.
- Reject wheel/Engine ABI or source-revision mismatches before creating runtime
  contexts.
- Inspect desktop imports to confirm no LLVM, MLIR, GLFW, CUDA Toolkit, or
  Vulkan loader import-library dependency enters `VernonRuntime`.
