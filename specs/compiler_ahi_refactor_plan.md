# Compiler and AHI backend decomposition

## Goal

Reduce `VernonCompiler.cpp` and `VernonRuntime.cpp` to thin C API entry points.
Backend-specific operations and state must live in backend-specific modules.
The refactor must preserve the public C ABI, generated artifacts, reflection,
pipeline manifests, and runtime behavior.

This is a structural refactor. It does not add a backend, asynchronous
execution, or a new public C++ interface.

## Current status (2026-07-23)

The structural backend decomposition is complete.

- `VernonCompiler.cpp` is a thin compiler C API shell with target dispatch in
  compiler modules.
- `VernonRuntime.cpp` includes no backend headers, contains no optional-backend
  preprocessor branches, and never accesses concrete backend state. All
  backend selection and state access crosses `runtime_dispatch`.
- CPU, CUDA, OpenGL, and Vulkan implementation files are backend-owned. CUDA
  and Vulkan sources are selected only when their Runtime capabilities are
  enabled.
- Runtime handles retain backend state opaquely with backend-specific
  deleters. The common state header has no GL, Vulkan, or CUDA dependency.
- `source/lib/runtime/` and `source/lib/compiler/` are independently
  configurable CMake projects. The repository root is only their composition
  build.
- Runtime exports `Vernon::Runtime`, packages the canonical Runtime source
  project in the wheel, and compiles private test hooks only with
  `BUILD_TESTING`.
- Runtime common code is compiled once into `VernonRuntimeInternals` and linked
  by Runtime and its tests; tests no longer compile duplicate implementation
  sources.
- `nlohmann_json` is a Runtime dependency, resolved once per composition
  build. Compiler does not declare it; compiler-only tests request it as a
  test dependency when Runtime is absent.
- Vernon Engine consumes `source/lib/runtime/` directly and does not set
  compiler/runtime build-enable flags.

Verification completed on Windows:

- standalone CPU/OpenGL, CUDA, and Vulkan Runtime configurations built;
- standalone Compiler configuration built;
- all-backend composition build passed 41 CTest tests, including compiler C
  API all-target artifacts, reflection, native/CLI surface parity, Runtime
  pipelines, and OpenGL/Vulkan/CUDA availability-gated tests;
- the complete wheel-backed Python suite passed 95 tests;
- a `BUILD_TESTING=OFF` wheel built and installed, and its packaged Runtime
  source project built independently;
- an installed Runtime package was consumed through `find_package` and
  `Vernon::Runtime`;
- the wheel-owned OpenGL context smoke test passed;
- Vernon Engine built `VCore` against only the Runtime subproject.

Contract note:

- The pre-existing working-tree changes to `VernonRuntime.h` that add texture
  constraint queries and remove the directory bundle loader remain separate
  API work. This refactor neither introduced nor reverted them; its
  ABI-preservation claim applies to the structural changes listed above.

### Performance follow-up

Prioritize work that removes repeated game-loop work:

1. Cache the resolved invocation plan and stable parameter/resource layout.
   `vernonRuntimePipelineInvoke` currently rebuilds planner vectors/maps and
   repeats layout validation for every invocation. Rebuild only when pipeline
   features, bindings, or mutable invocation state changes.
2. Parse manifest strings such as texture dimension, texture format, access,
   stage, topology, and interface kind into validated enums once during bundle
   load or pipeline resolution. Hot invocation code should compare integers,
   not strings.
3. Cache backend-native values that are stable for a resource or pipeline,
   such as `VkFormat`, GL texture mappings, descriptor layout keys, and
   topology. Do not repeatedly translate them during submission.
4. Keep texture-constraint reflection outside the game loop. Vernon currently
   queries constraints by parameter index while resolving a pipeline and
   stores them in `PipelineParameter`; texture bind/rebind then performs only
   constant-time enum checks. Avoid the name-based linear lookup on hot paths.
5. If `texture_format` remains a constraint, enforce it using the pre-parsed
   enum during invocation or binding validation. Do not reparse the manifest
   string per invocation.
6. Keep Vernon engine texture types independent from the Runtime C ABI and
   convert only in `runtime_texture_bridge`. These switch-based conversions are
   load/import-time capability checks and are not a useful optimization target.
7. Do not replace small enum switches with hand-written lookup tables without
   profile evidence. Optimizing compilers already lower dense switches to jump
   tables or constant tables; bounds checking and cache access can erase any
   theoretical gain.

Measure the invocation planner, allocation count, pipeline-cache hit rate, and
submission cost before and after these changes. GPU object creation, transfers,
and driver calls should be reported separately from CPU planning time.

## Design rules

1. Keep the public API in `VernonCompiler.h`, `VernonRuntime.h`, and
   `VernonCommon.h` unchanged.
2. Keep the compiler and runtime as separate shared libraries. `_native`
   remains their intentional Python composition root.
3. Keep backend selection explicit. C API entry points may contain one
   target/backend switch, but backend operations must not be interleaved.
4. Do not introduce an `IBackend` hierarchy, a backend registry, or virtual
   dispatch. The backend set is small and compile-time known.
5. Move behavior before changing representation. First move existing backend
   functions verbatim; only then replace mixed state structures.
6. Start with one coarse source file per backend. Split a backend further only
   when it has an independently testable responsibility.
7. Common code may define contracts, validation, metadata, planning, hashing,
   and data packing. It must not own GL, Vulkan, CUDA, LLVM, or LLD objects.
8. Backend modules must not include one another.

## Target architecture

```text
Public C API
    |
    +-- VernonCompiler.cpp
    |     validation + explicit target dispatch + result accessors
    |          |
    |          +-- common reflection and artifact contracts
    |          +-- CPU compiler
    |          +-- CUDA compiler
    |          +-- SPIR-V compiler
    |                 |
    |                 +-- SPIRV-Cross adapters
    |
    +-- VernonRuntime.cpp
          validation + explicit backend dispatch + handle accessors
               |
               +-- common bundle, metadata, planning, and tensor bridge
               +-- CPU AHI
               +-- CUDA AHI
               +-- OpenGL AHI
               +-- Vulkan AHI
```

`VernonRuntime` remains a GLFW-free AHI. Context owners such as `_gl_context`
or Vernon Engine provide the same context-access callbacks. The AHI does not
record which layer owns or created the context.

## Compiler decomposition

### Common compiler state

Introduce a private `compiler_internal.h` containing only backend-neutral
declarations:

- compiler diagnostic/error state;
- compile options normalized from the public struct;
- target-independent reflection records;
- named artifact records;
- opaque CPU execution state owned through a forward-declared pointer.

LLVM ORC, LLD, SPIRV-Cross, CUDA/NVVM, and backend pass headers must not be
included by this common header.

### Modules

Initial modules:

- `compiler_reflection.cpp`
  - entry-point reflection;
  - sampled-texture provenance;
  - parameter and output records;
  - reflection JSON serialization.
- `compiler_artifacts.cpp`
  - canonical target and artifact names;
  - artifact insertion and lookup;
  - result hash inputs.
- `compiler_spirv.cpp`
  - common Vulkan-compatible SPIR-V lowering;
  - SPIR-V validation and image-query materialization;
  - Vulkan artifacts.
- `compiler_spirv_cross.cpp`
  - GLSL, GLES, and MSL source generation from SPIR-V;
  - no MLIR lowering ownership.
- `compiler_cuda.cpp`
  - GPU/NVVM lowering;
  - PTX emission.
- `compiler_cpu.cpp`
  - CPU lowering and ABI wrapper generation;
  - LLVM object emission;
  - optional ORC execution state;
  - optional LLD native-library finalization.
- `VernonCompiler.cpp`
  - C API validation and lifetime;
  - explicit target dispatch;
  - result/artifact/reflection accessors.

The current Vulkan-oriented graphics path should be named according to its
actual responsibility: SPIR-V production. Vulkan is one consumer; GLSL, GLES,
and MSL are adapters over that result.

### Compiler invariants

- Validation still runs before compilation.
- Pass ordering remains explicit in each backend module.
- Reflection JSON, artifact names, artifact bytes, and content hashes remain
  byte-identical during extraction.
- CPU `module_hash` is computed at the same point relative to ABI wrapper and
  object generation.
- MLIR modules continue to be parsed where they are parsed today during the
  behavior-preserving stages. Sharing parsed modules is a separate
  optimization because it can alter diagnostics and pass side effects.
- DirectX remains explicitly unsupported; unused HLSL linkage may be removed.

## AHI decomposition

### Common runtime modules

- `runtime_common`
  - error state;
  - backend tag;
  - live-child counters;
  - common handle metadata.
- `pipeline_bundle`
  - schema-2 parsing;
  - content hashes and artifact resolution;
  - variant validation.
- `pipeline_metadata`
  - public parameter and output views;
  - reflection-backed lookup.
- `graphics_invocation_planner`
  - backend-neutral invocation validation and planning.
- `tensor_bridge`
  - tensor bounds and stride validation;
  - contiguous checks;
  - host packing required by backend ABIs.

### Backend modules

- `backend_cpu`
  - host buffers;
  - static/AOT entry registration and resolution;
  - CPU kernel launch.
- `backend_cuda`
  - dynamically loaded driver;
  - CUDA context, buffers, modules, and kernel launch.
- `backend_opengl`
  - context-access callbacks and proc loading;
  - owned/imported GL resources;
  - compute and graphics programs;
  - pipeline resolution and invocation encoding.
- `backend_vulkan`
  - loader, instance/device, queues, and command resources;
  - buffers, images, samplers, shaders, and compute pipelines;
  - graphics pipeline resolution, caching, and invocation encoding.
- `VernonRuntime.cpp`
  - public C API validation and lifetime;
  - one explicit backend dispatch per operation;
  - common public metadata accessors.

### Internal handle model

Public handles remain opaque C types. Internally, each handle has a small
tagged base and a backend-specific concrete representation:

```cpp
struct VernonRuntimeContext {
    VernonRuntimeBackend backend;
    ErrorState error;
    LiveHandleCounts live;
};

struct OpenGLContextAccess {
    void *user_data;
    MakeCurrentFn make_current;
    GetProcAddressFn get_proc_address;
};

struct OpenGLRuntimeContext final : VernonRuntimeContext {
    OpenGLContextAccess context;
    OpenGLDriver driver;
};
```

CPU, CUDA, and Vulkan contexts contain only their corresponding state. Apply
the same pattern to buffers, textures, samplers, kernels, and loaded pipelines.
Destruction checks the base tag and destroys the concrete type. Operations
validate that all supplied handles have the expected backend and context.

The base is an internal implementation detail, not a public C++ API. Avoid
`std::variant` for opaque handles because it still forces every translation
unit owning that variant to see every backend payload. Avoid virtual methods
because dispatch is already determined by the public backend tag.

`OpenGLContextAccess` is normalized from the public context descriptor at the
C API boundary. It deliberately has no `external`, `owned`, `GLFW`, or
`Vernon` discriminator: both the Python GLFW context owner and Vernon Engine
provide the same AHI contract, and their lifetimes remain outside the AHI.

### AHI invariants

- Runtime child handles retain their current context ownership rules.
- A context cannot be destroyed while children remain live.
- Imported OpenGL resources are never deleted by Runtime.
- Every OpenGL operation makes the associated context current first.
- Runtime-only builds do not link GLFW, compiler, LLVM, or LLD.
- CUDA and Vulkan loaders remain dynamically resolved.
- Pipeline schema 2 and invocation ABI version 3 remain unchanged.

## Refactor sequence

### Phase 0: freeze behavior

Before moving code:

- record compiler artifact SHA-256 values and reflection JSON for all enabled
  targets;
- run compiler C API and compile-surface parity tests;
- run runtime C API, common API, CPU pipeline, OpenGL context-access, planner, and
  available Vulkan/CUDA tests;
- verify a runtime-only build with `VERNON_ENABLE_COMPILER=OFF`;
- run the Python suite and standalone OpenGL example smoke test.

### Phase 1: extract backend-neutral code

Move compiler reflection/artifact logic and runtime bundle/metadata/tensor
logic into their target modules without changing behavior or data structures.

This phase establishes dependency direction before backend code moves.

### Phase 2: move runtime backend operations

Move operations one backend at a time while retaining the existing mixed
runtime structs:

1. OpenGL, because context-access behavior is already focused and tested;
2. CPU;
3. CUDA;
4. Vulkan.

After each move, `VernonRuntime.cpp` should delegate to backend functions but
may still own the old state declarations temporarily.

### Phase 3: make the runtime C API shell thin

Consolidate common argument validation and leave one explicit backend switch
per public operation. Remove backend headers and helper implementations from
`VernonRuntime.cpp`.

### Phase 4: split runtime state

Replace mixed state one handle family at a time:

1. runtime context;
2. buffer;
3. texture and sampler;
4. kernel;
5. loaded pipeline and backend pipeline state.

Run all runtime tests after each family. Do not convert every handle in one
change.

### Phase 5: remove duplicated runtime mechanics

Only after extraction is behavior-neutral:

- share OpenGL and GLES compute dispatch mechanics where their contracts match;
- centralize Vulkan image-layout transitions;
- consolidate tensor packing and backend compatibility checks;
- remove obsolete mixed-state fields and helpers.

### Phase 6: decompose compiler backends

Move compiler logic in this order:

1. reflection and artifact helpers;
2. SPIRV-Cross adapters;
3. common SPIR-V production;
4. CUDA;
5. CPU/JIT/LLD;
6. reduce `VernonCompiler.cpp` to its C API shell.

CPU moves last because it combines the most distinct responsibilities.

### Phase 7: build and packaging cleanup

Physically and logically separate the two distributable libraries:

```text
VernonDSL/
  source/
    lib/
      runtime/
        CMakeLists.txt
        source and private headers
      compiler/
        CMakeLists.txt
        source and private headers
    include/
    tests/
    CMakeLists.txt
  python/
  CMakeLists.txt
```

The wheel keeps the Runtime source package self-contained:

```text
vernon_dsl/runtime_src/
    CMakeLists.txt
    include/
    lib/runtime/
    cmake/
```

The directories are target ownership boundaries, not cosmetic source
grouping:

- `source/lib/runtime/CMakeLists.txt` always builds `VernonRuntime` and exports
  `Vernon::Runtime`. Entering this subproject must not require
  `VERNON_ENABLE_RUNTIME`.
- `source/lib/compiler/CMakeLists.txt` always builds `VernonDSLCompiler`.
  Entering this subproject must not require `VERNON_ENABLE_COMPILER`.
- The repository root is only a composition build for compiler, runtime,
  Python bindings, tools, and tests. Root-only `VERNON_BUILD_RUNTIME` and
  `VERNON_BUILD_COMPILER` options may exist for developer builds, but they are
  not part of the Runtime consumer interface.
- CMake inclusion is strictly top-down: the repository root adds only
  `source/`; `source/CMakeLists.txt` adds the selected library subprojects and
  owns cross-library Python/test composition. Library subprojects never add
  `source/` back as a subdirectory.
- Vernon Engine consumes only `source/lib/runtime/` through `add_subdirectory`,
  `FetchContent` with `SOURCE_SUBDIR`, or an installed `Vernon::Runtime`
  package. Its exposed choices are backend capabilities such as
  `VERNON_ENABLE_CUDA_RUNTIME` and `VERNON_ENABLE_VULKAN_RUNTIME`, not whether
  unrelated compiler components should be built.
- Runtime must be self-contained and have no MLIR, LLVM, LLD, nanobind,
  Compiler, or GLFW dependency. Public common C ABI headers required by Runtime
  must be included in its source/install package.
- Compiler and Runtime remain separate shared libraries. The Python native
  module is the intentional top-level composition point that may link both.
- Package the canonical `source/lib/runtime/` project and its sources directly.
  Do not maintain a second implementation tree that can drift from desktop
  Runtime tests.
- Add optional CUDA and Vulkan Runtime implementation files with conditional
  target sources. Disabled Runtime backends must not compile their driver
  translation units. SPIRV-Cross is an unconditional Compiler dependency
  because OpenGL, OpenGL ES, and Metal are part of the Compiler contract.
- Keep shared internal implementation objects coarse-grained; do not create
  one static library per source file.
- Stop compiling Runtime implementation `.cpp` files directly into test
  executables. Tests must exercise the same objects linked into Runtime.
- Keep internal test hooks private and compile them only when `BUILD_TESTING`
  is enabled.

Verification:

- Build Runtime directly from `source/lib/runtime/` with CPU/OpenGL only, CUDA
  enabled, and Vulkan enabled configurations as available.
- Build Compiler directly from `source/lib/compiler/` without building Runtime.
- Build Vernon Engine against only the Runtime subproject/package without
  setting compiler/runtime enable options.
- Build and install the wheel with `BUILD_TESTING=OFF`.
- Extract the wheel's Runtime source package, build it independently, and
  verify that an external smoke project can link `Vernon::Runtime`.

## Verification gates

Every phase must satisfy:

1. Public headers and ABI version constants are unchanged.
2. Compiler artifact names, bytes, reflection, and hashes are unchanged.
3. Compiler-only and runtime-only configurations still build.
4. Runtime tests do not silently compile duplicate implementation sources.
5. Compiler and runtime CTest suites pass for enabled backends.
6. Python native and pipeline tests pass.
7. GLFW-owned OpenGL and Vernon-owned OpenGL pass through the same AHI path.
8. Wheel installation and runtime source distribution smoke tests pass.

Vulkan tests that compile assets through Python are integration tests, not
runtime-only proof. Runtime-only verification must use prebuilt fixtures or
exclude those cases.

## Completion criteria

- `VernonCompiler.cpp` contains C API/lifetime code, accessors, and one target
  dispatch, but no backend pass pipeline or backend object management.
- `VernonRuntime.cpp` contains C API/lifetime code, accessors, and one backend
  dispatch, but no GL, Vulkan, CUDA, or CPU implementation.
- Common runtime context and handle bases contain no GL, Vulkan, CUDA, LLVM, or
  LLD payloads.
- Backend modules do not include one another.
- Runtime-only builds remain compiler- and GLFW-free.
- All verification gates pass without changing public contracts.

