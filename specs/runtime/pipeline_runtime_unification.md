# Pipeline Runtime Unification

## Decision

Treat a cooked pipeline asset as a function-like, context-bound executable,
but do not persist it as a literal native C function.

The public execution surface should be one stable C ABI call over a loaded
pipeline handle:

```c
typedef struct VernonPipelineBundle VernonPipelineBundle;
typedef struct VernonLoadedPipeline VernonLoadedPipeline;

typedef struct VernonFeatureSetView {
  const char *const *names;
  size_t count;
} VernonFeatureSetView;

VernonPipelineBundle *vernonRuntimeLoadPipelineBundle(
    VernonRuntimeContext *context,
    const void *bundle,
    size_t bundle_size);

VernonLoadedPipeline *vernonRuntimeResolvePipeline(
    VernonPipelineBundle *bundle,
    VernonFeatureSetView features);

VernonStatus vernonRuntimePipelineInvoke(
    VernonLoadedPipeline *pipeline,
    const VernonPipelineInvocation *invocation);
```

Reflection remains in the cooked asset for validation, tooling, and
diagnostics. Only the shared C++ runtime parses it; Vernon, Python, and pure-C
callers should not implement separate binding planners.

## Current architecture

Native execution is unified in `source/lib/VernonRuntime.cpp` and the
`Vernon::Runtime` target:

```text
Python DSL + manifests
        |
        v
Vernon compiler
        |
        +-- GLSL / MSL / SPIR-V
        +-- reflection JSON
        +-- variant mapping
        |
        v
shader.json + stages/*
        |
        +-- Python/Vernon/C callers load VernonPipelineBundle
        |
        +-- VernonRuntime validates reflection and executes the selected backend
```

The native runtime executes the maintained operations:

- `vernonRuntimeLaunch` dispatches compute;
- `vernonRuntimePipelineInvoke` submits compute/graphics pipeline bundles;
- `vernonRuntimeLoadComputeBundle` loads cooked compute assets.

Python exposes the same runtime through the single `_native` module. OpenGL and
OpenGL ES require a host-owned external context; Vulkan and CUDA resolve their
system drivers dynamically.

### Concrete gaps

1. There is no native graphics or composed-pipeline bundle loader equivalent
   to `vernonRuntimeLoadComputeBundle`.
2. Pipeline cooker schema version 2 and the legacy CLI shader bundle schema
   version 1 coexist.
3. Vernon parses variant stage records but does not retain their reflected
   argument interfaces for binding.
4. Python owns graphics argument merging, Tensor layout expansion, residency,
   MRT routing, index validation, and compute-to-graphics orchestration.
5. Vernon production render passes still use legacy named uniforms and do not
   use cooked DSL assets as their normal draw path.
6. Vernon and VernonRuntime currently own different graphics contexts and
   resource handle types.

## Target architecture

```text
Pipeline bundle
  - backend artifacts
  - exact variant map
  - merged parameter ABI
  - binding plan
  - capability requirements
  - tooling reflection
          |
          v
C++ PipelineRuntime
  - bundle validation
  - variant resolution
  - backend program creation
  - argument validation
  - binding planning
  - draw/dispatch orchestration
          |
          v
LoadedPipeline
          |
          +-- Vernon adapter
          +-- Python/nanobind adapter
          +-- pure C caller
```

The runtime-facing object has four distinct forms:

1. **Cooked pipeline bundle**: immutable, portable data without live GPU
   handles.
2. **Pipeline bundle handle**: parsed metadata, target artifacts, and exact
   variant map associated with a runtime context.
3. **Loaded pipeline**: backend- and context-bound executable state for one
   exact feature key, including linked programs and cached binding locations.
4. **Invocation packet**: per-call resources and dynamic draw/dispatch state.

## Unified compute and graphics execution

Compute and graphics share the same parameter ABI, resource views, pipeline
handle, and invocation entry point. A loaded pipeline is represented as an
ordered sequence of platform-independent execution steps:

```cpp
struct PipelineExecutable {
  std::vector<PipelineStep> steps{
      Dispatch{compute_entry},
      Barrier{compute_write, vertex_read},
      DrawIndexed{graphics_program},
  };
};
```

The common planner is responsible for:

- Tensor, Texture, scalar, shape, layout, and access validation;
- stable slot resolution;
- exact feature variant selection;
- compute grid inference;
- vertex, index, instance, topology, and MRT validation;
- stage ordering and abstract resource hazards;
- backend capability validation.

The resulting `PlannedInvocation` no longer contains Python conventions,
source parameter names, generated GLSL identifiers, or engine-specific scene
objects.

## Backend boundary and platform differences

Platform differences are hidden below `PlannedInvocation`, at the
`BackendDevice` boundary:

```cpp
class BackendDevice {
public:
  virtual Result<BufferHandle> createBuffer(const BufferDescriptor &) = 0;
  virtual Result<TextureHandle> createTexture(const TextureDescriptor &) = 0;

  virtual Result<PipelineHandle>
  createPipeline(const PipelineArtifacts &, const PipelineLayout &) = 0;

  virtual Result<Unit>
  invoke(PipelineHandle, const PlannedInvocation &) = 0;
};
```

The common invocation model does not eliminate backend artifacts or backend
commands. It makes those differences invisible to callers.

### Pipeline creation

- Vulkan loads SPIR-V and creates descriptor layouts and compute/graphics
  pipelines.
- OpenGL and OpenGL ES compile/link GLSL programs and create any required
  VAO/FBO state.
- Metal compiles/loads MSL libraries and creates compute/render pipeline
  states.
- CUDA loads PTX modules and resolves kernel functions.
- CPU loads AOT native-library entry points. Python `@kernel` execution uses
  the same validated native compute-bundle path; AST processing ends after
  Vernon MLIR generation and never executes the kernel.

### Resource binding

- Vulkan writes descriptor sets or descriptor buffers.
- OpenGL/GLES resolves uniform, SSBO, texture-unit, vertex, and framebuffer
  bindings.
- Metal binds buffers, textures, samplers, and argument buffers.
- CUDA packs kernel argument pointers and values.
- CPU builds the native argument/result memory layout.

### Submission and synchronization

- Vulkan records dispatch/draw commands and pipeline/image barriers.
- OpenGL/GLES issue dispatch/draw calls and memory barriers.
- Metal records compute/render command encoders and resource fences.
- CUDA launches kernels on streams and uses events where required.
- CPU invokes a function serially and applies host memory ordering. A future
  multithreaded CPU executor may parallelize the invocation grid without
  changing the pipeline or CPU entry-point ABI.

### Capability rules

Capabilities belong to the loaded backend, not to the DSL type system:

```cpp
struct BackendCapabilities {
  bool compute;
  bool graphics;
  bool compute_graphics_pipeline;
  bool external_memory_interop;
};
```

Examples:

- OpenGL 3.3 supports graphics but not composed compute+graphics.
- OpenGL 4.3 can support compute and graphics composition.
- OpenGL ES support depends on the requested version and implementation.
- Vulkan can support the complete compute+graphics model.
- Metal can support the complete compute+graphics model once its runtime
  backend is implemented.
- CUDA supports compute only. A pipeline containing a draw step is rejected
  unless a future explicit CUDA/graphics external-memory interop path is used.

Pipeline bundles are canonical JSON with target-specific stage artifacts:

```text
pipeline.bundle
  target
  variants[]
  stage_artifacts{}
    OpenGL/OpenGL ES: GLSL source
    Vulkan: base64 SPIR-V plus SHA-256
    CUDA: PTX source plus compute reflection
    CPU: relative host-native library sidecar plus SHA-256 and reflection
```

CUDA uses the unified pipeline bundle and invocation ABI for dispatch-only
pipelines. CUDA bundle loading rejects draw and barrier steps, then loads PTX
plus reflection through the normal compute artifact path. CPU uses that same
dispatch-only pipeline model, rejecting draw and barrier steps. Because its
native library is a path-relative sidecar rather than embedded bytes, CPU
pipeline loading requires the bundle directory. The runtime rejects rooted,
parent-traversing, and canonical-escape paths and validates host OS,
architecture, CPU invocation ABI, file size, and SHA-256 before opening the
library. Loading validates that the artifact family matches the selected
runtime backend.

## Pipeline invocation ABI

The pipeline invocation ABI is versioned independently of the compute launch
ABI.

```c
typedef struct VernonDeviceSampler VernonDeviceSampler;

typedef enum VernonDataType {
  VERNON_DATA_BOOL,
  VERNON_DATA_I32,
  VERNON_DATA_U32,
  VERNON_DATA_F16,
  VERNON_DATA_F32,
  VERNON_DATA_F64
} VernonDataType;

typedef enum VernonValueAccess {
  VERNON_ACCESS_READ,
  VERNON_ACCESS_WRITE,
  VERNON_ACCESS_READ_WRITE
} VernonValueAccess;

typedef struct VernonTensorView {
  VernonDeviceBuffer *buffer;
  VernonDataType dtype;
  VernonValueAccess access;
  uint32_t rank;
  const uint64_t *shape;
  const uint64_t *byte_strides;
  size_t byte_offset;
} VernonTensorView;

typedef struct VernonTextureView {
  VernonDeviceTexture *texture;
  VernonTextureFormat format;
  VernonValueAccess access;
  uint32_t dimension;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
} VernonTextureView;

typedef struct VernonInlineValue {
  VernonDataType dtype;
  uint32_t rank;
  const uint64_t *shape;
  const void *data;
  size_t data_size;
} VernonInlineValue;

typedef enum VernonPipelineArgumentKind {
  VERNON_PIPELINE_TENSOR,
  VERNON_PIPELINE_TEXTURE,
  VERNON_PIPELINE_SAMPLER,
  VERNON_PIPELINE_INLINE_VALUE
} VernonPipelineArgumentKind;

typedef struct VernonPipelineArgument {
  uint32_t slot;
  VernonPipelineArgumentKind kind;
  union {
    VernonTensorView tensor;
    VernonTextureView texture;
    VernonDeviceSampler *sampler;
    VernonInlineValue inline_value;
  };
} VernonPipelineArgument;

typedef struct VernonPipelineInvocation {
  uint32_t struct_size;
  uint32_t abi_version;

  const VernonPipelineArgument *arguments;
  size_t argument_count;

  const VernonIndexBinding *index_binding;
  const VernonColorAttachment *color_attachments;
  size_t color_attachment_count;

  VernonPrimitiveTopology topology;
  uint32_t vertex_count;
  uint32_t instance_count;
  VernonLaunchSize compute_grid;

  uint32_t viewport[4];
  uint32_t scissor[4];
} VernonPipelineInvocation;
```

The exact structure may evolve, but it should use additive `struct_size`
versioning and stable numeric slot IDs. Generated GLSL names must not be part
of the public invocation ABI.

### Runtime value and type ABI

The DSL, asset, and invocation layers reuse one logical type system:

```text
DSL type
  Tensor[f32, (N, 3)]
  Texture["2d", f32]
  vec4[f32]
        |
        v
asset parameter constraint
  kind / dtype / rank / shape / access / interface
        |
        v
invoke-time value
  VernonTensorView / VernonTextureView / VernonInlineValue
```

The asset stores type constraints; it never stores a concrete Python or C++
object layout. Invocation supplies context-bound resources and concrete
layouts.

`VernonTensorView` preserves shape, byte strides, and byte offset, so the same
ABI represents contiguous Tensors, interleaved vertex data, slices, and
runtime swizzles. Vectors, matrices, and primitive scalar values use
`VernonInlineValue` and do not require a device-buffer allocation.

Python `Tensor`/`TensorView` and Vernon C++ Tensor wrappers convert to borrowed
`VernonTensorView` values. Python and Vernon Texture wrappers similarly convert
to `VernonTextureView`. The C ABI does not expose Python object layouts, C++
templates, STL containers, or engine resource classes.

Invocation value rules:

- resource and shape pointers remain valid for the duration of the call;
- views do not transfer ownership;
- all resources belong to, or are imported into, the pipeline's device;
- runtime validation checks kind, dtype, rank, shape, access, context, and
  required alignment before backend binding;
- writable resources are marked device-dirty after successful execution;
- source names are used only to map ergonomic Python/C++ wrappers to stable
  numeric slots.

C++ can provide typed convenience conversion while retaining the C ABI:

```cpp
pipeline.invoke({
    PipelineArgument::tensor(position_slot, positions.view()),
    PipelineArgument::texture(albedo_slot, albedo.view()),
    PipelineArgument::value(roughness_slot, 0.5f),
});
```

### Stable slots

The cooker assigns each externally supplied parameter a stable slot. A slot
record contains:

- source name for diagnostics and generated wrappers;
- stages that consume it;
- interface kind;
- scalar dtype and value shape;
- read/write access;
- vertex location and instance divisor where applicable;
- resource set and binding where applicable;
- static or dynamic dimensions;
- optional/default status.

The runtime resolves slots to backend locations once when loading the
pipeline. Per-frame invocation passes slot/value pairs and does not parse JSON
or call `glGetUniformLocation` by source name.

## Asset contents

A unified pipeline bundle should contain:

- `pipeline_bundle_schema_version`;
- `invocation_abi_version`;
- pipeline ID and content hash;
- source dependency hashes;
- exact feature variant keys;
- variant-to-stage artifact mapping;
- merged external parameter table;
- host-only parameters such as indices, topology, targets, and compute grid;
- stage ordering and required barriers;
- backend and capability requirements;
- backend artifacts;
- optional readable reflection for tools.

The existing stage-level reflection remains useful, but runtime execution
should consume a normalized pipeline-level binding schema produced by the
cooker.

## Responsibility boundaries

### Cooker and compiler

- validate shader stages and cross-stage locations;
- specialize feature variants;
- emit backend artifacts;
- merge stage interfaces into a pipeline signature;
- assign stable slots;
- emit binding and capability metadata;
- content-address artifacts and dependencies.

### Shared C++ PipelineRuntime

- parse and validate the bundle once;
- resolve an exact variant;
- create backend programs and pipelines;
- cache concrete binding locations;
- validate concrete argument layouts and counts;
- bind buffers, uniforms, textures, samplers, indices, and render targets;
- submit dispatch, barriers, and draw operations;
- expose errors through the stable C API.

### Vernon adapter

- map engine buffers, textures, materials, framebuffers, and render-graph
  state to runtime resource handles and slots;
- retain scene/render-pass scheduling;
- provide dynamic state such as viewport, scissor, depth, and blend overrides;
- avoid parsing shader reflection.

### Python adapter

- map keyword names to precomputed slots;
- map Tensor, TensorView, Texture, and scalar objects to runtime arguments;
- retain Python/NumPy host ergonomics;
- avoid implementing backend binding rules.

## Why the asset should not be a literal C function

GPU execution cannot be represented by portable machine code alone. A loaded
pipeline depends on:

- a live graphics or compute context;
- driver-created program and pipeline objects;
- dynamic resource handles;
- render targets and framebuffer state;
- synchronization and queue ownership;
- backend capabilities and driver versions.

A native function embedded in an asset would also require per-OS and per-CPU
artifacts, relocation, code signing, and strict compiler ABI compatibility.
OpenGL programs and Vulkan pipeline caches are not portable substitutes.

The correct equivalent of a function is therefore:

```text
LoadedPipeline handle + stable invocation ABI + vernonRuntimePipelineInvoke()
```

## Optional generated C wrappers

The cooker may generate a typed header for applications that want a
function-like API:

```c
typedef struct MeshPipelineArguments {
  VernonDeviceBuffer *position;
  VernonDeviceBuffer *instance_offset;
  VernonDeviceBuffer *indices;
  VernonDeviceTexture *color;
  VernonDeviceTexture *object_id;
  uint32_t instance_count;
} MeshPipelineArguments;

static inline VernonStatus meshPipelineDraw(
    VernonLoadedPipeline *pipeline,
    const MeshPipelineArguments *arguments);
```

The wrapper only constructs a `VernonPipelineInvocation` with generated slot
IDs and calls `vernonRuntimePipelineInvoke`. It is not the stored executable
asset and does not contain backend binding logic.

## Runtime library ownership

The OpenGL, OpenGL ES, Vulkan, Metal, CUDA, and CPU implementations should not
move directly into Vernon Engine scene/render modules. Doing so would make the
standalone Python runtime, pure-C applications, compiler integration tests,
and compute-only deployments depend on the complete engine.

They should be extracted into independently linkable native targets:

```text
VernonRuntimeCore
  - PipelineLayout
  - VernonValue
  - InvocationPlanner
  - LoadedPipeline
  - residency state
  - stable C ABI

VernonBackendOpenGL
VernonBackendVulkan
VernonBackendMetal
VernonBackendCUDA
VernonBackendCPU

Vernon Engine
  - RenderGraph
  - Scene
  - Material
  - AssetService
  - runtime/device adapter
```

The physical sources may live in the Vernon repository if the projects are
consolidated, but the runtime targets must not depend on Scene, Material,
Editor, RenderGraph, or other engine application layers.

The dependency direction is:

```text
Python/nanobind ---+
pure C/C++ --------+--> VernonRuntimeCore --> backend implementations
Vernon Engine -----+
```

CUDA remains a compute backend and must not depend on the engine render
subsystem. OpenGL and OpenGL ES may share an implementation family with
different capability profiles.

Long term, Vernon `GraphicsRuntime` should wrap or own a shared
`BackendDevice`, rather than retaining a second OpenGL/Vulkan implementation.
This removes duplicate resource, binding, and draw code while preserving
Vernon's higher-level render graph.

## Graphics context and resource ownership

This is the main integration blocker.

Vernon owns its production graphics context, graphics thread, texture storage,
and render graph. `VernonRuntime` consumes that current context through host
callbacks and imports Vernon buffer/texture handles; it never creates a second
context.

The shared runtime needs one of these integration models:

1. **External device adapter**: Vernon supplies a backend callback table for
   program creation, resource binding, draw, dispatch, and synchronization.
2. **External context mode**: VernonRuntime operates on the current Vernon
   graphics context and imports Vernon resource handles.
3. **Shared GraphicsDevice abstraction**: both projects use one native device
   interface and one resource-handle model.

The third option gives the cleanest long-term ownership model. The first is
the least invasive migration path.

The runtime should support two creation modes:

```cpp
// Standalone Python or pure C: the runtime creates and owns the device.
vernonRuntimeCreate(VERNON_RUNTIME_VULKAN, device_index);

// Vernon Engine: the runtime uses the engine's existing device/context.
vernonRuntimeCreateExternalDevice(&vernon_device_adapter);
```

External-device mode must preserve Vernon's graphics thread, swapchain,
surface, queue, and render-graph ownership. Runtime resource views reference
the same underlying buffers and textures; they must not silently copy them
into a hidden secondary context.

## Render graph boundary

The pipeline runtime should own shader-specific execution:

- program/pipeline binding;
- parameter and resource binding;
- vertex/index/instance configuration;
- MRT routing;
- draw or dispatch submission;
- pipeline-local barriers.

Vernon should continue to own:

- render-pass ordering;
- scene traversal and batching;
- transient resource allocation;
- framebuffer lifetime;
- depth, blend, and raster policy when controlled by the render graph;
- cross-pass synchronization.

This prevents a pipeline asset from becoming an opaque mini render engine.

## Migration plan

### Phase 1: Native graphics reflection and planner — implemented

- define the pipeline-level parameter schema and invocation ABI;
- implement its parser in VernonRuntime;
- move matrix expansion, vertex binding, uniform packing, MRT routing, index
  validation, and interface merging out of Python;
- make Python call the native planner.

### Phase 2: Unified bundle format and loader — implemented

- converge legacy shader schema version 1 and pipeline schema version 2;
- add `vernonRuntimeLoadPipelineBundle`;
- retain parsed reflection and variant mappings on `VernonPipelineBundle`;
- add exact feature-key resolution to cached `VernonLoadedPipeline` handles;
- make JIT compilation construct the same in-memory bundle representation as
  AOT cooking.

### Phase 3: Shared backend drivers — implemented

- keep engine-independent OpenGL/GLES, Vulkan, CUDA, platform loading, and
  hashing helpers under `source/lib/runtime`;
- expose backend capability queries and common runtime resource handles;
- resolve CUDA and Vulkan drivers dynamically and consume host OpenGL/GLES
  callbacks;
- preserve explicit errors for unsupported stage combinations.

### Phase 4: Vernon device adapter — remaining integration

- let the runtime use Vernon graphics context and resources;
- make `ShaderProvider` return a loaded pipeline asset rather than only a
  linked `Shader`;
- replace named material bindings with slot-based invocation;
- integrate loaded variants into production render passes.

### Phase 5: Residency and typed wrappers — remaining integration

- move optional dirty-generation tracking into the shared native runtime;
- generate C/C++ argument wrappers from pipeline signatures;
- remove obsolete Python and Vernon binding paths.

## Compatibility rules

- compute launch ABI and pipeline invocation ABI are versioned independently;
- all public structs use additive `struct_size` extension;
- bundle loaders reject unsupported ABI versions before creating GPU objects;
- exact feature variant matching remains deterministic;
- tooling reflection may grow independently of the compact invocation schema;
- source names remain diagnostic metadata, not backend binding keys;
- dynamic dimensions require either load-time specialization or explicit
  invocation constraints.

## Prior art and intended extension

This architecture combines established ideas rather than assuming each
component is novel.

### Taichi AOT and TiRT

Taichi demonstrates the deployment model:

```text
Python kernels
  -> architecture-specific AOT module (.tcm)
  -> C runtime / header-only C++ wrapper
  -> load module, resolve kernel, set typed arguments, launch
```

Taichi also supports importing or exporting a Vulkan device for integration
with an existing native renderer. Its AOT modules are compiled for a selected
architecture, and backend support maturity varies. The key precedent for
Vernon is that Python authoring does not require Python in the deployed C++
application.

Vernon should follow this loader and C-ABI model while extending the asset from
compute kernels/graphs to complete compute plus raster pipelines, feature
variants, instancing, indexed draws, MRT, and render-graph integration.

### LuisaCompute

LuisaCompute demonstrates the runtime/backend model:

- C++ and Python DSL frontends;
- a unified Device, Stream, resource, and command abstraction;
- CUDA, DirectX, Metal, Vulkan, and CPU backends;
- callable/kernel compilation and persistent shader bytecode caches;
- compute and raster command support.

Its common operating model is JIT compilation with persistent backend caches.
Vernon's intended addition is an explicit, versioned, distributable pipeline
bundle and stable C invocation ABI shared by Python JIT, native AOT deployment,
and the Vernon Engine asset system.

### Vernon differentiation

The intended combined system is:

```text
Python Tensor-first DSL
+ shared host/device definitions
+ compute/vertex/fragment composition
+ one Tensor/Texture/inline-value invocation ABI
+ multi-target backend artifacts
+ AOT callable pipeline bundles
+ Vernon RenderGraph, Material, and Scene integration
```

One logical bundle may contain artifacts for several targets, but every final
artifact remains backend-specific. Cross-platform means the source signature,
asset identity, and invocation ABI remain stable while the loader selects the
artifact matching the current `BackendDevice`.

## Implementation status

The public runtime now loads versioned pipeline bundles, resolves feature
variants, validates reflected argument slots, and executes
`VernonPipelineInvocation` directly. Python uses this path through `_native`;
Vernon can consume the same `Vernon::Runtime` target and external-context
resource imports. Legacy standalone program/draw entry points are removed.
