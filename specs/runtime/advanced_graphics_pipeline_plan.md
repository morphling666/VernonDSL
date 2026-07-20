# Advanced Graphics Pipeline Plan

Status: implemented for Vulkan and external-context OpenGL/OpenGL ES pipeline
bundles.

## Goal

Extend the direct Tensor-first pipeline API with instancing, indexed drawing,
interactive shader variants, and multiple render targets:

```python
render = vd.pipeline(
    animate,
    vertex_main,
    fragment_main,
    features={"SKIN", "PICKING"},
)

render(
    positions=positions,
    instances=instances,
    indices=indices,
    targets={
        "color": color_target,
        "object_id": id_target,
    },
)
```

The API must remain declarative. Shader annotations select vertex, instance,
uniform, and resource behavior; callers provide Tensors and Textures without
constructing device, encoder, descriptor, or draw objects.

## Pipeline contract

### Construction

Add keyword options after the stage list:

```python
vd.pipeline(
    *stages,
    features: Iterable[str] = (),
)
```

- Canonicalize features as a sorted unique tuple.
- Reject undeclared features and illegal stage orders before native
  compilation.
- Include the canonical feature set, dependency hashes, target, target
  options, stage entries, and merged external interface in the cache key.
- Compile each stage with the same feature set and validate the specialized
  vertex-to-fragment interface.

### Invocation

Reserve these host-only names:

- `indices`: optional index Tensor;
- `target`: compatibility spelling for one fragment output;
- `targets`: named mapping for one or more fragment outputs.

All other names continue to route through the merged reflected stage
interface. Supplying both `target` and `targets` is an error.

## Instancing

The existing annotation remains the source of truth:

```python
transform: Annotated[vd.mat4[vd.f32], vd.instance(divisor=1)]
```

Implementation requirements:

- infer one common instance count from all instance Tensors;
- validate leading dimensions, trailing value shapes, locations, and divisors;
- reject conflicting instance counts;
- issue non-indexed or indexed instanced draws as appropriate;
- preserve divisor metadata in compiler reflection and native binding records;
- support divisor `1` first; gate larger divisors on the relevant backend
  capability.

OpenGL uses vertex attribute divisors. Vulkan bakes vertex/instance input rates
into the cached graphics-pipeline vertex layout.

## Index buffers

`indices` remains a normal contiguous Tensor. The MVP accepts:

```python
indices = vd.Tensor.from_numpy(np.asarray([...], dtype=np.uint32))
```

Rules:

- rank must be one;
- dtype must be `u32` initially;
- draw count is the Tensor length;
- topology remains triangle list, so the count must be divisible by three;
- an index Tensor is internal draw state and never appears as a shader
  parameter;
- cache the native index allocation through normal Tensor residency.

`VernonPipelineInvocation` carries an optional index binding containing the
buffer, element type, byte offset, and index count. OpenGL uses
`glDrawElementsInstanced`; Vulkan uses `vkCmdBindIndexBuffer` and
`vkCmdDrawIndexed`.

## Interactive shader variants

Reuse the existing module graph and AOT feature semantics:

- `feature("NAME")`;
- `When[FEATURE, T]`;
- `if FEATURE` and `if not FEATURE`;
- stable interface locations reserved before specialization.

Interactive `vd.pipeline(..., features=...)` must call the same
`compile_file(..., features=..., entry=...)` path as the cooker. No implicit
fallback is allowed when a variant is absent or invalid.

The native program cache key includes:

- canonical feature tuple;
- specialized source and dependency hashes;
- merged specialized interface;
- target and target options;
- generated artifact hashes.

AOT `shader-pipeline` manifests continue to enumerate permitted feature keys.
Interactive and cooked variants must produce equivalent canonical keys and
interface validation diagnostics.

## Host/device shared definitions

Keep device-only behavior as the default and add an explicit shared option:

```python
@vd.struct(shared=True)
class Light:
    position: vd.vec3[vd.f32]
    intensity: vd.f32

    @vd.func(shared=True)
    def contribution(self, point: vd.vec3[vd.f32]) -> vd.f32:
        distance = vd.norm(self.position - point)
        return self.intensity / (distance * distance)
```

Semantics:

- `@vd.func` and `@vd.struct` remain device-only by default;
- `shared=True` means the same source definition can execute or instantiate on
  host Python and can also compile into shader code;
- shared structs use value semantics and provide a generated host constructor
  for their annotated fields;
- shared functions execute their original Python body on the host;
- DSL scalar/vector/matrix constructors and pure numeric intrinsics dispatch
  to Python/NumPy equivalents during host execution;
- shared functions may call other shared functions, but may not call
  device-only helpers when executing on the host;
- struct bodies may contain annotated fields and `@vd.func` methods;
- a method lowers to a namespaced private function with the struct value as an
  explicit first argument, so `light.contribution(point)` becomes equivalent
  to `Light__contribution(light, point)` in device IR;
- methods inherit normal function-domain rules: `@vd.func` is device-only and
  `@vd.func(shared=True)` is callable on both sides;
- a shared method requires a shared struct; a shared struct may also contain
  device-only methods;
- `self` has immutable value semantics in device code; field mutation,
  inheritance, virtual dispatch, properties, and dynamic method replacement
  are not supported;
- methods may call other methods on `self`, subject to the normal recursion and
  domain checks;
- resources, builtins, texture operations, atomics, and shader-stage entries
  remain device-only;
- recursion remains forbidden in device compilation.

“Shared” refers to source and behavior, not a guaranteed packed binary ABI.
Transferable heterogeneous struct buffers require a separate layout and
serialization design.

## Multiple render targets

### Shader declaration

Use a named `@vd.struct` fragment result:

```python
@vd.struct
class GBuffer:
    color: Annotated[vd.vec4[vd.f32], vd.location(0)]
    normal: Annotated[vd.vec4[vd.f32], vd.location(1)]
    object_id: Annotated[vd.vec4[vd.f32], vd.location(2)]


@vd.fragment
def fragment_main(...) -> GBuffer:
    return GBuffer(color, normal, object_id)
```

Extend struct-field reflection so stage-output field names, types, and
locations survive Python lowering, MLIR, SPIR-V lowering, and artifact
reflection. Flatten the fragment result into one native output per field.

### Target routing

`targets` maps reflected output field names to Textures:

```python
targets={
    "color": color_texture,
    "normal": normal_texture,
    "object_id": id_texture,
}
```

Validation:

- keys must exactly match enabled fragment outputs;
- every Texture must belong to the active runtime generation;
- all targets must have equal width and height;
- viewport and scissor are inferred from that shared extent;
- MVP targets are RGBA8; typed formats are a later extension.

For a single unnamed or scalar fragment result, retain
`target=texture`. Internally normalize it to one location-zero target.

## Native graphics ABI

Bump `graphics_draw_abi_version` to 2 and extend the additive draw description
with:

- optional index binding;
- an array of `(output location, Texture)` color attachments;
- attachment count.

Version 1 callers with one `target` remain valid. Version 2 normalizes the
legacy field into attachment location zero.

Backend work:

- OpenGL: configure `glDrawBuffers`, attach every Texture to
  `GL_COLOR_ATTACHMENT0 + location`, validate framebuffer completeness, and
  select array or indexed instanced drawing.
- Vulkan: create render-pass/framebuffer or dynamic-rendering state for the
  reflected attachment locations, include attachment formats/count in the
  graphics pipeline cache key, and select indexed/non-indexed commands.
- OpenGL ES consumes GLES-profile bundles through a host-owned external
  context and never routes through desktop OpenGL.

## Ordering and residency

Submission remains:

```text
upload host-dirty Tensors
-> optional compute dispatch
-> compute-to-graphics barrier
-> bind vertex, instance, and optional index buffers
-> bind variant resources and targets
-> indexed or non-indexed instanced draw
-> mark all targets device-dirty
```

Changing one uniform, instance Tensor, index Tensor, or target must not
reallocate or upload unrelated resident resources. Variant changes select a
different compiled program but do not invalidate compatible Tensor or Texture
allocations.

## Verification

### Frontend and composition

- valid and invalid feature sets;
- feature-specialized stage interfaces;
- deterministic variant cache keys;
- missing, extra, or incompatible merged parameters;
- named struct output reflection and location overlap diagnostics.

### Shared definitions

- shared scalar, vector, matrix, and struct functions produce matching host
  and CPU-reference results;
- shared-to-shared calls and imported shared helpers;
- host construction and field access for shared structs;
- device-only and shared struct methods, including method-to-method calls;
- rejection of mutable `self`, inheritance, recursive methods, and invalid
  shared-method/device-struct combinations;
- host calls to device-only definitions remain errors;
- shared functions reject device-only operations with precise diagnostics.

### Instancing

- one triangle rendered by multiple instances;
- conflicting instance counts;
- invalid divisors and trailing shapes;
- indexed and non-indexed instanced paths.

### Index buffers

- indexed quad rendered from four vertices and six `u32` indices;
- invalid rank, dtype, range, and triangle-list count;
- repeated draws reuse the resident index allocation.

### Multiple render targets

- fragment shader writes distinct values to at least two RGBA8 targets;
- named target routing is independent of mapping iteration order;
- missing/extra names and mismatched extents are rejected;
- every output becomes device-dirty and reads back correctly.

### Backends and assets

- OpenGL graphics-only and compute-to-graphics pipelines;
- Vulkan compute/graphics pipeline bundles;
- host-context OpenGL ES with matching GLSL ES artifacts;
- interactive and cooked variants produce equivalent stage selection;
- complete Release CTest and Python suites.

## Design records

Record these decisions in the relevant design documents:

- index buffers are Tensors but are reserved draw state, not shader arguments;
- named MRT routing is based on reflected fragment result fields;
- feature specialization precedes merged-interface validation;
- Vulkan graphics pipeline keys include vertex layout, index type, instance
  rates, target formats/count, and feature key.
