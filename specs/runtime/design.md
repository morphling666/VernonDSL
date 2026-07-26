# Runtime design

## CPU compute bundles

Persistent CPU bundles contain a target relocatable object with stable,
module-hashed C entry wrappers. The manifest records the target triple, object
format, CPU invocation ABI, exported symbol, artifact size, and SHA-256.
Applications link the object at build time and register its wrapper with
VernonRuntime; Runtime validates the external descriptor but never parses or
relocates object files. Python immediate execution may use a host-native
shared-library descriptor after embedded LLD finalizes the object into an
ephemeral DLL/so/dylib. LLVM IR and ORC JIT are not persistent runtime bundle
formats.

## Backend loading

`VernonRuntime` owns the Win32/POSIX library loader used by CPU native-library
descriptors, CUDA, and Vulkan. CPU relocatable objects resolve through the
static entry registry. CUDA Driver and Vulkan loader symbols are resolved at
runtime; Vulkan headers are compile-only. This keeps LLVM, LLD, GLFW, CUDA
Toolkit libraries, and the Vulkan loader import library outside the deployable
runtime dependency closure.

OpenGL function resolution is isolated in `backend_opengl_driver`; unlike CUDA
and Vulkan it consumes context callbacks. OpenGL and OpenGL ES are distinct
external-context backends and only accept pipeline manifests for their matching
GLSL profile. Cooked GLSL is an external content-addressed artifact;
interactive GLSL remains inline.

Context ownership is a separate layer: `Context Owner -> external callbacks ->
AHI`. `VernonRuntime` is the GLFW-free AHI and always consumes the same
external-context contract. Vernon Engine supplies callbacks for its existing
context. The Python wheel's separate `_gl_context` extension links GLFW, owns a
hidden window/context, and keeps that owner alive until after all AHI children
and the runtime are destroyed. Runtime-only builds never discover GLFW.

The OpenGL function table, external-context callbacks, buffers, images,
samplers, shader compilation, programs, vertex arrays, and attachment
framebuffer objects are owned by VernonRHI. Runtime retains format and
invocation policy while PipelineAsset and direct compute-artifact loaders both
produce `VernonLoadedPipeline` and execute through RuntimeCore and the RHI
provider adapter.
`VernonOpenGLContextCallbacks` is the single context contract for Python and
Engine owners; backend behavior never branches on context origin.

## Vulkan graphics bundles

Cooked pipeline schema 2 stores each SPIR-V stage as a content-addressed
external `.spv` artifact. The manifest records its relative path, byte size,
and SHA-256, all of which the common artifact resolver validates before Vulkan
sees the bytes. Non-persistent interactive execution uses the same descriptor
shape with inline base64 storage because it has no durable asset directory.
Resolution creates immutable shader modules and content-keyed descriptor-set
layouts, pipeline layouts, and graphics pipelines. Graphics pipeline keys
include render-pass attachment compatibility, topology, vertex input, and the
full layout key; invocation retains descriptor writes and allocation, render
passes, framebuffers, attachment views, and command recording as transient
state. RGBA8 offscreen images use optimal tiling and staging buffers for host
upload/readback. Unbound graphics uniforms use the
compiler's single push-constant block ABI; descriptor-bound uniforms remain a
later extension. Block members follow source argument order with Vulkan
scalar/vector/matrix alignment, and Runtime reproduces that layout from
reflection before checking the device's `maxPushConstantsSize`. Vertex and
fragment blocks occupy disjoint ranges; Runtime relocates the fragment
SPIR-V member offsets by the planned range base so one stage cannot overwrite
the other stage's values. A negative Vulkan viewport height preserves the
OpenGL clip-space and host-readback Y convention. One-mip RGBA8 Cube uploads
store tightly packed faces in `+X, -X, +Y, -Y, +Z, -Z` order.

## Python native module

Python exposes compiler services and all runtime backends through one `_native`
module. The module links the compiler DLL and `VernonRuntime`; the runtime DLL
itself retains its dependency boundary. Interactive GPU pipelines serialize
pipeline schema 2 with inline artifact descriptors; the cooker emits the same
schema with external descriptors only. CPU execution uses native AOT bundles
and remains compute-only because Vernon does not provide a software rasterizer.

## Target execution architecture

Execution is split across RuntimeCore and backend-specific providers:

- `VernonRHI` is the hardware layer. Capability facets separate Compute,
  Graphics, and NativeInterop; CUDA implements Compute only.
- `VernonRuntimeCore` owns PipelineAsset parsing, reflection, binding plans, and
  prepared-pipeline caches. It calls an opaque `RuntimeDeviceProvider` SPI and
  does not depend on VernonRHI types.
- `VernonRuntimeRHIAdapter` implements that SPI with VernonRHI. A foreign
  engine may instead implement the SPI with its own RHI and depend only on
  RuntimeCore.
- `RuntimeCpuProvider` implements the Compute facet directly over CPU AOT/JIT
  entries and host buffers. It does not link VernonRHI; dispatch is synchronous.

RuntimeCore never owns textures, framebuffers, render graphs, queues, or
resource state. Provider-owned resource references are non-owning opaque
handles with immutable metadata. The provider owns allocation, barriers,
submission, completion, and transient descriptor/upload storage.

Pipeline loading prepares layouts and immutable pipelines once. Invocation
uses pre-resolved slots and prepared bindings; it must not parse manifests,
perform name lookup, or create pipeline/layout objects on the hot path.
The OpenGL provider covers compute plus graphics pipelines with storage
buffers, inline scalar buffers, float uniforms, sampled images, explicit or
implicit samplers, vertex inputs, index buffers, and color attachments. Shader
compilation, uniform-location lookup, program linking, VAO, binding storage,
and attachment FBO creation happen during pipeline preparation. Dispatch and
draw encode into those stable objects. RuntimeCore preallocates both sides of
its resource-reference swap so repeated binding updates do not allocate merely
because an optional sampler changes between a native object and the provider
default.

Runtime implementation is separated by responsibility:
`runtime_backend_dispatch` handles contexts and resources,
`runtime_pipeline_direct` adapts direct compute artifacts and CPU entries into
synthetic compute pipelines, and `runtime_pipeline_dispatch` routes
PipelineAsset resolution plus compute/graphics invocation. Pipeline
preparation, destruction, and invocation live in the CPU, CUDA, Vulkan, D3D12,
and OpenGL pipeline translation units; the dispatch file contains only backend
routing and synchronization. There is no Runtime Kernel handle or kernel
dispatch layer. RHI provider implementations are split into common, CUDA,
D3D12, OpenGL, and Vulkan translation units; CPU provider preparation remains
with RuntimeCpuProvider.

## Distribution and Engine ownership

Distribution exports `VernonRHI`, `VernonRuntimeCore`, and the optional
`VernonRuntimeRHIAdapter` as separate targets. Vernon Engine and the Python
wheel use all three. A foreign engine may use RuntimeCore with its own provider
without adopting VernonRHI.

RHI and provider handles are scoped to their creating device and must never
cross devices or independently loaded Runtime/RHI copies.
When Runtime is a shared library, VernonRHI is also shared. Engine and Runtime
must resolve one RHI module so process-wide device handles cannot accidentally
address duplicate static registries.

D3D12 device selection, COM device/queue/allocator/list/fence ownership,
resource creation, and resource destruction live in VernonRHI. RuntimeCore
owns format and invocation planning; the D3D12 provider owns preparation and
encoding. Three command frames rotate allocator/list ownership. Persistently
mapped upload/readback rings and persistent RTV/resource/sampler descriptor
rings grow geometrically and reuse storage after completed submissions. The
RHI slot storage is address-stable because RuntimeCore resource references
point directly at native resource state. The
current C ABI submission remains synchronous, so ring wrap cannot overwrite
in-flight GPU data; asynchronous submission will require fence-tagged ring
segments.

D3D12 graphics preparation is lazy because render-target formats, topology,
and concrete vertex strides arrive with the first invocation. The resulting
root signature (including descriptor tables), input layout, and PSO are cached
as one prepared entry keyed by those invocation-stable properties. Repeated
draws reuse the complete entry without creating new D3D12 pipeline objects.

Small graphics uniforms use stage-local D3D12 root constants and Vulkan push
constants instead of per-vertex buffers. Host matrices use the Python/NumPy
row-major convention; Vulkan transposes square matrix payloads while packing
push constants, while D3D12 HLSL and OpenGL consume the row-major payload
through their native matrix binding conventions.

D3D12 NativeInterop uses external-recording ownership. Borrowed devices,
queues, and open direct command lists are referenced without `AddRef`; imported
buffers, images, and descriptor ranges receive RHI generational handles but
remain owner-managed COM objects. Destroying those RHI handles only invalidates
the slots. RHI does not close, execute, signal, synchronize, or release the
borrowed objects; the owner submits the recorded command list and controls
completion.

Vulkan loader dispatch and instance, physical/logical device, queue, command
pool, buffer/memory, image/view/memory, and sampler ownership live in
VernonRHI. Runtime keeps shader reflection and invocation compatibility while
delegating allocation, destruction, command recording, submission, and device
completion to the RHI device state.

Runtime-visible Vulkan buffers use device-local memory. Host transfers pass
through persistent mapped upload/readback rings. Three command buffers and
fences rotate from one RHI-owned command pool. RHI resource slots use
address-stable storage because RuntimeCore opaque resource references point at
their native buffer, image, view, or sampler state. Descriptor sets come from
one device pool and remain valid with their cached RuntimeCore binding sets;
the pool is not reset per submission, and each set is individually returned
when its binding set is destroyed. Submission remains synchronous for
compatibility, so completed submissions safely reset transient ring offsets.

Vulkan graphics prefers dynamic rendering when Vulkan 1.3, or Vulkan 1.2 with
`VK_KHR_dynamic_rendering`, exposes the feature. Older devices use cached render
passes keyed by attachment-location formats; framebuffers remain invocation
specific because they contain the concrete image views and extent.

Graphics draw invocations clear every color attachment to transparent black.
An optional D32 attachment is cleared to one and enables less-than depth
testing and depth writes in Vulkan, D3D12, and OpenGL. Python exposes this
render-only resource as `DepthTexture`; sampled depth remains a separate
extension because it requires an explicit shader-readable format/view contract.

The deployable Runtime does not depend on LLVM, MLIR, GLFW, the CUDA Toolkit,
or a statically linked Vulkan loader. Compiler and asset cooking remain host
tools. CPU deployment uses registered AOT object entry points; CUDA and Vulkan
resolve system drivers dynamically; OpenGL/GLES consume an externally owned
context.

Vulkan NativeInterop adopts an externally owned instance, physical/logical
device, queue, and command buffer without submitting, synchronizing, resetting,
or destroying them. Imported buffers, images, and image views use generational
RHI handles while retaining explicit size, format, subresource, usage, layout,
and queue-capability metadata. Image destruction is rejected while an imported
view still references it; releasing an RHI handle never releases the borrowed
native object.

## CUDA driver loading

The optional CUDA backend dynamically resolves the stable Driver API from
`nvcuda.dll` on Windows or `libcuda.so.1` on Linux. Vernon emits PTX through
LLVM and therefore does not require CUDA Toolkit headers, import libraries, or
`nvcc`; a compatible installed NVIDIA display driver is sufficient.

CUDA driver loading, primary-context ownership, streams, device allocation,
modules, and functions belong to VernonRHI. Each device keeps one persistent
non-default stream, and synchronous compatibility transfers reuse a
size-matched pinned staging pool around asynchronous Driver API copies. Compute
pipeline preparation fixes the argument pointer layout once; dispatch only
updates preallocated descriptors and enqueues on the persistent stream. Both
direct artifact pipelines and PipelineAsset compute pipelines use this path,
avoiding module, stream, staging, and argument-vector allocation on the hot
path.

## CUDA structured control flow

`VernonToGPU` creates `gpu.module` directly. Because that operation is an
isolated symbol table, the top-level SCF conversion in MLIR's standard NVVM
pipeline does not rewrite control flow inside the kernel. Vernon therefore
runs value-Tensor and SCF lowering as nested `gpu.module` passes before
invoking the standard pipeline. Static value Tensors with at most 16 elements
are flattened in row-major order to register vectors. The limit covers
`Matrix[f32, 4, 4]`
and bounds the register pressure from scalarized operations. Larger static
values use elementwise-to-Linalg, one-shot bufferization to private memrefs,
and Linalg-to-SCF loops. Dynamic local value Tensors are rejected; addressable
runtime N-D Tensor parameters are specialized and remain memrefs. The SCF
structural conversion updates loop-carried values and region arguments
consistently. Buffer intrinsics cloned below `scf.if` or `scf.while` are
rewritten after the complete kernel body is cloned, so nested loads and stores
do not remain illegal Vernon operations.

Standalone PTX loaded through the CUDA Driver API is not linked with
`libdevice`. CUDA lowering emits square root and approximate cosine as LLVM
intrinsics before MLIR's libdevice-call conversion. NVPTX therefore selects
native `sqrt.rn` and `cos.approx` instructions, keeping artifacts
self-contained while allowing `norm` and `cos` inside kernels.

## Vulkan compute runtime

The Vulkan backend dynamically resolves the system loader and executes the
same specialized kernel source through SPIR-V. Tensor resources and scalar
inputs are storage-buffer descriptors in set zero; scalar descriptors are
materialized per launch. Runtime-owned Tensor buffers use host-visible,
host-coherent memory so the existing synchronous upload/download contract does
not require a staging queue. This favors a small portable runtime over peak
transfer throughput; a future asynchronous API may add device-local buffers.

SPIR-V lowering uses MLIR's GPU ABI materialization after Vernon value-Tensor
normalization. CUDA's LLVM math pass is never used: Vulkan retains standard
math operations for SPIR-V lowering, and Metal source is cross-compiled from
the same SPIR-V module.

Metal MSL remains a cook-only compiler product. DirectX cooking emits
Shader Model 6 DXIL containers and `VernonRuntime` exposes a Windows-only
D3D12 backend. The backend owns its device, direct queue, command allocator,
fence, buffers, textures, samplers, descriptor heaps, and offscreen render
targets. Deployments load pre-cooked DXIL and do not load DXC. Synchronous
submission keeps transient upload/readback and descriptor storage alive until
the fence completes. Tests select WARP through an internal hook; normal device
creation skips software adapters.

### Pipeline runtime requirements

Schema-2 PipelineAssets may contain a hash-covered `runtime_requirements`
object. Its target-discriminated values are derived from the emitted artifact:
CPU target triple/object format/invocation ABI, GLSL profile and API version,
SPIR-V version plus Vulkan 1.1 and compute workgroup limits, or PTX version,
address size, and minimum compute capability. Required reflection features are
stored once in sorted order. DirectX additionally records D3D12, minimum
feature level, Shader Model, root-signature version, and compute workgroup
limits. Metal omits this field because it has no Runtime backend.

`target_options` records how compilation was requested; it is not a runtime
capability contract. `runtime_requirements` records the minimum capabilities
needed by the resulting artifact. Runtime validates the latter after the
manifest hash and target, but before resolving or loading artifacts. The field
is mandatory. CUDA intentionally does not infer a driver version from PTX;
compute capability is checked early, and the CUDA driver JIT remains
authoritative for PTX compatibility.

## OpenGL and OpenGL ES runtime

OpenGL and OpenGL ES are external-context AHI backends. The context owner
supplies `make_current` and `get_proc_address` callbacks through
`vernonRuntimeCreateOpenGLWithCallbacks`; Engine may register its existing
context with `register_external_opengl_context`. Otherwise Python creates a
hidden context through the separately linked `_gl_context` GLFW extension.
`VernonRuntime` itself never creates or links a window-system context.

Each backend accepts only pipeline bundles compiled for its matching GLSL
profile. Compute requires OpenGL 4.3 or OpenGL ES 3.1 and binds reflected
storage buffers before issuing a shader-storage barrier. Graphics accepts
imported host buffer, texture, and sampler names without deleting them. It also
creates owned child resources for standalone execution; all GL allocation,
transfer, deletion, and invocation operations first make the associated
context current.
Pipeline ABI v3 represents every numeric argument as one `VernonTensorView`.
The storage discriminator selects borrowed host memory or a runtime-owned
device buffer; shape and positive byte strides describe logical indexing, and
byte offset/size bound every reachable element. Host pointers are borrowed only
for the synchronous invocation. Device compute views remain contiguous
whole-buffer views because the compute launch ABI has no offset/stride fields.

OpenGL uploads native column-major matrices directly with `transpose=false`.
Desktop OpenGL also uploads native row-major matrices directly with
`transpose=true`; OpenGL ES and arbitrary strided layouts are packed into a
small column-major scratch because GLES requires `transpose=false`. Vulkan
push constants are packed by logical indices and Tensor strides.

## Tensor indexing

Addressable Tensor and explicit TensorView parameters lower from
`!vernon.tensor_view` to backend memrefs or storage resources. Multidimensional
indices are flattened in NumPy-compatible row-major order:
`linear = ((i0 * d1 + i1) * d2 + i2) ...`.
Strided host packing identifies the maximal row-major contiguous suffix and
copies one block per outer index. This preserves arbitrary positive-stride
views while avoiding per-element index division and tiny copies for common
padded-row and sliced-batch layouts.

## Pipeline runtime boundary

`VernonLoadedPipeline` is the only Runtime program handle. It represents either
one compute stage or a validated tuple of graphics stages; compute and graphics
entries are never combined in one pipeline. A persistent PipelineAsset resolves
to the same handle. Direct CPU entries and raw GPU artifacts synthesize a
compute-only variant and return `VernonLoadedPipeline` as well.

PipelineAsset target architecture and options are selected by the cooker. A
compute PipelineAsset resolves only against a compute backend, while a graphics
PipelineAsset resolves only against a graphics backend. The target profile is
part of artifact identity and the cooked manifest, not the source declaration.

The graphics stage tuple is extensible through the language's versioned stage
registry and topology rules. Runtime loading must reject a language-valid
stage topology with an explicit unsupported-target result when that backend
does not implement it. Manifest parsing must not hard-code vertex-plus-fragment
as the only representable topology.

Schema-2 binds each variant directly through its `program` stage-to-artifact
map. It contains either one compute program or one graphics-stage tuple; it has
no dispatch/barrier/draw step list and cannot encode host orchestration.

Multi-program orchestration, render-pass and attachment state, framebuffer or
renderbuffer abstraction, resource transitions, and compute/graphics backend
pairing are deferred. The archived Pass-graph proposal in
`specs/backup/execution_graph_design.md` is non-normative.

Tensor allocations belong to one runtime generation. Reinitializing the
runtime invalidates cached native handles. Within a generation, unchanged
host data is not uploaded again, shader writes remain device-resident, and
`to_numpy()` is the synchronization point that downloads device-dirty data.

## Graphics invocation state

Index buffers remain ordinary resident `u32` Tensors, but `indices` is a
host-only draw argument and never enters a shader interface. Primitive topology
is likewise host draw state so one specialized shader can render triangle,
line, or point lists.

Named fragment struct fields are flattened before SPIR-V lowering. Their
reflected source names and locations route `targets` independently of mapping
iteration order. Feature specialization runs before the vertex/fragment
interfaces are merged and validated, making the specialized reflection the
only runtime binding contract.

Pipeline invocation ABI version 3 carries index bindings, color attachments,
topology, viewport, scissor, compute grid, and reflected argument slots.
Backend-specific command encoding consumes this common invocation without
exposing legacy program/draw entry points.

Runtime planning separates the two pipeline kinds before backend dispatch.
Compute validation, argument packing, and grid inference produce one
`PlannedComputeLaunch`; graphics attachment, resource-pairing, vertex-input,
and draw-state validation produce one `PlannedGraphicsInvocation`. Backends
consume the corresponding plan and do not repeat frontend invocation planning.

## Compiler-generated graphics values

Compiler reflection marks generated entry arguments with
`vernon.implicit`. An implicit sampler carries
`vernon.implicit = "sampler"` and `vernon.implicit_texture = <source name>`;
its existing `sampled_texture_bindings` relation identifies every concrete
texture descriptor it samples. A generated resolution argument carries
`vernon.implicit = "resolution"` and has type `tensor<2xf32>`.

The schema-2 cooker removes these arguments from the external `parameters`
table and records them in `internal_parameters`. Each internal record has
`source: "implicit_sampler"` or `source: "system_value"`; resolution also has
`system_value: "resolution"`. Stage uses retain concrete uniform names,
argument indices, descriptor set/binding values, and sampled-texture
relations. This is the backend ABI: public parameter enumeration and
invocation slots expose only source parameters.

An explicitly declared sampler is not compiler-generated and remains an
external sampler slot, overriding implicit sampler generation.
`VernonTextureView.sampler` optionally supplies the policy for an implicit
sampler and must belong to the pipeline context. OpenGL binds that sampler
object, while Vulkan combines it with the reflected texture descriptor. A null
field uses the backend runtime default: OpenGL sampler object zero or the
runtime-context-owned cached Vulkan linear/repeat sampler. Explicit sampler
parameters ignore the texture-view field.
The runtime supplies resolution as `(width, height)` from a non-empty
invocation viewport, falling back to the common color-attachment extent.
Generated resolution is an OpenGL uniform or a Vulkan push constant according
to its reflected stage use.

## Shared definitions and struct methods

Function domains follow CUDA-style compatibility while retaining Python
decorator syntax. `@func` has the device domain and `@func(shared=True)` has
both host and device domains. A caller's domains must be a subset of the
callee's domains: device code can call shared helpers, but shared helpers
cannot call device-only helpers or operations.

Shared functions execute their original Python body with NumPy-backed scalar,
vector, matrix, and pure numeric intrinsic implementations. Shared structs
copy and freeze numeric fields at construction, providing host value semantics
without promising a packed host/device binary ABI.

Struct methods are source-normalized to private functions named
`Struct__method`, with a struct-typed explicit first argument. Imported struct
names keep their module prefix before method mangling. Distinct shared and
device-only methods may coexist on a shared struct; duplicate names are
rejected rather than treated as host/device overloads. Device compilation
continues to forbid recursion and mutable `self`.

Graphics manifest schema 3 carries explicit vertex-attribute leaves. A bound
Tensor has shape `(record_count, *logical_shape)`, contiguous row-major inner
dimensions, an arbitrary positive record stride, and an independent base
offset. Leaves select locations, dtype formats, component counts, and relative
byte offsets; an instance divisor changes only fetch rate. OpenGL, Vulkan, and
DirectX consume this common list and reject formats or location spans that the
actual device cannot represent rather than applying rank or scalar-count
limits.
