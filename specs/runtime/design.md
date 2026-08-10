# Runtime design

## CPU cooked pipelines

CPU cooking emits the canonical `*.pipeline.json` manifest, a target
relocatable `.o`/`.obj` with stable module-hashed C entry wrappers, and
generated registration `.c`/`.h` sources. The manifest records
`PIPELINE_VERSION`, the target triple, object format, exported symbol, artifact
size, and SHA-256. Applications link the object and generated registration
source at build time and call its registration function before loading the
pipeline. Runtime validates the external descriptor but never parses or
relocates object files. The removed `vernon-compile --compute-bundle` and
`compute.json` format are not pipeline-14 deployment APIs. Python immediate
execution uses compiler-owned LLJIT to load the host object directly. LLVM IR
and ORC JIT are not persistent runtime bundle formats.

## CPU range-phase execution

Ordinary CPU compute and CPU autodiff use one range-only execution model.
Runtime passes a `VernonCpuRangeV1` to the compiled entry; it does not invoke a
scalar kernel entry once per logical invocation and has no blocking
cooperative-workgroup compatibility executor.

One scheduler owns a persistent worker pool bounded by
`VERNON_CPU_THREAD_BUDGET`. A callback processes one contiguous
workgroup-linear lane interval in compiled code. Scheduling is hierarchical:

- grids with at least as many workgroups as runners are claimed in bounded
  groups, and each claimed workgroup executes as one range;
- fewer, larger workgroups divide each phase into deterministic non-empty lane
  ranges assigned across the available runners;
- one workgroup of at most 64 lanes executes inline to avoid queue and wake-up
  overhead.

The pool never creates one OS thread or queue item per lane. A dispatch with
many groups streams group state through bounded workers; a large individual
workgroup uses at most the configured worker count. Multiple callers share the
same bounded pool. Scheduling is currently non-preemptive at the runner-job
level, so one sufficiently parallel long dispatch may occupy the pool until it
completes; fair multi-dispatch latency is a future QoS feature, not part of the
synchronous execution contract.

Each active workgroup owns:

- a lazily allocated 16 KiB primal shared arena;
- lazily allocated pullback shared-adjoint sites;
- cache-aligned eight-lane blocks for lane-private allocations and coroutine
  frames;
- the current phase, expected barrier site, outstanding-range count, and
  latched failure diagnostic.

Shared allocation sites are established before the first yield and sealed
after it. Workgroup and lane allocation use checked size/alignment arithmetic;
inconsistent sites, overflow, allocation failure, or a new sealed shared site
fail the dispatch. Lane storage belongs to logical lanes rather than worker
threads and therefore survives phase migration.

A compiled range starts with `outcome = complete`. A lowered barrier changes
it to `yielded` and records the site. The last range in a phase advances the
workgroup only when all ranges either complete or yield at the same site.
Mixed completion/yield, different sites, exceptions, invalid outcomes, and
entry errors latch one failure, stop new phases, wake the dispatch waiter, and
release group state when the job is destroyed. Workers never wait at a
workgroup barrier.

## Backend loading

`VernonRuntime` owns the Win32/POSIX library loader used by CPU native-library
descriptors. CPU relocatable objects resolve through the static entry registry.
VernonRHI owns CUDA Driver and Vulkan loader discovery; Vulkan headers are
compile-only. This keeps LLVM, GLFW, CUDA Toolkit libraries, and the
Vulkan loader import library outside the deployable runtime dependency closure.
Vulkan loader discovery is runtime-only and ordered: `VERNON_VULKAN_LOADER`, a
loader under `VULKAN_SDK`, the normal OS loader name, then Homebrew fallbacks on
macOS. Failure diagnostics retain every attempted location. Vernon always loads
the Khronos loader rather than an ICD such as MoltenVK directly, so standard ICD
discovery remains intact. Build-time `VERNON_ENABLE_VULKAN_RUNTIME` only
includes the backend; runtime availability still depends on a loader, ICD, and
usable device.

Before creating an instance, Vernon enumerates advertised instance extensions
and enables `VK_KHR_portability_enumeration` plus its instance flag only when
present. It similarly enumerates the selected device's extensions, enables
`VK_KHR_portability_subset` only when advertised, and queries its supported
feature structure in the device feature chain. Portability behavior is therefore
capability-driven rather than selected by the host platform.

OpenGL function resolution is isolated in `backend_opengl_driver`; unlike CUDA
and Vulkan it consumes context callbacks. OpenGL and OpenGL ES are distinct
external-context backends and only accept pipeline manifests for their matching
GLSL profile. Cooked GLSL is an external content-addressed artifact;
interactive GLSL remains inline.

Context ownership is a separate layer: `Context Owner -> external callbacks ->
VernonRHI -> Runtime adapter`. Vernon Engine and Python each create one RHI
device for their existing or hidden context, then create Runtime for that
device. The Python wheel's separate `_gl_context` extension links GLFW and
keeps the context owner alive until after Runtime and RHI children are
destroyed. Runtime-only builds never discover GLFW.

The OpenGL function table, external-context callbacks, buffers, images,
samplers, shader compilation, programs, vertex arrays, and attachment
framebuffer objects are owned by VernonRHI. Runtime retains format and
invocation policy while PipelineAsset and direct compute-artifact loaders both
produce `VernonLoadedPipeline` and execute through RuntimeCore and the RHI
provider adapter.
`VernonOpenGLContextCallbacks` is the single context contract for Python and
Engine owners; backend behavior never branches on context origin.

## Vulkan graphics bundles

The current `PIPELINE_VERSION` format stores each SPIR-V stage as a content-addressed
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
store tightly packed faces in `+X, -X, +Y, -Y, +Z, -Z` order on Vulkan,
D3D12, and OpenGL. D3D12 stages each array face in its own aligned placed
footprint because `GetCopyableFootprints` reports per-call byte size
independently of its base offset. Desktop OpenGL enables seamless cube-map
filtering so linear samples crossing face boundaries match Vulkan and D3D12;
OpenGL ES already requires seamless cube filtering. OpenGL feature variants
retain stable logical parameter slots, but linked programs may optimize
inactive uniforms away; provider preparation omits those locations and their
paired sampler binds instead of treating a `-1` location as a malformed
artifact. D3D12 sampled depth uses an R32 typeless allocation with a D32 DSV
and R32 float SRV; Vulkan and OpenGL use their native D32 sampled depth
formats.

## Python native module

Python exposes compiler services and all runtime backends through one `_native`
module. The module links the compiler DLL and `VernonRuntime`; the runtime DLL
itself retains its dependency boundary. Interactive GPU pipelines serialize
the current `PIPELINE_VERSION` format with inline artifact descriptors; the
cooker emits the same format with external descriptors only. CPU execution uses native AOT bundles
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
  entries and contiguous host Tensor bytes. It does not link VernonRHI;
  dispatch is synchronous.

RuntimeCore never owns textures, framebuffers, execution graphs, queues, or
resource state. It transactionally retains opaque provider resource references
held by prepared bindings. The provider owns allocation, logical records,
barriers, submission, completion, and transient descriptor/upload storage.
Python `Texture` owns color image storage, including two-dimensional,
three-dimensional, and six-face Cube resources. Three-dimensional textures use
`(depth, height, width, channels)` host order and are shader resources rather
than attachments. `RenderTarget` owns or groups two-dimensional attachment
references and manages its optional depth attachment explicitly. Its
`depth_texture` is a shader-readable resource view of that attachment; depth is
not a public host-transfer `Texture` format. Immutable `SamplerState` objects
bind shader `Sampler` parameters independently of image ownership.

## Reflection-driven structured Value binding

The public C++ `PipelineInvocationBuilder` packs field trees, Tensor
structure-of-arrays views, and per-element callbacks into runtime-owned
canonical bytes. It resolves every leaf through the reflected field/index path
under one source parameter name and validates dtype, static leaf shape, outer
Tensor shape, duplicates, and completeness. Native C++ struct layout is never
inferred or reinterpreted. Scalar static-Tensor leaves carry their shape in
reflection because scalar count alone cannot distinguish layouts such as
`Tensor[f32, (2, 2)]` and `Tensor[f32, (4,)]`.

Backend adapters share one deterministic vertex-attribute capability check for
expanded dtype/component/location leaves, then apply device-specific format
feature queries. This keeps OpenGL, Vulkan, and D3D12 rejection policy aligned
without moving hardware ownership into RuntimeCore.

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
`runtime_backend_dispatch` handles CPU contexts and borrowed RHI adapters,
`runtime_pipeline_direct` adapts direct compute artifacts and CPU entries into
synthetic compute pipelines, and `runtime_pipeline_dispatch` routes
PipelineAsset resolution plus compute/graphics invocation. Pipeline
preparation, destruction, and invocation live in the CPU, CUDA, Vulkan, D3D12,
and OpenGL pipeline translation units; the dispatch file contains only backend
routing and synchronization. There is no Runtime Kernel handle or kernel
dispatch layer. RHI provider implementations are split into common, CUDA,
D3D12, OpenGL, and Vulkan translation units; CPU provider preparation remains
with RuntimeCpuProvider.

GPU invocation always records through one VernonRHI command encoder. A
non-null provider encoder resolves to the encoder's active native command list,
command buffer, context, or stream; provider draw and dispatch callbacks record
commands but never finish or submit it. Immediate invocation creates an
ephemeral encoder, records once, finishes, and submits once. ExecutionGraph
uses the same contract across all compiled scopes and submits once after the
last scope. Concurrent ExecutionGraph executions on one device serialize their
complete command-encoder lifecycle through a device session because the
synchronous RHI contract permits only one active encoder per device; this
policy is independent of Runtime contexts and graph contents. Encoder lookup
uses an O(1) generation-checked registry and a
per-encoder lock; backend submission and fence waits never hold the registry
lock. Recording retains an O(1)-deduplicated set of RHI resources plus prepared
pipeline and binding objects. Owned submissions release those references after
the synchronous backend completion; borrowed native command targets retain
them until encoder destruction, which is the owner's completion signal.
Failed recordings are abandoned with the same cleanup path.
The stable `0.1.2` API does not expose asynchronous submission, deferred graph
execution, or multiple frames in flight. Callers must not infer those
capabilities from backend-native queues or streams. Adding them requires
completion-serial tracking and deferred reclamation for every enabled backend,
as defined by the
[`asynchronous GPU resource lifetime`](../roadmap.md#asynchronous-gpu-resource-lifetime)
roadmap.

ExecutionGraph render scopes own the first attachment load operations and the
last attachment store operations. Providers consume those scope operations
instead of per-draw values. Vulkan encodes them in dynamic rendering or a
compatible fallback render pass; D3D12 and OpenGL apply final discard when the
scope ends. Vulkan barriers map resource state plus explicit stage/access
masks, OpenGL emits one merged destination barrier, and D3D12 emits UAV
barriers for same-state write hazards.

ExecutionGraph resource identity combines the owning graph with the RHI
resource kind, slot, and generation. Re-importing one RHI image through
different views shares one hazard identity while each attachment retains its
view metadata. Render-pass fusion requires identical attachment geometry and
read-only policy. An intermediate clear is encoded explicitly inside the
scope; an intermediate discard, or preserve after a discarded store, splits
the scope because retaining the attachment would change observable contents.
Culled passes never record encoder references, and failed compilation discards
its partial schedule and scopes before any command encoder is created.

Runtime has no resource ownership API. GPU Tensor, image, sampler, index, and
attachment bindings are `VernonRuntimeProviderResourceReference` values
obtained from the same RHI device used to create Runtime. Host Tensor bindings
are accepted only by CPU or inline-value paths. Runtime allocation/import
handles and backend resource fallback branches are intentionally absent.

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
encoding. Synchronous submission reuses one allocator/list command frame. Persistently
mapped upload/readback rings and persistent RTV/resource/sampler descriptor
rings grow geometrically and reuse storage after completed submissions. The
Provider references encode stable RHI slot-and-generation keys rather than slot
addresses or COM pointers. Public destruction invalidates the owner handle
immediately; an owned native resource remains in its logical record until all
prepared bindings release it. The current C ABI submission remains synchronous,
so ring wrap cannot overwrite in-flight GPU data; asynchronous submission will
require fence-tagged ring segments.

D3D12 graphics preparation is lazy because render-target formats, topology,
and concrete vertex strides arrive with the first invocation. The resulting
root signature (including descriptor tables), input layout, and PSO are cached
as one prepared entry keyed by those invocation-stable properties. Repeated
draws reuse the complete entry without creating new D3D12 pipeline objects.

Small graphics uniforms use stage-local D3D12 root constants and Vulkan push
constants instead of per-vertex buffers. Host matrices use the Python/NumPy
row-major convention; Vulkan transposes square matrix payloads while packing
push constants, while D3D12 HLSL and OpenGL consume the row-major payload
through their native matrix binding conventions. Stable parameter slots remain
name-sorted, but root/push-constant offsets follow each shader entry's source
argument index; otherwise multiple same-stage matrices can be silently
exchanged. Vulkan offsets also honor each reflected physical alignment. D3D12
offsets follow HLSL constant-buffer register packing: scalar/vector values may
share a 16-byte register but never cross one, while matrices and aggregate
values begin on a new register. Encoding writes one contiguous root-constant
or push-constant range per shader stage; changing only those inline values does
not invalidate descriptor state.

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
completion to the RHI device state. Owned-device indices use high-performance
ordering on both desktop APIs: D3D12 follows DXGI GPU preference, while Vulkan
orders usable discrete, integrated, virtual, and CPU devices, breaking
same-type ties by device-local memory.

Runtime-visible Vulkan buffers use device-local memory. Host transfers pass
through persistent mapped upload/readback rings. Readback memory prefers a
host-cached, host-coherent type and falls back to host-coherent memory when the
device does not expose that combination. One command buffer and fence are
reused from the RHI-owned command pool. RuntimeCore opaque resource
references resolve generation-checked logical records to native buffer, image,
or sampler state when bindings are encoded. Each resource-content revision
owns an immutable descriptor-set snapshot plus one combined aligned inline
buffer. Unchanged snapshots are shared across command encoders; in-flight
command references retain stale revisions until completion, while the current
revision stays cached with the prepared binding set. Binding updates therefore
cannot mutate earlier commands, and push-constant-only animation does not
allocate descriptor sets. Submission remains synchronous, so completed
submissions safely reset transient ring offsets.

Vulkan graphics prefers dynamic rendering when Vulkan 1.3, or Vulkan 1.2 with
`VK_KHR_dynamic_rendering`, exposes the feature. Older devices use cached render
passes and framebuffers keyed by retained attachment generations, formats,
color/depth/stencil load-store operations, and extent. The persistent cache is
bounded; overflow objects belong to the active command and are released at
completion.

Rendering scopes carry explicit `Clear`, `Preserve`, or `Discard` load
operations and `Preserve` or `Discard` store operations. Clear values belong to
the attachment use, not the pipeline. An optional D32 attachment enables
less-than depth testing and depth writes in Vulkan, D3D12, and OpenGL. Python
keeps that attachment under `RenderTarget` ownership and exposes a
shader-readable `depth_texture` reference when depth sampling is required.
Importing the attachment through both its target and shader-resource identities
resolves to one native graph resource, so attachment-to-sampling hazards remain
visible.

## PBR integration reference

`examples/pbr.py` is the executable multi-pass graphics reference. A shadow
pass writes a generic D32 texture; the dependent PBR pass samples that texture
and an RGBA8 Cube environment through separate immutable samplers. The
procedural cube and plane keep backend validation deterministic and avoid a
model-loader dependency. The BRDF uses Cook-Torrance with the GGX distribution
from Walter et al. (2007), Schlick Fresnel (1994), and the separable Smith
masking approximation. These choices match the metallic-roughness workflow
while keeping generated shaders small enough for all three desktop backends.
Shadow visibility uses a fixed four-tap half-texel PCF kernel and a 0.004
receiver bias; the compact deterministic kernel is intended for portability
validation rather than production-quality filtering. Its sampler clamps at the
edge. OpenGL shadow UV uses an upward Y scale, while Vulkan and D3D12 use a
downward Y scale to match their render-target coordinate convention.

Native backend graph tests observe adapter calls made after native draw or
dispatch recording and assert one RHI submission per graph. The
`VERNON_ENABLE_SANITIZERS` CMake option enables AddressSanitizer; lifetime
stress repeatedly invokes prepared bindings after public owner destruction.
Vulkan validation runs set `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`
for the same graph and PBR tests.

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
runtime N-D Tensor parameters remain memrefs. Their dynamic shape, stride, and
offset are invocation data and do not require AOT shape specialization. The
SCF structural conversion updates loop-carried values and region arguments
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

On Apple platforms, cooked MSL bundles are consumed by the Metal Runtime.
Apple Silicon macOS wheels expose the stable compute and offscreen graphics
subset. The Runtime compiles the cooked MSL for the selected device, prepares
reflection-driven resource layouts and pipeline state, and executes through
VernonRHI.

DirectX cooking emits Shader Model 6 DXIL containers and `VernonRuntime`
exposes a Windows-only D3D12 backend. The backend owns its device, direct queue,
command allocator, fence, buffers, textures, samplers, descriptor heaps, and
offscreen render targets. Deployments load pre-cooked DXIL and do not load DXC.
Synchronous submission keeps transient upload/readback and descriptor storage
alive until the fence completes. Tests select WARP through an internal hook;
normal device creation skips software adapters.

### Pipeline runtime requirements

PipelineAssets may contain a hash-covered `runtime_requirements`
object. Its target-discriminated values are derived from the emitted artifact:
CPU target triple/object format, GLSL profile and API version,
SPIR-V version plus Vulkan 1.1 and compute workgroup limits, or PTX version,
address size, and minimum compute capability. Required reflection features are
stored once in sorted order. DirectX additionally records D3D12, minimum
feature level, Shader Model, root-signature version, and compute workgroup
limits. Metal records the Apple platform, MSL version, minimum OS version, and
required features for early Runtime validation.

The canonical `target` object records a `kind` and that backend's typed
`options`; it is not a runtime capability contract. `runtime_requirements`
records the minimum capabilities needed by the resulting artifact. Runtime
validates the latter after the manifest hash and target, but before resolving
or loading artifacts. The field is mandatory. CUDA intentionally does not infer a driver version from PTX;
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
storage buffers and storage images before issuing the required memory barrier.
The current language contract permits filtered texture sampling only in
fragment shaders, so a compute pipeline containing a sampled-image parameter is
rejected during preparation rather than failing later during dispatch.
Graphics accepts imported host buffer, texture, and sampler names without
deleting them. It also creates owned child resources for standalone execution;
all GL allocation, transfer, deletion, and invocation operations first make
the associated context current.
The current `PIPELINE_VERSION` invocation ABI represents every numeric argument
as one `VernonTensorView`. The storage discriminator selects immutable host
Value transport, borrowed TensorStorage memory, or a runtime-owned device
allocation. TensorView records carry shape, signed byte strides, and byte
offset/size so full-owner and explicit subview dispatch preserve identical
logical addressing. A host pointer used for an asynchronous dispatch remains
borrowed, and its TensorStorage owner is retained until backend completion.

OpenGL uploads native column-major matrices directly with `transpose=false`.
Desktop OpenGL also uploads native row-major matrices directly with
`transpose=true`; OpenGL ES and arbitrary strided layouts are packed into a
small column-major scratch because GLES requires `transpose=false`. Vulkan
push constants are packed by logical indices and Tensor strides.

## Tensor indexing

Explicit TensorView parameters and `workgroup_storage` results lower from the
same address-space-parameterized `!vernon.tensor_view` family to backend
memrefs or storage resources. Owned workgroup views are flattened in
row-major order. External views use their validated projection:
`linear = offset + sum(index[d] * stride[d])`.
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

Pipeline 14 uses one canonical `*.pipeline.json` schema for compute and
graphics. Its optional root `autodiff` object contains differentiated-program
metadata; it is absent for ordinary primal-only assets. Pipeline-13
transform/profile fields are not current aliases.

The current `PIPELINE_VERSION` manifest binds each variant directly through its
`program` stage-to-artifact map. It contains either one compute program or one
graphics-stage tuple; it has no dispatch/barrier/draw step list and cannot
encode host orchestration.

`VernonExecutionGraph` owns multi-program orchestration above VernonRHI.
Class-based render and compute passes declare resource uses separately from
execution. Compilation infers RAW, WAR, and WAW edges, culls dead passes,
produces deterministic scheduling and barriers, and fuses adjacent compatible
render passes. Runtime pipeline invocations encode bindings and draw/dispatch
commands into the graph-provided typed encoder; attachment ownership and clear
policy remain outside Runtime.

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

`PIPELINE_VERSION` covers the invocation carrying index bindings, attachment
operations, topology, viewport, scissor, compute workgroup grid, an optional active
encoder, and reflected argument slots.
Backend-specific command encoding consumes this common invocation without
exposing legacy program/draw entry points.

Runtime planning separates the two pipeline kinds before backend dispatch.
The compute grid stores workgroup counts; total invocation extent is the
component-wise product of grid and reflected workgroup size. Compute
validation, argument packing, and grid inference produce one
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

The cooker removes these arguments from the external `parameters`
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
Generated resolution is an OpenGL uniform, Vulkan push constant, or DirectX 12
root constant according to its reflected stage use. All three graphics runtime
paths represent `source: "system_value"` in their binding plans and upload the
effective resolution before each draw.

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

The current `PIPELINE_VERSION` graphics manifest carries explicit
vertex-attribute leaves. A bound
Tensor has shape `(record_count, *logical_shape)`, contiguous row-major inner
dimensions, an arbitrary positive record stride, and an independent base
offset. Leaves select locations, dtype formats, component counts, and relative
byte offsets; an instance divisor changes only fetch rate. OpenGL, Vulkan, and
DirectX consume this common list and reject formats or location spans that the
actual device cannot represent rather than applying rank or scalar-count
limits.
