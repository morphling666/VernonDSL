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

## Launch ownership

Runtime contexts own one backend device/context. Buffers and loaded kernels
retain their context in language bindings; the C API rejects context
destruction while handles remain live. Transfers are synchronous and a launch
retains all argument storage through synchronization.

Backend dispatch is centralized in `runtime_dispatch`. `VernonRuntime.cpp`
owns public validation and handle lifetime but never includes backend headers
or reads concrete backend state. This keeps disabled CUDA/Vulkan translation
units out of the build and prevents backend payload types from leaking into
the C API shell.

## Distribution and Engine ownership

`VernonRuntime` is a standalone source project and CMake target. The Python
wheel owns a private host Runtime used by `_native` and also carries the
version-matched Runtime source closure for consumers that must compile it with
their own toolchain. Vernon Engine builds that source as part of the Engine
configuration. Host-wheel and Engine-built Runtime contexts, resources, and
handles are separate and must never cross.

The deployable Runtime does not depend on LLVM, MLIR, GLFW, the CUDA Toolkit,
or a statically linked Vulkan loader. Compiler and asset cooking remain host
tools. CPU deployment uses registered AOT object entry points; CUDA and Vulkan
resolve system drivers dynamically; OpenGL/GLES consume an externally owned
context.

Engine-owned Vulkan instance/device/queue and borrowed image/buffer adoption
remain a future integration boundary. That work must preserve queue ownership,
layout, synchronization, and lifetime metadata explicitly rather than treating
native handles as untyped integers.

## CUDA driver loading

The optional CUDA backend dynamically resolves the stable Driver API from
`nvcuda.dll` on Windows or `libcuda.so.1` on Linux. Vernon emits PTX through
LLVM and therefore does not require CUDA Toolkit headers, import libraries, or
`nvcc`; a compatible installed NVIDIA display driver is sufficient.

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

Metal MSL and DirectX HLSL PipelineAssets are cook-only compiler products.
They deliberately have no `VernonRuntimeBackend`; target inspection and bundle
loading return unsupported until matching runtime backends are implemented.

### Pipeline runtime requirements

Schema-2 PipelineAssets may contain a hash-covered `runtime_requirements`
object. Its target-discriminated values are derived from the emitted artifact:
CPU target triple/object format/invocation ABI, GLSL profile and API version,
SPIR-V version plus Vulkan 1.1 and compute workgroup limits, or PTX version,
address size, and minimum compute capability. Required reflection features are
stored once in sorted order. Metal and DirectX omit this field because they
have no Runtime backend.

`target_options` records how compilation was requested; it is not a runtime
capability contract. `runtime_requirements` records the minimum capabilities
needed by the resulting artifact. Runtime validates the latter after the
manifest hash and target, but before resolving or loading artifacts. Missing
requirements preserve legacy schema-2 loading behavior. CUDA intentionally
does not infer a driver version from PTX; compute capability is checked early,
and the CUDA driver JIT remains authoritative for PTX compatibility.

## OpenGL and OpenGL ES runtime

OpenGL and OpenGL ES are external-context AHI backends. The context owner
supplies `make_current` and `get_proc_address` callbacks through
`vernonRuntimeCreateExternalOpenGLForBackend`; Engine may register its existing
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

Kernel is the compute program form. Pipeline is the graphics program form and
contains a validated tuple of graphics stages. A persistent PipelineAsset wraps
exactly one of those forms; compute and graphics entries are never combined in
one pipeline.

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
