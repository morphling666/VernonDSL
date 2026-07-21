# Runtime design

## CPU compute bundles

Persistent CPU bundles contain a target relocatable object with stable,
module-hashed C entry wrappers. The manifest records the target triple, object
format, CPU invocation ABI, exported symbol, artifact size, and SHA-256.
Applications link the object at build time and register its wrapper with
VernonRuntime; Runtime validates the external descriptor but never parses or
relocates object files. Host-native shared-library bundles remain readable for
migration and Python immediate execution, where embedded LLD finalizes a host
object into an ephemeral DLL/so/dylib. LLVM IR and ORC JIT are not persistent
runtime bundle formats.

## Backend loading

`VernonRuntime` owns the Win32/POSIX library loader retained for legacy CPU AOT
bundles, CUDA, and Vulkan. New CPU objects resolve through the static entry
registry. CUDA Driver and Vulkan loader symbols are resolved at runtime; Vulkan
headers are compile-only. This keeps LLVM, LLD, GLFW, CUDA Toolkit libraries,
and the Vulkan loader import library outside the deployable runtime dependency
closure.

OpenGL function resolution is isolated in `backend_opengl_driver`; unlike CUDA
and Vulkan it consumes a host callback because the host owns the current
context. OpenGL and OpenGL ES are distinct external-context backends and only
accept pipeline manifests for their matching GLSL profile. Cooked GLSL is an
external content-addressed artifact; interactive GLSL remains inline. Context
creation is host policy.

## Vulkan graphics bundles

Cooked pipeline schema 2 stores each SPIR-V stage as a content-addressed
external `.spv` artifact. The manifest records its relative path, byte size,
and SHA-256, all of which the common artifact resolver validates before Vulkan
sees the bytes. Non-persistent interactive execution uses the same descriptor
shape with inline base64 storage because it has no durable asset directory.
Resolution creates immutable shader modules. Invocation creates render-pass
and pipeline state from the concrete attachment and vertex layouts, submits
synchronously, then releases that transient state. This first implementation
favors correct layout specialization; a later cache can key the same state
without changing the bundle ABI. RGBA8 offscreen images use optimal tiling and
staging buffers for host upload/readback. Unbound graphics uniforms use the
compiler's single push-constant block ABI; descriptor-bound uniforms remain a
later extension.

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
are flattened in row-major order to register vectors. The limit covers `mat4`
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

## OpenGL and OpenGL ES runtime

OpenGL and OpenGL ES are external-context backends. The host supplies
`make_current` and `get_proc_address` callbacks through
`vernonRuntimeCreateExternalOpenGLForBackend`; Python registers the same
addresses with `register_external_opengl_context`. `VernonRuntime` never
creates or links a window-system context.

Each backend accepts only pipeline bundles compiled for its matching GLSL
profile. Compute requires OpenGL 4.3 or OpenGL ES 3.1 and binds reflected
storage buffers before issuing a shader-storage barrier. Graphics imports host
buffer and texture handles, so resources remain owned by the host context.

## Tensor indexing

Addressable Tensor parameters lower to Vernon buffers. Multidimensional
indices are flattened in NumPy-compatible row-major order:
`linear = ((i0 * d1 + i1) * d2 + i2) ...`.

## Unified compute and graphics execution

The compute launch ABI remains version 1. Maintained graphics execution uses
`VernonPipelineBundle`, `VernonLoadedPipeline`, and
`vernonRuntimePipelineInvoke`. Bundle resolution validates feature variants,
stage artifacts, reflected parameter slots, and output layouts before
submission. OpenGL 3.3 or OpenGL ES 3.0 is sufficient for graphics;
compute/graphics compositions require OpenGL 4.3 or OpenGL ES 3.1.

Pipeline submission order is fixed: upload host-dirty Tensors, dispatch the
optional compute entry, issue the storage/vertex-input barrier, bind the
offscreen target and reflected arguments, draw, then mark the target
device-dirty. Readback is lazy. This order is required because a Tensor may be
written through an SSBO and consumed immediately as a vertex input without a
host round trip.

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

Pipeline invocation ABI version 1 carries index bindings, color attachments,
topology, viewport, scissor, compute grid, and reflected argument slots.
Backend-specific command encoding consumes this common invocation without
exposing legacy program/draw entry points.

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
