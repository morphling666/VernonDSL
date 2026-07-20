# Runtime design

## CPU compute bundles

CPU bundles persist the compiler's textual LLVM IR artifact. The standalone
runtime loads it with LLVM ORC and keeps the JIT alive for the lifetime of the
loaded kernel. This keeps bundle loading independent of CPython and
`VernonDSLCompiler`, while avoiding platform-specific shared-library link steps
in the asset cooker.

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

## OpenGL compute runtime

The OpenGL backend uses GLFW only to create a hidden OpenGL 4.3 core context
and resolve compute entry points. Tensor and scalar arguments are shader
storage buffers bound according to reflection; launches are synchronous and
issue a shader-storage memory barrier before host access.

OpenGL ES source remains available as GLSL ES 3.10 through SPIRV-Cross.
Runtime `opengles` currently executes the same supported compute subset through
the desktop OpenGL compatibility path because EGL and a GLES implementation
are not system components on every desktop platform. A native GLES runtime
will require an explicit EGL provider rather than silently depending on one
installed by another application.

## Tensor indexing

Addressable Tensor parameters lower to Vernon buffers. Multidimensional
indices are flattened in NumPy-compatible row-major order:
`linear = ((i0 * d1 + i1) * d2 + i2) ...`.

## Unified compute and graphics execution

The compute launch ABI remains version 1 and `vernonRuntimeLaunch` is
unchanged. Graphics is an additive ABI: versioned context creation,
RGBA8 Texture lifetime and transfer operations, graphics program loading,
reflected vertex bindings, uniform uploads, offscreen draw submission, and an
explicit compute-to-graphics barrier. OpenGL 3.3 is sufficient for a
vertex/fragment pipeline; compositions containing compute require OpenGL 4.3.

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

## Advanced OpenGL draw state

Index buffers remain ordinary resident `u32` Tensors, but `indices` is a
host-only draw argument and never enters a shader interface. Primitive topology
is likewise host draw state so one specialized shader can render triangle,
line, or point lists.

Named fragment struct fields are flattened before SPIR-V lowering. Their
reflected source names and locations route `targets` independently of mapping
iteration order. Feature specialization runs before the vertex/fragment
interfaces are merged and validated, making the specialized reflection the
only runtime binding contract.

Graphics draw ABI version 2 appends index, color-attachment, and topology
fields after the version-1 prefix. `struct_size` gates access to those fields;
version-1 descriptions normalize their single `target` to location zero.

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
