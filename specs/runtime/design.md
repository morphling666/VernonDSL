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

## Known limitation: CUDA `while` lowering

Straight-line Python kernels compile to PTX and execute through the dynamically
loaded CUDA Driver API, including Tensor memref arguments. The fractal kernel
does not yet compile for CUDA because it contains `scf.while`; the NVVM
pipeline reports:

`failed to legalize operation 'scf.while' that was explicitly marked illegal`

Experiments that inserted module-level SCF-to-CF and CF-to-LLVM passes were
reverted: they moved the failure to `gpu.func` or `cf.br`, which indicates that
the conversion must be placed and scoped correctly inside the `gpu.module`
rather than appended blindly to the top-level pipeline.

Next steps:

1. Add a minimal CUDA compiler regression test containing one loop-carried
   `scf.while`.
2. Capture the IR after `VernonToGPU` and before GPU-to-NVVM lowering.
3. Add the required nested conversion pipeline at `gpu.module` scope and
   verify that SCF, CF, and `gpu.func` are legalized in the expected order.
4. Add a Python CUDA numerical test for the fractal after the minimal test
   passes.

## Tensor indexing

Addressable Tensor parameters lower to Vernon buffers. Multidimensional
indices are flattened in NumPy-compatible row-major order:
`linear = ((i0 * d1 + i1) * d2 + i2) ...`.
