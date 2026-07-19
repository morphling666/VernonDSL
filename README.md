# VernonDSL

VernonDSL is a tensor-first Python language and MLIR compiler for graphics and
compute shaders.

The source language uses one compound numeric type,
`Tensor[element_type, shape]`. Familiar vector and matrix names are aliases,
not independent type families. The compiler shares type checking and
computation IR across all targets, then selects a target-specific lowering:

- graphics: Vernon interface semantics to SPIR-V;
- portable compute: MLIR GPU dialect to SPIR-V;
- CUDA compute: MLIR GPU dialect to NVVM/NVPTX;
- CPU reference: standard MLIR to LLVM.

SPIR-V is intentionally not treated as the universal backend IR.

## Current status

The compiler is under active development. A target is reported as available
only when its complete artifact pipeline works; unimplemented targets return
an explicit unsupported-target status.

The native compiler library currently provides:

- textual MLIR parsing and verification;
- deterministic entry-point reflection;
- a stable C ABI and target capability query;
- graphics lowering to SPIR-V and Vulkan `.spv` artifacts;
- compute outlining to MLIR `gpu.module`/`gpu.func` and Vulkan SPIR-V;
- CUDA compute lowering through NVVM to PTX;
- SPIRV-Cross source artifacts for OpenGL, OpenGL ES, and Metal;
- `vernon-opt`, the Vernon MLIR pass driver.

CUDA/NVPTX and CPU/LLVM are distinct branches; they are never routed through
SPIR-V. The CPU backend JIT-compiles the current graphics and compute numeric
subset, including buffer load/store and callback-based 2D texture sampling,
and exports guarded C ABI entry points for reference execution. DirectX
remains unavailable until the HLSL produced by SPIRV-Cross is completed by a
DXC-to-DXIL artifact step.

## Build on Windows

Build and install LLVM/MLIR first as described in
[`BUILD_POWERSHELL.md`](BUILD_POWERSHELL.md), then:

```powershell
cmake -S . -B build `
  -DMLIR_DIR="$PWD/llvm-project/install/lib/cmake/mlir" `
  -DVERNON_ENABLE_SPIRV_CROSS=ON
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
```

The public native API is declared in
[`source/include/vernon-c/Compiler.h`](source/include/vernon-c/Compiler.h).
Compiler architecture decisions are recorded in
[`specs/compiler/design.md`](specs/compiler/design.md).

## Compile a Python shader

The frontend parses Python source without importing or executing it:

```powershell
uv run --frozen vernon-compile-python `
  python/tests/smoke_shader.py `
  -o build/smoke.mlir

build/source/tools/vernon_opt/Release/vernon-opt.exe `
  build/smoke.mlir `
  --vernon-validate

build/source/Release/vernon-compile.exe `
  --target vulkan build/smoke.mlir `
  --output-dir build/vulkan
```

Shader regression examples are in `examples/`: `blinn_phong.py` implements
Blinn-Phong lighting, `blinn_phong_vertices.py` covers static, custom-instance,
and skinned vertex variants, and `planet_terrain.py` exercises a non-template
terrain vertex path.

Native consumers should include `<vernon-c/Compiler.h>`. Validation and
compilation results own their diagnostics, reflection, artifacts, and any CPU
entry points; destroy them with `vernonCompileResultDestroy`.

## Native artifacts and reflection

`vernonCompilerCompileMlir` can return multiple named artifacts. Enumerate them
with `vernonCompileResultGetArtifactCount`, `GetArtifactName`, and
`GetArtifactData`; returned string views remain valid until the result is
destroyed. Current artifact formats are SPIR-V (`.spv`), GLSL/GLES (`.glsl`),
Metal source (`.metal`), CUDA PTX (`.ptx`), and CPU LLVM IR (`.ll`).

Reflection is deterministic JSON. Each entry records its stage, workgroup size
when applicable, argument/result types, interface attributes, and CPU ABI
layout. CPU callers allocate byte buffers using `cpu_arguments_size` and
`cpu_results_size`; each value's aligned range is described by `cpu_offset` and
`cpu_size`. Retrieve the function with `vernonCompileResultGetCpuEntry` and
pass those buffers in `VernonCpuInvocation`. The wrapper returns
`VERNON_STATUS_INVALID_ARGUMENT` for null or undersized required buffers.

The CLI accepts `--output-dir` to write all named artifacts and
`--reflection <file>` to select the reflection output path:

```powershell
build/source/Release/vernon-compile.exe `
  --target cpu build/material.mlir `
  --output-dir build/cpu `
  --reflection build/cpu/material.json
```
