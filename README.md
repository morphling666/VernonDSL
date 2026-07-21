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

That JIT is a compiler reference facility, not a deployable runtime format.
`VernonRuntime` accepts CPU native-library AOT bundles only.

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

Configure a runtime-only build without LLVM/MLIR using the canonical options:

```powershell
cmake -S . -B runtime_build `
  -DVERNON_ENABLE_COMPILER=OFF `
  -DVERNON_ENABLE_RUNTIME=ON
cmake --build runtime_build --config Release --target VernonRuntime --parallel 4
```

The public native APIs are declared in
[`source/include/vernon-c/Compiler.h`](source/include/vernon-c/Compiler.h) and
[`source/include/vernon-c/Runtime.h`](source/include/vernon-c/Runtime.h).
The canonical runtime CMake target is `VernonRuntime`, exported as
`Vernon::Runtime`.
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

## Run an explicit compute kernel

Runtime kernels use an explicit global grid and runtime-owned contiguous
Tensors. Builtin arguments are synthesized and omitted from the call:

```python
from typing import Annotated
import vernon_dsl as vd

vd.init(arch=vd.cpu)  # vd.cuda and vd.vulkan use the same native Kernel API
output = vd.Tensor.zeros(dtype=vd.f32, shape=(8,))

@vd.kernel(workgroup_size=(8, 1, 1))
def scale(
    output: vd.Tensor[vd.f32, (None,)],
    factor: vd.f32,
    gid: Annotated[
        vd.Tensor[vd.u32, (3,)],
        vd.builtin("global_invocation_id"),
    ],
) -> None:
    output[gid[0]] = vd.f32(gid[0]) * factor

scale(output, 2.0, grid=(8, 1, 1))
values = output.to_numpy()
```

All three compute backends lower the restricted Python AST to Vernon MLIR on
the first specialized call, compile a target artifact, cache the loaded native
kernel, and launch through `VernonRuntime`. CPU execution creates a temporary
validated AOT compute bundle using `vernon-compile` and clang; set
`VERNON_COMPILER` when the compiler executable is not beside `_native`, in the
checkout build directory, or on `PATH`. There is no Python interpreter
fallback for `@kernel`.

`VERNON_ENABLE_RUNTIME` builds the standalone `VernonRuntime` C API with CPU
AOT execution. `VERNON_ENABLE_CUDA_RUNTIME` dynamically loads the
CUDA Driver API from `nvcuda.dll`/`libcuda.so.1`; no CUDA Toolkit or `nvcc`
installation is required. `VERNON_ENABLE_VULKAN_RUNTIME` dynamically loads the
system Vulkan loader and supports compute plus offscreen graphics pipeline
bundles. OpenGL and OpenGL ES use host-owned external contexts and native GLSL
for the matching profile; the runtime never creates a GLFW context.
`vd.register_external_opengl_context(...)` must be called before selecting
either external backend. Compiler capabilities remain independent of
runtime/device availability. `VERNON_ENABLE_PYTHON_BINDINGS` builds the single
nanobind `_native` compiler/runtime module when Python 3.11 and nanobind are
available.

External-context registration accepts the backend (`vd.opengl` or
`vd.opengles`), opaque host user-data address, `make_current` callback address,
`get_proc_address` callback address, and the actual context version. The host
must keep the context and callbacks alive for the runtime lifetime.

`VernonRuntime` links only its JSON parser and operating-system libraries.
Vulkan headers are compile-time-only; LLVM/MLIR, GLFW, the CUDA Toolkit, and
the Vulkan loader import library are outside its dependency closure.

Run the fractal directly on either GPU backend, or emit Metal source for use on
macOS:

```powershell
uv run python fractal.py --arch cuda
uv run python fractal.py --arch vulkan
uv run python fractal.py --emit-metal build/fractal.metal
```

Run the advanced graphics example with an indexed quad, instance attributes,
interactive `PICKING` specialization, and two named render targets:

```powershell
uv sync --extra examples
uv run python examples/advanced_pipeline.py --arch vulkan --frames 2 --headless `
  --output build/advanced-color.png `
  --id-output build/advanced-object-id.png
```

For one end-to-end example that combines shared definitions with three-stage
compute/vertex/fragment composition, three feature variants, indexed
instancing, named MRT outputs, and per-frame input/index/uniform rebinding:

```powershell
uv run python examples/complete_pipeline.py --arch vulkan --frames 3 --headless `
  --output build/complete-color.png `
  --id-output build/complete-object-id.png `
  --method-mlir build/complete-methods.mlir
```

Run the shared-definition example to exercise immutable host structs,
shared and device-only methods, NumPy intrinsics, CPU device parity, and
method lowering:

```powershell
uv run python examples/shared_struct_methods.py --arch cpu `
  --mlir build/shared-struct-methods.mlir
uv run python examples/shared_struct_methods.py --arch vulkan `
  --mlir build/shared-struct-methods.mlir
```

Persist a kernel for C or C++ loading with:

```powershell
vernon-compile-python kernel.py --entry scale -o build/scale.mlir
build/source/Release/vernon-compile.exe `
  --target cpu build/scale.mlir `
  --compute-bundle build/scale
```

OpenGL and OpenGL ES source versions are selectable per compilation. Omitting
the option uses the backend default:

```powershell
build/source/Release/vernon-compile.exe `
  --target opengl build/runtime.mlir `
  --glsl-version 330 `
  --output-dir build/opengl
```

To cook an OpenGL shader directly into a Vernon asset, provide its stable
asset ID and a normal output directory:

```powershell
build/source/Release/vernon-compile.exe `
  --target opengl build/runtime.mlir `
  --glsl-version 330 `
  --bundle assets/shaders/runtime `
  --asset-id shaders/runtime
```

The cooked directory contains `shader.json` plus readable generated vertex
and fragment GLSL files. The manifest records schema-versioned reflection,
source dependency hashes, the module hash, and an explicit
entry/stage/artifact table.
Mount the containing asset root in Vernon, then load `Shader` asset
`shaders/runtime`.

For variant families, author `*.shader-module.json` and
`*.shader-pipeline.json`, then cook every explicitly included feature key:

```powershell
uv run --frozen vernon-cook-shader `
  examples/variant_mesh.shader-pipeline.json `
  --asset-root examples `
  --compiler build/source/Release/vernon-compile.exe `
  --target opengl `
  -o build/variant_mesh_asset
```

The resulting `shader.json` maps exact canonical feature keys to shared stage
artifacts. For the four `INSTANCE`/`SKIN` combinations, four specialized
vertex files share one unchanged fragment file. Vernon rejects missing
variants rather than falling back.

`feature("NAME")`, `When[FEATURE, T]`, `if FEATURE`, and `if not FEATURE` are
specialized before type checking. Interface locations are inferred from
declaration order and type span before specialization; disabled fields reserve
their ranges, keeping surviving locations stable across variants. Explicit
locations remain available and overlapping ranges are diagnosed.

Reflected resource binding and material `(set, binding)` values remain the next
asset-pipeline milestone.

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
