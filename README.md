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

The current Python frontend language version is 3. Entry points keep explicit
ABI annotations, while `@func` helpers may infer parameters and results from
their call sites. Python `int`/`float` mean `i32`/`f32`; safe numeric widening
and integer true division are deterministic across targets. Use
`vd.Vector([...])` and `vd.Matrix([...])` for inferred value construction.
The active specifications and reading order are in
[`specs/README.md`](specs/README.md). The language contract, including
migration rules, is [`specs/language/contract.md`](specs/language/contract.md).
Checked phases in the
[`language-v4 roadmap`](specs/language/future_language_roadmap.md) are already
implemented, but the frontend remains version 3 until every v4 acceptance gate
passes.

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
- SPIRV-Cross source artifacts for OpenGL, OpenGL ES, and Metal, plus
  Shader Model 6 DXIL for DirectX;
- `vernon-opt`, the Vernon MLIR pass driver.

### Target support matrix

Compiler capability means that the complete artifact pipeline is linked.
Runtime availability additionally depends on the operating system, driver,
device, and requested feature set.

| Target | Compiler artifact | Compiler stages | VernonRuntime execution | Required environment |
| --- | --- | --- | --- | --- |
| CPU | LLVM IR and relocatable object; in-process reference entry | Compute and numeric stage reference | Compute only | Host or supported cross-target toolchain |
| Vulkan | SPIR-V | Compute and graphics | Compute and offscreen graphics | Vulkan 1.1 loader and compatible device |
| OpenGL | Desktop GLSL | Compute and graphics | Compute and graphics | External or Python-owned context; compute requires 4.3 |
| OpenGL ES | ESSL | Compute and graphics | Compute and graphics | External context; compute requires 3.1 |
| CUDA | PTX | Compute only | Compute only | Compatible NVIDIA driver |
| DirectX | Shader Model 6 DXIL | Compute and graphics | Compute and offscreen graphics | Windows D3D12; DXC is required only while cooking |
| Metal | MSL source | Compute and graphics | Not implemented | External Metal integration |

AMD/ROCDL is not a public compiler target. Unsupported stages, target options,
and runtime/device combinations fail explicitly rather than falling back to a
different backend.

CUDA/NVPTX and CPU/LLVM are distinct branches; they are never routed through
SPIR-V. The CPU backend JIT-compiles the current graphics and compute numeric
subset, including buffer load/store and callback-based 2D texture sampling,
and exports guarded C ABI entry points for reference execution. DirectX uses
the pinned official DXC redistributable while cooking and executes pre-cooked
DXIL through the Windows-only D3D12 Runtime backend. CMake downloads the
hash-verified package by default; offline builds may provide
`VERNON_DXC_EXECUTABLE` or disable `VERNON_FETCH_DXC`.

That JIT is a compiler reference facility, not a deployable runtime format.
`VernonRuntime` accepts CPU native-library AOT bundles only.

## Build and test

This section is the canonical setup guide. The compiler build expects the
matching LLVM checkout at `llvm-project/` because Vernon TableGen definitions
use MLIR source files that are not installed.

Prerequisites are CMake, a C/C++ toolchain, Python 3.11, and
[uv](https://docs.astral.sh/uv/). Build and install LLVM/MLIR once:

```powershell
uv pip install --target llvm-project/nvidia-nvcc `
  nvidia-cuda-nvcc-cu12==12.9.86
$libdevice = Resolve-Path `
  llvm-project/nvidia-nvcc/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
cmake -S llvm-project/llvm -B llvm-project/build `
  -DLLVM_ENABLE_PROJECTS="mlir;lld" `
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64;NVPTX;AMDGPU" `
  -DMLIR_NVVM_EMBED_LIBDEVICE=ON `
  -DMLIR_NVVM_LIBDEVICE_PATH="$libdevice" `
  -DCMAKE_INSTALL_PREFIX="$PWD/llvm-project/install"
cmake --build llvm-project/build --config Release --target install --parallel 4
```

Install the locked Python environment, including nanobind for the native
module, then configure, build, and test VernonDSL from the repository root:

```powershell
uv sync --extra build
uv run cmake -S . -B build `
  -DMLIR_DIR="$PWD/llvm-project/install/lib/cmake/mlir"
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
```

On a single-configuration Linux or macOS generator, use the equivalent:

```bash
uv pip install --target llvm-project/nvidia-nvcc \
  nvidia-cuda-nvcc-cu12==12.9.86
LIBDEVICE="$PWD/llvm-project/nvidia-nvcc/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc"
cmake -S llvm-project/llvm -B llvm-project/build \
  -DLLVM_ENABLE_PROJECTS="mlir;lld" \
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64;NVPTX;AMDGPU" \
  -DMLIR_NVVM_EMBED_LIBDEVICE=ON \
  -DMLIR_NVVM_LIBDEVICE_PATH="$LIBDEVICE" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/llvm-project/install"
cmake --build llvm-project/build --target install --parallel 4
uv sync --extra build
uv run cmake -S . -B build \
  -DMLIR_DIR="$PWD/llvm-project/install/lib/cmake/mlir" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
```

CMake writes `_native`, `_gl_context`, and their shared-library dependencies
to `python/vernon_dsl/`, which is the sole development import location.
`uv sync` installs dependencies but does not create an editable wheel; use
`uv build` explicitly when producing a release wheel. Run Python commands
through `uv run --frozen` with `python/` on `PYTHONPATH` when they are not
launched by CTest.

### Baseline benchmarks

The baseline runner emits JSON and uses fixed repository fixtures. Record the
commit, build configuration, target, driver/device, and command with every
result. CPU scenarios are the portable baseline:

```powershell
$env:PYTHONPATH = "$PWD/python"
uv run --frozen --no-sync python scripts/benchmark_baseline.py frontend --iterations 10
uv run --frozen --no-sync python scripts/benchmark_baseline.py native --target vulkan --iterations 10
uv run --frozen --no-sync python scripts/benchmark_baseline.py cook --target cpu --iterations 5
uv run --frozen --no-sync python scripts/benchmark_baseline.py kernel --arch cpu --iterations 20
```

The native scenario repeatedly compiles a fixed medium fragment shader and
isolates the owned MLIR compiler/artifact path from Python frontend time. Run
the same kernel scenario with `--arch cuda`, `vulkan`, `opengl`,
`opengles`, or `directx` on an available device. It reports cold
compile-and-dispatch, warm dispatch, and upload/dispatch/readback separately.
The multi-pass GPU baseline uses a fixed headless ocean workload:

```powershell
uv sync --extra build --extra examples --frozen
uv run --frozen --no-sync python scripts/benchmark_baseline.py showcase `
  --arch vulkan --frames 60 --grid 64 --size 256
```

Showcase time includes startup and compilation by design. Use an external GPU
timeline profiler when separating queue wait and device execution.

### Windows CI

GitHub Actions runs the supported CI configuration on `windows-2022` with
Visual Studio 17 2022. Runtime and style checks use Python 3.11.9, while wheel
builds and fresh-environment tests cover Python 3.11 through 3.14. The workflow
installs `uv` explicitly; it does not rely on software inherited from the
runner image.

The Runtime and style jobs do not check out or build LLVM. The Compiler job
builds a reduced `mlir;lld` LLVM installation on the first run, then caches the
installation by the pinned `llvm-project` submodule revision. It obtains
NVIDIA's redistributable `libdevice.10.bc` from the pinned
`nvidia-cuda-nvcc-cu12` package and embeds it in MLIR's NVVM target library, so
the published compiler wheel does not require a CUDA Toolkit installation.
Consequently, the first run after changing the LLVM revision, libdevice
version, or cache recipe is expected to be much slower. Later runs reuse the
matching installation.

Configure the Runtime subproject directly without LLVM/MLIR:

```powershell
cmake -S source/lib/runtime -B runtime_build
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

## Run an explicit compute pipeline

The `@kernel` frontend compiles to a compute pipeline with an explicit global
grid and runtime-owned contiguous Tensors. Builtin arguments are synthesized
and omitted from the call:

```python
from typing import Annotated
import vernon_dsl as vd

vd.init(arch=vd.cpu)  # vd.cuda and vd.vulkan use the same LoadedPipeline API
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
the first specialized call, compile a target artifact, cache a native
`LoadedPipeline`, and invoke it through `VernonRuntime`. CPU execution retains the owning
in-process compiler result and loads its JIT entry directly into the runtime;
`@kernel` does not invoke a compiler subprocess or create a temporary compute
bundle. CPU entries use RuntimeCpuProvider; GPU artifacts use
VernonRuntimeRHIAdapter. Both execute through RuntimeCore.

The `source/lib/runtime/` subproject builds the standalone `VernonRuntime` C
API with CPU AOT execution. `VERNON_ENABLE_CUDA_RUNTIME` dynamically loads the
CUDA Driver API from `nvcuda.dll`/`libcuda.so.1`; no CUDA Toolkit or `nvcc`
installation is required. `VERNON_ENABLE_VULKAN_RUNTIME` dynamically loads the
system Vulkan loader and supports compute plus offscreen graphics pipeline
bundles. OpenGL and OpenGL ES use the AHI external-context API and native GLSL
for the matching profile. Python wheels include a separate `_gl_context`
extension that owns a hidden GLFW context, so `vd.init(arch=vd.opengl)` runs
directly. Vernon Engine can continue to call
`vd.register_external_opengl_context(...)` to use its existing context.
`VERNON_ENABLE_DIRECTX12_RUNTIME` builds the Windows D3D12 compute/offscreen
graphics backend and links only system D3D12/DXGI libraries; runtime
deployments do not require DXC or its DLLs.
Compiler capabilities remain independent of runtime/device availability.
`VERNON_ENABLE_PYTHON_BINDINGS` builds the `_native` compiler/runtime module;
full Python builds also include `_gl_context` by default.

External-context registration accepts the backend (`vd.opengl` or
`vd.opengles`), opaque host user-data address, `make_current` callback address,
`get_proc_address` callback address, and the actual context version. The host
must keep the context and callbacks alive for the runtime lifetime.

`VernonRuntime` links only its JSON parser and operating-system libraries.
Vulkan headers are compile-time-only; LLVM/MLIR, GLFW, the CUDA Toolkit, and
the Vulkan loader import library are outside its dependency closure. GLFW is
linked only by the optional Python context-owner extension and is never
discovered by runtime-only/AHI builds.

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

Run the compute-driven showcase demos. Both record simulation, dynamic mesh
generation, shadow rendering, and the PBR main pass into one `ExecutionGraph`,
so buffer and depth-image barriers are inferred automatically:

```powershell
uv sync --extra examples

# Conservative hydraulic erosion, triplanar rock, wet runoff, and sunset PBR.
uv run python examples/dynamic_terrain_erosion.py --arch vulkan --frames 180 `
  --grid 224 --extent 13 --output build/dynamic-terrain-erosion.png

# Ping-pong finite-difference waves and compute-generated PBR mesh data.
uv run python examples/dynamic_ocean.py --arch vulkan --frames 180 `
  --grid 192 --extent 14 --output build/dynamic-ocean.png
```

Use `--headless` for automated screenshots. `--arch directx` and
`--arch opengl` exercise the same graph on their available Windows backends.
The simulations use one writer per vertex/cell and do not require atomics or
workgroup barriers. The erosion demo reconstructs neighboring runoff from the
previous state before rebuilding terrain normals and material channels.

The end-to-end example invokes separate compute and graphics programs while
sharing three feature variants, indexed instancing, named MRT outputs, and
per-frame input/index/uniform rebinding:

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

Emit a relocatable CPU object bundle with:

```powershell
vernon-compile-python kernel.py --entry scale -o build/scale.mlir
build/source/Release/vernon-compile.exe `
  --target cpu build/scale.mlir `
  --compute-bundle build/scale
```

On Windows this writes `module.obj`; ELF and Mach-O targets write `module.o`.
The bundle records the target triple, object format, stable exported wrapper,
size, and SHA-256. It is intended for static application linking, not direct
dynamic loading. To cross-compile an iOS object:

```powershell
build/source/Release/vernon-compile.exe `
  --target cpu build/scale.mlir `
  --target-triple arm64-apple-ios17.0 `
  --compute-bundle build/scale-ios
```

`--host-runtime-bundle` may be added only for immediate desktop execution. It
uses the LLD driver embedded in `VernonDSLCompiler` to finalize the host object
as a temporary DLL/so/dylib; persistent assets should retain the object.

OpenGL and OpenGL ES source versions are selectable per compilation. Omitting
the option uses the backend default:

```powershell
build/source/Release/vernon-compile.exe `
  --target opengl build/runtime.mlir `
  --glsl-version 330 `
  --output-dir build/opengl
```

Declare persistent assets beside their stage functions. The declaration is
read from the source AST and never imports or executes the module. The target
PipelineAsset API is:

```python
mesh_asset = vd.pipeline_asset(
    id="pipeline/mesh",
    program=(mesh_vertex, mesh_fragment),
    variants=((), (INSTANCE,), (SKIN,), (INSTANCE, SKIN)),
)
```

`program=` is either one compute Kernel or a non-empty tuple of graphics
stages. Graphics stage kinds come from their decorators, allowing the tuple
topology to grow beyond vertex-plus-fragment without changing the asset shape.
Target architecture and options are supplied by the cooker rather than source.
DirectX defaults to Shader Model 6.0 DXIL; `--hlsl-shader-model 60` selects the
runtime-compatible profile. Older HLSL source bundles are a broken contract
and must be re-cooked. Metal bundles remain compiler outputs only.

Cook the named declaration in process for one target:

```powershell
uv run --frozen vernon-cook-pipeline `
  examples/variant_mesh.py:mesh_asset `
  --target opengl `
  -o build/variant_mesh_asset
```

For a compute PipelineAsset, use the same cooker with `--target cpu`:

```powershell
uv run --frozen vernon-cook-pipeline `
  python/tests/pipeline_asset_fixture.py:scale_asset `
  --target cpu `
  -o build/cpu_scale
```

The cooker result contains `cpu_scale.pipeline.json` and a
content-addressed `artifacts/<sha256>.obj` or `.o`. Consumers should parse the
JSON manifest and link the referenced object; the cooker does not generate an
executable CMake fragment.

The `variant_mesh_asset.pipeline.json` maps exact canonical feature keys
to shared stage artifacts. For the four `INSTANCE`/`SKIN` combinations, four
specialized vertex files share one unchanged fragment file by content hash.
Vernon rejects missing variants rather than falling back.

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
Metal source (`.metal`), DirectX DXIL (`.dxil`), CUDA PTX (`.ptx`), and CPU
LLVM IR (`.ll`).

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
