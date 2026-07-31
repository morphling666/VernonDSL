# VernonDSL

VernonDSL is a tensor-first Python language and MLIR compiler for GPU graphics
and compute. A restricted, statically typed Python frontend lowers through
shared semantic IR into target-specific CPU, CUDA, Vulkan, OpenGL, DirectX, and
Metal artifacts.

For an end-to-end explanation of the language, compiler backends, Runtime/RHI,
ExecutionGraph, and offline cooking model, see
[`ARCHITECTURE.md`](ARCHITECTURE.md).

```python
from typing import Annotated

import vernon_dsl as vd

vd.init(arch=vd.cpu)


@vd.kernel(workgroup_size=(8, 1, 1))
def scale(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    factor: vd.f32,
    gid: Annotated[
        vd.Tensor[vd.u32, (3,)],
        vd.builtin("global_invocation_id"),
    ],
) -> None:
    output[gid[0]] = vd.f32(gid[0]) * factor


output = vd.storage.zeros(dtype=vd.f32, shape=(1024,))
scale(output, 2.0, grid=(1024, 1, 1))
values = output.to_numpy()
```

The current release line is `0.1.1a1`, a Windows-first alpha developer
preview. The released frontend remains language version 3 while the v4
acceptance roadmap is completed. See the
[`0.1.1a1 release notes`](https://github.com/morphling666/VernonDSL/blob/master/RELEASE_NOTES.md)
for the support contract and known
limitations.

## Demos

![VernonDSL ray-marched terrain](https://raw.githubusercontent.com/morphling666/VernonDSL/master/examples/assets/terrain-showcase.webp)

![VernonDSL animated Mandelbulb](https://raw.githubusercontent.com/morphling666/VernonDSL/master/examples/assets/mandelbulb-showcase.webp)

The Terrain and Mandelbulb showcases exercise real-time fragment pipelines,
ray marching, structured control flow, texture sampling, mathematical
intrinsics, offscreen rendering, and host readback.

From a source checkout, install the example dependency and run an interactive
60 FPS presentation:

```powershell
$env:PYTHONPATH = "$PWD/python"
uv sync --extra examples --frozen

uv run --frozen --no-sync python examples/terrain_showcase.py `
  --arch vulkan --preset showoff --fps 60

uv run --frozen --no-sync python examples/mandelbulb_showcase.py `
  --arch vulkan --preset showoff --fps 60
```

Press Escape or Q to exit. Use `--arch directx` or `--arch opengl` on Windows
when Vulkan is unavailable. The presenter targets the requested frame rate;
actual throughput depends on the GPU, driver, image size, and readback cost.
Use `--preset smoke --headless` for fast acceptance checks.

## Install from PyPI

Prebuilt wheels are currently provided for Windows and supported CPython 3.11
through 3.14:

```powershell
py -m pip install vernon-lang==0.1.1
```

Verify the installation:

```powershell
py -c "from vernon_dsl._versions import RELEASE_VERSION; print(RELEASE_VERSION)"
vernon-compile-python --help
vernon-cook-pipeline --help
```

Runtime availability depends on installed drivers and hardware:

- CPU: compute reference execution;
- CUDA: compute on a compatible NVIDIA driver;
- Vulkan: compute and offscreen graphics;
- DirectX 12: compute and offscreen graphics on Windows;
- OpenGL: compute and graphics through a Python-owned or external context;
- Metal: source artifact generation only; no Vernon runtime.

Vulkan is discovered when the runtime creates a device. Vernon tries
`VERNON_VULKAN_LOADER`, a loader under `VULKAN_SDK`, the platform loader name,
and, on macOS, Homebrew locations under `/opt/homebrew` and `/usr/local`.
`VERNON_VULKAN_LOADER` may name a specific Khronos loader DLL, shared object, or
dylib; do not point it at MoltenVK directly. macOS users can install the loader
and MoltenVK ICD with:

```bash
brew install molten-vk vulkan-loader
```

`VERNON_ENABLE_VULKAN_RUNTIME` controls whether Vulkan support is included in
the build. It does not indicate that a loader, ICD, or usable device is present
on the machine running Vernon. Device creation reports the attempted loader
locations and Vulkan initialization error when runtime discovery fails.

## Cook deployable artifacts

Declare a persistent asset beside its shader stages:

```python
mesh_asset = vd.pipeline_asset(
    id="pipeline/mesh",
    program=(mesh_vertex, mesh_fragment),
    variants=((), (INSTANCE,), (SKIN,), (INSTANCE, SKIN)),
)
```

Cook it for a deployment target without importing or executing the source
module:

```powershell
vernon-cook-pipeline examples/variant_mesh.py:mesh_asset `
  --target vulkan `
  -o build/variant_mesh
```

For a CPU compute asset:

```powershell
vernon-cook-pipeline python/tests/pipeline_asset_fixture.py:scale_asset `
  --target cpu `
  -o build/cpu_scale
```

The output contains a versioned `*.pipeline.json` manifest and
content-addressed files under `artifacts/`. Depending on the target, artifacts
are SPIR-V, GLSL/ESSL, DXIL, PTX, Metal source, LLVM IR, or relocatable CPU
objects. Metal bundles are compiler outputs only. Missing variants and
unsupported target combinations fail explicitly rather than silently falling
back.

When working from a source checkout where `tool.uv.package = false`, invoke the
cooker as a module:

```powershell
$env:PYTHONPATH = "$PWD/python"
uv run --frozen --no-sync python -m vernon_dsl.pipeline_asset_cli `
  examples/variant_mesh.py:mesh_asset --target vulkan -o build/variant_mesh
```

## Build from source

The alpha build is continuously tested on Windows with Visual Studio 2022.
The LLVM configuration helper also supports Linux and macOS. Required tools:

- Git, CMake, and a platform C++ toolchain;
- Visual Studio 2022 C++ tools and the Windows SDK on Windows;
- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/);
- the repository's pinned `llvm-project` submodule.

Initialize the repository and build the pinned LLVM/MLIR installation once:

```shell
git submodule update --init --depth 1 llvm-project
python scripts/configure_llvm.py --build
```

The helper selects the LLVM target for the host architecture and reuses the
generator recorded in an existing build directory. Otherwise it selects Visual
Studio 2022 on Windows, Ninja when available, or Unix Makefiles. CUDA/NVPTX is
enabled when `libdevice.10.bc` is found through `CUDA_PATH`, `CUDA_HOME`,
`CUDAToolkit_ROOT`, `CONDA_PREFIX`, `nvcc`, or the vendored
`llvm-project/nvidia-nvcc` directory. Use `--cuda off` to disable CUDA or
`--cuda on --libdevice PATH` to require it. Omit `--build` to configure only;
run `python scripts/configure_llvm.py --help` for all overrides.

Build VernonDSL:

On macOS:

```bash
mkdir osx_build
cd osx_build
cmake ..
cmake --build . --parallel
ctest --output-on-failure
```

On Linux:

```bash
mkdir linux_build
cd linux_build
cmake ..
cmake --build . --parallel
ctest --output-on-failure
```

On Windows PowerShell:

```powershell
mkdir windows_build
cd windows_build
cmake ..
cmake --build . --config Release --parallel
ctest -C Release --output-on-failure
```

CMake runs the frozen `uv sync`, uses MLIR and LLD from
`llvm-project/install`, selects the synchronized Python interpreter and host
compiler architecture, and disables runtime backends unsupported by the target
platform and architecture. Single-configuration generators default to Release.
Tests and the staged-file formatting Git hook are enabled by default. Each
setting remains available as a `-D` override.

CMake places the development `_native` and `_gl_context` modules in
`python/vernon_dsl/`. Source-checkout Python commands therefore use
`PYTHONPATH=python` and `uv run --frozen --no-sync`.

Build a release wheel:

```powershell
uv build --python 3.11 --wheel --no-cache --clear
uvx --from twine twine check dist/*.whl
```

Clean wheel builds fetch pinned third-party CMake dependencies. They can take
several minutes on the first run.

## Run tests

Run the Python suite:

```powershell
$env:PYTHONPATH = "$PWD/python"
uv run --frozen --no-sync pytest python/tests -q
```

Run native tests:

```powershell
ctest --test-dir build -C Release --output-on-failure
```

Run the release coverage gates:

```powershell
$env:PYTHONPATH = "$PWD/python"
uv run --frozen --no-sync coverage erase
uv run --frozen --no-sync coverage run -m pytest python/tests -q
uv run --frozen --no-sync coverage json -o coverage.json
uv run --frozen --no-sync python scripts/check_python_coverage.py coverage.json
```

Run GPU-optional showcase acceptance:

```powershell
$env:PYTHONPATH = "$PWD/python"
uv run --frozen --no-sync python scripts/run_showcase_smoke.py `
  --output-dir build/showcase-smoke `
  --summary build/showcase-smoke/summary.json
```

Unavailable GPU backends are reported as explicit skips. Use `--required` when
every requested backend must execute.

## Run examples

All source examples use the same development environment:

```powershell
$env:PYTHONPATH = "$PWD/python"
uv sync --extra examples --frozen
```

Headless showcase output:

```powershell
uv run --frozen --no-sync python examples/terrain_showcase.py `
  --arch vulkan --preset showoff --headless `
  --output build/terrain.png --result-json build/terrain.json

uv run --frozen --no-sync python examples/mandelbulb_showcase.py `
  --arch vulkan --preset showoff --headless `
  --output build/mandelbulb.png --result-json build/mandelbulb.json
```

Compute-rendered Julia set:

```powershell
uv run --frozen --no-sync python examples/fractal.py --arch cuda
uv run --frozen --no-sync python examples/fractal.py --arch vulkan
uv run --frozen --no-sync python examples/fractal.py `
  --emit-metal build/fractal.metal
```

End-to-end compute and graphics pipeline:

```powershell
uv run --frozen --no-sync python examples/complete_pipeline.py `
  --arch vulkan --frames 3 --headless `
  --output build/complete-color.png `
  --id-output build/complete-object-id.png
```

More examples and their third-party attributions are documented in
[`examples/README.md`](https://github.com/morphling666/VernonDSL/blob/master/examples/README.md).

## Future roadmap

VernonDSL remains an alpha project. The main path toward beta is:

1. finish and accept the language-v4 contract, including first-order
   pure-function autodiff and remaining synchronization gates;
2. add Linux CI and verify source builds outside Windows;
3. define stable public API, deprecation, ABI, and cache compatibility
   policies;
4. broaden repeatable GPU runtime coverage and production resource-lifetime
   behavior;
5. replace the remaining temporary compiler bridges tracked in the completion
   roadmap.

Stable/GA additionally requires production-ready binary distribution,
cross-platform release CI, release automation, security reporting, and a
published support policy.

Detailed plans:

- [Language v4 roadmap](https://github.com/morphling666/VernonDSL/blob/master/specs/language/future_language_roadmap.md)
- [Completion roadmap](https://github.com/morphling666/VernonDSL/blob/master/specs/completion_roadmap.md)
- [Release readiness](https://github.com/morphling666/VernonDSL/blob/master/RELEASE_READINESS.md)
- [Compiler and runtime design](https://github.com/morphling666/VernonDSL/blob/master/specs/compiler/design.md)

## License

VernonDSL is licensed under the
[Apache License 2.0](https://github.com/morphling666/VernonDSL/blob/master/LICENSE).
Third-party notices for adapted showcase shaders are retained under
`examples/`.
