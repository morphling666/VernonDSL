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

WIDTH = 640
HEIGHT = 320


@vd.func
def complex_square(z: vd.Vector[vd.f32, 2]) -> vd.Vector[vd.f32, 2]:
    return vd.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])


@vd.kernel(workgroup_size=(16, 16, 1))
def paint(
    pixels: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    time: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    y = gid[1]
    if x < WIDTH and y < HEIGHT:
        c = vd.Vector([-0.8, vd.cos(time) * 0.2])
        z = vd.Vector(
            [
                (vd.f32(x) / vd.f32(HEIGHT) - 1.0) * 2.0,
                (vd.f32(y) / vd.f32(HEIGHT) - 0.5) * 2.0,
            ]
        )
        iterations = 0
        while vd.norm(z) < 20.0 and iterations < 50:
            z = complex_square(z) + c
            iterations += 1
        pixels[y, x] = 1.0 - vd.f32(iterations) * 0.02


vd.init(arch=vd.cuda)
pixels = vd.storage.zeros(dtype=vd.f32, shape=(HEIGHT, WIDTH))
paint(pixels, 0.0, grid=(WIDTH, HEIGHT, 1))
image = pixels.to_numpy()
```

The current stable release is `0.1.1`. It supports CPython 3.11 through 3.14
on Windows x64, Linux x64, and Apple Silicon macOS. The frontend remains
language version 3 while the v4 roadmap is developed. See the
[`0.1.1 release notes`](https://github.com/morphling666/VernonDSL/blob/master/RELEASE_NOTES.md),
[`PUBLIC_API.md`](PUBLIC_API.md), and [`COMPATIBILITY.md`](COMPATIBILITY.md)
for the supported surface and compatibility contract.

## Demos

<table align="center" width="640">
  <tr>
    <td colspan="2" align="center">
      <img
        src="https://raw.githubusercontent.com/morphling666/VernonDSL/master/examples/assets/fractal-showcase.webp"
        alt="VernonDSL Julia set"
        width="620"
      >
      <br>
      <strong>Julia Set</strong>
      <br>
      <sub>Tensor compute, structured control flow, and device readback</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img
        src="https://raw.githubusercontent.com/morphling666/VernonDSL/master/examples/assets/terrain-showcase.webp"
        alt="VernonDSL ray-marched terrain"
        width="300"
      >
      <br>
      <strong>Ray-marched Terrain</strong>
    </td>
    <td width="50%" align="center">
      <img
        src="https://raw.githubusercontent.com/morphling666/VernonDSL/master/examples/assets/mandelbulb-showcase.webp"
        alt="VernonDSL animated Mandelbulb"
        width="300"
      >
      <br>
      <strong>Animated Mandelbulb</strong>
    </td>
  </tr>
</table>

The Julia Set showcases tensor compute. Terrain and Mandelbulb exercise
real-time fragment pipelines, ray marching, texture sampling, mathematical
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

Prebuilt wheels are provided for Windows x64, Linux x64, and Apple Silicon
macOS 15 or newer for CPython 3.11 through 3.14:

```powershell
py -m pip install vernon-lang==0.1.1
```

VernonDSL 0.1.1 is wheel-only. Intel macOS, source distributions, PyPy, and
other Python versions are not published.

Verify the installation:

```powershell
py -c "from importlib.metadata import version; print(version('vernon-lang'))"
vernon-compile-python --help
vernon-cook-pipeline --help
```

The wheel contains the Vernon compiler, CPU Runtime, and the native backend
implementations supported by its platform. It does not bundle GPU drivers.
Runtime availability therefore depends on the selected backend:

- CPU: compute reference execution with no additional system dependency;
- CUDA: compute with a compatible NVIDIA driver;
- Vulkan: compute and offscreen graphics with a loader and vendor ICD;
- DirectX 12: compute and offscreen graphics using Windows and its GPU driver;
- OpenGL: compute and graphics through a compatible system context;
- OpenGL ES: compute and graphics through a compatible owned or external
  context;
- Metal: compute and offscreen graphics through the system framework on
  supported Apple Silicon Macs, with no additional loader.

Cooked MSL bundles are consumed by the Runtime on Apple. Metal presentation and
swapchain management are outside the `0.1.1` contract. Argument-buffer
pipelines fail explicitly when the selected device cannot provide the required
tier or encoder.

CPU graphics, CUDA images and samplers, f16/f64 vertex attributes,
non-relaxed atomics, asynchronous dispatch, and multiple frames in flight are
outside the supported `0.1.1` subset. See
[`RELEASE_NOTES.md`](RELEASE_NOTES.md) for the complete release contract.

### Optional Vulkan setup

#### macOS

Metal is the zero-install GPU backend on macOS. Vulkan is optional because
macOS does not provide it natively. To use Vulkan, install the Khronos loader
and MoltenVK ICD with Homebrew:

```bash
brew install molten-vk vulkan-loader
```

Vernon also searches Homebrew locations under `/opt/homebrew` and `/usr/local`.
`VERNON_VULKAN_LOADER` may name the installed Khronos loader dylib; do not point
it directly at MoltenVK.

#### Linux

Linux users need only the driver stack for the backend they select. For Vulkan
on Ubuntu or Debian, install the loader and an appropriate vendor ICD. Mesa
provides Intel, AMD, and software Vulkan drivers:

```bash
sudo apt-get update
sudo apt-get install --yes libvulkan1 mesa-vulkan-drivers
```

NVIDIA systems should install the matching proprietary driver instead of
relying on Mesa for the device ICD.

#### Loader discovery

Vulkan is discovered when the Runtime creates a device. Vernon tries
`VERNON_VULKAN_LOADER`, a loader under `VULKAN_SDK`, and the platform loader
name.

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

The output contains the canonical pipeline-14 `*.pipeline.json` manifest and
content-addressed files under `artifacts/`. Depending on the target, artifacts
are SPIR-V, GLSL/ESSL, DXIL, PTX, Metal source, LLVM IR, or relocatable CPU
objects. A differentiated asset adds the optional root `autodiff` object to the
same manifest schema. CPU cooking also emits a `.o`/`.obj` plus generated
static-registration `.c` and `.h` sources; there is no separate `compute.json`
bundle. Cooked Metal bundles contain MSL consumed by the Runtime on Apple.
Missing variants and unsupported target combinations fail explicitly rather
than silently falling back.

Pipeline sources, MLIR, manifests, shader artifacts, native objects, and caches
are executable input, not sandboxed data. Do not compile or load untrusted
artifacts without isolation; see [`SECURITY.md`](SECURITY.md).

When working from a source checkout where `tool.uv.package = false`, invoke the
cooker as a module:

```powershell
$env:PYTHONPATH = "$PWD/python"
uv run --frozen --no-sync python -m vernon_dsl.pipeline_asset_cli `
  examples/variant_mesh.py:mesh_asset --target vulkan -o build/variant_mesh
```

## Build from source

Source and wheel builds are continuously tested on Windows, Linux, and macOS.
Required tools:

- Git, CMake, and a platform C++ toolchain;
- Visual Studio 2022 C++ tools and the Windows SDK on Windows;
- Python 3.11 through 3.14 and [uv](https://docs.astral.sh/uv/);
- the repository's pinned `llvm-project` submodule.

For a full Ubuntu or Debian source-build and graphics-test environment, install
the same native packages used by Linux CI:

```bash
sudo apt-get update
sudo apt-get install --yes \
  ninja-build patchelf pkg-config \
  libvulkan1 mesa-vulkan-drivers \
  libgl1-mesa-dev libegl1-mesa-dev \
  libwayland-dev libxkbcommon-dev wayland-protocols \
  xorg-dev xvfb
```

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

CMake runs the frozen `uv sync`, uses MLIR from `llvm-project/install`,
selects the synchronized Python interpreter and host
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

On macOS, set `MACOSX_DEPLOYMENT_TARGET=15.0` for the supported arm64 wheel,
matching the release CI.

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
ctest --test-dir osx_build -C Release --output-on-failure
```

Use `linux_build` or `windows_build` instead when following the corresponding
platform build example above.

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

The post-`0.1.1` roadmap includes:

1. finish and accept the language-v4 contract, including first-order
   pure-function autodiff;
2. broaden repeatable hardware-backed GPU acceptance;
3. design the ABI change required for asynchronous dispatch and deferred
   multi-frame resource reclamation;
4. replace the remaining temporary compiler bridges tracked in the completion
   roadmap.

Release support and security reporting are documented in
[`SUPPORT.md`](SUPPORT.md) and [`SECURITY.md`](SECURITY.md). Published wheel,
checksum, SBOM, and provenance requirements are documented in
[`RELEASE_ARTIFACTS.md`](RELEASE_ARTIFACTS.md).

Related design and release documents:

- [Project roadmap](https://github.com/morphling666/VernonDSL/blob/master/specs/roadmap.md)
- [Language v4 roadmap](https://github.com/morphling666/VernonDSL/blob/master/specs/language/future_language_roadmap.md)
- [Release readiness](https://github.com/morphling666/VernonDSL/blob/master/RELEASE_READINESS.md)
- [Compiler and runtime design](https://github.com/morphling666/VernonDSL/blob/master/specs/compiler/design.md)

## License

VernonDSL is licensed under the
[Apache License 2.0](https://github.com/morphling666/VernonDSL/blob/master/LICENSE).
Third-party notices for adapted showcase shaders are retained under
`examples/`.
