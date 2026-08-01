# VernonDSL 0.1.1a1 release notes

VernonDSL 0.1.1a1 is a Windows-first alpha developer preview. It is suitable
for evaluation and experimentation, not production deployment. APIs,
artifacts, and backend coverage may change before beta.

## Highlights

- Tensor-first Python frontend with shared graphics and compute semantics.
- CPU, CUDA, Vulkan, OpenGL, OpenGL ES, and DirectX compilation paths.
- Direct CPU compute dispatch and GPU-optional runtime execution.
- Offscreen graphics pipelines, host readback, PipelineAsset cooking, and
  bundled VernonRuntime CMake sources.
- Terrain and Mandelbulb headless showcases with deterministic presets and
  machine-readable acceptance results.

## Distribution and support contract

- Windows is the only CI platform and the only platform with prebuilt wheels.
- Wheels target supported CPython 3.11 through 3.14 versions.
- macOS source builds and CI provide an experimental Metal compute and
  offscreen graphics Runtime. It consumes cooked MSL bundles on Apple, but is
  not a stable wheel or GA capability.
- Graphics rendering is offscreen with host readback; VernonRuntime does not
  provide swapchain or window presentation.
- GPU tests and showcases may skip when the required hardware, loader, context,
  or driver is unavailable. Release results distinguish executed, skipped, and
  failed backends.
- Dynamic multi-rank `TensorView` layouts require AOT specialization.
- CPU graphics, CUDA image/sampler resources, f16/f64 vertex attributes, and
  non-relaxed atomic orderings are outside the supported alpha subset.
- Production-grade multi-frame GPU resource lifetime management is not
  provided.
- The released frontend remains language version 3. Language v4 is a roadmap
  target and will not be declared until all required acceptance gates pass.

## Development and installation

Source checkouts use a dependency-only uv environment. Set `PYTHONPATH` to the
repository's `python` directory and run commands with
`uv run --frozen --no-sync`. Built wheels install the package and the
`vernon-compile-python` and `vernon-cook-pipeline` console scripts normally.

## Acceptance evidence

The release commit must pass Ruff, Python tests and coverage gates, native
CTest, ASan, wheel build, `twine check`, installed-wheel CPU dispatch and
frontend compilation, and GPU-optional dual-showcase acceptance.

Local acceptance on 2026-07-30 produced the following results:

- 340 Python tests and 349 subtests passed.
- Language/frontend line coverage was 91.81%; inference/type-parser branch
  coverage was 85.04%.
- Native CTest passed 135/135, with CUDA image/sampler lifetime and OpenGL
  synchronization recorded as environment-dependent skips.
- AddressSanitizer CTest passed 83/83, with CUDA image/sampler lifetime skipped.
- Terrain and Mandelbulb executed successfully at 64x64 on Vulkan, DirectX 12,
  and OpenGL; no showcase backend was skipped.
- The CPython 3.11 Windows wheel passed `twine check`, installation into a
  clean virtual environment, CPU kernel dispatch and readback, frontend CLI
  compilation, bundled Runtime source lookup, and release-version comparison.

## Known limitations before beta

Linux and macOS CI, stable public API and deprecation policies, compatibility
guarantees for all version axes, broad GPU runtime coverage, and the remaining
compiler and language-v4 work are beta prerequisites. See
[`RELEASE_READINESS.md`](RELEASE_READINESS.md) for the detailed boundary.
