# VernonDSL 0.1.2 release notes

VernonDSL 0.1.2 is the upcoming cross-platform release of the language,
compiler, offline cooker, and Program-based Runtime/RHI APIs.

## Highlights

- Tensor-first Python frontend with shared graphics and compute semantics.
- CPU, CUDA, Vulkan, OpenGL, OpenGL ES, and DirectX compilation paths.
- Direct CPU compute dispatch and GPU-optional runtime execution.
- Offscreen graphics pipelines, host readback, ProgramAsset cooking, and
  bundled VernonRuntime CMake sources.
- Pipeline 17 makes canonical Program plus ArtifactSystem the only execution
  package. Runtime topology, bindings, and autodiff signatures are resolved
  from Program rather than a parallel pipeline manifest.
- CPU automatic differentiation uses one structured ABI for direct and
  cooked execution, including dynamic/partially dynamic Storage shapes,
  reusable pullbacks, aggregate leaves, aliases, and zero extents.
- GPU and graphics automatic differentiation and graph-level pullbacks are
  deferred and unsupported. Ordinary non-AD GPU compute and graphics support
  is unchanged.
- CPU cooking emits that package, a relocatable `.o`/`.obj`, and generated
  static-registration `.c`/`.h` sources. The native compiler's former
  `--compute-bundle`/`compute.json` packaging interface has been removed.
- Terrain and Mandelbulb headless showcases with deterministic presets and
  machine-readable acceptance results.

## Distribution and support contract

- Wheels target CPython 3.11 through 3.14 on Windows x64, Linux x64, and
  Apple Silicon macOS. Intel macOS and source distributions are not shipped.
- CPU provides reference compute execution.
- CUDA provides compute and buffers on compatible NVIDIA drivers.
- Vulkan provides compute and offscreen graphics on supported drivers.
- DirectX 12 provides compute and offscreen graphics on Windows.
- OpenGL and OpenGL ES provide compute and graphics when a compatible owned or
  external context is available.
- Metal provides compute and offscreen graphics on supported Apple Silicon
  devices. Argument-buffer pipelines are rejected when the device cannot
  provide the required argument-buffer tier or encoder.
- Graphics rendering is offscreen with host readback; VernonRuntime does not
  provide swapchain or window presentation.
- Module calls and VJPs execute canonical Programs; direct kernels and graphics
  pipelines use their resolved one-node Program endpoint.
- Dynamic TensorView shape, stride, and offset are invocation data and do not
  require recompilation.
- CPU graphics, CUDA image/sampler resources, f16/f64 vertex attributes, and
  non-relaxed atomic orderings are outside the supported `0.1.2` subset.
- Concurrent dispatch and multiple frames in flight are not guaranteed in
  `0.1.2`; backends may complete submissions inline.
- The released frontend remains language version 3. Language v4 is a roadmap
  target.

## Development and installation

Source checkouts use a dependency-only uv environment. Set `PYTHONPATH` to the
repository's `python` directory and run commands with
`uv run --frozen --no-sync`. Built wheels install the package and the
`vernon-compile-python` and `vernon-cook-program` console scripts normally.

## Compatibility and lifecycle

This release uses compiler contract 13 and pipeline contract 17. Incompatible
artifacts are rejected rather than silently loaded. The stable API, ABI,
deprecation, cache, support, and security policies are documented in
[`PUBLIC_API.md`](PUBLIC_API.md), [`COMPATIBILITY.md`](COMPATIBILITY.md),
[`SUPPORT.md`](SUPPORT.md), and [`SECURITY.md`](SECURITY.md).

The exact release commit must pass native and Python tests plus CPython
3.11–3.14 wheel build, metadata, and clean-install gates on Linux, macOS, and
Windows. Windows CI additionally enforces formatting, Ruff, Python coverage,
and Runtime AddressSanitizer gates. The release workflow validates one
12-wheel payload, generates checksums and an SBOM, and binds provenance to the
tagged commit before publication. After PyPI publication, CPython 3.11 on each
supported operating system repeats the installed-wheel CPU dispatch/readback,
frontend, cooker, and Runtime-source layout smoke test.
