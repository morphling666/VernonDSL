# Minimal external engine

[Run the live WebAssembly demo](https://morphling666.github.io/VernonDSL/)

This C++ engine embeds two cooked bundles and displays them in one DPI-aware
GLFW window or WebGL canvas. The left panel runs `examples/fractal.py` through
one persistent CPU Program instance; the right panel runs one persistent
graphics Program instance. Every frame begins a fresh canonical Program
invocation and binds its current Values and graphics controls. Presentation is
a separate host/RHI operation, and the shared frame loop composites both
images with aspect-preserving blits.

Python and nanobind are used only by the host-side cooking step. Browser builds
perform no filesystem or dynamic-library access.

Install a VernonDSL wheel in the active Python environment before configuring
this example. CMake asks that host Python for
`python -m vernon_dsl.runtime_source --cmake-dir` and builds only the Runtime
sources delivered inside the wheel. It does not fall back to the VernonDSL
repository's `source/` tree. When multiple Python installations are present,
pass the wheel environment's interpreter as `-DPython_EXECUTABLE=...`.

```sh
python -m pip install path/to/vernon_lang-*.whl
```

For cross-compilation, install the wheel for the build host; its portable
Runtime source payload is compiled for the target toolchain.

## Desktop split screen

```sh
python -m vernon_dsl.program_asset_cli \
  examples/external_engine/fractal_pipeline.py:asset \
  --target cpu \
  -o fractal-desktop-cooked

python -m vernon_dsl.program_asset_cli \
  examples/shader_lib/mandelbulb.py:mandelbulb_asset \
  --target opengl --opengl-version 330 \
  -o mandelbulb-desktop-cooked

cmake -S examples/external_engine -B split-desktop-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DVERNON_EXTERNAL_ENGINE_CPU_COOKED_DIR="$PWD/fractal-desktop-cooked" \
  -DVERNON_EXTERNAL_ENGINE_GRAPHICS_COOKED_DIR="$PWD/mandelbulb-desktop-cooked"
cmake --build split-desktop-build --config Release
ctest --test-dir split-desktop-build -C Release --output-on-failure
./split-desktop-build/vernon-external-engine
```

The same CMake project and C++ sources build on Windows, macOS, and Linux.
`ctest` runs the common CPU Program path headlessly; the interactive executable
adds the GLFW/OpenGL presentation path.

## WebAssembly split screen

```sh
python -m vernon_dsl.program_asset_cli \
  examples/external_engine/fractal_pipeline.py:asset \
  --target cpu \
  --cpu-triple wasm32-unknown-emscripten \
  -o fractal-wasm-cooked

python -m vernon_dsl.program_asset_cli \
  examples/shader_lib/mandelbulb.py:mandelbulb_asset \
  --target opengles --opengl-version 300 \
  -o mandelbulb-wasm-cooked

emcmake cmake -S examples/external_engine -B split-wasm-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DVERNON_EXTERNAL_ENGINE_CPU_COOKED_DIR="$PWD/fractal-wasm-cooked" \
  -DVERNON_EXTERNAL_ENGINE_GRAPHICS_COOKED_DIR="$PWD/mandelbulb-wasm-cooked"
cmake --build split-wasm-build --config Release
ctest --test-dir split-wasm-build -C Release --output-on-failure
python3 -m http.server 8003 --directory split-wasm-build
```

Open <http://localhost:8003/>. Cooked GLSL ES stages are integrity-checked and
embedded alongside the CPU manifest and statically linked wasm object.

The CTest above runs the same executable's CPU-only headless Node smoke:

```sh
node split-wasm-build/vernon-external-engine.js
```

Cooking always runs with the desktop Python toolchain. Only the Runtime and
external engine are cross-compiled. Emscripten builds copy `browser.html` to
`index.html` next to the generated JavaScript and WebAssembly files.

The `External Engine Pages` workflow cooks both bundles and builds the browser
application on pull requests, pushes to `master`, and manual runs. Every run
executes the headless Node checksum and an explicitly installed headless Chrome
startup smoke. Non-PR runs also deploy the three static files to GitHub Pages.
The repository's Pages source must be set to **GitHub Actions**.
