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

## Desktop split screen

```sh
cmake -S . -B host-build
cmake --build host-build --target vernon-dsl-native

PYTHONPATH=python .venv/bin/python -m vernon_dsl.program_asset_cli \
  examples/external_engine/fractal_pipeline.py:asset \
  --target cpu \
  -o fractal-desktop-cooked

PYTHONPATH="python:$PWD" .venv/bin/python -m vernon_dsl.program_asset_cli \
  examples/shader_lib/mandelbulb.py:mandelbulb_asset \
  --target opengl --opengl-version 330 \
  -o mandelbulb-desktop-cooked

cmake -S examples/external_engine -B split-desktop-build \
  -DVERNON_EXTERNAL_ENGINE_CPU_COOKED_DIR="$PWD/fractal-desktop-cooked" \
  -DVERNON_EXTERNAL_ENGINE_GRAPHICS_COOKED_DIR="$PWD/mandelbulb-desktop-cooked"
cmake --build split-desktop-build
./split-desktop-build/vernon-external-engine
```

## WebAssembly split screen

```sh
PYTHONPATH="python:$PWD" .venv/bin/python -m vernon_dsl.program_asset_cli \
  examples/external_engine/fractal_pipeline.py:asset \
  --target cpu \
  --cpu-triple wasm32-unknown-emscripten \
  -o fractal-wasm-cooked

PYTHONPATH="python:$PWD" .venv/bin/python -m vernon_dsl.program_asset_cli \
  examples/shader_lib/mandelbulb.py:mandelbulb_asset \
  --target opengles --opengl-version 300 \
  -o mandelbulb-wasm-cooked

emcmake cmake -S examples/external_engine -B split-wasm-build \
  -DVERNON_EXTERNAL_ENGINE_CPU_COOKED_DIR="$PWD/fractal-wasm-cooked" \
  -DVERNON_EXTERNAL_ENGINE_GRAPHICS_COOKED_DIR="$PWD/mandelbulb-wasm-cooked"
cmake --build split-wasm-build
python3 -m http.server 8003 --directory split-wasm-build
```

Open <http://localhost:8003/>. Cooked GLSL ES stages are integrity-checked and
embedded alongside the CPU manifest and statically linked wasm object.

The same executable retains a CPU-only headless Node smoke:

```sh
node split-wasm-build/vernon-external-engine.js
```

Cooking always runs with the desktop Python toolchain. Only the Runtime and
external engine are cross-compiled. Emscripten builds copy `browser.html` to
`index.html` next to the generated JavaScript and WebAssembly files.

The `External Engine Pages` workflow cooks both bundles, builds the browser
application, and deploys these three static files to GitHub Pages after pushes
to `master`. The repository's Pages source must be set to **GitHub Actions**.
