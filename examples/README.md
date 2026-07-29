# VernonDSL examples

## Visual showcases

Install the optional presenter dependency before running either showcase:

```powershell
uv sync --extra examples
```

Vulkan is the preferred backend:

```powershell
uv run python examples/aurora_showcase.py --arch vulkan --preset showoff `
  --headless --output build/aurora.png --result-json build/aurora.json

uv run python examples/mandelbulb_showcase.py --arch vulkan --preset showoff `
  --headless --output build/mandelbulb.png --result-json build/mandelbulb.json
```

Use `--preset smoke` for a small, quick backend check. Omit `--headless` to
display frames in an OpenCV window; Escape and Q close it. On Windows,
`--arch directx` and `--arch opengl` are fallback paths. Rendering is offscreen
on every backend, and each displayed frame is read back to the host.

Aurora demonstrates a two-pass `ExecutionGraph`, render-to-texture sampling,
procedural animation, blur, and temporal dithering. Mandelbulb demonstrates
bounded control flow, inverse trigonometric intrinsics, ray marching, soft
shadows, ambient occlusion, and host-driven camera uniforms.

The scripts print one JSON result containing timing, graph, image shape, alpha,
brightness, and variance fields. Empty, transparent, or effectively uniform
screenshots fail with a clear error.

## Attribution

- `shader_lib/aurora.py` is adapted from
  [jagajaga/coaurora](https://github.com/jagajaga/coaurora), MIT License,
  Copyright (c) 2026 Arseniy Seroka.
- `shader_lib/mandelbulb.py` is adapted from
  [matt-k-wong/WebGL-Mandelbulb](https://github.com/matt-k-wong/WebGL-Mandelbulb),
  MIT License, Copyright (c) 2026.

The copyright and permission notices from those MIT licenses must be retained
when substantial portions of these adapted shaders are redistributed.

## Advanced compute and PBR examples

`dynamic_ocean.py` and `dynamic_terrain_erosion.py` remain available as larger
examples of compute simulation, dynamic mesh generation, shadows, PBR/IBL, and
multi-pass resource scheduling.
