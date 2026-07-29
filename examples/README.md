# VernonDSL examples

## Visual showcases

Install the optional presenter dependency before running either showcase:

```powershell
uv sync --extra examples
```

Vulkan is the preferred backend:

```powershell
uv run python examples/terrain_showcase.py --arch vulkan --preset showoff `
  --headless --output build/terrain.png --result-json build/terrain.json `
  --animation-output build/terrain.webp

uv run python examples/mandelbulb_showcase.py --arch vulkan --preset showoff `
  --headless --output build/mandelbulb.png --result-json build/mandelbulb.json `
  --animation-output build/mandelbulb.webp
```

Use `--preset smoke` for a small, quick backend check. Omit `--headless` to
display frames in an OpenCV window; Escape and Q close it. On Windows,
`--arch directx` and `--arch opengl` are fallback paths. Rendering is offscreen
on every backend, and each displayed frame is read back to the host. The
optional `--animation-output` records showcase frames as a looping animated
WebP. `fractal.py` accepts the same option for its compute-rendered Julia set.

Terrain demonstrates texture-driven fractal noise, ray marching, finite-
difference normals, soft shadows, ambient occlusion, fog, and host-driven
quality uniforms. Mandelbulb demonstrates bounded control flow, inverse
trigonometric intrinsics, ray marching, soft shadows, ambient occlusion, and
host-driven camera uniforms.

The scripts print one JSON result containing timing, graph, image shape, alpha,
brightness, and variance fields. Empty, transparent, or effectively uniform
screenshots fail with a clear error.

## Attribution

- `shader_lib/terrain.py` is adapted from
  [kevinroast/webglshaders](https://github.com/kevinroast/webglshaders),
  MIT License, Copyright (c) 2015 Kevin Roast.
- `shader_lib/mandelbulb.py` is adapted from
  [matt-k-wong/WebGL-Mandelbulb](https://github.com/matt-k-wong/WebGL-Mandelbulb),
  MIT License, Copyright (c) 2026.

The copyright and permission notices from those MIT licenses must be retained
when substantial portions of these adapted shaders are redistributed.

## Advanced compute and PBR examples

`dynamic_ocean.py` and `dynamic_terrain_erosion.py` remain available as larger
examples of compute simulation, dynamic mesh generation, shadows, PBR/IBL, and
multi-pass resource scheduling.
