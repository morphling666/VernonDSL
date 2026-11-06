# Editor scenes

The Vernon Editor (`applications/robot_simulator`) draws its scene viewport
and its ImGui panels through VernonDSL graphics Programs on the OpenGL
backend. This report measures the main thread's own cost per frame in the
Editor's demo scenes, the regression benchmarks of
[`specs/runtime/execution_model.md`](../../specs/runtime/execution_model.md)
section 13.

## Provenance

- Source revision: VernonDSL `6fcb25d` and Vernon `90254343`, both with the
  uncommitted architecture v2 work.
- Platform: Windows 11 (10.0.26300), AMD Ryzen 9 9950X, NVIDIA GeForce
  RTX 4090, OpenGL 4.6, the Editor's default window, vsync off, Release.
- Command: `Editor.exe --no-vsync --frames 2400 --frame-stats [--demo
  <dog|bodies|planet>] [--simulate] [--orbit 10]`, run on the desktop
  through `scripts/run_on_desktop.ps1`.
- Metric: as in [`planet_walker_frame.md`](planet_walker_frame.md): per
  window of 300 frames, the mean frame time, the time the render graphs
  waited for earlier frames to retire, and the difference (host). The first
  window holds startup and is left out.

## Results

| Scene | Frame | Host (range of windows) |
|---|---:|---:|
| empty editor (panels only) | 0.60 ms | 0.58–0.60 ms |
| `dog`, editing | 0.62 ms | 0.59–0.70 ms |
| `dog`, simulating | 0.78 ms | 0.75–0.84 ms |
| `bodies`, editing | 0.68 ms | 0.62–0.71 ms |
| `bodies`, simulating | 3.76 ms | 2.72–3.45 ms |
| `planet`, standing | 4.48 ms | 1.62–1.73 ms |
| `planet`, orbiting at 10°/s | 4.50 ms | 2.07–2.74 ms |

- **The ImGui pass**, whose recording changes every frame, is within the
  0.6 ms of the empty editor, which also holds the panels' layout and the
  viewport; the target was a plan build path no slower than the old
  Execute's 0.7 ms.
- **Simulating `bodies`** adds about 2.2 ms of physics on the main thread,
  rigid bodies, particles and articulations stepping; drawing them costs
  what the editing case does.
- **`planet`** is PlanetWalker's terrain under the editor's panels: standing
  is PlanetWalker's 0.77 ms plus the panels and the hierarchy's scan of the
  world's placed entities. Orbiting changes the selected tiles as walking
  does.

## After step 5

The same command after section 16.3 step 5 of
[`architecture_v2.md`](../../specs/runtime/architecture_v2.md), with
`--device opengl` and `--device vulkan`, at VernonDSL `f840106` and Vernon
`90254343`, both with uncommitted work. One run per scene and device,
except OpenGL `bodies` simulating and `planet` orbiting, which took three:

| Scene | OpenGL frame | OpenGL host | Vulkan frame | Vulkan host |
|---|---:|---:|---:|---:|
| empty editor (panels only) | 0.57 ms | 0.53–0.58 ms | 0.62 ms | 0.60–0.61 ms |
| `dog`, editing | 0.58 ms | 0.51–0.65 ms | 0.65 ms | 0.64–0.65 ms |
| `dog`, simulating | 0.69 ms | 0.65–0.71 ms | 0.77 ms | 0.74–0.78 ms |
| `bodies`, editing | 0.56 ms | 0.48–0.62 ms | 0.67 ms | 0.66–0.70 ms |
| `bodies`, simulating | 3.92–4.08 ms | 2.75–4.20 ms | 2.99 ms | 2.19–3.22 ms |
| `planet`, standing | 4.49 ms | 1.58–1.67 ms | 4.21 ms | 2.45–2.60 ms |
| `planet`, orbiting at 10°/s | 4.49–4.58 ms | 2.15–3.19 ms | 4.27 ms | 2.81–3.56 ms |

- **OpenGL** is at or under the table above in every scene but simulating
  `bodies`, whose frame is 0.15–0.3 ms longer in all three runs, about 5%.
  Its host time varies by 1 ms between windows as the bodies settle, so
  the cause is not separated from the simulation's own variation, and no
  older build remains to compare against.
- **Vulkan** host time is 0.05–0.1 ms above OpenGL in the small scenes, and
  0.9 ms above it in `planet` standing, where PlanetWalker's gap is 0.37 ms
  (`planet_walker_frame.md`). The remaining 0.5 ms is Editor-only work on
  Vulkan, not yet profiled. Simulating `bodies` is 0.9 ms shorter a frame
  on Vulkan.

## Hierarchy

Before this measurement the Scene Hierarchy panel visited every descendant
of every root, querying the world and formatting each entity's description,
even under collapsed tree nodes. With the planet's tiles owning thousands of
sites and buildings, that cost 1.5 ms standing (3.2 ms host) and about
2 ms orbiting (4.1–5.3 ms host). It now visits only the parts of open nodes.
