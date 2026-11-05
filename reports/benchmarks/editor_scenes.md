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

## Hierarchy

Before this measurement the Scene Hierarchy panel visited every descendant
of every root, querying the world and formatting each entity's description,
even under collapsed tree nodes. With the planet's tiles owning thousands of
sites and buildings, that cost 1.5 ms standing (3.2 ms host) and about
2 ms orbiting (4.1–5.3 ms host). It now visits only the parts of open nodes.
