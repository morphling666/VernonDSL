# PlanetWalker frame time, standing and walking

The Vernon engine's `PlanetWalker` renders through VernonDSL graphics Programs
on the OpenGL backend. This report measures the main thread's own cost per
frame with the walker standing and walking, against the gate of
[`specs/runtime/architecture_v2.md`](../../specs/runtime/architecture_v2.md)
section 16: under 1 ms standing and under 2 ms walking.

## Provenance

- Source revision: VernonDSL `6fcb25d` and Vernon `90254343`, both with the
  uncommitted architecture v2 work (linked schedules on the Core HAL, the
  device-cull tier, prepared rendering scopes in packet batches).
- Platform: Windows 11 (10.0.26300), AMD Ryzen 9 9950X, NVIDIA GeForce
  RTX 4090, OpenGL 4.6, a 1280×720 window, vsync off, the default Pudong
  site on the planet terrain, Release build.
- Command: `PlanetWalker.exe --root <Vernon> --no-vsync --frames 4000
  --frame-stats [--walk]`, run on the desktop through
  `scripts/run_on_desktop.ps1`. `--walk` walks forward and turns at 0.1 of
  the turn rate without input.
- Metric: `--frame-stats` prints, per window of 300 frames, the mean wall
  time of `app.update()` (frame), the part of it the render graphs spent
  blocked retiring the frame they invoke into (wait), and their difference
  (host). The retire wait spins in the OpenGL driver, so thread CPU time
  cannot separate it; host is the main thread's own work. The first window
  holds startup and the first terrain selection and is left out below.

## Results

| Case | Frame | Wait | Host (mean) | Host (worst window) | Gate |
|---|---:|---:|---:|---:|---:|
| standing | 4.24 ms | 3.47 ms | 0.77 ms | 0.81 ms | 1 ms |
| walking | 5.28 ms | 3.88 ms | 1.40 ms | 1.72 ms | 2 ms |

With `--device vulkan`, on the same Core device path and 1500 frames,
after the per-call RHI work this session reduced (batched uploads, cached
encoder resolution, layout transitions skipped when already done):

| Case | Frame | Wait | Host (mean) | Host (worst window) | Gate |
|---|---:|---:|---:|---:|---:|
| standing | 4.32 ms | 2.52 ms | 1.80 ms | 2.07 ms | 1 ms |
| walking | 4.59 ms | 2.44 ms | 2.16 ms | 2.34 ms | 2 ms |

Vulkan missed both gates on that path: each call encoded the frame again
through RuntimeCore packets, the RHI adapter and the RHI's command layer,
about 0.26 ms per schedule call against 0.11 ms on OpenGL, and its UI draw
and present cost about 0.24 ms against 0.09 ms.

### Vulkan HAL device (section 16.3 step 1)

With the native Vulkan device of `hal/vulkan/vulkan_device.cpp` (recordings
submitted again, slots pooled, indirect compute dispatches, uploads in the
same submission), VernonDSL and Vernon at the same revisions with the step 1
work, standing over 3000 frames and walking over 1500:

| Case | Frame | Wait | Host (mean) | Host (worst window) | Gate |
|---|---:|---:|---:|---:|---:|
| standing | 4.21 ms | 3.32 ms | 0.89 ms | 0.93 ms | 1 ms |
| walking | 4.54 ms | 3.10 ms | 1.45 ms | 1.63 ms | 2 ms |

OpenGL in the same session, 1500 frames, on the unchanged Core path:

| Case | Frame | Wait | Host (mean) | Host (worst window) | Gate |
|---|---:|---:|---:|---:|---:|
| standing | 4.41 ms | 3.61 ms | 0.80 ms | 0.84 ms | 1 ms |
| walking | 4.61 ms | 3.31 ms | 1.30 ms | 1.42 ms | 2 ms |

A standing Vulkan frame spends about 0.30 ms in the two schedule calls
(pass planning 0.08 ms, the UI's recording rebuilt every call 0.05 ms,
submission 0.08 ms) and 0.13 ms presenting, which still blits through the
RHI. While walking, the frame schedule builds a new recording of its 1740
draws on about one frame in eighty, when the shadow, depth or forward plan
changes.

### OpenGL tape device (section 16.3 step 2)

With the tape device of `hal/opengl/opengl_device.cpp`, against the Core
device measured in the same session by pointing OpenGL schedules back at
it, 1500 frames each. `--device opengl33` caps the context at OpenGL 3.3,
which runs the frame's compatibility tier: CPU culling and no device
streams.

| Device | Case | Frame | Wait | Host (mean) | Host (worst window) |
|---|---|---:|---:|---:|---:|
| OpenGL 4.6, tape | standing | 4.41 ms | 3.83 ms | 0.58 ms | 0.62 ms |
| OpenGL 4.6, Core | standing | 4.40 ms | 3.59 ms | 0.81 ms | 0.85 ms |
| OpenGL 4.6, tape | walking | 4.65 ms | 3.53 ms | 1.12 ms | 1.31 ms |
| OpenGL 4.6, Core | walking | 4.59 ms | 3.23 ms | 1.36 ms | 1.54 ms |
| OpenGL 3.3, tape | standing | 4.60 ms | 1.97 ms | 2.63 ms | 2.75 ms |
| OpenGL 3.3, Core | standing | 4.62 ms | 1.90 ms | 2.72 ms | 2.84 ms |
| OpenGL 3.3, tape | walking | 4.94 ms | 1.36 ms | 3.58 ms | 3.79 ms |
| OpenGL 3.3, Core | walking | 4.86 ms | 1.23 ms | 3.62 ms | 3.71 ms |

The tape saves about 0.23 ms a standing frame on OpenGL 4.6. On OpenGL 3.3
the host time is the compatibility tier's CPU culling and per-record draws,
which both devices share, so both miss the 1 ms and 2 ms gates there.

Both OpenGL cases are GPU-bound: the main thread waits for the device for about
80% of a frame. Walking frames take longer on the GPU because the view
covers new, denser ground as the walker moves.

### HAL resource API (section 16.3 step 3)

Step 3 adds `VernonHal.h` and moves the CPU and CUDA devices; Vernon does
not call the new API yet, so no frame path changed. Vulkan standing over
3000 frames, the rest over 1500, host time excluding frames 0-299:

| Device | Case | Frame | Wait | Host (mean) | Host (worst window) |
|---|---|---:|---:|---:|---:|
| Vulkan | standing | 4.12 ms | 3.20 ms | 0.92 ms | 0.94 ms |
| Vulkan | walking | - | - | 1.45-1.48 ms | 1.63-1.75 ms |
| OpenGL 4.6 | standing | 4.28 ms | 3.73 ms | 0.55 ms | 0.59 ms |
| OpenGL 4.6 | walking | 4.51 ms | 3.42 ms | 1.09 ms | 1.29 ms |
| OpenGL 3.3 | standing | 4.46 ms | 2.04 ms | 2.42 ms | 2.50 ms |
| OpenGL 3.3 | walking | 4.76 ms | 1.43 ms | 3.34 ms | 3.50 ms |

Vulkan walking is the range of three runs. A fourth run reached 1.69 ms
mean and 2.04 ms worst; the walking window's host time follows which plans
rebuild in it, and varies by about 0.1 ms between identical runs.

### Vernon on HAL objects (section 16.3 step 4, part 2)

Vernon's buffers, images, samplers and surfaces are `VernonHal.h`
objects; schedules still take references made from them. On the OpenGL 3.3
tier the host cull now copies and uploads only the instance rows that changed
(Vernon `specs/render/gpu_scene.md` section 8.2). Ranges are over three
3000-frame runs standing and two 1500-frame runs walking, excluding frames
0-299:

| Device | Case | Frame | Host (mean) | Host (worst window) |
|---|---|---:|---:|---:|
| Vulkan | standing | 4.19 ms | 0.92-0.94 ms | 0.96-0.97 ms |
| Vulkan | walking | 4.49 ms | 1.44 ms | 1.72 ms |
| OpenGL 4.6 | standing | 4.27 ms | 0.57-0.58 ms | 0.62-0.63 ms |
| OpenGL 4.6 | walking | 4.52 ms | 1.11 ms | 1.23 ms |
| OpenGL 3.3 | standing | 4.18-4.20 ms | 2.32-2.35 ms | 2.40-2.42 ms |
| OpenGL 3.3 | walking | 4.45-4.48 ms | 3.11-3.12 ms | 3.27-3.41 ms |

Before the row change, OpenGL 3.3 measured 2.57-2.60 ms standing and
3.51-3.55 ms walking: each frame copied every batch's whole range and
re-uploaded nearly all of it, because the cull's slot order changes the rows
every frame. OpenGL 4.6 standing sits 0.02 ms above step 3's single
measurement and within step 2's 0.58 ms. Its only new per-frame work is the
HAL reference calls, about 5 µs in a profile.

### Linked schedules on every device (section 16.3 step 4, part 3)

Primal Programs link on every HAL device, and CPU and CUDA VJPs, with
link-time checkpointing, run as linked schedules. Standing, 1500 frames
per run, excluding frames 0-299; walking was not measured again:

| Device | Case | Frame | Host (mean) |
|---|---|---:|---:|
| Vulkan | standing | 4.25-4.29 ms | 0.87-0.97 ms |
| OpenGL 4.6 | standing | 4.34 ms | 0.57-0.65 ms |
| OpenGL 3.3 | standing | 4.27 ms | 2.35 ms |

Host time matches part 2. The frame is about 0.08-0.1 ms longer on every
device, all of it in time the GPU spends executing passes. The GPU ran at
99% utilization at 2670 MHz. A Vulkan frame's pass timestamps, captured by a
temporary print that has since been removed, cover the whole frame:

| Pass | GPU time |
|---|---:|
| sky irradiance | 0.15 ms |
| instance cull | 0.06 ms |
| shadow | 0.93 ms |
| depth | 1.01 ms |
| forward | 2.23 ms |
| sky, translucent, bloom, exposure, display | 0.06 ms |

None of these shaders changed. The shift is the same fraction on Vulkan and on
both OpenGL tiers, so it is more likely to come from GPU clock state than from
any pass doing more work. No older build was available to compare against.

### Native Vulkan and OpenGL buffers (section 16.3 step 4, part 5)

Vernon's buffers are HAL allocations named by serial, which each build maps
to the native buffer once. 1500 frames per run, excluding frames 0-299:

| Device | Case | Runs | Frame | Host (mean) | Host (worst window) |
|---|---|---:|---:|---:|---:|
| Vulkan | standing | 2 | 4.28-4.31 ms | 0.93-0.98 ms | 0.96-1.00 ms |
| Vulkan | walking | 2 | 4.54-4.57 ms | 1.41-1.49 ms | 1.55-1.67 ms |
| OpenGL 4.6 | standing | 1 | 4.36 ms | 0.56 ms | 0.59 ms |
| OpenGL 4.6 | walking | 2 | 4.58 ms | 1.09-1.11 ms | 1.24-1.30 ms |
| OpenGL 3.3 | standing | 1 | 4.28 ms | 2.34 ms | 2.43 ms |
| OpenGL 3.3 | walking | 1 | 4.56 ms | 3.16 ms | 3.29 ms |

Host time matches parts 2 and 3. The first walking run on Vulkan and on
OpenGL 4.6 after each build measured higher (1.77 ms and 1.27 ms mean) and is
left out; the reruns match part 2.

### Graphics VJPs (section 16.3 step 4, part 6)

VJPs with tapes and leaf copies link on Vulkan and OpenGL. 1500 frames per
run, excluding frames 0-299:

| Device | Case | Runs | Frame | Host (mean) | Host (worst window) |
|---|---|---:|---:|---:|---:|
| Vulkan | standing | 4 | 4.18-4.19 ms | 0.94-0.96 ms | 0.96-0.98 ms |
| Vulkan | walking | 3 | 4.6 ms | 1.55-1.58 ms | 1.73-1.86 ms |
| OpenGL 4.6 | standing | 3 | 4.24-4.27 ms | 0.82-0.85 ms | 0.83-0.86 ms |
| OpenGL 4.6 | walking | 2 | 4.6 ms | 1.42 ms | 1.61 ms |

Vulkan walking is about 0.1 ms above part 5, within the variance part 3
notes. OpenGL 4.6 host time is about 0.26 ms above part 5, standing and
walking, with the same frame time, so the main thread waits that much less.
Temporary timers, since removed, put the two schedule calls at about 0.30 ms
a frame and `SwapBuffers` at 22 µs; neither recordings rebuilt per call (the
UI's, as before) nor buffer allocations account for the rest. The cause is
not found yet; no build of part 5 remains to compare against.

### Core device deleted (section 16.3 step 5, part 1)

The Core device is deleted, and internal buffers, leaf carriers and tape
launch counts come from the device's own buffer pool instead of the RHI.
1500 frames per run, excluding frames 0-299:

| Device | Case | Runs | Frame | Host (mean of windows) | Host (worst window) |
|---|---|---:|---:|---:|---:|
| Vulkan | standing | 1 | 4.13-4.14 ms | 0.90 ms | 0.92 ms |
| Vulkan | walking | 1 | 3.7-4.9 ms | 1.43 ms | 1.69 ms |
| OpenGL 4.6 | standing | 2 | 4.21-4.24 ms | 0.52 ms | 0.56 ms |
| OpenGL 4.6 | walking | 1 | 3.7-5.1 ms | 1.19 ms | 1.47 ms |

OpenGL 4.6 standing host time is 0.3 ms under part 6, about where part 5
was, so the rise part 6 recorded does not reproduce here. Which change, or
whether the machine's load at the time, accounts for it is not identified.

### Native images, views, samplers and surfaces (section 16.3 step 5, part 2)

Vulkan and OpenGL images, views, samplers and surfaces are HAL objects named
by serial, and the RHI bridge is deleted. 1500 frames per run, excluding
frames 0-299:

| Device | Case | Runs | Frame | Host (mean of windows) | Host (worst window) |
|---|---|---:|---:|---:|---:|
| Vulkan | standing | 1 | 4.13 ms | 0.89 ms | 0.92 ms |
| Vulkan | walking | 2 | 3.7-4.9 ms | 1.32-1.39 ms | 1.61 ms |
| OpenGL 4.6 | standing | 1 | 4.22 ms | 0.49 ms | 0.51 ms |
| OpenGL 4.6 | walking | 1 | 3.7-4.9 ms | 1.01 ms | 1.22 ms |
| OpenGL 3.3 | standing | 1 | 4.14 ms | 2.25 ms | 2.32 ms |

Every case is at or under part 1. The first Vulkan walking run after the
build measured 1.61 ms mean and 1.95 ms worst and is left out, as part 5
does with such runs; the two reruns are in the table.

### Step 5 complete (section 16.3 step 5)

The Program invocation runtime and the ProgramGraph linker are deleted;
every call is a linked schedule, and a composed Program graph links per
node. VernonDSL `f840106` and Vernon `90254343`, both with uncommitted
work. 1500 frames per run, excluding frames 0-299:

| Device | Case | Runs | Frame | Host (mean of windows) | Host (worst window) |
|---|---|---:|---:|---:|---:|
| Vulkan | standing | 1 | 4.18-4.20 ms | 0.88 ms | 0.89 ms |
| Vulkan | walking | 2 | 3.8-4.9 ms | 1.36-1.48 ms | 1.58-1.72 ms |
| OpenGL 4.6 | standing | 1 | 4.18 ms | 0.51 ms | 0.54 ms |
| OpenGL 4.6 | walking | 1 | 3.7-4.9 ms | 1.03 ms | 1.31 ms |
| OpenGL 3.3 | standing | 1 | 4.10-4.11 ms | 2.17 ms | 2.23 ms |
| OpenGL 3.3 | walking | 1 | 3.7-4.8 ms | 2.87 ms | 2.95 ms |

Standing is at or under part 2 on every device, and OpenGL 3.3 walking is
0.25 ms under part 5. As in part 2, the first Vulkan walking run measured
higher (1.60 ms mean, 1.93 ms worst) and is left out; the two reruns span
the 0.1 ms variance part 3 notes.

### DirectX 12 device (section 16.3, phase 4)

PlanetWalker runs on the DirectX 12 HAL device (`--device directx`). The
compiler reads struct members through their construction before
serialization, which changes the SPIR-V of every target. VernonDSL `364d8af`
and Vernon `90254343`, both with uncommitted work. 1500 frames per run,
excluding frames 0-299:

| Device | Case | Runs | Frame | Host (mean of windows) | Host (worst window) |
|---|---|---:|---:|---:|---:|
| DirectX 12 | standing | 1 | 4.37-4.38 ms | 1.35 ms | 1.41 ms |
| DirectX 12 | walking | 2 | 4.0-5.6 ms | 1.91-2.25 ms | 2.05-2.58 ms |
| Vulkan | standing | 1 | 4.21-4.22 ms | 0.92 ms | 0.94 ms |
| Vulkan | walking | 2 | 3.8-5.0 ms | 1.42-1.44 ms | 1.57-1.58 ms |
| OpenGL 4.6 | standing | 1 | 4.18-4.20 ms | 0.54 ms | 0.58 ms |
| OpenGL 4.6 | walking | 2 | 3.8-4.9 ms | 1.05-1.06 ms | 1.16-1.33 ms |

Vulkan and OpenGL 4.6 match step 5 within the 0.1 ms variance part 3 notes.
DirectX 12 spends about 0.43 ms more host time than Vulkan standing and
0.5-0.8 ms more walking, so it misses the 1 ms standing gate and, in its
first walking run, the 2 ms walking gate. The first walking run after the
build is the higher one, as the first Vulkan walking runs were in parts 2
and 5. No
earlier DirectX measurement exists to compare against, and where the extra
host time goes is not profiled yet.

### Terrain selection by distance only

Terrain selection no longer culls tiles against the view frustum (Vernon
`specs/scene/design.md`, "Selection ignores the view direction").
`--turn` walks in a tight circle at the full turn rate, so the follow camera
swings behind the walker. Frame stats now also print each window's longest
frame. DirectX 12, 1800 frames per run, excluding frames 0-299:

| Case | Selection | Frame | Host (mean of windows) | Longest frame |
|---|---|---:|---:|---:|
| turning | frustum and distance | 6.6-8.2 ms | 3.58-4.87 ms | 132-175 ms |
| turning | distance | 5.1-6.5 ms | 1.58-2.36 ms | 7-68 ms |
| walking | frustum and distance | 4.0-5.2 ms | 1.68-2.11 ms | 34-94 ms |
| walking | distance | 4.3-5.4 ms | 1.58-2.00 ms | 37-111 ms |
| standing | frustum and distance | 4.37-4.41 ms | 1.35-1.42 ms | 4.9-5.9 ms |
| standing | distance | 4.79-4.80 ms | 1.43-1.51 ms | 5.4-6.0 ms |

Turning no longer rebuilds the tiles the view sweeps across, which halves
its host time and its worst stalls. The tiles behind the camera cost about
0.4 ms of device time standing (the retire wait rises from 3.0 to 3.3 ms).
They are culled from the camera passes, so the likely cost is their casters
in the shadow cascades; this is not profiled yet.
The remaining stalls of 40-110 ms come from tiles crossing the 1 km
selection radius, which still build synchronously, and the batch rebuild
that follows.

### Temporal antialiasing

Scene viewports resolve every frame against a history (Vernon
`specs/render/design.md`, "Temporal antialiasing"): one full-screen resolve
and one history copy at the 1920x1080 frame, both before bloom. Standing,
with distance-only selection, excluding frames 0-299:

| Device | Frame without | Frame with | Host with |
|---|---:|---:|---:|
| DirectX 12 | 4.79-4.80 ms | 4.94-4.96 ms | 1.38-1.46 ms |
| Vulkan | | 4.39-4.41 ms | 0.84-0.93 ms |
| OpenGL 4.6 | | 4.52-4.53 ms | 0.60-0.66 ms |

On DirectX 12 the two passes cost about 0.16 ms of device time and no host
time. Vulkan and OpenGL were not measured with distance-only selection
alone. Walking and turning on DirectX 12 match the distance-only runs within
their variance: host 1.56-1.87 ms walking and 1.50-2.19 ms turning.

With contrast-adaptive sharpening in the display pass and material textures
sampled at a -0.5 mip bias, DirectX 12 standing measures 4.87-4.89 ms a frame
and 1.36-1.43 ms host, within the variance of the run without them.

## Standing frame

Measured once with temporary scope timers (since removed), which added
about 0.06 ms to the host time:

| Step | Time |
|---|---:|
| `SceneViewport::render` without the retire wait | 0.67 ms |
| linked schedule call (packet replay and OpenGL calls) | 0.46 ms |
| CPU culling and instance rows (`collectDraws`) | 0.15 ms |
| scene tables, batches, frame values | 0.02 ms |
| UI draw and present | 0.10 ms |
| systems other than the view (walk, animation) | 0.09 ms |

Nothing is planned or validated in a steady frame: stored scopes replay,
and the instance rows of the opaque passes are culled on the device.

## Walking frame

Walking adds terrain selection, tile building and storage loads, and batch
rebuilds when tiles spawn or retire. A batch rebuild costs about 12 ms; it
happens on the frames that change the set of dressed tiles, which the
window means spread out. Storage loads cost up to about 0.8 ms in a frame
in dense regions.

## Before the v2 runtime

The same app before architecture v2 (VernonDSL `6fcb25d` without the work,
measured by wall time over 600 frames): 15.2 ms a frame standing and 36.8 ms
walking, both CPU-bound. Walking changed the recording's structure every
frame, so every draw was planned, validated (`validateGraphicsRecordings`
40% of the main thread) and hazard-checked again; a standing frame spent 30%
in CPU culling and 30% decoding the stored batch.
