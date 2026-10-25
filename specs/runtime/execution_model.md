# Execution model: compiled plans, argument arenas, bindless resources

Status: accepted. Phase 1 is implemented for rendering scopes as stored
graphics scopes, whose argument arena is device memory written by copies in
command order ([`stored_plans.md`](stored_plans.md) section 12) rather than
the per-slot mapped memory of section 4. Applies to every invocation kind: graphics,
compute, VJP pullbacks and modules. Amends, once accepted,
[`stored_plans.md`](stored_plans.md) sections 2, 3, 10 and 11.2 and
[`binding_groups.md`](binding_groups.md). The Engine-side GPU scene (Vernon
`specs/render/gpu_scene.md`) is a separate layer and not part of this
document.

## 1. Goal

**A steady-state Execute costs O(changed payload bytes) + O(1) submissions
of CPU time, independent of how many draws, dispatches or kernels it
issues.**

Measured today (Editor, 1000 boxes, OpenGL, about 6000 draws): Execute takes
3.0 ms, about 500 ns per draw, of which the GL driver is about 0.1 ms. Every
Execute plans, validates, checks hazards, fills groups, builds packets and
decodes them, although nothing structural changed.

## 2. Why industry runtimes cannot do this, and we can

UE5's RHI, bgfx, Diligent, NVRHI and wgpu are immediate-mode translators.
The engine above them re-issues every command every frame, and the RHI
rediscovers state: redundant-bind filtering, descriptor and PSO hashing,
transition tracking, native translation. UE5's render graph recompiles
every frame. Their cost is commands × frames, because a general C++ engine
cannot promise that the next frame resembles this one. Eager deep-learning
runtimes pay the same per kernel launch, which is why LLM serving falls
back to CUDA Graphs.

Vernon compiles programs. The binding layouts, the reads and writes of
every stage, the barriers and the command sequence are known when a plan
is built. The runtime therefore compiles each plan once into the native
form that replays without translation, and an Execute only supplies the
bytes that changed.

## 3. Payload classes

Stored plans split inputs into structure and payload
(`stored_plans.md` section 2). Payload splits again by where it lives:

| Class | Examples | Lives in | A change costs |
|---|---|---|---|
| Memory payload | uniforms, kernel scalars, inline values, tensor extents and strides, draw and instance counts, dispatch grids | the argument arena (section 4) | a copy of the changed bytes |
| Command payload | viewport, scissor, stencil reference, clear values | the native program | re-encoding the native program from the stored packets, without a plan rebuild |
| Structure | pipelines, resources, command order, capacities (section 8) | the stored plan | a plan rebuild |

Counts and grids are memory payload. Where a tier has indirect draws and
dispatches and they measure faster, native programs read them from the arena
(section 5). A native program that encodes them directly re-encodes its
segment when one changes (section 9); the compatibility tier's tape reads them
from the payload at every replay (section 11). Command payload changes rarely:
on a resize or an edited clear color.

## 4. Argument arena

Every memory payload of a plan has an offset in one argument arena, fixed
when the plan is built. Each draw and dispatch reads its arguments from
its **argument record** at a fixed offset.

- The arena is device memory owned by the stored plan, one region for every
  submission. An Execute writes its changed ranges with copies recorded in
  command order ahead of the replay, each ordered after the earlier
  submissions' reads by a barrier (`glBufferSubData` on OpenGL); an
  unchanged payload copies nothing. Per-slot mapped regions were rejected:
  they need one set of commands per slot. Stored graphics scopes implement
  this with updatable groups (`stored_plans.md` section 12).
- A record's offset reaches its shader as a constant baked into the native
  program when the plan is built: a push constant (Vulkan), a root constant
  (D3D12), a buffer offset (Metal), a kernel parameter (CUDA), a uniform
  (OpenGL). The arena's base address is fixed, so a native program never
  changes when payload changes.
- Groups become bytes in the arena (section 6); there is no per-bind fill.

## 5. Native programs

A stored plan keeps its packet batches (`stored_plans.md` section 11) as
the backend-neutral form. On backends that replay natively, encoding them
once more produces a **native program** per submission slot, valid while
the plan's command payload is unchanged:

| Backend | Native program | Execute |
|---|---|---|
| Vulkan | a primary command buffer recorded without `ONE_TIME_SUBMIT` | resubmit it |
| D3D12 | a closed command list | `ExecuteCommandLists` again |
| Metal | indirect command buffers for draws and dispatches, executed from a per-submission command buffer | one execute call per pass |
| CUDA | a CUDA graph built node by node from the packets (not by stream capture) | `cuGraphLaunch`; a changed grid updates its node with `cuGraphExecKernelNodeSetParams` |
| WebGPU | render bundles per pass | xecuteBundles per pass |
| OpenGL/ES | none; the packets are pre-resolved into a tape of native calls | decode the tape |
| CPU | none; the tape of kernel calls | run the tape |

Entry barriers for resources shared with other plans
(`stored_plans.md` section 5) are recorded into a small per-Execute command
buffer submitted ahead of the native program. Interior barriers live in it.

This reverses `stored_plans.md` section 10's rejection of native retained
command buffers. That rejection assumed patch sites inside commands, which
freeze in a retained buffer, and that CUDA had no equivalent. With memory
payload in the arena, nothing patched lives in a command, and CUDA graphs
are the CUDA equivalent.

## 6. Bindless resources

Resources live in one descriptor heap per device; shaders index it, and a
group is the bytes of its indices and inline values:

| Backend | Heap |
|---|---|
| Vulkan | descriptor indexing (core 1.2) with update-after-bind, or `VK_EXT_descriptor_buffer` |
| D3D12 | Shader Model 6.6 `ResourceDescriptorHeap` and `SamplerDescriptorHeap` |
| Metal | tier-2 argument buffers with GPU resource IDs |
| CUDA | device pointers and texture objects, natively |
| WebGPU, OpenGL/ES, CPU | none; bound groups, or the tape binds resources conventionally |

- A resource gets a heap slot when it is created and keeps it for its
  lifetime. A destroyed resource's slot is reused only after every
  submission that may read it retires.
- A sealed set's bytes are written once into persistent memory; an instance
  binding's bytes are memory payload in the arena. Group creation, caching
  and per-bind filling are removed: `GroupStore`, `InstanceGroups` and the
  adapters' per-group descriptor sets, tables and argument buffers go.
- Bindless needs code generation per target: SPIR-V runtime arrays with
  `NonUniform`, HLSL heap indexing, MSL argument-buffer structs. The
  layout of a group's bytes comes from reflection, as binding layouts do
  today.

## 7. Hazards, barriers and memory

Interior barriers, transient allocation and transient aliasing are decided
once per plan build, from the stages' declared reads and writes, and live
in the native program. Per Execute, the runtime checks each external
resource once (generation and entry state), not each command.

## 8. Shapes: payload unless a capacity changes

TensorView shapes stay dynamic in the language. This section decides only
when a changed shape reuses a stored plan.

`stored_plans.md` section 2 makes shape structural, because shape decides
allocation sizes, transfer ranges and Node frames. An LLM decode step grows
its KV cache by one token per step, so every step would rebuild its plan.
Batches of varying size and variable-resolution images behave the same.

Proposed: shape affects two things, and only one is structural.

- **Capacity** (structural): the size of every allocation the shape
  decides, rounded up to a capacity class (powers of two, or a maximum the
  program declares).
- **Extent** (memory payload): the dimensions, strides and grids kernels
  read. Kernels take their extents from their argument records; grids are
  indirect dispatches on Vulkan, D3D12, Metal and OpenGL, and updated graph
  nodes on CUDA.

Example: decoding with a KV cache of capacity class 4096 reuses one plan
for sequence lengths 2049 to 4096. Only the extent bytes change per token.
Length 4097 builds the plan of class 8192 once.

## 9. Plan segments

A stored plan is a sequence of **segments**, one per graphics Node
recording and one per run of other Nodes between them. Each segment has its
own signature and native program. A changed recording rebuilds its own
segment only, and the segments around it replay. The Editor runs its whole
frame graph as one invocation, so without segments an edited UI or debug
pass would rebuild the scene passes' thousands of draws with it.

Barriers between segments are computed from the segments' entry and exit
states when one of them is rebuilt.

## 10. Seams

Each risky choice sits behind one interface, so a measurement that goes
against it changes one component, not the design:

| Choice | Seam | If it measures badly |
|---|---|---|
| indirect counts and grids | per patch site, in the adapter's native-program encoder; packets name counts as payload either way | the adapter encodes that site directly and re-encodes its segment when the count changes |
| bindless groups | the provider group object; the Runtime and Core see an opaque group in either form | that backend's adapter and code generation target keep bound groups, which OpenGL/ES needs regardless |
| native programs | per backend, below the packet batches | that backend replays the packet decode, the phase-1 path |
| plan build cost | segments (section 9) | only the changed segment pays it; the build path is today's Execute path |

What no measurement is expected to undo: the structure and payload split,
the argument arena, and packets as the provider ABI. Those are the forms
UE5 uniform buffers, Unity's SRP Batcher and CUDA graphs also converge on.

## 11. Tiers

Vernon renders on two tiers (Vernon `specs/render/architecture.md` section
8). Sections 3 to 10 are the same on both; the tier changes only what the
adapter does below the packet batches:

| | High tier (Vulkan, D3D12, Metal) | WebGPU (high tier) | Compatibility tier (OpenGL 3.3, OpenGL ES 3.0, WebGL 2) |
|---|---|---|---|
| Stored plans, segments, payload split | yes | yes | yes |
| Argument arena | device-local storage or uniform memory written by staging copies and barriers | a buffer written with `writeBuffer` | a uniform buffer written with `glBufferSubData` for the changed ranges |
| A draw's argument record | push or root constant, buffer offset | dynamic offset of one bind group | `glBindBufferRange` on the arena when the record changes; the tape skips unchanged ones |
| Counts and grids | indirect where it measures faster, otherwise direct | indirect or direct (no first instance in indirect without the optional feature) | direct, read by the tape from the payload |
| Native programs | yes | render bundles | none: the tape |
| Bindless | yes, after measurement | no: bound groups | no: conventional binds in the tape |
| Compute, modules, VJP, shape capacities | yes | yes | none (the tier has no compute) |

On the compatibility tier the floor is therefore one tape decode and the
driver calls per command, with no planning, validation or group work per
Execute. That is where boxes1000 gains most: of its 3.0 ms Execute, the GL
driver calls are about 0.1 ms.

The arena's uniform-buffer form bounds a record by the tier's uniform block
size (16 KB guaranteed on OpenGL 3.3 and OpenGL ES 3.0) and aligns records
to the device's uniform buffer offset alignment. Arguments that exceed it
stay in the groups that hold them today.

## 12. Submission

- Executes of one frame or step accumulate and submit together, with
  timeline semaphores rather than a fence per Execute.
- Work the Command DAG shows independent of graphics may go to an async
  compute queue, decided when the plan is built.
- Parallel encoding matters only for plan builds, since replay does not
  encode.

## 13. Targets and benchmarks

Comparisons are measured, not claimed:

The Editor scenes are the regression benchmarks for every phase, measured
with `--profile` and `--no-vsync`. Today they run on OpenGL, the
compatibility tier; each phase that changes a high-tier adapter also runs
them on that backend once the Editor runs there:

| Scene | Baseline (after the 2b revert) | Target |
|---|---|---|
| boxes1000 (OpenGL, about 6000 draws) | 5.25 ms frame, 3.0 ms Execute | Execute bounded by driver time plus the tape (about 0.3 ms) |
| cave + acrobot, simulating | measured before phase 1 | no regression in any phase; Execute reduced by phase 1 |
| a scene whose recording changes every frame (the ImGui pass) | `ui.end` 0.7 ms | plan build path no slower than today's Execute |

Comparisons with other runtimes:

| Workload | Compared against | Target |
|---|---|---|
| 10000 draws, distinct materials, static | the same scene in UE 5.7 (render and RHI thread time in Unreal Insights) | Vulkan, D3D12 and Metal: Execute under 20 µs plus payload copy; OpenGL tape: under 30 ns per command plus driver time |
| a chain of 1000 compute kernels | `cuLaunchKernel` in a loop; a CUDA graph built by hand | within 10% of the hand-built graph |
| small-model LLM decode, growing sequence | PyTorch with CUDA graphs; llama.cpp | tokens per second at least equal; no plan builds within a capacity class |
| training step (forward + pullback + update) | PyTorch with CUDA graphs | CPU time per step independent of layer count |

## 14. Risks to measure

The O(changed payload) bound is the floor for CPU submission cost; no
runtime that submits work can do less. The parts below are choices that
reach it, and each must earn its place by measurement:

- **Indirect draws for every count.** Indirect draws cost GPU front-end
  time on some hardware (tile-based mobile GPUs, Metal indirect command
  buffers). Counts that do not change per Execute may stay direct command
  payload; the benchmarks decide.
- **Bindless access.** Non-uniform heap indexing costs on some GPUs, and
  OpenGL/ES has no heap. The OpenGL tape keeps conventional binds.
- **Structure that changes every frame** (UI, debug drawing, editing):
  such passes take the plan build path every Execute, so the build path
  keeps its own target (section 13) and is not slower than today.
- **Swapchain images.** A native program that renders into the presented
  image differs per image; a plan keeps one program per swapchain image, or
  renders offscreen and copies.
- **GPU time.** This document bounds CPU time. GPU time depends on barrier
  precision and aliasing (section 7) and is reported by the same
  benchmarks.

## 15. Phases

1. **Stored plans with segments and an arena.** `stored_plans.md` sections 5 to 7, with
   memory payload in the arena instead of command patch sites; replay
   through the packet decode on every backend. Removes planning,
   validation and group fills from steady-state Executes.
2. **Native programs.** CUDA graphs and Vulkan first, then D3D12 and
   Metal; the OpenGL and CPU tapes.
3. **Bindless.** Prototyped on Vulkan first and measured against bound
   groups on the Editor scenes; then code generation for the other targets,
   the heap, groups as bytes. Bound-group code is deleted per backend only
   after that backend measures faster.
4. **Shape capacities.** Section 8, with the LLM decode benchmark.
5. **Submission batching and async compute.**

Each phase ends with the benchmarks of section 13 that apply to it, before
and after.

## 16. Rejected alternatives

- **Faster per-command translation** (the industry design): its cost still
  scales with commands × Executes.
- **Bind-time bytes per group** (step 2b, reverted): every bind refilled its
  values; boxes slowed from 4.9 to 5.8 ms.
- **Dynamic offsets per group** (`stored_plans.md` section 3 as written):
  still one bind per draw, and Vulkan guarantees only 8 dynamic uniform and
  4 dynamic storage buffers per layout.
- **CUDA stream capture** to build graphs: captures whatever the stream
  does, including runtime-internal work. Building nodes from packets is
  deterministic and needs no capture mode.
- **Shape-polymorphic plans with no capacity classes**: allocations would
  be resized inside a replay, which a native program cannot do.
