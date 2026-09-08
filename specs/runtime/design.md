# Runtime design

Status: current Runtime, provider, RHI, and Program execution architecture.

Program semantics and deployment are defined by
[`../program/architecture.md`](../program/architecture.md) and
[`../program/execution_manifest.md`](../program/execution_manifest.md).
Graphics-specific semantics are defined by
[`../program/graphics_execution.md`](../program/graphics_execution.md).

## 1. Ownership layers

Runtime execution is split into four layers:

- **Runtime API** owns Program bundles, executables, instances, invocations,
  pullbacks, validation diagnostics, and public resource references.
- **RuntimeCore** owns manifest resolution, Stage binding plans, Program
  invocation planning, provider-neutral caches, and publication policy.
- **Runtime provider** owns physical resources, command construction,
  submission, completion, barriers, and transient binding storage.
- **VernonRHI** implements the built-in GPU provider over CUDA, Vulkan,
  DirectX 12, Metal, OpenGL, and OpenGL ES.

The CPU provider implements compute directly and does not link VernonRHI.
Foreign engines may implement the provider SPI without depending on
VernonRHI.

RuntimeCore stores opaque provider resource references. It does not own native
textures, framebuffers, descriptor heaps, queues, command encoders, or fences.

## 2. Canonical Program lifecycle

The only deployment lifecycle is:

```text
manifest bytes
  -> VernonProgramBundle
  -> resolve feature variant
  -> VernonProgramExecutable
  -> VernonProgramInstance
  -> VernonProgramInvocation
  -> bind Values and controls
  -> forward
  -> optional VernonPullback
```

`VernonProgramBundle` is immutable deployment data.
`VernonProgramExecutable` owns one immutable `ResolvedExecutionPlan`.
`VernonProgramInstance` owns persistent binding state.
`VernonProgramInvocation` owns concrete invocation descriptors and temporary
physical state.

Runtime has no pipeline-bundle compatibility loader, profile executor, direct
Stage asset submit path, or unbound Program forward path. An optimized
single-node schedule remains a Program execution optimization.

## 3. Resolve boundary

Resolution has two explicit results:

1. `ResolvedProgram` validates logical Program structure, Value SSA, Storage
   versions, controls, Stage contracts, boundaries, and derived dependencies.
2. `ResolvedExecutionPlan` selects exact physical Stage implementations,
   endpoint projections, carriers, residency, transfers, hazards, backend
   barriers, graphics scopes, publication transactions, and residual plans.

Every execution path consumes the physical plan. Invocation code and backends
do not rediscover policy from names, carrier shapes, resource aliases, or
sentinel values.

The resolver validates, before loading target code:

- Program and compiler contract versions;
- feature-key selection;
- target identity and runtime requirements;
- artifact size, digest, format, and entry point;
- reflection and Program endpoint compatibility;
- device/context capabilities;
- workgroup, graphics, resource, and ABI limits.

## 4. Binding and materialization

Public boundary reflection assigns stable slots. Binding uses
`VernonProgramArgument`; names are diagnostics and lookup conveniences, not
transport identity.

Program Values and Storages are canonical logical state. A
`MaterializedNodeFrame` creates transient Stage-local arguments according to
the resolved endpoint projections. It packs and unpacks physical aggregate
carriers without changing canonical Value layout.

TensorView bindings carry:

- owner/resource identity;
- element layout and access;
- rank and concrete shape;
- signed byte strides;
- byte offset and extent.

Dynamic shape, stride, offset, byte length, and compute grid are invocation
data. They are not artifact specialization inputs. Zero values are never used
as “infer this later” sentinels.

## 5. Transfers and residency

`ResolvedTransferExecutor` executes only transfers present in the resolved
plan:

- host upload;
- device-to-device buffer or image copy;
- host readback;
- provider-native transition.

Transfer ordering participates in the same command dependency graph as Stage
execution. Compatible intermediate Values remain device-resident. Runtime does
not synchronously read a device result to host merely to feed a later device
Node.

Checked arithmetic is required for every address, offset, extent, stride,
shape product, and transfer range.

## 6. Publication

`PublicationTransaction` implements Program boundary publication:

- `commit_after_success` stages output and commits only after successful
  completion;
- `in_place` binds caller-visible storage directly.

Rollback discards staged buffer/image publications. Device-resident
publication uses provider copy commands. Host upload/download is used only
when the resolved boundary residency requires host transport.

Invocation failure poisons only state whose contract cannot be restored.
Reusable Program instances and pullbacks preserve their documented state after
recoverable transactional failures.

## 7. Pullbacks

Forward-with-VJP returns a pullback that owns:

- the immutable executable plan;
- immutable retained Value and Storage snapshots;
- retained provider resources and exact versions;
- immutable tape;
- replay and checkpoint descriptions.

Mutable invocation scratch and command encoders are not retained inside the
pullback. Each apply creates a fresh `ProgramInvocationState`, binds canonical
cotangent and gradient slots, executes the backward plan, and transactionally
publishes fresh gradients.

Pullback application uses only `VernonProgramArgument` boundary bindings.
Derivative leaf/group reflection is metadata, not another execution ABI.

## 8. Private Command DAG

The private Command DAG owns:

- Value and resource hazards;
- upload, device-copy, Stage execution, and readback ordering;
- native barriers and resource transitions;
- render-scope formation and fusion;
- replay and checkpoint scheduling;
- submission, completion, and transient resource lifetime.

It is not serialized as Program topology and is not a public Python authoring
API. Runtime records and submits commands internally. Engine-owned execution
graph embedding does not change Program binding semantics.

## 9. CPU execution

CPU cooking emits:

- one canonical `*.program.json`;
- a relocatable `.o` or `.obj`;
- generated static-registration `.c` and `.h` sources.

Applications link the object and registration source before loading the
manifest. Runtime resolves symbols through the static CPU entry registry and
does not parse or relocate object files.

Interactive CPU execution may use compiler-owned LLJIT. LLVM IR and ORC state
are not persistent deployment formats.

### Range-phase scheduler

CPU compute and CPU VJP use one range-phase entry ABI. A bounded worker pool
executes contiguous lane ranges rather than creating one task or OS thread per
lane.

Each workgroup owns its shared arena, lane-private state, current phase,
barrier site, outstanding range count, and latched diagnostic. A lowered
barrier yields a range. A phase advances only when all ranges complete or
yield at the same barrier site.

Mixed completion/yield, mismatched barrier sites, allocation errors, invalid
outcomes, and entry exceptions fail the dispatch and release group state.
Workers never block at a workgroup barrier.

The web profile uses the same scheduler with calling-thread execution and no
worker pool.

## 10. GPU backend loading

VernonRHI owns runtime discovery of CUDA and Vulkan loaders. Vulkan headers are
compile-time dependencies only. Vulkan loader discovery uses the standard
Khronos loader and preserves ordinary ICD discovery.

Backend inclusion at build time does not imply runtime availability. Runtime
availability requires a loadable API, usable device/context, and all artifact
requirements.

### CUDA

CUDA is compute-only. It loads PTX through the Driver API and validates
address size, PTX requirements, compute capability, workgroup limits, and
required features before pipeline creation.

### Vulkan

Vulkan supports compute and offscreen graphics. Instance and device extensions
are enabled only when advertised. SPIR-V modules, descriptor layouts, pipeline
layouts, graphics/compute pipelines, barriers, and resource transfers are
owned by VernonRHI.

### DirectX 12

DirectX 12 loads pre-cooked DXIL and never invokes DXC at deployment. The
provider owns device, queue, allocators, fences, resources, descriptor heaps,
and transient upload/readback storage. Tests may select WARP through an
internal facility; normal creation selects hardware adapters.

### Metal

Metal consumes cooked MSL on supported Apple platforms. Runtime validates MSL,
OS, device feature, argument-buffer, and workgroup requirements before
creating compute or graphics pipelines.

### OpenGL and OpenGL ES

OpenGL and OpenGL ES require a host-owned current context. The owner supplies
`make_current` and `get_proc_address` callbacks. Runtime and VernonRHI do not
link a window-system library.

Each backend accepts only artifacts for its matching GLSL profile. Compute
requires OpenGL 4.3+ or OpenGL ES 3.1+. Capabilities that depend on API version
are queried after context creation.

The context owner outlives RHI and Runtime children. All GL allocation,
transfer, deletion, and invocation operations first make the associated
context current.

## 11. Runtime requirements

Every Stage artifact carries hash-covered `runtime_requirements` derived from
the emitted target code and reflection. Requirements describe the minimum
execution environment; target options describe compilation inputs.

Runtime validates requirements before code loading. Examples include:

- CPU target triple, object format, address size, and calling ABI;
- CUDA PTX version, address size, compute capability, and workgroup limits;
- Vulkan API/SPIR-V versions, limits, and features;
- DirectX API, feature level, Shader Model, root signature, and limits;
- Metal platform, MSL version, minimum OS, and features;
- OpenGL/OpenGL ES API version, profile, and required extensions.

Compiler target availability and runtime device availability are independent.

## 12. Images and graphics

Program image bindings are image-view references. Provider descriptor queries
return immutable parent allocation metadata, selected view metadata, hazard
identity, and ownership kind.

Graphics Nodes bind normalized render-pass, draw, dynamic-state, and shader
resource inputs. Color/depth attachments are resource transitions rather than
shader arguments. Runtime plans image subresource hazards by parent image plus
aspect, mip, and layer ranges.

Compiler-generated resolution and implicit Sampler endpoints are system
endpoints. They are resolved from validated graphics state and do not become
public Program parameters.

The full image contract is [`image_resources.md`](image_resources.md).

## 13. Resource lifetime and failure

Program bindings, provider references, transient descriptor storage, command
resources, and completion objects remain alive until the provider reports
completion.

Destroying a public owner does not release a resource still retained by an
in-flight invocation or pullback. Runtime generation changes invalidate
cached native handles.

Public C entry points contain exceptions. Backend, provider, allocation,
validation, and execution failures become stable status codes and diagnostics.
No failure silently changes backend, narrows dtype, serializes an invalid
dispatch, or falls back to a compatibility path.
