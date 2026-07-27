# ExecutionGraph completion plan

## Status

Active implementation plan. Remove this file after the accepted contracts and
invariants have been merged into the canonical specifications.

The current branch exposes `ExecutionGraph`, command encoders, attachment
operations, and pipeline encoding, but those layers do not yet form one native
GPU execution path. Passing graph compilation tests is not sufficient to call
the execution architecture complete.

## Scope

Complete VernonDSL execution from Python pass declaration through RuntimeCore
and VernonRHI to backend command recording and submission.

This plan does not include Vernon Engine migration, asynchronous multi-frame
execution, or compatibility branches for the incomplete command path.

## Current gaps

### Command recording is descriptive

- `source/lib/rhi/rhi_command.cpp` validates encoder state and records
  statistics, but does not issue backend commands.
- Runtime RHI adapters ignore the provider command-encoder argument and execute
  through their existing immediate paths.
- `vernonRuntimePipelineEncode` therefore still reaches an immediate backend
  invocation rather than recording into the supplied encoder.
- Python `ExecutionGraph` computes scopes and barriers independently, but
  `execute()` does not encode its barriers and each pipeline invocation may
  submit separately.

Consequently, reported render-scope fusion is not physical fusion and graph
barriers do not synchronize backend work.

### Resource identity is unstable

The DirectX 12 retention workaround keeps COM objects in a FIFO deque keyed by
a recyclable RHI slot address. It can keep an old native object alive, but it
does not make a stale provider reference resolve to the correct logical
resource after slot reuse. Independent binding sets also need not release
references in retain order.

The existing slot-reuse regression test checks one retain/release sequence. It
does not encode with the retained reference, reverse destruction order, or
cover image, sampler, attachment, and replacement-binding cases.

This activates phase 1 of
[`deferred_gpu_resource_lifetime_plan.md`](deferred_gpu_resource_lifetime_plan.md).
Phases 2 and 3 remain deferred.

### Public ABI layouts changed

The current changes modify public layouts while retaining their previous
version numbers:

- `VernonRuntimeProviderDrawDescriptor` gains attachment operations and
  scissor fields while the provider ABI remains 4.
- `VernonRhiGraphicsPipelineDescriptor` and
  `VernonRhiRenderingDescriptor` change while the RHI API remains 4.
- `VernonPipelineInvocation` gains an encoder field while the invocation ABI
  remains 6.

Several consumers require `struct_size >= sizeof(current_type)`, so these
changes are not automatically compatible with older callers. Do not change a
version merely to reserve future fields, but do not reuse an existing version
for an incompatible layout.

### Graph validation is incomplete

- C++ `importBuffer` and `importImage` create a new graph resource every time,
  even for the same generation-checked RHI handle.
- C++ resource validation checks only the numeric graph resource ID; a resource
  from another graph can be accepted when its ID is in range.
- Barrier stage masks are not derived from the declared use.
- Python pass flags are mutable booleans. Changing `never_cull`, `side_effect`,
  or `no_merge` after compilation does not mark the graph dirty.
- C++ and Python contain separate scheduling, culling, hazard, barrier, and
  fusion implementations.

### Integration validation is incomplete

The PBR example is currently a one-pass smoke test. Its shadow and environment
features do not exercise sampled depth or cubemap resources. It therefore does
not validate physical pass fusion, cross-pass barriers, or the intended
multi-pass resource chain.

## Implementation order

### Phase 1: make ABI changes explicit

1. Inventory every changed public structure and function across VernonRHI,
   Runtime, RuntimeCore, and the provider SPI.
2. For append-only structures, either accept documented older `struct_size`
   values and supply defaults for missing tails, or assign a new version.
3. For inserted or reordered fields, assign a new ABI/API version unless the
   old layout is restored.
4. Add tests using the previous structure size and version so compatibility is
   demonstrated rather than assumed.
5. Keep one implementation path; do not add old-layout and new-layout backend
   branches.

Acceptance:

- no two incompatible layouts advertise the same version;
- every accepted older structure size has deterministic defaults;
- bundle and provider version diagnostics name the expected and received
  versions.

### Phase 2: implement stable logical resource records

Implement phase 1 of the deferred resource lifetime plan before relying on
retained bindings or graph recording:

1. Replace provider references to recyclable slot addresses with stable
   generation-checked resource keys.
2. Retain resource records transactionally when bindings are created or
   updated, then release prior records.
3. Invalidate public handles immediately on destruction while preventing
   record aliasing and premature native destruction.
4. Replace the DirectX 12 COM deque and all no-op GPU retention callbacks in
   the same change.
5. Keep submission synchronous during this phase.

Acceptance:

- a binding retained across owner destruction still resolves to the original
  resource after public slot reuse;
- binding sets may be destroyed in any order;
- buffer, image, sampler, attachment, and imported-resource tests pass on every
  enabled backend;
- repeated updates do not grow retention storage without bound.

### Phase 3: lower one native command encoder

1. Give each RHI command encoder a backend recording object or an explicit
   borrowed external recording target.
2. Lower barriers, rendering scopes, pipeline binding, dynamic state, clears,
   draws, dispatches, finish, and submit to the selected backend.
3. Make provider `encode_draw` and `encode_dispatch` record into a non-null
   encoder without beginning, ending, or submitting their own command stream.
4. Implement immediate invocation as an ephemeral encoder followed by one
   finish and submit, not as a separate backend path.
5. Make `ExecutionGraph::execute` create one encoder, encode all compiled
   scopes, finish once, and submit once.
6. Define failure cleanup so partially recorded work and retained resources are
   released without submission.

Acceptance:

- two compatible render passes produce one backend begin/end rendering pair;
- one graph execution produces one backend submission;
- barriers cause backend transitions or memory barriers;
- no official RHI adapter ignores the command-encoder argument;
- command statistics are observations of actual recording, not the
  implementation of recording.

### Phase 4: use one graph compiler from Python

1. Bind Python resources and pipeline invocations to the native graph and
   encoder contract.
2. Preserve the class-based `declare()` and `execute()` API while making the
   native graph authoritative for dependencies, culling, barriers, and fusion.
3. Remove the duplicate Python scheduling and barrier implementation.
4. Replace mutable pass flags with APIs that invalidate compiled state.
5. Make graph resource resolution validate graph identity and generation.

Acceptance:

- Python schedule and scope inspection report the native compiled graph;
- changing dependencies, flags, resources, or attachments forces
  recompilation;
- there is one hazard and fusion implementation for C++ and Python execution.

### Phase 5: complete graph correctness

1. Deduplicate repeated imports by resource kind, handle index, and handle
   generation, or reject duplicate imports explicitly.
2. Reject resources from another graph even when numeric IDs match.
3. Derive stage, access, and resource-state transitions from declared uses.
4. Validate attachment format, extent, layer, sample count, location, load,
   store, and read-only compatibility before fusion.
5. Specify intermediate clear and discard behavior inside a fused scope.
6. Ensure culling and failed compilation release all unsubmitted references.

Acceptance:

- aliased imports cannot hide RAW, WAR, or WAW hazards;
- invalid cross-graph resources fail deterministically;
- fused and unfused execution produce equivalent attachment results;
- validation errors identify the pass and resource involved.

### Phase 6: integration and cleanup

1. Add native backend tests that observe real command calls and submission
   counts rather than only graph metadata or encoder statistics.
2. Add stale-binding stress tests under ASan and backend validation layers.
3. Implement sampled depth and cubemap resource chains.
4. Add a real shadow plus PBR multi-pass example and run it on DirectX 12,
   Vulkan, and OpenGL.
5. Reconcile `runtime/design.md`, `runtime/rhi_execution_plan.md`, and the PBR
   validation plan with the implemented ownership and submission boundary.
6. Delete temporary plans after their accepted conclusions are moved into
   canonical design documents.

Acceptance:

- the PBR integration exercises a produced depth resource, a sampled resource,
  and at least two dependent passes;
- full CTest and Python suites pass on all available backends;
- ASan reports no host lifetime errors;
- the working tree contains no command transcripts or generated LLVM install
  artifacts.

## Deferred work

After these phases are complete, phases 2 and 3 of the deferred resource
lifetime plan may add submission serials, completion tracking, deferred
destruction, asynchronous graph submission, and multiple frames in flight.
Those features must not be used to postpone stable logical identity or real
command recording.
