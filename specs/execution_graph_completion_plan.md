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

## Completed foundations and current gaps

### Native command recording is complete

- Every official GPU adapter requires a non-null provider command encoder.
- Immediate invocation creates an ephemeral RHI encoder and uses the same
  adapter recording path as graph execution.
- RHI barriers are batched per backend call, and failed or abandoned recordings
  roll back optimistic native resource-state tracking.
- Fused scopes use a bounded backend attachment-object cache; entries in use by
  the active encoder cannot be evicted, and overflow objects are command-owned.
- Per-encoder resource, cleanup, and rollback deduplication is O(1).
- OpenGL tracks unconsumed compute writes so submit emits a fallback memory
  barrier only when no explicit resource barrier consumed them.
- Synchronous DirectX 12 and Vulkan devices own one reusable command frame.

### Resource identity is stable

Provider references and graph imports retain generation-checked logical
resource records. Public destruction invalidates handles immediately without
allowing slot reuse to alias a retained record.

### Public ABI baseline is complete

There is one current command layout and one backend recording path. Callers
must provide the current `struct_size`; no compatibility normalization is
performed, and the existing RHI, provider, and invocation version constants
remain unchanged.

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

### Phase 1: establish the completed ABI baseline (complete)

1. Keep only the completed public structures used by the native command path.
2. Remove old layout definitions, compatibility normalization, and old backend
   branches.
3. Require the current `struct_size` at every affected boundary.
4. Keep the existing version constants unchanged for this unreleased baseline.

Acceptance:

- there is one current layout and one implementation path;
- older incomplete structure sizes are rejected rather than normalized;
- no compatibility-only data structure or backend branch remains;
- RHI API 4, provider ABI 4, and invocation ABI 6 remain unchanged.

### Phase 2: implement stable logical resource records (complete)

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

### Phase 3: lower one native command encoder (complete)

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
7. Retain deduplicated RHI resources and prepared provider objects until owned
   submission completes or the owner destroys a borrowed encoder.
8. Use a generation-checked O(1) registry with per-encoder synchronization;
   backend fence waits must not hold the registry lock.
9. Apply fused-scope load/store operations once, map barriers from declared
   state/stage/access, and keep descriptor storage valid until submission.
10. Roll back optimistic backend resource-state tracking when an encoder is
    abandoned before submission.
11. Batch native barriers and reuse immutable descriptor/rendering objects
    within their retained logical-resource lifetime.
12. Bound persistent backend caches, make command deduplication O(1), and keep
    only one native command frame while submission remains synchronous.
13. Snapshot mutable Vulkan bindings per encoder revision and resolve numeric
    binding slots in O(n) without repeated layout scans.

Acceptance:

- two compatible render passes produce one backend begin/end rendering pair;
- one graph execution produces one backend submission;
- barriers cause backend transitions or memory barriers;
- no official RHI adapter ignores the command-encoder argument;
- command statistics are observations of actual recording, not the
  implementation of recording;
- destroying resource, pipeline, or binding owners after encode cannot
  invalidate recorded commands;
- fused final discard and same-state write barriers preserve backend semantics;
- repeated fused draws neither recreate attachment descriptors nor overwrite
  descriptor storage still referenced by the active command stream.
- abandoned recordings leave tracked native resource state unchanged;
- repeated identical binding sets and Vulkan fallback scopes reuse cached
  backend objects without unbounded attachment retention;
- repeated DirectX 12 compute dispatches with unchanged bindings reuse
  descriptors within the active encoder;
- multiple Vulkan invocations in one encoder retain their original descriptor
  and inline-value snapshots;
- explicit OpenGL resource barriers suppress the redundant submit fallback;
- no manual compute-to-graphics compatibility API remains.

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
