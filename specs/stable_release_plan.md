# VernonDSL stable release plan

## Scope

This plan promotes the Windows-first `0.1.1a1` alpha to a production-grade
stable release. Removing the prerelease suffix is the final step, not the
definition of readiness.

The initial stable support range must be explicit:

- Windows x64 and Linux x64 are the baseline supported platforms;
- CPython 3.11 through 3.14 are the baseline Python versions;
- every advertised compiler and Runtime backend must have repeatable CI or
  hardware-backed acceptance evidence;
- macOS execution and a Metal Runtime remain unsupported unless they acquire
  their own implementation, packaging, and CI gates.

Autodiff is not a `0.1.1` feature or release gate. It remains future
language-v4 work.

## Stable contracts

Before release, publish:

- the supported Python, C, and C++ API surface and which modules remain
  internal;
- the deprecation period and removal process;
- release-version and SemVer policy;
- compiler-contract, pipeline, native ABI, manifest, artifact, and cache
  compatibility rules;
- support and security reporting policies.

Patch releases must preserve the declared public API and native ABI.
Incompatible compiler-contract or pipeline changes must bump the applicable
version axis and deterministically reject or invalidate incompatible artifacts
and caches.

The `0.1.1` Runtime contract is synchronous and permits at most one owned
submission in flight per device. Invocation, `ExecutionGraph.execute()`, and
owned RHI submission complete backend execution before returning. Asynchronous
dispatch, deferred graph execution, and multiple frames in flight are
explicitly outside the stable contract; enabling any of them requires the
serial-tracked reclamation model in
[`deferred_gpu_resource_lifetime_plan.md`](deferred_gpu_resource_lifetime_plan.md)
and a compatible API/ABI version change.

## Unified TensorView ABI and storage lowering

The general TensorView descriptor ABI and the removal of temporary storage
lowering bridges are one stable-release item.

Status (2026-07-30): **complete**.

- [x] One dispatch descriptor contract is shared by Python binding, reflection,
      manifests, C/C++ Runtime input, compiler projection, and enabled backends.
- [x] `physical_index` pass state is replaced by internal physical load, store,
      and atomic operations.
- [x] `VernonToGPU` uses MLIR one-to-many type conversion to lower TensorView
      arguments to scalar storage leaves and rewrites physical load, store, and
      atomic operations without an unrealized-cast bridge.
- [x] Windows incremental builds stage dependency DLLs through target-file
      dependencies in addition to destination `POST_BUILD` copies, so compiler
      and Runtime updates do not require relinking `_native`.
- [x] Concrete shape, stride, and offset specialization paths and cache inputs
      are removed.
- [x] CPU and enabled GPU Runtime bindings consume per-invocation descriptor
      values and reject rank, static-extent, element-layout, access, bounds, and
      backend index-range violations before dispatch.
- [x] Acceptance covers consecutive use of one artifact with dynamic shapes,
      contiguous and transposed addressing, sliced non-zero offsets, and
      negative strides without recompilation.

For each statically declared rank, the descriptor must carry:

- the physical data or backend resource reference;
- the element offset;
- one runtime extent per dimension;
- one signed element stride per dimension;
- the element type and access information required for validation.

Shape, stride, and offset are dispatch data. Production compilation must not
write concrete layouts into MLIR attributes, specialize code artifacts by
layout, or include concrete layouts in code-artifact cache identity.

The migration must:

1. define one descriptor contract consumed by Python binding, reflection,
   manifests, compiler lowering, the C/C++ ABI, and every supported Runtime;
2. replace `physical_index` pass-state attributes with internal physical
   load, store, and atomic operations whose types and operands express that
   projection has completed;
3. replace the ToGPU unrealized-cast/manual-clone bridge with MLIR one-to-many
   type conversion from a logical TensorView to its physical descriptor or
   backend argument sequence;
4. migrate CPU and every enabled GPU backend before deleting the old
   specialization and fallback paths;
5. preserve deterministic symbols, reflection, artifacts, and diagnostics.

Acceptance requires one precompiled artifact to execute consecutively with
different legal shapes, contiguous and transposed layouts, sliced views,
negative strides, and non-zero offsets without recompilation or a
layout-specific cache entry. Rank, element type, access, descriptor bounds, and
backend capability errors must be rejected deterministically before dispatch.

## Remaining compiler and Runtime gates

Status (2026-07-31):

- [x] remove cross-language Value ABI duplication and use one declarative
      layout source or native planner;
- [x] minimize derivable interface metadata in compiler contract 7 / pipeline
      10 and reject the retired per-use fields;
- [x] execute aggregate workgroup tests on available CUDA, Vulkan, OpenGL, and
      DirectX runtimes, including nested values, padding, control flow,
      independent workgroups, and barrier-visible writes;
- [x] prove the synchronization, relaxed-atomic, logical resource-lifetime,
      and external ownership behavior advertised for the synchronous stable
      backend contract;
- [x] define CUDA as compute/buffer-only and reject image/sampler creation,
      while covering OpenGL image/sampler retention and synchronization through
      its context-backed acceptance paths;
- [x] explicitly exclude asynchronous submission, deferred execution, and
      multiple frames in flight from the `0.1.1` stable contract.

Only language behavior already promised by `0.1.1` is a release gate.
Autodiff and the remaining language-v4 expansion are tracked separately.

## Cross-platform release engineering

- add Linux compiler, Runtime, CTest, Python, source-build, wheel, and
  fresh-install CI alongside Windows CI;
- build and verify a wheel for every supported platform and CPython version;
- define and test the source-distribution policy;
- add scheduled or self-hosted GPU validation with validation/debug layers;
- add fuzzing for source, manifest, reflection, TensorView, and ExecutionGraph
  inputs;
- publish benchmark results with stable regression thresholds;
- verify every candidate wheel through compile, dispatch, readback, CLI, and
  bundled Runtime-source checks in a clean environment.

No platform or backend may be advertised as stable without repeatable evidence.

## Trusted release automation

The release workflow must:

1. verify that a `vX.Y.Z` tag matches `versions.toml` and all generated version
   files;
2. require the tag to identify the exact commit that passed all required
   checks;
3. build once and promote the same verified artifacts to GitHub Releases and
   PyPI;
4. use PyPI Trusted Publishing with OIDC instead of a long-lived API token;
5. publish checksums, build provenance, and an SBOM;
6. verify the files exposed by PyPI and perform a clean post-publication
   installation.

Release tags are immutable. A failed or superseded candidate receives a new
version rather than moving an existing public tag.

## Promotion sequence

1. Publish a beta after the contracts, TensorView ABI, cross-platform CI, and
   backend correctness gates pass.
2. Use beta feedback to close API, installation, and compatibility defects.
3. Publish a release candidate and freeze public API and native ABI.
4. Permit only release-blocking fixes during the release-candidate period.
5. Publish stable only after the candidate passes every supported platform,
   backend, upgrade, cache, and artifact-compatibility gate.

Stable completion means that the tag, GitHub Release, PyPI files, provenance,
and post-publication verification all identify the same exact green commit.
