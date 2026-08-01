# VernonDSL 0.1.1 stable release plan

## Scope

This plan promotes the current `0.1.1a1` development line to the production
`0.1.1` release. The next public candidate should be `0.1.1rc1`; removing the
prerelease suffix is the final promotion step, not the definition of readiness.

The intended stable support range is:

- Windows x64 and Linux x64 are the baseline supported platforms;
- CPython 3.11 through 3.14 are the baseline Python versions;
- every advertised compiler and Runtime backend must have repeatable CI or
  hardware-backed acceptance evidence;
- macOS supports Metal compute and offscreen graphics with cooked MSL bundles;
- macOS packaging must explicitly cover Apple Silicon, define the minimum
  deployment target, and pass clean-wheel installation and execution tests;
- Metal does not include swapchain or window presentation in `0.1.1`;
- a hosted virtual Metal device that skips Argument Buffer coverage is useful
  CI evidence for the supported subset, but it is not a substitute for the
  physical Apple Silicon acceptance gate.

Autodiff is not a `0.1.1` feature or release gate. It remains future
language-v4 work.

## Current readiness

Completed foundations:

- [x] unified runtime TensorView ABI and storage lowering;
- [x] compiler contract 8 and pipeline contract 11 migration;
- [x] synchronous resource lifetime and synchronization contract;
- [x] Windows, Linux, and macOS CI definitions;
- [x] Metal compute and offscreen graphics Runtime implementation;
- [x] explicit Metal Argument Buffer capability probing and deterministic
      unsupported-target handling;
- [x] Vulkan-on-MoltenVK CI configuration for virtual Apple hardware.

Release blockers:

- [ ] update `versions.toml` to `0.1.1rc1` and regenerate every derived version
      file;
- [ ] replace the obsolete Windows-first alpha statements in `README.md`,
      `RELEASE_NOTES.md`, `RELEASE_READINESS.md`, and active specifications;
- [ ] define the shipped Windows, Linux, macOS x64, and macOS arm64 wheel
      matrix, including whether macOS uses separate architecture wheels or
      `universal2`;
- [ ] define and test the minimum supported macOS deployment target;
- [ ] run Metal compute, graphics, Argument Buffer, dispatch, and readback
      acceptance on a physical Apple Silicon Mac;
- [ ] verify each supported CPython 3.11-3.14 wheel in a clean environment;
- [ ] add a trusted release workflow that promotes the exact tested artifacts
      to GitHub Releases and PyPI;
- [ ] publish the stable API, ABI, compatibility, support, and security
      contracts;
- [ ] obtain a fully green release-candidate commit on every required CI job.

No blocker may be converted into an undocumented skip. A platform, wheel, or
backend that cannot satisfy its gate must be removed from the stable support
claim before release.

## Supported backend declaration

The `0.1.1` release notes must describe backends by tested capability rather
than only listing their names:

- CPU: compute reference execution;
- CUDA: compute and buffers only;
- Vulkan: compute and offscreen graphics on supported drivers;
- OpenGL/OpenGL ES: compute and graphics with a supplied compatible context;
- DirectX 12: compute and offscreen graphics on Windows;
- Metal: compute and offscreen graphics on supported Apple devices, with
  Argument Buffer pipelines rejected as unsupported when the device cannot
  create the required encoders.

Unavailable hardware may cause optional development tests to skip. A stable
backend claim, however, requires at least one repeatable hardware-backed
release gate that executes rather than skips its advertised capabilities.

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
- [x] minimize derivable interface metadata, reject the retired per-use fields,
      and converge the candidate on compiler contract 8 / pipeline 11;
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

- require Windows, Linux, and macOS compiler, Runtime, CTest, Python,
  source-build, wheel, and fresh-install CI;
- build and verify a wheel for every supported platform and CPython version;
- define and test the source-distribution policy and minimum deployment
  targets;
- audit Linux wheels for manylinux portability and macOS wheels for architecture
  and deployment-target correctness;
- execute installed-wheel compile, dispatch, readback, CLI, and bundled
  Runtime-source smoke tests;
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

1. Change the version source to `0.1.1rc1`, regenerate version files, and update
   all release-facing documentation.
2. Freeze the public Python, C, and C++ API and the native ABI.
3. Merge the release candidate and require the exact merged commit to pass all
   platform, wheel, backend, compatibility, and packaging gates.
4. Build release artifacts once from that commit and publish `v0.1.1rc1` as a
   GitHub pre-release and PyPI prerelease.
5. Permit only release-blocking fixes. Any code change produces a new release
   candidate and a complete rerun of the gates.
6. After candidate validation, change only the release version and final
   release notes, then run the complete matrix again.
7. Tag the exact green commit as immutable `v0.1.1` and promote the verified
   artifacts through the trusted release workflow.
8. Verify the files exposed by GitHub Releases and PyPI through a clean
   post-publication installation.

Stable completion means that the tag, GitHub Release, PyPI files, provenance,
and post-publication verification all identify the same exact green commit.
