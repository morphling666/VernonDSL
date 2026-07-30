# VernonDSL 0.1.1a1 release readiness

Assessment date: 2026-07-30

## Recommendation

`0.1.1a1` is ready as a local Windows-first alpha release candidate. Tag and
publish only after the exact merged `master` commit passes the complete Windows
CI matrix.

- Go: GitHub pre-release and Windows wheels after final CI.
- No-go: beta or stable release.
- Do not publish Linux/macOS wheels or claim a Metal runtime.

`0.1.0` already exists on PyPI, so `0.1.1a1` is the next valid PEP 440 preview
version. `versions.toml` is the single manually edited version source.

## Local acceptance evidence

- Ruff lint and formatting passed.
- Python: 340 tests and 349 subtests passed.
- Language/frontend line coverage: 91.81%.
- Inference/type-parser branch coverage: 85.04%, above the 85% gate.
- Native Release CTest: 135/135 passed.
- AddressSanitizer CTest: 83/83 passed.
- Generated-version drift check passed.
- CPython 3.11 Windows wheel built and passed `twine check`.
- A fresh environment installed the wheel, compiled and dispatched a CPU
  kernel, read results back, compiled through the installed frontend CLI, and
  validated bundled Runtime sources and release version.
- Terrain and Mandelbulb smoke acceptance executed on Vulkan, DirectX 12, and
  OpenGL at 64x64.

Environment-dependent native skips:

- CUDA image/sampler lifetime;
- OpenGL synchronization.

The latest referenced remote CI for commit `962ab1d` passed:
<https://github.com/morphling666/VernonDSL/actions/runs/30472583146>. It is
evidence for that commit only; the final merged release commit requires its own
green run.

## Release contract

The alpha contract is published in [`RELEASE_NOTES.md`](RELEASE_NOTES.md):

- Windows is the only CI and prebuilt-wheel platform.
- Supported wheels target CPython 3.11 through 3.14.
- Metal produces source artifacts but has no Vernon runtime.
- Graphics is offscreen with host readback, not swapchain presentation.
- GPU tests may skip when hardware, loaders, contexts, or drivers are absent.
- Dynamic multi-rank TensorView layouts require AOT specialization.
- CPU graphics, CUDA image/sampler resources, f16/f64 vertex attributes, and
  non-relaxed atomics are outside the supported alpha subset.
- Production-grade asynchronous multi-frame resource reclamation is deferred.
- Released builds remain frontend version 3; language v4 is a roadmap target.

## Final release steps

1. Commit and push the release candidate.
2. Merge through a reviewed PR into `master`.
3. Require the Windows style, Runtime, ASan, compiler, Python 3.11-3.14 wheel,
   fresh-wheel, and showcase jobs to pass on the merged commit.
4. Tag that exact commit as `v0.1.1a1`.
5. Create a GitHub pre-release using `RELEASE_NOTES.md`.
6. Attach only CI-produced wheels that passed fresh-environment verification.
7. Publish the same verified artifacts to PyPI.

## Before beta

- add Linux CI and verify compiler, Runtime, install, and source builds outside
  Windows;
- define a stable public API and deprecation policy;
- publish release, compiler-contract, pipeline, ABI, and cache compatibility
  guarantees;
- close the high-priority compiler architecture work in
  [`specs/completion_roadmap.md`](specs/completion_roadmap.md);
- execute aggregate workgroup storage tests on available GPU runtimes;
- decide the language-v4 autodiff and synchronization gates;
- improve repeatable hardware-backed GPU acceptance reporting.

## Before stable / GA

- continuously test every supported platform and distribution path;
- prove resource lifetime and synchronization across supported runtimes;
- publish security reporting, support, ABI, and cache policies;
- implement or explicitly remove major deferred promises;
- automate release creation and artifact publication from an exact green tag.
