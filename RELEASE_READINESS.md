# VernonDSL 0.1.2 release readiness

Status: active release checklist.

`0.1.2` is publishable only from an exact commit for which every required gate
is green. [`versions.toml`](versions.toml) is the only manually edited version
source.

## Release contract

- latest released version 0.1.1;
- release target 0.1.2;
- Compiler Contract 1;
- Program Version 1;
- CPython 3.11 through 3.14;
- Windows x64, Linux x64, and Apple Silicon macOS wheels;
- wheel-only distribution with no sdist, Intel macOS, PyPy, or 32-bit wheel.

Backend capabilities are defined by [`RELEASE_NOTES.md`](RELEASE_NOTES.md) and
[`SUPPORT.md`](SUPPORT.md). Public and compatibility boundaries are defined by
[`PUBLIC_API.md`](PUBLIC_API.md) and
[`COMPATIBILITY.md`](COMPATIBILITY.md).

## Required gates

- [ ] `versions.toml` and all generated files report release 0.1.2, Compiler
      Contract 1, and Program Version 1.
- [ ] Native builds and all CTest tests pass on Linux, macOS, and Windows.
- [ ] The complete Python suite and MLIR lit suite pass on every required host.
- [ ] Cross-backend language and Program matrices run every applicable case;
      every skip identifies an unavailable platform, device, context, API
      version, or capability.
- [ ] CPU, Vulkan, CUDA, DirectX 12, Metal, OpenGL, and OpenGL ES execute on
      the release hardware assigned to their gates.
- [ ] A physical Apple Silicon Mac runs Metal compute, graphics, Program VJP,
      argument-buffer, dispatch, and readback acceptance.
- [ ] The independent wasm32 build/runtime gate passes, including external
      engine browser rendering and CPU checksum verification.
- [ ] Formatting, Ruff, Python coverage, generated-version validation, and
      Runtime sanitizers pass.
- [ ] Windows x64, manylinux x64, and macOS arm64 wheels build for CPython
      3.11–3.14 and pass metadata/platform auditing.
- [ ] Every wheel installs in a clean environment and passes CPU
      dispatch/readback, frontend, cooker, and bundled Runtime-source checks.
- [ ] Release artifacts contain exactly the expected wheels, SHA256SUMS, SPDX
      SBOM, and provenance.
- [ ] The release workflow publishes through the configured PyPI Trusted
      Publisher and verifies installation from PyPI before finalizing GitHub
      Release.

## Publication procedure

1. Merge all release changes through review.
2. Require every release gate on the exact merged commit.
3. Verify the PyPI environment and Trusted Publisher configuration.
4. Create immutable tag `v0.1.2`.
5. Build, audit, attest, and stage the complete artifact set.
6. Publish to PyPI.
7. Install and verify the published wheels on Linux, macOS, and Windows.
8. Finalize the GitHub Release only after published verification succeeds.

Do not move or reuse a failed tag. Any source or packaging change after tag
creation requires a new version and a complete gate rerun.
