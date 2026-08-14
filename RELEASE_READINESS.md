# VernonDSL 0.1.2 release readiness

Assessment date: 2026-08-02

## Release decision

`0.1.2` is publishable only from an exact commit for which every required gate
below is green. `versions.toml` is the single manually edited version source.

## Supported distribution

- Windows x64, Linux x64, and Apple Silicon macOS wheels;
- CPython 3.11 through 3.14;
- wheel-only distribution for `0.1.2`; no sdist or Intel macOS wheel;
- compiler contract 12 and pipeline contract 16.

Backend capabilities and limitations are defined in
[`RELEASE_NOTES.md`](RELEASE_NOTES.md) and [`SUPPORT.md`](SUPPORT.md). Public
surface and compatibility guarantees are defined in
[`PUBLIC_API.md`](PUBLIC_API.md) and [`COMPATIBILITY.md`](COMPATIBILITY.md).

## Required gates

- [ ] `versions.toml` and all generated version files report release `0.1.2`,
      compiler contract 12, and pipeline contract 16.
- [ ] Native CTest and Python tests pass on Linux, macOS, and Windows on the
      exact release commit.
- [ ] Windows formatting, Ruff, Python coverage, and Runtime AddressSanitizer
      gates pass on that commit; generated-version validation passes in the
      release workflow.
- [ ] Windows x64, manylinux x64, and macOS arm64 wheels build for CPython
      3.11–3.14, pass metadata/platform auditing, and install in clean
      environments.
- [ ] Every installed wheel passes CPU dispatch/readback, frontend and cooker
      module CLI, and bundled Runtime-source presence/version checks.
- [ ] A physical Apple Silicon Mac executes Metal compute, graphics, Argument
      Buffer, dispatch, and readback acceptance. Hosted virtual Metal skips are
      not sufficient for this gate.
- [ ] The release workflow produces one immutable set of wheels, SHA256 sums,
      SBOM, and provenance, stages those files in a GitHub Release, and
      publishes the wheels to PyPI through Trusted Publishing.
- [ ] Clean CPython 3.11 environments on Windows, Linux, and macOS install
      `vernon-lang==0.1.2` from PyPI and repeat the installed-wheel smoke test
      before the GitHub Release leaves draft state.

## Publication procedure

1. Merge the release changes through review.
2. Require all platform and wheel checks on that exact merged commit.
3. Verify the PyPI `pypi` environment and Trusted Publisher configuration.
4. Create the immutable `v0.1.2` tag on that commit.
5. Let the release workflow build, verify, attest, stage, publish, verify from
   PyPI, and then finalize the GitHub Release.
6. Verify GitHub Release and PyPI filenames, hashes, version, and clean install.

Do not move or reuse a failed tag. Any source or packaging change requires a
new version and a complete rerun of the release gates.
