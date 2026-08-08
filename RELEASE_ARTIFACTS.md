# Release artifacts

VernonDSL 0.1.2 is a wheel-only release. A complete release contains exactly
12 wheels: CPython 3.11–3.14 for Windows x64, Linux x64, and macOS 15 arm64.

The trusted release workflow validates the complete matrix before publication
and produces:

- the 12 wheel files published unchanged to PyPI and GitHub Releases;
- `SHA256SUMS` covering every wheel;
- an SPDX JSON software bill of materials;
- GitHub build-provenance attestations bound to the release files.

The release tag, wheel metadata, generated version files, GitHub Release, and
PyPI project must all identify the same release version. GitHub provenance
attestations bind the payload files to the exact workflow and tagged commit.
Publishing requires an immutable `vX.Y.Z` tag merged into `master`.

A manual dry run builds and validates the complete payload without publishing.
After publication, clean CPython 3.11 environments on Windows, Linux, and macOS
install from PyPI and repeat CPU dispatch/readback, frontend and cooker module
CLI, and bundled Runtime-source presence/version checks. Release tags are never
moved or reused.
