# Compatibility policy

VernonDSL has independent release, compiler-contract, and pipeline version
axes. The 0.1.1 release uses compiler contract 9 and pipeline contract 12.

## Release and ABI

Patch releases in the 0.1 series preserve the documented Python API and public
C ABI. Wheels are supported only on their tagged Python, operating-system, and
architecture combination. VernonDSL 0.1.1 ships CPython 3.11–3.14 wheels for
Windows x64, Linux x64, and macOS arm64.
It is a wheel-only release: no source distribution, Intel macOS wheel, PyPy
wheel, or 32-bit wheel is published.

The bundled Runtime source package has the same release version as its wheel.
Applications embedding that source must rebuild when changing VernonDSL
versions; mixing headers or generated files from different releases is
unsupported.

## Compiler and pipeline contracts

Compiler input must carry the exact supported `vernon.compiler_contract_version`.
Pipeline manifests must carry the exact supported `pipeline_version`.
Incompatible input, reflection, manifests, and artifacts are rejected before
publication or execution rather than interpreted using best-effort fallback.

Generated artifacts and caches include their contract inputs. A compiler- or
pipeline-contract bump invalidates incompatible cache entries. Artifact bytes
are portable only to the target, options, backend capabilities, and ABI
recorded by their manifest.

## Deprecation

Public APIs receive a documented deprecation period of at least one minor
release before removal. Compatibility may be broken without that period only
to correct an exploitable security issue, memory-safety issue, or behavior that
could silently produce incorrect programs. Such changes are documented in the
release notes.
