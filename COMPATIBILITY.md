# Compatibility policy

VernonDSL has independent release, compiler-contract, and Program version
axes. Their current values are defined only by `versions.toml`. There is no
pipeline-version axis.

Contract numbers are frozen within a release line. Implementing another
backend, fixing a bug, adding tests, or completing already documented behavior
does not change any version. A contract number changes only after an explicitly
approved release cut changes source semantics, serialized schema, or public
ABI. Agents and implementation changes must not infer or perform such a bump
automatically.

## Release and ABI

Published patch releases preserve their documented Python API and public C
ABI. Wheels are supported only on their tagged Python, operating-system, and
architecture combination. The release matrix targets CPython 3.11–3.14 on
Windows x64, Linux x64, and macOS arm64. VernonDSL is wheel-only: no source
distribution, Intel macOS wheel, PyPy wheel, or 32-bit wheel is published.

The bundled Runtime source package has the same release version as its wheel.
Applications embedding that source must rebuild when changing VernonDSL
versions; mixing headers or generated files from different releases is
unsupported.

## Compiler and Program contracts

Compiler input must carry the exact supported `vernon.compiler_contract_version`.
Cooked Program bundles must carry the exact supported `program_version`.
The current Program contract packages one canonical Program and direct
logical-Stage artifacts;
execution topology and autodiff signatures come only from Program.
Older pipeline/program-bundle documents are rejected rather than normalized or
interpreted as current-schema aliases.
Incompatible input, reflection, manifests, and artifacts are rejected before
publication or execution rather than interpreted using best-effort fallback.

Generated artifacts and caches include their contract inputs. An explicitly
approved future Compiler Contract or Program Version cut invalidates
incompatible cache entries. Artifact bytes are portable only to the target,
options, backend capabilities, and ABI recorded by their manifest.

## Deprecation

Public APIs receive a documented deprecation period of at least one minor
release before removal. Compatibility may be broken without that period only
to correct an exploitable security issue, memory-safety issue, or behavior that
could silently produce incorrect programs. Such changes are documented in the
release notes.
