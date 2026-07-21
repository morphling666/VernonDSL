# Unified Pipeline Asset Plan

Status: proposed; not started.

This plan defines the cooked asset contract between VernonDSL and Vernon. It
replaces backend-specific runtime packaging with one pipeline asset directory
format while preserving compatibility with existing `pipeline.bundle`,
`shader.json`, and `compute.json` readers during migration.

## Problem

VernonDSL currently produces several related formats:

- OpenGL `shader.json` bundles with external GLSL files;
- runtime `pipeline.bundle` JSON with inline GLSL or PTX;
- runtime `pipeline.bundle` JSON with base64 SPIR-V;
- CPU `pipeline.bundle` JSON with an external native-library sidecar;
- legacy single-kernel `compute.json` bundles.

Vernon consequently has two overlapping consumers:

- `CompiledShaderBundle` / `ShaderProvider` for OpenGL `shader.json`;
- `PipelineProvider` / VernonRuntime for `pipeline.bundle`.

CPU appears more complex because the native library must remain on disk and
Vernon special-cases the directory loader. The hand-written CPU bundle block in
`Vernon/render/tests/CMakeLists.txt` is test fixture generation, not production
asset loading, and its `VERNON_CPU_PIPELINE_BUNDLE_PATH` output currently has no
Vernon source consumer.

## Decisions

### Asset naming

Use Vernon's existing `<asset-name>.<asset-type>.json` convention:

```text
<asset-directory>/
  <asset-name>.pipeline.json
  artifacts/
    <sha256>.<backend-extension>
```

Examples:

```text
cpu_compute_asset/
  cpu_compute_asset.pipeline.json
  artifacts/
    <sha256>.dll

cuda_compute_asset/
  cuda_compute_asset.pipeline.json
  artifacts/
    <sha256>.ptx

basic_material/
  basic_material.pipeline.json
  artifacts/
    <sha256>.vert.glsl
    <sha256>.frag.glsl
```

Backend names are not asset types. Do not introduce `.cpu.json`,
`.cuda.json`, or `.vulkan.json`. The manifest's `target` field selects the
backend. A cooker invocation produces one target-specific asset directory;
multi-target manifests are outside this migration.

The `artifacts/` directory is mandatory for cooked assets. Extensions aid
inspection and tooling, but SHA-256 is the artifact identity.

### Cooked versus interactive storage

All cooked/deployment artifacts are external:

- GLSL and GLSL ES source;
- PTX;
- SPIR-V;
- CPU `.dll`, `.so`, or `.dylib`.

The JSON manifest contains no compiled payload bytes. This avoids duplicate
source in JSON, SPIR-V base64 expansion, and backend-specific loading rules,
while enabling content-addressed deduplication and independent caching.

Non-persistent Python interactive/JIT execution may retain inline artifacts
because it has no durable asset directory. Inline and external descriptors use
the same schema, but the cooker always emits external storage.

### Runtime ownership

`PipelineProvider` and VernonRuntime are the only production consumers of
VernonDSL pipeline assets. Vernon must not parse backend artifact internals.

`CompiledShaderBundle` and `shader.json` remain read-only compatibility paths
during migration. `compute.json` remains a legacy single-kernel API and is not
the production pipeline asset format.

## Pipeline schema 2

The new manifest is UTF-8 JSON:

```json
{
  "schema_version": 2,
  "type": "pipeline",
  "id": "pipelines/cpu_compute_asset",
  "target": "cpu",
  "invocation_abi_version": 1,
  "features": [],
  "variants": [],
  "stage_artifacts": {},
  "content_hash": "<canonical-manifest-sha256>"
}
```

Every cooked stage uses one artifact descriptor:

```json
{
  "id": "<stage-id>",
  "stage": "compute",
  "entry": "main",
  "target": "cuda",
  "reflection": {},
  "artifact": {
    "format": "ptx",
    "storage": "external",
    "path": "artifacts/<sha256>.ptx",
    "size": 1234,
    "sha256": "<sha256>"
  }
}
```

Interactive execution may instead use:

```json
{
  "format": "ptx",
  "storage": "inline",
  "encoding": "utf8",
  "data": "...",
  "size": 1234,
  "sha256": "<sha256>"
}
```

SPIR-V inline data uses `encoding: "base64"`. Cooked SPIR-V is a normal
external `.spv` file and requires no base64 encoding.

CPU stage records additionally retain:

- exported symbol;
- operating system;
- architecture;
- CPU invocation ABI version.

These are stage metadata, not filename or directory components.

## Validation invariants

- Manifest discovery uses the `.pipeline.json` suffix.
- Each manifest path is interpreted relative to its own parent directory.
- External artifact paths must be relative, normalized, and contained beneath
  that directory; absolute paths and `..` escapes are rejected.
- The loader verifies file size and SHA-256 before backend use.
- Duplicate asset IDs, stage IDs, feature keys, and parameter slots are
  rejected.
- CPU additionally validates operating system, architecture, symbol, and ABI
  before loading the native library.
- The manifest `content_hash` is computed from canonical JSON with the
  `content_hash` field omitted.
- Exact feature-variant matching remains unchanged.
- CPU and CUDA remain dispatch-only until their runtime capabilities are
  deliberately extended.

## Implementation phases

### 1. Record and parse schema 2

Update:

- `specs/runtime/pipeline_runtime_unification.md`;
- Vernon's `specs/render/design.md`;
- `source/include/VernonRuntime.h` documentation;
- `source/lib/VernonRuntime.cpp`.

Add a common artifact resolver in VernonRuntime that:

1. parses the descriptor;
2. resolves inline bytes or an external path;
3. validates containment, size, and SHA-256;
4. returns validated bytes or a validated native-library path;
5. dispatches the result to the selected backend.

Accept both legacy `vernon_pipeline_bundle` schema 1 and pipeline schema 2
during migration. New producers emit only schema 2.

### 2. Unify VernonDSL production

Refactor:

- `python/vernon_dsl/shader_assets.py`;
- `python/vernon_dsl/shader_asset_cli.py`;
- `python/vernon_dsl/runtime.py`;
- relevant cooker and runtime tests.

The cooker must:

1. compile each stage;
2. hash its exact artifact bytes;
3. write `artifacts/<sha256>.<ext>`;
4. deduplicate identical artifact content;
5. write `<asset-name>.pipeline.json`;
6. never inline cooked artifact payloads.

Interactive runtime generation may emit schema-2 inline descriptors.

Stop producing `pipeline.bundle`. Continue producing `shader.json` only while
legacy tooling needs it, and mark it deprecated for execution.

### 3. Unify Vernon consumption

Refactor Vernon's
`Vernon/render/asset/shader/pipeline_asset.cpp`:

- discover `*.pipeline.json`;
- retain legacy `pipeline.bundle` discovery temporarily;
- always pass the manifest directory through
  `vernonRuntimeLoadPipelineBundleWithOptions`;
- remove the CPU-only `LoadPipelineBundleFromDirectory` branch;
- leave target-specific context and invocation behavior below the common
  loader.

Route VernonDSL graphics assets through `PipelineProvider`. Keep
`CompiledShaderBundle` only for explicitly legacy OpenGL assets, then remove it
after all production assets migrate.

### 4. Remove hand-written test packaging

Remove the unused CPU fixture generation from
`Vernon/render/tests/CMakeLists.txt`. Vernon CMake must not construct bundle
JSON.

Add focused cross-repository integration fixtures that mount complete asset
directories produced by the VernonDSL cooker. Keep low-level CPU artifact
tampering tests in VernonDSL, where path escape, hash mismatch, platform
mismatch, and ABI mismatch can be tested without duplicating the schema in
Vernon.

### 5. Retire compatibility paths

After all callers migrate:

- stop scanning `pipeline.bundle`;
- stop using `shader.json` for execution;
- deprecate or remove the OpenGL-only `CompiledShaderBundle` parser;
- keep `compute.json` only if the public single-kernel API still requires it.

Removal is a separate compatibility decision and must not be combined with the
initial schema-2 rollout.

## Verification

### VernonDSL

- Cooker tests for CPU, CUDA, OpenGL, OpenGL ES, and Vulkan verify the exact
  `<name>.pipeline.json + artifacts/<sha256>.<ext>` layout.
- Verify deduplication, deterministic canonical JSON, and stable content hashes.
- Verify corrupt size/hash, missing artifact, absolute path, parent traversal,
  symlink escape, wrong OS/architecture, and wrong CPU ABI diagnostics.
- Verify schema-1 loading compatibility.
- Verify interactive inline GLSL, PTX, and SPIR-V.
- Run all VernonRuntime CPU/CUDA/OpenGL/Vulkan pipeline tests and Python tests.

### Vernon

- Build the Release `tests` target.
- Test `PipelineProvider` discovery and duplicate-ID behavior for
  `.pipeline.json`.
- Mount cooker-produced CPU and CUDA compute assets and verify persistent
  tensor dispatch.
- Mount a cooker-produced OpenGL pipeline asset and invoke it using imported
  engine buffers and textures.
- Run GUI OpenGL verification interactively when WGL context creation is not
  available to automation.
- Confirm Vernon contains no backend artifact parsing and no CMake-generated
  bundle JSON.

## Expected end state

```text
VernonDSL compiler/cooker
  -> <asset-name>.pipeline.json
  -> artifacts/<sha256>.<ext>
  -> Vernon PipelineProvider
  -> VernonRuntime common artifact resolver
  -> CPU / CUDA / OpenGL / OpenGL ES / Vulkan backend
```

The asset directory and manifest contract are backend-independent. Only
backend execution and genuinely backend-specific metadata remain specialized.
