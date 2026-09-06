# Unified Program Asset Architecture

## 1. Status

This document defines the target architecture for the coordinated Program Asset release. The source tree is currently
mid-migration: the Program names and part of the canonical execution path exist, but legacy stage deployment, schema
dispatch, direct submit/autodiff entry points, and sentinel controls have not all been removed. The implementation is
not release-complete until §13 passes.

Contract changes land atomically at the release boundary, without compatibility aliases, dual schemas, or partially
renamed APIs. Intermediate working-tree state is not a supported contract and must never be described as one.

Normative documents this plan depends on and must not contradict:

- `specs/program_execution_manifest.md` — the canonical Program deployment representation (the `program` object).
- `specs/graphics_execution_architecture.md` — unified graphics execution.
- `specs/unified_program_vjp.md` — Program-level VJP model.
- `specs/unified_module_pipeline.md` — the one Program model shared by Module and Pipeline authoring.

This plan defines the **asset envelope, the layering, and the lifecycle**. It does not redefine the canonical
`program` object; that belongs to `specs/program_execution_manifest.md`.

## 2. Goal

Make **Program Asset** the only cookable deployment concept.

Every authored program — bare compute Kernel, graphics `vd.pipeline(...)`, Kernel VJP, Module, Module VJP — lowers to
one Program IR, serializes to one manifest schema, and executes through one C/C++ load and invocation API. Everything
else is deleted, not deprecated.

Success is measured by subtraction: after this migration there is exactly one cook front end, one cooked schema, one
loader, one invocation path, and one autodiff ABI. If any of those still has two forms, the migration is not done.

## 3. Vocabulary: one word, one meaning

`Pipeline` currently means three different things. That ambiguity is the root cause of most of the duplication below,
so fix the vocabulary first.

| Term | Exactly one meaning after this migration |
| --- | --- |
| **Program** | A complete callable graph: values, storages, stages, graphs, ABI. |
| **Program Asset** | A Program in cookable (declared) or deployed (cooked) form. |
| **Pipeline** | Two narrow, non-asset uses only: the live `vd.pipeline(...)` graphics authoring object, and an internal native compute/graphics stage pipeline owned by the Program executor. |
| **Variant** | One feature-key specialization of a Program Asset. |
| **Node** | One invocation in a Program graph. It owns Program Value operands/results, resource accesses, controls, and endpoint projections. |
| **Stage** | A reusable portable implementation contract referenced by one or more Nodes. It owns no invocation-specific Value binding. |
| **Artifact** | One target-native compiled blob plus its entry point. |
| **Resolved Execution Plan** | The immutable per-variant physical plan: node implementations, carriers, transfers, residency, hazards, and publication transactions. |

`Pipeline` is never again an asset type, a bundle type, a manifest `type` value, a cooked file suffix, or a public
load/invoke type.

## 4. Layered architecture

Eight layers, strictly one-way dependencies. A layer may depend only on layers above it in this list. No layer may
import a private symbol from a lower layer, and no layer may re-derive a decision an upper layer already made.

```
L0  Authoring        vd.kernel / vd.pipeline / vd.Module / vd.ad.vjp
                     -> typed Python objects
L1  Declaration      program_asset(...) -> ProgramAssetDeclaration
                     -> id, program object, variant keys
L2  Capture          declaration -> CapturedProgram
                     -> canonical Program IR + per-stage implementation requests
L3  Compile          implementation requests -> CompiledStage
                     -> provider chain, target artifacts, portable reflection
L4  Deployment       CapturedProgram + CompiledStage -> ProgramManifest + blobs
                     -> .program.json
--------------------------------------------- process boundary
L5  Load             .program.json -> VernonProgramBundle
                     -> immutable authenticated deployment description
L6  Resolve          bundle + feature key -> VernonProgramExecutable
                     -> immutable ResolvedExecutionPlan + backend stage pipelines
L7  Execute          VernonProgramInstance -> VernonProgramInvocation
                     -> concrete bindings, forward, pullback
```

### L1 Declaration

`program_asset(*, id, program, variants)` returns `ProgramAssetDeclaration`. It validates only what is knowable from
the authored objects: that `program` is one of the five accepted typed forms, and that variant keys are canonical.

Accepted `program` forms, and nothing else:

| Authored form | Program shape |
| --- | --- |
| compute Kernel | one-node compute Program |
| `vd.pipeline(...)` | one-node graphics Program with symbolic render-pass, draw, and dynamic-state boundaries |
| Kernel VJP (`vd.ad.vjp(kernel, ...)`) | forward/backward Program graphs via `CapturedVjpDslProvider` |
| Module | canonical Program parser; the body may mix compute Kernel calls and `vd.pipeline(...)` calls |
| Module VJP | canonical Program parser with VJP request; compute-only body |

Tuples of graphics entry functions are **not** accepted. `vd.pipeline(...)` is the only graphics authoring form. This
removes tuple topology validation from the declaration layer entirely, since `vd.pipeline` already owns it.

### L2 Capture — one front end, dispatching on Python type

**This is the single most important structural change in the plan.**

Capture takes a `ProgramAssetDeclaration` and dispatches on the **Python type of `declaration.program`**. There is one
capture entry point, five typed branches, and one output type:

```
capture_program(declaration) -> CapturedProgram
```

`CapturedProgram` carries the canonical Program IR and the set of implementation requests needed to realize it. It is
the same type for all five branches. Downstream layers cannot tell which branch produced it, and must not ask.

Kind discrimination therefore happens exactly once, at exactly one place, on real objects. Nothing downstream carries
a kind tag, a nullable "canonical" field, or a schema selector.

All five branches produce the same non-null `CapturedProgram`. Kernel structured VJP is an implementation provider
used while compiling differentiated compute requests; it is not a sixth capture result, a stage plan alternative, or
a deployment ABI. `CapturedProgram`, variant deployment types, and `VernonProgramBundle` contain no nullable
"canonical" member and no legacy alternative.

**Static AST parsing is not a kind oracle.** The current tree guesses a `program_kind` of `"stages"` or `"module"`
from the AST shape of the `program=` argument, then the Module branch executes the source anyway. The "parse without
execution" property is already false for half of all inputs, and the guess is wrong for `vd.pipeline(...)`: it parses
as `ast.Call` → `"module"` → cook fails with `declared a host Program that is not a Vernon Module`. That is the bug
behind "make actual `vd.Pipeline` declarations work".

Resolution: the cook front end evaluates the asset source and reads the typed declaration. The static parser is
demoted to an **optional declaration-site lint** with no role in dispatch, keeping only the checks that are genuinely
static and useful as fast preflight:

- the `id` is a literal;
- variant keys are canonical, unique, locally declared, and within the variant cap;
- referenced entry names resolve inside the project.

If that lint is not worth its own maintenance cost, delete `parsing.py` outright rather than keeping a second,
partially-correct model of what an asset is. Do not keep it as a dispatch input.

Also delete the parse-time `{"type": "python_pipeline_asset", ...}` stringly-typed dict. It duplicates the typed
descriptor beside it, and a canonical-JSON blob is not a substitute for a type.

### L2 capability boundary

Enforce capability limits during capture, before any provider lowering, with one diagnostic vocabulary:

- Graphics is supported in both graphics forms. A `vd.pipeline(...)` asset and a Module that calls one both
  lower to a `GraphicsCallOp` node with symbolic render-pass, draw, and dynamic-state controls. Do not add a
  "graphics Modules are rejected" rule; that capability already exists and is covered by
  `test_module_graphics_controls.py`.
- Reject graphics VJP in every form: a Module VJP whose capture contains a graphics call, and a graphics
  Program carrying a transform. Both are rejected by `capture_program`; the public API has no graphics
  custom-rule-set branch.
- Reject unsupported autodiff resources.

These are declared capability rules evaluated at one layer, not ad-hoc `isinstance` guards scattered across
declaration, parsing, and cooking.

### L3/L4 Compile and Deployment

Compile turns implementation requests into `CompiledStage` values. Deployment turns `CapturedProgram` plus
`CompiledStage` values into the manifest and blob set. Deployment is a **pure function**:

```
build_program_manifest(captured, compiled_stages, target) -> ProgramManifest
```

It owns the whole envelope for every variant. It does not sniff its input to choose a schema, does not accept a
nullable Program, and does not import anything from the cook orchestrator.

### L6 Resolve — one physical planning authority

Resolve selects a variant and constructs one immutable `ResolvedExecutionPlan`. This is the only layer that projects
logical Program semantics onto physical backend execution. Its plan contains:

- one record per `(graph, node)`, never one mutable binding record per Stage;
- the selected StageArtifact and backend pipeline for each Node;
- exact logical-value/leaf to physical-endpoint projections;
- required host/device residency and explicit upload, readback, and device-copy edges;
- resource alias domains, derived RAW/WAR/WAW dependencies, and backend barriers;
- graphics scope planning and render-pass compatibility;
- tape, residual, replay, and checkpoint storage requirements;
- publication transactions and their commit mode.

Stage contracts and backend pipelines may be shared by several Nodes; Node projections may not. L7 executes this
plan and must not inspect target carriers to rediscover residency, scan Storage aliases to rebuild transfer policy, or
choose publication behavior from the concrete resource kind.

The current `bundle/serialize.py` does the opposite on all three counts: it picks between two schemas by duck-typing
`variant.canonical_program`, and reaches upward with a function-local
`from .._shader_assets.cooking import _canonical_deployment` to dodge a circular import. Deleting that import is a
required outcome; a function-local import of a private upper-layer symbol is the signature of inverted layering, and
the fix is to move canonical deployment construction down into the deployment layer where it belongs.

## 5. One deployment schema

One schema, all variants, all program forms. Top-level envelope:

```json
{
  "compiler_contract_version": 14,
  "program_version": 19,
  "type": "program",
  "id": "shaders/example",
  "target": { "kind": "vulkan", "options": {} },
  "blobs": {},
  "variants": [
    {
      "key": ["FEATURE"],
      "program": {},
      "artifact_system": {
        "runtime_requirements": {},
        "artifacts": {}
      }
    }
  ],
  "content_hash": ""
}
```

- `program` is the canonical Program object defined by `specs/program_execution_manifest.md`.
- `compiler_contract_version` and `program_version` jointly select the compiler/reflection and Program contracts;
- root `target` is the one target shared by every variant in the bundle;
- root `blobs` is the content-addressed authenticated byte store shared across variants;
- variant `artifact_system.runtime_requirements` is the selected variant's aggregate requirement;
- variant `artifact_system.artifacts` is keyed directly by that variant's logical Program Stage ID. There is no
  Stage-to-artifact binding table. Identical code ranges are shared through root Blob IDs.
- `content_hash` is the SHA-256 of the canonical JSON of the document with `content_hash` removed.

`program_version` replaces the `pipeline_version` manifest key. Only the name changes; the **value** is not bumped by
this migration, per the contract-version rule in `AGENTS.md`.

This rename lands in **Phase 3**, with the schema rewrite — not in the Phase 1 surface rename. It reaches the
compiler contract surface, so it must not be spread across two phases, and Phase 3 rewrites the manifest anyway.

That rename is **not** an edit to `python/vernon_dsl/_versions.py`, `source/include/VernonVersions.h`, or
`cmake/VernonVersions.cmake`. All three are generated and marked `Do not edit`. The rename is made in
`versions.toml` (the `pipeline` key) and `tools/generate_versions.py`, and the generated outputs follow. The
generator emits six coupled names, and all six must move together or the migration will need a shim:

| Generated | After |
| --- | --- |
| `PIPELINE_VERSION` (Python) | `PROGRAM_VERSION` |
| `VERNON_PIPELINE_VERSION` (C header) | `VERNON_PROGRAM_VERSION` |
| `VERNON_PIPELINE_VERSION_STRING` (C header) | `VERNON_PROGRAM_VERSION_STRING` |
| `VERNON_PIPELINE_JSON_FIELD` (C header, literal `"pipeline_version":N`) | `VERNON_PROGRAM_JSON_FIELD`, literal `"program_version":N` |
| `VERNON_PIPELINE_VERSION` (CMake) | `VERNON_PROGRAM_VERSION` |
| `vernon.pipeline_version` (MLIR module attribute) | `vernon.program_version` |

The MLIR module attribute and the `VERNON_PIPELINE_JSON_FIELD` literal are the two easiest to miss: the attribute is
also rewritten by a regex inside the generator, and the literal is a hardcoded JSON prefix used as a fast-path
matcher. Renaming the manifest key without both leaves a manifest whose declared key no longer matches the string the
runtime scans for.

### Deleted from the schema

The rightmost column cites the §9 rule the field violates, so each removal is verifiable rather than aesthetic.

| Deleted | Why | Rule |
| --- | --- | --- |
| `type: "pipeline"` | Second cooked schema. | 11 |
| `type: "program_bundle"` | Third name for the same thing. | 11 |
| `stage_artifacts` | Stage-based schema; superseded by `artifact_system.artifacts`. | 11 |
| `stage_bindings` | Unconditionally an identity map (`stage_bindings[s] = s`). An always-identity indirection is not indirection. `artifact_system.artifacts` is already keyed by logical stage; index it directly. | 5 |
| variant `parameters`, `outputs`, `internal_parameters` | Duplicate the Program's own `parameters` and `abi`. | 9 |
| variant `program` as `{stage_name: stage_id_hash}` | Second meaning for the key `program`; the canonical Program object is the only meaning. | 1, 2 |
| bundle-level `autodiff` with the `profiles` table | Second deployment ABI for autodiff. See §7. | 9 |
| `program.shape_symbols`, `program.shape_constraints`, `program.alias_preconditions`, `residual_contract.shape_symbols` | Required by the C++ parser **and required to be empty** (`program_execution_manifest.cpp` rejects any non-empty value). A field that must exist and must always be empty carries no information. Remove it from the schema and from the parser; add it back, non-empty and meaningful, when the feature lands. | 4 |

Keep target-native stage artifacts and `implementation.metadata` behind canonical stage contracts. Backend pipeline
construction stays private to the Program executor.

The physical transfer schedule is not serialized. The loader authenticates artifacts; L6 derives the
`ResolvedExecutionPlan` from Program Value/Storage semantics, Node endpoint projections, portable artifact reflection,
and backend capabilities.

Two nearby cases are **not** deletions, and the rule-4 sweep must not remove them. A graphics `forward` graph's
`captures` must be empty because captures belong to the backward graph, and a graphics `system` endpoint's
`abi.bindings` must be empty because a system value has no bindings. Both are semantic rules about a specific variant,
not placeholders for unimplemented features. If the second is retained, the review question is whether `abi` belongs
on a system endpoint at all — not whether the emptiness check is a workaround.

### Multi-variant deployment is not optional

`_canonical_deployment` currently raises `canonical deployment currently requires exactly one variant`, and
`_compile_module_bundle_plan` works around it by building a separate plan per variant and then re-splitting with
`dataclasses.replace(plan, variants=(variant,))` inside `materialize_bundle`. Both the restriction and the workaround
go away: the deployment layer takes all variants and emits all variants in one pass.

## 6. One load and execution lifecycle

Four types, four transitions, one direction. No kind flags, no optional "actually a Program" fields, no re-entry.

| Type | Created by | Mutability | Holds |
| --- | --- | --- | --- |
| `VernonProgramBundle` | `vernonRuntimeLoadProgramBundleWithOptions` | **immutable** | parsed manifest for all variants; no backend objects |
| `VernonProgramExecutable` | select one variant by feature key | **immutable** | that variant's Program, materialized backend stage pipelines, and one `ResolvedExecutionPlan` |
| `VernonProgramInstance` | `vernonRuntimeProgramInstanceCreate` | mutable | persistent binding state, resource leases, caches, telemetry |
| `VernonProgramInvocation` | `vernonRuntimeProgramInstanceBeginInvocation` | mutable | one transaction: concrete bindings, grid, render pass, draw, dynamic state |

Public surface, and nothing beyond it:

- `VernonProgramBundle`, `VernonProgramBundleLoadOptions`, `VernonProgramExecutable`
- `vernonRuntimeLoadProgramBundleWithOptions`
- Program boundary reflection (parameters, outputs, image constraints, AD boundaries)
- `vernonRuntimeProgramInstanceCreate` / `Destroy` / `BeginInvocation` / `GetTelemetry`
- `vernonRuntimeProgramInvocationBind*` (Program Values/resources, render pass, draw command, dynamic state)
- `vernonRuntimeProgramInvocationForward`
- `vernonRuntimeProgramInvocationRollback` / `Destroy`
- `vernonProgramPullbackApply*` / `vernonProgramPullbackDestroy`

Runtime owns command recording, submission, synchronization, and readback for this API. An external command encoder is
not a Program binding and is not part of the Program Asset lifecycle. Embedding a resolved execution plan in a larger
engine-owned ExecutionGraph is a separate optional facility with its own capability and completion contract; it must
not add a second Program loader, invocation ABI, or forward path.

### Deleted from the runtime surface

| Deleted | Why | Rule |
| --- | --- | --- |
| `vernonRuntimeExecutableBundleInspectKind`, `VernonExecutableBundleKind` | Bundle-kind discrimination. One schema needs no sniffing. | 1 |
| `vernonRuntimeLoadPipelineBundleWithOptions` | Second loader. | 9 |
| `vernonRuntimeResolvePipeline` | Its job becomes variant selection producing an immutable `VernonProgramExecutable`. It currently mutates the loaded object in place. | 8 |
| `vernonRuntimeLoadedPipelineIsManagedProgram` | Runtime kind query for a distinction that no longer exists. | 1 |
| `VernonLoadedPipeline` | Conflates deployment description with resolved executable; signals "is a Program" through an optional `topology` field and "is differentiated" through an optional `differentiated` field. Split into `VernonProgramBundle` and `VernonProgramExecutable`. | 7 |
| `vernonRuntimePipelineSubmit`, `VernonPipelineInvocation` | Legacy cooked submit path. Runtime submission is owned by `VernonProgramInvocation`. | 9 |
| `vernonAdPipelineForward`, `vernonAdPipelineEncodeForward` | Direct cooked autodiff entry points. See §7. | 9 |
| `vernonRuntimeProgramForward` (unbound) | Second forward path beside the instance/invocation model. | 9 |
| Python `CookedVjpPipeline`, `load_cooked_vjp_asset` | Second Python cooked type and loader. | 9 |
| `Runtime.load_cooked_asset` dual dispatch | One loader, no dispatch. | 1 |
| Obsolete pipeline manifest parsers and runtime structures | Dead with the schema. | 11 |

Internal stage pipelines remain implementation details of the Program executor.

### Retained, explicitly not part of this migration

These are separate facilities, not a second cooked-asset loader, and this plan does not delete them:

- `vernonRuntimeLoadArtifact` and `vernonRuntimeLoadCpuEntry` for direct non-asset AOT artifacts.
- `vernonRuntimeRegisterStaticCpuEntry` / `RegisterCpuEntry` / `UnregisterCpuEntry`, and the CPU static registration
  translation unit emitted during cooking, which remain part of CPU-target Program Asset deployment.

Do not remove the existing unified graphics execution architecture while performing this migration. Graphics scope
planning, prepared draw handling, Program graphics execution, invocation context, and target implementation metadata
are current functionality, not migration leftovers.

## 7. One autodiff ABI

Route Kernel VJP through canonical Program autodiff. Structured VJP generation stays an internal implementation
provider behind `CapturedVjpDslProvider` and must not create a second deployment ABI.

Concretely, the bundle-level `autodiff.profiles` table and `vernonAdPipelineForward` form a complete parallel
deployment and execution path for stage-based Kernel VJP. Both are deleted. After the migration:

- VJP structure lives in the Program's own graphs and AD boundaries;
- forward-with-pullback is requested through `vernonRuntimeProgramInvocationForward`;
- pullback is applied through `vernonProgramPullbackApply*`;
- planning policy, tape policy, and residual decisions are compile-time inputs to capture, not manifest tables read
  by the runtime.

Do not reintroduce `CarrierProjection` as a runtime or manifest compatibility mechanism.

## 8. Runtime-value boundary guarantee

The cooked manifest is a compile-time deployment contract, not an invocation snapshot. Information that can only be
known when C++ invokes a Program must never be serialized into the manifest, filled in by Python cooking, or used to
specialize a cooked Program.

The manifest **may** contain only:

- stable Program Value and control-slot identities;
- symbolic references connecting nodes, boundaries, and implementations;
- type, rank, access, layout, and static compatibility constraints;
- workgroup size, and grid axes explicitly declared static by the source Program;
- graphics compatibility constraints: image dimension, format class, aspect, sample-count class;
- captured constants and target implementation metadata that are genuinely invariant across invocations.

The manifest **must not** contain:

- concrete values for dynamic grid axes;
- concrete TensorView shapes, strides, offsets, pointers, or buffer identities;
- concrete extents or identities for borrowed images and attachment views;
- concrete owned dynamic-Storage extents;
- bound render attachments or framebuffer identity;
- invocation draw counts, index ranges, viewport, scissor, blend constants, stencil references, or any other dynamic
  state;
- any Python-side placeholder value substituted for data unresolved until runtime;
- values cached from a previous invocation.

### Where dynamic values do live

| Value | Manifest holds | Runtime resolves from |
| --- | --- | --- |
| Dynamic grid axes | symbolic control slot and dataflow | current invocation bindings |
| TensorView shape/stride/offset | rank, dtype, layout, access constraints | invocation descriptors |
| Borrowed images, attachment views | dimension, format class, aspect, sample-count class | invocation render-pass bindings |
| Owned dynamic Storage extent | symbolic extent expression | invocation binding of its inputs |
| Draw command, dynamic graphics state | boundary declaration only | per-invocation control bindings |

### Dynamic grid axes are Program Value controls

Today the compute launch planner treats a zero `compute_grid` as "please infer", then walks `variant.parameters` and
infers the grid from the **first** tensor argument whose shape happens to be available. That is a sentinel value plus
an order-dependent implicit heuristic: two workarounds standing in for a missing symbolic control.

Replace it with explicit dataflow. A dynamic grid axis is a Program Value control slot. The manifest records the slot
and the expression that derives it. The runtime evaluates that expression against the current invocation bindings.
There is no zero sentinel, no parameter scan, and no dependence on parameter order. An unbound grid control is an
error naming the slot, not a fallback.

The C ABI does not store a separate `compute_grid` in `VernonProgramInvocation`. A convenience wrapper may accept
three integers only by resolving and binding the Program's three declared grid-control Value slots through the normal
binding transaction. Zero is an ordinary invalid dispatch extent, never an absence marker.

Attachment dimensions follow the same rule. The manifest contains format/sample compatibility and symbolic extent
dependencies, not `[0, 0, 1]` or another placeholder. Concrete dimensions come from the currently bound attachment
view and are validated without mutating the executable.

### Publication and externally visible mutation

Publication semantics are declared by ProgramABI and are independent of whether a concrete binding is host memory or
an RHI resource:

- `commit_after_success` writes into invocation-owned staging and publishes only after every planned command,
  validation, tape check, and required synchronization succeeds. Host and device destinations use the same
  transaction semantics.
- `in_place` explicitly permits externally visible mutation during execution and therefore does not promise rollback
  after submission. It is valid only for an ABI boundary documented as an in-place side effect.

Runtime must not silently turn `commit_after_success` into `in_place` because the destination is a borrowed device
resource. Failure before commit leaves every `commit_after_success` destination unchanged. Pullback application uses
the same publication contract.

### Immutability of the deployment description

Loading a Program produces an immutable deployment description. Beginning or executing an invocation creates separate
runtime state; it must not mutate the loaded manifest and must not persist concrete runtime values back into the
executable. The same loaded Program must support different valid shapes, grids, image extents, attachments, and
dynamic states without recooking.

Enforce the boundary at both ends:

- Python cooking fails if unresolved runtime data has been concretized or embedded in emitted JSON.
- C++ manifest parsing and validation rejects concrete invocation snapshots in symbolic control and resource fields.
- C++ invocation validation resolves and validates current values against manifest constraints without changing those
  constraints.
- Tests inspect cooked JSON to prove dynamic values are absent, then invoke the same loaded Program with multiple
  runtime configurations to prove no compile-time specialization occurred.

## 9. No workarounds

A change in this migration is a workaround, and is not acceptable, if it does any of the following.

1. **Discriminates a kind more than once.** Kind is decided at L2 on real Python objects. No downstream `type` sniff,
   `program_kind` string, `is_managed_program` query, or bundle-kind enum.
2. **Selects behavior by duck-typing its input.** No `if all(variant.canonical_program is not None)`. Schema and path
   selection follow from types, not from probing optional fields.
3. **Uses a sentinel value to mean "absent" or "infer".** Zero grid means zero, not "infer". Absence is modeled by
   the type.
4. **Requires a field that must be empty.** No `shape_symbols` that must exist and must be `[]`.
5. **Keeps an always-identity indirection.** No `stage_bindings` mapping every key to itself.
6. **Imports a private symbol from a lower layer, or imports function-locally to dodge a cycle.** A needed cycle-break
   means the code is in the wrong layer; move it.
7. **Signals a variant through a nullable field on a shared struct.** No optional `topology` meaning "this is really a
   Program", no optional `differentiated` meaning "this one has autodiff".
8. **Mutates an immutable-by-contract object.** Variant selection produces a new executable; it does not write into
   the loaded bundle.
9. **Adds a second ABI for an existing capability.** One autodiff ABI, one forward path, one loader.
10. **Leaves a restriction described as "currently".** Either the restriction is part of the contract and is documented
    as such, or it is removed.
11. **Keeps a compatibility alias, typedef, wrapper, or dual schema.** Rename hard; delete the old name in the same
    change.
12. **Re-derives a resolved physical decision during execution.** L7 does not rescan target carriers, Storage aliases,
    or resource kinds to infer residency, transfers, or publication. Those decisions belong to L6.
13. **Attaches Node-specific state to a reusable Stage.** Endpoint projections, controls, and Value bindings are
    indexed by graph and Node; only immutable implementation contracts and backend pipelines may be Stage-shared.

The deletion tables in §5 and §6 cite these rule numbers directly. The remaining named workarounds map as follows:
`program_kind` AST guessing violates rule 1; the `materialize_bundle` schema sniff violates rule 2; the zero
`compute_grid` sentinel and its parameter scan violate rule 3; the `serialize.py` function-local import of
`_canonical_deployment` violates rule 6; and `canonical deployment currently requires exactly one variant` violates
rule 10.

## 10. Public surface rename

| Current | After |
| --- | --- |
| `pipeline_asset` | `program_asset` |
| `PipelineAssetDeclaration` | `ProgramAssetDeclaration` |
| `cook_pipeline_asset` | `cook_program_asset` |
| `load_pipeline` | `load_program` |
| `CookedPipeline` | `CookedProgram` |
| `CookedVjpPipeline`, `load_cooked_vjp_asset` | deleted |
| `PipelineCompileError` | `ProgramCompileError` |
| `python/vernon_dsl/pipeline_assets.py` | `python/vernon_dsl/program_assets.py` |
| `python/vernon_dsl/pipeline_asset_cli.py` | `python/vernon_dsl/program_asset_cli.py` |
| `vernon-cook-pipeline` | `vernon-cook-program` |
| `{dir}.pipeline.json` | `{dir}.program.json` |
| `pipeline_version` (manifest key, plus the five coupled generated names in §5) | `program_version` |
| `VernonPipelineBundle` | `VernonProgramBundle` |
| `VernonPipelineBundleLoadOptions` | `VernonProgramBundleLoadOptions` |
| `VernonLoadedPipeline` | `VernonProgramExecutable` |
| `vernonRuntimeLoadPipelineBundleWithOptions` | `vernonRuntimeLoadProgramBundleWithOptions` |

Rename deployment-layer `_shader_assets` modules whose names no longer describe their responsibility. These are
capture, compile, and deployment modules; none of them is about shaders specifically.

Deliberately **not** renamed: the `python/vernon_dsl/bundle/` package keeps its name, because "bundle" survives as the
C-side deployment noun in `VernonProgramBundle`. What is deleted is the manifest `type: "program_bundle"` value, not
the word.

Delete every old public name. No typedefs, no wrapper aliases, no re-export shims.

## 11. Required verification

Public C++ cook → load → invoke coverage for all five authored forms:

- Kernel;
- graphics `vd.pipeline(...)`;
- Kernel VJP;
- Module, covering both a compute body and a graphics body;
- Module VJP.

Deployment-boundary tests proving one cooked Program is invocable without recooking across:

- multiple Tensor shapes;
- multiple dispatch grids;
- multiple attachment extents;
- multiple dynamic graphics states.

Resolved-plan tests proving:

- one reusable Stage referenced by multiple Nodes receives distinct Node endpoint projections;
- a device result consumed by a later inline/uniform endpoint is represented by an ordered transfer edge and observes
  the producer's result;
- Storage aliases receive one resolve-time residency/transfer plan rather than invocation-time rescanning;
- `commit_after_success` leaves host and device destinations unchanged under failure injection, while explicitly
  declared `in_place` effects document their non-rollback behavior.

Also:

- port numerical, tape, and failure-injection autodiff tests to Program APIs;
- explicit negative tests for graphics Module VJP and graphics Kernel VJP, asserting the single §4 graphics-VJP
  diagnostic. A graphics Module is a *positive* case and needs a cooked-asset test, not a rejection test;
- a multi-variant cooked Program test, covering the deleted one-variant restriction;
- source and schema guards preventing reintroduction of the old names, the deleted `type` values, and the deleted
  manifest fields;
- a layering guard asserting the L0→L7 dependency direction, including that the deployment layer imports nothing from
  the cook orchestrator;
- update every fixture, example, CMake cook rule, installed-wheel check, CLI invocation, public document, and spec;
- delete obsolete code only after all callers use the canonical path.

Tests that encode a removed design must be removed or rewritten, not skipped. `test_python_pipeline_asset_is_parsed_
without_execution` asserted a property §4 retires for cooking; it now covers the lint, which is the part that still
promises not to execute the source.

One vestige is worth recording rather than removing blind: `CompiledStage.module_manifest`, and the descriptor
`canonical_manifest` feeding it, are never serialized and are absent from `CompiledStage.identity`, so both stage
paths carry provenance that nothing reads back. Phase 3 rewrites the deployment layer and should decide their fate
there, where the surrounding code is already being changed.

## 12. Migration discipline

The executable checklist lives in `/Users/yuanxinyu/.cursor/plans/统一_program_asset_bdd7bbe3.plan.md`; this normative
document defines outcomes rather than recording transient attempts.

The migration order is architectural:

1. freeze the corrected manifest, Stage/Node, resolved-plan, invocation, and publication contracts;
2. introduce non-null typed capture, compile, deployment, and resolve products;
3. make all authored forms and variants produce those products;
4. atomically switch to one schema and one loader;
5. move all physical carrier, transfer, residency, hazard, and publication decisions into L6;
6. delete tuple/stage cooking, direct submit/autodiff paths, sentinels, and nullable kind signals;
7. migrate consumers and complete §11 and §13.

Do not repair failures by adding a compatibility branch to the old model. When a root cause exposes a missing
authority, add that authority at its owning layer, migrate its consumers, and delete the displaced derivation in the
same checklist item.

## 13. Completion gate

- Run formatters from `.venv`.
- Use one `debug-build/` tree; do not build different trees in parallel.
- Complete a full build.
- Pass full CTest.
- Pass full Python tests.
- Pass type and lint checks.
- Verify installed-package and example workflows.
- Confirm no symbol, manifest field, or `type` value from the deleted lists in §5, §6, §7, and §10 remains as an
  active contract or implementation anywhere in source, tests, fixtures, examples, or public docs. The deletion
  tables in this migration record are the only identifier-level allowlist.
- Do not bump the compiler or Program contract version until the release step.
