# VernonDSL specifications

`specs/` contains only current contracts, stable architecture, accepted future
designs, active plans, and the forward roadmap. Git history is the archive for
completed migrations and superseded plans. Measurement output belongs in
`reports/`.

## Version authority

[`versions.toml`](../versions.toml) is the only manually edited version source:

- `RELEASE_VERSION` identifies the package release;
- `COMPILER_CONTRACT_VERSION` versions source semantics, compiler IR
  contracts, reflection, and Stage compilation;
- `PROGRAM_VERSION` versions Program manifests, deployment artifacts, Runtime
  and provider ABI, and RHI compatibility.

Current values are read from `versions.toml` and are frozen for the active
release line. Backend implementation, bug fixes, tests, and completion of
documented behavior do not require a version change.

A future version changes only through an explicit release/contract decision.
If implementation work conflicts with the frozen contract, report that
conflict; do not bump versions or add a compatibility path automatically.

There is no independent numeric frontend or pipeline version. `Pipeline`
denotes the graphics authoring object or a native backend pipeline, not a
deployment contract axis.

## Normative contracts

1. [`language/contract.md`](language/contract.md) defines language semantic
   categories, types, effects, entry interfaces, specialization, and supported
   differentiation boundaries.
2. [`language/tensor_view.md`](language/tensor_view.md) defines Tensor,
   TensorStorage, TensorView, workgroup storage, indexing, and descriptor
   semantics in detail.
3. [`program/execution_manifest.md`](program/execution_manifest.md) is the
   field-level Program JSON, artifact, resolve, ABI, and fail-closed validation
   contract.
4. [`autodiff/contract.md`](autodiff/contract.md) defines compute Program VJP,
   residuals, pullbacks, tangent layout, and deployment behavior.

Normative field definitions must appear in one of these documents only.
Architecture and tutorials link to the contract instead of redefining fields.

## Architecture

1. [`program/architecture.md`](program/architecture.md) defines the one-Program
   model shared by Kernel, graphics pipeline, Module, and VJP.
2. [`program/graphics_execution.md`](program/graphics_execution.md) defines
   graphics normalization, controls, attachments, image versions, and
   render-scope planning.
3. [`compiler/design.md`](compiler/design.md) defines compiler layering,
   lowering, reflection, target routing, and artifact cooking.
4. [`compiler/invocation_index_ownership.md`](compiler/invocation_index_ownership.md)
   defines ordinary device-write injectivity and dispatch residuals.
5. [`runtime/design.md`](runtime/design.md) defines Program execution,
   RuntimeCore/provider/RHI ownership, transfers, publication, scheduling, and
   backend behavior.
6. [`runtime/image_resources.md`](runtime/image_resources.md) defines current
   image owner/view/provider/RHI behavior.
7. [`autodiff/program_adjoint_ssa.md`](autodiff/program_adjoint_ssa.md) defines
   interior Program cotangent SSA and destination-passing accumulation.

## Guides and platform documents

- [`autodiff/tutorial.md`](autodiff/tutorial.md) is the non-normative VJP guide.
- [`platform/wasm.md`](platform/wasm.md) defines current wasm32 CPU deployment.
- [`examples/design.md`](examples/design.md) records non-obvious example
  algorithms and third-party design provenance.

## Active plans and future designs

- [`testing/cross_backend_language_testing_plan.md`](testing/cross_backend_language_testing_plan.md)
  tracks contract-driven, capability-based language test normalization.
- [`testing/program_acceptance.md`](testing/program_acceptance.md) defines
  permanent Kernel/Module/Program/VJP regression gates.
- [`language/future_language_roadmap.md`](language/future_language_roadmap.md)
  lists remaining language-v4 gates.
- [`future/host_language.md`](future/host_language.md) is a non-normative future
  Host-language and native-interop design.
- [`roadmap.md`](roadmap.md) lists project-wide unfinished work.

## Program boundary

Every public executable is a Program:

```text
Kernel / vd.pipeline / initialized Module / Program transform
  -> capture and MLIR Program IR
  -> Program + target ArtifactSystem
  -> load Program bundle
  -> resolve immutable execution plan
  -> create instance
  -> begin invocation
  -> bind
  -> invoke
```

Standalone compute and graphics executables are one-node Programs. A Module
differs only in node count. Runtime owns command recording and submission
internally.

## Documentation rules

- Update the authoritative contract in the same change as a contract change.
- Describe current behavior in present tense; migration history belongs in Git.
- Mark every non-normative document as architecture, tutorial, active plan, or
  future design.
- Do not duplicate manifest fields, API contracts, or acceptance gates.
- Do not use removed API names as current architecture.
- Do not add benchmark output, session handoffs, or temporary investigations
  to `specs/`.
- Every Markdown file under `specs/` must be linked from this index directly or
  through its owning section.
