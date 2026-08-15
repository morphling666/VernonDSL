# VernonDSL specifications

`specs/` contains current contracts, architecture decisions, and the active
project roadmap. Git history is the archive for superseded implementation plans
and milestone checklists.

## Reading order

1. [`language/contract.md`](language/contract.md) defines the normative
   language-v4 target and host semantics. Released builds remain frontend
   version 3 until all required v4 gates pass.
2. [`language/tensor_view.md`](language/tensor_view.md) defines the accepted
   Tensor, TensorStorage, TensorView, workgroup, projection, and ABI contract.
3. [`autodiff.md`](autodiff.md) defines the VJP program transform,
   pullback semantics, differentiated asset cooking, and C/C++ deployment API.
4. [`autodiff_tutorial.md`](autodiff_tutorial.md) explains the implemented
   Phase 1/2 VJP stack from Tape IR and dynamic control flow through CPU
   multithreading and ExecutionGraph VJP.
5. [`autodiff_cost_aware_residual_plan.md`](autodiff_cost_aware_residual_plan.md)
   records completed Phase 1/2 memory work and the active production-target
   residual-source, cost-model, and bounded-replay checklist.
6. [`compiler/design.md`](compiler/design.md) defines compiler boundaries,
   lowering invariants, reflection, target routing, and artifact cooking.
7. [`runtime/design.md`](runtime/design.md) defines deployment ABI, backend
   ownership, resource behavior, and execution semantics.
8. [`runtime/image_resources.md`](runtime/image_resources.md) defines the
   long-term Image, ImageView, sampled/storage binding, provider, and RHI
   architecture. It is a future contract rather than 0.1.2 behavior.
9. [`roadmap.md`](roadmap.md) summarizes completed milestones and current future
   work across language, compiler, Runtime, backends, and release engineering.
10. [`examples/design.md`](examples/design.md) records non-obvious showcase
   algorithms and third-party design provenance.

Supporting future designs:

- [`language/future_language_roadmap.md`](language/future_language_roadmap.md)
  lists incomplete language-v4 acceptance gates.
- [`host_language.md`](host_language.md) defines the proposed interpreted and
  AOT Host domain, C++ API schema, and desktop/browser acceptance demo.
- [`platform_execution_plan.md`](platform_execution_plan.md) sequences
  performance measurement, statically linked browser WebAssembly CPU support,
  asynchronous GPU execution, and GPU autodiff.
- [`web_wasm.md`](web_wasm.md) describes the static WebAssembly CPU architecture,
  build, linking, and browser deployment path.

## Version policy

`versions.toml` is the only manually edited version source:

- `RELEASE_VERSION` identifies a package release;
- `COMPILER_CONTRACT_VERSION` versions source semantics and compiler contracts;
- `PIPELINE_VERSION` versions manifests, artifacts, Runtime/provider ABI, and
  RHI compatibility.

There is no independent numeric `FRONTEND_VERSION`. Documents should refer to
the released frontend-v3 position or the language-v4 target, not invent another
version axis. Historical schema numbers must not be used as names for the
current `PIPELINE_VERSION` contract.

## Pipeline boundary

`PipelineAsset` wraps either one compute Kernel or one graphics stage tuple.
The cooker selects the target and emits one current-version manifest plus
content-addressed artifacts. `VernonExecutionGraph` owns host orchestration,
hazards, scheduling, and render scopes. Autodiff may wrap a PipelineAsset
program in a declarative VJP `ProgramExpression`; compiler-internal
`ProgramGraph` remains distinct from the deployment `VernonExecutionGraph`.

When implementation changes a non-obvious invariant, update the applicable
canonical document in the same change. Temporary investigations belong in
issues and should not become permanent specifications.
