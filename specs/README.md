# VernonDSL specifications

`specs/` contains current contracts, architecture decisions, and active future
work. Completed plans and superseded proposals are deleted; Git history is the
archive.

## Reading order

1. [`language/contract.md`](language/contract.md) defines the normative
   language-v4 target and host semantics. Released builds remain frontend
   version 3 until all required v4 gates pass.
2. [`language/tensor_view.md`](language/tensor_view.md) defines the accepted
   Tensor, TensorStorage, TensorView, workgroup, projection, and ABI contract.
3. [`compiler/design.md`](compiler/design.md) defines compiler boundaries,
   lowering invariants, reflection, target routing, and artifact cooking.
4. [`runtime/design.md`](runtime/design.md) defines deployment ABI, backend
   ownership, resource behavior, and execution semantics.
5. [`language/future_language_roadmap.md`](language/future_language_roadmap.md)
   lists incomplete language-v4 acceptance gates.
6. [`stable_release_plan.md`](stable_release_plan.md) defines the production
   contracts, TensorView ABI, release engineering, and promotion gates for the
   first stable release.
7. [`completion_roadmap.md`](completion_roadmap.md) lists active work required
   for beta and stable releases.

Supporting active documents:

- [`deferred_gpu_resource_lifetime_plan.md`](deferred_gpu_resource_lifetime_plan.md)
  defines deferred in-flight resource reclamation and multi-frame execution.
- [`examples/design.md`](examples/design.md) records non-obvious showcase
  algorithms and third-party design provenance.

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
hazards, scheduling, and render scopes. The low-priority `ProgramGraph` roadmap
item is private compiler IR for autodiff and is not a deployment graph.

When implementation changes a non-obvious invariant, update the applicable
canonical document in the same change. Temporary investigations belong in
issues and should not become permanent specifications.
