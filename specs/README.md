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
4. [`unified_program_vjp.md`](unified_program_vjp.md) defines the active
   breaking convergence architecture: every standalone compute executable,
   standalone graphics executable, and Module is a Program; standalone
   executables are one-node Programs; and VJP is a forward/backward graph pair
   linked by residual state.
5. [`program_execution_manifest.md`](program_execution_manifest.md) is the
   normative field-level Program deployment specification. It defines every
   Program value origin and lifetime, payload, TensorView descriptor,
   ValueLayout, graph/node field, root-to-stage endpoint/leaf binding,
   derivative signature, resolve algorithm, and fail-closed validation rule.
6. [`autodiff_tutorial.md`](autodiff_tutorial.md) explains the implemented
   Phase 1/2 VJP stack from Tape IR and dynamic control flow through CPU
   multithreading and canonical Program VJP.
   [`module_program_test_gaps.md`](module_program_test_gaps.md) is the active
   Module/Program capability and regression contract for input parity,
   persistent binding, output selection, opaque resources, and acceptance
   gates.
7. [`autodiff_cost_aware_residual_plan.md`](autodiff_cost_aware_residual_plan.md)
   records completed Phase 1/2 memory work and the active production-target
   residual-source, cost-model, and bounded-replay checklist.
8. [`compiler/design.md`](compiler/design.md) defines compiler boundaries,
   lowering invariants, reflection, target routing, and artifact cooking.
   Ordinary device-store injectivity (affine unique-axis, mixed-radix,
   workgroup leader, and dispatch residuals) is
   [`compiler/invocation_index_ownership.md`](compiler/invocation_index_ownership.md).
9. [`runtime/design.md`](runtime/design.md) defines deployment ABI, backend
   ownership, resource behavior, and execution semantics.
10. [`runtime/image_resources.md`](runtime/image_resources.md) defines the
   long-term Image, ImageView, sampled/storage binding, provider, and RHI
   architecture. It is a future contract rather than 0.1.2 behavior.
11. [`roadmap.md`](roadmap.md) summarizes completed milestones and current future
   work across language, compiler, Runtime, backends, and release engineering.
12. [`examples/design.md`](examples/design.md) records non-obvious showcase
   algorithms and third-party design provenance.

Supporting future designs:

- [`language/future_language_roadmap.md`](language/future_language_roadmap.md)
  lists incomplete language-v4 acceptance gates.
- [`host_language.md`](host_language.md) defines the proposed interpreted and
  AOT Host domain, C++ API schema, and desktop/browser acceptance demo.
- [`web_wasm.md`](web_wasm.md) describes the static WebAssembly CPU architecture,
  build, linking, and browser deployment path.
- [`autodiff.md`](autodiff.md) is the current autodiff contract; Program VJP
  follow-up is [`unified_program_vjp.md`](unified_program_vjp.md).
- [`program_vjp_adjoint_ssa.md`](program_vjp_adjoint_ssa.md) is the interior
  adjoint rule for Program VJP: SSA handles, DPS TensorView add, kernel/public
  `read`/`write` ABI, no kernel that returns Storage.

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

`ProgramAsset` accepts a compute Kernel, graphics pipeline, initialized Module
or explicit Program transform. The coordinated breaking target has exactly one
canonical deployment path:

```text
source executable
  -> Program { stages, parameters, storages, values, graphs, abi }
  -> ResolveProgram
  -> ExecuteProgram
```

A standalone compute or graphics executable, including a single shader, is a
one-node Program; a Module differs only in node count. Compute kernel stages
are locked in [`unified_module_pipeline.md`](unified_module_pipeline.md)
section 1.1: lower without tensors, specialize without launch extents, bind
runtime shape in C++. Do not bake `vd.dyn` into the native artifact.

The cooker selects the
target and emits exactly one Program plus content-addressed stage artifacts.
The Program remains platform-neutral: its `stages` are portable contracts,
while each cooked variant maps them to target code through
`stage_bindings`.
Each stage artifact names authenticated target code modules and exact entry
points (`.o`/`.obj`, PTX, SPIR-V, GLSL/GLES, MSL, or DXIL); graphics carries
ordered vertex and fragment modules. Node dependencies are derived from Value
SSA and ResourceAccess hazards rather than serialized as a second edge list.
The breaking loader does not
accept an old artifact through compatibility normalization, legacy profiles,
name binding, copied `parameters`/`internal_parameters`/`uses` tracks, or
direct stage topology. Any single-node fast path begins after `ResolveProgram`
and preserves the same resolved binding plan.

The target Program contract is a static DAG of compute and graphics nodes only.
Transfer and deployment control-flow nodes are not representable.
`StorageDescriptor` is the resource authority; control metadata is entry-only,
and resource results use ordinary public output signature entries.
The private runtime Command DAG owns hazards, scheduling, render scopes, and
submission below `ExecuteProgram`. Compiler-internal Program transforms remain
distinct from that native command scheduler.

The graphics target architecture uses the same Program Value binding path as
compute. Source/IR `RenderTarget`, `Attachment`, and `RenderState` builtins
normalize to graphics-node attachments, state, and draw fields; they are not a
second public resource-aggregate ABI. Shader resources bind independently, and
primitive topology is static artifact specialization. Current version numbers
remain unchanged until the coordinated compiler/pipeline contract release; the
release is intentionally artifact-incompatible rather than a parallel
architecture.

When implementation changes a non-obvious invariant, update the applicable
canonical document in the same change. Temporary investigations belong in
issues and should not become permanent specifications.
