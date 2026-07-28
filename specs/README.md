# VernonDSL specifications

`specs` contains only current contracts, implementation direction, and active
deferred work. Completed and superseded plans are removed; Git history is the
archive.

## Canonical documents

- [`language/contract.md`](language/contract.md) is the normative language and
  host-semantic contract.
- [`language/future_language_roadmap.md`](language/future_language_roadmap.md)
  records incomplete language/runtime phases and acceptance gates.
- [`compiler/design.md`](compiler/design.md) records compiler boundaries,
  lowering invariants, reflection, and asset cooking.
- [`runtime/design.md`](runtime/design.md) records deployment ABI, backend
  ownership, resource behavior, and execution semantics.
- [`examples/design.md`](examples/design.md) records non-obvious algorithm and
  synchronization choices used by the GPU showcase programs.
- [`completion_roadmap.md`](completion_roadmap.md) prioritizes completion,
  refactoring, performance, backend consistency, and production engineering.

An implementation must update the applicable canonical document when it adds a
non-obvious invariant or changes an accepted contract. Temporary investigation
notes should be tracked as issues or plans outside `specs` and removed after
their conclusions are merged here.

## Pipeline asset reading order

The active persistent-pipeline contract is specified in:

1. [`language/contract.md` — functions, interfaces, and specialization](language/contract.md#6-functions-interfaces-and-specialization)
2. [`compiler/design.md` — Pipeline asset declarations](compiler/design.md#pipeline-asset-declarations)
3. [`runtime/design.md` — Pipeline runtime boundary](runtime/design.md#pipeline-runtime-boundary)

The shared contract is:

- `PipelineAsset` wraps either one compute Kernel or one graphics stage tuple;
- Kernel is compute-only and Pipeline is graphics-only;
- graphics stage topology is extensible independently from backend support;
- `variants=` enumerates accepted feature keys;
- target architecture and options are cooker inputs, not source fields.

`VernonExecutionGraph` is the active host-orchestration API. Its ownership,
hazard, scheduling, render-scope, and command-recording contract is maintained
in [`runtime/design.md`](runtime/design.md#pipeline-runtime-boundary), not in a
separate completion plan.
The separate low-priority `ProgramGraph` roadmap item is compiler IR inside one
program for autodiff; it is not a deployment graph.

Stable logical resource records are active phase-1 work. Multi-frame resource
ownership and reclamation remain deferred in
[`deferred_gpu_resource_lifetime_plan.md`](deferred_gpu_resource_lifetime_plan.md).
