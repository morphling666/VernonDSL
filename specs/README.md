# VernonDSL specifications

`specs` contains current contracts and implementation direction. Deferred
non-normative proposals that remain useful for future redesign live under
`backup/`; other historical plans are available from Git history.

## Canonical documents

- [`language/contract.md`](language/contract.md) is the normative language and
  host-semantic contract.
- [`language/future_language_roadmap.md`](language/future_language_roadmap.md)
  records incomplete language/runtime phases and acceptance gates.
- [`compiler/design.md`](compiler/design.md) records compiler boundaries,
  lowering invariants, reflection, and asset cooking.
- [`runtime/design.md`](runtime/design.md) records deployment ABI, backend
  ownership, resource behavior, and execution semantics.

An implementation must update the applicable canonical document when it adds a
non-obvious invariant or changes an accepted contract. Temporary investigation
notes should be tracked as issues or plans outside `specs` and removed after
their conclusions are merged here.

## Program asset reading order

The active persistent-program contract is specified in:

1. [`language/contract.md` — functions, interfaces, and specialization](language/contract.md#6-functions-interfaces-and-specialization)
2. [`compiler/design.md` — Program asset declarations](compiler/design.md#program-asset-declarations)
3. [`runtime/design.md` — Program runtime boundary](runtime/design.md#program-runtime-boundary)

The shared contract is:

- `ProgramAsset` wraps either one compute Kernel or one graphics stage tuple;
- Kernel is compute-only and Pipeline is graphics-only;
- graphics stage topology is extensible independently from backend support;
- `variants=` enumerates accepted feature keys;
- target architecture and options are cooker inputs, not source fields.

Execution-graph and Pass proposals are deferred. Their previous non-normative
design is preserved in
[`backup/execution_graph_design.md`](backup/execution_graph_design.md).
