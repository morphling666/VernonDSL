# Future designs

Documents in this directory are proposals. They are not current VernonDSL
language, compiler, Program, manifest, Runtime, RHI, or ABI contracts.

Current normative architecture remains under the corresponding directories in
`specs/`. A future design becomes current only through an explicit versioned
release with implementation, compatibility rules, and acceptance coverage.

## Compiler reading order

1. [`architecture.md`](architecture.md) — goals, layers, authorities,
   invariants, and the complete joint optimization boundary.
2. [`optimization_ir.md`](optimization_ir.md) — analyzable operation
   interfaces, partition/placement IR, distributed primitives, fusion, physical
   tasks, schedules, and verification.
3. [`joint_planner.md`](joint_planner.md) — topology, recipe-free search, cost
   model, profiles, measurement, Pareto selection, and Agent proposals.
4. [`numerical_representation.md`](numerical_representation.md) — mixed
   precision, quantization, accumulation, storage/transfer representation, and
   numerical quality.
5. [`backend_lowering.md`](backend_lowering.md) — CUDA Tile IR, portable target
   paths, artifacts, capability qualification, and fallback.
6. [`implementation_roadmap.md`](implementation_roadmap.md) — current code
   entry points, PR sequence, validation workloads, metrics, and release gates.

Each compiler fact has one owner:

| Question | Owner |
| --- | --- |
| Why, what, and which layer owns a fact? | `architecture.md` |
| What IR is represented and what is legal? | `optimization_ir.md` |
| How are candidates searched, costed, and learned from? | `joint_planner.md` |
| What are the numerical and representation semantics? | `numerical_representation.md` |
| How does a verified leaf become a target artifact? | `backend_lowering.md` |
| In what order is the system delivered and accepted? | `implementation_roadmap.md` |

## Orthogonal language track

- [`host_language.md`](host_language.md) — restricted Host language, native
  interop, desktop AOT, and browser WebAssembly. This is a separate track from
  the distributed device optimizer.

## Core vocabulary

- `Program`: sole semantic authority.
- `Partition`: logical pieces independent of resources.
- `Placement`: mapping pieces and replicas to topology resources.
- `PartialValue`: contribution requiring a declared combine.
- `Redistribution`: required partition, placement, replication, or
  representation change.
- `AsyncTask`: explicit physical work, dependency, and completion.
- `FusionRegion`: semantic group eligible for one implementation.
- `TileTask` and other typed physical tasks: target-oriented work.
- `PlanCandidate`: one complete numerical, distributed, fusion, and target
  alternative.
- `PhysicalPlanVariant`: future immutable resolved-plan alternative selectable
  by Runtime. It is distinct from the current compile-time Program typed
  specialization variant and requires a versioned installation/selection
  contract.

Human strategy names such as DP, TP, PP, CP, and EP are optional validation
recipes. They are not core IR, automatic-planner actions, or cost features.

## Shared rules

- Future documents state their non-contract status.
- They do not silently redefine current behavior.
- New public fields and serialized metadata require a separately versioned
  contract.
- Program semantics remain separate from physical representation.
- Compiler-induced communication is explicit before communication fusion.
- Cost models rank legal plans; they do not own correctness.
- Agent proposals pass the same verifiers, references, and measurements.
- A future versioned Runtime policy may select only installed validated
  `PhysicalPlanVariant` records.
- Backend fast paths have a valid fallback or explicit unsupported diagnostic.
