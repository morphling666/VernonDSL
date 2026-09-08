# Future designs

Documents in this directory are proposals. They are not current VernonDSL
language, compiler, Program, manifest, or Runtime contracts.

Current normative architecture remains under the corresponding directories in
`specs/`. A future design becomes current only through an explicit versioned
release with implementation and acceptance coverage.

## Distributed compilation

- [`distributed_compiler.md`](distributed_compiler.md): domain-independent
  architecture, boundaries, ordering, backend strategy, and implementation
  phases.
- [`distributed_planner.md`](distributed_planner.md): DeviceMesh, placement,
  user constraints, partition propagation, communication scheduling, and cost
  modeling.
- [`megakernel_tile_ir.md`](megakernel_tile_ir.md): tile tasks, explicit
  orchestration, communication fusion, bounded megakernels, and backend
  lowering.
- [`mixed_precision.md`](mixed_precision.md): quantized types, scaled tensors,
  exact operation capabilities, accumulation, calibration, and portable
  fallbacks.

## Validation workloads

- [`distributed_dl_compiler.md`](distributed_dl_compiler.md): LLM recipes and
  acceptance workloads over the generic distributed compiler. DP, PP, TP, CP,
  and EP are use-case source-level recipe terms, not core IR.

## Language and deployment

- [`host_language.md`](host_language.md): restricted Host language, native
  interop, desktop AOT, and browser WebAssembly design.

## Shared rules

- Future documents must state their contract status explicitly.
- They must not silently redefine current behavior.
- New public fields and serialized metadata require a separately versioned
  contract.
- Implementation phases must include deterministic rejection and acceptance
  criteria.
- Backend-specific fast paths require a semantically valid fallback or an
  explicit unsupported-target diagnostic.
