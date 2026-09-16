# Optimization IR and legality

Status: future design, not a current VernonDSL contract.

This document defines the compiler representations between semantic Program IR
and target leaf lowering. The architecture and layer authorities are defined
in [`architecture.md`](architecture.md). Search and profitability belong to
[`joint_planner.md`](joint_planner.md); numerical representation belongs to
[`numerical_representation.md`](numerical_representation.md).

## 1. Design principles

- Program IR is the only semantic authority.
- Analysis is exposed through interfaces, not a universal serialized op.
- Global partition is separate from local tile layout.
- Placement mismatch becomes explicit redistribution.
- Communication is explicit before overlap or communication fusion.
- Physical tasks remain typed by domain.
- Tokens represent region availability and completion.
- Verification failure selects another schedule or an ordinary fallback.

## 2. Analyzable operation protocol

An operation exposes the meaningful subset of:

| Interface | Contract |
| --- | --- |
| `IterationDomain` | Logical points, bounds, symbols, and dynamic constraints |
| `AccessRelation` | Relation from logical work to read or written Value/Storage regions |
| `IteratorKinds` | Parallel, reduction, ordered, scan, or data-dependent dimensions |
| `ScalarSemantics` | Scalar body, contribution, combiner, identity, algebraic properties |
| `EffectsAndAliases` | Reads, writes, atomics, allocation, synchronization, aliases, publication |
| `Dependencies` | Semantic ordering not already implied by SSA, Storage versions, and effects |
| `PartitionRules` | Factor correspondence, legal splits, replication, and partial results |
| `FusionRules` | Composition conditions and required completion points |
| `ReferenceSemantics` | Trustworthy conservative implementation |
| `CostFeatures` | Typed features not derivable from generic relations |

An operation need not implement every interface. Unknown information causes
conservative barriers or candidate rejection, not invented semantics.

These interfaces are not specific to deterministic fusion passes. They are the
shared, typed optimization API for handwritten rules, solvers, agents, and
external synthesizers. A candidate generator may derive a region using fixed
rules or propose a new algorithmic replacement using learned reasoning. In
both cases the source operations retain semantic authority and the candidate
must preserve or explicitly refine their contracts.

`FusionRules` does not enumerate profitable neighboring operation names. It
states exceptional composition conditions, required completion or
materialization points, and legal refinement preconditions that cannot be
derived from iteration, access, scalar, effect, alias, dependency, and
numerical semantics.

### 2.1 Structured computation

For structured tensor computation, the complete contract is:

```text
iteration domain
+ operand indexing maps
+ iterator kinds
+ scalar body and reduction combiner
+ effects and aliases
+ numerical policy
```

Indexing maps answer which operand coordinates are accessed at an iteration
point. They do not identify a sum, maximum, ordered update, race, or legal
reassociation by themselves.

For matrix multiplication:

```text
domain: (m, n, k)
A:      (m, n, k) -> (m, k)
B:      (m, n, k) -> (k, n)
C:      (m, n, k) -> (m, n)
iterator kinds: parallel(m), parallel(n), reduction(k)
scalar body: next = add(current, mul(a, b))
```

The reduction contract includes identity, associativity, commutativity, order,
overflow, and floating-point reassociation policy.

### 2.2 General access relations

`AccessRelation` also represents:

- stencil neighborhoods;
- indirect table, graph, or particle access;
- image subresources and attachments;
- symbolic containment and bounds;
- uniqueness or possible-conflict information;
- an explicit conservative unknown relation.

The compiler must not pretend that data-dependent access is affine.

### 2.3 Opaque stages

An opaque Stage remains legal with typed inputs/outputs, effects, aliases,
capabilities, optional partition/fusion constraints, conservative cost
metadata, alternatives, and a fallback. Backend code is never inspected to
reconstruct missing semantics.

### 2.4 Semantic regions and progressive analyzability

Candidate generators consume a `SemanticRegion`, which references one or more
canonical Program operations and their observable boundary:

```text
SemanticRegion {
  Program operations and provenance
  external Values, Storages, and regions
  observable effects and dependencies
  implemented analysis interfaces
  algebraic and numerical laws
  conservative reference semantics
}
```

The view may expose progressively more structure:

```text
typed opaque reference
  -> effects and conservative access
  -> iteration, scalar, and reduction semantics
  -> partition and composition laws
  -> verified algorithmic refinements
```

A missing advanced interface limits transformations but does not require
replacing the operation or hard-coding its domain name into the planner.

### 2.5 Optimization decision graph

Planning decisions are recorded as a persistent dependency graph rather than
only mutating one monolithic plan:

```text
DecisionNode {
  identity_and_kind
  generator_and_parent_decisions
  semantic_objects_and_regions
  assumptions_and_constraints
  selected_action_or_refinement
  exact_derived_consequences
  alternatives_and_rejection_reasons
}
```

For example, a factor split may imply a `PartialValue`; its consumer placement
may imply a `Redistribution`; the selected transfer representation may imply a
conversion; and chunk availability may enable a fusion candidate. These edges
explain why each physical task exists and allow measured local failures to
replace the affected decisions without discarding unrelated choices.

A `PlanCandidate` is one complete, consistent, evaluable projection of an
`OptimizationDecisionGraph`. The graph is compiler search and provenance
state; it is not serialized as mutable Runtime policy.

## 3. Logical distribution IR

### 3.1 Core objects

```text
Partition {
  semantic_object
  logical_factors_or_regions
  pieces
  padding_or_ragged_rules
}

Placement {
  pieces
  compute_and_memory_resources
  owner_and_lifetime
  persistent_or_temporary
}

PartialValue {
  semantic_value
  contributions
  combine_semantics
  numerical_policy
}

Replication {
  semantic_value_or_storage
  replicas
  owners
  consistency_and_update_rule
}
```

Partition applies to Values, Storages, iteration domains, index sets,
operation sets, and Program regions. Tensor sharding is one specialization.

Persistent placement describes retained state such as parameters, caches, and
simulation state. Temporary placement describes operation-local execution.
They may differ.

### 3.2 Partition propagation

Operator partition rules declare:

- correspondence between operand and result factors;
- independently splittable factors;
- factors producing partial Values;
- required replication;
- divisibility, padding, and ragged constraints;
- forward and derivative relationships;
- semantic redistribution alternatives.

Propagation:

1. projects user constraints to logical factors;
2. propagates forward and backward to a fixed point;
3. creates partial Values when reduction factors split;
4. records unresolved legal choices;
5. diagnoses conflicting hard constraints with provenance;
6. projects factors back to Value and Storage pieces.

It proves consistency. It does not select routes, collective algorithms, or
tile schedules.

## 4. Redistribution

A producer placement or representation that does not satisfy a consumer
creates:

```text
Redistribution {
  semantic_value
  source_partition_and_placement
  destination_partition_and_placement
  logical_region
  source_and_destination_representation
  replication_or_partial_state
  combine_semantics
  numerical_policy
  legal_routes_and_fallbacks
}
```

Logical redistribution may become a local view, copy, conversion, point-to-
point transfer, replication, reduction, collective, remote operation, or
hierarchical sequence. It does not contain a library name.

## 5. Asynchronous task IR

The task graph is the strongest common physical layer:

```text
AsyncTask {
  semantic_and_logical_provenance
  typed_kind
  input_and_output_regions
  affected_resource_versions
  predecessors_and_completion
  placement_and_eligible_engines
  lifetime_and_residency
  capabilities
  fallback_boundary
}
```

Typed kinds include compute, communication, copy, conversion,
synchronization, graphics, allocation, prefetch, eviction, publication, and
I/O where applicable.

The graph owns physical dependencies. It references canonical Program Values,
Storages, effects, and versions rather than reconstructing them.

## 6. Distributed primitives

### 6.1 Physical tasks

| Task | Semantic result |
| --- | --- |
| Local view or conversion | Value and optional completion |
| Copy, send, receive | Region availability and completion |
| Remote put or get | Remote visibility and completion |
| Replicate or multicast chunk | Per-destination region availability |
| Reduce-to-owner chunk | Partial consumption and owner completion |
| Collective chunk | Participant epoch and region availability |
| Signal or wait | Scoped ordering and visibility |

Every communication task declares:

- source/destination placement and memory space;
- logical Value region and physical byte range;
- participants, team, route, and failure domain;
- addressability and staging;
- ordering, visibility, completion, and epoch;
- engine, stream, or worker eligibility;
- progress and concurrent residency;
- buffer lifetime and ownership;
- combine and numerical semantics;
- Runtime or multi-task fallback.

Bulk Runtime communication is the reference implementation. Device-side
communication requires proof of capability, topology, addressability,
ordering, visibility, progress, and concurrent residency.

### 6.2 Availability tokens

Communication and compute expose:

```text
AvailableRegion {
  value
  logical_region
  placement
  representation
  completion_scope
  producer_token
}
```

A consumer tile may start when all regions required by its access relation are
available. It need not wait for an entire logical communication operation.
Tokens are compiler ordering values; they do not create new Program semantics.

## 7. Fusion regions

A `FusionRegion` records semantic eligibility:

- input/output Values and regions;
- operations and communication provenance;
- effects and aliases;
- local shapes and placements;
- numerical requirements;
- allowed split points;
- candidate implementation families.

It is not yet a kernel schedule.

Fusion-region proposers include deterministic producer-consumer growth,
bounded graph search, user constraints, reusable algorithmic rewrites, and
agents. An agent may choose a boundary or replace the region with a synthesized
algorithm not present in the deterministic rule set. Such a replacement
records source-to-replacement Value and region mappings, preconditions,
numerical refinement, effects, completion, and a reference or split fallback.
Agent selection does not itself establish eligibility.

### 7.1 Region growth

Deterministic producer-consumer fusion:

1. Select a consumer output region.
2. Use consumer access relations to derive required inputs.
3. Compute producer preimages to derive required producer iterations and
   operand regions.
4. Check reduction completion and scalar algebra.
5. Check materialization requirements and other consumers.
6. Check effects, ownership, aliases, and numerical policy.
7. Estimate target-independent resource lower bounds.
8. Grow until a legality boundary, configured bound, or dominated estimate.

For `relu(matmul(a, b))`, ReLU may consume a completed output tile without
materializing the full intermediate. It cannot move inside the reduction loop
because `relu(sum(x))` is not generally `sum(relu(x))`.

### 7.2 Communication-aware fusion

Compiler-induced communication is materialized first:

```text
partition and placement
  -> redistribution
  -> communication tasks
  -> chunk availability
  -> communication-compute fusion candidates
```

Generic patterns include:

- receive chunk to consumer tile;
- producer tile to send/reduce chunk;
- completed partial combine to nonlinear epilogue;
- conversion chunk to native compute tile.

Patterns match regions, dependencies, and combine semantics. They do not match
TP, all-gather-plus-GEMM, halo, or another workload strategy name.

For each communication edge the compiler:

1. derives producer and consumer regions;
2. enumerates legal chunking from access, alignment, route, and participant
   constraints;
3. builds ordinary, overlapped, and fused candidates;
4. propagates availability tokens;
5. generates compatible local physical schedules;
6. verifies communication and persistent obligations;
7. retains an ordinary split fallback;
8. sends complete candidates to the joint planner.

## 8. Typed physical IR

### 8.1 Shared contract

Every physical task declares:

- typed inputs and outputs;
- logical region and physical ownership;
- layout and memory requirements;
- effects and atomics;
- completion and synchronization;
- resource estimate or exact requirement;
- target capabilities;
- split point and fallback.

### 8.2 Tile tasks

`TileTask` covers structured/indexable work:

- load, store, copy, and prefetch;
- elementwise map and conversion;
- contraction and convolution;
- reduction and optional scan;
- stencil and neighborhood;
- quantize, dequantize, pack, and unpack;
- bounded communication chunks participating in a tiled schedule.

Tasks are compiler IR, not opaque separately compiled kernels.

### 8.3 Other physical tasks

- `SparseTask`: indirect access, preprocessing, conflict policy, queues,
  atomics, and load balancing.
- `RenderScopeTask`: attachments, load/store/clear/resolve, draws, image
  transitions, and native render-scope composition.
- `CommunicationTask`: point-to-point, remote, collective, route, participant,
  visibility, and progress.

### 8.4 Layout

An immutable local layout records logical shape, memory scope, mapping over
named hardware axes, replication, offset, packing, alignment, swizzle,
low-precision metadata, and producer-consumer compatibility.

A global `Shard(dim)` does not determine register, subgroup, shared-memory, or
matrix-fragment layout. Backends may not infer missing correctness facts from
pointer arithmetic.

### 8.5 Execution scope

Portable scopes are Lane, Subgroup, Workgroup, WorkgroupCluster, Device, and
DistributedTeam. Target terms such as warp, CTA, TMA, or wave appear only
after target selection.

### 8.6 Persistent schedules

`PersistentSchedule` composes compatible tasks into a bounded resident
structure and records:

- roles and ownership;
- ordering tokens, barriers, channels, and epochs;
- pipeline stages and queues;
- memory offsets and lifetimes;
- communication routes;
- progress, termination, and failure behavior;
- split fallback.

It is an optional optimization form, not the required result for every
Program.

### 8.7 Implementation candidates

A verified physical region may be implemented by an ordinary lowering pass,
schedule enumeration, an agent acting as a lowering pass, or another
registered synthesizer:

```text
ImplementationCandidate {
  source_region_and_semantic_provenance
  selected_optimization_decisions
  generator_identity_and_inputs
  target_and_capability_preconditions
  shape_layout_and_representation_assumptions
  input_output_region_mapping
  numerical_refinement
  effects_memory_and_synchronization_contract
  source_physical_or_target_IR
  resource_requirements_or_analysis
  validation_evidence
  fallback
}
```

The generator may emit typed Vernon physical IR, a registered external DSL,
target IR, or lower target representation. Intermediate forms are aids to
generation and verification, not mandatory semantic authorities. A lower or
less inspectable output requires stronger target parsing, translation
validation, reference testing, capability restrictions, and fallback.

Cross-platform semantics means that the same verified source region may be
regenerated into different target implementations. It does not require one
low-level implementation to execute unchanged on every backend.

## 9. Composable verification

Every candidate passes shared checks:

1. semantic provenance and reference agreement;
2. Value, Storage-version, and region dependency preservation;
3. ownership and completion;
4. memory bounds and lifetime;
5. effects, hazards, aliases, and visibility;
6. numerical and representation policy;
7. resource capacity and concurrent residency;
8. capability satisfaction;
9. fallback validity.

For a synthesized region or direct target implementation, these checks are
translation validation of the concrete result rather than trust in the
generator that produced it. Validation proves what is statically tractable and
records numerical or semantic obligations discharged by differential testing.
Incomplete evidence narrows the accepted shape and capability bucket or
rejects the candidate; it never silently upgrades a candidate to a general
rule.

The acceptance result is an `EvidenceBundle` associated with, but not authored
by, the candidate:

```text
EvidenceBundle {
  verifier_and_translation_validation_results
  numerical_and_differential_results
  compilation_and_resource_results
  analytical_estimates_and_uncertainty
  measured_profiles
  counterexamples_and_qualification_limits
}
```

Keeping evidence separate prevents a generator from granting authority to its
own correctness or performance claims.

Typed checks add:

- structured compute: coverage, bounds, reduction, layout;
- sparse: index bounds, conflicts, queue termination;
- graphics: attachment compatibility, version continuity, transitions;
- communication: participants, epochs, route, remote visibility, progress;
- persistent schedules: cycles, barrier phases, residency, termination.

Communication-compute fusion is rejected if a nonlinear consumer observes an
incomplete partial Value, participant order changes, communication cannot make
progress, fused compute consumes required engines or residency, representation
changes semantics, a cycle is possible, resources exceed limits, or no valid
split fallback exists.

Verification failure never weakens synchronization. It selects another
schedule or the ordinary fallback.

## 10. Resource requirements

Physical IR tracks:

- compute-unit capacity and occupancy;
- registers and local memories;
- barriers, channels, signals, and phase slots;
- persistent workers and concurrent residency;
- code size and instruction-cache pressure;
- compute sharing between compute and communication;
- copy and communication engines;
- staging buffers and lifetime.

The resource representation is extensible and typed. One
`shared_memory_bytes` field cannot prove progress, render compatibility,
communication overlap, or instruction-cache behavior.

## 11. Verification requirements

IR verifiers require inspectable dependencies, regions, layouts, scopes,
effects, completion, and split points. Alias, reduction, barrier, participant,
resource, residency, progress, ownership, and cycle failures reject
deterministically. Delivery and acceptance gates are owned solely by
[`implementation_roadmap.md`](implementation_roadmap.md).
