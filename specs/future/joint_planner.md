# Joint planner, candidate generation, and evidence

Status: future design, not a current VernonDSL contract.

This document defines topology, recipe-free joint search, the common candidate
generator protocol, cost modeling, profiles, optimization memory, measurement,
and Pareto selection. IR and legality belong to
[`optimization_ir.md`](optimization_ir.md). Numerical quality and representation belong to
[`numerical_representation.md`](numerical_representation.md).

## 1. Planner responsibility

The planner jointly evaluates:

- numerical and storage representation;
- partition, placement, and replication;
- partial Values and redistribution;
- communication route, algorithm, and chunking;
- asynchronous dependencies and overlap;
- fusion and split points;
- tile, sparse, graphics, communication, and persistent schedules;
- memory lifetime and residency;
- target lowering and future `PhysicalPlanVariant` records.

It searches legal, memory-feasible plans. It does not prove a global optimum
or change Program semantics.

## 2. Inputs

```text
PlannerInput {
  semantic_program_and_analysis
  user_constraints
  resource_topology
  numerical_and_quality_policy
  objectives_and_budgets
  versioned_profiles
}
```

Named machine-learning strategies are not required inputs. Recipes may lower
to ordinary constraints and provide optional seeds, but recipe-free and
recipe-seeded runs are measured separately.

## 3. Resource topology

A logical `DeviceMesh` is an optional named view over a physical topology; it
is not the topology itself.

```text
HardwareTopologyProfile {
  identity_and_versions
  compute_nodes
  memory_nodes
  engines
  directed_links
  routes
  failure_domains
}
```

| Element | Required facts |
| --- | --- |
| Compute node | Backend, exact operation capabilities, concurrency, measured throughput |
| Memory node | Capacity, address space, ownership, persistence, measured bandwidth/latency |
| Engine | Copy/communication operations, concurrency, progress behavior |
| Link | Transport, direction, addressability, startup, bandwidth distribution, contention |
| Route | Direct/staged path, collective or remote support, hierarchy, failure domain |

Topology facts are versioned and measured. Peak marketing bandwidth is a
fallback prior only when measurements do not exist.

Heterogeneous devices are not interchangeable ranks. Placement accounts for
backend, capabilities, capacity, transfer format, network cost, compilation
availability, and different preferred local shapes.

## 4. Generic action space

Automatic search uses only generic transformations:

- split or merge a logical factor, index set, or Program region;
- replicate, move, pin, prefetch, or evict a piece;
- create partial contributions with a declared combine;
- choose persistent versus temporary placement;
- select a storage or transfer representation;
- select redistribution route, algorithm, and chunk;
- assign streams, queues, engines, and buffers;
- fuse or split a physical region;
- change tile, local layout, scope, role, pipeline, or resources.

TP, EP, DP, CP, PP, attention, MoE, halo, and similar domain names are not
actions or hidden cost features. A generic search may rediscover the same
physical plan when it is profitable.

The action space is the initial vocabulary, not a privileged deterministic
path or the maximum future intelligence of the compiler. A generator may
synthesize a new algorithmic rewrite, fusion region, physical implementation,
or target lowering outside the current rule closure. The concrete result still
enters the same typed plan, translation-validation, evaluation, and fallback
protocols.

### 4.1 Candidate generator protocol

All proposal mechanisms implement:

```text
CandidateGenerator {
  accepted_problem_kinds
  required_semantic_and_analysis_interfaces
  accepted_constraints_profiles_and_prior_evidence
  generated_decisions_regions_or_implementations
  declared_assumptions_and_expected_evidence
  identity_version_and_replay_information
}
```

Initial implementations include:

- handwritten rewrites and region-growth passes;
- constraint, graph, ILP, and equality-saturation solvers;
- schedule enumerators and autotuners;
- retrieval and adaptation from optimization memory;
- learned proposers and agents;
- external synthesis services.

A generator may propose one `DecisionNode`, a consistent decision subgraph, a
complete `PlanCandidate`, or an `ImplementationCandidate`. Generator identity
changes provenance and reproducibility requirements, not semantic authority or
acceptance criteria.

## 5. Bounded joint search

Search remains hierarchical but revisable:

```mermaid
flowchart TB
    Hard["Apply hard constraints"]
    Analyze["Propagate semantic relations"]
    Generate["CandidateGenerator set"]
    Outer["Partition, representation, placement beam"]
    Redist["Materialize redistribution"]
    Async["Generate communication and async schedules"]
    Inner["Chunk, fusion, layout, tile beam"]
    Verify["Verify complete candidates"]
    Evaluate["Model, compile, reference, measure"]
    Pareto["Retain explainable Pareto variants"]

    Hard --> Analyze --> Generate
    Generate --> Outer --> Redist --> Async --> Inner
    Generate --> Inner
    Inner --> Verify --> Evaluate --> Pareto
    Evaluate -. bounded feedback .-> Outer
    Evaluate -. evidence and failures .-> Generate
```

Initial algorithms:

| Decision | Initial method | Hard pruning |
| --- | --- | --- |
| Partition seed | Constraint propagation and bounded factor enumeration | Semantics, resource count, memory lower bound |
| Placement | Beam search with optional min-cut or small ILP | Capacity, capability, forbidden resources |
| Redistribution | Route/algorithm/chunk enumeration | Addressability, participants, progress |
| Fusion | Pattern DAG and bounded region growth | Effects, liveness, numerics, synchronization, resources |
| Physical schedule | Template enumeration and autotuning | Shape, layout, target capability |

Every outer candidate materializes communication and receives an inner
implementation estimate. Material measured deviations may trigger a bounded,
cached outer reconsideration.

The outer/global loop owns representation, partition, placement, replication,
redistribution, checkpointing, fusion boundaries, and asynchronous
composition. The inner/local loop owns algorithmic replacement, tiles,
layouts, pipelines, memory hierarchy, target operations, and leaf lowering.
They exchange explicit decision and evidence edges rather than running once in
a fixed order.

## 6. Cost model role

The optimization system separates authority:

| Component | Authority |
| --- | --- |
| Semantic rules and verifiers | Define the feasible set |
| Generic transformations | Define the reusable baseline action vocabulary |
| Agent or learned proposer | Supply an implicit prior over useful unexplored candidates |
| Explicit accounting and cost model | Approximate multi-objective value with uncertainty |
| Hardware measurement | Supply ground-truth observations |
| Planner, agent, or external synthesizer | Propose candidates and implementations |

A favorable model score cannot legalize an invalid candidate.

The model evaluates a complete `PlanCandidate`, not a kernel after distribution
has already been fixed.

Agent intuition does not replace the explicit model. Exact accounting,
calibrated topology facts, event simulation, versioned uncertainty, and
explanations provide a shared evaluator across generators. Agent intuition may
challenge the ranking and request exploration of an uncertain or
out-of-distribution candidate; it may not declare its own candidate faster.
Measurement remains performance evidence.

## 7. Candidate features

### 7.1 Distribution and representation

- partition geometry, divisibility, padding, raggedness, imbalance;
- persistent and temporary placement;
- replicated and partial Value sizes;
- storage/compute/transfer encodings and metadata;
- conversion, scale synchronization, and numerical quality.

### 7.2 Communication and schedule

- bytes by source, destination, route, representation, and chunk;
- startup, hierarchy, participants, algorithm, staging, and contention;
- events, barriers, channels, epochs, streams, and engines;
- legal overlap windows and event-graph critical path;
- buffer lifetime and peak memory.

### 7.3 Local implementation

- typed operation counts and operation-class FLOPs;
- bytes and transactions by memory level;
- reuse and materialization avoided;
- layout conversions;
- registers, local memory, workspace, occupancy, residency;
- launch count, code size, pipeline fill/drain;
- compilation and tuning complexity.

Features use typed domain payloads. GEMM uses M/N/K and layouts; stencil uses
radius and local geometry; sparse work uses occupancy and neighbor
distributions; graphics uses attachments and resolution. Tensor shape is not a
universal workload identity.

## 8. Multi-fidelity evaluation

```text
legality and exact capabilities
-> memory, resource, and progress lower bounds
-> analytical compute and communication bounds
-> resource-constrained event-graph simulation
-> exact or nearest versioned profile lookup
-> calibrated residual model with uncertainty
-> compile and reference comparison
-> hardware measurement
```

### 8.1 Compute baseline

```text
kernel_lower_bound =
  max(
    work_by_operation_class / measured_operation_throughput,
    bytes_L2 / measured_L2_bandwidth,
    bytes_device_memory / measured_device_memory_bandwidth
  )
  + launch
  + mandatory_synchronization
```

Measured residuals account for occupancy, pipeline fill/drain, layout
conversion, cache regime, instruction mix, and code size.

### 8.2 Communication baseline

```text
communication =
  startup
  + bytes / effective_bandwidth
  + route_and_hierarchy
  + packing_and_conversion
  + synchronization
  + measured_contention
```

Collectives are algorithm- and hierarchy-specific. A single bandwidth number
is insufficient.

### 8.3 Event-graph simulation

End-to-end latency and throughput come from a resource-constrained event
simulation. A communication kernel consuming compute units is not free because
it is asynchronous. Overlap is never modeled by subtracting one duration from
another.

Peak memory is simulated across persistent state, semantic live ranges,
communication staging, kernel workspace, representation conversions, and
transform/checkpoint state.

### 8.4 Fusion delta

Benefits:

- removed launches;
- removed global materialization and staging;
- reduced transferred bytes;
- earlier region availability.

Costs:

- increased register and local-memory use;
- reduced occupancy or concurrent residency;
- compute resources consumed by communication;
- additional barriers, channels, and code size;
- pipeline imbalance and fill/drain;
- stricter progress and capability requirements;
- compilation and tuning time.

The same model compares fewer-communication/poor-local-tile plans against
more-communication/better-overlap plans.

## 9. Profiles and measurements

Profiles are keyed by:

- device, architecture, driver, OS;
- compiler, backend, Runtime, and communication implementation;
- topology identity;
- candidate and generator identity;
- typed workload bucket;
- representation, layout, and schedule;
- warm/cold and contention conditions.

They record distributions, counters, correctness status, objectives, and
Pareto status rather than only means. Tail objectives use tail measurements.
Stale profiles initialize search but cannot certify a changed target.

### 9.1 Active measurement

The model reports uncertainty. The tuner selects candidates likely to improve
the Pareto frontier or reduce decision-relevant uncertainty. It does not spend
equal measurement budget on clearly dominated candidates.

Every candidate records:

- parent candidate;
- one generic transformation;
- changed features;
- predicted cost/resource delta;
- verification and compilation result;
- measured delta.

This supports cross-layer credit assignment: whether a result came from
partition geometry, communication, conversion, fusion, or local scheduling.

### 9.2 Model evaluation

Report:

- top-k recall;
- rank correlation;
- selected-plan regret;
- uncertainty calibration;
- Pareto-front recall;
- compilation and measurement budget.

Average runtime prediction error alone is not an adequate planner metric.

### 9.3 Optimization memory

Optimization memory is a versioned evidence store, not a committed source
kernel library:

```text
OptimizationMemoryRecord {
  semantic_and_analysis_fingerprint
  shape_phase_topology_and_target_conditions
  reusable_optimization_decision_subgraph
  implementation_and_artifact_identity
  EvidenceBundle
  failed_attempts_and_counterexamples
  generator_and_toolchain_identity
}
```

Retrieval is itself a `CandidateGenerator`: it finds related records, adapts
their decisions to the current problem, and submits the result to normal
verification. Exact artifacts may be reused only when all qualified identities
match.

Knowledge is promoted by evidence:

```text
one qualified success     -> cached implementation
repeated related success  -> parameterized schedule schema
cross-workload success    -> algorithmic rewrite or lowering pattern
stable prediction gain    -> shared cost feature or residual model
```

Failures and qualification boundaries remain first-class memory because they
prevent repeated expensive exploration.

## 10. Objectives and Pareto selection

Objectives may include:

- latency, tail latency, and throughput;
- peak and persistent memory;
- exposed communication and transfer bytes;
- utilization and contention;
- energy;
- numerical error and quality;
- recomputation;
- compilation and tuning time;
- failure-domain and availability policy.

The result is an explainable Pareto set. A future versioned deployment policy
selects one or more `PhysicalPlanVariant` records. These are not the current
compile-time Program typed specialization variants.

## 11. Explainability

Every selected plan records:

- user and recipe constraints, if any;
- propagated partitions and placements;
- representations and inserted conversions;
- redistributions and physical communication;
- rejected alternatives and reasons;
- estimated and measured costs with uncertainty;
- fusion and physical schedules;
- capability dependencies;
- ordinary fallback;
- recipe-free or recipe-seeded provenance.

## 12. Runtime adaptation

Runtime may observe shape/work buckets, declared phases, load distributions,
available devices/links, task durations, cache hits, and contention. Under a
future versioned contract it may select only among compatible installed
`PhysicalPlanVariant` records.

Runtime may not change hard placement, Program numerics, synchronization,
representation, or an immutable installed plan. New decisions require
compilation and validation.

### 12.1 Tiered optimization

Deployment does not wait for open-ended synthesis:

```text
Tier 0  deterministic reference or baseline lowering
Tier 1  reusable rules, retrieval, and explicit cost-guided search
Tier 2  bounded autotuning and active measurement
Tier 3  agentic algorithm, graph, and kernel synthesis
```

Each tier may install a better qualified `PhysicalPlanVariant` without mutating
an executing one. Expensive tiers focus on high-regret, high-uncertainty,
out-of-distribution, or high-value regions. Compilation and measurement budget
are explicit objectives.

## 13. Generator specialization: agents

### 13.1 Inputs and outputs

An agent receives canonical IR summaries, generic action schemas, topology and
capabilities, relevant profile slices, model uncertainty, verifier
obligations, and prior failures.

It may propose:

- complete graph-level numerical, partition, placement, redistribution, and
  asynchronous plans;
- partition or placement transformations;
- task decompositions and availability regions;
- compositional fusion, split, or algorithmic replacement regions;
- physical schedule templates or concrete physical IR;
- target lowering implementations in any registered inspectable target IR or
  source adapter;
- search-order and pruning hypotheses;
- new cost features, analytical components, event simulators, residual models,
  or candidate-specific cost extensions.

Typed builders are preferred. Free-form DSL must parse into the same typed IR
and pass the same checks. An agent is therefore allowed to act both as a
graph-level optimizer and as a kernel-lowering pass. It need not encode a
successful implementation as a deterministic compiler rule before that
implementation can be evaluated or installed.

### 13.2 One protocol, different proposal policies

Operation analysis interfaces serve every generator:

```text
operation semantics and analysis
  -> CandidateGenerator
  -> typed FusionRegion, PlanCandidate, or ImplementationCandidate
  -> shared verification and evaluation
```

Rules provide fast, reproducible coverage of known cases; solvers provide
systematic bounded exploration; retrieval amortizes prior work; and agents
provide a broader but uncertain proposal distribution that may escape the
current transformation closure. These are policies behind one protocol, not
separate compiler paths. Rules are retained as validated, amortized knowledge
rather than treated as the maximum intelligence available to the compiler.

### 13.3 Evaluation

```text
construct
-> verify
-> compile
-> reference/differential test
-> benchmark
-> classify and record
```

Failures are semantic, capability, resource, progress, compilation,
correctness, numerical, or performance failures.

The same agent must not make an uncalibrated performance claim authoritative
for the candidate it generated. It may request measurement, explain why the
current model is out of distribution, or propose an independently testable
cost extension.

### 13.4 Generated cost extensions

For a candidate outside the current explicit model, an agent may produce:

```text
GeneratedCostExtension {
  applicability_and_feature_extractor
  exact_accounting_dependencies
  analytical_decomposition_or_event_model
  topology_and_toolchain_dependencies
  calibration_parameters
  uncertainty_and_out_of_distribution_policy
  required_microbenchmarks_and_holdout_cases
}
```

The extension is advisory until dimensional and monotonicity checks,
microbenchmarks, held-out calibration, and candidate measurements establish
its useful range. A small candidate set may be measured directly instead.
Repeatedly validated extensions may enter the versioned shared cost model;
failed extensions remain failure records.

The resulting division of authority is:

```text
agent intuition      = proposal and exploration prior
explicit cost model  = shared calibrated critic
hardware measurement = performance evidence
verifier             = correctness authority
```

### 13.5 Insight promotion

```text
InsightRecord {
  semantic_preconditions
  topology_and_capability_preconditions
  transformation_or_feature
  predicted_resource_and_cost_delta
  measured_evidence
  uncertainty
  counterexamples_and_failure_boundaries
  toolchain_identity
}
```

An insight becomes a deterministic compiler rule only after held-out replay
across shapes, devices, versions, and negative cases. Generated artifacts stay
in caches or bundles rather than becoming a source kernel library. A
successful one-off generated implementation may be installed as a qualified
physical-plan variant without first becoming a general rule. Repeated success
may be distilled into an algorithmic rewrite, schedule schema, target lowering
pattern, cost feature, or residual model.

As agents improve they may control more proposal policy. They never replace
semantic authority, verification, reference checks, measurement, or immutable
plan installation.

## 14. Verification requirements

Planner evaluation must distinguish recipe-free from seeded search, compare
small spaces with exhaustive ground truth, measure selected-plan regret, and
preserve explanation and fallback. Cost estimates never bypass legality.
Delivery and acceptance gates are owned solely by
[`implementation_roadmap.md`](implementation_roadmap.md).
