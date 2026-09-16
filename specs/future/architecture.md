# Future compiler architecture

Status: future design, not a current VernonDSL contract.

This document is the architectural entry point for Vernon's future optimizing
compiler. It defines the questions answered by each layer, the authority of
each representation, and the invariants shared by machine learning,
simulation, scientific computing, sparse workloads, graphics, and
heterogeneous deployment.

Detailed contracts are split into:

- [`optimization_ir.md`](optimization_ir.md): analysis interfaces, distributed
  primitives, fusion, tile tasks, schedules, and verification;
- [`joint_planner.md`](joint_planner.md): topology, joint search, cost models,
  profiles, measurement, and agents;
- [`numerical_representation.md`](numerical_representation.md): mixed
  precision, quantization, storage encodings, accumulation, and quality;
- [`backend_lowering.md`](backend_lowering.md): target IRs, artifacts, and
  fallback policy;
- [`implementation_roadmap.md`](implementation_roadmap.md): implementation
  sequence and acceptance workloads.

Nothing here changes the current Program, compiler, manifest, Runtime, RHI, or
public ABI. A future design becomes current only through a separately
versioned release with implementation and acceptance coverage.

## 1. Goal

The compiler jointly selects:

- numerical and physical representation;
- logical partition and physical placement;
- replication and partial-result structure;
- redistribution and communication;
- asynchronous overlap and memory lifetime;
- fusion boundaries;
- tile, sparse, graphics, communication, and persistent schedules;
- target leaf lowering and validated Runtime variant.

These decisions interact. A compressed representation may reduce network
traffic but require conversion and scale synchronization. A partition with
minimal communication may create poor local tiles. A fused communication
kernel may remove staging while consuming the compute resources required for
forward progress.

The compiler therefore evaluates complete plans rather than optimizing each
decision once in isolation.

## 2. Central architectural claim

Vernon shares one optimization protocol across domains without inventing one
universal operation or one universal kernel:

> Program IR owns meaning. Analysis interfaces expose structure. Logical plans
> own partition and placement. The asynchronous task graph owns required
> physical work and dependencies. Typed physical IR owns target-oriented
> schedules. Verifiers define legality. Models and measurements select among
> legal candidates.

Machine learning strategy names such as data, tensor, pipeline, context, and
expert parallelism are optional source recipes and validation terminology.
They are not core IR operations, planner actions, or cost-model features.

## 3. End-to-end model

```mermaid
flowchart TB
    Program["1. Semantic Program IR"]
    Analysis["2. Analyzable relations"]
    Logical["3. Logical distribution"]
    Tasks["4. Asynchronous task graph"]
    Physical["5. Typed physical specialization"]
    Target["6. Target leaf lowering"]
    Resolved["7. Future physical-plan variants"]
    Planner["Joint planner"]
    Verify["Composable verifiers"]
    Measure["Compile, reference, measure"]
    Profiles[("Versioned profiles")]

    Program --> Analysis --> Logical --> Tasks --> Physical --> Target
    Target --> Verify --> Measure --> Resolved
    Planner --> Logical
    Planner --> Physical
    Profiles --> Planner
    Measure --> Profiles
```

There are two kinds of structure:

- IR layers progressively own more physical decisions;
- optimization services generate, reject, rank, compile, and measure
  alternatives.

The joint planner is a service, not another semantic IR. Cost feedback may
select another candidate but cannot change Program meaning.

## 4. Layer authorities

### 4.1 Semantic Program IR

Program IR is the sole authority for:

- logical Values and Storages;
- typed compute, graphics, sparse, coordination, and control operations;
- shape, index, numerical, and control semantics;
- effects, aliases, Storage-version transitions, and observable ordering;
- public boundaries;
- semantic transforms such as VJP.

It does not own device count, topology placement, communication algorithm,
stream, tile size, local layout, persistent worker role, measured duration, or
selected target artifact.

Runtime invocation facts such as concrete dynamic dimensions, strides,
offsets, resources, and launch grids remain invocation state. They do not
become semantic identity.

### 4.2 Analyzable relations

Operations expose only the analysis interfaces meaningful to them:

- iteration or index domain;
- access relations;
- effects, aliases, and semantic dependencies;
- iterator and reduction meaning;
- legal partition and fusion relations;
- numerical policy;
- conservative reference implementation.

The interfaces do not form a second semantic graph. Unknown structure reduces
optimization opportunities but does not invalidate an opaque Stage.

### 4.3 Logical distribution

This layer owns:

- logical pieces of Values, Storages, domains, index sets, operations, or
  Program regions;
- placement on compute and memory resources;
- replication and consistency requirements;
- partial Values and their combine semantics;
- representation requirements;
- logical redistribution caused by producer-consumer mismatch.

Logical distribution does not select a collective library, stream, worker
role, kernel-local layout, or native operation.

### 4.4 Asynchronous task graph

After placement, every required action is explicit:

- compute;
- communication or copy;
- representation conversion;
- synchronization and completion;
- graphics;
- allocation, prefetch, eviction, publication, or I/O where applicable.

Tasks reference Program facts and logical-plan provenance. They do not redefine
Values, effects, or meaning. Overlap is represented by dependencies and
resources, not by subtracting estimated durations.

### 4.5 Typed physical specialization

Different domains retain different physical tasks:

- `TileTask` for structured and indexable compute;
- `SparseTask` for indirect or data-dependent work;
- `RenderScopeTask` for graphics;
- `CommunicationTask` for transfer, remote access, and collective work;
- `PersistentSchedule` for bounded resident composition.

They share ownership, resource, completion, capability, verification, and
fallback protocols without pretending their domain semantics are identical.

### 4.6 Target leaf lowering

A target backend receives a verified leaf region and implements its declared
semantics and synchronization. It may use CUDA Tile IR, GPU/NVGPU/NVVM,
SPIR-V, the current SPIRV-Cross MSL/GLSL/HLSL routes, DXC, LLVM, or
compiler-owned intrinsics. A future dedicated target IR requires its own
accepted design and does not silently replace the current route.

Target IR does not become the authority for global partition, cross-device
progress, Program numerics, or fallback.

### 4.7 Physical plan variants

The future compiler may produce immutable `PhysicalPlanVariant` records
containing artifacts, task execution, capability requirements, and fallback.
Runtime may select among installed physical-plan variants using declared
shape, phase, resource, or telemetry buckets. This concept is distinct from
the current compile-time Program typed specialization variant and requires a
separately versioned serialization, installation, and selection contract. A
materially new plan is compiled, validated, and installed as another physical
plan variant.

Runtime does not invent synchronization, representations, partitioning, or
unchecked schedules.

## 5. Authority matrix

| Fact | Program | Logical plan | Async tasks | Physical IR | Resolved plan |
| --- | :---: | :---: | :---: | :---: | :---: |
| Values, Storages, operations, effects | owner | reference | reference | projection | validated |
| Index, shape, numerical semantics | owner | projection | projection | local projection | validated |
| Partition, placement, replication, partial combine | constraint | owner | materialized | local projection | validated |
| Copy, communication, conversion, completion | — | requirement | owner | optional fused form | selected |
| Queue, engine, lifetime, residency | — | — | owner | requirement | selected |
| Local layout, ownership scope, pipeline | — | — | — | owner | selected |
| Artifact, capability dependency, fallback | — | — | candidate | requirement | owner |

`—` means that the layer must not reconstruct or independently own the fact.

## 6. Communication distinction

Two meanings of communication remain separate:

1. A Program may contain an explicitly authored semantic exchange, combine,
   or coordination operation.
2. The compiler introduces physical redistribution because selected
   placements or representations are incompatible.

The second is not inserted into semantic Program IR. It becomes explicit after
logical planning and before communication-aware fusion. NCCL, MPI, NVSHMEM, a
driver API, or a backend remote instruction never defines Program meaning.

## 7. Joint optimization boundary

A complete candidate contains:

```text
PlanCandidate {
  semantic_provenance
  numerical_and_representation_plan
  partition_and_placement_plan
  partial_values_and_redistributions
  asynchronous_task_graph
  fusion_regions
  typed_physical_schedules
  target_leaf_lowerings
  resource_and_capability_requirements
  split_fallbacks
}
```

Search is hierarchical to remain bounded, but earlier choices are revisable.
Every partition candidate materializes its communication and receives a local
fusion/tile evaluation. Material differences between estimates and measured
lower-level costs may trigger bounded reconsideration of representation,
partition, or placement.

## 8. Domain independence

Shared infrastructure includes:

- canonical Program authority;
- analysis-interface protocol;
- partition and placement vocabulary;
- asynchronous dependencies and completion;
- resource and capability framework;
- composable verification;
- profiles, measurements, explanations, and Pareto selection;
- future immutable physical-plan variants and ordinary fallback.

Domain-specific components include:

- semantic operations;
- precision of access relations;
- partition and fusion rules;
- physical task kinds;
- cost features and schedule templates;
- reference implementations;
- peak backend paths.

Generality means a domain plugs typed components into common services. It does
not mean reducing sparse traversal or graphics to Linalg, or requiring every
Program to become one megakernel.

## 9. Program transforms

Semantic transforms run before distribution:

```text
Primal Program
  -> semantic transform such as VJP
  -> transformed Program
  -> numerical and distribution planning
  -> checkpoint and recompute planning
  -> physical fusion and target lowering
```

Transform, distribution, checkpoint, fusion, and tile planners may exchange
cost feedback. None may take over another layer's semantic authority.

## 10. User control

Users may constrain:

- partition factors, ranges, index sets, operations, or Program regions;
- allowed or required devices, memory tiers, and failure domains;
- replication, consistency, and combine policy;
- allowed redistribution classes, routes, or byte budgets;
- numerical encoding, accumulation, reproducibility, and quality;
- required or forbidden fusion boundaries;
- residency and memory budgets;
- latency, throughput, energy, accuracy, and tuning budgets.

Constraint strength is `hard`, `prefer`, `open`, or `closed`. Hard constraints
are never silently weakened because a model predicts a faster alternative.
Every selected plan explains how constraints were applied.

## 11. Kernel provenance

Executable compute originates from user-authored or compiler-generated typed
Vernon IR. Generated IR and artifacts belong in the compiler cache or
deployment bundle, not in an expanding committed kernel-template library.

External source DSLs may serve as experimental adapters or performance
references, but deployment does not require an external kernel package unless
a separately declared target toolchain requires it.

## 12. Architectural invariants

- Program IR remains the sole semantic authority.
- Analysis interfaces expose facts; they do not become a second semantic graph.
- Global partition and placement remain separate from kernel-local layout.
- Numerical meaning remains separate from storage and target representation.
- Compiler-induced communication is explicit before communication fusion.
- The asynchronous graph owns physical dependencies.
- Physical tasks remain typed.
- Capabilities, resources, progress, and fallback are explicit and fail closed.
- Every optimized candidate has a reference, split point, fallback, or
  explicit unsupported-target diagnostic.
- Cost and measurement select candidates but never mutate semantics.
- Under a future versioned selection contract, Runtime selects only installed
  validated `PhysicalPlanVariant` records.
- Agents propose candidates and insights; they do not own correctness.
- Current public contracts remain unchanged until a versioned release.

## 13. Non-goals

This architecture does not claim:

- one IR operation models every domain equally well;
- the planner proves a global optimum;
- affine maps model arbitrary sparse or graphics behavior;
- every Program becomes one persistent kernel;
- a cost model predicts exact performance;
- named human parallelism strategies define the automatic search;
- Runtime dynamically creates plans;
- backend code may supply missing Program semantics.
