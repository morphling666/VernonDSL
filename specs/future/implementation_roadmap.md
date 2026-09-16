# Future compiler implementation roadmap

Status: future implementation plan, not a current VernonDSL contract.

This document turns the architecture in [`architecture.md`](architecture.md)
into incremental implementation and acceptance milestones. It does not commit
dates or change current public contracts.

## 1. Delivery policy

- Each phase preserves the current Program, manifest, Runtime, and ABI until a
  separately versioned release.
- New compiler IR remains inspectable before serialization is proposed.
- Every optimization has an ordinary reference or explicit unsupported path.
- Correctness and deterministic rejection precede profitability tuning.
- Recipe-free automatic planning is evaluated separately from recipe-seeded
  planning.
- Generated IR and artifacts remain cache/deployment outputs.

## 2. Current implementation entry points

| Future work | Existing entry point |
| --- | --- |
| Program compute and graphics Nodes | `source/include/mlir/Dialect/VernonProgram/IR/VernonProgramOps.td` |
| TensorView load/store/atomic/reduction | `source/include/mlir/Dialect/Vernon/IR/VernonOps.td` |
| Effects, barriers, validation | `source/lib/Dialect/Vernon/Transforms/VernonValidation.cpp` |
| Shape and contraction analysis | `source/include/mlir/Dialect/Vernon/Transforms/VernonTensorShapeSemantics.h` |
| Storage projection | `source/include/mlir/Dialect/Vernon/Transforms/VernonStorageProjection.h` |
| GPU lowering | `source/lib/Dialect/Vernon/Transforms/VernonToGPU.cpp` |
| Tensor/SPIR-V lowering | `source/lib/Dialect/Vernon/Transforms/VernonLowerGPUTensors.cpp`, `VernonConvertGPUToSPIRV.cpp` |
| Target capability rejection | `source/lib/compiler/compiler_dispatch.cpp` |
| Resolved transfer/dependency/residency | `source/lib/runtime/resolved_execution_plan.h` |
| Manifest resolution | `source/lib/runtime/program_execution_manifest.cpp` |
| Backend invocation | `source/lib/runtime/runtime_pipeline_backend.h` |
| Pass registration | `source/tools/vernon_opt/vernon_opt.cpp` |

Suggested new compiler-side locations:

```text
source/include/mlir/Interfaces/VernonAnalyzableRegion.td
source/include/mlir/Dialect/VernonDist/IR/
source/include/mlir/Dialect/VernonTask/IR/
source/include/mlir/Dialect/VernonTile/IR/
source/lib/Dialect/VernonDist/Analysis/
source/lib/Dialect/VernonDist/Transforms/
source/lib/Dialect/VernonTile/Transforms/
source/lib/planner/
```

Compiler planning IR may eventually lower into a versioned extension of the
resolved-plan authority. The current `ResolvedExecutionPlan` models
single-resource Host/Device residency and HostUpload/DeviceCopy/Readback; it
does not yet provide distributed ownership, transport, participants, or
failure semantics. Runtime executes accepted validated choices rather than
recovering partition semantics or running compiler search.

## 3. PR 0: Contracts and baselines

Deliver:

- final analysis-interface and IR field drafts;
- `PlanCandidate`, topology, profile, measurement, and insight schemas;
- CUDA Tile IR bytecode/toolchain compatibility policy;
- deterministic reference Programs and benchmark harness.

Baselines:

- matmul or reduction plus epilogue;
- two-device stencil or tiled reduction;
- grouped INT4 fused-dequant contraction;
- ordinary replicated/unfused execution.

Exit:

- references execute deterministically;
- expected numerical policy and measured metrics are recorded;
- no public contract fields are added.

## 4. PR 1: Analyzable interfaces

Deliver MLIR interfaces for iteration/access, iterator/reduction semantics,
effects/aliases, partition, fusion, reference semantics, and cost features.

Initial operations:

- elementwise map and broadcast;
- matmul/contraction;
- associative reduction;
- TensorView load/store/atomic;
- stencil/neighborhood;
- pack/unpack and quantization conversion.

Tests:

- exact affine region derivation;
- conservative indirect/unknown behavior;
- reduction algebra and reassociation diagnostics;
- opaque Stage fusion barrier;
- analysis dumps stable enough for review.

Exit: partition and fusion facts are explicit without changing serialized
Program contracts.

## 5. Distributed execution contract gate

Before compiler work reaches multi-resource Runtime execution, accept a
separately versioned contract defining:

- resource and process identity;
- allocation and Value ownership;
- topology discovery and supplied topology profiles;
- transport, participants, ordering, visibility, and completion;
- forward progress, timeout, cancellation, and failure behavior;
- deployment, compatibility, security, and artifact distribution;
- immutable physical-plan installation and selection.

Compiler-only partition analysis may precede this gate. Runtime communication,
multi-process execution, and serialized distributed plans may not.

## 6. PR 2: Logical partition and placement

Deliver:

- compiler-side `Partition`, `Placement`, `Replication`, and `PartialValue`;
- hard/prefer/open/closed constraint projection;
- bidirectional factor propagation;
- two-resource topology input;
- deterministic conflict diagnostics.

Initial scope:

- equal shard and replicate;
- one split reduction factor creating partial Values;
- persistent versus temporary placement;
- no named parallelism recipe in the automatic path.

Exit: a two-resource reference plan projects valid local domains and rejects
incompatible domains, topology, and memory.

## 7. PR 3: Redistribution and async tasks

Deliver:

- logical `Redistribution`;
- typed compute, copy, send/receive, reduce/combine, conversion, signal/wait,
  and completion tasks;
- Runtime communication reference executor;
- resource, lifetime, and dependency graph.

Tests:

- shard mismatch creates transfer;
- split reduction creates combine;
- representation mismatch creates conversion;
- participants, ownership, and completion are inspectable;
- reference ordering matches ordinary execution.

Exit: the asynchronous event graph reproduces reference semantics and reports
communication and peak memory.

## 8. PR 4: Fusion and region availability

Deliver:

- bounded producer-consumer region growth;
- `FusionRegion`;
- chunk and `AvailableRegion` tokens;
- ordinary, overlapped, and communication-fused candidate generation;
- target-independent legality and resource lower bounds.

Initial patterns:

- elementwise chain;
- completed contraction/reduction tile to epilogue;
- receive chunk to consumer tile;
- producer tile to send or reduce chunk;
- conversion chunk to native compute tile.

Tests inject alias, incomplete reduction, participant, ordering, and cycle
failures.

Exit: the same semantic region produces inspectable unfused, overlapped, and
fused candidates with split fallback.

## 9. PR 5: Vernon physical IR

Deliver:

- typed tile Values and immutable layouts;
- load/store/map/contraction/reduction/conversion tasks;
- communication chunks;
- scope, role, pipeline, memory lifetime, barriers, channels, and tokens;
- bounded `PersistentSchedule`;
- composable verifier.

Verifier covers coverage, bounds, effects, aliases, numerics, layouts,
ownership, completion, barriers, epochs, resources, progress, residency,
termination, and fallback.

Exit: local fused kernels match unfused references; injected failures all fail
closed.

## 10. PR 6: Numerical representation

Deliver:

- exact quantized type and `ScaledValue`;
- storage, transfer, compute, accumulator, and result policy;
- scalar bit-exact references;
- packing ABI;
- exact operation capability tuples;
- calibration/quality record;
- distributed representation and scale synchronization tasks.

Initial implementations:

- grouped symmetric INT4;
- FP16/BF16 with FP32 accumulation;
- fused dequant plus wider contraction;
- one dynamic scale path.

Exit: one CUDA and one non-CUDA path preserve compressed storage and report
native/emulated status, conversion cost, communication bytes, and quality.

## 11. PR 7: Backend leaf lowering

Deliver:

- portable Linalg/Vector/GPU to SPIR-V or LLVM path;
- CUDA Tile IR bytecode emission for qualified leaf regions;
- CUDA Driver JIT loading;
- optional `tileiras` AOT;
- complete artifact/cache identity;
- ordinary fallback.

CuTe DSL and Triton may provide performance baselines but are not production
intermediates.

Exit: one FusionRegion runs through CUDA Tile IR and portable fallback with
equivalent results; unsupported bytecode/capabilities reject or fall back.

## 12. PR 8: Profiles and joint cost model

Deliver:

- versioned hardware topology and operation profiles;
- complete PlanCandidate feature extraction;
- compute/communication lower bounds;
- resource-constrained event simulation;
- profile lookup and calibrated residual;
- uncertainty and active measurement;
- plan explanation and Pareto records.

Tests:

- analytical lower bounds are conservative;
- event simulation exposes true overlap constraints;
- legal-but-slower fusion is rejected;
- stale profile cannot certify a final plan;
- selected-plan regret is measured on enumerable spaces.

Exit: selected plans report estimated and measured compute, communication,
overlap, peak memory, quality, uncertainty, and fallback.

## 13. PR 9: Recipe-free search and agents

Deliver:

- hierarchical beam/Pareto search over generic actions;
- bounded outer feedback;
- exhaustive small-case oracle;
- structured Agent proposal import/export;
- sandboxed verify/compile/reference/benchmark loop;
- failure and insight records.

Exit:

- automatic search receives no TP/EP/DP/CP/PP label;
- it rediscovers known Pareto plans on small cases;
- it exceeds one fixed recipe baseline on held-out workload/topology;
- Agent value is reported under fixed compile and measurement budgets;
- only held-out reproduced insights are promoted to deterministic rules.

## 14. Primary acceptance workloads

### 14.1 Two-device stencil or tiled reduction

Exercises generic spatial/index partition, partial Values, redistribution,
overlap, communication fusion, and fallback.

Required variants:

- ordinary communication then compute;
- communication/compute overlap;
- optional chunk-fused path;
- single-resource reference.

Required evidence:

- explicit local domains and redistribution;
- equivalent results;
- deterministic topology/memory rejection;
- measured compute, communication, overlap, resources, and peak memory.

### 14.2 Distributed contraction and epilogue

Program:

```text
y = epilogue(matmul(a, b), bias)
```

Exercises factor propagation, operand replication/sharding alternatives,
partial reduction, communication chunk availability, epilogue completion,
mixed precision, and local tiles.

The compiler receives generic semantics rather than a tensor-parallel label.

### 14.3 Quantized contraction

Exercises canonical INT4 packing, grouped scales, transfer representation,
fused dequantization, accumulation, exact capability, and numerical quality.

Required variants:

- compressed transfer plus fused conversion;
- wider transfer/compute fallback;
- CUDA Tile IR leaf when supported;
- scalar/CPU reference.

### 14.4 Sparse/data-dependent workload

Exercises indirect access summaries, uneven partitions, explicit movement,
load balancing, conservative fusion, and distribution-sensitive profiles.

It validates that the common architecture does not require affine TileTask
semantics for every operation.

### 14.5 Compute-to-graphics workload

Exercises Program-region placement, shared Storage versions, image
transitions, compute tiles, graphics tasks, publication, and coarse
heterogeneous transfer.

It validates that the async task graph is the common waist rather than Linalg.

## 15. LLM stress validation

LLM workloads combine contractions, reductions, persistent parameters/caches,
data-dependent routing, low precision, multiple useful partition factors,
bulk and fine-grained communication, and distinct latency/throughput phases.

Human names such as DP, TP, PP, CP, and EP are used only to label known
baseline recipes. They lower to generic constraints and tasks.

Validation families:

| Workload | Generic behavior |
| --- | --- |
| Dense block | Factor partition, replication/partial Value, redistribution, contraction and epilogue |
| Sparse expert block | Data-dependent index partition, Storage placement/replication, dispatch, grouped compute, combine |
| Pipeline deployment | Program-region placement, boundary transfer, buffering, dependencies |
| Quantized linear block | Encoding, transfer representation, conversion, accumulation, capability |

Profiles include training, prefill, decode, and sparse/skewed execution as
objective and data-distribution policies, not IR semantics.

Recipe-free and recipe-seeded runs are reported separately. A recipe name must
not appear in the recipe-free planner state, action schema, capability query,
or feature vector.

Success:

- generic plans are inspectable;
- hard user constraints are preserved;
- ordinary, overlapped, and fused variants agree;
- a future versioned Runtime policy may select separate prefill/decode
  `PhysicalPlanVariant` records;
- measured heterogeneous topology drives coarse/fine transfer;
- no LLM-specific concept is required by Program, physical IR, or Runtime.

## 16. Metrics

Correctness:

- reference agreement and numerical-quality bounds;
- deterministic invalid-plan rejection;
- fallback coverage.

Performance:

- latency/throughput distributions;
- exposed communication;
- peak/persistent memory;
- occupancy, residency, code size;
- conversion and scale synchronization;
- compilation and tuning budget.

Planner/model:

- top-k recall and rank correlation;
- selected-plan regret;
- uncertainty calibration;
- Pareto-front recall;
- recipe-free versus seeded result.

Agent:

- Pareto improvement per evaluation budget;
- reproducible insight count;
- held-out success and counterexample rate.

## 17. Versioned-release gates

Before any future representation becomes public:

- field ownership and serialization are finalized;
- old/new readers have deterministic compatibility behavior;
- Runtime ABI and manifest versions are explicit;
- every backend has a legal implementation or unsupported diagnostic;
- fallback and reference tests pass;
- current compiler and Runtime regressions remain green;
- generated artifacts contain complete identity and capability requirements.
