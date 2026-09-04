# Vernon Compiler & Distributed Runtime — Architecture Plan

## 0. Current State

Vernon currently provides:

```text
Python API
    │
    ▼
Torch-like Module / Tensor API
    │
    ▼
Program Graph
    │
    ├── builtin ops
    │     ├── matmul
    │     ├── add
    │     ├── conv
    │     └── ...
    │
    └── custom Kernel DSL
          ├── arbitrary control flow
          ├── thread-level programming model
          ├── graphics shader support
          └── automatic backward generation
    │
    ▼
Backend lowering
    ├── CPU
    ├── CUDA
    ├── SPIR-V
    └── Metal
```

The system already supports:

* Torch-like `Module`
* graph construction
* forward execution
* automatic backward generation
* builtin operators
* custom kernel DSL nodes
* arbitrary control flow inside kernels
* thread-level execution model
* multiple hardware backends

Example:

```python
x = vd.matmul(a, b)
y = my_kernel(x)
z = vd.relu(y)
```

The primary missing capabilities are:

1. Graph-level fusion
2. Kernel DSL lowering and optimization
3. Distributed execution primitives/runtime

---

# 1. Overall Architecture

The next architecture should introduce a clear separation between:

```text
Frontend IR
    ↓
Graph / Tensor IR
    ↓
High-level optimization
    ↓
Fusion / Partitioning
    ↓
Kernel IR
    ↓
Kernel DSL
    ↓
Backend lowering
    ↓
CPU / CUDA / SPIR-V / Metal
```

and separately:

```text
Distributed Graph
    ↓
Partition
    ↓
Device Placement
    ↓
Communication Graph
    ↓
Distributed Schedule
    ↓
Local Execution Graph
```

The target architecture is:

```text
                         Python API
                            │
                            ▼
                    ┌─────────────────┐
                    │   Vernon Graph  │
                    │       IR        │
                    └────────┬────────┘
                             │
             ┌───────────────┼────────────────┐
             │               │                │
             ▼               ▼                ▼
          Fusion       Autodiff/Grad      Distributed
             │               │                │
             └───────────────┼────────────────┘
                             ▼
                    Optimized Graph IR
                             │
                             ▼
                    Kernel Generation IR
                             │
                             ▼
                       Kernel DSL
                             │
                             ▼
                    Backend Lowering
                             │
          ┌──────────┬───────┼────────┬──────────┐
          ▼          ▼       ▼        ▼          ▼
         CPU       CUDA    SPIR-V   Metal    Network
```

The key principle:

> **Graph IR describes what is computed. Kernel DSL describes how a kernel executes.**

Do not collapse these two abstractions.

---

# 2. Phase I — Stabilize the IR Boundary

Before implementing fusion, formalize the boundary between:

### Graph-level computation

```text
matmul
add
relu
softmax
conv
attention
...
```

and:

### Kernel-level computation

```text
load
store
thread id
shared memory
barrier
loop
branch
vector arithmetic
matrix operation
...
```

A builtin op should therefore have two representations:

```text
BuiltinOp
    │
    ├── semantic definition
    │
    └── lowering implementation
```

For example:

```text
MatMulOp
    │
    ├── shape semantics
    ├── dtype semantics
    ├── autodiff rule
    └── lowering
          │
          ▼
      Kernel IR / DSL
```

This allows:

```text
matmul + add + relu
```

to remain a graph-level expression until fusion decides what should become a kernel.

---

# 3. Phase II — Fusion

## 3.1 Do not fuse the entire graph

Fusion should produce **fusion groups**, not one giant kernel.

Example:

```text
Graph:

A
│
▼
MatMul
│
▼
Add
│
▼
ReLU
│
▼
Softmax
│
▼
MatMul
```

could become:

```text
FusionGroup 0:
    MatMul
    Add
    ReLU

FusionGroup 1:
    Softmax

FusionGroup 2:
    MatMul
```

The optimized graph becomes:

```text
A
│
▼
[FusedMatMulAddRelu]
│
▼
[Softmax]
│
▼
[MatMul]
```

Fusion groups should be first-class internal compiler objects.

---

# 3.2 Fusion should have multiple strategies

Do not implement one universal "fusion pass".

Use:

```text
FusionEngine
    │
    ├── ElementwiseFusion
    ├── ProducerConsumerFusion
    ├── ReductionFusion
    ├── MatMulEpilogueFusion
    ├── ConvEpilogueFusion
    ├── AttentionFusion
    └── CustomFusionPatterns
```

Examples:

```text
add → relu
```

is generic elementwise fusion.

```text
matmul → add → relu
```

is GEMM epilogue fusion.

```text
QKV → attention
```

is a domain-specific pattern.

---

# 3.3 Pattern matching

Introduce a pattern representation:

```text
Pattern:
    MatMul
      ↓
    Add
      ↓
    ReLU
```

and matching:

```text
Graph
  ↓
PatternMatcher
  ↓
MatchedSubgraph
  ↓
FusionGroup
```

Patterns should be declarative where possible.

Example conceptual API:

```python
@vd.fusion_pattern
def matmul_add_relu(a, b, bias):
    x = vd.matmul(a, b)
    x = vd.add(x, bias)
    return vd.relu(x)
```

The exact API can be decided later.

The important architectural property is:

> Fusion patterns should not require rewriting the entire compiler.

---

# 3.4 Fusion legality

Fusion must check:

```text
shape compatibility
dtype compatibility
layout compatibility
aliasing
memory dependency
side effects
reduction dependencies
device compatibility
```

A fusion candidate is therefore:

```text
Candidate
    │
    ├── pattern match
    ├── legality check
    └── cost model
```

Only then:

```text
FusionGroup
```

is created.

---

# 3.5 Fusion cost model

Fusion is not always beneficial.

Consider:

```text
A → B → C
```

Fusion may reduce:

```text
global memory traffic
kernel launch overhead
intermediate allocation
```

but increase:

```text
register pressure
shared memory usage
occupancy loss
instruction count
compilation time
```

Therefore:

```text
FusionDecision =
    legality
    +
    estimated_memory_saving
    +
    estimated_launch_saving
    -
    estimated_register_cost
    -
    estimated_occupancy_loss
```

Initially use heuristics.

Later replace with backend-specific cost models.

---

# 4. Phase III — Fusion → Kernel DSL

The preferred Vernon architecture is:

```text
Graph IR
    ↓
Fusion Groups
    ↓
Kernel IR
    ↓
Kernel DSL
    ↓
Backend
```

Do NOT immediately lower fusion directly into CUDA/SPIR-V/Metal.

This keeps your existing Kernel DSL useful.

Example:

```text
MatMul
   ↓
Add
   ↓
ReLU
```

becomes conceptually:

```text
FusedKernel
{
    load A
    load B

    x = matmul(A, B)

    x = x + bias

    x = max(x, 0)

    store x
}
```

Then your existing kernel compiler handles:

```text
Kernel DSL
    ↓
CUDA
SPIR-V
Metal
CPU
```

This also means a new backend does not require implementing the fusion system again.

---

# 5. Phase IV — Kernel DSL Optimization

The Kernel DSL should remain a general-purpose low-level programming model.

It must support:

```text
thread
block/workgroup
memory
vector
control flow
loops
branches
barriers
atomics
matrix operations
builtins
```

Fusion-generated kernels and user-written kernels should enter the same optimization pipeline.

```text
User Kernel DSL
       │
       ├──────────────┐
       │              │
       ▼              ▼
Fusion-generated   User-written
       │              │
       └──────┬───────┘
              ▼
          Kernel IR
              │
              ▼
       Kernel Optimizer
```

---

# 6. Kernel Optimization Pipeline

Introduce explicit kernel optimization passes.

### 6.1 Canonicalization

```text
constant folding
dead code elimination
algebraic simplification
control-flow simplification
```

### 6.2 Memory optimization

```text
load/store elimination
load hoisting
store sinking
memory coalescing
reuse analysis
local/shared memory promotion
```

### 6.3 Thread optimization

```text
thread-local value propagation
divergence analysis
barrier optimization
workgroup mapping
```

### 6.4 Vectorization

```text
scalar
  ↓
vector
  ↓
SIMD
```

### 6.5 Backend-specific lowering

```text
generic kernel IR
       │
 ┌─────┼─────────────┐
 ↓     ↓             ↓
CUDA  SPIR-V       Metal
```

Each backend can apply specialized transformations.

---

# 7. Matrix Operations

Your Kernel DSL already allows:

```python
x = vd.matmul(A, B)
```

This should remain a semantic operation in the kernel IR for as long as possible.

Do not immediately expand it into scalar multiply/add operations.

Instead:

```text
MatMul
   │
   ├── generic implementation
   │
   ├── CUDA Tensor Core implementation
   ├── SPIR-V matrix implementation
   ├── Metal matrix implementation
   └── CPU SIMD implementation
```

This gives the backend the opportunity to select specialized hardware.

---

# 8. Precision Architecture

Do not model FP4/FP8 purely as ordinary scalar dtypes.

Separate:

```text
Logical dtype
Storage encoding
Compute dtype
Accumulator dtype
Quantization metadata
```

Conceptually:

```text
Tensor
├── logical_dtype
├── storage_dtype
├── compute_dtype
├── accumulation_dtype
├── layout
└── quantization_metadata
```

Example:

```text
MoE Weight

logical dtype:
    FP8

storage:
    FP4 block quantized

scale:
    E8M0

compute:
    FP8

accumulate:
    BF16/FP32
```

This becomes important for future LLM support.

---

# 9. Phase V — Distributed Runtime

Do not begin by implementing:

```text
Megatron
DeepSpeed
FSDP
```

Instead provide **distributed primitives**.

The core abstraction should be:

```text
Device
    │
    ├── local CPU
    ├── CUDA GPU
    ├── Metal GPU
    └── SPIR-V GPU

Process
    │
    └── DeviceSet

Communication
    ├── send
    ├── recv
    ├── broadcast
    ├── reduce
    ├── all_reduce
    ├── all_gather
    ├── reduce_scatter
    └── all_to_all
```

---

# 10. Distributed Tensor Abstraction

Introduce something conceptually similar to:

```text
DistributedTensor
```

with:

```text
global shape
global dtype
placement
partition
```

Example:

```text
Tensor [8192, 8192]

placement:

GPU0 → rows 0:2048
GPU1 → rows 2048:4096
GPU2 → rows 4096:6144
GPU3 → rows 6144:8192
```

The graph remains device-independent.

Placement is metadata.

---

# 11. Device Topology

The runtime should explicitly represent:

```text
DeviceTopology
```

Example:

```text
             100Gb Ethernet
        ┌─────────┬─────────┐
        ↓         ↓         ↓
      GPU0      GPU1      GPU2
        │
      NVLink
        │
      GPU3
```

Edges contain:

```text
bandwidth
latency
transport
```

For example:

```text
NVLink:
    bandwidth = X
    latency = Y

PCIe:
    bandwidth = X
    latency = Y

Ethernet:
    bandwidth = X
    latency = Y
```

This information should be available to the planner.

---

# 12. Distributed Planner

The long-term target is:

```text
Graph
  ↓
Partition
  ↓
Placement
  ↓
Communication insertion
  ↓
Scheduling
```

Instead of exposing only:

```python
model.to("cuda:0")
```

the runtime can eventually reason about:

```text
where should this computation execute?
```

based on:

```text
memory
compute capability
bandwidth
latency
topology
tensor size
```

---

# 13. TP / DP / PP / EP as Planning Strategies

Do not make TP/DP/PP/EP fundamental IR concepts.

Treat them as higher-level partition strategies.

### Data Parallelism

```text
Graph
 ├── replica 0 → GPU0
 ├── replica 1 → GPU1
 └── replica 2 → GPU2
```

### Tensor Parallelism

```text
Op
 ↓
partition
 ├── GPU0
 ├── GPU1
 └── GPU2
```

### Pipeline Parallelism

```text
Layer 0~20  → device group A
Layer 21~40 → device group B
Layer 41~60 → device group C
```

### Expert Parallelism

```text
Expert 0~31   → GPU0
Expert 32~63  → GPU1
...
```

These strategies can eventually be composed.

```text
PP
 └── TP
      └── EP
           └── DP
```

But the core runtime only needs:

```text
partition
placement
communication
schedule
```

---

# 14. Communication-Aware Scheduling

The planner should be able to produce:

```text
compute
   │
   ├──────────────┐
   │              │
   ▼              ▼
compute       prefetch
                  │
                  ▼
               network
                  │
                  ▼
             next tensor
```

The objective is:

```text
communication latency
        ↓
overlap with computation
```

This becomes especially important for heterogeneous environments:

```text
CUDA GPU
Metal GPU
CPU
Vulkan GPU
network
SSD
```

---

# 15. Memory Hierarchy

Eventually model memory as:

```text
Device Memory
    │
    ├── registers
    ├── shared/local memory
    ├── VRAM/unified memory
    ├── host RAM
    └── SSD
```

Each edge has:

```text
capacity
bandwidth
latency
```

The planner can then perform:

```text
placement
+
prefetch
+
eviction
```

For very large MoE models:

```text
SSD
 ↓ prefetch
RAM
 ↓ prefetch
VRAM
 ↓
compute
```

This should be represented by the runtime rather than hardcoded into the LLM implementation.

---

# 16. Distributed Launch Architecture

The first version should remain simple.

Use a process-group model:

```text
machine 0
    └── Vernon process rank 0

machine 1
    └── Vernon process rank 1

machine 2
    └── Vernon process rank 2
```

Each process initializes:

```text
world_size
rank
local_devices
master_endpoint
```

Then:

```text
DistributedRuntime
    ↓
ProcessGroup
    ↓
CommunicationBackend
```

Communication backends can later include:

```text
TCP
RDMA
NCCL
MPI
custom transport
```

Do not build a scheduler/server first.

Start with a deterministic process-group runtime.

---

# 17. Automatic Differentiation + Distributed Execution

The existing autodiff system should remain graph-based.

Conceptually:

```text
Forward Graph
      │
      ▼
Backward Graph
      │
      ▼
Distributed Partitioning
```

Do not implement distributed backward separately.

For example:

```text
W
│
▼
MatMul
│
▼
Loss
```

Backward generates:

```text
dW
dX
```

Then distributed planner inserts:

```text
AllReduce
ReduceScatter
AllGather
```

where required by placement.

This keeps autodiff independent from distributed strategy.

---

# 18. Training Runtime

Once distributed primitives exist, training becomes:

```text
Module
  ↓
Forward Graph
  ↓
Fusion
  ↓
Kernel Generation
  ↓
Backward Graph
  ↓
Distributed Partition
  ↓
Communication insertion
  ↓
Scheduling
  ↓
Execution
```

Optimizer remains a relatively separate layer:

```text
loss
 ↓
backward
 ↓
gradients
 ↓
optimizer
 ↓
parameter update
```

Initially implement standard optimizers:

```text
SGD
Adam
AdamW
```

There is no need to invent a new optimizer system.

---

# 19. Recommended Implementation Order

Do NOT implement everything simultaneously.

## Milestone 1 — IR foundation

Goal:

```text
Graph IR
+
Kernel IR
+
clear lowering boundary
```

Deliverables:

* explicit op representation
* tensor metadata
* shape/dtype propagation
* kernel boundary
* backend capability model

---

## Milestone 2 — Basic fusion

Implement:

```text
elementwise fusion
producer-consumer fusion
matmul epilogue fusion
```

Target:

```text
matmul
→ add
→ relu
```

becomes one kernel.

Do not start with attention.

---

## Milestone 3 — Fusion → Kernel DSL

Build:

```text
FusionGroup
    ↓
Kernel Generator
    ↓
Kernel DSL
```

This is the most important architectural milestone.

After this:

```text
user kernel
```

and:

```text
compiler-generated fused kernel
```

share the same optimization/lowering path.

---

## Milestone 4 — Kernel optimizer

Implement:

1. constant folding
2. dead code elimination
3. common subexpression elimination
4. memory optimization
5. loop optimization
6. vectorization
7. backend-specific optimization

Then benchmark against unfused kernels.

---

## Milestone 5 — Pattern fusion

Add:

```text
softmax
layernorm
attention
GELU
RoPE
MLP
```

as patterns.

Especially:

```text
QKV
 ↓
Attention
 ↓
Projection
```

and:

```text
Linear
 ↓
Activation
 ↓
Linear
```

---

## Milestone 6 — Distributed primitives

Implement:

```text
ProcessGroup

send
recv
broadcast
all_reduce
all_gather
reduce_scatter
all_to_all
```

First support:

```text
CPU ↔ CPU
CUDA ↔ CUDA
```

Then heterogeneous devices.

---

## Milestone 7 — Distributed tensor / placement

Add:

```text
DistributedTensor
Placement
Partition
DeviceTopology
```

Then support explicit:

```text
replicate
shard
```

before attempting automatic planning.

---

## Milestone 8 — Distributed planner

Start with explicit strategies:

```text
DataParallel
TensorParallel
PipelineParallel
ExpertParallel
```

Then introduce:

```text
cost model
+
topology-aware placement
+
communication-aware scheduling
```

---

## Milestone 9 — Memory/offload planner

Add:

```text
VRAM
RAM
SSD
```

as memory tiers.

Implement:

```text
prefetch
eviction
overlap
```

This is especially valuable for large MoE inference.

---

# 20. What NOT to Build Yet

Avoid these early:

### Do not build your own CUDA-like backend

Your Kernel DSL already provides the abstraction.

### Do not implement every fusion pattern

Start with generic fusion + a few high-value patterns.

### Do not implement Megatron/DeepSpeed

Implement primitives.

### Do not build a distributed scheduler/server

Start with process groups.

### Do not make every dtype universal

Use backend capabilities.

### Do not make TP/DP/PP/EP part of the core Graph IR

They should be partitioning strategies.

### Do not force users to express everything as ops

Custom Kernel DSL remains a first-class escape hatch.

---

# 21. Final Architecture

The intended long-term system is:

```text
                         Vernon Python API
                                │
                                ▼
                       ┌──────────────────┐
                       │   Graph IR       │
                       │                  │
                       │ builtin ops      │
                       │ custom kernels   │
                       └────────┬─────────┘
                                │
                ┌───────────────┼────────────────┐
                │               │                │
                ▼               ▼                ▼
             Autodiff        Fusion          Placement
                │               │                │
                │               ▼                │
                │        Fusion Groups            │
                │               │                │
                └───────────────┼────────────────┘
                                ▼
                       Optimized Graph
                                │
                                ▼
                       Distributed Graph
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
              Local Graph            Communication
                    │                       │
                    └───────────┬───────────┘
                                ▼
                         Kernel Generation
                                │
                                ▼
                           Kernel DSL
                                │
                         Kernel Optimizer
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
             CPU              CUDA           SPIR-V/Metal
              │                 │                 │
              └─────────────────┼─────────────────┘
                                ▼
                           Device Runtime
```

## 22. The Core Design Principle

The entire system should ultimately revolve around four independent abstractions:

```text
WHAT
Graph IR
    ↓
WHAT CAN BE FUSED
Fusion / Optimization
    ↓
HOW
Kernel DSL
    ↓
WHERE
Placement / Distributed Runtime
```

And a fifth layer decides:

```text
WHEN
Scheduler
```

So the conceptual model becomes:

```text
                WHAT
                 │
              Graph IR
                 │
        ┌────────┴────────┐
        │                 │
      FUSION           PLACEMENT
        │                 │
        └────────┬────────┘
                 │
              SCHEDULE
                 │
                 ▼
          Kernel Generation
                 │
                 ▼
            Kernel DSL
                 │
                 ▼
             Backend
```

This architecture preserves what Vernon already does well:

> **A single device-independent computational graph can contain ordinary compute, arbitrary custom kernels, graphics-style kernels, and eventually distributed computation.**

Fusion does not replace Kernel DSL.

Distributed execution does not replace Graph IR.

And Kernel DSL does not need to understand whether a kernel came from:

```text
physics
graphics
ML
fusion
user code
```

They all ultimately become executable graph nodes with backend-specific lowering.

---

# 23. Definition of "Done"

The next major version can be considered architecturally complete when the following program works:

```python
class Model(vd.Module):

    def forward(self, x):
        x = vd.matmul(x, self.w1)
        x = vd.add(x, self.bias)
        x = vd.relu(x)

        x = custom_kernel(x)

        return vd.matmul(x, self.w2)
```

and Vernon can automatically produce:

```text
Graph
 │
 ├── [Fused MatMul + Add + ReLU]
 │
 ├── [User Kernel]
 │
 └── [MatMul]
       │
       ▼
Distributed placement
       │
       ▼
Kernel DSL
       │
       ▼
CUDA / Metal / SPIR-V / CPU
```

while the same infrastructure can eventually execute:

```text
Game graphics
Physics simulation
Neural network training
LLM inference
Distributed LLM training
```

without introducing separate execution models for each domain.
