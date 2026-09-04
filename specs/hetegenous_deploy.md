# DeepSeek-V4-Pro 家庭异构推理集群方案

> 目标：在约 **¥50,000** 新增预算下，利用现有设备构建尽可能高性能的 DeepSeek-V4-Pro 推理集群。
>
> 核心目标不是单纯堆 GPU，而是验证 Vernon 的：
>
> * Graph partitioning
> * Pipeline Parallelism (PP)
> * Expert Parallelism (EP)
> * topology-aware placement
> * hierarchical memory
> * expert caching
> * prefetch
> * communication scheduling
>
> **模型假设：DeepSeek-V4-Pro，约 1.6T total parameters / 49B active parameters，MoE。**

---

# 1. 已有硬件

| Node   | CPU / SoC         | Accelerator |               Memory |  Storage | Network    |
| ------ | ----------------- | ----------- | -------------------: | -------: | ---------- |
| Node A | AMD Ryzen 9 9950X | RTX 4090    | 32GB RAM + 24GB VRAM | 2TB NVMe | 当前 100Mbps |
| Node B | Apple M5          | M5 GPU      |  32GB unified memory |  2TB SSD | 当前 100Mbps |

已有资源：

```text
GPU / Unified Memory

RTX 4090      24 GB
M5 UMA        32 GB
-------------------
              56 GB
```

System memory：

```text
4090 node     32 GB
M5 node       32 GB
-------------------
              64 GB
```

因此现有机器不足以直接容纳约 865GB 量级的 V4-Pro checkpoint，需要构建 hierarchical memory。

---

# 2. 预算原则

## 不优先购买 RTX 6000 Ada

RTX 6000 Ada 的主要优势：

```text
48GB VRAM
高显存带宽
ECC
```

但单位成本太高。

对于本项目：

```text
1 × RTX 6000 Ada
```

只增加约：

```text
48GB accelerator memory
```

无法从根本上解决 1.6T 模型的存储问题。

因此优先级应该是：

```text
10GbE
  ↓
大 RAM
  ↓
NVMe
  ↓
GPU
```

而不是：

```text
GPU
GPU
GPU
GPU
```

---

# 3. 推荐新增硬件

## Node C：高内存 GPU Node

目标：

```text
RTX 5090
+
128GB~192GB RAM
+
NVMe
+
10GbE
```

推荐：

```text
CPU:
    Ryzen 9 / Threadripper / 类似高 PCIe lane CPU

GPU:
    RTX 5090 32GB

RAM:
    ≥128GB
    理想 192GB

Storage:
    4TB NVMe

Network:
    10GbE
```

用途：

```text
PP Stage
+
local expert cache
+
prefetch
```

---

# 4. Node D：第二个高内存节点

如果预算允许：

```text
GPU:
    RTX 5090 32GB
```

或者根据实际价格选择：

```text
RTX 4090
二手高显存 NVIDIA GPU
其他具有 CUDA 支持的 GPU
```

重点仍然是：

```text
RAM ≥128GB
NVMe ≥4TB
10GbE
```

如果预算不足以同时购买第二张高端 GPU：

> 优先保留大 RAM 和 10GbE，把第二个节点 GPU 降级。

---

# 5. 网络：必须升级

现有：

```text
100Mbps
```

理论吞吐：

```text
100 Mbps / 8
≈ 12.5 MB/s
```

对于 distributed inference 太慢。

必须升级：

```text
10GbE
```

理论吞吐：

```text
10 Gbps / 8
≈ 1.25 GB/s
```

推荐拓扑：

```text
                 10GbE Switch
                       │
          ┌────────────┼────────────┐
          │            │            │
        Node A       Node B       Node C
        4090           M5          5090
          │            │            │
        10GbE         10GbE        10GbE
```

所有机器使用有线连接。

不要使用 Wi-Fi。

---

# 6. 最终硬件拓扑

推荐目标：

```text
                           10GbE
                             │
                    ┌────────┴────────┐
                    │   10GbE Switch  │
                    └────────┬────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
       ▼                     ▼                     ▼

   Node A                 Node B                 Node C
   9950X                  Apple M5               5090
   RTX 4090               M5 GPU                 32GB VRAM
   24GB VRAM              32GB UMA
   32GB RAM               32GB UMA               128~192GB RAM
   2TB NVMe               2TB SSD                4TB NVMe

       │                     │                     │
       └─────────────────────┴─────────────────────┘
                         10GbE
```

如果预算允许加入 Node D：

```text
                           10GbE
                             │
                     ┌───────┴───────┐
                     │   10GbE SW    │
                     └───────┬───────┘
                             │
       ┌──────────┬──────────┼──────────┬──────────┐
       │          │          │          │
       ▼          ▼          ▼          ▼
      A           B          C          D
    4090         M5        5090       5090
```

---

# 7. Vernon 的 Device Topology

Vernon 不应该把设备简单看成：

```text
GPU0
GPU1
GPU2
GPU3
```

而应该建立 topology graph：

```text
DeviceTopology

Node A
 ├── RTX4090
 ├── RAM 32GB
 └── NVMe 2TB

Node B
 ├── M5 GPU
 ├── Unified Memory 32GB
 └── SSD 2TB

Node C
 ├── RTX5090
 ├── RAM 128~192GB
 └── NVMe 4TB

Network:
 └── 10GbE
```

每条 edge 都具有：

```text
latency
bandwidth
memory capacity
memory bandwidth
```

例如：

```text
RTX5090 ↔ local RAM
    cost = low

RTX5090 ↔ RTX4090
    cost = 10GbE

M5 ↔ RTX5090
    cost = 10GbE
```

---

# 8. Hierarchical Memory

V4-Pro 的权重远大于 GPU memory，因此需要：

```text
                    Model Weights
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          Active Cache          Cold Storage
              │                     │
          ┌───┴───┐                 │
          ▼       ▼                 ▼
         VRAM     RAM              NVMe
```

三个 tier：

```text
Tier 0:
    GPU / Unified Memory

Tier 1:
    System RAM

Tier 2:
    NVMe SSD
```

执行原则：

```text
NVMe
  ↓ prefetch
RAM
  ↓ prefetch
VRAM
  ↓
Compute
```

---

# 9. Graph Partition

DeepSeek-V4-Pro：

```text
~61 transformer layers
```

第一版不要直接尝试：

```text
TP + PP + EP
```

而应该先验证：

```text
PP
+
local expert placement
```

例如：

```text
                    V4-Pro
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
     Stage 0       Stage 1       Stage 2
     Node A        Node C        Node D
     L0~20         L21~40        L41~60
```

M5 可以作为：

```text
auxiliary compute
+
CPU/GPU offload
+
prefetch node
```

而不是强制成为主 PP stage。

原因：

> PP 对低带宽网络更加友好，而 M5 与 NVIDIA GPU 之间不存在高速 GPU interconnect。

---

# 10. 为什么优先 PP

PP 的通信模式：

```text
Stage 0
   │
   │ activation
   ▼
Stage 1
   │
   │ activation
   ▼
Stage 2
```

通信次数相对少。

而 TP：

```text
Layer
 ├── GPU0
 ├── GPU1
 ├── GPU2
 └── GPU3
       ↕
     synchronization
```

几乎每个 layer 都可能需要通信。

因此：

```text
10GbE:
    PP 适合
    TP 不理想
```

---

# 11. Prefill Strategy

Prefill 是最适合 pipeline 的阶段。

例如：

```text
prompt
    ↓
tokens
    ↓
micro-batches
```

把 token 分成：

```text
MB0
MB1
MB2
MB3
...
```

Pipeline：

```text
              Stage 0     Stage 1     Stage 2

MB0           ████████
                         ████████
                                    ████████

MB1                    ████████
                                    ████████
                                               ████████

MB2                               ████████
                                          ████████
```

填充 pipeline 后：

```text
Stage 0: █ █ █ █ █
Stage 1:   █ █ █ █ █
Stage 2:     █ █ █ █ █
```

因此：

> Prefill 可以通过 microbatch 尽可能隐藏 PP communication latency。

---

# 12. Prefill + Expert Parallelism

Prefill 有大量 token：

```text
batch × sequence
```

因此可以形成比较大的 expert batch。

推荐：

```text
PP
 +
local EP
```

而不是全局 EP。

例如：

```text
Stage 0
 ├── Router
 └── local experts

Stage 1
 ├── Router
 └── local experts

Stage 2
 ├── Router
 └── local experts
```

尽量避免：

```text
Stage 0
   │
   ├────────→ Stage 1 expert
   ├────────→ Stage 2 expert
   └────────→ Stage 3 expert
```

形成跨节点 All-to-All。

---

# 13. Expert Placement

不要简单：

```text
Node A:
    experts 0~127

Node B:
    experts 128~255

Node C:
    experts 256~383
```

更理想的是：

```text
routing statistics
       ↓
expert popularity
       ↓
expert placement
```

例如：

```text
Node A:
    hot experts
    + local experts

Node B:
    hot experts
    + local experts

Node C:
    hot experts
    + local experts
```

---

# 14. Hot Expert Replication

如果某些 expert 被频繁访问：

```text
Expert 17
```

可以复制：

```text
Node A ── Expert 17
Node B ── Expert 17
Node C ── Expert 17
```

而不是：

```text
Node A
   │
   │ network
   ▼
Node C
   │
Expert 17
```

trade-off：

```text
Memory ↑
Network traffic ↓↓↓
```

对于家庭网络环境，这是非常值得考虑的。

---

# 15. Expert Cache

每个 node：

```text
NVMe
  ↓
RAM
  ↓
GPU
```

维护：

```text
Expert Cache
```

例如：

```text
VRAM:
    hot experts

RAM:
    warm experts

NVMe:
    cold experts
```

---

# 16. Expert Prefetch

理想执行：

```text
Router
   │
   ▼
expert IDs
   │
   ├──────────────┐
   │              │
   ▼              ▼
compute current   prefetch next
expert            expert
   │              │
   │              ▼
   │             RAM
   │              │
   │              ▼
   │             VRAM
   │
   ▼
next token
```

目标：

```text
I/O latency
    ≈
hidden behind
computation
```

---

# 17. Decode Strategy

Decode 与 Prefill 完全不同。

Decode：

```text
token t
  ↓
token t+1
  ↓
token t+2
```

每个 token 都依赖前一个 token。

因此 pipeline bubble 很难完全隐藏。

---

## Decode 第一优先级

```text
locality
```

而不是：

```text
maximum parallelism
```

目标：

```text
token
 ↓
local router
 ↓
local expert
 ↓
local compute
```

尽可能避免：

```text
token
 ↓
router
 ↓
10GbE
 ↓
remote expert
```

---

# 18. Decode 中减少 EP

Prefill：

```text
PP + local EP
```

Decode：

```text
PP
+
expert replication
+
expert cache
+
local routing
```

如果某个 expert 不在本地：

```text
remote expert
```

才触发 network。

---

# 19. Decode Expert Cache

维护：

```text
Hot Expert Cache
```

例如：

```text
VRAM:

Expert 17
Expert 82
Expert 201
Expert 301
...

RAM:

next-level experts

NVMe:

all remaining experts
```

然后：

```text
previous routing statistics
        ↓
predict likely experts
        ↓
prefetch
        ↓
next token
```

---

# 20. Communication Policy

Vernon scheduler 应区分：

### Local

```text
GPU ↔ GPU
GPU ↔ RAM
```

### Intra-node

```text
PCIe
```

### Inter-node

```text
10GbE
```

并使用不同 cost。

例如：

```text
cost(local)
    << cost(PCIe)
    << cost(10GbE)
```

因此 planner 应自动倾向：

```text
local compute
```

而不是：

```text
remote compute
```

---

# 21. All-to-All 避免策略

第一版：

```text
NO global EP
```

采用：

```text
PP
+
local EP
+
expert replication
+
expert cache
```

只在必要时：

```text
remote expert dispatch
```

并且：

```text
batch tokens
```

而不是：

```text
one token → one network request
```

正确：

```text
tokens
 ↓
router
 ↓
group by expert
 ↓
pack
 ↓
single communication
 ↓
expert batch
```

---

# 22. Prefill 最终执行计划

```text
Prompt
  │
  ▼
Tokenize
  │
  ▼
Microbatch
  │
  ▼
┌───────────────────────────────────────┐
│ Pipeline                              │
│                                       │
│ Stage 0 → Stage 1 → Stage 2           │
│   │          │          │              │
│ local EP   local EP   local EP        │
│   │          │          │              │
│ expert      expert     expert          │
│ cache       cache      cache           │
└───────────────────────────────────────┘
  │
  ▼
KV Cache
```

重点：

```text
large batch
+
pipeline
+
local expert batching
+
prefetch
```

---

# 23. Decode 最终执行计划

```text
KV Cache
   │
   ▼
Next Token
   │
   ▼
Router
   │
   ▼
Expert IDs
   │
   ├──── local → execute
   │
   └──── remote
          │
          ▼
      batched dispatch
          │
          ▼
       execute
          │
          ▼
       combine
          │
          ▼
      next token
```

同时：

```text
current compute
      │
      └──────────→ prefetch next experts
```

---

# 24. Vernon Planner 的目标

最终不应该让用户手写：

```text
PP=3
EP=2
TP=1
```

而应该提供：

```text
DeviceTopology
MemoryTopology
Graph
CostModel
```

然后：

```text
                   Graph IR
                      │
                      ▼
              Device Topology
                      │
                      ▼
                 Cost Model
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Partition   Placement   Fusion
          │           │           │
          └───────────┼───────────┘
                      ▼
               Prefetch Planner
                      │
                      ▼
              Communication Plan
                      │
                      ▼
                  Runtime
```

---

# 25. 第一阶段实现顺序

不要一开始实现完整：

```text
TP + PP + EP + expert caching + prefetch
```

推荐顺序：

### Phase 1 — 10GbE

```text
所有机器互联
```

测量：

```text
latency
bandwidth
```

---

### Phase 2 — PP

```text
61 layers
→
3 stages
```

验证：

```text
activation transport
```

---

### Phase 3 — Prefill Pipeline

加入：

```text
microbatch
```

测量：

```text
pipeline utilization
bubble
communication overlap
```

---

### Phase 4 — Expert Placement

加入：

```text
router
expert locality
```

---

### Phase 5 — Expert Cache

```text
VRAM
 ↓
RAM
 ↓
NVMe
```

---

### Phase 6 — Prefetch

```text
current expert compute
        +
next expert I/O
```

测量：

```text
I/O hidden %
```

---

### Phase 7 — Hot Expert Replication

根据真实 routing statistics：

```text
hot experts
 ↓
replicate
```

---

### Phase 8 — Local EP

最后才加入：

```text
local EP
```

避免一开始就引入 global All-to-All。

---

# 26. 最终目标

最终的 Vernon execution plan 应类似：

```text
                    DeepSeek-V4-Pro
                           │
                           ▼
                     Graph IR
                           │
                           ▼
                  Topology Analyzer
                           │
                           ▼
              ┌────────────────────────┐
              │ Partition Planner      │
              │                        │
              │ PP                     │
              │ Local EP               │
              │ Expert Placement       │
              │ Expert Replication     │
              └───────────┬────────────┘
                          │
                          ▼
                  Memory Planner
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
             VRAM        RAM         NVMe
              │           │           │
              └───────────┼───────────┘
                          ▼
                  Prefetch Planner
                          │
                          ▼
               Communication Planner
                          │
                          ▼
                    10GbE Network
                          │
                          ▼
                       Runtime
```

---

# 27. 核心原则

整个系统最终遵循四条原则：

```text
1. Compute locality
   尽量在本地执行。

2. Communication minimization
   尽量避免跨机器 All-to-All。

3. Memory hierarchy
   VRAM → RAM → NVMe。

4. Latency hiding
   Prefetch / pipeline / batching
   尽可能把通信和 I/O 隐藏在计算之后。
```

最终不是：

> “用很多消费级 GPU 模拟 H100 cluster。”

而是：

> **利用 topology-aware compiler/runtime，把大量异构、低带宽设备组织成一个 hierarchical compute + memory system。**

这也是这个实验对 Vernon 最有价值的地方。
