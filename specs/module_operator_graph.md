# Module、Operation Graph 与 Program Asset 规范

## 1. 目标

VernonDSL 应允许用户用普通 Python 结构组织可 Cook 的程序：

```python
class Simulation(vd.Module):
    def forward(self, position, velocity, dt):
        for _ in range(STEPS):
            velocity = update_velocity(position, velocity, dt)
            position = integrate(position, velocity, dt)
        return position, velocity
```

用户不需要：

- 为每个 kernel 或 graphics pipeline 手工定义 Operator 包装；
- 为普通 graphics invocation 手工派生 `ComputePass` 或 `RenderPass`；
- 把 `if`、`elif`、`else`、`for`、`while` 改写成专用 graph-building API；
- 手工插入反向图中的梯度累加 Add；
- 预先注册 `AdvectOperator`、`PressureSolveOperator`、`ShadowPassOperator` 等固定领域算子类型。

Frontend 必须先形成可优化的 structured Operation Graph，再执行 AD、fusion、实现选择和 kernel
partition。Kernel、ComputePass 和 RenderPass 是 lowering 结果，不是用户组织程序的主要抽象。

## 2. 非目标

本规范不要求：

- 把 graphics differentiation 纳入 active VJP；
- 自动微分 opaque 外部 binary；
- 把一次 invocation 的具体帧资源作为 Program-owned object Cook 进 asset；
- 把 C++ ExecutionGraph 序列化为 Program Asset；
- 在当前发布前修改既有 compiler contract 或 pipeline contract。

## 3. 分层模型

系统采用六层单向 lowering：

```text
Python Module
    ↓
Typed Structured Operation Graph
    ↓
Operator-level AD and graph optimization
    ↓
Kernel partition and implementation mapping
    ↓
ComputePass / RenderPass fragments
    ↓
ExecutionGraph / Command DAG / RHI
```

### 3.1 Module

`Module` 是用户可见的程序组织单位，负责：

- 定义 `forward()` 入口；
- 组织子 Module、kernel 和 graphics pipeline；
- 保存 Cook-time 配置和 typed defines；
- 暴露 Program 的公开输入与输出；
- 作为 Cook、load 和 invoke 的逻辑单位。

`Module` 不拥有具体帧资源，不负责 queue、barrier、clear/load/store 或 command submission。

`Module.__init__()` 可以保存 typed/static 配置并注册子 Module。子 Module 通过属性赋值获得稳定 qualified
name，配置进入 graph/cache identity。具体 `TensorStorage`、`TensorView` 和每帧资源不得注册为持久
Module 属性；invocation-private 中间资源由 `forward()` 内的 `empty()`、`empty_like()`、`zeros()`
或 `zeros_like()` 创建。

普通顶层调用只执行 primal，不构建 reverse variant，也不分配或保留 tape：

```python
outputs = module(inputs)
```

需要微分时必须显式构造 VJP variant：

```python
outputs, pullback = vd.ad.vjp(
    module,
    wrt=("input",),
    outputs=("result",),
)(inputs)
gradients = pullback(cotangents)
```

`pullback` 独占该次 invocation 的 reverse program、compiled checkpoint plan 和 tape；多个尚未调用的
pullback 必须互不覆盖。子 Module 调用继续 capture 到同一个 Program Operation Graph，而不是嵌套创建
独立 submission。

### 3.2 Program

Cook 后的 Module 成为一个 Program。Program 可以内部包含零个、一个或多个 kernel，也可以包含
graphics pipeline invocation。

对 C++ Runtime 和 ExecutionGraph，Program 暴露统一 contract：

```text
Program
- public value/resource signature
- aggregate effects
- typed variant key
- forward entry
- optional pullback entry
- internal execution fragment
```

一个多 kernel Program 对外仍是一个逻辑 FusedOp/ProgramInvocation；“fused”表示逻辑调度边界，
不保证只有一个 GPU dispatch。

### 3.3 Operation Graph

Operation Graph 是编译器内部语义边界，不是用户继承的 Python `Operator` 类。

它至少包含：

- primitive value operations；
- storage load/store 和 resource effects；
- `KernelCallOp`；
- `GraphicsPipelineCallOp`；
- Module/function call；
- structured `IfOp`、`ForOp`、`WhileOp`；
- tuple/struct/tensor value；
- return/yield 和 loop-carried values。

由于存在控制流，最终表示必须支持 SSA Value、Operation、Region 和 Block；简单的线性
`OperatorDag` 不能作为最终模型。

## 4. 用户编程模型

### 4.1 普通控制流

用户继续写普通控制流：

```python
def forward(self, state):
    if MODE == 0:
        state = method_a(state)
    elif MODE == 1:
        state = method_b(state)
    else:
        state = method_c(state)

    while state.error > tolerance:
        state = iterate(state)
    return state
```

Compile-time define 控制的分支在 type/effect inference 前 specialization。Runtime 条件 lower
为 structured control flow。Reverse-mode AD 使用已有 control-history/Tape 机制记录不能安全重算的
predicate、迭代次数和退出轨迹。

### 4.2 Module 输入

`forward()` 参数定义 Program 的公开输入。所有输入都通过统一的 Program Value binding 进入
invocation；不存在额外的 `structural_inputs`、`configure()` 或 `graphics_config` 路径。输入包括：

- Tensor/Value、TensorView、uniform、普通 scalar 和显式 RawBuffer；
- Texture/ImageView、Sampler；
- source/Program IR 的 `RenderTarget`、`Attachment` 和 `RenderState` typed builtin。

这些 graphics builtin 是 source convenience 和 IR 语义，不是最终 public resource-aggregate ABI。
Final manifest 必须把它们规范化到每个 graphics node 的 `attachments`、`state` 和 `draw` 字段，并把
shader-visible Texture/ImageView/typed byte storage/Sampler 作为独立 Program Values 绑定。Shader resource
不得嵌套在 ABI-stable struct 中，也不得依赖 host struct layout。

ImageView 只是在 source/runtime 层携带 descriptor 的 Texture alias，不新增 Program dialect type。
Final Program 中 root 与 view 都使用 `!vernon.texture`；view Origin 保存 parent version 与
aspect/mip/layer descriptor。Sampled Texture 的 dialect format 必须是 `"unknown"`，具体 allocation
format 与 extent 由 image StorageDescriptor 表达；只有 storage Texture 的
`read`/`write`/`read_write` access
携带具体 format。

Graphics calls do not return replacement Texture objects. Deployment lowering
assigns one Storage identity per resource and explicit internal value versions
per use.

Module 属性只应用于：

- 子 Module；
- kernel/pipeline program；
- typed define 和 Cook-time 配置；
- 明确注册的 Program constant。

具体 mesh、camera、simulation state、RenderTarget 和每帧资源不得因为赋值给 `self` 而隐式
固化进 asset。

### 4.3 Kernel 和 pipeline 调用

Module 中直接调用 Vernon kernel，Frontend 自动为每次调用形成 `KernelCallOp`；用户不定义
Operator 包装：

```python
@vd.kernel
def integrate(position, velocity, dt):
    ...


class Step(vd.Module):
    def forward(self, position, velocity, dt):
        integrate(position, velocity, dt)
```

Module 中直接调用 graphics pipeline，Frontend 自动形成 `GraphicsPipelineCallOp`。Module 可以包含
多个 compute/graphics invocation，并保留它们的程序顺序、数据依赖和 resource effects。

这些节点由被调用的 kernel/pipeline 及其 specialization 标识，不属于预先固定的领域 Operator
taxonomy。用户可以用普通函数或子 Module 组织 advection、pressure solve、shadow rendering 等领域
逻辑，但 compiler 看到的仍是可组合、可分析和可优化的 primitive/call nodes：

```python
class FluidStep(vd.Module):
    def forward(self, state, dt):
        advect_kernel(state.velocity, state.velocity_next, dt)
        pressure_kernel(state.velocity_next, state.pressure)
        project_kernel(state.velocity_next, state.pressure)
        return state


class Pbr(vd.Module):
    def forward(self, scene):
        shadow_pipeline(...)
        pbr_pipeline(...)
```

显式 kernel 是默认 optimization boundary。普通 primitive 运算和透明 helper 保持可 inline、可 fusion。

## 5. Typed compile-time defines 与 variants

### 5.1 定义

Compile-time specialization 使用 typed define：

```python
SHADOW = vd.define("SHADOW", bool, default=False)
QUALITY = vd.define("QUALITY", vd.i32, default=2)
MSAA = vd.define("MSAA", vd.i32, default=1)
```

`feature("NAME")` 最终定义为 boolean `define` 的兼容语法，不再拥有独立语义。

Define：

- 不是 runtime binding；
- 具有稳定名称、类型和 default；
- 在普通 `if`、`elif`、`else`、`for` 和常量表达式中使用；
- 自动传播到 Module、子 Module、kernel、shader 和 helper specialization；
- 进入 cache identity 和下一版 asset variant key；
- 不允许程序运行期间修改。

### 5.2 Variant

Variant 是 define environment：

```python
vd.variant(
    SHADOW=True,
    QUALITY=3,
    MSAA=4,
)
```

Specialization 顺序必须是：

```text
bind defines
→ eliminate compile-time control flow
→ determine active parameters
→ type/effect inference
→ build Operation Graph
```

Module compiler 从 specialization 后实际保留的程序推导 variant signature。用户不需要把普通
`if/elif/else` 手工翻译为 `When`。

`When[predicate, T]` 只用于必须显式声明 conditional ABI 的 stage 或外部接口。Runtime control flow
不改变 binding schema。

### 5.3 Contract rollout

当前发布线不修改 compiler/pipeline contract：

1. 先实现内部 typed define、specialization environment 和 Module frontend；
2. 当前 artifact 继续使用现有 contract；
3. 下一次版本化 contract 才加入 typed variant map 和 C++ typed variant selection；
4. 禁止把整数 define 临时编码成 `"NAME=value"` feature 字符串。

## 6. Graphics 与 RenderPass 边界

### 6.1 Graphics 在 active VJP 之外

`GraphicsPipelineCallOp` 是 forward-only effectful operation。若 requested gradient path
穿过 graphics result，compiler 必须给出明确 unsupported diagnostic；graphics 在 AD 计算之后仅用于
显示或输出时不阻塞其他 differentiable subgraphs。

### 6.2 Graphics builtin 的规范化边界

Source 和 Program IR 可使用：

```text
Attachment
= image view/use
+ load: load | clear | discard
+ clear value when load == clear
+ store: store | discard
+ optional resolve target

RenderState
= blend
+ depth/stencil
+ raster
+ color write mask
+ multisample
```

Clear 属于 `Attachment`，不属于 `RenderState`。`RenderTarget` 只是 source convenience：它可以在
内部创建 Texture 并聚合 color/depth attachments，但 final manifest 不把 `RenderTarget` 或其他
resource aggregate 暴露为 public ABI。它只保留规范化后的 graphics node：

Resolve destination 的 load 隐式为 discard；multisample source 与 resolve destination 的 store
policy 相互独立。

```text
graphics node
- attachments
- state
- draw
```

`draw` 包含 optional `index_buffer: TensorView`、默认值为 1 的 `instance_count`，以及 direct draw
必须显式提供的 `vertex_count`；deployment 不从 vertex inputs 推导 draw count。不存在 `DrawArgs`。Primitive topology 是
pipeline/artifact 的 static specialization，不是 runtime draw value。

### 6.3 Attachment continuity 与 native render-pass fusion

Attachment use 同时表达 view/resource identity 和本次使用的 load/store/clear/resolve 语义。
ExecutionGraph 根据 parent image、aspect/mip/layer、attachment geometry、前后 use continuity、
load/store/resolve compatibility 和中间 RAW/WAR/WAW hazard 决定是否形成同一个 native render
pass。中间 discard、被 discard 后的 load，或 attachment-to-sampling hazard 必须切断 fusion。

Fusion 不要求相邻 draw 的 `RenderState` 相等；blend、depth/stencil、raster、write mask 和
multisample state 可在同一 native render pass 内按 draw 改变，只要 backend capability 允许且
attachment continuity/hazard 条件成立。

### 6.4 不要求用户派生 Pass

普通调用应允许：

```python
invocation = module.bind(
    scene=scene,
    shadow_depth=shadow_depth,
    color=color,
    depth=depth,
)
graph.invoke(invocation)
```

所有 Program Values，包括 attachment views，使用同一个 bind 路径。内部可以使用
invocation-backed RenderPass adapter 复用已发布的 C++ contract；手写 `RenderPass` 仅保留为插入
自定义 encoder command 的低层 escape hatch，不构成 target public invocation ABI。

## 7. Operator-level AD

### 7.1 AD 发生在 kernel partition 之前

AD pipeline：

```text
typed primal Operation Graph
→ operator-level reverse graph
→ residual/control planning
→ primal and pullback optimization
→ kernel partition
→ backend lowering
```

不得先把整个 Module 固化为一组 opaque kernels，再尝试恢复 operator semantics。

### 7.2 Primitive VJP

基础 Operation 具有内部 VJP/JVP rule：

- arithmetic；
- elementwise；
- reduction；
- gather/scatter；
- sampling；
- reshape/view；
- storage operations 中可合法微分的子集。

领域逻辑可以直接调用一个或多个 kernel，也可以由 primitive、普通函数和子 Module 组合。它们不会因为
名称或用途被固化为 `AdvectOperator`、`PressureSolveOperator` 等内建领域节点；每次 kernel 和
graphics pipeline 调用本身自动成为 Operation Graph node。

### 7.3 梯度累加 Add

Reverse graph 构造遇到 fan-in 时自动插入语义级 `AddOp`/accumulation operation：

```text
cotangent A ─┐
             AddOp → accumulated cotangent
cotangent B ─┘
```

随后 optimizer 可以：

- 合并多级 Add；
- 与生产者/消费者 fusion；
- 选择 deterministic reduction；
- 选择 in-place、tree reduction 或 staged reduction；
- 映射到 CPU implementation 或 GPU compute kernel；
- 与相邻 backward operations 一起 partition。

`AddOp` 由 Python/compiler 的 reverse graph transform 生成，和其他 Operation 一样经过统一
implementation mapping。CPU、Metal、Vulkan、CUDA 等只是同一个 generated Add kernel 的不同 backend；
不得在 graph semantic 或 lowering API 中增加 `allow_cpu`、`device_add` 等 backend 分支。

`GraphAutodiffValue::add()`、materialization-time gradient expression 和 Add-only `OperatorDag`
已删除。低层 `ExecutionGraph` 若收到未经过 Program transform 的 cotangent fan-in 会直接拒绝，
避免重新引入隐式累积语义或维护两套 operator contract。

C++ `ExecutionGraph` 只接收 Program lowering 后的 compute/render fragments，不暴露 `add()`、
`lowerAdd()` 或其他按具体数学 Operation 命名的 graph-autodiff virtual method。它可以调度由 `AddOp`
生成的普通 ComputePass，但不负责发现 cotangent fan-in 或构造 Add。

### 7.4 Custom kernel AD

透明 Vernon `KernelCallOp` 可以调用 kernel-level AD 生成 forward/backward kernels，并把生成的
pullback 作为该 Operation 的 lowering。

Opaque external kernel 必须：

- 注册显式 VJP；
- 或标记 non-differentiable。

多 kernel custom Program 的 differentiation contract 位于 Program/Operation Graph 层，不能只对其中
一个 kernel 求导。

## 8. Optimization 与 lowering

### 8.1 Operation Graph 优化

Kernel partition 前至少允许：

- constant folding 和 define specialization；
- dead branch/dead operation elimination；
- common subexpression elimination；
- algebraic simplification；
- elementwise/reduction fusion；
- layout and shape propagation；
- alias/effect-aware scheduling；
- rematerialization/checkpoint selection；
- forward/backward joint optimization。

### 8.2 Implementation mapping

一个 Operation 可以映射为：

- inline IR；
- 一个 generated compute kernel；
- 多个 compute kernels；
- external library call；
- graphics pipeline call；
- CPU implementation；
- explicit unsupported diagnostic。

Mapping 必须基于 target capabilities、dtype/layout、alias/effects 和 planning policy；不能静默 host
fallback 或改变数值语义。

### 8.3 Pass lowering

Lowering 生成：

- ComputePass fragments；
- RenderPass fragments；
- normalized resource effects；
- internal transient resources；
- Command DAG dependencies。

C++ ExecutionGraph 调度 ProgramInvocation 的最终 execution fragment。它不重新执行 operator-level AD，
但可以展开内部 pass 以完成 hazard analysis、barrier、render fusion 和 submission coalescing。

## 9. Program Asset 与 C++ 调用

Cooked Program Asset 包含：

- public signatures；
- define declarations 和版本化 variant metadata；
- optimized Operation Graph 或其稳定序列化表示；
- selected implementation/fusion plan；
- content-addressed StageArtifacts：exact target、runtime requirements、
  authenticated code Blob ranges 和 entry points；
- internal execution fragment；
- optional forward/pullback entries。

不包含具体帧资源和外层 ExecutionGraph declaration。
Program 不序列化第二份 node dependency list；ResolveProgram 从 Value SSA、
ResourceAccess version chain 与 view overlap 推导完整 producer/RAW/WAR/WAW DAG。
每个 Storage 的 root versions 必须形成 non-branching chain。

C++ 目标接口：

```cpp
auto program = runtime.loadProgram("simulation.asset", defines);
auto invocation = program.bind(values);
graph.invoke(invocation);
```

Graphics invocation 与 compute 一样只绑定 Program Values：

```cpp
auto invocation = program.bind(values);
graph.invoke(invocation);
```

Runtime 从 final manifest 的 graphics-node `attachments`、`state` 和 `draw` 字段实例化执行节点；
不需要第二个 attachment declaration/configuration API。

## 10. 核心不变量

- 用户 Module 不需要定义 Operator 或 Pass 包装类。
- 普通控制流在 source 保持普通控制流。
- Operation Graph 在 kernel partition 前保留足够的语义用于 AD、fusion 和 mapping。
- 显式 kernel 是可分析的 call operation，但默认是 fusion boundary。
- Graphics 使用统一 Program Value binding；不存在 structural/configure 双路径。
- Attachment 拥有 view/use、load/clear/store/resolve；clear 不属于 RenderState。
- RenderTarget 是 source convenience，不是 final manifest 的 public resource aggregate。
- Topology 是 static specialization；Draw 不使用 `DrawArgs`。
- Shader resources 独立绑定，不得嵌套在 ABI-stable struct 中。
- Runtime 条件不改变 ABI；compile-time define variant 可以改变 active signature。
- AD fan-in 以语义级 accumulation operation 表达。
- 不存在 Operator-level Add 与 materialization-time Add 两套长期实现。
- Program Asset 与 ExecutionGraph 分离；Program 可被多个 C++ graphs 复用。
- Unsupported layout、dtype、backend capability 或 differentiation path 必须 fail closed。

## 11. 分阶段实施

### Phase 1：Module 与 declarative graphics vertical slice

- 增加用户可见 `Module` 和 `forward()` 调用；
- capture 多个 pipeline invocations；
- 用统一 `bind(values)` 规范化并调用 graphics Program；
- 迁移 `examples/pbr.py`，删除用户定义的普通 `RenderPass` subclasses；
- 保留低层 Pass API 作为 custom encoder escape hatch。

### Phase 2：Typed define internal foundation

- 增加 `define()`、type/default validation 和 specialization environment；
- 支持 define 驱动的普通 `if/elif/else` 和静态循环；
- 统一现有 boolean Feature 的内部语义；
- 不修改当前 serialized contract。

### Phase 3：Structured Operation Graph

- 用 SSA Value/Operation/Region/Block 取代 Add-only physical `OperatorDag`；
- 引入 primitive、KernelCall、GraphicsPipelineCall 和 structured control operations；
- 先完成 compute-only `Module` vertical slice、子 Module qualified naming 和 invocation-private transient；
- 每次 resource write 产生 `ResourceVersion`，reverse AD 按版本化 value 累积 cotangent；
- 完成 shape/type/effect/alias verification；
- 建立 optimization 和 implementation mapping 接口。

### Phase 4：Operator-level AD

- 从 primitive VJP 构建 reverse Operation Graph；
- 自动插入 accumulation AddOp；
- Module backward 只执行已经生成的 reverse Operation Graph，不调用 C++ graph-level Add；
- 接入 control-history/Tape、rematerialization 和 checkpoint planning；
- 透明 custom kernel 接入 kernel-level AD；
- materialization-time Add hack 和过渡 OperatorDag 已从 runtime、Python binding 与 build target 删除。

### Phase 5：Kernel partition 与 Program lowering

- 完成 forward/backward joint fusion；
- 将 optimized graph partition 到 compute kernels 和 graphics calls；
- lower 为 Program execution fragment 与现有 ExecutionGraph/Command DAG；
- 验证 CPU、Metal、Vulkan，并在对应机器验证 CUDA、DirectX、OpenGL。

### Phase 6：下一版 asset contract

- 版本化 typed define map；
- 序列化 Module Program metadata、variant signature 和 internal graph/fragment；
- 增加 C++ typed variant selection、统一 Program load/bind/invoke；
- 仅在正式 contract version 切换时删除旧 serialized representation。

## 12. 验收标准

- PBR 示例不再要求用户定义 `declare()/execute()` Pass subclasses。
- 相同 Module 能 Cook 多个 typed define variants，且 active signature 正确。
- Module 内普通 `if/elif/else/for/while` 保持源码语法并正确 specialization/lowering。
- 两个以上 kernel 的 Program 可作为一个逻辑 invocation 被 C++ ExecutionGraph 调度。
- AD fan-in 生成 Operation Graph AddOp，并可与相邻 backward work fusion。
- CPU/GPU gradient、failure transaction、resource lifetime 和 deterministic tests 全部通过。
- Graphics forward-only 路径在无 gradient dependency 时可与 differentiable compute 共存。
- 当前发布线 compiler/pipeline contracts 在版本切换前保持不变。
