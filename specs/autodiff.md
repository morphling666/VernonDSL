# Autodiff design

Implemented behavior and remaining capability boundaries are tracked in
[`autodiff-implementation-status.md`](autodiff-implementation-status.md).

> **Status:** normative design target. The source tree implements the
> declarative `ProgramExpression`/`ProgramTransformSpec` surface, static asset
> parsing, typed straight-line pure-Value `ProgramGraph`, derivative-rule
> validation, bounded tape/reverse planning and identity, analytical CPU
> ProgramGraph pullbacks for Scalar, Tensor, Tuple, and Struct Values, static
> indexing scatter adjoints, bounded branches, pure-Value early returns,
> literal-range loops, and literal-bounded loops with a dynamic leading break,
> plus monomorphized pure-helper inlining with finite-difference validation.
> Deterministic primal, forward-with-tape, and backward symbols and their
> cotangent/gradient/tape ABI plans are reflected by the frontend and reserved
> in the bundle schema. Typed input/storage/output/tape/cotangent/gradient
> resource roles are preserved in differentiated GPU reflection. TensorView
> state remains resource SSA in GPU forward and backward profiles. Reverse
> accumulation uses `vernon.reduce_sum` and `vernon.scatter_add`, with optional
> grid-independent disjoint evidence, then lowers per target to direct, atomic,
> or canonical deterministic serial reduction. Runtime output, tape, and
> cotangent carriers have dynamic `(z,y,x)` physical shape. The invocation grid
> is absent from assets, identities, manifests, and artifacts. These additions
> advance the compiler contract to 10 and the pipeline contract to 13; older
> native components reject them.
> Independent native forward-with-tape and backward MLIR modules compile for
> the supported f32/f64 Scalar, static-Tensor, and TensorView stateful subset.
> CPU uses a direct host Value ABI and currently requires one reflected Value
> leaf per input. CUDA, Vulkan, Metal, OpenGL, and OpenGL ES use explicit
> resources for primal Values and gradients, plus dynamic three-dimensional
> resources for tape leaves, cotangents, and outputs. DirectX uses the same
> lowering but still requires DXC-backed validation.
> The cooker emits deterministic primal, forward-with-tape, and backward
> artifacts for the supported compute VJP subset on CPU, CUDA, Vulkan, Metal,
> OpenGL, OpenGL ES, and DirectX. Unsupported programs are rejected before
> output creation without a primal-only substitute. Runtime
> bundle loading strictly validates transform/profile identities, complete
> variant coverage, profile-stage metadata, and artifact references before
> resolution. Loaded pipelines expose output and gradient metadata from either
> direct host profiles or resource-backed GPU profiles. CPU cooking keeps
> relocatable objects and emits deterministic C
> registration source for primal, forward-with-tape, and backward symbols. The
> initial f32/f64 Scalar and equal-shape static Tensor CPU profiles execute
> through an opaque C pullback handle with reusable C++ RAII ownership and
> NumPy binding. Cooked relocatable objects, generated registration code, C
> Runtime loading, and Python `LoadedPipeline.vjp()` are covered end to end.
> Resource-backed compute profiles resolve independent forward and backward
> pipelines. The immediate C API caches them as a one-node compiled
> ExecutionGraph, so immediate and composed execution share buffer ownership,
> tape retention, hazard planning, and reverse submission. Vulkan and Metal cooked
> GPU VJPs are covered end to end for pure Values, statically indexed
> TensorView mutation with fresh Storage gradients, and injective
> global-invocation gather/scatter dispatches, including pullback use
> after destroying the loaded primal pipeline, interleaved pullbacks with
> independent tape, and concurrent application of distinct pullbacks.
> Cooked CUDA and DirectX acceptance runs when those native backends are
> available; OpenGL/ES acceptance runs when suitable compute contexts are
> available. CUDA still requires hardware execution in CI, and DirectX requires
> DXC-backed Windows validation.
> The CPU reference and native Runtime accept a positive invocation-time 3D
> grid and execute X-fastest. The ProgramGraph only attaches disjoint evidence
> when every active resource access uses the same direct index mapping and
> that mapping contains all three global-invocation axes. CUDA lowers
> conflicting f32 accumulation to atomics; conflicting CUDA f64 gradients and
> portable targets run one canonical X-fastest backward loop for deterministic
> reduction. This serial
> scheduling is currently selected during target-profile emission; scalable
> contribution sorting/segmented reduction and fully backend-owned scheduling
> remain future work. GPU runtime rejects overlapping writable Storage bindings
> before dispatch.
> It evaluates only the selected side of bounded branches. CPU Native profiles
> use region-based forward and backward control flow for bounded branches,
> pure-Value early returns, and literal-bounded loops with a dynamic leading
> `if condition: break` guard. They statically unroll the maximum loop bound and
> lower static Tensor broadcasting through explicit splat/broadcast operations
> with reduction adjoints. C++ `AutodiffGraph` composes multiple cooked GPU VJP
> profiles into an ExecutionGraph forward plan and a reusable reverse plan,
> retains tape resources, inserts resource barriers, routes cotangents, and
> accumulates fan-out contributions. Early returns or dynamic loops with
> Storage effects, general break/continue/return loop control, and graphics
> backward lowering remain unavailable.

This document defines Vernon's first public automatic-differentiation model.
The design uses reverse-mode vector-Jacobian products (VJPs), preserves the
Value/Storage/Resource split, and supports the same authored Kernel or graphics
stages in ordinary and differentiated cooked programs.

## 1. Design principles

- Source `@kernel`, `@vertex`, and `@fragment` signatures never gain a
  `grad_or_not` parameter.
- `vd.ad.vjp(program, ...)` is the only initial public reverse-mode transform.
  `grad`, `value_and_grad`, an implicit Tape, and a combined `apply` shortcut
  are not separate initial APIs.
- `wrt` is the only declaration of differentiated inputs. There is no
  `requires_grad` flag, implicit `.grad` state, global gradient clearing, or
  context exit that silently executes a backward pass.
- One execution returns `(outputs, pullback)`. Applying the pullback to output
  cotangents returns gradients for the declared `wrt` inputs.
- Primal and differentiated artifacts have separate deterministic identities.
  An ordinary primal invocation never pays tape or backward costs.
- Unsupported differentiation is rejected before artifacts or Runtime state
  are produced. The compiler never substitutes a zero gradient silently.

The initial public surface is deliberately VJP-only. JVP, full-Jacobian
materialization, higher-order differentiation, and convenience aliases may be
added later without changing the VJP contract.

## 2. Mathematical contract

For a specialized program

```text
Y = F(X)
```

the pullback computes

```text
dX = J_F(X)^T dY
```

where `dY` has the same differentiable Value/Storage structure as `Y`, and
`dX` has the structure selected by `wrt`.

Autodiff does not require `Y` to be scalar. For
`Y.shape = (2, 2, 2)` and `X.shape = (2, 2)`, the full Jacobian has logical
shape `(2, 2, 2, 2, 2, 2)`. The initial API does not materialize that matrix; it
computes its product with a supplied output cotangent.

If the output is one floating Scalar, `pullback()` may omit the cotangent and
uses the unique natural seed `1`. Tensor, Tuple, Struct, Storage, and
multi-output results require an explicitly matching cotangent. There is no
implicit Tensor `sum` or `mean`.

A downstream scalar objective commonly supplies a non-scalar cotangent. For
example,

```text
L = mean((image - target)^2)
d_image = 2 * (image - target) / element_count
```

and the renderer pullback maps `d_image` to scene gradients. If the objective
is part of the same differentiable ExecutionGraph, this intermediate
cotangent is generated by graph reverse traversal.

## 3. Value, Storage, and Resource gradients

Floating Scalar leaves are differentiable Values. Tensor, Tuple, and Struct
Values derive gradient structure recursively from floating leaves.

- A floating Scalar gradient is an ordinary Scalar Value.
- An immutable Tensor gradient is an ordinary Tensor Value with the same
  logical shape.
- A TensorView or mutable Storage gradient is a newly owned Storage result.
- Integer and Boolean leaves are non-differentiable unless an explicit custom
  rule consumes them without requesting a gradient.
- Resource handles and sampler state are non-differentiable.

The initial gradient accumulation types are:

```text
f16 primal -> f32 gradient
f32 primal -> f32 gradient
f64 primal -> f64 gradient
```

Gradient Storage is separate from primal Storage and never changes primal
type, identity, ownership, or layout. The first API does not expose
`accumulate_into` or persistent `.grad` state. Explicit reusable accumulation
buffers are a later, independent Runtime feature.

## 4. Source and asset declaration

`pipeline_asset()` remains the only cookable declaration. Its `program=`
operand accepts either an ordinary Kernel/graphics stage tuple or a
declarative `ProgramExpression`.

```python
render_rules_v1 = vd.ad.rule_set(
    id="render/v1",
    rasterization=raster_vjp,
    visibility=visibility_vjp,
    depth=depth_vjp,
    blend=blend_vjp,
    texture=texture_vjp,
)

render_asset = vd.pipeline_asset(
    id="pipeline/render",
    program=vd.ad.vjp(
        (vertex_main, fragment_main),
        wrt=("vertices.position", "material.roughness"),
        rules=render_rules_v1,
    ),
    variants=((),),
)
```

`vd.ad.vjp(...)` in this position is a statically parsed program transform,
not an execution call and not another asset type. Rule sets are immutable
module-level declarations parsed without importing or executing the source
module.

An ordinary `program=(vertex, fragment)` or `program=kernel` declaration cooks
only a primal profile. A VJP program expression cooks, under the same asset ID:

- an ordinary primal profile;
- a `forward_with_tape` profile;
- a backward profile;
- versioned tape, cotangent, gradient, `wrt`, and custom-rule reflection.

The cooker CLI and descriptor reference remain unchanged:

```text
vernon-cook-pipeline scene.py:render_asset --target metal -o build/render
```

There is no `ad_pipeline_asset()` and no `autodiff=True` Boolean.

## 5. Python execution

After the Runtime resolves the cooked pipeline:

```python
image, pullback = pipeline.vjp(bindings, grid=(grid_x, grid_y, grid_z))
gradients = pullback(d_image)

d_roughness = gradients["material.roughness"]
d_vertices = gradients["vertices.position"]
```

The binding names and paths accepted by `wrt` are canonical reflected source
paths. They are fixed at cook time and cannot be changed by a Runtime call.
Structured output cotangents and aggregate gradients use the same flattened,
fully qualified leaf paths, such as `output.color` and `material.roughness`;
the cooked transform records the exact cotangent paths and every profile
variant must expose identical gradient paths.

The positive three-dimensional grid is supplied for each forward invocation;
it is not part of the pipeline asset, manifest identity, ProgramGraph, profile
identity, or cooked artifact identity. Invocation carriers use physical
`(z, y, x)` order, with X fastest. A pullback retains its forward grid and
always runs backward over that same logical invocation domain.

A pullback owns immutable saved Values, tape Storage, and temporary GPU
resources from its forward invocation. It may be called sequentially with
multiple cotangents; each call returns fresh gradients and does not consume or
mutate the tape. Destroying the pullback releases the tape. The synchronous
Runtime contract does not promise concurrent calls on one pullback.

## 6. Typed transform and cache identity

Autodiff runs after source loading, feature/constant specialization,
monomorphization, type inference, and effect validation, but before target
lowering.

An immutable `ProgramTransformSpec` records:

- transform kind (`vjp`);
- canonical `wrt` leaf paths;
- output cotangent paths;
- gradient element-type policy;
- semantic accumulation operations and determinism requirements;
- bounded tape/checkpoint policy;
- derivative-rule and custom-rule-set versions.

The transform spec participates in frontend semantic identity, compiler cache
identity, symbols, reflection, cooked stage identity, and the pipeline
manifest. It does not alter primal helper specialization keys.

The compiler-internal `ProgramGraph` represents one specialized program's
typed Value flow, structured control flow, Storage effects, alias regions,
differentiability boundaries, saved Values, and reverse dependencies. It is
not a deployment graph.

## 7. Stateful Kernel differentiation

Stateful reverse mode requires all of the following:

- functionalize local mutation and TensorView writes where semantics permit;
- preserve typed read/write/atomic/barrier effects;
- prove alias, injectivity, and owner-lifetime constraints;
- generate `ReduceSum` and `ScatterAdd` adjoints, with `DisjointScatter`
  evidence only when injectivity is proven independently of one invocation's
  grid;
- define bounded branch/loop tape layouts and reject unbounded tape;
- keep primal, tape, cotangent, and gradient bindings explicit in reflection.

A target lowers semantic accumulation to disjoint direct stores, a supported
atomic-add fast path, or canonical X-fastest deterministic serial reduction.
Backend capability does not change the frontend derivative graph.

## 8. Graphics and custom VJP rules

Differentiating vertex and fragment functions independently does not define a
differentiable graphics pipeline. The Pipeline ProgramGraph models varying
interpolation, rasterization, visibility, depth, blending, and texture
sampling as versioned stage-boundary primitives.

Ordinary differentiable arithmetic uses built-in versioned rules.
Rasterization, visibility, depth, blend, and texture sampling require a named
custom VJP rule set. Each rule declares:

- primal inputs and outputs;
- saved Values and tape bounds;
- accepted cotangent and produced-gradient ABI;
- supported formats, topology, sampling modes, and backend capabilities;
- behavior at discontinuities.

Missing or incompatible rules are compile errors. Resource handles and sampler
state remain non-differentiable. A texture rule may produce gradients for
coordinates and for texel data explicitly exposed as differentiable Storage;
it does not differentiate the Texture handle.

Rule-set identity participates in compiler and pipeline contracts, cache keys,
reflection, and manifests.

## 9. ExecutionGraph composition

`VernonExecutionGraph` remains host orchestration. It is not serialized as
another shader asset and is distinct from compiler-internal `ProgramGraph`.
`VernonAutodiffGraph.h` provides the GPU composition API: add resolved pipeline
nodes, declare named external input ports, connect one node output to a
downstream input, select the sink, and compile the topology. Each
`CompiledAutodiffGraph::forward` call accepts one positive graph-wide grid and
a fresh external value set.
Forward resources and tape are graph-owned and retained by the resulting
`AutodiffGraphPullback`.

The graph validates a DAG and matching connected Value ABIs. Forward profiles
are encoded as compute passes, so ExecutionGraph derives write/read barriers
from the shared `GraphBuffer`; pass execution resolves native handles through
`ExecutionResources`. Pullback walks reverse topological order, routes
connected gradients to upstream cotangents, sums fan-out and repeated external
gradient paths in shared GPU buffers, and creates fresh gradient buffers on
every application. Compiled plans and pullbacks remain reusable after loaded
pipelines and bundles are destroyed. Current public graph composition is C++
and GPU-only and has one selected sink output.

## 10. C and C++ deployment API

C++ loads the same manifest and selects the already cooked VJP profile:

```cpp
auto program = bundle.resolvePipeline("pipeline/render");
auto [image, pullback] = program.vjp(bindings, {gridX, gridY, gridZ});
auto gradients = pullback({{"color", dImage}});
```

The application-facing C ABI uses an opaque pullback handle:

```c
VernonPullback *pullback = NULL;
VernonLaunchSize grid = {grid_x, grid_y, grid_z};
vernonAdPipelineForward(program, grid, &inputs, &outputs, &pullback);
vernonPullbackApply(pullback, &cotangents, &gradients);
vernonPullbackDestroy(pullback);
```

The C structures use `struct_size` and reserved fields. C++ provides RAII over
the same ownership rules. Deployment never runs Python or performs the
autodiff transform.

## 11. Versioning and acceptance

This feature requires intentional compiler- and pipeline-contract advances.
No existing manifest may be reinterpreted as containing VJP profiles.

Acceptance requires:

1. freeze the VJP, pullback, gradient-result, tape-ownership, mutation, and
   custom-rule contracts;
2. add `ProgramExpression`, `ProgramTransformSpec`, deterministic symbols,
   reflection, and cooked manifest schemas;
3. implement pure-Value VJP and finite-difference CPU reference tests;
4. implement Storage functionalization, bounded tape, explicit gradient
   Storage results, alias validation, and accumulation capability checks;
5. implement stateful compute Kernel acceptance;
6. implement rasterization, visibility, depth, blend, and texture custom VJPs
   plus cross-stage graphics backward execution;
7. compose cooked VJP profiles through ExecutionGraph in Python and C++;
8. compare every backend that advertises a given AD capability against the CPU
   reference.

Any missing custom rule, unbounded tape, unsupported target capability,
uncertain write conflict, malformed cotangent, or stale contract version must
fail before execution.
