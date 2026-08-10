# Autodiff design

> **Status:** accepted contract and implemented compute VJP surface. Compiler
> contract 11 and pipeline contract 15 provide deterministic
> `primal`/`forward_with_tape`/`backward` profiles, typed derivative groups,
> checked dynamic tape, explicit accumulation plans, and invocation-time
> `(x,y,z)` workgroup grids with physical invocation carriers. CPU `dynamic_v2` supports
> direct and cooked void Kernels with explicit Storage objectives, recursive
> Scalar/Tensor/Tuple/Struct Storage elements, dynamic and signed-stride
> TensorViews, mutable scratch versioning, structured branches and loops,
> runtime gather/scatter accumulation, reusable pullbacks, and fresh packed
> tangent owners. Direct and cooked execution normalize the same native
> derivative metadata and share one structured CPU executable and pullback
> implementation.
>
> CPU `dynamic_v2` is the only supported autodiff execution and cooking path.
> GPU and graphics autodiff, graph-level pullbacks, graphics backward
> lowering, custom compute VJPs, higher-order AD, persistent `.grad`, and
> multi-kernel temporal differentiation are deferred and unsupported. GPU
> targets may still compile and run ordinary non-AD compute and graphics
> pipelines.

This document defines Vernon's first public automatic-differentiation model.
The design uses reverse-mode vector-Jacobian products (VJPs) and preserves the
Value/Storage/Resource split. The implemented surface differentiates CPU
Kernels; differentiated graphics stages remain future work.

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

Aggregate Storage cotangents and gradients preserve this logical structure at
the public API boundary. Compiler profiles may transport canonical ABI leaves
independently, but the host reconstructs them through one structural
`TangentLayout` and one packed tangent owner per primal Storage owner.

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

and the renderer pullback maps `d_image` to scene gradients. Graph-level
reverse traversal is not currently supported; callers compose intermediate
cotangents explicitly.

## 3. Value, Storage, and Resource gradients

Floating Scalar leaves are differentiable Values. Tensor, Tuple, and Struct
Values derive gradient structure recursively from floating leaves.

- A floating Scalar gradient is an ordinary Scalar Value.
- An immutable Tensor gradient is an ordinary Tensor Value with the same
  logical shape.
- A TensorView or mutable Storage gradient is newly owned Storage at the public
  pullback boundary. In the compiler ABI it is a writable TensorView argument
  of the backward profile, not a fixed-shape SSA result.
- Vector, Matrix, Tensor, Tuple, and Struct Storage elements derive one
  structural tangent schema recursively. Their public gradient is one packed
  tangent `TensorStorage`, not one owner per ABI leaf.
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

Dynamic Storage extents remain dynamic through differentiation. Backward
profiles receive descriptor-backed shape-source TensorViews for primal Storage
identities, descriptor-backed cotangent TensorViews, and writable gradient
TensorViews. Invocation-local adjoint buffers use those runtime extents;
compiler analysis and reflection must not specialize `dyn` dimensions. This
does not change the tape allocator ABI.

External gradient destinations carry mandatory ownership metadata derived by
autodiff analysis: `invocation_private`, `workgroup_shared`, `atomic_shared`,
or `none`. Static and dynamic TensorViews use the same rule; shape size never
selects ownership. Missing metadata fails profile loading, and unsupported
shared accumulation fails lowering instead of selecting shared staging or
serial execution.

Forward-with-tape and backward profiles each reflect the same internal
dispatch contract used by ordinary compute. CPU AD validates both contracts
against the requested primal grid before constructing an effect transaction,
allocating tape, or staging gradients, and validates the backward contract
again when a reusable pullback is applied.

The tangent layout is independent from the primal layout. Dtype promotion may
change element size, alignment, product offsets, and element stride.
Non-differentiable product children remain logical `Zero` nodes and occupy no
tangent bytes. A tangent layout is structural metadata, not a generated
nominal DSL Struct declaration, and it must never reinterpret primal bytes.

The recursive tangent schema is:

```text
T(f16) = Scalar(f32)
T(f32) = Scalar(f32)
T(f64) = Scalar(f64)
T(bool | i32 | u32) = Zero
T(Tensor[S, shape]) = Tensor(shape, T(S))
T(Vector[S, n]) = Tensor((n,), T(S))
T(Matrix[S, m, n]) = Tensor((m, n), T(S))
T(Tuple[A, ...]) = Product(0:T(A), ...)
T(Struct{a:A, ...}) = Product(a:T(A), ...)
```

`TangentLayout` adapts this schema to the same canonical host ABI planner used
for ordinary Values. It does not maintain an independent alignment or offset
algorithm. Product-valued Tensor elements retain their element coordinates in
canonical derivative paths; `Zero` projections fail deterministically.

All compatible `wrt` views of one primal allocation accumulate into one fresh
tangent owner. Separately bound overlapping read/write aliases remain invalid.
A legal in-place `read_write` binding is differentiated through explicit
Storage versions.

## 4. Source and asset declaration

`pipeline_asset()` remains the only cookable declaration. For supported
autodiff cooking, its `program=` operand is a CPU Kernel VJP
`ProgramExpression`.

```python
loss_asset = vd.pipeline_asset(
    id="pipeline/loss",
    program=vd.ad.vjp(
        loss_kernel,
        wrt=("parameters",),
        outputs=("loss",),
    ),
    variants=((),),
)
```

`vd.ad.vjp(...)` in this position is a statically parsed program transform,
not an execution call and not another asset type.

An ordinary `program=(vertex, fragment)` or `program=kernel` declaration cooks
only a primal profile. A VJP program expression cooks, under the same asset ID:

- an ordinary primal profile;
- a `forward_with_tape` profile;
- a backward profile;
- versioned tape, cotangent, gradient, `wrt`, and custom-rule reflection.

The cooker CLI and descriptor reference remain unchanged:

```text
vernon-cook-pipeline model.py:loss_asset --target cpu -o build/loss
```

There is no `ad_pipeline_asset()` and no `autodiff=True` Boolean.

## 5. Python execution

After the Runtime resolves the cooked pipeline:

```python
pipeline = vd.load_cooked_vjp_asset(
    "build/render/render.pipeline.json",
    features=(),
)
image, pullback = pipeline.vjp(bindings, grid=(grid_x, grid_y, grid_z))
gradients = pullback(d_image)

d_roughness = gradients["material.roughness"]
d_vertices = gradients["vertices.position"]
```

The binding names and paths accepted by `wrt` are canonical reflected source
paths. They are fixed at cook time and cannot be changed by a Runtime call.
Compiler and Runtime profiles transport structured output cotangents and
aggregate gradients through flattened, fully qualified leaf paths such as
`output.color` and `material.roughness`. The public pullback API groups those
leaves by the declared output or `wrt` path and packs aggregate Storage through
its reflected `TangentLayout`; leaf paths are not separate public Storage
owners. Every profile variant must expose identical groups and leaf paths.

The Runtime stores one canonical typed derivative-group table in
gradient-then-cotangent order. Direct execution supplies it through POD metadata
views; cooked execution derives it once from the validated transform and
backward profile. Both paths validate the same canonical paths, unique group
ownership, protocol, and executable signature. Python always reads groups from
the loaded native pipeline; it does not maintain a second direct or cooked
grouping model.

The public C Runtime exposes indexed reflection rather than a scalar-output
special case:

- `vernonRuntimeLoadedPipelineGetAdOutputCount` and
  `vernonRuntimeLoadedPipelineGetAdOutputByIndex`;
- the equivalent cotangent and gradient count/index pairs;
- derivative-group count/index queries and indexed leaf-path queries.

Each value query returns one `VernonAdValueMetadataView` containing the
canonical path, tangent dtype, rank, and logical shape. This is the only public
output/cotangent/gradient reflection API and supports aggregate and independent
multi-output objectives without reinterpretation.

For launches larger than one invocation, packed aggregate cotangents prepend the
physical `(grid.z*workgroup.z, grid.y*workgroup.y, grid.x*workgroup.x)` carrier
dimensions to the primal owner shape. TensorView
offsets and signed strides apply to the trailing owner dimensions, so
carrier packing does not erase subview descriptors.

The positive three-dimensional workgroup grid is supplied for each forward invocation;
it is not part of the pipeline asset, manifest identity, ProgramGraph, profile
identity, or cooked artifact identity. Each workgroup contains the reflected
`workgroup_size`; invocation carriers use physical `(z, y, x)` order, with X
fastest. A pullback retains its forward grid and
always runs backward over that same logical invocation domain.

A pullback owns immutable saved Values and CPU tape Storage from its forward
invocation. It may be called sequentially with
multiple cotangents; each call returns fresh gradients and does not consume or
mutate the tape. Destroying the pullback releases the tape. The synchronous
Runtime contract does not promise concurrent calls on one pullback.

### CPU range-phase execution

CPU primal, `forward_with_tape`, and `backward` profiles all use the ordinary
range-phase scheduler described by
[`runtime/design.md`](runtime/design.md#cpu-range-phase-execution). There is no
AD scalar-entry adapter, worker-local lane identity, fixed-tape execution
branch, protocol guessing, or serial fallback.

For each scheduler range, Runtime lazily prepares the active lanes' packed
argument/result frames and supplies flattened-global pointer tables to the
compiled range entry. The generated wrapper invokes the lowered program for
every lane in the contiguous interval without a Runtime callback per lane.
Frames persist while a lane is yielded and are released immediately after
final completion; pointer-table bounds are explicit in `VernonCpuRangeV1`.

Each logical forward invocation owns one `HostDynamicTape`, independent of the
worker executing its current phase. The allocator registry provides checked
O(1) descriptor ownership lookup, and each tape serializes its mutable
metadata. A barrier yield retains the live tape and coroutine frame. Final lane
completion seals and validates the tape, transfers an immutable
`HostTapeSnapshot` to the pending pullback, and releases the mutable tape.
Snapshots retain their memory-policy charge until the pullback releases them.
Compiler-generated code sees only the versioned semantic allocator callbacks
and opaque handles; Runtime alone owns payload, region, record, and index
representation.

Backward execution creates fresh mutable lane/workgroup phase state for each
pullback application while retaining the immutable forward snapshots. Reverse
barriers are lowered through the same coroutine/yield machinery as primal
barriers. Invocation-private gradients follow logical lanes across workers;
workgroup-shared gradients remain in the group arena; atomic-shared updates use
the declared target capability. External gradients are accumulated and
published only after every backward workgroup completes successfully.

Before dispatch, checked policy accounting covers tape reservation,
cotangent carriers, invocation-private or workgroup-shared gradient staging,
and arithmetic overflow. Forward Storage/output writes use
`HostEffectTransaction` shadows and commit exactly once only after every lane
has completed and every snapshot is valid. Failure discards shadows, mutable
tapes, untransferred snapshots, lane frames, group arenas, staging, and
dispatch reservations. Parallel range failures collect diagnostics without
writing shared invocation state from worker threads.

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

## 8. Deferred graphics and custom VJP rules

GPU and graphics autodiff are unsupported in the current implementation. A
future design must account for the fact that differentiating vertex and
fragment functions independently does not define a differentiable graphics
pipeline. The Pipeline ProgramGraph would model varying
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

`VernonExecutionGraph` remains ordinary host orchestration and is not a
differentiation API. Graph-level autodiff and graph pullbacks are deferred and
unsupported.

## 10. C and C++ deployment API

C++ can load a CPU manifest and select its cooked VJP profile:

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

Compiler contract 11 and pipeline contract 15 are the current CPU VJP boundary.
Differentiated assets use the canonical pipeline manifest with one optional
root `autodiff` object; pipeline-14 transform/profile fields are not aliases in
the current schema. No
older manifest is reinterpreted as containing current structured profiles.
CPU VJP has one structured `dynamic_v2` compiler/runtime path. Differentiated
GPU and graphics assets are not supported.

CPU acceptance covers direct and cooked structured Storage VJP, recursive
aggregate tangents, multiple outputs, owner aliases, signed-stride descriptors,
multi-invocation cotangent carriers, dynamic control flow, scratch overwrite,
finite differences, and deterministic reusable pullbacks. Runtime acceptance
also freezes tape allocator ABI, ownership, memory charging, failure latching,
and public exception containment. GPU acceptance covers only ordinary non-AD
compute and graphics pipelines.

Any missing derivative or custom rule, unbounded tape, unsupported target
capability, uncertain write conflict, malformed group/cotangent/signature,
protocol mismatch, or stale contract version fails before execution.
