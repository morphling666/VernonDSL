# Archived host Pass-graph design

> **Status: archived, non-normative, and not an implementation commitment.**
>
> This document preserves the previous Phase 3B/3C host orchestration proposal.
> It was removed from the active compiler, language, and runtime contracts
> because graphics invocation state, attachment semantics, and heterogeneous
> compute/graphics backend ownership are not sufficiently specified. Pipeline
> assets do not depend on this proposal. A future graph design may replace any
> part of it.

## Previous semantic boundary

The proposed graph was a forward-only host semantic object outside parsed
device code. It owned an ordered dependency DAG of compute dispatches,
graphics draws, and inferred resource transitions. A synchronous run retained
all bound resources through final synchronization. Manifest JSON serialized a
validated lowering of the graph, but was not intended to become the semantic
source of topology, effects, or cache identity.

The deployment-facing successor used one common `Pass` concept:

- `ComputePass`: one Kernel dispatch.
- `RenderPass`: one graphics Pipeline draw with attachment/load-store state.

Pipeline was graphics-only. Compute-to-graphics composition was expressed as a
Pass dependency rather than a compute stage inside Pipeline. Resource
transitions were generated lowering records rather than author-authored Passes.
Compatible adjacent RenderPasses could be fused into one backend render scope
without changing their public identity, phase order, or effects.

## Previous declaration sketch

```python
SHADOWS = vd.feature("SHADOWS")


def build_frame(graph: vd.GraphBuilder) -> None:
    simulate = graph.compute_pass(
        id="simulate",
        program=simulate_particles,
        state=vd.compute_state(grid=(1024, 1, 1)),
    )

    forward_dependency = simulate
    if SHADOWS:
        forward_dependency = graph.render_pass(
            id="shadow",
            program=shadow_pipeline,
            depends_on=(simulate,),
            state=vd.render_state(
                colors=(),
                depth=vd.depth_attachment("shadow_depth"),
            ),
            draw=vd.draw(indexed=True),
        )

    graph.render_pass(
        id="forward",
        program=particle_pipeline,
        depends_on=(forward_dependency,),
        state=vd.render_state(
            colors=(vd.color_attachment("color"),),
            depth=vd.depth_attachment("depth"),
        ),
        draw=vd.draw(instances=10000),
    )


frame_asset = vd.execution_graph_asset(
    id="graphs/frame",
    build=build_frame,
    variants=((), (SHADOWS,)),
    targets={"vulkan": {}, "opengl": {}},
)
```

`GraphBuilder`, `ComputePass`, `RenderPass`, and `PassHandle` were host asset
declaration types rather than device-language Values. Pass IDs were stable and
unique within one feature key. `depends_on` accepted PassHandles. Source order
broke ties but did not replace explicit resource dependencies. The cooker was
to evaluate only a restricted declaration AST and compile-time feature
conditions without importing or executing the module.

This syntax is specifically archived because its compute state, draw
description, render state, source-level targets, and backend-independent
attachment model were unresolved. It must not be treated as the preferred
future syntax.

## Previous specialization and lowering proposal

For each canonical feature key, one specialization environment was intended to:

1. specialize every referenced Kernel and graphics Pipeline entry;
2. evaluate feature-guarded Pass inclusion;
3. freeze one ordered Pass topology;
4. merge specialized reflection into canonical state and binding slots;
5. infer resource uses, dependencies, and abstract transitions;
6. serialize topology and matching program artifact references.

The graph feature key was also the program specialization key. There was no
second shader-variant selector. Identical specialized entries remained
content-addressed and could deduplicate across graph feature keys.

Planning distinguished resource state from encoder state. RAW, WAR, WAW,
host-write visibility, and image-layout changes produced synchronization.
Program, framebuffer, viewport, scissor, depth, stencil, blend, and descriptor
changes produced encoder deltas. No raw OpenGL/Vulkan command or
`ExternalCommand` escape hatch was proposed.

## Previous per-Pass preparation proposal

Every Pass had two ordered CPU preparation phases:

1. State filled declared typed state slots through a `StateWriter`.
2. Binding filled reflected program and resource slots through a
   `BindingWriter`.

The runtime then encoded the dispatch or draw. A Pass without a function used
cooked literals or defaults. A C++ application associated optional State and
Binding functions with a stable Pass ID; function names, code, and application
objects were not serialized.

`StateWriter` was intended to expose only declared state. ComputePass could set
its grid. RenderPass could set compatible attachment handles,
load/store/clear values, viewport, scissor, draw ranges, and declared
raster/depth/stencil/blend fields. Attachment format and sample count remained
cooked compatibility constraints.

`BindingWriter` accepted canonical reflection paths mapped to Values,
TensorViews, Textures, or Samplers. Runtime did not interpret application C++
aggregate types. Callbacks could run application logic but could not issue
backend commands, mutate topology, read a preceding GPU result, or create
graph-visible state outside their writer contract.

All Passes were prepared and frozen before any backend submission. Invalid
state, incompatible resources, or absent required bindings failed without
partial execution. Mid-graph GPU-to-CPU callbacks were excluded.

## Previous cooked asset proposal

The proposed schema 3 used `kind: "execution_graph"`:

```json
{
  "schema": 3,
  "kind": "execution_graph",
  "id": "graphs/frame",
  "features": ["SHADOWS"],
  "programs": [],
  "variants": [
    {
      "feature_key": ["SHADOWS"],
      "passes": [
        {
          "id": "simulate",
          "kind": "compute",
          "program": "simulate_particles",
          "depends_on": [],
          "state_contract": {},
          "binding_contract": {},
          "resource_uses": []
        },
        {
          "id": "forward",
          "kind": "render",
          "program": "particle_pipeline",
          "depends_on": ["simulate"],
          "attachments": {},
          "draw": {},
          "state_contract": {},
          "binding_contract": {},
          "resource_uses": []
        }
      ],
      "transitions": []
    }
  ]
}
```

The asset stored feature-keyed Pass records, program artifact references,
specialized reflection, attachment and draw records, typed state and binding
slots, resource uses, and abstract transition steps. It did not store callback
code, application object identity, native handles, frame values, or
backend-native command blobs.

Schema 2 remained the existing program/Pipeline migration format. The proposal
treated schema-2 ordered execution steps as loader-only compatibility input
that could normalize to an empty-feature-key internal graph.

## Previous native surface proposal

```c
typedef struct VernonExecutionGraphAsset VernonExecutionGraphAsset;
typedef struct VernonLoadedExecutionGraph VernonLoadedExecutionGraph;
typedef struct VernonPassStateWriter VernonPassStateWriter;
typedef struct VernonPassBindingWriter VernonPassBindingWriter;

typedef VernonStatus (*VernonPassStateFunction)(
    VernonPassStateWriter *writer,
    void *userdata);

typedef VernonStatus (*VernonPassBindingFunction)(
    VernonPassBindingWriter *writer,
    void *userdata);

VernonStatus vernonRuntimeExecutionGraphSetStateFunction(
    VernonLoadedExecutionGraph *graph,
    const char *pass_id,
    VernonPassStateFunction function,
    void *userdata);

VernonStatus vernonRuntimeExecutionGraphSetBindingFunction(
    VernonLoadedExecutionGraph *graph,
    const char *pass_id,
    VernonPassBindingFunction function,
    void *userdata);
```

The common runtime, rather than Python or Engine adapters, was to parse the
asset, resolve an exact feature key, validate canonical slots, and map planned
operations to a backend. Public structures remained size/version guarded and
writer functions returned status values rather than throwing across the C ABI.

## Previous implementation phases

Phase 3B described the implemented host dispatch/dependency/transition graph:

- represent host dispatches, dependencies, and transitions outside device code;
- lower graph nodes to deterministic runtime steps;
- validate incompatible aliases, missing transitions, lifetime, and backend
  capabilities;
- defer reverse traversal and gradient bindings.

Phase 3C proposed:

- replace public dispatch/draw/transition nodes with ComputePass and RenderPass;
- make Pipeline graphics-only;
- prepare each Pass in State, Binding, encode order;
- use one feature key for program specialization and graph topology;
- cook schema-3 execution assets;
- expose `vernonRuntimeExecutionGraph*`;
- retain schema-2 execution assets as compatibility inputs.

These phase definitions are no longer active roadmap commitments. Multi-program
orchestration, render-pass and attachment semantics, dynamic graphics state,
cross-backend resource ownership, synchronization, and graph-level autodiff
must be redesigned together before host orchestration returns to the canonical
specifications. This proposal is unrelated to the compiler-internal
ProgramGraph used for autodiff inside one specialized program.
