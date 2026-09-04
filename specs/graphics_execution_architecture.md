# Unified Graphics Execution Architecture

Status: implementation plan for the next coordinated compiler/pipeline release.

This document defines the breaking cleanup that unifies standalone graphics
`Pipeline`, graphics calls captured inside `Module`, and cooked graphics assets.
It supersedes the temporary managed-graphics implementation built around
`dsl_state`, `staticState`, deprecated flat invocation fields, and
`ProgramPipelineMode::DirectEndpoint`.

The migration is intentionally clean:

- bump the compiler contract and pipeline ABI together;
- reject older bundles through the ordinary version gate;
- do not dual-read old and new graphics schemas;
- do not retain field-level compatibility shims;
- do not keep a second standalone graphics compiler or execution path.

## 1. Required invariants

1. Every callable Kernel or graphics Pipeline is a Program. A standalone call is
   a one-node Program; a Module is a multi-node Program.
2. Node count does not select a different compiler planner, manifest, loader,
   binding lifecycle, graphics planner, or backend invocation contract.
3. Compute and graphics are distinct typed operation variants. They share one
   dispatch boundary but retain operation-specific payloads and planners.
4. `GraphicsPipelineState` is immutable pipeline state.
5. `RenderPass`, `DrawCommand`, and `DynamicState` are typed invocation
   controls. They are not ordinary shader parameters and do not use general
   parameter binding slots.
6. The manifest has one authoritative representation for every graphics field.
   Runtime execution never parses Python-specific JSON.
7. Attachment dependencies are Program resource-version dependencies. Runtime
   image aliasing and transitions are derived from concrete image/view identity.
8. Backends consume only normalized `ResolvedComputeInvocation` or
   `ResolvedGraphicsInvocation` values. Backends do not infer defaults or choose
   between deprecated and current fields.

## 2. Unified operation model

Python capture may retain separate `ComputeCall` and `GraphicsCall` payload
classes, but all generic code dispatches through one closed `ExecutionKind`.
The C++ in-memory manifest representation uses a tagged union:

```cpp
enum class ExecutionKind {
    Compute,
    Graphics,
};

using NodeOperation = std::variant<ComputeOperation, GraphicsOperation>;

struct Node {
    uint32_t id;
    std::string name;
    std::string stage;
    std::vector<uint32_t> operands;
    std::vector<uint32_t> results;
    std::vector<EndpointBinding> bindings;
    std::vector<ResourceAccess> accesses;
    NodeOperation operation;
};
```

The serialized operation tag is converted to this variant exactly once in the
manifest parser. Code after parsing must use variant visitation, not string
comparisons such as `node.operation == "graphics"`.

Backend selection and execution-kind selection are separate concerns. Runtime
dispatch resolves one executor from `(backend, ExecutionKind)` or equivalent
typed provider capabilities. CPU and CUDA reject unsupported graphics through a
single capability check; graphics invocation must never succeed as a no-op.

## 3. One Program path

### 3.1 Interactive standalone calls

`Pipeline.__call__` constructs or reuses a one-node canonical Program and binds
its shader arguments and three graphics control slots. It uses the same:

- `plan_program_result`;
- Program canonicalization and deployment;
- Program specialization/cache identity;
- persistent binding transaction;
- Program loader;
- Program executor;
- graphics invocation planner;
- backend adapter

as a Pipeline call captured inside `Module.forward`.

The direct Python implementations of graphics attachment encoding, draw
encoding, default blend expansion, viewport/scissor fallback, and submission
are removed from `python/vernon_dsl/_runtime/pipeline.py`.

### 3.2 Cooked graphics

A cooked standalone graphics asset packages the same one-node Program produced
for interactive execution. The legacy stage-tuple bundle planner is removed for
graphics. Shader stages are compiled once per Program specialization and reused
when assembling the graphics artifact.

### 3.3 Runtime loading

`ProgramPipelineMode::DirectEndpoint` is removed. The runtime has one Program
loader and one Program invocation lifecycle.

A one-node fast path is permitted only after the Program has passed the same
parse, resolve, binding, control snapshot, and invocation normalization steps as
a multi-node Program. It is an internal scheduling optimization, not another
semantic mode.

## 4. Graphics manifest schema

The coordinated release replaces the current graphics operation schema. It
does not emit or accept `dsl_state`.

Conceptually, a graphics node contains:

```json
{
  "tag": "graphics",
  "pipeline_state": {
    "topology": "triangle_list",
    "rasterization": {},
    "depth_stencil": {},
    "color_blends": []
  },
  "render_pass": {
    "control": 3,
    "colors": [
      {
        "location": 0,
        "access": 4,
        "formats": ["rgba8_unorm"],
        "sample_counts": [1]
      }
    ],
    "depth_stencil": null
  },
  "draw": {
    "control": 4,
    "default": {
      "tag": "direct",
      "vertex_count": 3,
      "instance_count": 1
    }
  },
  "dynamic_state": {
    "control": 5
  }
}
```

Exact canonical ordering and integer encoding remain compiler-defined, but the
ownership rules are fixed:

- `pipeline_state` is the sole static pipeline-state authority;
- `render_pass.control` selects the typed invocation RenderPass;
- render-pass attachment records identify Program resource accesses and
  reflected compatibility constraints, not duplicate runtime load/store values;
- `draw.control` selects an optional typed DrawCommand;
- `draw.default` is the compiler-resolved fallback when no DrawCommand is bound;
- `dynamic_state.control` selects an optional typed DynamicState;
- queues and predecessor edges are not serialized.

`GraphicsOperation` in `program_execution_manifest.h` stores fully parsed typed
state, attachment signatures, direct/indexed draw defaults, and control slots.
All enum, attachment, blend, depth/stencil, draw, and control-slot validation
happens during parse/resolve.

Delete:

- `GraphicsOperation::staticState`;
- `prepareManagedGraphicsState`;
- execution-time JSON parsing;
- compiler-generated placeholder graphics state that is later ignored;
- undocumented graphics override fields.

## 5. Pipeline invocation ABI

`VernonProgramSubmitDescriptor` keeps typed pointers:

```c
const VernonGraphicsState *graphics_state;
const VernonRenderPass *render_pass;
const VernonDrawCommand *draw_command;
const VernonDynamicState *dynamic_state;
```

Remove the deprecated flat graphics fields:

- topology;
- color attachments and depth attachment;
- index binding;
- vertex and instance counts;
- viewport and scissor;
- stencil reference.

There is no fallback from typed pointers to flat fields.

The runtime resolves one immutable invocation:

```text
typed manifest pipeline state
  + reflected attachment constraints
  + typed invocation controls
  = ResolvedGraphicsInvocation
```

`ResolvedGraphicsInvocation` owns or safely references all normalized arrays and
contains:

- normalized shader arguments;
- color/depth/stencil attachment views and operations;
- attachment formats, extents, samples, and render area;
- direct or indexed draw parameters;
- viewport, scissor, and stencil reference;
- normalized rasterization, depth/stencil, and blend state;
- the complete graphics variant key;
- image and buffer resource accesses required by the command DAG.

Every graphics backend consumes this object. Backend code may translate it to
the native API, but may not resolve DSL defaults or reinterpret manifest fields.

## 6. Attachment output DSL

Add:

```python
color = vd.color_output(render_pass, location=0)
depth = vd.depth_output(render_pass)
```

These functions expose the attachment version produced by the most recent
graphics node that uses the given RenderPass control.

The public value is an immutable, read-only `AttachmentOutputView`:

- direct execution resolves it to a read-only `TextureView`;
- Module capture resolves it to a symbolic projection of an existing attachment
  storage owner;
- it never creates or owns image storage;
- color output carries its location and color aspect;
- depth output carries its depth aspect and concrete texture format.

Capture rejects:

- requesting an attachment not present in the RenderPass prototype;
- reading before a producing graphics node;
- reading an attachment whose producing operation uses `store=discard`;
- using a depth output where no depth attachment exists;
- an incompatible sampled/storage usage.

The Program graph wires an output projection to the graphics attachment
`after` version. A later shader sample, storage access, copy, host publication,
or attachment use consumes that exact SSA version.

Resource identity must not use `id(render_pass)`. Compile-time projections use
the RenderPass control slot plus aspect/location and explicit storage alias
information. Runtime hazards use concrete parent image/view identity so two
different controls that alias the same image remain correct.

## 7. Image hazards and render-scope composition

Graphics command nodes declare concrete accesses for:

- color attachment reads/writes;
- depth and stencil attachment reads/writes;
- sampled images;
- storage images;
- index and vertex buffers.

The command DAG derives RAW, WAR, and WAW edges and required RHI states from
these accesses.

Adjacent graphics nodes may share one render scope only when all of the
following hold:

1. attachment parent images, views, aspects, locations, formats, sample counts,
   extents, and layer ranges are compatible;
2. the next node preserves the current attachment contents;
3. previous store and next load semantics permit eliding the boundary;
4. neither node requests a clear/discard operation that requires a new scope;
5. no intervening sampled/storage/transfer/compute access requires the image in
   another state;
6. depth/stencil read/write modes are compatible;
7. the backend advertises the required render-scope capability.

A scope always ends for incompatible attachments, clear/discard boundaries,
attachment-to-sampled dependencies, compute/transfer nodes, or explicit
publication.

The current policy that batches every adjacent graphics node into one provider
rendering scope is removed. Composition is decided by a dedicated
`GraphicsScopePlanner`, not by ad-hoc checks in the Program execution loop.

## 8. Implementation sequence

### Phase 1: parity gates

- Add direct Pipeline versus one-node Module pixel parity.
- Add direct, Module, and cooked artifact/reflection parity.
- Cover static state, indexed draw, dynamic state, persistent controls,
  attachment format specialization, and error parity.

### Phase 2: typed dispatch

- Add shared Python and C++ `ExecutionKind`.
- Convert runtime `Node` to a typed operation variant.
- Centralize capability and backend executor dispatch.
- Remove string and empty-stage-map inference outside parser boundaries.

### Phase 3: one Program pipeline

- Normalize standalone Pipeline to one-node Program.
- Move all graphics control binding to the persistent Program transaction.
- Remove the standalone graphics planner/deployment/invocation path.
- Move cooked graphics to Program Assets.
- Remove duplicate Module shader compilation.
- Remove `ProgramPipelineMode::DirectEndpoint`.

### Phase 4: breaking manifest and ABI

- Bump compiler contract and pipeline ABI together.
- Emit and parse the new typed graphics operation.
- Remove `dsl_state`, `staticState`, placeholder state, execution-time JSON, and
  deprecated flat invocation fields.
- Replace fixtures with new-version golden fixtures.
- Verify old bundles fail only at the version gate.

### Phase 5: attachment outputs and scheduling

- Implement `AttachmentOutputView`, `color_output`, and `depth_output`.
- Emit explicit attachment-after to sampled/read resource chains.
- Add concrete image accesses to command nodes.
- Introduce `GraphicsScopePlanner`.
- Split and merge scopes according to the rules in section 7.

### Phase 6: cleanup

- Remove dead legacy planners, serializers, loader modes, cache keys, branches,
  tests, and comments.
- Update all graphics examples to use attachment outputs for multipass flow.
- Update the Program manifest, runtime design, unified Pipeline/Module, C API,
  and language contract specifications.

## 9. Acceptance criteria

The migration is complete only when:

1. standalone Pipeline, one-node Module, multi-node Module, and cooked
   standalone graphics use the same canonical Program schema and loader;
2. no runtime code contains `dsl_state`, `staticState`, execution-time graphics
   JSON parsing, or deprecated flat invocation reads;
3. no semantic decision depends on `variant.compute.empty()`,
   `node.operation == "graphics"`, or `ProgramPipelineMode::DirectEndpoint`;
4. direct and Module rendering produce identical pixels, validation errors,
   variant keys, and persistent-control behavior;
5. multipass color/depth output tests prove attachment-write to sampled-read
   ordering;
6. aliased images remain correct across different RenderPass controls;
7. render scopes merge only when the compatibility rules permit it;
8. one build directory is used and full build, CTest, Python, formatting, lint,
   OpenGL, Metal, Vulkan, and DirectX gates run sequentially;
9. unavailable backend gates skip with an explicit capability reason;
10. repository search finds no obsolete graphics schema or compatibility path.

## 10. Primary implementation files

Python:

- `python/vernon_dsl/render.py`
- `python/vernon_dsl/__init__.py`
- `python/vernon_dsl/operation_graph.py`
- `python/vernon_dsl/program.py`
- `python/vernon_dsl/program_frontend/parser.py`
- `python/vernon_dsl/_runtime/pipeline.py`
- `python/vernon_dsl/_runtime/program_autodiff.py`
- `python/vernon_dsl/_runtime/binding.py`
- `python/vernon_dsl/_program_assets/cooking.py`
- `python/vernon_dsl/bundle/planner.py`

Compiler and runtime:

- `source/include/VernonVersions.h`
- `source/include/VernonRuntime.h`
- `source/include/VernonRuntime.hpp`
- `source/lib/compiler/compiler_program_graphics.cpp`
- `source/lib/compiler/compiler_program_lowering.cpp`
- `source/lib/runtime/program_execution_manifest.h`
- `source/lib/runtime/program_execution_manifest.cpp`
- `source/lib/runtime/program_execution_backend.cpp`
- `source/lib/runtime/target_binding_plan.cpp`
- `source/lib/runtime/graphics_invocation_planner.h`
- `source/lib/runtime/graphics_invocation_planner.cpp`
- `source/lib/runtime/runtime_pipeline_dispatch.cpp`
- `source/lib/runtime/VernonRuntime.cpp`
- `source/lib/execution_graph/execution_command_model.h`
- graphics backend adapters under `source/lib/runtime/runtime_pipeline_*.cpp`

Tests:

- `python/tests/test_pipeline_runtime.py`
- `python/tests/test_module_graphics_controls.py`
- runtime manifest, graphics planner, C API, command DAG, and backend graphics
  tests under `source/tests/`.
