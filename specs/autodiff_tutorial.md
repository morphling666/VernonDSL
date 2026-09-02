# VernonDSL automatic differentiation

This guide describes the compiler-contract 13 and Pipeline 17 architecture.
Canonical Program is the only differentiation topology; Python pass graphs and
legacy pipeline AD manifests are not part of the API.

## 1. VJP model

`vd.ad.vjp` computes a vector-Jacobian product. For a function `y = f(x)`, the
forward call returns `y` and a pullback. Applying cotangent `dy` to the pullback
returns `dx = dy J_f(x)`.

For a `vd.Module`, differentiation is declared with logical input and output
paths:

```python
transformed = vd.ad.vjp(module, wrt=("parameters.weight",), outputs=("loss",))
outputs, pullback = transformed(inputs)
gradients = pullback({"loss": output_cotangent})
```

Paths address logical aggregate leaves. A root argument uses its argument name;
nested structures use dotted field paths and numeric tuple indices.

`wrt=("particle",)` selects the whole aggregate. The Program signature expands
that root to canonical differentiable leaves and reconstructs the gradient tree
at the Python boundary.

## 2. Canonical Program topology

The frontend captures a Module invocation and emits one typed Program with:

- a forward graph;
- an optional backward graph;
- public primal, cotangent, and gradient signature paths;
- explicit Storage ownership and resource-version effects;
- residual captures required by the backward graph;
- stage references resolved through ArtifactSystem.

The compiler owns differentiation and residual planning. Runtime does not infer
a reverse graph from host pass descriptors, and Python does not construct a
second execution topology.

At load time, Runtime resolves Program and ArtifactSystem into `ResolvedProgram`.
The forward and backward graphs then lower to the private C++ Command DAG for
hazard analysis, barriers, render-scope formation, and submission.

## 3. Residuals and pullback lifetime

The forward graph may capture values needed by the backward graph. Runtime
stores those values in Program-owned pullback state. A pullback retains:

- the immutable `ResolvedProgram`;
- captured residual Storage;
- the primal resources required by the contract;
- backend submission state.

Pullbacks are reusable when their Program contract permits reuse. Gradient
publication is transactional: failed backward execution does not expose
partially accumulated results.

No-Tape means the differentiated stage needs no local AD tape. It does not mean
that a multi-node Program has no residual values.

## 4. Shapes and physical ABI

Logical tensor shape is distinct from the physical carrier ABI. Dynamic shape,
stride, and offset are invocation data and do not require recompilation.

For kernel-local reverse execution, a cotangent carrier may include a leading
invocation domain. That domain is represented by the canonical binding layout;
callers must not synthesize a constant leading extent. Program binding derives
the physical view from the declared carrier shape, dispatch invocation count,
and canonical strides.

Aggregate values are flattened exactly once by the compiler's Value ABI.
Program signatures, stage bindings, runtime materialization, and gradient
reconstruction all consume the same canonical leaf paths and layouts.

## 5. Gradient accumulation

Each backward node writes its local gradient contribution according to the
stage ABI. Program-level fan-in is explicit in the backward graph and combines
contributions deterministically. The runtime must not infer fan-in from names
or apply an additional Python-side accumulation pass.

The current gradient element policy is:

- `f16` primal accumulates into `f32`;
- `f32` primal accumulates into `f32`;
- `f64` primal accumulates into `f64`.

Fresh pullback application starts from zeroed gradient destinations unless the
Program explicitly declares another initialization operation.

## 6. Planning policy

`planning_policy` accepts `min_memory`, `balanced`, or `min_runtime`. It controls
compiler residual selection and bounded GPU replay where a stage requires tape.
The policy is part of the Program transform identity and therefore participates
in compilation caching.

Runtime still enforces hard physical memory limits. If one complete workgroup
cannot fit the bounded replay budget, pullback application fails explicitly.

## 7. Direct kernel VJP

A direct kernel expression remains a one-node convenience surface:

```python
expression = vd.ad.vjp(kernel, wrt=("source",), outputs=("loss",))
outputs, pullback = expression(source, loss, grid=(groups_x, 1, 1))
gradients = pullback(output_cotangent)
```

It compiles through the same typed stage and Program contracts. The native
direct-endpoint loader is separate from managed Module loading, so endpoint
invocation semantics are explicit rather than inferred from node count or
Storage ownership.

## 8. Diagnostics

Program failures use stable phase and path information:

- parse errors identify malformed Program or ArtifactSystem fields;
- resolve errors identify missing stages, incompatible contracts, or invalid
  signatures;
- invocation errors identify missing bindings, shape/layout mismatches, or
  unavailable controls;
- execute errors identify backend submission or pullback failures.

When debugging numerical gradients, inspect in this order:

1. logical signature paths and aggregate leaf expansion;
2. resolved value layouts and Storage aliases;
3. dispatch controls and physical carrier extents;
4. residual capture and tape batch layout;
5. backward fan-in and final gradient publication.

Avoid fixing a stage ABI mismatch by reshaping data in Python. The Program
signature and resolved carrier layout are the authority.

## 9. Unsupported behavior

- Python `ExecutionGraph`, pass descriptors, and graph-level VJP APIs were
  removed in Pipeline 17.
- Compiler-contract 12 and Pipeline 16 artifacts are rejected rather than
  normalized.
- Graphics differentiation requires explicit versioned derivative rules for
  rasterization, visibility, depth, blending, and texture operations; ordinary
  graphics execution does not imply those rules exist.
