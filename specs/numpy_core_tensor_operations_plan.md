# NumPy Core Tensor Operations Implementation Plan

## Goal

Define and implement a coherent NumPy-compatible numeric Tensor subset rather
than adding unrelated intrinsics one at a time. Frontend inference, CPU/CUDA
lowering, SPIR-V lowering, graphics stages, runtime execution, and host/shared
execution must agree on:

- result shape;
- axis handling;
- dtype promotion and accumulation dtype;
- broadcasting;
- error conditions;
- scalar versus rank-zero results.

The supported subset is static-shape and device-oriented. NumPy APIs requiring
dynamic output sizes or unbounded temporary storage are outside the first
implementation.

## Existing baseline

Vernon currently supports:

- scalar/Tensor elementwise arithmetic with NumPy trailing-dimension
  broadcasting;
- NumPy-compatible `matmul` rank promotion and batch broadcasting;
- common math functions such as `sin`, `cos`, `exp`, `log`, `sqrt`, and `abs`;
- shader-oriented `dot`, `cross`, `normalize`, `norm`, and `reflect`;
- binary elementwise `min`, `max`, `pow`, and ternary `clamp`.

The current `min`/`max` names do not match NumPy: Vernon treats them as binary
elementwise operations while `np.min`/`np.max` are reductions. This must be
resolved as part of the public API normalization.

## Scope and priorities

### Priority 0: API consistency

Add NumPy names and semantics:

- `minimum(left, right)` and `maximum(left, right)` for elementwise operations;
- `power(left, right)` for elementwise exponentiation;
- `clip(value, minimum, maximum)`;
- `min(value, axis=None, keepdims=False)` and
  `max(value, axis=None, keepdims=False)` for reductions.

Keep shader-style `pow` and `clamp` only if they are documented aliases that
lower to exactly the same canonical operation. Do not let aliases acquire
different broadcasting or dtype behavior.

Because changing `min`/`max` is source-breaking, perform one intentional
language API version cutover and update all in-tree shaders/tests. Do not
overload `min(a, b)` as elementwise while also supporting reduction semantics;
that would preserve ambiguity.

### Priority 1: static shape/view operations

Implement:

- `reshape`
- `transpose`
- `swapaxes`
- `squeeze`
- `expand_dims`
- `broadcast_to`

These operations produce Values with static shapes. They do not allocate
runtime Storage and should lower through canonical compile-time index maps.

### Priority 2: reductions

Implement:

- `sum`
- `prod`
- `min`
- `max`
- `mean`
- `any`
- `all`

Support:

- `axis=None`;
- one integer axis;
- a tuple of unique integer axes;
- negative axes;
- `keepdims`;
- scalar and Tensor results according to NumPy shape rules.

### Priority 3: selection and dtype conversion

Implement:

- `where(condition, left, right)`
- `astype(value, dtype)`
- complete comparison and logical broadcasting needed by `where`, `any`, and
  `all`.

The three `where` operands follow NumPy broadcast rules. One-argument
`np.where(condition)` is excluded because it has dynamic-sized index outputs.

### Priority 4: static composition and indexing

Implement:

- `concatenate`
- `stack`
- `take`
- `take_along_axis`
- basic static slicing
- `argmin`
- `argmax`

Dynamic advanced indexing, boolean index result compaction, and mutation
through advanced indices are separate work.

### Priority 5: general contractions

After reductions and static index maps are stable, implement:

- `tensordot`
- a constrained static-shape `einsum`

Both must lower through one generic contraction planner shared with the
scalarized fallback of `matmul`. Do not add expression-specific `einsum`
patterns.

## Canonical semantic planners

Create independent, testable planners in Python and matching C++ code. Keep
logical semantics separate from backend physical vector/matrix choices.

### Axis normalization

One canonical helper accepts rank and `axis`:

- normalize negative axes;
- reject out-of-range axes;
- reject duplicate axes;
- return axes in deterministic ascending order;
- distinguish `axis=None` from an empty axis tuple.

Use it for reductions, squeeze, transpose, take, and contractions where
applicable.

### Static index maps

Represent shape transforms with a planner that maps each result linear index to
one source linear index. Use it for:

- reshape;
- transpose/swapaxes;
- squeeze/expand_dims;
- broadcast_to;
- concatenate/stack;
- take with compile-time indices where possible.

Backends consume the same map and may optimize it later. Initial correctness
must not depend on native vector shuffle or matrix operations.

### Reduction plan

The reduction planner records:

- source shape;
- normalized axes;
- retained and reduced dimensions;
- result shape with and without `keepdims`;
- reduction element count;
- source linear index for each result/reduction coordinate;
- identity and combination operation.

### Contraction plan

Generalize the current static `matmul` plan to describe:

- left/right batch dimensions;
- contracted dimension pairs;
- retained dimensions;
- result shape;
- broadcasted batch coordinates;
- source indices for each result/reduction coordinate.

`matmul`, `tensordot`, and constrained `einsum` must use this common model.

## NumPy dtype behavior

Do not infer dtype separately in each intrinsic.

### Elementwise and selection

Use the existing safe Vernon numeric promotion where it intentionally differs
from unsafe implicit narrowing, but document every difference from NumPy.
`where` computes the common dtype of its value operands after broadcasting the
condition.

### Reductions

Specify accumulation and result dtype explicitly:

- floating input keeps its floating dtype unless an explicit supported dtype
  is provided;
- integer `sum`/`prod` must not silently use backend-dependent widths;
- `mean` requires a floating result;
- `any`/`all` return bool;
- `min`/`max` retain input dtype;
- `argmin`/`argmax` return one canonical index dtype.

The first version may reject optional `dtype`, `out`, `initial`, and `where`
reduction parameters, but diagnostics must name the unsupported option rather
than silently ignoring it.

### Casts

`astype` supports only explicitly listed numeric conversions. Backend
capability differences such as unavailable `f64` must produce deterministic
target diagnostics, never implicit narrowing.

## Frontend implementation

Update:

- `python/vernon_dsl/intrinsics.py`
- `python/vernon_dsl/__init__.py`
- `python/vernon_dsl/frontend/tensor_shapes.py`
- `python/vernon_dsl/frontend/type_solver.py`
- `python/vernon_dsl/frontend/inference.py`
- `python/vernon_dsl/frontend/lowering.py`
- shared host execution implementations

Required work:

- add editor-visible NumPy-compatible signatures;
- parse axis, shape, permutation, and dtype arguments as compile-time values;
- use canonical planners for result inference;
- emit a small set of canonical Vernon intrinsics:
  - `shape_transform`;
  - `reduce`;
  - `where`;
  - `cast`;
  - `concatenate`;
  - `contraction`;
- avoid adding one IR operation for every public alias;
- provide source diagnostics matching the invalid concept: axis, shape,
  permutation, broadcast, dtype, or unsupported dynamic behavior.

Host/shared execution should delegate to NumPy after applying Vernon's explicit
validation so host and device reject the same unsupported signatures.

## Compiler implementation

Place matching C++ planners under
`source/include/mlir/Dialect/Vernon/Transforms/` and
`source/lib/Dialect/Vernon/Transforms/`.

### CPU and CUDA

- Lower static shape transforms through vector/LLVM aggregate extraction and
  reconstruction.
- Lower reductions with deterministic scalar combination order.
- Lower `where` to scalar/vector select operations.
- Lower casts through explicit arithmetic conversion operations.
- Keep CUDA static Tensor-by-value entry restrictions independent from local
  Tensor operation support.

### SPIR-V graphics and compute

- Use scalarized aggregate fallbacks for all legal static shapes.
- Use native vector shuffle, matrix, or group operations only as verified
  optimizations.
- Never construct SPIR-V vectors with unsupported component counts.
- Ensure vertex, fragment, and compute stages share the same operation
  lowering.
- Validate required capabilities for integer widths, `f16`, and `f64`.

### Reduction ordering

Initial reductions use a deterministic sequential logical order on every
backend. Parallel tree reductions may be added later behind an explicitly
documented numerical-reproducibility policy. Do not silently produce different
association rules per backend.

## API-specific semantics

### reshape

- Require positive static result extents.
- Support at most one inferred `-1`.
- Require equal source/result element counts.
- Preserve row-major logical order.

### transpose and swapaxes

- Default `transpose` reverses dimensions.
- Explicit permutations must contain every axis exactly once.
- `swapaxes` normalizes and swaps two axes.

### squeeze and expand_dims

- `squeeze(axis=None)` removes every extent-one dimension.
- Explicit squeeze axes must all have extent one.
- `expand_dims` accepts normalized insertion axes using NumPy ordering.

### broadcast_to

- Reuse the existing trailing-dimension broadcast planner.
- Reject writes or mutable aliases through a broadcasted Value; this operation
  produces an immutable Value, not a Storage view.

### reductions

- `axis=None` reduces all dimensions.
- Reducing every dimension produces a scalar unless `keepdims=True`.
- An empty axis tuple performs no reduction and preserves shape.
- Define behavior for zero-sized dimensions only when Vernon permits zero
  static extents; otherwise retain the current positive-extent restriction.

### where

- Condition must be bool or explicitly convertible under the language rule.
- Condition and both values broadcast to one result shape.
- Both branches are Values; side-effecting lazy branch evaluation is not part
  of this intrinsic.

### concatenate and stack

- Require a compile-time sequence of Tensor Values.
- `concatenate` requires equal non-concatenated dimensions.
- `stack` requires identical input shapes and inserts one axis.

### argmin and argmax

- Define first-index tie behavior matching NumPy.
- Reject empty reduction domains.
- Preserve deterministic traversal order.

### einsum

The first version supports:

- explicit output subscripts;
- alphabetic labels only;
- static shapes;
- repeated labels representing diagonal/contraction;
- ellipsis only after batch broadcasting is implemented in the generic
  contraction planner.

Reject unsupported syntax rather than falling back to host execution.

## Verification

### Differential shape and dtype tests

For every planner, compare valid and invalid cases with NumPy:

- ranks zero through at least five;
- negative and tuple axes;
- `keepdims`;
- singleton and broadcast dimensions;
- reshape inference;
- arbitrary transpose permutations;
- scalar/Tensor combinations;
- dtype combinations.

Property-generated tests should use bounded static shapes to avoid large
allocations.

### Frontend tests

- Verify inferred MLIR result types and canonical intrinsic attributes.
- Verify invalid axes, shapes, permutations, broadcasts, and dtypes produce
  stable `CompileError` diagnostics.
- Verify aliases (`power`/`pow`, `clip`/`clamp`) emit identical canonical IR.
- Verify aggregate-element Tensors are rejected by numeric operations until a
  numeric field is selected.

### Runtime backend parity

Compare results with NumPy on:

- CPU;
- CUDA;
- Vulkan compute;
- OpenGL compute;
- OpenGL ES compute;
- DirectX compute;
- OpenGL/OpenGL ES/Vulkan/DirectX vertex and fragment stages where applicable.

Include:

- rank greater than two;
- element counts greater than four;
- broadcasting before and after shape transforms;
- reductions over leading, middle, trailing, multiple, and all axes;
- `where` with independently broadcast condition/value shapes;
- contraction batches;
- expected target rejection for unavailable dtypes.

Use tolerances appropriate to dtype and operation. Exact integer/bool results
must use exact comparison.

### Full regression

After focused tests:

- run the complete Python suite;
- build and run all compiler tests;
- build and run all runtime/provider tests;
- regenerate and validate pipeline assets;
- verify source parameter names remain stable when generated artifacts use
  internal names.

## Deferred operations

Do not include these in the initial NumPy Core milestone:

- one-argument `where`/`nonzero`;
- `unique`;
- general boolean indexing with compacted dynamic results;
- dynamic-shape concatenate/stack;
- `sort` and `partition`;
- `cumsum`/`cumprod`;
- FFT;
- matrix inverse, determinant, decomposition, eigenvalue, and SVD APIs.

They require separate policies for dynamic output, temporary Storage,
parallelism, numerical stability, or backend libraries.

## Implementation order

1. Normalize public names: `minimum`, `maximum`, `power`, `clip`, and reduction
   `min`/`max`.
2. Add shared axis normalization and static index-map planners.
3. Implement `reshape`, `transpose`, `swapaxes`, `squeeze`, `expand_dims`, and
   `broadcast_to`.
4. Add reduction planner and `sum`, `prod`, `min`, `max`, `mean`, `any`, and
   `all`.
5. Add comparisons, logical broadcasting, `where`, and `astype`.
6. Add `concatenate`, `stack`, basic slicing, `take`, `argmin`, and `argmax`.
7. Generalize contraction planning and add `tensordot`, then constrained
   `einsum`.
8. Optimize native vector/matrix/reduction paths only after cross-backend
   scalarized correctness tests pass.

## Completion criteria

- Public operation names and accepted signatures have documented NumPy
  semantics.
- Python and C++ planners produce identical shape/index plans.
- Frontend inference and every backend lowering consume those plans.
- No backend-specific physical representation changes logical shape or dtype.
- Numeric aggregate operations clearly reject non-numeric Tensor elements.
- Focused NumPy differential tests pass on all available compute and graphics
  backends.
- Complete Python, compiler, runtime, provider, and asset suites pass.
