# Local Value updates and static structure

Status: proposal. Nothing here is current language, compiler, or autodiff
contract. Accepted parts move into `language/contract.md`,
`language/tensor_view.md`, and `autodiff/contract.md` before implementation.

## 0. Problem

A batched physics solver needs three things the language does not have:

1. **Updating a local matrix.** Factorizations (Cholesky, LDLᵀ), the
   articulated-body algorithm and Newton steps update entries of a local
   matrix in loops. Today a `Tensor` Value cannot be updated
   (`indexed assignment requires TensorView storage`), and the only workaround
   is global scratch Storage: an 18×18 Cholesky per env costs 32 ms for 4096
   envs on CUDA.
2. **Block access.** Spatial algebra works on 3×3 and 6×6 blocks. A view or
   matrix can be indexed only one scalar at a time, and a matrix cannot be
   built from computed rows.
3. **Compile-time structure.** Algorithms whose shape is fixed per variant
   (per robot, per block mask) need loops expanded at compile time, so that
   indices become constants and Values scalarize.

Sparsity is designed separately in [`sparse_tensors.md`](sparse_tensors.md),
which builds on Parts A and C here. This design keeps the three semantic
categories (Value, Storage, Resource) and keeps Tensor an immutable Value.

## 1. Design summary

| Part | Feature | Semantic change |
| --- | --- | --- |
| A | `a[i, j] = v`, `a.field = v` on local Values | Sugar for rebinding to a functionally updated Value |
| B | Leading-index and static-slice access on Values and views; `vd.zeros`, `vd.eye`, row construction | New Value operations; view loads of sub-Tensors |
| C | `vd.static(...)` loops, branches and locals; constant-Value specializations | Explicit compile-time expansion before typing |
| D | Sparsity | Moved to `sparse_tensors.md` |
| E | Autodiff | Rules for insert, extract and slice; a replay residual policy for Value code |

Parts A and B unblock dense solvers. Part C makes compile-time structure
explicit and makes AD of the expanded code straight-line. Static-pattern sparse
tensors are built on A and C.

## 2. Part A: local Value updates

### 2.1 Semantics

An assignment whose target is rooted at a **local name bound to a Value**
rebinds that name:

```text
a[i, j] = v        ==>  a = insert(a, (i, j), v)
a[i] = row         ==>  a = insert(a, (i,), row)          # Part B leading index
a[0:3, 3:6] = blk  ==>  a = insert_slice(a, static_box, blk)
b.vel = v          ==>  b = with_field(b, "vel", v)
b.vel[2] = z       ==>  b = with_field(b, "vel", insert(b.vel, (2,), z))
s[k].pos = p       ==>  s = insert(s, (k,), with_field(s[k], "pos", p))
a[i, j] += v       ==>  t = (i, j); a = insert(a, t, a[t] + v)
```

- Paths compose left to right: Tensor indices, static slices and Struct
  fields.
- Value semantics: after `c = a; a[0] = 1.0`, `c` is unchanged. There are no
  aliases between Values, so no alias analysis is needed.
- Evaluation order follows Python: the right-hand side first, then the target
  path's index expressions, each exactly once.
- The rebinding obeys the existing control-flow convergence rules. A Value
  updated in a loop is loop-carried, and one updated in a branch merges like
  any other local.
- Parameters are Values passed by value. Updating a parameter inside a
  `@vd.func` changes only the callee's binding, and the callee returns the
  result. There are no `inout` parameters.
- Targets rooted at a TensorView keep their current meaning: a Storage store.
  A target rooted at anything else (a Tuple, a params value, a Resource, or a
  Struct field reached through a TensorView element) is rejected with its own
  diagnostic.
- Tuple elements are not updatable. Tuples stay structural values built by
  construction and destructuring.

### 2.2 Indices and bounds

- An index with a compile-time value (a literal, a specialization, or a
  `vd.static` local from Part C) is bounds-checked at compile time.
- A dynamic index that is out of bounds is a contract violation, as a dynamic
  `range` step of zero is today. CPU and CUDA report it through the existing
  runtime-assertion path. SPIR-V targets cannot report it, so for them we must
  pick one fixed behavior (open question 1).

### 2.3 IR and lowering

Two typed ops carry the semantics:

```text
%t1 = vernon.value.insert %t0[%i, %j], %v        : tensor<...>, f32
%t1 = vernon.value.insert_slice %t0[box], %blk   : tensor<...>, tensor<...>
```

Plus the existing extract op and Struct `with_field`. Lowering guarantees:

1. **In place when dead.** An insert whose source Value has no later use
   updates the same storage. A copy is materialized only when the old Value is
   still live. Loop-carried updates reuse their `iter_args` slot. This is
   MLIR one-shot bufferization over function-local memory; the guarantee is
   normative, so performance does not depend on optimizer luck.
2. **Constant indices become registers.** When every access to a Value has
   compile-time indices (always the case after Part C expansion), it is
   scalarized: `insertvalue`/`extractvalue` on LLVM and NVVM,
   `OpCompositeInsert` on SPIR-V. Entries that are never read are dead and
   cost nothing.
3. **Dynamic indices use a private array.** LLVM uses an `alloca` (CUDA local
   memory). SPIR-V uses an `OpVariable` in the `Function` storage class with
   `OpAccessChain`. The compiler already materializes this function-local copy
   for dynamic reads (`compiler/design.md`, language representation
   boundary); stores reuse it.
4. **Pressure is reported, not hidden.** Reflection records each entry's
   private bytes per invocation, plus target spill counts where the toolchain
   reports them (ptxas `-v`). A large private footprint is a diagnostic note,
   not an error.

## 3. Part B: blocks and construction

### 3.1 Partial indexing and static slices

- `x[i]` on a rank-r Tensor Value with r ≥ 2 yields a rank r−1 Value (NumPy
  leading-index semantics). More generally, k leading integer indices yield
  rank r−k.
- Static slices `x[a:b, c:d]` require compile-time bounds and a step of 1, and
  produce a static shape. Static slices and integer indices may be mixed. A
  start that is dynamic while the size is static is deferred to a
  `vd.dynamic_slice(x, starts, sizes)` intrinsic (open question 4).
- The same forms on a TensorView in a load position **load a Value**:
  `m[e]` on a `(dyn, 4, 4)` view yields `Matrix[f32, 4, 4]`. In a store
  position they store a Value. This follows the existing rule that a view load
  materializes a Value. Passing a subview to a helper as Storage stays a host
  `view(...)` operation.
- `TensorView[Matrix[f32, 4, 4], (dyn,)]` and `TensorView[f32, (dyn, 4, 4)]`
  are one type by nested-Tensor canonicalization, so either binds an
  `(N, 4, 4)` storage. Today the first form is rejected ("rank 3, expected
  1"), which contradicts the contract; that is a bug to fix independently.

### 3.2 Construction

- `vd.zeros(T, shape)`, `vd.full(T, shape, value)` and `vd.eye(T, n)` are
  Value constructors with compile-time shapes.
- `vd.Matrix([r0, r1, ...])` and `vd.Tensor([...])` accept computed Values of
  equal shape as elements, not only literals. The result shape is the outer
  count followed by the element shape.

## 4. Part C: static evaluation

Specialization is the language's only compile-time mechanism
(`language/static_specialization.md`). Part C extends it with explicit
expansion instead of adding a second mechanism.

### 4.1 Static constructs

```python
for k in vd.static(range(N)):        # unrolled; k is a compile-time i32
for row in vd.static(TABLE):         # iterates the leading axis of a constant
if vd.static(TABLE[k] >= 0):         # static branch, same rules as section 6.2
i = vd.static(TABLE[k])              # compile-time local
while vd.static(i >= 0):             # static loop; condition is compile-time
    ...
    i = vd.static(TABLE[i])
```

Rules:

- `vd.static(e)` requires `e` to be a compile-time expression. The
  compile-time expression subset gains constant Tensor/Tuple/Struct values,
  constant indexing into them, `.shape`, and static locals.
- A name assigned through `vd.static(...)`, or a static loop variable, is a
  compile-time local. Assigning the same name both through `vd.static(...)` and
  dynamically in one function fails with `static local is also assigned a
  runtime value`.
- Expansion runs before typing, like static branches. Each unrolled body is
  typed on its own, so the same `h[k, i]` becomes a different constant index in
  each copy.
- `break` and `continue` inside a static loop are allowed only under a static
  condition. A runtime `break` inside a static loop is rejected; the loop
  should be a runtime loop instead.
- **Expansion budget.** Total expanded statements per entry are capped
  (default 65,536), with the diagnostic `static expansion exceeds budget`.
  This fails closed instead of silently producing a huge artifact.
- Literal `range(...)` loops keep their current runtime-loop meaning. Unrolling
  is never implicit, because implicit unrolling would change the code size and
  AD control flow of existing programs.

Precedent: Taichi `ti.static` and NVIDIA Warp `wp.static` (Warp ≥ 1.5) use the
same explicit marker.

### 4.2 Constant Values

- **Module constants.** A module-level `NAME = vd.Tensor([...])` (or a Tuple,
  or a shared Struct constructor) whose leaves fold to scalars is a constant
  Value. This extends today's int/float/bool module constants.
- **Specializations.** `vd.specialization(name, vd.constant)` accepts any host
  Value of a `@vd.struct(shared=True)` type, a Tensor or a Tuple, with scalar
  leaves. Its identity in the specialization key is its type identity plus its
  canonical ABI bytes. `variants=` enumerates robots or masks exactly as it
  enumerates materials today.
- Indexing a constant Value with a compile-time index folds. Indexing it with
  a runtime index lowers to a constant table.
- Host code must not generate source text. Structure reaches the compiler as a
  specialization value, consistent with the rejection of generated wrapper
  sources.
- Iterating a static sparse pattern (`sparse_tensors.md` section 4) is
  compile-time by construction and needs no `vd.static` marker. `vd.static`
  is only for loops over ordinary compile-time values.

## 5. Part D: sparsity (moved)

Sparsity is a general feature for physics and deep learning, designed in
[`sparse_tensors.md`](sparse_tensors.md). A static-pattern `SparseTensor` is
a Value, built on Part A (updates of pattern entries) and Part C (compile-time
pattern iteration). Runtime patterns are Storage.

Spatial algebra (6-vectors, spatial inertia, motion and force transforms) and
the articulated algorithms (CRBA, ABA, contact Jacobians) belong to the Vernon
engine's physics module as a DSL library. That library derives the mass-matrix
pattern from the robot's kinematic tree on the host. Kernels see only a
pattern, never a parent array. Topology is static per robot. Numeric
parameters (masses, inertias, friction, motor constants) stay runtime Storage,
so domain randomization never recompiles.

## 6. Part E: autodiff

### 6.1 Rules

All are linear in their floating operands. Integer indices are
non-differentiable.

```text
y = insert(t, idx, v):   t̄ += insert(ȳ, idx, 0)      v̄ += ȳ[idx]
y = t[idx]:              t̄  = insert(t̄, idx, t̄[idx] + ȳ)
y = insert_slice(t,b,s): t̄ += insert_slice(ȳ, b, 0)  s̄ += ȳ[b]
y = t[b] (slice):        t̄  = insert_slice(t̄, b, t̄[b] + ȳ)
y = with_field(s,f,v):   s̄ += with_field(ȳ, f, 0)    v̄ += ȳ.f
```

The adjoint code updates local adjoint Values with the same insert machinery,
so Part A is also what keeps generated backward kernels in registers.

### 6.2 Residuals

- The linear ops above save no floating residuals. A dynamic index whose value
  cannot be recomputed is saved as an integer.
- Statically expanded code is straight-line, so it needs no loop tape. This is
  the main reason structural algorithms should use Part C instead of runtime
  loops.
- **Replay policy (proposed default for Value-local code).** The backward
  invocation recomputes the kernel's Value-local primal from its versioned
  Storage inputs, then runs the reverse sweep in registers (Warp-style replay).
  Per-element taping of an unrolled factorization would cost hundreds of
  scalars per env of device memory. Taping remains for Storage versions and
  runtime-loop control. How this composes with the current residual planning
  (`autodiff/contract.md` section 12, `planning_policy`) is open question 3.
- Runtime loops that update a Value with dynamic indices stay inside the
  current AD loop subset (literal-bounded loops, one leading `break` guard).
  Their tape bound is trip count × Value size and is checked against the
  existing bounded-tape rule.

### 6.3 Out of scope

Differentiating through an iterative contact solver by unrolling its
iterations is accurate but expensive. System identification and sim-to-real
gradients want implicit differentiation at the solver's fixed point, which
needs custom compute VJP declarations; those remain deferred. Parts A–E make
that later rule cheap to write.

## 7. Rejected alternatives

- **Mutable local Storage** (`vd.local_storage`, a private-address-space
  TensorView). Pass-by-reference to helpers is its only advantage over Part A,
  and it costs alias and effect rules, mutation functionalization for AD, and
  a fourth address space in reflection. Revisit only if a measured workload
  needs out-parameters.
- **Implicitly unrolling literal ranges.** This changes code size, cache
  identity and AD control flow of existing programs.
- **Compile-time function evaluation** (`@vd.func(static=True)`). This is a
  general partial evaluator. Host tooling precomputes topology data instead.
  Revisit with the "deterministic constant evaluation" roadmap item.
- **Generating per-robot source text.** This was already rejected for
  wrappers in `static_specialization.md`. A constant specialization value
  carries the same information with a stable identity.

## 8. Phasing

1. Part A plus Part B forward on every compute backend, with CPU/GPU parity.
2. Part E rules and the replay residual policy, with finite-difference tests.
3. Part C static evaluation and constant specializations.
4. Sparse tensor phase S1 (`sparse_tensors.md` section 11), then the Vernon
   physics library: spatial algebra, CRBA, ABA, contact Jacobians, validated
   against the current CPU `multibody` and `articulation` code.

## 9. Acceptance tests

- Update forms (index, slice, field, nested path, augmented) agree with a
  NumPy reference. Value-semantics copies leave the earlier binding unchanged.
- A dynamic-index update inside runtime loops runs on CPU, CUDA, Vulkan and
  DirectX. An out-of-bounds dynamic index produces the specified behavior on
  each target class.
- In-place guarantee: the IR test checks that a dead-source insert emits no
  copy, and that a live-source insert emits exactly one.
- An all-constant-index kernel lowers to no private array (scalarized) on
  every backend.
- `m[e]` view load and store, mixed slice and index, and the canonicalized
  Matrix-element view bind to the same storage.
- The static expansion budget, static/runtime local conflicts, and runtime
  `break` in a static loop produce their diagnostics.
- A constant specialization's cache identity changes with its bytes and its
  type identity, and is stable across checkouts.
- VJP finite-difference checks for every Part E rule, a statically expanded
  dense Cholesky, and a runtime-loop Cholesky.
- Performance gate: 4096-env 18×18 solve on CUDA at least 10× faster than the
  current 32 ms global-scratch baseline.

## 10. Open questions

1. **Out-of-bounds dynamic index on SPIR-V.** Options: (a) loads return zero
   and stores are dropped, (b) indices are clamped, or (c) dynamic indices into
   local Values are rejected on SPIR-V. (a) is cheap and memory-safe, but it
   silently hides bugs on GPU.
2. **Static expansion budget.** Is a default of 65,536 statements, overridable
   per entry, right?
3. **Replay residual policy.** Does it become the default for Value-local
   code, or a new `planning_policy` value?
4. **`vd.dynamic_slice`.** Is it needed in phase 1? Contact-row blocks would
   use it.
5. **Library home.** Does the articulated-body library live in Vernon
   (proposed) or in a VernonDSL `lib` package?
