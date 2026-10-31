# Sparse tensors

Status: proposal. Nothing here is current contract. Adopting it amends
`language/contract.md` section 3.5 ("Dense-only core") and adds sparse rules to
`autodiff/contract.md`. It builds on local Value updates and static expansion in
[`value_updates_and_static_structure.md`](value_updates_and_static_structure.md).

## 0. Why sparsity is a language feature

Writing CSR by hand over dense Storage, which is today's rule, works for one
kernel. It fails as a general feature because:

- the compiler cannot check that writes stay inside the pattern, or that two
  kernels agree on one encoding;
- the gradient of a sparse matrix is dense unless autodiff knows the pattern.
  A 1M-nonzero weight would get an N² gradient;
- one algorithm cannot run on several encodings (CSR, block, N:M, ragged), so
  every library op is written once per format;
- a pattern known at compile time cannot be scalarized into registers.

## 1. Workloads

| Workload | Pattern known | Granularity | Lives in |
| --- | --- | --- | --- |
| Robot joint-space mass matrix, contact Jacobian rows | compile time | element / block | Value (registers) |
| Cloth/FEM stiffness, mesh or graph adjacency (GNN) | when the mesh loads; fixed after | element / 3×3 block | Storage |
| Pruned weights, 2:4 structured sparsity | when the model loads | N:M | Storage |
| Block-sparse attention (causal, sliding window, BigBird) | compile time or load time | block | Storage |
| MoE routing, variable-length sequence batches | every step | ragged rows | Storage |
| Top-k / dynamic sparse attention | every step | block / element | Storage |
| Broadphase pairs, contacts, sparse voxels / point clouds | every step | coordinate (COO) | Storage |

Two axes describe every row. The **binding time** of the pattern (compile time
or runtime) decides Value versus Storage. The **format** (how a pattern is
encoded) is independent of binding time.

## 2. Model

A sparse tensor has four separable parts:

1. **Logical tensor.** Element type `T` and shape, as for a dense Tensor.
   Entries not in the pattern are **structural zeros**.
2. **Format.** One *level type* per stored dimension, plus optional blocking.
   This is the TACO / MLIR `sparse_tensor` vocabulary:
   - `dense`: every coordinate is present;
   - `compressed`: positions plus coordinates, the CSR inner level;
   - `singleton`: one coordinate per parent entry, the COO inner level;
   - `n_of_m(n, m)`: at most n of every m are present, as in 2:4;
   - `blocks=(b0, b1, ...)`: levels index blocks, and the element is a dense
     `b0 × b1` Tensor (BSR).

   Properties `ordered` (coordinates sorted within a segment) and `unique` (no
   duplicates) are part of the format. CSR is `(dense, compressed)`, COO is
   `(compressed(non-unique), singleton)`, and BSR is CSR with `blocks=(16, 16)`.
   Ragged batches are `(dense, compressed)` whose element is a row Tensor: MoE
   tokens per expert are `Sparse[Vector[f32, D], (E, T), (dense, compressed)]`.
   Index widths are format parameters `pos=` and `crd=`, separately for
   positions and coordinates (proposed defaults in section 13.4).
3. **Pattern.** The concrete positions and coordinates. It is **immutable**
   once a sparse value or storage exists.
4. **Values.** A dense sequence of `nnz` elements in pattern order. They are
   mutable only where Storage access allows it.

The element type `T` may be any ABI-stable Value, including blocks and
Structs.

## 3. Types

No new semantic category is added. A sparse tensor is either a Value or a
Storage, decided by the binding time of its pattern.

### 3.1 Static pattern: `SparseTensor` Value

```python
vd.SparseTensor[T, shape, PATTERN]
```

- `PATTERN` is a compile-time pattern constant: a module constant or
  `vd.specialization(name, vd.pattern)`. Its identity in type and cache keys is
  its canonical bytes (format plus positions and coordinates).
- Physically it is `Tensor[T, (nnz,)]` with every pattern query folded at
  compile time. With the local-update lowering it lives in registers, and
  structural zeros never exist.
- Its canonical ABI is the `nnz` values in pattern order. So
  `TensorView[SparseTensor[f32, (18, 18), H], (dyn,)]` stores per-env mass
  matrices at nnz floats each, not 324.
- It is a distinct type from `Tensor`. `vd.sparse.to_dense(x)` and
  `vd.sparse.project(dense, PATTERN)` (which keeps only pattern entries) are
  the explicit conversions.

### 3.2 Runtime pattern: `SparseStorage` and `SparseView`

```python
vd.SparseView[T, shape, FORMAT, access]
```

- `FORMAT` is compile-time. Positions, coordinates, shape (`vd.dyn` allowed)
  and nnz are invocation-time descriptor data, as dense views already work.
  Compiled artifacts never specialize on a concrete pattern.
- `access` applies to the values. The pattern is always read-only inside
  kernels.
- The host owner is `SparseStorage`: one values `TensorStorage` and one
  integer `TensorStorage` per positions or coordinates array, at the format's
  `pos`/`crd` widths. Several sparse storages
  may share one pattern, which is safe because patterns are immutable. This is
  how a gradient shares its primal's pattern.
- Binding expands one source argument into its level descriptors, the same
  mechanism aggregate TensorView leaves use today. Reflection records the
  expansion.
- Host interop: construct from `scipy.sparse` and coordinate or mask arrays;
  export to `scipy.sparse`. Zero-copy `torch.sparse_*` depends on DLPack
  support, a separate item.

## 4. Operations

The same source works on both binding times. A static pattern resolves every
operation at compile time; a runtime pattern executes it.

| Operation | Static pattern | Runtime pattern |
| --- | --- | --- |
| `a[i, j]` read | compile-time resolved: an entry, or the constant zero | O(1) dense levels, O(m) N:M; rejected on compressed levels (proposal 13.2) |
| `a[i, j] = v` | allowed only for pattern entries, checked at compile time: `(i, j) is not in pattern` | same levels as reads; absence is a contract violation |
| `e = vd.sparse.find(a, i, j)` | compile-time entry or invalid handle | O(log k) binary search; requires `ordered`; `e.valid` reports presence |
| `for e in vd.sparse.entries(a)` and `entries(a, row=i)` | expanded at compile time; `e.index` is compile-time | runtime loop over positions |
| `a[e]`, `a[e] = v` with an entry handle | O(1) | O(1) |
| `vd.sparse.has(a, i, j)` | compile-time bool | runtime bool |
| `vd.sparse.intersect(r0, r1)`, `union(r0, r1)` | compile-time pairs of entries | merge loop; requires `ordered` |
| `vd.sparse.nnz(a)`, `a.row(i)` | constants | descriptor reads |

- Iterating a static pattern is inherently compile-time, so loops over
  `entries` of a `SparseTensor` need no `vd.static` marker. Their loop
  variables are compile-time.
- Updates of a `SparseTensor` Value use local-update sugar (`a[e] = v`,
  `a[i, j] = v`) and keep the pattern.
- No operation changes a pattern. A different pattern means a different
  value or storage.

**Compile-time pattern functions.** A closed, deterministic set of pure
functions over pattern constants joins the compile-time expression subset:
`transpose`, `lower`, `upper`, `union`, `intersect`, `product(A, B)` (the
pattern of A·B), and `cholesky_fill(A)` (the symbolic factorization). They need
no general compile-time evaluation, and the frontend implements them natively.
Heuristic choices such as fill-reducing orderings stay with the host
(proposal 13.1).

## 5. Building patterns

- **On the host.** `vd.sparse.Pattern.from_coordinates`, `from_mask`,
  `from_scipy`, and domain helpers in domain libraries. A robot's tree pattern
  comes from Vernon's URDF loader, so kernels never see a parent array.
- **On the device.** Patterns that change every step are built in two phases.
  A count kernel runs first, then a prefix sum gives positions, then a fill
  kernel writes coordinates. The results are ordinary TensorStorages, and
  `SparseStorage.assemble(format, positions, coordinates, values,
  validate=...)` wraps them without copying. This depends on the scan and
  compaction primitives (a separate blocker).
- **Capacity.** A device-built pattern has a fixed nnz capacity and a
  device-side count. Overflow sets an error flag that the next completion
  reports; entries are never silently dropped.
- **Validation.** Safety properties (monotonic positions, in-bounds
  coordinates) are always guaranteed. The semantic properties `ordered` and
  `unique` are guaranteed according to where the pattern came from
  (proposal 13.5).

## 6. Example: generic static sparse Cholesky

The host derives `H` from the robot's kinematic tree, and the kernel is generic
over any static pattern:

```python
H = vd.specialization("mass_pattern", vd.pattern)    # lower triangle, from URDF
L = vd.sparse.cholesky_fill(H)                       # tree patterns: L == H (no fill)
N = H.shape[0]


@vd.func
def cholesky(h: vd.SparseTensor[vd.f32, (N, N), H]) -> vd.SparseTensor[vd.f32, (N, N), L]:
    l = vd.sparse.project(h, L)
    for e in vd.sparse.entries(l):                    # row-major, j <= i; compile-time
        i, j = e.index
        s = l[e]
        for p, q in vd.sparse.intersect(l.row(i), l.row(j), below=j):
            s = s - l[p] * l[q]
        l[e] = vd.sqrt(s) if i == j else s / l[j, j]
    return l
```

- No parent array, no physics knowledge, and no `vd.static` appear in the
  kernel.
- For the quadruped, `L` has 117 entries, all held in registers, with no fill.
  For an arbitrary pattern the same code runs with whatever fill
  `cholesky_fill` computes.
- Featherstone's tree LTL (IJRR 2005) is the special case where the pattern is
  a tree and the elimination runs leaves first. The library may provide that
  ordering as `vd.sparse.reverse_order`, but the algorithm stays generic.

The same `cholesky` source compiles against a `SparseView` (a runtime pattern),
where it is correct but serial per row. Large runtime-pattern factorization
needs parallel algorithms (level scheduling, supernodes), which are library
work.

## 7. Deep learning shapes

- **Block-sparse attention.** The mask is `(dense, compressed)` with
  `blocks=(64, 64)`. A workgroup per query block iterates
  `entries(mask, row=qb)` and computes only present key blocks. A causal or
  sliding-window mask is a static pattern, so block skipping is resolved at
  compile time; a per-request mask is a runtime pattern.
- **MoE.** Routed tokens form a ragged `SparseView[Vector[f32, D], (E, dyn),
  (dense, compressed)]`, built on the device every step (section 5). A grouped
  GEMM iterates experts by segment.
- **2:4 weights.** `(dense, n_of_m(2, 4))`. The portable lowering is scalar.
  CUDA sparse tensor-core MMA is a capability-gated lowering of library matmul,
  not a language change.
- **GNN and mesh operators.** CSR `SparseView`. SpMM and SDDMM are library ops
  whose gradients come from section 8.

## 8. Autodiff

- **The pattern is non-differentiable** (integer data). Values are
  differentiable.
- **Sparse tangents.**
  `T(SparseTensor[S, shape, P]) = SparseTensor[T(S), shape, P]`. A
  `SparseView` gradient is a new values storage that shares the primal's
  immutable pattern. Gradients are never densified implicitly.
- **Structural zeros are constants, not parameters.** Reading an absent entry
  yields a constant zero with no gradient. This is the defining difference
  from masked dense math, and the reason pruned weights stay pruned under
  training.
- **Rules.** Entry reads and writes are extract and insert on the values
  sequence, using the local-update autodiff rules. A runtime lookup becomes a
  gather on the found slot, or a constant when the entry is absent.
- **Loops bounded by the pattern become an accepted AD loop class.** Their trip
  counts come from immutable integer pattern data, so the backward pass
  re-derives them and needs no control tape. Today only literal-bounded loops
  are accepted.
- **Accumulation.** Backward through `y = A x` scatters into `x̄` along the
  pattern's columns. This uses the existing accumulation plans (atomic, or
  deterministic reduction). A format may carry a precomputed transpose
  (`with_transpose=True`) so that the backward pass is a conflict-free gather.

## 9. Lowering

- **Static patterns.** Values are a register-resident `Tensor[T, (nnz,)]`, and
  every pattern query folds. No pattern data exists at runtime.
- **Runtime patterns.** Each level lowers to the standard position and
  coordinate walk (`pos[i] .. pos[i+1]`). The IR uses MLIR `sparse_tensor`
  encoding attributes as the format representation, so tooling and interchange
  share one vocabulary. Kernels are lowered by Vernon's own lowering, not
  MLIR's automatic sparsifier, because they must reach SPIR-V and must express
  solvers, not only algebra.
- **Optimizer.** `SparseView` accesses give the future optimization IR's
  `SparseTask` its indirect-access summaries (`optimization_ir.md` section
  8.3).

## 10. Rejected alternatives

- **Masks over dense tensors.** They save no memory or compute, and their
  gradients flow to masked entries.
- **Static access patterns only.** This was the previous draft of
  `value_updates_and_static_structure.md`: dense matrices with sparse loops. It
  has no compact Storage ABI, no checked writes, no sparse tangents, and does
  not extend to runtime patterns.
- **SNode / transparent sparse layout trees (Taichi).** They hide allocation
  and cost behind dense syntax, and the contract already rejects them.
- **A single CSR type.** It cannot express BSR, N:M, ragged or COO. Per-level
  formats compose instead.
- **An automatic sparsifier (TACO / MLIR style, "write dense, compile sparse")
  as the main model.** It is excellent for tensor algebra but cannot express
  factorizations and solvers, and its codegen targets LLVM, not SPIR-V. It may
  later lower library algebra ops over these same types.

## 11. Phasing

1. **S1.** Static `SparseTensor`, host `Pattern`, entries/has/lookup/project,
   and compile-time pattern functions. This is the physics path: mass matrix
   and contact Jacobians.
2. **S2.** Runtime `SparseStorage`/`SparseView` with `dense`, `compressed`,
   `singleton` and `blocks`; scipy interop; sparse tangents; pattern-bounded
   AD loops.
3. **S3.** Device-built patterns, capacity and overflow, and `assemble`
   validation. Requires scan and compaction.
4. **S4.** `n_of_m`, runtime co-iteration, and library ops: SpMV, SpMM, SDDMM,
   grouped GEMM, block-sparse attention, parallel sparse factorization.

## 12. Acceptance tests

- Every format round-trips with `scipy.sparse`. Invalid patterns fail
  `assemble` with stable reasons.
- A static-pattern write outside the pattern fails at compile time. A
  runtime-pattern write to an absent entry is reported on CPU.
- `SparseTensor` lowers with no private array and no pattern storage (checked
  by reflection) on every backend.
- The generic `cholesky` matches `numpy.linalg.cholesky` for the quadruped
  tree pattern and for a random static pattern with fill.
- A one-source kernel runs against both a static pattern and a `SparseView`
  and gives equal results.
- Gradients are sparse with the primal's pattern; finite differences agree;
  absent entries get no gradient. Pattern-bounded loops differentiate without
  a control tape.
- Block-sparse attention with a static causal mask and with a runtime mask
  agree with a dense masked reference.

## 13. Proposed resolutions

Status: proposals, not accepted decisions. Each records the recommendation
and its reasoning for review.

### 13.1 Pattern functions: frontend for exact functions, host for heuristics

- **Proposal.** Structural functions with one correct answer (`transpose`,
  `lower`, `upper`, `union`, `intersect`, `product`, `cholesky_fill`) are
  evaluated by the frontend from one native implementation. The same
  implementation is exposed to Python for inspection. Heuristic choices
  (fill-reducing orderings such as AMD or nested dissection, block sizes,
  partitioning) are computed by the host and passed in as specializations.
- **Reason.** A host-passed derived pattern can disagree with its source
  silently. If `L` lacks a fill entry, writes there fail to compile, but reads
  return the structural zero, so the factorization is wrong with no error.
  Deriving `L` inside the compiler removes that failure mode, and the cache
  identity follows from the inputs alone. Heuristics have many valid answers
  and must not change with a compiler version, so the user owns them.
- **Bound.** Evaluation counts against the static expansion budget, because
  `product` can grow quadratically.

### 13.2 Random lookup: cost visible in syntax

- **Proposal.** `a[i, j]` is accepted where it is cheap: static patterns,
  `dense` levels and `n_of_m` levels. On a runtime `compressed` level it is a
  compile error that names `vd.sparse.find`.
  `e = vd.sparse.find(a, i, j)` returns an entry handle, with `e.valid`
  reporting presence. It requires `ordered`, and is rejected on unordered
  levels. Then `a[e]` reads or writes in O(1).
- **Reason.** A binary search hidden behind `[]` is a divergent GPU loop the
  author cannot see. A read-modify-write would search twice. And writing an
  absent entry is an error, so the caller has to handle "not found" anyway.

### 13.3 Fill value: structural zero only

- **Proposal.** Absent entries are always the zero of `T`. There is no
  per-tensor fill.
- **Reason.** Zero is the additive identity and annihilates under
  multiplication, which keeps `A·x`, `product` and fill patterns well defined.
  A `−∞` fill makes `A·x` produce `−∞·0 = NaN`. Sparse attention needs no
  fill, because softmax iterates present entries only, which is correct and
  faster than masking. Ops with other identities (logsumexp from `−∞`,
  max-plus semirings) take the identity as a parameter of the library
  reduction over `entries`. Autodiff is unaffected, since structural zeros
  are constants either way.

### 13.4 Index widths: separate position and coordinate widths

- **Proposal.** Defaults are `crd=vd.i32` and `pos=vd.u32`, following MLIR's
  split of position width from coordinate width.
  - Coordinates are user-visible indices and must mix with `range` and dense
    view indexing without casts, so they are `i32`.
  - Positions are internal offsets, never user arithmetic. `u32` raises the
    limit to 2³² nnz (16 GB of f32 values) at no ergonomic cost.
- **Deferred.** 64-bit widths need an `i64` scalar the language does not have,
  plus per-backend capabilities (OpenGL ES lacks 64-bit integers). Very large
  tensors are sharded across devices anyway. Because widths are format
  parameters, adding 64-bit later is a new format value, not a redesign.

### 13.5 `ordered` / `unique`: separate safety from semantics

- **Proposal.**
  - **Safety properties are always guaranteed.** Positions are monotonic and
    ≤ nnz, and coordinates are in bounds. Without them, other buffers could be
    accessed out of bounds. They are validated at `assemble`, or checked at
    use where validation is skipped. There is no opt-out.
  - **Semantic properties (`ordered`, `unique`) are guaranteed according to
    where the pattern came from:**
    - Host-built patterns (scipy, coordinate lists, masks) are canonicalized
      (sorted and deduplicated) or validated every time. It is a one-time
      cost.
    - Library device builders (count, prefix sum, fill) produce ordered,
      unique patterns by construction, and need no check.
    - Raw user arrays assembled on the device are validated by default, with
      `validate=False` as an opt-out for hot per-step paths. A violation then
      gives unspecified results but stays memory-safe, because the safety
      checks still apply.
- **Reason.** Always enforcing the semantic properties puts a validation kernel
  on every MoE or contact step. Never enforcing them lets co-iteration, `find`
  and duplicate-summing return wrong answers silently. Deciding by source keeps
  the hot paths cheap and the risky one checked.
