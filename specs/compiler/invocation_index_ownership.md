# Invocation index ownership

This document is the mathematical account of the ordinary-write injectivity
proof implemented in `VernonGlobalIdIndexProof.cpp`. Language residual
conditions are `vernon.dispatch_contract` in
[`language/contract.md`](../language/contract.md) §5. Runtime enforcement is
`validateDispatchContract` in `source/lib/runtime/dispatch_contract.h`.

The checker is a **sound, incomplete decision procedure** on a restricted
index language. Failure means “unproven”, not “there exists a collision”.
Success means: under the residual launch constraints emitted with the proof,
distinct invocations cannot produce the same index tuple.

## 1. Launch, invocations, and global id

A compute dispatch has workgroup count $G = (G_0, G_1, G_2)$ and workgroup
size $W = (W_0, W_1, W_2)$, all positive integers. An **invocation** is a pair
$(q, \ell)$ with workgroup id $q_a \in \{0,\ldots,G_a-1\}$ and local id
$\ell_a \in \{0,\ldots,W_a-1\}$. The **global invocation id** is

$$
g_a = q_a W_a + \ell_a \in \{0,\ldots, G_a W_a - 1\}.
$$

The reconstruction $g_a = q_a W_a + \ell_a$ is the unique mixed-radix encoding
of $g_a$ with lower digit extent $W_a$. The compiler treats an IR expression of
that form (with the function’s static $W_a$) as identical to `gid[a]`.

A **scalar** global id is a single integer that folds all three axes. The
invocation proof rejects it: it does not name a unique axis in
$g \in \mathbb{N}^3$.

Write $\mathrm{Inv}$ for the set of invocations of a concrete launch. The map
from $\mathrm{Inv}$ to $g$ is bijective.

## 2. The ownership claim

Let $i = (i_0,\ldots,i_{r-1})$ be the index tuple of an ordinary device
`TensorView` store, each $i_d$ an integer expression of $g$ (and of
launch-invariant data: constants, `get_shape` extents).

**Definition (invocation-owned write).** The store is invocation-owned on a
launch when the map

$$
\Phi : \mathrm{Inv} \to \mathbb{Z}^r, \qquad \Phi(\iota) = i\bigl(g(\iota)\bigr)
$$

is injective.

If $\Phi$ is injective, two distinct invocations cannot store to the same
element. That is the only property required of an ordinary (non-atomic,
non-accumulating) device write.

The claim is **not** a bounds proof. Out-of-range indices are a separate
TensorView check. Injectivity may hold on a larger integer lattice than the
allocation.

The claim is **conditional** on residual launch constraints. The compiler
records those constraints as `vernon.dispatch_contract`:

- `unit_grid_axes`: every listed axis $a$ must have $G_a = 1$;
- `requires_unit_workgroup`: every axis must have $W_a = 1$.

Axes that appear in the index proof (covered axes) are left unconstrained.
Axes that the index expression does not distinguish are constrained so that
they do not range over multiple invocations.

`requires_unit_workgroup` additionally forces every `unit_grid_axes` slot to
be set (parser): a unit-workgroup contract is a fully serialized launch
$G = W = (1,1,1)$.

## 3. Affine fragment

### 3.1 Term

An expression is in the **affine fragment** when it normalizes to

$$
t(g) = c + \sum_{a=0}^{2} \alpha_a g_a
$$

with $c, \alpha_a \in \mathbb{Z}$. Allowed constructors: constants, `gid[a]`,
reconstructed $g_a$, `addi`/`subi`, multiplication by a **constant**. Same-width
`bitcast` and `index_cast` of at most i32 are transparent.

`div`, `rem`, and `trunci` are **not** affine. They abort the affine path
(`invalidArithmetic`) rather than becoming “unknown”.

### 3.2 Unique-axis lemma

Say $t$ **uses a unique axis** $a$ when $\alpha_a \neq 0$ and $\alpha_b = 0$
for all $b \neq a$. Then

$$
t(g) = \alpha_a g_a + c,
$$

and $g_a \mapsto t(g)$ is injective on $\mathbb{Z}$.

### 3.3 Affine invocation ownership

Let $i_0,\ldots,i_{r-1}$ be store indices. Process dimensions in order:

1. If $i_d$ is affine and uses a unique axis $a_d$ that has not already been
   used, record $a_d$ as covered.
2. If $i_d$ is not affine (and not invalid arithmetic), treat it as an
   **unknown suffix** and ignore it, provided no later dimension returns to a
   proven unique axis (unknown may only trail a unique-axis prefix).
3. Scalar `gid` is rejected.
4. Duplicate unique axes are rejected (two dimensions both determined by
   $g_0$ is not a matching).

Let $A$ be the set of covered axes. If $A = \{0,1,2\}$, $\Phi$ is injective on
all of $\mathrm{Inv}$ with no extra grid constraint: each axis of $g$ appears
in exactly one proven coordinate, each with nonzero coefficient.

If $A$ is a nonempty proper subset, unused axes $b \notin A$ are emitted as
`unit_grid_axes`, and the static workgroup size on those axes must already be
$W_b = 1$. Then every invocation is distinguished by $g|_{A}$, and the
unique-axis prefix is injective on $\mathrm{Inv}$. The full tuple is then
injective: distinct invocations differ on the prefix.

**Trailing unknown after a complete three-axis prefix** is therefore sound:
the prefix already separates all invocations.

**Unknown in the middle** is rejected (a later unique axis after an unproven
coordinate is not a prefix).

This is lemma `proveAffineInvocationOwnedIndex`.

### 3.4 Strict invocation ownership

`proveStrictInvocationOwnedIndex` is the same unique-axis matching with no
unknown dimensions, no unit-axis residual, and $|A| = 3$. It is the
`scatter_add disjoint` obligation: exclusivity with a full, unconstrained
grid.

## 4. Mixed-radix fragment

### 4.1 Euclidean digits

On the unsigned naturals, for $E \ge 1$,

$$
L = \left\lfloor \frac{L}{E} \right\rfloor \cdot E + (L \bmod E),
\qquad 0 \le (L \bmod E) < E.
$$

`arith.divui` / `arith.remui` are this pair. Signed `divsi` / `remsi` are
toward-zero, not Euclidean, and are rejected.

A **positive extent** is either a compile-time constant $E \ge 1$ or a
`get_shape(s)[d]` term (`ShapeDim`). Dynamic extents are a **residual of
defined arithmetic**: if a dynamic extent is $0$ at runtime, `urem`/`udiv`
are undefined and the kernel is not in the model.

### 4.2 Unwrapped mixed radix

Fix $r \ge 2$ and extents $E_1,\ldots,E_{r-1} \ge 1$. Define
$\varphi : \mathbb{N} \to \mathbb{N}^r$ by

$$
\begin{aligned}
d_{r-1} &= L \bmod E_{r-1}, \\
L^{(r-2)} &= \left\lfloor \frac{L}{E_{r-1}} \right\rfloor, \\
d_{k} &= L^{(k)} \bmod E_{k}, \\
L^{(k-1)} &= \left\lfloor \frac{L^{(k)}}{E_{k}} \right\rfloor
\quad (k = r-2,\ldots,1), \\
q_0 &= L^{(0)}.
\end{aligned}
$$

The leading coordinate $q_0$ is **not** reduced modulo an $E_0$. Then
$\varphi(L) = (q_0, d_1, \ldots, d_{r-1})$ (store order $i_0 = q_0$,
$i_k = d_k$) is injective: the inverse is the positional polynomial

$$
L = q_0 \prod_{k=1}^{r-1} E_k
  + \sum_{j=1}^{r-2} d_j \prod_{k=j+1}^{r-1} E_k
  + d_{r-1}.
$$

No bound $L < \prod_{k} E_k$ is required. Distinct $L$ give distinct tuples on
all of $\mathbb{N}$. Extra threads with large $L$ write coordinates that may
be out of allocation bounds; they do not alias a smaller $L$.

Wrapping the leading digit, $q_0 \bmod E_0$, destroys injectivity on
$\mathbb{N}$ unless the launch also constrains
$L < \prod_{k=0}^{r-1} E_k$. That form is **not** accepted.

### 4.3 Composition with a unique-axis linear form

Let $L(g) = \alpha g_a + c$ be affine unique-axis with $\alpha \neq 0$.
If $L(\mathrm{Inv}) \subseteq \mathbb{N}$ (in particular $g_a \ge 0$,
$\alpha = 1$, $c = 0$: $L = g_a$), then

$$
\iota \mapsto \varphi\bigl(L(g(\iota))\bigr)
$$

is injective on $\mathrm{Inv}$ once unused axes of $g$ are constrained as in
§3.3.

Extents need not be the stored TensorView’s shape. Any positive constants or
`ShapeDim` terms that are equal as terms (same source value and dimension, or
same constant) are allowed. Injectivity is a property of $\varphi$, not of
in-bounds addressing.

The compiler reconstructs this form **from independently normalized index
terms**, not by searching sibling IR users. Two `get_shape` extracts of the
same source and dimension are the same `ShapeDim` term.

This is lemma `proveMixedRadixInvocationOwnedIndex`. It is tried only when
the affine path fails. Rank $1$ mixed radix is the affine fragment
($i_0 = g_a$).

Program `copy`/`add` kernels use this lemma: linearized $L = g_0$, digits
from `output.shape`.

## 5. Workgroup ownership and leader guards

A store is **workgroup-owned** when indices are affine in `workgroup_id`
(unique-axis matching, unused workgroup axes listed as `unit_grid_axes`)
rather than in $g$. Many local lanes of one workgroup then share the index,
so the store is **not** invocation-owned unless only one lane executes it.

A **leader guard** is a nest of `scf.if` whose condition is
`local_invocation_id[a] == c` with $0 \le c < W_a$, covering every axis with
$W_a > 1$ (or a scalar local id compared against the linearized lane count).
Then at most one lane per workgroup takes the then-region.

`proveLeaderGuardedWorkgroupOwnedIndex` is workgroup unique-axis ownership
conjoined with a leader guard. Residual `unit_grid_axes` constrain unused
**grid** axes of $q$, not $W$.

## 6. Single-invocation fallback

If no lemma applies, the workgroup is already $(1,1,1)$, and the indices do
not mention scalar `gid`, the compiler may still accept the store by
constraining **every** grid axis to $1$ and setting
`requires_unit_workgroup`. That is a serialized launch:
$|\mathrm{Inv}| = 1$, so $\Phi$ is vacuously injective.

This is the path for constant-index ordinary writes. It is **not** used when
mixed-radix or affine ownership already distinguishes $g_0$.

## 7. What is not proved

- Arbitrary functions of $g_0$ (for example $g_0 \bmod 4$ alone).
- Leading-digit wrap without a launch bound on $L$.
- Signed remainder/division.
- 32-bit wraparound of affine forms (same omission as the affine fragment).
- In-bounds / allocation size.
- Barrier alias analysis for mixed-radix tuples (reads still normalize only
  in the affine fragment; unproven communication stays fail-closed).

Negative unique-axis coefficients composed with `remui` are accepted only as
far as the unsigned bit-pattern interpretation of $L$; the intended source
form is $L = g_a \ge 0$.

## 8. Implementation map

| Lemma | Function |
| --- | --- |
| Affine unique-axis | `proveAffineInvocationOwnedIndex` |
| Mixed-radix of unique-axis $L$ | `proveMixedRadixInvocationOwnedIndex` |
| Dispatcher | `proveInvocationOwnedIndex` |
| Full three-axis affine, no residual | `proveStrictInvocationOwnedIndex` |
| Workgroup unique-axis | `proveWorkgroupOwnedIndex` |
| Leader-guarded workgroup | `proveLeaderGuardedWorkgroupOwnedIndex` |
| Ordinary-store policy and contract | `KernelMemoryPhaseAnalysis` in `VernonValidation.cpp` |
| Launch check | `validateDispatchContract` |
