# Program VJP interior adjoint SSA

Design for cotangent accumulation on multi-node Programs. Architecture target
remains [`unified_program_vjp.md`](unified_program_vjp.md). Public gradient
rules remain [`autodiff.md`](autodiff.md) §3 and
[`language/contract.md`](language/contract.md) (TensorView gradients are newly
owned Storage; compute-node ABI is a writable TensorView, not a kernel that
returns Storage). Do not bump `COMPILER_CONTRACT_VERSION` / `PIPELINE_VERSION`.

## 1. Symptom

`test_module_forward_and_backward_use_program_operation_graph` fails cook with
`Program VJP cannot accumulate derivative values`.

`addValues` already emits `vernon.intrinsic "add"` for matching
`Tensor` / `TensorView` pairs. It returns null only when the two SSA types
differ. FanIn passes because both contributions are the same type. Fluid fails
because one adjoint is a public cotangent (`read` TensorView) and the other is
a nested kernel VJP result (`write` TensorView).

That is not “Tensor vs tuple-of-leaves”. Density and velocity are returned
**and** read by a later kernel (`smoke_loss`, `transport_density`). Reverse
walk therefore adds an output cotangent to a kernel-input gradient.

## 2. SSA handles vs kernel DSL

Program IR may use SSA. That SSA is a **handle** (Value, or TensorView over
Storage). It is not a license for the kernel DSL to return Storage.

Compute kernels are multithreaded and return `None`. They read TensorViews and
write dest TensorViews. A thread cannot allocate and return the whole buffer.
`@vd.kernel` add is already this shape: rank-specialized, `vd.dyn` extents,
`output: TensorView[..., write]`, `-> None`. Launch extents come from the
view descriptor (`get_shape` / linearized `gid`), not from baking a static
shape into the kernel type.

So:

| Kind | Program SSA | Kernel DSL |
|------|-------------|------------|
| Immutable Tensor | value SSA (`arith.addf`, tensor constructor) | not a Storage return |
| TensorView / Storage | handle to an allocation; result aliases a dest (`result_resource_sources`) | void + write dest |
| Alloc (`empty` / `zeros` / `*_like`) | host-side new Storage, handle as result | not a compute kernel |

`addValues` for TensorView must be destination-passing: alloc a dest, compute
`vernon.builtin.add(left, right, dest)` with `operand_accesses = [read, read,
write]`, result aliases `dest`. Do not invent a kernel that returns Storage.
Value-producing `%sum = add %left, %right` is only for Tensor **Values**.

`createZero` for a TensorView is `zeros` / `zeros_like` (alloc), not a kernel
result.

## 3. Three layers

Access `read` / `write` / `read_write` is a **capability** of a TensorView.
It is valid in three different places. Mixing those places is the bug.

| Layer | Where | Access lives | Type of the SSA handle |
|-------|--------|----------------|------------------------|
| Public backward ABI | backward function args / results | on the TensorView type | cotangent `read`; gradient `write` |
| Kernel VJP ABI | nested `compute.vjp` parameters | `vernon_program.operand_accesses` | same as primal: Program TensorView (`read_write` on fluid) |
| Interior adjoint map | `adjoints[primalSsa]` | not on the type | `derivativeType(primal)`: keep the primal constructor and access |

Primal already follows this. Fluid `%v14` dest and `%v15` result are both
`!vernon.tensor_view<f32, [-1, -1], "read_write", "device">`. The dest is
write-only in `operand_accesses`; `result_resource_sources` aliases the same
storage as the result. Program VJP currently types nested VJP **results** as
`gradientDestType` (`write`) and seeds `adjoints[result]` from function args
typed `cotangentType` (`read`). `addValues` then compares ABI types.

Kernel ABI assignment of `read` (cotangent) and `write` (gradient dest) is
correct. Public signature assignment is correct. Interior handles must not use
`cotangentType` / `gradientDestType`.

## 4. Target construction

Nested `compute.vjp` is a primal compute node that happens to implement a
derivative, not a second type system.

1. Interior `adjoints[v]` has type `derivativeType(v)`. For a fluid TensorView
   that is still `read_write` with the derivative payload. For a Tensor value
   that is still the tensor constructor. Do not flatten aggregates to scalars.
2. Nested VJP results use that interior type, with `result_resource_sources`
   aliasing write dests, matching primal. Gradient dest **operands** keep
   `operand_accesses = "write"`. Cotangent **operands** keep
   `operand_accesses = "read"`.
3. TensorView fan-in is DPS add into a fresh dest (strict type equality on the
   two **read** operands). Tensor value fan-in stays value `add`. Do not relax
   type equality and do not pun `read`/`write` as one value.
4. Public backward args stay `cotangentType` (`read`). Public results stay
   `gradientDestType` (`write`).
5. Function entry: a public cotangent is a **read operand**, never a dest.
   If the interior map needs an owned adjoint buffer, **copy** into a fresh
   allocation. Do not alias the caller’s cotangent as `write` or `read_write`.
6. Function exit: the public gradient is newly owned Storage viewed as `write`.
   If the interior handle is already that dest and was not accumulated further,
   return it. If DPS add wrote a new dest, that dest is the owner.

## 5. Casts

Access-changing `cast` is not the solution. It is an adapter, and most
directions are unsound.

| Cast | Meaning | Allowed |
|------|---------|---------|
| `write` → interior after the producer finished | dest handle becomes the result alias | prefer `result_resource_sources`, not a standalone `cast` |
| `read` → `write` | capability upgrade | no; would write caller cotangents |
| `read` → `read_write` | capability upgrade | no; later passes may treat it as a dest |
| ignore access in `addValues` | pun ABI roles as one value | no |

Do not insert interior casts to make `add` type-check. Fix result typing so
both read operands of DPS add are the interior type.

Do not accumulate TensorView adjoints by first converting them to Tensor
Values. That would make the DSL return Storage.

## 6. Builtin add is rank + dyn

The kernel is already `_program_add_{dtype}_rank{n}` with `TensorView[...,
vd.dyn, ..., write|read]` and `-> None`. Rank selects the kernel; extents stay
dynamic and come from the descriptor.

Host wiring today still rejects dyn extents in places (`implementationGrid`
wants a static ranked result to fill `grid`; `_lower_builtin_add` /
`program_add_invocation` require `extent >= 0`). That is **not** the kernel
contract. If fluid cook hits it after typing is fixed, fix the host to pass
rank plus descriptor extents (same as other dyn TensorView kernels). Do not
specialize `vd.dyn` into the artifact and do not treat static shape as the
add design.

## 7. How large is the change

Localized to Program VJP construction plus any host add request that still
bakes static extents:

- `VernonProgramVjp.cpp`: nested VJP result types, adjoint map, TensorView
  accumulate as DPS add (alloc dest + compute), seed/return boundary
- C++ tests that require nested VJP **result** access `write`
  (`ProgramVjpStopsAtStorageAllocIntrinsics`). Kernel dest access on the
  operand stays `write`
- FanIn Tensor-value accumulate stays as it is

Not in this change: contract versions, Module ABI names, flattening
aggregates, parent `variant.parameters` synthesis, unique-ifying CPU symbols,
relaxing `addValues`, user-facing `accumulate_into`.

## 8. Acceptance

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/tests/test_module.py -q
PYTHONPATH=python .venv/bin/python -m pytest \
  python/tests/test_smoke_fluid_graph.py::SmokeFluidGraphTests::test_module_forward_and_backward_use_program_operation_graph \
  python/tests/test_smoke_fluid_graph.py::SmokeFluidGraphTests::test_single_step_velocity_gradient_matches_finite_difference \
  -q --tb=short
```

P2 is closed when those two fluid cases **cook**. Numeric pullback may lag one
step. Existing green AD tests must stay green; do not drop below the floor in
[`program_vjp_debug_priority.md`](program_vjp_debug_priority.md).
