# Program VJP debug priority

Working checklist for finishing unified Program VJP after the non-AD baseline
is green. Architecture target remains [`unified_program_vjp.md`](unified_program_vjp.md).
Tape identity remains [`program_execution_manifest.md`](program_execution_manifest.md)
§12.2 (`!vernon.ad_tape` is a Program Value + `residual_contract` capture).
Do not bump `COMPILER_CONTRACT_VERSION` / `PIPELINE_VERSION` until the
coordinated release.

Status after nested kernel VJP **pair** wiring (`forward_with_tape` +
`!vernon.ad_tape` capture). FanIn stays numeric (`captures == [0]`). CPU
Fluid **single-step Program VJP cooks, runs, and matches finite difference**.
Remaining Fluid red is wrappers (host-static `Module.forward`, Program
pullback telemetry, checkpoint/budget, GPU Program AD gate), not missing tape
bindings.

## Current pass status (do not regress)

Last full gate 2026-09-01: `test_smoke_fluid_graph.py` is **9 passed / 3
failed**; all `GpuExecutionGraphAutodiffTests` pass. P3, the P2 Fluid anchors,
the phase-2 benchmark, and all non-AD tests are green.

| Suite | Result |
|-------|--------|
| **ctest** (`build/`, 510 tests) | **509 passed / 1 failed** (OpenGL/CUDA/unavailable-backend skips are not failures). Only file-level `vernon-python-smoke-fluid-graph` remains red. |
| **pytest** `python/tests` | **455 passed / 3 failed / 60 skipped / 397 subtests passed** |

**Every change must keep this floor.** Failed count must not go up. Currently
passing tests must stay passing. A patch that turns a listed failure green is
allowed only if nothing else turns red. After each AD change, re-run the full
commands in “Suggested daily gate sequence” and refresh the tables below if
the counts move.

### C++ that must stay green

Everything in `ctest` except the one AD name in “Current failures”. That
includes all non-AD compiler/runtime/RHI/ExecutionGraph tests, plus these AD
targets that already pass:

- `VernonAutodiffRulesTest.TensorRulesPromoteF16AndRejectInvalidShapes`
- `VernonStructuredVjpTest.ProgramVjpScalarizesCanonicalAggregateDerivativeLeaves`
- `VernonStructuredVjpTest.AutodiffDerivativeValueLayoutUsesPhysicalPayloadAbi`
- `VernonStructuredVjpTest.ProgramVjpStopsAtStorageAllocIntrinsics`
- `VernonStructuredVjpTest.SpecializeKernelHostConstantsInlinesScalarArguments`
- `VernonStructuredVjpTest.SpecializeKernelHostConstantsRejectsResourceArguments`
- remaining `VernonStructuredVjpTest.*` / `VernonAutodiffRulesTest.*` /
  `VernonAutodiffAnalysisTest.*` / `RuntimeStructuredScalarAutodiff.*` /
  `RuntimeStructuredAggregateAutodiff.*` / `RuntimeGpuAutodiff.*` (non-OpenGL) /
  CPU `ExecutionGraphAutodiff.*`
- `VernonPythonCookedAutodiffTest`
- `VernonPythonNativeAutodiffNumericCpuTest`
- `VernonPythonDynamicV2ParityTest`
- `vernon-python-test_storage_vjp_contract`
- ctest Python non-AD targets (`vernon-python-native-compiled-program`,
  compile-surface parity, kernel-memory, examples, …)

### Python that must stay green

All of `python/tests` except the 3 names in “Current failures”. In
particular:

- **Entire files green:** `test_module.py`, `test_storage_vjp_contract.py`,
  `test_autodiff.py`, `test_compiler.py`, `test_pipeline_compile.py`,
  `test_pipeline_runtime.py`, `test_kernel_runtime.py`, `test_shader_assets.py`,
  `test_tensor_storage.py`, `test_tensor_shapes.py`, `test_attribute_abi.py`,
  `test_language_contract.py`, `test_native_compiled_program.py`,
  `test_numpy_tensor_runtime.py`, `test_blinn_phong_numeric.py`,
  `test_inference_backend_numeric.py`, `test_dependency_boundaries.py`,
  `test_type_analysis_coverage.py`, `test_shared_definitions.py`,
  `test_production_checks_under_optimization.py`,
  `test_showcase_smoke_runner.py`, `test_versions.py`
- **`test_execution_graph.py`:** all cases, including structured aggregate
  `GpuExecutionGraphAutodiffTests`, stay green
- **`test_smoke_fluid_graph.py`:** all cases except the three methods in
  “Current failures” stay green

Non-AD C++ and Python suites must remain fully green.

## Baseline that must stay green

Minimum AD anchors after a local edit, before a full suite (these pass today):

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/tests/test_storage_vjp_contract.py -q
PYTHONPATH=python .venv/bin/python -m pytest \
  python/tests/test_execution_graph.py -k 'vjp and not Gpu' -q
PYTHONPATH=python .venv/bin/python -m pytest python/tests/test_module.py -q
PYTHONPATH=python .venv/bin/python -m pytest \
  python/tests/test_smoke_fluid_graph.py::SmokeFluidGraphTests::test_module_forward_and_backward_use_program_operation_graph \
  python/tests/test_smoke_fluid_graph.py::SmokeFluidGraphTests::test_single_step_velocity_gradient_matches_finite_difference \
  -q --tb=short
ctest --test-dir build --output-on-failure -R 'VernonPythonDynamicV2ParityTest'
```

Do not land a change that only passes these anchors. Full `ctest` + `pytest`
must still match the floor above. The currently red commands are under each
P-section and in “Current failures”.

## What already landed

- Program alloc ops are not reverse-walked; write dests are objectives.
- Canonical cook is one forward graph plus optional backward +
  `residual_contract`.
- Derivative Values keep the primal tensor constructor; struct payloads map to
  ABI leaves. No `tangent<>` identity rebase.
- Canonical Program load does not invent parent `variant.parameters`. Python
  `program_vjp` binds by signature path / AD leaf metadata, not `value.name`.
- Storage is the physical allocation; Values share `host_data` and keep their
  own layout.
- Python JIT CPU entries intern by hashed symbol on `Runtime`: one address,
  first `CompileResult` pinned, later cooks reuse that pointer. The C register
  API still rejects two different pointers for one symbol.
- `GetParameterValueLeaf` returns element-scope payload leaves for shaped
  Tensors (cells / struct fields). Outer extents stay on the parameter.
  Packed host_value ABI is not this API.
- Program VJP interior adjoints use `derivativeType` (primal constructor +
  access). Nested `compute.vjp` results alias write dests. TensorView fan-in
  is DPS `vernon.builtin.add`; Tensor Value fan-in stays value `add`. Public
  cotangents stay `read`; public gradients stay `write`. Host builtin add/copy
  cook by rank + `vd.dyn`, not baked static extents.
- Nested VJP dests are `gradient.<source>` with `write` on the edge. Only
  active result cotangents are bound. Role matching, not primal names.
  `resourceAccessSatisfies` is the access check.
- `rebuildCaptures` recomputes from ABI-matched operands. No signature seed.
  Like-source captures only when extents are dynamic. Runtime trusts
  `residual_contract`. Static FanIn dests do not inflate captures past `[0]`.
- Program TensorView **type** stays `read_write`; access lives on the edge.
  Host numpy may carry trailing Vector/Matrix leaf axes; bind uses the
  TensorView prefix. Reflection keeps `vd.dyn` as `-1` / `0`; runtime bind
  instantiates from the bound buffer, **including empty extent 0**.
- Program compute `constant_names` / `constant_values` specialize the attached
  kernel implementation (`arith.constant` in kernel IR, arguments erased)
  before nested structured VJP. Host-static `np.int32` does not become a
  Program value.
- Nested kernel VJP is a **pair**. When backward needs a sealed tape, Program
  forward uses `callee.forward_with_tape`, captures `!vernon.ad_tape` on
  `residual_contract`, and `callee.vjp` consumes that Value. `gid` stays a
  wrapper-injected builtin. Tape builtins stay packed fields of
  `!vernon.ad_tape`, not extra manifest fields. Do not inject an empty
  `HostTapeAllocator` into backward-only dispatch.

## Ordering principle

P0, P1, interior adjoint SSA, language ABI copy, host-static kernel-constant
specialization, nested Vector TensorView dest-passing ABI, residual capture
set, empty-dyn bind, and **P2 nested `forward_with_tape` pair** are closed.

```text
Module.forward host-static comparisons (P4a: rollout `assert loss is not None`)
  → Program pullback telemetry / checkpoint / budget / optimization (P4b)
  → GPU ExecutionGraph `particles` ABI (P3)
  → GPU Program AD (CPU-only `compile_program_autodiff` gate)
  → phase-2 benchmark
```

One minimal failing test per layer until that layer is green, then expand.

## P0 — Module Program VJP (closed)

**Was:** `Program VJP wrt argument is not differentiable`, then
`cannot register canonical Program CPU entry '__vernon_cpu_…_module_square'`
when primal kernel JIT and Program VJP both loaded the same hashed symbol.

**Now:** FanIn cooks, runs, and pulls back (`square=4`, `cube=8`, grad `28`
at `x=2`). CPU JIT intern reuses the first address per symbol and pins that
`CompileResult`, so primal-then-VJP in one Runtime is legal — same model as
linking one `.o` from many Programs.

**Green (whole file):**

```bash
PYTHONPATH=python .venv/bin/python -m pytest python/tests/test_module.py -q --tb=short
```

Do not unique-ify symbols per Program, and do not let the C API accept two
pointers for one symbol. Do not change
`self.assertEqual([value["value_id"] for value in signature["captures"]], [0])`.

## P1 — Kernel AD leaf / shape ABI (closed)

**Was:** `GetParameterValueLeaf("value")` returned the packed Tensor ABI
(`scalar_count=2`, `static_rank=1`) while the parameter already carried
`shape=[2]`. Python `vjp()` concatenated those extents and doubled rank.

**Now:** that API returns element-scope payload leaves (cells / struct fields).
Outer Tensor extents stay on `FindParameter().rank` / `static_shape`. Packed
host_value ABI is unchanged for dispatch.

**Green:**

```bash
ctest --test-dir build --output-on-failure -R \
  'RuntimeStructuredAggregateAutodiff.ExecutesAggregateInputAndStorageObjectivePullback|VernonPythonCookedAutodiffTest|VernonPythonNativeAutodiffNumericCpuTest|VernonPythonDynamicV2ParityTest'
```

GPU `particles` was re-checked after this and still reports `conflicting ABI
values for 'particles'`. Treat that as P3, not a leftover of this encoding.

## P2 — Nested kernel VJP pair (`forward_with_tape`) (closed)

**Was:** Program forward cloned **untaped primal**; backward was `callee.vjp`
expecting a sealed tape. FanIn stayed green (`validNoTape`). Fluid
`smoke_loss` loops → `validTape` → `backward.3` (`smoke_loss.vjp`) died with
`CPU inline binding requires host data` on packed tape argument 0.

**Now:** when nested backward needs tape, forward attaches
`forward_with_tape`, emits `!vernon.ad_tape`, captures it on
`residual_contract`, and `compute.vjp` consumes that Value.
`CapturedVjpDslProvider` lowers both profiles of the structured VJP pair.
CPU Fluid single-step Program VJP cooks, pulls back, and matches finite
difference.

Kernel profiles (landed contract; do not invent new manifest tape fields):

| profile | role | CPU packed ABI |
|---|---|---|
| `primal` | numeric only | no tape |
| `forward_with_tape` | same compute, records loops | allocator only (write) |
| `backward` | reads a **sealed** snapshot | allocator **and** root region |

`gid` stays a hidden builtin. Tape contents are a Program Value.
`ad_tape_allocator` / `ad_tape_root_region` are that Value’s CPU packed layout
(like TensorView descriptors). Manifest §12.2 already specifies this as
Program Storage + `residual_contract`.

**Do not** inject an empty `HostTapeAllocator` into backward-only dispatch.
Read callbacks are legal only after seal.

**Green (stay green):**

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/tests/test_smoke_fluid_graph.py::SmokeFluidGraphTests::test_module_forward_and_backward_use_program_operation_graph \
  python/tests/test_smoke_fluid_graph.py::SmokeFluidGraphTests::test_single_step_velocity_gradient_matches_finite_difference \
  -q --tb=short
```

## P3 — GPU direct autodiff load (closed)

**Was:** `cannot load direct autodiff profiles: autodiff parameters define
conflicting ABI values for 'particles'`.

Aggregate derivative reflection now projects differentiable leaves, canonical
Program endpoints bind `(value, leaf)`, and the planner preserves heterogeneous
physical leaf uses under one logical aggregate owner. Root and dotted `wrt`
paths pass. GPU **Program** AD remains a separate P4 gate because
`compile_program_autodiff` is CPU-only.

**Green:**

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/tests/test_execution_graph.py::GpuExecutionGraphAutodiffTests::test_gpu_graph_vjp_preserves_structured_storage_gradients \
  python/tests/test_execution_graph.py::GpuExecutionGraphAutodiffTests::test_gpu_graph_vjp_rejects_unlowered_aggregate_fan_in \
  -q --tb=short
ctest --test-dir build --output-on-failure -R 'vernon-python-execution-graph'
```

## P4 — Smoke wrappers (current)

P2 single-step pullback and the rollout wrappers are green. Remaining
`SmokeFluidGraphTests` split:

- **P4b telemetry** — `test_phase2_tape_telemetry_for_ci_grids` reaches
  `ProgramNativePullback.pass_telemetry`, but the expected logical loss pass
  is absent (`StopIteration`). Do not restore fake
  `estimated_tape_bytes = allocated_bytes` or `recomputation_factor = 1.0`.
- **Dynamic checkpoint budget** —
  `test_dynamic_checkpoint_runtime_enforces_physical_budget` reaches its
  accounting assertion; bounded rematerialization reports
  `logical_residual_bytes == 512` instead of `0`.
- **GPU Program AD** —
  `interactive Program autodiff currently requires the CPU runtime`
  (`compile_program_autodiff`). This is why
  `test_available_gpu_backends_validate_optimistic_static_tape_hints` is red.
  Non-AD GPU Fluid forward already matches CPU.
- `vernon-python-autodiff-phase2-benchmark` is green.

**Red (checkpoint budget / telemetry / GPU Program AD):**

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/tests/test_smoke_fluid_graph.py::SmokeFluidGraphTests::test_available_gpu_backends_validate_optimistic_static_tape_hints \
  python/tests/test_smoke_fluid_graph.py::SmokeFluidGraphTests::test_dynamic_checkpoint_runtime_enforces_physical_budget \
  python/tests/test_smoke_fluid_graph.py::SmokeFluidGraphTests::test_phase2_tape_telemetry_for_ci_grids \
  -q --tb=short
```

Order:

1. P4b Program pullback telemetry (`pass_telemetry`)
2. Dynamic checkpoint physical-budget accounting
3. GPU Program AD (CPU-only compile gate)

## Explicit non-goals while climbing the ladder

- Starting with smoke fluid or phase-2 benchmark
- Flattening Program aggregate derivatives back to a tuple of scalars
- Re-synthesizing parent `variant.parameters` from `value.name`
- Parallel rewrites of Program execution manifest + kernel leaf ABI + GPU load
  without a green gate between them
- Reintroducing deprecated direct AD / intern / fake-zero workarounds
- Injecting an empty `HostTapeAllocator` into backward-only dispatch
- Adding new manifest fields for tape; §12.2 / `!vernon.ad_tape` is the contract
- Bumping compiler or pipeline contract versions before the release cut
- Shipping a change that drops below **509 ctest / 455 pytest** passing, or
  that adds any failure not listed below

## Suggested daily gate sequence

| Step | Command focus | Advance when |
|------|---------------|--------------|
| 1 | Full `test_module.py` | Stay green (already) |
| 2 | C++ aggregate + cooked/numeric CPU AD + DynamicV2 | Stay green (P1) |
| 3 | Fluid `module_forward` + `single_step` | Stay green (P2) |
| 4 | P4a host-static rollout comparison | `assert loss is not None` is host-static |
| 5 | P4b telemetry + checkpoint wrappers | Program pullback surface + numeric checkpoint |
| 6 | GPU ExecutionGraph AD pair (P3) | Conflicting `particles` ABI |
| 7 | GPU Program AD + phase-2 benchmark | Product-level AD |

Full suite (expect only unfinished AD layers to fail until the ladder is done):

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure -j"$(sysctl -n hw.ncpu)"
PYTHONPATH=python .venv/bin/python -m pytest python/tests -q --tb=line
```

## Current failures (only allowed remaining red)

Refresh this list when a layer turns green. This ctest name and three pytest
names are the **only** permitted failures. Anything else red is a
regression.

**ctest (1)** — one shot (file-level; the Python methods inside are the three
names below):

```bash
ctest --test-dir build --output-on-failure -R 'vernon-python-smoke-fluid-graph'
```

- `vernon-python-smoke-fluid-graph`

**pytest (3)** — one shot:

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/tests/test_smoke_fluid_graph.py \
  -q --tb=short
```

- `SmokeFluidGraphTests::test_available_gpu_backends_validate_optimistic_static_tape_hints`
- `SmokeFluidGraphTests::test_dynamic_checkpoint_runtime_enforces_physical_budget`
- `SmokeFluidGraphTests::test_phase2_tape_telemetry_for_ci_grids`
