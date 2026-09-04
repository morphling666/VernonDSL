# Module/Program capability and regression contract

Status: implemented and verified. This document records the supported surface,
the resolved historical gaps, and the regression gates. It does not introduce
a compiler or pipeline contract version change.

This document records the coverage lost when the Python `ExecutionGraph` test
file was deleted, the tests restored as canonical Program tests, and the
implementation gaps those tests now expose. Do not reintroduce Python pass
descriptors or a second execution manifest to make these tests pass.

## Required public model

`Module.forward` inputs must be a strict superset of direct Kernel DSL inputs:

- every by-value argument accepted by a decorated Kernel must also be accepted
  as a runtime Module argument with the same annotation and conversion rules;
- every Kernel resource argument must preserve the same logical type, access,
  shape, alias, and runtime binding semantics through Module capture;
- Module additionally accepts `TensorStorage` as an owner-level input and may
  infer the matching Tensor/TensorView type from its use inside `forward`;
- positional and keyword binding follows the Python `forward` signature;
- a runtime value argument remains an invocation value. It must not silently
  become a specialization constant merely because Module capture occurs in
  Python.

At minimum, parity covers the existing Kernel binder's scalar, vector, matrix,
tuple/struct, Tensor/TensorView, texture, and sampler categories wherever that
backend supports them. `TensorStorage` is the Module-only convenience owner.

Module outputs continue to be represented by storage trees. Differentiable
aggregate gradients are returned as packed `TensorStorage` owners whose field
views are accessed with `gradient["field"]`; they are not reconstructed as
Python dictionaries.

## Canonical Program boundary and binding model

Managed Program execution has one complete forward invocation ABI and a
separate autodiff projection of that ABI. These are related contracts, not two
competing execution paths.

The forward invocation ABI has stable, typed slots for every public Module
input and output:

- ABI-stable Values: scalar, Vector, Matrix, Tuple/shared Struct, and Tensor;
- Storage resources: TensorStorage owners and TensorView projections;
- opaque resources: Texture views and Sampler descriptors;
- output Storage trees.

Graphics Programs are forward-only. Module VJP rejects a Module definition
containing any Texture or Sampler parameter, including an otherwise inactive
primal resource. Texture and Sampler slots never enter `VernonAdValue`, cannot
be selected by `wrt`, and never acquire tangent, cotangent, or gradient entries.
This strict Module boundary is intentional until opaque-resource snapshots are
part of the versioned VJP capability contract.

`outputs=` is the public reverse-boundary selection. Internally, the selected
paths must be carried into Program IR (currently as
`vernon_program.vjp_outputs`) because `vjp_wrt` describes input differentiation
only. This internal field must not prune the forward function results:

- forward execution preserves the complete output tree and all declared
  effects;
- reverse activity is seeded only from selected output paths;
- the cotangent signature contains only selected differentiable outputs.

### Versioned capability baseline

`VernonProgramCapabilities.h` is the compiler/runtime authority for this
release's capability IDs, support decisions, and stable unsupported
diagnostics. Python preflight queries that same matrix through the private
compiler bridge; it must not maintain a second table.

- Direct compute Kernel forward accepts Value, Tensor/TensorView, and backend-supported
  opaque resources. Compute sampling through Sampler is not supported.
- Graphics Pipeline forward accepts Texture and Sampler bindings.
- Module/Program compute forward accepts ABI-stable Values, Storage/TensorView,
  and Texture resources. Sampler remains graphics-only.
- Direct Kernel VJP and Module VJP support differentiable Value/Storage leaves.
  Module VJP rejects every Texture or Sampler parameter at transform creation.
- Graphics VJP is unsupported.
- CPU `f16` follows the canonical `f16` primal to `f32` gradient policy.
  GPU `f16` is unsupported in the current contract on every backend; no
  implicit widening or backend fallback is permitted. Enabling it requires a
  future versioned capability change.

### Shape and layout authority

Carrier shape and Value shape have different owners:

- TensorStorage/TensorView outer extents belong to the Program resource
  descriptor;
- the Value stored in each resource element is described by its canonical
  ValueLayout, including Vector/Matrix/Tensor or nested Struct/Tuple leaf
  shapes;
- a standalone Vector/Matrix/Tensor invocation Value has no outer Program
  resource shape; its complete shape appears exactly once in ValueLayout;
- Texture dimension, extent, format, and access belong to the opaque resource
  descriptor, not ValueLayout;
- Sampler has neither carrier shape nor ValueLayout.

Consequently, a standalone Matrix Value must not reflect `[2, 2]` both as a
Program shape and as a ValueLayout leaf shape. Storage whose element is that
same Matrix legitimately has an outer Storage shape and an inner `[2, 2]`
ValueLayout shape.

### Persistent binding state

Each loaded Program owns a persistent binding table keyed by stable public
slot. Repeated invocation updates only slots whose canonical execution token
changed:

- Value token: logical type, canonical layout hash, and packed canonical bytes;
- Storage token: owner identity/generation plus view offset, shape, strides,
  access, and dirty resource version/ranges;
- Texture token: resource/view generation and descriptor identity;
- Sampler token: immutable sampler descriptor identity;
- output token: owner/view identity and writable generation.

Prepared bindings and resident external resources survive loop iterations.
Unchanged slots reuse their prepared argument and device residency; changed
Value slots upload only their canonical payload, and changed Storage uses its
dirty ranges. Runtime-owned `ProgramInstance` transactions publish immutable
invocation snapshots only after successful execution. `ProgramInvocationFrame`
borrows snapshot-owned public bindings and owns only invocation-local internal
allocations; pullbacks retain their own snapshot, tape, and residual state.

This model forbids using a Python packed-value cache as a substitute for device
binding persistence. Host packing reuse is useful, but it does not satisfy the
contract unless the runtime also reuses the prepared slot and resident
resource.

## Why the previous full-suite result was invalid

`python/tests/test_execution_graph.py` was deleted while its CTest entry still
used unittest discovery with that filename. Discovering no matching file can
exit successfully, so CTest reported a pass while executing zero tests.

The replacement is `python/tests/test_program_execution.py`, and the CTest
entry `vernon-python-module-program` now names that file. Acceptance must check
that tests are actually collected, not merely that discovery exits with zero.

The deleted file mixed three responsibilities:

- Python graph/pass-builder API behavior;
- multi-operation execution and VJP behavior;
- direct resource and graphics behavior.

Pipeline 17 intentionally removes the first public API. Its implementation
details remain covered by native Command DAG tests. The second category must be
rewritten through Module/Program and may not be dropped. The third category
belongs to direct Kernel, graphics Pipeline, and resource tests.

## Restored semantic coverage

`python/tests/test_program_execution.py` currently covers:

- ordered execution of dependent kernels inside one Module;
- repeated Module invocation and Program specialization reuse;
- direct Kernel VJP and one-kernel Module VJP parity;
- multi-kernel reverse composition;
- independent retained state for multiple pullbacks;
- scalar branch fan-in;
- multiple differentiated inputs;
- repeated/aliased operand fan-in;
- dynamic scalar Module inputs in primal and VJP calls;
- direct and Module aggregate root VJP;
- aggregate leaf-path and root-path selection;
- aggregate branch fan-in;
- selecting one differentiable objective while an unrelated branch executes;
- non-contiguous TensorView gradient scatter on GPU;
- GPU scalar fan-in and structured aggregate gradients;
- direct texture-view validation that does not belong to Module.

The direct Kernel and Module forms are both required. Passing a Kernel directly
to `vd.ad.vjp` must remain supported; wrapping that same Kernel in a Module must
produce equivalent primal and gradient values.

## Confirmed test correction

One restored assertion was wrong: aggregate gradients were asserted to be
Python dictionaries. The canonical owner contract, already exercised by
`test_storage_vjp_contract.py`, returns one packed `TensorStorage`. Changing
that assertion to `TensorStorage` is a test correction, not an implementation
workaround.

The restored tests remain permanent requirements. None is an expected failure,
xfail, or compatibility-only case.

## Resolved implementation gaps

The following subsections preserve the original failures and their required
root fixes. All four are implemented; the diagnostics are historical and must
not reappear.

### 1. Canonical Module value descriptors

Historical regression tests:

- `test_module_scalar_argument_is_bound_per_invocation`
- `test_module_vjp_differentiates_scalar_argument`

Historical failure:

```text
Module.forward() argument 'amount' must be TensorStorage or TensorView, got float32
```

The former runtime-value inference modeled every Module input as a
storage-backed value. `ModuleDefinition` and `RuntimeParameterDescriptor` now
provide one immutable typed definition, and `GraphValueInput` preserves
by-value invocation identity through capture.

Implemented root fix:

- introduce one canonical Module input descriptor capable of representing both
  by-value Kernel types and storage-owner/view types;
- reuse Kernel frontend type checking and host-value packing rules rather than
  creating another dtype conversion table;
- create capture placeholders for by-value inputs so a scalar used in a kernel
  call remains a Program argument;
- include logical type and layout in specialization identity, but not the
  runtime scalar value;
- materialize and bind those Program arguments through the canonical runtime
  signature for both primal and backward execution.

Do not convert scalars to rank-zero storage in tests or in the public API. That
would change the Kernel contract rather than implement parity.

### 2. Structural aggregate fan-in lowering

Historical regression test:

- `test_module_aggregate_vjp_accumulates_branch_fan_in`

Historical failure:

```text
unsupported binary operation for Tuple
```

The historical path generated a Python `vernon.builtin.add` kernel that added
the complete aggregate element. Fan-in is now an explicit typed Program
accumulate operation lowered from canonical tangent `ValueLayout` leaves.

Implemented root fix:

- define Program gradient accumulation over canonical differentiable leaves;
- use `ValueLayout`/tangent layout as the authority;
- add floating tangent leaves independently with their declared accumulation
  dtype;
- preserve aggregate owner shape, field paths, alignment, and aliases;
- exclude non-differentiable leaves instead of attempting to add integer tags;
- keep fan-in as an explicit Program operation lowered by the compiler/runtime
  operator path.

Do not add ad-hoc tuple parsing to tests, flatten owners in Python calling code,
or special-case the `Particle` fixture.

### 3. Operand-use and storage-owner identity

Historical regression test:

- `test_module_vjp_accumulates_aliased_operands_once_per_use`

Historical failure:

```text
Program aggregate binding does not match its canonical Value ABI for request 'backward:1'
```

The primal binds one Storage owner to both `left` and `right`. Reverse mode must
produce one contribution per operand use and accumulate both into the same
owner gradient. Finalization formerly rejected the repeated physical binding;
the compiler now preserves operand-use identity separately from `aliasOwner`.

Implemented root fix:

- distinguish operand-use identity from Storage-owner identity;
- validate each physical endpoint binding against the canonical value or leaf
  it projects;
- permit multiple legal uses of one owner in a node;
- explicitly fan contributions into one gradient owner;
- publish exactly one gradient for the public `source` path.

For `source * source` at `source = 3`, acceptance requires gradient `6`, not
`3`, `12`, or two separately returned gradients.

### 4. Selected Module VJP outputs

Historical regression test:

- `test_module_vjp_ignores_unrelated_non_differentiable_branch`

Historical failure:

```text
Program autodiff currently requires all Module outputs
```

The requested objective is `square`; `unrelated` is a valid primal output but
is outside the reverse boundary. Requiring a cotangent for every returned
output is unnecessarily restrictive and differs from direct Kernel VJP's
explicit `outputs=` selection.

Implemented root fix:

- preserve the complete primal output tree;
- build the cotangent signature from selected `outputs=` paths only;
- mark reverse activity from those selected values;
- omit inactive branches unless their effects are dependencies of active work;
- require cotangents only for selected non-scalar objectives;
- continue executing and returning unrelated primal outputs.

Do not make the test pass by selecting all outputs and supplying a synthetic
zero cotangent.

## Acceptance criteria

### Module input parity

- A Module with `forward(source: TensorStorage, amount: vd.f32)` accepts
  `np.float32` at runtime.
- Reusing one specialization with amounts `2` and `5` produces distinct correct
  results without recompiling by value.
- The same by-value scalar may be selected by `wrt` and returns the correct
  scalar gradient.
- Add focused parity tests for every Kernel by-value category currently
  supported by host binding. The Module and direct Kernel paths must accept and
  reject the same values for the same annotation.
- TensorStorage owner inference remains supported and does not weaken explicit
  TensorView access validation.

### Direct Kernel and one-kernel Module VJP

- `vd.ad.vjp(kernel, ...)` executes directly and returns the expected gradient.
- `vd.ad.vjp(ModuleWrappingKernel(), ...)` executes through canonical Program
  and returns the same primal and gradient.
- Neither route calls the removed Python pass API.
- Both routes work after cache reuse and runtime reinitialization according to
  existing generation rules.

### Program reverse composition

- A two-node square chain at `x = 3` returns `x^4 = 81` and gradient `108`.
- Two pullbacks created from different forward calls retain independent state.
- Scalar fan-in combines square and cube branches deterministically.
- Aggregate fan-in combines every differentiable leaf and returns one packed
  owner.
- Repeated aliases accumulate once per operand use into one public gradient.
- Multiple distinct inputs return gradients under their declared paths.
- Selecting one output leaves unrelated primal output behavior intact and does
  not require its cotangent.

### Aggregate and layout behavior

- `wrt=("particles",)` and
  `wrt=("particles.velocity", "particles.mass")` both succeed.
- The root gradient is a packed `TensorStorage`; field views contain `f32`
  accumulation values.
- Non-differentiable `tag` leaves are absent from the tangent layout.
- Non-contiguous and negative-stride TensorViews scatter gradients back to the
  owning Storage correctly.

### Backend coverage

- All CPU tests in `ProgramExecutionTests` pass.
- Available GPU tests in `ProgramGpuExecutionTests` pass without forcing the
  architecture back to CPU.
- Direct Kernel VJP backend coverage remains in
  `test_storage_vjp_contract.py`; Module coverage supplements rather than
  replaces it.

### Discovery and regression gates

Run all of the following:

```shell
PYTHONPATH=python .venv/bin/python -m pytest python/tests/test_program_execution.py -q
PYTHONPATH=python .venv/bin/python -m pytest python/tests -q --tb=short
ctest --test-dir build -R '^vernon-python-module-program$' --output-on-failure
ctest --test-dir build --output-on-failure
```

The focused Program file must report a non-zero collected test count. Verified
on 2026-09-03:

```text
test_program_execution.py: 22 passed
python/tests: 455 passed, 52 capability skips, 405 subtests passed
CTest: 524/524 passed
```

The capability skips are existing unavailable-backend gates, not skips or
xfails for the four historical CPU regressions.

## Permanent test-change policy

- Treat the restored tests as requirements.
- Change a test only when it contradicts a documented canonical contract, as
  the dictionary-versus-packed-owner assertion did.
- Do not replace scalar arguments with rank-zero TensorStorage.
- Do not add zero cotangents for unselected outputs.
- Do not delete alias or aggregate fan-in cases.
- Do not accept expected failures.
- Fix one ownership layer: frontend typing, canonical Program construction,
  compiler finalization, or runtime binding. Avoid Python-side compatibility
  branches and duplicate ABI logic.
- Do not restore retired compiler/pipeline artifacts, public pass descriptors,
  parameter reconstruction, or a second Program execution manifest.
