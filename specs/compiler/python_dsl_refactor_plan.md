# Python DSL language and modularity refactor plan

## Goal

Make `vernon_dsl` a versioned, testable, tensor-first Python dialect that feels
like ordinary Python without sacrificing deterministic cross-backend typing.
The refactor first establishes the language contract and type inference, then
separates compiler phases and runtime/tooling modules. General control-flow
support is a subsequent project.

Breaking Python DSL and public API changes are allowed, with a v2-to-v3
migration guide. The work covers all of `python/vernon_dsl`, but language and
frontend changes must land before Runtime, bundle, and asset module moves.

## Current findings

The frontend is not a general Python-to-MLIR compiler. It currently supports a
closed AST subset with required positional annotations, static or specialized
Tensor shapes, limited statements, and a closed intrinsic set.

- Scalar arithmetic and `while` loop-carried values lower reasonably generally.
- `if` merges existing SSA values but cannot contain `return`.
- `for` accepts only literal `range(...)`, positive steps, and no loop-carried
  assignment.
- `and` and `or` eagerly evaluate operands instead of using Python
  short-circuit semantics.
- Dynamic `range`, `break`, `continue`, conditional expressions, chained
  comparisons, destructuring, and structured early return are unsupported.
- Many operations emit `vernon.intrinsic`; producing textual MLIR does not prove
  that every backend supports the operation and type combination.
- Exact type checking currently happens while emitting MLIR. Consequently,
  helper parameters and returns require annotations and mixed literals often
  require unnecessary casts.
- `f16`, `f64`, and `Array` have mostly textual-IR coverage rather than a proven
  CPU/CUDA/Vulkan/graphics execution contract.

Confirmed cleanup candidates:

- `shader_assets.serialize_runtime_pipeline_bundle`
- the unreferenced `CpuAotRuntime`
- duplicate `types.Tensor` and `types.Texture` constructors superseded by the
  public runtime classes' `__class_getitem__`
- `ShaderAssetError`
- the `@compute` compatibility spelling
- native Python `Compiler.compile` and `compile_program` compatibility methods,
  after migrating parity tests to `compile_program_result`

Keep the schema-1 rejection guards, module graph, shared definitions,
development native-loader fallback, and wheel Runtime source locator. They are
active boundaries, not obsolete implementations.

## Language v3 decisions

### Public spellings

- `@kernel` is the only compute-entry decorator; remove `@compute`.
- `vd.Tensor` and `vd.Texture` remain dual-purpose runtime objects and
  annotation spellings. Remove only the duplicate constructors in `types.py`.
- Keep `Buffer` as an explicit low-level resource annotation.
- Remove `Array` from the stable language until it has expression semantics and
  backend execution coverage.
- Generated builtin intrinsics such as `vertex_id()` are the preferred authoring
  surface. Keep raw `builtin("...")` as the low-level ABI spelling.
- Accept Python `int` and `float` as the default `i32` and `f32` annotations and
  explicit casts. Keep `vd.i32`, `vd.u32`, `vd.f16`, `vd.f32`, and `vd.f64` for
  exact control.
- Prefer `import vernon_dsl as vd` in maintained examples.

### Inference boundary

Entry points (`@kernel`, `@vertex`, and `@fragment`) retain parameter and
externally visible result annotations because they define the Runtime and
reflection ABI. `@func` parameters and returns may be wholly or partially
unannotated.

Helpers are monomorphized from concrete call-site argument types. The
specialization key is:

```text
(qualified function, concrete argument types, enabled features)
```

The call graph remains non-recursive. A helper used with different scalar
precision or Tensor shapes receives separate, deterministically named MLIR
instances.

### Numeric policy

Use Python-like syntax with safe, deterministic static conversion:

- Keep integer and floating literals untyped until operand, constructor,
  assignment, call, or return context constrains them.
- Default unconstrained integer and floating literals to `i32` and `f32`.
- `/` follows Python true-division semantics. Integer operands produce `f32`
  unless a wider floating context exists.
- Allow safe widening, including `f16 -> f32 -> f64` and integer-to-floating
  promotion.
- Do not implicitly narrow floating precision or convert floating values to
  integers.
- Keep `bool` outside ordinary numeric arithmetic.
- Reject mixed types that have no safe common representation. In particular,
  dynamic `i32`/`u32` mixing remains an error until the language has an `i64`
  or an explicit common-type rule.
- Operator and intrinsic spellings use the same type solver. For example,
  `pow(x, 2)` and `x ** 2` must infer identically.

Local variables infer their type from first assignment. Reassignment and branch
merges must preserve the type or use a safe common widening. Return types are
inferred from all reachable returns and must agree after promotion.

### Pythonic value construction

Add `vd.Vector([...])` and `vd.Matrix([...])`. List or tuple literals are
accepted only in these constructor contexts initially. Shape and common element
type are inferred. Existing fixed-size constructors may remain as thin aliases
during migration.

Once expression types are known, support intrinsic method sugar such as
`z.norm()` by resolving it to the same canonical intrinsic as `vd.norm(z)`.
Do not create a second lowering path.

The intended non-parallel authoring result is:

```python
@vd.func
def complex_sqr(z):
    return vd.Vector(
        [
            z[0] ** 2 - z[1] ** 2,
            z[1] * z[0] * 2,
        ]
    )


@vd.kernel
def paint(t: float, pixels: vd.Tensor[vd.f32, (None, None)], ...):
    # Explicit invocation indexing remains until automatic parallel fields are
    # designed separately.
    ...
```

Automatic `for i, j in pixels`, global fields, and automatic parallelization
are explicitly outside this refactor.

## Target architecture

```text
Python source
  -> project loading and feature specialization
  -> normalized AST
  -> typed semantic model and helper monomorphization
  -> MLIR lowering
  -> native compiler
  -> reflection and bundle planning
  -> Runtime execution or offline artifact cooking
```

Dependencies flow only forward:

- language contracts are dependency roots;
- frontend never imports Runtime;
- bundle planning does not import frontend or Runtime;
- offline cooking and interactive execution share native loading, stage
  compilation, and bundle planning;
- MLIR lowering consumes typed semantic nodes and performs no inference.

The typed semantic model must include branch merge, lvalue, effect, and
termination concepts even though general control flow is deferred. This avoids
redesigning the frontend when that work begins.

## Phase 1: language contract and characterization

Add `specs/language/contract.md` as the canonical language reference. Specify:

- decorators and function domains;
- entry ABI versus inferred helper rules;
- type and promotion algebra;
- supported/rejected statements and expressions;
- Tensor/Buffer/Texture semantics;
- intrinsic and builtin mappings;
- feature specialization and interface-location stability;
- frontend-only versus backend-executable support;
- stable diagnostics and language-version rules.

Add `python/tests/test_language_contract.py` with:

- accepted and rejected AST syntax cases;
- numeric promotion and contextual-literal cases;
- stable diagnostics for unsupported constructs;
- `FrontendCompileRequest` shape and captured-constant specialization;
- frontend emission versus CPU/CUDA/Vulkan/graphics execution cases;
- characterization for `f16`, `f64`, `Array`, casts, tensor comparisons,
  indexing, and entry-struct returns.

Bump `vernon.frontend_version` and semantic cache identity from 2 to 3.

## Phase 2: breaking cleanup

Before moving modules:

1. Migrate tests and examples from `@compute` to `@kernel`, then remove
   `@compute` from decorators, module graph, compiler, cooker, and exports.
2. Delete confirmed dead and compatibility-only Python symbols listed above.
3. Make `int`/`float` documented and tested default type spellings.
4. Reject eager `and`/`or` until real short-circuit lowering exists.
5. Reject nested `return` consistently, including inside `while`.
6. Fix the internal `index` type representation.
7. Update README, examples, CLI tests, native parity tests, and migration notes.

Do not remove schema rejection logic or the development/wheel delivery helpers.

## Phase 3: typed semantic model and inference

Implement inference before further module splitting:

1. Introduce source types, literal type variables, concrete types, typed
   expressions, typed statements, and typed function instances.
2. Parse entry ABI annotations independently from helper inference.
3. Build and validate the call graph before lowering.
4. Instantiate helpers recursively from concrete call sites; reject recursion
   and unresolved parameters.
5. Solve contextual literals and common numeric types centrally.
6. Infer locals, branch merge types, constructor shape/element types, and
   helper return types.
7. Resolve operators, intrinsic functions, and typed intrinsic methods to one
   canonical semantic operation.
8. Lower only fully typed functions to MLIR.

Required tests include:

- unannotated `complex_sqr(z)`;
- separate f32/f64 and shape specializations of one helper;
- `* 2`, `iterations * 0.02`, and integer true division without manual casts;
- Vector/Matrix common-element inference;
- operator/intrinsic parity;
- deterministic symbols, MLIR, reflection, and cache keys;
- clear errors for ambiguous literals, unsafe narrowing, and unresolved helper
  types;
- matching CPU/CUDA/Vulkan numeric results.

## Phase 4: language and frontend modules

Create:

```text
vernon_dsl/language/
  scalar_types.py
  syntax.py
  ast_utils.py

vernon_dsl/frontend/
  model.py
  request.py
  type_parser.py
  inference.py
  monomorphize.py
  interfaces.py
  analysis.py
  lowering.py
```

- `scalar_types.py` is the sole scalar/NumPy/MLIR type registry.
- `syntax.py` owns closed decorator, intrinsic, builtin, and metadata contracts.
- `ast_utils.py` replaces duplicated dotted-name/decorator helpers.
- `compiler.py` becomes a thin public facade and phase orchestrator.
- Preserve byte-identical MLIR during mechanical extractions; semantic changes
  must have dedicated tests and commits.

Keep `shader_contracts.py` and `struct_methods.py` cohesive unless their contents
are fully absorbed by the new language/analysis layers.

## Phase 5: project and asset AST normalization

Centralize feature, decorator, import, and declaration parsing so
`module_graph.py` and `pipeline_asset(...)` parsing cannot drift.

Preserve and test this pass order:

```text
load imports
  -> specialize features
  -> normalize struct methods
  -> prune entry and validate call graph
  -> infer and type-check
  -> lower
```

Method normalization must remain before entry pruning because method calls hide
helper reachability.

## Phase 6: bundle, native, Runtime, and cooker modules

Split `pipeline_compile.py` into:

```text
vernon_dsl/bundle/
  types.py
  reflection.py
  parameters.py
  planner.py
  serialize.py
```

Keep it pure and preserve manifest JSON, stage IDs, hashes, slots, and artifact
selection byte-for-byte.

Add a shared native loader and stage compiler so the cooker no longer reaches
`runtime._native` and interactive/offline stage compilation does not duplicate
cache orchestration.

Split `runtime.py` incrementally into internal modules:

```text
vernon_dsl/_runtime/
  session.py
  resources.py
  kernel.py
  pipeline.py
```

All architecture, native context, generation, and child-lifetime state belongs
to `session.py`. Keep `runtime.py` as the public re-export facade.

Split `shader_assets.py` into declaration, descriptor parsing, cooking, and
artifact I/O modules without changing schema-2 output.

## Validation gates

After every phase:

1. Run Ruff lint and format checks.
2. Run all Python tests.
3. Run the complete native CTest suite.
4. Compare MLIR, reflection JSON, bundle JSON, artifact hashes, semantic cache
   keys, and generated helper symbols where the phase is intended to be
   behavior-preserving.
5. Run CPU, CUDA, Vulkan, OpenGL, and OpenGL ES tests where available.
6. Build a clean wheel, install it in a fresh environment, and verify compiler,
   bundled Runtime source, and native execution.

Update the compiler design record when introducing the promotion lattice,
monomorphization key, typed semantic model, or a non-obvious pass-order
constraint.

## v2-to-v3 migration output

Provide:

- `@compute` to `@kernel` mechanical migration;
- removed import paths and symbols;
- redundant casts that inference makes unnecessary;
- explicit casts required for unsafe narrowing;
- changes to `int`/`float` meaning;
- helper specialization effects on generated symbols and cache keys;
- unsupported constructs that now fail earlier and consistently.

## Follow-up: general Python syntax

Begin only after language v3 inference and module boundaries are stable:

1. dynamic `range` and `scf.for` iter_args;
2. true short-circuit `and`/`or`, conditional expressions, and chained
   comparisons;
3. `break` and `continue`;
4. structured early return;
5. additional destructuring and local aggregate syntax;
6. integer intrinsic expansion and matrix `@`;
7. separately designed global fields and automatic parallel loops.

Each addition requires language-contract updates, typed semantic nodes,
frontend rejection tests, MLIR structural tests, and backend execution tests.
