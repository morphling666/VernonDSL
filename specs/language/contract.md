# Vernon Python language contract

This document defines Python frontend language version 3. The frontend parses
source with `ast` and never imports or executes a shader module.

## Functions and ABI

- `@kernel`, `@vertex`, and `@fragment` declare externally visible entries.
  Their parameters and non-`None` results require annotations.
- `@func` declares a non-recursive, stage-polymorphic helper. Helper parameter
  and result annotations are optional and may be partial.
- Helpers are specialized by `(qualified function, concrete argument types,
  enabled features)`. Generated names and specialization order are
  deterministic.
- Unreachable helpers are pruned before inference. Reachable helpers must
  resolve completely; there is no diagnostic obligation for an unreachable
  generic body.
- Functions accept required positional parameters only. Calls do not accept
  keyword arguments.
- `@struct` declares an immutable aggregate used by the language and entry
  interfaces. `@func(shared=True)` is restricted to host/device common types
  and operations.

## Types

`bool`, `i32`, `u32`, `f16`, `f32`, and `f64` are scalar types. Python `int`
and `float` are aliases for `i32` and `f32` in annotations and explicit casts.
`Tensor[element, shape]` is the numeric aggregate type. `vec*` and `mat*` are
fixed-shape aliases. `Buffer`, `Texture`, and `Sampler` are explicit resources.
`Array` is not a language-v3 type.

Entry Tensor dimensions may be `None` and must be specialized to positive
runtime dimensions before lowering. Value Tensor shapes are static.

Integer and floating literals remain contextual until typing is solved.
Unconstrained literals default to `i32` and `f32`. Safe implicit conversions
are integer-to-floating and `f16 -> f32 -> f64`. Floating narrowing,
floating-to-integer conversion, and dynamic `i32`/`u32` mixing require an
explicit cast. `bool` does not participate in numeric arithmetic. `/` is true
division; integer operands produce at least `f32`.

Locals infer from their first assignment. Reassignment and control-flow merges
must preserve the type or use a safe common widening. All reachable returns
must have a safe common type.

`Vector([...])` and `Matrix([...])` are the aggregate constructors. Their
literal list or tuple must be rectangular and have a common element type.
Legacy fixed-size constructors are aliases. Typed intrinsic methods resolve to
the same canonical operation as their function spelling.

## Supported syntax

The stable statement subset is `pass`, expression statements, simple local or
indexed assignment, annotated assignment, augmented assignment, `return`,
`if`, literal positive-step `range` loops, and `while`. Entry-interface
metadata, feature branches, and imported DSL declarations are normalized
before typing.

The expression subset includes names, numeric and Boolean literals, arithmetic,
unary operations, one comparison, calls, supported attributes, and indexing.
`and` and `or`, conditional expressions, chained comparisons, dynamic `range`,
`break`, `continue`, destructuring, recursion, and nested/early `return` are
rejected with frontend diagnostics.

## Intrinsics and backend support

Generated builtin functions are the preferred authoring API. `builtin("...")`
is the low-level ABI annotation. Operators, intrinsic functions, and intrinsic
methods share one typed operation registry.

Frontend acceptance guarantees well-typed Vernon MLIR, not that every target
implements the operation. Backend capability tests define executable support.
In particular, `f16` and `f64` require target coverage; unsupported target/type
combinations must fail explicitly rather than silently narrow.

## Specialization and diagnostics

Project processing order is:

1. load imports;
2. specialize features and captured constants;
3. normalize struct methods;
4. prune to the selected entry and validate the call graph;
5. infer and type-check;
6. lower typed nodes.

Method normalization precedes pruning because receiver syntax can hide helper
reachability. Interface locations are allocated before feature pruning, so
disabled fields reserve their locations.

Semantic cache identity includes frontend version 3, all source dependency
digests, entry, enabled features, concrete Tensor shapes, captured constants,
workgroup size, and helper specialization types. Diagnostics include source
path, one-based line and column, and a stable reason string.

Pipeline schema-1 rejection, source module graph validation, development native
loading, and wheel Runtime source discovery remain supported delivery
boundaries and are independent of the language version.
