# Python DSL v3 completion plan

This document tracks work that remains after the initial language-v3 API and
module-boundary refactor. Passing the existing suite is not sufficient for
completion: each section below has explicit semantic and execution gates.

## 1. Language contract test matrix

- [x] Test every safe scalar promotion: `f16 -> f32 -> f64`, `i32 -> float`,
      and `u32 -> float`.
- [x] Reject implicit floating narrowing, floating-to-integer conversion,
      Boolean arithmetic, and dynamic `i32`/`u32` mixing.
- [x] Test contextual literals on both operand sides, assignments, calls,
      returns, constructors, comparisons, and `/`.
- [x] Test partial helper annotations, inferred void/value returns, conflicting
      returns, unresolved helpers, and recursion diagnostics.
- [x] Test separate helper specializations for f32/f64 and concrete Tensor
      shapes, including imported qualified helpers and feature variants.
- [x] Test `Vector` and `Matrix` inference, empty/ragged input rejection, mixed
      element promotion, and legacy constructor parity.
- [x] Test intrinsic method/function and operator/function parity, especially
      `z.norm()`/`norm(z)` and `x ** y`/`pow(x, y)`.
- [x] Enforce annotations on all entry ABI parameters and externally visible
      results.
- [x] Test branch merge and loop-carried widening plus stable diagnostics for
      unsupported control flow.

## 2. Complete semantic inference

- [x] Represent integer and floating literals as type variables until a
      surrounding constraint or defaulting phase resolves them.
- [x] Use one promotion/conversion solver for operators, intrinsics,
      constructors, assignments, calls, branch merges, and returns.
- [x] Analyze loop-carried values to a fixed point instead of one pass.
- [x] Infer every operation supported by lowering, including structs,
      `matmul`, textures, generated builtins, resources, and method sugar.
- [x] Reject reachable unresolved helper types. Unreachable generic helpers
      must have an explicit and documented policy.
- [x] Include concrete helper specialization keys in semantic cache inputs.

## 3. Typed semantic model boundary

- [x] Keep Tensor as one semantic type and represent compute-parameter
      addressability/access on typed values instead of `ConcreteType.kind`.
- [x] Build typed expressions and statements for every reachable function.
- [x] Record lvalues, effects, branch merges, and termination in the model.
- [x] Validate the complete typed call graph before lowering.
- [x] Make MLIR lowering consume only typed semantic nodes; it must not perform
      type inference or overload selection.
- [x] Add tests that typed-model dumps and diagnostics are deterministic.

## 4. Backend execution coverage

- [x] Execute inferred scalar and Tensor helpers on CPU.
- [x] Compare supported inference programs on CPU, CUDA, and Vulkan.
- [x] Cover integer true division, mixed literals, Vector/Matrix construction,
      operator/intrinsic parity, and multiple helper specializations.
- [x] Characterize f16/f64 support per backend and require explicit
      unsupported-target diagnostics where execution is unavailable.
- [x] Compare MLIR, reflection, specialization symbols, semantic cache keys,
      bundle JSON, and artifact hashes across repeated builds.

Tests requiring unavailable CUDA or Vulkan devices may skip with a precise
capability reason; CPU reference execution is mandatory.

## 5. Finish physical module separation

- [x] Move bundle types, reflection, parameter merging, planning, and
      serialization implementations out of the current planner monolith.
- [x] Move Runtime resource, kernel, pipeline, and session implementations out
      of the current session monolith while keeping state ownership in session.
- [x] Move shader declaration, descriptor parsing, cooking, and artifact I/O
      implementations into their modules without circular re-export wrappers.
- [x] Add dependency tests that frontend does not import Runtime and pure bundle
      planning does not import frontend or Runtime.
- [x] Preserve byte-identical schema-2 manifests and artifact selection.

## 6. Measurable quality gates

- [x] Add `coverage` to the build/test dependency group.
- [x] Publish line and branch coverage for `python/vernon_dsl`.
- [x] Start with an observed baseline; then require at least 90% line coverage
      for language/frontend modules and 85% branch coverage for inference and
      type parsing.
- [x] Keep Ruff lint/format, all Python tests, complete CTest, clean wheel
      install, bundled Runtime source lookup, and native smoke execution green.

Observed Windows baseline after the physical split (2026-07-23): 82% total
coverage, 87.2% line/76.1% branch coverage for language+frontend, and 87.1%
line/76.5% branch coverage for inference+type parsing+solver. The target gates
now enforce 90% language/frontend line coverage and 85% inference/type-parser
branch coverage in Windows CI. The first passing measurement was 90.06% and
87.08%, respectively.

## Execution order

1. Add failing contract tests for section 1.
2. Complete the solver and semantic model in sections 2-3.
3. Add backend numeric and deterministic-output tests in section 4.
4. Finish physical module separation in section 5.
5. Measure coverage, close uncovered branches, and enforce section 6.
6. Only after this plan is complete, begin
   `specs/language/future_language_roadmap.md`.

Check an item only after its tests pass. If implementation reveals a language
decision not covered by `specs/language/contract.md`, update that contract and
the compiler design record before proceeding.
