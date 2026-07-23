# Post-refactor language roadmap

This roadmap starts only after the Python frontend typed-model refactor,
backend parity coverage, physical module separation, and coverage gates in
`specs/compiler/python_dsl_v3_followup.md` are complete. The current refactor
must establish the semantic boundary; it must not grow new syntax at the same
time.

## Design position

Vernon is a statically typed GPU/graphics DSL, not unrestricted Python.
Dynamic Python `list` and `dict` semantics require allocation, ownership,
capacity management, hashing, and failure behavior that do not map uniformly
to shader and compute backends.

`Tensor[Struct, shape]` is not a substitute for a list:

- Tensor is homogeneous and rectangular with a fixed or specialized shape.
- List has logical length, capacity, mutation, and potentially dynamic growth.
- Numeric Tensor lowering currently assumes scalar elements in many
  operations.
- `Buffer[Struct]` can represent fixed AoS storage, but not dynamic list
  ownership or growth.

Runtime `dict` is therefore not a core-language priority. Compile-time maps and
library data structures over buffers are preferable.

## Phase 1: fixed-size value aggregates

- [ ] Add a new, precisely specified `Array[T, N]`; do not restore the removed
      v2 placeholder API.
- [ ] Allow Array elements to be scalar, Tensor value, Struct, enum, or another
      fixed Array when ABI rules permit.
- [ ] Add heterogeneous Tuple types, tuple construction, indexing, and
      destructuring.
- [ ] Add non-owning `Slice[T]` or `TensorView[T, rank]` with runtime extents,
      strides, access mode, and explicit lifetime restrictions.
- [ ] Define Array/Tuple/Struct ABI layout consistently for CPU, CUDA, Vulkan,
      OpenGL compute, and host/device shared functions.

## Phase 2: complete structured control flow

- [ ] Add short-circuit `and` and `or`.
- [ ] Add `break` and `continue`.
- [ ] Support early returns with explicit region/CFG termination.
- [ ] Add conditional expressions.
- [ ] Define dynamic range bounds and steps, including backend legality and
      termination diagnostics.

## Phase 3: explicit generics

- [ ] Add type parameters and constraints such as `Numeric`, `Integer`, and
      `Float`.
- [ ] Add const generics for dimensions and capacities.
- [ ] Make explicit and inferred specializations share one deterministic key.
- [ ] Add type aliases and compile-time assertions.
- [ ] Retain call-site inference as convenience, not as the only way to express
      polymorphism.

## Phase 4: GPU memory and synchronization model

- [ ] Model private, function-local, workgroup/shared, storage, uniform, and
      host-visible memory explicitly.
- [ ] Define references/views separately from owned values.
- [ ] Add workgroup arrays and backend-validated alignment/layout.
- [ ] Add atomic operations with explicit supported element types.
- [ ] Add barriers and memory ordering/scope semantics shared by CUDA, SPIR-V,
      and CPU reference execution.
- [ ] Extend effect analysis to reads, writes, atomics, barriers, and aliasing.

## Phase 5: algebraic data types

- [ ] Add enums with deterministic underlying representation.
- [ ] Add tagged unions and `Option[T]`.
- [ ] Add exhaustive `match`.
- [ ] Define copy, equality, and ABI rules for nested Struct/Array/enum values.
- [ ] Decide and document mutable Struct-field policy and AoS/SoA conversion.

## Phase 6: compile-time data

- [ ] Add deterministic constant-expression evaluation.
- [ ] Add compile-time conditionals and generated fixed lookup tables.
- [ ] Add `ConstMap[K, V, N]` for compile-time-known keys.
- [ ] Lower small fixed maps to sorted tables, switch trees, or perfect hashes
      according to a documented deterministic policy.

## Non-core library work

- [ ] Implement fixed-capacity vector/list containers over Array or Buffer.
- [ ] Implement GPU hash maps over Buffer plus atomics only after the atomic and
      memory models are stable.
- [ ] Keep allocation strategy, capacity overflow, collision behavior, and
      concurrency policy explicit in library types.

## Acceptance criteria for every phase

- The language contract is updated before implementation.
- Typed semantic nodes fully represent the feature before lowering changes.
- CPU reference behavior is covered by numeric tests.
- CUDA and Vulkan behavior is compared where the feature is supported.
- Unsupported target combinations fail with explicit diagnostics.
- Cache identity and generated symbols remain deterministic.
