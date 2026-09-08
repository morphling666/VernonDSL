# Cross-Backend Language Testing Plan

Status: active implementation plan.

## Goals

- Use `specs/language/contract.md` as the source of truth for supported language regions.
- Cover every construct across its valid and invalid stage, type, effect, and address-space regions.
- Require frontend acceptance, target compilation, cooked Program execution, and a numeric or effect oracle whenever runtime behavior is part of the contract.
- Run backend-independent semantic tests on every applicable backend.
- Skip only when the platform, device, API version, or required capability is unavailable, with an explicit reason.
- Run WASM as an independent required CPU/WASM release gate.

## Test model

The suite uses a contract matrix:

- Exhaustively cover each individual construct across its valid and invalid regions.
- Cover important pairwise interactions between constructs.
- Do not claim exhaustive coverage of every possible source-program combination.

Compile and runtime results are separate:

- Every enabled compiler target must compile every applicable positive case and reject every applicable negative case.
- Runtime tests execute on every backend that can create the required context/device and satisfies the case requirements.
- A compile, load, bind, or execution failure is a failure, not a skip.
- Backend-native ABI and interop tests remain backend-specific.

Backend artifacts remain target-specific, but fixture registration, manifest lookup, parameterized test bodies, and capability filtering are shared. Tests must not cook assets at runtime.

## Phase 0: Audit and coverage inventory

- [x] Audit the current frontend, compiler, cooked asset, runtime, and CTest coverage.
- [x] Audit current compiler/runtime capability APIs, backend enumeration, CMake bundle generation, and skip behavior.
- [x] Select contract-matrix coverage with important pairwise interactions.
- [x] Select WASM as an independent required release gate.
- [x] Create a feature inventory from:
  - `specs/language/contract.md`
  - `specs/language/tensor_view.md`
  - `specs/language/future_language_roadmap.md`
  - `python/vernon_dsl/shader_contracts.py`
- [x] Give every supported language feature and region a stable test ID.
- [x] Record for every test ID:
  - valid and invalid regions;
  - required compile capabilities;
  - required runtime capabilities;
  - required test layers;
  - expected diagnostic or runtime oracle.
- [x] Correct outdated language specification statements found during the inventory.

## Phase 1: Canonical backend test matrix

- [ ] Add `source/tests/support/backend_test_matrix.h`.
- [ ] Define one backend row for each compiler/runtime pair:
  - CPU;
  - CUDA;
  - Vulkan;
  - DirectX 12;
  - Metal;
  - OpenGL;
  - OpenGL ES.
- [ ] Define test requirements for:
  - compute;
  - graphics;
  - storage buffers;
  - device atomics;
  - f32 atomic add;
  - f64 atomic add;
  - texture and sampler operations;
  - minimum OpenGL/OpenGL ES API versions;
  - backend-specific ABI or interop capabilities.
- [ ] Derive availability from existing compiler, runtime, and context capability APIs.
- [ ] Distinguish platform-not-built, device/context-unavailable, and capability-unsupported skip reasons.
- [ ] Consolidate RHI device/context ownership in `source/tests/support/runtime_rhi_test_utils.h`.
- [ ] Remove duplicated `OwnedGpuRuntime` implementations after migration.
- [ ] Add `python/tests/backend_test_matrix.py` with the same backend and requirement semantics.
- [ ] Reject catch-all exception-based skips in the shared harness.

## Phase 2: Fixture and manifest matrix

- [ ] Replace repeated per-backend fixture blocks in `source/tests/CMakeLists.txt` with one enabled-target loop.
- [ ] Add OpenGL ES to the fixture target matrix.
- [ ] Generate a test-only `(fixture_id, target) -> manifest path` table.
- [ ] Ensure each target-specific artifact is built once and reused by all tests for that target.
- [ ] Remove grouped `VERNON_*_METAL_MANIFEST`, `VERNON_*_VULKAN_MANIFEST`, and equivalent path macros after consumers migrate.
- [ ] Parameterize `source/tests/runtime/runtime_module_program_gpu_c_api_test.cpp` as the reference suite.
- [ ] Run compute Module forward, TensorView chain, dynamic shape/grid reuse, and Module VJP on every applicable backend.
- [ ] Verify expected run and skip sets on the current platform.

## Phase 3: Contract-driven language cases

- [ ] Add `python/tests/language_contract_cases.py`.
- [ ] Give each case:
  - stable contract ID;
  - source construct;
  - valid regions;
  - invalid regions;
  - capability requirements;
  - expected diagnostic;
  - runtime oracle or explicit `runtime_not_applicable`.
- [ ] Migrate reusable case tables from:
  - `python/tests/test_language_contract.py`;
  - `python/tests/test_compiler.py`;
  - `python/tests/test_type_analysis_coverage.py`.
- [ ] Cover scalar, vector, matrix, Tensor, aggregate, and conversion rules.
- [ ] Cover expressions, indexing, control flow, helper specialization, and builtins.
- [ ] Cover TensorView access modes, address spaces, shape/rank rules, and alias constraints.
- [ ] Cover Texture and Sampler operations by legal stage and access mode.
- [ ] Cover atomics, barriers, workgroup memory, and dispatch constraints.
- [ ] Cover compute, vertex, and fragment entry regions.
- [ ] Cover Module composition and supported VJP regions.
- [ ] Add a coverage audit that fails when a supported region lacks its required test layers.

## Phase 4: Compile and runtime acceptance

- [ ] Run every positive frontend case through semantic analysis.
- [ ] Run every negative frontend case and verify its diagnostic.
- [ ] Run applicable cases through MLIR verification and target compilation.
- [ ] Cook runnable cases as canonical Program assets.
- [ ] Execute each runnable case with a numeric, publication, ordering, or effect oracle.
- [ ] Run CPU-compatible cases on CPU.
- [ ] Run GPU compute cases on every compute-capable backend.
- [ ] Run graphics cases on every graphics-capable backend.
- [ ] Create a real host-owned OpenGL/OpenGL ES context before querying context capabilities.
- [ ] Require OpenGL 4.3+ or OpenGL ES 3.1+ for compute tests.
- [ ] Treat load, bind, dispatch, synchronization, and result mismatches as failures.
- [ ] Verify that only capability checks produce skips.

## Phase 5: Existing suite migration

- [ ] Parameterize `source/tests/runtime/runtime_gpu_autodiff_test.cpp`.
- [ ] Parameterize `source/tests/runtime/runtime_module_graphics_program_c_api_test.cpp`.
- [ ] Migrate `python/tests/test_kernel_runtime.py` to the shared Python backend matrix.
- [ ] Migrate `python/tests/test_module_graphics_controls.py` to the shared Python backend matrix.
- [ ] Remove duplicated Metal/Vulkan semantic test methods.
- [ ] Remove duplicated hardcoded backend lists.
- [ ] Keep Vulkan native interop tests backend-specific.
- [ ] Keep Metal argument-buffer ABI tests backend-specific.
- [ ] Keep DirectX 12 native state/interop tests backend-specific.
- [ ] Keep OpenGL/OpenGL ES callback and API-version tests backend-specific.

## Phase 6: CI and release gates

- [ ] Register the complete frontend language-contract suite in CTest.
- [ ] Remove the discrepancy between full pytest coverage and CTest Python coverage.
- [ ] Run compile matrices in Linux, macOS, and Windows jobs for every enabled target.
- [ ] Run runtime matrices against all capabilities available on each CI machine.
- [ ] Report backend, missing capability, and probe diagnostic for every skip.
- [ ] Add an independent required WASM build and runtime job.
- [ ] Run applicable frontend, compile, cooked Program, and numeric cases under WASM.
- [ ] Add source guards against:
  - new backend-specific copies of backend-independent semantic tests;
  - new duplicated backend lists;
  - unregistered language acceptance cases;
  - catch-and-skip behavior;
  - runtime asset cooking.

## Phase 7: Final validation

- [ ] Run the complete native build.
- [ ] Run all CTest tests.
- [ ] Run the complete Python test suite.
- [ ] Run all MLIR lit tests.
- [ ] Run the WASM build and runtime gate.
- [ ] Validate expected run/skip sets for each tested platform.
- [ ] Confirm every supported contract region has the required compile and runtime coverage.
- [ ] Update the language testing specification with the final matrix and gate commands.
