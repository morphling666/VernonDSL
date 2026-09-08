# VernonDSL roadmap

Status: forward-looking project roadmap.

Current contracts and architecture are indexed in [`README.md`](README.md).
Completed milestones and migrations belong in release notes and Git history.

## 1. Release verification

- Complete the contract-driven cross-backend language matrix in
  [`testing/cross_backend_language_testing_plan.md`](testing/cross_backend_language_testing_plan.md).
- Run compiler matrices independently from runtime device matrices on Linux,
  macOS, and Windows.
- Require explicit capability reasons for every skipped backend case.
- Keep full CTest, Python, MLIR lit, native example, packaging, and WASM gates
  green.
- Add deterministic and malformed-input fuzzing for source, Program,
  ArtifactSystem, reflection, TensorView, and Command DAG boundaries.

## 2. Language v4

Remaining declaration gates are defined in
[`language/future_language_roadmap.md`](language/future_language_roadmap.md):

- complete cross-backend workgroup storage, barrier, and atomic execution
  evidence;
- complete contract-matrix coverage for every supported language region;
- preserve deterministic semantic IR, cache identity, reflection, and target
  diagnostics.

New language features must not bypass typed semantic IR or introduce another
Value/Storage/Resource category.

## 3. Autodiff

- Add bounded complete-workgroup replay for captured static/dynamic GPU tape
  without GPU-to-host tape readback.
- Select specialized operator VJPs only when measured plans beat generic replay
  and preserve the same Program ABI.
- Complete independent wasm32 Program VJP validation before advertising browser
  VJP.
- Specify custom compute VJP typing, capture, identity, and deployment before
  exposing a public declaration.
- Keep graphics VJP rejected until differentiability domains and versioned
  rules exist for rasterization, visibility, depth, blend, and texture
  sampling.
- Treat JVP, batched transforms, higher-order AD, Hessians, HVPs, and persistent
  gradient buffers as independent future proposals.

## 4. Execution and scheduling

- Measure and improve fairness when multiple CPU dispatches share the bounded
  worker pool.
- Extend asynchronous multi-frame execution only through resolved command
  plans, explicit completion, and retained resource ownership.
- Continue optimizing transfer coalescing, device residency, render-scope
  fusion, checkpoint selection, and command submission without changing
  Program semantics.
- Keep engine-owned graph/encoder embedding separate from the canonical
  Program load, bind, and invoke API.
- Add distributed execution only after a separate ownership, topology,
  synchronization, failure, and deployment contract is accepted.

## 5. Graphics and resources

- Expand capability-tested format, sample-count, storage-image, cube-map, and
  mipmap coverage across applicable backends.
- Add graphics stages or topology only through the stage registry and a
  coordinated compiler contract.
- Preserve explicit draw counts, attachment versions, subresource hazards, and
  publication semantics while improving native render-pass fusion.
- Keep CPU graphics outside the product contract unless a software rasterizer
  is designed and accepted explicitly.

## 6. Platform work

- Maintain native CPU/GPU and wasm32 CPU release gates as separate capability
  profiles.
- Add a pthread-enabled web profile only with explicit Worker,
  `SharedArrayBuffer`, COOP/COEP, scheduling, and packaging requirements.
- Improve Apple, Windows, and Linux packaging without adding target-specific
  source-language behavior.
- Keep compiler target availability distinct from runtime hardware
  availability.

## 7. Host language

The proposed Host domain is specified in
[`future/host_language.md`](future/host_language.md). Before implementation it
requires:

- a versioned Host semantic and extern ABI;
- deterministic interpreted and AOT behavior;
- desktop and wasm32 linking/lifetime rules;
- one acceptance workload with matching interpreted, native, and browser
  results.

It must compose Programs through the public Program lifecycle rather than
embedding Python or exposing private Runtime scheduling objects.

## 8. Performance policy

- Optimize only from reproducible reports stored outside `specs/`.
- Keep correctness, failure transaction, determinism, and memory-budget gates
  separate from timing gates.
- Record backend, device, driver/API, compiler/Program versions, workload,
  policy, and measurement method with each report.
- Do not turn workload-size heuristics into hidden semantic or compatibility
  behavior.
