# Vernon benchmarks

Benchmarks are opt-in measurements, not correctness tests. They are never
registered with CTest and machine-local latency baselines are not repository
gates.

Configure the benchmark tooling explicitly:

```sh
cmake -S . -B benchmark-build -DVERNON_BUILD_BENCHMARKS=ON
cmake --build benchmark-build --target VernonBenchmarkFixtures
PYTHONPATH=python .venv/bin/python benchmarks/run.py --build benchmark-build --layer binding
```

`fixtures.json` is the workload catalog. `result.schema.json` defines the
single output format consumed by reports and future comparison tooling.
`environment.surface` distinguishes direct Python DSL calls from cooked C++
Runtime calls. Both surfaces are generated from `workloads.py`; the C++ surface
never has a separately authored test Program. A fixture becomes runnable only
after both drivers are implemented. Catalog validation and listing include
planned fixtures without presenting them as measurements.

Binding, Program, backend, and graphics fixtures are runnable on both
surfaces. Autodiff and application fixtures remain planned and are
intentionally absent from the catalog until both drivers are implemented. The
zero-node lifecycle fixture is intentionally deferred until the DSL has a
meaningful real workload for that boundary.

## Layers and boundaries

- `binding`: unchanged bind, changed bind, dynamic shape, and control updates.
  `call_ns` measures the complete public call after load/resolve and warmup.
  The Python surface includes DSL dispatch and the C++ surface includes
  begin/bind/execute/commit/destroy on a prepared instance. Validation and
  telemetry reads are outside the interval.
- `program`: square and 1/8/64-node chains. The same node operation and tensor
  extent are used to separate fixed invocation cost from per-node cost.
  `call_ns` is the synchronized public call on both surfaces. Load, compilation,
  cooking, resolve, and warmup are excluded.
- `backend`: empty kernel, elementwise, and reduction on every available
  backend. `call_ns` is the synchronized public-call latency; it does not claim
  to isolate device execution from host orchestration.
- `autodiff`: forward, retained-pullback creation, replay, and backward for
  static and dynamic tape workloads.
- `graphics`: minimal draw, multi-draw, and attachment/state-change workloads;
  `call_ns` is the synchronized public-call latency. Isolated command
  recording, submission, and device timestamps require backend timestamp
  support and are not inferred from this measurement.
- `application`: fluid simulation end-to-end frame/step throughput. It is a
  macrobenchmark and cannot establish a lifecycle allocation invariant.

Each driver must warm up before sampling, validate its output outside the
measured interval, and emit one JSON result matching `result.schema.json`.
The runner expands every runnable fixture over the canonical seven-backend
matrix shared with the tests: CPU, CUDA, Vulkan, DirectX 12, Metal, OpenGL, and
OpenGL ES. Unbuilt targets, unavailable devices/contexts, and unsupported
declared capabilities produce an explicit `skipped` result. Execution,
synchronization, or validation failures are errors.

Allocation and lock requirements remain deterministic tests or source-policy
checks under `source/tests`; they are not inferred from noisy latency
thresholds.
