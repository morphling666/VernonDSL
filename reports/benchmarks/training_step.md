# Training step: host time against layer count

The phase 2 gate of `specs/runtime/architecture_v2.md` section 16 asks for a
training step whose CPU time does not grow with its layer count. This report
measures how far the linked VJP is from it. A step is one call of
`vd.ad.vjp(Net(), wrt=weights)` and one call of its pullback. `Net` chains
`layers` elementwise multiplications, each by its own 64-element weight, and
the pullback returns one gradient per weight. Both calls take the linked
path: two schedules, the residuals in one native residual block, and on CUDA
one graph launch per schedule.

A step runs in one of two modes:

- **Unbound.** Every weight is passed on every call, and the pullback
  returns fresh gradients.
- **Bound.** The step is `step.bind(w0=..., ...)`, which binds the weights
  once (language contract section 8). The pullback writes the gradients
  into Storages that the bound step keeps, so a call passes only its batch.

## Provenance

- Source revision: VernonDSL `6fcb25d` with uncommitted work (`-dirty`);
  compiler contract 1, Program version 2, release 0.1.2.
- Command: `python examples/training_step/run.py --arch cpu --result-json reports/benchmarks/training_step_cpu.json`,
  and the same with `--arch cuda`, each also with `--bound` and
  `training_step_bound_<arch>.json`, from the repository root with
  `PYTHONPATH=python`.
- Platform: Windows 11 (10.0.26300), AMD Ryzen 9 9950X, NVIDIA GeForce RTX
  4090, driver 610.88 (WDDM); Python 3.11.9, numpy 1.26.4.
- Method:
  - Three warm-up steps, then the mean of 300 steps back to back.
  - The forward and backward columns are the time spent inside the two
    native schedule calls. Python is the rest of the step.
  - The idle step is the median of 200 single steps, each started after the
    device has finished the previous one. On CUDA, back-to-back steps wait
    for the four-slot record ring, which the GPU's kernels drain. The idle
    step therefore measures the host, and the back-to-back step measures
    the device.

## Results (µs per step)

Unbound, CPU:

| layers | step | native forward | native backward | Python | idle step |
|---:|---:|---:|---:|---:|---:|
| 1 | 20.7 | 1.8 | 4.4 | 14.5 | 21.3 |
| 8 | 45.9 | 6.0 | 9.6 | 30.3 | 47.3 |
| 64 | 242.5 | 37.1 | 49.3 | 156.1 | 244.3 |
| 256 | 959.4 | 140.2 | 189.4 | 629.9 | 988.9 |

Unbound, CUDA:

| layers | step | native forward | native backward | Python | idle step |
|---:|---:|---:|---:|---:|---:|
| 1 | 94.5 | 10.5 | 8.8 | 75.3 | 41.3 |
| 8 | 147.3 | 12.4 | 16.5 | 118.3 | 77.2 |
| 64 | 508.4 | 43.8 | 61.8 | 402.8 | 352.5 |
| 256 | 2368.2 | 153.3 | 262.5 | 1952.4 | 1461.3 |

Bound, CPU:

| layers | step | native forward | native backward | Python | idle step |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.3 | 1.7 | 4.1 | 9.5 | 15.0 |
| 8 | 23.8 | 5.1 | 8.5 | 10.2 | 23.8 |
| 64 | 87.1 | 29.5 | 40.9 | 16.7 | 86.5 |
| 256 | 300.8 | 114.6 | 148.8 | 37.4 | 299.0 |

Bound, CUDA:

| layers | step | native forward | native backward | Python | idle step |
|---:|---:|---:|---:|---:|---:|
| 1 | 89.7 | 8.8 | 7.8 | 73.1 | 28.5 |
| 8 | 120.7 | 12.6 | 12.0 | 96.1 | 36.6 |
| 64 | 358.7 | 32.0 | 48.3 | 278.4 | 103.0 |
| 256 | 1832.3 | 108.1 | 172.4 | 1551.8 | 316.7 |

The CPU rows varied by up to 15% between runs of one build in the same
session: the bound 256-layer step measured 258 µs an hour before these
runs, which repeated at 299–309 µs, and a build with the earlier argument
copies measured the same. Compare CPU rows within one run.

On CUDA, a back-to-back step's Python column is mostly waiting. Dropping a
pullback frees its residual block once the block's backward call has
finished on the device, which a back-to-back step reaches before the record
ring fills. In the previous measurement that wait was in the native
backward column.

For comparison:

- A 64-layer CUDA step took 1916 µs when the VJP ran as a Program, at the
  start of phase 2's VJP work.
- Before the compiler stopped copying gradients at the boundary (below),
  the unbound 256-layer steps took 1192.5 µs on the CPU and 2742.0 µs on
  CUDA.

At 256 layers, before the VJP took Program arguments, a temporary timer
inside the native bound backward call (769 slots) split its host time as
follows:

| part | µs |
|---|---:|
| validating and binding the arguments, about 0.04 per bound buffer | 28–31 |
| write-disjointness and bound-version checks | 10 |
| the schedule call: record copy, node comparison and `cuGraphLaunch` of about 513 nodes | 67–98 |
| publishing the 256 gradient writes | 11 |

## After step 5

The same commands after section 16.3 step 5 of `architecture_v2.md`, at
`f840106` with uncommitted work, with `_step5` added to each result file's
name. µs per step, step / idle step:

| layers | unbound CPU | bound CPU | unbound CUDA | bound CUDA |
|---:|---:|---:|---:|---:|
| 1 | 19.5 / 20.1 | 14.9 / 14.6 | 99.5 / 38.2 | 91.5 / 28.1 |
| 8 | 42.9 / 44.5 | 22.0 / 21.6 | 150.1 / 76.8 | 120.7 / 36.5 |
| 64 | 226.5 / 228.5 | 75.1 / 75.2 | 512.4 / 359.4 | 364.0 / 105.0 |
| 256 | 948.6 / 921.1 | 253.7 / 252.2 | 2660.9 / 1471.9 | 2029.7 / 339.0 |

The CPU rows are at or under the tables above. The CUDA idle steps, which
measure the host, are within 2% unbound and 7% bound. The back-to-back
256-layer CUDA steps are 11–12% higher, almost all of it in the Python
column, which is waiting for the device.

## Reading

- **Python is nearly flat in a bound step.** It grows from 10 µs to 37 µs
  over 256 layers, about 0.1 µs per layer, most of it registering the
  bound inputs a pullback preserves on copy-on-write. An unbound step's Python grows
  by about 3 µs per layer, because each layer passes its weight, gets a
  fresh gradient and has its version recorded.
- **The native host time still grows.** The idle bound CUDA step grows by
  about 1.1 µs per layer, up from 0.9 before the VJP took Program
  arguments. A CUDA buffer is a HAL allocation whose reference carries its
  device address, so binding one looks nothing up. A forward call's
  captured weights are checked again when its backward call reads them.
  The other per-layer costs are these:
  - The driver's `cuGraphLaunch` costs about 0.09 µs per node on WDDM, and
    each layer adds a forward node and a backward node. `cuGraphUpload`
    after node updates did not reduce it.
  - Each bound buffer is revalidated on every call, about 0.04 µs each.
    The contract keeps this check, so a host write to a bound weight
    between steps is seen.
  - Each gradient write is published to its owner.
- **On the CPU the native columns are the kernels.** On the CPU arch the
  schedule runs the 2 × `layers` kernels itself, so its native time grows
  with the work rather than with overhead.
- **The boundary copies are gone.** A public gradient used to be copied
  out of the kernel VJP's zeroed destination, one copy node per layer. The
  compiler now returns that destination as the gradient whenever it is
  only ever written. This removed about a third of the backward nodes and
  most of the backward kernel time.
- **What remains for a flat step.** A step whose host time does not grow at
  all needs fewer nodes per layer. That is compile-time fusion of the
  layers' kernels, which a runtime link cannot do (section 7). Everything
  else a bound step pays per layer is the per-buffer validation and
  publication that the contract requires.
