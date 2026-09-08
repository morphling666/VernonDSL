# Cost-aware CPU smoke autodiff benchmark

CPU, warmed forward/backward smoke runs. Pressure iterations and measured steps are recorded per row; tape context limit=268435456 bytes.

Frozen baseline: 469.8 logical bytes per active cell; compiler Tape hints per lane={'advect': 208, 'initialize': 88, 'jacobi': 88, 'project': 88, 'transport': 160}.

| Grid | Pressure | Steps | Logical residual | Logical/cell | Resident | Allocated | Retained allocation | Temporary Tape peak | Checkpoint | Peak managed | Resident/logical | RSS | RSS delta | Stable RSS | Forward | Backward | Recompute | Python callbacks |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 1 | 0 | 0.000 | 0 | 0 | 94588 | 436336 | 0 | 530924 | 0.000 | 121798656 | 0 | False | 0.001502s | 0.003217s | 1.306 | 9 |
| 64 | 1 | 1 | 0 | 0.000 | 0 | 0 | 377212 | 436336 | 0 | 813548 | 0.000 | 122617856 | 98304 | False | 0.002626s | 0.011420s | 1.306 | 9 |
| 128 | 1 | 1 | 0 | 0.000 | 0 | 0 | 1507708 | 436336 | 0 | 2501044 | 0.000 | 124092416 | 0 | False | 0.006680s | 0.044232s | 1.306 | 9 |
| 256 | 1 | 1 | 0 | 0.000 | 0 | 0 | 6029692 | 436336 | 0 | 9775540 | 0.000 | 137428992 | 0 | False | 0.024382s | 0.180191s | 1.306 | 9 |
| 512 | 1 | 1 | 0 | 0.000 | 0 | 0 | 24117628 | 436336 | 0 | 38873524 | 0.000 | 190136320 | 0 | False | 0.091892s | 0.807533s | 1.306 | 9 |
| 1024 | 1 | 1 | 0 | 0.000 | 0 | 0 | 96469372 | 436336 | 0 | 155265460 | 0.000 | 435994624 | 0 | False | 0.364605s | 5.183884s | 1.306 | 9 |

Logical residual bytes are payload bytes retained by reusable pullbacks. A no-Tape or bounded-replay pullback reports zero retained Tape bytes; checkpoint, retained primal-version, transaction, and gradient staging memory remain visible in their separate counters. RSS is sampled from the worker process after backward; each grid runs in a fresh process.

The JSON artifact also records maxima across all measured steps and RSS growth after the first measured step, plus every per-step RSS sample. Stable RSS means the final max(5, steps/5) samples fit within max(1 MiB, 1% of that window's high-water mark), so pressure runs distinguish stable selected residual/checkpoint storage from unconditional per-step Tape retention.

Recomputation factor is 1 + compiler-estimated rematerialization cost / active operation count, plus any graph-checkpoint replay factor.

CPU bounded replay uses complete-workgroup segments. A pure static balanced/min-runtime plan may retain whole-dispatch Tape only when its exact construction allocation fits the hard context budget.
