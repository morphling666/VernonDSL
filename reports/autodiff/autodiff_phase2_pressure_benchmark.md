# Cost-aware CPU smoke autodiff benchmark

CPU, warmed forward/backward smoke runs. Pressure iterations and measured steps are recorded per row; tape context limit=268435456 bytes.

Frozen baseline: 469.8 logical bytes per active cell; compiler Tape hints per lane={'advect': 208, 'initialize': 88, 'jacobi': 88, 'project': 88, 'transport': 160}.

| Grid | Pressure | Steps | Logical residual | Logical/cell | Resident | Allocated | Retained allocation | Temporary Tape peak | Checkpoint | Peak managed | Resident/logical | RSS | RSS delta | Stable RSS | Forward | Backward | Recompute | Python callbacks |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 10 | 100 | 0 | 0.000 | 0 | 0 | 205684 | 436336 | 0 | 642020 | 0.000 | 122552320 | 671744 | True | 0.266859s | 0.712309s | 1.397 | 27 |

Logical residual bytes are payload bytes retained by reusable pullbacks. A no-Tape or bounded-replay pullback reports zero retained Tape bytes; checkpoint, retained primal-version, transaction, and gradient staging memory remain visible in their separate counters. RSS is sampled from the worker process after backward; each grid runs in a fresh process.

The JSON artifact also records maxima across all measured steps and RSS growth after the first measured step, plus every per-step RSS sample. Stable RSS means the final max(5, steps/5) samples fit within max(1 MiB, 1% of that window's high-water mark), so pressure runs distinguish stable selected residual/checkpoint storage from unconditional per-step Tape retention.

Recomputation factor is 1 + compiler-estimated rematerialization cost / active operation count, plus any graph-checkpoint replay factor.

CPU bounded replay uses complete-workgroup segments. A pure static balanced/min-runtime plan may retain whole-dispatch Tape only when its exact construction allocation fits the hard context budget.
