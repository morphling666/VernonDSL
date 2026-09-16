# Cost-aware smoke autodiff benchmark

Warmed forward/backward smoke runs. Pressure iterations and measured steps are recorded per row; tape context limit=268435456 bytes.

Frozen baseline: 469.8 logical bytes per active cell; compiler Tape hints per lane={'advect': 208, 'initialize': 88, 'jacobi': 88, 'project': 88, 'transport': 160}.

| Backend | Grid | Pressure | Steps | Logical residual | Logical/cell | Resident | Allocated | Retained allocation | Temporary Tape peak | Checkpoint | Peak managed | Resident/logical | RSS | RSS delta | Stable RSS | Forward | Backward | Recompute | Python callbacks |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cpu | 256 | 4 | 3 | 0 | 0.000 | 0 | 0 | 8389156 | 436336 | 0 | 12135004 | 0.000 | 146374656 | 2916352 | False | 0.113542s | 0.808470s | 1.350 | 15 |
| cpu | 512 | 4 | 3 | 0 | 0.000 | 0 | 0 | 33554980 | 436336 | 0 | 48310876 | 0.000 | 227033088 | 22020096 | False | 0.446342s | 3.687969s | 1.350 | 15 |
| cpu | 1024 | 4 | 3 | 0 | 0.000 | 0 | 0 | 134218276 | 436336 | 0 | 193014364 | 0.000 | 583942144 | 92438528 | False | 1.809840s | 22.536969s | 1.350 | 15 |
| metal | 256 | 4 | 3 | 0 | 0.000 | 0 | 0 | 8388684 | 17846284 | 0 | 26234968 | 0.000 | 226639872 | 68173824 | False | 0.065624s | 0.150105s | 1.350 | 15 |
| metal | 512 | 4 | 3 | 0 | 0.000 | 0 | 0 | 33554508 | 71385100 | 0 | 104939608 | 0.000 | 543784960 | 274513920 | False | 0.117132s | 0.426833s | 1.350 | 15 |
| metal | 1024 | 4 | 3 | 0 | 0.000 | 0 | 0 | 134217804 | 134215212 | 0 | 268433016 | 0.000 | 1772879872 | 1063583744 | False | 0.332789s | 1.599534s | 1.350 | 15 |
| vulkan | 256 | 4 | 3 | 0 | 0.000 | 0 | 0 | 8388684 | 17846284 | 0 | 26234968 | 0.000 | 143261696 | 3063808 | False | 0.066071s | 0.114740s | 1.350 | 15 |
| vulkan | 512 | 4 | 3 | 0 | 0.000 | 0 | 0 | 33554508 | 71385100 | 0 | 104939608 | 0.000 | 195379200 | 16875520 | False | 0.102794s | 0.275441s | 1.350 | 15 |
| vulkan | 1024 | 4 | 3 | 0 | 0.000 | 0 | 0 | 134217804 | 134215212 | 0 | 268433016 | 0.000 | 401178624 | 71483392 | False | 0.275113s | 0.952273s | 1.350 | 15 |

Logical residual bytes are payload bytes retained by reusable pullbacks. A no-Tape or bounded-replay pullback reports zero retained Tape bytes; checkpoint, retained primal-version, transaction, and gradient staging memory remain visible in their separate counters. RSS is sampled from the worker process after backward; each grid runs in a fresh process.

The JSON artifact also records maxima across all measured steps and RSS growth after the first measured step, plus every per-step RSS sample. Stable RSS means the final max(5, steps/5) samples fit within max(1 MiB, 1% of that window's high-water mark), so pressure runs distinguish stable selected residual/checkpoint storage from unconditional per-step Tape retention.

Recomputation factor is 1 + compiler-estimated rematerialization cost / active operation count, plus any graph-checkpoint replay factor.

CPU bounded replay uses complete-workgroup segments. A pure static balanced/min-runtime plan may retain whole-dispatch Tape only when its exact construction allocation fits the hard context budget.
