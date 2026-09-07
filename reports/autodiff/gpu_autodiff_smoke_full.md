# Cost-aware smoke autodiff benchmark

Warmed forward/backward smoke runs. Pressure iterations and measured steps are recorded per row; tape context limit=268435456 bytes.

Frozen baseline: 469.8 logical bytes per active cell; compiler Tape hints per lane={'advect': 208, 'initialize': 88, 'jacobi': 88, 'project': 88, 'transport': 160}.

| Backend | Grid | Pressure | Steps | Logical residual | Logical/cell | Resident | Allocated | Retained allocation | Temporary Tape peak | Checkpoint | Peak managed | Resident/logical | RSS | RSS delta | Stable RSS | Forward | Backward | Recompute | Python callbacks |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cpu | 32 | 4 | 3 | 0 | 0.000 | 0 | 0 | 131620 | 436336 | 0 | 567956 | 0.000 | 123486208 | 81920 | False | 0.005917s | 0.014027s | 1.350 | 15 |
| cpu | 64 | 4 | 3 | 0 | 0.000 | 0 | 0 | 524836 | 436336 | 0 | 961172 | 0.000 | 124960768 | 294912 | False | 0.010708s | 0.050706s | 1.350 | 15 |
| cpu | 128 | 4 | 3 | 0 | 0.000 | 0 | 0 | 2097700 | 436336 | 0 | 3091036 | 0.000 | 126582784 | 622592 | False | 0.031441s | 0.200243s | 1.350 | 15 |
| cpu | 256 | 4 | 3 | 0 | 0.000 | 0 | 0 | 8389156 | 436336 | 0 | 12135004 | 0.000 | 145063936 | 2752512 | False | 0.115754s | 0.823736s | 1.350 | 15 |
| cpu | 512 | 4 | 3 | 0 | 0.000 | 0 | 0 | 33554980 | 436336 | 0 | 48310876 | 0.000 | 225689600 | 21397504 | False | 0.444974s | 3.644568s | 1.350 | 15 |
| cpu | 1024 | 4 | 3 | 0 | 0.000 | 0 | 0 | 134218276 | 436336 | 0 | 193014364 | 0.000 | 583090176 | 92438528 | False | 1.794419s | 22.475390s | 1.350 | 15 |
| metal | 32 | 4 | 3 | 0 | 0.000 | 0 | 0 | 131148 | 278860 | 0 | 410008 | 0.000 | 127025152 | 1572864 | False | 0.055357s | 0.063575s | 1.350 | 15 |
| metal | 64 | 4 | 3 | 0 | 0.000 | 0 | 0 | 524364 | 1115404 | 0 | 1639768 | 0.000 | 131350528 | 4472832 | False | 0.056402s | 0.069422s | 1.350 | 15 |
| metal | 128 | 4 | 3 | 0 | 0.000 | 0 | 0 | 2097228 | 4461580 | 0 | 6558808 | 0.000 | 147734528 | 16285696 | False | 0.057487s | 0.085260s | 1.350 | 15 |
| metal | 256 | 4 | 3 | 0 | 0.000 | 0 | 0 | 8388684 | 17846284 | 0 | 26234968 | 0.000 | 226836480 | 67600384 | False | 0.066585s | 0.149367s | 1.350 | 15 |
| metal | 512 | 4 | 3 | 0 | 0.000 | 0 | 0 | 33554508 | 71385100 | 0 | 104939608 | 0.000 | 544260096 | 274038784 | False | 0.117820s | 0.427232s | 1.350 | 15 |
| metal | 1024 | 4 | 3 | 0 | 0.000 | 0 | 0 | 134217804 | 134215212 | 0 | 268433016 | 0.000 | 1770930176 | 1063550976 | False | 0.332241s | 1.571678s | 1.350 | 15 |
| vulkan | 32 | 4 | 3 | 0 | 0.000 | 0 | 0 | 131148 | 278860 | 0 | 410008 | 0.000 | 131170304 | 81920 | False | 0.058189s | 0.065073s | 1.350 | 15 |
| vulkan | 64 | 4 | 3 | 0 | 0.000 | 0 | 0 | 524364 | 1115404 | 0 | 1639768 | 0.000 | 131334144 | 98304 | False | 0.058704s | 0.066099s | 1.350 | 15 |
| vulkan | 128 | 4 | 3 | 0 | 0.000 | 0 | 0 | 2097228 | 4461580 | 0 | 6558808 | 0.000 | 132939776 | 606208 | False | 0.060215s | 0.076443s | 1.350 | 15 |
| vulkan | 256 | 4 | 3 | 0 | 0.000 | 0 | 0 | 8388684 | 17846284 | 0 | 26234968 | 0.000 | 142573568 | 2670592 | False | 0.066706s | 0.114886s | 1.350 | 15 |
| vulkan | 512 | 4 | 3 | 0 | 0.000 | 0 | 0 | 33554508 | 71385100 | 0 | 104939608 | 0.000 | 196296704 | 19447808 | False | 0.102587s | 0.276252s | 1.350 | 15 |
| vulkan | 1024 | 4 | 3 | 0 | 0.000 | 0 | 0 | 134217804 | 134215212 | 0 | 268433016 | 0.000 | 399687680 | 71467008 | False | 0.275445s | 0.949179s | 1.350 | 15 |

Logical residual bytes are payload bytes retained by reusable pullbacks. A no-Tape or bounded-replay pullback reports zero retained Tape bytes; checkpoint, retained primal-version, transaction, and gradient staging memory remain visible in their separate counters. RSS is sampled from the worker process after backward; each grid runs in a fresh process.

The JSON artifact also records maxima across all measured steps and RSS growth after the first measured step, plus every per-step RSS sample. Stable RSS means the final max(5, steps/5) samples fit within max(1 MiB, 1% of that window's high-water mark), so pressure runs distinguish stable selected residual/checkpoint storage from unconditional per-step Tape retention.

Recomputation factor is 1 + compiler-estimated rematerialization cost / active operation count, plus any graph-checkpoint replay factor.

CPU bounded replay uses complete-workgroup segments. A pure static balanced/min-runtime plan may retain whole-dispatch Tape only when its exact construction allocation fits the hard context budget.
