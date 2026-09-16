# VernonDSL reports

This directory contains measurements and historical baselines. Reports are
evidence, not normative contracts. A report is reproducible only when it
contains the provenance listed below; older files without that information are
historical observations rather than reproducible release baselines.

- `autodiff/`: autodiff smoke, memory-pressure, and comparison results.
- `benchmarks/`: focused performance measurements.
- `baselines/`: machine-specific regression baselines.

Each new report must record the source revision, generating command, workload,
policy, platform, device, driver/API, compiler contract, Program version, and
measurement method.
