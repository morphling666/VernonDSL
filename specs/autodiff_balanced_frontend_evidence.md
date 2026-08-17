# Balanced frontend whole-dispatch evidence

The public frontend declaration uses `planning_policy="balanced"`. The compiler
serializes `whole_dispatch_retention_permitted: true`; the Runtime consumes that
decision without inferring it from workload size or policy.

The isolated Runtime test
`RuntimeStructuredScalarAutodiff.BalancedRetainsOnlyExactlyAdmittedWholeDispatchTape`
covers both 512 and 1024 grids under two explicit test budgets:

- An exact whole-dispatch budget retains Tape, reports a positive retained
  allocation, and uses no temporary replay Tape.
- A single-workgroup budget rejects whole-dispatch Tape retention, reports only
  the retained primal allocation (12,324 / 24,612 bytes), and bounds temporary
  replay Tape by one workgroup.

The Runtime test reconstructs and compares the machine-readable artifact, so
`autodiff_balanced_frontend_evidence.json` cannot drift from measured behavior.
