# Native source tree

This directory contains the Vernon MLIR dialect, validation and lowering
passes, compiler C API and CLI, lightweight multibackend runtime, native Python
bindings, and native tests.

Public headers are under `include/`; implementations are under `lib/`; command
line tools are under `tools/`; and native tests are under `tests/`. The dialect
uses one Tensor type. Vector and matrix names belong to the source-language
API and lower according to use; they are not separate MLIR type families.

Build the source tree through the repository's top-level CMake project. See the
canonical [source build guide](../README.md#build-from-source) and
[test guide](../README.md#run-tests). Architecture and stable-contract
decisions are recorded in
[`specs/compiler/design.md`](../specs/compiler/design.md) and
[`specs/runtime/design.md`](../specs/runtime/design.md).
