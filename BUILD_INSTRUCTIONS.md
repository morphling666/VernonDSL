# Building VernonDSL

The canonical prerequisites, LLVM/MLIR installation, full compiler build,
tests, Python environment, and runtime-only build commands are maintained in
[`README.md`](README.md#build-and-test).

Use out-of-source builds from the repository root. A full compiler build needs
both the installed MLIR CMake package and the matching `llvm-project` source
checkout for TableGen includes. A runtime-only build sets
`VERNON_ENABLE_COMPILER=OFF` and does not require LLVM/MLIR.

This file intentionally does not duplicate setup commands; update the README
when the supported build procedure changes.
