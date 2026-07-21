# PowerShell build notes

[`README.md`](README.md#build-and-test) is the canonical Windows build and test
guide. Run its `cmake -S ... -B ...` commands from the repository root; they do
not require changing into the build directory and avoid source-directory
ambiguity.

If configuration cannot find MLIR, verify the canonical install location:

```powershell
Test-Path "$PWD/llvm-project/install/lib/cmake/mlir/MLIRConfig.cmake"
```

If that returns `False`, build and install LLVM/MLIR with the commands in the
README before configuring VernonDSL. Runtime-only builds do not require
LLVM/MLIR.
