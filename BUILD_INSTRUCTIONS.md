# Building the Vernon Dialect (Standalone)

The Vernon dialect is built as a standalone project that uses MLIR as a third-party dependency. The `llvm-project` directory remains untouched.

## Prerequisites

1. **Build and install MLIR/LLVM first:**
   ```bash
   cd llvm-project
   mkdir build
   cd build
   cmake ../llvm -DLLVM_ENABLE_PROJECTS="mlir" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install
   cmake --build . --target install
   ```

   This will install MLIR to `llvm-project/install/`

2. **Set MLIR_DIR when configuring CMake:**
   ```bash
   cmake -DMLIR_DIR=llvm-project/install/lib/cmake/mlir ..
   ```

   Or set it as an environment variable:
   ```bash
   # On Windows (PowerShell):
   $env:MLIR_DIR = "$PWD\llvm-project\install\lib\cmake\mlir"
   
   # On Linux/Mac:
   export MLIR_DIR="$PWD/llvm-project/install/lib/cmake/mlir"
   ```

## Build Steps

### Windows (PowerShell)

1. **Create build directory (at project root):**
   ```powershell
   mkdir build
   cd build
   ```

2. **Configure CMake:**
   ```powershell
   cmake .. -DMLIR_DIR="$PSScriptRoot\llvm-project\install\lib\cmake\mlir"
   ```
   
   Or use relative path:
   ```powershell
   cmake .. -DMLIR_DIR="..\llvm-project\install\lib\cmake\mlir"
   ```

3. **Build:**
   ```powershell
   cmake --build . --target MLIRVernonDialect
   ```

   Or build everything:
   ```powershell
   cmake --build .
   ```

**Alternative: Use the provided build script:**
```powershell
.\build.ps1
```

### Linux/Mac (Bash)

1. **Create build directory (at project root):**
   ```bash
   mkdir build
   cd build
   ```

2. **Configure CMake:**
   ```bash
   cmake .. -DMLIR_DIR=../llvm-project/install/lib/cmake/mlir
   ```

3. **Build:**
   ```bash
   cmake --build . --target MLIRVernonDialect
   ```

## Project Structure

```
VernonDSL/
├── CMakeLists.txt              # Root CMakeLists (finds MLIR as third-party)
├── source/
│   ├── CMakeLists.txt
│   ├── include/mlir/Dialect/Vernon/IR/
│   │   ├── VernonDialect.td
│   │   ├── VernonTypes.td
│   │   ├── VernonOps.td
│   │   ├── Vernon.h
│   │   └── CMakeLists.txt      # TableGen configuration
│   └── lib/Dialect/Vernon/IR/
│       ├── VernonDialect.cpp
│       ├── VernonTypes.cpp
│       ├── VernonOps.cpp
│       └── CMakeLists.txt      # Library build configuration
└── llvm-project/               # MLIR as third-party (untouched)
    └── install/                # MLIR installation (after building)
        └── lib/cmake/mlir/     # MLIR CMake config files
```

## Using the Dialect

After building, link against `MLIRVernonDialect` in your CMake project:

```cmake
target_link_libraries(your_target MLIRVernonDialect)
```

And use in your code:
```cpp
#include "mlir/Dialect/Vernon/IR/Vernon.h"

// Register the dialect
ctx->loadDialect<mlir::vernon::VernonDialect>();
```

## Notes

- The `llvm-project` directory is treated as a third-party dependency and is not modified
- MLIR must be built and installed before building the Vernon dialect
- The build follows the MLIR standalone dialect pattern (see `llvm-project/mlir/examples/standalone/`)
