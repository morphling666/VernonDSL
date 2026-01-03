# Building Vernon DSL on Windows (PowerShell)

## Quick Start

1. **Make sure you're in the project root directory** (where `CMakeLists.txt` is located)

2. **Create and enter the build directory:**
   ```powershell
   mkdir build -ErrorAction SilentlyContinue
   cd build
   ```

3. **Configure CMake (use backslashes for Windows paths):**
   ```powershell
   cmake .. -DMLIR_DIR="..\llvm-project\install\lib\cmake\mlir"
   ```

4. **Build:**
   ```powershell
   cmake --build . --target MLIRVernonDialect
   ```

## Alternative: Use Absolute Path

If relative paths don't work, use an absolute path:

```powershell
$mlirDir = Resolve-Path "..\llvm-project\install\lib\cmake\mlir"
cmake .. -DMLIR_DIR="$mlirDir"
```

## Common Issues

### Issue: "CMake Error: The source directory does not appear to contain CMakeLists.txt"

**Solution:** Make sure you're running cmake from the `build` directory, not from the project root:
```powershell
# Wrong - running from project root:
cd VernonDSL
cmake .. -DMLIR_DIR=...  # This looks for CMakeLists.txt in parent directory

# Correct - running from build directory:
cd VernonDSL
mkdir build
cd build
cmake .. -DMLIR_DIR="..\llvm-project\install\lib\cmake\mlir"  # This looks for CMakeLists.txt in parent (project root)
```

### Issue: Path not found

**Solution:** Verify MLIR is installed:
```powershell
Test-Path "..\llvm-project\install\lib\cmake\mlir"
```

If this returns `False`, you need to build and install MLIR first (see BUILD_INSTRUCTIONS.md).
