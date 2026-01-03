#!/bin/bash
# Build script for Vernon DSL
# cmake ../llvm -DLLVM_ENABLE_PROJECTS="mlir;clang" -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -Thost=x64 -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_INSTALL_PREFIX="../install"
# cmake --build . --config Release --target install
# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: CMakeLists.txt not found. Please run this script from the project root."
    exit 1
fi

# Check if MLIR is installed
MLIR_DIR="llvm-project/install/lib/cmake/mlir"
if [ ! -d "$MLIR_DIR" ]; then
    echo "Error: MLIR not found at $MLIR_DIR"
    echo "Please build and install MLIR first:"
    echo "  cd llvm-project"
    echo "  mkdir build"
    echo "  cd build"
    echo "  cmake ../llvm -DLLVM_ENABLE_PROJECTS=\"mlir\" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install"
    echo "  cmake --build . --target install"
    exit 1
fi

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
    echo "Created build directory"
fi

# Configure and build
cd build

echo "Configuring CMake..."
cmake .. -DMLIR_DIR="../$MLIR_DIR"

if [ $? -eq 0 ]; then
    echo "Building..."
    cmake --build . --target MLIRVernonDialect
    if [ $? -eq 0 ]; then
        echo "Build successful!"
    else
        echo "Build failed!"
        cd ..
        exit 1
    fi
else
    echo "CMake configuration failed!"
    cd ..
    exit 1
fi

cd ..
