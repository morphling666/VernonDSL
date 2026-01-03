# Build script for Vernon DSL

# Check if we're in the right directory
if (-not (Test-Path "CMakeLists.txt")) {
    Write-Host "Error: CMakeLists.txt not found. Please run this script from the project root."
    exit 1
}

# Check if MLIR is installed
$mlirDir = "llvm-project\install\lib\cmake\mlir"
if (-not (Test-Path $mlirDir)) {
    Write-Host "Error: MLIR not found at $mlirDir"
    Write-Host "Please build and install MLIR first:"
    Write-Host "  cd llvm-project"
    Write-Host "  mkdir build"
    Write-Host "  cd build"
    Write-Host "  cmake ..\llvm -DLLVM_ENABLE_PROJECTS=`"mlir`" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..\install"
    Write-Host "  cmake --build . --target install"
    exit 1
}

# Create build directory if it doesn't exist
if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build" | Out-Null
    Write-Host "Created build directory"
}

# Configure and build
Set-Location build

Write-Host "Configuring CMake..."
cmake .. -DMLIR_DIR="$PSScriptRoot\llvm-project\install\lib\cmake\mlir"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Building..."
    cmake --build . --target MLIRVernonDialect
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Build successful!"
    } else {
        Write-Host "Build failed!"
        Set-Location ..
        exit 1
    }
} else {
    Write-Host "CMake configuration failed!"
    Set-Location ..
    exit 1
}

Set-Location ..
