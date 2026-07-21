param(
    [string]$BuildDirectory = "build",
    [string]$Configuration = "Release",
    [string]$MlirDirectory = "",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path (Join-Path $PSScriptRoot "CMakeLists.txt"))) {
    throw "CMakeLists.txt not found next to build.ps1."
}

if (-not $MlirDirectory) {
    $MlirDirectory = Join-Path $PSScriptRoot "llvm-project\install\lib\cmake\mlir"
}
if (-not (Test-Path (Join-Path $MlirDirectory "MLIRConfig.cmake"))) {
    throw "MLIRConfig.cmake not found under '$MlirDirectory'. Pass -MlirDirectory explicitly."
}

$buildPath = Join-Path $PSScriptRoot $BuildDirectory
$cmakeArguments = @(
    "-S", $PSScriptRoot,
    "-B", $buildPath,
    "-DMLIR_DIR=$MlirDirectory",
    "-DCMAKE_BUILD_TYPE=$Configuration"
)
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $cmakeArguments += "-DPython_EXECUTABLE=$venvPython"
}
cmake @cmakeArguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build $buildPath --config $Configuration --parallel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $SkipTests) {
    ctest --test-dir $buildPath -C $Configuration --output-on-failure
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
