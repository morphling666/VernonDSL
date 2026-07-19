param(
  [Parameter(Mandatory = $true)]
  [string]$ProjectRoot,
  [Parameter(Mandatory = $true)]
  [string]$BinaryDir,
  [Parameter(Mandatory = $true)]
  [string]$VernonOpt
)

$ErrorActionPreference = "Stop"
$inputPath = Join-Path $ProjectRoot "python\tests\smoke_shader.py"
$outputPath = Join-Path $BinaryDir "frontend-smoke.mlir"
$validatedPath = Join-Path $BinaryDir "frontend-smoke.validated.mlir"

Push-Location $ProjectRoot
try {
  & uv run --frozen vernon-compile-python $inputPath -o $outputPath
  if ($LASTEXITCODE -ne 0) {
    throw "Python frontend compilation failed"
  }

  & $VernonOpt --vernon-validate $outputPath -o $validatedPath
  if ($LASTEXITCODE -ne 0) {
    throw "Native Vernon MLIR validation failed"
  }
} finally {
  Pop-Location
}

Write-Host "Python frontend to native MLIR validation passed."
