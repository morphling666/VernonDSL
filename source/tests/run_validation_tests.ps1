param(
  [string]$VernonOpt = (
    Join-Path $PSScriptRoot "..\..\build\source\tools\vernon_opt\Release\vernon-opt.exe"
  )
)

$ErrorActionPreference = "Stop"
$valid = Join-Path $PSScriptRoot "integration\validate-valid.mlir"
$invalid = Join-Path $PSScriptRoot "integration\validate-invalid.mlir"

& $VernonOpt --vernon-validate $valid | Out-Null
if ($LASTEXITCODE -ne 0) {
  throw "Expected the valid Vernon interface test to pass"
}

$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $VernonOpt
$startInfo.Arguments = "--vernon-validate `"$invalid`""
$startInfo.UseShellExecute = $false
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$process = [System.Diagnostics.Process]::Start($startInfo)
$diagnostics = $process.StandardError.ReadToEnd()
$process.WaitForExit()
if ($process.ExitCode -eq 0) {
  throw "Expected the invalid Vernon interface test to fail"
}

$expected = @(
  "cannot have both 'vernon.location' and 'vernon.builtin'",
  "uniform must provide both 'vernon.set' and 'vernon.binding', or neither",
  "'vernon.instance_divisor' is only valid on vertex inputs",
  "compute entry requires 'vernon.workgroup_size'",
  "but the vertex output has type"
)
foreach ($message in $expected) {
  if (-not $diagnostics.Contains($message)) {
    throw "Missing expected diagnostic: $message`n$diagnostics"
  }
}

Write-Host "Vernon validation positive and negative tests passed."
