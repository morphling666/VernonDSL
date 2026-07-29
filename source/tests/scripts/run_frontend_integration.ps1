param(
  [Parameter(Mandatory = $true)]
  [string]$ProjectRoot,
  [Parameter(Mandatory = $true)]
  [string]$BinaryDir,
  [Parameter(Mandatory = $true)]
  [string]$VernonOpt,
  [Parameter(Mandatory = $true)]
  [string]$VernonCompiler
)

$ErrorActionPreference = "Stop"
$inputPath = Join-Path $ProjectRoot "python\tests\smoke_shader.py"
$outputPath = Join-Path $BinaryDir "frontend-smoke.mlir"
$validatedPath = Join-Path $BinaryDir "frontend-smoke.validated.mlir"
$shaderPath = Join-Path $ProjectRoot "examples\shadowed_material.py"
$shaderMlirPath = Join-Path $BinaryDir "shadowed-material.mlir"
$vulkanOutput = Join-Path $BinaryDir "shadowed-material-vulkan"
$cpuOutput = Join-Path $BinaryDir "shadowed-material-cpu"
$runtimeShaderPath = Join-Path $ProjectRoot "examples\runtime_shader.py"
$runtimeMlirPath = Join-Path $BinaryDir "runtime-shader.mlir"
$openGlOutput = Join-Path $BinaryDir "runtime-shader-opengl"
$env:PYTHONPATH = Join-Path $ProjectRoot "python"
$versions = & python -c "from vernon_dsl._versions import *; print(f'{COMPILER_CONTRACT_VERSION},{PIPELINE_VERSION}')"
$compilerContractVersion, $pipelineVersion = $versions.Trim().Split(",")

Push-Location $ProjectRoot
try {
  & uv run --frozen python -m vernon_dsl.cli $inputPath -o $outputPath
  if ($LASTEXITCODE -ne 0) {
    throw "Python frontend compilation failed"
  }

  & $VernonOpt --vernon-validate $outputPath -o $validatedPath
  if ($LASTEXITCODE -ne 0) {
    throw "Native Vernon MLIR validation failed"
  }

  & uv run --frozen python -m vernon_dsl.cli $shaderPath -o $shaderMlirPath
  if ($LASTEXITCODE -ne 0) {
    throw "Multi-file shader frontend compilation failed"
  }

  Remove-Item -Recurse -Force $vulkanOutput -ErrorAction SilentlyContinue
  & $VernonCompiler --target vulkan $shaderMlirPath --output-dir $vulkanOutput
  if ($LASTEXITCODE -ne 0) {
    throw "Vulkan shader artifact compilation failed"
  }
  $spirvPath = Join-Path $vulkanOutput "module.spv"
  $spirvBytes = [System.IO.File]::ReadAllBytes($spirvPath)
  if ($spirvBytes.Length -lt 20 -or
      [BitConverter]::ToUInt32($spirvBytes, 0) -ne 0x07230203) {
    throw "Vulkan artifact is not a valid SPIR-V module"
  }
  $reflection = Get-Content (Join-Path $vulkanOutput "reflection.json") -Raw |
    ConvertFrom-Json
  if (-not ($reflection.dependencies.path -contains
            "examples/shader_lib/shadow.py")) {
    throw "Reflection does not contain transitive shader dependencies"
  }

  Remove-Item -Recurse -Force $cpuOutput -ErrorAction SilentlyContinue
  & $VernonCompiler --target cpu $shaderMlirPath --output-dir $cpuOutput
  $cpuArtifact = if (
    [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
      [System.Runtime.InteropServices.OSPlatform]::Windows)) {
    "module.obj"
  } else {
    "module.o"
  }
  if ($LASTEXITCODE -ne 0 -or
      -not (Test-Path (Join-Path $cpuOutput $cpuArtifact))) {
    throw "CPU reference shader compilation failed"
  }

  & uv run --frozen python -m vernon_dsl.cli $runtimeShaderPath -o $runtimeMlirPath
  if ($LASTEXITCODE -ne 0) {
    throw "Runtime shader frontend compilation failed"
  }
  Remove-Item -Recurse -Force $openGlOutput -ErrorAction SilentlyContinue
  & $VernonCompiler --target opengl $runtimeMlirPath --output-dir $openGlOutput `
    --glsl-version 330
  if ($LASTEXITCODE -ne 0) {
    throw "OpenGL runtime shader compilation failed"
  }
  $vertexSource = Get-Content (Join-Path $openGlOutput "runtime_vertex.vert.glsl") -Raw
  $fragmentSource = Get-Content (Join-Path $openGlOutput "runtime_fragment.frag.glsl") -Raw
  if (-not $vertexSource.StartsWith("#version 330") -or
      -not $fragmentSource.StartsWith("#version 330")) {
    throw "OpenGL artifacts must target Vernon's GLSL 3.3 runtime"
  }
  $openGlReflection = Get-Content (Join-Path $openGlOutput "reflection.json") -Raw |
    ConvertFrom-Json
  if ($openGlReflection.compiler_contract_version -ne [int]$compilerContractVersion -or
      $openGlReflection.pipeline_version -ne [int]$pipelineVersion -or
      $openGlReflection.target -ne "opengl" -or
      $openGlReflection.target_options.glsl_version -ne 330 -or
      $openGlReflection.artifacts.Count -ne 2 -or
      -not ($openGlReflection.artifacts.filename -contains "runtime_vertex.vert.glsl") -or
      -not ($openGlReflection.artifacts.filename -contains "runtime_fragment.frag.glsl")) {
    throw "OpenGL output is missing reflected runtime artifacts"
  }
} finally {
  Pop-Location
}

Write-Host "Multi-file Python shader produced Vulkan, CPU, OpenGL 3.3, and Vernon asset artifacts."
