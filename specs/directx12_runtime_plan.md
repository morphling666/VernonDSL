# DX12 Runtime Backend Plan

## Goal

Upgrade the existing `directx` target from cook-only HLSL source to Shader
Model 6 DXIL, then add a complete Windows D3D12 Runtime backend for compute
and graphics.

DXC is a pinned Cook/build dependency distributed with compiler tooling.
Runtime-only deployments load pre-cooked DXIL and do not require users to
install DXC, `dxcompiler.dll`, or `dxil.dll`.

This intentionally changes the current `directx` artifact contract. Existing
HLSL DirectX bundles are not runtime-compatible and must be re-cooked.

## 1. DXC and DXIL Cooking

- Add a pinned official DXC dependency and install its compiler/validator
  components beside native compiler tooling. Do not link DXC into
  `VernonRuntime`.
- Keep SPIR-V to HLSL as an intermediate compiler step, then invoke DXC with
  stage-specific `vs_6_0`, `ps_6_0`, or `cs_6_0` profiles.
- Emit deterministic `.dxil` binary artifacts and propagate DXC diagnostics
  through the C API, Python API, and CLI.
- Change the default DirectX Shader Model from 5.0 to 6.0 and reject settings
  below 6.0.
- Update artifact format/extension handling, reflection, compiler surface
  parity, cooker tests, wheel packaging, and install rules.

Primary files:

- `source/lib/compiler/compiler_spirv_cross.cpp`
- new `source/lib/compiler/compiler_dxc.{h,cpp}`
- `source/lib/compiler/compiler_dispatch.cpp`
- `source/lib/compiler/compiler_artifacts.cpp`
- `source/lib/runtime/VernonRuntimeTarget.cmake`
- `python/vernon_dsl/bundle/types.py`
- `python/vernon_dsl/_shader_assets/cooking.py`

## 2. DirectX Runtime Requirements

- Generate hash-covered DirectX requirements containing:
  - D3D API 12
  - minimum Feature Level
  - minimum Shader Model
  - root-signature version
  - compute workgroup dimensions when applicable
  - sorted compiler-reflected required features
- Requirements participate in the manifest content hash but not stage artifact
  identity.
- Extend the strict target-discriminated Runtime parser for DirectX and DXIL.
- Reject legacy HLSL DirectX bundles with an explicit unsupported-contract
  diagnostic.

Primary files:

- `python/vernon_dsl/bundle/requirements.py`
- `source/lib/runtime/pipeline_manifest.{h,cpp}`
- `source/lib/runtime/VernonRuntime.cpp`
- `source/lib/runtime/runtime_dispatch.{h,cpp}`

## 3. D3D12 Device and Resource Layer

- Add `VERNON_RUNTIME_DIRECTX12` to the stable Runtime API.
- Add a Windows-only `VERNON_ENABLE_DIRECTX12_RUNTIME` build option.
- Implement adapter enumeration, device creation, command queue/list,
  allocators, fences, and deterministic teardown.
- Production defaults to a hardware adapter. Tests use an internal hook to
  force WARP without changing the public API.
- Probe and retain:
  - supported Feature Level
  - highest recognized Shader Model, retrying lower versions after
    `E_INVALIDARG`
  - root-signature version
  - resource-binding tier
  - compute thread-group limits
- Fail before artifact loading when requirements exceed actual capabilities,
  reporting both required and actual values.
- Implement default-heap buffers/textures, upload/readback staging,
  SRV/UAV/CBV/RTV/sampler descriptors, state transitions, fence synchronization,
  and device-removed diagnostics.

Primary files:

- `source/include/VernonRuntime.h`
- new `source/lib/runtime/backend_directx12.{h,cpp}`
- `source/lib/runtime/runtime_state.h`
- `source/lib/runtime/runtime_dispatch.{h,cpp}`
- `source/lib/runtime/VernonRuntimeTarget.cmake`

## 4. Compute Execution

- Build stable root signatures from manifest reflection and parameter uses.
- Place resources in CBV/SRV/UAV and sampler descriptor tables.
- Materialize scalar and matrix values through upload constant buffers.
- Create and cache compute PSOs from DXIL.
- Bind descriptor heaps/root parameters and dispatch using reflected
  workgroup dimensions.
- Reuse existing TensorView validation, copy, synchronization, and error
  contracts.

## 5. Graphics Execution

- Add `graphics_directx12_encoder.{h,cpp}` and reuse the canonical graphics
  invocation planner.
- Map vertex inputs, textures, samplers, uniforms, render targets, draw
  topology, viewport, and scissor state to D3D12.
- Cache graphics PSOs by DXIL stages, input layout, topology, root signature,
  and render-target formats.
- Implement RTV creation, vertex/index binding, draw calls, implicit samplers,
  and render-target/readback state transitions.
- First release supports the current vertex-plus-fragment topology, compute,
  owned resources, and synchronous invocation.
- Swapchains, windows, and external D3D12 resource import are out of scope.

Primary files:

- new `source/lib/runtime/graphics_directx12_encoder.{h,cpp}`
- `source/lib/runtime/graphics_invocation_planner.{h,cpp}`
- `source/lib/runtime/runtime_dispatch.{h,cpp}`

## 6. Tests and Verification

- Compiler/Python tests:
  - DXC success and diagnostics
  - valid DXIL container output
  - stage-specific SM6 profiles
  - deterministic artifacts and deduplication
  - content hashes and DirectX requirements
  - explicit rejection of legacy HLSL contracts
- Device-independent C++ tests:
  - requirements parsing
  - capability comparisons
  - descriptor/root-signature planning
  - format and input-layout mappings
- Windows WARP integration tests:
  - compute dispatch and readback
  - triangle rendering
  - sampled textures and samplers
  - render targets
  - buffer/texture upload and readback
  - PSO cache reuse
  - artificially raised Feature Level and Shader Model rejection
- Re-cook `scale_asset`, triangle/sampled fixtures, and cube-map fixtures.
- Run the complete Windows Release build, CTest suite, Python tests, and
  formatting/lint checks.

## 7. Documentation

Update:

- `README.md`
- `specs/compiler/design.md`
- `specs/runtime/design.md`

Document the breaking `directx` artifact change, DXC's Cook-only dependency,
end-user system/driver requirements, capability validation, supported feature
surface, and current exclusions.

## Implementation Order

1. Integrate pinned DXC and upgrade `directx` cooking to SM6 DXIL.
2. Generate and validate DirectX runtime requirements.
3. Implement D3D12 device probing, synchronization, and resources.
4. Implement compute root signatures, descriptors, PSOs, and dispatch.
5. Implement graphics encoding and PSO caching.
6. Add WARP/compiler tests, re-cook fixtures, run full verification, and
   update documentation.
