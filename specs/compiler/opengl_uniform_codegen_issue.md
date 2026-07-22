# OpenGL single-value uniform code generation issue

## Summary

VernonDSL currently emits invalid and unnecessarily wrapped GLSL for unbound
graphics uniforms when cross-compiling SPIR-V to OpenGL.

The CubeMap vertex stage declares three ordinary matrix uniforms:

```python
projection: Annotated[vd.mat4[vd.f32], vd.uniform()]
view: Annotated[vd.mat4[vd.f32], vd.uniform()]
model: Annotated[vd.mat4[vd.f32], vd.uniform()]
```

The generated OpenGL 3.3 artifact contains:

```glsl
struct _6 { mat4 _m0; };
uniform _6 projection;
struct _6 { mat4 _m0; };
uniform _6 view;
struct _6 { mat4 _m0; };
uniform _6 model;
```

All three declarations reuse the same struct type name. Redefining `_6` in the
same GLSL scope is invalid. Even if unique names were emitted, these one-member
wrapper structs are not the intended OpenGL interface.

## Expected output

```glsl
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
```

Shader expressions should reference `projection`, `view`, and `model` directly
rather than `projection._m0`, `view._m0`, and `model._m0`.

## Reproduction

Source:

```text
C:/Users/12825/backup/Vernon/frontend/python/engine/render/shader/cube_map.py:cube_map_asset
```

Cook command from the VernonDSL checkout:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m vernon_dsl.shader_asset_cli `
  "C:/Users/12825/backup/Vernon/frontend/python/engine/render/shader/cube_map.py:cube_map_asset" `
  --target opengl `
  --output "C:/Users/12825/backup/Vernon/assets/builtin/shaders/cube_map"
```

Affected generated artifact:

```text
C:/Users/12825/backup/Vernon/assets/builtin/shaders/cube_map/artifacts/c828c8c2a982a4f1a08b434de5770113c9f44b35d1fd9ddc9af3034b52311c21.vert.glsl
```

## Suspected cause

`createInterfaceVariable` in
`source/lib/Dialect/Vernon/Transforms/VernonToSpirv.cpp` wraps every
`spirv::StorageClass::Uniform` value in a one-member `Block` struct. Separate
matrix arguments therefore create structurally identical anonymous SPIR-V
struct types. SPIRV-Cross assigns the same fallback GLSL name (`_6`) to those
types and emits each definition independently.

The pipeline planner currently compensates for the wrapper by recording
OpenGL uniform names as `<source_name>._m0` in
`python/vernon_dsl/pipeline_compile.py`. If OpenGL uniforms are flattened, this
mapping must become the original source name.

## Constraints

- Vulkan behavior must not regress. Actual Vulkan compilation now aggregates
  unbound uniforms into one aligned push-constant block per graphics stage.
- OpenGL, OpenGL ES, and Metal cross-compilation intentionally use the
  non-aggregated path so source-level uniforms can remain independently named.
- Do not solve this in Vernon with generated-text regex rewriting. The compiler
  and reflection/planner output must agree on the final interface.
- Explicitly descriptor-bound uniform blocks must remain blocks.

## Required tests

1. Compile a vertex stage with at least three `mat4` uniforms to OpenGL 3.3.
2. Assert the GLSL declares each as a plain `uniform mat4` using its source
   name.
3. Assert no duplicate struct declarations and no `._m0` references exist.
4. Validate or compile the generated GLSL with a real GLSL compiler/context.
5. Assert pipeline reflection uses `projection`, `view`, and `model` as the
   OpenGL runtime lookup names.
6. Keep the Vulkan packed-block offset/stride regression and CubeMap runtime
   test passing.

## Current verification state

- The CubeMap Vulkan cook/upload/draw/MRT/readback test passes.
- The full VernonDSL CTest suite passed 23/23 before this issue was recorded.
- The generated OpenGL fragment shader is structurally correct; the observed
  failure is in the vertex uniform declarations.
- `glslangValidator` was not available in the current shell, so the generated
  OpenGL file has not yet been validated by that tool.
