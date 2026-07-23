# Python DSL v2 to v3

- Replace `@compute` with `@kernel`.
- Import runtime/annotation resources from `vernon_dsl`: `vd.Tensor` and
  `vd.Texture` now provide the only public spellings. The duplicate
  `vernon_dsl.types` constructors are gone.
- Replace `Array[T, N]` with `Tensor[T, (N,)]` where value-Tensor semantics are
  intended. `Array` has no stable v3 replacement for resource storage.
- Replace `ShaderAssetError` with `PipelineCompileError`.
- Use native `Compiler.compile_program_result`; `compile` and `compile_program`
  compatibility methods were removed.
- Python `int` and `float` annotations/casts mean `i32` and `f32`.
- Remove redundant casts around literals in floating expressions. Integer
  true division now produces `f32`, and safe floating widening is implicit.
  Floating narrowing and floating-to-integer conversion still require an
  explicit cast.
- Helpers may omit parameter and result annotations. Each concrete argument
  signature creates a deterministic specialized symbol and contributes to
  semantic cache identity.
- Replace `and`/`or` in device code with structured control flow. They now fail
  early rather than being eagerly evaluated. Nested return, dynamic `range`,
  conditional expressions, and chained comparisons also fail consistently.
