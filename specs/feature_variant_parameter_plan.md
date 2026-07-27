# Typed Feature and Variant Parameter Plan

## Goals

- Extend boolean-only `vd.feature` into typed compile-time values without
  introducing textual macro expansion or runtime shader branches.
- Let `vd.When` explicitly remove a parameter or select different parameter
  types for different variants.
- Separate the stable logical host ABI from backend bindings removed by
  compile-time specialization or dead-code elimination.

## Feature values

`vd.feature("NAME")` remains compatible and means a Boolean feature whose
default is `False`. Typed values use a default from which the feature type is
inferred:

```python
SHADOW = vd.feature("SHADOW")
QUALITY = vd.feature("QUALITY", default=0)
BIAS = vd.feature("BIAS", default=0.001)
```

The first implementation supports `bool`, 32-bit integers, and finite `f32`.
It rejects implicit cross-type assignments, integer overflow, NaN, infinity,
and values that depend on runtime data.

Existing Boolean variant input remains valid:

```python
features={"SHADOW"}
```

It is shorthand for `{"SHADOW": True}`. Value features use canonical
assignments:

```python
features={"SHADOW": True, "QUALITY": 2}
```

Feature declarations record name, type, and default. Variant identities record
sorted typed assignments and include them in frontend specialization identity,
artifact hashes, PipelineAsset variants, and runtime variant selection.
Boolean-only asset manifests remain readable; newly written manifests use the
typed representation.

## Compile-time evaluation

One target-independent evaluator in the source module graph handles feature
values, captured constants, literals, parentheses, arithmetic, comparison,
and Boolean expressions. The same evaluator is used by:

- `if` and `elif` specialization;
- `vd.When` conditions;
- static type and shape expressions;
- helper specialization and cache identity.

Evaluation and control-flow pruning happen before type inference and lowering.
No feature value becomes a uniform, descriptor, specialization constant, or
runtime IR branch. Invalid operations such as division by zero, overflow,
mixed incompatible types, and non-deterministic expressions report the source
location.

## Conditional parameter types

The existing form remains supported:

```python
value: vd.When[SHADOW, ShadowData]
```

It is equivalent to one conditional case with no fallback. The extended form
uses ordered cases:

```python
value: vd.When[
    vd.Case[QUALITY >= 2, vd.Vector[vd.f32, 4]],
    vd.Case[SHADOW, vd.Texture["2d", vd.f32]],
    vd.Else[vd.Matrix[vd.f32, 4, 4]],
]
```

Cases use first-match semantics equivalent to `if` / `elif` / `else`.

- If one case matches, the parameter is replaced by that case's ordinary type.
- If no case matches and an `Else` exists, its type is selected.
- If no case matches and no `Else` exists, the parameter is removed.
- Only the selected type participates in type, binding, and backend capability
  validation. Unselected branches have no diagnostic obligation.

Different cases may select Value, Storage, or Resource types with different
interface metadata. Resource set/binding conflicts are checked after selection.
Attribute locations are reserved before selection using the maximum location
span of all candidate attribute types, preserving deterministic locations
across variants.

Across bundle variants, a parameter name retains one stable logical slot even
when its selected kind or type changes. A variant in which the parameter is
absent does not contain that parameter row.

## Logical ABI versus physical bindings

An ordinary parameter that becomes unused after feature specialization remains
part of the variant's logical host ABI. Optimization must not silently change
the required host argument list.

The compiler marks each reflected parameter use as physically active or
inactive after specialization. An inactive use remains logically visible but
does not create:

- a Vulkan descriptor or push-constant member;
- a D3D12 descriptor or root-constant member;
- an OpenGL uniform, texture, sampler, or buffer bind;
- resource retention, transition, or command-encoding work.

`vd.When` is the explicit mechanism for changing the logical ABI. If its
parameter is removed, it disappears from specialized source, entry reflection,
the bundle variant parameter table, and runtime argument validation.

For example, moving sampling entirely under a disabled feature keeps ordinary
parameters logical but physically inactive:

```python
def fragment(optional_image: TextureType, optional_sampler: SamplerType):
    result = default_color
    if OPTIONAL_IMAGE:
        result = vd.texture_sample(optional_image, optional_sampler, uv)
    return result
```

Annotating both parameters with `vd.When[OPTIONAL_IMAGE, ...]` removes them
from the disabled variant instead.

## Reflection and runtime work

1. Extend feature declarations and compile requests from string sets to
   canonical typed assignments while preserving the Boolean shorthand.
2. Add the shared compile-time evaluator and specialize control flow and
   `vd.When` before frontend type collection.
3. Emit explicit physical activity in compiler reflection. Missing
   `sampled_texture_bindings` must not be used as an implicit inactivity signal.
4. Extend bundle parameter uses and the native manifest parser with an
   optional `active` field whose absent value means `true`.
5. Update OpenGL, Vulkan, and D3D12 pipeline resolvers so inactive logical
   arguments map to ignored runtime bindings and never reach provider layouts.
6. Keep OpenGL `-1` uniform locations as a final driver-DCE fallback rather
   than the primary activity detector.

## Tests and acceptance

- Preserve all existing Boolean feature and `vd.When[A, T]` behavior.
- Test typed defaults and assignments, imported features, expression folding,
  invalid types, overflow, division by zero, and canonical cache identities.
- Test ordered cases, first-match behavior, `Else`, no-match removal, different
  Value/Texture/Matrix types, maximum location reservation, and selected-branch
  binding conflicts.
- Restore the inactive texture test so `texture_sample` exists only inside the
  feature branch; no unconditional sample may be added to satisfy reflection.
- Verify ordinary inactive parameters remain required logical arguments but
  allocate no backend binding.
- Verify `vd.When` parameters disappear when unmatched and use the selected
  type when matched.
- Cover specialized source/MLIR, compiler reflection, bundle JSON, native
  manifest parsing, and OpenGL/Vulkan/D3D12 execution.
- Run the complete Python compiler/runtime suites, native CTest suite, and
  `pbr.py --no-cubemap` on all three graphics backends.
