# Function slots and cook-time composition

Status: accepted, compiler-contract additive. Program manifests are unchanged:
a composed Kernel or Pipeline cooks into an ordinary Program.

A *template* is a Kernel or graphics Pipeline whose source calls *function
slots*: named, typed placeholders for `@vd.func` implementations. Binding
slots to functions produces a new Kernel or Pipeline at cook time. One surface
function can thus be cooked into many contexts (forward fragment, masked
raster, compute resolve) and one context into many geometry kinds (static
fetch, skinned fetch), without textual code generation and without the
language learning any domain concept such as "material".

Composition is source-level specialization. It is not a runtime dispatch, not
a Program variant axis, and not a function value: slots are resolved before
compilation and each composition compiles to straight-line calls.

## 1. Declaring a slot

```python
@vd.function_slot
def surface(params: MaterialParams, inputs: SurfaceInputs) -> SurfaceOutput: ...
```

- The body is `...` (or a docstring followed by `...`); a slot has no default
  implementation.
- Annotated parameters and the return annotation form the slot *prototype*.
  An unannotated parameter is generic: any bound type is accepted, and the
  template's own uses must type-check after binding.
- Slots are module-level symbols. Templates call them like functions:
  `out = surface(params, inputs)`. A slot may be called from the entry or from
  any `@vd.func` reachable from it.
- A slot is identified by its defining module path and name.

## 2. Binding

```python
forward_static = vd.compose(forward_template, {surface: default_lit, fetch: static_fetch})
```

`vd.compose(template, bindings)` returns a new Kernel or Pipeline of the same
kind as `template`. For a Pipeline, the bindings apply to both stages.

Binding rules, each a `TypeError` at `compose` time or a compile error with the
slot name when it needs source analysis:

- every slot reachable from an entry of the template must be bound;
- every binding must name a slot reachable from some entry (no unused
  bindings);
- a bound value must be a `@vd.func` (an entry decorator, a lambda, or a slot
  is rejected);
- composing an already-composed template may bind only slots that are still
  unbound; rebinding a slot is rejected;
- a bound function must not itself call slots.

The bound function's first `n` parameters, where `n` is the prototype arity,
correspond positionally to the prototype parameters. Each annotated prototype
parameter type must equal the bound parameter type exactly; the prototype
return type, when annotated, must equal the bound return type exactly.

## 3. Lifted parameters

Parameters of the bound function after the prototype arity are *lifted*: they
become entry parameters of every entry that calls the slot directly. This is
how a fetch function brings its vertex and bone buffers, or a surface function
its textures, without the template naming them.

- A lifted parameter must be resource-like: a `TensorView`, `Texture`,
  `Sampler`, or `uniform()` value. Attribute, varying, and builtin parameters,
  and explicit `set`/`binding`, are rejected.
- A lifted parameter is named `<slot>__<parameter>` in the composed entry and
  in host calls, Program parameters, and reflection.
- The slot call passes the lifted entry parameters as trailing arguments.
- A slot with lifted parameters may be called only from an entry. A call
  from a helper is an error, because the helper's signature would otherwise
  change per composition.
- A lifted name that collides with an existing entry parameter is an error.

Lifting runs after reachability pruning, so only entries that actually call
the slot gain parameters.

## 4. Derived types

`vd.parameter_type(slot, index)` returns the annotation of prototype parameter
`index`, so templates and host code can name slot types without importing the
implementation module.

## 5. Identity and caching

Bindings are part of frontend compile identity. The frontend request carries
the sorted `(slot module, slot name) -> (function module, function name)` map
and the bound functions' source dependencies; frontend and compile caches key
on it. Two compositions of the same template with different bindings never
share frontend results, and a composed template's semantic inputs include the
bound sources.

Host-side signatures of a composed Kernel or Pipeline (argument
normalization, annotations, and Program parameter typing) are derived from the
composed entry, including lifted parameters, not from the template's Python
function.

## 6. Enumeration and cooking

```python
assets = vd.composition_assets(
    id="materials/{surface}/forward/{fetch}",
    template=forward_template,
    slots={"surface": {"default_lit": default_lit, "unlit": unlit},
           "fetch": {"static_mesh": static_fetch, "skinned_mesh": skinned_fetch}},
    variants=...,
)
```

`vd.composition_assets` returns one Program asset declaration per element of
the cartesian product of the slot choices, sorted by asset id. Each id is the
format string with every `{slot}` replaced by the choice key. Choice keys are
non-empty `[a-z0-9_]` strings; every slot of the template appears exactly once
in `slots` and in the id pattern. Duplicate ids are rejected. `variants` keeps
its ordinary meaning (scalar specialization keys) and applies to each
composition; compositions are never variant keys.

The cook API and CLI accept a set of declarations. All compositions compile
before any bundle is written, so a failed composition leaves no partial output
for the set.

## 7. Diagnostics

Stable error fragments: `unbound function slot`, `unused function slot
binding`, `function slot binding must be a @vd.func`, `function slot already
bound`, `bound function calls a function slot`, `does not match function slot
prototype`, `lifted parameter`, `duplicate composition asset id`.

## 8. Acceptance

- a Kernel and a Pipeline composed with different bindings compute or render
  the results of their bound functions;
- lifted TensorView and Texture parameters appear in reflection and host calls
  with `<slot>__<parameter>` names;
- every binding error in section 2 and lifting error in section 3 is rejected
  with its diagnostic;
- frontend caches do not alias different bindings of one template;
- composition enumeration is complete, sorted, and rejects duplicate ids;
- cooking a set writes nothing when one member fails.
