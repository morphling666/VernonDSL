# Static specialization

Status: accepted design. Sections 2 through 5 are implemented and summarized
in [`contract.md`](contract.md) section 6, with the limitations listed at the
end of section 5. Sections 6 and 7 are not yet
implemented; on implementation their rules replace the remaining
specialization paragraphs of `contract.md` section 6 and the specialization-key
paragraph of
[`../program/execution_manifest.md`](../program/execution_manifest.md), and
this document then keeps only the rationale.

Specialization is the language's only compile-time selection mechanism. This
design generalizes it instead of adding a composition feature:

1. every specialization condition is static, and `When` takes any
   compile-time predicate;
2. binding groups aggregate entry interface items the way Structs aggregate
   Values, and methods resolve on inferred types;
3. a specialization value may be a type.

## 1. Motivation

Library entries such as a visibility-buffer resolve kernel combine axes that
different code owns:

- library-owned closed sets (shading model, quality level), selected by value;
- library-owned geometry kinds, each with its own buffers and fetch code;
- content-owned surface code (a game's custom material) with its own
  textures, which the library must not import.

Today an integer specialization is substituted as a constant after module
loading: every branch is type-checked, every callee is emitted, and every
branch's resources stay in the entry signature. `When` accepts only a Boolean
feature. There is no way for an entry to use code it does not import.

Rejected alternatives:

- function slots and `compose`: a second binding mechanism beside
  specialization, plus naming rules for lifted parameters;
- function values or closures: the frontend loads sources without executing
  Python, and function values have no stable cache identity;
- a variant per implementation in the library template: the template must
  import all content, and the entry carries every implementation's resources;
- generated wrapper sources: textual code generation;
- resources inside `@vd.struct`: a Struct is an ABI-stable Value that can be
  a Tensor or TensorView element, which a resource cannot be;
- bindless handles only: GLSL 3.30 and GLSL ES 3.00 index sampler arrays only
  with constant expressions, so the fallback tier needs ordinary bindings.

A type value fits the domain: a material is one unit of parameters,
resources and code, which is a binding group with methods.

## 2. Compile-time expressions and static branches

A compile-time expression contains only literals, captured constants,
specialization names, and these operators:

- `==` and `!=` on every specialization kind;
- `<`, `<=`, `>`, `>=`, `+`, `-`, `*`, `//`, and `%` on numeric kinds;
- `and`, `or`, `not`.

A type-valued specialization compares only with `==` and `!=` against a type
reference.

When the test of an `if`, `elif`, or conditional expression references a
specialization, the whole test must be a compile-time expression; otherwise
compilation fails with `specialization condition mixes runtime values`. Such a
branch is static:

- it is decided when the specialization key is bound, before method
  resolution, reachability, and typing;
- only the selected branch remains; discarded branches have no diagnostic
  obligation and contribute no callees, resources, or reachability;
- a name bound only in a discarded branch is undefined afterwards.

Source dependency identity stays independent of the key: a module referenced
only from a discarded branch is still loaded and recorded as a dependency, so
editing it invalidates cached variants conservatively.

Boolean feature tests are the Boolean case of this rule. Any other use of a
scalar specialization is a typed constant, as today. Using an unassigned
scalar specialization fails with `missing specialization`; an unassigned
Boolean is false.

## 3. Conditional parameters

`When[condition, T]` accepts any compile-time expression as `condition`.
A disabled parameter is absent from the entry interface, host call signature,
reflection, and Program parameters. Using a disabled parameter in a selected
branch is the existing `disabled value` error.

Attribute and varying locations are assigned before any `When` is evaluated,
so disabled parameters keep stable location reservations. Resource bindings
are planned after evaluation, so disabled resources take no binding.

Graphics Pipeline assets accept every specialization kind. Both stages of a
pipeline compile with the same specialization key.

The host evaluates annotations with each specialization name bound to its
value in the key, and a `When` annotation resolves to its wrapped type; the
host's parameter set comes from the specialized entry. A `Specialization`
object refuses `bool()`, ordering, and arithmetic, so on Python 3.11 through
3.13 a module whose annotations contain conditions other than `==` and `!=`
needs `from __future__ import annotations`. `==` keeps its value meaning
because specializations are mapping keys in `variants=` and
`specializations=`.

## 4. Binding groups

```python
@vd.struct(shared=True)
class HologramParams:
    tint: vd.Vector[vd.f32, 4]
    speed: vd.f32

@vd.binding_group
class Hologram:
    params: HologramParams
    noise: vd.Texture["2d", vd.f32]

    @vd.func
    def surface(self, inputs: SurfaceInputs) -> SurfaceOutput: ...
```

A `@vd.binding_group` class declares a named group of entry interface items.
Its fields are:

- resources: `TensorView`, `Texture`, `Sampler`;
- Values, including Structs, which bind as immutable uniforms in graphics
  stages and as ordinary Value parameters in Kernels;
- nested binding groups.

Fields carry no interface metadata: no `attribute`, `varying`, `builtin`,
`set`, or `binding`. So one group is valid in every stage and the binding
planner places all of its leaves. Graphics TensorView fields follow the
read-only rule of
[`../program/graphics_storage_reads.md`](../program/graphics_storage_reads.md).

A binding group is not a Value. It may appear only as an entry parameter, a
helper parameter, a forwarded argument, or the receiver of a field access or
method call. It cannot be constructed, assigned to a local, returned,
compared, used as a Struct field or a Tensor or TensorView element, or passed
to `@func(shared=True)` code. `@vd.struct` is unchanged.

Methods are `@vd.func` and cannot reassign fields. A binding group has no
`shared` option, because its purpose is device-only resources; host/device
code belongs on its Value Structs, which may be `shared=True`.

At an entry, each leaf becomes one interface item named by its dotted path,
for example `material.noise`, `material.params`, or `geometry.positions`.
Reflection, Program parameters, and host keyword binding use the dotted
names. On the host, the class is a plain container:
`Hologram(params=HologramParams(...), noise=texture)` binds leaf by leaf.

A binding group is one allocation unit. The binding link
([`../program/graphics_storage_reads.md`](../program/graphics_storage_reads.md)
section 3) assigns all generated bindings of one group occurrence, including
spilled uniforms, storage leaves, and nested groups, as one consecutive range
in field order. The same group under the same pipeline name in both stages
reuses that range. All ranges are in descriptor set 0 today. Because a group
is always contiguous, a later contract can give each group its own descriptor
set or Metal argument buffer, grouped by update frequency, without changing
source.

Differentiation treats a binding group leaf by leaf, under the rules of
[`../autodiff/contract.md`](../autodiff/contract.md) section 3. Value leaves
are ordinary Value parameters, and TensorView leaves are Storage parameters;
their cotangents use the leaf's dotted name. Texture and Sampler leaves are
non-differentiable, and a transform that rejects a resource kind also rejects
that kind inside a group.

With bindless resource tables (Engine extension E3), typed handles become
ABI-stable Values that a Struct may hold, so a material's parameters and
texture references move into one Struct that serves as the GPU material-table
row. Binding groups remain the interface for directly bound resources.

## 5. Typed method resolution

`x.m(...)` resolves on the inferred Struct or binding-group type of `x` during
typed helper specialization, not by syntactic lookup of the receiver. Generic
helpers may call methods on unannotated parameters, and each helper
specialization resolves its calls independently.

- Methods are reachable only through resolved calls; unreachable methods are
  pruned without diagnostics, like unreachable generic bodies.
- A missing method fails at the call site with `'T' has no method 'm'` and
  the helper specialization chain.
- Methods remain non-first-class.

Binding groups at the program boundary:

- Every backend accepts every group leaf kind, including Struct and Tuple
  Value leaves in GPU Kernels. A leaf's scalar types still follow the target
  capabilities; for example, SPIR-V targets reject `f64`.
- `Module.forward()` accepts binding-group parameters. They must be
  annotated, because the Module captures one Program parameter per leaf from
  the annotation. `forward` may pass a group to the Kernels and Pipelines it
  calls, and may construct a group from leaves it holds.
- A VJP `wrt` or `outputs` path may name a group root, or a nested group, as
  shorthand for its leaves in field order. For `wrt` the root expands to
  every differentiable leaf. These are floating Value leaves, with Struct and
  Tuple Values expanded to their floating fields, plus whole floating
  TensorView leaves. For `outputs` it expands to the writable floating
  TensorView leaves. A root with no such leaf is rejected with
  `names a binding group with no ... leaves`. Expansion happens before the
  paths become the transform identity, so cotangents and cooked manifests see
  only leaf paths.

## 6. Type-valued specializations

`vd.specialization(name, vd.type)` declares a specialization whose value is a
`@vd.binding_group` class, a `@vd.struct` class, or a scalar type.

- The specialization name is a type wherever an annotation is allowed:
  `material: MATERIAL` for a binding group, or `Annotated[MATERIAL, uniform()]`
  for a Struct in a graphics stage.
- At key binding, the module that defines the value joins the source module
  graph as if imported. The template's own imports are unchanged.
- ProgramAsset `variants=` and Kernel specialization assignments take the
  class object itself, for example `{MATERIAL: Hologram}`. The frontend loads
  the declaration from its defining source; it never executes that module.
- A type specialization has no default; a missing assignment is an error.
- Conformance is checked at each use, not declared. When the
  constrained-generics roadmap item adds constraints, they go on the
  declaration, for example
  `vd.specialization("MATERIAL", vd.type, bound=Surface)`.

## 7. Identity and manifest

The frontend request, frontend caches, and semantic identity include each
type assignment's canonical identity and the source digests of its defining
module graph. Two keys that differ only in a type value never share frontend
results.

A declared type's canonical identity is its defining module's name in the
source module graph, then `:` and the class name. The module name is the one
the graph already uses for imports and emitted symbols: the module path
relative to the project root (the nearest directory containing
`pyproject.toml`), with `.` separators and no `.py` suffix. A type value whose
module is outside the project root has no portable identity and is rejected.
A scalar type's identity is its scalar name. The manifest key entry is

```json
{"name":"MATERIAL","value":{"tag":"type","value":"engine.materials.hologram:Hologram"}}
```

and Runtime compares the string bytewise when selecting a variant. The
`type` tag is part of the unpublished `PROGRAM_VERSION` 2 contract.

## 8. Processing order

The contract's processing order becomes:

1. load and validate source imports;
2. bind the canonical specialization key: substitute scalar constants,
   evaluate static branches and `When` conditions, and load type-valued
   modules;
3. normalize generated builtins;
4. prune to selected entries and validate the call graph;
5. infer and specialize typed helpers, resolving method calls and flattening
   binding groups;
6. lower typed nodes;
7. validate target capabilities and materialize artifacts.

## 9. Diagnostics

Stable fragments: `specialization condition mixes runtime values`,
`disabled value`, `binding group cannot be`, `binding group field cannot
declare`, `has no method`, `type specialization requires a declared type or
scalar type`, `missing type specialization`, `type value has no portable
identity`.

## 10. Acceptance

- An integer static branch prunes the callees and type errors of discarded
  branches, and a `When[MODE == 2, ...]` resource disappears from the
  interface and bindings when `MODE != 2`, for Kernels and graphics stages.
- Graphics Pipeline assets cook and render with integer and type variants.
- A binding group with TensorView, Texture, Sampler, Struct, and nested-group
  fields works as an entry parameter in a Kernel and in both graphics stages
  on every backend, with dotted names in reflection and host calls.
- Every binding-group misuse in section 4 is rejected with its diagnostic,
  and a binding group used as a Struct field or Tensor element is rejected.
- A group's generated bindings form one consecutive range in field order,
  and a group declared by both stages of a pipeline shares that range.
- A Kernel VJP over an entry with a binding group returns cotangents for its
  floating Value and TensorView leaves under their dotted names, and none for
  Texture or Sampler leaves.
- One generic helper calls the same method on two different binding groups.
- A template that does not import its material cooks against two
  binding-group values, frontend caches do not alias them, and editing one
  value's source invalidates only the keys that use it.
- Manifest keys carry module-name identities that do not depend on the
  checkout location, and a type value outside the project root is rejected.

## 11. Non-goals

Function values, closures, and lambdas; runtime dispatch; resources inside
Structs; declared interface constraints on type specializations, which belong
to the constrained-generics roadmap item; `When` on binding-group fields;
type values other than declared types and scalar types.
