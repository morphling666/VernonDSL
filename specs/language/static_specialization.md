# Static specialization

Status: implemented. The rules are normative in
[`contract.md`](contract.md) sections 6.1 through 6.7 and, for manifest keys,
in [`../program/execution_manifest.md`](../program/execution_manifest.md).
This document keeps the rationale.

Specialization is the language's only compile-time selection mechanism. The
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

Before this design, an integer specialization was substituted as a constant
after module loading: every branch was type-checked, every callee emitted, and
every branch's resources stayed in the entry signature. `When` accepted only a
Boolean feature, and an entry could not use code it did not import.

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

## 2. Static branches

Branches are decided before method resolution, reachability, and typing, so a
discarded branch carries no diagnostics, callees, or resources. That is what
lets one template hold per-axis code whose branches would not type-check
together.

Source dependency identity stays independent of the key: a module referenced
only from a discarded branch is still a dependency, so editing it invalidates
cached variants conservatively.

## 3. Conditional parameters

Locations are assigned before `When` evaluation so disabled parameters keep
stable location reservations across variants; bindings are planned after
evaluation so disabled resources take no binding.

`==` on a `Specialization` keeps its Python value meaning because
specializations are mapping keys in `variants=` and `specializations=`. The
other operators therefore raise on the host, and annotations that use them
must be deferred.

## 4. Binding groups

A group carries no interface metadata so one declaration is valid in every
stage and the binding planner, not the author, places its leaves.

A group is one consecutive binding range so that a later contract can give
each group its own descriptor set or Metal argument buffer, grouped by update
frequency, without changing source. With bindless resource tables (Engine
extension E3), typed handles become ABI-stable Values that a Struct may hold,
so a material's parameters and texture references can move into one Struct
that serves as the GPU material-table row. Binding groups remain the interface
for directly bound resources.

A group has no `shared` option because its purpose is device-only resources;
host/device code belongs on its Value Structs.

## 5. Typed method resolution

Resolving methods on the inferred receiver type, rather than syntactically,
lets one generic helper call the same method on different binding groups,
which is how a template calls whichever material a key binds. Keeping methods
non-first-class preserves the no-function-values rule of section 1.

## 6. Type-valued specializations

The value's defining module joins the source graph at key binding so that the
template never imports content: the asset, not the library, names the
material. The frontend reads the declaration from source rather than from the
host class, because compilation never executes Python and the value's source
must enter dependency identity. The host passes only the class's canonical
identity, so the asset module that names the class is the one place the
content module is imported.

Conformance is checked at use until the constrained-generics roadmap item adds
declared bounds, for example
`vd.specialization("MATERIAL", vd.type, bound=Surface)`.

The canonical identity reuses the module name the source graph already uses
for imports, relative to the project root, so manifest keys do not depend on
the checkout location; a value with no such path has no portable identity and
is rejected rather than given a machine-local one. The `type` tag is part of
the unpublished `PROGRAM_VERSION` 2 contract.

A Struct keeps its declared name across modules (instead of the module prefix
helpers carry) because host Values, reflection, and manifests name a Struct by
its class; type values always live in another module, so a prefixed name would
make every type-valued Struct unbindable from the host.

## 7. Non-goals

Function values, closures, and lambdas; runtime dispatch; resources inside
Structs; declared interface constraints on type specializations, which belong
to the constrained-generics roadmap item; `When` on binding-group fields;
type values other than declared types and scalar types.
