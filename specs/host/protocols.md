# Protocols

Status: proposed. It designs roadmap item 4 of the
[language growth order](roadmap.md#language-growth-order), and Vernon's
general traits (Vernon `specs/architecture.md`, sections 7.2 and 11). Items
move into [`language.md`](language.md) and [`design.md`](design.md) as they
land.

## Purpose

Host code has no inheritance, and it keeps none: a class layout is a C
struct, reuse is composition through class fields, and polymorphism comes
from protocols, as Rust's traits give it:

- a protocol names a set of method signatures;
- a type declares that it implements a protocol, and the checker verifies it;
- code generic over a protocol is specialized for each implementing type
  (static dispatch, Rust's `impl Trait`);
- each implementation is also a C function table, which an engine, a
  scheduler or a C++ plugin calls through (dynamic dispatch, Rust's
  `dyn Trait`).

The function table lives beside an instance, never inside it, so an
instance's bytes are its declared fields alone. That keeps state migration,
JSON, `fini`, and C++ access unchanged, and it is what an engine holds
anyway: a pointer to the instance and a pointer to how to drive it.

## Declaration

```python
from typing import Protocol

import vernon_dsl as vd
from vernon_dsl import Vector, f32, f64, i32


@vd.protocol(meta=vd.meta(role="system"))
class System(Protocol):
    """Something the scheduler updates once per tick."""

    @vd.host(effects="write")
    def update(self, dt: f64) -> i32: ...

    def priority(self) -> i32:
        return 0


@vd.protocol
class Collider(Protocol):
    def bounds(self) -> Aabb: ...
    def ray(self, origin: Vector[f32, 3], direction: Vector[f32, 3]) -> f32: ...


@vd.host(implements=(System,))
class Mover:
    speed: f32 = 1.0

    @vd.host(effects="write")
    def update(self, dt: f64) -> i32: ...


@vd.struct(shared=True, implements=(Collider,))
class SphereShape:
    center: Vector[f32, 3]
    radius: f32

    @vd.func(shared=True)
    def bounds(self) -> Aabb: ...

    @vd.func(shared=True)
    def ray(self, origin: Vector[f32, 3], direction: Vector[f32, 3]) -> f32: ...
```

- **`@vd.protocol`** marks a class deriving from `typing.Protocol` and
  nothing else. Its body holds methods, a docstring, and nothing else: no
  fields, class constants or nested classes. `meta=` and docstrings are kept
  in reflection, as for classes.
- **Methods** take `self` unannotated and annotate everything else. A method
  whose body is `...` is required. A method with a body is a **default**: an
  implementer that does not define it gets this body.
- **Effects.** `@vd.host(effects=...)` on a protocol method declares its
  effect bound; an undecorated method is `io`, as for class methods.
- **Signatures** follow the rules of exported methods, because each method
  is also a function-table entry: parameters and the result are values, a
  `str` is neither, and a class is never passed.
- **Rejected:** protocols deriving from protocols (supertraits), generic
  protocols (`Protocol[T]`), associated types and constants, dunder methods,
  and private (leading-underscore) methods.
- A protocol nothing implements is allowed; it is reflected but has no
  tables.

## Conformance

A type implements a protocol only by naming it: `implements=(P, ...)` on
`@vd.host` for classes and on `@vd.struct` for Structs. The checker then
requires, for each protocol method:

- a method of the same name on the type, unless the protocol gives a
  default;
- the same parameter and result types exactly, with no variance;
- an effect no greater than the protocol method's;
- for a Struct, a `@vd.func(shared=True)` method, so its effect is `pure`
  and its `self` is a value.

A conforming type may have more methods and fields than the protocol names.
A missing method, a mismatched signature or a greater effect is a
diagnostic at the `implements=` entry, naming the protocol method.

- **Why explicit.** Python's `Protocol` is structural: any type with
  matching methods conforms. Rust's traits are nominal. Host code takes the
  nominal rule:
  - the error appears at the declaration, not at a distant generic call;
  - the AOT build knows exactly which function tables to emit, without
    emitting one for every type that happens to match;
  - an implementation is part of a module's reflection and identity, so a
    plugin or a reload sees the same set of implementations a build did.
- Python type checkers still see `typing.Protocol`, so pyright checks
  conformance structurally as well.
- **Retroactive implementations** (implementing a protocol for a type
  declared in another module, Rust's `impl Trait for Type` outside the
  type's crate) are out of scope; see [Open points](#open-points).

## Static dispatch

A parameter annotated with a protocol makes a function generic over its
implementers:

```python
@vd.host(effects="write")
def _step_all(first: System, second: System, dt: f64) -> i32:
    return first.update(dt) + second.update(dt)


@vd.host(implements=(System,))
class World:
    mover: Mover
    spinner: Spinner

    @vd.host(effects="write")
    def update(self, dt: f64) -> i32:
        return _step_all(self.mover, self.spinner, dt)
```

- **Specialization.** Each distinct tuple of implementing types at a call
  makes a separate copy, as a generic `@func` helper does. Calls through the
  parameter are direct calls to the implementer's method, so they inline,
  and call-site reflection stays exact: a copy lists the externs its
  implementer's methods reach.
- **Arguments.** A class implementer is passed as a place: `self`, a class
  field (`self.mover`), or a protocol parameter passed on. It is borrowed for
  the call, as a method receiver is, and is never copied, stored or
  returned. A Struct implementer is passed as a value.
- **Through a protocol parameter** Host code may call the protocol's methods
  and pass the parameter on to another protocol parameter. It cannot read
  fields, compare, or assign the parameter, because the protocol has no
  fields and the implementer is not known to the code.
- **Effects.** A call through a protocol parameter has the protocol method's
  declared effect, so a generic function checks once against the bound, not
  per implementer.
- **Not exported.** A function with a protocol parameter has no single C
  signature, so it cannot be exported: root-file `@host` functions and
  public methods with protocol parameters are rejected. Private methods,
  `@host` functions of imported modules, and `@func` helpers may have them.
  The C boundary is the function table instead.
- **Defaults** are compiled per implementer, as a specialization with `self`
  bound to the implementer. A default may call only the protocol's methods
  on `self`, since that is all it knows.
- **Python mode** calls the method on the object it receives. The checker
  has already verified every specialization, so duck typing and the
  compiled copies agree.

## Function tables

For each `(type, protocol)` pair in an `implements=`, AOT emits a function
table, and the reflection table lists it. The header declares one C struct
per protocol:

```c
typedef struct DemoSystem_vtable {
    uint32_t struct_size;
    /* The implementing type: its size, init and fini, to allocate and release instances. */
    const VhType *type;
    int32_t (*update)(void *self, double dt, int32_t *out);
    int32_t (*priority)(void *self, int32_t *out);
} DemoSystem_vtable;

extern const DemoSystem_vtable demo_Mover__as_System;
```

- **Entries** are in protocol declaration order and use the exported method
  convention, `int32_t (params..., R *out)` returning 0 or a status, with
  the instance pointer first. A Struct implementer's entry receives a
  pointer to the Struct's bytes and only reads them. A default the type
  does not override points at its specialized copy.
- **`struct_size` first**, and new protocol methods append, so a C++ caller
  built against an older protocol keeps reading newer tables.
- **Symbols** are `<prefix><Type>__as_<Protocol>`.
- **C++ implementations.** A C++ type implements a protocol by filling in
  the same struct, with a `VhType` describing its layout, and handing it to
  the application, so a plugin registers a `Collider` the build never saw.
  `bindgen.md`'s generator can emit the C++ wrapper that does this.
- **Python mode** builds the same tables with ctypes beside
  `HostModule.table`, whose entries are the addresses its method records
  already use, so a scheduler drives Python-mode and compiled instances
  through one code path.

## Reflection

`VhModule` gains appended members, read when `struct_size` covers them:

```c
typedef struct VhProtocol {
    uint32_t struct_size;
    const char *name;
    const char *doc;
    const VhMeta *meta;
    uint32_t meta_count;
    /* Signatures in declaration order; `symbol` and `address` are NULL, and `calls` is empty. */
    const VhFunction *methods;
    uint32_t method_count;
} VhProtocol;

typedef struct VhImpl {
    uint32_t struct_size;
    const VhProtocol *protocol;
    const VhType *type;
    /* The <Protocol>_vtable, whose entries are in `protocol->methods` order. */
    const void *table;
} VhImpl;
```

- `VhModule.protocols` lists the protocols the module declares or
  implements, and `VhModule.impls` every implementation, ordered by
  protocol then type name. `vh_find_impl(module, protocol_name, type_name)`
  finds one.
- An application finds every `System` in a module by scanning `impls`,
  which is Vernon's `query<System>()` for Host code. The implementer's own
  `VhType` methods still carry their exact call-site reflection, so a
  scheduler derives each system's reads and writes from it.
- The reflection JSON gains `"protocols"` (name, doc, meta, method
  signatures) and `"impls"` (protocol, type, symbol).
- **Hashes.** The module hash covers each protocol's method signatures and
  the implementation set. A library's identity covers the protocols it
  declares, since C++ code implements them against the header.
- **Reload.** `vh_migrate` is unchanged, because tables are not in
  instances. After `HostModule.reload()` an application looks implementations
  up again by protocol and type name, as it does functions.

## Device code

Vernon's traits are one declaration used by world queries, CPU systems and
device code. Device code has no function pointers, so it uses static
dispatch only:

- a protocol whose implementers are Structs with `@func(shared=True)`
  methods is usable as a parameter type in `@func` helpers called from
  kernels, specialized per Struct as on the host;
- dispatch over several Struct kinds in one dispatch (Vernon's geometry
  kinds' `fetch`) is a tag switch over a closed set, a separate design.

This phase follows the Host phases and is listed so the Host design does
not preclude it: Struct implementers and `pure` value receivers are what it
needs.

## Phases

1. Declaration and conformance: `@vd.protocol`, `implements=`, defaults,
   diagnostics, and Python-mode behavior.
2. Static dispatch in both modes: protocol parameters, specialization,
   defaults, effects and call-site reflection, with parity tests.
3. Function tables: AOT tables, the header struct, Python-mode tables, the
   `VhProtocol`/`VhImpl` reflection records, hashes, and a C consumer test
   that drives instances through a table.
4. Vernon's `HostSystem` adapter schedules `System` implementations from
   `impls`.
5. Device static dispatch for Struct implementers.

## Open points

- **Retroactive implementations**: a block such as
  `@vd.impl(Collider, for_=SphereShape)` holding the methods, for types a
  library module declares. It needs a rule like Rust's orphan rule, so two
  modules cannot both implement one pair.
- **Dynamic dispatch inside Host code**: `dyn` class fields and
  `list[dyn System]` need fat pointers (instance and table) and borrowed
  element places. They are designed together with `list[C]` of Host
  classes, which raises the same questions.
- **Supertraits**: `@vd.protocol(requires=(A,))` once a use needs one; a
  table would then hold its required protocols' tables.
