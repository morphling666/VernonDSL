# Native library bindings

Status: partly implemented. The declarations (`vd.library`, `meta=` on
externs and handles, the computed schema identity) and the level 1
generator, with spans, owned results, generic externs, type descriptors,
Struct and handle `cpp=`/`convert=`, dependent handles, Python handle
classes, the effect check and stub placement, are in
[`design.md`](design.md#libraries); the language growth below is proposed. It realizes the roadmap decisions "One binding per native
function" and "The glue is generated"
([`roadmap.md`](roadmap.md#decisions)). Items move into
[`design.md`](design.md) as they land.

## Purpose

Host code is glue over native C and C++ libraries. A library is reached the
same way at every stage:

| Stage | Host code runs as | A native function is reached through |
|---|---|---|
| Development | Python mode | its address, from the library's nanobind module or shared library |
| Desktop release | AOT object | the same C function, linked |
| Mobile and web | AOT object, linked statically | the same C function, linked |

The C function is the same in every row, so a binding cannot drift between
modes and Python mode needs no per-library conversion code. Python mode
already calls native addresses: `ExternRegistry.bind_symbols` and
`bind_library` bind them, and `native.py` calls each through a ctypes
prototype built from the extern's signature, passing Structs and Tensors as
packed Host ABI bytes.

## The Host extern module is the schema

A library's binding is an ordinary Host module of `@vd.handle`,
`@vd.struct`, `@vd.enum` and `@vd.extern` declarations. Host code imports
it, Python mode binds it, AOT objects link it, and the generator reads it.
There is no second declaration language, so the declarations every tool
reads are one source.

```python
import vernon_dsl as vd
from vernon_dsl import Vector, f64

vd.library("physics", version=0, meta=vd.meta(cpp_namespace="physics::impl"))


@vd.handle(meta=vd.meta(cpp="physics::World", header="physics/world.h"))
class World:
    """A simulated world; destroyed with world_destroy."""


@vd.handle
class Body:
    """A body of a World, always passed with it."""


@vd.struct
class BodyState:
    position: Vector[f64, 3]
    velocity: Vector[f64, 3]


@vd.extern(effects="io", abi="status")
def world_create() -> World: ...


@vd.extern(effects="io", abi="status", meta=vd.meta(destroys="world"))
def world_destroy(world: World) -> None: ...


@vd.extern(effects="write", abi="status")
def body_set_state(world: World, body: Body, state: BodyState) -> None: ...
```

What a declaration must say, and why nothing less works:

- **The C symbol.** `vd.library(name, version)` sets the default symbol of
  the module's externs to `<name>_v<version>_<function>`; an explicit
  `symbol=` still wins, for existing C APIs (implemented: design.md,
  "Libraries").
- **The C layout of every type.** Structs and handles are Host types, whose
  layout the Host ABI fixes on every target.
- **The effect.** C and C++ signatures cannot express effects on state
  outside the arguments, and Host code needs them for its effect checks
  and for engines that schedule by them.
- **For C++ libraries only, how to reach the C++ side.** These are `vd.meta`
  keys that only the generator reads, so the language learns nothing about
  C++.

## Three levels

A library pays only for what it needs.

### Level 0: a library with a C API

Write the externs with explicit `symbol=` against the existing functions
(SDL, sqlite, ONNX Runtime, Box2D v3 and most codecs have one). Nothing is
generated:

- AOT objects link the library.
- Python mode binds it with `ExternRegistry.bind_library(path)`, or with
  `bind_module` when a nanobind module publishes its addresses, which
  `vd.library(..., binding="pkg.module")` makes automatic (design.md,
  "Libraries").

A C function whose parameters the Host ABI cannot express (callbacks,
tensor storages) needs a small hand-written C wrapper.

### Level 1: a C++ library

Implemented ([design.md](design.md#level-1-bindings-vernon_dslbindgen)).
`python -m vernon_dsl.bindgen generate <module.py>` reads the same module
and emits:

| File | Contents |
|---|---|
| `<name>_api.h` | the C header: the Host types' C spellings and the prototypes; VernonDSL already emits headers for Host modules |
| `<name>_api_thunks.cpp` | the C functions: handle lookup, conversions, a call to `<cpp_namespace>::<function>`, status and last error |
| `<name>_api_impl.h` | the declarations the library implements, in C++ types |
| `<name>_api_nanobind.cpp` | the nanobind module: Python functions over the C functions, and the symbol table |
| `<name>_api.pyi` | Python stubs of the nanobind module |

The C++ mapping keys:

- **On `vd.library`:** `cpp_namespace`, where the implementation lives,
  and optionally `effect_check`, a function there that every non-pure
  status thunk asks before calling the implementation (design.md, "Level
  1 bindings").
- **On a handle:** `cpp` and `header`, the object type. A handle with `cpp`
  gets a handle table.
  - An extern returning it inserts what the implementation returns
    (`vh::Result<std::unique_ptr<T>>`).
  - An extern with `destroys="<parameter>"` removes it.
  - `depends_on="<Handle>"` destroys it with the parent it was created
    with.
  - With `convert="<function>"` as well, the bits are a C++ value the
    library converts, and no table holds it.
  - A handle without `cpp` passes its bits through, for ids the library
    manages itself.
  - The externs the handle's class body declares are methods of the nanobind
    class and its stub, `counter.add(5)` and the static `Counter.create(mode)`,
    and only those: the module has no function of their C names, so Python
    calls them as Host code does.
- **On a Struct:** `cpp` and `header`, a C++ type of the C layout, which
  the thunks cast after generated `static_assert`s on size, alignment and
  trivial copyability. With `convert="<function>"` as well, the layouts may
  differ and the implementation defines both conversions in
  `cpp_namespace`. Without `cpp` the implementation takes the C struct.

The library's own C++, such as its other bindings, reaches the tabled
objects through `<cpp_namespace>::handles` in `<name>_api_impl.h`:
`find(handle)` returns the object behind a live handle or null,
`handle_of(object)` returns the live handle of a tabled object or a null
one, so library code can hand an object it reached by pointer back to
Host code, and `destroy_all<Handle>()` destroys every object of a handle as destroying
each would, for a teardown hook. The nanobind module's contents are
`void <name>_v<version>_nanobind(nb::module_ &)`; defining
`<NAME>_NANOBIND_HOSTED` leaves out `NB_MODULE`, so an existing extension
holds them as a submodule and shares the handle tables, which a second
extension module would duplicate.

`python -m vernon_dsl.bindgen check <module.py>` fails when a committed
output differs from what the module generates. The outputs are emitted in
the project's style rather than run through clang-format or ruff, so the
check does not depend on tool versions.

### Application concepts

An application's own concepts are its declarations and its C++, never a
generator extension, so VernonDSL keeps no engine concepts and an
application generates nothing of its own. Vernon's components show the
pattern:

- **A component is a Struct** whose `vd.meta` names it as one, with the
  fields every component of its kind shares.
- **Access is generic externs**, `component_get(world, entity, type[T])`
  and `component_set(world, entity, value: T)`, which the implementation
  dispatches on the descriptor in a C++ registry of its component types. A
  new component is a Struct and a registry entry, not new externs.
- **Effect checks** are the library's `effect_check` function, which reads
  the call's effect, metadata and bound Struct; **scene files** are the
  registry's, which reads and writes fields through the library's type
  descriptors.

## Binding a module and checking it

- **Schema identity** and **`ExternRegistry.bind_module(module)`** are
  implemented ([design.md](design.md), "Libraries"). The generator
  publishes the same `library_identity` it computes for loading, so no
  declaration spells it by hand.
- **The nanobind module** defines `vh_schema`, `vh_symbols()` (a dict
  from each C symbol to its address) and `vh_last_error()`.
- **AOT objects** link the symbols. The generated `<name>_v<version>_schema()`
  lets an application compare the linked library with the identity its Host
  modules were built against.

## Extern metadata in reflection

`@vd.extern(meta=...)` and `@vd.handle(meta=...)` are implemented
([design.md](design.md), "Metadata and docstrings"). A method's call list
carries each reachable extern's metadata, and a generic extern's entry its
bound Struct: an engine reads its own keys from them, and Vernon derives a
system's reads and writes from which component types its generic calls
bind.

## Error convention

Status externs return 0 or a nonzero status. A generated thunk stores the
message in a thread-local last error, which a generated
`<name>_v<version>_last_error` copies out. Implementations return
`vh::Result<T>` from the C++ helper header `vernon_host/bindgen.hpp`, and
the thunks unwrap it. A failing extern ends the calling Host function with
its status in both modes. Python mode raises `HostExecutionError` with the
status and the last error's text.

## Language growth this needs

Library APIs need more of the extern ABI than gameplay code has used:

- **TensorStorage spans.** A borrowed `{data, shape}` parameter for buffers
  handed to physics, image and audio libraries. Until then, a function
  taking a storage has no Host extern, and its nanobind function takes a
  numpy array.
- **Callbacks.** A function pointer and a context pointer, for libraries
  that call back into Host code.
- **Triples.** Android (`aarch64-linux-android`) and macOS arm64
  (`aarch64-apple-darwin`) objects in the tested targets.

## Tests

- A level 0 module bound with `bind_library` against a plain C library.
- A Host module whose externs differ from the bound library's fails to load.
- Reflection: a method's calls carry their extern metadata in both tables.

## Rejected alternatives

- **A separate schema vocabulary** (`api`, `handle`, `struct`, `function`,
  as Vernon's schema had): it restates what Host declarations already say
  and then generates those declarations again, so the schema and the Host
  module become two sources for one fact.
- **Reading C++ headers with libclang** (as SWIG, cppyy and Rust's
  `bindgen` do): headers state neither effects nor ownership nor a stable
  C layout, so annotations are needed anyway. Template-heavy C++ is also
  hard to read reliably, and the build would need libclang on every host.
- **Introspecting nanobind signatures:** nanobind functions take Python
  objects, which give no C ABI for AOT objects to link.
- **Generator extensions for application concepts** (a Vernon `component`
  expanding into per-component externs, thunk bodies and scene codecs): each
  component would grow the C API by four or more functions and the
  application would own a second generator. Generic externs over
  descriptors give the same access through one set of functions, and keep
  the application's code ordinary C++.
