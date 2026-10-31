# Host language roadmap

Status: active plan, not implemented. It extends [`design.md`](design.md) and
[`language.md`](language.md), which describe the implemented language; each
item moves into those documents when it lands. It supersedes the ordering of
[`../future/host_language.md`](../future/host_language.md) where they differ.

The first consumer is the Vernon engine, which builds its systems, components
and plugins on the Host language (Vernon `specs/world/design.md`, "Systems
from plugins"). Nothing here is specific to it.

## Decisions

- **General purpose.** The language has no engine concepts. An application
  expresses its own (systems, components, phases) through `vd.meta` keys,
  its own extern modules and its own conventions, as Vernon does.
- **A feature enters only with AOT semantics.** A construct is accepted only
  if it compiles without CPython and means the same in both modes, apart from
  the listed Python-mode differences. Within that rule the language supports
  as much of Python as it can, in the order below. Dynamic typing, arbitrary
  imports, `eval`, and Python objects in compiled code stay out; an
  application that needs them runs ordinary Python beside Host code, on
  platforms that have Python.
- **Compiled code is authoritative.** Python mode stays ordinary CPython for
  debugging, with the documented numeric differences, and is not made
  bit-identical. Anything that needs reproducible results across machines
  (replays, goldens, lockstep) runs the compiled object, which tests already
  link (`LinkedHost`).
- **Python spellings.** A Host construct is written as in typed Python:
  `list[T]`, `dict[K, V]` and `str`, not new names.
- **Dynamic data is owned by a class instance.** Containers, strings and
  device storages live in instance fields and are freed with the instance. Nothing references into
  them from outside the instance, so there is no garbage collector and no
  reference counting, and migration and the place rules extend to them.
- **Callers reach Host code only through the module's reflection table.** An
  application calls methods by the addresses in `VhModule` and never caches
  them elsewhere. Swapping the table is then the whole of a reload in Python
  mode today and of a compiled reload later, which becomes a loader change
  rather than an ABI change.
- **Hot reload is Python-mode reload first.** `HostModule.reload` and
  `migrate` already keep instance state in application memory across an
  edit. Reloading compiled code (a shared library swapped under `vh_migrate`)
  comes later; iOS forbids it in release builds and the web needs side
  modules.

## Landed

The additions of reflection version 2 and the Python-mode debugging support are implemented
and described in [`design.md`](design.md#reflection-table):

- call-site reflection;
- record `struct_size`s with striding accessors;
- the `fini` slot;
- the JSON form of values;
- `HostModule.take_error`.

Host modules are Program assets, described in
[`design.md`](design.md#build-integration). A cooked object registers its table
with VernonRuntime under the asset id. Python mode registers its own table
under the same id, so application code finds a module by id in both modes. An asset may list several modules; they cook into one object and table.
The asset gains variants once Host code has specializations.

Call-site reflection is the type-granular first step toward the region
effects of Vernon `architecture.md` section 11. Tracing inside compiled code
comes after the Python-mode tools.

`list[T]`, `dict[K, V]` and `str` class fields, `len` of a `str`,
reflection version 3 with the allocator slot and generated `fini`, and
migration that moves containers are implemented and described in
[`design.md`](design.md#containers) and
[`language.md`](language.md#containers). `str` locals, `+` and f-strings
are implemented and described in [`design.md`](design.md#strings) and
[`language.md`](language.md#strings).

## Device resources

Host code drives device code with the same types and helpers as Python
code, so an orchestration written in Python (allocate, launch, chain
kernels, read back) moves to Host code unchanged.

- **Types.**
  - `TensorStorage` is owned like a container: it lives in a class field,
    or in a local that is freed when its function returns, and `fini`
    releases it.
  - `TensorView` is a value over a storage, with the shape, strides and
    offset the Python runtime gives it. It does not own its storage, so a
    view may live only in a local or in a field of the instance owning that
    storage. A view in another instance would outlive a reload or a `fini`.
- **Helpers.** The `vernon_dsl` storage helpers (`zeros`, `empty`, uploads
  from host Tensors and lists, downloads, view slicing and reshaping) and
  kernel and Program launches are Host built-ins with the Python signatures.
- **Python mode** passes the real runtime objects through, so a Host
  method and a Python script share storages.
- **Compiled code** calls a C ABI of the runtime over opaque storage
  handles, with views as plain structs. A module records the kernels and
  Programs it launches in reflection, and its AOT build embeds their
  compiled artifacts for the target.
- **Effects.** A launch has the `write` effect on the storages its kernel
  writes, and a download has `read`. Call-site reflection lists storage
  accesses like extern calls.
- **The application chooses the device**, as it supplies the allocator. A
  module receives the runtime context with its other bindings and never
  initializes a device itself.

This makes TensorView and device resources part of the Host ABI, which
[`../future/host_language.md`](../future/host_language.md#invocation-abi)
left out of its first version.

## Language growth order

1. Small and broadly needed (bitwise operators and shifts have landed, in
   device code too):
   - `IntEnum`, and `match` over enums and scalars;
   - `Optional[T]` for value types;
   - recursion, which only device rules forbid. It is rejected today by
     checks shared with device code (the module call graph, helper
     specialization and typed-effect resolution); a recursive `@host`
     function types against its declared result instead;
   - comprehensions over `range` and fixed Tensors, lowered to loops;
   - the [known gaps](language.md#known-gaps): `is`, Tensor `==`, Tuple
     indexing and nested `def`.
2. Owned containers and strings: `list`, `dict` and `str` fields, and
   `str` locals with f-strings, have landed.
3. Device resources, as above, on the ownership rules of the containers.
4. `Protocol` interfaces, lowered to function-pointer tables and listed in
   reflection. They give polymorphism without inheritance, so a class layout
   stays a C struct, and they serve Vernon's general traits (Vernon
   `architecture.md` section 11).
5. Errors: `raise HostError(code)` and `try`/`except` over statuses, on the
   existing status convention.

## v0.1.3 scope

Vernon uses the package release directly, so v0.1.3 fixes what it builds on:

- **Frozen ABI**:
  - the reflection table at `VH_REFLECT_VERSION` 3, with the allocator
    slot (landed);
  - the exported-function and status convention;
  - the generic-extern descriptor;
  - the migration rules;
  - the Host module manifest (`host_module_version` 1) and registration by
    asset id (landed).
- **Language**: group 1 of the growth order. Groups 2 to 5 follow as minor
  releases, extending the table through `struct_size` without breaking
  callers.
- **Acceptance**: a C program and a Python-mode run use one Host module
  through the same table, including call-site reflection and `fini`; a
  reload in Python mode keeps an instance's state through the table swap;
  every group 1 construct runs the parity table in both modes.
