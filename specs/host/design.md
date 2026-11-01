# Host language design

Status: architecture of the implemented Host language. What programs may
contain and mean is in [`language.md`](language.md). The motivation and the
longer-range plan are in [`../future/host_language.md`](../future/host_language.md).

The Host language is the CPU side of VernonDSL: typed Python source with a
fixed semantics. In development it runs as CPython code; for deployment it is
compiled to LLVM IR and emitted as a native or wasm32 object. It is general
purpose. Engines build their own concepts (systems, components, inspectors) on
its classes, metadata, reflection and generic externs; the language has no
engine-specific constructs.

## Pipeline

```text
source.py -> module graph -> class lowering -> typed Host model (frontend.py)
          -> the source itself runs as CPython (native.py)            development
          -> LLVM IR text (llvm_emitter.py, reflect.py)
             -> _native: O2 pipeline -> object (.obj/.o/.wasm.o) + header + JSON    deployment
```

- **Development runs the user's source as ordinary Python.** Nothing is
  generated or rewritten. The type check runs once per module, as a lint and
  as the source of the reflection table, and every call afterwards runs the
  user's own function objects. Breakpoints, stepping, locals, tracebacks,
  `pdb`, coverage and profilers therefore work as in any Python code, at the
  speed of CPython with numpy scalars.
- **Values in Python mode are numpy values.** `f32(x)` is `np.float32`,
  integers are numpy integers, and `Vector`, `Matrix` and `Tensor` are
  read-only ndarray subclasses. Handles are Python ints. Struct instances are
  the `@struct(shared=True)` Python classes.
- **Why not generate Python, and why no JIT.** An earlier backend generated
  Python that applied the compiled scalar rules operation by operation. It was
  exact, but the rewrite cost CPython speed and put generated code under the
  debugger. An ORC JIT ran deployed code at native speed, but needed a native
  compile of the whole import chain on every edit and lost Python's debugging.
  The tree-walking interpreter before both ran slower than CPython. Native
  Python keeps the development loop of an interpreted language. The price is
  a documented set of numeric differences, which are accepted.
- **Accepted differences from compiled code.** Python mode converts values to
  their declared types only at boundaries (parameters, results, field stores,
  extern calls), and follows numpy and Python between them. The resulting
  differences in integer and float width, conversions, NaN handling and error
  reporting are listed in [`language.md`](language.md#python-mode-differences);
  `parity_cases.PYTHON_DIFFERENCES` pins them in the tests.
- **Rules that make the source valid Python.** Structs and helpers must be
  declared `shared=True`, because Python constructs and calls them, and values
  are immutable, so sharing one is never observable. Mutation goes only
  through class instances, which are references in Python and pointers in
  compiled code. [`language.md`](language.md#values-and-places) states the
  rules.
- **One value model for Host and device code.** Tuples, Structs and Tensors
  are SSA values in device code, so they are values in Host code too, with
  no exception for class fields: an earlier rule let `self.history[i] = x`
  write a Tensor field in place, which made the same spelling legal or not
  depending on where the Tensor lived. Updates build new values instead,
  with JAX's `v.at[i].set(x)` and `vd.replace(p, f=x)`, which every mode
  and backend lowers; elements that change in place are a TensorStorage.
  Device lowering is described in
  [`../language/contract.md`](../language/contract.md#value-updates).
- **Selections are planned once, in `language/slicing.py`.** Inference, the
  emitter, device lowering and the Python runtime share `plan_selection`,
  which turns a subscript into one `AxisIndex`, `AxisRange` (constant
  positions) or `AxisWindow` (a dynamic start) per axis. Python mode sees
  only integers, so `runtime_key` rewrites each window `slice(a, a + n)`
  into the clamped (Tensor) or checked (TensorStorage) window before numpy
  indexes with it; numpy would otherwise shorten a window that runs off the
  end, which compiled code cannot do with a static shape. The emitter
  computes each selected element's row-major offset, a constant plus the
  clamped or checked window start, and gathers loads into a new Tensor or
  scatters stores from a scalar or a same-shaped Tensor. `.at` spills the
  receiver, scatters into the spilled copy and loads it back; a storage
  slice addresses the storage's elements directly.
- **Dispatch.** The outermost call into Host code (a `HostFunction`, a class
  method called from Python, or a C trampoline) activates a `HostRuntime` in
  a `ContextVar`. Calls inside Host code are plain Python calls, and each
  extern call dispatches to the active runtime's registry. `HostModule.load`
  binds every reachable extern before anything runs, so a missing binding
  fails before any effect. A class imported as an ordinary module runs with
  the default registry and binds each extern at its first call instead.
- **Instances are ctypes structures in the compiled layout.** A
  `@vd.host class` gets a `ctypes.Structure` named `<Class>State`. Fields are
  descriptors over it:
  - scalars read as numpy scalars;
  - Tensor fields read as read-only copies;
  - Struct fields read as fresh values;
  - class fields read as views of the parent's memory.

  `check_layout` fails if ctypes lays a class out differently from the
  compiler. `address_of(instance)` is the pointer C callers receive.
- **C callers see one table in both modes.** In Python mode, the reflection
  table is built with ctypes from the same description the LLVM constants are
  emitted from (`reflect.describe`). Its method and init addresses are
  `CFUNCTYPE` trampolines into the Python methods, so a C++ application calls
  Python-mode code exactly as it calls the AOT object. An exception other
  than a Host status cannot unwind into C, so the trampoline records it in
  `errors` and returns 1. `HostModule.take_error()` hands the latest one to
  the application reporting that status, which chains it onto its own error
  so the traceback still points at the Python line that raised.
- Every exported function uses one convention,
  `int32_t <prefix><name>(params..., Result *out)`, and returns zero or a
  status. Struct parameters are passed by pointer, and handles as their
  `uint64_t` bits.

## Build integration

- `python -m vernon_dsl.hostlang compile <file.py>...` compiles whole modules
  (every `@host` function and class, and the `<prefix>module` table). Each
  `--entry` restricts a single file to named entries. `include-dir` prints the directory
  of `vernon_host/reflect.h`.
- **Host modules are Program assets.** `vd.program_asset(id=..., program=
  <imported @host module or tuple of them>, prefix=...)` declares one, and
  the default prefix is the id with non-identifier characters replaced by
  `_`. A tuple cooks into one object and one table named by the id, as an
  application's systems, components and plugins share one module. Applications then
  keep one asset pipeline, with ids, content hashes and build rules, for
  device Programs and Host code. That matters because Host code will launch
  device Programs, so a plugin ships both as assets. Host assets take no
  variants until Host code has specializations.
- **Several modules, one object.** The modules share the C symbol prefix, so
  `check_combinable` rejects an exported name or class defined by two of
  them, and one Struct name with two layouts. Host classes, like Structs and
  handles, keep their declared names when imported (`module_graph`), so a
  class one module imports from another is one table type with one set of
  methods. Every other function is private to its program. The frontend
  names root helpers bare and imported ones by their module path, so two
  programs can emit the same name with different bodies. The emitter
  therefore names public functions (exports, initializers, methods)
  `vh.<symbol>` and all others `vh.<program index>.<symbol>`. In Python
  mode the sources run in order, and while a later one runs, importing an
  earlier one by name finds the module this `HostModule` ran, so both
  modes bind the same class objects.
- **Packages ship Host sources, and the deployable unit compiles them
  together.** A Host package is a Python package, so its wheel already holds
  its `.py` sources. An application or Program asset compiles everything it
  reaches from other packages into its own object, found along `sys.path` or
  through an editable install's finder. Packages are named by import name,
  so an editable checkout and an installed wheel compile alike. Whole-program
  compilation is chosen over objects per library:
  - generic functions specialize for the importer's types;
  - calls between packages inline;
  - no binary interface between packages needs to stay stable;
  - one object fits every target.

  Objects per library would also put a private copy of a shared class's
  functions in each library's object, fragile once instances cross between
  them. A boundary between separately built units (the engine, binary
  plugins) is a `vd.library` of externs instead (bindgen.md). Host classes,
  handles and enums keep bare names in the object, so two packages' Host
  classes of one name cannot share an object. In Python mode a package
  module runs in its package (`__package__` set, its import root
  importable), so its imports bind the classes an ordinary import does.
- **Cooking** (`_program_assets/host_cooking.py`) targets `cpu` only, since
  the object is native code for one triple. The output directory holds:
  - `<dir>.program.json`, with `"type": "host_module"`, the id, the target,
    the content hash, and `host_module` (table symbol, prefix, header,
    reflection);
  - the object, content-addressed under `artifacts/` and listed in `blobs`
    like a CPU Program's, so `vernon_add_program_asset` archives it and links
    it whole-archive unchanged;
  - `<dir>.h` and `<dir>.json`. For CPU assets, `vernon_add_program_asset`
    adds the asset directory and the `reflect.h` directory to consumers'
    include paths, and `HEADER_VARIABLE` names the header.

  `vd.load_program` rejects a Host manifest, because there is nothing to load
  at run time. The code is linked.
- **Registration by id.** The cooked object's static constructor passes
  `{id, &<prefix>module}` to `vernonRuntimeRegisterStaticHostModule`, and its
  destructor withdraws it. The record goes by pointer, the one form every C
  ABI passes alike. Consumers link VernonRuntime and find the table with
  `vernonRuntimeFindHostModule(id)`. A lookup returns the newest live
  registration. In Python mode, `HostModule.from_asset(asset)` loads the
  sources with the asset's prefix and names the table by its id, and the application registers that table
  under `asset_id`. The development table then shadows the linked one until
  it is withdrawn, and a reload can register its new table the same way.
  Application code looks the module up by id in both modes.
- `HostModule.load(..., prefix=...)` names Python-mode symbols as an AOT build
  with the same prefix does, so both modes present identical tables. The
  plain `hostlang compile` command stays as the runtime-free tool: its object
  registers nothing.

## Classes

`@vd.host class` declares a class with mutable state: fields with optional
defaults, and methods taking a mutable `self`.

- **Lowering** (`classes.py`). A class becomes a `@struct` layout marked
  `HostStruct.place`.
  - Each method is hoisted to a function `Class__method(self: Class, ...)`.
  - A synthesized `Class____init__` zeroes the instance, writes the field
    defaults, and initializes class-typed fields recursively. Zeroed memory
    is an empty list or `str`.
  - Calls such as `self.child.step()` resolve through the same typed method
    table as struct methods.
- **Methods.** Undecorated methods are `effects="io"`; `@vd.host(effects=...)`
  narrows one method. A leading underscore makes a method private: it is
  callable inside the module, but not exported.
- **Places.** An instance is a place, not a value: a class parameter is the
  instance pointer, and field and element accesses rooted at it compile to
  address arithmetic with loads and stores. `+=` must keep the field's type.
- **Ownership** (`ownership.py`, run after typing). Python shares an instance
  where C would copy it, so the checker admits only programs in which no
  instance is reachable through two owners, which makes the two
  indistinguishable without copying anything. The model is Rust's, without
  lifetimes in the syntax:
  - Parameters, locals bound to a place and loop elements borrow;
    constructors, calls returning a class and pops create; owned locals hold
    a created instance until it moves. A move is a `memcpy` and a zeroed
    source, so releasing every owned slot at `vh.exit` needs no drop flags;
    `_MoveFlow`, a forward dataflow over statements with a loop fixpoint,
    rejects a use that some path reaches after a move.
  - Borrows are paths: a root, then field names, with `*` for any element.
    Restructuring a path invalidates the borrows it is a prefix of: whole
    assignment of a class field or element, `append`, `pop`, `clear`, `del`,
    and rebinding an owned or container local. A summary records, per
    function, the parameters it may restructure through, transitively; it
    is a fixpoint over the call graph, since `@host` functions may recurse.
  - A borrowed local is bound once, rooted at a parameter and reached through
    fields only. An element moves whenever its container grows, so naming
    one outside a loop over the container would need every later operation
    proved harmless; indexing it at each use stays simple and safe.
- **Results and temporaries.** A class result is built in the caller's out
  pointer, which holds no instance. A created instance passed as an argument
  lives in a zeroed temporary slot the caller releases after the call, so
  `weigh(make_item(3), 1)` frees what `make_item` allocated.
- **Containers of classes.** Elements are stored inline, like values. The
  runtime's list and dict functions move bytes and release only str keys, so
  the emitter releases elements: `vh.release_elements.<type>` runs each
  element's fini before `clear`, `[]`, `{}` and the container's own fini, and
  `del` and stores release the one value they replace. Python mode's
  `HostList` and `HostDict` do the same over the same records, and move an
  instance in by copying its state and zeroing the Python object's.
- **Recursive types.** A class may hold a list or dict of itself; a class
  inline in itself would have no finite size and stays rejected. The layout
  hash spells a class met inside its own spelling by name, the header
  declares such a class ahead (`typedef struct Node Node;`), and the C++
  `vh::List<T>` checks `T` only in the operations that copy elements, so the
  class body can name its own incomplete type.
- **Why these rules.** Value semantics for classes would need copies of every
  nested class and hidden temporaries in the ABI. Place semantics keep a
  method a plain C function over a pointer the application allocated. That is
  what an engine stores in its own tables and hands to a scheduler; moves and
  borrows add owned instances without changing that ABI.
- **No inheritance, dunder methods or `self` annotations.** Composition
  through class fields covers reuse without vtables, so the layout stays a C
  struct.

## Metadata and docstrings

- `vd.meta(**entries)` attaches metadata in six places:
  - classes: `@vd.host(meta=...)`;
  - Structs: `@vd.struct(meta=...)`;
  - fields: `Annotated[T, vd.meta(...)]`;
  - handles: `@vd.handle(meta=...)`, on the handle's `VhType`;
  - enums: `@vd.enum(meta=...)`, on the enum's `VhType`;
  - externs: `@vd.extern(meta=...)`, on every `VhCall` reaching the extern
    and in the AOT reflection JSON. Binding generators keep their C++
    mapping here ([bindgen.md](bindgen.md)).
- A string literal directly after a field is that field's docstring.
- Metadata values must be compile-time constants (numbers, strings, booleans,
  None, and tuples of them). They are stored as JSON text, so the language
  fixes no key vocabulary: an engine interprets keys such as `min`, `unit` or
  `role`.
- Everything a tool knows about a field beyond its name, type, offset and
  docstring is metadata its author wrote, as with Unreal's `UPROPERTY`
  specifiers. The language reflects no defaults, ranges or UI hints of its
  own.

## Reflection table

Every module emits `const VhModule <prefix>module`, which mirrors
`include/vernon_host/reflect.h` (`VH_REFLECT_VERSION` 3).
`reflect.include_dir()` locates the header. `HostModule.table` is the table
built with ctypes in Python mode; an application linking the object reads the
exported constant.

- **Contents.** It lists every named type a module reaches:
  - its kind, size and alignment;
  - its fields (name, type, offset, doc, metadata);
  - for a class, its methods (exported symbol, or null if private; address;
    parameters; result; declared effect; calls) and its `init`;
  - for a type owning memory (a list, a dict, a `str`, or a class holding
    one), its `fini`;
  - for a dict (kind 15), its value type in `element` and its key type in
    `key`, a member appended to `VhType`;
  - for a TensorStorage (kind 16), its element type, rank, and static
    extents in `shape`, with 0 for `dyn`;
  - for an enum (kind 17), `i32` in `element` and its members in
    declaration order as `VhEnumerator {name, int64_t value}` records in
    `enumerators`, appended members of `VhType`; `vh_enum_name` finds a
    value's first member;
  - plus the module's free functions, and the module's allocator slot.
- **Call-site reflection.** Each function lists the externs it can reach,
  through any chain of helpers and methods, as `VhCall` entries: the C
  symbol, the extern's declared effect, for a generic extern the Struct
  descriptor that call passes, and the extern's metadata (`meta` and
  `meta_count`, appended members). The bound type at a call site is static,
  so the list is exact. An application derives type-granular access from
  it, such as the components a system reads and writes, instead of trusting
  a hand-written list.
- **Growth.** The module, type, field, function and call records start with
  their `struct_size`, and later versions only append members. C code indexes
  record arrays through `vh_field_at`, `vh_method_at`, `vh_function_at` and
  `vh_call_at`, which stride by the table's size, so code built against
  version 2 keeps reading newer tables. The sizes are LLVM constant
  expressions (`ptrtoint` of a `getelementptr` past a null record), so they
  are the target's sizes, wasm32 included.
- **JSON form.** `value_json.to_json` and `from_json` read and write any
  described value at an address; `HostModule.to_json(instance)` and
  `from_json(instance, edit)` apply them to class instances.
  - Structs and classes are objects in field order; Tuples, Tensors and
    TensorStorages (nested by shape, as numpy's `tolist`) and lists are
    lists; `str`s are strings; handles are their
    bits; enums are their first matching member's name, or the number when
    no member has the value, and read back from either; non-finite floats
    are `"nan"`, `"inf"` and `"-inf"`.
  - A dict is an object in insertion order. JSON keys are strings, so a
    `str` key or an enum member's name is the key itself, and any other key
    is written as its own JSON text: `{"7": 1.5}`, `{"true": 2}`,
    `{"RUN": 3}`.
  - An edit may name only some fields, and every check runs before any write,
    so a rejected edit changes nothing. A list, dict, `str` or TensorStorage
    is replaced whole, growing through the module's allocator; a storage's
    nested lists must be rectangular and match its static extents.
  - `vh_write_json` writes the same form from C, with snprintf's sizing
    contract, for C++ inspectors and error reports.
- **Named type symbols.** Named types are emitted as `@"vh.type.<kind>.<Name>"`
  constants. Generic externs pass these same addresses as type descriptors.
- **Layout hashes.** A layout hash is FNV-1a 64 over a canonical layout string
  that includes field names, so a rename changes the hash. The module hash
  combines every type layout and function signature.
- **Why a table in the object instead of only JSON.** A deployed binary on a
  phone or in a browser has no files beside it, and C++ cannot reflect on
  itself. An in-object table gives a C++ inspector, a serializer or a
  scheduler the same view in development and in AOT builds. The JSON beside AOT
  objects is for build tools.

## Containers

`list[T]`, `dict[K, V]` and `str` class fields own memory; the instance
frees it.

- **Record.** Both are a `VhList` record, `{uint64_t data, size, capacity}`:
  24 bytes, aligned to 8, on every target. `data` is a `uint64_t` rather
  than a pointer, so the record and every class holding one have the same
  layout and layout hash on wasm32 as on 64-bit targets. A `str` holds UTF-8
  bytes, and `len` counts its code points (the emitted `vh.str_len`) to
  match Python.
- **Dict record.** A dict is a `VhDict`,
  `{uint64_t entries, size, used, capacity, slots, slot_count}`: 48 bytes,
  aligned to 8.
  - `entries` is an array of `{uint64_t hash; K key; V value}` in C layout,
    in insertion order. `used` counts entries written, and a removed entry
    keeps its place with hash `VH_DICT_REMOVED` (`UINT64_MAX`).
  - `slots` is an open-addressing table with linear probing and a
    power-of-two `slot_count` of at least twice the capacity, and at least
    8. A slot is 0 when empty, `index + 1` for an entry, or `UINT64_MAX`
    for a removed one.
  - Keys hash as FNV-1a over their bytes, then the murmur3 finalizer, so
    nearby integers and handles spread across slots. A `str` key hashes its
    UTF-8 bytes; its entry holds an owned `VhList`, and lookups pass it as
    `{uint64_t data, size}` without copying.
  - An insert into a full entry array rebuilds it: if live entries would
    exceed half the capacity, the capacity doubles (at least 4); otherwise
    it is kept and the rebuild only drops removed entries. Removing the last
    entry resets `used` to 0.
- **Why this dict.** It is CPython's compact dict: insertion order makes
  iteration, JSON and migration deterministic and equal in both modes, and
  the dense entry array iterates without scanning empty slots. Floats are
  not keys because NaN and `-0.0` have no single equality, which would make
  bytewise matching differ from Python.
- **Dict runtime.** `vernonRuntimeHostDictFind`, `Insert`, `Remove`,
  `Clear` and `Release` take the record, a `VernonHostDictLayout` (entry
  size and alignment, key offset and size, value offset, key kind) and the
  allocator. The emitter passes each dict type's layout as a private
  constant, so one runtime serves every `K` and `V`. `Insert` returns the
  entry, new entries with a zeroed value, so `d[k] = v` and `d[k] += v` are
  one call.
- **Allocation.** Growing calls `vernonRuntimeHostListReserve`, which at
  least doubles the capacity, and freeing calls
  `vernonRuntimeHostListRelease`. Both take the `VhAllocator` in the
  module's allocator slot, which `vh_set_allocator` writes and which is
  null for VernonRuntime's default (`malloc`). An engine or a wasm host thus
  owns container memory. The functions live in VernonRuntime rather than in
  each object, so a module and the Python-mode runtime grow and free the
  same records the same way. A module using no container references
  neither.
- **fini.** The emitter generates `vh.fini.<type>` for each type owning
  memory: a list, dict or `str` releases its record (a dict also its `str`
  keys), and a class releases its
  container fields, recursively through class fields. Each leaves the
  records zeroed, so a released instance reads as empty. The application
  calls `fini` before freeing an instance it allocated.
- **Copies.** `vh.copy.<type>(target, source) -> status` deep-copies into
  zeroed memory, beside `fini`. A list reserves the source's size, then
  memcpys elements that own nothing or copies each in turn, counting it in
  the target's size first. A dict inserts each live key in insertion order
  (a `str` key's record starts with the `{data, size}` view `Insert` takes)
  and copies the value into the entry `Insert` zeroed. A class memcpys its
  bytes, then zeroes and copies each field that owns memory. A failure
  returns the status with a partial copy that `fini` frees, so the
  assignment copies into a temporary record and only then releases the
  target and moves the copy in. Python mode's `HostList`, `HostDict` and
  class `__deepcopy__` call the same steps through ctypes (`copy_at` on
  each codec), so `copy.deepcopy` of a Python-mode instance is the
  compiled copy. Python mode's list and dict field setters likewise fill a
  record of their own before releasing the field's.
- **Evaluation order.** `append(v)` evaluates `v` before growing, and an
  index or key is evaluated before the size and data are read, because
  evaluating either may change the container. An augmented assignment
  evaluates its target's indices once and reuses them for the load and the
  store.
- **Iteration** follows Python's list iterator: each step compares the
  position with the current size and rereads the data pointer, so a body
  that appends visits the new items and one that pops ends early. This
  needs no rule against mutation during iteration and gives the same result
  in both modes. A dict loop instead fails with `CONTAINER_CHANGED` when the
  size differs from the size at the loop's start, as Python's dict iterator
  does; an insert may grow and move the entries, so continuing has no
  Python-equal meaning.
- **C++ access.** Generated headers spell container fields `VH_STR`,
  `VH_LIST(T)` and `VH_DICT(K, V)`. `reflect.h` defines them as the
  `VhList` and `VhDict` records in C. In C++ the header includes
  `vernon_host/containers.hpp`, which defines them as `vh::Str`,
  `vh::List<T>` and `vh::Dict<K, V>`. These have the records' layout and
  offer `std::string_view`, range-for (dicts yield `{key, value}` in
  insertion order), `find`, `push_back`, `assign`, `erase` and `clear`.
  - Each class's dict layout comes from `struct {uint64_t hash; K key;
    V value;}` at compile time, so C++ needs no layout constant.
  - Reads take no allocator. Writes take a `vh::Allocator`, normally built
    from the module (`vh_get_allocator`), because memory must return to the
    allocator it came from.
  - The classes call the same VernonRuntime functions as compiled code, so
    an engine edits an instance's containers between Host calls with no
    separate format.
  - They cannot be copied, since two owners would free the memory twice.
  - A header includes `containers.hpp` only when the module has container
    fields, so a C++ consumer of a module without them needs no VernonRuntime
    headers.
- **Why not the C++ standard containers.** `std::vector`, `std::string` and
  `std::unordered_map` cannot be the field type, for these reasons:
  - Their layouts differ between standard libraries and build modes, but an
    instance must have one layout and layout hash on every target and in
    Python mode, which reads it through ctypes.
  - Compiled Host code operates on fields inline, from IR that cannot
    instantiate templates.
  - Migration moves containers bytewise, which libstdc++'s self-referencing
    short `std::string` does not survive.
  - The allocator is chosen at run time, while a standard allocator is part
    of the type.
  - `std::unordered_map` has no defined iteration order.

  The vh:: classes give C++ code the standard interface over a fixed record
  instead.
- **Python mode** uses the same memory. A list field reads as a `HostList`
  view over the record that calls the VernonRuntime functions through
  ctypes, with the allocator of the class's runtime, a dict field reads as a
  `HostDict` view over the same functions, and a `str` field reads
  as a Python `str` copy. An instance Python creates gets a
  `weakref.finalize` that releases its containers. The finalizer holds the
  instance's state rather than the instance, so a view that outlives it reads
  zeroed, empty records. An instance C creates through the table is freed
  by C, through the table's `fini` trampoline.
- **Fields and locals, never passed.** Containers as parameters or results
  would need moves or copies at every boundary. A container is owned by
  exactly one instance or one function call, and freed by it, so there is
  no reference counting:
  - A field is freed by the instance's `fini`.
  - A local is one of the function's owned records, as `str` and
    TensorStorage locals are: zeroed at entry, released in the `vh.exit`
    block every return and failing status goes through. Its name is the
    record's address, as `self.d` is for a field, so every container
    operation takes either.
  - Neither escapes: a container is used only in place, never assigned to
    another name, passed or returned. Python would alias where C would copy
    the record.
- **Locals are Python containers in Python mode.** Host code runs as
  written, so `counts: dict[K, V] = {}` makes a Python `dict`, and Python
  mode does not rewrite it into a `HostDict` over allocated memory. That
  keeps a local a plain value in a debugger. It fits the existing rule
  that between boundaries locals hold whatever Python computes. The cost is
  that a local's failures are Python's (`KeyError`, `IndexError`,
  `RuntimeError`) where a field's are `HostExecutionError` statuses.
  Rebinding a local that a loop iterates is rejected, because that is the
  one use where a new Python container and a cleared C record differ.

## Tensor storages

- **Record.** A `TensorStorage[T, (e0, ...)]` of rank r is
  `{uint64_t data; uint64_t shape[r]}`: 8 × (1 + r) bytes, aligned to 8, on
  every target. The elements are dense and row-major, aligned to their
  size, and allocated from the module's `VhAllocator`. A storage with no
  elements has null `data`. Static extents are checked whenever a storage
  is created or written, not stored separately.
- **Why not `VernonTensorView` as the record.** A view is a borrowed
  description of memory with a pointer-sized layout, its own `shape` and
  `byte_strides` pointers, an access mode and a residency. None of these
  belong in an instance that must have one layout and layout hash on
  wasm32, 64-bit targets and Python mode, and that migration moves
  bytewise. The record owns, the view borrows: `vernon_host/tensor_view.h`
  builds a host-resident `VernonTensorView` over a record
  (`vh_storage_tensor_view`), and `vh::TensorStorage::view()` wraps it, so a
  storage reaches every API that takes views without a copy. It is a
  separate header so `reflect.h` keeps depending only on the C library.
- **Runtime.** `vernonRuntimeHostStorageAllocate`, `Release` and `Fill`
  take the record, the rank, the element size and alignment, and the
  allocator. `Allocate` rejects a negative extent or a rank above 8 with
  `INVALID_SHAPE` and an overflowing byte size with `OUT_OF_MEMORY`, and
  zero-fills. Compiled code and
  `vh::TensorStorage` allocate into a temporary record, then release the
  target and move the temporary in, so a failure leaves the target intact.
  `empty` and `empty_like` zero-fill as well, so both modes read the same
  elements.
- **Ownership.** Fields are freed by the class's `fini`. A local is an
  owned record in the frame, zeroed at entry and released at `vh.exit`, as
  `str` locals are. A storage is never copied by assignment (only created
  into), passed, or returned, so it has exactly one owner and needs no
  reference count. Fields whose extents are all static, without a default,
  are allocated zeroed by `init`, so their indexing never meets a missing
  shape.
- **Indexing.** Indices are evaluated before the record is read; negative
  indices wrap once, then each is checked against its extent, so
  `s[i, j]` is `INDEX_OUT_OF_RANGE` exactly where numpy raises
  `IndexError`. Tensor values index the same way (`flat_offset` in the
  emitter), so the two kinds of tensor never disagree on an index.
- **Values in and out.** `vd.from_values(t, dtype=T)` and
  `copy_from_values` copy a spilled Tensor into the elements with one
  `memcpy`; `to_values` loads them back as a Tensor. They are named for
  values rather than numpy because in compiled code the other side is a
  Tensor SSA value; only in Python mode is it a numpy array.
- **Element types** are `bool`, `i32`, `u32`, `f32` and `f64`: the scalars
  that `VernonDataType` and the Python-mode `TensorStorage` have, less
  `f16`, which is not a Host value. Neither has 64-bit integers, so a view
  of an `i64` storage could not be described.
- **C++ access.** Generated headers spell a storage `VH_STORAGE(T, rank)`.
  In C it is the anonymous record, read with `vh_storage_data`,
  `vh_storage_shape` and `vh_storage_count`; in C++ it is
  `vh::TensorStorage<T, rank>`, which offers unchecked `s(i, j)`, checked
  `get({i, j})`, `shape`, `size`, `fill`, `allocate`, `release` and
  `view`.
- **Python mode** reads a field as a real `vd.TensorStorage` whose array is
  `np.frombuffer` over the record's elements, so writes through it and
  compiled code see each other. The object is cached by record address and
  reused while the record's data pointer and shape are unchanged. Replacing
  or releasing the record detaches the cached object, which copies its
  array, so a Python reference never reads freed memory.

## Strings

- **Owned records.** Each `str` local, and each f-string or `+` expression
  site, is a `VhList` record in the function's frame, zeroed at entry. A
  site reuses its record across loop iterations, truncating it before
  writing, so a loop formats without allocating once the record is large
  enough.
- **Cleanup.** A function owning `str` records stores its status in a slot
  and branches to one `vh.exit` block from every return, failed check and
  propagated extern status. `vh.exit` releases every record and returns the
  status. A record never written is empty, so releasing it is free.
- **Assignment** builds the value first and then copies it into the
  target, reserving only if the target must grow. `s = s + t` and `s += t`
  thus never read a target they have begun to overwrite.
- **Field operands.** A `str` field is borrowed, not copied, where nothing
  can change it before its use. In `==`/`!=` and extern arguments, a later
  operand that makes a call could assign the field, so the field is first
  copied into an owned record, as Python would hold the old object. A dict
  loop copies each `str` key into its local, because the body may delete
  the entry.
- **Formatter.** `vernonRuntimeHostStrAppend`, `AppendInteger` and
  `AppendFloat` append one value to a record under a
  `VernonHostFormatSpec {fill, align, sign, type, width, precision}`; the
  emitter passes each distinct spec as a private constant, and null for
  none. The text appended may lie in the record itself, so `s += s` works.
  An integer arrives as 64 bits with a signed flag; an `f32` is widened to
  `double` first, which is Python's own `float(np.float32(x))`.
- **Float repr.** A float without a spec is Python's `repr`: the shortest
  digit string that reads back as the same `double`, found by trying
  `%.{p}e` for p = 0..16 and checking with `strtod`. It is written
  positionally when the decimal exponent is in [-4, 16), and in exponent
  form otherwise, as CPython's `float_repr_style = 'short'` does. Formatting
  through the C library keeps the runtime free of a Ryu or Grisu table, at
  the cost of at most 17 `snprintf` calls per float, which logging can
  afford.
- **Why this subset.** The spec subset is what logs and diagnostics use and
  what both modes format identically. `!r` would need Python's quoting
  rules, `=` the expression's source text, the thousands separators locale
  rules, and nested specs a run-time parser; each can be added without
  changing the record or the runtime calls.

## Generic externs

- **Declaration.** An extern parameter `type: type[T]` (with
  `T = TypeVar("T")`) makes the extern generic over Structs. Other parameters
  and the result may mention `T`. Without one, the first parameter annotated
  `T` binds it: Host code passes a Struct value there, as in
  `write(entity, Health(...))`, and its static type is the bound Struct.
- **Call sites.** Each call site's Struct specializes the extern
  (`name__Struct`) for type checking: the type argument names it, or
  inference types the value. A value-bound extern is declared for every
  Struct of the program before inference and the uncalled specializations
  are dropped after, because only inference knows a value's type. The C
  symbol is shared: the descriptor `const VhType *` is inserted at the
  type parameter's position, or just before the binding value, and every
  `T`-typed value is passed by pointer. Both forms therefore have the same C
  signature, and a library can switch a setter between them without
  changing its C API.
- **Which form.** A function with a `T` value to pass takes the value form,
  so a call does not name its Struct twice; a getter, `has` or `remove`,
  which has no value, takes `type[T]`.
- **Module graph.** A combined module drops its modules' assignments, so
  each extern keeps the names its own module binds to `TypeVar`; an
  importer's names never make an extern generic.
- **Python bindings.** A binding serves every specialization. It receives the
  C parameters, the descriptor included, and receives and returns `T` values
  as field dicts; in the value form Python mode takes the descriptor from
  the value's class.
- **Why.** An engine exposes one `read`/`write`/`has` entry point over
  component storage it never compiled against. That storage is keyed by the
  descriptor and copies `descriptor->size` bytes, which is how script-declared
  components reach a C++ engine without generated C++ per type.

## Spans and owned results

The rules are in language.md (Externs). They let a library take and return
variable-size data, such as a component's name or its child list, without
any engine-specific declarations.

- **Spans borrow.** A list argument passes the field's own data pointer and
  size. A dict argument is laid out as a temporary list of `{K key; V value;}`
  entries, which the function owns and releases at exit. Elements are plain
  values, so C reads them as bytes. Python passes the container object
  itself, and compiled code reads it when the argument is reached. A later
  argument that calls a Host function, method or extern could change or
  rebind the container in between, so the validator rejects one.
  Copying would not help: a `str` field is copied because Python strs are
  immutable, but a copied list would miss changes Python sees.
- **The callee fills an empty record.** C APIs often have callers pass a
  buffer and retry when `BUFFER_TOO_SMALL` comes back. That calls the
  function twice, so an `io` extern's effect would happen twice, and a
  result can change between the calls. Instead the extern receives a zeroed
  `VhList` and the allocator to fill it with, and is called once. Every
  result is a list, so one record type and one release path serve `str`,
  list and dict.
- **A dict result is a list of entries.** C has no hash table to hand back,
  and building one would tie the library to VernonRuntime's layout. The
  caller inserts the entries in order into a new dict, so a repeated key
  keeps its last value, as in a Python dict literal.
- **Received like a copy.** The call is only ever the value of an
  assignment. The result is built in a record of the function's own, then
  the target is released and the result moved in, as `assign_copy` does. A
  failed call leaves the target unchanged, and the function's exit releases
  whatever the extern left in the record.
- **Never a NULL allocator.** VernonRuntime's NULL allocator means its own
  `malloc` and `free`. Shipped builds link the Host object, VernonRuntime
  and the libraries statically (iOS, web, packaged desktop), so there is one
  heap. During development, Python mode loads `vernon_dsl._native` and each
  library's nanobind module as separate shared libraries. On Windows each
  can link its own C runtime, so a library's `malloc` and VernonRuntime's
  `free` would use different heaps. Callers therefore pass the module's
  allocator or `vernonRuntimeHostDefaultAllocator()`, a plain function of
  the static runtime. That works the same in both setups and costs one
  indirect call per result in a static build.

## Enums

- **Declared as `IntEnum`.** A `@vd.enum` class is a standard
  `enum.IntEnum`, so the same class is the Python-mode value, a binding
  generator's input and an ordinary Python enum to tools that never heard
  of Host code.
- **Always `int32_t`.** C leaves the size of an `enum` type to the compiler,
  so the header spells the type `typedef int32_t Name;` and the members as
  an anonymous `enum` of constants; Struct layouts and the call ABI never
  depend on the C compiler. One fixed underlying type keeps the layout,
  reflection and JSON rules single; a library with wider values binds them
  as integers.
- **Equality only.** Ordering, arithmetic, comparing with an integer and
  `Mode(x)` are rejected, so every use of an enum names its constants and
  a renumbering cannot change what Host code means. `i32(mode)` is the
  explicit way out.
- **Unknown values survive.** C code may pass a value no member has. Python
  mode then holds a plain `int` instead of raising, and JSON writes the
  number, so a value crosses both modes and every tool unchanged.
- **Names across modules.** An enum keeps its declared name in the module
  graph, as Structs and handles do, because reflection and the header name
  it.

## Libraries

- **Default symbols.** `vd.library(name, version)` gives the module's externs
  declared without `symbol=` the symbol `<name>_v<version>_<function>`, so
  libraries do not collide and a version change renames every symbol.
- **One rule in both modes.** The module graph applies it while emitting the
  declaring module's externs, by their declared names, so an importing
  module's prefix never reaches the symbol. In Python mode `vd.library`
  records itself in the calling module's globals, where `@vd.extern` reads
  it. Both reject a `vd.library` after an extern, the one order in which
  they would disagree.
- **Schema identity.** A library's identity is `"<name>-v<version>:<hash>"`
  (`hostlang.library_identity`), and it is the schema of the module's
  externs that do not declare `schema=`, so the existing check rejects a
  binding from another revision: at load for a compiled module or an eager
  runtime, at the first call otherwise.
  - The hash covers the module's Struct and handle layouts, its enums'
    members and values, and each extern's
    symbol, effect, ABI and canonical parameter and result layouts. A
    generic extern counts as its signature as written, since only call
    sites type it.
  - Metadata is not covered: it changes how a generator reaches C++, not
    the C functions a binding calls.
  - It is computed by loading the library module alone, with no `@host`
    code required, and cached by the digests of the files it was computed
    from. Python mode's `ExternFunction.schema` computes it on first use.
- **`ExternRegistry.bind_module(module)`** binds a generated module's
  `vh_symbols()` address table under its `vh_schema`, and its
  `vh_last_error()` when it has one: Python mode then adds the library's
  message to the `HostExecutionError` of a failing status extern.

### Level 1 bindings (`vernon_dsl.bindgen`)

`python -m vernon_dsl.bindgen generate|check <module.py> [--output-dir DIR]
[--pyi PATH]` reads a `vd.library` module with `meta=vd.meta(cpp_namespace=...)` and emits
`<name>_api.h`, `<name>_api_impl.h`, `<name>_api_thunks.cpp`,
`<name>_api_nanobind.cpp` and `<name>_api.pyi`. `check` exits 1 when a
committed output differs. The rest of the proposal is in
[`bindgen.md`](bindgen.md).

- **One C function per extern, in both modes.** The thunks are the C
  functions: the nanobind module publishes their addresses for Python mode,
  and AOT objects link them. The nanobind module's Python functions call the
  same C functions, for scripts that use the library without Host code.
- **The C header** spells types as AOT headers do (`aot.type_declarations`,
  with the camel-cased library name as type prefix) and declares
  `<name>_v<version>_last_error(buffer, capacity)` and
  `<name>_v<version>_schema()`, the identity also defined as
  `<NAME>_V<VERSION>_SCHEMA`. A `str` argument is `VhStrArg` (`reflect.h`).
- **Type descriptors.** `<name>_v<version>_types(&count)` lists `VhType`
  descriptors of the library's Structs, handles and enums, which the thunks
  define as constants from the records `reflect.describe` builds for Host
  objects. A Struct's layout hash is therefore the one Host code passes to
  a generic extern, and a library can read field names, offsets and
  metadata from them, as for scene files. They describe layout only: no
  `fini`, `init` or methods.
- **The C++ side** (`vernon_host/bindgen.hpp`): status externs return
  `vh::Result<T>` or `vh::Status`; direct externs return the value and must
  not throw. Structs arrive as `const T &`, Tensors as `const T (&)[N]`,
  `str` as `std::string_view`, enums and handles without `cpp=` as their C
  types.
- **Value handles**, with `cpp=` and `convert=<name>`: the bits are a C++
  value, such as an entity id, which the library converts with
  `Cpp <name>(C)` and `C <name>(const Cpp &)` in `cpp_namespace`. No table
  holds them; the implementation receives and returns the C++ value.
- **Handles with `cpp=`** live in a `vh::HandleTable<T>` the thunks own:
  index and generation in the 64 bits, generations from 1, so the zero
  handle and a destroyed handle's bits never find an object. A creating
  extern's implementation returns `vh::Result<std::unique_ptr<T>>`; an
  extern with `destroys="<parameter>"` takes the object out of the table
  before calling the implementation, which receives the `std::unique_ptr`,
  so the handle is gone even when the call fails. Taking or returning such a
  handle requires `abi="status"`, since a stale handle must fail. Its
  parameter is `const T &` for `pure` and `read` externs and `T &` otherwise.
- **Dependent handles**, with `depends_on="<Handle>"`: each extern creating
  one takes exactly one parent of that handle, whose bits the table records.
  Destroying the parent first takes every object created with it and
  destroys each, its own dependents first, through its destroying extern's
  implementation, which must take only the handle. The cascade runs after
  the parent's handle is gone, so no new dependent is created with it, and a
  failure does not stop the others: the first is the result. This is how an
  object that refers to another (a viewport to its world) cannot outlive it.
- **Statuses.** Thunks report `vh::kFailed` (1), `vh::kInvalidHandle` (2)
  and `vh::kOutOfMemory` (3) themselves, and a thrown C++ exception becomes
  one of these, because C callers cannot unwind it. An implementation's
  status of zero or below is reported as `kFailed`, so positive statuses
  stay the library's and negative ones Host code's. The message goes to a
  thread-local last error.
- **Spans and owned results**, in C++17 types (no `std::span`). A list
  parameter is `vh::Span<T>` and a dict parameter `vh::Span<vh::Entry<K,
  V>>`, borrowed for the call. An owned result is `vh::Result<std::string>`
  or `vh::Result<std::vector<T>>` (a dict as a vector of entries, in
  order); the thunk copies it into `*out` through the caller's allocator
  with `vh::assign_result`, and fails with `vh::kOutOfMemory` when that
  allocator cannot hold it. The nanobind module passes its own malloc
  allocator and returns Python lists and dicts.
- **Generic externs** keep their `type[T]` parameter as `const VhType *` in
  C and `const VhType &` in C++, and a value-bound one has a parameter
  `type` there; `T` values are `const void *` (a result, `void *out`) of
  `type.size` bytes. The thunk rejects a null descriptor. In nanobind the
  argument is the bound Struct's class, whose `vh_type` is the library's
  descriptor: a value must be an instance of it, and a result is a new one.
  A value-bound function takes no class; the value's class gives the
  descriptor.
- **The effect check**, with `effect_check=<name>` on `vd.library`: every
  `abi="status"` thunk of a non-pure extern calls
  `vh::Status <name>(const VhCall &call[, const T &object])` in
  `cpp_namespace` after its handles resolve and before the implementation,
  and returns the refusal as the call's status. `call` is a static
  `VhCall` with the extern's effect, symbol and `meta=` entries; a generic
  extern's thunk passes a copy whose `type` is the call's descriptor.
  `object` is that of the extern's first table handle parameter, and
  `<name>_api_impl.h` declares one overload per object type the externs
  pass, plus the one without. A destroying extern finds its object, asks,
  and only then takes it, so a refused destroy leaves the handle live.
  - Why a hook rather than a check per implementation: an application's
    rules (Vernon's "structural changes only from a structural system")
    then hold for every function, including ones added later, and read the
    same facts the schema declares, its effects and metadata, instead of
    restating them in C++.
  - Direct externs skip it: they cannot fail, so they cannot report a
    refusal. A library that needs to refuse one declares it `abi="status"`.
- **Stub placement.** `--pyi PATH` writes the stub to `PATH` instead of
  the output directory, and `check` compares it there, so it can sit beside
  the Python module that re-exports the extension's names, where editors
  find it. The stub gives each Struct its field constructor and re-exports
  `HostExecutionError`, which the nanobind module also sets as an
  attribute, so callers catch failures from the module they call.
- **Structs with `cpp=`**: without `convert=`, the thunks cast between the
  C and C++ types after `static_assert`s on size, alignment and trivial
  copyability. With `convert=<name>`, `<name>_api_impl.h` declares
  `Cpp <name>(const C &)` and `C <name>(const Cpp &)` in `cpp_namespace`,
  which the library defines. A list or dict of converted Structs or value
  handles converts item by item into a `std::vector` the `vh::Span` views,
  and a result's items back. Table handles cannot be items, as each would
  need a table lookup.
- **Python handle classes.** Each handle is a class over its C struct, with
  `bits`, equality and hashing by bits, and `__index__`, so Host calls in
  Python mode take it. A table handle with a one-parameter destroying extern
  has a subclass `<Handle>.Owned`, which its creating functions return and
  which destroys the object when collected. An explicit destroy zeroes the
  object's bits, so collecting it does nothing. A dependent handle's
  `Owned` object keeps its parent's Python object alive (`keep_alive`), so
  collection cannot destroy the parent under it. `Handle(bits)` names an
  object without owning it.
- **Not yet generated** (rejected with `BindgenError`): Tuple parameters or
  results, and lists of `str`.
- **No formatter runs.** Outputs are emitted in the project's style, so
  `check` does not depend on clang-format or ruff versions.

## State migration

- `HostModule.reload()` re-imports changed dependencies and runs the edited
  source as a new module; `HostModule.migrate(instance)` builds the new
  instance.
- `vh_migrate` (C, in `reflect.h`) and its Python mirror follow the same
  rules:
  - equal layout hashes copy the bytes;
  - otherwise the new instance is initialized from defaults, and each field
    with the same name and the same field layout hash is copied;
  - class fields whose layout changed migrate recursively.
- Container memory moves rather than copies: the target's default is
  released through its `fini`, and the source's records are zeroed
  (`vh_forget`), so both instances can be finalized without a double free.
  TensorStorages move the same way. A list, dict or storage whose element,
  key or value layout changed is a retyped field and keeps its new default. The two tables must share an allocator.
- Renamed or retyped fields take their new defaults. This keeps the rule
  predictable without needing annotations for renames.

## Tests

AOT code is tested by linking it, not by a JIT. `python/tests/host_aot.py`
(`LinkedHost`) compiles programs to an object, generates a C shim that
forwards each extern to a function-pointer slot, and links both into a shared
library with MSVC or `cc`. ctypes loads the library and fills the slots from
the registry. These tests skip without a C toolchain.

`python/tests/test_host_language.py`:

- runs the scalar parity table exactly against the linked library;
- runs it in Python mode, where only `PYTHON_DIFFERENCES` may differ;
- compares the two modes on random finite float inputs within f32 tolerance;
- covers extern binding, reload, and diagnostics.

`python/tests/test_host_classes.py` covers:

- class behavior, reflection contents (call-site reflection and record sizes
  included), the JSON form, generic externs, and migration across an edited
  source;
- Python mode under a debugger: line events at source lines, readable locals,
  a `pdb` breakpoint in a method, and tracebacks at the failing source line;
- C callers reaching Python-mode methods through the table's trampolines;
- diagnostics;
- objects for `x86_64-pc-windows-msvc`, `x86_64-unknown-linux-gnu`,
  `aarch64-apple-ios` and `wasm32-unknown-emscripten`;
- a C program (`host_programs/systems_consumer.c`) that links the native
  object and uses it through the header and through the table. That program
  implements the generic externs as a descriptor-keyed byte store.

The ctest `VernonPythonHostConsumerTest` covers the build path a C++
application uses:

- `vernon_add_program_asset` cooks `host_programs/systems_asset.py` (asset
  `demo/systems`: `systems.py` and `abilities.py`, which imports a class
  from `systems.py` and names a private helper as `systems.py` names an
  export) during the build.
- The nanobind module `vernon_host_consumer_fixture`
  (`source/tests/fixtures/host_consumer_fixture.cpp`) links the object and
  VernonRuntime and defines the externs in C++. Its `drive(id)` finds the
  table registered under the id and walks it by reflection alone. It
  allocates and initializes instances, edits fields by offset, calls methods
  and functions through table addresses, reads call-site reflection and
  writes JSON.
- `source/tests/python/host_consumer_runtime_test.py` checks that the object
  registered itself, and drives it. It then registers the `from_asset`
  Python-mode table under the same id, with the fixture's C++ functions as
  its externs, and drives it again. Both reports must be identical, and
  withdrawing the Python-mode table restores the linked one.

`python/tests/test_host_copies.py` runs `host_programs/copies.py`, which
copies lists, dicts, storages, class instances and containers of them and
then changes the copy, in Python mode and against the linked library. Both
must give identical results and release everything under a counting
allocator. It also covers `copy.deepcopy` of a whole instance from Python,
and the copy diagnostics.

`python/tests/test_host_packages.py` builds two `src/` layout projects, where
one imports a class and a function from the other. The other is found
through `sys.path` or through a setuptools-style editable finder. The
importer's module must compile, run in Python mode and run linked, with the
other's code named by its import name in a checkout and in an installed
copy alike. It also covers imports inside one `src/` layout package and the
diagnostic for names a Python import brought in.

`python/tests/test_host_containers.py` runs one script of every list and
`str` operation on `host_programs/inventory.py` in Python mode and against
the linked library, and requires identical results. The `LinkedHost` shim
forwards the container functions to the VernonRuntime inside
`vernon_dsl._native`. The file also covers `HostList` reads and writes, the
finalizer, `fini` through the Python-mode table, JSON, migration moving
containers, the header's `VhList` fields, and the container diagnostics.
`python/tests/test_host_dicts.py` does the same for every dict operation on
`host_programs/scoreboard.py`, with integer, `bool`, handle and `str` keys,
plus 1000-key growth with removals in Python mode. The consumer fixture's
`Poison` holds a `str`, a `list[f32]`, a `dict[Entity, f32]` and a
`dict[str, i32]`; the fixture installs a counting `VhAllocator` and checks
that every allocation is released in both modes. Through the generated
header's typed fields, it also reads what Host code wrote. It then appends a
drain and adds a level that the next Host calls read.
`python/tests/test_host_strs.py` runs `host_programs/messages.py` in Python
mode, where its f-strings are Python's own, and against the linked library
with a counting `VhAllocator`. It requires identical logs and results and
every allocation released, including on early returns; the program covers
every format type, fill, alignment and sign, `str` locals, `+` and `+=`,
borrowed and copied field operands, dict keys bound in loops, and
non-ASCII text. It also covers the `str` diagnostics.
`python/tests/test_host_composition.py` runs `host_programs/composition.py`,
where a class holds classes (one two levels down) that each own a list,
dict, `str` and `TensorStorage`, through nested fields and nested method
calls. It requires identical results in Python mode and AOT, and that the
outer `fini` releases every allocation.
`python/tests/test_host_class_values.py` runs `host_programs/class_values.py`
in Python mode and AOT with a counting allocator: class parameters and
results, owned and borrowed locals, moves into fields, elements and dict
values, pops, temporaries, and a `Node` tree holding `list[Node]` built by
recursive methods; nothing may stay allocated after `fini`. It also checks
JSON rebuilding lists and dicts of classes from their defaults, Python
callers moving instances, the header declaring the recursive class for C and
C++, and a diagnostic for each ownership rule.
`python/tests/test_host_container_locals.py` runs
`host_programs/container_locals.py` in Python mode and against the linked
library with a counting `VhAllocator`. It covers list and dict locals
(enum, `str` and integer keys, Struct elements), declarations in a loop
body, early returns, a missing key and a failing extern, and requires
identical results with every allocation released. It also covers the
container local diagnostics.
`python/tests/test_host_enums.py` runs `host_programs/enums.py` in Python
mode and against the linked library: enum parameters, results, aliases,
Struct and Tuple items, values no member has, and an extern taking and
returning an enum. It also covers enum dict keys and class fields through
the Python-mode table and JSON, the reflection records, the header and
reflection JSON, the library identity, the diagnostics, and a C program
(`host_programs/enums_consumer.c`) that uses the header's constants and
writes enum JSON with `vh_write_json`.
`python/tests/test_host_storages.py` runs `host_programs/storages.py` in
Python mode and against the linked library with a counting allocator, and
requires identical results, statuses and logs, and every allocation
released. The program covers creation into fields and locals, failed
creations, numpy-style indexing with negative indices, `shape`, `len`,
`fill`, and both copies. The file also covers the Python-mode objects over
instance memory, JSON, migration, the header's `VH_STORAGE` fields and the
storage diagnostics. The consumer fixture's `Trail` holds two storages; the
fixture indexes them through `vh::TensorStorage`, resizes one, checks its
`VernonTensorView`, and moves it with `vh_migrate`, in both modes.
`python/tests/test_bindgen.py` covers level 1 bindings on the sample
library `bindgen_sample/tally.py`, a C++ `tally::Counter` (`cpp/`) whose
generated outputs are committed in `generated/`. CMake builds its nanobind
module `tally_api` with `BUILD_TESTING`, next to the sample. The tests run
the drift check (and that `check` reports an edited or missing output) and
call the nanobind functions. They run `bindgen_sample/tally_user.py` in
Python mode through `bind_module` and from a linked AOT object over the same
library, and require identical results and statuses for successful runs, a
library failure, a destroyed handle, and live-object counts. The sample
passes list and dict spans (with a repeated dict key), receives `str`, list
and dict results, converts a `Range` Struct, casts a `Sample` Struct, and
stores a Host Struct through generic externs keyed by its descriptor. They
also link a C program (`bindgen_sample/consumer.c`) from the Host object,
the thunks, the library and VernonRuntime's container functions compiled
with `VERNON_RUNTIME_STATIC`, as a static mobile or web build links them,
and cover the generator's rejections.
`python/tests/test_host_transfers.py` runs `host_programs/transfers.py`
against Python bindings and against C implementations
(`host_programs/transfers_impl.c`, built with `vh_list_assign`). It runs
them in Python mode and from the linked library with a counting allocator,
and requires identical results, with every allocation released and released
at its allocated size. The program passes list and dict fields and locals
as spans and receives `str`, list and dict results into fields and locals,
including repeated keys, a failing call and calls in a loop. One run has no
allocator installed and uses the runtime default across the C library's
heap. The file also covers the declaration and position diagnostics.
`RuntimeCApi.HostStrFormattingMatchesPython` checks the formatter against
about 40 values formatted by Python.
`test_generated_header_compiles_as_c_records_and_typed_cpp` compiles a
generated header as C11 and, with the typed classes, as C++17.
`RuntimeCApi.HostContainerClassesChangeTheRecordsThroughTheRuntime` covers
the C++ classes.
`RuntimeCApi.HostListsGrowThroughTheModuleAllocator`,
`HostDictsKeepInsertionOrderAcrossRemovalAndGrowth` and
`HostDictsOwnTheirStrKeys` cover the runtime functions.

`test_host_classes.py` covers the declaration rules, cooking for the native
and wasm32 triples, the rejection of other targets, and combined modules:
their single object and header, private function naming, Python mode over
several sources, and the rejection of a name two modules define.
`RuntimeCApi.HostModuleRegistrationFindsTheNewestLiveTable` covers the
registry.
