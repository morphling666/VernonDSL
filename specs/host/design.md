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
- **Places.** An instance is a place, not a value: `self` is the instance
  pointer, and field and element accesses rooted at it compile to address
  arithmetic with loads and stores.
  - A class value cannot be copied into a local, returned, or held in a
    Struct, tuple or array.
  - A class value can only be a class field or `self`; the application owns
    the top-level instances.
  - The only writable places are fields reached through `self`.
  - `+=` must keep the field's type.
  - Containment cycles are rejected.
- **Why these rules.** Value semantics for classes would need copies of every
  nested class and hidden temporaries in the ABI. Place semantics keep a
  method a plain C function over a pointer the application allocated. That is
  what an engine stores in its own tables and hands to a scheduler.
- **No class locals in v1.** Instances exist only in application memory or as
  fields of other classes. This removes lifetime and aliasing questions until
  a real use needs a local.
- **No inheritance, dunder methods or `self` annotations.** Composition
  through class fields covers reuse without vtables, so the layout stays a C
  struct.

## Metadata and docstrings

- `vd.meta(**entries)` attaches metadata in three places:
  - classes: `@vd.host(meta=...)`;
  - Structs: `@vd.struct(meta=...)`;
  - fields: `Annotated[T, vd.meta(...)]`.
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
  - plus the module's free functions, and the module's allocator slot.
- **Call-site reflection.** Each function lists the externs it can reach,
  through any chain of helpers and methods, as `VhCall` entries: the C
  symbol, the extern's declared effect, and for a generic extern the Struct
  descriptor that call passes. The bound type at a call site is static, so the
  list is exact. An application derives type-granular access from it, such
  as the components a system reads and writes, instead of trusting a
  hand-written list.
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
    bits; non-finite floats are `"nan"`, `"inf"` and `"-inf"`.
  - A dict is an object in insertion order. JSON keys are strings, so a
    non-`str` key is written as its own JSON text: `{"7": 1.5}`,
    `{"true": 2}`.
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
- **Why fields only.** Containers as locals, parameters or results would
  need moves or copies at every boundary and destructors on every exit
  path. Keeping them in instance fields keeps the ownership rule to "the
  instance frees it", which the place rules already enforce: a container is
  reached only through `self` and never escapes. `str` locals are the one
  exception, since a `str` is a value: it is copied on assignment, never
  aliased, and never leaves its function except as an extern argument,
  which only borrows it. TensorStorage locals are the other, below.

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
  and the result may mention `T`.
- **Call sites.** Each call site's Struct argument specializes the extern
  (`name__Struct`) for type checking. The C symbol is shared: the descriptor
  `const VhType *` is inserted at the parameter's position, and every
  `T`-typed value is passed by pointer.
- **Python bindings.** A binding serves every specialization. It receives the
  descriptor and receives and returns `T` values as field dicts.
- **Why.** An engine exposes one `read`/`write`/`has` entry point over
  component storage it never compiled against. That storage is keyed by the
  descriptor and copies `descriptor->size` bytes, which is how script-declared
  components reach a C++ engine without generated C++ per type.

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
