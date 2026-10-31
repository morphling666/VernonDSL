# Host language reference

Status: normative for the implemented Host language. It states what a program
may contain and what it means. The reasons behind the design, the pipeline and
the reflection table are in [`design.md`](design.md).

A Host program is a Python file that `vernon_dsl.hostlang` type-checks. It runs
in two modes:

- **Python mode**: CPython runs the file as written, with numpy values.
- **Compiled mode**: the file is compiled ahead of time to a native or wasm32
  object.

A program the checker accepts means the same in both modes, apart from the
differences listed in [Python-mode differences](#python-mode-differences).
Anything else that differs, or any accepted program that fails to compile, is
a checker bug; the ones known today are listed in [Known gaps](#known-gaps).

The rules are enforced by `hostlang/frontend.py` and the shared frontend.
Compiled semantics are in `hostlang/llvm_emitter.py`, and Python-mode values
in `hostlang/native.py`. `python/tests/test_host_language.py` and
`python/tests/test_host_classes.py` test them.

## Program structure

A program is one module and the modules it imports. Its top level may declare:

| Declaration | Purpose |
|---|---|
| `@vd.host` / `@vd.host(effects=...)` function | Host code; exported when defined in the root file |
| `@vd.host` / `@vd.host(meta=...)` class | mutable state with methods |
| `@vd.struct(shared=True)` class | immutable value type |
| `@vd.handle` / `@vd.handle(c_type=...)` class | opaque reference to an object the application owns |
| `@vd.extern(...)` function | a C function the application provides |
| `@vd.func(shared=True)` function | pure helper, also usable by device code |

- Each function has exactly one decorator. Other decorators, such as device
  stage decorators, are rejected in Host programs.
- Parameters are required and positional: no defaults, `*args`, keyword-only
  or positional-only parameters. Calls pass positional arguments only.
- `@host` parameters and results must be annotated. A function returning
  nothing is annotated `-> None`. Every path of a function with a result must
  return.
- Recursion, direct or indirect, is rejected.

## Types

### Scalars

| Type | Compiled | Python mode |
|---|---|---|
| `bool` | `bool` | `np.bool_` |
| `i32`, `u32` | `int32_t`, `uint32_t` | `np.int32`, `np.uint32` |
| `i64`, `u64` | `int64_t`, `uint64_t` | `np.int64`, `np.uint64` |
| `f32`, `f64` | `float`, `double` | `np.float32`, `np.float64` |

`f16` is a device type and not a Host value.

### Aggregates

- **Tensors.** `Vector[T, N]`, `Matrix[T, R, C]` and `Tensor[T, D0, D1, ...]`
  have scalar elements and fixed extents. They are laid out as row-major C
  arrays. In Python mode they are read-only `Vector`/`Matrix`/`Tensor`
  ndarrays. Elements of Structs, handles or classes are rejected.
- **Tuples.** `Tuple[T0, T1, ...]` is a fixed product of values, built with
  `Tuple(a, b, ...)` and unpacked with `a, b = ...`. In Python mode it is a
  Python tuple.
- **Structs.** A `@vd.struct(shared=True)` class has annotation-only fields of
  value types, without defaults. Construct it positionally, `Point(x, y)`, and
  read its fields with `p.x`. A Struct may have `@vd.func(shared=True)` methods.
  Options:
  - `c_type="Name"` names the C typedef;
  - `meta=vd.meta(...)` attaches metadata.

  In Python mode a Struct is an immutable instance of the declared class.
- **Handles.** A handle names an object that lives in the application, such
  as a file, a texture or a scene node, which Host code passes to externs but
  cannot look inside. Each `@vd.handle` class is a distinct type, so a `Node`
  cannot be passed where a `File` is expected, although both are 64 bits.
  `c_type="Name"` gives it its own C typedef, so C callers keep the
  distinction too. A handle class has no body. Compiled code passes it as its
  `uint64_t` bits; in Python mode it is an `int`. `Node()` is the null handle,
  0. Handles compare with `==` and `!=`.
- **str.** UTF-8 text. A `str` is a class field (see
  [Containers](#containers)), a local, a string literal, an f-string, or an
  extern `str` parameter, which receives `{const char *data; uint64_t size}`
  (see [Strings](#strings)). It cannot be a `@host` parameter or result, or a
  Struct field.
- **list[T].** A growable sequence of values, owned by the class instance
  whose field it is (see [Containers](#containers)). `T` is a value type
  without containers or classes.
- **dict[K, V].** A hash map that keeps insertion order, owned by the class
  instance whose field it is (see [Containers](#containers)). `K` is an
  integer, `bool`, handle or `str`; floats are rejected because NaN and
  signed zero have no single equality. `V` is a value type without
  containers or classes.
- **TensorStorage[T, (e0, e1, ...)].** A dense row-major array of `bool`,
  `i32`, `u32`, `f32` or `f64` elements, in host memory owned by the class
  instance or the function that holds it (see
  [Tensor storages](#tensor-storages)). The rank is fixed at 1 to 8; each
  extent is a constant or `dyn`.
- **Host classes.** See [Classes](#classes).

Other device types (tensor views, textures, samplers) are rejected.

## Values and places

Every type except a Host class is a **value**. Values are immutable in both
modes: there is no way to change a value, only to build a new one. So sharing
a value and copying it cannot be told apart, and Python's reference semantics
match compiled code's SSA values.

- A Struct field cannot be assigned: `p.x = 1.0` is rejected, and in Python
  mode it raises `AttributeError`. `vd.replace(p, x=1.0)` builds a copy with
  the named fields changed; the keywords are evaluated in source order.
- A Tensor element cannot be assigned, wherever the Tensor is held:
  `v[0] = 1.0` is rejected, and in Python mode it raises `ValueError`.
  `v.at[i, j].set(x)` builds a copy with one element changed, and
  `v.at[i, j].add(x)` one with `x` added to it, as in JAX. `.at` takes a
  [selection](#slices) as a read does, so `v.at[1:3, :].set(b)` replaces a
  block; its elements must be scalars. Elements that change in place belong
  in a [TensorStorage](#tensor-storages).
- Rebinding a local is not mutation: `v = v + 1.0` builds a new value, and
  so does `self.history = self.history.at[i].set(x)` for a field.

The same rules hold in device code, so a value means the same thing in
Host, `@func` and kernel code.

A **Host class** instance is a **place**: storage the application owns.

- `self` is the only name for an instance in Host code. A method's `self` is
  a pointer in compiled code and a reference in Python mode.
- The writable paths all start at `self`:
  - a scalar, Tensor, Struct, handle or `str` field: `self.speed = x`;
  - an element of a list field: `self.history[i] = x`;
  - an element of a TensorStorage field: `self.heights[i, j] = x`;
  - the value for a key of a dict field: `self.scores[k] = x`;
  - a list or dict field, through its methods, `del`, or `self.items = []`
    and `self.scores = {}`;
  - a field of a nested class field: `self.timer.fired += 1`.
- A Struct field reached through `self` is a value, so `self.rest.scale = x`
  is rejected; assign the whole field, `self.rest = Transform(...)`.
- A class-typed field cannot be assigned as a whole; assign its fields.
- Instances cannot be copied into locals, passed (except as the receiver of a
  method call), returned, or held in a Struct, Tuple or Tensor.

## Names and locals

- A local's type is the type of its first assignment and cannot change. An
  annotated first assignment sets it: `count: i64 = 0`.
- Parameters may be rebound, except `self`.
- A `for` target is an `i32` local over a `range`, and a local of the
  element type over a list. Over a dict it is a key local, a value local
  for `values()`, or a `k, v` pair of locals for `items()`. A `str` key
  local holds a copy of the key, so it stays readable after `del` removes
  the entry.

## Literals and conversions

- An integer literal is typed from its context: the other operand of a binary
  operator, or the declared type it is assigned or returned to. Without
  context it is `i32`. A float literal is typed the same way and defaults to
  `f32`. A literal that does not fit its type is rejected.
- Implicit conversions are only lossless:
  - integer to float;
  - float to a wider float.

  Narrowing, such as `f64` to `f32`, is rejected. So is mixing integers of
  different width or signedness, which has no common type: `i32 + i64`
  requires a cast.
- Explicit conversions are `bool()`, `i32()`, `u32()`, `i64()`, `u64()`,
  `f32()` and `f64()`, each on one scalar. `int()` means `i32()` and
  `float()` means `f32()`.
  - Float to integer truncates toward zero, saturates out of range, and
    converts NaN to 0.
  - Float to `bool` is `x != 0`, so NaN is true.
- `+=` and the other augmented assignments on a field must keep the field's
  type. The target's indices and keys are evaluated once, as in Python:
  `self.d[f()] += 1` calls `f` once.

## Expressions

- **Arithmetic.** `+ - * / // % **` on numeric scalars and elementwise on
  Tensors. A scalar broadcasts against a Tensor; two Tensors must have the
  same shape. There is no arithmetic on `bool`.
  - Integers wrap at their width.
  - `/` on two `i32` or two `u32` gives `f32`, and on two `i64` or two `u64`
    gives `f64`.
  - `//` and `%` require integers. For signed operands they floor, as in
    Python. A zero divisor is a `DIVISION_BY_ZERO` error.
  - On floats, `%` takes the divisor's sign, as in Python.
  - `**` requires floats.
  - Floats round in the operand width and are never fused.
- **Bitwise.** `& | ^ << >>` and `~` on integers, elementwise on Tensors.
  A left or unsigned right shift by the bit width or more, or by a negative
  amount, is 0; a signed right shift by such an amount fills with the sign.
  This is numpy's definition, so both modes agree; device code shares it.
- **Comparisons.** `== != < <= > >=` with exactly one operator; chained
  comparisons are rejected. In compiled code, every comparison involving NaN
  is false, `!=` included. `k in self.d` and `k not in self.d` test a dict
  field's keys.
- **Logic.** `and`, `or` and `not` take `bool` operands and short-circuit.
  `a if c else b` evaluates one branch.
- **Unary.** `-x`, `+x` and `~x`.
- **Strings.** `a + b` joins two `str` values, and f-strings format values
  into one (see [Strings](#strings)).
- **Not supported:** `@`, `in` on anything but a dict field, slices of
  anything but a Tensor or TensorStorage, list and dict displays other than the `[]` and `{}` assigned to a container
  field, set literals, comprehensions and lambdas.
- **Indexing.**
  - A Tensor takes one index per dimension: `v[i]`, `m[r, c]`, `t[i, j, k]`.
    Indices may be dynamic. A negative index counts from the end, as in
    numpy; one still out of range is an `INDEX_OUT_OF_RANGE` error. A
    constant index out of range is rejected. Fewer indices, or a slice,
    select several elements ([Slices](#slices)).
  - A list takes one index. A negative index counts from the end, as in
    Python; one still out of range is an `INDEX_OUT_OF_RANGE` error.
  - A dict takes one key. Reading a missing key is a `KEY_NOT_FOUND` error;
    storing one inserts it at the end of the order.
  - Vectors also take swizzles of `x y z w` or `r g b a`: `v.y`, `v.xz`.
- **Conditions.** An `if`/`while` test or `IfExp` condition may be any scalar,
  converted as by `bool()`.

### Slices

A Tensor or TensorStorage of scalar elements takes a numpy selection, with
one item per axis. A Tensor's shape is static, so every selection has a
length known at compile time:

- an integer index picks one position and drops the axis;
- a missing trailing axis, `:`, or `...` (at most once) keeps a whole axis;
- a slice of integer literals follows numpy: `1:3`, `:-1`, `::2`, `::-1`;
- `a:b` with `b > a` is the **window** of `b - a` elements starting at `a`,
  where a negative `a` counts from the end. A dynamic start is written as a
  window `i:i+n` with `n` a positive literal, and the start has no calls,
  since Python evaluates it in both bounds.

```python
row = m[1]                            # Tensor[f32, (4,)]
corner = m[:2, -2:]                   # Tensor[f32, (2, 2)]
pair = m[1, i : i + 2]                # a window from a dynamic start
m = m.at[1:3, i : i + 2].set(corner)  # a block replaced
self.heights[r, 0:2] = 1.5            # a block written in place
```

A window that runs off the end is where numpy would return fewer elements,
so the two kinds of container decide it the same way in every mode:

- a Tensor value clamps the window's start so that the window fits, as
  JAX's `dynamic_slice` does: `v[i : i + 2]` of a `(4,)` Tensor at `i = 3`
  reads elements 2 and 3;
- a TensorStorage fails with `INDEX_OUT_OF_RANGE` (`IndexError` in Python
  mode);
- a constant window that does not fit is rejected, so numpy and the window
  rule agree on every constant slice that compiles.

A slice that selects nothing is rejected. An axis of `dyn` extent takes only
an index or a window. A selection reads as a new Tensor value, a copy. A
value written into a selection, with `.at`, `=` or `+=` on a storage, is a
scalar for every element or a Tensor of exactly the selection's shape.

## Statements

The supported statements are:

- assignment to a name, a tuple of names, or a writable place
  ([Values and places](#values-and-places));
- annotated assignment and augmented assignment;
- expression statements;
- `return`;
- `if`/`elif`/`else`;
- `while` with an optional `else`;
- `for i in range(stop)`, `range(start, stop)` and `range(start, stop, step)`,
  with `i32` bounds and an optional `else`. A zero step is an
  `INVALID_RANGE_STEP` error;
- `for x in self.items` over a list field, with an optional `else`. As with
  Python's list iterator, each step reads the list's current length, so the
  body may append to or pop from the list;
- `for k in self.d`, `for k in self.d.keys()`, `for v in self.d.values()`
  and `for k, v in self.d.items()` over a dict field, in insertion order,
  with an optional `else`. As in Python, a step after the dict's size
  changed is a `CONTAINER_CHANGED` error; assigning to an existing key does
  not change the size. `for` over anything else is rejected;
- `del self.d[k]` on a dict field, which is a `KEY_NOT_FOUND` error if `k`
  is missing;
- `break`, `continue`, `pass`.

`with`, `try`, `raise`, `assert`, other `del` targets, `global` and
`nonlocal` are rejected.

## Calls

Host code may call:

- `@host` functions;
- `@func(shared=True)` helpers, which may leave parameters unannotated to be
  generic. Each distinct argument typing makes a separate copy;
- externs;
- methods, on `self` and its class fields, and Struct `@func` methods on
  values;
- constructors: Structs, handles, `Vector`, `Matrix`, `Tensor`, `Tuple`;
- conversions;
- on a list field: `len(self.items)` (an `i32`), `self.items.append(v)`,
  `self.items.pop()` and `self.items.clear()`;
- on a dict field: `len(self.d)` (an `i32`), `self.d.get(k, default)`,
  `self.d.pop(k)`, `self.d.pop(k, default)` and `self.d.clear()`. `get`
  requires its default, since there is no `None`. `keys()`, `values()` and
  `items()` appear only as a `for` iterable;
- `len` of a `str`: its length in code points, as in Python, not bytes;
- the intrinsics:
  - `abs`, `min`, `max`, `clamp`;
  - `sqrt`, `floor`, `pow`, `exp`, `log`;
  - `sin`, `cos`, `acos`, `atan2`;
  - `dot`, `cross`, `norm`, `normalize`.

Transcendental functions call the platform libm in compiled code. In Python
mode `min` and `max` are Python's builtins.

Helpers cannot call externs or `@host` functions. Device operations are
rejected.

## Effects

Effects are ordered `pure < read < write < io`.

- A `@host` function or method declares its effect with `effects=...`. The
  default is `io`.
- An extern declares the effect of calling it; the default is `io`.
- A function's effect is the greatest effect of what it calls, and must not
  exceed its declared effect.
- `@func` helpers are `pure`.

## Externs

```python
@vd.extern(symbol="app_find_node", effects="read", abi="status", schema="app-v3")
def find(name: str) -> Node: ...
```

- The body must be `...`. Parameters and the result must be annotated.
- `symbol` is the C name; it defaults to the function name.
- `abi="direct"` (the default) calls `R symbol(params...)` and allows only
  scalar or handle results. `abi="status"` calls
  `int32_t symbol(params..., R *out)`. A nonzero status ends the calling Host
  function with that status.
- `schema` must match the schema the application binds the symbol with.
- A `str` parameter takes any `str` expression. A `str` field passed before
  an argument that makes a call is copied first, so the extern receives the
  text as it was when that argument was reached, as in Python.
- **Generic externs.** A parameter `type: type[T]`, with
  `T = TypeVar("T")`, makes the extern generic over Structs. Other parameters
  and the result may use `T`. Each call passes a Struct class, and the C
  function receives that Struct's `const VhType *` descriptor at the
  parameter's position, with `T` values by pointer.
- In Python mode, an extern call goes to the registry of the active runtime.
  A Python binding receives Structs as field dicts and Tensors as tuples, and
  returns the same forms. An exception it raises propagates to the caller.
- Every exported function is called as
  `int32_t <prefix><name>(params..., R *out)`. It returns 0 or a status.
  Structs and Tensors are passed by pointer, and handles as `uint64_t`.

## Classes

```python
@vd.host(meta=vd.meta(category="motion"))
class Mover:
    """Docstring, kept in reflection."""

    target: Node
    speed: Annotated[f32, vd.meta(min=0.0)] = 2.0
    """Field docstring."""
    timer: Timer

    @vd.host(effects="write")
    def update(self, dt: f64) -> i32: ...
```

- **Fields** have value types, container types or Host class types, with an
  optional default. The default is a constant or constructor expression.
- **Initialization.** A new instance is zeroed. Then each field, in
  declaration order, takes its default or, for a class field, is initialized
  recursively.
- **Methods** take `self` unannotated. An undecorated method is `effects="io"`;
  `@vd.host(effects=...)` sets one method's effect. A leading underscore makes
  a method private: callable inside the module, but not exported.
- **Metadata.** `vd.meta(**entries)` on a class, Struct or field holds
  compile-time constants: numbers, strings, booleans, `None`, and tuples of
  them. The language fixes no keys.
- **Rejected:**
  - `__init__` and other dunder methods;
  - inheritance;
  - annotating `self`;
  - a class that contains itself;
  - class-typed locals, parameters and results.
- **In Python mode:**
  - `Mover()` creates an initialized instance, and `m.update(dt)` runs with
    the module's runtime;
  - scalar fields read as numpy scalars;
  - Tensor fields read as read-only copies;
  - Struct fields read as fresh values, and class fields as views;
  - list fields read as `HostList` views of the instance's memory, dict
    fields as `HostDict` views, and `str` fields as Python `str` copies;
  - `address_of(m)` is the pointer C code receives.

## Containers

```python
@vd.host
class Poison:
    name: str = "poison"
    drains: list[f32]

    def update(self, drained: f32) -> f32:
        self.drains.append(drained)
        total = f32(0.0)
        for value in self.drains:
            total += value
        return total if self.name == "poison" else self.drains[-1]
```

- `list[T]`, `dict[K, V]` and `str` fields own memory, which is freed with
  the instance. Nothing outside the instance refers into it, so there is no
  garbage collection or reference counting. A dict owns copies of its `str`
  keys.
- A container field is used in place, never copied:
  - a list is indexed, iterated, passed to `len`, or has `append`, `pop` or
    `clear` called on it; `self.items = []` clears it;
  - a dict is indexed, iterated, tested with `in`, passed to `len`, has its
    methods called, or has a key deleted; `self.d = {}` clears it;
  - a `str` field is used as any `str` (see [Strings](#strings)).
- A field default is `[]` for a list, `{}` for a dict, or a string literal
  for a `str`; without one the field starts empty.
- `pop` on an empty list is an `INDEX_OUT_OF_RANGE` error, `pop(k)` of a
  missing key is `KEY_NOT_FOUND`, and failing to grow a container is
  `OUT_OF_MEMORY`.

## Tensor storages

```python
@vd.host
class Grid:
    heights: TensorStorage[f32, (dyn, dyn)]
    tile: TensorStorage[i32, (2, 3)]

    def resize(self, rows: i32, columns: i32) -> i32:
        self.heights = vd.zeros(dtype=f32, shape=(rows, columns))
        self.heights[-1, 0] = f32(2.0)
        height, width = self.heights.shape
        return height * width

    def blurred_total(self) -> f32:
        scratch = vd.zeros_like(self.heights)
        scratch.fill(f32(1.0))
        return scratch[0, 0] + self.tile.to_values()[1, 2]
```

A `TensorStorage` is a container like a `list`, used as numpy arrays are.
It has nothing to do with kernels, graphics or the module's devices: its
elements are host memory from the module's allocator.

- It is a class field or a local. A field without a default starts zeroed
  at its extents when they are all static; otherwise it starts with no
  elements and every extent 0. A local is
  freed when the function returns, on every path. It cannot be a
  parameter, a result, a Struct field, an extern argument, or used in an
  `@func` helper.
- It is assigned only from a creation call:
  - `vd.zeros(dtype=T, shape=(...))` and `vd.empty(dtype=T, shape=(...))`;
  - `vd.zeros_like(s)` and `vd.empty_like(s)`;
  - `vd.from_values(t, dtype=T)`, a copy of a `Tensor` value whose element
    type is `T`, at the Tensor's shape.

  The new elements replace the old ones, which are freed. A failed creation
  leaves the target as it was. The created shape must match each static
  extent, and a field default may be a creation call.
- It is used in place:
  - `s[i, j]` reads and assigns an element, with one index per axis, where
    negative indices wrap;
  - `s[1:3, j]` and the other [slices](#slices) read a `Tensor` copy of the
    selected elements, and `s[1:3, j] = b` and `s[1:3, j] += b` write them in
    place. As in Python, `=` computes the value before checking the
    selection, and `+=` reads the selection first;
  - `s.shape` is a `Tuple` of `i32` extents, and `len(s)` is the first;
  - `s.fill(v)` sets every element;
  - `s.copy_from_values(t)` copies a `Tensor` of the same element type and
    rank in;
  - `s.to_values()` copies the elements out as a `Tensor`, which needs every
    extent static.

  The names say what crosses: a storage holds elements in place, and a
  `Tensor` value carries a copy of them in or out.
- An index out of range is `INDEX_OUT_OF_RANGE`. A created shape that does
  not match a static extent, or a `copy_from_values` of another shape, is
  `INVALID_SHAPE`.
- There are no views: a slice is a copy out or a write in, never a storage
  that aliases another.
- `i64` and `u64` elements are rejected, since a numpy array of the
  runtime's element types has no 64-bit integer.

In Python mode a field reads as a `vd.TensorStorage` over the instance's
memory, so element writes through it are what compiled code reads. Reading
the field again gives the same object until it is replaced. A storage kept
after its field is replaced, or after the instance is released, keeps a copy
of its elements.

## Strings

```python
@vd.host
class Logbook:
    last: str

    def report(self, ship: Ship, speed: f32, docked: bool) -> i32:
        status = "docked" if docked else "under way"
        line = f"ship {ship} at {speed:.1f} m/s ({status})"
        line += "!"
        log(line)
        self.last = line
        return len(line)
```

- A `str` local owns its text, which is freed when the function returns,
  on every path. Assigning to it or to a `str` field copies the text, so a
  `str` is a value like any other.
- A `str` is used to:
  - assign a local or a field, or `+=` onto one;
  - join with `+`, format in an f-string, or pick in `a if c else b`;
  - compare with `==` or `!=`;
  - index or test a dict with `str` keys;
  - pass to `len` or to an extern `str` parameter.
- `@func` helpers cannot use `str`, since they are shared with device code.
- **f-strings** format integers, floats, `bool`, handles and `str`. Without
  a format spec a value formats as Python's `str()`: a handle as its bits,
  a `bool` as `True` or `False`.
  - A float formats as Python formats the `float` it converts to, so an
    `f32` 0.1 is `0.10000000149011612`. This is also what numpy scalars
    give in Python mode.
  - The format spec is `[[fill]align][sign][0][width][.precision][type]`,
    with Python's meaning. Integers and handles take the types `d`, `x`,
    `X`, `o` and `b`; floats take `f`, `e` and `E`. A `str` takes only
    fill, align and width, and a `bool` takes no spec. Width is at most
    2^20 and precision at most 64.
  - Rejected: conversions (`!r`, `!s`, `!a`), the `=` form, and specs
    with replacement fields, such as `{x:{width}}`.

## Runtime errors

Compiled code reports a failure as the function's status:

| Status | Value | Python mode |
|---|---|---|
| `DIVISION_BY_ZERO` | -1 | 0 and a numpy `RuntimeWarning` |
| `INDEX_OUT_OF_RANGE` | -2 | `IndexError` on a Tensor, `.at` or TensorStorage; `HostExecutionError` on a list |
| `INVALID_RANGE_STEP` | -3 | `ValueError` from `range` |
| `OUT_OF_MEMORY` | -4 | `HostExecutionError` with that status |
| `KEY_NOT_FOUND` | -5 | `HostExecutionError` with that status |
| `CONTAINER_CHANGED` | -6 | `HostExecutionError` with that status |
| `INVALID_SHAPE` | -7 | `ValueError` from the TensorStorage creation or copy |
| a nonzero extern status | the extern's | `HostExecutionError` with that status |

An exception escaping Python-mode code under a C caller is recorded in the
module's `errors`, and the C call returns 1.

## Python-mode differences

These differences are accepted. Python mode converts values to their declared
types only at boundaries:

- parameters on entry from Python;
- results returned to Python;
- field stores;
- extern arguments and results.

Between boundaries, locals hold whatever Python and numpy compute. Programs
whose results depend on the following differ between the modes:

- **Integer width of intermediates.** A local built only from literals is an
  unbounded Python `int` (`x = 2147483647; x + 1` is 2147483648). Under numpy
  1.x, a numpy integer combined with a Python `int` promotes to `int64`, so
  `u32(0) - 1` is -1. Under numpy 2 it wraps to 4294967295. The value is
  converted at the next boundary, where an `i32` result out of range raises
  `OverflowError`.
- **Float width of intermediates.** Under numpy 1.x, an `f32` scalar combined
  with a Python float is `f64`; under numpy 2 it stays `f32`. `float(x)` and
  integer `/` produce `f64` in Python. Compositions such as `normalize`
  round differently, by about an ulp, and cancellation can magnify this.
- **Conversions.** `int(x)` is an unbounded Python `int`. Out-of-range float
  to integer conversions follow the CPU or raise, instead of saturating.
- **NaN and signed zero.** `!=` is true when an operand is NaN. `-x` of `0.0`
  is `-0.0`. `min` and `max` follow Python's NaN behavior.
- **Errors.** As in [Runtime errors](#runtime-errors).

`parity_cases.PYTHON_DIFFERENCES` lists the parity cases that differ, and the
test fails if any other case does.

This project installs numpy 1.x on Python versions before 3.13 and numpy 2 on
3.13 and later.

## Known gaps

These are checker bugs: programs that should be rejected, or that crash
instead of reporting a diagnostic.

- `is` and `is not` pass the checker. Python compares identity, and the AOT
  compiler crashes.
- `==` on Tensors types as `bool`. Python returns an elementwise array, and
  the AOT compiler crashes.
- Indexing a `Tuple` value (`t[1]`), and a nested `def`, crash the checker with
  an internal `KeyError`.
- `int()` and `float()` could be rejected in favor of `i32()` and `f32()`,
  which mean the same in both modes.
