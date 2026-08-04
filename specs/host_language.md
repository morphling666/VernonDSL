# Host language and native interop design

## Status

This document is a post-0.1.1 design and implementation plan. None of the Host
language, C++ extern, desktop Host AOT, or browser WebAssembly interfaces
described here are part of the 0.1.1 public contract.

The first acceptance target is one gameplay example whose source:

- executes through a typed interpreter during development;
- calls a schema-defined C++ API through nanobind;
- compiles to a native desktop `.o` or `.obj`;
- compiles to a `wasm32-unknown-emscripten` relocatable object;
- links with the same C++ implementation for desktop and browser;
- produces the same deterministic state and checksum in all three modes.

GPU dispatch, hot reload, engine serialization, networking, state migration,
and general browser Runtime support are later work.

## Goals

- Add a restricted Host domain without weakening the existing device-language
  rules.
- Use one typed representation for interpreted and AOT execution.
- Let Host code call C++ without depending on the C++ ABI, STL layouts, or
  Python objects.
- Reuse nanobind for development bindings without making nanobind metadata the
  deployment contract.
- Keep Python and nanobind out of desktop and browser deployment artifacts.
- Generate native and WebAssembly objects from the same specialized Host IR.
- Keep the existing CPU compute entry ABI and GPU compilation paths unchanged.

## Non-goals for the first version

- Calling arbitrary Python from compiled Host code.
- Calling ordinary C++ functions directly from GPU kernels.
- C++ exceptions crossing an ABI boundary.
- Passing C++ classes, STL containers, Python objects, or implicit ownership
  across the Host ABI.
- Host-to-GPU kernel dispatch or ExecutionGraph construction from compiled Host
  code.
- A browser Python runtime, WASM wheel, dynamic loader, or general browser
  plugin system.
- Function-table hot reload, state migration, network replication, or rollback
  semantics.

## Architecture

```text
C++ API schema
  -> stable C declarations and C++ thunks
  -> nanobind development module
  -> Vernon extern declarations and metadata
  -> Python type stubs

Python-syntax Host source
  -> module graph
  -> typed inference and effect analysis
  -> Typed Host IR
       -> development interpreter -> nanobind extern registry
       -> Host MLIR -> LLVM -> native .o/.obj
       -> Host MLIR -> LLVM -> wasm32 relocatable object

native object + C++ thunks -> desktop executable
wasm object + C++ thunks -> em++ -> .wasm + JavaScript loader
```

The module graph, type parser, specialization cache, typed expression model,
and diagnostics remain shared with the existing frontend. Host execution forks
after typed inference because compute entries, graphics stages, and Host
functions have different effects and ABIs.

## Host language surface

### Host entry

`@vd.host` declares a Host entry. It is not a shader stage and is not
stage-polymorphic:

```python
@vd.host
def update(world: WorldHandle, entity: EntityHandle, dt: vd.f32) -> State:
    position = game.get_position(world, entity)
    velocity = game.get_velocity(world, entity)
    next_position = position + velocity * dt
    game.set_position(world, entity, next_position)
    return State(position=next_position, velocity=velocity)
```

Host entries support:

- required positional arguments with explicit annotations;
- fixed-width scalar values, enums, fixed vectors and matrices;
- ABI-stable structs and opaque integer handles;
- local immutable values, assignment, conditionals, bounded loops, calls to
  pure Host helpers, and declared extern calls;
- explicit `pure`, `read`, `write`, or `io` effects.

Host entries reject:

- workgroup, invocation, vertex, fragment, texture, sampler, and device-resource
  operations;
- arbitrary module imports or Python calls;
- recursive call graphs;
- implicit exception or ownership behavior;
- device `@kernel`, `@vertex`, or `@fragment` calls in the first version.

`@func(shared=True)` remains a pure host/device numeric helper. It does not
become the Host orchestration entry point.

### External functions

Generated extern declarations carry a symbol, signature, effect, schema
version, and source API identity:

```python
@vd.extern(
    symbol="vernon_demo_get_position_v1",
    effects="read",
)
def get_position(world: WorldHandle, entity: EntityHandle) -> Vec2: ...
```

During interpretation, the symbol resolves to a callable registered by the
generated nanobind module. During AOT lowering, the call remains an external C
symbol resolved by the native or Emscripten linker.

## C++ API schema

One backend-neutral schema is the source of truth for the example API. The
schema records:

- API name and schema version;
- function names and versioned C symbols;
- fixed-width argument and result types;
- struct fields and physical layout;
- opaque handle types;
- `pure`, `read`, `write`, or `io` effects;
- status/error behavior.

The generator produces:

1. a C header with `extern "C"` declarations;
2. C++ thunks that adapt the engine implementation to the stable C ABI;
3. a nanobind module for development execution;
4. Vernon extern declarations and compiler metadata;
5. a JSON manifest and Python `.pyi` declarations.

Generated files are committed and checked for drift. The generator must be
deterministic.

### ABI rules

The first ABI supports:

- `bool`, fixed-width integers, `f32`, and `f64`;
- fixed vectors and matrices with declared layout;
- ABI-stable value structs;
- enums with fixed underlying types;
- opaque 64-bit handles;
- explicit UTF-8 or byte spans only if pointer/length ownership is declared.

The first ABI excludes:

- C++ references, class pointers, vtables, and name-mangled symbols;
- `std::string`, `std::vector`, and other STL types;
- Python objects and nanobind objects;
- exceptions crossing the C boundary;
- implicit allocation or ownership transfer.

C++ thunks catch exceptions and convert them to explicit status/error results.
C++ object lifetime remains owned by the host implementation.

Direct versioned C symbols are sufficient for the first demo. A versioned
function table with `struct_size`, context pointer, and reserved slots can be
added later for plugin hot reload without changing Host source semantics.

## Development interpreter

The interpreter consumes `TypedFunctionInstance` rather than executing the
original Python function body. This preserves the same:

- scalar casts and overflow behavior;
- struct construction and field access;
- branch and loop semantics;
- function-call resolution;
- effect validation;
- diagnostics;
- extern signatures.

The interpreter registry maps schema-qualified C symbols to generated nanobind
callables. The registry rejects missing symbols, schema mismatches, incompatible
signatures, and undeclared effects before execution.

The initial interpreter is correctness-oriented. ORC JIT may later compile hot
Host functions, but it must consume the same typed Host IR and extern metadata.

## Host AOT

### Host MLIR

Host entries lower to ordinary `func`/Vernon MLIR with a distinct
`vernon.host_entry` attribute. Extern declarations carry symbol, schema, and
effect attributes. Host validation rejects device-only types and operations
before lowering.

The Host lowering lane must not reuse compute workgroup wrappers. It produces a
versioned Host entry wrapper and reflection containing:

- exported symbol;
- target triple and object format;
- schema hash;
- argument and result layouts;
- imported C symbols and effects.

### Invocation ABI

Use a pointer-width-safe Host invocation contract rather than the existing
64-bit CPU compute wrapper. The C struct is compiled for the destination target
and uses explicit sizes for values and buffers.

The Host compiler must validate target data layout before emitting wrappers.
Native and wasm32 artifacts use identical logical arguments but target-specific
pointer layout.

### Desktop

An empty target triple selects the compiler host. Explicit desktop triples
produce PIC `.o` or `.obj` artifacts. The demo links the object and generated
C++ thunks into a small native executable.

Development ORC JIT is optional and host-only. Cross-compiled objects do not
create an ORC execution state.

### Browser WebAssembly

The compiler emits a relocatable object for
`wasm32-unknown-emscripten`. It does not attempt to load or execute that object
through the native Runtime.

Requirements:

- build the pinned LLVM with the WebAssembly target enabled;
- remove 64-bit assumptions only from the new Host ABI path;
- set the wasm target triple and data layout before LLVM conversion;
- preserve imported and exported versioned C symbols;
- use a pinned Emscripten toolchain for final linking;
- keep libc and browser dependencies outside the Host object where possible.

`em++` links the Host object, generated C++ thunks, and browser host into the
final `.wasm` and JavaScript loader. Python and nanobind are not linked into the
browser artifact.

## Demo

The demo owns a small C++ world containing entities with position and velocity.
The schema exposes:

- create or initialize world state;
- get position;
- get velocity;
- set position;
- emit an event or log record;
- compute or return a state checksum.

The Host DSL updates one entity inside fixed bounds and calls only these C++
functions. A fixed input sequence produces a deterministic final position,
event count, and checksum.

### Development mode

- Build the generated nanobind demo module.
- Load the same Host DSL source.
- Run it through the typed interpreter.
- Record the expected trajectory and checksum.

### Desktop deployment

- Compile the same Host DSL source to the current native triple.
- Link the generated object with the same C++ world and thunks.
- Run a standalone executable with no Python dependency.
- Compare the complete state and checksum with interpreted execution.

### Browser deployment

- Compile the same Host DSL source to `wasm32-unknown-emscripten`.
- Link it with the same C++ world and thunks through `em++`.
- Animate the entity on an HTML canvas.
- Expose completion, state, and checksum to JavaScript.
- Run a headless browser test and compare the checksum with desktop.

## Implementation phases

### Phase 1: contract and schema

- finalize Host syntax, types, effects, diagnostics, and artifact metadata;
- implement schema parser and deterministic generators;
- generate the demo C ABI, C++ thunks, extern declarations, nanobind wrapper,
  manifest, and stubs;
- add schema, generated-file, and C-header compile tests.

### Phase 2: typed Host interpreter

- add `@vd.host` and `@vd.extern`;
- extend module graph, call graph, type inference, and effects;
- implement the Typed Host IR interpreter and extern registry;
- run the gameplay demo through nanobind;
- add interpreter and diagnostic tests.

### Phase 3: native Host AOT

- add Host MLIR entry and extern attributes;
- add Host validation and target-independent lowering;
- emit native object, reflection, and pointer-width-safe wrappers;
- link and run the standalone desktop demo;
- add interpreter/native parity tests.

### Phase 4: wasm32 AOT

- enable the LLVM WebAssembly target;
- emit Emscripten-compatible relocatable objects;
- add pinned Emscripten discovery and explicit build errors;
- link the browser demo and expose state/checksum to JavaScript;
- add object import/export and headless browser tests.

### Phase 5: integration and hardening

- add generated-file checks, formatting, docs, and CMake install rules;
- run existing compiler, Runtime, Python, and wheel regressions;
- add optional Emscripten CI without making Emscripten a normal build
  dependency;
- document ABI compatibility and extension rules.

## Acceptance

The first Host language milestone is complete when:

- one schema generates every development and deployment binding;
- one Host DSL source runs without modification in interpreted, native, and
  browser modes;
- interpreter and desktop produce identical complete state and checksum;
- browser produces the same checksum under headless execution;
- desktop and browser artifacts contain no Python or nanobind dependency;
- invalid Host/device calls, ABI types, missing symbols, effect mismatches, and
  schema mismatches fail deterministically;
- existing device compiler, CPU compute ABI, Runtime, and wheel tests remain
  green.
