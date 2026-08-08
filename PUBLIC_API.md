# Public API policy

VernonDSL 0.1.1 treats the following installed interfaces as public:

- names exported by `vernon_dsl.__all__`;
- `vernon_dsl.runtime_source` helpers for locating the bundled Runtime source;
- the `vernon-compile-python` and `vernon-cook-pipeline` command-line tools;
- application-facing C declarations in `VernonCommon.h`,
  `VernonGraphicsState.h`, `VernonOpenGLContext.h`, `VernonRHI.h`,
  `VernonExecutionGraph.h`, `VernonRuntime.h`, `VernonVersions.h`,
  `vernon-c/Common.h`, and `vernon-c/Runtime.h`;
- the `VernonRHI.hpp` and `VernonRuntime.hpp` C++ wrappers;
- the standalone CMake project bundled under `vernon_dsl/runtime_src` and its
  documented Runtime targets;
- canonical `*.pipeline.json` manifests and artifacts accepted by compiler
  contract 10 and pipeline contract 14.

The wheel does not install a prebuilt `lib/cmake/VernonRuntime` package into the
environment. Embedders locate `vernon_dsl/runtime_src` and configure that
directory as a standalone CMake source project. A prefix installation using
the `VernonDevelopment` component has a different development layout and is
not part of the PyPI wheel contract.

`VernonRuntimeCore.h`, `VernonRuntimeProvider.h`, and
`VernonRuntimeRHIAdapter.h` are shipped embedder SPI used to implement backend
providers. They are not application-facing stable API and may change when the
compiler or pipeline contract changes. Compiler C headers installed only by the
development component are likewise outside the wheel's stable API.

The native `vernon-compile` executable and compiler shared library are bundled
to implement the installed Python tools. They are not separately supported
command-line or native-link interfaces. Python modules, native symbols,
undocumented implementation sources and CMake targets under
`vernon_dsl/runtime_src`, test hooks, and repository build scripts are internal
unless another public document explicitly says otherwise.

## Stability

Patch releases preserve documented Python behavior and the application-facing
C ABI described above. Runtime embedders rebuild from the bundled source for
each VernonDSL release; ABI stability does not make binaries built from mixed
release sources or headers compatible. Additive APIs may be introduced in
minor releases. A public API is deprecated for at least one minor release
before removal unless continued support would create a security or correctness
defect.

Struct-based C APIs use `struct_size` and reserved fields for compatible
extension. Callers must zero-initialize structures, set `struct_size`, and leave
reserved fields zero. Enum numeric values and exported C function signatures
are stable within the 0.1 release series.

The Runtime contract is synchronous. Invocation, owned RHI submission, and
`ExecutionGraph.execute()` complete backend work before returning. Asynchronous
dispatch, deferred execution, swapchain presentation, and multiple frames in
flight are not public 0.1.1 behavior.
