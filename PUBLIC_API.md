# Public API policy

VernonDSL treats the following installed interfaces as public:

- names exported by `vernon_dsl.__all__`;
- `vernon_dsl.runtime_source` helpers for locating the bundled Runtime source;
- the `vernon-compile-python` and `vernon-cook-program` command-line tools;
- application-facing C declarations in `VernonCommon.h`,
  `VernonGraphicsState.h`, `VernonOpenGLContext.h`, `VernonRHI.h`,
  `VernonRuntime.h`, `VernonVersions.h`,
  `vernon-c/Common.h`, and `vernon-c/Runtime.h`;
- the `VernonRHI.hpp` and `VernonRuntime.hpp` C++ wrappers;
- the standalone CMake project bundled under `vernon_dsl/runtime_src` and its
  documented Runtime targets;
- canonical Program bundles and artifacts accepted by the current Compiler
  Contract and Program Version.

The wheel does not install a prebuilt `lib/cmake/VernonRuntime` package into the
environment. Embedders locate `vernon_dsl/runtime_src` and configure that
directory as a standalone CMake source project. A prefix installation using
the `VernonDevelopment` component has a different development layout and is
not part of the PyPI wheel contract.

`VernonRuntimeCore.h`, `VernonRuntimeProvider.h`, and
`VernonRuntimeRHIAdapter.h` are shipped embedder SPI used to implement backend
providers. They are not application-facing stable API and may change when the
compiler contract or Program version changes. Compiler C headers installed only by the
development component are likewise outside the wheel's stable API.

The native `vernon-compile` executable and compiler shared library are bundled
to implement the installed Python tools. They are not separately supported
command-line or native-link interfaces. Python modules, native symbols,
undocumented implementation sources and CMake targets under
`vernon_dsl/runtime_src`, test hooks, and repository build scripts are internal
unless another public document explicitly says otherwise.

## Stability

Version 0.1.1 was preliminary and is not an API compatibility baseline for the
canonical Program architecture. Version 0.1.2 removes its direct pipeline/Stage
execution and public ExecutionGraph authoring surfaces without compatibility
wrappers. Beginning with 0.1.2, patch releases preserve documented Python
behavior and the application-facing C ABI described above.

Runtime embedders rebuild from the bundled source for each VernonDSL release;
ABI stability does not make binaries built from mixed release sources or
headers compatible. Additive APIs may be introduced in minor releases. APIs
first published in 0.1.2 or later are deprecated for at least one minor release
before removal unless continued support would create a security or correctness
defect.

Struct-based C APIs use `struct_size` and reserved fields for compatible
extension. Callers must zero-initialize structures, set `struct_size`, and leave
reserved fields zero. Enum numeric values and exported C function signatures
are stable once published in a release. Unreleased API drafts may be
replaced without compatibility wrappers before their first release.

Canonical Program execution uses bundle → executable → instance → invocation
→ bind → forward.
`vernonRuntimeProgramInvocationForward` records, submits, and completes the
resolved Program plan inside the Runtime; a Program caller does not provide a
command encoder or submit descriptor. Version 0.1.2 has no public direct-Stage
loader, binding, submit, or encode facility and no public ExecutionGraph
authoring model. Stage objects and the Command DAG are private post-resolution
runtime implementation.

`VernonProgramGraph` is the public pre-resolution composition builder. It adds
loaded cooked Program bundles as node-scoped components, connects compatible
symbolic graph Values and ordered Storage-version chains, and resolves to the
same `VernonProgramExecutable` lifecycle. Node boundary and graphics-control
tokens are stable and scoped. Graph Storage handles bind a shared resource
once. Global resolve emits static graphics fusion-candidate regions;
invocation materializes only their resolved fused or split paths. Only
graph-level invocation forward submits work. ProgramGraph does not accept
resolved executables, raw Stages, native resources, encoders, or callbacks.

Borrowed Vulkan and DirectX 12 command targets are queued by their external
owner. Their completions remain pending until that owner has observed its GPU
fence and calls `vernonRhiCompletionSignal`; retained resources are not
released before that signal. All borrowed completions must be signaled before
destroying the Vernon RHI device.

Python `Kernel(...)` and graphics `Pipeline(...)` calls are synchronous
convenience operations. Module execution and VJP are resolved from canonical
Program; the native Command DAG is not exposed as a Python pass API.

Sparse host updates use `vernonRhiDeviceUploadBufferRanges`, which validates a
complete range list before mutation and lets each backend execute the list as
one transfer transaction.

The current public contract does not guarantee concurrent execution or multiple frames
in flight. Backends may complete work inline while preserving the same
submission and lifetime semantics. Swapchain presentation remains outside the
public Runtime contract.
