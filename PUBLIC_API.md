# Public API policy

VernonDSL 0.1.2 treats the following installed interfaces as public:

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
- canonical Program bundles and artifacts accepted by compiler contract 13 and
  pipeline contract 17.

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
are stable once published in a 0.1 release. Unreleased API drafts may be
replaced without compatibility wrappers before their first release.

Execution is submission-based. Runtime pipelines use
`vernonRuntimeProgramSubmit`, and RHI command encoders are consumed by
`vernonRhiDeviceSubmit`. A submission may already be
complete; callers use its state query or `wait()` method. C callers explicitly
destroy submission/completion handles, while C++ and Python submissions use
managed lifetime. Submission destruction and device shutdown drain unfinished
work before releasing retained resources.

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

The 0.1.2 contract does not guarantee concurrent execution or multiple frames
in flight. Backends may complete work inline while preserving the same
submission and lifetime semantics. Swapchain presentation remains outside the
public Runtime contract.
