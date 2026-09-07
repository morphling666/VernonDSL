# Browser WebAssembly CPU Runtime

Status: current wasm32 CPU deployment architecture.

Vernon's first browser target is `wasm32-unknown-emscripten`. The compiler emits relocatable
WebAssembly objects, the cooker keeps those objects separate from the manifest, and the application
links them together with the static Runtime and generated registration source.

## Architecture

The browser path uses the same compiler, Program, and CPU entry contracts:

1. Compile each CPU stage to `module.wasm.o`.
2. Cook the object as `artifacts/<sha256>.wasm.o` and emit
   `vernon_cpu_registration_<hash>.c`.
3. Link the object, registration source, and `VernonRuntime` into the Emscripten application.
4. Load the cooked manifest from application memory with
   `vernonRuntimeLoadProgramBundleWithOptions(..., nullptr)`.
5. Resolve the stage through the static CPU entry registry. The web profile never opens the
   artifact path or calls a dynamic-library API.

The external artifact descriptor remains in the manifest as build identity and deployment metadata.
For statically linked objects, integrity is established by the application link step; executable
bytes cannot be re-hashed as a standalone object at Runtime load time.

## Build and cook

Build LLVM with the WebAssembly target enabled. A host build of the compiler performs cooking:

```sh
python -m vernon_dsl.program_asset_cli \
  path/to/pipelines.py:asset \
  --target cpu \
  --cpu-triple wasm32-unknown-emscripten \
  -o cooked
```

Configure the browser Runtime with Emscripten:

```sh
emcmake cmake -S source/lib/runtime -B wasm-build \
  -DVERNON_RUNTIME_PROFILE=web \
  -DVERNON_RUNTIME_LIBRARY_TYPE=STATIC \
  -DBUILD_TESTING=OFF
emcmake cmake --build wasm-build
```

The final application link must include:

- `wasm-build/libVernonRuntime.a` and its installed static Runtime dependencies;
- every `cooked/artifacts/*.wasm.o` used by the application;
- every generated `cooked/vernon_cpu_registration_*.c` source;
- the application code that supplies manifest JSON bytes.

[`examples/external_engine/`](../../examples/external_engine/) provides a minimal
cross-platform C++ application and CMake project that assembles exactly these
inputs. The same engine source builds natively or through `emcmake`.

## Execution policy

The initial web profile uses the existing CPU range-phase and barrier scheduler with its
calling-thread execution policy. It creates no `std::thread` worker pool and therefore does not
require Emscripten `-pthread`, `SharedArrayBuffer`, COOP/COEP headers, or Worker pool setup.

A future pthread-enabled web configuration can select the existing worker-pool policy after those
deployment requirements are explicit. It does not require a second dispatcher implementation.

## Current verification

The regular compiler tests emit and inspect a real WebAssembly object. Web-profile Runtime tests
load and resolve multiple statically registered Programs without a bundle directory, and exercise
the calling-thread scheduler. The external-engine example loads both bundles in one executable:
the animated `examples/fractal.py` kernel runs through a CPU Execution Graph in the left panel while
an RHI Execution Graph renders Mandelbulb through desktop OpenGL or Emscripten WebGL2 in the right
panel. The web build embeds integrity-checked GLSL ES stages and the statically linked CPU object,
and retains a CPU-only headless Node checksum smoke.
