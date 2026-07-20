#include "VernonCompiler.h"
#include "VernonRuntime.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace {

std::string stringView(VernonStringView view) {
  return view.data ? std::string(view.data, view.size) : std::string();
}

struct Compiler {
  Compiler() : context(vernonCompilerCreate()) {
    if (!context)
      throw std::runtime_error("cannot create Vernon compiler");
  }
  ~Compiler() { vernonCompilerDestroy(context); }

  nb::tuple compile(const std::string &mlir, VernonTarget target) {
    VernonCompileResult *raw =
        vernonCompilerCompileMlir(context, mlir.data(), mlir.size(), target);
    std::unique_ptr<VernonCompileResult, decltype(&vernonCompileResultDestroy)>
        result(raw, &vernonCompileResultDestroy);
    if (!result ||
        vernonCompileResultGetStatus(result.get()) != VERNON_STATUS_OK)
      throw std::runtime_error(
          result ? stringView(vernonCompileResultGetDiagnostics(result.get()))
                 : "compiler returned no result");
    if (vernonCompileResultGetArtifactCount(result.get()) != 1)
      throw std::runtime_error("kernel compilation must produce one artifact");
    VernonStringView artifact =
        vernonCompileResultGetArtifactData(result.get(), 0);
    return nb::make_tuple(
        nb::bytes(artifact.data, artifact.size),
        stringView(vernonCompileResultGetReflection(result.get())));
  }

  VernonCompilerContext *context{};
};

struct Runtime;

struct Buffer {
  Buffer(Runtime *owner, size_t size, size_t alignment);
  ~Buffer() { vernonRuntimeBufferFree(handle); }

  void upload(const nb::bytes &data) {
    if (data.size() != size)
      throw std::runtime_error("upload size does not match buffer");
    if (vernonRuntimeCopyFromHost(handle, 0, data.c_str(), data.size()) !=
        VERNON_STATUS_OK)
      throw std::runtime_error("runtime buffer upload failed");
  }

  nb::bytes download() const {
    std::string data(size, '\0');
    if (vernonRuntimeCopyToHost(handle, 0, data.data(), data.size()) !=
        VERNON_STATUS_OK)
      throw std::runtime_error("runtime buffer download failed");
    return nb::bytes(data.data(), data.size());
  }

  Runtime *owner{};
  VernonDeviceBuffer *handle{};
  size_t size{};
};

struct LoadedKernel {
  LoadedKernel(Runtime *owner, VernonLoadedKernel *handle)
      : owner(owner), handle(handle) {}
  ~LoadedKernel() { vernonRuntimeKernelUnload(handle); }

  void launch(uint32_t x, uint32_t y, uint32_t z, const nb::list &values) {
    std::vector<VernonLaunchArgument> arguments;
    std::vector<std::string> scalars;
    arguments.reserve(values.size());
    scalars.reserve(values.size());
    for (nb::handle value : values) {
      if (nb::isinstance<Buffer>(value)) {
        Buffer *buffer = nb::cast<Buffer *>(value);
        if (buffer->owner != owner)
          throw std::runtime_error("buffer belongs to another runtime");
        arguments.push_back({VERNON_LAUNCH_TENSOR, buffer->handle, nullptr, 0});
      } else if (nb::isinstance<nb::bytes>(value)) {
        nb::bytes bytes = nb::borrow<nb::bytes>(value);
        scalars.emplace_back(bytes.c_str(), bytes.size());
        arguments.push_back({VERNON_LAUNCH_SCALAR, nullptr,
                             scalars.back().data(), scalars.back().size()});
      } else {
        throw std::runtime_error("launch values must be Buffer or bytes");
      }
    }
    if (vernonRuntimeLaunch(handle, {x, y, z}, arguments.data(),
                            arguments.size()) != VERNON_STATUS_OK)
      throw std::runtime_error("native kernel launch failed");
  }

  Runtime *owner{};
  VernonLoadedKernel *handle{};
};

struct Runtime {
  explicit Runtime(VernonRuntimeBackend backend)
      : handle(vernonRuntimeCreate(backend, 0)) {
    if (!handle)
      throw std::runtime_error("requested runtime backend is unavailable");
  }
  ~Runtime() { vernonRuntimeDestroy(handle); }

  std::unique_ptr<Buffer> allocate(size_t size, size_t alignment) {
    return std::make_unique<Buffer>(this, size, alignment);
  }

  std::unique_ptr<LoadedKernel> load(const nb::bytes &artifact,
                                     const std::string &reflection,
                                     const std::string &entry) {
    VernonLoadedKernel *kernel = vernonRuntimeLoadArtifact(
        handle, artifact.c_str(), artifact.size(), reflection.data(),
        reflection.size(), entry.data(), entry.size());
    if (!kernel)
      throw std::runtime_error("cannot load native kernel: " +
                               stringView(vernonRuntimeGetLastError(handle)));
    return std::make_unique<LoadedKernel>(this, kernel);
  }

  void synchronize() {
    if (vernonRuntimeSynchronize(handle) != VERNON_STATUS_OK)
      throw std::runtime_error("runtime synchronization failed");
  }

  VernonRuntimeContext *handle{};
};

Buffer::Buffer(Runtime *owner, size_t size, size_t alignment)
    : owner(owner), size(size) {
  handle = vernonRuntimeBufferAllocate(owner->handle, size, alignment);
  if (!handle)
    throw std::runtime_error("runtime buffer allocation failed");
}

} // namespace

NB_MODULE(_native, module) {
  module.doc() = "VernonDSL native compiler and kernel runtime bindings";
  nb::enum_<VernonTarget>(module, "Target")
      .value("CPU", VERNON_TARGET_CPU)
      .value("CUDA", VERNON_TARGET_CUDA)
      .value("VULKAN", VERNON_TARGET_VULKAN);
  nb::enum_<VernonRuntimeBackend>(module, "RuntimeBackend")
      .value("CPU", VERNON_RUNTIME_CPU)
      .value("CUDA", VERNON_RUNTIME_CUDA);
  nb::class_<Compiler>(module, "Compiler")
      .def(nb::init<>())
      .def("compile", &Compiler::compile);
  nb::class_<Runtime>(module, "Runtime")
      .def(nb::init<VernonRuntimeBackend>())
      .def("allocate", &Runtime::allocate, nb::keep_alive<0, 1>())
      .def("load", &Runtime::load, nb::keep_alive<0, 1>())
      .def("synchronize", &Runtime::synchronize);
  nb::class_<Buffer>(module, "Buffer")
      .def("upload", &Buffer::upload)
      .def("download", &Buffer::download);
  nb::class_<LoadedKernel>(module, "LoadedKernel")
      .def("launch", &LoadedKernel::launch);
  module.def("runtime_available", [](VernonRuntimeBackend backend) {
    return vernonRuntimeGetCapabilities(backend).available != 0;
  });
}
