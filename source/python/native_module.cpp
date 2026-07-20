#include "VernonCompiler.h"
#include "VernonRuntime.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
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

  nb::tuple compileProgram(const std::string &mlir, VernonTarget target,
                           uint32_t glslVersion) {
    VernonCompileOptions options{};
    options.struct_size = sizeof(options);
    options.glsl_version = glslVersion;
    VernonCompileResult *raw = vernonCompilerCompileMlirWithOptions(
        context, mlir.data(), mlir.size(), target, &options);
    std::unique_ptr<VernonCompileResult, decltype(&vernonCompileResultDestroy)>
        result(raw, &vernonCompileResultDestroy);
    if (!result ||
        vernonCompileResultGetStatus(result.get()) != VERNON_STATUS_OK)
      throw std::runtime_error(
          result ? stringView(vernonCompileResultGetDiagnostics(result.get()))
                 : "compiler returned no result");
    nb::list artifacts;
    for (size_t index = 0;
         index < vernonCompileResultGetArtifactCount(result.get()); ++index) {
      VernonStringView name =
          vernonCompileResultGetArtifactName(result.get(), index);
      VernonStringView data =
          vernonCompileResultGetArtifactData(result.get(), index);
      artifacts.append(
          nb::make_tuple(stringView(name), nb::bytes(data.data, data.size)));
    }
    return nb::make_tuple(
        artifacts, stringView(vernonCompileResultGetReflection(result.get())));
  }

  VernonCompilerContext *context{};
};

struct Runtime;
struct Texture;
struct LoadedProgram;

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

struct Texture {
  Texture(Runtime *owner, uint32_t width, uint32_t height);
  ~Texture() { vernonRuntimeTextureFree(handle); }

  void upload(const nb::bytes &data) {
    if (vernonRuntimeTextureCopyFromHost(handle, data.c_str(), data.size()) !=
        VERNON_STATUS_OK)
      throw std::runtime_error("runtime texture upload failed");
  }

  nb::bytes download() const {
    std::string data(static_cast<size_t>(width) * height * 4, '\0');
    if (vernonRuntimeTextureCopyToHost(handle, data.data(), data.size()) !=
        VERNON_STATUS_OK)
      throw std::runtime_error("runtime texture readback failed");
    return nb::bytes(data.data(), data.size());
  }

  Runtime *owner{};
  VernonDeviceTexture *handle{};
  uint32_t width{};
  uint32_t height{};
};

struct LoadedProgram {
  LoadedProgram(Runtime *owner, VernonRuntimeContext *runtimeHandle,
                VernonLoadedProgram *handle)
      : owner(owner), runtimeHandle(runtimeHandle), handle(handle) {}
  ~LoadedProgram() { vernonRuntimeProgramUnload(handle); }

  void draw(Texture *target, const nb::list &attachmentValues,
            const nb::list &values, const nb::list &uniformValues,
            uint32_t vertexCount, uint32_t instanceCount, Buffer *indexBuffer,
            uint32_t indexCount, size_t indexOffset,
            VernonPrimitiveTopology topology,
            const std::tuple<float, float, float, float> &clear) {
    if (target && target->owner != owner)
      throw std::runtime_error("target belongs to another runtime");
    std::vector<VernonColorAttachment> attachments;
    attachments.reserve(attachmentValues.size());
    for (nb::handle value : attachmentValues) {
      nb::tuple row = nb::cast<nb::tuple>(value);
      if (row.size() != 2)
        throw std::runtime_error("color attachment requires two fields");
      Texture *texture = nb::cast<Texture *>(row[1]);
      if (!texture || texture->owner != owner)
        throw std::runtime_error("color attachment belongs to another runtime");
      attachments.push_back({nb::cast<uint32_t>(row[0]), texture->handle});
    }
    VernonIndexBinding index{};
    if (indexBuffer) {
      if (indexBuffer->owner != owner)
        throw std::runtime_error("index buffer belongs to another runtime");
      index = {indexBuffer->handle, VERNON_INDEX_U32, indexOffset, indexCount};
    }
    std::vector<VernonDrawBinding> bindings;
    bindings.reserve(values.size());
    for (nb::handle value : values) {
      nb::tuple row = nb::cast<nb::tuple>(value);
      if (row.size() != 6)
        throw std::runtime_error("draw binding requires six fields");
      Buffer *buffer = nb::cast<Buffer *>(row[1]);
      if (!buffer || buffer->owner != owner)
        throw std::runtime_error("draw buffer belongs to another runtime");
      bindings.push_back({
          nb::cast<uint32_t>(row[0]),
          buffer->handle,
          nb::cast<uint32_t>(row[2]),
          nb::cast<uint32_t>(row[3]),
          nb::cast<size_t>(row[4]),
          nb::cast<uint32_t>(row[5]),
      });
    }
    std::vector<VernonUniformBinding> uniforms;
    std::vector<std::string> uniformNames;
    std::vector<std::vector<float>> uniformData;
    uniforms.reserve(uniformValues.size());
    uniformNames.reserve(uniformValues.size());
    uniformData.reserve(uniformValues.size());
    for (nb::handle value : uniformValues) {
      nb::tuple row = nb::cast<nb::tuple>(value);
      if (row.size() != 3)
        throw std::runtime_error("uniform binding requires three fields");
      uniformNames.push_back(nb::cast<std::string>(row[0]));
      nb::bytes bytes = nb::cast<nb::bytes>(row[1]);
      uint32_t count = nb::cast<uint32_t>(row[2]);
      if (bytes.size() != static_cast<size_t>(count) * sizeof(float))
        throw std::runtime_error(
            "uniform byte size does not match value count");
      uniformData.emplace_back(count);
      std::memcpy(uniformData.back().data(), bytes.c_str(), bytes.size());
      uniforms.push_back(
          {uniformNames.back().c_str(), uniformData.back().data(), count});
    }
    VernonDrawDescription description{};
    description.struct_size = sizeof(description);
    description.target = target ? target->handle : nullptr;
    description.bindings = bindings.data();
    description.binding_count = bindings.size();
    description.uniforms = uniforms.data();
    description.uniform_count = uniforms.size();
    description.vertex_count = vertexCount;
    description.instance_count = instanceCount;
    description.index_binding = indexBuffer ? &index : nullptr;
    description.color_attachments =
        attachments.empty() ? nullptr : attachments.data();
    description.color_attachment_count = attachments.size();
    description.topology = topology;
    description.clear_color[0] = std::get<0>(clear);
    description.clear_color[1] = std::get<1>(clear);
    description.clear_color[2] = std::get<2>(clear);
    description.clear_color[3] = std::get<3>(clear);
    if (vernonRuntimeDraw(handle, &description) != VERNON_STATUS_OK)
      throw std::runtime_error(
          "graphics draw failed: " +
          stringView(vernonRuntimeGetLastError(runtimeHandle)));
  }

  Runtime *owner{};
  VernonRuntimeContext *runtimeHandle{};
  VernonLoadedProgram *handle{};
};

struct Runtime {
  explicit Runtime(VernonRuntimeBackend backend, uint16_t apiMajor = 0,
                   uint16_t apiMinor = 0) {
    VernonRuntimeCreateOptions options{};
    options.struct_size = sizeof(options);
    options.api_version_major = apiMajor;
    options.api_version_minor = apiMinor;
    handle = vernonRuntimeCreateWithOptions(backend, &options);
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

  std::unique_ptr<Texture> createTexture(uint32_t width, uint32_t height) {
    return std::make_unique<Texture>(this, width, height);
  }

  std::unique_ptr<LoadedProgram> loadGraphics(const nb::bytes &vertex,
                                              const nb::bytes &fragment) {
    VernonLoadedProgram *program = vernonRuntimeProgramLoadGraphics(
        handle, {vertex.c_str(), vertex.size()},
        {fragment.c_str(), fragment.size()});
    if (!program)
      throw std::runtime_error("cannot load graphics program: " +
                               stringView(vernonRuntimeGetLastError(handle)));
    return std::make_unique<LoadedProgram>(this, handle, program);
  }

  void barrier() {
    if (vernonRuntimeComputeToGraphicsBarrier(handle) != VERNON_STATUS_OK)
      throw std::runtime_error("compute-to-graphics barrier failed");
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

Texture::Texture(Runtime *owner, uint32_t width, uint32_t height)
    : owner(owner), width(width), height(height) {
  handle = vernonRuntimeTextureCreate2D(owner->handle, width, height,
                                        VERNON_TEXTURE_RGBA8_UNORM);
  if (!handle)
    throw std::runtime_error("runtime texture allocation failed");
}

} // namespace

NB_MODULE(_native, module) {
  module.doc() = "VernonDSL native compiler and kernel runtime bindings";
  nb::enum_<VernonTarget>(module, "Target")
      .value("CPU", VERNON_TARGET_CPU)
      .value("CUDA", VERNON_TARGET_CUDA)
      .value("VULKAN", VERNON_TARGET_VULKAN)
      .value("METAL", VERNON_TARGET_METAL)
      .value("OPENGL", VERNON_TARGET_OPENGL)
      .value("OPENGL_ES", VERNON_TARGET_OPENGL_ES);
  nb::enum_<VernonRuntimeBackend>(module, "RuntimeBackend")
      .value("CPU", VERNON_RUNTIME_CPU)
      .value("CUDA", VERNON_RUNTIME_CUDA)
      .value("VULKAN", VERNON_RUNTIME_VULKAN)
      .value("OPENGL", VERNON_RUNTIME_OPENGL)
      .value("OPENGL_ES", VERNON_RUNTIME_OPENGL_ES);
  nb::enum_<VernonPrimitiveTopology>(module, "PrimitiveTopology")
      .value("TRIANGLE_LIST", VERNON_TOPOLOGY_TRIANGLE_LIST)
      .value("LINE_LIST", VERNON_TOPOLOGY_LINE_LIST)
      .value("POINT_LIST", VERNON_TOPOLOGY_POINT_LIST);
  nb::class_<Compiler>(module, "Compiler")
      .def(nb::init<>())
      .def("compile", &Compiler::compile)
      .def("compile_program", &Compiler::compileProgram, nb::arg("mlir"),
           nb::arg("target"), nb::arg("glsl_version") = 0);
  nb::class_<Runtime>(module, "Runtime")
      .def(nb::init<VernonRuntimeBackend, uint16_t, uint16_t>(),
           nb::arg("backend"), nb::arg("api_major") = 0,
           nb::arg("api_minor") = 0)
      .def("allocate", &Runtime::allocate, nb::keep_alive<0, 1>())
      .def("load", &Runtime::load, nb::keep_alive<0, 1>())
      .def("create_texture", &Runtime::createTexture, nb::keep_alive<0, 1>())
      .def("load_graphics", &Runtime::loadGraphics, nb::keep_alive<0, 1>())
      .def("barrier", &Runtime::barrier)
      .def("synchronize", &Runtime::synchronize);
  nb::class_<Buffer>(module, "Buffer")
      .def("upload", &Buffer::upload)
      .def("download", &Buffer::download);
  nb::class_<LoadedKernel>(module, "LoadedKernel")
      .def("launch", &LoadedKernel::launch);
  nb::class_<Texture>(module, "Texture")
      .def_prop_ro("width", [](const Texture &value) { return value.width; })
      .def_prop_ro("height", [](const Texture &value) { return value.height; })
      .def("upload", &Texture::upload)
      .def("download", &Texture::download);
  nb::class_<LoadedProgram>(module, "LoadedProgram")
      .def("draw", &LoadedProgram::draw, nb::arg("target"),
           nb::arg("attachments"), nb::arg("bindings"), nb::arg("uniforms"),
           nb::arg("vertex_count"), nb::arg("instance_count") = 1,
           nb::arg("index_buffer") = nullptr, nb::arg("index_count") = 0,
           nb::arg("index_offset") = 0,
           nb::arg("topology") = VERNON_TOPOLOGY_TRIANGLE_LIST,
           nb::arg("clear") = std::make_tuple(0.0f, 0.0f, 0.0f, 0.0f));
  module.def("runtime_available", [](VernonRuntimeBackend backend) {
    return vernonRuntimeGetCapabilities(backend).available != 0;
  });
}
