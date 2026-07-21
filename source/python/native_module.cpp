#include "VernonCompiler.h"
#include "VernonRuntime.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstring>
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
                           uint32_t glslVersion,
                           const std::string &targetTriple,
                           const std::string &cpu,
                           const std::string &cpuFeatures) {
    VernonCompileOptions options{};
    options.struct_size = sizeof(options);
    options.glsl_version = glslVersion;
    options.cpu_target_triple =
        VernonStringView{targetTriple.data(), targetTriple.size()};
    options.cpu_name = VernonStringView{cpu.data(), cpu.size()};
    options.cpu_features =
        VernonStringView{cpuFeatures.data(), cpuFeatures.size()};
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
struct LoadedPipeline;

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

struct LoadedPipeline {
  LoadedPipeline(Runtime *owner, VernonRuntimeContext *runtime,
                 VernonPipelineBundle *bundle, VernonLoadedPipeline *pipeline)
      : owner(owner), runtime(runtime), bundle(bundle), pipeline(pipeline) {}
  ~LoadedPipeline() {
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(bundle);
  }

  void
  invoke(const nb::list &argumentValues, Buffer *indexBuffer,
         uint32_t indexCount, size_t indexOffset,
         const nb::list &attachmentValues, uint32_t topology,
         uint32_t vertexCount, uint32_t instanceCount,
         const std::tuple<uint32_t, uint32_t, uint32_t> &computeGrid,
         const std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> &viewport,
         const std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> &scissor) {
    std::vector<VernonPipelineArgument> arguments;
    std::vector<std::vector<uint64_t>> shapes;
    std::vector<std::vector<uint64_t>> strides;
    std::vector<std::string> inlineStorage;
    arguments.reserve(argumentValues.size());
    shapes.reserve(argumentValues.size());
    strides.reserve(argumentValues.size());
    inlineStorage.reserve(argumentValues.size());
    for (nb::handle value : argumentValues) {
      nb::tuple row = nb::cast<nb::tuple>(value);
      const std::string kind = nb::cast<std::string>(row[0]);
      VernonPipelineArgument argument{};
      if (kind == "tensor" && row.size() == 8) {
        Buffer *buffer = nb::cast<Buffer *>(row[2]);
        if (!buffer || buffer->owner != owner)
          throw std::runtime_error(
              "pipeline buffer belongs to another runtime");
        shapes.push_back(nb::cast<std::vector<uint64_t>>(row[5]));
        strides.push_back(nb::cast<std::vector<uint64_t>>(row[6]));
        argument.slot = nb::cast<uint32_t>(row[1]);
        argument.kind = VERNON_PIPELINE_TENSOR;
        argument.tensor = {
            buffer->handle,
            static_cast<VernonDataType>(nb::cast<uint32_t>(row[3])),
            static_cast<VernonValueAccess>(nb::cast<uint32_t>(row[4])),
            static_cast<uint32_t>(shapes.back().size()),
            shapes.back().data(),
            strides.back().data(),
            nb::cast<size_t>(row[7]),
        };
      } else if (kind == "inline" && row.size() == 5) {
        nb::bytes bytes = nb::cast<nb::bytes>(row[3]);
        inlineStorage.emplace_back(bytes.c_str(), bytes.size());
        shapes.push_back(nb::cast<std::vector<uint64_t>>(row[4]));
        argument.slot = nb::cast<uint32_t>(row[1]);
        argument.kind = VERNON_PIPELINE_INLINE_VALUE;
        argument.inline_value = {
            static_cast<VernonDataType>(nb::cast<uint32_t>(row[2])),
            static_cast<uint32_t>(shapes.back().size()),
            shapes.back().data(),
            inlineStorage.back().data(),
            inlineStorage.back().size(),
        };
      } else {
        throw std::runtime_error("invalid pipeline argument row");
      }
      arguments.push_back(argument);
    }
    std::vector<VernonColorAttachment> attachments;
    for (nb::handle value : attachmentValues) {
      nb::tuple row = nb::cast<nb::tuple>(value);
      Texture *texture = nb::cast<Texture *>(row[1]);
      if (row.size() != 2 || !texture || texture->owner != owner)
        throw std::runtime_error("invalid color attachment");
      attachments.push_back({nb::cast<uint32_t>(row[0]), texture->handle});
    }
    VernonIndexBinding index{};
    if (indexBuffer)
      index = {indexBuffer->handle, VERNON_INDEX_U32, indexOffset, indexCount};
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
    invocation.arguments = arguments.data();
    invocation.argument_count = arguments.size();
    invocation.index_binding = indexBuffer ? &index : nullptr;
    invocation.color_attachments = attachments.data();
    invocation.color_attachment_count = attachments.size();
    invocation.topology = static_cast<VernonPrimitiveTopology>(topology);
    invocation.vertex_count = vertexCount;
    invocation.instance_count = instanceCount;
    invocation.compute_grid = {std::get<0>(computeGrid),
                               std::get<1>(computeGrid),
                               std::get<2>(computeGrid)};
    const uint32_t viewportValues[] = {
        std::get<0>(viewport), std::get<1>(viewport), std::get<2>(viewport),
        std::get<3>(viewport)};
    const uint32_t scissorValues[] = {
        std::get<0>(scissor), std::get<1>(scissor), std::get<2>(scissor),
        std::get<3>(scissor)};
    std::memcpy(invocation.viewport, viewportValues, sizeof(viewportValues));
    std::memcpy(invocation.scissor, scissorValues, sizeof(scissorValues));
    if (vernonRuntimePipelineInvoke(pipeline, &invocation) != VERNON_STATUS_OK)
      throw std::runtime_error("pipeline invocation failed: " +
                               stringView(vernonRuntimeGetLastError(runtime)));
  }

  Runtime *owner{};
  VernonRuntimeContext *runtime{};
  VernonPipelineBundle *bundle{};
  VernonLoadedPipeline *pipeline{};
};

struct Runtime {
  explicit Runtime(VernonRuntimeContext *handle) : handle(handle) {}
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

  static std::unique_ptr<Runtime>
  createExternalOpenGL(VernonRuntimeBackend backend, uintptr_t userData,
                       uintptr_t makeCurrent, uintptr_t getProcAddress,
                       uint16_t apiMajor, uint16_t apiMinor) {
    VernonExternalOpenGLContext external{};
    external.struct_size = sizeof(external);
    external.user_data = reinterpret_cast<void *>(userData);
    external.make_current =
        reinterpret_cast<VernonOpenGLMakeCurrentFn>(makeCurrent);
    external.get_proc_address =
        reinterpret_cast<VernonOpenGLGetProcAddressFn>(getProcAddress);
    external.api_version_major = apiMajor;
    external.api_version_minor = apiMinor;
    VernonRuntimeContext *handle =
        vernonRuntimeCreateExternalOpenGLForBackend(backend, &external);
    if (!handle)
      throw std::runtime_error("external OpenGL context is invalid");
    return std::make_unique<Runtime>(handle);
  }

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

  std::unique_ptr<LoadedKernel>
  loadComputeBundle(const std::string &directory) {
    VernonLoadedKernel *kernel =
        vernonRuntimeLoadComputeBundle(handle, directory.c_str());
    if (!kernel)
      throw std::runtime_error("cannot load compute bundle: " +
                               stringView(vernonRuntimeGetLastError(handle)));
    return std::make_unique<LoadedKernel>(this, kernel);
  }

  std::unique_ptr<Texture> createTexture(uint32_t width, uint32_t height) {
    return std::make_unique<Texture>(this, width, height);
  }

  std::unique_ptr<LoadedPipeline>
  loadPipeline(const nb::bytes &data,
               const std::vector<std::string> &features) {
    VernonPipelineBundle *bundle =
        vernonRuntimeLoadPipelineBundle(handle, data.c_str(), data.size());
    if (!bundle)
      throw std::runtime_error("cannot load pipeline bundle: " +
                               stringView(vernonRuntimeGetLastError(handle)));
    std::vector<const char *> names;
    for (const std::string &feature : features)
      names.push_back(feature.c_str());
    VernonLoadedPipeline *pipeline =
        vernonRuntimeResolvePipeline(bundle, {names.data(), names.size()});
    if (!pipeline) {
      vernonRuntimePipelineBundleDestroy(bundle);
      throw std::runtime_error("cannot resolve pipeline bundle: " +
                               stringView(vernonRuntimeGetLastError(handle)));
    }
    return std::make_unique<LoadedPipeline>(this, handle, bundle, pipeline);
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
           nb::arg("target"), nb::arg("glsl_version") = 0,
           nb::arg("target_triple") = "", nb::arg("cpu") = "",
           nb::arg("cpu_features") = "");
  nb::class_<Runtime>(module, "Runtime")
      .def(nb::init<VernonRuntimeBackend, uint16_t, uint16_t>(),
           nb::arg("backend"), nb::arg("api_major") = 0,
           nb::arg("api_minor") = 0)
      .def_static("create_external_opengl", &Runtime::createExternalOpenGL,
                  nb::arg("backend"), nb::arg("user_data"),
                  nb::arg("make_current"), nb::arg("get_proc_address"),
                  nb::arg("api_major"), nb::arg("api_minor"))
      .def("allocate", &Runtime::allocate, nb::keep_alive<0, 1>())
      .def("load", &Runtime::load, nb::keep_alive<0, 1>())
      .def("load_compute_bundle", &Runtime::loadComputeBundle,
           nb::keep_alive<0, 1>())
      .def("create_texture", &Runtime::createTexture, nb::keep_alive<0, 1>())
      .def("load_pipeline", &Runtime::loadPipeline, nb::keep_alive<0, 1>())
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
  nb::class_<LoadedPipeline>(module, "LoadedPipeline")
      .def("invoke", &LoadedPipeline::invoke, nb::arg("arguments"),
           nb::arg("index_buffer") = nullptr, nb::arg("index_count") = 0,
           nb::arg("index_offset") = 0, nb::arg("attachments") = nb::list(),
           nb::arg("topology") =
               static_cast<uint32_t>(VERNON_TOPOLOGY_TRIANGLE_LIST),
           nb::arg("vertex_count") = 0, nb::arg("instance_count") = 1,
           nb::arg("compute_grid") = std::make_tuple(0u, 0u, 0u),
           nb::arg("viewport") = std::make_tuple(0u, 0u, 0u, 0u),
           nb::arg("scissor") = std::make_tuple(0u, 0u, 0u, 0u));
  module.attr("DATA_BOOL") = static_cast<uint32_t>(VERNON_DATA_BOOL);
  module.attr("DATA_I32") = static_cast<uint32_t>(VERNON_DATA_I32);
  module.attr("DATA_U32") = static_cast<uint32_t>(VERNON_DATA_U32);
  module.attr("DATA_F16") = static_cast<uint32_t>(VERNON_DATA_F16);
  module.attr("DATA_F32") = static_cast<uint32_t>(VERNON_DATA_F32);
  module.attr("DATA_F64") = static_cast<uint32_t>(VERNON_DATA_F64);
  module.attr("ACCESS_READ_WRITE") =
      static_cast<uint32_t>(VERNON_ACCESS_READ_WRITE);
  module.attr("TOPOLOGY_TRIANGLE_LIST") =
      static_cast<uint32_t>(VERNON_TOPOLOGY_TRIANGLE_LIST);
  module.attr("TOPOLOGY_LINE_LIST") =
      static_cast<uint32_t>(VERNON_TOPOLOGY_LINE_LIST);
  module.attr("TOPOLOGY_POINT_LIST") =
      static_cast<uint32_t>(VERNON_TOPOLOGY_POINT_LIST);
  module.def("runtime_available", [](VernonRuntimeBackend backend) {
    return vernonRuntimeGetCapabilities(backend).available != 0;
  });
}
