#include "VernonCompiler.h"
#include "VernonRuntime.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

std::string stringView(VernonStringView view) { return view.data ? std::string(view.data, view.size) : std::string(); }

struct Compiler {
    Compiler() : context(vernonCompilerCreate()) {
        if (!context)
            throw std::runtime_error("cannot create Vernon compiler");
    }
    ~Compiler() { vernonCompilerDestroy(context); }

    VernonCompilerContext *context{};
};

struct Runtime;
struct CompiledProgram;
struct Texture;
struct Sampler;
struct LoadedPipeline;
struct PipelineInvocationBuilder;

using SharedCompileResult = std::shared_ptr<VernonCompileResult>;

struct CompiledProgram {
    CompiledProgram(VernonCompileResult *result, VernonTarget target, uint32_t glslVersion, uint32_t hlslShaderModel,
                    std::string targetTriple, std::string cpu, std::string cpuFeatures)
        : result(result, &vernonCompileResultDestroy), target(target), glslVersion(glslVersion),
          hlslShaderModel(hlslShaderModel), targetTriple(std::move(targetTriple)), cpu(std::move(cpu)),
          cpuFeatures(std::move(cpuFeatures)) {
        if (!this->result)
            throw std::runtime_error("compiler returned no result");
    }

    bool ok() const { return vernonCompileResultGetStatus(result.get()) == VERNON_STATUS_OK; }

    VernonStatus status() const { return vernonCompileResultGetStatus(result.get()); }

    std::string diagnostics() const { return stringView(vernonCompileResultGetDiagnostics(result.get())); }

    std::string reflection() const { return stringView(vernonCompileResultGetReflection(result.get())); }

    nb::list artifacts() const {
        nb::list values;
        for (size_t index = 0; index < vernonCompileResultGetArtifactCount(result.get()); ++index) {
            VernonStringView name = vernonCompileResultGetArtifactName(result.get(), index);
            VernonStringView data = vernonCompileResultGetArtifactData(result.get(), index);
            values.append(nb::make_tuple(stringView(name), nb::bytes(data.data, data.size)));
        }
        return values;
    }

    bool hasCpuEntry(const std::string &entry) const {
        return vernonCompileResultGetCpuEntry(result.get(), entry.data(), entry.size()) != nullptr;
    }

    void requireSuccess() const {
        if (!ok()) {
            std::string message = diagnostics();
            throw std::runtime_error(message.empty() ? "compilation failed" : message);
        }
    }

    SharedCompileResult result;
    VernonTarget target;
    uint32_t glslVersion{};
    uint32_t hlslShaderModel{};
    std::string targetTriple;
    std::string cpu;
    std::string cpuFeatures;
};

std::unique_ptr<CompiledProgram> compileProgramResult(Compiler &compiler, const std::string &mlir, VernonTarget target,
                                                      uint32_t glslVersion, uint32_t hlslShaderModel,
                                                      const std::string &targetTriple, const std::string &cpu,
                                                      const std::string &cpuFeatures) {
    VernonCompileOptions options{};
    options.struct_size = sizeof(options);
    options.glsl_version = glslVersion;
    options.hlsl_shader_model = hlslShaderModel;
    options.cpu_target_triple = VernonStringView{targetTriple.data(), targetTriple.size()};
    options.cpu_name = VernonStringView{cpu.data(), cpu.size()};
    options.cpu_features = VernonStringView{cpuFeatures.data(), cpuFeatures.size()};
    return std::make_unique<CompiledProgram>(
        vernonCompilerCompileMlirWithOptions(compiler.context, mlir.data(), mlir.size(), target, &options), target,
        glslVersion, hlslShaderModel, targetTriple, cpu, cpuFeatures);
}

struct Buffer {
    Buffer(Runtime *owner, size_t size, size_t alignment);
    ~Buffer() { vernonRuntimeBufferFree(handle); }

    void upload(const nb::bytes &data) {
        if (data.size() != size)
            throw std::runtime_error("upload size does not match buffer");
        if (vernonRuntimeCopyFromHost(handle, 0, data.c_str(), data.size()) != VERNON_STATUS_OK)
            throw std::runtime_error("runtime buffer upload failed");
    }

    nb::bytes download() const {
        std::string data(size, '\0');
        if (vernonRuntimeCopyToHost(handle, 0, data.data(), data.size()) != VERNON_STATUS_OK)
            throw std::runtime_error("runtime buffer download failed");
        return nb::bytes(data.data(), data.size());
    }

    Runtime *owner{};
    VernonDeviceBuffer *handle{};
    size_t size{};
};

struct LoadedKernel {
    LoadedKernel(Runtime *owner, VernonLoadedKernel *handle, SharedCompileResult retainedResult = {})
        : owner(owner), handle(handle), retainedResult(std::move(retainedResult)) {}
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
                arguments.push_back({VERNON_LAUNCH_SCALAR, nullptr, scalars.back().data(), scalars.back().size()});
            } else {
                throw std::runtime_error("launch values must be Buffer or bytes");
            }
        }
        if (vernonRuntimeLaunch(handle, {x, y, z}, arguments.data(), arguments.size()) != VERNON_STATUS_OK)
            throw std::runtime_error("native kernel launch failed");
    }

    Runtime *owner{};
    VernonLoadedKernel *handle{};
    // ORC entry pointers are valid only while their compile result owns the JIT.
    SharedCompileResult retainedResult;
};

struct Texture {
    Texture(Runtime *owner, uint32_t width, uint32_t height);
    ~Texture() { vernonRuntimeTextureFree(handle); }

    void upload(const nb::bytes &data) {
        if (vernonRuntimeTextureCopyFromHost(handle, data.c_str(), data.size()) != VERNON_STATUS_OK)
            throw std::runtime_error("runtime texture upload failed");
    }

    nb::bytes download() const {
        std::string data(static_cast<size_t>(width) * height * 4, '\0');
        if (vernonRuntimeTextureCopyToHost(handle, data.data(), data.size()) != VERNON_STATUS_OK)
            throw std::runtime_error("runtime texture readback failed");
        return nb::bytes(data.data(), data.size());
    }

    Runtime *owner{};
    VernonDeviceTexture *handle{};
    uint32_t width{};
    uint32_t height{};
};

struct Sampler {
    Sampler(Runtime *owner, VernonDeviceSampler *handle) : owner(owner), handle(handle) {}
    ~Sampler() { vernonRuntimeSamplerFree(handle); }

    Runtime *owner{};
    VernonDeviceSampler *handle{};
};

struct PipelineParameterMetadata {
    uint32_t slot{};
    std::string name;
    VernonPipelineArgumentKind kind{};
    VernonDataType dtype{};
    VernonValueAccess access{};
    std::vector<uint64_t> shape;
};

struct PipelineOutputMetadata {
    std::string name;
    VernonPipelineArgumentKind kind{};
    VernonDataType dtype{};
    VernonValueAccess access{};
    std::vector<uint64_t> shape;
    uint32_t location{};
};

PipelineParameterMetadata parameterMetadata(const VernonPipelineParameterView &view) {
    PipelineParameterMetadata result;
    result.slot = view.slot;
    result.name = stringView(view.name);
    result.kind = view.kind;
    result.dtype = view.dtype;
    result.access = view.access;
    if (view.rank)
        result.shape.assign(view.static_shape, view.static_shape + view.rank);
    return result;
}

PipelineOutputMetadata outputMetadata(const VernonPipelineOutputView &view) {
    PipelineOutputMetadata result;
    result.name = stringView(view.name);
    result.kind = view.kind;
    result.dtype = view.dtype;
    result.access = view.access;
    result.location = view.location;
    if (view.rank)
        result.shape.assign(view.static_shape, view.static_shape + view.rank);
    return result;
}

struct PipelineInvocationBuilder {
    PipelineInvocationBuilder(Runtime *owner, VernonRuntimeContext *runtime, VernonLoadedPipeline *pipeline)
        : owner(owner), runtime(runtime), pipeline(pipeline) {}

    PipelineParameterMetadata resolveParameter(const nb::object &identifier) {
        VernonPipelineParameterView view{};
        if (nb::isinstance<nb::str>(identifier)) {
            const std::string name = nb::cast<std::string>(identifier);
            if (vernonRuntimeLoadedPipelineFindParameter(pipeline, {name.data(), name.size()}, &view) !=
                VERNON_STATUS_OK)
                throw std::invalid_argument("unknown pipeline parameter '" + name + "'");
            return parameterMetadata(view);
        }
        if (!nb::isinstance<nb::int_>(identifier))
            throw std::invalid_argument("pipeline parameter must be a name or slot");
        const uint32_t slot = nb::cast<uint32_t>(identifier);
        const size_t count = vernonRuntimeLoadedPipelineGetParameterCount(pipeline);
        for (size_t index = 0; index < count; ++index) {
            if (vernonRuntimeLoadedPipelineGetParameterByIndex(pipeline, index, &view) == VERNON_STATUS_OK &&
                view.slot == slot)
                return parameterMetadata(view);
        }
        throw std::invalid_argument("unknown pipeline parameter slot " + std::to_string(slot));
    }

    struct OwnedArgument {
        VernonPipelineArgument value{};
        std::vector<uint64_t> shape;
        std::vector<int64_t> strides;
        nb::object hostOwner;
    };

    OwnedArgument &addArgument(const PipelineParameterMetadata &parameter, VernonPipelineArgumentKind kind) {
        if (parameter.kind != kind)
            throw std::invalid_argument("pipeline parameter '" + parameter.name + "' has a different reflected kind");
        if (!slots.insert(parameter.slot).second)
            throw std::invalid_argument("pipeline parameter was already bound");
        arguments.emplace_back();
        OwnedArgument &result = arguments.back();
        result.value.slot = parameter.slot;
        result.value.kind = kind;
        return result;
    }

    static VernonDataType numpyDataType(const nb::object &array) {
        const std::string name = nb::cast<std::string>(array.attr("dtype").attr("name"));
        if (name == "bool")
            return VERNON_DATA_BOOL;
        if (name == "int32")
            return VERNON_DATA_I32;
        if (name == "uint32")
            return VERNON_DATA_U32;
        if (name == "float16")
            return VERNON_DATA_F16;
        if (name == "float32")
            return VERNON_DATA_F32;
        if (name == "float64")
            return VERNON_DATA_F64;
        throw std::invalid_argument("unsupported NumPy pipeline dtype '" + name + "'");
    }

    PipelineInvocationBuilder &hostTensor(const nb::object &identifier, const nb::object &array) {
        if (!nb::isinstance(array, nb::module_::import_("numpy").attr("ndarray")))
            throw std::invalid_argument("host tensor must be a NumPy ndarray");
        const PipelineParameterMetadata parameter = resolveParameter(identifier);
        OwnedArgument &argument = addArgument(parameter, VERNON_PIPELINE_TENSOR);
        argument.shape = nb::cast<std::vector<uint64_t>>(array.attr("shape"));
        const std::vector<int64_t> signedStrides = nb::cast<std::vector<int64_t>>(array.attr("strides"));
        if (signedStrides.size() != argument.shape.size())
            throw std::invalid_argument("NumPy Tensor shape/stride mismatch");
        argument.strides.reserve(signedStrides.size());
        const size_t elementSize = nb::cast<size_t>(array.attr("dtype").attr("itemsize"));
        size_t before = 0;
        size_t after = 0;
        for (size_t dimension = 0; dimension < signedStrides.size(); ++dimension) {
            if (!argument.shape[dimension])
                throw std::invalid_argument("NumPy Tensor dimensions must be positive");
            const int64_t stride = signedStrides[dimension];
            const uint64_t magnitude =
                stride < 0 ? static_cast<uint64_t>(-(stride + 1)) + 1 : static_cast<uint64_t>(stride);
            const uint64_t steps = argument.shape[dimension] - 1;
            if (magnitude > std::numeric_limits<size_t>::max() ||
                (steps && magnitude > std::numeric_limits<size_t>::max() / steps))
                throw std::invalid_argument("NumPy Tensor byte span overflows");
            const size_t extent = static_cast<size_t>(steps * magnitude);
            size_t &bound = stride < 0 ? before : after;
            if (extent > std::numeric_limits<size_t>::max() - bound)
                throw std::invalid_argument("NumPy Tensor byte span overflows");
            bound += extent;
            argument.strides.push_back(stride);
        }
        if (after > std::numeric_limits<size_t>::max() - before ||
            elementSize > std::numeric_limits<size_t>::max() - before - after)
            throw std::invalid_argument("NumPy Tensor byte span overflows");
        const size_t span = before + after + elementSize;
        const VernonDataType dtype = numpyDataType(array);
        if (dtype != parameter.dtype)
            throw std::invalid_argument("host tensor dtype does not match pipeline reflection");
        argument.hostOwner = array;
        argument.value.tensor.struct_size = sizeof(VernonTensorView);
        argument.value.tensor.storage = VERNON_TENSOR_HOST;
        argument.value.tensor.host_data =
            reinterpret_cast<const void *>(nb::cast<uintptr_t>(array.attr("ctypes").attr("data")) - before);
        argument.value.tensor.dtype = dtype;
        argument.value.tensor.access = parameter.access;
        argument.value.tensor.rank = static_cast<uint32_t>(argument.shape.size());
        argument.value.tensor.shape = argument.shape.data();
        argument.value.tensor.byte_strides = argument.strides.data();
        argument.value.tensor.byte_offset = before;
        argument.value.tensor.byte_size = span;
        return *this;
    }

    PipelineInvocationBuilder &deviceTensor(const nb::object &identifier, Buffer *buffer, uint32_t dtype,
                                            uint32_t access, const std::vector<uint64_t> &shape,
                                            const std::vector<int64_t> &strides, size_t offset) {
        if (!buffer || buffer->owner != owner)
            throw std::invalid_argument("pipeline buffer belongs to another runtime");
        if (shape.size() != strides.size() || shape.empty())
            throw std::invalid_argument("device tensor shape and strides must have equal non-zero rank");
        const PipelineParameterMetadata parameter = resolveParameter(identifier);
        OwnedArgument &argument = addArgument(parameter, VERNON_PIPELINE_TENSOR);
        argument.shape = shape;
        argument.strides = strides;
        argument.value.tensor.struct_size = sizeof(VernonTensorView);
        argument.value.tensor.storage = VERNON_TENSOR_DEVICE;
        argument.value.tensor.buffer = buffer->handle;
        argument.value.tensor.dtype = static_cast<VernonDataType>(dtype);
        argument.value.tensor.access = static_cast<VernonValueAccess>(access);
        argument.value.tensor.rank = static_cast<uint32_t>(shape.size());
        argument.value.tensor.shape = argument.shape.data();
        argument.value.tensor.byte_strides = argument.strides.data();
        argument.value.tensor.byte_offset = offset;
        argument.value.tensor.byte_size = buffer->size;
        return *this;
    }

    PipelineInvocationBuilder &texture(const nb::object &identifier, Texture *texture, Sampler *sampler) {
        if (!texture || texture->owner != owner || (sampler && sampler->owner != owner))
            throw std::invalid_argument("pipeline texture or sampler belongs to another runtime");
        const PipelineParameterMetadata parameter = resolveParameter(identifier);
        OwnedArgument &argument = addArgument(parameter, VERNON_PIPELINE_TEXTURE);
        argument.value.texture = {texture->handle,
                                  VERNON_TEXTURE_RGBA8_UNORM,
                                  parameter.access,
                                  VERNON_TEXTURE_2D,
                                  texture->width,
                                  texture->height,
                                  1,
                                  sampler ? sampler->handle : nullptr};
        return *this;
    }

    PipelineInvocationBuilder &sampler(const nb::object &identifier, Sampler *sampler) {
        if (!sampler || sampler->owner != owner)
            throw std::invalid_argument("pipeline sampler belongs to another runtime");
        const PipelineParameterMetadata parameter = resolveParameter(identifier);
        OwnedArgument &argument = addArgument(parameter, VERNON_PIPELINE_SAMPLER);
        argument.value.sampler = sampler->handle;
        return *this;
    }

    PipelineInvocationBuilder &colorAttachment(uint32_t location, Texture *texture) {
        if (!texture || texture->owner != owner)
            throw std::invalid_argument("color attachment belongs to another runtime");
        attachments.push_back({location, texture->handle});
        return *this;
    }

    PipelineInvocationBuilder &indexBinding(Buffer *buffer, uint32_t count, size_t offset) {
        if (!buffer || buffer->owner != owner)
            throw std::invalid_argument("index buffer belongs to another runtime");
        index = {buffer->handle, VERNON_INDEX_U32, offset, count};
        hasIndex = true;
        return *this;
    }

    PipelineInvocationBuilder &setTopology(uint32_t value) {
        if (value > static_cast<uint32_t>(VERNON_TOPOLOGY_POINT_LIST))
            throw std::invalid_argument("invalid primitive topology");
        topology = static_cast<VernonPrimitiveTopology>(value);
        return *this;
    }

    PipelineInvocationBuilder &counts(uint32_t vertices, uint32_t instances) {
        vertexCount = vertices;
        instanceCount = instances;
        return *this;
    }

    PipelineInvocationBuilder &grid(uint32_t x, uint32_t y, uint32_t z) {
        computeGrid = {x, y, z};
        return *this;
    }

    PipelineInvocationBuilder &setViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
        viewport[0] = x;
        viewport[1] = y;
        viewport[2] = width;
        viewport[3] = height;
        return *this;
    }

    PipelineInvocationBuilder &setScissor(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
        scissor[0] = x;
        scissor[1] = y;
        scissor[2] = width;
        scissor[3] = height;
        return *this;
    }

    void invoke() {
        std::vector<VernonPipelineArgument> values;
        values.reserve(arguments.size());
        for (const OwnedArgument &argument : arguments)
            values.push_back(argument.value);
        VernonPipelineInvocation invocation{};
        invocation.struct_size = sizeof(invocation);
        invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
        invocation.arguments = values.empty() ? nullptr : values.data();
        invocation.argument_count = values.size();
        invocation.index_binding = hasIndex ? &index : nullptr;
        invocation.color_attachments = attachments.empty() ? nullptr : attachments.data();
        invocation.color_attachment_count = attachments.size();
        invocation.topology = topology;
        invocation.vertex_count = vertexCount;
        invocation.instance_count = instanceCount;
        invocation.compute_grid = computeGrid;
        std::memcpy(invocation.viewport, viewport, sizeof(viewport));
        std::memcpy(invocation.scissor, scissor, sizeof(scissor));
        if (vernonRuntimePipelineInvoke(pipeline, &invocation) != VERNON_STATUS_OK)
            throw std::runtime_error("pipeline invocation failed: " + stringView(vernonRuntimeGetLastError(runtime)));
    }

    Runtime *owner{};
    VernonRuntimeContext *runtime{};
    VernonLoadedPipeline *pipeline{};
    std::deque<OwnedArgument> arguments;
    std::unordered_set<uint32_t> slots;
    std::vector<VernonColorAttachment> attachments;
    VernonIndexBinding index{};
    bool hasIndex{};
    VernonPrimitiveTopology topology{VERNON_TOPOLOGY_TRIANGLE_LIST};
    uint32_t vertexCount{};
    uint32_t instanceCount{};
    VernonLaunchSize computeGrid{};
    uint32_t viewport[4]{};
    uint32_t scissor[4]{};
};

struct LoadedPipeline {
    LoadedPipeline(Runtime *owner, VernonRuntimeContext *runtime, VernonPipelineBundle *bundle,
                   VernonLoadedPipeline *pipeline)
        : owner(owner), runtime(runtime), bundle(bundle), pipeline(pipeline) {}
    ~LoadedPipeline() {
        vernonRuntimeLoadedPipelineDestroy(pipeline);
        vernonRuntimePipelineBundleDestroy(bundle);
    }

    std::unique_ptr<PipelineInvocationBuilder> invocationBuilder() {
        return std::make_unique<PipelineInvocationBuilder>(owner, runtime, pipeline);
    }

    std::vector<PipelineParameterMetadata> parameters() const {
        std::vector<PipelineParameterMetadata> result;
        const size_t count = vernonRuntimeLoadedPipelineGetParameterCount(pipeline);
        result.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            VernonPipelineParameterView view{};
            if (vernonRuntimeLoadedPipelineGetParameterByIndex(pipeline, index, &view) != VERNON_STATUS_OK)
                throw std::runtime_error("cannot read loaded pipeline parameter");
            result.push_back(parameterMetadata(view));
        }
        return result;
    }

    std::vector<PipelineOutputMetadata> outputs() const {
        std::vector<PipelineOutputMetadata> result;
        const size_t count = vernonRuntimeLoadedPipelineGetOutputCount(pipeline);
        result.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            VernonPipelineOutputView view{};
            if (vernonRuntimeLoadedPipelineGetOutputByIndex(pipeline, index, &view) != VERNON_STATUS_OK)
                throw std::runtime_error("cannot read loaded pipeline output");
            result.push_back(outputMetadata(view));
        }
        return result;
    }

    Runtime *owner{};
    VernonRuntimeContext *runtime{};
    VernonPipelineBundle *bundle{};
    VernonLoadedPipeline *pipeline{};
};
struct Runtime {
    explicit Runtime(VernonRuntimeContext *handle) : handle(handle) {}
    explicit Runtime(VernonRuntimeBackend backend, uint16_t apiMajor = 0, uint16_t apiMinor = 0) {
        VernonRuntimeCreateOptions options{};
        options.struct_size = sizeof(options);
        options.api_version_major = apiMajor;
        options.api_version_minor = apiMinor;
        handle = vernonRuntimeCreateWithOptions(backend, &options);
        if (!handle)
            throw std::runtime_error("requested runtime backend is unavailable");
    }
    ~Runtime() { vernonRuntimeDestroy(handle); }

    static std::unique_ptr<Runtime> createExternalOpenGL(VernonRuntimeBackend backend, uintptr_t userData,
                                                         uintptr_t makeCurrent, uintptr_t getProcAddress,
                                                         uint16_t apiMajor, uint16_t apiMinor) {
        VernonExternalOpenGLContext external{};
        external.struct_size = sizeof(external);
        external.user_data = reinterpret_cast<void *>(userData);
        external.make_current = reinterpret_cast<VernonOpenGLMakeCurrentFn>(makeCurrent);
        external.get_proc_address = reinterpret_cast<VernonOpenGLGetProcAddressFn>(getProcAddress);
        external.api_version_major = apiMajor;
        external.api_version_minor = apiMinor;
        VernonRuntimeContext *handle = vernonRuntimeCreateExternalOpenGLForBackend(backend, &external);
        if (!handle)
            throw std::runtime_error("external OpenGL context is invalid");
        return std::make_unique<Runtime>(handle);
    }

    std::unique_ptr<Buffer> allocate(size_t size, size_t alignment) {
        return std::make_unique<Buffer>(this, size, alignment);
    }

    std::unique_ptr<LoadedKernel> load(const nb::bytes &artifact, const std::string &reflection,
                                       const std::string &entry) {
        VernonLoadedKernel *kernel =
            vernonRuntimeLoadArtifact(handle, artifact.c_str(), artifact.size(), reflection.data(), reflection.size(),
                                      entry.data(), entry.size());
        if (!kernel)
            throw std::runtime_error("cannot load native kernel: " + stringView(vernonRuntimeGetLastError(handle)));
        return std::make_unique<LoadedKernel>(this, kernel);
    }

    std::unique_ptr<LoadedKernel> loadCpuEntry(const CompiledProgram &program, const std::string &entry) {
        program.requireSuccess();
        if (program.target != VERNON_TARGET_CPU)
            throw std::runtime_error("CPU entries can be loaded only from CPU compiled programs");
        if (entry.empty())
            throw std::runtime_error("CPU entry name must not be empty");
        VernonCpuEntryPoint entryPoint =
            vernonCompileResultGetCpuEntry(program.result.get(), entry.data(), entry.size());
        if (!entryPoint)
            throw std::runtime_error("CPU entry '" + entry + "' was not found in compiled program");
        const std::string reflection = program.reflection();
        VernonLoadedKernel *kernel = vernonRuntimeLoadCpuEntry(handle, entryPoint, reflection.data(), reflection.size(),
                                                               entry.data(), entry.size());
        if (!kernel)
            throw std::runtime_error("cannot load CPU entry: " + stringView(vernonRuntimeGetLastError(handle)));
        return std::make_unique<LoadedKernel>(this, kernel, program.result);
    }

    std::unique_ptr<LoadedKernel> loadComputeBundle(const std::string &directory) {
        VernonLoadedKernel *kernel = vernonRuntimeLoadComputeBundle(handle, directory.c_str());
        if (!kernel)
            throw std::runtime_error("cannot load compute bundle: " + stringView(vernonRuntimeGetLastError(handle)));
        return std::make_unique<LoadedKernel>(this, kernel);
    }

    std::unique_ptr<Texture> createTexture(uint32_t width, uint32_t height) {
        return std::make_unique<Texture>(this, width, height);
    }

    std::unique_ptr<Sampler> importOpenGLSampler(uint32_t name) {
        VernonDeviceSampler *sampler = vernonRuntimeImportOpenGLSampler(handle, name);
        if (!sampler)
            throw std::runtime_error("cannot import OpenGL sampler");
        return std::make_unique<Sampler>(this, sampler);
    }

    std::unique_ptr<LoadedPipeline> loadPipeline(const nb::bytes &data, const std::vector<std::string> &features) {
        VernonPipelineBundle *bundle = vernonRuntimeLoadPipelineBundle(handle, data.c_str(), data.size());
        if (!bundle)
            throw std::runtime_error("cannot load pipeline bundle: " + stringView(vernonRuntimeGetLastError(handle)));
        std::vector<const char *> names;
        for (const std::string &feature : features)
            names.push_back(feature.c_str());
        VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {names.data(), names.size()});
        if (!pipeline) {
            vernonRuntimePipelineBundleDestroy(bundle);
            throw std::runtime_error("cannot resolve pipeline bundle: " +
                                     stringView(vernonRuntimeGetLastError(handle)));
        }
        return std::make_unique<LoadedPipeline>(this, handle, bundle, pipeline);
    }

    std::unique_ptr<LoadedPipeline> loadPipelineAsset(const nb::bytes &data, const std::string &directory,
                                                      const std::vector<std::string> &features) {
        VernonPipelineBundleLoadOptions options{};
        options.struct_size = sizeof(options);
        options.bundle_directory = directory.c_str();
        VernonPipelineBundle *bundle =
            vernonRuntimeLoadPipelineBundleWithOptions(handle, data.c_str(), data.size(), &options);
        if (!bundle)
            throw std::runtime_error("cannot load pipeline bundle: " + stringView(vernonRuntimeGetLastError(handle)));
        std::vector<const char *> names;
        for (const std::string &feature : features)
            names.push_back(feature.c_str());
        VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {names.data(), names.size()});
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

Buffer::Buffer(Runtime *owner, size_t size, size_t alignment) : owner(owner), size(size) {
    handle = vernonRuntimeBufferAllocate(owner->handle, size, alignment);
    if (!handle)
        throw std::runtime_error("runtime buffer allocation failed");
}

Texture::Texture(Runtime *owner, uint32_t width, uint32_t height) : owner(owner), width(width), height(height) {
    handle = vernonRuntimeTextureCreate2D(owner->handle, width, height, VERNON_TEXTURE_RGBA8_UNORM);
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
        .value("DIRECTX", VERNON_TARGET_DIRECTX)
        .value("OPENGL", VERNON_TARGET_OPENGL)
        .value("OPENGL_ES", VERNON_TARGET_OPENGL_ES);
    nb::enum_<VernonStatus>(module, "Status")
        .value("OK", VERNON_STATUS_OK)
        .value("INVALID_ARGUMENT", VERNON_STATUS_INVALID_ARGUMENT)
        .value("PARSE_ERROR", VERNON_STATUS_PARSE_ERROR)
        .value("VERIFICATION_ERROR", VERNON_STATUS_VERIFICATION_ERROR)
        .value("UNSUPPORTED_TARGET", VERNON_STATUS_UNSUPPORTED_TARGET)
        .value("INTERNAL_ERROR", VERNON_STATUS_INTERNAL_ERROR);
    nb::enum_<VernonRuntimeBackend>(module, "RuntimeBackend")
        .value("CPU", VERNON_RUNTIME_CPU)
        .value("CUDA", VERNON_RUNTIME_CUDA)
        .value("VULKAN", VERNON_RUNTIME_VULKAN)
        .value("OPENGL", VERNON_RUNTIME_OPENGL)
        .value("OPENGL_ES", VERNON_RUNTIME_OPENGL_ES)
        .value("DIRECTX12", VERNON_RUNTIME_DIRECTX12);
    nb::enum_<VernonPrimitiveTopology>(module, "PrimitiveTopology")
        .value("TRIANGLE_LIST", VERNON_TOPOLOGY_TRIANGLE_LIST)
        .value("LINE_LIST", VERNON_TOPOLOGY_LINE_LIST)
        .value("POINT_LIST", VERNON_TOPOLOGY_POINT_LIST);
    nb::class_<Compiler>(module, "Compiler")
        .def(nb::init<>())
        .def("compile_program_result", &compileProgramResult, nb::arg("mlir"), nb::arg("target"),
             nb::arg("glsl_version") = 0, nb::arg("hlsl_shader_model") = 0, nb::arg("target_triple") = "",
             nb::arg("cpu") = "", nb::arg("cpu_features") = "");
    nb::class_<CompiledProgram>(module, "CompiledProgram")
        .def_prop_ro("ok", &CompiledProgram::ok)
        .def_prop_ro("status", &CompiledProgram::status)
        .def_prop_ro("diagnostics", &CompiledProgram::diagnostics)
        .def_prop_ro("artifacts", &CompiledProgram::artifacts)
        .def_prop_ro("reflection", &CompiledProgram::reflection)
        .def_prop_ro("target", [](const CompiledProgram &value) { return value.target; })
        .def_prop_ro("glsl_version", [](const CompiledProgram &value) { return value.glslVersion; })
        .def_prop_ro("hlsl_shader_model", [](const CompiledProgram &value) { return value.hlslShaderModel; })
        .def_prop_ro("target_triple", [](const CompiledProgram &value) { return value.targetTriple; })
        .def_prop_ro("cpu", [](const CompiledProgram &value) { return value.cpu; })
        .def_prop_ro("cpu_features", [](const CompiledProgram &value) { return value.cpuFeatures; })
        .def("has_cpu_entry", &CompiledProgram::hasCpuEntry);
    nb::class_<Runtime>(module, "Runtime")
        .def(nb::init<VernonRuntimeBackend, uint16_t, uint16_t>(), nb::arg("backend"), nb::arg("api_major") = 0,
             nb::arg("api_minor") = 0)
        .def_static("create_external_opengl", &Runtime::createExternalOpenGL, nb::arg("backend"), nb::arg("user_data"),
                    nb::arg("make_current"), nb::arg("get_proc_address"), nb::arg("api_major"), nb::arg("api_minor"))
        .def("allocate", &Runtime::allocate, nb::keep_alive<0, 1>())
        .def("load", &Runtime::load, nb::keep_alive<0, 1>())
        .def("load_cpu_entry", &Runtime::loadCpuEntry, nb::keep_alive<0, 1>())
        .def("load_compute_bundle", &Runtime::loadComputeBundle, nb::keep_alive<0, 1>())
        .def("create_texture", &Runtime::createTexture, nb::keep_alive<0, 1>())
        .def("import_opengl_sampler", &Runtime::importOpenGLSampler, nb::keep_alive<0, 1>())
        .def("load_pipeline", &Runtime::loadPipeline, nb::keep_alive<0, 1>())
        .def("load_pipeline_asset", &Runtime::loadPipelineAsset, nb::keep_alive<0, 1>())
        .def("synchronize", &Runtime::synchronize);
    nb::class_<Buffer>(module, "Buffer").def("upload", &Buffer::upload).def("download", &Buffer::download);
    nb::class_<LoadedKernel>(module, "LoadedKernel").def("launch", &LoadedKernel::launch);
    nb::class_<Texture>(module, "Texture")
        .def_prop_ro("width", [](const Texture &value) { return value.width; })
        .def_prop_ro("height", [](const Texture &value) { return value.height; })
        .def("upload", &Texture::upload)
        .def("download", &Texture::download);
    nb::class_<Sampler>(module, "Sampler");
    nb::class_<PipelineParameterMetadata>(module, "PipelineParameter")
        .def_ro("slot", &PipelineParameterMetadata::slot)
        .def_ro("name", &PipelineParameterMetadata::name)
        .def_prop_ro("kind", [](const PipelineParameterMetadata &value) { return static_cast<uint32_t>(value.kind); })
        .def_prop_ro("dtype", [](const PipelineParameterMetadata &value) { return static_cast<uint32_t>(value.dtype); })
        .def_prop_ro("access",
                     [](const PipelineParameterMetadata &value) { return static_cast<uint32_t>(value.access); })
        .def_ro("shape", &PipelineParameterMetadata::shape);
    nb::class_<PipelineOutputMetadata>(module, "PipelineOutput")
        .def_ro("name", &PipelineOutputMetadata::name)
        .def_prop_ro("kind", [](const PipelineOutputMetadata &value) { return static_cast<uint32_t>(value.kind); })
        .def_prop_ro("dtype", [](const PipelineOutputMetadata &value) { return static_cast<uint32_t>(value.dtype); })
        .def_prop_ro("access", [](const PipelineOutputMetadata &value) { return static_cast<uint32_t>(value.access); })
        .def_ro("shape", &PipelineOutputMetadata::shape)
        .def_ro("location", &PipelineOutputMetadata::location);
    nb::class_<PipelineInvocationBuilder>(module, "PipelineInvocationBuilder")
        .def("host_tensor", &PipelineInvocationBuilder::hostTensor, nb::arg("parameter"), nb::arg("array"),
             nb::rv_policy::reference_internal)
        .def("device_tensor", &PipelineInvocationBuilder::deviceTensor, nb::arg("parameter"), nb::arg("buffer"),
             nb::arg("dtype"), nb::arg("access"), nb::arg("shape"), nb::arg("strides"), nb::arg("offset") = 0,
             nb::rv_policy::reference_internal, nb::keep_alive<1, 3>())
        .def("texture", &PipelineInvocationBuilder::texture, nb::arg("parameter"), nb::arg("texture"),
             nb::arg("sampler") = nullptr, nb::rv_policy::reference_internal, nb::keep_alive<1, 3>(),
             nb::keep_alive<1, 4>())
        .def("sampler", &PipelineInvocationBuilder::sampler, nb::arg("parameter"), nb::arg("sampler"),
             nb::rv_policy::reference_internal, nb::keep_alive<1, 3>())
        .def("color_attachment", &PipelineInvocationBuilder::colorAttachment, nb::arg("location"), nb::arg("texture"),
             nb::rv_policy::reference_internal, nb::keep_alive<1, 3>())
        .def("index_binding", &PipelineInvocationBuilder::indexBinding, nb::arg("buffer"), nb::arg("count"),
             nb::arg("offset") = 0, nb::rv_policy::reference_internal, nb::keep_alive<1, 2>())
        .def("topology", &PipelineInvocationBuilder::setTopology, nb::arg("topology"),
             nb::rv_policy::reference_internal)
        .def("counts", &PipelineInvocationBuilder::counts, nb::arg("vertex_count") = 0, nb::arg("instance_count") = 0,
             nb::rv_policy::reference_internal)
        .def("grid", &PipelineInvocationBuilder::grid, nb::arg("x"), nb::arg("y"), nb::arg("z"),
             nb::rv_policy::reference_internal)
        .def("viewport", &PipelineInvocationBuilder::setViewport, nb::arg("x"), nb::arg("y"), nb::arg("width"),
             nb::arg("height"), nb::rv_policy::reference_internal)
        .def("scissor", &PipelineInvocationBuilder::setScissor, nb::arg("x"), nb::arg("y"), nb::arg("width"),
             nb::arg("height"), nb::rv_policy::reference_internal)
        .def("invoke", &PipelineInvocationBuilder::invoke);
    nb::class_<LoadedPipeline>(module, "LoadedPipeline")
        .def("invocation_builder", &LoadedPipeline::invocationBuilder, nb::keep_alive<0, 1>())
        .def_prop_ro("parameters", &LoadedPipeline::parameters)
        .def_prop_ro("outputs", &LoadedPipeline::outputs);
    module.attr("DATA_BOOL") = static_cast<uint32_t>(VERNON_DATA_BOOL);
    module.attr("DATA_I32") = static_cast<uint32_t>(VERNON_DATA_I32);
    module.attr("DATA_U32") = static_cast<uint32_t>(VERNON_DATA_U32);
    module.attr("DATA_F16") = static_cast<uint32_t>(VERNON_DATA_F16);
    module.attr("DATA_F32") = static_cast<uint32_t>(VERNON_DATA_F32);
    module.attr("DATA_F64") = static_cast<uint32_t>(VERNON_DATA_F64);
    module.attr("ACCESS_READ") = static_cast<uint32_t>(VERNON_ACCESS_READ);
    module.attr("ACCESS_WRITE") = static_cast<uint32_t>(VERNON_ACCESS_WRITE);
    module.attr("ACCESS_READ_WRITE") = static_cast<uint32_t>(VERNON_ACCESS_READ_WRITE);
    module.attr("PIPELINE_TENSOR") = static_cast<uint32_t>(VERNON_PIPELINE_TENSOR);
    module.attr("PIPELINE_TEXTURE") = static_cast<uint32_t>(VERNON_PIPELINE_TEXTURE);
    module.attr("PIPELINE_SAMPLER") = static_cast<uint32_t>(VERNON_PIPELINE_SAMPLER);
    module.attr("TOPOLOGY_TRIANGLE_LIST") = static_cast<uint32_t>(VERNON_TOPOLOGY_TRIANGLE_LIST);
    module.attr("TOPOLOGY_LINE_LIST") = static_cast<uint32_t>(VERNON_TOPOLOGY_LINE_LIST);
    module.attr("TOPOLOGY_POINT_LIST") = static_cast<uint32_t>(VERNON_TOPOLOGY_POINT_LIST);
    module.def("runtime_available",
               [](VernonRuntimeBackend backend) { return vernonRuntimeGetCapabilities(backend).available != 0; });
}
