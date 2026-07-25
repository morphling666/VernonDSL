#include "VernonCompiler.h"
#include "VernonRHI.h"
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

struct RhiHostState;
struct Runtime;
RhiHostState *runtimeRhiHost(const Runtime *runtime);
struct CompiledProgram;
struct LoadedPipeline;
struct PipelineInvocationBuilder;

struct RhiHostState {
    RhiHostState(VernonRhiBackend backend, uint32_t deviceIndex) : backend(backend) {
        VernonRhiOwnedDeviceDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.backend = backend;
        descriptor.device_index = deviceIndex;
        device = vernonRhiCreateDevice(&descriptor);
        if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
            throw std::runtime_error("cannot create Vernon RHI device");
    }
    RhiHostState(VernonRhiBackend backend, const VernonOpenGLContextCallbacks &callbacks) : backend(backend) {
        VernonRhiOwnedDeviceDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.backend = backend;
        descriptor.opengl_callbacks = &callbacks;
        device = vernonRhiCreateDevice(&descriptor);
        if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
            throw std::runtime_error("cannot create external OpenGL RHI device");
    }
    ~RhiHostState() { vernonRhiDestroyDevice(device); }

    VernonRhiBackend backend;
    VernonRhiDevice device{};
};

struct RhiBuffer {
    RhiBuffer(std::shared_ptr<RhiHostState> host, size_t size) : host(std::move(host)), size(size) {
        VernonRhiBufferDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.size = size;
        descriptor.usage = VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION |
                           VERNON_RHI_BUFFER_UNIFORM | VERNON_RHI_BUFFER_STORAGE | VERNON_RHI_BUFFER_VERTEX |
                           VERNON_RHI_BUFFER_INDEX;
        descriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
        if (vernonRhiDeviceCreateBuffer(this->host->device, &descriptor, &handle) != VERNON_RHI_STATUS_OK)
            throw std::runtime_error("cannot create Vernon RHI buffer");
    }
    ~RhiBuffer() { vernonRhiDeviceDestroyBuffer(host->device, handle); }

    void upload(const nb::bytes &data) {
        if (data.size() != size ||
            vernonRhiDeviceUploadBuffer(host->device, handle, 0, data.c_str(), data.size()) != VERNON_RHI_STATUS_OK)
            throw std::runtime_error("RHI buffer upload failed");
    }
    nb::bytes download() const {
        std::string data(size, '\0');
        if (vernonRhiDeviceDownloadBuffer(host->device, handle, 0, data.data(), data.size()) != VERNON_RHI_STATUS_OK)
            throw std::runtime_error("RHI buffer download failed");
        return nb::bytes(data.data(), data.size());
    }

    std::shared_ptr<RhiHostState> host;
    VernonRhiBuffer handle{};
    size_t size{};
};

struct RhiImage {
    RhiImage(std::shared_ptr<RhiHostState> host, uint32_t width, uint32_t height)
        : host(std::move(host)), width(width), height(height) {
        VernonRhiImageDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.dimension = VERNON_RHI_IMAGE_2D;
        descriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
        descriptor.width = width;
        descriptor.height = height;
        descriptor.depth = 1;
        descriptor.mip_levels = 1;
        descriptor.array_layers = 1;
        descriptor.sample_count = 1;
        descriptor.usage = VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION |
                           VERNON_RHI_IMAGE_SAMPLED | VERNON_RHI_IMAGE_COLOR_ATTACHMENT;
        if (vernonRhiDeviceCreateImage(this->host->device, &descriptor, &handle) != VERNON_RHI_STATUS_OK)
            throw std::runtime_error("cannot create Vernon RHI image");
    }
    ~RhiImage() { vernonRhiDeviceDestroyImage(host->device, handle); }

    void upload(const nb::bytes &data) {
        if (data.size() != static_cast<size_t>(width) * height * 4)
            throw std::runtime_error("RHI image upload size does not match RGBA8 extent");
        VernonRhiImageUploadDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.width = width;
        descriptor.height = height;
        descriptor.depth = 1;
        descriptor.source_format = VERNON_RHI_IMAGE_DATA_RGBA;
        descriptor.source_type = VERNON_RHI_IMAGE_DATA_UINT8;
        descriptor.data = data.c_str();
        if (vernonRhiDeviceUploadImage(host->device, handle, &descriptor, 1) != VERNON_RHI_STATUS_OK)
            throw std::runtime_error("RHI image upload failed");
    }
    nb::bytes download() const {
        std::string data(static_cast<size_t>(width) * height * 4, '\0');
        if (vernonRhiDeviceDownloadImage(host->device, handle, data.data(), data.size()) != VERNON_RHI_STATUS_OK)
            throw std::runtime_error("RHI image download failed");
        return nb::bytes(data.data(), data.size());
    }

    std::shared_ptr<RhiHostState> host;
    VernonRhiImage handle{};
    uint32_t width{};
    uint32_t height{};
};

struct RhiSampler {
    explicit RhiSampler(std::shared_ptr<RhiHostState> host) : host(std::move(host)) {
        VernonRhiSamplerDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.min_filter = VERNON_RHI_FILTER_LINEAR;
        descriptor.mag_filter = VERNON_RHI_FILTER_LINEAR;
        descriptor.mip_filter = VERNON_RHI_FILTER_LINEAR;
        descriptor.address_u = VERNON_RHI_ADDRESS_REPEAT;
        descriptor.address_v = VERNON_RHI_ADDRESS_REPEAT;
        descriptor.address_w = VERNON_RHI_ADDRESS_REPEAT;
        descriptor.max_anisotropy = 1.0f;
        if (vernonRhiDeviceCreateSampler(this->host->device, &descriptor, &handle) != VERNON_RHI_STATUS_OK)
            throw std::runtime_error("cannot create Vernon RHI sampler");
    }
    ~RhiSampler() { vernonRhiDeviceDestroySampler(host->device, handle); }

    std::shared_ptr<RhiHostState> host;
    VernonRhiSampler handle{};
};

struct RhiHost {
    RhiHost(VernonRhiBackend backend, uint32_t deviceIndex)
        : state(std::make_shared<RhiHostState>(backend, deviceIndex)) {}
    explicit RhiHost(std::shared_ptr<RhiHostState> state) : state(std::move(state)) {}

    static std::unique_ptr<RhiHost> createExternalOpenGL(VernonRhiBackend backend, uintptr_t userData,
                                                         uintptr_t makeCurrent, uintptr_t getProcAddress,
                                                         uint16_t apiMajor, uint16_t apiMinor) {
        if (backend != VERNON_RHI_BACKEND_OPENGL && backend != VERNON_RHI_BACKEND_OPENGL_ES)
            throw std::invalid_argument("external OpenGL host requires an OpenGL backend");
        VernonOpenGLContextCallbacks callbacks{};
        callbacks.struct_size = sizeof(callbacks);
        callbacks.user_data = reinterpret_cast<void *>(userData);
        callbacks.make_current = reinterpret_cast<VernonOpenGLMakeCurrentFn>(makeCurrent);
        callbacks.get_proc_address = reinterpret_cast<VernonOpenGLGetProcAddressFn>(getProcAddress);
        callbacks.api_version_major = apiMajor;
        callbacks.api_version_minor = apiMinor;
        return std::make_unique<RhiHost>(std::make_shared<RhiHostState>(backend, callbacks));
    }

    std::unique_ptr<RhiBuffer> createBuffer(size_t size) { return std::make_unique<RhiBuffer>(state, size); }
    std::unique_ptr<RhiImage> createImage(uint32_t width, uint32_t height) {
        return std::make_unique<RhiImage>(state, width, height);
    }
    std::unique_ptr<RhiSampler> createSampler() { return std::make_unique<RhiSampler>(state); }
    std::unique_ptr<Runtime> createRuntime();
    void synchronize() {
        if (vernonRhiDeviceSynchronize(state->device) != VERNON_RHI_STATUS_OK)
            throw std::runtime_error("RHI device synchronization failed");
    }

    std::shared_ptr<RhiHostState> state;
};

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
        if (name == "uint8")
            return VERNON_DATA_U8;
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

    PipelineInvocationBuilder &rhiTensor(const nb::object &identifier, RhiBuffer *buffer, uint32_t dtype,
                                         uint32_t access, const std::vector<uint64_t> &shape,
                                         const std::vector<int64_t> &strides, size_t offset) {
        if (!buffer || shape.size() != strides.size() || shape.empty())
            throw std::invalid_argument("RHI Tensor shape and strides must have equal non-zero rank");
        const PipelineParameterMetadata parameter = resolveParameter(identifier);
        OwnedArgument &argument = addArgument(parameter, VERNON_PIPELINE_TENSOR);
        argument.shape = shape;
        argument.strides = strides;
        argument.value.tensor.struct_size = sizeof(VernonTensorView);
        argument.value.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
        if (vernonRuntimeReferenceRhiBuffer(runtime, buffer->handle, 0, buffer->size,
                                            &argument.value.tensor.resource) != VERNON_STATUS_OK)
            throw std::invalid_argument("RHI buffer belongs to another Runtime device");
        argument.value.tensor.dtype = static_cast<VernonDataType>(dtype);
        argument.value.tensor.access = static_cast<VernonValueAccess>(access);
        argument.value.tensor.rank = static_cast<uint32_t>(shape.size());
        argument.value.tensor.shape = argument.shape.data();
        argument.value.tensor.byte_strides = argument.strides.data();
        argument.value.tensor.byte_offset = offset;
        argument.value.tensor.byte_size = buffer->size;
        return *this;
    }

    PipelineInvocationBuilder &rhiTexture(const nb::object &identifier, RhiImage *texture, RhiSampler *sampler) {
        if (!texture || (sampler && sampler->host != texture->host))
            throw std::invalid_argument("RHI texture and sampler belong to different devices");
        const PipelineParameterMetadata parameter = resolveParameter(identifier);
        OwnedArgument &argument = addArgument(parameter, VERNON_PIPELINE_TEXTURE);
        if (vernonRuntimeReferenceRhiImage(runtime, texture->handle, &argument.value.texture.resource) !=
            VERNON_STATUS_OK)
            throw std::invalid_argument("RHI image belongs to another Runtime device");
        if (sampler && vernonRuntimeReferenceRhiSampler(runtime, sampler->handle,
                                                        &argument.value.texture.sampler_resource) != VERNON_STATUS_OK)
            throw std::invalid_argument("RHI sampler belongs to another Runtime device");
        argument.value.texture.format = VERNON_TEXTURE_RGBA8_UNORM;
        argument.value.texture.access = parameter.access;
        argument.value.texture.dimension = VERNON_TEXTURE_2D;
        argument.value.texture.width = texture->width;
        argument.value.texture.height = texture->height;
        argument.value.texture.depth = 1;
        return *this;
    }

    PipelineInvocationBuilder &rhiSampler(const nb::object &identifier, RhiSampler *sampler) {
        if (!sampler)
            throw std::invalid_argument("RHI sampler is null");
        const PipelineParameterMetadata parameter = resolveParameter(identifier);
        OwnedArgument &argument = addArgument(parameter, VERNON_PIPELINE_SAMPLER);
        if (vernonRuntimeReferenceRhiSampler(runtime, sampler->handle, &argument.value.resource) != VERNON_STATUS_OK)
            throw std::invalid_argument("RHI sampler belongs to another Runtime device");
        return *this;
    }

    PipelineInvocationBuilder &rhiColorAttachment(uint32_t location, RhiImage *texture) {
        if (!texture)
            throw std::invalid_argument("RHI color attachment is null");
        VernonColorAttachment attachment{};
        attachment.location = location;
        if (vernonRuntimeReferenceRhiImage(runtime, texture->handle, &attachment.resource) != VERNON_STATUS_OK)
            throw std::invalid_argument("RHI color attachment belongs to another Runtime device");
        attachment.width = texture->width;
        attachment.height = texture->height;
        attachment.format = VERNON_TEXTURE_RGBA8_UNORM;
        attachments.push_back(attachment);
        return *this;
    }

    PipelineInvocationBuilder &rhiIndexBinding(RhiBuffer *buffer, uint32_t count, size_t offset) {
        if (!buffer || offset > buffer->size)
            throw std::invalid_argument("RHI index buffer is null");
        index = {};
        if (vernonRuntimeReferenceRhiBuffer(runtime, buffer->handle, offset, buffer->size - offset, &index.resource) !=
            VERNON_STATUS_OK)
            throw std::invalid_argument("RHI index buffer belongs to another Runtime device");
        index.type = VERNON_INDEX_U32;
        index.offset = offset;
        index.index_count = count;
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
                   VernonLoadedPipeline *pipeline, SharedCompileResult retainedResult = {})
        : owner(owner), runtime(runtime), bundle(bundle), pipeline(pipeline),
          retainedResult(std::move(retainedResult)) {}
    ~LoadedPipeline() {
        vernonRuntimeLoadedPipelineDestroy(pipeline);
        vernonRuntimePipelineBundleDestroy(bundle);
    }

    std::unique_ptr<PipelineInvocationBuilder> invocationBuilder() {
        return std::make_unique<PipelineInvocationBuilder>(owner, runtime, pipeline);
    }

    void invoke(PipelineInvocationBuilder &builder) {
        if (builder.pipeline != pipeline)
            throw std::invalid_argument("invocation builder belongs to another pipeline");
        builder.invoke();
    }

    static size_t dataTypeSize(VernonDataType type) {
        switch (type) {
        case VERNON_DATA_BOOL:
        case VERNON_DATA_U8:
            return 1;
        case VERNON_DATA_F16:
            return 2;
        case VERNON_DATA_I32:
        case VERNON_DATA_U32:
        case VERNON_DATA_F32:
            return 4;
        case VERNON_DATA_F64:
            return 8;
        }
        throw std::invalid_argument("unsupported compute pipeline data type");
    }

    void invokeCompute(uint32_t x, uint32_t y, uint32_t z, const nb::list &values) {
        const std::vector<PipelineParameterMetadata> metadata = parameters();
        if (values.size() != metadata.size())
            throw std::invalid_argument("compute pipeline value count does not match reflection");
        std::deque<std::string> scalarStorage;
        std::deque<std::vector<uint64_t>> shapes;
        std::deque<std::vector<int64_t>> strides;
        std::vector<VernonPipelineArgument> arguments;
        arguments.reserve(metadata.size());
        for (size_t index = 0; index < metadata.size(); ++index) {
            const PipelineParameterMetadata &parameter = metadata[index];
            if (parameter.kind != VERNON_PIPELINE_TENSOR)
                throw std::invalid_argument("direct compute pipelines accept only Tensor/value parameters");
            shapes.push_back(parameter.shape);
            strides.emplace_back(parameter.shape.size());
            size_t stride = dataTypeSize(parameter.dtype);
            for (size_t dimension = parameter.shape.size(); dimension-- != 0;) {
                strides.back()[dimension] = static_cast<int64_t>(stride);
                stride *= static_cast<size_t>(parameter.shape[dimension]);
            }
            VernonPipelineArgument argument{};
            argument.slot = parameter.slot;
            argument.kind = VERNON_PIPELINE_TENSOR;
            argument.tensor.struct_size = sizeof(VernonTensorView);
            argument.tensor.dtype = parameter.dtype;
            argument.tensor.access = parameter.access;
            argument.tensor.rank = static_cast<uint32_t>(shapes.back().size());
            argument.tensor.shape = shapes.back().empty() ? nullptr : shapes.back().data();
            argument.tensor.byte_strides = strides.back().empty() ? nullptr : strides.back().data();
            nb::handle value = values[index];
            if (nb::isinstance<Buffer>(value)) {
                Buffer *buffer = nb::cast<Buffer *>(value);
                if (!buffer || buffer->owner != owner)
                    throw std::invalid_argument("compute pipeline buffer belongs to another runtime");
                argument.tensor.storage = VERNON_TENSOR_DEVICE;
                argument.tensor.buffer = buffer->handle;
                argument.tensor.byte_size = buffer->size;
            } else if (nb::isinstance<RhiBuffer>(value)) {
                RhiBuffer *buffer = nb::cast<RhiBuffer *>(value);
                if (!buffer || runtimeRhiHost(owner) != buffer->host.get())
                    throw std::invalid_argument("compute pipeline RHI buffer belongs to another device");
                argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
                if (vernonRuntimeReferenceRhiBuffer(runtime, buffer->handle, 0, buffer->size,
                                                    &argument.tensor.resource) != VERNON_STATUS_OK)
                    throw std::invalid_argument("compute pipeline RHI buffer is invalid");
                argument.tensor.byte_size = buffer->size;
            } else if (nb::isinstance<nb::bytes>(value)) {
                nb::bytes bytes = nb::borrow<nb::bytes>(value);
                scalarStorage.emplace_back(bytes.c_str(), bytes.size());
                argument.tensor.storage = VERNON_TENSOR_HOST;
                argument.tensor.host_data = scalarStorage.back().data();
                argument.tensor.byte_size = scalarStorage.back().size();
            } else {
                throw std::invalid_argument("compute pipeline values must be Buffer or bytes");
            }
            arguments.push_back(argument);
        }
        VernonPipelineInvocation invocation{};
        invocation.struct_size = sizeof(invocation);
        invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
        invocation.arguments = arguments.data();
        invocation.argument_count = arguments.size();
        invocation.compute_grid = {x, y, z};
        if (vernonRuntimePipelineInvoke(pipeline, &invocation) != VERNON_STATUS_OK)
            throw std::runtime_error("compute pipeline invocation failed: " +
                                     stringView(vernonRuntimeGetLastError(runtime)));
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
    // ORC entry pointers are valid only while their compile result owns the JIT.
    SharedCompileResult retainedResult;
};
struct Runtime {
    explicit Runtime(VernonRuntimeContext *handle, std::shared_ptr<RhiHostState> rhiHost = {})
        : handle(handle), rhiHost(std::move(rhiHost)) {}
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
        VernonOpenGLContextCallbacks callbacks{};
        callbacks.struct_size = sizeof(callbacks);
        callbacks.user_data = reinterpret_cast<void *>(userData);
        callbacks.make_current = reinterpret_cast<VernonOpenGLMakeCurrentFn>(makeCurrent);
        callbacks.get_proc_address = reinterpret_cast<VernonOpenGLGetProcAddressFn>(getProcAddress);
        callbacks.api_version_major = apiMajor;
        callbacks.api_version_minor = apiMinor;
        VernonRuntimeContext *handle = vernonRuntimeCreateOpenGLWithCallbacks(backend, &callbacks);
        if (!handle)
            throw std::runtime_error("external OpenGL context is invalid");
        return std::make_unique<Runtime>(handle);
    }

    std::unique_ptr<Buffer> allocate(size_t size, size_t alignment) {
        return std::make_unique<Buffer>(this, size, alignment);
    }

    std::unique_ptr<LoadedPipeline> load(const nb::bytes &artifact, const std::string &reflection,
                                         const std::string &entry) {
        VernonLoadedPipeline *pipeline =
            vernonRuntimeLoadArtifact(handle, artifact.c_str(), artifact.size(), reflection.data(), reflection.size(),
                                      entry.data(), entry.size());
        if (!pipeline)
            throw std::runtime_error("cannot load compute pipeline: " + stringView(vernonRuntimeGetLastError(handle)));
        return std::make_unique<LoadedPipeline>(this, handle, nullptr, pipeline);
    }

    std::unique_ptr<LoadedPipeline> loadCpuEntry(const CompiledProgram &program, const std::string &entry) {
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
        VernonLoadedPipeline *pipeline = vernonRuntimeLoadCpuEntry(handle, entryPoint, reflection.data(),
                                                                   reflection.size(), entry.data(), entry.size());
        if (!pipeline)
            throw std::runtime_error("cannot load CPU entry: " + stringView(vernonRuntimeGetLastError(handle)));
        return std::make_unique<LoadedPipeline>(this, handle, nullptr, pipeline, program.result);
    }

    std::unique_ptr<LoadedPipeline> loadComputeBundle(const std::string &directory) {
        VernonLoadedPipeline *pipeline = vernonRuntimeLoadComputeBundle(handle, directory.c_str());
        if (!pipeline)
            throw std::runtime_error("cannot load compute bundle: " + stringView(vernonRuntimeGetLastError(handle)));
        return std::make_unique<LoadedPipeline>(this, handle, nullptr, pipeline);
    }

    std::unique_ptr<LoadedPipeline> loadPipeline(const nb::bytes &data, const std::vector<std::string> &features) {
        VernonPipelineBundle *bundle =
            vernonRuntimeLoadPipelineBundleWithOptions(handle, data.c_str(), data.size(), nullptr);
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
    std::shared_ptr<RhiHostState> rhiHost;
};

RhiHostState *runtimeRhiHost(const Runtime *runtime) { return runtime ? runtime->rhiHost.get() : nullptr; }

std::unique_ptr<Runtime> RhiHost::createRuntime() {
    VernonRuntimeBackend backend;
    switch (state->backend) {
    case VERNON_RHI_BACKEND_CUDA:
        backend = VERNON_RUNTIME_CUDA;
        break;
    case VERNON_RHI_BACKEND_VULKAN:
        backend = VERNON_RUNTIME_VULKAN;
        break;
    case VERNON_RHI_BACKEND_DIRECTX12:
        backend = VERNON_RUNTIME_DIRECTX12;
        break;
    case VERNON_RHI_BACKEND_OPENGL:
        backend = VERNON_RUNTIME_OPENGL;
        break;
    case VERNON_RHI_BACKEND_OPENGL_ES:
        backend = VERNON_RUNTIME_OPENGL_ES;
        break;
    }
    VernonRuntimeContext *runtime = vernonRuntimeCreateForRhiDevice(backend, state->device);
    if (!runtime)
        throw std::runtime_error("cannot create Runtime for Vernon RHI device");
    return std::make_unique<Runtime>(runtime, state);
}

Buffer::Buffer(Runtime *owner, size_t size, size_t alignment) : owner(owner), size(size) {
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
    nb::enum_<VernonRhiBackend>(module, "RhiBackend")
        .value("CUDA", VERNON_RHI_BACKEND_CUDA)
        .value("VULKAN", VERNON_RHI_BACKEND_VULKAN)
        .value("DIRECTX12", VERNON_RHI_BACKEND_DIRECTX12)
        .value("OPENGL", VERNON_RHI_BACKEND_OPENGL)
        .value("OPENGL_ES", VERNON_RHI_BACKEND_OPENGL_ES);
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
    nb::class_<RhiHost>(module, "RhiHost")
        .def(nb::init<VernonRhiBackend, uint32_t>(), nb::arg("backend"), nb::arg("device_index") = 0)
        .def_static("create_external_opengl", &RhiHost::createExternalOpenGL, nb::arg("backend"), nb::arg("user_data"),
                    nb::arg("make_current"), nb::arg("get_proc_address"), nb::arg("api_major"), nb::arg("api_minor"))
        .def("create_buffer", &RhiHost::createBuffer)
        .def("create_image", &RhiHost::createImage)
        .def("create_sampler", &RhiHost::createSampler)
        .def("create_runtime", &RhiHost::createRuntime, nb::keep_alive<0, 1>())
        .def("synchronize", &RhiHost::synchronize);
    nb::class_<RhiBuffer>(module, "RhiBuffer")
        .def_prop_ro("size", [](const RhiBuffer &value) { return value.size; })
        .def("upload", &RhiBuffer::upload)
        .def("download", &RhiBuffer::download);
    nb::class_<RhiImage>(module, "RhiImage")
        .def_prop_ro("width", [](const RhiImage &value) { return value.width; })
        .def_prop_ro("height", [](const RhiImage &value) { return value.height; })
        .def("upload", &RhiImage::upload)
        .def("download", &RhiImage::download);
    nb::class_<RhiSampler>(module, "RhiSampler");
    nb::class_<Runtime>(module, "Runtime")
        .def(nb::init<VernonRuntimeBackend, uint16_t, uint16_t>(), nb::arg("backend"), nb::arg("api_major") = 0,
             nb::arg("api_minor") = 0)
        .def_static("create_external_opengl", &Runtime::createExternalOpenGL, nb::arg("backend"), nb::arg("user_data"),
                    nb::arg("make_current"), nb::arg("get_proc_address"), nb::arg("api_major"), nb::arg("api_minor"))
        .def("allocate", &Runtime::allocate, nb::keep_alive<0, 1>())
        .def("load", &Runtime::load, nb::keep_alive<0, 1>())
        .def("load_cpu_entry", &Runtime::loadCpuEntry, nb::keep_alive<0, 1>())
        .def("load_compute_bundle", &Runtime::loadComputeBundle, nb::keep_alive<0, 1>())
        .def("load_pipeline", &Runtime::loadPipeline, nb::keep_alive<0, 1>())
        .def("load_pipeline_asset", &Runtime::loadPipelineAsset, nb::keep_alive<0, 1>())
        .def("synchronize", &Runtime::synchronize);
    nb::class_<Buffer>(module, "Buffer").def("upload", &Buffer::upload).def("download", &Buffer::download);
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
        .def("rhi_tensor", &PipelineInvocationBuilder::rhiTensor, nb::arg("parameter"), nb::arg("buffer"),
             nb::arg("dtype"), nb::arg("access"), nb::arg("shape"), nb::arg("strides"), nb::arg("offset") = 0,
             nb::rv_policy::reference_internal, nb::keep_alive<1, 3>())
        .def("rhi_texture", &PipelineInvocationBuilder::rhiTexture, nb::arg("parameter"), nb::arg("texture"),
             nb::arg("sampler") = nullptr, nb::rv_policy::reference_internal, nb::keep_alive<1, 3>(),
             nb::keep_alive<1, 4>())
        .def("rhi_sampler", &PipelineInvocationBuilder::rhiSampler, nb::arg("parameter"), nb::arg("sampler"),
             nb::rv_policy::reference_internal, nb::keep_alive<1, 3>())
        .def("rhi_color_attachment", &PipelineInvocationBuilder::rhiColorAttachment, nb::arg("location"),
             nb::arg("texture"), nb::rv_policy::reference_internal, nb::keep_alive<1, 3>())
        .def("rhi_index_binding", &PipelineInvocationBuilder::rhiIndexBinding, nb::arg("buffer"), nb::arg("count"),
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
        .def(
            "invoke",
            [](LoadedPipeline &pipeline, uint32_t x, uint32_t y, uint32_t z, const nb::list &values) {
                pipeline.invokeCompute(x, y, z, values);
            },
            nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("values"))
        .def(
            "invoke", [](LoadedPipeline &pipeline, PipelineInvocationBuilder &builder) { pipeline.invoke(builder); },
            nb::arg("builder"))
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
