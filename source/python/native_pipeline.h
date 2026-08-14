#ifndef VERNON_PYTHON_NATIVE_PIPELINE_H
#define VERNON_PYTHON_NATIVE_PIPELINE_H

#include "VernonCompiler.h"
#include "VernonRuntime.h"
#include "compiler_python_bridge.h"
#include "native_rhi.h"
#include "runtime/tensor_bridge.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <future>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nb = nanobind;

std::string nativeStringView(VernonStringView view);

struct Compiler {
    Compiler() : context(vernonCompilerCreate()) {
        if (!context)
            throw std::runtime_error("cannot create Vernon compiler");
        const VernonCpuRuntimeHelpersV1 cpuHelpers{sizeof(VernonCpuRuntimeHelpersV1), &vernonCpuWorkgroupAddressV1,
                                                   &vernonCpuLaneAddressV1, &vernonCpuWorkgroupBarrierV1,
                                                   &vernonCpuWorkgroupIsLeaderV1};
        if (vernonCompilerRegisterCpuRuntimeHelpersV1(context, &cpuHelpers) != VERNON_STATUS_OK) {
            vernonCompilerDestroy(context);
            context = nullptr;
            throw std::runtime_error("cannot register CPU workgroup helpers with the Vernon compiler");
        }
    }
    ~Compiler() { vernonCompilerDestroy(context); }

    VernonCompilerContext *context{};
};

bool targetAvailable(VernonTarget target);

nb::dict targetCapabilities(VernonTarget target);

nb::list planValueAbi(const std::string &moduleText, const std::vector<std::string> &logicalDtypes);

struct StructuredVjp {
    using ResultPtr = std::unique_ptr<VernonPythonStructuredVjp, decltype(&vernonCompilerDestroyPythonStructuredVjp)>;

    explicit StructuredVjp(VernonPythonStructuredVjp *result)
        : result(result, &vernonCompilerDestroyPythonStructuredVjp) {}

    VernonPythonStructuredVjpView current() const { return vernonCompilerGetPythonStructuredVjpView(result.get()); }

    uint64_t tapeBytes() const { return current().tape_bytes; }
    uint64_t activeOperationCount() const { return current().active_operation_count; }
    uint64_t recomputationCost() const { return current().recomputation_cost; }

    nb::list derivativeRules() const {
        VernonPythonStructuredVjpView transformed = current();
        nb::list rules;
        for (size_t index = 0; index < transformed.derivative_rule_count; ++index)
            rules.append(nativeStringView(transformed.derivative_rules[index]));
        return rules;
    }

    nb::list requiredPrimalPaths() const {
        VernonPythonStructuredVjpView transformed = current();
        nb::list paths;
        for (size_t index = 0; index < transformed.required_primal_path_count; ++index)
            paths.append(nativeStringView(transformed.required_primal_paths[index]));
        return paths;
    }

    nb::dict sourceKindCounts() const {
        VernonPythonStructuredVjpView transformed = current();
        nb::dict counts;
        for (size_t index = 0; index < transformed.source_kind_count; ++index)
            counts[nativeStringView(transformed.source_kind_counts[index].name).c_str()] =
                transformed.source_kind_counts[index].value;
        return counts;
    }

    nb::dict costComponents() const {
        VernonPythonStructuredVjpView transformed = current();
        nb::dict costs;
        for (size_t index = 0; index < transformed.cost_component_count; ++index)
            costs[nativeStringView(transformed.cost_components[index].name).c_str()] =
                transformed.cost_components[index].value;
        return costs;
    }

    std::string selectedPolicy() const { return nativeStringView(current().selected_policy); }

    nb::dict profiles(const std::string &identity) {
        if (vernonCompilerFinalizePythonStructuredVjp(result.get(), {identity.data(), identity.size()}) !=
            VERNON_STATUS_OK) {
            const std::string diagnostics = nativeStringView(current().diagnostics);
            throw std::invalid_argument(diagnostics.empty() ? "cannot finalize structured VJP profiles" : diagnostics);
        }
        VernonPythonStructuredVjpView transformed = current();
        nb::dict profiles;
        profiles["forward_with_tape"] = nativeStringView(transformed.forward_module);
        profiles["backward"] = nativeStringView(transformed.backward_module);
        return profiles;
    }

    ResultPtr result;
};

std::unique_ptr<StructuredVjp> buildStructuredVjp(const std::string &moduleText, const std::string &entry,
                                                  const std::vector<std::string> &wrtPaths,
                                                  const std::vector<std::string> &outputPaths,
                                                  const std::string &forwardSymbol, const std::string &backwardSymbol);

struct Runtime;
RhiHostState *runtimeRhiHost(const Runtime *runtime);
struct CompiledProgram;
struct LoadedPipeline;
struct PipelineInvocationBuilder;

using SharedCompileResult = std::shared_ptr<VernonCompileResult>;

struct CompiledProgram {
    CompiledProgram(VernonCompileResult *result, VernonTarget target)
        : result(result, &vernonCompileResultDestroy), target(target) {
        if (!this->result)
            throw std::runtime_error("compiler returned no result");
    }

    bool ok() const { return vernonCompileResultGetStatus(result.get()) == VERNON_STATUS_OK; }

    VernonStatus status() const { return vernonCompileResultGetStatus(result.get()); }

    std::string diagnostics() const { return nativeStringView(vernonCompileResultGetDiagnostics(result.get())); }

    std::string reflection() const { return nativeStringView(vernonCompileResultGetReflection(result.get())); }

    nb::list artifacts() const {
        nb::list values;
        for (size_t index = 0; index < vernonCompileResultGetArtifactCount(result.get()); ++index) {
            VernonStringView name = vernonCompileResultGetArtifactName(result.get(), index);
            VernonStringView data = vernonCompileResultGetArtifactData(result.get(), index);
            values.append(nb::make_tuple(nativeStringView(name), nb::bytes(data.data, data.size)));
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
};

struct CpuTargetOptionStrings {
    std::string triple;
    std::string processor;
    std::string features;
};

CpuTargetOptionStrings parseCpuTargetOptions(const nb::dict &targetOptions);

std::unique_ptr<CompiledProgram> compileProgramResult(Compiler &compiler, const std::string &mlir, VernonTarget target,
                                                      const nb::dict &targetOptions);

std::vector<std::unique_ptr<CompiledProgram>> compileCpuProgramResults(const std::vector<std::string> &modules,
                                                                       const nb::dict &targetOptions);

struct PipelineParameterMetadata {
    uint32_t slot{};
    std::string name;
    VernonPipelineArgumentKind kind{};
    uint32_t elementByteSize{};
    uint32_t elementAlignment{};
    std::string layoutHash;
    std::vector<VernonValueLeafView> elementLeaves;
    VernonValueAccess access{};
    VernonImageBindingRole imageBindingRole{VERNON_IMAGE_BINDING_SAMPLED};
    VernonTextureFormat storageImageFormat{};
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

PipelineParameterMetadata parameterMetadata(const VernonPipelineParameterView &view);

PipelineOutputMetadata outputMetadata(const VernonPipelineOutputView &view);

struct PythonRuntimeSubmission {
    explicit PythonRuntimeSubmission(VernonSubmission *value) : handle(value) {}
    ~PythonRuntimeSubmission() { vernonSubmissionDestroy(handle); }
    PythonRuntimeSubmission(const PythonRuntimeSubmission &) = delete;
    PythonRuntimeSubmission &operator=(const PythonRuntimeSubmission &) = delete;

    void wait() {
        if (vernonSubmissionWait(handle) != VERNON_STATUS_OK)
            throw std::runtime_error("pipeline submission failed");
    }

    uint32_t state() const {
        VernonSubmissionState value{};
        if (vernonSubmissionGetState(handle, &value) != VERNON_STATUS_OK)
            throw std::runtime_error("cannot query pipeline submission");
        return static_cast<uint32_t>(value);
    }

    VernonSubmission *handle{};
};

struct PreparedPipelineArgument {
    VernonPipelineArgument value{};
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;
    std::string layoutHash;
    std::vector<VernonValueLeafView> elementLeaves;
    nb::object owner;
};

struct PipelineInvocationBuilder {
    PipelineInvocationBuilder(Runtime *owner, VernonRuntimeContext *runtime, VernonLoadedPipeline *pipeline)
        : owner(owner), runtime(runtime), pipeline(pipeline) {}
    PipelineInvocationBuilder(const PipelineInvocationBuilder &) = delete;
    PipelineInvocationBuilder &operator=(const PipelineInvocationBuilder &) = delete;

    PipelineParameterMetadata resolveParameter(const nb::object &identifier) {
        VernonPipelineParameterView view{};
        if (nb::isinstance<nb::str>(identifier)) {
            const std::string name = nb::cast<std::string>(identifier);
            if (vernonRuntimeLoadedPipelineFindParameter(pipeline, {name.data(), name.size()}, &view) !=
                VERNON_STATUS_OK)
                throw std::invalid_argument("unknown pipeline parameter '" + name + "'");
            PipelineParameterMetadata metadata = parameterMetadata(view);
            if (view.kind == VERNON_PIPELINE_IMAGE) {
                VernonPipelineImageConstraintView constraint{};
                constraint.struct_size = sizeof(constraint);
                if (vernonRuntimeLoadedPipelineFindImageConstraint(pipeline, {name.data(), name.size()}, &constraint) !=
                    VERNON_STATUS_OK)
                    throw std::runtime_error("cannot query pipeline image constraint");
                metadata.imageBindingRole = constraint.binding_role;
                metadata.storageImageFormat = constraint.storage_format;
            }
            return metadata;
        }
        if (!nb::isinstance<nb::int_>(identifier))
            throw std::invalid_argument("pipeline parameter must be a name or slot");
        const uint32_t slot = nb::cast<uint32_t>(identifier);
        const size_t count = vernonRuntimeLoadedPipelineGetParameterCount(pipeline);
        for (size_t index = 0; index < count; ++index) {
            if (vernonRuntimeLoadedPipelineGetParameterByIndex(pipeline, index, &view) == VERNON_STATUS_OK &&
                view.slot == slot) {
                PipelineParameterMetadata metadata = parameterMetadata(view);
                if (view.kind == VERNON_PIPELINE_IMAGE) {
                    VernonPipelineImageConstraintView constraint{};
                    constraint.struct_size = sizeof(constraint);
                    if (vernonRuntimeLoadedPipelineGetImageConstraintByParameterIndex(pipeline, index, &constraint) !=
                        VERNON_STATUS_OK)
                        throw std::runtime_error("cannot query pipeline image constraint");
                    metadata.imageBindingRole = constraint.binding_role;
                    metadata.storageImageFormat = constraint.storage_format;
                }
                return metadata;
            }
        }
        throw std::invalid_argument("unknown pipeline parameter slot " + std::to_string(slot));
    }

    std::unique_ptr<PreparedPipelineArgument> createArgument(const PipelineParameterMetadata &parameter,
                                                             VernonPipelineArgumentKind kind) {
        if (parameter.kind != kind)
            throw std::invalid_argument("pipeline parameter '" + parameter.name + "' has a different reflected kind");
        auto prepared = std::make_unique<PreparedPipelineArgument>();
        PreparedPipelineArgument &result = *prepared;
        result.value.slot = parameter.slot;
        result.value.kind = kind;
        if (kind == VERNON_PIPELINE_TENSOR) {
            result.layoutHash = parameter.layoutHash;
            result.elementLeaves = parameter.elementLeaves;
            result.value.tensor.element_layout = {
                sizeof(VernonValueLayoutView), parameter.elementByteSize,
                parameter.elementAlignment,    {result.layoutHash.data(), result.layoutHash.size()},
                result.elementLeaves.data(),   result.elementLeaves.size(),
            };
        }
        return prepared;
    }

    PipelineInvocationBuilder &preparedArgument(PreparedPipelineArgument &prepared) {
        if (!slots.insert(prepared.value.slot).second)
            throw std::invalid_argument("pipeline parameter was already bound");
        arguments.push_back(&prepared);
        return *this;
    }

    PipelineInvocationBuilder &ownedArgument(std::unique_ptr<PreparedPipelineArgument> prepared) {
        PreparedPipelineArgument &value = *prepared;
        ownedArguments.push_back(std::move(prepared));
        return preparedArgument(value);
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

    std::unique_ptr<PreparedPipelineArgument> prepareHostTensor(const nb::object &identifier, const nb::object &array) {
        if (!nb::isinstance(array, nb::module_::import_("numpy").attr("ndarray")))
            throw std::invalid_argument("host tensor must be a NumPy ndarray");
        const PipelineParameterMetadata parameter = resolveParameter(identifier);
        auto prepared = createArgument(parameter, VERNON_PIPELINE_TENSOR);
        PreparedPipelineArgument &argument = *prepared;
        const std::vector<uint64_t> arrayShape = nb::cast<std::vector<uint64_t>>(array.attr("shape"));
        const std::vector<int64_t> arrayStrides = nb::cast<std::vector<int64_t>>(array.attr("strides"));
        if (arrayStrides.size() != arrayShape.size())
            throw std::invalid_argument("NumPy Tensor shape/stride mismatch");
        const size_t elementSize = nb::cast<size_t>(array.attr("dtype").attr("itemsize"));
        size_t rank = arrayShape.size();
        if (elementSize != parameter.elementByteSize) {
            size_t trailingSize = elementSize;
            while (rank && trailingSize < parameter.elementByteSize) {
                const size_t dimension = --rank;
                if (arrayStrides[dimension] != static_cast<int64_t>(trailingSize) ||
                    arrayShape[dimension] > std::numeric_limits<size_t>::max() / trailingSize)
                    throw std::invalid_argument("host tensor aggregate element storage must be contiguous");
                trailingSize *= static_cast<size_t>(arrayShape[dimension]);
            }
            if (trailingSize != parameter.elementByteSize)
                throw std::invalid_argument("host tensor element size does not match pipeline reflection");
        }
        argument.shape.assign(arrayShape.begin(), arrayShape.begin() + static_cast<std::ptrdiff_t>(rank));
        argument.strides.assign(arrayStrides.begin(), arrayStrides.begin() + static_cast<std::ptrdiff_t>(rank));
        for (uint64_t extent : argument.shape)
            if (!extent)
                throw std::invalid_argument("NumPy Tensor dimensions must be positive");
        VernonTensorView layoutProbe{};
        layoutProbe.element_layout = argument.value.tensor.element_layout;
        layoutProbe.rank = static_cast<uint32_t>(rank);
        layoutProbe.shape = argument.shape.data();
        layoutProbe.byte_strides = argument.strides.data();
        size_t before = 0;
        size_t after = 0;
        size_t span = 0;
        if (!vernon::runtime::tensorRelativeByteBounds(layoutProbe, before, after) ||
            !vernon::runtime::tensorRequiredSpan(layoutProbe, span))
            throw std::invalid_argument("NumPy Tensor byte span overflows");
        if (array.attr("dtype").attr("fields").is_none() && parameter.elementLeaves.size() == 1 &&
            parameter.elementLeaves[0].scalar_count == 1 && parameter.elementLeaves[0].byte_offset == 0 &&
            numpyDataType(array) != static_cast<VernonDataType>(parameter.elementLeaves[0].dtype))
            throw std::invalid_argument("host tensor dtype does not match pipeline reflection");
        argument.owner = array;
        const uintptr_t data = nb::cast<uintptr_t>(array.attr("ctypes").attr("data"));
        uintptr_t allocation = data - before;
        size_t allocationSize = span;
        nb::object base = array.attr("base");
        while (!base.is_none() && nb::isinstance(base, nb::module_::import_("numpy").attr("ndarray"))) {
            const uintptr_t candidate = nb::cast<uintptr_t>(base.attr("ctypes").attr("data"));
            const size_t candidateSize = nb::cast<size_t>(base.attr("nbytes"));
            if (candidate > data || data - candidate > candidateSize || before > data - candidate ||
                after + parameter.elementByteSize > candidateSize - (data - candidate))
                break;
            allocation = candidate;
            allocationSize = candidateSize;
            base = base.attr("base");
        }
        argument.value.tensor.struct_size = sizeof(VernonTensorView);
        argument.value.tensor.storage = VERNON_TENSOR_HOST;
        argument.value.tensor.host_data = reinterpret_cast<const void *>(allocation);
        argument.value.tensor.access = parameter.access;
        argument.value.tensor.rank = static_cast<uint32_t>(argument.shape.size());
        argument.value.tensor.shape = argument.shape.data();
        argument.value.tensor.byte_strides = argument.strides.data();
        argument.value.tensor.byte_offset = data - allocation;
        argument.value.tensor.byte_size = allocationSize;
        if (argument.value.tensor.access != VERNON_ACCESS_READ &&
            !vernon::runtime::tensorByteLayoutInjective(argument.value.tensor))
            throw std::invalid_argument("writable NumPy Tensor must have an internally injective layout");
        return prepared;
    }

    PipelineInvocationBuilder &hostTensor(const nb::object &identifier, const nb::object &array) {
        return ownedArgument(prepareHostTensor(identifier, array));
    }

    std::unique_ptr<PreparedPipelineArgument> prepareRhiTensor(const nb::object &identifier, RhiBuffer *buffer,
                                                               uint32_t access, const std::vector<uint64_t> &shape,
                                                               const std::vector<int64_t> &strides, size_t offset) {
        if (!buffer || shape.size() != strides.size() || shape.empty())
            throw std::invalid_argument("RHI Tensor shape and strides must have equal non-zero rank");
        const PipelineParameterMetadata parameter = resolveParameter(identifier);
        auto prepared = createArgument(parameter, VERNON_PIPELINE_TENSOR);
        PreparedPipelineArgument &argument = *prepared;
        argument.owner = nb::cast(buffer, nb::rv_policy::reference);
        argument.shape = shape;
        argument.strides = strides;
        argument.value.tensor.struct_size = sizeof(VernonTensorView);
        argument.value.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
        if (vernonRuntimeReferenceRhiBuffer(runtime, buffer->handle, 0, buffer->size,
                                            &argument.value.tensor.resource) != VERNON_STATUS_OK)
            throw std::invalid_argument("RHI buffer belongs to another Runtime device");
        argument.value.tensor.access = static_cast<VernonValueAccess>(access);
        argument.value.tensor.rank = static_cast<uint32_t>(shape.size());
        argument.value.tensor.shape = argument.shape.data();
        argument.value.tensor.byte_strides = argument.strides.data();
        argument.value.tensor.byte_offset = offset;
        argument.value.tensor.byte_size = buffer->size;
        if (argument.value.tensor.access != VERNON_ACCESS_READ &&
            !vernon::runtime::tensorByteLayoutInjective(argument.value.tensor))
            throw std::invalid_argument("writable RHI Tensor must have an internally injective layout");
        return prepared;
    }

    PipelineInvocationBuilder &rhiTensor(const nb::object &identifier, RhiBuffer *buffer, uint32_t access,
                                         const std::vector<uint64_t> &shape, const std::vector<int64_t> &strides,
                                         size_t offset) {
        return ownedArgument(prepareRhiTensor(identifier, buffer, access, shape, strides, offset));
    }

    std::unique_ptr<PreparedPipelineArgument> prepareRhiTexture(const nb::object &identifier, RhiImageView *view) {
        const PipelineParameterMetadata parameter = resolveParameter(identifier);
        const bool storage = parameter.imageBindingRole == VERNON_IMAGE_BINDING_STORAGE;
        const uint32_t requiredUsage = storage ? VERNON_RHI_IMAGE_STORAGE : VERNON_RHI_IMAGE_SAMPLED;
        if (!view || !(view->image->usage & requiredUsage))
            throw std::invalid_argument("RHI texture usage does not match the pipeline parameter");
        if (storage && view->format != parameter.storageImageFormat)
            throw std::invalid_argument("RHI texture format does not match the storage texture parameter");
        auto prepared = createArgument(parameter, VERNON_PIPELINE_IMAGE);
        PreparedPipelineArgument &argument = *prepared;
        argument.owner = nb::cast(view, nb::rv_policy::reference);
        if (vernonRuntimeReferenceRhiImageView(runtime, view->handle, &argument.value.image.view) != VERNON_STATUS_OK)
            throw std::invalid_argument("RHI image view belongs to another Runtime device");
        return prepared;
    }

    PipelineInvocationBuilder &rhiTexture(const nb::object &identifier, RhiImageView *view) {
        return ownedArgument(prepareRhiTexture(identifier, view));
    }

    std::unique_ptr<PreparedPipelineArgument> prepareRhiSampler(const nb::object &identifier, RhiSampler *sampler) {
        if (!sampler)
            throw std::invalid_argument("RHI sampler is null");
        const PipelineParameterMetadata parameter = resolveParameter(identifier);
        auto prepared = createArgument(parameter, VERNON_PIPELINE_SAMPLER);
        PreparedPipelineArgument &argument = *prepared;
        argument.owner = nb::cast(sampler, nb::rv_policy::reference);
        if (vernonRuntimeReferenceRhiSampler(runtime, sampler->handle, &argument.value.resource) != VERNON_STATUS_OK)
            throw std::invalid_argument("RHI sampler belongs to another Runtime device");
        return prepared;
    }

    PipelineInvocationBuilder &rhiSampler(const nb::object &identifier, RhiSampler *sampler) {
        return ownedArgument(prepareRhiSampler(identifier, sampler));
    }

    PipelineInvocationBuilder &rhiColorAttachment(uint32_t location, RhiImageView *view, uint32_t loadOperation,
                                                  uint32_t storeOperation, const std::array<float, 4> &clearColor) {
        if (!view || !(view->image->usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT) ||
            (view->format == VERNON_TEXTURE_D32_FLOAT || view->format == VERNON_TEXTURE_D32_FLOAT_S8_UINT) ||
            loadOperation > VERNON_RHI_LOAD_DISCARD || storeOperation > VERNON_RHI_STORE_DISCARD)
            throw std::invalid_argument("RHI color attachment is null");
        VernonColorAttachment attachment{};
        attachment.location = location;
        if (vernonRuntimeReferenceRhiImageView(runtime, view->handle, &attachment.view) != VERNON_STATUS_OK)
            throw std::invalid_argument("RHI color attachment belongs to another Runtime device");
        attachment.load_operation = static_cast<VernonRuntimeProviderLoadOperation>(loadOperation);
        attachment.store_operation = static_cast<VernonRuntimeProviderStoreOperation>(storeOperation);
        std::copy(clearColor.begin(), clearColor.end(), attachment.clear_color);
        attachments.push_back(attachment);
        return *this;
    }

    PipelineInvocationBuilder &rhiDepthAttachment(RhiImageView *view, uint32_t loadOperation, uint32_t storeOperation,
                                                  float clearDepth) {
        if (!view || !(view->image->usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT) ||
            (view->format != VERNON_TEXTURE_D32_FLOAT && view->format != VERNON_TEXTURE_D32_FLOAT_S8_UINT) ||
            loadOperation > VERNON_RHI_LOAD_DISCARD || storeOperation > VERNON_RHI_STORE_DISCARD || clearDepth < 0.0f ||
            clearDepth > 1.0f)
            throw std::invalid_argument("RHI depth attachment must use D32 format");
        depthAttachment = {};
        if (vernonRuntimeReferenceRhiImageView(runtime, view->handle, &depthAttachment.view) != VERNON_STATUS_OK)
            throw std::invalid_argument("RHI depth attachment belongs to another Runtime device");
        depthAttachment.load_operation = static_cast<VernonRuntimeProviderLoadOperation>(loadOperation);
        depthAttachment.store_operation = static_cast<VernonRuntimeProviderStoreOperation>(storeOperation);
        depthAttachment.clear_depth = clearDepth;
        hasDepthAttachment = true;
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

    std::unique_ptr<PythonRuntimeSubmission> submit(VernonRuntimeProviderObject *encoder) {
        std::vector<VernonPipelineArgument> values;
        values.reserve(arguments.size());
        for (const PreparedPipelineArgument *argument : arguments)
            values.push_back(argument->value);
        VernonPipelineInvocation invocation{};
        invocation.struct_size = sizeof(invocation);
        invocation.abi_version = VERNON_PIPELINE_VERSION;
        invocation.arguments = values.empty() ? nullptr : values.data();
        invocation.argument_count = values.size();
        invocation.index_binding = hasIndex ? &index : nullptr;
        invocation.color_attachments = attachments.empty() ? nullptr : attachments.data();
        invocation.color_attachment_count = attachments.size();
        invocation.depth_attachment = hasDepthAttachment ? &depthAttachment : nullptr;
        invocation.topology = topology;
        invocation.vertex_count = vertexCount;
        invocation.instance_count = instanceCount;
        invocation.compute_grid = computeGrid;
        std::memcpy(invocation.viewport, viewport, sizeof(viewport));
        std::memcpy(invocation.scissor, scissor, sizeof(scissor));
        if (encoder) {
            if (vernonRuntimePipelineEncode(*encoder, pipeline, &invocation) != VERNON_STATUS_OK)
                throw std::runtime_error("pipeline encoding failed: " +
                                         nativeStringView(vernonRuntimeGetLastError(runtime)));
            return {};
        }
        VernonSubmission *submission{};
        if (vernonRuntimePipelineSubmit(pipeline, &invocation, &submission) != VERNON_STATUS_OK)
            throw std::runtime_error("pipeline submission failed: " +
                                     nativeStringView(vernonRuntimeGetLastError(runtime)));
        return std::make_unique<PythonRuntimeSubmission>(submission);
    }

    std::unique_ptr<PythonRuntimeSubmission> submit() { return submit(nullptr); }

    template <typename Encoder> void encode(const Encoder &encoder) {
        VernonRuntimeProviderObject providerEncoder{};
        if (vernonRuntimeReferenceRhiCommandEncoder(runtime, encoder.native(), &providerEncoder) != VERNON_STATUS_OK)
            throw std::invalid_argument("command encoder belongs to another Runtime device");
        (void)submit(&providerEncoder);
    }

    Runtime *owner{};
    VernonRuntimeContext *runtime{};
    VernonLoadedPipeline *pipeline{};
    std::vector<PreparedPipelineArgument *> arguments;
    std::vector<std::unique_ptr<PreparedPipelineArgument>> ownedArguments;
    std::unordered_set<uint32_t> slots;
    std::vector<VernonColorAttachment> attachments;
    VernonDepthAttachment depthAttachment{};
    VernonIndexBinding index{};
    bool hasDepthAttachment{};
    bool hasIndex{};
    VernonPrimitiveTopology topology{VERNON_TOPOLOGY_TRIANGLE_LIST};
    uint32_t vertexCount{};
    uint32_t instanceCount{};
    VernonLaunchSize computeGrid{};
    uint32_t viewport[4]{};
    uint32_t scissor[4]{};
};

const char *numpyDtypeName(VernonDataType dtype);

size_t autodiffDtypeSize(VernonDataType dtype);

VernonDataType autodiffTangentDtype(VernonDataType primal);

std::string formatShape(const std::vector<uint64_t> &shape);

#endif
