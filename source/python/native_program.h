#ifndef VERNON_PYTHON_NATIVE_PROGRAM_H
#define VERNON_PYTHON_NATIVE_PROGRAM_H

#include "VernonCompiler.h"
#include "VernonRuntime.h"
#include "compiler_python_bridge.h"
#include "native_rhi.h"
#include "runtime/tensor_bridge.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nb = nanobind;

std::string nativeStringView(VernonStringView view);

struct Compiler {
    Compiler() : context(vernonCompilerCreate()) {
        if (!context)
            throw std::runtime_error("cannot create Vernon compiler");
        const VernonCpuRuntimeHelpers cpuHelpers{sizeof(VernonCpuRuntimeHelpers), &vernonCpuWorkgroupAddress,
                                                 &vernonCpuLaneAddress, &vernonCpuWorkgroupBarrier,
                                                 &vernonCpuWorkgroupIsLeader};
        if (vernonCompilerRegisterCpuRuntimeHelpers(context, &cpuHelpers) != VERNON_STATUS_OK) {
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
    bool wholeDispatchRetentionPermitted() const { return current().whole_dispatch_retention_permitted != 0; }

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
std::pair<std::string, std::string> buildProgramBuiltin(const std::string &operation, const std::string &elementType,
                                                        uint32_t rank, const std::vector<std::string> &leafDtypes);

std::string specializeKernelHostConstants(const std::string &moduleText, const std::string &entry,
                                          const std::vector<std::string> &names, const nb::list &values);

struct RuntimeState;
RhiHostState *runtimeRhiHost(const RuntimeState *runtime);
struct CompiledProgram;
struct PythonProgramExecutable;
struct ProgramInvocationBuilder;

using SharedCompileResult = std::shared_ptr<VernonCompileResult>;

// One hashed CPU symbol maps to one JIT address, matching linked .o artifacts.
// Later cooks of the same kernel keep this CompileResult alive instead of
// registering a second ORC copy.
struct InternedCpuJit {
    SharedCompileResult result;
    VernonCpuEntryPoint entry{};
};

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
std::unique_ptr<CompiledProgram> planProgramResult(Compiler &compiler, const std::string &program);
std::unique_ptr<CompiledProgram>
finalizeProgramResult(Compiler &compiler, const std::string &plan,
                      const std::vector<std::tuple<std::string, std::string, std::string, std::string>> &kernels,
                      const std::vector<std::tuple<std::string, std::string, std::vector<uint64_t>>> &shapeFacts);
std::unique_ptr<CompiledProgram> analyzeProgramResult(Compiler &compiler, const std::string &mlir);
std::unique_ptr<CompiledProgram> verifyProgramResult(Compiler &compiler, const std::string &mlir);

std::vector<std::unique_ptr<CompiledProgram>> compileCpuProgramResults(const std::vector<std::string> &modules,
                                                                       const nb::dict &targetOptions);

struct ProgramParameterMetadata {
    struct PathComponent {
        bool field{};
        std::string name;
        uint64_t index{};
    };

    uint32_t slot{};
    std::string name;
    VernonProgramArgumentKind kind{};
    uint32_t elementByteSize{};
    uint32_t elementAlignment{};
    std::string layoutHash;
    std::vector<VernonValueLeafView> elementLeaves;
    std::vector<std::vector<PathComponent>> elementLeafPaths;
    std::vector<std::vector<uint64_t>> elementLeafShapes;
    VernonValueAccess access{};
    VernonImageBindingRole imageBindingRole{VERNON_IMAGE_BINDING_SAMPLED};
    VernonTextureFormat storageImageFormat{};
    std::vector<uint64_t> shape;
};

ProgramParameterMetadata parameterMetadata(const VernonProgramParameterView &view);

struct PreparedProgramArgument {
    PreparedProgramArgument() = default;
    PreparedProgramArgument(const PreparedProgramArgument &other)
        : value(other.value), shape(other.shape), strides(other.strides), layoutHash(other.layoutHash),
          elementLeaves(other.elementLeaves), owner(other.owner) {
        refreshViews();
    }
    PreparedProgramArgument &operator=(const PreparedProgramArgument &) = delete;

    void refreshViews() {
        if (value.kind != VERNON_PROGRAM_TENSOR)
            return;
        value.tensor.shape = shape.empty() ? nullptr : shape.data();
        value.tensor.byte_strides = strides.empty() ? nullptr : strides.data();
        value.tensor.element_layout.layout_hash = {layoutHash.data(), layoutHash.size()};
        value.tensor.element_layout.leaves = elementLeaves.empty() ? nullptr : elementLeaves.data();
    }

    VernonProgramArgument value{};
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;
    std::string layoutHash;
    std::vector<VernonValueLeafView> elementLeaves;
    nb::object owner;
};

struct ProgramInvocationBuilder {
    ProgramInvocationBuilder(std::shared_ptr<RuntimeState> owner, VernonRuntimeContext *runtime,
                             VernonProgramExecutable *executable)
        : owner(std::move(owner)), runtime(runtime), executable(executable) {}
    ProgramInvocationBuilder(const ProgramInvocationBuilder &) = delete;
    ProgramInvocationBuilder &operator=(const ProgramInvocationBuilder &) = delete;

    ProgramParameterMetadata resolveParameter(const nb::object &identifier) {
        VernonProgramParameterView view{};
        if (nb::isinstance<nb::str>(identifier)) {
            const std::string name = nb::cast<std::string>(identifier);
            if (vernonRuntimeProgramExecutableFindParameter(executable, {name.data(), name.size()}, &view) !=
                VERNON_STATUS_OK)
                throw std::invalid_argument("unknown executable parameter '" + name + "'");
            ProgramParameterMetadata metadata = parameterMetadata(view);
            if (view.kind == VERNON_PROGRAM_IMAGE) {
                VernonProgramImageConstraintView constraint{};
                constraint.struct_size = sizeof(constraint);
                if (vernonRuntimeProgramExecutableFindImageConstraint(executable, {name.data(), name.size()},
                                                                      &constraint) != VERNON_STATUS_OK)
                    throw std::runtime_error("cannot query executable image constraint");
                metadata.imageBindingRole = constraint.binding_role;
                metadata.storageImageFormat = constraint.storage_format;
            }
            return metadata;
        }
        if (!nb::isinstance<nb::int_>(identifier))
            throw std::invalid_argument("executable parameter must be a name or slot");
        const uint32_t slot = nb::cast<uint32_t>(identifier);
        const VernonProgramBoundaryRole roles[] = {
            VERNON_PROGRAM_BOUNDARY_INPUT,
            VERNON_PROGRAM_BOUNDARY_OUTPUT,
            VERNON_PROGRAM_BOUNDARY_COTANGENT,
            VERNON_PROGRAM_BOUNDARY_GRADIENT,
        };
        for (VernonProgramBoundaryRole role : roles) {
            const size_t boundaryCount = vernonRuntimeProgramExecutableGetBoundaryCount(executable, role);
            for (size_t index = 0; index < boundaryCount; ++index)
                if (vernonRuntimeProgramExecutableGetBoundaryByIndex(executable, role, index, &view) ==
                        VERNON_STATUS_OK &&
                    view.slot == slot && view.kind != VERNON_PROGRAM_IMAGE)
                    return parameterMetadata(view);
        }
        const size_t count = vernonRuntimeProgramExecutableGetParameterCount(executable);
        for (size_t index = 0; index < count; ++index) {
            if (vernonRuntimeProgramExecutableGetParameterByIndex(executable, index, &view) == VERNON_STATUS_OK &&
                view.slot == slot) {
                ProgramParameterMetadata metadata = parameterMetadata(view);
                if (view.kind == VERNON_PROGRAM_IMAGE) {
                    VernonProgramImageConstraintView constraint{};
                    constraint.struct_size = sizeof(constraint);
                    if (vernonRuntimeProgramExecutableGetImageConstraintByParameterIndex(
                            executable, index, &constraint) != VERNON_STATUS_OK)
                        throw std::runtime_error("cannot query executable image constraint");
                    metadata.imageBindingRole = constraint.binding_role;
                    metadata.storageImageFormat = constraint.storage_format;
                }
                return metadata;
            }
        }
        throw std::invalid_argument("unknown executable parameter slot " + std::to_string(slot));
    }

    std::unique_ptr<PreparedProgramArgument> createArgument(const ProgramParameterMetadata &parameter,
                                                            VernonProgramArgumentKind kind) {
        if (parameter.kind != kind)
            throw std::invalid_argument("executable parameter '" + parameter.name + "' has a different reflected kind");
        auto prepared = std::make_unique<PreparedProgramArgument>();
        PreparedProgramArgument &result = *prepared;
        result.value.slot = parameter.slot;
        result.value.kind = kind;
        if (kind == VERNON_PROGRAM_TENSOR) {
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

    ProgramInvocationBuilder &preparedArgument(PreparedProgramArgument &prepared) {
        if (!slots.insert(prepared.value.slot).second)
            throw std::invalid_argument("executable parameter was already bound");
        arguments.push_back(&prepared);
        return *this;
    }

    ProgramInvocationBuilder &ownedArgument(std::unique_ptr<PreparedProgramArgument> prepared) {
        PreparedProgramArgument &value = *prepared;
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
        throw std::invalid_argument("unsupported NumPy executable dtype '" + name + "'");
    }

    std::unique_ptr<PreparedProgramArgument> prepareHostTensor(const nb::object &identifier, const nb::object &array) {
        if (!nb::isinstance(array, nb::module_::import_("numpy").attr("ndarray")))
            throw std::invalid_argument("host tensor must be a NumPy ndarray");
        const ProgramParameterMetadata parameter = resolveParameter(identifier);
        auto prepared = createArgument(parameter, VERNON_PROGRAM_TENSOR);
        PreparedProgramArgument &argument = *prepared;
        const std::vector<uint64_t> arrayShape = nb::cast<std::vector<uint64_t>>(array.attr("shape"));
        const std::vector<int64_t> arrayStrides = nb::cast<std::vector<int64_t>>(array.attr("strides"));
        if (arrayStrides.size() != arrayShape.size())
            throw std::invalid_argument("NumPy Tensor shape/stride mismatch");
        const size_t elementSize = nb::cast<size_t>(array.attr("dtype").attr("itemsize"));
        size_t rank = arrayShape.size();
        if (elementSize != parameter.elementByteSize) {
            size_t trailingSize = elementSize;
            while (rank > parameter.shape.size()) {
                const size_t dimension = --rank;
                if (arrayStrides[dimension] != static_cast<int64_t>(trailingSize) ||
                    arrayShape[dimension] > std::numeric_limits<size_t>::max() / trailingSize)
                    throw std::invalid_argument("host tensor aggregate element storage must be contiguous");
                trailingSize *= static_cast<size_t>(arrayShape[dimension]);
            }
            if (trailingSize != parameter.elementByteSize)
                throw std::invalid_argument("host tensor element size does not match executable reflection");
        }
        argument.shape.assign(arrayShape.begin(), arrayShape.begin() + static_cast<std::ptrdiff_t>(rank));
        argument.strides.assign(arrayStrides.begin(), arrayStrides.begin() + static_cast<std::ptrdiff_t>(rank));
        VernonTensorView layoutProbe{};
        layoutProbe.element_layout = argument.value.tensor.element_layout;
        layoutProbe.rank = static_cast<uint32_t>(rank);
        layoutProbe.shape = argument.shape.data();
        layoutProbe.byte_strides = argument.strides.data();
        auto bounds = vernon::runtime::tensorRelativeByteBounds(layoutProbe);
        auto span = vernon::runtime::tensorRequiredSpan(layoutProbe);
        if (bounds.isErr() || span.isErr())
            throw std::invalid_argument("NumPy Tensor byte span overflows");
        const auto relativeBounds = bounds.value();
        if (elementSize == parameter.elementByteSize && array.attr("dtype").attr("fields").is_none() &&
            parameter.elementLeaves.size() == 1 && parameter.elementLeaves[0].scalar_count == 1 &&
            parameter.elementLeaves[0].byte_offset == 0 &&
            numpyDataType(array) != static_cast<VernonDataType>(parameter.elementLeaves[0].dtype))
            throw std::invalid_argument("host tensor dtype does not match executable reflection");
        argument.owner = array;
        const uintptr_t data = nb::cast<uintptr_t>(array.attr("ctypes").attr("data"));
        uintptr_t allocation = data - relativeBounds.before;
        size_t allocationSize = span.value();
        nb::object base = array.attr("base");
        while (!base.is_none() && nb::isinstance(base, nb::module_::import_("numpy").attr("ndarray"))) {
            const uintptr_t candidate = nb::cast<uintptr_t>(base.attr("ctypes").attr("data"));
            const size_t candidateSize = nb::cast<size_t>(base.attr("nbytes"));
            if (candidate > data || data - candidate > candidateSize || relativeBounds.before > data - candidate ||
                relativeBounds.after + parameter.elementByteSize > candidateSize - (data - candidate))
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

    std::unique_ptr<PreparedProgramArgument> prepareRhiTensor(const nb::object &identifier, RhiBuffer *buffer,
                                                              uint32_t access, const std::vector<uint64_t> &shape,
                                                              const std::vector<int64_t> &strides, size_t offset) {
        if (!buffer || shape.size() != strides.size())
            throw std::invalid_argument("RHI Tensor shape and strides must have equal rank");
        const ProgramParameterMetadata parameter = resolveParameter(identifier);
        auto prepared = createArgument(parameter, VERNON_PROGRAM_TENSOR);
        PreparedProgramArgument &argument = *prepared;
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
        argument.value.tensor.shape = argument.shape.empty() ? nullptr : argument.shape.data();
        argument.value.tensor.byte_strides = argument.strides.empty() ? nullptr : argument.strides.data();
        argument.value.tensor.byte_offset = offset;
        argument.value.tensor.byte_size = buffer->size;
        if (argument.value.tensor.access != VERNON_ACCESS_READ &&
            !vernon::runtime::tensorByteLayoutInjective(argument.value.tensor))
            throw std::invalid_argument("writable RHI Tensor must have an internally injective layout");
        return prepared;
    }

    std::unique_ptr<PreparedProgramArgument> prepareRhiTexture(const nb::object &identifier, RhiImageView *view) {
        const ProgramParameterMetadata parameter = resolveParameter(identifier);
        const bool storage = parameter.imageBindingRole == VERNON_IMAGE_BINDING_STORAGE;
        const uint32_t requiredUsage = storage ? VERNON_RHI_IMAGE_STORAGE
                                       : parameter.imageBindingRole == VERNON_IMAGE_BINDING_COLOR_ATTACHMENT
                                           ? VERNON_RHI_IMAGE_COLOR_ATTACHMENT
                                       : parameter.imageBindingRole == VERNON_IMAGE_BINDING_DEPTH_STENCIL_ATTACHMENT
                                           ? VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT
                                           : VERNON_RHI_IMAGE_SAMPLED;
        if (!view || !(view->image->usage & requiredUsage))
            throw std::invalid_argument("RHI texture usage does not match the executable parameter");
        if (storage && view->format != parameter.storageImageFormat)
            throw std::invalid_argument("RHI texture format does not match the storage texture parameter");
        auto prepared = createArgument(parameter, VERNON_PROGRAM_IMAGE);
        PreparedProgramArgument &argument = *prepared;
        argument.owner = nb::cast(view, nb::rv_policy::reference);
        if (vernonRuntimeReferenceRhiImageView(runtime, view->handle, &argument.value.image.view) != VERNON_STATUS_OK)
            throw std::invalid_argument("RHI image view belongs to another Runtime device");
        return prepared;
    }

    std::unique_ptr<PreparedProgramArgument> prepareRhiSampler(const nb::object &identifier, RhiSampler *sampler) {
        if (!sampler)
            throw std::invalid_argument("RHI sampler is null");
        const ProgramParameterMetadata parameter = resolveParameter(identifier);
        auto prepared = createArgument(parameter, VERNON_PROGRAM_SAMPLER);
        PreparedProgramArgument &argument = *prepared;
        argument.owner = nb::cast(sampler, nb::rv_policy::reference);
        if (vernonRuntimeReferenceRhiSampler(runtime, sampler->handle, &argument.value.resource) != VERNON_STATUS_OK)
            throw std::invalid_argument("RHI sampler belongs to another Runtime device");
        return prepared;
    }

    ProgramInvocationBuilder &rhiColorAttachment(uint32_t location, const nb::object &viewObject,
                                                 uint32_t loadOperation, uint32_t storeOperation,
                                                 const std::array<float, 4> &clearColor) {
        auto *view = nb::cast<RhiImageView *>(viewObject);
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
        renderResourceOwners.push_back(viewObject);
        return *this;
    }

    ProgramInvocationBuilder &rhiDepthAttachment(const nb::object &viewObject, uint32_t loadOperation,
                                                 uint32_t storeOperation, float clearDepth) {
        auto *view = nb::cast<RhiImageView *>(viewObject);
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
        renderResourceOwners.push_back(viewObject);
        return *this;
    }

    ProgramInvocationBuilder &rhiIndexBinding(const nb::object &bufferObject, uint32_t count, size_t offset) {
        auto *buffer = nb::cast<RhiBuffer *>(bufferObject);
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
        drawResourceOwner = bufferObject;
        return *this;
    }

    ProgramInvocationBuilder &setTopology(uint32_t value) {
        if (value > static_cast<uint32_t>(VERNON_TOPOLOGY_POINT_LIST))
            throw std::invalid_argument("invalid primitive topology");
        topology = static_cast<VernonPrimitiveTopology>(value);
        if (hasGraphicsState)
            graphicsState.topology = topology;
        return *this;
    }

    ProgramInvocationBuilder &counts(uint32_t vertices, uint32_t instances) {
        vertexCount = vertices;
        instanceCount = instances;
        return *this;
    }

    ProgramInvocationBuilder &setViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
        viewport[0] = x;
        viewport[1] = y;
        viewport[2] = width;
        viewport[3] = height;
        return *this;
    }

    ProgramInvocationBuilder &setScissor(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
        scissor[0] = x;
        scissor[1] = y;
        scissor[2] = width;
        scissor[3] = height;
        return *this;
    }

    ProgramInvocationBuilder &setGraphicsState(const nb::object &state) {
        const nb::object raster = state.attr("rasterization");
        const nb::object depth = state.attr("depth_stencil");
        graphicsState = {};
        graphicsState.struct_size = sizeof(graphicsState);
        graphicsState.topology = topology;
        graphicsState.rasterization.cull_mode = nb::cast<uint32_t>(raster.attr("cull_mode"));
        graphicsState.rasterization.front_face = nb::cast<uint32_t>(raster.attr("front_face"));
        graphicsState.rasterization.depth_clamp = nb::cast<bool>(raster.attr("depth_clamp"));
        graphicsState.rasterization.depth_bias_constant = nb::cast<float>(raster.attr("depth_bias_constant"));
        graphicsState.rasterization.depth_bias_slope = nb::cast<float>(raster.attr("depth_bias_slope"));
        graphicsState.rasterization.depth_bias_enabled = graphicsState.rasterization.depth_bias_constant != 0.0f ||
                                                         graphicsState.rasterization.depth_bias_slope != 0.0f;
        graphicsState.depth_stencil.depth_test = nb::cast<bool>(depth.attr("depth_test"));
        graphicsState.depth_stencil.depth_write = nb::cast<bool>(depth.attr("depth_write"));
        graphicsState.depth_stencil.depth_compare = nb::cast<uint32_t>(depth.attr("depth_compare"));
        graphicsState.depth_stencil.stencil_test = nb::cast<bool>(depth.attr("stencil_test"));
        const auto copyFace = [](const nb::object &source, VernonStencilFaceState &target) {
            target.stencil_fail = nb::cast<uint32_t>(source.attr("stencil_fail"));
            target.depth_fail = nb::cast<uint32_t>(source.attr("depth_fail"));
            target.pass = nb::cast<uint32_t>(source.attr("pass_operation"));
            target.compare = nb::cast<uint32_t>(source.attr("compare"));
        };
        copyFace(depth.attr("front"), graphicsState.depth_stencil.front);
        copyFace(depth.attr("back"), graphicsState.depth_stencil.back);
        graphicsState.depth_stencil.stencil_read_mask = nb::cast<uint32_t>(depth.attr("stencil_read_mask"));
        graphicsState.depth_stencil.stencil_write_mask = nb::cast<uint32_t>(depth.attr("stencil_write_mask"));
        colorBlends.clear();
        for (nb::handle item : state.attr("color_blends")) {
            const nb::tuple pair = nb::cast<nb::tuple>(item);
            const uint32_t location = nb::cast<uint32_t>(pair[0]);
            if (location != colorBlends.size())
                throw std::invalid_argument("graphics color blend locations must be contiguous from zero");
            const nb::object source = nb::borrow<nb::object>(pair[1]);
            VernonColorBlendState blend{};
            blend.blend_enabled = nb::cast<bool>(source.attr("enabled"));
            blend.source_color_factor = nb::cast<uint32_t>(source.attr("source_color"));
            blend.destination_color_factor = nb::cast<uint32_t>(source.attr("destination_color"));
            blend.color_operation = nb::cast<uint32_t>(source.attr("color_operation"));
            blend.source_alpha_factor = nb::cast<uint32_t>(source.attr("source_alpha"));
            blend.destination_alpha_factor = nb::cast<uint32_t>(source.attr("destination_alpha"));
            blend.alpha_operation = nb::cast<uint32_t>(source.attr("alpha_operation"));
            blend.write_mask = nb::cast<uint32_t>(source.attr("write_mask"));
            colorBlends.push_back(blend);
        }
        graphicsState.color_blends = colorBlends.empty() ? nullptr : colorBlends.data();
        graphicsState.color_blend_count = colorBlends.size();
        hasGraphicsState = true;
        return *this;
    }

    ProgramInvocationBuilder &setStencilReference(uint32_t value) {
        if (value > 0xff)
            throw std::invalid_argument("stencil reference must be in [0, 255]");
        stencilReference = value;
        return *this;
    }

    void refreshGraphicsControls() const {
        if (!hasGraphicsState)
            return;
        renderPass = {};
        renderPass.struct_size = sizeof(renderPass);
        renderPass.color_attachments = attachments.empty() ? nullptr : attachments.data();
        renderPass.color_attachment_count = attachments.size();
        renderPass.depth_attachment = hasDepthAttachment ? &depthAttachment : nullptr;
        drawCommand = {};
        drawCommand.struct_size = sizeof(drawCommand);
        drawCommand.index_binding = hasIndex ? &index : nullptr;
        drawCommand.vertex_count = vertexCount;
        drawCommand.instance_count = instanceCount;
        dynamicState = {};
        dynamicState.struct_size = sizeof(dynamicState);
        std::memcpy(dynamicState.viewport, viewport, sizeof(viewport));
        std::memcpy(dynamicState.scissor, scissor, sizeof(scissor));
        dynamicState.stencil_reference = stencilReference;
    }

    void collectArguments(std::vector<VernonProgramArgument> &values) const {
        values.clear();
        values.reserve(arguments.size());
        for (const PreparedProgramArgument *argument : arguments)
            values.push_back(argument->value);
        refreshGraphicsControls();
    }

    std::shared_ptr<RuntimeState> owner;
    VernonRuntimeContext *runtime{};
    VernonProgramExecutable *executable{};
    std::vector<PreparedProgramArgument *> arguments;
    std::vector<std::unique_ptr<PreparedProgramArgument>> ownedArguments;
    std::unordered_set<uint32_t> slots;
    std::vector<VernonColorAttachment> attachments;
    std::vector<nb::object> renderResourceOwners;
    nb::object drawResourceOwner;
    VernonDepthAttachment depthAttachment{};
    VernonIndexBinding index{};
    bool hasDepthAttachment{};
    bool hasIndex{};
    VernonPrimitiveTopology topology{VERNON_TOPOLOGY_TRIANGLE_LIST};
    uint32_t vertexCount{};
    uint32_t instanceCount{};
    uint32_t viewport[4]{};
    uint32_t scissor[4]{};
    VernonGraphicsState graphicsState{};
    std::vector<VernonColorBlendState> colorBlends;
    bool hasGraphicsState{};
    uint32_t stencilReference{};
    mutable VernonRenderPass renderPass{};
    mutable VernonDrawCommand drawCommand{};
    mutable VernonDynamicState dynamicState{};
};

inline void appendBindingTokenField(std::string &result, char tag, const char *data, size_t size) {
    result += tag;
    result += std::to_string(size);
    result += ':';
    result.append(data, size);
}

inline void appendCanonicalBindingToken(std::string &result, PyObject *value) {
    if (PyTuple_Check(value)) {
        const Py_ssize_t count = PyTuple_GET_SIZE(value);
        result += 't';
        result += std::to_string(count);
        result += ':';
        for (Py_ssize_t index = 0; index < count; ++index)
            appendCanonicalBindingToken(result, PyTuple_GET_ITEM(value, index));
        return;
    }
    if (PyBytes_Check(value)) {
        char *data = nullptr;
        Py_ssize_t size = 0;
        if (PyBytes_AsStringAndSize(value, &data, &size) != 0)
            throw nb::python_error();
        appendBindingTokenField(result, 'b', data, static_cast<size_t>(size));
        return;
    }
    if (PyUnicode_Check(value)) {
        Py_ssize_t size = 0;
        const char *data = PyUnicode_AsUTF8AndSize(value, &size);
        if (!data)
            throw nb::python_error();
        appendBindingTokenField(result, 's', data, static_cast<size_t>(size));
        return;
    }
    if (PyLong_Check(value)) {
        nb::object text = nb::steal<nb::object>(PyObject_Str(value));
        if (!text.is_valid())
            throw nb::python_error();
        Py_ssize_t size = 0;
        const char *data = PyUnicode_AsUTF8AndSize(text.ptr(), &size);
        if (!data)
            throw nb::python_error();
        appendBindingTokenField(result, 'i', data, static_cast<size_t>(size));
        return;
    }
    throw std::invalid_argument("Program binding token must contain only tuple, string, integer, or bytes fields");
}

inline std::string canonicalBindingToken(const nb::object &value) {
    std::string result;
    appendCanonicalBindingToken(result, value.ptr());
    return result;
}

inline void retainPythonProgramBinding(void *object) {
    const PyGILState_STATE state = PyGILState_Ensure();
    Py_INCREF(static_cast<PyObject *>(object));
    PyGILState_Release(state);
}

inline void releasePythonProgramBinding(void *object) {
    const PyGILState_STATE state = PyGILState_Ensure();
    Py_DECREF(static_cast<PyObject *>(object));
    PyGILState_Release(state);
}

struct PythonInvocationOutcome {
    VernonStatus status{VERNON_STATUS_OK};
    VernonInvocationSubmissionState submission{VERNON_INVOCATION_NOT_SUBMITTED};
    std::vector<VernonBoundaryMutation> mutations;
    std::string error;

    bool ok() const { return status == VERNON_STATUS_OK; }
    nb::list mutationList() const {
        nb::list result;
        for (const VernonBoundaryMutation &mutation : mutations)
            result.append(nb::make_tuple(static_cast<uint32_t>(mutation.kind), mutation.slot,
                                         static_cast<uint32_t>(mutation.state)));
        return result;
    }
};

// Nanobind owns only Python object conversion. Transactional binding state,
// token comparison, snapshots, telemetry, and payload leases live in the
// runtime ProgramInstance referenced by this adapter.
struct PythonProgramInvocationAdapter {
    PythonProgramInvocationAdapter(std::shared_ptr<RuntimeState> owner, VernonRuntimeContext *runtime,
                                   VernonProgramExecutable *executable, VernonProgramInstance *nativeInstance)
        : builder(std::make_unique<ProgramInvocationBuilder>(std::move(owner), runtime, executable)),
          nativeInvocation(vernonRuntimeProgramInstanceBeginInvocation(nativeInstance)) {
        if (!nativeInvocation)
            throw std::runtime_error("failed to begin native Program invocation");
    }
    PythonProgramInvocationAdapter(const PythonProgramInvocationAdapter &) = delete;
    PythonProgramInvocationAdapter &operator=(const PythonProgramInvocationAdapter &) = delete;
    ~PythonProgramInvocationAdapter() { vernonRuntimeProgramInvocationDestroy(nativeInvocation); }

    ProgramInvocationBuilder &builderView() const { return *builder; }
    ProgramInvocationBuilder &newControlBuilder() {
        controlBuilders.push_back(
            std::make_unique<ProgramInvocationBuilder>(builder->owner, builder->runtime, builder->executable));
        return *controlBuilders.back();
    }

    void setAutodiffOptions(const nb::object &checkpointMemoryBudget, const std::string &checkpointPolicy) {
        VernonProgramAutodiffInvocationOptions options{};
        options.struct_size = sizeof(options);
        options.abi_version = VERNON_PROGRAM_AUTODIFF_INVOCATION_OPTIONS_VERSION;
        options.has_checkpoint_memory_budget = !checkpointMemoryBudget.is_none();
        if (options.has_checkpoint_memory_budget)
            options.checkpoint_memory_budget = nb::cast<uint64_t>(checkpointMemoryBudget);
        options.checkpoint_policy = {checkpointPolicy.data(), checkpointPolicy.size()};
        if (vernonRuntimeProgramInvocationSetAutodiffOptions(nativeInvocation, &options) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to set Program invocation autodiff options");
    }

    void bind(uint32_t slot, const nb::object &token, const nb::callable &prepare, uint64_t uploadBytes = 0,
              uint64_t uploadRanges = 0, bool uploadPrecedesLookup = false) {
        const std::string key = canonicalBindingToken(token);
        VernonProgramBindingToken bindingToken{sizeof(bindingToken), key.data(), key.size()};
        uint8_t reused = 0;
        const uint64_t reusedUploadBytes = uploadPrecedesLookup ? uploadBytes : 0;
        const uint64_t reusedUploadRanges = uploadPrecedesLookup ? uploadRanges : 0;
        if (vernonRuntimeProgramInvocationTryReuse(nativeInvocation, slot, &bindingToken, reusedUploadBytes,
                                                   reusedUploadRanges, &reused) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to query native Program binding reuse");
        if (reused)
            return;
        nb::object prepared = prepare();
        auto *argument = nb::cast<PreparedProgramArgument *>(prepared);
        if (!argument)
            throw std::invalid_argument("binding prepare callback did not return a prepared argument");
        builder->preparedArgument(*argument);
        VernonProgramResourceLease lease{sizeof(lease), prepared.ptr(), retainPythonProgramBinding,
                                         releasePythonProgramBinding};
        if (vernonRuntimeProgramInvocationBind(nativeInvocation, &bindingToken, &argument->value, &lease, uploadBytes,
                                               uploadRanges) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to bind native Program argument");
    }

    void bindRenderPass(uint32_t slot, const nb::object &token, ProgramInvocationBuilder &control) {
        if (!control.hasGraphicsState)
            throw std::invalid_argument("Program RenderPass control builder has no typed graphics state");
        control.refreshGraphicsControls();
        const std::string key = canonicalBindingToken(token);
        VernonProgramBindingToken bindingToken{sizeof(bindingToken), key.data(), key.size()};
        std::vector<VernonProgramResourceLease> leases;
        leases.reserve(control.renderResourceOwners.size());
        for (const nb::object &owner : control.renderResourceOwners)
            leases.push_back({sizeof(VernonProgramResourceLease), owner.ptr(), retainPythonProgramBinding,
                              releasePythonProgramBinding});
        if (vernonRuntimeProgramInvocationBindRenderPass(nativeInvocation, slot, &bindingToken, &control.renderPass,
                                                         leases.data(), leases.size()) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to bind native Program RenderPass control");
    }

    void bindDrawCommand(uint32_t slot, const nb::object &token, ProgramInvocationBuilder &control) {
        if (!control.hasGraphicsState)
            throw std::invalid_argument("Program DrawCommand control builder has no typed graphics state");
        control.refreshGraphicsControls();
        const std::string key = canonicalBindingToken(token);
        VernonProgramBindingToken bindingToken{sizeof(bindingToken), key.data(), key.size()};
        VernonProgramResourceLease lease{};
        const VernonProgramResourceLease *leasePointer = nullptr;
        if (control.drawResourceOwner.is_valid()) {
            lease = {sizeof(lease), control.drawResourceOwner.ptr(), retainPythonProgramBinding,
                     releasePythonProgramBinding};
            leasePointer = &lease;
        }
        if (vernonRuntimeProgramInvocationBindDrawCommand(nativeInvocation, slot, &bindingToken, &control.drawCommand,
                                                          leasePointer) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to bind native Program DrawCommand control");
    }

    void bindDynamicState(uint32_t slot, const nb::object &token, ProgramInvocationBuilder &control) {
        if (!control.hasGraphicsState)
            throw std::invalid_argument("Program DynamicState control builder has no typed graphics state");
        control.refreshGraphicsControls();
        const std::string key = canonicalBindingToken(token);
        VernonProgramBindingToken bindingToken{sizeof(bindingToken), key.data(), key.size()};
        if (vernonRuntimeProgramInvocationBindDynamicState(nativeInvocation, slot, &bindingToken,
                                                           &control.dynamicState) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to bind native Program DynamicState control");
    }

    PythonInvocationOutcome execute(bool retainPullback) {
        PythonInvocationOutcome result;
        result.mutations.resize(vernonRuntimeProgramExecutableGetMutationCapacity(builder->executable));
        VernonInvocationMutationOutcome outcome{sizeof(VernonInvocationMutationOutcome),
                                                VERNON_INVOCATION_NOT_SUBMITTED,
                                                result.mutations.data(),
                                                result.mutations.size(),
                                                0,
                                                {}};
        result.status = vernonRuntimeProgramInvocationExecute(nativeInvocation, retainPullback, &outcome);
        result.submission = outcome.submission;
        result.mutations.resize(outcome.mutation_count);
        if (result.status != VERNON_STATUS_OK)
            result.error = nativeStringView(vernonRuntimeGetLastError(builder->runtime));
        return result;
    }

    void commit() {
        if (vernonRuntimeProgramInvocationCommit(nativeInvocation, nullptr) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to commit native Program invocation");
        finished = true;
    }
    VernonPullback *commitPullback() {
        VernonPullback *pullback = nullptr;
        if (vernonRuntimeProgramInvocationCommit(nativeInvocation, &pullback) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to commit native Program invocation");
        finished = true;
        return pullback;
    }
    void rollback() {
        vernonRuntimeProgramInvocationRollback(nativeInvocation);
        finished = true;
    }
    bool isFinished() const { return finished; }

    std::unique_ptr<ProgramInvocationBuilder> builder;
    std::vector<std::unique_ptr<ProgramInvocationBuilder>> controlBuilders;
    VernonProgramInvocation *nativeInvocation{};
    bool finished{};
};

struct PythonProgramInstanceAdapter {
    PythonProgramInstanceAdapter(std::shared_ptr<RuntimeState> owner, VernonRuntimeContext *runtime,
                                 VernonProgramExecutable *executable)
        : owner(std::move(owner)), runtime(runtime), executable(executable),
          nativeInvocationInstance(vernonRuntimeProgramInstanceCreate(executable)) {
        if (!nativeInvocationInstance)
            throw std::runtime_error("failed to create native Program instance");
    }
    PythonProgramInstanceAdapter(const PythonProgramInstanceAdapter &) = delete;
    PythonProgramInstanceAdapter &operator=(const PythonProgramInstanceAdapter &) = delete;
    ~PythonProgramInstanceAdapter() { vernonRuntimeProgramInstanceDestroy(nativeInvocationInstance); }

    std::unique_ptr<PythonProgramInvocationAdapter> beginInvocation() {
        return std::make_unique<PythonProgramInvocationAdapter>(owner, runtime, executable, nativeInvocationInstance);
    }

    nb::dict telemetryView() const {
        VernonProgramBindingTelemetry telemetry{};
        telemetry.struct_size = sizeof(telemetry);
        if (vernonRuntimeProgramInstanceGetTelemetry(nativeInvocationInstance, &telemetry) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to query native Program binding telemetry");
        nb::dict result;
        result["prepare_count"] = telemetry.prepare_count;
        result["reuse_count"] = telemetry.reuse_count;
        result["rollback_count"] = telemetry.rollback_count;
        result["upload_bytes"] = telemetry.upload_bytes;
        result["upload_ranges"] = telemetry.upload_ranges;
        return result;
    }

    std::shared_ptr<RuntimeState> owner;
    VernonRuntimeContext *runtime{};
    VernonProgramExecutable *executable{};
    VernonProgramInstance *nativeInvocationInstance{};
};

const char *numpyDtypeName(VernonDataType dtype);

size_t autodiffDtypeSize(VernonDataType dtype);

VernonDataType autodiffTangentDtype(VernonDataType primal);

std::string formatShape(const std::vector<uint64_t> &shape);

#endif
