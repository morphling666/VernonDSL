#ifndef VERNON_PYTHON_NATIVE_STAGE_H
#define VERNON_PYTHON_NATIVE_STAGE_H

#include "native_program.h"

struct Runtime;

struct PythonStageExecutable;

struct StageInvocationBuilder {
    StageInvocationBuilder(VernonRuntimeContext *runtime, VernonStageExecutable *stage)
        : runtime(runtime), stage(stage) {}
    StageInvocationBuilder(const StageInvocationBuilder &) = delete;
    StageInvocationBuilder &operator=(const StageInvocationBuilder &) = delete;

    ProgramParameterMetadata resolveParameter(const nb::object &identifier) const {
        VernonProgramParameterView view{};
        if (nb::isinstance<nb::str>(identifier)) {
            const std::string name = nb::cast<std::string>(identifier);
            if (vernonRuntimeStageExecutableFindParameter(stage, {name.data(), name.size()}, &view) != VERNON_STATUS_OK)
                throw std::invalid_argument("unknown Stage parameter '" + name + "'");
            return parameterMetadata(view);
        }
        if (!nb::isinstance<nb::int_>(identifier))
            throw std::invalid_argument("Stage parameter must be a name or slot");
        const uint32_t slot = nb::cast<uint32_t>(identifier);
        for (size_t index = 0; index < vernonRuntimeStageExecutableGetParameterCount(stage); ++index) {
            if (vernonRuntimeStageExecutableGetParameterByIndex(stage, index, &view) == VERNON_STATUS_OK &&
                view.slot == slot)
                return parameterMetadata(view);
        }
        throw std::invalid_argument("unknown Stage parameter slot " + std::to_string(slot));
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
        throw std::invalid_argument("unsupported NumPy Stage dtype '" + name + "'");
    }

    StageInvocationBuilder &hostTensor(const nb::object &identifier, const nb::object &array) {
        if (!nb::isinstance(array, nb::module_::import_("numpy").attr("ndarray")))
            throw std::invalid_argument("host tensor must be a NumPy ndarray");
        const ProgramParameterMetadata parameter = resolveParameter(identifier);
        if (parameter.kind != VERNON_PROGRAM_TENSOR)
            throw std::invalid_argument("Stage parameter '" + parameter.name + "' is not a Tensor");
        auto prepared = std::make_unique<PreparedProgramArgument>();
        PreparedProgramArgument &argument = *prepared;
        argument.value.slot = parameter.slot;
        argument.value.kind = VERNON_PROGRAM_TENSOR;
        argument.layoutHash = parameter.layoutHash;
        argument.elementLeaves = parameter.elementLeaves;
        argument.value.tensor.element_layout = {
            sizeof(VernonValueLayoutView), parameter.elementByteSize,
            parameter.elementAlignment,    {argument.layoutHash.data(), argument.layoutHash.size()},
            argument.elementLeaves.data(), argument.elementLeaves.size(),
        };
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
                throw std::invalid_argument("host tensor element size does not match Stage reflection");
        }
        argument.shape.assign(arrayShape.begin(), arrayShape.begin() + static_cast<std::ptrdiff_t>(rank));
        argument.strides.assign(arrayStrides.begin(), arrayStrides.begin() + static_cast<std::ptrdiff_t>(rank));
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
            throw std::invalid_argument("host tensor dtype does not match Stage reflection");
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
        slots.insert(parameter.slot);
        arguments.push_back(&argument);
        ownedArguments.push_back(std::move(prepared));
        return *this;
    }

    StageInvocationBuilder &grid(uint32_t x, uint32_t y, uint32_t z) {
        computeGrid = {x, y, z};
        return *this;
    }

    std::unique_ptr<PythonRuntimeSubmission> submit() {
        std::vector<VernonProgramArgument> values;
        values.reserve(arguments.size());
        for (const PreparedProgramArgument *argument : arguments)
            values.push_back(argument->value);
        VernonStageInvocationDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.abi_version = VERNON_PROGRAM_VERSION;
        descriptor.arguments = values.empty() ? nullptr : values.data();
        descriptor.argument_count = values.size();
        descriptor.compute_grid = computeGrid;
        VernonSubmission *submission = nullptr;
        if (vernonRuntimeStageSubmit(stage, &descriptor, &submission) != VERNON_STATUS_OK || !submission)
            throw std::runtime_error(nativeStringView(vernonRuntimeGetLastError(runtime)));
        return std::make_unique<PythonRuntimeSubmission>(submission);
    }

    VernonRuntimeContext *runtime{};
    VernonStageExecutable *stage{};
    VernonLaunchSize computeGrid{1, 1, 1};
    std::unordered_set<uint32_t> slots;
    std::vector<std::unique_ptr<PreparedProgramArgument>> ownedArguments;
    std::vector<PreparedProgramArgument *> arguments;
};

struct PythonStageExecutable {
    PythonStageExecutable(VernonRuntimeContext *runtime, VernonStageExecutable *stage,
                          SharedCompileResult retainedResult = {})
        : runtime(runtime), stage(stage), retainedResult(std::move(retainedResult)) {}
    ~PythonStageExecutable() { vernonRuntimeStageExecutableDestroy(stage); }
    PythonStageExecutable(const PythonStageExecutable &) = delete;
    PythonStageExecutable &operator=(const PythonStageExecutable &) = delete;

    std::unique_ptr<StageInvocationBuilder> invocationBuilder() {
        return std::make_unique<StageInvocationBuilder>(runtime, stage);
    }

    std::vector<ProgramParameterMetadata> parameters() const {
        std::vector<ProgramParameterMetadata> result;
        const size_t count = vernonRuntimeStageExecutableGetParameterCount(stage);
        result.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            VernonProgramParameterView view{};
            if (vernonRuntimeStageExecutableGetParameterByIndex(stage, index, &view) != VERNON_STATUS_OK)
                throw std::runtime_error("cannot query Stage parameter");
            result.push_back(parameterMetadata(view));
        }
        return result;
    }

    VernonRuntimeContext *runtime{};
    VernonStageExecutable *stage{};
    SharedCompileResult retainedResult;
};

#endif
