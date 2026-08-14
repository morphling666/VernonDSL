#include "native_pipeline.h"

std::string nativeStringView(VernonStringView view) {
    return view.data ? std::string(view.data, view.size) : std::string();
}

bool targetAvailable(VernonTarget target) {
    Compiler compiler;
    return vernonCompilerGetTargetCapabilities(compiler.context, target).available != 0;
}

nb::dict targetCapabilities(VernonTarget target) {
    Compiler compiler;
    const VernonTargetCapabilities capabilities = vernonCompilerGetTargetCapabilities(compiler.context, target);
    nb::dict result;
    result["available"] = capabilities.available != 0;
    result["graphics"] = capabilities.supports_graphics != 0;
    result["compute"] = capabilities.supports_compute != 0;
    result["device_storage_atomics"] = capabilities.supports_device_storage_atomics != 0;
    result["f32_device_atomic_add"] = capabilities.supports_f32_device_atomic_add != 0;
    return result;
}

nb::list planValueAbi(const std::string &moduleText, const std::vector<std::string> &logicalDtypes) {
    std::vector<VernonStringView> dtypes;
    dtypes.reserve(logicalDtypes.size());
    for (const std::string &dtype : logicalDtypes)
        dtypes.push_back({dtype.data(), dtype.size()});
    std::unique_ptr<VernonPythonValueAbiPlan, decltype(&vernonCompilerDestroyPythonValueAbiPlan)> plan(
        vernonCompilerPlanPythonValueAbi({moduleText.data(), moduleText.size()}, dtypes.data(), dtypes.size()),
        &vernonCompilerDestroyPythonValueAbiPlan);
    if (!plan)
        throw std::bad_alloc();
    const VernonPythonValueAbiPlanView view = vernonCompilerGetPythonValueAbiPlanView(plan.get());
    if (view.status != VERNON_STATUS_OK) {
        const std::string diagnostics = nativeStringView(view.diagnostics);
        throw std::invalid_argument(diagnostics.empty() ? "native Value ABI planning failed" : diagnostics);
    }

    nb::list nodes;
    for (size_t index = 0; index < view.node_count; ++index) {
        const VernonPythonValueAbiNodeView &node = view.nodes[index];
        std::vector<uint64_t> offsets;
        if (node.field_count)
            offsets.assign(node.field_offsets, node.field_offsets + node.field_count);
        nb::object elementStride = nb::none();
        if (node.has_element_stride)
            elementStride = nb::int_(node.element_stride);
        nodes.append(nb::make_tuple(node.byte_size, node.alignment, std::move(offsets), std::move(elementStride)));
    }
    return nodes;
}

std::unique_ptr<StructuredVjp> buildStructuredVjp(const std::string &moduleText, const std::string &entry,
                                                  const std::vector<std::string> &wrtPaths,
                                                  const std::vector<std::string> &outputPaths,
                                                  const std::string &forwardSymbol, const std::string &backwardSymbol) {
    std::vector<VernonStringView> paths;
    paths.reserve(wrtPaths.size());
    for (const std::string &path : wrtPaths)
        paths.push_back({path.data(), path.size()});
    std::vector<VernonStringView> outputs;
    outputs.reserve(outputPaths.size());
    for (const std::string &path : outputPaths)
        outputs.push_back({path.data(), path.size()});
    auto view = [](const std::string &value) { return VernonStringView{value.data(), value.size()}; };
    VernonPythonStructuredVjp *result = vernonCompilerBuildPythonStructuredVjp(
        view(moduleText), view(entry), paths.data(), paths.size(), outputs.data(), outputs.size(), view(forwardSymbol),
        view(backwardSymbol));
    if (!result)
        throw std::bad_alloc();
    std::unique_ptr<StructuredVjp> transformedResult = std::make_unique<StructuredVjp>(result);
    VernonPythonStructuredVjpView transformed = transformedResult->current();
    if (transformed.status != VERNON_STATUS_OK) {
        std::string diagnostics = nativeStringView(transformed.diagnostics);
        throw std::invalid_argument(diagnostics.empty() ? "structured VJP transform failed" : diagnostics);
    }
    return transformedResult;
}

CpuTargetOptionStrings parseCpuTargetOptions(const nb::dict &targetOptions) {
    for (auto item : targetOptions) {
        const std::string key = nb::cast<std::string>(item.first);
        if (key != "triple" && key != "processor" && key != "features")
            throw std::invalid_argument("unknown option '" + key + "' for CPU target");
    }
    auto readString = [&](const char *name) {
        return targetOptions.contains(name) ? nb::cast<std::string>(targetOptions[name]) : std::string();
    };
    return {readString("triple"), readString("processor"), readString("features")};
}

std::unique_ptr<CompiledProgram> compileProgramResult(Compiler &compiler, const std::string &mlir, VernonTarget target,
                                                      const nb::dict &targetOptions) {
    auto readString = [&](const char *name) {
        return targetOptions.contains(name) ? nb::cast<std::string>(targetOptions[name]) : std::string();
    };
    auto rejectUnknown = [&](std::initializer_list<std::string_view> allowed) {
        for (auto item : targetOptions) {
            const std::string key = nb::cast<std::string>(item.first);
            if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
                throw std::invalid_argument("unknown option '" + key + "' for selected target");
        }
    };
    VernonCompileOptions options{};
    options.struct_size = sizeof(options);
    options.target = target;
    CpuTargetOptionStrings cpu;
    if (target == VERNON_TARGET_CPU) {
        cpu = parseCpuTargetOptions(targetOptions);
        options.as.cpu.triple = VernonStringView{cpu.triple.data(), cpu.triple.size()};
        options.as.cpu.processor = VernonStringView{cpu.processor.data(), cpu.processor.size()};
        options.as.cpu.features = VernonStringView{cpu.features.data(), cpu.features.size()};
    } else if (target == VERNON_TARGET_OPENGL || target == VERNON_TARGET_OPENGL_ES) {
        rejectUnknown({"version"});
        options.as.opengl.version =
            targetOptions.contains("version") ? nb::cast<uint32_t>(targetOptions["version"]) : 0;
    } else if (target == VERNON_TARGET_METAL) {
        rejectUnknown({"platform"});
        const std::string platform = readString("platform");
        options.as.metal.platform = platform.empty() || platform == "macos" ? VERNON_METAL_PLATFORM_MACOS
                                    : platform == "ios"                     ? VERNON_METAL_PLATFORM_IOS
                                                        : static_cast<VernonMetalPlatform>(UINT32_MAX);
    } else if (target == VERNON_TARGET_DIRECTX) {
        rejectUnknown({"shader_model"});
        options.as.directx.shader_model =
            targetOptions.contains("shader_model") ? nb::cast<uint32_t>(targetOptions["shader_model"]) : 0;
    } else {
        rejectUnknown({});
    }
    return std::make_unique<CompiledProgram>(
        vernonCompilerCompileMlirWithOptions(compiler.context, mlir.data(), mlir.size(), &options), target);
}

std::vector<std::unique_ptr<CompiledProgram>> compileCpuProgramResults(const std::vector<std::string> &modules,
                                                                       const nb::dict &targetOptions) {
    const CpuTargetOptionStrings cpu = parseCpuTargetOptions(targetOptions);

    std::vector<std::future<std::unique_ptr<CompiledProgram>>> futures;
    std::vector<std::unique_ptr<CompiledProgram>> results;
    futures.reserve(modules.size());
    results.reserve(modules.size());
    {
        nb::gil_scoped_release release;
        for (const std::string &module : modules) {
            futures.push_back(std::async(std::launch::async, [module, cpu] {
                Compiler compiler;
                VernonCompileOptions options{};
                options.struct_size = sizeof(options);
                options.target = VERNON_TARGET_CPU;
                options.as.cpu.triple = VernonStringView{cpu.triple.data(), cpu.triple.size()};
                options.as.cpu.processor = VernonStringView{cpu.processor.data(), cpu.processor.size()};
                options.as.cpu.features = VernonStringView{cpu.features.data(), cpu.features.size()};
                return std::make_unique<CompiledProgram>(
                    vernonCompilerCompileMlirWithOptions(compiler.context, module.data(), module.size(), &options),
                    VERNON_TARGET_CPU);
            }));
        }
        for (auto &future : futures)
            results.push_back(future.get());
    }
    return results;
}

PipelineParameterMetadata parameterMetadata(const VernonPipelineParameterView &view) {
    PipelineParameterMetadata result;
    result.slot = view.slot;
    result.name = nativeStringView(view.name);
    result.kind = view.kind;
    result.elementByteSize = view.element_layout.byte_size;
    result.elementAlignment = view.element_layout.alignment;
    result.layoutHash = nativeStringView(view.element_layout.layout_hash);
    if (view.element_layout.leaf_count)
        result.elementLeaves.assign(view.element_layout.leaves,
                                    view.element_layout.leaves + view.element_layout.leaf_count);
    result.access = view.access;
    if (view.rank)
        result.shape.assign(view.static_shape, view.static_shape + view.rank);
    return result;
}

PipelineOutputMetadata outputMetadata(const VernonPipelineOutputView &view) {
    PipelineOutputMetadata result;
    result.name = nativeStringView(view.name);
    result.kind = view.kind;
    result.dtype = view.dtype;
    result.access = view.access;
    result.location = view.location;
    if (view.rank)
        result.shape.assign(view.static_shape, view.static_shape + view.rank);
    return result;
}

const char *numpyDtypeName(VernonDataType dtype) {
    if (dtype == VERNON_DATA_BOOL)
        return "bool_";
    if (dtype == VERNON_DATA_U8)
        return "uint8";
    if (dtype == VERNON_DATA_I32)
        return "int32";
    if (dtype == VERNON_DATA_U32)
        return "uint32";
    if (dtype == VERNON_DATA_F16)
        return "float16";
    if (dtype == VERNON_DATA_F32)
        return "float32";
    if (dtype == VERNON_DATA_F64)
        return "float64";
    throw std::invalid_argument("Python autodiff Value dtype is unsupported");
}

size_t autodiffDtypeSize(VernonDataType dtype) {
    if (dtype == VERNON_DATA_BOOL || dtype == VERNON_DATA_U8)
        return 1;
    if (dtype == VERNON_DATA_F16)
        return 2;
    if (dtype == VERNON_DATA_I32 || dtype == VERNON_DATA_U32 || dtype == VERNON_DATA_F32)
        return 4;
    if (dtype == VERNON_DATA_F64)
        return 8;
    throw std::invalid_argument("Python autodiff Value dtype is unsupported");
}

VernonDataType autodiffTangentDtype(VernonDataType primal) {
    return primal == VERNON_DATA_F16 ? VERNON_DATA_F32 : primal;
}

std::string formatShape(const std::vector<uint64_t> &shape) {
    std::string result = "[";
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index)
            result += ", ";
        result += std::to_string(shape[index]);
    }
    return result + "]";
}
