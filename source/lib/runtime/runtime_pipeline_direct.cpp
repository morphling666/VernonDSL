#include "runtime_dispatch.h"

#include "backend_cpu.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <memory>
#include <utility>

namespace vernon::runtime {
namespace {

bool buildDirectComputeVariant(const nlohmann::json &root, const std::string &entry, Variant &variant,
                               ReflectedEntry &reflection, std::string &error) {
    if (!parseReflection(root, entry, reflection, error))
        return false;
    const nlohmann::json *selected = nullptr;
    if (root.contains("entries") && root["entries"].is_array())
        for (const auto &candidate : root["entries"])
            if (candidate.is_object() && candidate.value("name", std::string()) == entry) {
                selected = &candidate;
                break;
            }
    if (!selected || !selected->contains("arguments") || !(*selected)["arguments"].is_array()) {
        error = "compute reflection does not contain the selected entry arguments";
        return false;
    }
    variant.compute = entry;
    variant.program.emplace("compute", entry);
    uint32_t slot = 0;
    const auto &arguments = (*selected)["arguments"];
    for (size_t reflectedIndex = 0; reflectedIndex < arguments.size(); ++reflectedIndex) {
        const auto &argument = arguments[reflectedIndex];
        if (!argument.is_object() || argument.value("kind", std::string()) == "builtin")
            continue;
        Parameter parameter;
        parameter.slot = slot;
        parameter.name = argument.value("name", "argument" + std::to_string(slot));
        parameter.kind = "tensor";
        parameter.source = "direct";
        parameter.dtype = argument.value("dtype", std::string());
        const bool opaqueValue = !pipelineDataType(parameter.dtype).has_value();
        if (opaqueValue)
            parameter.dtype = "u8";
        parameter.access =
            argument.value("access", argument.value("kind", std::string()) == "tensor" ? "read_write" : "read");
        if (argument.contains("shape") && argument["shape"].is_array())
            parameter.shape = argument["shape"].get<std::vector<uint64_t>>();
        else if (opaqueValue && argument.value("kind", std::string()) != "tensor")
            parameter.shape = {argument.value("cpu_size", uint64_t{0})};
        ParameterUse use;
        use.stage = "compute";
        use.index = static_cast<uint32_t>(reflectedIndex);
        use.interfaceKind = argument.value("kind", std::string()) == "tensor" ? "storage" : "value";
        use.dtype = parameter.dtype;
        use.shape = parameter.shape;
        use.descriptorSet = argument.value("vernon.set", uint32_t{0});
        use.binding =
            argument.value("vernon.binding", argument.value("binding", static_cast<uint32_t>(reflectedIndex)));
        parameter.uses.push_back(std::move(use));
        variant.parameters.push_back(std::move(parameter));
        ++slot;
    }
    return true;
}

} // namespace

VernonStatus registerBackendStaticCpuEntry(VernonStringView symbol, VernonCpuEntryPoint entryPoint) {
    return registerStaticCpuEntry(symbol, entryPoint);
}

VernonLoadedPipeline *loadBackendCpuEntryPipeline(VernonRuntimeContext &context, VernonCpuEntryPoint entryPoint,
                                                  const char *reflectionData, size_t reflectionSize,
                                                  const char *entryData, size_t entrySize) {
    const nlohmann::json parsed =
        nlohmann::json::parse(reflectionData, reflectionData + reflectionSize, nullptr, false);
    if (parsed.is_discarded())
        return nullptr;
    auto pipeline = std::make_unique<VernonLoadedPipeline>();
    pipeline->context = &context;
    ReflectedEntry reflection;
    const std::string entry(entryData, entrySize);
    if (!buildDirectComputeVariant(parsed, entry, pipeline->variant, reflection, context.error))
        return nullptr;
    CpuKernelState kernel;
    ReflectedEntry loadedReflection;
    if (!loadCpuEntry(entryPoint, reflectionData, reflectionSize, entryData, entrySize, kernel, loadedReflection,
                      context.error))
        return nullptr;
    auto state = std::make_unique<CpuPipelineState>();
    if (!prepareCpuComputePipeline(context, std::move(kernel), std::move(loadedReflection), *state))
        return nullptr;
    installRuntimeBackendState(*pipeline, state.release());
    ++context.livePipelines;
    return pipeline.release();
}

VernonLoadedPipeline *loadBackendCpuNativePipeline(VernonRuntimeContext &context, const CpuNativeArtifact &artifact) {
    auto pipeline = std::make_unique<VernonLoadedPipeline>();
    pipeline->context = &context;
    ReflectedEntry reflected;
    if (!buildDirectComputeVariant(artifact.reflection, artifact.entry, pipeline->variant, reflected, context.error))
        return nullptr;
    CpuKernelState kernel;
    ReflectedEntry loadedReflection;
    if (!loadCpuNativeArtifact(artifact, kernel, loadedReflection, context.error))
        return nullptr;
    auto state = std::make_unique<CpuPipelineState>();
    if (!prepareCpuComputePipeline(context, std::move(kernel), std::move(loadedReflection), *state))
        return nullptr;
    installRuntimeBackendState(*pipeline, state.release());
    ++context.livePipelines;
    return pipeline.release();
}

VernonLoadedPipeline *loadBackendArtifactPipeline(VernonRuntimeContext &context, const void *artifact,
                                                  size_t artifactSize, const char *reflectionData,
                                                  size_t reflectionSize, const char *entryData, size_t entrySize) {
    if (!artifact || !artifactSize || !reflectionData || !reflectionSize || !entryData || !entrySize)
        return nullptr;
    const nlohmann::json parsed =
        nlohmann::json::parse(reflectionData, reflectionData + reflectionSize, nullptr, false);
    if (parsed.is_discarded())
        return nullptr;
    auto pipeline = std::make_unique<VernonLoadedPipeline>();
    pipeline->context = &context;
    ReflectedEntry reflection;
    const std::string entry(entryData, entrySize);
    if (!buildDirectComputeVariant(parsed, entry, pipeline->variant, reflection, context.error))
        return nullptr;
    VernonPipelineBundle bundle;
    bundle.context = &context;
    Stage stage;
    stage.stage = "compute";
    stage.entry = entry;
    stage.reflection.assign(reflectionData, reflectionSize);
    std::copy_n(reflection.workgroup, 3, stage.workgroup);
    if (context.backend == VERNON_RUNTIME_CUDA || isOpenGLBackend(context.backend))
        stage.source.assign(static_cast<const char *>(artifact), artifactSize);
    else
        stage.binary.assign(static_cast<const uint8_t *>(artifact),
                            static_cast<const uint8_t *>(artifact) + artifactSize);
    bundle.stages.emplace(entry, std::move(stage));
    if (!resolveBackendPipeline(bundle, pipeline->variant, *pipeline))
        return nullptr;
    ++context.livePipelines;
    return pipeline.release();
}

} // namespace vernon::runtime
