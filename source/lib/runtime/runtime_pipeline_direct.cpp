#include "runtime_dispatch.h"

#include "backend_cpu.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <memory>
#include <utility>

namespace vernon::runtime {
namespace {

bool buildDirectComputeVariant(const nlohmann::json &root, const std::string &entry, Variant &variant,
                               ReflectedEntry &reflection, VernonRuntimeBackend backend, std::string &error) {
    if (!parseReflection(root, entry, reflection, backend, error))
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
    const std::string transport = backend == VERNON_RUNTIME_CPU    ? "host_value"
                                  : backend == VERNON_RUNTIME_CUDA ? "kernel_parameter"
                                                                   : "storage_buffer";
    const std::string physicalProfile = physicalValueProfileName(backend, transport);
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
        const std::string reflectedKind = argument.value("kind", std::string());
        parameter.kind = reflectedKind == "texture" || reflectedKind == "sampler" ? reflectedKind : "tensor";
        parameter.source = "direct";
        if (parameter.kind == "tensor" &&
            (!argument.contains("element_layout") ||
             !parsePipelineValueLayout(argument["element_layout"], parameter.elementLayout, error)))
            return false;
        parameter.access =
            argument.value("access", argument.value("kind", std::string()) == "tensor" ? "read_write" : "read");
        if (argument.contains("shape") && argument["shape"].is_array())
            parameter.shape = argument["shape"].get<std::vector<uint64_t>>();
        else if (argument.contains("source_shape") && argument["source_shape"].is_array())
            for (const auto &extent : argument["source_shape"]) {
                if (!extent.is_number_integer() || extent.get<int64_t>() < -1) {
                    error = "TensorView source shape contains an invalid extent";
                    return false;
                }
                parameter.shape.push_back(extent.get<int64_t>() < 0 ? 0 : static_cast<uint64_t>(extent.get<int64_t>()));
            }
        ParameterUse use;
        use.stage = "compute";
        use.index = static_cast<uint32_t>(reflectedIndex);
        const std::string &physicalKind = reflection.arguments[reflectedIndex].kind;
        use.interfaceKind = physicalKind == "tensor"                                 ? "storage"
                            : physicalKind == "texture" || physicalKind == "sampler" ? "resource"
                                                                                     : "value";
        use.dtype = argument.value("dtype", std::string());
        use.shape = parameter.shape;
        if (argument.contains("tensor_view_descriptor")) {
            const auto &descriptor = argument["tensor_view_descriptor"];
            TensorViewDescriptorUse parsedDescriptor;
            if (!descriptor.is_object() || !descriptor.contains("rank") || !descriptor["rank"].is_number_unsigned() ||
                !descriptor.contains("offset_binding") || !descriptor["offset_binding"].is_number_unsigned() ||
                !descriptor.contains("extent_bindings") || !descriptor["extent_bindings"].is_array() ||
                !descriptor.contains("stride_bindings") || !descriptor["stride_bindings"].is_array()) {
                error = "TensorView reflection has invalid descriptor metadata";
                return false;
            }
            parsedDescriptor.rank = descriptor["rank"].get<uint32_t>();
            parsedDescriptor.offsetBinding = descriptor["offset_binding"].get<uint32_t>();
            parsedDescriptor.extentBindings = descriptor["extent_bindings"].get<std::vector<uint32_t>>();
            parsedDescriptor.strideBindings = descriptor["stride_bindings"].get<std::vector<uint32_t>>();
            if (!parsedDescriptor.rank || parsedDescriptor.extentBindings.size() != parsedDescriptor.rank ||
                parsedDescriptor.strideBindings.size() != parsedDescriptor.rank ||
                (!use.shape.empty() && use.shape.size() != parsedDescriptor.rank)) {
                error = "TensorView descriptor rank does not match shape";
                return false;
            }
            use.tensorViewDescriptor = std::move(parsedDescriptor);
        }
        use.descriptorSet = argument.value("vernon.set", uint32_t{0});
        use.binding =
            argument.value("vernon.binding", argument.value("binding", static_cast<uint32_t>(reflectedIndex)));
        if (use.interfaceKind == "value") {
            const auto physicalLayouts = argument.find("physical_layouts");
            if (physicalLayouts == argument.end() || !physicalLayouts->is_object()) {
                error = "compute value reflection has no physical layout table";
                return false;
            }
            const auto physical = physicalLayouts->find(physicalProfile);
            if (physical == physicalLayouts->end() || !physical->is_object() ||
                physical->value("profile", std::string()) != physicalProfile || !physical->contains("size") ||
                !(*physical)["size"].is_number_unsigned() || !physical->contains("alignment") ||
                !(*physical)["alignment"].is_number_unsigned() || !physical->contains("byte_strides") ||
                !(*physical)["byte_strides"].is_array()) {
                error = "compute value reflection has no physical profile for selected target";
                return false;
            }
            PhysicalValueLayout layout;
            layout.profile = physicalProfile;
            layout.transport = transport;
            layout.size = (*physical)["size"].get<uint64_t>();
            layout.alignment = (*physical)["alignment"].get<uint64_t>();
            layout.byteStrides = (*physical)["byte_strides"].get<std::vector<uint64_t>>();
            if (physical->contains("element_leaf_offsets")) {
                if (!(*physical)["element_leaf_offsets"].is_array()) {
                    error = "compute value reflection has invalid element leaf offsets";
                    return false;
                }
                layout.elementLeafOffsets = (*physical)["element_leaf_offsets"].get<std::vector<uint64_t>>();
            }
            if (layout.byteStrides.size() != parameter.shape.size()) {
                error = "compute value physical stride rank does not match its logical shape";
                return false;
            }
            use.physicalValueLayout = std::move(layout);
        }
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
    if (!buildDirectComputeVariant(parsed, entry, pipeline->variant, reflection, VERNON_RUNTIME_CPU, context.error))
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
    if (!buildDirectComputeVariant(artifact.reflection, artifact.entry, pipeline->variant, reflected,
                                   VERNON_RUNTIME_CPU, context.error))
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
    if (!buildDirectComputeVariant(parsed, entry, pipeline->variant, reflection, context.backend, context.error))
        return nullptr;
    VernonPipelineBundle bundle;
    bundle.context = &context;
    Stage stage;
    stage.stage = "compute";
    stage.entry = entry;
    stage.reflection.assign(reflectionData, reflectionSize);
    std::copy_n(reflection.workgroup, 3, stage.workgroup);
    if (context.backend == VERNON_RUNTIME_CUDA || context.backend == VERNON_RUNTIME_METAL ||
        isOpenGLBackend(context.backend))
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
