#include "runtime_dispatch.h"

#include "backend_cpu.h"
#include "shape_layout.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <memory>
#include <utility>

namespace vernon::runtime {
namespace {

bool parseAutodiffResourceRole(const nlohmann::json &argument, AutodiffResourceRole &role, std::string &error) {
    const std::string value = argument.value("vernon.autodiff_role", "");
    if (value.empty())
        return true;
    if (const std::optional<program_plan::TapeCarrier> carrier = program_plan::tapeCarrierFromRoleName(value)) {
        switch (*carrier) {
        case program_plan::TapeCarrier::TapeData:
            role = AutodiffResourceRole::Tape;
            break;
        case program_plan::TapeCarrier::ReplaySegment:
            role = AutodiffResourceRole::ReplaySegment;
            break;
        case program_plan::TapeCarrier::ReplayStatus:
            role = AutodiffResourceRole::ReplayStatus;
            break;
        case program_plan::TapeCarrier::LaunchMetadata:
            role = AutodiffResourceRole::LaunchMetadata;
            break;
        }
    } else if (value == "input")
        role = AutodiffResourceRole::Input;
    else if (value == "storage")
        role = AutodiffResourceRole::Storage;
    else if (value == "output")
        role = AutodiffResourceRole::Output;
    else if (value == "cotangent")
        role = AutodiffResourceRole::Cotangent;
    else if (value == "gradient")
        role = AutodiffResourceRole::Gradient;
    else if (value == "primal")
        role = AutodiffResourceRole::Primal;
    else if (value == "retained_primal")
        role = AutodiffResourceRole::RetainedPrimal;
    else {
        error = "compute artifact reflection contains an unknown autodiff resource role";
        return false;
    }
    return true;
}

bool materializeComputeEndpoint(const nlohmann::json &root, const std::string &entry, Variant &variant,
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
        parameter.name =
            argument.value("vernon.source_name", argument.value("name", "argument" + std::to_string(slot)));
        const std::string reflectedKind = argument.value("kind", std::string());
        parameter.kind = reflectedKind == "image" || reflectedKind == "sampler" ? reflectedKind : "tensor";
        parameter.source = "direct";
        if (!parseAutodiffResourceRole(argument, parameter.autodiffRole, error))
            return false;
        parameter.autodiffSource = argument.value("vernon.autodiff_source", "");
        const std::string carrier = argument.value("vernon.autodiff_carrier", "");
        if (!carrier.empty() && carrier != "invocation_linear") {
            error = "compute artifact reflection contains an unknown autodiff carrier";
            return false;
        }
        parameter.invocationCarrier = carrier == "invocation_linear";
        if (parameter.kind == "tensor") {
            if (argument.contains("element_layout")) {
                if (!parsePipelineValueLayout(argument["element_layout"], parameter.elementLayout, error))
                    return false;
            } else if (argument.contains("value_layout")) {
                ValueLayout layout;
                if (!parsePipelineValueLayout(argument["value_layout"], layout, error))
                    return false;
                parameter.valueLayout = std::move(layout);
            } else {
                error = "compute value reflection has no canonical layout";
                return false;
            }
        }
        if (argument.contains("access")) {
            if (!argument["access"].is_string()) {
                error = "compute argument reflection has invalid access metadata";
                return false;
            }
            parameter.access = argument["access"].get<std::string>();
        } else if (reflectedKind == "scalar" || reflectedKind == "tensor_value") {
            parameter.access = "read";
        } else {
            error = "compute resource reflection has no access metadata";
            return false;
        }
        if (!pipelineValueAccess(parameter.access)) {
            error = "compute argument reflection has invalid access metadata";
            return false;
        }
        if (parameter.kind == "image") {
            parameter.dimension = argument.value("dimension", std::string());
            parameter.bindingRole = argument.value("binding_role", std::string());
            parameter.sampleResultClass = argument.value("sample_result_class", std::string());
            parameter.exactStorageFormat = argument.value("exact_storage_format", std::string());
            if (!pipelineTextureDimension(parameter.dimension) ||
                (parameter.bindingRole != "sampled" && parameter.bindingRole != "storage") ||
                (parameter.bindingRole == "storage" && !pipelineTextureFormat(parameter.exactStorageFormat))) {
                error = "compute image reflection has invalid role, dimension, or format metadata";
                return false;
            }
        }
        if (argument.contains("shape") && argument["shape"].is_array())
            parameter.shape = argument["shape"].get<std::vector<uint64_t>>();
        else if (argument.contains("source_shape") && argument["source_shape"].is_array()) {
            std::vector<int64_t> reflected;
            for (const auto &extent : argument["source_shape"]) {
                if (!extent.is_number_integer()) {
                    error = "TensorView source shape contains an invalid extent";
                    return false;
                }
                reflected.push_back(extent.get<int64_t>());
            }
            const std::optional<shape::DeclaredShape> decoded = shape::decodeReflectedShape(reflected);
            if (!decoded) {
                error = "TensorView source shape contains an invalid extent";
                return false;
            }
            parameter.shape = shape::encodeRuntimeContractShape(*decoded);
        }
        ParameterUse use;
        use.stage = "compute";
        use.index = static_cast<uint32_t>(reflectedIndex);
        const std::string &physicalKind = reflection.arguments[reflectedIndex].kind;
        use.interfaceKind = physicalKind == "tensor"                               ? "storage"
                            : physicalKind == "image" || physicalKind == "sampler" ? "resource"
                                                                                   : "value";
        use.dtype = argument.value("dtype", std::string());
        use.shape = parameter.shape;
        use.transport = transport;
        if (argument.contains("value_layout")) {
            ValueLayout layout;
            if (!parsePipelineValueLayout(argument["value_layout"], layout, error))
                return false;
            use.valueLayout = std::move(layout);
        }
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
            if (parsedDescriptor.extentBindings.size() != parsedDescriptor.rank ||
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
                physical->value("profile", std::string()) != physicalProfile) {
                error = "compute value reflection has no physical profile for selected target";
                return false;
            }
            InterfacePlan plan;
            if (!parsePipelineInterfacePlan(*physical, plan, error))
                return false;
            if (plan.root && !plan.root->byteStrides.empty() && plan.root->byteStrides.size() != use.shape.size()) {
                error = "compute value interface plan rank does not match its reflected shape";
                return false;
            }
            if (use.valueLayout && plan.canonicalLayoutHash != use.valueLayout->layoutHash) {
                error = "compute value interface plan hash does not match value_layout";
                return false;
            }
            use.interfacePlan = std::move(plan);
        }
        parameter.uses.push_back(std::move(use));
        variant.parameters.push_back(std::move(parameter));
        ++slot;
    }
    return true;
}

} // namespace

bool buildReflectedComputeVariant(const Stage &stage, VernonRuntimeBackend backend, Variant &variant,
                                  std::string &error) {
    const nlohmann::json reflection = nlohmann::json::parse(stage.reflection, nullptr, false);
    if (reflection.is_discarded()) {
        error = "compute artifact reflection is invalid JSON";
        return false;
    }
    ReflectedEntry entry;
    return materializeComputeEndpoint(reflection, stage.entry, variant, entry, backend, error);
}

bool isDirectPipelineTopology(const Variant &variant) {
    return variant.program.size() == 1 && !variant.compute.empty() &&
           variant.program.find("compute") != variant.program.end();
}

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
    if (!materializeComputeEndpoint(parsed, entry, pipeline->variant, reflection, VERNON_RUNTIME_CPU,
                                    invocationDiagnostic(context)))
        return nullptr;
    pipeline->workgroupSize = {reflection.workgroup[0], reflection.workgroup[1], reflection.workgroup[2]};
    pipeline->dispatchContract = reflection.dispatchContract;
    pipeline->readFootprints = reflection.readFootprints;
    pipeline->writeFootprints = reflection.writeFootprints;
    CpuKernelState kernel;
    ReflectedEntry loadedReflection;
    if (!loadCpuEntry(entryPoint, reflectionData, reflectionSize, entryData, entrySize, kernel, loadedReflection,
                      invocationDiagnostic(context)))
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
    Stage stage;
    Variant variant;
    ReflectedEntry reflection;
    if (!buildDirectComputeStage(context, artifact, artifactSize, reflectionData, reflectionSize, entryData, entrySize,
                                 stage, variant, reflection))
        return nullptr;
    auto pipeline = std::make_unique<VernonLoadedPipeline>();
    pipeline->context = &context;
    pipeline->variant = std::move(variant);
    pipeline->workgroupSize = {reflection.workgroup[0], reflection.workgroup[1], reflection.workgroup[2]};
    pipeline->dispatchContract = reflection.dispatchContract;
    pipeline->readFootprints = reflection.readFootprints;
    pipeline->writeFootprints = reflection.writeFootprints;
    VernonPipelineBundle bundle;
    bundle.context = &context;
    bundle.stages.emplace(stage.entry, std::move(stage));
    if (!resolveBackendPipeline(bundle, pipeline->variant, *pipeline))
        return nullptr;
    ++context.livePipelines;
    return pipeline.release();
}

VernonLoadedPipeline *loadBackendTypedComputePipeline(VernonRuntimeContext &context, Variant variant,
                                                      ReflectedEntry reflection, const void *artifact,
                                                      size_t artifactSize, const std::string &entry,
                                                      VernonCpuEntryPoint cpuEntry,
                                                      const std::vector<NativeResourceSlot> &nativeSlots) {
    auto pipeline = std::make_unique<VernonLoadedPipeline>();
    pipeline->context = &context;
    pipeline->variant = std::move(variant);
    pipeline->workgroupSize = {reflection.workgroup[0], reflection.workgroup[1], reflection.workgroup[2]};
    pipeline->dispatchContract = reflection.dispatchContract;
    pipeline->readFootprints = reflection.readFootprints;
    pipeline->writeFootprints = reflection.writeFootprints;
    if (context.backend == VERNON_RUNTIME_CPU) {
        CpuKernelState kernel;
        kernel.entry = cpuEntry;
        auto state = std::make_unique<CpuPipelineState>();
        if (!prepareCpuComputePipeline(context, std::move(kernel), std::move(reflection), *state))
            return nullptr;
        installRuntimeBackendState(*pipeline, state.release());
        ++context.livePipelines;
        return pipeline.release();
    }
    if (!artifact || !artifactSize)
        return nullptr;
    Stage stage;
    stage.stage = "compute";
    stage.entry = entry;
    stage.reflected = std::move(reflection);
    stage.nativeSlots = nativeSlots;
    stage.dispatchContract = pipeline->dispatchContract;
    stage.workgroup[0] = pipeline->workgroupSize.x;
    stage.workgroup[1] = pipeline->workgroupSize.y;
    stage.workgroup[2] = pipeline->workgroupSize.z;
    stage.readFootprints = pipeline->readFootprints;
    stage.writeFootprints = pipeline->writeFootprints;
    if (context.backend == VERNON_RUNTIME_CUDA || context.backend == VERNON_RUNTIME_METAL ||
        isOpenGLBackend(context.backend))
        stage.source.assign(static_cast<const char *>(artifact), artifactSize);
    else
        stage.binary.assign(static_cast<const uint8_t *>(artifact),
                            static_cast<const uint8_t *>(artifact) + artifactSize);
    VernonPipelineBundle bundle;
    bundle.context = &context;
    bundle.stages.emplace(stage.entry, std::move(stage));
    if (!resolveBackendPipeline(bundle, pipeline->variant, *pipeline))
        return nullptr;
    ++context.livePipelines;
    return pipeline.release();
}

bool buildDirectComputeStage(VernonRuntimeContext &context, const void *artifact, size_t artifactSize,
                             const char *reflectionData, size_t reflectionSize, const char *entryData, size_t entrySize,
                             Stage &stage, Variant &variant, ReflectedEntry &reflection) {
    if (!artifact || !artifactSize || !reflectionData || !reflectionSize || !entryData || !entrySize)
        return false;
    const nlohmann::json parsed =
        nlohmann::json::parse(reflectionData, reflectionData + reflectionSize, nullptr, false);
    if (parsed.is_discarded())
        return false;
    const std::string entry(entryData, entrySize);
    if (!materializeComputeEndpoint(parsed, entry, variant, reflection, context.backend, invocationDiagnostic(context)))
        return false;
    stage.stage = "compute";
    stage.entry = entry;
    stage.reflection.assign(reflectionData, reflectionSize);
    std::copy_n(reflection.workgroup, 3, stage.workgroup);
    stage.dispatchContract = reflection.dispatchContract;
    stage.readFootprints = reflection.readFootprints;
    stage.writeFootprints = reflection.writeFootprints;
    if (context.backend == VERNON_RUNTIME_CUDA || context.backend == VERNON_RUNTIME_METAL ||
        isOpenGLBackend(context.backend))
        stage.source.assign(static_cast<const char *>(artifact), artifactSize);
    else
        stage.binary.assign(static_cast<const uint8_t *>(artifact),
                            static_cast<const uint8_t *>(artifact) + artifactSize);
    return true;
}

} // namespace vernon::runtime
