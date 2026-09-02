#include "runtime_pipeline_backend.h"

#if defined(VERNON_HAS_METAL_RUNTIME)
#include "VernonRuntimeRHIAdapter.h"
#include "backend_metal.h"
#include "pipeline_metadata.h"
#include "rhi/rhi_internal.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#endif

namespace vernon::runtime {

#if defined(VERNON_HAS_METAL_RUNTIME)
namespace {

VernonStatus fail(VernonRuntimeContext &context, std::string error,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    invocationDiagnostic(context) = std::move(error);
    return status;
}

struct MetalResourceLocation {
    uint32_t argumentBufferIndex{UINT32_MAX};
    uint32_t memberId{UINT32_MAX};
    uint32_t directBufferIndex{UINT32_MAX};
    uint32_t count{};
};

bool parseMetalResourceSlots(const nlohmann::json &reflection, std::vector<NativeResourceSlot> &slots,
                             std::string &error) {
    const auto jsonSlots = reflection.find("metal_resource_slots");
    if (jsonSlots == reflection.end() || !jsonSlots->is_array()) {
        error = "Metal reflection has no argument-buffer resource sidecar";
        return false;
    }
    for (const auto &slot : *jsonSlots) {
        if (!slot.is_object())
            continue;
        if (!slot.contains("argument_buffer_index") || !slot.contains("member_id") ||
            !slot.contains("direct_buffer_index")) {
            error = "Metal reflection uses the obsolete direct-resource slot contract";
            return false;
        }
        NativeResourceSlot parsed;
        parsed.entry = slot.value("entry_point", std::string());
        parsed.stage = slot.value("stage", std::string());
        parsed.kind = slot.value("kind", std::string());
        parsed.name = slot.value("name", std::string());
        parsed.set = slot.value("set", UINT32_MAX);
        parsed.binding = slot.value("binding", UINT32_MAX);
        parsed.argumentBufferIndex = slot.value("argument_buffer_index", UINT32_MAX);
        parsed.memberId = slot.value("member_id", UINT32_MAX);
        parsed.directBufferIndex = slot.value("direct_buffer_index", UINT32_MAX);
        parsed.count = slot.value("count", 0u);
        slots.push_back(std::move(parsed));
    }
    return true;
}

const std::vector<NativeResourceSlot> *metalSlotsForStage(const Stage &stage, const nlohmann::json *reflection,
                                                          std::vector<NativeResourceSlot> &parsed, std::string &error) {
    if (!stage.nativeSlots.empty() || stage.reflection.empty())
        return &stage.nativeSlots;
    if (!reflection || !parseMetalResourceSlots(*reflection, parsed, error))
        return nullptr;
    return &parsed;
}

bool resolveMetalResourceLocation(const std::vector<NativeResourceSlot> &slots, const std::string &entry,
                                  const char *stage, const char *kind, uint32_t set, uint32_t binding,
                                  MetalResourceLocation &output, std::string &error,
                                  const std::string *name = nullptr) {
    bool found = false;
    for (const NativeResourceSlot &slot : slots) {
        if (slot.entry != entry || slot.stage != stage || slot.kind != kind || (name && slot.name != *name) ||
            (set != UINT32_MAX && slot.set != set) || (binding != UINT32_MAX && slot.binding != binding))
            continue;
        MetalResourceLocation location;
        location.argumentBufferIndex = slot.argumentBufferIndex;
        location.memberId = slot.memberId;
        location.directBufferIndex = slot.directBufferIndex;
        location.count = slot.count;
        const bool descriptor = set != UINT32_MAX;
        const bool validDescriptor = location.argumentBufferIndex < 8 && location.memberId != UINT32_MAX &&
                                     location.directBufferIndex == UINT32_MAX;
        const bool validDirect = location.argumentBufferIndex == UINT32_MAX && location.memberId == UINT32_MAX &&
                                 location.directBufferIndex < 31;
        if (found || !location.count || (descriptor ? !validDescriptor : !validDirect)) {
            error = "Metal reflection contains an ambiguous or invalid argument-buffer resource";
            return false;
        }
        output = location;
        found = true;
    }
    if (!found)
        error = "Metal reflection has no compiled " + std::string(kind) + " location for " + stage + " binding (" +
                std::to_string(set) + ", " + std::to_string(binding) + ")";
    return found;
}

struct MetalArgumentBufferUsage {
    uint64_t buffers{};
    uint64_t textures{};
    uint64_t samplers{};
    bool writableTexture{};
};

bool collectMetalArgumentBufferUsage(const std::vector<NativeResourceSlot> &slots, const std::string &entry,
                                     const char *stage, MetalArgumentBufferUsage &usage, std::string &error) {
    for (const NativeResourceSlot &slot : slots) {
        if (slot.entry != entry || slot.stage != stage)
            continue;
        if (!slot.count) {
            error = "Metal reflection contains a zero-sized resource binding";
            return false;
        }
        if (slot.kind == "uniform_buffer" || slot.kind == "storage_buffer")
            usage.buffers += slot.count;
        else if (slot.kind == "sampled_image" || slot.kind == "storage_image") {
            usage.textures += slot.count;
            usage.writableTexture |= slot.kind == "storage_image";
        } else if (slot.kind == "sampler")
            usage.samplers += slot.count;
    }
    return true;
}

bool validateMetalArgumentBufferUsage(const MetalArgumentBufferUsage &usage, uint32_t deviceTier,
                                      bool encodingSupported, std::string &error) {
    if ((usage.buffers || usage.textures || usage.samplers) && !encodingSupported) {
        error = "Metal argument-buffer encoding is unavailable on this device";
        return false;
    }
    if (deviceTier < 1 && (usage.buffers > 31 || usage.textures > 31 || usage.samplers > 16 || usage.writableTexture)) {
        error = "Metal pipeline requires Argument Buffers Tier 2 (resources: " + std::to_string(usage.buffers) +
                " buffers, " + std::to_string(usage.textures) + " textures, " + std::to_string(usage.samplers) +
                " samplers" + (usage.writableTexture ? ", writable texture" : "") + ")";
        return false;
    }
    return true;
}

} // namespace
#endif

bool validateMetalArgumentBufferLimitsForTesting(uint64_t buffers, uint64_t textures, uint64_t samplers,
                                                 bool writableTexture, uint32_t deviceTier) {
#if defined(VERNON_HAS_METAL_RUNTIME)
    const MetalArgumentBufferUsage usage{buffers, textures, samplers, writableTexture};
    std::string error;
    return validateMetalArgumentBufferUsage(usage, deviceTier, true, error);
#else
    (void)buffers;
    (void)textures;
    (void)samplers;
    (void)writableTexture;
    (void)deviceTier;
    return false;
#endif
}

bool resolveMetalPipeline(VernonPipelineBundle &bundle, const Variant &variant, VernonLoadedPipeline &pipeline) {
#if defined(VERNON_HAS_METAL_RUNTIME)
    if (variant.compute.empty()) {
        const Stage &vertex = bundle.stages.at(variant.vertex);
        const Stage &fragment = bundle.stages.at(variant.fragment);
        const nlohmann::json vertexReflection = nlohmann::json::parse(vertex.reflection, nullptr, false);
        const nlohmann::json fragmentReflection = nlohmann::json::parse(fragment.reflection, nullptr, false);
        std::vector<NativeResourceSlot> vertexSlots;
        std::vector<NativeResourceSlot> fragmentSlots;
        const std::vector<NativeResourceSlot> *vertexNative =
            metalSlotsForStage(vertex, vertexReflection.is_discarded() ? nullptr : &vertexReflection, vertexSlots,
                               invocationDiagnostic(*bundle.context));
        const std::vector<NativeResourceSlot> *fragmentNative =
            metalSlotsForStage(fragment, fragmentReflection.is_discarded() ? nullptr : &fragmentReflection,
                               fragmentSlots, invocationDiagnostic(*bundle.context));
        if (!vertexNative || !fragmentNative) {
            if (invocationDiagnostic(*bundle.context).empty())
                invocationDiagnostic(*bundle.context) = "Metal graphics reflection is invalid";
            return false;
        }
        MetalArgumentBufferUsage argumentBufferUsage;
        if (!collectMetalArgumentBufferUsage(*vertexNative, vertex.entry, "vertex", argumentBufferUsage,
                                             invocationDiagnostic(*bundle.context)) ||
            !collectMetalArgumentBufferUsage(*fragmentNative, fragment.entry, "fragment", argumentBufferUsage,
                                             invocationDiagnostic(*bundle.context)) ||
            !validateMetalArgumentBufferUsage(argumentBufferUsage, metalState(*bundle.context).argumentBuffersTier,
                                              metalState(*bundle.context).argumentBufferEncodingSupported,
                                              invocationDiagnostic(*bundle.context)))
            return false;
        struct Candidate {
            VernonRuntimeProviderBindingLayoutEntry layout{};
            std::vector<VernonRuntimeProviderVertexAttribute> attributes;
            MetalPipelineState::GraphicsBinding binding;
        };
        std::vector<Candidate> candidates;
        uint32_t nextProviderSlot = 0;
        uint32_t vertexBinding = 0;
        const auto addUse = [&](const Parameter &parameter, const ParameterUse &use, bool internal) {
            if (use.stage != "vertex" && use.stage != "fragment")
                return false;
            Candidate candidate;
            if (nextProviderSlot == UINT32_MAX)
                return false;
            candidate.layout.slot = nextProviderSlot++;
            candidate.layout.argument_index = use.index;
            candidate.layout.stage_mask =
                use.stage == "vertex" ? VERNON_RUNTIME_PROVIDER_STAGE_VERTEX : VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT;
            candidate.layout.array_count = 1;
            candidate.binding.externalSlot = parameter.slot;
            candidate.binding.descriptorSet = use.descriptorSet;
            candidate.binding.descriptorBinding = use.binding;
            const Stage &nativeStage = use.stage == "vertex" ? vertex : fragment;
            const std::vector<NativeResourceSlot> &nativeSlots =
                use.stage == "vertex" ? *vertexNative : *fragmentNative;
            auto resolveDescriptor = [&](const char *kind) {
                MetalResourceLocation location;
                if (!resolveMetalResourceLocation(nativeSlots, nativeStage.entry, use.stage.c_str(), kind,
                                                  use.descriptorSet, use.binding, location,
                                                  invocationDiagnostic(*bundle.context)))
                    return false;
                candidate.layout.set = location.argumentBufferIndex;
                candidate.layout.binding = location.memberId;
                candidate.layout.array_count = location.count;
                return true;
            };
            if (parameter.kind == "sampler") {
                if (use.sampledImageBindings.empty())
                    return false;
                for (size_t index = 0; index < use.sampledImageBindings.size(); ++index) {
                    const SampledImageBinding &binding = use.sampledImageBindings[index];
                    Candidate sampler = candidate;
                    if (index != 0) {
                        if (nextProviderSlot == UINT32_MAX)
                            return false;
                        sampler.layout.slot = nextProviderSlot++;
                    }
                    sampler.layout.kind = VERNON_RUNTIME_PROVIDER_SAMPLER;
                    sampler.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                    sampler.layout.set = binding.descriptorSet;
                    sampler.layout.binding = binding.binding;
                    sampler.binding.descriptorSet = binding.descriptorSet;
                    sampler.binding.descriptorBinding = binding.binding;
                    MetalResourceLocation location;
                    if (!resolveMetalResourceLocation(nativeSlots, nativeStage.entry, use.stage.c_str(), "sampler",
                                                      binding.descriptorSet, binding.binding, location,
                                                      invocationDiagnostic(*bundle.context)))
                        return false;
                    sampler.layout.set = location.argumentBufferIndex;
                    sampler.layout.binding = location.memberId;
                    sampler.layout.array_count = location.count;
                    sampler.binding.source = internal ? MetalPipelineState::GraphicsBinding::IMPLICIT_SAMPLER
                                                      : MetalPipelineState::GraphicsBinding::EXTERNAL_SAMPLER;
                    candidates.push_back(std::move(sampler));
                }
                return true;
            }
            if (parameter.kind == "tensor" && use.interfaceKind == "uniform" && use.interfacePlan &&
                use.interfacePlan->root) {
                const auto &shape = use.shape.empty() ? parameter.shape : use.shape;
                const std::optional<VernonDataType> dtype = pipelineDataType(use.dtype);
                uint64_t count = 1;
                for (uint64_t dimension : shape) {
                    if (!dimension || count > UINT32_MAX / dimension)
                        return false;
                    count *= dimension;
                }
                if (!dtype || !use.interfacePlan->root->size || use.interfacePlan->root->size > UINT32_MAX ||
                    !use.interfacePlan->root->alignment || use.interfacePlan->root->alignment > UINT32_MAX)
                    return false;
                candidate.layout.kind = use.transport == "storage_buffer"   ? VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER
                                        : use.transport == "uniform_buffer" ? VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER
                                                                            : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                candidate.layout.element_size = static_cast<uint32_t>(use.interfacePlan->root->size);
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM;
                candidate.layout.element_count = static_cast<uint32_t>(count);
                candidate.layout.vector_count = shape.size() == 2 ? static_cast<uint32_t>(shape[0]) : 1;
                candidate.layout.element_alignment = static_cast<uint32_t>(use.interfacePlan->root->alignment);
                candidate.layout.set = use.descriptorSet;
                candidate.layout.binding = use.binding;
                candidate.binding.source = internal ? MetalPipelineState::GraphicsBinding::RESOLUTION
                                                    : MetalPipelineState::GraphicsBinding::EXTERNAL_UNIFORM;
                const ValueLayout &canonical = parameter.valueLayout ? *parameter.valueLayout : parameter.elementLayout;
                std::optional<TensorCopyPlan> packing =
                    parameter.valueLayout
                        ? compileWholeValueCopyPlan(pipelineValueLayout(canonical), *use.interfacePlan->root)
                        : compileElementStreamCopyPlan(pipelineValueLayout(canonical), shape, *use.interfacePlan->root);
                if (!packing || packing->elementSize != canonical.byteSize)
                    return false;
                candidate.binding.packing = std::move(*packing);
                candidate.binding.storage.resize(candidate.layout.element_size);
                if (candidate.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER) {
                    if (!resolveDescriptor("uniform_buffer"))
                        return false;
                } else if (candidate.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
                    if (!resolveDescriptor("storage_buffer"))
                        return false;
                } else {
                    MetalResourceLocation location;
                    const std::string inlineName = !use.uniformName.empty() ? use.uniformName : parameter.name;
                    if (!resolveMetalResourceLocation(nativeSlots, nativeStage.entry, use.stage.c_str(),
                                                      "inline_constant", UINT32_MAX, UINT32_MAX, location,
                                                      invocationDiagnostic(*bundle.context), &inlineName))
                        return false;
                    candidate.layout.set = UINT32_MAX;
                    candidate.layout.binding = location.directBufferIndex;
                }
            } else if (parameter.kind == "tensor" && use.interfaceKind == "input" && use.stage == "vertex" &&
                       use.location != UINT32_MAX && !use.attributeLeaves.empty() && vertexBinding < 15) {
                candidate.layout.kind = VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER;
                candidate.layout.element_size = parameter.elementLayout.byteSize;
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_VERTEX_INPUT;
                candidate.layout.binding = 16 + vertexBinding++;
                candidate.layout.divisor = use.divisor;
                for (const AttributeLeaf &leaf : use.attributeLeaves) {
                    const std::optional<VernonDataType> attributeType = pipelineDataType(leaf.dtype);
                    if (!attributeType)
                        return false;
                    candidate.attributes.push_back({candidate.layout.binding, use.location + leaf.locationOffset,
                                                    static_cast<uint32_t>(*attributeType), leaf.componentCount,
                                                    leaf.byteOffset});
                }
                candidate.binding.source = MetalPipelineState::GraphicsBinding::EXTERNAL_VERTEX;
            } else if (parameter.kind == "image" && use.interfaceKind == "resource") {
                const bool storage = parameter.bindingRole == "storage";
                candidate.layout.kind =
                    storage ? VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE : VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE;
                if (!configureImageBindingLayout(parameter, candidate.layout))
                    return false;
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                candidate.layout.set = use.descriptorSet;
                candidate.layout.binding = use.binding;
                candidate.binding.source = MetalPipelineState::GraphicsBinding::EXTERNAL_TEXTURE;
                if (!resolveDescriptor(storage ? "storage_image" : "sampled_image"))
                    return false;
            } else {
                return false;
            }
            candidates.push_back(std::move(candidate));
            return true;
        };
        bool supported = true;
        for (const Parameter &parameter : variant.parameters) {
            for (const ParameterUse &use : parameter.uses)
                if (!addUse(parameter, use, false)) {
                    supported = false;
                    break;
                }
            if (!supported)
                break;
        }
        for (const Parameter &parameter : variant.internalParameters)
            for (const ParameterUse &use : parameter.uses)
                if (!supported || !addUse(parameter, use, true)) {
                    supported = false;
                    break;
                }
        if (!supported) {
            if (invocationDiagnostic(*bundle.context).empty())
                invocationDiagnostic(*bundle.context) =
                    "Metal RuntimeCore graphics path does not support this parameter layout";
            return false;
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &left, const auto &right) { return left.layout.slot < right.layout.slot; });
        auto state = std::make_unique<MetalPipelineState>();
        for (auto &candidate : candidates) {
            state->rhiGraphicsLayout.push_back(candidate.layout);
            state->rhiGraphicsVertexAttributes.insert(state->rhiGraphicsVertexAttributes.end(),
                                                      candidate.attributes.begin(), candidate.attributes.end());
            state->rhiGraphicsBindingPlan.push_back(std::move(candidate.binding));
        }
        state->rhiGraphicsValues.resize(candidates.size());
        const VernonRuntimeProviderShaderDescriptor shaders[2]{{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                                VERNON_RUNTIME_PROVIDER_STAGE_VERTEX,
                                                                {"msl", 3},
                                                                vertex.source.data(),
                                                                vertex.source.size(),
                                                                {vertex.entry.data(), vertex.entry.size()},
                                                                {},
                                                                {0, 0, 0, 0}},
                                                               {sizeof(VernonRuntimeProviderShaderDescriptor),
                                                                VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT,
                                                                {"msl", 3},
                                                                fragment.source.data(),
                                                                fragment.source.size(),
                                                                {fragment.entry.data(), fragment.entry.size()},
                                                                {},
                                                                {0, 0, 0, 0}}};
        VernonRuntimeCorePipelineDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.kind = VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE;
        descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_GRAPHICS;
        descriptor.shaders = shaders;
        descriptor.shader_count = 2;
        descriptor.bindings = state->rhiGraphicsLayout.data();
        descriptor.binding_count = state->rhiGraphicsLayout.size();
        descriptor.vertex_attributes = state->rhiGraphicsVertexAttributes.data();
        descriptor.vertex_attribute_count = state->rhiGraphicsVertexAttributes.size();
        descriptor.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
        descriptor.sample_count = 1;
        const VernonStatus status =
            vernonRuntimeCorePreparePipeline(vernonRuntimeRhiAdapterGetProvider(metalState(*bundle.context).adapter),
                                             &descriptor, &state->rhiGraphicsPipeline);
        if (status != VERNON_STATUS_OK) {
            const VernonStringView providerError =
                vernonRuntimeRhiAdapterGetLastError(metalState(*bundle.context).adapter);
            invocationDiagnostic(*bundle.context) = providerError.data
                                                        ? std::string(providerError.data, providerError.size)
                                                        : "failed to prepare Metal provider graphics pipeline";
            return false;
        }
        installRuntimeBackendState(pipeline, state.release());
        return true;
    }
    const Stage &stage = bundle.stages.at(variant.compute);
    ReflectedEntry reflection;
    if (!resolveStageReflection(stage, VERNON_RUNTIME_METAL, reflection, invocationDiagnostic(*bundle.context)))
        return false;
    const nlohmann::json parsed = nlohmann::json::parse(stage.reflection, nullptr, false);
    std::vector<NativeResourceSlot> parsedSlots;
    const std::vector<NativeResourceSlot> *nativeSlots = metalSlotsForStage(
        stage, parsed.is_discarded() ? nullptr : &parsed, parsedSlots, invocationDiagnostic(*bundle.context));
    if (!nativeSlots)
        return false;
    MetalArgumentBufferUsage argumentBufferUsage;
    if (!collectMetalArgumentBufferUsage(*nativeSlots, stage.entry, "compute", argumentBufferUsage,
                                         invocationDiagnostic(*bundle.context)) ||
        !validateMetalArgumentBufferUsage(argumentBufferUsage, metalState(*bundle.context).argumentBuffersTier,
                                          metalState(*bundle.context).argumentBufferEncodingSupported,
                                          invocationDiagnostic(*bundle.context)))
        return false;

    struct Candidate {
        VernonRuntimeProviderBindingLayoutEntry layout{};
        uint64_t resourceOffset{};
        ComputeBindingSource source;
    };
    std::vector<Candidate> candidates;
    uint32_t internalSlot = 0;
    std::vector<uint32_t> argumentBindings(reflection.arguments.size());
    uint32_t flattenedBinding = 0;
    for (size_t index = 0; index < reflection.arguments.size(); ++index) {
        argumentBindings[index] = flattenedBinding;
        if (reflection.arguments[index].kind != "builtin")
            flattenedBinding +=
                static_cast<uint32_t>(std::max(reflection.arguments[index].storageLeaves.size(), size_t{1}));
    }
    for (const Parameter &parameter : variant.parameters)
        internalSlot = std::max(internalSlot, parameter.slot);
    for (const Parameter &parameter : variant.parameters) {
        for (const ParameterUse &use : parameter.uses) {
            if (use.stage != "compute" && use.stage != variant.compute)
                continue;
            if (use.index >= reflection.arguments.size()) {
                invocationDiagnostic(*bundle.context) = "Metal parameter use exceeds reflected argument table";
                return false;
            }
            const ReflectedArgument &argument = reflection.arguments[use.index];
            const size_t leafCount = std::max(argument.storageLeaves.size(), size_t{1});
            for (size_t leafIndex = 0; leafIndex < leafCount; ++leafIndex) {
                Candidate candidate;
                candidate.layout.slot = leafIndex == 0 ? parameter.slot : ++internalSlot;
                candidate.layout.set = argument.descriptorSet;
                candidate.layout.binding = argument.storageLeaves.empty()
                                               ? argumentBindings[use.index] + static_cast<uint32_t>(leafIndex)
                                               : argument.storageLeaves[leafIndex].binding;
                candidate.layout.kind = parameter.kind == "image"   ? (parameter.bindingRole == "sampled"
                                                                           ? VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE
                                                                           : VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE)
                                        : argument.kind == "tensor" ? VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER
                                                                    : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                candidate.layout.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
                candidate.layout.access = parameter.access == "read" ? 1u : parameter.access == "write" ? 2u : 3u;
                if (parameter.kind == "image" && !configureImageBindingLayout(parameter, candidate.layout))
                    return false;
                candidate.layout.array_count = 1;
                candidate.layout.argument_index = use.index;
                candidate.layout.element_size = static_cast<uint32_t>(
                    argument.storageLeaves.empty() ? (parameter.kind == "image"   ? 1
                                                      : argument.kind == "tensor" ? argument.tensorElementSize
                                                                                  : argument.physical.size)
                                                   : argument.storageLeaves[leafIndex].elementSize);
                candidate.resourceOffset = 0;
                candidate.source = {ComputeBindingSourceKind::Argument, use.index, 0};
                MetalResourceLocation location;
                const char *resourceKind =
                    parameter.kind == "image" ? (parameter.bindingRole == "sampled" ? "sampled_image" : "storage_image")
                                              : "storage_buffer";
                if (candidate.layout.binding == UINT32_MAX || candidate.layout.element_size == 0 ||
                    !resolveMetalResourceLocation(*nativeSlots, stage.entry, "compute", resourceKind,
                                                  candidate.layout.set, candidate.layout.binding, location,
                                                  invocationDiagnostic(*bundle.context)))
                    return false;
                candidate.layout.set = location.argumentBufferIndex;
                candidate.layout.binding = location.memberId;
                candidate.layout.array_count = location.count;
                candidates.push_back(candidate);
            }
            if (use.tensorViewDescriptor) {
                const auto addDescriptor = [&](ComputeBindingSourceKind kind, uint32_t dimension, uint32_t binding) {
                    Candidate candidate;
                    candidate.layout.slot = ++internalSlot;
                    candidate.layout.set = argument.descriptorSet;
                    candidate.layout.binding = binding;
                    candidate.layout.kind = VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                    candidate.layout.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
                    candidate.layout.access = 1;
                    candidate.layout.array_count = 1;
                    candidate.layout.argument_index = use.index;
                    candidate.layout.element_size = 4;
                    candidate.source = {kind, use.index, dimension};
                    MetalResourceLocation location;
                    if (!resolveMetalResourceLocation(*nativeSlots, stage.entry, "compute", "storage_buffer",
                                                      candidate.layout.set, binding, location,
                                                      invocationDiagnostic(*bundle.context)))
                        return false;
                    candidate.layout.set = location.argumentBufferIndex;
                    candidate.layout.binding = location.memberId;
                    candidate.layout.array_count = location.count;
                    candidates.push_back(candidate);
                    return true;
                };
                if (!addDescriptor(ComputeBindingSourceKind::TensorOffset, 0, use.tensorViewDescriptor->offsetBinding))
                    return false;
                for (uint32_t dimension = 0; dimension < use.tensorViewDescriptor->rank; ++dimension)
                    if (!addDescriptor(ComputeBindingSourceKind::TensorExtent, dimension,
                                       use.tensorViewDescriptor->extentBindings[dimension]))
                        return false;
                for (uint32_t dimension = 0; dimension < use.tensorViewDescriptor->rank; ++dimension)
                    if (!addDescriptor(ComputeBindingSourceKind::TensorStride, dimension,
                                       use.tensorViewDescriptor->strideBindings[dimension]))
                        return false;
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &left, const auto &right) { return left.layout.slot < right.layout.slot; });
    auto state = std::make_unique<MetalPipelineState>();
    for (const Candidate &candidate : candidates) {
        state->rhiComputeLayout.push_back(candidate.layout);
        state->rhiComputeResourceOffsets.push_back(candidate.resourceOffset);
        state->rhiComputeBindingSources.push_back(candidate.source);
    }
    state->rhiComputeValues.resize(candidates.size());
    state->rhiComputeDescriptorValues.resize(candidates.size());
    std::copy_n(stage.workgroup, 3, state->rhiComputeWorkgroup);
    const VernonRuntimeProviderShaderDescriptor shader{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                       VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE,
                                                       {"msl", 3},
                                                       stage.source.data(),
                                                       stage.source.size(),
                                                       {stage.entry.data(), stage.entry.size()},
                                                       {},
                                                       {0, 0, 0, 0}};
    VernonRuntimeCorePipelineDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.kind = VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
    descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_COMPUTE;
    descriptor.shaders = &shader;
    descriptor.shader_count = 1;
    descriptor.bindings = state->rhiComputeLayout.data();
    descriptor.binding_count = state->rhiComputeLayout.size();
    std::copy_n(state->rhiComputeWorkgroup, 3, descriptor.workgroup_size);
    const VernonStatus status =
        vernonRuntimeCorePreparePipeline(vernonRuntimeRhiAdapterGetProvider(metalState(*bundle.context).adapter),
                                         &descriptor, &state->rhiComputePipeline);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError = vernonRuntimeRhiAdapterGetLastError(metalState(*bundle.context).adapter);
        invocationDiagnostic(*bundle.context) = providerError.data
                                                    ? std::string(providerError.data, providerError.size)
                                                    : "failed to prepare Metal provider compute pipeline";
        return false;
    }
    installRuntimeBackendState(pipeline, state.release());
    return true;
#else
    (void)bundle;
    (void)variant;
    (void)pipeline;
    return false;
#endif
}

void destroyMetalPipeline(VernonLoadedPipeline &pipeline) {
#if defined(VERNON_HAS_METAL_RUNTIME)
    MetalPipelineState &state = runtimeBackendState<MetalPipelineState>(pipeline);
    vernonRuntimeCoreBindingsDestroy(state.rhiComputeBindings);
    vernonRuntimeCorePipelineDestroy(state.rhiComputePipeline);
    vernonRuntimeCoreBindingsDestroy(state.rhiGraphicsBindings);
    destroyGraphicsVariant(state.rhiGraphicsVariant);
    vernonRuntimeCorePipelineDestroy(state.rhiGraphicsPipeline);
#else
    (void)pipeline;
#endif
}

VernonStatus invokeMetalGraphicsPipeline(VernonLoadedPipeline &pipeline, const VernonPipelineInvocation &invocation,
                                         const PlannedGraphicsInvocation &plan) {
#if defined(VERNON_HAS_METAL_RUNTIME)
    MetalPipelineState &state = runtimeBackendState<MetalPipelineState>(pipeline);
    if (!state.rhiGraphicsPipeline)
        return VERNON_STATUS_OK;
    VernonRuntimeRhiAdapter &adapter = *metalState(*pipeline.context).adapter;
    for (size_t index = 0; index < state.rhiGraphicsLayout.size(); ++index) {
        const auto &layout = state.rhiGraphicsLayout[index];
        auto &prepared = state.rhiGraphicsBindingPlan[index];
        auto &value = state.rhiGraphicsValues[index];
        value = {};
        value.slot = layout.slot;
        value.kind = layout.kind;
        if (prepared.source == MetalPipelineState::GraphicsBinding::EXTERNAL_VERTEX) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_TENSOR ||
                found->second->tensor.storage != VERNON_TENSOR_RHI_RESOURCE ||
                !found->second->tensor.resource.resource.value || !found->second->tensor.byte_strides ||
                found->second->tensor.byte_strides[0] <= 0)
                return fail(*pipeline.context, "Metal RHI vertex argument is missing or invalid");
            value.payload.buffer.resource = found->second->tensor.resource;
            value.payload.buffer.resource.offset += found->second->tensor.byte_offset;
            value.payload.buffer.stride = static_cast<uint32_t>(found->second->tensor.byte_strides[0]);
        } else if (prepared.source == MetalPipelineState::GraphicsBinding::EXTERNAL_STORAGE) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_TENSOR ||
                found->second->tensor.storage != VERNON_TENSOR_RHI_RESOURCE ||
                !found->second->tensor.resource.resource.value)
                return fail(*pipeline.context, "Metal RHI storage argument is missing");
            value.payload.buffer.resource = found->second->tensor.resource;
            value.payload.buffer.resource.offset += found->second->tensor.byte_offset;
        } else if (prepared.source == MetalPipelineState::GraphicsBinding::EXTERNAL_TEXTURE) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_IMAGE ||
                !found->second->image.view.resource.value)
                return fail(*pipeline.context, "Metal RHI image argument is missing");
            value.payload.image.view = found->second->image.view;
        } else if (prepared.source == MetalPipelineState::GraphicsBinding::EXTERNAL_SAMPLER) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_SAMPLER ||
                !found->second->resource.resource.value)
                return fail(*pipeline.context, "Metal RHI sampler argument is missing");
            value.payload.sampler.resource = found->second->resource;
        } else if (prepared.source == MetalPipelineState::GraphicsBinding::IMPLICIT_SAMPLER) {
            const auto sampled = plan.sampledResources.find({prepared.descriptorSet, prepared.descriptorBinding});
            if (sampled == plan.sampledResources.end())
                return fail(*pipeline.context, "Metal RHI implicit sampler binding is missing");
            if (sampled->second.samplerResource.resource.value)
                value.payload.sampler.resource = sampled->second.samplerResource;
            else
                value.flags = VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE;
        } else if (prepared.source == MetalPipelineState::GraphicsBinding::RESOLUTION) {
            std::memcpy(prepared.storage.data(), plan.resolution.data(),
                        std::min(prepared.storage.size(), sizeof(plan.resolution)));
            value.payload.inline_value.data = prepared.storage.data();
            value.payload.inline_value.size = prepared.storage.size();
        } else {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_TENSOR)
                return fail(*pipeline.context, "Metal RHI uniform argument is missing");
            const std::optional<std::vector<uint8_t>> packed = packTensor(found->second->tensor, prepared.packing);
            if (!packed || packed->size() != prepared.storage.size())
                return fail(*pipeline.context, "Metal RHI uniform Tensor is invalid");
            prepared.storage = *packed;
            value.payload.inline_value.data = prepared.storage.data();
            value.payload.inline_value.size = prepared.storage.size();
        }
    }
    VernonStatus status =
        state.rhiGraphicsBindings
            ? vernonRuntimeCoreUpdateBindings(state.rhiGraphicsBindings, state.rhiGraphicsValues.data(),
                                              state.rhiGraphicsValues.size())
            : vernonRuntimeCoreCreateBindings(state.rhiGraphicsPipeline, state.rhiGraphicsValues.data(),
                                              state.rhiGraphicsValues.size(), &state.rhiGraphicsBindings);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError = vernonRuntimeRhiAdapterGetLastError(&adapter);
        return fail(*pipeline.context,
                    providerError.data && providerError.size ? std::string(providerError.data, providerError.size)
                                                             : "failed to update Metal RHI graphics bindings",
                    status);
    }
    if (plan.attachments.empty() || plan.attachments.size() > 8)
        return fail(*pipeline.context, "Metal RHI draw requires one to eight color attachments");
    std::array<VernonRuntimeProviderColorAttachment, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> attachments{};
    std::vector<uint32_t> formats;
    formats.reserve(plan.attachments.size());
    for (size_t index = 0; index < plan.attachments.size(); ++index) {
        const VernonColorAttachment &source = *plan.attachments[index];
        attachments[index].location = source.location;
        attachments[index].view = source.view;
        attachments[index].load_operation = source.load_operation;
        attachments[index].store_operation = source.store_operation;
        std::copy(std::begin(source.clear_color), std::end(source.clear_color), attachments[index].clear_color);
        const uint32_t format = rhi::metalTexturePixelFormat(plan.attachmentFormats[index]);
        if (!format)
            return fail(*pipeline.context, "Metal RHI color attachment is stale");
        formats.push_back(format);
    }
    VernonRuntimeProviderResourceReference depthAttachment{};
    uint32_t depthFormat = 0;
    if (plan.depthAttachment) {
        depthAttachment = plan.depthAttachment->view;
        depthFormat = rhi::metalTexturePixelFormat(plan.depthFormat);
        if (!depthFormat)
            return fail(*pipeline.context, "Metal RHI depth attachment is stale");
    }
    PlannedGraphicsState graphicsState;
    const bool hasStencil = plan.depthAttachment && plan.depthFormat == VERNON_TEXTURE_D32_FLOAT_S8_UINT;
    if (!planGraphicsState(invocation, formats.size(), plan.depthAttachment != nullptr, hasStencil, graphicsState,
                           invocationDiagnostic(*pipeline.context)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    std::vector<uint32_t> vertexStrides;
    for (size_t index = 0; index < state.rhiGraphicsLayout.size(); ++index) {
        if (state.rhiGraphicsLayout[index].kind != VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER)
            continue;
        const uint32_t binding = state.rhiGraphicsLayout[index].binding;
        if (vertexStrides.size() <= binding)
            vertexStrides.resize(binding + 1);
        vertexStrides[binding] = state.rhiGraphicsValues[index].payload.buffer.stride;
    }
    GraphicsVariantKey variantKey{static_cast<uint32_t>(invocation.topology),
                                  formats,
                                  depthFormat,
                                  1,
                                  vertexStrides,
                                  graphicsState.rasterization,
                                  graphicsState.depthStencil,
                                  graphicsState.colorBlends};
    status = ensureGraphicsVariant(state.rhiGraphicsPipeline, variantKey, state.rhiGraphicsVariant);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError = vernonRuntimeRhiAdapterGetLastError(&adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to prepare Metal RHI graphics variant",
                    status);
    }
    const bool hasViewport = invocation.viewport[2] && invocation.viewport[3];
    VernonRuntimeCoreDrawInvocation draw{};
    draw.struct_size = sizeof(draw);
    draw.command_encoder = invocation.command_encoder;
    draw.vertex_count = plan.vertexCount;
    draw.instance_count = plan.instanceCount;
    draw.color_attachments = attachments.data();
    draw.color_attachment_count = plan.attachments.size();
    draw.depth_stencil_view = depthAttachment;
    draw.depth_load_operation =
        plan.depthAttachment ? plan.depthAttachment->load_operation : VERNON_RUNTIME_PROVIDER_LOAD_DISCARD;
    draw.depth_store_operation =
        plan.depthAttachment ? plan.depthAttachment->store_operation : VERNON_RUNTIME_PROVIDER_STORE_DISCARD;
    draw.clear_depth = plan.depthAttachment ? plan.depthAttachment->clear_depth : 1.0f;
    draw.stencil_load_operation =
        hasStencil ? plan.depthAttachment->stencil_load_operation : VERNON_RUNTIME_PROVIDER_LOAD_DISCARD;
    draw.stencil_store_operation =
        hasStencil ? plan.depthAttachment->stencil_store_operation : VERNON_RUNTIME_PROVIDER_STORE_DISCARD;
    draw.clear_stencil = hasStencil ? plan.depthAttachment->clear_stencil : 0;
    draw.stencil_reference = graphicsState.stencilReference;
    draw.viewport[0] = hasViewport ? invocation.viewport[0] : 0;
    draw.viewport[1] = hasViewport ? invocation.viewport[1] : 0;
    draw.viewport[2] = hasViewport ? invocation.viewport[2] : plan.attachmentWidth;
    draw.viewport[3] = hasViewport ? invocation.viewport[3] : plan.attachmentHeight;
    const bool hasScissor = invocation.scissor[2] && invocation.scissor[3];
    for (size_t index = 0; index < 4; ++index)
        draw.scissor[index] = hasScissor ? invocation.scissor[index] : draw.viewport[index];
    draw.topology = invocation.topology;
    if (plan.indexBinding) {
        draw.index_buffer = plan.indexBinding->resource;
        draw.index_buffer.offset += plan.indexBinding->offset;
        draw.index_count = plan.indexBinding->index_count;
        draw.index_type = plan.indexBinding->type;
    }
    status = vernonRuntimeCoreEncodeGraphicsVariantDrawInvocation(state.rhiGraphicsVariant.handle,
                                                                  state.rhiGraphicsBindings, &draw);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError = vernonRuntimeRhiAdapterGetLastError(&adapter);
        return fail(*pipeline.context,
                    providerError.data && providerError.size ? std::string(providerError.data, providerError.size)
                                                             : "failed to encode Metal RHI draw",
                    status);
    }
    return VERNON_STATUS_OK;
#else
    (void)pipeline;
    (void)invocation;
    (void)plan;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

VernonStatus invokeMetalComputePipeline(VernonLoadedPipeline &pipeline, const PlannedComputeLaunch &launch) {
#if defined(VERNON_HAS_METAL_RUNTIME)
    MetalPipelineState &state = runtimeBackendState<MetalPipelineState>(pipeline);
    VernonRuntimeRhiAdapter &adapter = *metalState(*pipeline.context).adapter;
    for (size_t index = 0; index < state.rhiComputeLayout.size(); ++index) {
        const auto &layout = state.rhiComputeLayout[index];
        const ComputeBindingSource &source = state.rhiComputeBindingSources[index];
        auto &value = state.rhiComputeValues[index];
        value = {};
        value.slot = layout.slot;
        value.kind = layout.kind;
        if (layout.argument_index >= launch.arguments.size())
            return fail(*pipeline.context, "Metal prepared argument index is invalid");
        const ComputeLaunchArgument &argument = launch.arguments[layout.argument_index];
        if (source.kind != ComputeBindingSourceKind::Argument) {
            std::optional<int64_t> descriptor = computeBindingDescriptorValue(argument, source);
            if (!descriptor || *descriptor < INT32_MIN || *descriptor > INT32_MAX)
                return fail(*pipeline.context, "Metal TensorView descriptor exceeds the shader index range");
            state.rhiComputeDescriptorValues[index] = static_cast<int32_t>(*descriptor);
            value.payload.inline_value.data = &state.rhiComputeDescriptorValues[index];
            value.payload.inline_value.size = sizeof(int32_t);
            continue;
        }
        if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            const auto *tensor = std::get_if<ComputeTensorArgument>(&argument);
            if (!tensor || !tensor->resource.resource.value)
                return fail(*pipeline.context, "Metal prepared storage binding requires an RHI Tensor");
            value.payload.buffer.resource = tensor->resource;
            value.payload.buffer.resource.offset += state.rhiComputeResourceOffsets[index];
        } else if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE ||
                   layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE) {
            const auto *image = std::get_if<ComputeImageArgument>(&argument);
            if (!image || !image->view.resource.value)
                return fail(*pipeline.context, "Metal prepared image binding requires an RHI Texture");
            value.payload.image.view = image->view;
        } else {
            const auto *scalar = std::get_if<ComputeScalarArgument>(&argument);
            if (!scalar || !scalar->data || !scalar->size)
                return fail(*pipeline.context, "Metal prepared inline binding requires packed host data");
            value.payload.inline_value.data = scalar->data;
            value.payload.inline_value.size = scalar->size;
        }
    }
    VernonStatus status =
        state.rhiComputeBindings
            ? vernonRuntimeCoreUpdateBindings(state.rhiComputeBindings, state.rhiComputeValues.data(),
                                              state.rhiComputeValues.size())
            : vernonRuntimeCoreCreateBindings(state.rhiComputePipeline, state.rhiComputeValues.data(),
                                              state.rhiComputeValues.size(), &state.rhiComputeBindings);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError = vernonRuntimeRhiAdapterGetLastError(&adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to prepare Metal invocation bindings",
                    status);
    }
    const uint32_t groups[3]{launch.grid.x, launch.grid.y, launch.grid.z};
    status = vernonRuntimeCoreEncodeDispatch(state.rhiComputePipeline, state.rhiComputeBindings, launch.commandEncoder,
                                             groups, nullptr, 0);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError = vernonRuntimeRhiAdapterGetLastError(&adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to encode Metal provider dispatch",
                    status);
    }
    return VERNON_STATUS_OK;
#else
    (void)pipeline;
    (void)launch;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

} // namespace vernon::runtime
