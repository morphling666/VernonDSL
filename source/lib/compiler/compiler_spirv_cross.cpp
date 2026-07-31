#include "compiler_spirv_cross.h"

#include "spirv_glsl.hpp"
#include "spirv_hlsl.hpp"
#include "spirv_msl.hpp"
#include "llvm/ADT/StringRef.h"

#include <cstring>
#include <exception>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace vernon::compiler {
namespace {

llvm::StringRef stageSuffix(spv::ExecutionModel model) {
    switch (model) {
    case spv::ExecutionModelVertex:
        return "vert";
    case spv::ExecutionModelFragment:
        return "frag";
    case spv::ExecutionModelGLCompute:
        return "comp";
    default:
        return "stage";
    }
}

llvm::StringRef stageName(spv::ExecutionModel model) {
    switch (model) {
    case spv::ExecutionModelVertex:
        return "vertex";
    case spv::ExecutionModelFragment:
        return "fragment";
    case spv::ExecutionModelGLCompute:
        return "compute";
    default:
        return "unknown";
    }
}

enum class MetalResourceClass { Buffer, Texture, Sampler };

struct MetalBinding {
    std::string entryPoint;
    std::string name;
    uint32_t resourceId{};
    spv::ExecutionModel stage{};
    uint32_t descriptorSet{};
    uint32_t binding{};
    MetalResourceClass resourceClass{};
    std::string kind;
    uint32_t index{};
    bool pushConstant{};
};

using MetalBindingKey = std::tuple<MetalResourceClass, uint32_t, uint32_t, std::string, std::string>;
using MetalBindingBaseKey = std::tuple<MetalResourceClass, uint32_t, uint32_t, std::string>;

constexpr uint32_t metalPushConstantBufferIndex = 15;
constexpr uint32_t metalMaxDescriptorBufferCount = metalPushConstantBufferIndex;
constexpr uint32_t metalMaxTextureCount = 128;
constexpr uint32_t metalMaxSamplerCount = 16;

bool collectMetalBindings(const std::vector<uint32_t> &words, const spirv_cross::EntryPoint &entry,
                          std::vector<MetalBinding> &bindings, std::string &diagnostics) {
    spirv_cross::CompilerMSL compiler(words);
    compiler.set_entry_point(entry.name, entry.execution_model);
    const auto activeVariables = compiler.get_active_interface_variables();
    const spirv_cross::ShaderResources resources = compiler.get_shader_resources(activeVariables);
    auto add = [&](const spirv_cross::Resource &resource, MetalResourceClass resourceClass, std::string kind,
                   bool requiresDescriptor = true) {
        if (requiresDescriptor && (!compiler.has_decoration(resource.id, spv::DecorationDescriptorSet) ||
                                   !compiler.has_decoration(resource.id, spv::DecorationBinding))) {
            diagnostics = "Metal resource '" + resource.name + "' has no descriptor set and binding";
            return false;
        }
        bindings.push_back(
            {entry.name, resource.name, resource.id, entry.execution_model,
             requiresDescriptor ? compiler.get_decoration(resource.id, spv::DecorationDescriptorSet) : UINT32_MAX,
             requiresDescriptor ? compiler.get_decoration(resource.id, spv::DecorationBinding) : UINT32_MAX,
             resourceClass, std::move(kind)});
        return true;
    };
    for (const auto &resource : resources.uniform_buffers)
        if (!add(resource, MetalResourceClass::Buffer, "uniform_buffer"))
            return false;
    for (const auto &resource : resources.storage_buffers)
        if (!add(resource, MetalResourceClass::Buffer, "storage_buffer"))
            return false;
    for (const auto &resource : resources.gl_plain_uniforms)
        if (!add(resource, MetalResourceClass::Buffer, "inline_constant", false))
            return false;
    for (const auto &resource : resources.separate_images) {
        const auto &type = compiler.get_type(resource.type_id);
        if (!add(resource, MetalResourceClass::Texture, type.image.sampled == 2 ? "storage_image" : "sampled_image"))
            return false;
    }
    for (const auto &resource : resources.storage_images)
        if (!add(resource, MetalResourceClass::Texture, "storage_image"))
            return false;
    for (const auto &resource : resources.separate_samplers)
        if (!add(resource, MetalResourceClass::Sampler, "sampler"))
            return false;
    for (const auto &resource : resources.sampled_images) {
        if (!add(resource, MetalResourceClass::Texture, "sampled_image") ||
            !add(resource, MetalResourceClass::Sampler, "sampler"))
            return false;
    }
    if (!resources.subpass_inputs.empty() || !resources.atomic_counters.empty() ||
        !resources.acceleration_structures.empty()) {
        diagnostics = "Metal target does not support a reflected shader resource class";
        return false;
    }
    if (!resources.push_constant_buffers.empty())
        bindings.push_back({entry.name, resources.push_constant_buffers.front().name,
                            resources.push_constant_buffers.front().id, entry.execution_model, 0, 0,
                            MetalResourceClass::Buffer, "inline_constant", metalPushConstantBufferIndex, true});
    return true;
}

bool assignMetalBindingIndices(std::vector<MetalBinding> &bindings, std::string &diagnostics) {
    using EntryBindingKey =
        std::tuple<std::string, spv::ExecutionModel, MetalResourceClass, uint32_t, uint32_t, std::string>;
    std::map<EntryBindingKey, size_t> entryBindingCounts;
    std::set<MetalBindingBaseKey> ambiguousBindings;
    for (const MetalBinding &binding : bindings) {
        if (binding.pushConstant)
            continue;
        const EntryBindingKey entryKey{binding.entryPoint,    binding.stage,   binding.resourceClass,
                                       binding.descriptorSet, binding.binding, binding.kind};
        if (++entryBindingCounts[entryKey] > 1)
            ambiguousBindings.emplace(binding.resourceClass, binding.descriptorSet, binding.binding, binding.kind);
    }
    auto key = [&](const MetalBinding &binding) {
        const MetalBindingBaseKey base{binding.resourceClass, binding.descriptorSet, binding.binding, binding.kind};
        return MetalBindingKey{binding.resourceClass, binding.descriptorSet, binding.binding, binding.kind,
                               ambiguousBindings.count(base) ? binding.name : std::string{}};
    };

    std::map<MetalBindingKey, uint32_t> indices;
    uint32_t nextBuffer = 0;
    uint32_t nextTexture = 0;
    uint32_t nextSampler = 0;
    for (const MetalBinding &binding : bindings) {
        if (binding.pushConstant)
            continue;
        indices.try_emplace(key(binding), 0);
    }
    for (auto &[key, index] : indices) {
        switch (std::get<0>(key)) {
        case MetalResourceClass::Buffer:
            if (nextBuffer >= metalMaxDescriptorBufferCount) {
                diagnostics = "Metal target supports at most 15 shader buffer bindings; slots 15-30 are reserved "
                              "for inline constants and vertex buffers";
                return false;
            }
            index = nextBuffer++;
            break;
        case MetalResourceClass::Texture:
            if (nextTexture >= metalMaxTextureCount) {
                diagnostics = "Metal target supports at most 128 texture bindings";
                return false;
            }
            index = nextTexture++;
            break;
        case MetalResourceClass::Sampler:
            if (nextSampler >= metalMaxSamplerCount) {
                diagnostics = "Metal target supports at most 16 sampler bindings";
                return false;
            }
            index = nextSampler++;
            break;
        }
    }
    for (MetalBinding &binding : bindings)
        if (!binding.pushConstant)
            binding.index = indices.at(key(binding));
    return true;
}

void configureMetalCompiler(spirv_cross::CompilerMSL &compiler, VernonMetalPlatform platform) {
    spirv_cross::CompilerMSL::Options options;
    options.platform = platform == VERNON_METAL_PLATFORM_IOS ? spirv_cross::CompilerMSL::Options::iOS
                                                             : spirv_cross::CompilerMSL::Options::macOS;
    options.msl_version = spirv_cross::CompilerMSL::Options::make_msl_version(2, 4);
    options.argument_buffers = false;
    compiler.set_msl_options(options);
}

void applyMetalBindings(spirv_cross::CompilerMSL &compiler, llvm::StringRef entryPoint, spv::ExecutionModel stage,
                        const std::vector<MetalBinding> &bindings) {
    std::map<uint32_t, spirv_cross::MSLResourceBinding> resources;
    for (const MetalBinding &binding : bindings) {
        if (binding.entryPoint != entryPoint || binding.stage != stage)
            continue;
        if (binding.pushConstant) {
            spirv_cross::MSLResourceBinding resource;
            resource.stage = stage;
            resource.desc_set = spirv_cross::ResourceBindingPushConstantDescriptorSet;
            resource.binding = spirv_cross::ResourceBindingPushConstantBinding;
            resource.count = 1;
            resource.msl_buffer = binding.index;
            compiler.add_msl_resource_binding(resource);
            continue;
        }
        compiler.set_decoration(binding.resourceId, spv::DecorationDescriptorSet, 31);
        compiler.set_decoration(binding.resourceId, spv::DecorationBinding, binding.resourceId);
        auto iterator = resources.try_emplace(binding.resourceId).first;
        spirv_cross::MSLResourceBinding &resource = iterator->second;
        resource.stage = stage;
        resource.desc_set = 31;
        resource.binding = binding.resourceId;
        resource.count = 1;
        if (binding.resourceClass == MetalResourceClass::Buffer)
            resource.msl_buffer = binding.index;
        else if (binding.resourceClass == MetalResourceClass::Texture)
            resource.msl_texture = binding.index;
        else
            resource.msl_sampler = binding.index;
    }
    for (const auto &resource : resources)
        compiler.add_msl_resource_binding(resource.second);
}

} // namespace

bool crossCompileSpirv(std::vector<Artifact> &artifacts, std::string &diagnostics, VernonTarget target,
                       uint32_t glslVersion, uint32_t hlslShaderModel, VernonMetalPlatform metalPlatform,
                       std::vector<TargetResourceSlot> *targetResourceSlots) {
    std::vector<Artifact> translated;
    try {
        std::vector<MetalBinding> metalBindings;
        if (target == VERNON_TARGET_METAL) {
            for (const Artifact &artifact : artifacts) {
                if (artifact.data.size() % sizeof(uint32_t) != 0) {
                    diagnostics = "SPIR-V artifact size is not aligned to 32-bit words";
                    return false;
                }
                std::vector<uint32_t> words(artifact.data.size() / sizeof(uint32_t));
                std::memcpy(words.data(), artifact.data.data(), artifact.data.size());
                spirv_cross::Compiler probe(words);
                for (const spirv_cross::EntryPoint &entry : probe.get_entry_points_and_stages())
                    if (!collectMetalBindings(words, entry, metalBindings, diagnostics))
                        return false;
            }
            if (!assignMetalBindingIndices(metalBindings, diagnostics))
                return false;
        }
        for (const Artifact &artifact : artifacts) {
            if (artifact.data.size() % sizeof(uint32_t) != 0) {
                diagnostics = "SPIR-V artifact size is not aligned to 32-bit words";
                return false;
            }
            std::vector<uint32_t> words(artifact.data.size() / sizeof(uint32_t));
            std::memcpy(words.data(), artifact.data.data(), artifact.data.size());
            spirv_cross::Compiler probe(words);
            const auto entryPoints = probe.get_entry_points_and_stages();
            std::map<uint32_t, std::string> varyingNames;
            if (target != VERNON_TARGET_METAL) {
                auto sourceName = [](std::string name) {
                    for (llvm::StringRef suffix : {"_vertex", "_fragment"}) {
                        if (llvm::StringRef(name).ends_with(suffix)) {
                            name.resize(name.size() - suffix.size());
                            break;
                        }
                    }
                    return name;
                };
                for (const spirv_cross::EntryPoint &entry : entryPoints) {
                    spirv_cross::CompilerGLSL interfaceCompiler(words);
                    interfaceCompiler.set_entry_point(entry.name, entry.execution_model);
                    const spirv_cross::ShaderResources resources = interfaceCompiler.get_shader_resources();
                    const auto *variables =
                        entry.execution_model == spv::ExecutionModelVertex     ? &resources.stage_outputs
                        : entry.execution_model == spv::ExecutionModelFragment ? &resources.stage_inputs
                                                                               : nullptr;
                    if (!variables)
                        continue;
                    for (const spirv_cross::Resource &variable : *variables) {
                        if (!interfaceCompiler.has_decoration(variable.id, spv::DecorationLocation))
                            continue;
                        const uint32_t location =
                            interfaceCompiler.get_decoration(variable.id, spv::DecorationLocation);
                        std::string name = sourceName(variable.name);
                        if (entry.execution_model == spv::ExecutionModelVertex)
                            varyingNames[location] = std::move(name);
                        else
                            varyingNames.try_emplace(location, std::move(name));
                    }
                }
            }
            for (const spirv_cross::EntryPoint &entry : entryPoints) {
                std::string source;
                std::string extension;
                auto canonicalizeInterface = [&](auto &compiler) {
                    const spirv_cross::ShaderResources resources = compiler.get_shader_resources();
                    const auto *variables =
                        entry.execution_model == spv::ExecutionModelVertex     ? &resources.stage_outputs
                        : entry.execution_model == spv::ExecutionModelFragment ? &resources.stage_inputs
                                                                               : nullptr;
                    if (variables) {
                        for (const spirv_cross::Resource &variable : *variables) {
                            if (!compiler.has_decoration(variable.id, spv::DecorationLocation))
                                continue;
                            const uint32_t location = compiler.get_decoration(variable.id, spv::DecorationLocation);
                            const auto name = varyingNames.find(location);
                            compiler.set_name(variable.id, name != varyingNames.end()
                                                               ? name->second
                                                               : "vernon_location_" + std::to_string(location));
                        }
                    }
                    for (const spirv_cross::Resource &uniform : resources.uniform_buffers) {
                        const std::string variableName = compiler.get_name(uniform.id);
                        if (variableName.empty())
                            continue;
                        compiler.set_name(uniform.base_type_id, variableName + "_block");
                        compiler.set_name(uniform.id, variableName);
                    }
                };
                if (target == VERNON_TARGET_METAL) {
                    spirv_cross::CompilerMSL compiler(words);
                    compiler.set_entry_point(entry.name, entry.execution_model);
                    configureMetalCompiler(compiler, metalPlatform);
                    applyMetalBindings(compiler, entry.name, entry.execution_model, metalBindings);
                    source = compiler.compile();
                    extension = "metal";
                } else if (target == VERNON_TARGET_DIRECTX) {
                    spirv_cross::CompilerHLSL compiler(words);
                    compiler.set_entry_point(entry.name, entry.execution_model);
                    canonicalizeInterface(compiler);
                    spirv_cross::CompilerHLSL::Options options;
                    options.shader_model = hlslShaderModel;
                    compiler.set_hlsl_options(options);
                    source = compiler.compile();
                    extension = "hlsl";
                } else {
                    spirv_cross::CompilerGLSL compiler(words);
                    compiler.set_entry_point(entry.name, entry.execution_model);
                    // Canonical location names let GLSL 3.30 link vertex outputs to
                    // fragment inputs. Prefer the source-derived vertex output name;
                    // vertex inputs and fragment outputs retain their own source names.
                    canonicalizeInterface(compiler);
                    spirv_cross::CompilerGLSL::Options options;
                    options.es = target == VERNON_TARGET_OPENGL_ES;
                    options.version = glslVersion != 0                                        ? glslVersion
                                      : options.es                                            ? 310
                                      : entry.execution_model == spv::ExecutionModelGLCompute ? 430
                                                                                              : 330;
                    options.enable_420pack_extension = options.es;
                    compiler.set_common_options(options);
                    source = compiler.compile();
                    extension = options.es ? "gles" : "glsl";
                }
                std::string name = entry.name + "." + stageSuffix(entry.execution_model).str() + "." + extension;
                translated.push_back(Artifact{std::move(name), std::move(source)});
            }
        }
        if (target == VERNON_TARGET_METAL && targetResourceSlots) {
            targetResourceSlots->clear();
            for (const MetalBinding &binding : metalBindings)
                targetResourceSlots->push_back({binding.entryPoint, stageName(binding.stage).str(), binding.kind,
                                                binding.name, binding.descriptorSet, binding.binding, binding.index,
                                                1});
        }
    } catch (const std::exception &exception) {
        diagnostics = exception.what();
        return false;
    }
    if (translated.empty()) {
        diagnostics = "SPIRV-Cross found no shader entry points";
        return false;
    }
    artifacts = std::move(translated);
    return true;
}

} // namespace vernon::compiler
