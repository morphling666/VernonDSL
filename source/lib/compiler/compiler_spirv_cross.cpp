#include "compiler_spirv_cross.h"

#include "runtime/interface_names.h"
#include "spirv_glsl.hpp"
#include "spirv_hlsl.hpp"
#include "spirv_msl.hpp"
#include "llvm/ADT/StringRef.h"

#include <cstring>
#include <exception>
#include <map>
#include <set>
#include <stdexcept>
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
    uint32_t argumentBufferIndex{UINT32_MAX};
    uint32_t memberId{UINT32_MAX};
    uint32_t directBufferIndex{UINT32_MAX};
    uint32_t count{1};
    bool pushConstant{};
};

constexpr uint32_t metalPushConstantBufferIndex = 15;
constexpr uint32_t metalMaxBufferCount = 31;
constexpr uint32_t metalMaxArgumentBuffers = spirv_cross::kMaxArgumentBuffers;

uint32_t metalResourceCount(const spirv_cross::CompilerMSL &compiler, const spirv_cross::Resource &resource,
                            std::string &diagnostics) {
    const auto &type = compiler.get_type(resource.type_id);
    uint64_t count = 1;
    for (uint32_t dimension : type.array) {
        if (dimension == 0 || count > UINT32_MAX / dimension) {
            diagnostics = "Metal resource '" + resource.name + "' has an unsupported descriptor array";
            return 0;
        }
        count *= dimension;
    }
    return static_cast<uint32_t>(count);
}

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
        const uint32_t count = metalResourceCount(compiler, resource, diagnostics);
        if (!count)
            return false;
        bindings.push_back(
            {entry.name, resource.name, resource.id, entry.execution_model,
             requiresDescriptor ? compiler.get_decoration(resource.id, spv::DecorationDescriptorSet) : UINT32_MAX,
             requiresDescriptor ? compiler.get_decoration(resource.id, spv::DecorationBinding) : UINT32_MAX,
             resourceClass, std::move(kind), UINT32_MAX, UINT32_MAX, UINT32_MAX, count});
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
                            MetalResourceClass::Buffer, "inline_constant", UINT32_MAX, UINT32_MAX,
                            metalPushConstantBufferIndex, 1, true});
    return true;
}

bool assignMetalArgumentBindings(std::vector<MetalBinding> &bindings, std::string &diagnostics) {
    using EntryKey = std::pair<std::string, spv::ExecutionModel>;
    using SetKey = std::tuple<std::string, spv::ExecutionModel, uint32_t>;
    using MemberKey =
        std::tuple<std::string, spv::ExecutionModel, uint32_t, uint32_t, MetalResourceClass, std::string, std::string>;
    std::map<MemberKey, uint32_t> memberCounts;
    for (const auto &binding : bindings) {
        if (binding.pushConstant || binding.descriptorSet == UINT32_MAX)
            continue;
        if (binding.descriptorSet >= metalMaxArgumentBuffers) {
            diagnostics = "Metal argument buffers support descriptor set indices 0-7";
            return false;
        }
        const MemberKey key{binding.entryPoint,    binding.stage, binding.descriptorSet, binding.binding,
                            binding.resourceClass, binding.kind,  binding.name};
        const auto [iterator, inserted] = memberCounts.emplace(key, binding.count);
        if (!inserted && iterator->second != binding.count) {
            diagnostics = "Metal descriptor binding has inconsistent array counts";
            return false;
        }
    }

    std::map<MemberKey, uint32_t> memberIds;
    std::map<SetKey, uint32_t> nextMemberId;
    for (const auto &[key, count] : memberCounts) {
        const SetKey setKey{std::get<0>(key), std::get<1>(key), std::get<2>(key)};
        uint32_t &next = nextMemberId[setKey];
        if (next > UINT32_MAX - count) {
            diagnostics = "Metal argument-buffer member IDs overflow";
            return false;
        }
        memberIds[key] = next;
        next += count;
    }

    // Descriptor set g is group g, which every stage binds as argument buffer g, so direct buffers start after the
    // argument-buffer indices of every group.
    std::map<EntryKey, uint32_t> nextDirectBuffer;
    for (auto &binding : bindings) {
        if (binding.pushConstant)
            continue;
        const EntryKey entry{binding.entryPoint, binding.stage};
        if (binding.descriptorSet != UINT32_MAX) {
            const MemberKey memberKey{binding.entryPoint,    binding.stage, binding.descriptorSet, binding.binding,
                                      binding.resourceClass, binding.kind,  binding.name};
            binding.argumentBufferIndex = binding.descriptorSet;
            binding.memberId = memberIds.at(memberKey);
            continue;
        }
        uint32_t &index = nextDirectBuffer.try_emplace(entry, metalMaxArgumentBuffers).first->second;
        if (index == metalPushConstantBufferIndex)
            ++index;
        if ((binding.stage != spv::ExecutionModelGLCompute && index >= metalPushConstantBufferIndex) ||
            index >= metalMaxBufferCount) {
            diagnostics = "Metal stage has too many direct inline buffer bindings after argument-buffer allocation";
            return false;
        }
        binding.directBufferIndex = index++;
    }
    return true;
}

void configureMetalCompiler(spirv_cross::CompilerMSL &compiler, VernonMetalPlatform platform) {
    spirv_cross::CompilerMSL::Options options;
    options.platform = platform == VERNON_METAL_PLATFORM_IOS ? spirv_cross::CompilerMSL::Options::iOS
                                                             : spirv_cross::CompilerMSL::Options::macOS;
    options.msl_version = spirv_cross::CompilerMSL::Options::make_msl_version(2, 4);
    options.argument_buffers = true;
    options.argument_buffers_tier = spirv_cross::CompilerMSL::Options::ArgumentBuffersTier::Tier2;
    compiler.set_msl_options(options);
}

void applyMetalBindings(spirv_cross::CompilerMSL &compiler, llvm::StringRef entryPoint, spv::ExecutionModel stage,
                        const std::vector<MetalBinding> &bindings) {
    using ResourceKey = std::pair<uint32_t, uint32_t>;
    std::map<ResourceKey, spirv_cross::MSLResourceBinding> resources;
    std::set<uint32_t> argumentBuffers;
    for (const MetalBinding &binding : bindings) {
        if (binding.entryPoint != entryPoint || binding.stage != stage)
            continue;
        if (binding.pushConstant) {
            spirv_cross::MSLResourceBinding resource;
            resource.stage = stage;
            resource.desc_set = spirv_cross::ResourceBindingPushConstantDescriptorSet;
            resource.binding = spirv_cross::ResourceBindingPushConstantBinding;
            resource.count = 1;
            resource.msl_buffer = binding.directBufferIndex;
            compiler.add_msl_resource_binding(resource);
            continue;
        }
        if (binding.descriptorSet == UINT32_MAX) {
            compiler.set_decoration(binding.resourceId, spv::DecorationDescriptorSet, 31);
            compiler.set_decoration(binding.resourceId, spv::DecorationBinding, binding.resourceId);
            auto iterator = resources.try_emplace({31, binding.resourceId}).first;
            auto &resource = iterator->second;
            resource.stage = stage;
            resource.desc_set = 31;
            resource.binding = binding.resourceId;
            resource.count = binding.count;
            resource.msl_buffer = binding.directBufferIndex;
            continue;
        }
        argumentBuffers.insert(binding.argumentBufferIndex);
        auto iterator = resources.try_emplace({binding.descriptorSet, binding.binding}).first;
        auto &resource = iterator->second;
        resource.stage = stage;
        resource.desc_set = binding.descriptorSet;
        resource.binding = binding.binding;
        resource.count = binding.count;
        if (binding.resourceClass == MetalResourceClass::Buffer)
            resource.msl_buffer = binding.memberId;
        else if (binding.resourceClass == MetalResourceClass::Texture)
            resource.msl_texture = binding.memberId;
        else
            resource.msl_sampler = binding.memberId;
    }
    for (uint32_t index : argumentBuffers) {
        spirv_cross::MSLResourceBinding resource;
        resource.stage = stage;
        resource.desc_set = index;
        resource.binding = spirv_cross::kArgumentBufferBinding;
        resource.count = 1;
        resource.msl_buffer = index;
        compiler.add_msl_resource_binding(resource);
    }
    for (const auto &resource : resources)
        compiler.add_msl_resource_binding(resource.second);
}

void bindDirectXRootConstants(spirv_cross::CompilerHLSL &compiler, spv::ExecutionModel stage) {
    spirv_cross::HLSLResourceBinding binding;
    binding.stage = stage;
    binding.desc_set = spirv_cross::ResourceBindingPushConstantDescriptorSet;
    binding.binding = spirv_cross::ResourceBindingPushConstantBinding;
    binding.cbv.register_binding = ::vernon::runtime::kDirectXRootConstantRegister;
    binding.cbv.register_space = ::vernon::runtime::kDirectXRootConstantRegisterSpace;
    compiler.add_hlsl_resource_binding(binding);
}

void shiftDirectXDescriptorSets(spirv_cross::Compiler &compiler) {
    const spirv_cross::ShaderResources resources = compiler.get_shader_resources();
    for (const auto *list : {&resources.uniform_buffers, &resources.storage_buffers, &resources.storage_images,
                             &resources.sampled_images, &resources.separate_images, &resources.separate_samplers})
        for (const spirv_cross::Resource &resource : *list)
            if (compiler.has_decoration(resource.id, spv::DecorationBinding))
                compiler.set_decoration(resource.id, spv::DecorationDescriptorSet,
                                        compiler.get_decoration(resource.id, spv::DecorationDescriptorSet) +
                                            ::vernon::runtime::kDirectXDescriptorSetRegisterSpaceOffset);
}

// D3D12 links a pixel shader's input signature to the vertex shader's output signature by register, and SPIRV-Cross
// places SV_Position after the varyings on both sides, so a fragment that declares fewer varyings than its vertex
// stage writes, or reads no position, would shift every register after them. Each stage compiles alone, so it cannot
// see the other's varyings. Instead both signatures start with SV_Position, the pixel shader's even when it reads no
// position. A fragment declares a prefix of the vertex outputs by location, so its signature is then always a
// register prefix of the vertex shader's.
bool leadWithDirectXPosition(std::string &source, spv::ExecutionModel stage, std::string &diagnostics) {
    if (stage != spv::ExecutionModelVertex && stage != spv::ExecutionModelFragment)
        return true;
    const std::string header =
        stage == spv::ExecutionModelVertex ? "struct SPIRV_Cross_Output\n{\n" : "struct SPIRV_Cross_Input\n{\n";
    const size_t open = source.find(header);
    if (open == std::string::npos)
        return true;
    const size_t first = open + header.size();
    const size_t close = source.find("};\n", first);
    if (close == std::string::npos) {
        diagnostics = "DirectX stage signature struct is not closed";
        return false;
    }
    std::string position;
    size_t line = first;
    while (line < close) {
        const size_t end = source.find('\n', line) + 1;
        if (llvm::StringRef(source).slice(line, end).contains(" : SV_Position;")) {
            position = source.substr(line, end - line);
            source.erase(line, end - line);
            break;
        }
        line = end;
    }
    // A vertex stage that writes no position feeds no pixel shader.
    if (position.empty() && stage == spv::ExecutionModelVertex)
        return true;
    if (position.empty())
        position = "    float4 vernon_position : SV_Position;\n";
    source.insert(first, position);
    return true;
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
            if (!assignMetalArgumentBindings(metalBindings, diagnostics))
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
                    shiftDirectXDescriptorSets(compiler);
                    bindDirectXRootConstants(compiler, entry.execution_model);
                    spirv_cross::CompilerHLSL::Options options;
                    options.shader_model = hlslShaderModel;
                    options.support_nonzero_base_vertex_base_instance = true;
                    compiler.set_hlsl_options(options);
                    compiler.set_hlsl_aux_buffer_binding(spirv_cross::HLSL_AUX_BINDING_BASE_VERTEX_INSTANCE,
                                                         ::vernon::runtime::kDirectXDrawOffsetsRegister,
                                                         ::vernon::runtime::kDirectXRootConstantRegisterSpace);
                    source = compiler.compile();
                    if (!leadWithDirectXPosition(source, entry.execution_model, diagnostics))
                        return false;
                    extension = "hlsl";
                } else {
                    spirv_cross::CompilerGLSL compiler(words);
                    compiler.set_entry_point(entry.name, entry.execution_model);
                    if (entry.execution_model == spv::ExecutionModelGLCompute) {
                        const spirv_cross::ShaderResources resources = compiler.get_shader_resources();
                        constexpr size_t portableComputeStorageBlockLimit = 16;
                        if (resources.storage_buffers.size() > portableComputeStorageBlockLimit) {
                            diagnostics = "OpenGL compute entry '" + entry.name + "' requires " +
                                          std::to_string(resources.storage_buffers.size()) +
                                          " genuine storage-buffer blocks; the portable limit is " +
                                          std::to_string(portableComputeStorageBlockLimit);
                            return false;
                        }
                    }
                    if (target == VERNON_TARGET_OPENGL_ES) {
                        const spirv_cross::ShaderResources resources = compiler.get_shader_resources();
                        for (const spirv_cross::Resource &resource : resources.storage_images) {
                            const auto &type = compiler.get_type(resource.type_id);
                            const bool coreReadWriteFormat = type.image.format == spv::ImageFormatR32f ||
                                                             type.image.format == spv::ImageFormatR32i ||
                                                             type.image.format == spv::ImageFormatR32ui;
                            const bool singleDirection =
                                compiler.has_decoration(resource.id, spv::DecorationNonReadable) ||
                                compiler.has_decoration(resource.id, spv::DecorationNonWritable);
                            if (!coreReadWriteFormat && !singleDirection) {
                                diagnostics = "OpenGL ES requires a read_write storage image to use r32_float, "
                                              "r32_sint, or r32_uint format";
                                return false;
                            }
                        }
                    }
                    // Canonical location names let GLSL 3.30 link vertex outputs to
                    // fragment inputs. Prefer the source-derived vertex output name;
                    // vertex inputs and fragment outputs retain their own source names.
                    canonicalizeInterface(compiler);
                    for (const spirv_cross::Resource &uniform : compiler.get_shader_resources().uniform_buffers) {
                        if (!compiler.has_decoration(uniform.id, spv::DecorationBinding)) {
                            diagnostics = "OpenGL uniform block '" + compiler.get_name(uniform.id) + "' has no binding";
                            return false;
                        }
                        compiler.set_name(uniform.base_type_id,
                                          ::vernon::runtime::openGLUniformBlockName(
                                              compiler.get_decoration(uniform.id, spv::DecorationBinding)));
                    }
                    spirv_cross::CompilerGLSL::Options options;
                    options.es = target == VERNON_TARGET_OPENGL_ES;
                    options.version = glslVersion != 0                                        ? glslVersion
                                      : options.es                                            ? 310
                                      : entry.execution_model == spv::ExecutionModelGLCompute ? 430
                                                                                              : 330;
                    options.enable_420pack_extension = options.es;
                    // Image row 0 is the top (normalized device y = +1) on every target, but GL rasterizes y = +1
                    // to the last row. Negating y in the vertex stage also reverses winding, which the OpenGL
                    // adapter compensates by inverting the front face.
                    options.vertex.flip_vert_y = entry.execution_model == spv::ExecutionModelVertex;
                    // Clip depth is [0, w] on every target, but GL clips to [-w, w]; the vertex stage remaps it, so
                    // stored depth is clip z / w as elsewhere.
                    options.vertex.fixup_clipspace = entry.execution_model == spv::ExecutionModelVertex;
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
                targetResourceSlots->push_back(
                    TargetResourceSlot{binding.entryPoint, stageName(binding.stage).str(), binding.kind, binding.name,
                                       binding.descriptorSet, binding.binding, binding.argumentBufferIndex,
                                       binding.memberId, binding.directBufferIndex, binding.count});
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
