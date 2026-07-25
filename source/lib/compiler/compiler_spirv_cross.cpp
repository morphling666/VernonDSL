#include "compiler_spirv_cross.h"

#include "spirv_glsl.hpp"
#include "spirv_hlsl.hpp"
#include "spirv_msl.hpp"
#include "llvm/ADT/StringRef.h"

#include <cstring>
#include <exception>
#include <map>
#include <string>
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

} // namespace

bool crossCompileSpirv(std::vector<Artifact> &artifacts, std::string &diagnostics, VernonTarget target,
                       uint32_t glslVersion, uint32_t hlslShaderModel) {
    std::vector<Artifact> translated;
    try {
        for (const Artifact &artifact : artifacts) {
            if (artifact.data.size() % sizeof(uint32_t) != 0)
                return false;
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
                    if (!variables)
                        return;
                    for (const spirv_cross::Resource &variable : *variables) {
                        if (!compiler.has_decoration(variable.id, spv::DecorationLocation))
                            continue;
                        const uint32_t location = compiler.get_decoration(variable.id, spv::DecorationLocation);
                        const auto name = varyingNames.find(location);
                        compiler.set_name(variable.id, name != varyingNames.end()
                                                           ? name->second
                                                           : "vernon_location_" + std::to_string(location));
                    }
                };
                if (target == VERNON_TARGET_METAL) {
                    spirv_cross::CompilerMSL compiler(words);
                    compiler.set_entry_point(entry.name, entry.execution_model);
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
