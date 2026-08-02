#include "compiler_artifacts.h"

#include "VernonVersions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include <optional>
#include <utility>

namespace vernon::compiler {
namespace {

llvm::StringRef asStringRef(std::string_view value) { return llvm::StringRef(value.data(), value.size()); }

llvm::json::Array cpuFeatures(std::string_view features) {
    llvm::json::Array result;
    llvm::StringRef remaining = asStringRef(features);
    while (!remaining.empty()) {
        auto [feature, rest] = remaining.split(',');
        if (!feature.empty())
            result.emplace_back(feature.str());
        remaining = rest;
    }
    return result;
}

llvm::StringRef targetName(VernonTarget target) {
    switch (target) {
    case VERNON_TARGET_CPU:
        return "cpu";
    case VERNON_TARGET_OPENGL:
        return "opengl";
    case VERNON_TARGET_OPENGL_ES:
        return "opengles";
    case VERNON_TARGET_VULKAN:
        return "vulkan";
    case VERNON_TARGET_METAL:
        return "metal";
    case VERNON_TARGET_DIRECTX:
        return "directx";
    case VERNON_TARGET_CUDA:
        return "cuda";
    }
    return "unknown";
}

llvm::StringRef artifactFormat(llvm::StringRef filename) {
    llvm::StringRef extension = llvm::sys::path::extension(filename);
    if (extension == ".spv")
        return "spirv";
    if (extension == ".glsl")
        return "glsl";
    if (extension == ".gles")
        return "gles";
    if (extension == ".metal")
        return "msl";
    if (extension == ".hlsl")
        return "hlsl";
    if (extension == ".dxil")
        return "dxil";
    if (extension == ".ptx")
        return "ptx";
    if (extension == ".ll")
        return "llvm_ir";
    if (extension == ".o" || extension == ".obj")
        return "relocatable_object";
    return "unknown";
}

} // namespace

bool addArtifactTable(std::string &reflection, std::string &diagnostics, const std::vector<Artifact> &artifacts,
                      const CompileOptions &compileOptions,
                      const std::vector<TargetResourceSlot> &targetResourceSlots) {
    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(reflection);
    if (!parsed) {
        diagnostics = "compiler reflection is not valid JSON: " + llvm::toString(parsed.takeError());
        return false;
    }
    llvm::json::Object *root = parsed->getAsObject();
    if (!root) {
        diagnostics = "compiler reflection root must be a JSON object";
        return false;
    }

    const VernonTarget target = compileTargetKind(compileOptions);
    (*root)["compiler_contract_version"] = int64_t{VERNON_COMPILER_CONTRACT_VERSION};
    (*root)["pipeline_version"] = int64_t{VERNON_PIPELINE_VERSION};
    llvm::json::Object options;
    llvm::json::Object output;
    if (const auto *cpu = std::get_if<CpuCodegenOptions>(&compileOptions)) {
        const llvm::StringRef triple = asStringRef(cpu->targetTriple);
        options["triple"] = llvm::Triple::normalize(triple.empty() ? llvm::sys::getDefaultTargetTriple() : triple);
        if (!cpu->cpu.empty())
            options["processor"] = cpu->cpu;
        if (!cpu->features.empty())
            options["features"] = cpuFeatures(cpu->features);
    } else if (const auto *opengl = std::get_if<OpenGLCompileOptions>(&compileOptions)) {
        if (opengl->version != 0)
            options["version"] = static_cast<int64_t>(opengl->version);
    } else if (const auto *metal = std::get_if<MetalCompileOptions>(&compileOptions)) {
        const bool ios = metal->platform == VERNON_METAL_PLATFORM_IOS;
        options["platform"] = ios ? std::string("ios") : std::string("macos");
        output["language"] = "msl";
        output["version"] = llvm::json::Array{int64_t{2}, int64_t{4}};
        output["minimum_os_version"] =
            ios ? llvm::json::Array{int64_t{15}, int64_t{0}} : llvm::json::Array{int64_t{11}, int64_t{0}};
    } else if (const auto *directx = std::get_if<DirectXCompileOptions>(&compileOptions)) {
        options["shader_model"] = static_cast<int64_t>(directx->shaderModel);
    }
    llvm::json::Object targetSpec;
    targetSpec["kind"] = targetName(target).str();
    targetSpec["options"] = std::move(options);
    if (!output.empty())
        targetSpec["output"] = std::move(output);
    (*root)["target"] = std::move(targetSpec);

    llvm::json::Array table;
    llvm::json::Array *entries = root->getArray("entries");
    if (entries) {
        for (auto [artifactIndex, artifact] : llvm::enumerate(artifacts)) {
            llvm::StringRef artifactName = artifact.name;
            for (auto [entryIndex, entryValue] : llvm::enumerate(*entries)) {
                llvm::json::Object *entry = entryValue.getAsObject();
                if (!entry)
                    continue;
                std::optional<llvm::StringRef> entryName = entry->getString("name");
                std::optional<llvm::StringRef> stage = entry->getString("stage");
                if (!entryName || !stage)
                    continue;

                // Cross-compiled artifacts carry the entry name. Binary module
                // artifacts are emitted in the same deterministic order as entries;
                // a single module may contain every entry.
                const bool namedArtifact =
                    artifactName.starts_with(*entryName) && artifactName.drop_front(entryName->size()).starts_with(".");
                const bool sharedArtifact = artifacts.size() == 1;
                const bool parallelArtifact = artifacts.size() == entries->size() && artifactIndex == entryIndex;
                if (!namedArtifact && !sharedArtifact && !parallelArtifact)
                    continue;

                llvm::json::Object row;
                row["entry_point"] = entryName->str();
                row["stage"] = stage->str();
                row["target"] = targetName(target).str();
                row["format"] = artifactFormat(artifactName).str();
                row["filename"] = artifact.name;
                table.emplace_back(std::move(row));
            }
        }
    }
    (*root)["artifacts"] = std::move(table);
    if (target == VERNON_TARGET_METAL) {
        llvm::json::Array slots;
        for (const TargetResourceSlot &slot : targetResourceSlots) {
            llvm::json::Object row;
            row["entry_point"] = slot.entryPoint;
            row["stage"] = slot.stage;
            row["kind"] = slot.kind;
            row["name"] = slot.name;
            row["set"] = static_cast<int64_t>(slot.descriptorSet);
            row["binding"] = static_cast<int64_t>(slot.binding);
            row["argument_buffer_index"] = static_cast<int64_t>(slot.argumentBufferIndex);
            row["member_id"] = static_cast<int64_t>(slot.memberId);
            row["direct_buffer_index"] = static_cast<int64_t>(slot.directBufferIndex);
            row["count"] = static_cast<int64_t>(slot.count);
            slots.emplace_back(std::move(row));
        }
        (*root)["metal_resource_slots"] = std::move(slots);
    }

    reflection.clear();
    llvm::raw_string_ostream stream(reflection);
    stream << llvm::json::Value(std::move(*root));
    return true;
}

} // namespace vernon::compiler
