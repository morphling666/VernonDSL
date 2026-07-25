#include "compiler_artifacts.h"

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

void addArtifactTable(std::string &reflection, const std::vector<Artifact> &artifacts, VernonTarget target,
                      uint32_t glslVersion, std::string_view cpuTargetTriple, std::string_view cpu,
                      std::string_view cpuFeatures, uint32_t hlslShaderModel) {
    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(reflection);
    if (!parsed)
        return;
    llvm::json::Object *root = parsed->getAsObject();
    if (!root)
        return;

    (*root)["schema_version"] = int64_t{2};
    (*root)["target"] = targetName(target).str();
    llvm::json::Object targetOptions;
    if (target == VERNON_TARGET_OPENGL || target == VERNON_TARGET_OPENGL_ES)
        targetOptions["glsl_version"] = static_cast<int64_t>(glslVersion);
    if (target == VERNON_TARGET_DIRECTX)
        targetOptions["hlsl_shader_model"] = static_cast<int64_t>(hlslShaderModel);
    if (target == VERNON_TARGET_CPU) {
        const llvm::StringRef triple = asStringRef(cpuTargetTriple);
        targetOptions["target_triple"] =
            llvm::Triple::normalize(triple.empty() ? llvm::sys::getDefaultTargetTriple() : triple);
        if (!cpu.empty())
            targetOptions["cpu"] = std::string(cpu);
        if (!cpuFeatures.empty())
            targetOptions["cpu_features"] = std::string(cpuFeatures);
    }
    (*root)["target_options"] = std::move(targetOptions);

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

    reflection.clear();
    llvm::raw_string_ostream stream(reflection);
    stream << llvm::json::Value(std::move(*root));
}

} // namespace vernon::compiler
