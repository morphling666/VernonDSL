#include "stage_artifact.h"

#include "content_hash.h"

namespace vernon::runtime {
namespace {

const char *hostArchitecture() {
#if defined(_M_X64) || defined(__x86_64__)
    return "x86_64";
#elif defined(_M_ARM64) || defined(__aarch64__)
    return "aarch64";
#else
    return "unknown";
#endif
}

} // namespace

StageArtifactResult<void> validateProgramBundleHash(const nlohmann::json &root, bool required) {
    if (!root.contains("content_hash")) {
        if (!required)
            return StageArtifactResult<void>{vernon::ok()};
        return StageArtifactResult<void>{vernon::err(StageArtifactError::MissingContentHash)};
    }
    if (!root["content_hash"].is_string())
        return StageArtifactResult<void>{vernon::err(StageArtifactError::InvalidContentHash)};
    const std::string expected = root["content_hash"].get<std::string>();
    if (!isSha256Hex(expected))
        return StageArtifactResult<void>{vernon::err(StageArtifactError::InvalidContentHash)};
    nlohmann::json canonical = root;
    canonical.erase("content_hash");
    const std::string bytes = canonical.dump(-1, ' ', false, nlohmann::json::error_handler_t::strict);
    if (sha256Hex(bytes.data(), bytes.size()) != expected)
        return StageArtifactResult<void>{vernon::err(StageArtifactError::ContentHashMismatch)};
    return StageArtifactResult<void>{vernon::ok()};
}

StageArtifactResult<void> validateCpuRuntimeRequirements(const std::string &targetTriple,
                                                         const std::string &objectFormat) {
#if defined(VERNON_RUNTIME_PROFILE_WEB)
    if (targetTriple.rfind("wasm32-", 0) != 0 || targetTriple.find("emscripten") == std::string::npos ||
        objectFormat != "wasm")
        return StageArtifactResult<void>{vernon::err(StageArtifactError::UnsupportedCpuTarget)};
    return StageArtifactResult<void>{vernon::ok()};
#else
#if defined(_WIN32)
    constexpr const char *hostFormat = "coff";
    constexpr const char *hostOsToken = "windows";
#elif defined(__APPLE__)
    constexpr const char *hostFormat = "macho";
    constexpr const char *hostOsToken = "darwin";
#else
    constexpr const char *hostFormat = "elf";
    constexpr const char *hostOsToken = "linux";
#endif
    const std::string architecture = hostArchitecture();
    const bool architectureMatches = targetTriple.rfind(architecture, 0) == 0 ||
                                     (architecture == "x86_64" && targetTriple.rfind("amd64", 0) == 0) ||
                                     (architecture == "aarch64" && targetTriple.rfind("arm64", 0) == 0);
    if (!architectureMatches || targetTriple.find(hostOsToken) == std::string::npos || objectFormat != hostFormat)
        return StageArtifactResult<void>{vernon::err(StageArtifactError::UnsupportedCpuTarget)};
    return StageArtifactResult<void>{vernon::ok()};
#endif
}

std::string renderStageArtifactError(StageArtifactError error, std::string_view cpuTarget) {
    switch (error) {
    case StageArtifactError::MissingContentHash:
        return "Program bundle content_hash is missing";
    case StageArtifactError::InvalidContentHash:
        return "Program bundle content_hash is invalid";
    case StageArtifactError::ContentHashMismatch:
        return "Program bundle content_hash does not match canonical content";
    case StageArtifactError::UnsupportedCpuTarget:
        return cpuTarget.empty() ? "Stage artifact requires an unsupported CPU target"
                                 : "Stage artifact requires unsupported CPU target " + std::string(cpuTarget);
    }
    return "unknown Stage artifact error";
}

} // namespace vernon::runtime
