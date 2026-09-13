#include "stage_artifact.h"

#include "content_hash.h"

#include <algorithm>
#include <fstream>
#include <limits>

namespace vernon::runtime {
namespace {

std::vector<uint8_t> readFile(const std::filesystem::path &path) {
    std::error_code error;
    const uintmax_t fileSize = std::filesystem::file_size(path, error);
    if (error || fileSize > std::numeric_limits<size_t>::max() ||
        fileSize > static_cast<uintmax_t>(std::numeric_limits<std::streamsize>::max()))
        return {};
    std::ifstream input(path, std::ios::binary);
    std::vector<uint8_t> bytes(static_cast<size_t>(fileSize));
    if (!bytes.empty() &&
        !input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        return {};
    return bytes;
}

bool isSha256(const std::string &value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char character) {
               return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
           });
}

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
    if (!isSha256(expected))
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

StageArtifactResult<ResolvedCpuNativeArtifact> resolveCpuNativeArtifact(const CpuNativeArtifact &artifact) {
    const bool nativeLibrary = artifact.format == "native_library";
    const bool relocatableObject = artifact.format == "relocatable_object";
    if (artifact.entry.empty() || artifact.symbol.empty() || artifact.relativeLibrary.empty() || !artifact.size ||
        !isSha256(artifact.sha256) || (!nativeLibrary && !relocatableObject) ||
        (artifact.staticallyLinked && !relocatableObject) || artifact.targetTriple.empty() ||
        (artifact.objectFormat != "coff" && artifact.objectFormat != "elf" && artifact.objectFormat != "macho" &&
         artifact.objectFormat != "wasm"))
        return StageArtifactResult<ResolvedCpuNativeArtifact>{vernon::err(StageArtifactError::InvalidCpuArtifact)};
    if (artifact.relativeLibrary.is_absolute() || artifact.relativeLibrary.has_root_path() ||
        std::find(artifact.relativeLibrary.begin(), artifact.relativeLibrary.end(), std::filesystem::path("..")) !=
            artifact.relativeLibrary.end())
        return StageArtifactResult<ResolvedCpuNativeArtifact>{vernon::err(StageArtifactError::InvalidCpuArtifactPath)};

    ReflectedEntry parsed;
    std::string reflectionError;
    if (!parseReflection(artifact.reflection, artifact.entry, parsed, VERNON_RUNTIME_CPU, reflectionError))
        return StageArtifactResult<ResolvedCpuNativeArtifact>{vernon::err(StageArtifactError::InvalidReflection)};
    ResolvedCpuNativeArtifact resolved;
    resolved.reflection = std::move(parsed);
    if (artifact.staticallyLinked) {
        return StageArtifactResult<ResolvedCpuNativeArtifact>{vernon::ok(std::move(resolved))};
    }

    std::error_code filesystemError;
    const std::filesystem::path canonicalRoot = std::filesystem::canonical(artifact.root, filesystemError);
    if (filesystemError || !std::filesystem::is_directory(canonicalRoot, filesystemError))
        return StageArtifactResult<ResolvedCpuNativeArtifact>{vernon::err(StageArtifactError::InvalidBundleDirectory)};
    const std::filesystem::path candidate =
        std::filesystem::weakly_canonical(canonicalRoot / artifact.relativeLibrary, filesystemError);
    if (filesystemError)
        return StageArtifactResult<ResolvedCpuNativeArtifact>{vernon::err(StageArtifactError::InvalidCpuArtifactPath)};
    const std::filesystem::path relative = candidate.lexically_relative(canonicalRoot);
    if (relative.empty() || relative.is_absolute() || *relative.begin() == std::filesystem::path(".."))
        return StageArtifactResult<ResolvedCpuNativeArtifact>{vernon::err(StageArtifactError::InvalidCpuArtifactPath)};

    const std::vector<uint8_t> bytes = readFile(candidate);
    if (bytes.empty() || artifact.size != bytes.size() || sha256Hex(bytes.data(), bytes.size()) != artifact.sha256)
        return StageArtifactResult<ResolvedCpuNativeArtifact>{
            vernon::err(StageArtifactError::CpuArtifactIntegrityMismatch)};
    resolved.libraryPath = candidate;
    return StageArtifactResult<ResolvedCpuNativeArtifact>{vernon::ok(std::move(resolved))};
}

std::string renderStageArtifactError(StageArtifactError error, const CpuNativeArtifact *artifact) {
    switch (error) {
    case StageArtifactError::MissingContentHash:
        return "Program bundle content_hash is missing";
    case StageArtifactError::InvalidContentHash:
        return "Program bundle content_hash is invalid";
    case StageArtifactError::ContentHashMismatch:
        return "Program bundle content_hash does not match canonical content";
    case StageArtifactError::UnsupportedCpuTarget:
        return artifact ? "Stage artifact requires unsupported CPU target " + artifact->targetTriple + " / " +
                              artifact->objectFormat
                        : "Stage artifact requires an unsupported CPU target";
    case StageArtifactError::InvalidCpuArtifact:
        return "unsupported or invalid CPU AOT artifact";
    case StageArtifactError::InvalidCpuArtifactPath:
        return "CPU AOT artifact path is invalid or escapes the bundle directory";
    case StageArtifactError::InvalidReflection:
        return "CPU AOT artifact reflection is invalid";
    case StageArtifactError::InvalidBundleDirectory:
        return "Program bundle directory is invalid";
    case StageArtifactError::CpuArtifactIntegrityMismatch:
        return "CPU AOT artifact size or SHA-256 mismatch";
    }
    return "unknown Stage artifact error";
}

} // namespace vernon::runtime
