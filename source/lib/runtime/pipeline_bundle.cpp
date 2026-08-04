#include "pipeline_bundle.h"

#include "content_hash.h"

#include <algorithm>
#include <array>
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

std::optional<std::vector<uint8_t>> decodeBase64Strict(const std::string &encoded) {
    static const std::array<int8_t, 256> decodeTable = [] {
        std::array<int8_t, 256> table{};
        table.fill(-1);
        constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int8_t index = 0; index < 64; ++index)
            table[static_cast<uint8_t>(alphabet[index])] = index;
        return table;
    }();
    if (encoded.size() % 4 != 0)
        return std::nullopt;
    std::vector<uint8_t> result;
    result.reserve(encoded.size() / 4 * 3);
    for (size_t offset = 0; offset < encoded.size(); offset += 4) {
        uint32_t value = 0;
        size_t padding = 0;
        for (size_t index = 0; index < 4; ++index) {
            const unsigned char character = encoded[offset + index];
            if (character == '=') {
                if (index < 2 || offset + 4 != encoded.size())
                    return std::nullopt;
                ++padding;
                value <<= 6;
                continue;
            }
            if (padding)
                return std::nullopt;
            const int8_t decoded = decodeTable[character];
            if (decoded < 0)
                return std::nullopt;
            value = (value << 6) | static_cast<uint32_t>(decoded);
        }
        if (padding > 2 || (padding == 1 && (value & 0xffu) != 0) || (padding == 2 && (value & 0xffffu) != 0))
            return std::nullopt;
        result.push_back(static_cast<uint8_t>(value >> 16));
        if (padding < 2)
            result.push_back(static_cast<uint8_t>(value >> 8));
        if (!padding)
            result.push_back(static_cast<uint8_t>(value));
    }
    return result;
}

const char *hostOperatingSystem() {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#else
    return "unknown";
#endif
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

bool validateManifestHash(const nlohmann::json &root, bool required, std::string &error) {
    if (!root.contains("content_hash")) {
        if (!required)
            return true;
        error = "pipeline manifest content_hash is missing";
        return false;
    }
    if (!root["content_hash"].is_string()) {
        error = "pipeline manifest content_hash is invalid";
        return false;
    }
    const std::string expected = root["content_hash"].get<std::string>();
    if (!isSha256(expected)) {
        error = "pipeline manifest content_hash is invalid";
        return false;
    }
    nlohmann::json canonical = root;
    canonical.erase("content_hash");
    const std::string bytes = canonical.dump(-1, ' ', false, nlohmann::json::error_handler_t::strict);
    if (sha256Hex(bytes.data(), bytes.size()) != expected) {
        error = "pipeline manifest content_hash does not match canonical content";
        return false;
    }
    return true;
}

bool validateCpuRuntimeRequirements(const std::string &targetTriple, const std::string &objectFormat,
                                    std::string &error) {
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
    if (!architectureMatches || targetTriple.find(hostOsToken) == std::string::npos || objectFormat != hostFormat) {
        error = "pipeline requires CPU target " + targetTriple + " / " + objectFormat + ", runtime provides " +
                architecture + "-" + hostOsToken + " / " + hostFormat;
        return false;
    }
    return true;
}

bool resolveArtifact(const nlohmann::json &descriptor, const std::optional<std::filesystem::path> &directory,
                     ResolvedArtifact &output, std::string &error) {
    if (!descriptor.is_object() || !descriptor.contains("format") || !descriptor["format"].is_string() ||
        !descriptor.contains("storage") || !descriptor["storage"].is_string() || !descriptor.contains("size") ||
        !descriptor["size"].is_number_unsigned() || !descriptor.contains("sha256") ||
        !descriptor["sha256"].is_string()) {
        error = "pipeline artifact descriptor is invalid";
        return false;
    }
    output.format = descriptor["format"].get<std::string>();
    const std::string storage = descriptor["storage"].get<std::string>();
    const uint64_t declaredSize = descriptor["size"].get<uint64_t>();
    const std::string expectedHash = descriptor["sha256"].get<std::string>();
    if (output.format.empty() || !isSha256(expectedHash)) {
        error = "pipeline artifact format or SHA-256 is invalid";
        return false;
    }

    if (storage == "inline") {
        if (!descriptor.contains("encoding") || !descriptor["encoding"].is_string() || !descriptor.contains("data") ||
            !descriptor["data"].is_string()) {
            error = "inline pipeline artifact is invalid";
            return false;
        }
        const std::string encoding = descriptor["encoding"].get<std::string>();
        const std::string data = descriptor["data"].get<std::string>();
        if (encoding == "utf8")
            output.bytes.assign(data.begin(), data.end());
        else if (encoding == "base64") {
            auto decoded = decodeBase64Strict(data);
            if (!decoded) {
                error = "inline pipeline artifact base64 is invalid";
                return false;
            }
            output.bytes = std::move(*decoded);
        } else {
            error = "inline pipeline artifact encoding is unsupported";
            return false;
        }
    } else if (storage == "external") {
        if (!directory || !descriptor.contains("path") || !descriptor["path"].is_string()) {
            error = "external pipeline artifacts require a bundle directory and path";
            return false;
        }
        const std::filesystem::path relative = std::filesystem::u8path(descriptor["path"].get<std::string>());
        if (relative.empty() || relative.is_absolute() || relative.has_root_path() ||
            relative == std::filesystem::path(".") || relative.lexically_normal() != relative ||
            std::find(relative.begin(), relative.end(), std::filesystem::path("..")) != relative.end()) {
            error = "external pipeline artifact path is not normalized";
            return false;
        }
        std::error_code filesystemError;
        const std::filesystem::path root = std::filesystem::canonical(*directory, filesystemError);
        if (filesystemError || !std::filesystem::is_directory(root, filesystemError)) {
            error = "pipeline bundle directory is invalid";
            return false;
        }
        output.path = std::filesystem::canonical(root / relative, filesystemError);
        if (filesystemError || !std::filesystem::is_regular_file(output.path, filesystemError)) {
            error = "external pipeline artifact cannot be resolved";
            return false;
        }
        const std::filesystem::path contained = output.path.lexically_relative(root);
        if (contained.empty() || contained.is_absolute() || *contained.begin() == std::filesystem::path("..")) {
            error = "external pipeline artifact escapes the bundle directory";
            return false;
        }
        output.bytes = readFile(output.path);
        output.external = true;
    } else {
        error = "pipeline artifact storage is unsupported";
        return false;
    }

    if (output.bytes.size() != declaredSize || sha256Hex(output.bytes.data(), output.bytes.size()) != expectedHash) {
        error = "pipeline artifact size or SHA-256 mismatch";
        return false;
    }
    return true;
}

bool resolveCpuNativeArtifact(const CpuNativeArtifact &artifact, std::filesystem::path &libraryPath,
                              ReflectedEntry *reflection, std::string &error) {
    const bool nativeLibrary = artifact.format == "native_library";
    const bool relocatableObject = artifact.format == "relocatable_object";
    if (artifact.entry.empty() || artifact.symbol.empty() || artifact.relativeLibrary.empty() ||
        artifact.sha256.size() != 64 || (!nativeLibrary && !relocatableObject) ||
        (nativeLibrary &&
         (artifact.operatingSystem != hostOperatingSystem() || artifact.architecture != hostArchitecture())) ||
        (relocatableObject &&
         (artifact.targetTriple.empty() || (artifact.objectFormat != "coff" && artifact.objectFormat != "elf" &&
                                            artifact.objectFormat != "macho" && artifact.objectFormat != "wasm")))) {
        error = "unsupported or invalid CPU AOT artifact";
        return false;
    }
    if (artifact.relativeLibrary.is_absolute() || artifact.relativeLibrary.has_root_path() ||
        std::find(artifact.relativeLibrary.begin(), artifact.relativeLibrary.end(), std::filesystem::path("..")) !=
            artifact.relativeLibrary.end()) {
        error = "CPU AOT artifact path is invalid";
        return false;
    }

    std::error_code filesystemError;
    const std::filesystem::path canonicalRoot = std::filesystem::canonical(artifact.root, filesystemError);
    if (filesystemError || !std::filesystem::is_directory(canonicalRoot, filesystemError)) {
        error = "CPU AOT bundle directory is invalid";
        return false;
    }
    const std::filesystem::path candidate =
        std::filesystem::weakly_canonical(canonicalRoot / artifact.relativeLibrary, filesystemError);
    if (filesystemError) {
        error = "CPU AOT artifact path cannot be resolved";
        return false;
    }
    const std::filesystem::path relative = candidate.lexically_relative(canonicalRoot);
    if (relative.empty() || relative.is_absolute() || *relative.begin() == std::filesystem::path("..")) {
        error = "CPU AOT artifact escapes the bundle directory";
        return false;
    }

    const std::vector<uint8_t> bytes = readFile(candidate);
    if (bytes.empty() || artifact.size != bytes.size() || sha256Hex(bytes.data(), bytes.size()) != artifact.sha256) {
        error = "CPU AOT artifact size or SHA-256 mismatch";
        return false;
    }
    ReflectedEntry parsed;
    if (!parseReflection(artifact.reflection, artifact.entry, parsed, VERNON_RUNTIME_CPU, error))
        return false;
    libraryPath = candidate;
    if (reflection)
        *reflection = std::move(parsed);
    return true;
}

bool parseCpuComputeBundle(const std::filesystem::path &root, CpuNativeArtifact &artifact, std::string &error) {
    const std::vector<uint8_t> manifestBytes = readFile(root / "compute.json");
    const nlohmann::json manifest = nlohmann::json::parse(manifestBytes.begin(), manifestBytes.end(), nullptr, false);
    if (manifest.is_discarded() || !manifest.is_object() ||
        manifest.value("pipeline_version", 0) != VERNON_PIPELINE_VERSION || manifest.value("target", "") != "cpu" ||
        manifest.value("artifact_format", "") != "native_library" || !manifest.contains("reflection")) {
        error = "unsupported or invalid CPU AOT bundle";
        return false;
    }
    artifact = {};
    artifact.root = root;
    artifact.relativeLibrary = std::filesystem::u8path(manifest.value("artifact", ""));
    artifact.entry = manifest.value("entry", "");
    artifact.symbol = manifest.value("symbol", "");
    artifact.operatingSystem = manifest.value("operating_system", "");
    artifact.architecture = manifest.value("architecture", "");
    artifact.size = manifest.value("artifact_size", uint64_t{0});
    artifact.sha256 = manifest.value("artifact_sha256", "");
    artifact.reflection = manifest["reflection"];
    return true;
}

} // namespace vernon::runtime
