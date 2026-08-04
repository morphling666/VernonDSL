#ifndef VERNON_RUNTIME_PIPELINE_BUNDLE_H
#define VERNON_RUNTIME_PIPELINE_BUNDLE_H

#include "VernonRuntime.h"
#include "pipeline_metadata.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace vernon::runtime {

struct CpuNativeArtifact {
    std::filesystem::path root;
    std::filesystem::path relativeLibrary;
    std::string format{"native_library"};
    std::string entry;
    std::string symbol;
    std::string operatingSystem;
    std::string architecture;
    std::string targetTriple;
    std::string objectFormat;
    uint64_t size{};
    std::string sha256;
    nlohmann::json reflection;
};

struct ResolvedArtifact {
    std::string format;
    std::vector<uint8_t> bytes;
    std::filesystem::path path;
    bool external{};
};

struct Stage {
    std::string stage;
    std::string entry;
    std::string autodiffProfile;
    std::string autodiffProfilesIdentity;
    std::string source;
    std::string reflection;
    std::vector<uint8_t> binary;
    std::optional<CpuNativeArtifact> cpuArtifact;
    uint32_t workgroup[3]{1, 1, 1};
};

bool validateManifestHash(const nlohmann::json &root, bool required, std::string &error);
bool validateCpuRuntimeRequirements(const std::string &targetTriple, const std::string &objectFormat,
                                    std::string &error);

bool resolveArtifact(const nlohmann::json &descriptor, const std::optional<std::filesystem::path> &directory,
                     ResolvedArtifact &output, std::string &error);

bool resolveCpuNativeArtifact(const CpuNativeArtifact &artifact, std::filesystem::path &libraryPath,
                              ReflectedEntry *reflection, std::string &error);

bool parseCpuComputeBundle(const std::filesystem::path &root, CpuNativeArtifact &artifact, std::string &error);

} // namespace vernon::runtime

#endif
