#ifndef VERNON_RUNTIME_STAGE_ARTIFACT_H
#define VERNON_RUNTIME_STAGE_ARTIFACT_H

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
    std::string targetTriple;
    std::string objectFormat;
    uint64_t size{};
    std::string sha256;
    nlohmann::json reflection;
    bool staticallyLinked{};
};

struct NativeResourceSlot {
    std::string entry;
    std::string stage;
    std::string kind;
    std::string name;
    uint32_t set{};
    uint32_t binding{};
    uint32_t argumentBufferIndex{UINT32_MAX};
    uint32_t memberId{UINT32_MAX};
    uint32_t directBufferIndex{UINT32_MAX};
    uint32_t count{1};
};

struct LoadedStageArtifact {
    std::string entry;
    std::string source;
    std::string reflection;
    std::optional<ReflectedEntry> reflected;
    std::vector<NativeResourceSlot> nativeSlots;
    std::vector<uint8_t> binary;
    std::optional<CpuNativeArtifact> cpuArtifact;
    uint32_t workgroup[3]{1, 1, 1};
    DispatchContract dispatchContract;
    std::vector<TensorViewWriteFootprint> readFootprints;
    std::vector<TensorViewWriteFootprint> writeFootprints;
};

bool validateProgramBundleHash(const nlohmann::json &root, bool required, std::string &error);
bool validateCpuRuntimeRequirements(const std::string &targetTriple, const std::string &objectFormat,
                                    std::string &error);

bool resolveCpuNativeArtifact(const CpuNativeArtifact &artifact, std::filesystem::path &libraryPath,
                              ReflectedEntry *reflection, std::string &error);

} // namespace vernon::runtime

#endif
