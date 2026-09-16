#ifndef VERNON_RUNTIME_STAGE_ARTIFACT_H
#define VERNON_RUNTIME_STAGE_ARTIFACT_H

#include "VernonResult.hpp"
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

enum class StageArtifactError : uint8_t {
    MissingContentHash,
    InvalidContentHash,
    ContentHashMismatch,
    UnsupportedCpuTarget,
    InvalidCpuArtifact,
    InvalidCpuArtifactPath,
    InvalidReflection,
    InvalidBundleDirectory,
    CpuArtifactIntegrityMismatch,
};

template <typename T> using StageArtifactResult = vernon::Result<T, StageArtifactError>;

struct ResolvedCpuNativeArtifact {
    std::filesystem::path libraryPath;
    ReflectedEntry reflection;
};

StageArtifactResult<void> validateProgramBundleHash(const nlohmann::json &root, bool required);
StageArtifactResult<void> validateCpuRuntimeRequirements(const std::string &targetTriple,
                                                         const std::string &objectFormat);
StageArtifactResult<ResolvedCpuNativeArtifact> resolveCpuNativeArtifact(const CpuNativeArtifact &artifact);
std::string renderStageArtifactError(StageArtifactError error, const CpuNativeArtifact *artifact = nullptr);

} // namespace vernon::runtime

#endif
