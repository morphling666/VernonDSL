#ifndef VERNON_RUNTIME_STAGE_ARTIFACT_H
#define VERNON_RUNTIME_STAGE_ARTIFACT_H

#include "VernonResult.hpp"
#include "VernonRuntime.h"
#include "pipeline_metadata.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vernon::runtime {

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
};

template <typename T> using StageArtifactResult = vernon::Result<T, StageArtifactError>;

StageArtifactResult<void> validateProgramBundleHash(const nlohmann::json &root, bool required);
StageArtifactResult<void> validateCpuRuntimeRequirements(const std::string &targetTriple,
                                                         const std::string &objectFormat);
// `cpuTarget` names the CPU target an UnsupportedCpuTarget artifact requires, as "triple / object format".
std::string renderStageArtifactError(StageArtifactError error, std::string_view cpuTarget = {});

} // namespace vernon::runtime

#endif
