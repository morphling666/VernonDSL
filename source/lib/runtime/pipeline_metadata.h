#ifndef VERNON_RUNTIME_PIPELINE_METADATA_H
#define VERNON_RUNTIME_PIPELINE_METADATA_H

#include "VernonRuntime.h"

#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vernon::runtime {

struct ValueLayout;

struct ReflectedStorageLeaf {
    size_t elementSize{};
    size_t byteOffset{};
    uint32_t binding{UINT32_MAX};
};

struct ReflectedArgument {
    std::string kind;
    std::string builtin;
    size_t cpuOffset{};
    size_t cpuSize{};
    size_t physicalSize{};
    size_t tensorBytes{};
    size_t tensorElements{};
    size_t tensorElementSize{};
    size_t alignment{1};
    uint32_t descriptorSet{};
    uint32_t binding{UINT32_MAX};
    std::vector<ReflectedStorageLeaf> storageLeaves;
};

struct ReflectedEntry {
    std::vector<ReflectedArgument> arguments;
    size_t cpuArgumentsSize{};
    uint32_t workgroup[3]{1, 1, 1};
};

bool parseReflection(const nlohmann::json &root, const std::string &selected, ReflectedEntry &output,
                     std::string &error);

std::optional<VernonPipelineArgumentKind> pipelineArgumentKind(const std::string &kind);

std::optional<VernonDataType> pipelineDataType(const std::string &dtype);

VernonValueLayoutView pipelineValueLayout(const ValueLayout &layout);

std::optional<VernonValueAccess> pipelineValueAccess(const std::string &access);

} // namespace vernon::runtime

#endif
