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

struct PhysicalArgumentLayout {
    size_t offset{};
    size_t size{};
    size_t alignment{1};
};

struct TensorViewDescriptorLayout {
    uint32_t rank{};
    uint32_t offsetBinding{UINT32_MAX};
    std::vector<uint32_t> extentBindings;
    std::vector<uint32_t> strideBindings;
};

struct ReflectedArgument {
    std::string kind;
    std::string builtin;
    PhysicalArgumentLayout physical;
    size_t tensorBytes{};
    size_t tensorElements{};
    size_t tensorElementSize{};
    uint32_t descriptorSet{};
    uint32_t binding{UINT32_MAX};
    std::vector<ReflectedStorageLeaf> storageLeaves;
    std::optional<TensorViewDescriptorLayout> tensorViewDescriptor;
};

struct PackedArgumentsLayout {
    size_t size{};
};

struct ReflectedEntry {
    std::vector<ReflectedArgument> arguments;
    std::optional<PackedArgumentsLayout> packedArguments;
    uint32_t workgroup[3]{1, 1, 1};
};

bool parseReflection(const nlohmann::json &root, const std::string &selected, ReflectedEntry &output,
                     VernonRuntimeBackend backend, std::string &error);

const char *physicalValueProfileName(VernonRuntimeBackend backend, const std::string &transport);

std::optional<VernonPipelineArgumentKind> pipelineArgumentKind(const std::string &kind);

std::optional<VernonDataType> pipelineDataType(const std::string &dtype);

VernonValueLayoutView pipelineValueLayout(const ValueLayout &layout);

std::optional<VernonValueAccess> pipelineValueAccess(const std::string &access);

} // namespace vernon::runtime

#endif
