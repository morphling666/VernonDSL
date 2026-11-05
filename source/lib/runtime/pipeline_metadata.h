#ifndef VERNON_RUNTIME_PIPELINE_METADATA_H
#define VERNON_RUNTIME_PIPELINE_METADATA_H

#include "VernonRuntime.h"
#include "dispatch_contract.h"
#include "stage_binding_plan.h"

#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vernon::runtime {

struct PhysicalArgumentLayout {
    size_t offset{};
    size_t size{};
    size_t alignment{1};
};

// `size` bytes of a value at `physicalOffset` of its physical bytes and `canonicalOffset` of its canonical ones.
struct ValueByteCopy {
    size_t physicalOffset{};
    size_t canonicalOffset{};
    size_t size{};
};

struct ReflectedArgument {
    std::string sourceName;
    std::string kind;
    std::string builtin;
    std::string autodiffRole;
    // The access the compiled module declares, which selects native descriptor kinds and write hazards.
    std::string access;
    // "invocation_linear" for a GPU backward kernel's cotangent carrier, whose leading axis is the dispatch's.
    std::string autodiffCarrier;
    std::optional<VernonDataType> dtype;
    bool result{};
    // A value argument whose physical bytes are its canonical value's bytes, such as one scalar or a Struct whose
    // leaves keep their canonical offsets, so a linked schedule copies it whole.
    bool canonicalBytes{};
    // A CPU result's physical bytes as its canonical value's, which a linked schedule copies out of the results frame.
    std::vector<ValueByteCopy> resultCopies;
    uint32_t index{UINT32_MAX};
    PhysicalArgumentLayout physical;
    size_t tensorBytes{};
    size_t tensorElements{};
    size_t tensorElementSize{};
    uint32_t descriptorSet{};
    uint32_t binding{UINT32_MAX};
    std::vector<ReflectedStorageLeaf> storageLeaves;
    std::vector<int64_t> sourceShape;
};

struct TensorViewWriteFootprint {
    uint32_t argument{};
    std::string owner;
    bool wholeView{true};
    std::vector<uint64_t> indices;
};

struct PackedArgumentsLayout {
    size_t size{};
};

struct ReflectedEntry {
    std::vector<ReflectedArgument> arguments;
    std::optional<MetadataCarrier> metadataCarrier;
    std::optional<PackedArgumentsLayout> packedArguments;
    std::optional<PackedArgumentsLayout> packedResults;
    uint32_t workgroup[3]{1, 1, 1};
    DispatchContract dispatchContract;
    std::vector<TensorViewWriteFootprint> readFootprints;
    std::vector<TensorViewWriteFootprint> writeFootprints;
};

struct LoadedStageArtifact;

bool parseReflection(const nlohmann::json &root, const std::string &selected, ReflectedEntry &output,
                     VernonRuntimeBackend backend, std::string &error);
bool resolveStageReflection(const LoadedStageArtifact &stage, VernonRuntimeBackend backend, ReflectedEntry &output,
                            std::string &error);

const char *physicalValueProfileName(VernonRuntimeBackend backend, const std::string &transport);

std::optional<VernonProgramArgumentKind> pipelineArgumentKind(const std::string &kind);

std::optional<VernonDataType> pipelineDataType(const std::string &dtype);

VernonValueLayoutView pipelineValueLayout(const ValueLayout &layout);

std::optional<VernonValueAccess> pipelineValueAccess(const std::string &access);

bool configureImageBindingLayout(const Parameter &parameter, VernonRuntimeProviderBindingLayoutEntry &layout);

} // namespace vernon::runtime

#endif
