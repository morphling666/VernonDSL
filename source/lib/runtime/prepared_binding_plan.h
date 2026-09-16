#ifndef VERNON_RUNTIME_PREPARED_BINDING_PLAN_H
#define VERNON_RUNTIME_PREPARED_BINDING_PLAN_H

#include "VernonRuntimeProvider.h"
#include "compute_launch_planner.h"
#include "graphics_invocation_planner.h"
#include "pipeline_metadata.h"
#include "stage_binding_plan.h"
#include "tensor_bridge.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vernon::runtime {

struct PreparedBindingSource {
    uint32_t argumentIndex{UINT32_MAX};
    uint64_t resourceOffset{};
    bool metadataCarrier{};
};

struct PreparedCpuBinding {
    std::string builtin;
    size_t packedOffset{};
    size_t packedSize{};
    bool result{};
    std::optional<VernonDataType> resultReduction;
};

struct PreparedMetadataCarrier {
    MetadataCarrier plan;
    VernonRuntimeProviderBindingLayoutEntry layout{};
};

// Immutable, ordinal-aligned physical binding sequence for one compute Stage.
// layouts and sources have identical length. CPU plans additionally carry one
// packed call-frame record per physical parameter ordinal.
struct PreparedComputeBindingPlan {
    VernonRuntimeBackend backend{VERNON_RUNTIME_CPU};
    std::vector<VernonRuntimeProviderBindingLayoutEntry> layouts;
    std::vector<PreparedBindingSource> sources;
    std::optional<PreparedMetadataCarrier> metadataCarrier;
    std::vector<PreparedCpuBinding> cpuBindings;
    size_t packedArgumentSize{};
    size_t packedResultSize{};
    size_t tapeAllocatorOffset{std::numeric_limits<size_t>::max()};
    size_t tapeRootOffset{std::numeric_limits<size_t>::max()};

    bool empty() const { return layouts.empty(); }
    size_t size() const { return layouts.size(); }
};

enum class PreparedGraphicsBindingSourceKind : uint8_t {
    ExternalUniform,
    ExternalVertex,
    ExternalImage,
    ExternalSampler,
    ImplicitSampler,
    Resolution,
};

struct PreparedGraphicsBindingSource {
    PreparedGraphicsBindingSourceKind kind{PreparedGraphicsBindingSourceKind::ExternalUniform};
    uint32_t externalSlot{};
    // Canonical shader descriptor identity. This intentionally remains unchanged when a
    // backend remaps the provider layout (for example, Metal argument buffers).
    uint32_t descriptorSet{UINT32_MAX};
    uint32_t descriptorBinding{UINT32_MAX};
    TensorCopyPlan packing;
    std::vector<uint8_t> storage;
    std::string nativeName;
};

struct OpenGLNativeUniformShape {
    uint32_t scalarCount{};
    uint32_t matrixColumns{1};
};

using ResolvePreparedGraphicsNativeBinding = bool (*)(void *userData, std::string_view stage, std::string_view kind,
                                                      uint32_t descriptorSet, uint32_t descriptorBinding,
                                                      const std::string *name,
                                                      VernonRuntimeProviderBindingLayoutEntry &layout,
                                                      std::string &error);

struct PreparedGraphicsBindingOptions {
    VernonRuntimeBackend backend{VERNON_RUNTIME_OPENGL};
    ResolvePreparedGraphicsNativeBinding resolveNativeBinding{};
    void *nativeBindingUserData{};
};

// Immutable, ordinal-aligned physical binding sequence for one graphics Stage.
// layouts and sources have identical length; provider slot order is canonical.
struct PreparedGraphicsBindingPlan {
    VernonRuntimeBackend backend{VERNON_RUNTIME_OPENGL};
    std::vector<VernonRuntimeProviderBindingLayoutEntry> layouts;
    std::vector<PreparedGraphicsBindingSource> sources;
    std::vector<VernonRuntimeProviderVertexAttribute> vertexAttributes;

    bool empty() const { return layouts.empty(); }
    size_t size() const { return layouts.size(); }
};

bool buildPreparedComputeBindingPlan(const StageBindingPlan &stagePlan, const ReflectedEntry &reflection,
                                     VernonRuntimeBackend backend, PreparedComputeBindingPlan &output,
                                     std::string &error);
bool validatePreparedComputeBindingPlan(const PreparedComputeBindingPlan &plan, std::string &error);

bool resolveOpenGLNativeUniformShape(std::string_view dtype, const std::vector<uint64_t> &shape,
                                     OpenGLNativeUniformShape &result);
bool buildPreparedGraphicsBindingPlan(const StageBindingPlan &stagePlan, const PreparedGraphicsBindingOptions &options,
                                      PreparedGraphicsBindingPlan &output, std::string &error);
bool validatePreparedGraphicsBindingPlan(const PreparedGraphicsBindingPlan &plan, std::string &error);
bool fillPreparedGraphicsBindingValues(PreparedGraphicsBindingPlan &prepared,
                                       const PlannedGraphicsInvocation &invocation,
                                       std::vector<VernonRuntimeProviderBindingValue> &values, std::string &error);

} // namespace vernon::runtime

#endif
