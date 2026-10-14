#ifndef VERNON_RUNTIME_RECORDED_GRAPHICS_H
#define VERNON_RUNTIME_RECORDED_GRAPHICS_H

#include "VernonResult.hpp"
#include "program_execution_manifest.h"
#include "resolved_execution_plan.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace vernon::runtime::recorded {

// Stage-local endpoint identity; endpoints of different pipelines correspond when their keys are equal.
struct EndpointKey {
    std::string module;
    std::string sourceName;

    bool operator==(const EndpointKey &other) const { return module == other.module && sourceName == other.sourceName; }
};

// One Stage endpoint of a graphics Node, in Stage parameter order.
struct NodeEndpoint {
    EndpointKey key;
    // Set when the endpoint reads a bindable Program parameter that no Node writes; every other endpoint is fixed.
    std::optional<uint32_t> perDrawSlot;
};

// How one Stage endpoint of a pipeline is supplied while the pipeline draws inside a recorded pass.
struct PipelineEndpoint {
    EndpointKey key;
    // The pass Node fixes this endpoint to the Node's own endpoint at this Stage parameter index.
    std::optional<size_t> fixedPassParameter;
    // Per-draw endpoint: the parameter of the pipeline's Program feeding it.
    std::optional<uint32_t> parameterSlot;
    // Per-draw endpoint that draws with the pass Node's own binding at this index until a recording rebinds it.
    std::optional<size_t> initialPassParameter;
};

// Diagnostic codes and phases of specs/program/recorded_draws.md.
inline constexpr const char *kPipelineConformance = "PROGRAM_PIPELINE_CONFORMANCE";
inline constexpr const char *kRecordingInvalid = "PROGRAM_RECORDING_INVALID";
inline constexpr const char *kBindingSetMismatch = "PROGRAM_BINDING_SET_MISMATCH";
inline constexpr const char *kBindingSetHazard = "PROGRAM_BINDING_SET_HAZARD";
inline constexpr const char *kPipelineCreationPhase = "pipeline_creation";
inline constexpr const char *kInvocationEntryPhase = "invocation_entry";

using BindableSlot = std::function<bool(uint32_t)>;
using NodeEndpointsResult = vernon::Result<std::vector<NodeEndpoint>, std::string>;
using PipelineEndpointsResult = vernon::Result<std::vector<PipelineEndpoint>, program::Diagnostic>;

NodeEndpointsResult classifyNodeEndpoints(const program::Program &program, const program::ResolvedNodePlan &node,
                                          const BindableSlot &bindable);

// Endpoints of pipeline 0, the pass Node's own Stage.
std::vector<PipelineEndpoint> passPipelineEndpoints(const std::vector<NodeEndpoint> &pass);

// Conformance rules 1 and 2 of specs/program/recorded_draws.md section 4.
PipelineEndpointsResult
conformPipeline(const program::GraphicsOperation &passGraphics, const program::ResolvedNodePlan &pass,
                const std::vector<NodeEndpoint> &passEndpoints, const program::GraphicsOperation &sourceGraphics,
                const program::ResolvedNodePlan &source, const std::vector<NodeEndpoint> &sourceEndpoints);

} // namespace vernon::runtime::recorded

#endif
