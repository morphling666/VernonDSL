#include "recorded_graphics.h"

#include <algorithm>

namespace vernon::runtime::recorded {
namespace {

PipelineEndpointsResult nonconforming(std::string path, std::string message) {
    return PipelineEndpointsResult{vernon::err(
        program::Diagnostic{kPipelineConformance, kPipelineCreationPhase, std::move(path), std::move(message)})};
}

NodeEndpointsResult unclassifiable(std::string message) { return NodeEndpointsResult{vernon::err(std::move(message))}; }

std::string endpointPath(const EndpointKey &key) { return key.module + ":" + key.sourceName; }

bool writtenByAnyNode(const program::Program &program, uint32_t value) {
    const std::optional<uint32_t> storage = program.values[value].storage;
    for (const program::Graph &graph : program.graphs)
        for (const program::Node &node : graph.nodes) {
            if (std::find(node.results.begin(), node.results.end(), value) != node.results.end())
                return true;
            for (const program::ResourceAccess &access : node.accesses)
                if (access.kind != program::AccessKind::Read && storage && access.storage == *storage)
                    return true;
            for (const program::EndpointBinding &binding : node.bindings)
                for (const program::ValueEndpointProjection &projection : binding.projections)
                    if (projection.direction == program::ValueBindingDirection::Result && projection.value == value)
                        return true;
        }
    return false;
}

bool sameLayout(const ValueLayout &left, const ValueLayout &right) {
    return left.layoutHash == right.layoutHash && left.byteSize == right.byteSize &&
           left.alignment == right.alignment && left.leaves.size() == right.leaves.size();
}

bool sameProjection(const std::optional<program::PhysicalEndpointProjection> &left,
                    const std::optional<program::PhysicalEndpointProjection> &right) {
    if (left.has_value() != right.has_value())
        return false;
    return !left ||
           (left->carrierByteSize == right->carrierByteSize && left->carrierAlignment == right->carrierAlignment &&
            left->leafByteOffset == right->leafByteOffset && left->leafByteSize == right->leafByteSize);
}

bool sameAttachment(const program::GraphicsAttachmentSignature &left,
                    const program::GraphicsAttachmentSignature &right) {
    return left.location == right.location && left.formats == right.formats &&
           left.sampleCounts == right.sampleCounts && left.aspects == right.aspects;
}

bool sameEndpointReflection(const Parameter &left, const program::TargetBinding &leftTarget, const Parameter &right,
                            const program::TargetBinding &rightTarget) {
    if (left.kind != right.kind || left.tensorArgument != right.tensorArgument || left.access != right.access ||
        left.addressSpace != right.addressSpace || left.dimension != right.dimension ||
        left.bindingRole != right.bindingRole || left.sampleResultClass != right.sampleResultClass ||
        left.exactStorageFormat != right.exactStorageFormat || left.shape != right.shape ||
        !sameLayout(left.elementLayout, right.elementLayout) ||
        left.valueLayout.has_value() != right.valueLayout.has_value() ||
        (left.valueLayout && !sameLayout(*left.valueLayout, *right.valueLayout)))
        return false;
    return leftTarget.carrier == rightTarget.carrier && leftTarget.source == rightTarget.source &&
           leftTarget.semantic == rightTarget.semantic && leftTarget.kind == rightTarget.kind &&
           leftTarget.reflectedKind == rightTarget.reflectedKind && leftTarget.role == rightTarget.role &&
           leftTarget.access == rightTarget.access && leftTarget.dimension == rightTarget.dimension &&
           leftTarget.imageFormat == rightTarget.imageFormat &&
           leftTarget.imageSampleResultClass == rightTarget.imageSampleResultClass &&
           sameProjection(leftTarget.endpointProjection, rightTarget.endpointProjection);
}

bool sameAttachmentSignature(const program::GraphicsOperation &left, const program::GraphicsOperation &right) {
    if (left.colorAttachments.size() != right.colorAttachments.size() ||
        left.depthStencilAttachment.has_value() != right.depthStencilAttachment.has_value())
        return false;
    for (size_t index = 0; index < left.colorAttachments.size(); ++index)
        if (!sameAttachment(left.colorAttachments[index], right.colorAttachments[index]))
            return false;
    return !left.depthStencilAttachment || sameAttachment(*left.depthStencilAttachment, *right.depthStencilAttachment);
}

} // namespace

NodeEndpointsResult classifyNodeEndpoints(const program::Program &program, const program::ResolvedNodePlan &node,
                                          const BindableSlot &bindable) {
    std::vector<NodeEndpoint> endpoints;
    endpoints.reserve(node.projections.size());
    for (const program::NodeEndpointProjection &projection : node.projections) {
        if (projection.value >= program.values.size())
            return unclassifiable("graphics Node endpoint exceeds its Program Values");
        NodeEndpoint endpoint;
        endpoint.key = {projection.target.endpoint.module, projection.target.stageSourceName};
        if (endpoint.key.module.empty() || endpoint.key.sourceName.empty())
            return unclassifiable("graphics Node endpoint has no Stage-local identity");
        if (std::any_of(endpoints.begin(), endpoints.end(),
                        [&](const NodeEndpoint &other) { return other.key == endpoint.key; }))
            return unclassifiable("graphics Node endpoint '" + endpointPath(endpoint.key) + "' is not unique");
        if (projection.target.access == "read" && !writtenByAnyNode(program, projection.value)) {
            const auto boundary = std::find_if(program.abi.boundarySlots.begin(), program.abi.boundarySlots.end(),
                                               [&](const program::BoundarySlot &slot) {
                                                   return slot.role == program::BoundaryRole::Input &&
                                                          slot.value == projection.value && bindable(slot.id);
                                               });
            if (boundary != program.abi.boundarySlots.end())
                endpoint.perDrawSlot = boundary->id;
        }
        endpoints.push_back(std::move(endpoint));
    }
    return NodeEndpointsResult{vernon::ok(std::move(endpoints))};
}

std::vector<PipelineEndpoint> passPipelineEndpoints(const std::vector<NodeEndpoint> &pass) {
    std::vector<PipelineEndpoint> endpoints;
    endpoints.reserve(pass.size());
    for (size_t index = 0; index < pass.size(); ++index) {
        PipelineEndpoint endpoint;
        endpoint.key = pass[index].key;
        if (pass[index].perDrawSlot) {
            endpoint.parameterSlot = pass[index].perDrawSlot;
            endpoint.initialPassParameter = index;
        } else {
            endpoint.fixedPassParameter = index;
        }
        endpoints.push_back(std::move(endpoint));
    }
    return endpoints;
}

PipelineEndpointsResult
conformPipeline(const program::GraphicsOperation &passGraphics, const program::ResolvedNodePlan &pass,
                const std::vector<NodeEndpoint> &passEndpoints, const program::GraphicsOperation &sourceGraphics,
                const program::ResolvedNodePlan &source, const std::vector<NodeEndpoint> &sourceEndpoints) {
    if (!pass.stage || !source.stage || passEndpoints.size() != pass.projections.size() ||
        sourceEndpoints.size() != source.projections.size() ||
        pass.stage->bindingProjection.parameters.size() != pass.projections.size() ||
        source.stage->bindingProjection.parameters.size() != source.projections.size())
        return nonconforming({}, "pipeline or pass Node has no resolved Stage endpoints");
    if (!sameAttachmentSignature(passGraphics, sourceGraphics))
        return nonconforming("attachments", "pipeline attachment signature differs from the pass Node");
    const std::vector<Parameter> &passParameters = pass.stage->bindingProjection.parameters;
    const std::vector<Parameter> &sourceParameters = source.stage->bindingProjection.parameters;
    std::vector<PipelineEndpoint> endpoints;
    endpoints.reserve(sourceEndpoints.size());
    for (size_t index = 0; index < sourceEndpoints.size(); ++index) {
        PipelineEndpoint endpoint;
        endpoint.key = sourceEndpoints[index].key;
        const auto counterpart =
            std::find_if(passEndpoints.begin(), passEndpoints.end(),
                         [&](const NodeEndpoint &candidate) { return candidate.key == endpoint.key; });
        const std::optional<size_t> passIndex =
            counterpart == passEndpoints.end()
                ? std::nullopt
                : std::optional<size_t>(static_cast<size_t>(counterpart - passEndpoints.begin()));
        const bool equalReflection =
            passIndex && sameEndpointReflection(passParameters[*passIndex], pass.projections[*passIndex].target,
                                                sourceParameters[index], source.projections[index].target);
        if (passIndex && !passEndpoints[*passIndex].perDrawSlot) {
            if (!equalReflection)
                return nonconforming(endpointPath(endpoint.key),
                                     "endpoint reflects differently from the pass Node's fixed endpoint");
            endpoint.fixedPassParameter = passIndex;
        } else {
            if (!sourceEndpoints[index].perDrawSlot)
                return nonconforming(
                    endpointPath(endpoint.key),
                    "endpoint is neither fixed by the pass Node nor bound to a parameter of the pipeline");
            endpoint.parameterSlot = sourceEndpoints[index].perDrawSlot;
            if (equalReflection)
                endpoint.initialPassParameter = passIndex;
        }
        endpoints.push_back(std::move(endpoint));
    }
    return PipelineEndpointsResult{vernon::ok(std::move(endpoints))};
}

} // namespace vernon::runtime::recorded
