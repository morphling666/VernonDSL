#include "execution_command_model.h"

#include <algorithm>
#include <limits>

namespace vernon::execution::detail {
namespace {

bool writes(AccessMode access) { return access != AccessMode::Read; }

bool finiteIntervalOverlaps(uint64_t leftOffset, uint64_t leftSize, uint64_t rightOffset, uint64_t rightSize) {
    const uint64_t leftEnd = leftSize == UINT64_MAX ? UINT64_MAX : leftOffset + leftSize;
    const uint64_t rightEnd = rightSize == UINT64_MAX ? UINT64_MAX : rightOffset + rightSize;
    return leftOffset < rightEnd && rightOffset < leftEnd;
}

bool imageIntervalOverlaps(uint32_t leftBase, uint32_t leftCount, uint32_t rightBase, uint32_t rightCount) {
    const uint64_t leftEnd = leftCount == UINT32_MAX ? UINT64_MAX : uint64_t{leftBase} + leftCount;
    const uint64_t rightEnd = rightCount == UINT32_MAX ? UINT64_MAX : uint64_t{rightBase} + rightCount;
    return uint64_t{leftBase} < rightEnd && uint64_t{rightBase} < leftEnd;
}

uint32_t accessBits(const CommandResourceAccess &access) {
    const bool reads = access.access != AccessMode::Write;
    const bool writesResource = access.access != AccessMode::Read;
    if (access.state == VERNON_RHI_STATE_COLOR_ATTACHMENT)
        return (reads ? VERNON_RHI_ACCESS_COLOR_READ : 0) | (writesResource ? VERNON_RHI_ACCESS_COLOR_WRITE : 0);
    if (access.state == VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT)
        return (reads ? VERNON_RHI_ACCESS_DEPTH_STENCIL_READ : 0) |
               (writesResource ? VERNON_RHI_ACCESS_DEPTH_STENCIL_WRITE : 0);
    if (access.state == VERNON_RHI_STATE_TRANSFER_SOURCE)
        return VERNON_RHI_ACCESS_TRANSFER_READ;
    if (access.state == VERNON_RHI_STATE_TRANSFER_DESTINATION)
        return VERNON_RHI_ACCESS_TRANSFER_WRITE;
    if (access.state == VERNON_RHI_STATE_SHADER_READ || access.state == VERNON_RHI_STATE_SHADER_WRITE)
        return (reads ? VERNON_RHI_ACCESS_SHADER_READ : 0) | (writesResource ? VERNON_RHI_ACCESS_SHADER_WRITE : 0);
    return VERNON_RHI_ACCESS_NONE;
}

VernonRhiImageSubresourceRange intersection(const VernonRhiImageSubresourceRange &left,
                                            const VernonRhiImageSubresourceRange &right) {
    const uint32_t baseMip = std::max(left.base_mip_level, right.base_mip_level);
    const uint32_t baseLayer = std::max(left.base_array_layer, right.base_array_layer);
    const uint64_t leftMipEnd =
        left.mip_level_count == UINT32_MAX ? UINT64_MAX : uint64_t{left.base_mip_level} + left.mip_level_count;
    const uint64_t rightMipEnd =
        right.mip_level_count == UINT32_MAX ? UINT64_MAX : uint64_t{right.base_mip_level} + right.mip_level_count;
    const uint64_t leftLayerEnd =
        left.array_layer_count == UINT32_MAX ? UINT64_MAX : uint64_t{left.base_array_layer} + left.array_layer_count;
    const uint64_t rightLayerEnd =
        right.array_layer_count == UINT32_MAX ? UINT64_MAX : uint64_t{right.base_array_layer} + right.array_layer_count;
    const uint64_t mipEnd = std::min(leftMipEnd, rightMipEnd);
    const uint64_t layerEnd = std::min(leftLayerEnd, rightLayerEnd);
    return {baseMip, mipEnd == UINT64_MAX ? UINT32_MAX : static_cast<uint32_t>(mipEnd - baseMip), baseLayer,
            layerEnd == UINT64_MAX ? UINT32_MAX : static_cast<uint32_t>(layerEnd - baseLayer),
            left.aspects & right.aspects};
}

} // namespace

bool normalizeCommandAccess(CommandResourceAccess &access, std::string &error) {
    if (!access.aliasDomain) {
        error = "command resource access has no alias domain";
        return false;
    }
    if (access.kind == ResourceKind::Buffer) {
        if (!access.bufferRange.byteSize) {
            error = "command buffer access has an empty byte range";
            return false;
        }
        if (access.bufferRange.byteSize != UINT64_MAX &&
            access.bufferRange.offset > std::numeric_limits<uint64_t>::max() - access.bufferRange.byteSize) {
            error = "command buffer access byte range overflows";
            return false;
        }
        access.imageSubresources = {};
        return true;
    }
    const auto &range = access.imageSubresources;
    if (!range.aspects || !range.mip_level_count || !range.array_layer_count) {
        error = "command image access has an empty subresource range";
        return false;
    }
    if ((range.mip_level_count != UINT32_MAX &&
         uint64_t{range.base_mip_level} + range.mip_level_count > uint64_t{UINT32_MAX} + 1) ||
        (range.array_layer_count != UINT32_MAX &&
         uint64_t{range.base_array_layer} + range.array_layer_count > uint64_t{UINT32_MAX} + 1)) {
        error = "command image access subresource range overflows";
        return false;
    }
    access.bufferRange = {};
    return true;
}

bool commandAccessesOverlap(const CommandResourceAccess &left, const CommandResourceAccess &right) {
    if (left.aliasDomain != right.aliasDomain || left.kind != right.kind)
        return false;
    if (left.kind == ResourceKind::Buffer)
        return finiteIntervalOverlaps(left.bufferRange.offset, left.bufferRange.byteSize, right.bufferRange.offset,
                                      right.bufferRange.byteSize);
    return (left.imageSubresources.aspects & right.imageSubresources.aspects) != 0 &&
           imageIntervalOverlaps(left.imageSubresources.base_mip_level, left.imageSubresources.mip_level_count,
                                 right.imageSubresources.base_mip_level, right.imageSubresources.mip_level_count) &&
           imageIntervalOverlaps(left.imageSubresources.base_array_layer, left.imageSubresources.array_layer_count,
                                 right.imageSubresources.base_array_layer, right.imageSubresources.array_layer_count);
}

bool commandAccessesConflict(const CommandResourceAccess &left, const CommandResourceAccess &right) {
    return commandAccessesOverlap(left, right) && (writes(left.access) || writes(right.access));
}

bool buildCommandDag(const std::vector<ExecutionResourceRecord> &resources,
                     const std::vector<std::unique_ptr<ExecutionPass>> &passes,
                     const std::vector<CompiledScope> &scopes, CommandDag &dag, std::string &error) {
    dag = {};
    std::vector<uint32_t> versions(resources.size());
    dag.nodes.reserve(scopes.size());
    for (uint32_t scopeIndex = 0; scopeIndex < scopes.size(); ++scopeIndex) {
        const CompiledScope &scope = scopes[scopeIndex];
        if (scope.passIndices.empty()) {
            error = "compiled scope has no passes";
            return false;
        }
        if (scope.passIndices.front() >= passes.size()) {
            error = "compiled scope refers to an invalid pass";
            return false;
        }
        const bool derivative = !scope.rendering && (passes[scope.passIndices.front()]->flags() & PassDerivative);
        if (derivative && scope.passIndices.size() != 1) {
            error = "derivative command scope must contain exactly one pass";
            return false;
        }
        CommandNode node;
        node.kind = scope.rendering ? CommandNodeKind::Render
                    : derivative    ? CommandNodeKind::Derivative
                                    : CommandNodeKind::Compute;
        node.queue = scope.rendering ? CommandQueueClass::Graphics : CommandQueueClass::Compute;
        node.scopeIndices.push_back(scopeIndex);
        node.predecessors = scope.predecessors;
        std::sort(node.predecessors.begin(), node.predecessors.end());
        node.predecessors.erase(std::unique(node.predecessors.begin(), node.predecessors.end()),
                                node.predecessors.end());
        std::vector<bool> written(resources.size());
        for (uint32_t passIndex : scope.passIndices) {
            if (passIndex >= passes.size()) {
                error = "compiled scope refers to an invalid pass";
                return false;
            }
            for (const ResourceUse &use : passes[passIndex]->uses()) {
                if (use.resource.id >= resources.size()) {
                    error = "command access refers to an invalid resource";
                    return false;
                }
                if (writes(use.access) && !written[use.resource.id]) {
                    if (versions[use.resource.id] == UINT32_MAX) {
                        error = "command resource version overflows";
                        return false;
                    }
                    ++versions[use.resource.id];
                    written[use.resource.id] = true;
                }
                const ExecutionResourceRecord &record = resources[use.resource.id];
                CommandResourceAccess access;
                access.resource = use.resource.id;
                access.aliasDomain = record.resourceKey ? record.resourceKey : uint64_t{use.resource.id} + 1;
                access.version = versions[use.resource.id];
                access.access = use.access;
                access.kind = use.resource.kind;
                access.state = use.state;
                access.stageMask = use.stageMask;
                if (!access.stageMask &&
                    (access.state == VERNON_RHI_STATE_SHADER_READ || access.state == VERNON_RHI_STATE_SHADER_WRITE))
                    access.stageMask = scope.rendering ? VERNON_RHI_STAGE_VERTEX | VERNON_RHI_STAGE_FRAGMENT
                                                       : VERNON_RHI_STAGE_COMPUTE;
                if (use.resource.kind == ResourceKind::Image)
                    access.imageSubresources = use.imageSubresources;
                if (!normalizeCommandAccess(access, error))
                    return false;
                node.accesses.push_back(access);
            }
        }
        dag.nodes.push_back(std::move(node));
    }
    return validateCommandDag(dag, error);
}

bool buildCommandBarriers(const std::vector<ExecutionResourceRecord> &resources, const CommandDag &dag,
                          std::vector<std::vector<VernonRhiBarrier>> &barriers, std::string &error) {
    barriers.assign(dag.nodes.size(), {});
    std::vector<CommandResourceAccess> lastAccesses;
    for (uint32_t nodeIndex = 0; nodeIndex < dag.nodes.size(); ++nodeIndex) {
        const CommandNode &node = dag.nodes[nodeIndex];
        std::vector<CommandResourceAccess> firstAccesses;
        std::vector<CommandResourceAccess> finalAccesses;
        for (const CommandResourceAccess &access : node.accesses) {
            if (std::none_of(finalAccesses.begin(), finalAccesses.end(), [&](const CommandResourceAccess &current) {
                    return commandAccessesOverlap(current, access);
                }))
                firstAccesses.push_back(access);
            finalAccesses.erase(std::remove_if(finalAccesses.begin(), finalAccesses.end(),
                                               [&](const CommandResourceAccess &current) {
                                                   return commandAccessesOverlap(current, access);
                                               }),
                                finalAccesses.end());
            finalAccesses.push_back(access);
        }
        std::vector<const CommandResourceAccess *> orderedAccesses;
        orderedAccesses.reserve(firstAccesses.size());
        for (const CommandResourceAccess &access : firstAccesses)
            orderedAccesses.push_back(&access);
        std::stable_sort(orderedAccesses.begin(), orderedAccesses.end(),
                         [](const CommandResourceAccess *left, const CommandResourceAccess *right) {
                             return left->resource < right->resource;
                         });
        for (const CommandResourceAccess *accessPointer : orderedAccesses) {
            const CommandResourceAccess &access = *accessPointer;
            if (access.resource >= resources.size()) {
                error = "command barrier access refers to an invalid resource";
                return false;
            }
            const ExecutionResourceRecord &record = resources[access.resource];
            if (record.resource.kind != access.kind) {
                error = "command barrier access kind does not match its resource";
                return false;
            }
            for (const CommandResourceAccess &previous : lastAccesses) {
                if (!commandAccessesOverlap(previous, access) ||
                    (previous.state == access.state && !writes(previous.access) && !writes(access.access)))
                    continue;
                VernonRhiBarrier barrier{};
                barrier.struct_size = sizeof(barrier);
                barrier.source_stage_mask = previous.stageMask;
                barrier.destination_stage_mask = access.stageMask;
                barrier.source_access = accessBits(previous);
                barrier.destination_access = accessBits(access);
                barrier.old_state = previous.state;
                barrier.new_state = access.state;
                barrier.is_image = access.kind == ResourceKind::Image;
                if (barrier.is_image) {
                    barrier.image = record.image;
                    barrier.image_subresources = intersection(previous.imageSubresources, access.imageSubresources);
                } else {
                    barrier.buffer = record.buffer;
                }
                barriers[nodeIndex].push_back(barrier);
            }
        }
        for (const CommandResourceAccess &access : finalAccesses) {
            lastAccesses.erase(std::remove_if(lastAccesses.begin(), lastAccesses.end(),
                                              [&](const CommandResourceAccess &previous) {
                                                  return commandAccessesOverlap(previous, access);
                                              }),
                               lastAccesses.end());
            lastAccesses.push_back(access);
        }
    }
    return true;
}

bool validateCommandDag(const CommandDag &dag, std::string &error) {
    std::vector<std::vector<bool>> ancestors(dag.nodes.size(), std::vector<bool>(dag.nodes.size()));
    for (uint32_t index = 0; index < dag.nodes.size(); ++index) {
        const CommandNode &node = dag.nodes[index];
        if ((node.kind == CommandNodeKind::Compute || node.kind == CommandNodeKind::Render) &&
            node.scopeIndices.empty()) {
            error = "command node has no compiled scope";
            return false;
        }
        uint32_t previous = UINT32_MAX;
        for (uint32_t predecessor : node.predecessors) {
            if (predecessor >= index) {
                error = "command DAG is not topologically ordered";
                return false;
            }
            if (previous != UINT32_MAX && predecessor <= previous) {
                error = "command node predecessors are not unique and ordered";
                return false;
            }
            previous = predecessor;
            ancestors[index][predecessor] = true;
            for (uint32_t ancestor = 0; ancestor < index; ++ancestor)
                ancestors[index][ancestor] = ancestors[index][ancestor] || ancestors[predecessor][ancestor];
        }
        for (CommandResourceAccess access : node.accesses)
            if (!normalizeCommandAccess(access, error))
                return false;
        for (uint32_t earlier = 0; earlier < index; ++earlier)
            for (const CommandResourceAccess &left : dag.nodes[earlier].accesses)
                for (const CommandResourceAccess &right : node.accesses)
                    if (commandAccessesConflict(left, right) && !ancestors[index][earlier]) {
                        error = "conflicting command resource accesses have no dependency";
                        return false;
                    }
    }
    return true;
}

} // namespace vernon::execution::detail
