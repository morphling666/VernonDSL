#include "execution_graph/execution_graph_internal.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace vernon::execution::detail {
namespace {

bool sameBinding(const RhiCommandResourceBinding &left, const RhiCommandResourceBinding &right) {
    if (left.aliasDomain != right.aliasDomain || left.kind != right.kind)
        return false;
    if (left.kind == ResourceKind::Buffer)
        return left.buffer.index == right.buffer.index && left.buffer.generation == right.buffer.generation;
    return left.image.index == right.image.index && left.image.generation == right.image.generation;
}

bool sameAccessRegion(const CommandResourceAccess &left, const CommandResourceAccess &right) {
    if (left.aliasDomain != right.aliasDomain || left.kind != right.kind)
        return false;
    if (left.kind == ResourceKind::Buffer)
        return left.bufferRange.offset == right.bufferRange.offset &&
               left.bufferRange.byteSize == right.bufferRange.byteSize;
    return left.imageSubresources.base_mip_level == right.imageSubresources.base_mip_level &&
           left.imageSubresources.mip_level_count == right.imageSubresources.mip_level_count &&
           left.imageSubresources.base_array_layer == right.imageSubresources.base_array_layer &&
           left.imageSubresources.array_layer_count == right.imageSubresources.array_layer_count &&
           left.imageSubresources.aspects == right.imageSubresources.aspects;
}

} // namespace

bool validateRhiCommandExecutionPlan(const RhiCommandExecutionPlan &plan, std::string &error) {
    if (!validateCommandDag(plan.commands, error))
        return false;
    if (plan.encoders.size() != plan.commands.nodes.size()) {
        error = "RHI command plan encoder count does not match its command DAG";
        return false;
    }
    for (size_t index = 0; index < plan.commands.nodes.size(); ++index) {
        const bool status = plan.commands.nodes[index].kind == CommandNodeKind::Status;
        if ((!status && (!plan.encoders[index].encode || plan.encoders[index].complete)) ||
            (status && (plan.encoders[index].encode || !plan.encoders[index].complete))) {
            error = "RHI command plan has an invalid node encoder";
            return false;
        }
        for (const CommandResourceAccess &access : plan.commands.nodes[index].accesses) {
            const auto binding =
                std::find_if(plan.bindings.begin(), plan.bindings.end(), [&](const RhiCommandResourceBinding &value) {
                    return value.aliasDomain == access.aliasDomain && value.kind == access.kind;
                });
            if (binding == plan.bindings.end()) {
                error = "RHI command plan has an unbound resource access";
                return false;
            }
        }
    }
    for (const CommandResourceAccess &initial : plan.initialAccesses) {
        CommandResourceAccess normalized = initial;
        if (!normalizeCommandAccess(normalized, error))
            return false;
        const auto binding =
            std::find_if(plan.bindings.begin(), plan.bindings.end(), [&](const RhiCommandResourceBinding &value) {
                return value.aliasDomain == initial.aliasDomain && value.kind == initial.kind;
            });
        if (binding == plan.bindings.end()) {
            error = "RHI command plan has an unbound initial resource access";
            return false;
        }
    }
    for (size_t index = 0; index < plan.bindings.size(); ++index) {
        const RhiCommandResourceBinding &binding = plan.bindings[index];
        if ((binding.kind == ResourceKind::Buffer && binding.buffer.index == VERNON_RHI_INVALID_HANDLE_INDEX) ||
            (binding.kind == ResourceKind::Image && binding.image.index == VERNON_RHI_INVALID_HANDLE_INDEX)) {
            error = "RHI command plan contains an invalid physical resource binding";
            return false;
        }
        for (size_t other = index + 1; other < plan.bindings.size(); ++other)
            if (plan.bindings[index].aliasDomain == plan.bindings[other].aliasDomain &&
                plan.bindings[index].kind == plan.bindings[other].kind) {
                error = sameBinding(plan.bindings[index], plan.bindings[other])
                            ? "RHI command plan contains a duplicate resource binding"
                            : "RHI command plan maps one alias domain to different physical resources";
                return false;
            }
    }
    return true;
}

bool appendRhiCommandExecutionPlan(RhiCommandExecutionPlan &destination, RhiCommandExecutionPlan source, bool serialize,
                                   std::string &error) {
    if (source.commands.nodes.empty())
        return true;
    if (!validateRhiCommandExecutionPlan(source, error))
        return false;
    if (!destination.commands.nodes.empty() && !validateRhiCommandExecutionPlan(destination, error))
        return false;
    std::vector<CommandResourceAccess> appendedInitialAccesses;
    appendedInitialAccesses.reserve(source.initialAccesses.size());
    for (const CommandResourceAccess &initial : source.initialAccesses) {
        const bool alreadyUsed =
            std::any_of(destination.commands.nodes.begin(), destination.commands.nodes.end(), [&](const auto &node) {
                return std::any_of(node.accesses.begin(), node.accesses.end(),
                                   [&](const auto &access) { return commandAccessesOverlap(access, initial); });
            });
        if (alreadyUsed)
            continue;
        bool duplicate = false;
        for (const CommandResourceAccess &existing : destination.initialAccesses) {
            if (!commandAccessesOverlap(existing, initial))
                continue;
            if (!sameAccessRegion(existing, initial) || existing.state != initial.state ||
                existing.access != initial.access || existing.stageMask != initial.stageMask) {
                error = "combined RHI command plan has overlapping initial resource states";
                return false;
            }
            duplicate = true;
        }
        if (!duplicate)
            appendedInitialAccesses.push_back(initial);
    }
    if (source.commands.nodes.size() > std::numeric_limits<uint32_t>::max() - destination.commands.nodes.size()) {
        error = "combined RHI command plan exceeds the command index range";
        return false;
    }
    const uint32_t nodeBase = static_cast<uint32_t>(destination.commands.nodes.size());
    uint32_t resourceBase = 0;
    for (const CommandNode &node : destination.commands.nodes)
        for (const CommandResourceAccess &access : node.accesses) {
            if (access.resource == std::numeric_limits<uint32_t>::max()) {
                error = "RHI command plan resource index overflows";
                return false;
            }
            resourceBase = std::max(resourceBase, access.resource + 1);
        }
    for (const RhiCommandResourceBinding &binding : source.bindings) {
        const auto existing = std::find_if(
            destination.bindings.begin(), destination.bindings.end(), [&](const RhiCommandResourceBinding &value) {
                return value.aliasDomain == binding.aliasDomain && value.kind == binding.kind;
            });
        if (existing != destination.bindings.end() && !sameBinding(*existing, binding)) {
            error = "combined RHI command plan maps one alias domain to different physical resources";
            return false;
        }
    }
    for (CommandNode &node : source.commands.nodes) {
        const bool root = node.predecessors.empty();
        for (uint32_t &predecessor : node.predecessors)
            predecessor += nodeBase;
        if (serialize && nodeBase && root)
            node.predecessors.push_back(nodeBase - 1);
        for (CommandResourceAccess &access : node.accesses) {
            if (access.resource > std::numeric_limits<uint32_t>::max() - resourceBase) {
                error = "combined RHI command plan resource index overflows";
                return false;
            }
            access.resource += resourceBase;
        }
    }
    destination.initialAccesses.reserve(destination.initialAccesses.size() + appendedInitialAccesses.size());
    destination.commands.nodes.reserve(destination.commands.nodes.size() + source.commands.nodes.size());
    destination.encoders.reserve(destination.encoders.size() + source.encoders.size());
    destination.bindings.reserve(destination.bindings.size() + source.bindings.size());
    destination.retainedContexts.reserve(destination.retainedContexts.size() + source.retainedContexts.size());
    RhiCommandExecutionPlan result = std::move(destination);
    result.initialAccesses.insert(result.initialAccesses.end(),
                                  std::make_move_iterator(appendedInitialAccesses.begin()),
                                  std::make_move_iterator(appendedInitialAccesses.end()));
    result.commands.nodes.insert(result.commands.nodes.end(), std::make_move_iterator(source.commands.nodes.begin()),
                                 std::make_move_iterator(source.commands.nodes.end()));
    result.encoders.insert(result.encoders.end(), std::make_move_iterator(source.encoders.begin()),
                           std::make_move_iterator(source.encoders.end()));
    for (RhiCommandResourceBinding &binding : source.bindings) {
        const auto existing =
            std::find_if(result.bindings.begin(), result.bindings.end(), [&](const RhiCommandResourceBinding &value) {
                return value.aliasDomain == binding.aliasDomain && value.kind == binding.kind;
            });
        if (existing == result.bindings.end())
            result.bindings.push_back(std::move(binding));
    }
    result.retainedContexts.insert(result.retainedContexts.end(),
                                   std::make_move_iterator(source.retainedContexts.begin()),
                                   std::make_move_iterator(source.retainedContexts.end()));
    destination = std::move(result);
    return true;
}

VernonRhiStatus executeRhiCommandPlanAndWait(VernonRhiDevice device, uint32_t requiredCapabilities,
                                             const RhiCommandExecutionPlan &plan, RhiCommandDagExecutionStats *stats) {
    std::string error;
    if (!validateRhiCommandExecutionPlan(plan, error))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    return executeRhiCommandDagAndWait(device, requiredCapabilities, plan.commands, plan.encoders, stats,
                                       &plan.bindings, plan.initialAccesses.empty() ? nullptr : &plan.initialAccesses);
}

} // namespace vernon::execution::detail
