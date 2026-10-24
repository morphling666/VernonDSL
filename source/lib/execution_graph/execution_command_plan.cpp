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

bool validHandle(const RhiCommandResourceBinding &binding) {
    return binding.kind == ResourceKind::Buffer ? binding.buffer.index != VERNON_RHI_INVALID_HANDLE_INDEX
                                                : binding.image.index != VERNON_RHI_INVALID_HANDLE_INDEX;
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

const RhiCommandResourceBinding *ValidatedRhiCommandPlan::binding(const CommandAccessDomain &domain) const {
    const auto found = bindings_.find(domain);
    return found == bindings_.end() ? nullptr : &plan_.bindings[found->second];
}

bool ValidatedRhiCommandPlan::append(RhiCommandExecutionPlan fragment, std::string &error) {
    if (fragment.commands.nodes.empty())
        return true;
    if (fragment.encoders.size() != fragment.commands.nodes.size()) {
        error = "RHI command plan encoder count does not match its command DAG";
        return false;
    }
    if (std::any_of(fragment.encoders.begin(), fragment.encoders.end(),
                    [](const RhiCommandNodeEncoder &encoder) { return !encoder.encode; })) {
        error = "RHI command plan has an invalid node encoder";
        return false;
    }
    if (std::any_of(fragment.finishers.begin(), fragment.finishers.end(),
                    [](const RhiCommandFinisher &finisher) { return !finisher.finish; })) {
        error = "RHI command plan has an invalid finisher";
        return false;
    }
    // A fragment's accesses must name its own bindings, not only bindings the plan already holds.
    CommandAccessDomainMap<uint32_t> fragmentBindings;
    fragmentBindings.reserve(fragment.bindings.size());
    for (uint32_t index = 0; index < fragment.bindings.size(); ++index) {
        const RhiCommandResourceBinding &binding = fragment.bindings[index];
        if (!validHandle(binding)) {
            error = "RHI command plan contains an invalid physical resource binding";
            return false;
        }
        const CommandAccessDomain domain{binding.aliasDomain, binding.kind};
        const auto [existing, inserted] = fragmentBindings.try_emplace(domain, index);
        if (!inserted) {
            error = sameBinding(fragment.bindings[existing->second], binding)
                        ? "RHI command plan contains a duplicate resource binding"
                        : "RHI command plan maps one alias domain to different physical resources";
            return false;
        }
        if (const RhiCommandResourceBinding *bound = this->binding(domain); bound && !sameBinding(*bound, binding)) {
            error = "combined RHI command plan maps one alias domain to different physical resources";
            return false;
        }
    }
    const auto bound = [&](const CommandResourceAccess &access) {
        return fragmentBindings.count(commandAccessDomain(access)) != 0;
    };
    std::vector<CommandResourceAccess> initialAccesses;
    initialAccesses.reserve(fragment.initialAccesses.size());
    for (const CommandResourceAccess &initial : fragment.initialAccesses) {
        CommandResourceAccess normalized = initial;
        if (!normalizeCommandAccess(normalized, error))
            return false;
        if (!bound(initial)) {
            error = "RHI command plan has an unbound initial resource access";
            return false;
        }
        if (dag_.overlapsAdded(initial))
            continue;
        bool duplicate = false;
        if (const auto existing = initialAccesses_.find(commandAccessDomain(initial));
            existing != initialAccesses_.end())
            for (uint32_t index : existing->second) {
                const CommandResourceAccess &current = plan_.initialAccesses[index];
                if (!commandAccessesOverlap(current, initial))
                    continue;
                if (!sameAccessRegion(current, initial) || current.state != initial.state ||
                    current.access != initial.access || current.stageMask != initial.stageMask) {
                    error = "combined RHI command plan has overlapping initial resource states";
                    return false;
                }
                duplicate = true;
            }
        if (!duplicate)
            initialAccesses.push_back(initial);
    }
    const uint32_t nodeBase = dag_.size();
    if (fragment.commands.nodes.size() > std::numeric_limits<uint32_t>::max() - nodeBase) {
        error = "combined RHI command plan exceeds the command index range";
        return false;
    }
    uint32_t resourceBase = resourceBase_;
    for (CommandNode &node : fragment.commands.nodes) {
        const bool root = node.predecessors.empty();
        for (uint32_t &predecessor : node.predecessors)
            predecessor += nodeBase;
        if (nodeBase && root)
            node.predecessors.push_back(nodeBase - 1);
        for (CommandResourceAccess &access : node.accesses) {
            if (!bound(access))
                error = "RHI command plan has an unbound resource access";
            else if (access.resource >= std::numeric_limits<uint32_t>::max() - resourceBase_)
                error = "combined RHI command plan resource index overflows";
            else {
                access.resource += resourceBase_;
                resourceBase = std::max(resourceBase, access.resource + 1);
                continue;
            }
            dag_.truncate(nodeBase);
            return false;
        }
        if (!dag_.add(node, error)) {
            dag_.truncate(nodeBase);
            return false;
        }
    }
    resourceBase_ = resourceBase;
    for (CommandResourceAccess &initial : initialAccesses) {
        initialAccesses_[commandAccessDomain(initial)].push_back(static_cast<uint32_t>(plan_.initialAccesses.size()));
        plan_.initialAccesses.push_back(std::move(initial));
    }
    for (RhiCommandResourceBinding &binding : fragment.bindings)
        if (bindings_.try_emplace({binding.aliasDomain, binding.kind}, static_cast<uint32_t>(plan_.bindings.size()))
                .second)
            plan_.bindings.push_back(std::move(binding));
    plan_.commands.nodes.insert(plan_.commands.nodes.end(), std::make_move_iterator(fragment.commands.nodes.begin()),
                                std::make_move_iterator(fragment.commands.nodes.end()));
    plan_.encoders.insert(plan_.encoders.end(), std::make_move_iterator(fragment.encoders.begin()),
                          std::make_move_iterator(fragment.encoders.end()));
    plan_.retainedContexts.insert(plan_.retainedContexts.end(),
                                  std::make_move_iterator(fragment.retainedContexts.begin()),
                                  std::make_move_iterator(fragment.retainedContexts.end()));
    for (const RhiCommandFinisher &finisher : fragment.finishers)
        if (std::none_of(plan_.finishers.begin(), plan_.finishers.end(), [&](const RhiCommandFinisher &existing) {
                return existing.finish == finisher.finish && existing.context == finisher.context;
            }))
            plan_.finishers.push_back(finisher);
    return true;
}

} // namespace vernon::execution::detail
