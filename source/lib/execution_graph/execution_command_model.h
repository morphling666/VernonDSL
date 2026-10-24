#ifndef VERNON_EXECUTION_COMMAND_MODEL_H
#define VERNON_EXECUTION_COMMAND_MODEL_H

#include "execution_graph/command_graph.h"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace vernon::execution::detail {

enum class CommandNodeKind : uint8_t {
    Compute,
    Render,
    Transfer,
    Checkpoint,
    Replay,
    Derivative,
};

enum class CommandQueueClass : uint8_t { Ordered, Compute, Graphics, Transfer };

struct CommandResourceAccess {
    uint32_t resource{};
    uint64_t aliasDomain{};
    uint32_t version{};
    AccessMode access{AccessMode::Read};
    ResourceKind kind{ResourceKind::Buffer};
    GraphByteRange bufferRange{0, UINT64_MAX};
    VernonRhiImageSubresourceRange imageSubresources{0, UINT32_MAX, 0, UINT32_MAX, VERNON_RHI_IMAGE_ASPECT_COLOR};
    VernonRhiResourceState state{VERNON_RHI_STATE_COMMON};
    uint32_t stageMask{};
    uint32_t rhiAccessMask{};
};

struct CommandNode {
    CommandNodeKind kind{CommandNodeKind::Compute};
    CommandQueueClass queue{CommandQueueClass::Ordered};
    std::vector<uint32_t> scopeIndices;
    std::vector<uint32_t> predecessors;
    std::vector<CommandResourceAccess> accesses;
};

struct CommandDag {
    std::vector<CommandNode> nodes;
};

// Only accesses of one domain can overlap.
struct CommandAccessDomain {
    uint64_t aliasDomain{};
    ResourceKind kind{ResourceKind::Buffer};

    bool operator==(const CommandAccessDomain &other) const {
        return aliasDomain == other.aliasDomain && kind == other.kind;
    }
};

struct CommandAccessDomainHash {
    size_t operator()(const CommandAccessDomain &domain) const {
        return std::hash<uint64_t>{}(domain.aliasDomain ^ (static_cast<uint64_t>(domain.kind) << 63));
    }
};

template <typename Value>
using CommandAccessDomainMap = std::unordered_map<CommandAccessDomain, Value, CommandAccessDomainHash>;

inline CommandAccessDomain commandAccessDomain(const CommandResourceAccess &access) {
    return {access.aliasDomain, access.kind};
}

bool normalizeCommandAccess(CommandResourceAccess &access, std::string &error);
uint32_t commandAccessMask(const CommandResourceAccess &access);
bool commandAccessesOverlap(const CommandResourceAccess &left, const CommandResourceAccess &right);
bool commandAccessesConflict(const CommandResourceAccess &left, const CommandResourceAccess &right);

// Validates a command DAG node by node as it grows, comparing each access only with earlier accesses of its domain.
class CommandDagValidator {
public:
    // Validates `node` as the next node; a rejected node leaves the validator unchanged.
    bool add(const CommandNode &node, std::string &error);
    // Drops the nodes from `count` on.
    void truncate(uint32_t count);
    // Whether `access` overlaps an access of an added node.
    bool overlapsAdded(const CommandResourceAccess &access) const;
    uint32_t size() const { return static_cast<uint32_t>(ancestors_.size()); }

private:
    struct Added {
        uint32_t node{};
        CommandResourceAccess access;
    };

    // Per node, a bit per earlier node that precedes it.
    std::vector<std::vector<uint64_t>> ancestors_;
    CommandAccessDomainMap<std::vector<Added>> accesses_;
    // The domain of every added access, in insertion order, so truncation pops each domain's newest entries.
    std::vector<CommandAccessDomain> insertions_;
};

bool validateCommandDag(const CommandDag &dag, std::string &error);

// Sets the barrier's native buffer or image for `access`, or fails when the access names no native resource.
using CommandBarrierTarget = bool (*)(const void *context, const CommandResourceAccess &access,
                                      VernonRhiBarrier &barrier, std::string &error);

// Per node, the barriers from the last overlapping accesses (or `initialAccesses`) to its first accesses of each
// region; `dag` must be valid.
bool buildCommandBarriers(const CommandDag &dag, const std::vector<CommandResourceAccess> &initialAccesses,
                          CommandBarrierTarget target, const void *context,
                          std::vector<std::vector<VernonRhiBarrier>> &barriers, std::string &error);

bool buildCommandDag(const std::vector<ExecutionResourceRecord> &resources,
                     const std::vector<std::unique_ptr<ExecutionPass>> &passes,
                     const std::vector<CompiledScope> &scopes, CommandDag &dag, std::string &error);

} // namespace vernon::execution::detail

#endif
