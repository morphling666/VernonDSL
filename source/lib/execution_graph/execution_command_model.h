#ifndef VERNON_EXECUTION_COMMAND_MODEL_H
#define VERNON_EXECUTION_COMMAND_MODEL_H

#include "VernonExecutionGraph.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vernon::execution::detail {

enum class CommandNodeKind : uint8_t {
    Compute,
    Render,
    Transfer,
    Checkpoint,
    Replay,
    Status,
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

bool normalizeCommandAccess(CommandResourceAccess &access, std::string &error);
bool commandAccessesOverlap(const CommandResourceAccess &left, const CommandResourceAccess &right);
bool commandAccessesConflict(const CommandResourceAccess &left, const CommandResourceAccess &right);
bool buildCommandDag(const std::vector<ExecutionResourceRecord> &resources,
                     const std::vector<std::unique_ptr<ExecutionPass>> &passes,
                     const std::vector<CompiledScope> &scopes, CommandDag &dag, std::string &error);
bool buildCommandBarriers(const std::vector<ExecutionResourceRecord> &resources, const CommandDag &dag,
                          std::vector<std::vector<VernonRhiBarrier>> &barriers, std::string &error);
bool validateCommandDag(const CommandDag &dag, std::string &error);

} // namespace vernon::execution::detail

#endif
