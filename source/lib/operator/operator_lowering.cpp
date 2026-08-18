#include "operator/operator_lowering.h"

#include "operator/operator_validation.h"

#include <algorithm>
#include <limits>

namespace vernon::ops {
namespace {

bool appendAccess(const TensorViewDescriptor &view, uint32_t resource, execution::AccessMode access,
                  execution::detail::CommandNode &command, std::string &error) {
    uint64_t begin = 0;
    uint64_t end = 0;
    if (!tensorViewFootprint(view, begin, end)) {
        error = "cannot lower an invalid operator tensor footprint";
        return false;
    }
    execution::detail::CommandResourceAccess result;
    result.resource = resource;
    result.aliasDomain = view.ownerIdentity;
    result.access = access;
    result.kind = execution::ResourceKind::Buffer;
    result.bufferRange = {begin, end - begin};
    result.state = access == execution::AccessMode::Read ? VERNON_RHI_STATE_SHADER_READ : VERNON_RHI_STATE_SHADER_WRITE;
    result.stageMask = VERNON_RHI_STAGE_COMPUTE;
    command.accesses.push_back(result);
    return true;
}

} // namespace

bool lowerOperatorDagToCommandDag(const OperatorDag &operators, execution::detail::CommandDag &commands,
                                  std::string &error) {
    if (!validateOperatorDag(operators, error))
        return false;
    commands = {};
    const uint32_t invalid = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> commandIndices(operators.nodes().size(), invalid);
    for (uint32_t index = 0; index < operators.nodes().size(); ++index) {
        const OperatorNode &node = operators.nodes()[index];
        if (node.kind == OperatorKind::Leaf)
            continue;
        execution::detail::CommandNode command;
        command.kind = execution::detail::CommandNodeKind::Derivative;
        command.queue = execution::detail::CommandQueueClass::Compute;
        for (uint32_t input : node.inputs) {
            if (!appendAccess(operators.nodes()[input].output, input, execution::AccessMode::Read, command, error))
                return false;
            if (commandIndices[input] != invalid)
                command.predecessors.push_back(commandIndices[input]);
        }
        if (!appendAccess(node.output, index, execution::AccessMode::Write, command, error))
            return false;
        std::sort(command.predecessors.begin(), command.predecessors.end());
        command.predecessors.erase(std::unique(command.predecessors.begin(), command.predecessors.end()),
                                   command.predecessors.end());
        commandIndices[index] = static_cast<uint32_t>(commands.nodes.size());
        commands.nodes.push_back(std::move(command));
    }
    return true;
}

} // namespace vernon::ops
