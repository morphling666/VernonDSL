#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_MATERIALIZED_NODE_FRAME_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_MATERIALIZED_NODE_FRAME_H

#include "device_commands.h"
#include "program_invocation_state.h"

#include <functional>

namespace vernon::runtime::program_execution {

struct MaterializedNodeFrame {
    struct HostCopy {
        const uint8_t *source{};
        uint8_t *destination{};
        size_t sourceOffset{};
        size_t destinationOffset{};
        size_t size{};
    };

    std::vector<VernonProgramArgument> arguments;
    std::vector<std::vector<uint64_t>> shapes;
    std::vector<std::vector<int64_t>> strides;
    std::vector<std::vector<uint8_t>> hostEndpointCarriers;
    std::vector<std::shared_ptr<void>> deviceEndpointCarriers;
    std::vector<HostCopy> copiesBefore;
    std::vector<HostCopy> copiesAfter;
    std::vector<DeviceBufferCopy> deviceCopiesBefore;
    std::vector<DeviceBufferCopy> deviceCopiesAfter;

    bool prepareHost(std::string &error) const;
    bool commitHost(std::string &error) const;
};

using ResolvePhysicalEndpoint = std::function<const VernonProgramArgument *(uint32_t, const program::TargetBinding &)>;

bool materializeNodeFrame(const ProgramInvocationState &invocation, const program::Program &program,
                          const program::Node &node, const program::ResolvedNodePlan &nodePlan,
                          const ResolvePhysicalEndpoint &resolvePhysicalEndpoint, MaterializedNodeFrame &output,
                          std::string &error);

} // namespace vernon::runtime::program_execution

#endif
