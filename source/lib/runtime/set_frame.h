#ifndef VERNON_RUNTIME_SET_FRAME_H
#define VERNON_RUNTIME_SET_FRAME_H

#include "runtime/set_values.h"

#include <string>
#include <vector>

namespace vernon::runtime {

struct MaterializedNodeFrame {
    struct HostCopy {
        const uint8_t *source{};
        uint8_t *destination{};
        size_t sourceOffset{};
        size_t destinationOffset{};
        size_t size{};
    };

    std::vector<VernonProgramArgument> arguments;
    // Per argument, whether it is its Value's bound argument lowered by the plan alone, with no carrier, copy or
    // dispatch-derived extent.
    std::vector<char> borrowed;
    std::vector<std::vector<uint64_t>> shapes;
    std::vector<std::vector<int64_t>> strides;
    std::vector<std::vector<uint8_t>> hostEndpointCarriers;
    std::vector<HostCopy> copiesBefore;
    std::vector<HostCopy> copiesAfter;
    size_t hostEndpointCarrierCount{};

    void reset(size_t endpointCount);
    std::vector<uint8_t> &appendHostEndpointCarrier(size_t byteSize);
};

// Lowers the set's Values to the Stage arguments of `node`. Stage parameters whose entry in `materializedParameters`
// is zero get an empty placeholder argument carrying only their slot, so arguments stay index-aligned with the Stage
// parameters. A device Value the Stage reads in another layout is rejected: lowering it needs a copy on the device.
bool materializeNodeFrame(const SetValues &values, const program::Program &program, const program::Node &node,
                          const program::ResolvedNodePlan &nodePlan, MaterializedNodeFrame &output, std::string &error,
                          const std::vector<char> &materializedParameters);

} // namespace vernon::runtime

#endif
