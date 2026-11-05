#ifndef VERNON_RUNTIME_SCHEDULE_VJP_PLAN_H
#define VERNON_RUNTIME_SCHEDULE_VJP_PLAN_H

// How a linked VJP passes the forward Values its backward graph reads (specs/runtime/architecture_v2.md section
// 16.2): a forward argument the backward schedule reads again, a residual the forward schedule writes into the call's
// residual block, or a CPU autodiff tape the block owns.

#include "VernonResult.hpp"

#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace vernon::schedule {

struct VjpPlan {
    // Forward allocations the backward graph reads; the forward schedule exposes them after its boundary slots.
    std::vector<uint32_t> residuals;
    // Autodiff tapes that are slots (planVjp), which both schedules expose after their other exposed Storages.
    std::vector<uint32_t> tapes;
    // The Storages the backward graph reads, sorted: the backward schedule exposes them after its boundary slots.
    std::vector<uint32_t> captured;
    // Per captured Storage, the forward boundary slot whose argument it is, or UINT32_MAX for a residual.
    std::vector<uint32_t> capturedBoundary;
    // Per captured scalar Value a backward node binds, in capture order, the forward boundary slot that passes it.
    std::vector<uint32_t> scalarBoundaries;
    // Per forward node, in node order, the indices into `residuals` then `tapes` of the Storages whose captured Values
    // it produces, by which a linked VJP reports each node's residuals.
    struct Pass {
        uint32_t node{};
        std::string name;
        std::vector<uint32_t> exposed;
    };
    std::vector<Pass> passes;
};

// The plan of a Program with exactly a "forward" and a "backward" graph. Every tape is a slot when `cpuTapes` (the
// CPU, whose kernels take tapes from an allocator); otherwise a tape a forward kernel binds is a slot (CUDA), and a
// backward kernel may bind any other tape only for launch metadata. Fails for what a linked VJP does not pass.
[[nodiscard]] Result<VjpPlan, std::string> planVjp(const nlohmann::json &program, bool cpuTapes);

} // namespace vernon::schedule

#endif
