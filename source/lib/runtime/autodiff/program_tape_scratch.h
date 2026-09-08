#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_TAPE_SCRATCH_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_TAPE_SCRATCH_H

#include "host_tape_allocator.h"
#include "runtime/program_execution/device_buffer.h"
#include "runtime/program_execution/materialized_node_frame.h"

#include <array>

namespace vernon::runtime::ad {

struct ProgramTapeCarrierSnapshot {
    std::shared_ptr<const program_execution::DeviceBuffer> owner;
    VernonProgramArgument argument{};
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;
};

struct ProgramTapeSnapshot {
    std::vector<std::shared_ptr<const HostStaticTapeBatch>> hostBatches;
    std::vector<std::array<ProgramTapeCarrierSnapshot, 4>> carriers;
};

class ProgramTapeScratch {
public:
    explicit ProgramTapeScratch(size_t valueCount);

    void setHostBatch(uint32_t value, std::shared_ptr<HostStaticTapeBatch> batch);
    std::shared_ptr<HostStaticTapeBatch> hostBatch(uint32_t value) const;
    std::vector<std::shared_ptr<HostStaticTapeBatch>> releaseHostBatches();
    void importHostBatches(const std::vector<std::shared_ptr<HostStaticTapeBatch>> &batches);
    ProgramTapeSnapshot releaseSnapshot();
    void importSnapshot(const ProgramTapeSnapshot &snapshot);

    bool allocateCarrier(VernonRuntimeContext &context, uint32_t value, const program::TargetBinding &binding,
                         size_t byteSize, std::vector<uint64_t> shape, std::vector<int64_t> strides,
                         std::string &error);
    bool uploadCarrier(uint32_t value, program_plan::TapeCarrier carrier, const void *data, size_t byteSize,
                       std::string &error);
    bool downloadCarrier(uint32_t value, program_plan::TapeCarrier carrier, void *data, size_t byteSize,
                         std::string &error) const;
    const VernonProgramArgument *argument(uint32_t value, const program::TargetBinding &binding) const;

private:
    struct Carrier {
        std::shared_ptr<program_execution::DeviceBuffer> buffer;
        std::shared_ptr<const program_execution::DeviceBuffer> retainedBuffer;
        VernonProgramArgument argument{};
        std::vector<uint64_t> shape;
        std::vector<int64_t> strides;
        bool readOnly{};
    };

    static size_t carrierIndex(program_plan::TapeCarrier carrier);
    Carrier *carrier(uint32_t value, program_plan::TapeCarrier carrier);
    const Carrier *carrier(uint32_t value, program_plan::TapeCarrier carrier) const;

    std::vector<std::shared_ptr<HostStaticTapeBatch>> hostBatches_;
    std::vector<std::array<Carrier, 4>> carriers_;
};

} // namespace vernon::runtime::ad

#endif
