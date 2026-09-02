#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_VALUE_ARENA_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_VALUE_ARENA_H

#include "runtime/autodiff/host_tape_allocator.h"
#include "runtime/autodiff/runtime_gpu_resources.h"
#include "runtime/target_binding_plan.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

struct VernonRuntimeContext;
struct VernonResolvedProgramStage;

namespace vernon::runtime {
struct ExecutableProgram;
}

namespace vernon::runtime::ad {

struct MaterializedProgramArguments {
    std::vector<VernonPipelineArgument> arguments;
    std::vector<std::vector<uint64_t>> shapes;
    std::vector<std::vector<int64_t>> strides;
};

struct ProgramHostValue {
    std::vector<uint8_t> owned;
    std::vector<int64_t> strides;
    std::optional<shape::ConcreteShape> concreteShape;
    VernonPipelineArgument argument{};
    std::shared_ptr<HostStaticTapeBatch> tapeBatch;
};

class ProgramValueArena {
public:
    ProgramValueArena() = default;
    explicit ProgramValueArena(const std::vector<VernonPipelineArgument> &hostArguments);
    explicit ProgramValueArena(std::vector<ProgramHostValue> hostValues);

    bool materializeDevice(VernonRuntimeContext &context, const std::vector<char> &required, std::string &error);
    bool restoreDeviceValuesFromHost(const std::vector<char> &required, std::string &error);
    bool adoptRetainedValue(uint32_t value, const ProgramValueArena &retained, std::string &error);
    bool allocateCarrier(VernonRuntimeContext &context, uint32_t value, const program::TargetBinding &binding,
                         size_t byteSize, std::vector<uint64_t> shape, std::vector<int64_t> strides,
                         std::string &error);
    bool uploadCarrier(uint32_t value, program::CarrierSemantic semantic, const void *data, size_t byteSize,
                       std::string &error);
    bool downloadCarrier(uint32_t value, program::CarrierSemantic semantic, void *data, size_t byteSize,
                         std::string &error) const;
    bool downloadLogicalToHost(const std::vector<char> &required, std::string &error) const;
    bool materializeNodeArguments(const ExecutableProgram &execution, const ProgramNode &node,
                                  const VernonResolvedProgramStage &stage, MaterializedProgramArguments &output,
                                  std::string &error) const;

    const VernonPipelineArgument *argument(uint32_t value, const program::TargetBinding *binding = nullptr) const;
    VernonPipelineArgument *argument(uint32_t value, const program::TargetBinding *binding = nullptr);
    VernonRhiBuffer buffer(uint32_t value, const program::TargetBinding *binding = nullptr) const;

    const std::vector<VernonPipelineArgument> &logicalArguments() const { return logicalArguments_; }
    std::vector<VernonPipelineArgument> &logicalArguments() { return logicalArguments_; }
    bool deviceResident() const { return deviceResident_; }
    const std::vector<ProgramHostValue> &hostValues() const { return hostValues_; }
    std::vector<ProgramHostValue> &hostValues() { return hostValues_; }

private:
    struct Carrier {
        std::shared_ptr<gpu::DeviceBuffer> buffer;
        VernonPipelineArgument argument{};
        std::vector<uint64_t> shape;
        std::vector<int64_t> strides;
    };

    static size_t carrierIndex(program::CarrierSemantic semantic);
    Carrier *carrier(uint32_t value, program::CarrierSemantic semantic);
    const Carrier *carrier(uint32_t value, program::CarrierSemantic semantic) const;
    void rebindLogicalDescriptor(uint32_t value);

    std::vector<ProgramHostValue> hostValues_;
    std::vector<VernonPipelineArgument> logicalArguments_;
    std::vector<std::shared_ptr<gpu::DeviceBuffer>> logicalBuffers_;
    std::vector<std::array<Carrier, 4>> carriers_;
    bool deviceResident_{};
};

} // namespace vernon::runtime::ad

#endif
