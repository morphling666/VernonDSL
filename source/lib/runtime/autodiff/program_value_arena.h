#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_VALUE_ARENA_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_VALUE_ARENA_H

#include "runtime/autodiff/host_tape_allocator.h"
#include "runtime/autodiff/runtime_gpu_resources.h"
#include "runtime/program_execution_manifest.h"
#include "runtime/target_binding_plan.h"

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct VernonRuntimeContext;
struct VernonResolvedProgramStage;

namespace vernon::runtime {
struct ProgramInvocationContext;
}

namespace vernon::runtime::ad {

struct MaterializedProgramArguments {
    std::vector<VernonPipelineArgument> arguments;
    std::vector<std::vector<uint64_t>> shapes;
    std::vector<std::vector<int64_t>> strides;
};

enum class ProgramValueOwnership {
    OwnedInvocation,
    BorrowedHost,
    BorrowedDevice,
    RetainedResidual,
    TapeCarrier,
};

struct ProgramHostValue {
    std::vector<uint8_t> owned;
    std::vector<int64_t> strides;
    std::optional<shape::ConcreteShape> concreteShape;
    VernonPipelineArgument argument{};
    std::shared_ptr<HostStaticTapeBatch> tapeBatch;
    ProgramValueOwnership ownership{ProgramValueOwnership::OwnedInvocation};
};

struct ProgramDeviceUpload {
    VernonRhiBuffer destination{};
    const void *source{};
    size_t size{};
};

class ProgramInvocationFrame {
public:
    ProgramInvocationFrame() = default;
    explicit ProgramInvocationFrame(const std::vector<VernonPipelineArgument> &hostArguments);
    explicit ProgramInvocationFrame(std::vector<ProgramHostValue> hostValues);

    bool materializeDevice(VernonRuntimeContext &context, const std::vector<char> &required, std::string &error);
    bool restoreDeviceValuesFromHost(const std::vector<char> &required, std::string &error);
    bool adoptRetainedValue(uint32_t value, const ProgramInvocationFrame &retained, std::string &error);
    bool allocateCarrier(VernonRuntimeContext &context, uint32_t value, const program::TargetBinding &binding,
                         size_t byteSize, std::vector<uint64_t> shape, std::vector<int64_t> strides,
                         std::string &error);
    bool uploadCarrier(uint32_t value, program_plan::TapeCarrier carrier, const void *data, size_t byteSize,
                       std::string &error);
    bool downloadCarrier(uint32_t value, program_plan::TapeCarrier carrier, void *data, size_t byteSize,
                         std::string &error) const;
    bool downloadLogicalToHost(const std::vector<char> &required, std::string &error) const;
    bool materializeNodeArguments(const program::Program &program, const program::Node &node,
                                  const VernonResolvedProgramStage &stage, MaterializedProgramArguments &output,
                                  std::string &error) const;
    bool bindControlImageStorage(const program::Program &program, uint32_t storage,
                                 VernonRuntimeProviderResourceReference view, std::string &error);
    bool resolveControl(const program::Program &program, const program::ControlComponent &control, uint64_t &value,
                        std::string &error) const;

    const VernonPipelineArgument *argument(uint32_t value, const program::TargetBinding *binding = nullptr) const;
    VernonPipelineArgument *argument(uint32_t value, const program::TargetBinding *binding = nullptr);
    VernonRhiBuffer buffer(uint32_t value, const program::TargetBinding *binding = nullptr) const;

    const std::vector<VernonPipelineArgument> &logicalArguments() const { return logicalArguments_; }
    std::vector<VernonPipelineArgument> &logicalArguments() { return logicalArguments_; }
    const std::vector<ProgramDeviceUpload> &deviceUploads() const { return deviceUploads_; }
    bool deviceResident() const { return deviceResident_; }
    void setInvocationContext(const ProgramInvocationContext *context) { invocationContext_ = context; }
    const ProgramInvocationContext *invocationContext() const { return invocationContext_; }
    const std::vector<ProgramHostValue> &hostValues() const { return hostValues_; }
    std::vector<ProgramHostValue> &hostValues() { return hostValues_; }

private:
    struct Carrier {
        std::shared_ptr<gpu::DeviceBuffer> buffer;
        VernonPipelineArgument argument{};
        std::vector<uint64_t> shape;
        std::vector<int64_t> strides;
    };

    static size_t carrierIndex(program_plan::TapeCarrier carrier);
    Carrier *carrier(uint32_t value, program_plan::TapeCarrier carrier);
    const Carrier *carrier(uint32_t value, program_plan::TapeCarrier carrier) const;
    void rebindLogicalDescriptor(uint32_t value);

    std::vector<ProgramHostValue> hostValues_;
    std::vector<VernonPipelineArgument> logicalArguments_;
    std::vector<std::shared_ptr<gpu::DeviceBuffer>> logicalBuffers_;
    std::vector<ProgramDeviceUpload> deviceUploads_;
    std::vector<std::array<Carrier, 4>> carriers_;
    std::map<uint32_t, VernonRuntimeProviderResourceReference> controlImages_;
    const ProgramInvocationContext *invocationContext_{};
    bool deviceResident_{};
};

} // namespace vernon::runtime::ad

#endif
