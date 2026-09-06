#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_VALUE_ARENA_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_VALUE_ARENA_H

#include "runtime/autodiff/host_tape_allocator.h"
#include "runtime/autodiff/runtime_gpu_commands.h"
#include "runtime/autodiff/runtime_gpu_resources.h"
#include "runtime/program_execution_manifest.h"
#include "runtime/target_binding_plan.h"

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct VernonRuntimeContext;
struct VernonResolvedProgramNode;

namespace vernon::runtime {
struct ProgramInvocationContext;
}

namespace vernon::runtime::ad {

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
    std::vector<std::vector<uint8_t>> hostStorage;
    std::vector<std::shared_ptr<gpu::DeviceBuffer>> deviceStorage;
    std::vector<HostCopy> copiesBefore;
    std::vector<HostCopy> copiesAfter;
    std::vector<gpu::DeviceBufferCopy> deviceCopiesBefore;
    std::vector<gpu::DeviceBufferCopy> deviceCopiesAfter;

    bool prepareHost(std::string &error) const;
    bool commitHost(std::string &error) const;
};

enum class ProgramValueOwnership {
    OwnedInvocation,
    BorrowedHost,
    BorrowedDevice,
    RetainedResidual,
    TapeCarrier,
};

struct LogicalProgramValue {
    std::vector<uint8_t> owned;
    std::vector<int64_t> strides;
    std::optional<shape::ConcreteShape> concreteShape;
    VernonProgramArgument argument{};
    std::shared_ptr<HostStaticTapeBatch> tapeBatch;
    ProgramValueOwnership ownership{ProgramValueOwnership::OwnedInvocation};
};

bool resolveProgramControl(const program::Program &program, const std::vector<LogicalProgramValue> &hostValues,
                           const program::ControlComponent &control, uint64_t &value, std::string &error);

struct ProgramDeviceUpload {
    VernonRhiBuffer destination{};
    const void *source{};
    size_t size{};
};

class LogicalValueFrame {
public:
    LogicalValueFrame() = default;
    explicit LogicalValueFrame(const std::vector<VernonProgramArgument> &hostArguments);
    explicit LogicalValueFrame(std::vector<LogicalProgramValue> hostValues);
    LogicalValueFrame(LogicalValueFrame &&other) noexcept;
    LogicalValueFrame(const LogicalValueFrame &) = delete;
    LogicalValueFrame &operator=(const LogicalValueFrame &) = delete;
    LogicalValueFrame &operator=(LogicalValueFrame &&) = delete;

    bool materializeDevice(VernonRuntimeContext &context, const std::vector<char> &required, std::string &error);
    bool restoreDeviceValuesFromHost(const std::vector<char> &required, std::string &error);
    bool adoptRetainedValue(uint32_t value, const LogicalValueFrame &retained, std::string &error);
    void retainOnly(const std::vector<char> &retained);
    bool allocateCarrier(VernonRuntimeContext &context, uint32_t value, const program::TargetBinding &binding,
                         size_t byteSize, std::vector<uint64_t> shape, std::vector<int64_t> strides,
                         std::string &error);
    bool uploadCarrier(uint32_t value, program_plan::TapeCarrier carrier, const void *data, size_t byteSize,
                       std::string &error);
    bool downloadCarrier(uint32_t value, program_plan::TapeCarrier carrier, void *data, size_t byteSize,
                         std::string &error) const;
    bool downloadLogicalToHost(const std::vector<char> &required, std::string &error) const;
    bool materializeNodeArguments(const program::Program &program, const program::Node &node,
                                  const VernonResolvedProgramNode &nodePlan, MaterializedNodeFrame &output,
                                  std::string &error) const;
    bool bindControlImageStorage(const program::Program &program, uint32_t storage,
                                 VernonRuntimeProviderResourceReference view, std::string &error);
    bool resolveControl(const program::Program &program, const program::ControlComponent &control, uint64_t &value,
                        std::string &error) const;

    const VernonProgramArgument *argument(uint32_t value, const program::TargetBinding *binding = nullptr) const;
    VernonProgramArgument *argument(uint32_t value, const program::TargetBinding *binding = nullptr);
    VernonRhiBuffer buffer(uint32_t value, const program::TargetBinding *binding = nullptr) const;

    const std::vector<VernonProgramArgument> &logicalArguments() const { return logicalArguments_; }
    std::vector<VernonProgramArgument> &logicalArguments() { return logicalArguments_; }
    const std::vector<ProgramDeviceUpload> &deviceUploads() const { return deviceUploads_; }
    bool deviceResident() const { return deviceResident_; }
    void setInvocationContext(const ProgramInvocationContext *context) { invocationContext_ = context; }
    const ProgramInvocationContext *invocationContext() const { return invocationContext_; }
    const std::vector<LogicalProgramValue> &hostValues() const { return hostValues_; }
    std::vector<LogicalProgramValue> &hostValues() { return hostValues_; }

private:
    struct Carrier {
        std::shared_ptr<gpu::DeviceBuffer> buffer;
        VernonProgramArgument argument{};
        std::vector<uint64_t> shape;
        std::vector<int64_t> strides;
    };

    static size_t carrierIndex(program_plan::TapeCarrier carrier);
    Carrier *carrier(uint32_t value, program_plan::TapeCarrier carrier);
    const Carrier *carrier(uint32_t value, program_plan::TapeCarrier carrier) const;
    void rebindLogicalDescriptor(uint32_t value);

    std::vector<LogicalProgramValue> hostValues_;
    std::vector<VernonProgramArgument> logicalArguments_;
    std::vector<std::shared_ptr<gpu::DeviceBuffer>> logicalBuffers_;
    std::vector<ProgramDeviceUpload> deviceUploads_;
    std::vector<std::array<Carrier, 4>> carriers_;
    std::map<uint32_t, VernonRuntimeProviderResourceReference> controlImages_;
    const ProgramInvocationContext *invocationContext_{};
    bool deviceResident_{};
};

} // namespace vernon::runtime::ad

#endif
