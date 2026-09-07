#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_PROGRAM_INVOCATION_STATE_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_PROGRAM_INVOCATION_STATE_H

#include "runtime/program_execution/device_buffer.h"
#include "runtime/program_execution/program_image_binding.h"
#include "runtime/program_execution_manifest.h"
#include "runtime/resolved_execution_plan.h"
#include "runtime/target_binding_plan.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct VernonRuntimeContext;

namespace vernon::runtime {
struct ProgramInvocationContext;
} // namespace vernon::runtime

namespace vernon::runtime::program_execution {

enum class ProgramValueOwnership {
    OwnedInvocation,
    BorrowedHost,
    BorrowedDevice,
    RetainedResidual,
    TapeCarrier,
};

struct ProgramValueState {
    struct StagedDeviceInitial {
        VernonRuntimeProviderResourceReference source{};
        std::optional<VernonRhiBuffer> retainedSourceBuffer;
        size_t byteOffset{};
        size_t byteSize{};
        VernonValueLayoutView elementLayout{};
        std::vector<uint64_t> shape;
        std::vector<int64_t> strides;
    };

    std::vector<uint8_t> ownedHostBytes;
    std::vector<int64_t> strides;
    std::optional<shape::ConcreteShape> concreteShape;
    VernonProgramArgument argument{};
    std::optional<StagedDeviceInitial> stagedDeviceInitial;
    ProgramValueOwnership ownership{ProgramValueOwnership::OwnedInvocation};
};

struct ProgramStorageBacking {
    uint32_t owner{UINT32_MAX};
    size_t bytes{};
    bool sized{};
    std::optional<VernonProgramArgument> external;
    std::optional<VernonProgramArgument> initial;
};

struct ProgramDeviceUpload {
    uint32_t value{};
    VernonRhiBuffer destination{};
    const void *source{};
    size_t size{};
};

struct CanonicalValueSnapshot {
    uint32_t value{};
    ProgramValueState logical;
    VernonProgramArgument residentArgument{};
    std::shared_ptr<const void> deviceOwner;
};

bool resolveProgramControl(const program::Program &program, const std::vector<ProgramValueState> &values,
                           const program::ControlComponent &control, uint64_t &value, std::string &error);

class ProgramInvocationState {
public:
    ProgramInvocationState(const program::ResolvedExecutionPlan &plan, std::vector<ProgramValueState> values,
                           std::map<uint32_t, ProgramStorageBacking> storageBackings);
    ~ProgramInvocationState();
    ProgramInvocationState(const ProgramInvocationState &) = delete;
    ProgramInvocationState &operator=(const ProgramInvocationState &) = delete;
    ProgramInvocationState(ProgramInvocationState &&other) noexcept;
    ProgramInvocationState &operator=(ProgramInvocationState &&) = delete;

    CanonicalValueSnapshot snapshotValue(uint32_t value) const;
    bool importSnapshot(const CanonicalValueSnapshot &snapshot, std::string &error);
    bool importStorageSnapshots(const std::map<uint32_t, ProgramStorageBacking> &snapshots, std::string &error);
    void retainOnly(const std::vector<char> &retained);
    bool restoreDeviceValuesFromHost(program::GraphDirection graph, std::string &error);
    bool bindControlImageStorage(VernonRuntimeContext &context, const program::Program &program, uint32_t storage,
                                 VernonRuntimeProviderResourceReference view, std::string &error);
    bool allocateOwnedImageStorages(VernonRuntimeContext &context, const program::Program &program, std::string &error);
    bool resolveControl(const program::Program &program, const program::ControlComponent &control, uint64_t &value,
                        std::string &error) const;

    const VernonProgramArgument *argument(uint32_t value) const;
    VernonProgramArgument *argument(uint32_t value);
    VernonRhiBuffer buffer(uint32_t value) const;

    const program::ResolvedExecutionPlan &plan() const { return *plan_; }
    const std::vector<VernonProgramArgument> &arguments() const { return arguments_; }
    std::vector<VernonProgramArgument> &arguments() { return arguments_; }
    const std::vector<ProgramValueState> &values() const { return values_; }
    std::vector<ProgramValueState> &values() { return values_; }
    const std::map<uint32_t, ProgramStorageBacking> &storageBackings() const { return storageBackings_; }
    const std::vector<ProgramDeviceUpload> &deviceUploads() const { return deviceUploads_; }
    void setInvocationContext(const ProgramInvocationContext *context) { invocationContext_ = context; }
    const ProgramInvocationContext *invocationContext() const { return invocationContext_; }
    const VernonRuntimeProviderResourceReference *controlImage(uint32_t storage) const;
    const VernonProgramArgument *externalStorage(uint32_t storage) const;

private:
    struct OwnedImageStorage {
        VernonRhiImage image{};
        VernonRhiImageView view{};
    };

    friend class ResolvedTransferExecutor;

    void rebindDescriptor(uint32_t value);

    const program::ResolvedExecutionPlan *plan_;
    std::vector<ProgramValueState> values_;
    std::map<uint32_t, ProgramStorageBacking> storageBackings_;
    std::vector<VernonProgramArgument> arguments_;
    std::vector<std::shared_ptr<DeviceBuffer>> deviceValues_;
    std::vector<ProgramDeviceUpload> deviceUploads_;
    std::map<uint32_t, VernonRuntimeProviderResourceReference> controlImages_;
    std::map<uint32_t, BoundProgramImage> controlImageDescriptors_;
    VernonRhiDevice imageDevice_{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    std::vector<OwnedImageStorage> ownedImages_;
    const ProgramInvocationContext *invocationContext_{};
};

} // namespace vernon::runtime::program_execution

#endif
