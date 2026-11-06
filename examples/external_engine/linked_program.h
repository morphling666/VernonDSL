#ifndef VERNON_EXTERNAL_ENGINE_LINKED_PROGRAM_H
#define VERNON_EXTERNAL_ENGINE_LINKED_PROGRAM_H

#include "VernonRuntime.hpp"
#include "VernonSchedule.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vernon_external_engine {

inline std::string runtimeError(VernonRuntimeContext *runtime) {
    const VernonStringView error = vernonRuntimeGetLastError(runtime);
    return error.data && error.size ? std::string(error.data, error.size) : std::string("unknown runtime error");
}

// A cooked Program's forward graph linked as one schedule on a HAL device's runtime, and the argument record its
// calls submit. Arguments written into the record keep pointing at the caller's data, which every call reads again.
class LinkedProgram {
public:
    LinkedProgram() = default;
    LinkedProgram(const LinkedProgram &) = delete;
    LinkedProgram &operator=(const LinkedProgram &) = delete;
    ~LinkedProgram() { reset(); }

    bool link(VernonRuntimeContext *runtime, const char *manifest, size_t manifestSize, const char *bundleDirectory,
              std::string &error) {
        reset();
        runtime_ = runtime;
        VernonProgramBundleLoadOptions options{};
        options.struct_size = sizeof(options);
        options.bundle_directory = bundleDirectory;
        auto asset = vernon::runtime::ProgramAsset::load(runtime, manifest, manifestSize, &options);
        if (asset.isErr()) {
            error = runtimeError(runtime);
            return false;
        }
        auto executable = std::move(asset).value().resolve();
        if (executable.isErr()) {
            error = runtimeError(runtime);
            return false;
        }
        executable_.emplace(std::move(executable).value());

        VernonScheduleProgramDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.graph = {"forward", 7};
        descriptor.executable = executable_->get();
        schedule_ = vernonScheduleLinkProgram(runtime, &descriptor);
        if (!schedule_) {
            error = runtimeError(runtime);
            return false;
        }
        const uint8_t *recordTemplate = vernonScheduleRecordTemplate(schedule_);
        record_.assign(recordTemplate, recordTemplate + vernonScheduleRecordSize(schedule_));
        return true;
    }

    // The first slot of `kind`; SIZE_MAX when the schedule has none.
    size_t slotOfKind(uint32_t kind) const {
        const VernonScheduleSlot *slots = vernonScheduleSlots(schedule_);
        for (size_t slot = 0; slot < vernonScheduleSlotCount(schedule_); ++slot)
            if (slots[slot].kind == kind)
                return slot;
        return SIZE_MAX;
    }

    template <typename T> void write(size_t slot, const T &value) {
        std::memcpy(record_.data() + vernonScheduleSlots(schedule_)[slot].offset, &value, sizeof(value));
    }

    // Writes the argument of boundary parameter `name` into its ARGUMENT slot; false when the schedule takes none.
    bool bindHostTensor(const char *name, const void *data, size_t size, uint32_t rank = 0,
                        const uint64_t *shape = nullptr, const int64_t *strides = nullptr) {
        VernonProgramArgument argument = tensor(name, rank, shape, strides, size);
        argument.tensor.storage = VERNON_TENSOR_HOST;
        argument.tensor.host_data = data;
        return bind(name, argument);
    }

    bool bindDeviceTensor(const char *name, VernonRuntimeProviderResourceReference resource, size_t size, uint32_t rank,
                          const uint64_t *shape, const int64_t *strides) {
        VernonProgramArgument argument = tensor(name, rank, shape, strides, size);
        argument.tensor.storage = VERNON_TENSOR_DEVICE_RESOURCE;
        argument.tensor.resource = resource;
        return bind(name, argument);
    }

    bool bindImage(const char *name, VernonRuntimeProviderResourceReference view) {
        VernonProgramArgument argument{};
        argument.kind = VERNON_PROGRAM_IMAGE;
        argument.image.view = view;
        return bind(name, argument);
    }

    bool bindSampler(const char *name, VernonRuntimeProviderResourceReference sampler) {
        VernonProgramArgument argument{};
        argument.kind = VERNON_PROGRAM_SAMPLER;
        argument.resource = sampler;
        return bind(name, argument);
    }

    bool call(std::string &error) {
        uint64_t point = 0;
        if (vernonScheduleCall(schedule_, record_.data(), record_.size(), &point) != VERNON_STATUS_OK) {
            error = runtimeError(runtime_);
            return false;
        }
        point_ = point;
        return true;
    }

    // Waits for the device to complete the latest call.
    bool wait(std::string &error) {
        if (vernonScheduleWait(runtime_, point_) == VERNON_STATUS_OK)
            return true;
        error = runtimeError(runtime_);
        return false;
    }

    void reset() {
        if (schedule_)
            vernonScheduleDestroy(schedule_);
        schedule_ = nullptr;
        executable_.reset();
        record_.clear();
        runtime_ = nullptr;
        point_ = 0;
    }

private:
    VernonProgramArgument tensor(const char *name, uint32_t rank, const uint64_t *shape, const int64_t *strides,
                                 size_t size) const {
        VernonProgramArgument argument{};
        argument.kind = VERNON_PROGRAM_TENSOR;
        argument.tensor.struct_size = sizeof(argument.tensor);
        VernonProgramParameterView parameter{};
        if (find(name, parameter)) {
            argument.tensor.element_layout = parameter.element_layout;
            argument.tensor.access = parameter.access;
        }
        argument.tensor.rank = rank;
        argument.tensor.shape = shape;
        argument.tensor.byte_strides = strides;
        argument.tensor.byte_size = size;
        return argument;
    }

    bool find(const char *name, VernonProgramParameterView &parameter) const {
        return vernonRuntimeProgramExecutableFindParameter(executable_->get(), {name, std::strlen(name)}, &parameter) ==
               VERNON_STATUS_OK;
    }

    bool bind(const char *name, VernonProgramArgument argument) {
        VernonProgramParameterView parameter{};
        if (!find(name, parameter))
            return false;
        for (size_t slot = 0; slot < vernonScheduleSlotCount(schedule_); ++slot)
            if (vernonScheduleSlotBoundary(schedule_, slot) == parameter.slot) {
                argument.slot = parameter.slot;
                write(slot, argument);
                return true;
            }
        return false;
    }

    VernonRuntimeContext *runtime_{};
    std::optional<vernon::runtime::ProgramExecutable> executable_;
    VernonLinkedSchedule *schedule_{};
    std::vector<uint8_t> record_;
    uint64_t point_{};
};

} // namespace vernon_external_engine

#endif
