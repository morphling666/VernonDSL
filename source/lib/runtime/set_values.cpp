#include "set_values.h"

#include <cstring>

namespace vernon::runtime {

SetValues::SetValues(const program::ResolvedExecutionPlan &plan, std::vector<ProgramValueState> values)
    : plan_(&plan), values_(std::move(values)) {
    for (ProgramValueState &value : values_) {
        if (value.argument.kind != VERNON_PROGRAM_TENSOR)
            continue;
        if (value.concreteShape) {
            value.argument.tensor.rank = static_cast<uint32_t>(value.concreteShape->size());
            value.argument.tensor.shape = value.concreteShape->data();
        }
        if (!value.strides.empty())
            value.argument.tensor.byte_strides = value.strides.data();
    }
}

vernon::Result<uint64_t, std::string> SetValues::resolveControl(const program::Program &program,
                                                                const program::ControlComponent &control) const {
    using Resolved = vernon::Result<uint64_t, std::string>;
    if (control.kind == program::ControlKind::Static)
        return Resolved{vernon::ok(control.value)};
    const auto unavailable = [] {
        return Resolved{vernon::err(std::string("Program dispatch control is not an available host scalar"))};
    };
    const auto invalid = [] {
        return Resolved{vernon::err(std::string("Program dispatch control must be a non-negative integer scalar"))};
    };
    const uint32_t valueId = control.reference;
    if (valueId >= values_.size() || valueId >= program.values.size())
        return unavailable();
    const VernonProgramArgument &argument = values_[valueId].argument;
    if (argument.kind != VERNON_PROGRAM_TENSOR || argument.tensor.storage != VERNON_TENSOR_HOST ||
        !argument.tensor.host_data || argument.tensor.byte_offset > argument.tensor.byte_size)
        return unavailable();
    const std::string &dtype = program.values[valueId].canonicalType.dtype;
    const bool unsigned32 = dtype == "u32" || dtype == "ui32";
    const bool signed32 = dtype == "i32" || dtype == "si32";
    const bool unsigned64 = dtype == "u64" || dtype == "ui64" || dtype == "index";
    const bool signed64 = dtype == "i64" || dtype == "si64";
    if (!unsigned32 && !signed32 && !unsigned64 && !signed64)
        return invalid();
    const size_t scalarSize = unsigned32 || signed32 ? sizeof(uint32_t) : sizeof(uint64_t);
    if (scalarSize > argument.tensor.byte_size - argument.tensor.byte_offset)
        return unavailable();
    const uint8_t *data = static_cast<const uint8_t *>(argument.tensor.host_data) + argument.tensor.byte_offset;
    if (unsigned32) {
        uint32_t scalar{};
        std::memcpy(&scalar, data, sizeof(scalar));
        return Resolved{vernon::ok(static_cast<uint64_t>(scalar))};
    }
    if (signed32) {
        int32_t scalar{};
        std::memcpy(&scalar, data, sizeof(scalar));
        return scalar < 0 ? invalid() : Resolved{vernon::ok(static_cast<uint64_t>(scalar))};
    }
    if (unsigned64) {
        uint64_t scalar{};
        std::memcpy(&scalar, data, sizeof(scalar));
        return Resolved{vernon::ok(scalar)};
    }
    int64_t scalar{};
    std::memcpy(&scalar, data, sizeof(scalar));
    return scalar < 0 ? invalid() : Resolved{vernon::ok(static_cast<uint64_t>(scalar))};
}

const VernonProgramArgument *SetValues::argument(uint32_t value) const {
    return value < values_.size() ? &values_[value].argument : nullptr;
}

bool SetValues::borrowsBinding(uint32_t value) const {
    return value < values_.size() && values_[value].ownership != ProgramValueOwnership::Unbound;
}

} // namespace vernon::runtime
