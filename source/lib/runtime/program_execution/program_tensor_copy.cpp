#include "program_tensor_copy.h"

#include <algorithm>
#include <limits>

namespace vernon::runtime::program_execution {

bool planProgramTensorCopy(const VernonTensorView &source, const VernonTensorView &destination,
                           std::vector<ProgramTensorCopyRegion> &regions, std::string &error) {
    if (source.rank != destination.rank ||
        (source.rank && (!source.shape || !source.byte_strides || !destination.shape || !destination.byte_strides)) ||
        !source.element_layout.byte_size || source.element_layout.byte_size != destination.element_layout.byte_size)
        return error = "Program tensor copy has incompatible layouts (source rank " + std::to_string(source.rank) +
                       ", destination rank " + std::to_string(destination.rank) + ", source element bytes " +
                       std::to_string(source.element_layout.byte_size) + ", destination element bytes " +
                       std::to_string(destination.element_layout.byte_size) + ")",
               false;
    for (uint32_t axis = 0; axis < source.rank; ++axis)
        if (source.shape[axis] != destination.shape[axis])
            return error = "Program tensor copy has incompatible shapes", false;
    if (source.rank &&
        std::any_of(source.shape, source.shape + source.rank, [](uint64_t extent) { return extent == 0; }))
        return true;

    const size_t elementBytes = source.element_layout.byte_size;
    std::vector<uint64_t> index(source.rank);
    for (;;) {
        if (source.byte_offset > std::numeric_limits<int64_t>::max() ||
            destination.byte_offset > std::numeric_limits<int64_t>::max())
            return error = "Program tensor copy offset exceeds portable range", false;
        int64_t sourceOffset = static_cast<int64_t>(source.byte_offset);
        int64_t destinationOffset = static_cast<int64_t>(destination.byte_offset);
        const auto advance = [](int64_t &offset, uint64_t coordinate, int64_t stride) {
            if (stride >= 0) {
                const uint64_t positive = static_cast<uint64_t>(stride);
                if (positive &&
                    coordinate > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - offset) / positive)
                    return false;
                offset += static_cast<int64_t>(coordinate * positive);
                return true;
            }
            const uint64_t magnitude = static_cast<uint64_t>(-(stride + 1)) + 1;
            if (magnitude && coordinate > static_cast<uint64_t>(offset) / magnitude)
                return false;
            offset -= static_cast<int64_t>(coordinate * magnitude);
            return true;
        };
        for (uint32_t axis = 0; axis < source.rank; ++axis)
            if (!advance(sourceOffset, index[axis], source.byte_strides[axis]) ||
                !advance(destinationOffset, index[axis], destination.byte_strides[axis]))
                return error = "Program tensor copy layout overflows", false;
        const auto valid = [&](int64_t offset, size_t capacity) {
            return offset >= 0 && static_cast<uint64_t>(offset) <= capacity &&
                   elementBytes <= capacity - static_cast<size_t>(offset);
        };
        if (!valid(sourceOffset, source.byte_size) || !valid(destinationOffset, destination.byte_size))
            return error = "Program tensor copy exceeds Storage backing", false;
        const size_t sourceByte = static_cast<size_t>(sourceOffset);
        const size_t destinationByte = static_cast<size_t>(destinationOffset);
        if (!regions.empty() && regions.back().sourceOffset + regions.back().size == sourceByte &&
            regions.back().destinationOffset + regions.back().size == destinationByte) {
            regions.back().size += elementBytes;
        } else {
            regions.push_back({sourceByte, destinationByte, elementBytes});
        }
        if (source.rank == 0)
            break;
        uint32_t axis = source.rank;
        while (axis) {
            --axis;
            if (++index[axis] < source.shape[axis])
                break;
            index[axis] = 0;
        }
        if (axis == 0 && index[0] == 0)
            break;
    }
    return true;
}

} // namespace vernon::runtime::program_execution
