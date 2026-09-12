#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_PROGRAM_BOUNDARY_CONTRACT_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_PROGRAM_BOUNDARY_CONTRACT_H

#include "VernonRuntime.h"
#include "runtime/program_execution_manifest.h"

namespace vernon::runtime::program_execution {

constexpr VernonProgramArgumentKind boundaryArgumentKind(program::BoundaryCategory category) noexcept {
    switch (category) {
    case program::BoundaryCategory::Texture:
        return VERNON_PROGRAM_IMAGE;
    case program::BoundaryCategory::Sampler:
        return VERNON_PROGRAM_SAMPLER;
    case program::BoundaryCategory::Value:
    case program::BoundaryCategory::StorageView:
        return VERNON_PROGRAM_TENSOR;
    }
    return VERNON_PROGRAM_TENSOR;
}

constexpr VernonValueAccess boundaryValueAccess(program::BoundaryAccess access) noexcept {
    switch (access) {
    case program::BoundaryAccess::Read:
        return VERNON_ACCESS_READ;
    case program::BoundaryAccess::Write:
        return VERNON_ACCESS_WRITE;
    case program::BoundaryAccess::ReadWrite:
        return VERNON_ACCESS_READ_WRITE;
    }
    return VERNON_ACCESS_READ;
}

constexpr bool valueAccessSatisfies(VernonValueAccess required, VernonValueAccess supplied) noexcept {
    if (required > VERNON_ACCESS_READ_WRITE || supplied > VERNON_ACCESS_READ_WRITE)
        return false;
    switch (required) {
    case VERNON_ACCESS_READ:
        return supplied != VERNON_ACCESS_WRITE;
    case VERNON_ACCESS_WRITE:
        return supplied != VERNON_ACCESS_READ;
    case VERNON_ACCESS_READ_WRITE:
        return supplied == VERNON_ACCESS_READ_WRITE;
    }
    return false;
}

constexpr bool argumentMatchesBoundary(const program::BoundarySlot &boundary,
                                       const VernonProgramArgument &argument) noexcept {
    if (argument.kind != boundaryArgumentKind(boundary.category))
        return false;
    return argument.kind != VERNON_PROGRAM_TENSOR ||
           valueAccessSatisfies(boundaryValueAccess(boundary.access), argument.tensor.access);
}

} // namespace vernon::runtime::program_execution

#endif
