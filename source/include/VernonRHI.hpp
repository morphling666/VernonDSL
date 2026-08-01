#ifndef VERNON_RHI_HPP
#define VERNON_RHI_HPP

#include "VernonRHI.h"

#include <cstdint>
#include <type_traits>

namespace vernon::rhi {

template <typename Handle> constexpr Handle invalidHandle() noexcept {
    static_assert(std::is_trivially_copyable_v<Handle>);
    return {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
}

template <typename Handle> constexpr bool isValid(Handle handle) noexcept {
    static_assert(std::is_trivially_copyable_v<Handle>);
    return handle.index != VERNON_RHI_INVALID_HANDLE_INDEX && handle.generation != 0;
}

template <typename Handle> constexpr bool operator==(Handle left, Handle right) noexcept {
    return left.index == right.index && left.generation == right.generation;
}

template <typename Handle> constexpr bool operator!=(Handle left, Handle right) noexcept { return !(left == right); }

} // namespace vernon::rhi

#endif
