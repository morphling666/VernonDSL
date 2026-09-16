#ifndef VERNON_ERROR_HPP
#define VERNON_ERROR_HPP

#include "VernonCommon.h"
#include "VernonRHI.h"
#include "VernonResult.hpp"

#include <cstddef>
#include <cstdint>

namespace vernon {

enum class ErrorDomain : std::uint8_t {
    Lifecycle = 1,
    Rhi = 2,
    Runtime = 3,
    ResourceAccess = 4,
    Arithmetic = 5,
    Allocation = 6,
    Provider = 7,
};

struct ErrorContext {
    const char *operation{};
    std::uint64_t value{};
    std::uint32_t detail{};
};

enum class LifecycleErrorCode : std::uint8_t {
    NotOpen = 1,
    AlreadyClosed = 2,
    LiveChildren = 3,
    AdmissionSaturated = 4,
    PinRejected = 5,
    CounterUnderflow = 6,
    TeardownFailed = 7,
    StaleAdmission = 8,
    AllocationFailed = 9,
    InvalidConfiguration = 10,
};

struct LifecycleError {
    LifecycleErrorCode code{LifecycleErrorCode::NotOpen};
    ErrorContext context{};
};

enum class RhiErrorCode : std::uint8_t {
    InvalidArgument = 1,
    Unsupported = 2,
    ResourceExhausted = 3,
    BackendFailure = 4,
    LifecycleFailure = 5,
    SynchronizationFailure = 6,
};

struct RhiError {
    RhiErrorCode code{RhiErrorCode::BackendFailure};
    ErrorContext context{};
};

enum class RuntimeErrorCode : std::uint8_t {
    InvalidArgument = 1,
    ParseFailure = 2,
    VerificationFailure = 3,
    Unsupported = 4,
    ResourceExhausted = 5,
    RhiFailure = 6,
    LifecycleFailure = 7,
    InternalFailure = 8,
};

struct RuntimeError {
    RuntimeErrorCode code{RuntimeErrorCode::InternalFailure};
    ErrorContext context{};
};

enum class ResourceAccessErrorCode : std::uint8_t {
    InvalidRegion = 1,
    ConflictingAccess = 2,
    UnknownAuthority = 3,
    ResourceExhausted = 4,
    LifecycleFailure = 5,
};

struct ResourceAccessError {
    ResourceAccessErrorCode code{ResourceAccessErrorCode::InvalidRegion};
    ErrorContext context{};
};

enum class ProviderErrorCode : std::uint8_t {
    InvalidArgument = 1,
    Unsupported = 2,
    ResourceExhausted = 3,
    ResourceFailure = 4,
    LifecycleFailure = 5,
    BackendFailure = 6,
};

struct ProviderError {
    ProviderErrorCode code{ProviderErrorCode::BackendFailure};
    ErrorContext context{};
};

static_assert(std::is_trivially_copyable_v<ErrorContext>);
static_assert(std::is_trivially_copyable_v<LifecycleError>);
static_assert(std::is_trivially_copyable_v<RhiError>);
static_assert(std::is_trivially_copyable_v<RuntimeError>);
static_assert(std::is_trivially_copyable_v<ResourceAccessError>);
static_assert(std::is_trivially_copyable_v<ProviderError>);

[[nodiscard]] constexpr RhiError toRhiError(LifecycleError error) noexcept {
    return {(error.code == LifecycleErrorCode::AdmissionSaturated || error.code == LifecycleErrorCode::AllocationFailed)
                ? RhiErrorCode::ResourceExhausted
                : RhiErrorCode::LifecycleFailure,
            error.context};
}

[[nodiscard]] constexpr RhiError toRhiError(ArithmeticError error, ErrorContext context = {}) noexcept {
    if (error == ArithmeticError::Saturated)
        return {RhiErrorCode::ResourceExhausted, context};
    return {RhiErrorCode::InvalidArgument, context};
}

[[nodiscard]] constexpr RhiError toRhiError(AllocationError, ErrorContext context = {}) noexcept {
    return {RhiErrorCode::ResourceExhausted, context};
}

[[nodiscard]] constexpr RhiError toRhiError(ResourceAccessError error) noexcept {
    switch (error.code) {
    case ResourceAccessErrorCode::ResourceExhausted:
        return {RhiErrorCode::ResourceExhausted, error.context};
    case ResourceAccessErrorCode::LifecycleFailure:
        return {RhiErrorCode::LifecycleFailure, error.context};
    case ResourceAccessErrorCode::InvalidRegion:
    case ResourceAccessErrorCode::ConflictingAccess:
    case ResourceAccessErrorCode::UnknownAuthority:
        return {RhiErrorCode::InvalidArgument, error.context};
    }
    return {RhiErrorCode::BackendFailure, error.context};
}

[[nodiscard]] constexpr RuntimeError toRuntimeError(LifecycleError error) noexcept {
    return {(error.code == LifecycleErrorCode::AdmissionSaturated || error.code == LifecycleErrorCode::AllocationFailed)
                ? RuntimeErrorCode::ResourceExhausted
                : RuntimeErrorCode::LifecycleFailure,
            error.context};
}

[[nodiscard]] constexpr RuntimeError toRuntimeError(RhiError error) noexcept {
    switch (error.code) {
    case RhiErrorCode::InvalidArgument:
        return {RuntimeErrorCode::InvalidArgument, error.context};
    case RhiErrorCode::Unsupported:
        return {RuntimeErrorCode::Unsupported, error.context};
    case RhiErrorCode::ResourceExhausted:
        return {RuntimeErrorCode::ResourceExhausted, error.context};
    case RhiErrorCode::LifecycleFailure:
        return {RuntimeErrorCode::LifecycleFailure, error.context};
    case RhiErrorCode::BackendFailure:
    case RhiErrorCode::SynchronizationFailure:
        return {RuntimeErrorCode::RhiFailure, error.context};
    }
    return {RuntimeErrorCode::InternalFailure, error.context};
}

[[nodiscard]] constexpr RuntimeError toRuntimeError(ResourceAccessError error) noexcept {
    switch (error.code) {
    case ResourceAccessErrorCode::ResourceExhausted:
        return {RuntimeErrorCode::ResourceExhausted, error.context};
    case ResourceAccessErrorCode::LifecycleFailure:
        return {RuntimeErrorCode::LifecycleFailure, error.context};
    case ResourceAccessErrorCode::InvalidRegion:
        return {RuntimeErrorCode::InvalidArgument, error.context};
    case ResourceAccessErrorCode::ConflictingAccess:
    case ResourceAccessErrorCode::UnknownAuthority:
        return {RuntimeErrorCode::InternalFailure, error.context};
    }
    return {RuntimeErrorCode::InternalFailure, error.context};
}

[[nodiscard]] constexpr RuntimeError toRuntimeError(ProviderError error) noexcept {
    switch (error.code) {
    case ProviderErrorCode::InvalidArgument:
        return {RuntimeErrorCode::InvalidArgument, error.context};
    case ProviderErrorCode::Unsupported:
        return {RuntimeErrorCode::Unsupported, error.context};
    case ProviderErrorCode::ResourceExhausted:
        return {RuntimeErrorCode::ResourceExhausted, error.context};
    case ProviderErrorCode::LifecycleFailure:
        return {RuntimeErrorCode::LifecycleFailure, error.context};
    case ProviderErrorCode::ResourceFailure:
    case ProviderErrorCode::BackendFailure:
        return {RuntimeErrorCode::InternalFailure, error.context};
    }
    return {RuntimeErrorCode::InternalFailure, error.context};
}

[[nodiscard]] constexpr RuntimeError toRuntimeError(ArithmeticError error, ErrorContext context = {}) noexcept {
    return {error == ArithmeticError::Saturated ? RuntimeErrorCode::ResourceExhausted
                                                : RuntimeErrorCode::InvalidArgument,
            context};
}

[[nodiscard]] constexpr RuntimeError toRuntimeError(AllocationError, ErrorContext context = {}) noexcept {
    return {RuntimeErrorCode::ResourceExhausted, context};
}

[[nodiscard]] constexpr RuntimeError runtimeErrorFromStatus(VernonStatus status, ErrorContext context = {}) noexcept {
    switch (status) {
    case VERNON_STATUS_INVALID_ARGUMENT:
        return {RuntimeErrorCode::InvalidArgument, context};
    case VERNON_STATUS_PARSE_ERROR:
        return {RuntimeErrorCode::ParseFailure, context};
    case VERNON_STATUS_VERIFICATION_ERROR:
        return {RuntimeErrorCode::VerificationFailure, context};
    case VERNON_STATUS_UNSUPPORTED_TARGET:
        return {RuntimeErrorCode::Unsupported, context};
    case VERNON_STATUS_INTERNAL_ERROR:
    case VERNON_STATUS_OK:
        return {RuntimeErrorCode::InternalFailure, context};
    }
    return {RuntimeErrorCode::InternalFailure, context};
}

[[nodiscard]] constexpr VernonRhiStatus toVernonRhiStatus(RhiError error) noexcept {
    switch (error.code) {
    case RhiErrorCode::InvalidArgument:
    case RhiErrorCode::LifecycleFailure:
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    case RhiErrorCode::Unsupported:
        return VERNON_RHI_STATUS_UNSUPPORTED;
    case RhiErrorCode::ResourceExhausted:
        return VERNON_RHI_STATUS_RESOURCE_EXHAUSTED;
    case RhiErrorCode::BackendFailure:
    case RhiErrorCode::SynchronizationFailure:
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    return VERNON_RHI_STATUS_INTERNAL_ERROR;
}

[[nodiscard]] constexpr VernonRhiStatus toVernonRhiStatus(LifecycleError error) noexcept {
    return toVernonRhiStatus(toRhiError(error));
}

[[nodiscard]] constexpr VernonRhiStatus toVernonRhiStatus(ResourceAccessError error) noexcept {
    return toVernonRhiStatus(toRhiError(error));
}

[[nodiscard]] constexpr VernonRhiStatus toVernonRhiStatus(ArithmeticError error) noexcept {
    return toVernonRhiStatus(toRhiError(error));
}

[[nodiscard]] constexpr VernonRhiStatus toVernonRhiStatus(AllocationError error) noexcept {
    return toVernonRhiStatus(toRhiError(error));
}

[[nodiscard]] constexpr VernonRhiStatus toVernonRhiStatus(RuntimeError error) noexcept {
    switch (error.code) {
    case RuntimeErrorCode::InvalidArgument:
    case RuntimeErrorCode::ParseFailure:
    case RuntimeErrorCode::VerificationFailure:
    case RuntimeErrorCode::LifecycleFailure:
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    case RuntimeErrorCode::Unsupported:
        return VERNON_RHI_STATUS_UNSUPPORTED;
    case RuntimeErrorCode::ResourceExhausted:
        return VERNON_RHI_STATUS_RESOURCE_EXHAUSTED;
    case RuntimeErrorCode::RhiFailure:
    case RuntimeErrorCode::InternalFailure:
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    return VERNON_RHI_STATUS_INTERNAL_ERROR;
}

[[nodiscard]] constexpr VernonRhiStatus toVernonRhiStatus(ProviderError error) noexcept {
    return toVernonRhiStatus(toRuntimeError(error));
}

[[nodiscard]] constexpr VernonStatus toVernonStatus(RuntimeError error) noexcept {
    switch (error.code) {
    case RuntimeErrorCode::InvalidArgument:
        return VERNON_STATUS_INVALID_ARGUMENT;
    case RuntimeErrorCode::ParseFailure:
        return VERNON_STATUS_PARSE_ERROR;
    case RuntimeErrorCode::VerificationFailure:
        return VERNON_STATUS_VERIFICATION_ERROR;
    case RuntimeErrorCode::Unsupported:
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    case RuntimeErrorCode::ResourceExhausted:
    case RuntimeErrorCode::RhiFailure:
    case RuntimeErrorCode::LifecycleFailure:
    case RuntimeErrorCode::InternalFailure:
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    return VERNON_STATUS_INTERNAL_ERROR;
}

[[nodiscard]] constexpr VernonStatus toVernonStatus(LifecycleError error) noexcept {
    return toVernonStatus(toRuntimeError(error));
}

[[nodiscard]] constexpr VernonStatus toVernonStatus(RhiError error) noexcept {
    return toVernonStatus(toRuntimeError(error));
}

[[nodiscard]] constexpr VernonStatus toVernonStatus(ResourceAccessError error) noexcept {
    return toVernonStatus(toRuntimeError(error));
}

[[nodiscard]] constexpr VernonStatus toVernonStatus(ArithmeticError error) noexcept {
    return toVernonStatus(toRuntimeError(error));
}

[[nodiscard]] constexpr VernonStatus toVernonStatus(AllocationError error) noexcept {
    return toVernonStatus(toRuntimeError(error));
}

[[nodiscard]] constexpr VernonStatus toVernonStatus(ProviderError error) noexcept {
    return toVernonStatus(toRuntimeError(error));
}

struct ErrorView {
    ErrorDomain domain{};
    std::uint8_t code{};
    ErrorContext context{};
};

[[nodiscard]] constexpr ErrorView errorView(LifecycleError error) noexcept {
    return {ErrorDomain::Lifecycle, static_cast<std::uint8_t>(error.code), error.context};
}

[[nodiscard]] constexpr ErrorView errorView(RhiError error) noexcept {
    return {ErrorDomain::Rhi, static_cast<std::uint8_t>(error.code), error.context};
}

[[nodiscard]] constexpr ErrorView errorView(RuntimeError error) noexcept {
    return {ErrorDomain::Runtime, static_cast<std::uint8_t>(error.code), error.context};
}

[[nodiscard]] constexpr ErrorView errorView(ResourceAccessError error) noexcept {
    return {ErrorDomain::ResourceAccess, static_cast<std::uint8_t>(error.code), error.context};
}

[[nodiscard]] constexpr ErrorView errorView(ArithmeticError error) noexcept {
    return {ErrorDomain::Arithmetic, static_cast<std::uint8_t>(error), {}};
}

[[nodiscard]] constexpr ErrorView errorView(AllocationError error) noexcept {
    return {ErrorDomain::Allocation, static_cast<std::uint8_t>(error), {}};
}

[[nodiscard]] constexpr ErrorView errorView(ProviderError error) noexcept {
    return {ErrorDomain::Provider, static_cast<std::uint8_t>(error.code), error.context};
}

namespace detail {

inline void appendCharacter(char *output, std::size_t capacity, std::size_t &written, char value) noexcept {
    if (capacity != 0 && written + 1 < capacity)
        output[written] = value;
    ++written;
}

inline void appendText(char *output, std::size_t capacity, std::size_t &written, const char *text,
                       std::size_t maximum) noexcept {
    if (!text)
        return;
    for (std::size_t index = 0; index < maximum && text[index] != '\0'; ++index)
        appendCharacter(output, capacity, written, text[index]);
}

inline void appendUnsigned(char *output, std::size_t capacity, std::size_t &written, std::uint64_t value) noexcept {
    char digits[20];
    std::size_t count = 0;
    do {
        digits[count++] = static_cast<char>('0' + value % 10);
        value /= 10;
    } while (value != 0);
    while (count != 0)
        appendCharacter(output, capacity, written, digits[--count]);
}

} // namespace detail

[[nodiscard]] inline std::size_t renderEmergencyDiagnostic(ErrorView error, char *output,
                                                           std::size_t capacity) noexcept {
    std::size_t written = 0;
    detail::appendText(output, capacity, written, "domain=", 7);
    detail::appendUnsigned(output, capacity, written, static_cast<std::uint8_t>(error.domain));
    detail::appendText(output, capacity, written, " code=", 6);
    detail::appendUnsigned(output, capacity, written, error.code);
    if (error.context.operation) {
        detail::appendText(output, capacity, written, " operation=", 11);
        detail::appendText(output, capacity, written, error.context.operation, 64);
    }
    detail::appendText(output, capacity, written, " detail=", 8);
    detail::appendUnsigned(output, capacity, written, error.context.detail);
    detail::appendText(output, capacity, written, " value=", 7);
    detail::appendUnsigned(output, capacity, written, error.context.value);
    if (capacity != 0)
        output[written < capacity ? written : capacity - 1] = '\0';
    return written;
}

template <typename E>
[[nodiscard]] inline std::size_t renderEmergencyDiagnostic(E error, char *output, std::size_t capacity) noexcept {
    return renderEmergencyDiagnostic(errorView(error), output, capacity);
}

} // namespace vernon

#endif
