#ifndef VERNON_RHI_PUBLIC_C_BOUNDARY_H
#define VERNON_RHI_PUBLIC_C_BOUNDARY_H

#include "backend_dispatch.h"

#include <new>
#include <string_view>
#include <utility>

namespace vernon::rhi {

inline RhiError publicStatusError(VernonRhiStatus status, const char *operation) noexcept {
    switch (status) {
    case VERNON_RHI_STATUS_INVALID_ARGUMENT:
        return {RhiErrorCode::InvalidArgument, {operation, 0, 0}};
    case VERNON_RHI_STATUS_UNSUPPORTED:
        return {RhiErrorCode::Unsupported, {operation, 0, 0}};
    case VERNON_RHI_STATUS_RESOURCE_EXHAUSTED:
        return {RhiErrorCode::ResourceExhausted, {operation, 0, 0}};
    case VERNON_RHI_STATUS_INTERNAL_ERROR:
    default:
        return {RhiErrorCode::BackendFailure, {operation, 0, 0}};
    }
}

inline Result<void, RhiError> publicStatusResult(VernonRhiStatus status, const char *operation) noexcept {
    return status == VERNON_RHI_STATUS_OK ? Result<void, RhiError>{ok()}
                                          : Result<void, RhiError>{err(publicStatusError(status, operation))};
}

inline std::string_view deviceCreationErrorDiagnostic(RhiErrorCode code) noexcept {
    switch (code) {
    case RhiErrorCode::InvalidArgument:
        return "RHI device creation rejected invalid arguments";
    case RhiErrorCode::Unsupported:
        return "RHI device creation is unsupported";
    case RhiErrorCode::ResourceExhausted:
        return "RHI device creation exhausted resources";
    case RhiErrorCode::LifecycleFailure:
        return "RHI device creation was rejected by lifecycle state";
    case RhiErrorCode::BackendFailure:
    default:
        return "RHI backend device creation failed";
    }
}

template <typename Function> VernonRhiStatus publicStatusBoundary(Function &&function) noexcept {
    try {
        auto result = std::forward<Function>(function)();
        return result.isOk() ? VERNON_RHI_STATUS_OK : toVernonRhiStatus(result.error());
    } catch (const std::bad_alloc &) {
        return VERNON_RHI_STATUS_RESOURCE_EXHAUSTED;
    } catch (...) {
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
}

template <typename Handle, typename Function>
Handle publicHandleBoundary(Function &&function, Handle invalid) noexcept {
    setDeviceCreationError({});
    try {
        auto result = std::forward<Function>(function)();
        if (result.isOk())
            return std::move(result).value();
        if (deviceCreationError().size == 0)
            setDeviceCreationError(deviceCreationErrorDiagnostic(result.error().code));
    } catch (const std::bad_alloc &) {
        setDeviceCreationError("RHI device creation exhausted resources");
    } catch (...) {
        setDeviceCreationError("RHI device creation failed at the C boundary");
    }
    return invalid;
}

template <typename View, typename Function> View publicViewBoundary(Function &&function, View fallback = {}) noexcept {
    try {
        auto result = std::forward<Function>(function)();
        return result.isOk() ? std::move(result).value() : fallback;
    } catch (...) {
        return fallback;
    }
}

template <typename Value, typename Function>
Value publicQueryBoundary(Function &&function, Value fallback = {}) noexcept {
    try {
        return std::forward<Function>(function)();
    } catch (...) {
        return fallback;
    }
}

template <typename Function> void publicDestroyBoundary(Function &&function) noexcept {
    try {
        auto result = std::forward<Function>(function)();
        if (result.isOk())
            return;
        switch (result.error().code) {
        case RhiErrorCode::InvalidArgument:
        case RhiErrorCode::LifecycleFailure:
            // The legacy void destroy API cannot report admission refusals.
            return;
        case RhiErrorCode::Unsupported:
        case RhiErrorCode::ResourceExhausted:
        case RhiErrorCode::SynchronizationFailure:
        case RhiErrorCode::BackendFailure:
            resultContractViolation();
        }
    } catch (...) {
        resultContractViolation();
    }
}

} // namespace vernon::rhi

#endif
