#ifndef VERNON_RUNTIME_CAPI_BOUNDARY_H
#define VERNON_RUNTIME_CAPI_BOUNDARY_H

#include "VernonError.hpp"
#include "VernonRuntime.h"
#include "runtime/runtime_state.h"

#include <cstdint>
#include <new>
#include <string_view>
#include <utility>

namespace vernon::runtime::capi {

VernonStatus fail(VernonRuntimeContext *context, std::string_view error,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) noexcept;
VernonStatus publishRuntimeError(VernonRuntimeContext *context, vernon::RuntimeError error) noexcept;
VernonRuntimeOperationStatus runtimeOperationStatus(vernon::RuntimeErrorCode code) noexcept;
// Each Runtime context takes a process-unique generation, so a stale thread-local diagnostic never matches a context
// later created at the same address.
vernon::Result<uint64_t, vernon::RuntimeError> reserveDiagnosticGeneration() noexcept;

// Every C API entry runs its body through one of these boundaries: no C++ exception crosses the ABI, and failures
// leave their diagnostic on the context.

template <typename Callback>
VernonStatus runtimeResultBoundary(VernonRuntimeContext *context, Callback &&callback) noexcept {
    try {
        RuntimeDiagnosticScope diagnostic(context);
        auto result = std::forward<Callback>(callback)();
        return result.isOk() ? VERNON_STATUS_OK : publishRuntimeError(context, std::move(result).error());
    } catch (const std::bad_alloc &) {
        return publishRuntimeError(
            context, {vernon::RuntimeErrorCode::ResourceExhausted, {"Runtime boundary allocation", 0, 0}});
    } catch (...) {
        return publishRuntimeError(context,
                                   {vernon::RuntimeErrorCode::InternalFailure, {"Runtime boundary exception", 0, 0}});
    }
}

template <typename Callback>
VernonStatus runtimeStatusBoundary(VernonRuntimeContext *context, Callback &&callback) noexcept {
    try {
        RuntimeDiagnosticScope diagnostic(context);
        return std::forward<Callback>(callback)();
    } catch (const std::bad_alloc &) {
        return publishRuntimeError(
            context, {vernon::RuntimeErrorCode::ResourceExhausted, {"Runtime boundary allocation", 0, 0}});
    } catch (...) {
        return publishRuntimeError(context,
                                   {vernon::RuntimeErrorCode::InternalFailure, {"Runtime boundary exception", 0, 0}});
    }
}

template <typename Pointer, typename Callback>
Pointer runtimePointerBoundary(VernonRuntimeContext *context, Callback &&callback) noexcept {
    try {
        RuntimeDiagnosticScope diagnostic(context);
        return std::forward<Callback>(callback)();
    } catch (const std::bad_alloc &) {
        (void)publishRuntimeError(context,
                                  {vernon::RuntimeErrorCode::ResourceExhausted, {"Runtime boundary allocation", 0, 0}});
    } catch (...) {
        (void)publishRuntimeError(context,
                                  {vernon::RuntimeErrorCode::InternalFailure, {"Runtime boundary exception", 0, 0}});
    }
    return nullptr;
}

template <typename Pointer, typename Callback>
Pointer runtimePointerResultBoundary(VernonRuntimeContext *context, Callback &&callback) noexcept {
    return runtimePointerBoundary<Pointer>(context, [&]() -> Pointer {
        auto result = std::forward<Callback>(callback)();
        if (result.isErr()) {
            (void)publishRuntimeError(context, std::move(result).error());
            return nullptr;
        }
        return std::move(result).value().release();
    });
}

template <typename Pointer, typename Callback>
VernonRuntimeOperationStatus runtimeHandleResultBoundary(VernonRuntimeContext *context, Pointer **output,
                                                         Callback &&callback) noexcept {
    if (output)
        *output = nullptr;
    try {
        RuntimeDiagnosticScope diagnostic(context);
        if (!output) {
            (void)fail(context, "Runtime result output is null");
            return VERNON_RUNTIME_OPERATION_INVALID_ARGUMENT;
        }
        auto result = std::forward<Callback>(callback)();
        if (result.isErr()) {
            vernon::RuntimeError error = std::move(result).error();
            const VernonStringView current =
                context ? currentInvocationDiagnosticView(*context) : VernonStringView{nullptr, 0};
            if (!current.size)
                (void)publishRuntimeError(context, error);
            return runtimeOperationStatus(error.code);
        }
        *output = std::move(result).value();
        return VERNON_RUNTIME_OPERATION_OK;
    } catch (const std::bad_alloc &) {
        const vernon::RuntimeError error{vernon::RuntimeErrorCode::ResourceExhausted,
                                         {"Runtime boundary allocation", 0, 0}};
        (void)publishRuntimeError(context, error);
        return VERNON_RUNTIME_OPERATION_RESOURCE_EXHAUSTED;
    } catch (...) {
        const vernon::RuntimeError error{vernon::RuntimeErrorCode::InternalFailure,
                                         {"Runtime boundary exception", 0, 0}};
        (void)publishRuntimeError(context, error);
        return VERNON_RUNTIME_OPERATION_INTERNAL_FAILURE;
    }
}

template <typename Value, typename Callback>
Value runtimeValueBoundary(const VernonRuntimeContext *context, Value fallback, Callback &&callback) noexcept {
    auto *mutableContext = const_cast<VernonRuntimeContext *>(context);
    try {
        RuntimeDiagnosticScope diagnostic(context);
        return std::forward<Callback>(callback)();
    } catch (const std::bad_alloc &) {
        (void)publishRuntimeError(mutableContext,
                                  {vernon::RuntimeErrorCode::ResourceExhausted, {"Runtime boundary allocation", 0, 0}});
    } catch (...) {
        (void)publishRuntimeError(mutableContext,
                                  {vernon::RuntimeErrorCode::InternalFailure, {"Runtime boundary exception", 0, 0}});
    }
    return fallback;
}

template <typename Callback> void runtimeVoidBoundary(VernonRuntimeContext *context, Callback &&callback) noexcept {
    try {
        std::forward<Callback>(callback)();
    } catch (const std::bad_alloc &) {
        (void)publishRuntimeError(context,
                                  {vernon::RuntimeErrorCode::ResourceExhausted, {"Runtime boundary allocation", 0, 0}});
    } catch (...) {
        (void)publishRuntimeError(context,
                                  {vernon::RuntimeErrorCode::InternalFailure, {"Runtime boundary exception", 0, 0}});
    }
}

} // namespace vernon::runtime::capi

#endif
