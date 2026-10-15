#include "VernonRuntime.h"
#include "runtime/capi/boundary.h"
#include "runtime/program_execution/submission_queue.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_state.h"
#include "runtime/tensor_bridge.h"

#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

using namespace vernon::runtime;
using namespace vernon::runtime::capi;

namespace {

vernon::Result<std::unique_ptr<VernonRuntimeContext>, vernon::RuntimeError> createRuntimeContext() noexcept {
    auto created = VernonRuntimeContext::create();
    if (created.isErr())
        return created;
    auto generation = reserveDiagnosticGeneration();
    if (generation.isErr())
        return vernon::Result<std::unique_ptr<VernonRuntimeContext>, vernon::RuntimeError>{
            vernon::err(std::move(generation).error())};
    created.value()->diagnosticGeneration = generation.value();
    return created;
}

vernon::runtime::program_execution::SubmissionBounds submissionBounds(const VernonRuntimeCreateOptions *options) {
    constexpr vernon::runtime::program_execution::SubmissionBounds defaults =
        vernon::runtime::program_execution::defaultSubmissionBounds;
    if (!options)
        return defaults;
    return {options->transient_budget_bytes ? options->transient_budget_bytes : defaults.transientBudgetBytes,
            options->max_pending_submissions ? options->max_pending_submissions : defaults.maxPendingSubmissions};
}

vernon::Result<std::unique_ptr<VernonRuntimeContext>, vernon::RuntimeError>
createRuntimeContextForBackend(VernonRuntimeBackend backend, const VernonRuntimeCreateOptions *options) {
    using Result = vernon::Result<std::unique_ptr<VernonRuntimeContext>, vernon::RuntimeError>;
    if (options && options->struct_size < sizeof(VernonRuntimeCreateOptions))
        return Result{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                                       {"create_runtime_context", options->struct_size, 0}})};
    auto created = createRuntimeContext();
    if (created.isErr())
        return created;
    created.value()->backend = backend;
    auto initialized = initializeBackend(*created.value(), options ? options->device_index : 0);
    if (initialized.isErr())
        return Result{vernon::err(std::move(initialized).error())};
    created.value()->submissions.configure(submissionBounds(options), created.value()->rhiDevice);
    return created;
}

vernon::Result<std::unique_ptr<VernonRuntimeContext>, vernon::RuntimeError>
createRuntimeContextForRhiDevice(VernonRuntimeBackend backend, VernonRhiDevice device,
                                 const VernonRuntimeCreateOptions *options) {
    using Result = vernon::Result<std::unique_ptr<VernonRuntimeContext>, vernon::RuntimeError>;
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX ||
        (options && (options->struct_size < sizeof(VernonRuntimeCreateOptions) || options->device_index)))
        return Result{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                                       {"create_runtime_context_for_rhi_device", 0, 0}})};
    auto created = createRuntimeContext();
    if (created.isErr())
        return created;
    created.value()->backend = backend;
    auto initialized = initializeBackendForRhiDevice(*created.value(), device);
    if (initialized.isErr())
        return Result{vernon::err(std::move(initialized).error())};
    created.value()->submissions.configure(submissionBounds(options), created.value()->rhiDevice);
    return created;
}

template <typename Output, typename Reference>
VernonStatus referenceRhiObject(VernonRuntimeContext *context, Output *output, std::string_view invalid,
                                Reference &&reference) {
    return runtimeStatusBoundary(context, [&] {
        if (!context)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto contextPin = context->operations.tryPin();
        if (contextPin.isErr())
            return vernon::toVernonStatus(contextPin.error());
        if (!output)
            return fail(context, invalid);
        auto referenced = std::forward<Reference>(reference)(*context);
        if (referenced.isErr())
            return publishRuntimeError(context, std::move(referenced).error());
        *output = referenced.value();
        return VERNON_STATUS_OK;
    });
}

} // namespace

VernonValueLayoutView vernonRuntimeGetScalarValueLayout(VernonDataType dtype) {
    return runtimeValueBoundary(nullptr, VernonValueLayoutView{}, [&] {
        static constexpr VernonValueLeafView leaves[] = {
            {VERNON_DATA_BOOL, 1, 0}, {VERNON_DATA_I32, 1, 0}, {VERNON_DATA_U32, 1, 0}, {VERNON_DATA_F16, 1, 0},
            {VERNON_DATA_F32, 1, 0},  {VERNON_DATA_F64, 1, 0}, {VERNON_DATA_U8, 1, 0},
        };
        static constexpr const char *hashes[] = {
            "3ca886485debde52d9dae8389b51daf52bf265e898ece94a57241849eea52fc7",
            "5221c466df6b1fe9046f6d2e7597efdc98f5a3aa66bb176fe65d02de0c39607f",
            "5d0250c80dab299ac915d5d0d21170d208e2f4d97216c89d0263a3c2d3bf5dc8",
            "937b700417d47a346038256ddb7c3ed7062303c531efba4d6dfd5e21583deec4",
            "cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b",
            "8f6e354f03614c96a53ba7e4ff053d14f7ce2c8f5066a5193d43f28db91268da",
            "",
        };
        auto sizeResult = dataTypeSize(dtype);
        const size_t index = static_cast<size_t>(dtype);
        if (sizeResult.isErr() || index >= std::size(leaves) || !hashes[index][0])
            return VernonValueLayoutView{};
        const size_t size = std::move(sizeResult).value();
        return VernonValueLayoutView{sizeof(VernonValueLayoutView),
                                     static_cast<uint32_t>(size),
                                     static_cast<uint32_t>(size),
                                     {hashes[index], std::strlen(hashes[index])},
                                     &leaves[index],
                                     1};
    });
}

VernonRuntimeCapabilities vernonRuntimeGetCapabilities(VernonRuntimeBackend backend) {
    return runtimeValueBoundary(nullptr, VernonRuntimeCapabilities{}, [&] {
        static thread_local std::string diagnostic;
        diagnostic.clear();
        VernonRuntimeCapabilities result{};
        result.available = probeBackend(backend, diagnostic).isOk();
        if (backend == VERNON_RUNTIME_CPU && result.available) {
            result.supports_compute = 1;
            result.supports_storage_buffers = 1;
        } else if ((backend == VERNON_RUNTIME_CUDA || backend == VERNON_RUNTIME_VULKAN ||
                    backend == VERNON_RUNTIME_DIRECTX12 || backend == VERNON_RUNTIME_METAL) &&
                   result.available) {
            result.supports_compute = result.available;
            result.supports_storage_buffers = result.available;
        } else if (backend == VERNON_RUNTIME_OPENGL || backend == VERNON_RUNTIME_OPENGL_ES) {
            result.supports_graphics = 1;
        }
        result.diagnostic = {diagnostic.data(), diagnostic.size()};
        return result;
    });
}

VernonRuntimeContext *vernonRuntimeCreateWithOptions(VernonRuntimeBackend backend,
                                                     const VernonRuntimeCreateOptions *options) {
    return runtimePointerResultBoundary<VernonRuntimeContext *>(
        nullptr, [&] { return createRuntimeContextForBackend(backend, options); });
}

VernonRuntimeContext *vernonRuntimeCreateForRhiDevice(VernonRuntimeBackend backend, VernonRhiDevice device,
                                                      const VernonRuntimeCreateOptions *options) {
    return runtimePointerResultBoundary<VernonRuntimeContext *>(
        nullptr, [&] { return createRuntimeContextForRhiDevice(backend, device, options); });
}

VernonStatus vernonRuntimeDestroy(VernonRuntimeContext *context) {
    return runtimeStatusBoundary(context, [&] {
        if (!context)
            return VERNON_STATUS_OK;
        auto operations = context->operations.beginDestroy();
        if (operations.isErr())
            return fail(context, "runtime context has an active operation", vernon::toVernonStatus(operations.error()));
        auto close = context->owner.beginClose();
        if (close.isErr()) {
            (void)operations.value().rollback();
            return fail(context, "runtime context still owns live handles", vernon::toVernonStatus(close.error()));
        }
        destroyBackend(*context);
        if (close.value().commit().isErr() || operations.value().commit().isErr())
            vernon::resultContractViolation();
        vernon::runtime::clearInvocationDiagnostic(*context);
        delete context;
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeContextGetSubmissionTelemetry(VernonRuntimeContext *context,
                                                        VernonSubmissionTelemetry *output) {
    return runtimeStatusBoundary(context, [&] {
        if (!context || !output || output->struct_size < sizeof(*output))
            return fail(context, "invalid submission telemetry query");
        *output = context->submissions.telemetry();
        return VERNON_STATUS_OK;
    });
}

VernonStringView vernonRuntimeGetLastError(const VernonRuntimeContext *context) {
    return runtimeValueBoundary(nullptr, VernonStringView{nullptr, 0}, [&] {
        if (!context)
            return VernonStringView{nullptr, 0};
        return vernon::runtime::currentInvocationDiagnosticView(*context);
    });
}

VernonRuntimeCapabilities vernonRuntimeGetContextCapabilities(const VernonRuntimeContext *context) {
    return runtimeValueBoundary(context, VernonRuntimeCapabilities{}, [&] {
        VernonRuntimeCapabilities result{};
        if (!context)
            return result;
        auto contextPin = context->operations.tryPin();
        if (contextPin.isErr())
            return result;
        fillBackendCapabilities(*context, result);
        result.diagnostic = currentInvocationDiagnosticView(*context);
        return result;
    });
}

VernonStatus vernonRuntimeRegisterStaticCpuEntries(const VernonStaticCpuEntry *entries, size_t count) {
    return runtimeResultBoundary(nullptr, [&] { return registerBackendStaticCpuEntries(entries, count); });
}

VernonStatus vernonRuntimeUnregisterStaticCpuEntries(const VernonStaticCpuEntry *entries, size_t count) {
    return runtimeResultBoundary(nullptr, [&] { return unregisterBackendStaticCpuEntries(entries, count); });
}

VernonStatus vernonRuntimeRegisterCpuEntry(VernonRuntimeContext *context, VernonStringView symbol,
                                           VernonCpuEntryPoint entryPoint) {
    return runtimeStatusBoundary(context, [&] {
        if (!context || context->backend != VERNON_RUNTIME_CPU || !symbol.data || !symbol.size || !entryPoint)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto contextPin = context->operations.tryPin();
        if (contextPin.isErr())
            return vernon::toVernonStatus(contextPin.error());
        std::lock_guard<std::mutex> lock(context->cpuEntriesMutex);
        auto [found, inserted] =
            context->cpuEntries.emplace(std::string(symbol.data, symbol.size), std::make_pair(entryPoint, size_t{1}));
        if (!inserted) {
            if (found->second.first != entryPoint)
                return VERNON_STATUS_INVALID_ARGUMENT;
            ++found->second.second;
        }
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeUnregisterCpuEntry(VernonRuntimeContext *context, VernonStringView symbol,
                                             VernonCpuEntryPoint entryPoint) {
    return runtimeStatusBoundary(context, [&] {
        if (!context || !symbol.data || !symbol.size || !entryPoint)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto contextPin = context->operations.tryPin();
        if (contextPin.isErr())
            return vernon::toVernonStatus(contextPin.error());
        std::lock_guard<std::mutex> lock(context->cpuEntriesMutex);
        const auto found = context->cpuEntries.find(std::string(symbol.data, symbol.size));
        if (found == context->cpuEntries.end() || found->second.first != entryPoint)
            return VERNON_STATUS_INVALID_ARGUMENT;
        if (!--found->second.second)
            context->cpuEntries.erase(found);
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeReferenceRhiBuffer(VernonRuntimeContext *context, VernonRhiBuffer buffer, uint64_t offset,
                                             uint64_t size, VernonRuntimeProviderResourceReference *output) {
    return referenceRhiObject(context, output, "invalid RHI buffer reference", [&](VernonRuntimeContext &pinned) {
        return referenceBackendRhiBuffer(pinned, buffer, offset, size);
    });
}

VernonStatus vernonRuntimeReferenceRhiImageView(VernonRuntimeContext *context, VernonRhiImageView view,
                                                VernonRuntimeProviderResourceReference *output) {
    return referenceRhiObject(context, output, "invalid RHI image view reference",
                              [&](VernonRuntimeContext &pinned) { return referenceBackendRhiImageView(pinned, view); });
}

VernonStatus vernonRuntimeReferenceRhiSampler(VernonRuntimeContext *context, VernonRhiSampler sampler,
                                              VernonRuntimeProviderResourceReference *output) {
    return referenceRhiObject(context, output, "invalid RHI sampler reference", [&](VernonRuntimeContext &pinned) {
        return referenceBackendRhiSampler(pinned, sampler);
    });
}

VernonStatus vernonRuntimeReferenceRhiCommandEncoder(VernonRuntimeContext *context, VernonRhiCommandEncoder encoder,
                                                     VernonRuntimeProviderObject *output) {
    return referenceRhiObject(
        context, output, "invalid RHI command encoder reference",
        [&](VernonRuntimeContext &pinned) { return referenceBackendCommandEncoder(pinned, encoder); });
}
