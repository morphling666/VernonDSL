#include "VernonRuntime.h"
#include "runtime/capi/boundary.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_state.h"
#include "runtime/tensor_bridge.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace vernon::runtime;
using namespace vernon::runtime::capi;

namespace {

struct HostModuleRegistration {
    const VhModule *module;
    size_t references;
};

// Per id, registrations oldest first; lookups take the back so the newest live table wins.
struct HostModuleRegistry {
    std::mutex mutex;
    std::unordered_map<std::string, std::vector<HostModuleRegistration>> modules;
};

// Created on first use, because cooked objects register from static constructors that may run before this file's
// globals are initialized, and leaked, because their static destructors may unregister after Runtime statics are
// destroyed.
HostModuleRegistry &hostModuleRegistry() {
    static auto *registry = new HostModuleRegistry;
    return *registry;
}

bool validHostModule(const VernonStaticHostModule *module) {
    return module && module->id.data && module->id.size && module->module;
}

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
    return created;
}

} // namespace

namespace vernon::runtime {

RuntimeResult<std::unique_ptr<VernonRuntimeContext>>
createRuntimeContextForHal(VernonRuntimeBackend backend,
                           const std::function<RuntimeResult<void>(VernonRuntimeContext &)> &initialize) {
    using Result = RuntimeResult<std::unique_ptr<VernonRuntimeContext>>;
    auto created = createRuntimeContext();
    if (created.isErr())
        return created;
    created.value()->backend = backend;
    auto initialized = initialize(*created.value());
    if (initialized.isErr())
        return Result{vernon::err(std::move(initialized).error())};
    return created;
}

} // namespace vernon::runtime

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
        } else if ((backend == VERNON_RUNTIME_CUDA || backend == VERNON_RUNTIME_VULKAN) && result.available) {
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

VernonStatus vernonRuntimeRegisterStaticHostModule(const VernonStaticHostModule *module) {
    return runtimeStatusBoundary(nullptr, [&] {
        if (!validHostModule(module))
            return VERNON_STATUS_INVALID_ARGUMENT;
        HostModuleRegistry &registry = hostModuleRegistry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        auto &registrations = registry.modules[std::string(module->id.data, module->id.size)];
        const auto found = std::find_if(registrations.begin(), registrations.end(), [&](const auto &registration) {
            return registration.module == module->module;
        });
        if (found != registrations.end()) {
            ++found->references;
        } else {
            registrations.push_back({module->module, 1});
        }
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeUnregisterStaticHostModule(const VernonStaticHostModule *module) {
    return runtimeStatusBoundary(nullptr, [&] {
        if (!validHostModule(module))
            return VERNON_STATUS_INVALID_ARGUMENT;
        HostModuleRegistry &registry = hostModuleRegistry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto id = registry.modules.find(std::string(module->id.data, module->id.size));
        if (id == registry.modules.end())
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto &registrations = id->second;
        const auto found = std::find_if(registrations.begin(), registrations.end(), [&](const auto &registration) {
            return registration.module == module->module;
        });
        if (found == registrations.end())
            return VERNON_STATUS_INVALID_ARGUMENT;
        if (--found->references == 0)
            registrations.erase(found);
        if (registrations.empty())
            registry.modules.erase(id);
        return VERNON_STATUS_OK;
    });
}

const VhModule *vernonRuntimeFindHostModule(VernonStringView id) {
    return runtimeValueBoundary(nullptr, static_cast<const VhModule *>(nullptr), [&]() -> const VhModule * {
        if (!id.data || !id.size)
            return nullptr;
        HostModuleRegistry &registry = hostModuleRegistry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto found = registry.modules.find(std::string(id.data, id.size));
        return found == registry.modules.end() ? nullptr : found->second.back().module;
    });
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