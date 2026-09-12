#ifndef VERNON_PYTHON_NATIVE_RUNTIME_H
#define VERNON_PYTHON_NATIVE_RUNTIME_H

#include "native_lifecycle_test_hooks.h"
#include "native_program_autodiff.h"
#include "runtime/program_execution_backend.h"

#include <nanobind/stl/map.h>

#include <mutex>
#include <unordered_map>

struct RuntimeContextDeleter {
    void operator()(VernonRuntimeContext *handle) const noexcept {
        if (handle) {
            vernonRuntimeDestroy(handle);
            vernon::python::testing::noteRuntimeDestroyed();
        }
    }
};

struct ProgramBundleDeleter {
    void operator()(VernonProgramBundle *bundle) const noexcept {
        if (bundle)
            vernonRuntimeProgramBundleDestroy(bundle);
    }
};

struct ProgramExecutableDeleter {
    void operator()(VernonProgramExecutable *executable) const noexcept {
        if (executable)
            vernonRuntimeProgramExecutableDestroy(executable);
    }
};

using RuntimeContextOwner = std::unique_ptr<VernonRuntimeContext, RuntimeContextDeleter>;
using ProgramBundleOwner = std::unique_ptr<VernonProgramBundle, ProgramBundleDeleter>;
using ProgramExecutableOwner = std::unique_ptr<VernonProgramExecutable, ProgramExecutableDeleter>;

inline RuntimeContextOwner createRuntimeContext(VernonRuntimeBackend backend) {
    VernonRuntimeCreateOptions options{};
    options.struct_size = sizeof(options);
    RuntimeContextOwner handle(vernonRuntimeCreateWithOptions(backend, &options));
    if (!handle)
        throw std::runtime_error("requested runtime backend is unavailable");
    return handle;
}

struct RuntimeState {
    explicit RuntimeState(RuntimeContextOwner handle, std::shared_ptr<RhiHostState> rhiHost = {})
        : rhiHost(std::move(rhiHost)), context(std::move(handle)) {}

    VernonRuntimeContext *handle() const noexcept { return context.get(); }

    void releaseInternedCpuJit(const std::string &symbol, const std::shared_ptr<InternedCpuJit> &jit) {
        std::lock_guard<std::mutex> lock(internedCpuMutex);
        const auto found = internedCpuEntries.find(symbol);
        if (found == internedCpuEntries.end())
            return;
        std::shared_ptr<InternedCpuJit> current = found->second.lock();
        if (current == jit && current.use_count() == 2)
            internedCpuEntries.erase(found);
    }

    std::shared_ptr<RhiHostState> rhiHost;
    RuntimeContextOwner context;
    std::mutex internedCpuMutex;
    std::unordered_map<std::string, std::weak_ptr<InternedCpuJit>> internedCpuEntries;
};

struct Runtime {
    explicit Runtime(RuntimeContextOwner handle, std::shared_ptr<RhiHostState> rhiHost = {})
        : state(std::make_shared<RuntimeState>(std::move(handle), std::move(rhiHost))) {}
    explicit Runtime(VernonRuntimeBackend backend) : Runtime(createRuntimeContext(backend)) {}

    std::shared_ptr<InternedCpuJit> internCpuJit(const std::string &symbol, SharedCompileResult result,
                                                 VernonCpuEntryPoint entry) {
        std::lock_guard<std::mutex> lock(state->internedCpuMutex);
        const auto found = state->internedCpuEntries.find(symbol);
        if (found != state->internedCpuEntries.end()) {
            if (std::shared_ptr<InternedCpuJit> existing = found->second.lock())
                return existing;
        }
        auto interned = std::make_shared<InternedCpuJit>();
        interned->result = std::move(result);
        interned->entry = entry;
        if (found == state->internedCpuEntries.end())
            state->internedCpuEntries.emplace(symbol, interned);
        else
            found->second = interned;
        return interned;
    }

    void registerInternedCpuStages(const nb::list &compiledStages, const char *invalidMetadata,
                                   const char *missingEntryPrefix, const char *registerFailurePrefix,
                                   std::vector<SharedCompileResult> &retained,
                                   std::vector<std::pair<std::string, VernonCpuEntryPoint>> &registered,
                                   std::vector<std::shared_ptr<InternedCpuJit>> &interned) {
        size_t registeredCount = 0;
        const auto rollback = [&]() {
            for (size_t index = 0; index < registeredCount; ++index) {
                const auto &[symbol, entry] = registered[index];
                vernonRuntimeUnregisterCpuEntry(state->handle(), {symbol.data(), symbol.size()}, entry);
            }
            for (size_t index = 0; index < registered.size() && index < interned.size(); ++index)
                state->releaseInternedCpuJit(registered[index].first, interned[index]);
            registered.clear();
            retained.clear();
            interned.clear();
        };
        retained.reserve(compiledStages.size());
        registered.reserve(compiledStages.size());
        interned.reserve(compiledStages.size());
        try {
            for (nb::handle item : compiledStages) {
                nb::tuple stage = nb::cast<nb::tuple>(item);
                if (stage.size() != 3)
                    throw std::invalid_argument(invalidMetadata);
                const std::string symbol = nb::cast<std::string>(stage[0]);
                const std::string entryName = nb::cast<std::string>(stage[1]);
                const CompiledProgram &program = nb::cast<const CompiledProgram &>(stage[2]);
                program.requireSuccess();
                VernonCpuEntryPoint entry =
                    vernonCompileResultGetCpuEntry(program.result.get(), entryName.data(), entryName.size());
                if (!entry)
                    throw std::runtime_error(std::string(missingEntryPrefix) + "'" + entryName + "' was not found");
                std::shared_ptr<InternedCpuJit> internedJit = internCpuJit(symbol, program.result, entry);
                retained.push_back(internedJit->result);
                registered.emplace_back(symbol, internedJit->entry);
                interned.push_back(internedJit);
                if (vernonRuntimeRegisterCpuEntry(state->handle(), {symbol.data(), symbol.size()},
                                                  internedJit->entry) != VERNON_STATUS_OK)
                    throw std::runtime_error(std::string(registerFailurePrefix) + "'" + symbol + "'");
                ++registeredCount;
            }
        } catch (...) {
            rollback();
            throw;
        }
    }

    std::unique_ptr<PythonProgramExecutable> loadProgramAsset(const nb::bytes &data, const std::string &directory,
                                                              const nb::list &specializationRows) {
        VernonProgramBundleLoadOptions options{};
        options.struct_size = sizeof(options);
        options.bundle_directory = directory.c_str();
        std::vector<std::string> names;
        std::vector<VernonProgramSpecialization> specializations;
        names.reserve(specializationRows.size());
        specializations.reserve(specializationRows.size());
        for (nb::handle rowHandle : specializationRows) {
            const nb::dict row = nb::cast<nb::dict>(rowHandle);
            if (!row.contains("name") || !row.contains("value"))
                throw std::invalid_argument("Program specialization row is incomplete");
            const nb::dict value = nb::cast<nb::dict>(row["value"]);
            if (!value.contains("tag") || !value.contains("value"))
                throw std::invalid_argument("Program specialization value is incomplete");
            names.push_back(nb::cast<std::string>(row["name"]));
            const std::string tag = nb::cast<std::string>(value["tag"]);
            VernonProgramSpecialization specialization{};
            specialization.struct_size = sizeof(specialization);
            specialization.name = {names.back().data(), names.back().size()};
            if (tag == "bool") {
                specialization.kind = VERNON_PROGRAM_SPECIALIZATION_BOOL;
                specialization.value.boolean_value = nb::cast<bool>(value["value"]) ? 1 : 0;
            } else if (tag == "i32") {
                specialization.kind = VERNON_PROGRAM_SPECIALIZATION_I32;
                specialization.value.i32_value = nb::cast<int32_t>(value["value"]);
            } else if (tag == "u32") {
                specialization.kind = VERNON_PROGRAM_SPECIALIZATION_U32;
                specialization.value.u32_value = nb::cast<uint32_t>(value["value"]);
            } else if (tag == "f32") {
                specialization.kind = VERNON_PROGRAM_SPECIALIZATION_F32;
                specialization.value.f32_value = nb::cast<float>(value["value"]);
            } else if (tag == "f64") {
                specialization.kind = VERNON_PROGRAM_SPECIALIZATION_F64;
                specialization.value.f64_value = nb::cast<double>(value["value"]);
            } else {
                throw std::invalid_argument("Program specialization tag is invalid");
            }
            specializations.push_back(specialization);
        }
        const VernonProgramVariantSelector selector{
            sizeof(VernonProgramVariantSelector), specializations.data(), specializations.size(), {}};
        ProgramBundleOwner bundle(
            vernonRuntimeLoadProgramBundleWithOptions(state->handle(), data.c_str(), data.size(), &options));
        if (!bundle)
            throw std::runtime_error("cannot load Program bundle: " +
                                     nativeStringView(vernonRuntimeGetLastError(state->handle())));
        ProgramExecutableOwner executable(vernonRuntimeResolveProgram(bundle.get(), &selector));
        if (!executable) {
            const std::string error = nativeStringView(vernonRuntimeGetLastError(state->handle()));
            throw std::runtime_error("cannot resolve Program bundle: " + error);
        }
        auto result = std::make_unique<PythonProgramExecutable>(state, state->handle(), bundle.get(), executable.get());
        bundle.release();
        executable.release();
        return result;
    }

    std::unique_ptr<PythonProgramExecutable>
    loadInMemoryProgram(const nb::bytes &manifestData, const std::string &directory, const nb::list &compiledStages) {
        std::string error;
        std::vector<SharedCompileResult> retained;
        std::vector<std::pair<std::string, VernonCpuEntryPoint>> registered;
        std::vector<std::shared_ptr<InternedCpuJit>> interned;
        const auto unregister = [&]() {
            for (size_t index = 0; index < registered.size(); ++index) {
                const auto &[symbol, entry] = registered[index];
                vernonRuntimeUnregisterCpuEntry(state->handle(), {symbol.data(), symbol.size()}, entry);
                if (index < interned.size())
                    state->releaseInternedCpuJit(symbol, interned[index]);
            }
            registered.clear();
            retained.clear();
            interned.clear();
        };
        try {
            registerInternedCpuStages(compiledStages, "compiled in-memory Program stage metadata is invalid",
                                      "compiled in-memory Program CPU entry ",
                                      "cannot register in-memory Program CPU entry ", retained, registered, interned);
            VernonProgramBundleLoadOptions options{};
            options.struct_size = sizeof(options);
            options.bundle_directory = directory.c_str();
            ProgramBundleOwner bundle(vernonRuntimeLoadProgramBundleWithOptions(state->handle(), manifestData.c_str(),
                                                                                manifestData.size(), &options));
            if (!bundle)
                throw std::runtime_error("cannot load in-memory Program bundle: " +
                                         nativeStringView(vernonRuntimeGetLastError(state->handle())));
            ProgramExecutableOwner loaded(vernonRuntimeResolveProgram(bundle.get(), nullptr));
            if (!loaded) {
                error = nativeStringView(vernonRuntimeGetLastError(state->handle()));
                throw std::runtime_error(error);
            }
            auto executable = std::make_unique<PythonProgramExecutable>(state, state->handle(), bundle.get(),
                                                                        loaded.get(), std::move(retained));
            executable->internedCpuJits = std::move(interned);
            executable->registeredCpuEntries = std::move(registered);
            bundle.release();
            loaded.release();
            return executable;
        } catch (...) {
            unregister();
            throw;
        }
    }

    std::shared_ptr<RuntimeState> state;
};

RhiHostState *runtimeRhiHost(const RuntimeState *runtime);

std::unique_ptr<Runtime> createRhiRuntime(RhiHost &host);

#endif
