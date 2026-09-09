#ifndef VERNON_PYTHON_NATIVE_RUNTIME_H
#define VERNON_PYTHON_NATIVE_RUNTIME_H

#include "native_program_autodiff.h"
#include "runtime/program_execution_backend.h"

#include <nanobind/stl/map.h>

#include <mutex>
#include <unordered_map>

struct Runtime {
    explicit Runtime(VernonRuntimeContext *handle, std::shared_ptr<RhiHostState> rhiHost = {})
        : handle(handle), rhiHost(std::move(rhiHost)) {}
    explicit Runtime(VernonRuntimeBackend backend) {
        VernonRuntimeCreateOptions options{};
        options.struct_size = sizeof(options);
        handle = vernonRuntimeCreateWithOptions(backend, &options);
        if (!handle)
            throw std::runtime_error("requested runtime backend is unavailable");
    }
    ~Runtime() { vernonRuntimeDestroy(handle); }

    std::shared_ptr<InternedCpuJit> internCpuJit(const std::string &symbol, SharedCompileResult result,
                                                 VernonCpuEntryPoint entry) {
        std::lock_guard<std::mutex> lock(internedCpuMutex);
        std::weak_ptr<InternedCpuJit> &slot = internedCpuEntries[symbol];
        if (std::shared_ptr<InternedCpuJit> existing = slot.lock())
            return existing;
        auto interned = std::make_shared<InternedCpuJit>();
        interned->result = std::move(result);
        interned->entry = entry;
        slot = interned;
        return interned;
    }

    void registerInternedCpuStages(const nb::list &compiledStages, const char *invalidMetadata,
                                   const char *missingEntryPrefix, const char *registerFailurePrefix,
                                   std::vector<SharedCompileResult> &retained,
                                   std::vector<std::pair<std::string, VernonCpuEntryPoint>> &registered,
                                   std::vector<std::shared_ptr<InternedCpuJit>> &interned) {
        const auto rollback = [&]() {
            for (const auto &[symbol, entry] : registered)
                vernonRuntimeUnregisterCpuEntry(handle, {symbol.data(), symbol.size()}, entry);
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
                if (vernonRuntimeRegisterCpuEntry(handle, {symbol.data(), symbol.size()}, internedJit->entry) !=
                    VERNON_STATUS_OK)
                    throw std::runtime_error(std::string(registerFailurePrefix) + "'" + symbol + "'");
                retained.push_back(internedJit->result);
                registered.emplace_back(symbol, internedJit->entry);
                interned.push_back(std::move(internedJit));
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
        VernonProgramBundle *bundle =
            vernonRuntimeLoadProgramBundleWithOptions(handle, data.c_str(), data.size(), &options);
        if (!bundle)
            throw std::runtime_error("cannot load Program bundle: " +
                                     nativeStringView(vernonRuntimeGetLastError(handle)));
        VernonProgramExecutable *executable = vernonRuntimeResolveProgram(bundle, &selector);
        if (!executable) {
            const std::string error = nativeStringView(vernonRuntimeGetLastError(handle));
            vernonRuntimeProgramBundleDestroy(bundle);
            throw std::runtime_error("cannot resolve Program bundle: " + error);
        }
        return std::make_unique<PythonProgramExecutable>(this, handle, bundle, executable);
    }

    std::unique_ptr<PythonProgramExecutable>
    loadInMemoryProgram(const nb::bytes &manifestData, const std::string &directory, const nb::list &compiledStages) {
        std::string error;
        std::vector<SharedCompileResult> retained;
        std::vector<std::pair<std::string, VernonCpuEntryPoint>> registered;
        std::vector<std::shared_ptr<InternedCpuJit>> interned;
        const auto unregister = [&]() {
            for (const auto &[symbol, entry] : registered)
                vernonRuntimeUnregisterCpuEntry(handle, {symbol.data(), symbol.size()}, entry);
        };
        try {
            registerInternedCpuStages(compiledStages, "compiled in-memory Program stage metadata is invalid",
                                      "compiled in-memory Program CPU entry ",
                                      "cannot register in-memory Program CPU entry ", retained, registered, interned);
            VernonProgramBundleLoadOptions options{};
            options.struct_size = sizeof(options);
            options.bundle_directory = directory.c_str();
            VernonProgramBundle *bundle =
                vernonRuntimeLoadProgramBundleWithOptions(handle, manifestData.c_str(), manifestData.size(), &options);
            if (!bundle)
                throw std::runtime_error("cannot load in-memory Program bundle: " +
                                         nativeStringView(vernonRuntimeGetLastError(handle)));
            VernonProgramExecutable *loaded = vernonRuntimeResolveProgram(bundle, nullptr);
            if (!loaded) {
                error = nativeStringView(vernonRuntimeGetLastError(handle));
                vernonRuntimeProgramBundleDestroy(bundle);
                throw std::runtime_error(error);
            }
            auto executable =
                std::make_unique<PythonProgramExecutable>(this, handle, bundle, loaded, std::move(retained));
            executable->internedCpuJits = std::move(interned);
            executable->registeredCpuEntries = std::move(registered);
            return executable;
        } catch (...) {
            unregister();
            throw;
        }
    }

    VernonRuntimeContext *handle{};
    std::shared_ptr<RhiHostState> rhiHost;
    std::mutex internedCpuMutex;
    std::unordered_map<std::string, std::weak_ptr<InternedCpuJit>> internedCpuEntries;
};

RhiHostState *runtimeRhiHost(const Runtime *runtime);

std::unique_ptr<Runtime> createRhiRuntime(RhiHost &host);

#endif
