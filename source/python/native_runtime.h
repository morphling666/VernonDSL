#ifndef VERNON_PYTHON_NATIVE_RUNTIME_H
#define VERNON_PYTHON_NATIVE_RUNTIME_H

#include "native_program_autodiff.h"
#include "native_stage.h"
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

    std::unique_ptr<PythonStageExecutable> load(const nb::bytes &artifact, const std::string &reflection,
                                                const std::string &entry) {
        VernonStageExecutable *stage =
            vernonRuntimeLoadArtifact(handle, artifact.c_str(), artifact.size(), reflection.data(), reflection.size(),
                                      entry.data(), entry.size());
        if (!stage)
            throw std::runtime_error("cannot load compute pipeline: " +
                                     nativeStringView(vernonRuntimeGetLastError(handle)));
        return std::make_unique<PythonStageExecutable>(handle, stage);
    }

    std::unique_ptr<PythonStageExecutable> loadCpuEntry(const CompiledProgram &program, const std::string &entry) {
        program.requireSuccess();
        if (program.target != VERNON_TARGET_CPU)
            throw std::runtime_error("CPU entries can be loaded only from CPU compiled programs");
        if (entry.empty())
            throw std::runtime_error("CPU entry name must not be empty");
        VernonCpuEntryPoint entryPoint =
            vernonCompileResultGetCpuEntry(program.result.get(), entry.data(), entry.size());
        if (!entryPoint)
            throw std::runtime_error("CPU entry '" + entry + "' was not found in compiled program");
        const std::string reflection = program.reflection();
        VernonStageExecutable *stage = vernonRuntimeLoadCpuEntry(handle, entryPoint, reflection.data(),
                                                                 reflection.size(), entry.data(), entry.size());
        if (!stage)
            throw std::runtime_error("cannot load CPU entry: " + nativeStringView(vernonRuntimeGetLastError(handle)));
        return std::make_unique<PythonStageExecutable>(handle, stage, program.result);
    }

    std::unique_ptr<PythonProgramExecutable> loadProgramAsset(const nb::bytes &data, const std::string &directory,
                                                              const std::vector<std::string> &features) {
        VernonProgramBundleLoadOptions options{};
        options.struct_size = sizeof(options);
        options.bundle_directory = directory.c_str();
        std::vector<const char *> names;
        for (const std::string &feature : features)
            names.push_back(feature.c_str());
        VernonProgramBundle *bundle =
            vernonRuntimeLoadProgramBundleWithOptions(handle, data.c_str(), data.size(), &options);
        if (!bundle)
            throw std::runtime_error("cannot load Program bundle: " +
                                     nativeStringView(vernonRuntimeGetLastError(handle)));
        VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(bundle, {names.data(), names.size()});
        if (!pipeline) {
            const std::string error = nativeStringView(vernonRuntimeGetLastError(handle));
            vernonRuntimeProgramBundleDestroy(bundle);
            throw std::runtime_error("cannot resolve Program bundle: " + error);
        }
        return std::make_unique<PythonProgramExecutable>(this, handle, bundle, pipeline);
    }

    std::unique_ptr<PythonProgramExecutable>
    loadCanonicalProgram(const nb::bytes &manifestData, const std::string &directory, const nb::list &compiledStages) {
        std::string error;
        std::vector<SharedCompileResult> retained;
        std::vector<std::pair<std::string, VernonCpuEntryPoint>> registered;
        std::vector<std::shared_ptr<InternedCpuJit>> interned;
        const auto unregister = [&]() {
            for (const auto &[symbol, entry] : registered)
                vernonRuntimeUnregisterCpuEntry(handle, {symbol.data(), symbol.size()}, entry);
        };
        try {
            registerInternedCpuStages(compiledStages, "compiled canonical Program stage metadata is invalid",
                                      "compiled canonical Program CPU entry ",
                                      "cannot register canonical Program CPU entry ", retained, registered, interned);
            VernonProgramBundleLoadOptions options{};
            options.struct_size = sizeof(options);
            options.bundle_directory = directory.c_str();
            VernonProgramBundle *bundle =
                vernonRuntimeLoadProgramBundleWithOptions(handle, manifestData.c_str(), manifestData.size(), &options);
            if (!bundle)
                throw std::runtime_error("cannot load canonical Program bundle: " +
                                         nativeStringView(vernonRuntimeGetLastError(handle)));
            VernonProgramExecutable *loaded = vernonRuntimeResolveProgram(bundle, {nullptr, 0});
            if (!loaded) {
                error = nativeStringView(vernonRuntimeGetLastError(handle));
                vernonRuntimeProgramBundleDestroy(bundle);
                throw std::runtime_error(error);
            }
            auto pipeline =
                std::make_unique<PythonProgramExecutable>(this, handle, bundle, loaded, std::move(retained));
            pipeline->internedCpuJits = std::move(interned);
            pipeline->registeredCpuEntries = std::move(registered);
            return pipeline;
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
