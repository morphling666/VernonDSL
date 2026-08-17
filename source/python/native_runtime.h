#ifndef VERNON_PYTHON_NATIVE_RUNTIME_H
#define VERNON_PYTHON_NATIVE_RUNTIME_H

#include "native_execution_graph.h"
#include "native_pipeline_autodiff.h"

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

    nb::object createExecutionGraph() { return createNativeExecutionGraph(rhiHost); }

    std::unique_ptr<LoadedPipeline> load(const nb::bytes &artifact, const std::string &reflection,
                                         const std::string &entry) {
        VernonLoadedPipeline *pipeline =
            vernonRuntimeLoadArtifact(handle, artifact.c_str(), artifact.size(), reflection.data(), reflection.size(),
                                      entry.data(), entry.size());
        if (!pipeline)
            throw std::runtime_error("cannot load compute pipeline: " +
                                     nativeStringView(vernonRuntimeGetLastError(handle)));
        return std::make_unique<LoadedPipeline>(this, handle, nullptr, pipeline);
    }

    std::unique_ptr<LoadedPipeline> loadCpuEntry(const CompiledProgram &program, const std::string &entry) {
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
        VernonLoadedPipeline *pipeline = vernonRuntimeLoadCpuEntry(handle, entryPoint, reflection.data(),
                                                                   reflection.size(), entry.data(), entry.size());
        if (!pipeline)
            throw std::runtime_error("cannot load CPU entry: " + nativeStringView(vernonRuntimeGetLastError(handle)));
        return std::make_unique<LoadedPipeline>(this, handle, nullptr, pipeline,
                                                std::vector<SharedCompileResult>{program.result});
    }

    std::unique_ptr<LoadedPipeline> loadAutodiff(const CompiledProgram &primal, const std::string &primalName,
                                                 const CompiledProgram &forward, const std::string &forwardName,
                                                 const CompiledProgram &backward, const std::string &backwardName,
                                                 uint64_t staticTapeBytesHint, const std::string &residualStorage,
                                                 const std::string &selectedPolicy,
                                                 bool wholeDispatchRetentionPermitted, const nb::list &groupMetadata) {
        const CompiledProgram *programs[] = {&primal, &forward, &backward};
        const std::string *names[] = {&primalName, &forwardName, &backwardName};
        VernonCpuEntryPoint entries[3]{};
        std::string reflections[3];
        for (size_t index = 0; index < 3; ++index) {
            programs[index]->requireSuccess();
            if (programs[index]->target != primal.target)
                throw std::runtime_error("direct autodiff profiles must use one target");
            if (names[index]->empty())
                throw std::runtime_error("direct autodiff profile entry names must not be empty");
            if (primal.target == VERNON_TARGET_CPU) {
                entries[index] = vernonCompileResultGetCpuEntry(programs[index]->result.get(), names[index]->data(),
                                                                names[index]->size());
                if (!entries[index])
                    throw std::runtime_error("direct autodiff profile entry '" + *names[index] + "' was not found");
            }
            reflections[index] = programs[index]->reflection();
        }
        auto view = [](const std::string &value) { return VernonStringView{value.data(), value.size()}; };
        std::vector<vernon::runtime::AutodiffDerivativeGroup> derivativeGroups;
        derivativeGroups.reserve(groupMetadata.size());
        for (nb::handle item : groupMetadata) {
            nb::tuple metadata = nb::cast<nb::tuple>(item);
            if (metadata.size() != 3)
                throw std::invalid_argument("direct autodiff derivative group metadata is invalid");
            const std::string role = nb::cast<std::string>(metadata[0]);
            vernon::runtime::AutodiffDerivativeRole derivativeRole;
            if (role == "gradient")
                derivativeRole = vernon::runtime::AutodiffDerivativeRole::Gradient;
            else if (role == "cotangent")
                derivativeRole = vernon::runtime::AutodiffDerivativeRole::Cotangent;
            else
                throw std::invalid_argument("direct autodiff derivative group role is invalid");
            derivativeGroups.push_back(
                {derivativeRole, nb::cast<std::string>(metadata[1]), nb::cast<std::vector<std::string>>(metadata[2])});
        }
        std::vector<std::vector<VernonStringView>> derivativeLeafViews;
        std::vector<vernon::runtime::AutodiffDerivativeGroupView> derivativeGroupViews;
        derivativeLeafViews.reserve(derivativeGroups.size());
        derivativeGroupViews.reserve(derivativeGroups.size());
        for (const vernon::runtime::AutodiffDerivativeGroup &group : derivativeGroups) {
            std::vector<VernonStringView> &leaves = derivativeLeafViews.emplace_back();
            leaves.reserve(group.leafPaths.size());
            for (const std::string &leaf : group.leafPaths)
                leaves.push_back(view(leaf));
            derivativeGroupViews.push_back({group.role, view(group.declaredPath), leaves.data(), leaves.size()});
        }
        VernonLoadedPipeline *pipeline = nullptr;
        if (primal.target == VERNON_TARGET_CPU) {
            pipeline = vernon::runtime::loadBackendCpuAutodiffPipeline(
                *handle, entries[0], view(reflections[0]), view(primalName), entries[1], view(reflections[1]),
                view(forwardName), entries[2], view(reflections[2]), view(backwardName), derivativeGroupViews.data(),
                derivativeGroupViews.size(), staticTapeBytesHint, view(residualStorage), view(selectedPolicy),
                wholeDispatchRetentionPermitted);
        } else {
            VernonStringView artifacts[3]{};
            for (size_t index = 0; index < 3; ++index) {
                if (vernonCompileResultGetArtifactCount(programs[index]->result.get()) != 1)
                    throw std::runtime_error("direct GPU autodiff profile must produce exactly one artifact");
                artifacts[index] = vernonCompileResultGetArtifactData(programs[index]->result.get(), 0);
                if (!artifacts[index].data || !artifacts[index].size)
                    throw std::runtime_error("direct GPU autodiff profile artifact is empty");
            }
            const vernon::runtime::AutodiffGpuStageView stages[3]{
                {artifacts[0].data, artifacts[0].size, view(reflections[0]), view(primalName)},
                {artifacts[1].data, artifacts[1].size, view(reflections[1]), view(forwardName)},
                {artifacts[2].data, artifacts[2].size, view(reflections[2]), view(backwardName)},
            };
            pipeline = vernon::runtime::loadBackendGpuAutodiffPipeline(
                *handle, stages[0], stages[1], stages[2], derivativeGroupViews.data(), derivativeGroupViews.size(),
                staticTapeBytesHint, view(residualStorage), view(selectedPolicy));
        }
        if (!pipeline)
            throw std::runtime_error("cannot load direct autodiff profiles: " +
                                     nativeStringView(vernonRuntimeGetLastError(handle)));
        return std::make_unique<LoadedPipeline>(
            this, handle, nullptr, pipeline,
            std::vector<SharedCompileResult>{primal.result, forward.result, backward.result});
    }

    std::unique_ptr<LoadedPipeline> loadPipeline(const nb::bytes &data, const std::vector<std::string> &features) {
        VernonPipelineBundle *bundle =
            vernonRuntimeLoadPipelineBundleWithOptions(handle, data.c_str(), data.size(), nullptr);
        if (!bundle)
            throw std::runtime_error("cannot load pipeline bundle: " +
                                     nativeStringView(vernonRuntimeGetLastError(handle)));
        std::vector<const char *> names;
        for (const std::string &feature : features)
            names.push_back(feature.c_str());
        VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {names.data(), names.size()});
        if (!pipeline) {
            vernonRuntimePipelineBundleDestroy(bundle);
            throw std::runtime_error("cannot resolve pipeline bundle: " +
                                     nativeStringView(vernonRuntimeGetLastError(handle)));
        }
        return std::make_unique<LoadedPipeline>(this, handle, bundle, pipeline);
    }

    std::unique_ptr<LoadedPipeline> loadPipelineAsset(const nb::bytes &data, const std::string &directory,
                                                      const std::vector<std::string> &features) {
        VernonPipelineBundleLoadOptions options{};
        options.struct_size = sizeof(options);
        options.bundle_directory = directory.c_str();
        VernonPipelineBundle *bundle =
            vernonRuntimeLoadPipelineBundleWithOptions(handle, data.c_str(), data.size(), &options);
        if (!bundle)
            throw std::runtime_error("cannot load pipeline bundle: " +
                                     nativeStringView(vernonRuntimeGetLastError(handle)));
        std::vector<const char *> names;
        for (const std::string &feature : features)
            names.push_back(feature.c_str());
        VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {names.data(), names.size()});
        if (!pipeline) {
            vernonRuntimePipelineBundleDestroy(bundle);
            throw std::runtime_error("cannot resolve pipeline bundle: " +
                                     nativeStringView(vernonRuntimeGetLastError(handle)));
        }
        return std::make_unique<LoadedPipeline>(this, handle, bundle, pipeline);
    }

    VernonRuntimeContext *handle{};
    std::shared_ptr<RhiHostState> rhiHost;
};

RhiHostState *runtimeRhiHost(const Runtime *runtime);

std::unique_ptr<Runtime> createRhiRuntime(RhiHost &host);

#endif
