#include "backend_cpu.h"

#include "backend_stage_pipeline.h"
#include "cpu_workgroup_dispatch.h"
#include "runtime_state.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vernon::runtime {
namespace {

std::mutex &staticEntriesMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<std::string, VernonCpuEntryPoint> &staticEntries() {
    static std::unordered_map<std::string, VernonCpuEntryPoint> entries;
    return entries;
}

template <typename T> VernonRuntimeProviderObject toHandle(T *value) {
    return {static_cast<uint64_t>(reinterpret_cast<uintptr_t>(value))};
}

template <typename T> T *fromHandle(VernonRuntimeProviderObject value) {
    return reinterpret_cast<T *>(static_cast<uintptr_t>(value.value));
}

struct CpuPreparedShader {
    CpuKernelState kernel;
    ReflectedEntry reflection;
};

struct CpuPreparedPipeline;

struct CpuPreparedLayout {
    std::vector<VernonRuntimeProviderBindingLayoutEntry> entries;
    CpuPreparedPipeline *pipeline{};
};

struct CpuPreparedPipeline {
    CpuPreparedShader *shader{};
    CpuPreparedLayout *layout{};
    uint32_t workgroup[3]{1, 1, 1};
};

struct CpuPreparedBindings {
    struct ResultFrameCommit {
        void *data{};
        size_t frameOffset{};
        size_t size{};
    };

    CpuPreparedPipeline *pipeline{};
    std::vector<unsigned char> packed;
    std::vector<unsigned char> results;
    std::vector<ResultFrameCommit> resultFrameCommits;
};

VernonStatus fail(std::string &error, std::string message, VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    error = std::move(message);
    return status;
}

uint32_t cpuCapabilities(void *) { return VERNON_RUNTIME_PROVIDER_COMPUTE; }

VernonRuntimeProviderDeviceIdentity cpuDeviceIdentity(void *data) {
    return {0x435055u, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data)), 1};
}

VernonStatus prepareCpuShader(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                              VernonRuntimeProviderObject *output) {
    auto &context = *static_cast<CpuContextState *>(data);
    if (output)
        *output = {};
    if (!descriptor || !output || descriptor->stage != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE ||
        descriptor->format.size != 12 || std::memcmp(descriptor->format.data, "cpu-prepared", 12) != 0 ||
        descriptor->size != sizeof(CpuProviderShaderPayload) || !descriptor->data)
        return fail(context.error, "invalid CPU provider shader descriptor");
    auto *payload = static_cast<const CpuProviderShaderPayload *>(descriptor->data);
    if (!payload->kernel || !payload->kernel->entry || !payload->reflection)
        return fail(context.error, "CPU provider shader payload is incomplete");
    auto shader = std::unique_ptr<CpuPreparedShader>(new (std::nothrow) CpuPreparedShader());
    if (!shader)
        return fail(context.error, "cannot allocate CPU prepared shader", VERNON_STATUS_INTERNAL_ERROR);
    shader->kernel = std::move(*payload->kernel);
    shader->reflection = std::move(*payload->reflection);
    *output = toHandle(shader.release());
    return VERNON_STATUS_OK;
}

VernonStatus prepareCpuLayout(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                              VernonRuntimeProviderObject *output) {
    auto &context = *static_cast<CpuContextState *>(data);
    if (output)
        *output = {};
    if (!descriptor || !output || (descriptor->binding_count && !descriptor->bindings))
        return fail(context.error, "invalid CPU provider pipeline layout");
    auto layout = std::unique_ptr<CpuPreparedLayout>(new (std::nothrow) CpuPreparedLayout());
    if (!layout)
        return fail(context.error, "cannot allocate CPU provider layout", VERNON_STATUS_INTERNAL_ERROR);
    try {
        if (descriptor->binding_count)
            layout->entries.assign(descriptor->bindings, descriptor->bindings + descriptor->binding_count);
    } catch (const std::bad_alloc &) {
        return fail(context.error, "cannot allocate CPU provider bindings", VERNON_STATUS_INTERNAL_ERROR);
    }
    for (const auto &entry : layout->entries) {
        if ((entry.kind != VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
             entry.kind != VERNON_RUNTIME_PROVIDER_INLINE_VALUE) ||
            entry.stage_mask != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE)
            return fail(context.error, "CPU provider layout is not a canonical compute ABI");
    }
    *output = toHandle(layout.release());
    return VERNON_STATUS_OK;
}

VernonStatus prepareCpuPipeline(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                                VernonRuntimeProviderObject *output) {
    auto &context = *static_cast<CpuContextState *>(data);
    if (output)
        *output = {};
    if (!descriptor || !output || descriptor->kind != VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE ||
        descriptor->shader_count != 1 || !descriptor->shaders || !descriptor->workgroup_size[0] ||
        !descriptor->workgroup_size[1] || !descriptor->workgroup_size[2])
        return fail(context.error, "invalid CPU provider compute pipeline");
    auto pipeline = std::unique_ptr<CpuPreparedPipeline>(new (std::nothrow) CpuPreparedPipeline());
    if (!pipeline)
        return fail(context.error, "cannot allocate CPU provider pipeline", VERNON_STATUS_INTERNAL_ERROR);
    pipeline->shader = fromHandle<CpuPreparedShader>(descriptor->shaders[0]);
    pipeline->layout = fromHandle<CpuPreparedLayout>(descriptor->layout);
    if (!pipeline->shader || !pipeline->layout)
        return fail(context.error, "CPU provider pipeline references invalid prepared objects");
    std::copy_n(descriptor->workgroup_size, 3, pipeline->workgroup);
    pipeline->layout->pipeline = pipeline.get();
    *output = toHandle(pipeline.release());
    return VERNON_STATUS_OK;
}

VernonStatus retainCpuResource(void *data, VernonRuntimeProviderResourceReference resource) {
    auto &context = *static_cast<CpuContextState *>(data);
    if (resource.identity != static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data)) || !resource.resource.value ||
        !resource.size)
        return fail(context.error, "CPU provider resource reference is invalid");
    return VERNON_STATUS_OK;
}

void releaseCpuResource(void *, VernonRuntimeProviderResourceReference) {}

VernonStatus updateCpuBindingsImpl(CpuContextState &context, CpuPreparedBindings &bindings,
                                   const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    if (!bindings.pipeline || !bindings.pipeline->shader || !bindings.pipeline->layout ||
        valueCount != bindings.pipeline->layout->entries.size() || (valueCount && !values))
        return fail(context.error, "CPU provider binding set does not match its layout");
    const ReflectedEntry &reflection = bindings.pipeline->shader->reflection;
    bindings.resultFrameCommits.clear();
    size_t reflectedIndex = 0;
    for (const ReflectedArgument &argument : reflection.arguments) {
        const bool tapeBuiltin = argument.kind == "builtin" && (argument.builtin == VERNON_AD_TAPE_ALLOCATOR_BUILTIN ||
                                                                argument.builtin == VERNON_AD_TAPE_ROOT_REGION_BUILTIN);
        if (argument.kind == "builtin" && !tapeBuiltin)
            continue;
        if (reflectedIndex >= valueCount)
            return fail(context.error, "CPU provider reflection exceeds its binding layout");
        const auto &layout = bindings.pipeline->layout->entries[reflectedIndex];
        const auto &value = values[reflectedIndex];
        std::vector<unsigned char> &frame = argument.result ? bindings.results : bindings.packed;
        if (value.slot != layout.slot || value.kind != layout.kind || argument.physical.offset > frame.size() ||
            argument.physical.size > frame.size() - argument.physical.offset)
            return fail(context.error, "CPU provider binding does not match reflection");
        if (value.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            const auto &resource = value.payload.buffer.resource;
            if (!resource.resource.value ||
                resource.identity != static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&context)) || !resource.size)
                return fail(context.error, "CPU provider storage binding is invalid");
            const auto *storage =
                reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(resource.resource.value) + resource.offset);
            if (argument.result)
                bindings.resultFrameCommits.push_back(
                    {const_cast<uint8_t *>(storage), argument.physical.offset, argument.physical.size});
            if (argument.physical.size == sizeof(uintptr_t)) {
                const uintptr_t pointer = reinterpret_cast<uintptr_t>(storage);
                std::memcpy(frame.data() + argument.physical.offset, &pointer, sizeof(pointer));
            } else {
                if (resource.size < argument.physical.size)
                    return fail(context.error, "CPU provider storage binding is smaller than the inline argument");
                std::memcpy(frame.data() + argument.physical.offset, storage, argument.physical.size);
            }
        } else {
            if (!value.payload.inline_value.data || value.payload.inline_value.size != argument.physical.size)
                return fail(context.error, "CPU provider inline binding is invalid");
            if (argument.result)
                bindings.resultFrameCommits.push_back({const_cast<void *>(value.payload.inline_value.data),
                                                       argument.physical.offset, argument.physical.size});
            std::memcpy(frame.data() + argument.physical.offset, value.payload.inline_value.data,
                        value.payload.inline_value.size);
        }
        ++reflectedIndex;
    }
    return reflectedIndex == valueCount ? VERNON_STATUS_OK
                                        : fail(context.error, "CPU provider binding layout exceeds reflection");
}

VernonStatus createCpuBindingSet(void *data, const VernonRuntimeProviderBindingSetDescriptor *descriptor,
                                 VernonRuntimeProviderObject *output) {
    auto &context = *static_cast<CpuContextState *>(data);
    if (output)
        *output = {};
    auto *layout = descriptor ? fromHandle<CpuPreparedLayout>(descriptor->layout) : nullptr;
    if (!descriptor || !output || !layout)
        return fail(context.error, "invalid CPU provider binding descriptor");
    if (!layout->pipeline || !layout->pipeline->shader)
        return fail(context.error, "CPU provider binding set is missing its pipeline");
    auto bindings = std::unique_ptr<CpuPreparedBindings>(new (std::nothrow) CpuPreparedBindings());
    if (!bindings)
        return fail(context.error, "cannot allocate CPU provider binding set", VERNON_STATUS_INTERNAL_ERROR);
    bindings->pipeline = layout->pipeline;
    if (!layout->pipeline->shader->reflection.packedArguments)
        return fail(context.error, "CPU reflection has no packed argument layout");
    try {
        bindings->packed.resize(layout->pipeline->shader->reflection.packedArguments->size);
        if (layout->pipeline->shader->reflection.packedResults)
            bindings->results.resize(layout->pipeline->shader->reflection.packedResults->size);
    } catch (const std::bad_alloc &) {
        return fail(context.error, "cannot allocate CPU invocation storage", VERNON_STATUS_INTERNAL_ERROR);
    }
    const VernonStatus status = updateCpuBindingsImpl(context, *bindings, descriptor->values, descriptor->value_count);
    if (status != VERNON_STATUS_OK)
        return status;
    *output = toHandle(bindings.release());
    return VERNON_STATUS_OK;
}

VernonStatus updateCpuBindingSet(void *data, VernonRuntimeProviderObject handle,
                                 const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    auto &context = *static_cast<CpuContextState *>(data);
    auto *bindings = fromHandle<CpuPreparedBindings>(handle);
    return bindings ? updateCpuBindingsImpl(context, *bindings, values, valueCount)
                    : fail(context.error, "invalid CPU provider binding set");
}

VernonStatus encodeCpuDispatch(void *data, VernonRuntimeProviderObject,
                               const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &context = *static_cast<CpuContextState *>(data);
    auto *pipeline = descriptor ? fromHandle<CpuPreparedPipeline>(descriptor->pipeline) : nullptr;
    auto *bindings = descriptor ? fromHandle<CpuPreparedBindings>(descriptor->bindings) : nullptr;
    if (!descriptor || !pipeline || !bindings || bindings->pipeline != pipeline)
        return fail(context.error, "invalid CPU provider dispatch");
    const VernonStatus status =
        context.scheduler->dispatch(descriptor->group_count, pipeline->workgroup, [&](VernonCpuRangeV1 &range) {
            range.arguments = bindings->packed.data();
            range.arguments_size = bindings->packed.size();
            range.results = bindings->results.empty() ? nullptr : bindings->results.data();
            range.results_size = bindings->results.size();
            range.textures = nullptr;
            const VernonCpuInvocation invocation{&range, VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1, nullptr, 0, nullptr};
            return pipeline->shader->kernel.entry(&invocation);
        });
    if (status != VERNON_STATUS_OK)
        return fail(context.error,
                    context.scheduler->lastDiagnostic().empty()
                        ? (status == VERNON_STATUS_INVALID_ARGUMENT ? "CPU builtin reflection or dispatch is invalid"
                                                                    : "CPU provider entry invocation failed")
                        : context.scheduler->lastDiagnostic(),
                    status);
    for (const CpuPreparedBindings::ResultFrameCommit &destination : bindings->resultFrameCommits)
        std::memcpy(destination.data, bindings->results.data() + destination.frameOffset, destination.size);
    return status;
}

VernonStatus unsupportedCpuDraw(void *, VernonRuntimeProviderObject, const VernonRuntimeProviderDrawDescriptor *) {
    return VERNON_STATUS_UNSUPPORTED_TARGET;
}

void destroyCpuShader(void *, VernonRuntimeProviderObject handle) { delete fromHandle<CpuPreparedShader>(handle); }
void destroyCpuLayout(void *, VernonRuntimeProviderObject handle) { delete fromHandle<CpuPreparedLayout>(handle); }
void destroyCpuPipeline(void *, VernonRuntimeProviderObject handle) { delete fromHandle<CpuPreparedPipeline>(handle); }
void destroyCpuBindings(void *, VernonRuntimeProviderObject handle) { delete fromHandle<CpuPreparedBindings>(handle); }

} // namespace

bool initializeCpuContext(VernonRuntimeContext &context, uint32_t deviceIndex) {
    if (deviceIndex != 0)
        return false;
    auto state = std::unique_ptr<CpuContextState>(new (std::nothrow) CpuContextState());
    if (!state)
        return false;
    state->scheduler = CpuWorkgroupScheduler::create(CpuWorkgroupScheduler::defaultConfig(), state->error);
    if (!state->scheduler)
        return false;
    state->provider.struct_size = sizeof(VernonRuntimeDeviceProvider);
    state->provider.abi_version = VERNON_PROGRAM_VERSION;
    state->provider.user_data = state.get();
    state->provider.get_capabilities = cpuCapabilities;
    state->provider.get_device_identity = cpuDeviceIdentity;
    state->provider.prepare_shader = prepareCpuShader;
    state->provider.prepare_pipeline_layout = prepareCpuLayout;
    state->provider.prepare_pipeline = prepareCpuPipeline;
    state->provider.retain_resource = retainCpuResource;
    state->provider.release_resource = releaseCpuResource;
    state->provider.create_binding_set = createCpuBindingSet;
    state->provider.update_binding_set = updateCpuBindingSet;
    state->provider.encode_dispatch = encodeCpuDispatch;
    state->provider.encode_draw = unsupportedCpuDraw;
    state->provider.destroy_shader = destroyCpuShader;
    state->provider.destroy_pipeline_layout = destroyCpuLayout;
    state->provider.destroy_pipeline = destroyCpuPipeline;
    state->provider.destroy_binding_set = destroyCpuBindings;
    installRuntimeBackendState(context, state.release());
    return true;
}

const VernonRuntimeDeviceProvider *cpuProvider(VernonRuntimeContext &context) {
    return &runtimeBackendState<CpuContextState>(context).provider;
}

CpuWorkgroupScheduler &cpuWorkgroupScheduler(VernonRuntimeContext &context) {
    return *runtimeBackendState<CpuContextState>(context).scheduler;
}

uint64_t cpuProviderResourceIdentity(const VernonRuntimeContext &context) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&runtimeBackendState<CpuContextState>(context)));
}

VernonStringView cpuProviderLastError(const VernonRuntimeContext &context) {
    const std::string &error = runtimeBackendState<CpuContextState>(context).error;
    return {error.data(), error.size()};
}

bool prepareCpuComputePipeline(VernonRuntimeContext &context, CpuKernelState kernel, ReflectedEntry reflection,
                               CpuPipelineState &state) {
    state.entry = kernel.entry;
    if (reflection.packedArguments)
        state.packedSize = reflection.packedArguments->size;
    if (reflection.packedResults)
        state.packedResultSize = reflection.packedResults->size;
    for (const ReflectedArgument &argument : reflection.arguments) {
        const bool tapeBuiltin = argument.kind == "builtin" && (argument.builtin == VERNON_AD_TAPE_ALLOCATOR_BUILTIN ||
                                                                argument.builtin == VERNON_AD_TAPE_ROOT_REGION_BUILTIN);
        if (argument.kind == "builtin" && !tapeBuiltin)
            continue;
        if (!tapeBuiltin && argument.index == UINT32_MAX) {
            invocationDiagnostic(context) = "CPU compute reflection is missing a kernel argument index";
            return false;
        }
        const uint32_t layoutIndex = static_cast<uint32_t>(state.layout.size());
        VernonRuntimeProviderBindingLayoutEntry binding{};
        binding.slot = layoutIndex;
        binding.set = argument.descriptorSet;
        binding.binding = argument.binding == UINT32_MAX ? layoutIndex : argument.binding;
        binding.kind = argument.kind == "tensor" && !argument.tensorViewDescriptor
                           ? VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER
                           : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
        binding.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
        binding.access = 3;
        binding.array_count = 1;
        binding.argument_index = tapeBuiltin ? UINT32_MAX : argument.index;
        binding.element_size =
            static_cast<uint32_t>(binding.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ? argument.tensorElementSize
                                                                                         : argument.physical.size);
        if (!binding.element_size) {
            invocationDiagnostic(context) = "CPU compute reflection contains a zero-sized argument";
            return false;
        }
        state.layout.push_back(binding);
        state.layoutBuiltins.push_back(tapeBuiltin ? argument.builtin : std::string());
        state.packedOffsets.push_back(argument.physical.offset);
        state.packedFieldSizes.push_back(argument.physical.size);
        state.packedResults.push_back(argument.result);
        state.packedResultReductions.push_back(argument.result && argument.autodiffRole == "gradient" ? argument.dtype
                                                                                                      : std::nullopt);
        if (tapeBuiltin && argument.builtin == VERNON_AD_TAPE_ALLOCATOR_BUILTIN)
            state.tapeAllocatorOffset = argument.physical.offset;
        if (tapeBuiltin && argument.builtin == VERNON_AD_TAPE_ROOT_REGION_BUILTIN)
            state.tapeRootOffset = argument.physical.offset;
    }
    state.values.resize(state.layout.size());
    std::copy_n(reflection.workgroup, 3, state.workgroup);
    CpuProviderShaderPayload payload{&kernel, &reflection};
    const VernonRuntimeProviderShaderDescriptor shader{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                       VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE,
                                                       {"cpu-prepared", 12},
                                                       &payload,
                                                       sizeof(payload),
                                                       {"cpu", 3},
                                                       {},
                                                       {0, 0, 0, 0}};
    VernonRuntimeCorePipelineDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.kind = VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
    descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_COMPUTE;
    descriptor.shaders = &shader;
    descriptor.shader_count = 1;
    descriptor.bindings = state.layout.data();
    descriptor.binding_count = state.layout.size();
    descriptor.push_constant_size = sizeof(VernonLaunchSize);
    std::copy_n(state.workgroup, 3, descriptor.workgroup_size);
    const VernonStatus status = vernonRuntimeCorePreparePipeline(cpuProvider(context), &descriptor, &state.pipeline);
    if (status == VERNON_STATUS_OK)
        return true;
    const VernonStringView providerError = cpuProviderLastError(context);
    invocationDiagnostic(context) = providerError.data ? std::string(providerError.data, providerError.size)
                                                       : "failed to prepare CPU provider pipeline";
    return false;
}

void setCpuProgramTape(VernonStageExecutable &pipeline, VernonAdTapeAllocator *allocator, VernonAdRegionHandle root) {
    CpuPipelineState &state = runtimeBackendState<CpuPipelineState>(pipeline);
    state.tapeAllocator = allocator;
    state.tapeRoot = root;
}

VernonStatus registerStaticCpuEntry(VernonStringView symbol, VernonCpuEntryPoint entry) {
    if (!symbol.data || !symbol.size || !entry)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const std::string name(symbol.data, symbol.size);
    std::lock_guard<std::mutex> lock(staticEntriesMutex());
    staticEntries().try_emplace(name, entry);
    return VERNON_STATUS_OK;
}

bool findRegisteredCpuEntry(VernonRuntimeContext &context, const std::string &symbol, VernonCpuEntryPoint &entry,
                            std::string &error) {
    {
        std::lock_guard<std::mutex> lock(context.cpuEntriesMutex);
        const auto found = context.cpuEntries.find(symbol);
        if (found != context.cpuEntries.end())
            entry = found->second.first;
    }
    if (!entry) {
        std::lock_guard<std::mutex> lock(staticEntriesMutex());
        const auto found = staticEntries().find(symbol);
        if (found != staticEntries().end())
            entry = found->second;
    }
    if (entry)
        return true;
    error = "CPU AOT object symbol '" + symbol + "' was not statically registered";
    return false;
}

bool loadCpuEntry(VernonCpuEntryPoint entry, const char *reflection, size_t reflectionSize, const char *entryName,
                  size_t entryNameSize, CpuKernelState &state, ReflectedEntry &metadata, std::string &error) {
    try {
        const nlohmann::json parsed = nlohmann::json::parse(reflection, reflection + reflectionSize, nullptr, false);
        if (parsed.is_discarded() ||
            !parseReflection(parsed, std::string(entryName, entryNameSize), metadata, VERNON_RUNTIME_CPU, error))
            return false;
        state.entry = entry;
        return true;
    } catch (const std::exception &exception) {
        error = std::string("failed to load CPU entry: ") + exception.what();
        return false;
    }
}

bool loadCpuNativeArtifact(VernonRuntimeContext &context, const CpuNativeArtifact &artifact, CpuKernelState &state,
                           ReflectedEntry &metadata, std::string &error) {
    if (artifact.format == "relocatable_object") {
        // Relocatable CPU artifacts are linked into the embedding application.
        // Runtime resolution is deliberately metadata + registry only: loading
        // object bytes here would require a platform linker or ORC JIT.
        std::filesystem::path unused;
        if (!resolveCpuNativeArtifact(artifact, unused, &metadata, error) ||
            !findRegisteredCpuEntry(context, artifact.symbol, state.entry, error))
            return false;
        return true;
    }

    std::filesystem::path libraryPath;
    if (!resolveCpuNativeArtifact(artifact, libraryPath, &metadata, error))
        return false;
    const std::string nativePath = libraryPath.u8string();
    if (!state.nativeLibrary.open(nativePath.c_str(), error))
        return false;
    state.entry = reinterpret_cast<VernonCpuEntryPoint>(state.nativeLibrary.symbol(artifact.symbol.c_str()));
    if (state.entry)
        return true;
    error = "CPU AOT library does not export symbol '" + artifact.symbol + "'";
    return false;
}

} // namespace vernon::runtime
