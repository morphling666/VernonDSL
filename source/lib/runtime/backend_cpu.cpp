#include "backend_cpu.h"

#include "runtime_state.h"

#include <nlohmann/json.hpp>

#include <algorithm>
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
    std::vector<const ReflectedArgument *> globalInvocationIds;
};

struct CpuPreparedBindings {
    CpuPreparedPipeline *pipeline{};
    std::vector<unsigned char> packed;
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
    for (size_t index = 0; index < layout->entries.size(); ++index) {
        const auto &entry = layout->entries[index];
        if ((entry.kind != VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
             entry.kind != VERNON_RUNTIME_PROVIDER_INLINE_VALUE) ||
            entry.stage_mask != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE || entry.argument_index != index)
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
    try {
        for (const ReflectedArgument &argument : pipeline->shader->reflection.arguments)
            if (argument.kind == "builtin" && argument.builtin == "global_invocation_id")
                pipeline->globalInvocationIds.push_back(&argument);
    } catch (const std::bad_alloc &) {
        return fail(context.error, "cannot prepare CPU builtin bindings", VERNON_STATUS_INTERNAL_ERROR);
    }
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
    size_t reflectedIndex = 0;
    for (const ReflectedArgument &argument : reflection.arguments) {
        if (argument.kind == "builtin")
            continue;
        if (reflectedIndex >= valueCount)
            return fail(context.error, "CPU provider reflection exceeds its binding layout");
        const auto &layout = bindings.pipeline->layout->entries[reflectedIndex];
        const auto &value = values[reflectedIndex];
        if (value.slot != layout.slot || value.kind != layout.kind ||
            argument.physical.offset > bindings.packed.size() ||
            argument.physical.size > bindings.packed.size() - argument.physical.offset)
            return fail(context.error, "CPU provider binding does not match reflection");
        if (value.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            if (!value.resource.resource.value ||
                value.resource.identity != static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&context)) ||
                !value.resource.size)
                return fail(context.error, "CPU provider storage binding is invalid");
            const auto *storage = reinterpret_cast<const uint8_t *>(
                static_cast<uintptr_t>(value.resource.resource.value) + value.resource.offset);
            if (argument.physical.size == sizeof(uintptr_t)) {
                const uintptr_t pointer = reinterpret_cast<uintptr_t>(storage);
                std::memcpy(bindings.packed.data() + argument.physical.offset, &pointer, sizeof(pointer));
            } else {
                if (value.resource.size < argument.physical.size)
                    return fail(context.error, "CPU provider storage binding is smaller than the inline argument");
                std::memcpy(bindings.packed.data() + argument.physical.offset, storage, argument.physical.size);
            }
        } else {
            if (!value.inline_data || value.inline_size != argument.physical.size)
                return fail(context.error, "CPU provider inline binding is invalid");
            std::memcpy(bindings.packed.data() + argument.physical.offset, value.inline_data, value.inline_size);
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
    VernonCpuInvocation invocation{bindings->packed.data(), bindings->packed.size(), nullptr, 0, nullptr};
    uint32_t global[3]{descriptor->group_count[0] * pipeline->workgroup[0],
                       descriptor->group_count[1] * pipeline->workgroup[1],
                       descriptor->group_count[2] * pipeline->workgroup[2]};
    if (descriptor->push_constants && descriptor->push_constant_size == sizeof(VernonLaunchSize)) {
        const auto &exact = *static_cast<const VernonLaunchSize *>(descriptor->push_constants);
        global[0] = exact.x;
        global[1] = exact.y;
        global[2] = exact.z;
    }
    for (uint32_t z = 0; z < global[2]; ++z)
        for (uint32_t y = 0; y < global[1]; ++y)
            for (uint32_t x = 0; x < global[0]; ++x) {
                const uint32_t id[3]{x, y, z};
                for (const ReflectedArgument *builtin : pipeline->globalInvocationIds) {
                    if (builtin->physical.offset > bindings->packed.size() ||
                        builtin->physical.size > bindings->packed.size() - builtin->physical.offset)
                        return fail(context.error, "CPU builtin reflection is out of bounds");
                    std::memcpy(bindings->packed.data() + builtin->physical.offset, id,
                                std::min(builtin->physical.size, sizeof(id)));
                }
                const VernonStatus status = pipeline->shader->kernel.entry(&invocation);
                if (status != VERNON_STATUS_OK)
                    return fail(context.error, "CPU provider entry invocation failed", status);
            }
    return VERNON_STATUS_OK;
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
    state->provider.struct_size = sizeof(VernonRuntimeDeviceProvider);
    state->provider.abi_version = VERNON_PIPELINE_VERSION;
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

uint64_t cpuProviderResourceIdentity(const VernonRuntimeContext &context) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&runtimeBackendState<CpuContextState>(context)));
}

VernonStringView cpuProviderLastError(const VernonRuntimeContext &context) {
    const std::string &error = runtimeBackendState<CpuContextState>(context).error;
    return {error.data(), error.size()};
}

bool prepareCpuComputePipeline(VernonRuntimeContext &context, CpuKernelState kernel, ReflectedEntry reflection,
                               CpuPipelineState &state) {
    uint32_t argumentIndex = 0;
    for (const ReflectedArgument &argument : reflection.arguments) {
        if (argument.kind == "builtin")
            continue;
        VernonRuntimeProviderBindingLayoutEntry binding{};
        binding.slot = argumentIndex;
        binding.set = argument.descriptorSet;
        binding.binding = argument.binding == UINT32_MAX ? argumentIndex : argument.binding;
        binding.kind = argument.kind == "tensor" && !argument.tensorViewDescriptor
                           ? VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER
                           : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
        binding.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
        binding.access = 3;
        binding.array_count = 1;
        binding.argument_index = argumentIndex;
        binding.element_size =
            static_cast<uint32_t>(argument.kind == "tensor" ? argument.tensorElementSize : argument.physical.size);
        if (!binding.element_size) {
            context.error = "CPU compute reflection contains a zero-sized argument";
            return false;
        }
        state.layout.push_back(binding);
        ++argumentIndex;
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
    context.error = providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to prepare CPU provider pipeline";
    return false;
}

VernonStatus registerStaticCpuEntry(VernonStringView symbol, VernonCpuEntryPoint entry) {
    if (!symbol.data || !symbol.size || !entry)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const std::string name(symbol.data, symbol.size);
    std::lock_guard<std::mutex> lock(staticEntriesMutex());
    auto [found, inserted] = staticEntries().emplace(name, entry);
    return inserted || found->second == entry ? VERNON_STATUS_OK : VERNON_STATUS_INVALID_ARGUMENT;
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

bool loadCpuNativeArtifact(const CpuNativeArtifact &artifact, CpuKernelState &state, ReflectedEntry &metadata,
                           std::string &error) {
    std::filesystem::path libraryPath;
    if (!resolveCpuNativeArtifact(artifact, libraryPath, &metadata, error))
        return false;
    if (artifact.format == "relocatable_object") {
        std::lock_guard<std::mutex> lock(staticEntriesMutex());
        const auto found = staticEntries().find(artifact.symbol);
        if (found != staticEntries().end())
            state.entry = found->second;
    } else {
        const std::string nativePath = libraryPath.u8string();
        if (!state.nativeLibrary.open(nativePath.c_str(), error))
            return false;
        state.entry = reinterpret_cast<VernonCpuEntryPoint>(state.nativeLibrary.symbol(artifact.symbol.c_str()));
    }
    if (state.entry)
        return true;
    error = artifact.format == "relocatable_object"
                ? "CPU AOT object symbol '" + artifact.symbol + "' was not statically registered"
                : "CPU AOT library does not export symbol '" + artifact.symbol + "'";
    return false;
}

} // namespace vernon::runtime
