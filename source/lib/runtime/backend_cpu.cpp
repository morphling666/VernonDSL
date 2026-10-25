#include "backend_cpu.h"

#include "backend_stage_pipeline.h"
#include "cpu_workgroup_dispatch.h"
#include "runtime_state.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vernon::runtime {
namespace {

struct StaticEntryRegistration {
    VernonCpuEntryPoint entry;
    size_t references;
};

struct StaticEntryRegistry {
    std::mutex mutex;
    // Several loaded images may each carry a copy of one object, so one symbol
    // can map to several live definitions. Lookups use the oldest; a definition
    // leaves when the image that registered it unregisters it on unload.
    std::unordered_map<std::string, std::vector<StaticEntryRegistration>> entries;
};

// Intentionally leaked: object destructors unregister during process exit or
// image unload, possibly after this library's own statics were destroyed.
StaticEntryRegistry &staticEntryRegistry() {
    static auto *registry = new StaticEntryRegistry();
    return *registry;
}

bool validStaticCpuEntries(const VernonStaticCpuEntry *entries, size_t count) {
    if (!entries && count)
        return false;
    for (size_t index = 0; index < count; ++index)
        if (!entries[index].symbol.data || !entries[index].symbol.size || !entries[index].entry_point)
            return false;
    return true;
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

struct CpuGroupLayout {
    std::vector<VernonRuntimeProviderBindingLayoutEntry> entries;
};

struct CpuPreparedLayout {
    std::vector<const CpuGroupLayout *> groups;
};

struct CpuPreparedPipeline {
    CpuPreparedShader *shader{};
    CpuPreparedLayout *layout{};
    uint32_t workgroup[3]{1, 1, 1};
};

// Inputs own their inline bytes. Result values are destinations that the dispatch writes, so they keep the caller's
// storage, which a synchronous CPU dispatch reaches before the invocation returns.
struct CpuGroup {
    const CpuGroupLayout *layout{};
    std::vector<VernonRuntimeProviderBindingValue> values;
    std::vector<std::vector<unsigned char>> bytes;
};

struct CpuCommands {
    struct ResultFrameCommit {
        void *data{};
        size_t frameOffset{};
        size_t size{};
    };

    CpuPreparedPipeline *pipeline{};
    std::array<const CpuGroup *, VERNON_RUNTIME_PROVIDER_MAX_GROUPS> groups{};
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

VernonStatus prepareCpuGroupLayout(void *data, const VernonRuntimeProviderGroupLayoutDescriptor *descriptor,
                                   VernonRuntimeProviderObject *output) {
    auto &context = *static_cast<CpuContextState *>(data);
    if (output)
        *output = {};
    if (!descriptor || !output || (descriptor->entry_count && !descriptor->entries))
        return fail(context.error, "invalid CPU provider group layout");
    for (size_t index = 0; index < descriptor->entry_count; ++index) {
        const auto &entry = descriptor->entries[index];
        if ((entry.kind != VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
             entry.kind != VERNON_RUNTIME_PROVIDER_INLINE_VALUE) ||
            entry.stage_mask != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE)
            return fail(context.error, "CPU provider group layout is not a canonical compute ABI");
    }
    auto layout = std::make_unique<CpuGroupLayout>();
    layout->entries.assign(descriptor->entries, descriptor->entries + descriptor->entry_count);
    *output = toHandle(layout.release());
    return VERNON_STATUS_OK;
}

VernonStatus prepareCpuLayout(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                              VernonRuntimeProviderObject *output) {
    auto &context = *static_cast<CpuContextState *>(data);
    if (output)
        *output = {};
    if (!descriptor || !output || (descriptor->group_layout_count && !descriptor->group_layouts))
        return fail(context.error, "invalid CPU provider pipeline layout");
    auto layout = std::make_unique<CpuPreparedLayout>();
    for (size_t index = 0; index < descriptor->group_layout_count; ++index)
        layout->groups.push_back(fromHandle<CpuGroupLayout>(descriptor->group_layouts[index]));
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
    *output = toHandle(pipeline.release());
    return VERNON_STATUS_OK;
}

VernonStatus createCpuGroup(void *data, const VernonRuntimeProviderGroupDescriptor *descriptor,
                            VernonRuntimeProviderObject *output) {
    auto &context = *static_cast<CpuContextState *>(data);
    if (output)
        *output = {};
    auto *layout = descriptor ? fromHandle<CpuGroupLayout>(descriptor->group_layout) : nullptr;
    if (!descriptor || !output || !layout || descriptor->value_count != layout->entries.size())
        return fail(context.error, "invalid CPU provider group descriptor");
    auto group = std::make_unique<CpuGroup>();
    group->layout = layout;
    group->values.assign(descriptor->values, descriptor->values + descriptor->value_count);
    group->bytes.resize(group->values.size());
    for (size_t index = 0; index < group->values.size(); ++index) {
        VernonRuntimeProviderBindingValue &value = group->values[index];
        if (value.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            const auto &resource = value.payload.buffer.resource;
            if (!resource.resource.value ||
                resource.identity != static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&context)))
                return fail(context.error, "CPU provider storage binding is invalid");
            continue;
        }
        const auto *bytes = static_cast<const unsigned char *>(value.payload.inline_value.data);
        group->bytes[index].assign(bytes, bytes + value.payload.inline_value.size);
    }
    *output = toHandle(group.release());
    return VERNON_STATUS_OK;
}

VernonStatus beginCpuCommands(void *data, VernonRuntimeProviderObject, VernonRuntimeProviderObject *output) {
    auto &context = *static_cast<CpuContextState *>(data);
    if (output)
        *output = {};
    if (!output)
        return fail(context.error, "invalid CPU provider commands");
    *output = toHandle(new CpuCommands());
    return VERNON_STATUS_OK;
}

VernonStatus endCpuCommands(void *, VernonRuntimeProviderObject commands) {
    delete fromHandle<CpuCommands>(commands);
    return VERNON_STATUS_OK;
}

VernonStatus resumeCpuCommands(void *, VernonRuntimeProviderObject) { return VERNON_STATUS_OK; }

// Values of the bound groups, in group order, follow the reflected arguments and then the metadata carrier.
VernonStatus packCpuFrame(CpuContextState &context, CpuCommands &commands) {
    const CpuPreparedPipeline &pipeline = *commands.pipeline;
    const ReflectedEntry &reflection = pipeline.shader->reflection;
    if (!reflection.packedArguments)
        return fail(context.error, "CPU reflection has no packed argument layout");
    commands.packed.assign(reflection.packedArguments->size, 0);
    commands.results.assign(reflection.packedResults ? reflection.packedResults->size : 0, 0);
    commands.resultFrameCommits.clear();
    struct Bound {
        const VernonRuntimeProviderBindingLayoutEntry *layout{};
        const VernonRuntimeProviderBindingValue *value{};
        const std::vector<unsigned char> *bytes{};
    };
    std::vector<Bound> bound;
    for (size_t group = 0; group < pipeline.layout->groups.size(); ++group) {
        const CpuGroupLayout &layout = *pipeline.layout->groups[group];
        if (layout.entries.empty())
            continue;
        const CpuGroup &values = *commands.groups[group];
        for (size_t index = 0; index < layout.entries.size(); ++index)
            bound.push_back({&layout.entries[index], &values.values[index], &values.bytes[index]});
    }
    size_t reflectedIndex = 0;
    for (const ReflectedArgument &argument : reflection.arguments) {
        const bool tapeBuiltin = argument.kind == "builtin" && (argument.builtin == VERNON_AD_TAPE_ALLOCATOR_BUILTIN ||
                                                                argument.builtin == VERNON_AD_TAPE_ROOT_REGION_BUILTIN);
        if (argument.kind == "builtin" && !tapeBuiltin)
            continue;
        if (reflectedIndex >= bound.size())
            return fail(context.error, "CPU provider reflection exceeds its binding layout");
        const Bound &entry = bound[reflectedIndex];
        std::vector<unsigned char> &frame = argument.result ? commands.results : commands.packed;
        if (argument.physical.offset > frame.size() || argument.physical.size > frame.size() - argument.physical.offset)
            return fail(context.error, "CPU provider binding does not match reflection");
        if (entry.value->kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            const auto &resource = entry.value->payload.buffer.resource;
            const auto *storage =
                reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(resource.resource.value) + resource.offset);
            if (argument.result)
                commands.resultFrameCommits.push_back(
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
            if (entry.bytes->size() != argument.physical.size)
                return fail(context.error, "CPU provider inline binding is invalid");
            if (argument.result)
                commands.resultFrameCommits.push_back({const_cast<void *>(entry.value->payload.inline_value.data),
                                                       argument.physical.offset, argument.physical.size});
            std::memcpy(frame.data() + argument.physical.offset, entry.bytes->data(), entry.bytes->size());
        }
        ++reflectedIndex;
    }
    if (reflection.metadataCarrier) {
        if (reflectedIndex >= bound.size())
            return fail(context.error, "CPU metadata carrier exceeds its binding layout");
        const auto &carrier = *reflection.metadataCarrier;
        const Bound &entry = bound[reflectedIndex];
        if (carrier.interfacePlan.frameOffset > std::numeric_limits<size_t>::max() ||
            carrier.size > std::numeric_limits<size_t>::max())
            return fail(context.error, "CPU metadata carrier exceeds the host address range");
        const size_t offset = static_cast<size_t>(carrier.interfacePlan.frameOffset);
        if (entry.layout->kind != VERNON_RUNTIME_PROVIDER_INLINE_VALUE || entry.bytes->size() != carrier.size ||
            entry.layout->element_size != carrier.size || entry.layout->element_alignment != carrier.alignment ||
            offset > commands.packed.size() || carrier.size > commands.packed.size() - offset)
            return fail(context.error, "CPU metadata carrier does not match its packed frame ABI");
        std::memcpy(commands.packed.data() + offset, entry.bytes->data(), entry.bytes->size());
        ++reflectedIndex;
    }
    return reflectedIndex == bound.size() ? VERNON_STATUS_OK
                                          : fail(context.error, "CPU provider binding layout exceeds reflection");
}

VernonStatus dispatchCpu(CpuContextState &context, CpuCommands &commands, const uint32_t groupCount[3]) {
    if (!commands.pipeline)
        return fail(context.error, "invalid CPU provider dispatch");
    VernonStatus status = packCpuFrame(context, commands);
    if (status != VERNON_STATUS_OK)
        return status;
    const CpuPreparedPipeline &pipeline = *commands.pipeline;
    status = context.scheduler->dispatch(groupCount, pipeline.workgroup, [&](VernonCpuRange &range) {
        range.arguments = commands.packed.data();
        range.arguments_size = commands.packed.size();
        range.results = commands.results.empty() ? nullptr : commands.results.data();
        range.results_size = commands.results.size();
        range.textures = nullptr;
        const VernonCpuInvocation invocation{&range, VERNON_CPU_RANGE_ARGUMENTS_SIZE, nullptr, 0, nullptr};
        return pipeline.shader->kernel.entry(&invocation);
    });
    if (status != VERNON_STATUS_OK)
        return fail(context.error,
                    context.scheduler->lastDiagnostic().empty()
                        ? (status == VERNON_STATUS_INVALID_ARGUMENT ? "CPU builtin reflection or dispatch is invalid"
                                                                    : "CPU provider entry invocation failed")
                        : context.scheduler->lastDiagnostic(),
                    status);
    for (const CpuCommands::ResultFrameCommit &destination : commands.resultFrameCommits)
        std::memcpy(destination.data, commands.results.data() + destination.frameOffset, destination.size);
    return status;
}

// The Runtime keeps the batch's pipelines and groups alive for as long as the batch exists.
struct CpuPacketBatch {
    std::vector<uint32_t> packets;
    std::vector<VernonRuntimeProviderObject> objects;
    size_t payloadWords{};
};

bool cpuPacketIsValid(const VernonRuntimeProviderPacketBatchDescriptor &descriptor, uint32_t opcode,
                      const uint32_t *operands, uint32_t operandCount) {
    switch (opcode) {
    case VERNON_RUNTIME_PROVIDER_PACKET_BIND_PIPELINE:
        return operandCount == 1 && operands[0] < descriptor.object_count;
    case VERNON_RUNTIME_PROVIDER_PACKET_BIND_GROUP:
        return operandCount == 2 && operands[0] < VERNON_RUNTIME_PROVIDER_MAX_GROUPS &&
               operands[1] < descriptor.object_count;
    case VERNON_RUNTIME_PROVIDER_PACKET_DISPATCH:
        return operandCount == 1 && operands[0] <= descriptor.payload_word_count &&
               descriptor.payload_word_count - operands[0] >= 3;
    case VERNON_RUNTIME_PROVIDER_PACKET_PUSH_DEBUG_GROUP:
    case VERNON_RUNTIME_PROVIDER_PACKET_INSERT_DEBUG_MARKER:
    case VERNON_RUNTIME_PROVIDER_PACKET_POP_DEBUG_GROUP:
        return true;
    default:
        return false;
    }
}

VernonStatus createCpuPacketBatch(void *data, const VernonRuntimeProviderPacketBatchDescriptor *descriptor,
                                  VernonRuntimeProviderObject *output) {
    auto &context = *static_cast<CpuContextState *>(data);
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !output ||
        (descriptor->packet_word_count != 0 && !descriptor->packets) ||
        (descriptor->object_count != 0 && !descriptor->objects))
        return fail(context.error, "invalid CPU provider packet batch");
    if (descriptor->resource_count != 0 || descriptor->rendering_count != 0)
        return fail(context.error, "CPU provider does not encode graphics commands", VERNON_STATUS_UNSUPPORTED_TARGET);
    for (size_t word = 0; word < descriptor->packet_word_count;) {
        const uint32_t count = descriptor->packets[word] >> 16;
        if (count == 0 || count > descriptor->packet_word_count - word ||
            !cpuPacketIsValid(*descriptor, descriptor->packets[word] & 0xffffu, descriptor->packets + word + 1,
                              count - 1))
            return fail(context.error, "invalid CPU provider packet");
        word += count;
    }
    auto batch = std::make_unique<CpuPacketBatch>();
    batch->packets.assign(descriptor->packets, descriptor->packets + descriptor->packet_word_count);
    batch->objects.assign(descriptor->objects, descriptor->objects + descriptor->object_count);
    batch->payloadWords = descriptor->payload_word_count;
    *output = toHandle(batch.release());
    return VERNON_STATUS_OK;
}

VernonStatus encodeCpuPackets(void *data, VernonRuntimeProviderObject handle, VernonRuntimeProviderObject batchHandle,
                              const uint32_t *payload, size_t payloadWords) {
    auto &context = *static_cast<CpuContextState *>(data);
    CpuCommands &commands = *fromHandle<CpuCommands>(handle);
    const auto *batch = fromHandle<CpuPacketBatch>(batchHandle);
    if (!batch || payloadWords < batch->payloadWords || (batch->payloadWords != 0 && !payload))
        return fail(context.error, "invalid CPU provider packet encoding");
    commands.pipeline = nullptr;
    commands.groups.fill(nullptr);
    for (size_t word = 0; word < batch->packets.size();) {
        const uint32_t opcode = batch->packets[word] & 0xffffu;
        const uint32_t *operands = batch->packets.data() + word + 1;
        word += batch->packets[word] >> 16;
        if (opcode == VERNON_RUNTIME_PROVIDER_PACKET_BIND_PIPELINE) {
            commands.pipeline = fromHandle<CpuPreparedPipeline>(batch->objects[operands[0]]);
        } else if (opcode == VERNON_RUNTIME_PROVIDER_PACKET_BIND_GROUP) {
            commands.groups[operands[0]] = fromHandle<CpuGroup>(batch->objects[operands[1]]);
        } else if (opcode == VERNON_RUNTIME_PROVIDER_PACKET_DISPATCH) {
            const VernonStatus status = dispatchCpu(context, commands, payload + operands[0]);
            if (status != VERNON_STATUS_OK)
                return status;
        }
    }
    return VERNON_STATUS_OK;
}

void destroyCpuPacketBatch(void *, VernonRuntimeProviderObject handle) { delete fromHandle<CpuPacketBatch>(handle); }

void destroyCpuShader(void *, VernonRuntimeProviderObject handle) { delete fromHandle<CpuPreparedShader>(handle); }
void destroyCpuGroupLayout(void *, VernonRuntimeProviderObject handle) { delete fromHandle<CpuGroupLayout>(handle); }
void destroyCpuLayout(void *, VernonRuntimeProviderObject handle) { delete fromHandle<CpuPreparedLayout>(handle); }
void destroyCpuPipeline(void *, VernonRuntimeProviderObject handle) { delete fromHandle<CpuPreparedPipeline>(handle); }
void destroyCpuGroup(void *, VernonRuntimeProviderObject handle) { delete fromHandle<CpuGroup>(handle); }

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
    state->provider.prepare_group_layout = prepareCpuGroupLayout;
    state->provider.prepare_pipeline_layout = prepareCpuLayout;
    state->provider.prepare_pipeline = prepareCpuPipeline;
    state->provider.create_group = createCpuGroup;
    state->provider.begin_commands = beginCpuCommands;
    state->provider.end_commands = endCpuCommands;
    state->provider.resume_commands = resumeCpuCommands;
    state->provider.create_packet_batch = createCpuPacketBatch;
    state->provider.encode_packets = encodeCpuPackets;
    state->provider.destroy_shader = destroyCpuShader;
    state->provider.destroy_group_layout = destroyCpuGroupLayout;
    state->provider.destroy_pipeline_layout = destroyCpuLayout;
    state->provider.destroy_pipeline = destroyCpuPipeline;
    state->provider.destroy_group = destroyCpuGroup;
    state->provider.destroy_packet_batch = destroyCpuPacketBatch;
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
    if (!buildPreparedComputeBindingPlan({}, reflection, VERNON_RUNTIME_CPU, state.bindingPlan,
                                         invocationDiagnostic(context)))
        return false;
    state.values.resize(state.bindingPlan.size());
    if (!state.groups.prepare(*cpuProvider(context), state.bindingPlan.layouts,
                              std::vector<PipelineGroups::SlotDependencies>(state.bindingPlan.size()),
                              invocationDiagnostic(context)))
        return false;
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
    descriptor.group_layouts = state.groups.layouts();
    descriptor.group_layout_count = state.groups.size();
    descriptor.push_constant_size = sizeof(VernonLaunchSize);
    std::copy_n(state.workgroup, 3, descriptor.workgroup_size);
    const VernonStatus status = vernonRuntimeCorePreparePipeline(cpuProvider(context), &descriptor, &state.pipeline);
    if (status == VERNON_STATUS_OK)
        return true;
    invocationDiagnostic(context) =
        providerDiagnostic(cpuProviderLastError(context), "failed to prepare CPU provider pipeline");
    return false;
}

void setCpuProgramTape(VernonStageExecutable &pipeline, VernonAdTapeAllocator *allocator, VernonAdRegionHandle root) {
    CpuPipelineState &state = runtimeBackendState<CpuPipelineState>(pipeline);
    state.tapeAllocator = allocator;
    state.tapeRoot = root;
}

VernonStatus registerStaticCpuEntries(const VernonStaticCpuEntry *entries, size_t count) {
    if (!validStaticCpuEntries(entries, count))
        return VERNON_STATUS_INVALID_ARGUMENT;
    StaticEntryRegistry &registry = staticEntryRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    for (size_t index = 0; index < count; ++index) {
        auto &definitions = registry.entries[std::string(entries[index].symbol.data, entries[index].symbol.size)];
        const auto found = std::find_if(definitions.begin(), definitions.end(), [&](const auto &definition) {
            return definition.entry == entries[index].entry_point;
        });
        if (found != definitions.end())
            ++found->references;
        else
            definitions.push_back({entries[index].entry_point, 1});
    }
    return VERNON_STATUS_OK;
}

VernonStatus unregisterStaticCpuEntries(const VernonStaticCpuEntry *entries, size_t count) {
    if (!validStaticCpuEntries(entries, count))
        return VERNON_STATUS_INVALID_ARGUMENT;
    StaticEntryRegistry &registry = staticEntryRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    for (size_t index = 0; index < count; ++index) {
        const auto symbol = registry.entries.find(std::string(entries[index].symbol.data, entries[index].symbol.size));
        if (symbol == registry.entries.end())
            continue;
        auto &definitions = symbol->second;
        const auto found = std::find_if(definitions.begin(), definitions.end(), [&](const auto &definition) {
            return definition.entry == entries[index].entry_point;
        });
        if (found == definitions.end() || --found->references != 0)
            continue;
        definitions.erase(found);
        if (definitions.empty())
            registry.entries.erase(symbol);
    }
    return VERNON_STATUS_OK;
}

VernonCpuEntryPoint findStaticCpuEntry(const std::string &symbol) {
    StaticEntryRegistry &registry = staticEntryRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.entries.find(symbol);
    return found != registry.entries.end() ? found->second.front().entry : nullptr;
}

bool findRegisteredCpuEntry(VernonRuntimeContext &context, const std::string &symbol, VernonCpuEntryPoint &entry,
                            std::string &error) {
    {
        std::lock_guard<std::mutex> lock(context.cpuEntriesMutex);
        const auto found = context.cpuEntries.find(symbol);
        if (found != context.cpuEntries.end())
            entry = found->second.first;
    }
    if (!entry)
        entry = findStaticCpuEntry(symbol);
    if (entry)
        return true;
    error = "CPU AOT object symbol '" + symbol +
            "' was not statically registered; link the Program's CPU objects into the application (whole-archive "
            "when they are in a static library)";
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
        auto resolved = resolveCpuNativeArtifact(artifact);
        if (resolved.isErr()) {
            error = renderStageArtifactError(resolved.error(), &artifact);
            return false;
        }
        metadata = std::move(resolved).value().reflection;
        if (!findRegisteredCpuEntry(context, artifact.symbol, state.entry, error))
            return false;
        return true;
    }

    auto resolved = resolveCpuNativeArtifact(artifact);
    if (resolved.isErr()) {
        error = renderStageArtifactError(resolved.error(), &artifact);
        return false;
    }
    ResolvedCpuNativeArtifact native = std::move(resolved).value();
    metadata = std::move(native.reflection);
    const std::string nativePath = native.libraryPath.u8string();
    if (!state.nativeLibrary.open(nativePath.c_str(), error))
        return false;
    state.entry = reinterpret_cast<VernonCpuEntryPoint>(state.nativeLibrary.symbol(artifact.symbol.c_str()));
    if (state.entry)
        return true;
    error = "CPU AOT library does not export symbol '" + artifact.symbol + "'";
    return false;
}

} // namespace vernon::runtime
