#include "VernonRuntime.h"
#include "VernonExecutionGraph.h"
#include "execution_graph/execution_graph_internal.h"
#include "rhi/rhi_internal.h"
#include "runtime/autodiff/tape_allocator_abi.h"
#include "runtime/backend_cpu.h"
#include "runtime/compute_launch_planner.h"
#include "runtime/graphics_invocation_planner.h"
#include "runtime/graphics_scope_planner.h"
#include "runtime/pipeline_bundle.h"
#include "runtime/pipeline_manifest.h"
#include "runtime/pipeline_metadata.h"
#include "runtime/program_execution/device_commands.h"
#include "runtime/program_execution/materialized_node_frame.h"
#include "runtime/program_execution/program_forward.h"
#include "runtime/program_execution/program_invocation_state.h"
#include "runtime/program_execution/resolved_transfer_executor.h"
#include "runtime/program_execution_backend.h"
#include "runtime/program_graphics_executor.h"
#include "runtime/program_instance.h"
#include "runtime/program_invocation_context.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_state.h"
#include "runtime/target_implementation_metadata.h"
#include "runtime/tensor_bridge.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

struct RuntimeProgramBinding {
    VernonProgramArgument argument{};
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;
    std::string layoutHash;
    std::vector<VernonValueLeafView> leaves;
    std::shared_ptr<void> lease;

    void refresh() {
        if (argument.kind != VERNON_PROGRAM_TENSOR)
            return;
        argument.tensor.shape = shape.empty() ? nullptr : shape.data();
        argument.tensor.byte_strides = strides.empty() ? nullptr : strides.data();
        argument.tensor.element_layout.layout_hash = {layoutHash.data(), layoutHash.size()};
        argument.tensor.element_layout.leaves = leaves.empty() ? nullptr : leaves.data();
    }
};

struct RuntimeProgramControl {
    enum Kind : uint32_t { RenderPass = 0, DrawCommand = 1, DynamicState = 2 };
    Kind kind{RenderPass};
    VernonRenderPass renderPass{};
    VernonDrawCommand draw{};
    VernonDynamicState dynamic{};
    std::vector<VernonColorAttachment> colors;
    std::optional<VernonDepthAttachment> depth;
    std::optional<VernonIndexBinding> index;
    std::vector<std::shared_ptr<void>> leases;

    void refresh() {
        renderPass.color_attachments = colors.empty() ? nullptr : colors.data();
        renderPass.color_attachment_count = colors.size();
        renderPass.depth_attachment = depth ? &*depth : nullptr;
        draw.index_binding = index ? &*index : nullptr;
    }
};

struct RuntimeProgramInstanceState {
    explicit RuntimeProgramInstanceState(VernonProgramExecutable &value)
        : pipeline(&value), bindings(&value), controls(&value) {}
    VernonProgramExecutable *pipeline;
    vernon::runtime::program::ProgramInstance bindings;
    vernon::runtime::program::ProgramInstance controls;
};

struct VernonProgramInstance {
    explicit VernonProgramInstance(VernonProgramExecutable &value)
        : state(std::make_shared<RuntimeProgramInstanceState>(value)) {}
    std::shared_ptr<RuntimeProgramInstanceState> state;
};

struct VernonProgramInvocation {
    explicit VernonProgramInvocation(VernonProgramInstance &value)
        : instance(value.state), transaction(instance->bindings.beginInvocation()),
          controlTransaction(instance->controls.beginInvocation()) {}
    std::shared_ptr<RuntimeProgramInstanceState> instance;
    std::unique_ptr<vernon::runtime::program::BindingTransaction> transaction;
    std::unique_ptr<vernon::runtime::program::BindingTransaction> controlTransaction;
    std::shared_ptr<const vernon::runtime::program::InvocationSnapshot> snapshot;
    std::shared_ptr<const vernon::runtime::program::InvocationSnapshot> controlSnapshot;
    bool finished{};
};

namespace {
uint32_t programControlKey(uint32_t slot, RuntimeProgramControl::Kind kind);

struct RuntimeDiagnosticState {
    std::string pending;
    std::string published;
    size_t depth{};
};
thread_local std::unordered_map<const VernonRuntimeContext *, RuntimeDiagnosticState> invocationDiagnostics;
} // namespace

std::string &vernon::runtime::invocationDiagnostic(VernonRuntimeContext &context) {
    RuntimeDiagnosticState &state = invocationDiagnostics[&context];
    return state.depth ? state.pending : state.published;
}

const std::string *vernon::runtime::currentInvocationDiagnostic(const VernonRuntimeContext &context) {
    const auto found = invocationDiagnostics.find(&context);
    return found == invocationDiagnostics.end() ? nullptr : &found->second.published;
}

void vernon::runtime::clearInvocationDiagnostic(const VernonRuntimeContext &context) {
    invocationDiagnostics.erase(&context);
}

vernon::runtime::RuntimeDiagnosticScope::RuntimeDiagnosticScope(const VernonRuntimeContext *context) noexcept
    : context_(context) {
    if (!context_)
        return;
    try {
        RuntimeDiagnosticState &state = invocationDiagnostics[context_];
        if (state.depth++ == 0)
            state.pending.clear();
    } catch (...) {
        context_ = nullptr;
    }
}

vernon::runtime::RuntimeDiagnosticScope::~RuntimeDiagnosticScope() noexcept {
    if (!context_)
        return;
    const auto found = invocationDiagnostics.find(context_);
    if (found == invocationDiagnostics.end())
        return;
    RuntimeDiagnosticState &state = found->second;
    if (!state.depth)
        return;
    if (--state.depth == 0)
        state.published = std::move(state.pending);
}

namespace {

using namespace vernon::runtime;

ArtifactResolution artifactResolutionFor(const VernonRuntimeContext &context) {
    // CPU relocatable objects are build inputs linked by the embedding
    // application. The runtime consumes their manifest metadata and resolves
    // the linked entry through the static registry; it must never load object
    // bytes or acquire an LLVM/ORC dependency.
    if (context.backend == VERNON_RUNTIME_CPU)
        return ArtifactResolution::MetadataOnly;
    return ArtifactResolution::LoadBytes;
}

VernonStatus fail(VernonRuntimeContext *context, std::string_view error,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) noexcept {
    try {
        if (context)
            invocationDiagnostic(*context) = error;
    } catch (...) {
    }
    return status;
}

std::string pipelineTargetKind(const nlohmann::json &root) {
    if (!root.contains("target") || !root["target"].is_object())
        return {};
    const nlohmann::json &target = root["target"];
    if (!target.contains("kind") || !target["kind"].is_string() || !target.contains("options") ||
        !target["options"].is_object() || target.size() != 2)
        return {};
    const std::string kind = target["kind"].get<std::string>();
    const nlohmann::json &options = target["options"];
    auto hasOnly = [&](std::initializer_list<std::string_view> allowed) {
        for (const auto &[key, value] : options.items())
            if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
                return false;
        return true;
    };
    if (kind == "cpu") {
        if (!hasOnly({"triple", "processor", "features"}) || !options.contains("triple") ||
            !options["triple"].is_string() || options["triple"].get_ref<const std::string &>().empty())
            return {};
        if (options.contains("processor") && !options["processor"].is_string())
            return {};
        if (options.contains("features") &&
            (!options["features"].is_array() ||
             !std::all_of(options["features"].begin(), options["features"].end(), [](const nlohmann::json &feature) {
                 return feature.is_string() && !feature.get_ref<const std::string &>().empty();
             })))
            return {};
    } else if (kind == "opengl" || kind == "opengles") {
        if (!hasOnly({"version"}))
            return {};
        if (options.contains("version") &&
            (!options["version"].is_number_unsigned() || options["version"].get<uint64_t>() < 100 ||
             options["version"].get<uint64_t>() > 999))
            return {};
    } else if (kind == "metal") {
        if (!hasOnly({"platform"}) || !options.contains("platform") || !options["platform"].is_string() ||
            (options["platform"] != "macos" && options["platform"] != "ios"))
            return {};
    } else if (kind == "directx") {
        if (!hasOnly({"shader_model"}) || !options.contains("shader_model") ||
            !options["shader_model"].is_number_unsigned() || options["shader_model"].get<uint64_t>() < 60)
            return {};
    } else if (kind == "vulkan" || kind == "cuda") {
        if (!options.empty())
            return {};
    } else {
        return {};
    }
    return kind;
}

const ValueLayout &parameterLogicalLeafLayout(const Parameter &parameter) {
    /* Shaped Tensors keep outer extents on the parameter; leaves are cells. */
    if (!parameter.shape.empty() && !parameter.elementLayout.leaves.empty())
        return parameter.elementLayout;
    if (parameter.valueLayout)
        return *parameter.valueLayout;
    return parameter.elementLayout;
}

const program::Program *executableProgram(const VernonProgramExecutable &pipeline) {
    return &pipeline.executionPlan->resolvedProgram->program;
}

bool publicBoundarySlot(const program::Program &program, const program::BoundarySlot &slot) {
    if (slot.role == program::BoundaryRole::Input)
        return true;
    if (slot.role != program::BoundaryRole::Output || !program::findPublicationTarget(program.abi, slot.id))
        return false;
    return std::none_of(program.abi.boundarySlots.begin(), program.abi.boundarySlots.end(),
                        [&](const program::BoundarySlot &candidate) {
                            return candidate.role == program::BoundaryRole::Input &&
                                   candidate.aliasOwner.kind == slot.aliasOwner.kind &&
                                   candidate.aliasOwner.id == slot.aliasOwner.id;
                        });
}

size_t publicBoundaryCount(const program::Program &program) {
    return static_cast<size_t>(
        std::count_if(program.abi.boundarySlots.begin(), program.abi.boundarySlots.end(),
                      [&](const program::BoundarySlot &slot) { return publicBoundarySlot(program, slot); }));
}

const program::BoundarySlot *publicBoundaryAt(const program::Program &program, size_t publicIndex) {
    for (const program::BoundarySlot &slot : program.abi.boundarySlots)
        if (publicBoundarySlot(program, slot) && publicIndex-- == 0)
            return &slot;
    return nullptr;
}

const program::BoundarySlot *findPublicBoundary(const program::Program &program, VernonStringView name) {
    const auto found = std::find_if(
        program.abi.boundarySlots.begin(), program.abi.boundarySlots.end(), [&](const program::BoundarySlot &slot) {
            return publicBoundarySlot(program, slot) && name.size == slot.path.size() &&
                   (name.size == 0 || std::equal(name.data, name.data + name.size, slot.path.data()));
        });
    return found == program.abi.boundarySlots.end() ? nullptr : &*found;
}

const ValueLayout *boundaryLayoutView(const VernonProgramExecutable &pipeline, const program::BoundarySlot &slot) {
    const program::Program *program = executableProgram(pipeline);
    if (!program)
        return nullptr;
    const auto slotIndex = static_cast<size_t>(&slot - program->abi.boundarySlots.data());
    const program::ResolvedExecutionPlan &execution = *pipeline.executionPlan;
    return slotIndex < execution.boundaryLayoutViews.size() ? &execution.boundaryLayoutViews[slotIndex] : nullptr;
}

} // namespace

extern "C" {

VernonValueLayoutView vernonRuntimeGetScalarValueLayout(VernonDataType dtype) {
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
    const size_t size = dataTypeSize(dtype);
    const size_t index = static_cast<size_t>(dtype);
    if (!size || index >= std::size(leaves) || !hashes[index][0])
        return {};
    return {sizeof(VernonValueLayoutView),
            static_cast<uint32_t>(size),
            static_cast<uint32_t>(size),
            {hashes[index], std::strlen(hashes[index])},
            &leaves[index],
            1};
}

VernonRuntimeCapabilities vernonRuntimeGetCapabilities(VernonRuntimeBackend backend) {
    static thread_local std::string diagnostic;
    diagnostic.clear();
    VernonRuntimeCapabilities result{};
    result.available = probeBackend(backend, diagnostic);
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
}

VernonRuntimeContext *vernonRuntimeCreateWithOptions(VernonRuntimeBackend backend,
                                                     const VernonRuntimeCreateOptions *options) {
    if (options && options->struct_size < sizeof(VernonRuntimeCreateOptions))
        return nullptr;
    auto context = std::make_unique<VernonRuntimeContext>();
    context->backend = backend;
    if (!initializeBackend(*context, options ? options->device_index : 0))
        return nullptr;
    return context.release();
}

VernonRuntimeContext *vernonRuntimeCreateForRhiDevice(VernonRuntimeBackend backend, VernonRhiDevice device) {
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        return nullptr;
    auto context = std::make_unique<VernonRuntimeContext>();
    context->backend = backend;
    if (!initializeBackendForRhiDevice(*context, device))
        return nullptr;
    return context.release();
}

VernonStatus vernonRuntimeDestroy(VernonRuntimeContext *context) {
    if (!context)
        return VERNON_STATUS_OK;
    if (context->liveBundles || context->livePipelines || context->liveContextLeases) {
        vernon::runtime::RuntimeDiagnosticScope diagnostic(context);
        return fail(context, "runtime context still owns live handles");
    }
    destroyBackend(*context);
    vernon::runtime::clearInvocationDiagnostic(*context);
    delete context;
    return VERNON_STATUS_OK;
}

VernonStringView vernonRuntimeGetLastError(const VernonRuntimeContext *context) {
    if (!context)
        return {nullptr, 0};
    if (const std::string *diagnostic = vernon::runtime::currentInvocationDiagnostic(*context))
        return {diagnostic->data(), diagnostic->size()};
    return {nullptr, 0};
}

VernonRuntimeCapabilities vernonRuntimeGetContextCapabilities(const VernonRuntimeContext *context) {
    VernonRuntimeCapabilities result{};
    if (!context)
        return result;
    {
        RuntimeDiagnosticScope diagnostic(context);
        fillBackendCapabilities(*context, result);
    }
    if (const std::string *diagnostic = currentInvocationDiagnostic(*context))
        result.diagnostic = {diagnostic->data(), diagnostic->size()};
    return result;
}

VernonStageExecutable *vernonRuntimeLoadCpuEntry(VernonRuntimeContext *context, VernonCpuEntryPoint entryPoint,
                                                 const char *reflection, size_t reflectionSize, const char *entry,
                                                 size_t entrySize) {
    RuntimeDiagnosticScope diagnostic(context);
    if (!context || context->backend != VERNON_RUNTIME_CPU || !entryPoint || !reflection || !reflectionSize || !entry ||
        !entrySize)
        return nullptr;
    return loadBackendCpuEntryPipeline(*context, entryPoint, reflection, reflectionSize, entry, entrySize);
}

VernonStatus vernonRuntimeRegisterStaticCpuEntry(VernonStringView symbol, VernonCpuEntryPoint entryPoint) {
    return registerBackendStaticCpuEntry(symbol, entryPoint);
}

VernonStatus vernonRuntimeRegisterCpuEntry(VernonRuntimeContext *context, VernonStringView symbol,
                                           VernonCpuEntryPoint entryPoint) {
    RuntimeDiagnosticScope diagnostic(context);
    if (!context || context->backend != VERNON_RUNTIME_CPU || !symbol.data || !symbol.size || !entryPoint)
        return VERNON_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(context->cpuEntriesMutex);
    auto [found, inserted] =
        context->cpuEntries.emplace(std::string(symbol.data, symbol.size), std::make_pair(entryPoint, size_t{1}));
    if (!inserted) {
        if (found->second.first != entryPoint)
            return VERNON_STATUS_INVALID_ARGUMENT;
        ++found->second.second;
    }
    return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeUnregisterCpuEntry(VernonRuntimeContext *context, VernonStringView symbol,
                                             VernonCpuEntryPoint entryPoint) {
    RuntimeDiagnosticScope diagnostic(context);
    if (!context || !symbol.data || !symbol.size || !entryPoint)
        return VERNON_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(context->cpuEntriesMutex);
    const auto found = context->cpuEntries.find(std::string(symbol.data, symbol.size));
    if (found == context->cpuEntries.end() || found->second.first != entryPoint)
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (!--found->second.second)
        context->cpuEntries.erase(found);
    return VERNON_STATUS_OK;
}

VernonStageExecutable *vernonRuntimeLoadArtifact(VernonRuntimeContext *context, const void *artifact,
                                                 size_t artifactSize, const char *reflection, size_t reflectionSize,
                                                 const char *entry, size_t entrySize) {
    RuntimeDiagnosticScope diagnostic(context);
    if (!context || !artifact || !artifactSize || !reflection || !reflectionSize || !entry || !entrySize)
        return nullptr;
    if (context->backend == VERNON_RUNTIME_CPU) {
        fail(context,
             "CPU AOT artifacts must be loaded from a compute bundle so the "
             "native library target and content hash can be validated",
             VERNON_STATUS_UNSUPPORTED_TARGET);
        return nullptr;
    }
    return loadBackendArtifactPipeline(*context, artifact, artifactSize, reflection, reflectionSize, entry, entrySize);
}

namespace {

bool parseProgramDeployments(VernonRuntimeContext &context, const nlohmann::json &document,
                             std::vector<vernon::runtime::ProgramVariantDeployment> &result) {
    std::string previousKeyBytes;
    for (const nlohmann::json &variant : document["variants"]) {
        if (!variant.is_object() || variant.size() != 3 || !variant.contains("key") || !variant["key"].is_array() ||
            !variant.contains("program") || !variant["program"].is_object() || !variant.contains("artifact_system") ||
            !variant["artifact_system"].is_object()) {
            fail(&context, "Program bundle variant has unsupported or invalid members", VERNON_STATUS_PARSE_ERROR);
            return false;
        }
        vernon::runtime::ProgramVariantDeployment parsed;
        for (const nlohmann::json &feature : variant["key"]) {
            if (!feature.is_string()) {
                fail(&context, "Program bundle feature names must be strings", VERNON_STATUS_PARSE_ERROR);
                return false;
            }
            parsed.key.push_back(feature.get<std::string>());
        }
        std::vector<std::string> canonicalKey = parsed.key;
        std::sort(canonicalKey.begin(), canonicalKey.end());
        canonicalKey.erase(std::unique(canonicalKey.begin(), canonicalKey.end()), canonicalKey.end());
        if (canonicalKey != parsed.key || std::any_of(parsed.key.begin(), parsed.key.end(),
                                                      [](const std::string &feature) { return feature.empty(); })) {
            fail(&context, "Program bundle feature key is not canonical", VERNON_STATUS_PARSE_ERROR);
            return false;
        }
        const std::string keyBytes = variant["key"].dump();
        if (!previousKeyBytes.empty() && keyBytes <= previousKeyBytes) {
            fail(&context, "Program bundle variants are not canonically ordered", VERNON_STATUS_PARSE_ERROR);
            return false;
        }
        previousKeyBytes = keyBytes;
        if (std::any_of(result.begin(), result.end(), [&](const vernon::runtime::ProgramVariantDeployment &existing) {
                return existing.key == parsed.key;
            })) {
            fail(&context, "Program bundle contains duplicate feature variants", VERNON_STATUS_PARSE_ERROR);
            return false;
        }
        vernon::runtime::program::Diagnostic diagnostic;
        if (!vernon::runtime::program::parse(variant["program"], parsed.program, diagnostic) ||
            !vernon::runtime::program::parseArtifactSystem(
                document["target"], document["blobs"], variant["artifact_system"], parsed.artifactSystem, diagnostic)) {
            fail(&context,
                 diagnostic.code + (diagnostic.path.empty() ? ": " : " at " + diagnostic.path + ": ") +
                     diagnostic.message,
                 VERNON_STATUS_PARSE_ERROR);
            return false;
        }
        result.push_back(std::move(parsed));
    }
    if (result.empty()) {
        fail(&context, "Program bundle declares no variants", VERNON_STATUS_PARSE_ERROR);
        return false;
    }
    return true;
}

const vernon::runtime::ProgramVariantDeployment *
selectProgramDeployment(VernonRuntimeContext &context,
                        const std::vector<vernon::runtime::ProgramVariantDeployment> &variants,
                        VernonFeatureSetView features) {
    std::vector<std::string> requested;
    requested.reserve(features.count);
    for (size_t index = 0; index < features.count; ++index) {
        if (!features.names[index] || !*features.names[index]) {
            fail(&context, "Program bundle feature names must not be empty");
            return nullptr;
        }
        requested.emplace_back(features.names[index]);
    }
    std::sort(requested.begin(), requested.end());
    requested.erase(std::unique(requested.begin(), requested.end()), requested.end());
    const auto found =
        std::find_if(variants.begin(), variants.end(), [&](const vernon::runtime::ProgramVariantDeployment &variant) {
            return variant.key == requested;
        });
    if (found == variants.end()) {
        fail(&context, "Program bundle has no matching variant", VERNON_STATUS_PARSE_ERROR);
        return nullptr;
    }
    return &*found;
}

VernonProgramExecutable *resolveProgramDeployment(VernonRuntimeContext &context,
                                                  const vernon::runtime::ProgramVariantDeployment &variant,
                                                  const std::filesystem::path &bundleRoot) {
    std::string error;
    VernonProgramExecutable *pipeline = vernon::runtime::program::loadBackendProgramPipeline(
        context, variant.program, variant.artifactSystem, bundleRoot, error);
    if (!pipeline)
        fail(&context, error.empty() ? "cannot load Program bundle" : error, VERNON_STATUS_PARSE_ERROR);
    return pipeline;
}

} // namespace

VernonStatus vernonRuntimeProgramBundleInspectTarget(const void *bundleData, size_t bundleSize,
                                                     VernonRuntimeBackend *target) {
    if (!bundleData || !bundleSize || !target)
        return VERNON_STATUS_INVALID_ARGUMENT;
    try {
        const nlohmann::json root = nlohmann::json::parse(
            static_cast<const char *>(bundleData), static_cast<const char *>(bundleData) + bundleSize, nullptr, false);
        if (root.is_discarded() || !root.is_object())
            return VERNON_STATUS_PARSE_ERROR;
        static const std::set<std::string> members{"compiler_contract_version",
                                                   "program_version",
                                                   "type",
                                                   "id",
                                                   "target",
                                                   "blobs",
                                                   "variants",
                                                   "content_hash"};
        std::set<std::string> actual;
        for (const auto &[name, unused] : root.items())
            actual.insert(name);
        std::string manifestError;
        if (actual != members || root.value("compiler_contract_version", 0) != VERNON_COMPILER_CONTRACT_VERSION ||
            root.value("program_version", 0) != VERNON_PROGRAM_VERSION || root.value("type", "") != "program" ||
            !root.contains("id") || !root["id"].is_string() || root["id"].get_ref<const std::string &>().empty() ||
            !root.contains("target") || !root["target"].is_object() || root["target"].size() != 2 ||
            !root["target"].contains("kind") || !root["target"]["kind"].is_string() ||
            !root["target"].contains("options") || !root["target"]["options"].is_object() || !root.contains("blobs") ||
            !root["blobs"].is_object() || !root.contains("variants") || !root["variants"].is_array() ||
            !validateManifestHash(root, true, manifestError))
            return VERNON_STATUS_PARSE_ERROR;
        const std::string name = pipelineTargetKind(root);
        if (name.empty())
            return VERNON_STATUS_PARSE_ERROR;
        if (name == "cpu")
            *target = VERNON_RUNTIME_CPU;
        else if (name == "cuda")
            *target = VERNON_RUNTIME_CUDA;
        else if (name == "vulkan")
            *target = VERNON_RUNTIME_VULKAN;
        else if (name == "opengl")
            *target = VERNON_RUNTIME_OPENGL;
        else if (name == "opengles")
            *target = VERNON_RUNTIME_OPENGL_ES;
        else if (name == "directx")
            *target = VERNON_RUNTIME_DIRECTX12;
        else if (name == "metal")
            *target = VERNON_RUNTIME_METAL;
        else
            return VERNON_STATUS_UNSUPPORTED_TARGET;
        return VERNON_STATUS_OK;
    } catch (...) {
        return VERNON_STATUS_PARSE_ERROR;
    }
}

VernonProgramBundle *vernonRuntimeLoadProgramBundleWithOptions(VernonRuntimeContext *context, const void *bundleData,
                                                               size_t bundleSize,
                                                               const VernonProgramBundleLoadOptions *options) {
    if (!context)
        return nullptr;
    RuntimeDiagnosticScope diagnostic(context);
    if ((context->backend != VERNON_RUNTIME_CPU && context->backend != VERNON_RUNTIME_OPENGL &&
         context->backend != VERNON_RUNTIME_OPENGL_ES && context->backend != VERNON_RUNTIME_VULKAN &&
         context->backend != VERNON_RUNTIME_CUDA && context->backend != VERNON_RUNTIME_DIRECTX12 &&
         context->backend != VERNON_RUNTIME_METAL) ||
        !bundleData || !bundleSize || (options && options->struct_size < sizeof(VernonProgramBundleLoadOptions))) {
        invocationDiagnostic(*context) = "invalid pipeline bundle load invocation";
        return nullptr;
    }
    try {
        std::optional<std::filesystem::path> bundleDirectory;
        if (options && options->bundle_directory && options->bundle_directory[0] != '\0')
            bundleDirectory = std::filesystem::u8path(options->bundle_directory);
        const nlohmann::json root = nlohmann::json::parse(static_cast<const char *>(bundleData),
                                                          static_cast<const char *>(bundleData) + bundleSize);
        const char *expectedTarget = context->backend == VERNON_RUNTIME_CPU         ? "cpu"
                                     : context->backend == VERNON_RUNTIME_CUDA      ? "cuda"
                                     : context->backend == VERNON_RUNTIME_VULKAN    ? "vulkan"
                                     : context->backend == VERNON_RUNTIME_DIRECTX12 ? "directx"
                                     : context->backend == VERNON_RUNTIME_METAL
                                         ? "metal"
                                         : (context->backend == VERNON_RUNTIME_OPENGL_ES ? "opengles" : "opengl");
        static const std::set<std::string> canonicalMembers{"compiler_contract_version",
                                                            "program_version",
                                                            "type",
                                                            "id",
                                                            "target",
                                                            "blobs",
                                                            "variants",
                                                            "content_hash"};
        std::set<std::string> members;
        if (root.is_object())
            for (const auto &[name, unused] : root.items())
                members.insert(name);
        if (members != canonicalMembers ||
            root.value("compiler_contract_version", 0) != VERNON_COMPILER_CONTRACT_VERSION ||
            root.value("program_version", 0) != VERNON_PROGRAM_VERSION || root.value("type", "") != "program" ||
            !root["id"].is_string() || root["id"].get_ref<const std::string &>().empty() ||
            !root["target"].is_object() || root["target"].size() != 2 ||
            root["target"].value("kind", "") != expectedTarget || !root["target"].contains("options") ||
            !root["target"]["options"].is_object() || !root["blobs"].is_object() || !root["variants"].is_array()) {
            fail(context, "unsupported or invalid Program bundle", VERNON_STATUS_PARSE_ERROR);
            return nullptr;
        }
        auto bundle = std::make_unique<VernonProgramBundle>();
        bundle->context = context;
        bundle->id = root["id"].get<std::string>();
        if (bundleDirectory)
            bundle->bundleRoot = *bundleDirectory;
        if (!parseProgramDeployments(*context, root, bundle->deployments))
            return nullptr;
        if (!validateManifestHash(root, true, invocationDiagnostic(*context)))
            return nullptr;
        ++context->liveBundles;
        return bundle.release();
    } catch (const nlohmann::json::exception &error) {
        fail(context, std::string("invalid Program bundle: ") + error.what());
        return nullptr;
    } catch (const std::exception &error) {
        fail(context, std::string("failed to load Program bundle: ") + error.what(), VERNON_STATUS_INTERNAL_ERROR);
        return nullptr;
    } catch (...) {
        fail(context, "failed to load Program bundle", VERNON_STATUS_INTERNAL_ERROR);
        return nullptr;
    }
}

VernonStringView vernonRuntimeProgramBundleGetId(const VernonProgramBundle *bundle) {
    RuntimeDiagnosticScope diagnostic(bundle ? bundle->context : nullptr);
    return bundle ? VernonStringView{bundle->id.data(), bundle->id.size()} : VernonStringView{nullptr, 0};
}

namespace {

VernonValueLayoutView valueLayoutView(const ValueLayout &layout) {
    return {sizeof(VernonValueLayoutView),
            layout.byteSize,
            layout.alignment,
            {layout.layoutHash.data(), layout.layoutHash.size()},
            layout.abiLeaves.empty() ? nullptr : layout.abiLeaves.data(),
            layout.abiLeaves.size()};
}

bool fillParameterView(const Parameter &source, VernonProgramParameterView &destination) {
    const auto kind = pipelineArgumentKind(source.kind);
    const auto access = pipelineValueAccess(source.access);
    if (!kind || !access)
        return false;
    destination = {source.slot,
                   {source.name.data(), source.name.size()},
                   *kind,
                   source.kind == "tensor"
                       ? valueLayoutView(source.valueLayout ? *source.valueLayout : source.elementLayout)
                       : VernonValueLayoutView{},
                   *access,
                   static_cast<uint32_t>(source.shape.size()),
                   source.shape.empty() ? nullptr : source.shape.data()};
    return true;
}

bool fillBoundaryParameterView(const VernonProgramExecutable &pipeline, const program::BoundarySlot &source,
                               VernonProgramParameterView &destination) {
    VernonProgramArgumentKind kind = VERNON_PROGRAM_TENSOR;
    if (source.category == program::BoundaryCategory::Texture)
        kind = VERNON_PROGRAM_IMAGE;
    else if (source.category == program::BoundaryCategory::Sampler)
        kind = VERNON_PROGRAM_SAMPLER;
    VernonValueAccess access = VERNON_ACCESS_READ;
    if (source.access == program::BoundaryAccess::Write)
        access = VERNON_ACCESS_WRITE;
    else if (source.access == program::BoundaryAccess::ReadWrite)
        access = VERNON_ACCESS_READ_WRITE;
    const ValueLayout *layout = boundaryLayoutView(pipeline, source);
    if ((source.category == program::BoundaryCategory::Value ||
         source.category == program::BoundaryCategory::StorageView) &&
        (!source.layout || !layout))
        return false;
    destination = {source.id,
                   {source.path.data(), source.path.size()},
                   kind,
                   layout ? valueLayoutView(*layout) : VernonValueLayoutView{},
                   access,
                   static_cast<uint32_t>(source.outerShape.size()),
                   source.outerShape.empty() ? nullptr : source.outerShape.data()};
    return true;
}

VernonStatus fillImageConstraintView(const Parameter &source, VernonProgramImageConstraintView &destination) {
    if (source.kind != "image")
        return VERNON_STATUS_INVALID_ARGUMENT;
    const auto dimension = pipelineTextureDimension(source.dimension);
    if (!dimension)
        return VERNON_STATUS_PARSE_ERROR;
    destination.dimension = *dimension;
    destination.binding_role =
        source.bindingRole == "sampled" ? VERNON_IMAGE_BINDING_SAMPLED : VERNON_IMAGE_BINDING_STORAGE;
    destination.sample_result_class = VERNON_IMAGE_SAMPLE_FLOAT;
    if (destination.binding_role == VERNON_IMAGE_BINDING_STORAGE) {
        const auto format = pipelineTextureFormat(source.exactStorageFormat);
        if (!format)
            return VERNON_STATUS_PARSE_ERROR;
        destination.storage_format = *format;
    } else {
        destination.storage_format = static_cast<VernonTextureFormat>(0);
    }
    std::fill(std::begin(destination.reserved), std::end(destination.reserved), 0);
    return VERNON_STATUS_OK;
}

VernonStatus fillBoundaryImageConstraintView(const program::BoundarySlot &source,
                                             VernonProgramImageConstraintView &destination) {
    if (source.category != program::BoundaryCategory::Texture || !source.storage ||
        source.storage->descriptorKind != program::StorageDescriptorKind::Image)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const auto dimension = pipelineTextureDimension(source.storage->image.dimension);
    if (!dimension)
        return VERNON_STATUS_PARSE_ERROR;
    destination.dimension = *dimension;
    const auto hasUsage = [&](std::string_view usage) {
        return std::find(source.storage->image.usage.begin(), source.storage->image.usage.end(), usage) !=
               source.storage->image.usage.end();
    };
    destination.binding_role = hasUsage("color_attachment")           ? VERNON_IMAGE_BINDING_COLOR_ATTACHMENT
                               : hasUsage("depth_stencil_attachment") ? VERNON_IMAGE_BINDING_DEPTH_STENCIL_ATTACHMENT
                               : source.access == program::BoundaryAccess::Read ? VERNON_IMAGE_BINDING_SAMPLED
                                                                                : VERNON_IMAGE_BINDING_STORAGE;
    destination.sample_result_class = VERNON_IMAGE_SAMPLE_FLOAT;
    if (destination.binding_role == VERNON_IMAGE_BINDING_STORAGE) {
        const auto format = pipelineTextureFormat(source.storage->image.format);
        if (!format)
            return VERNON_STATUS_PARSE_ERROR;
        destination.storage_format = *format;
    } else {
        destination.storage_format = static_cast<VernonTextureFormat>(0);
    }
    std::fill(std::begin(destination.reserved), std::end(destination.reserved), 0);
    return VERNON_STATUS_OK;
}

bool fillOutputView(const Output &source, VernonProgramOutputView &destination) {
    const auto kind = pipelineArgumentKind(source.kind);
    const auto dtype = pipelineDataType(source.dtype);
    const auto access = pipelineValueAccess(source.access);
    if (!kind || !dtype || !access)
        return false;
    destination = {{source.name.data(), source.name.size()},
                   *kind,
                   *dtype,
                   *access,
                   static_cast<uint32_t>(source.shape.size()),
                   source.shape.empty() ? nullptr : source.shape.data(),
                   source.location};
    return true;
}

bool stringViewEquals(VernonStringView view, const std::string &value) {
    return view.size == value.size() && (!view.size || std::memcmp(view.data, value.data(), view.size) == 0);
}

} // namespace

void vernonRuntimeProgramBundleDestroy(VernonProgramBundle *bundle) {
    if (!bundle)
        return;
    RuntimeDiagnosticScope diagnostic(bundle->context);
    --bundle->context->liveBundles;
    delete bundle;
}

VernonProgramExecutable *vernonRuntimeResolveProgram(VernonProgramBundle *bundle, VernonFeatureSetView features) {
    if (!bundle)
        return nullptr;
    try {
        RuntimeDiagnosticScope diagnostic(bundle->context);
        if (features.count && !features.names) {
            invocationDiagnostic(*bundle->context) = "invalid pipeline feature set";
            return nullptr;
        }
        const vernon::runtime::ProgramVariantDeployment *selected =
            selectProgramDeployment(*bundle->context, bundle->deployments, features);
        return selected ? resolveProgramDeployment(*bundle->context, *selected, bundle->bundleRoot) : nullptr;
    } catch (const std::bad_alloc &) {
        fail(bundle->context, "cannot allocate resolved pipeline", VERNON_STATUS_INTERNAL_ERROR);
    } catch (const std::length_error &) {
        fail(bundle->context, "resolved pipeline allocation is too large", VERNON_STATUS_INTERNAL_ERROR);
    } catch (...) {
        fail(bundle->context, "unexpected pipeline resolution failure", VERNON_STATUS_INTERNAL_ERROR);
    }
    return nullptr;
}

size_t vernonRuntimeProgramExecutableGetParameterCount(const VernonProgramExecutable *pipeline) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline)
        return 0;
    return publicBoundaryCount(*executableProgram(*pipeline));
}

VernonStatus vernonRuntimeProgramExecutableGetParameterByIndex(const VernonProgramExecutable *pipeline, size_t index,
                                                               VernonProgramParameterView *parameter) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !parameter)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const program::BoundarySlot *slot = publicBoundaryAt(*executableProgram(*pipeline), index);
    if (!slot)
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillBoundaryParameterView(*pipeline, *slot, *parameter) ? VERNON_STATUS_OK : VERNON_STATUS_PARSE_ERROR;
}

VernonStatus vernonRuntimeProgramExecutableFindParameter(const VernonProgramExecutable *pipeline, VernonStringView name,
                                                         VernonProgramParameterView *parameter) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !parameter || (name.size && !name.data))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const program::BoundarySlot *slot = findPublicBoundary(*executableProgram(*pipeline), name);
    if (!slot)
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillBoundaryParameterView(*pipeline, *slot, *parameter) ? VERNON_STATUS_OK : VERNON_STATUS_PARSE_ERROR;
}

VernonStatus vernonRuntimeProgramExecutableGetParameterValueLeaf(const VernonProgramExecutable *pipeline,
                                                                 VernonStringView parameterName, size_t leafIndex,
                                                                 VernonProgramValueLeafView *leaf) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !leaf || leaf->struct_size < sizeof(VernonProgramValueLeafView) ||
        (parameterName.size && !parameterName.data))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const program::BoundarySlot *slot = findPublicBoundary(*executableProgram(*pipeline), parameterName);
    const ValueLayout *layout = slot ? boundaryLayoutView(*pipeline, *slot) : nullptr;
    if (!slot || !layout ||
        (slot->category != program::BoundaryCategory::Value &&
         slot->category != program::BoundaryCategory::StorageView) ||
        leafIndex >= layout->leaves.size())
        return VERNON_STATUS_INVALID_ARGUMENT;
    const ValueLeaf &source = layout->leaves[leafIndex];
    leaf->value = layout->abiLeaves[leafIndex];
    leaf->path = source.abiPath.empty() ? nullptr : source.abiPath.data();
    leaf->path_count = source.abiPath.size();
    leaf->static_shape = source.shape.empty() ? nullptr : source.shape.data();
    leaf->static_rank = static_cast<uint32_t>(source.shape.size());
    return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeProgramExecutableGetImageConstraintByParameterIndex(
    const VernonProgramExecutable *pipeline, size_t parameterIndex, VernonProgramImageConstraintView *constraint) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !constraint || constraint->struct_size < sizeof(VernonProgramImageConstraintView))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const program::BoundarySlot *slot = publicBoundaryAt(*executableProgram(*pipeline), parameterIndex);
    return slot ? fillBoundaryImageConstraintView(*slot, *constraint) : VERNON_STATUS_INVALID_ARGUMENT;
}

VernonStatus vernonRuntimeProgramExecutableFindImageConstraint(const VernonProgramExecutable *pipeline,
                                                               VernonStringView parameterName,
                                                               VernonProgramImageConstraintView *constraint) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !constraint || constraint->struct_size < sizeof(VernonProgramImageConstraintView) ||
        (parameterName.size && !parameterName.data))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const program::BoundarySlot *slot = findPublicBoundary(*executableProgram(*pipeline), parameterName);
    return slot ? fillBoundaryImageConstraintView(*slot, *constraint) : VERNON_STATUS_INVALID_ARGUMENT;
}

size_t vernonRuntimeProgramExecutableGetOutputCount(const VernonProgramExecutable *pipeline) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    return 0;
}

namespace {

/* Collects the forward graph's graphics nodes in graph order, which is the order the controls query indexes by. */
std::vector<const program::Node *> forwardGraphicsNodes(const VernonProgramExecutable *pipeline) {
    std::vector<const program::Node *> nodes;
    const program::Program *program = pipeline ? executableProgram(*pipeline) : nullptr;
    if (!program)
        return nodes;
    const program::Graph *graph = program::findGraph(*program, "forward");
    if (!graph)
        return nodes;
    for (const program::Node &node : graph->nodes) {
        if (program::executionKind(node) == program::ExecutionKind::Graphics)
            nodes.push_back(&node);
    }
    return nodes;
}

} // namespace

size_t vernonRuntimeProgramExecutableGetGraphicsNodeCount(const VernonProgramExecutable *pipeline) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    return forwardGraphicsNodes(pipeline).size();
}

VernonStatus vernonRuntimeProgramExecutableGetGraphicsControlsByIndex(const VernonProgramExecutable *pipeline,
                                                                      size_t index,
                                                                      VernonProgramGraphicsControlsView *output) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !output)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const std::vector<const program::Node *> nodes = forwardGraphicsNodes(pipeline);
    if (index >= nodes.size())
        return VERNON_STATUS_INVALID_ARGUMENT;
    const program::Node &node = *nodes[index];
    const program::GraphicsOperation &graphics = program::graphicsOperation(node);
    *output = {sizeof(VernonProgramGraphicsControlsView), node.id, graphics.renderPassControl,
               graphics.drawCommandControl, graphics.dynamicStateControl};
    return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeProgramExecutableGetOutputByIndex(const VernonProgramExecutable *pipeline, size_t index,
                                                            VernonProgramOutputView *output) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    (void)index;
    return (!pipeline || !output) ? VERNON_STATUS_INVALID_ARGUMENT : VERNON_STATUS_INVALID_ARGUMENT;
}

VernonStatus vernonRuntimeProgramExecutableFindOutput(const VernonProgramExecutable *pipeline, VernonStringView name,
                                                      VernonProgramOutputView *output) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    (void)name;
    return (!pipeline || !output) ? VERNON_STATUS_INVALID_ARGUMENT : VERNON_STATUS_INVALID_ARGUMENT;
}

void vernonRuntimeProgramExecutableDestroy(VernonProgramExecutable *pipeline) {
    if (!pipeline)
        return;
    RuntimeDiagnosticScope diagnostic(pipeline->context);
    --pipeline->context->livePipelines;
    delete pipeline;
}

void vernonRuntimeStageExecutableDestroy(VernonStageExecutable *stage) {
    if (!stage)
        return;
    RuntimeDiagnosticScope diagnostic(stage->context);
    --stage->context->livePipelines;
    delete stage;
}

size_t vernonRuntimeStageExecutableGetParameterCount(const VernonStageExecutable *stage) {
    return stage ? stage->bindingProjection.parameters.size() : 0;
}

VernonStatus vernonRuntimeStageExecutableGetParameterByIndex(const VernonStageExecutable *stage, size_t index,
                                                             VernonProgramParameterView *parameter) {
    if (!stage || !parameter || index >= stage->bindingProjection.parameters.size())
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillParameterView(stage->bindingProjection.parameters[index], *parameter) ? VERNON_STATUS_OK
                                                                                     : VERNON_STATUS_PARSE_ERROR;
}

VernonStatus vernonRuntimeStageExecutableFindParameter(const VernonStageExecutable *stage, VernonStringView name,
                                                       VernonProgramParameterView *parameter) {
    if (!stage || !parameter || (name.size && !name.data))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const auto found =
        std::find_if(stage->bindingProjection.parameters.begin(), stage->bindingProjection.parameters.end(),
                     [&](const Parameter &candidate) { return stringViewEquals(name, candidate.name); });
    return found != stage->bindingProjection.parameters.end() && fillParameterView(*found, *parameter)
               ? VERNON_STATUS_OK
               : VERNON_STATUS_INVALID_ARGUMENT;
}

VernonStatus vernonRuntimeStageExecutableGetParameterValueLeaf(const VernonStageExecutable *stage,
                                                               VernonStringView parameterName, size_t leafIndex,
                                                               VernonProgramValueLeafView *leaf) {
    if (!stage || !leaf || leaf->struct_size < sizeof(*leaf) || (parameterName.size && !parameterName.data))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const auto parameter =
        std::find_if(stage->bindingProjection.parameters.begin(), stage->bindingProjection.parameters.end(),
                     [&](const Parameter &candidate) { return stringViewEquals(parameterName, candidate.name); });
    if (parameter == stage->bindingProjection.parameters.end() || parameter->kind != "tensor")
        return VERNON_STATUS_INVALID_ARGUMENT;
    const ValueLayout &layout = parameterLogicalLeafLayout(*parameter);
    if (leafIndex >= layout.leaves.size())
        return VERNON_STATUS_INVALID_ARGUMENT;
    const ValueLeaf &source = layout.leaves[leafIndex];
    leaf->value = layout.abiLeaves[leafIndex];
    leaf->path = source.abiPath.empty() ? nullptr : source.abiPath.data();
    leaf->path_count = source.abiPath.size();
    leaf->static_shape = source.shape.empty() ? nullptr : source.shape.data();
    leaf->static_rank = static_cast<uint32_t>(source.shape.size());
    return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeStageExecutableGetImageConstraintByParameterIndex(
    const VernonStageExecutable *stage, size_t parameterIndex, VernonProgramImageConstraintView *constraint) {
    if (!stage || !constraint || constraint->struct_size < sizeof(*constraint) ||
        parameterIndex >= stage->bindingProjection.parameters.size())
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillImageConstraintView(stage->bindingProjection.parameters[parameterIndex], *constraint);
}

VernonStatus vernonRuntimeStageExecutableFindImageConstraint(const VernonStageExecutable *stage,
                                                             VernonStringView parameterName,
                                                             VernonProgramImageConstraintView *constraint) {
    if (!stage || !constraint || constraint->struct_size < sizeof(*constraint) ||
        (parameterName.size && !parameterName.data))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const auto found =
        std::find_if(stage->bindingProjection.parameters.begin(), stage->bindingProjection.parameters.end(),
                     [&](const Parameter &parameter) { return stringViewEquals(parameterName, parameter.name); });
    return found == stage->bindingProjection.parameters.end() ? VERNON_STATUS_INVALID_ARGUMENT
                                                              : fillImageConstraintView(*found, *constraint);
}

size_t vernonRuntimeStageExecutableGetOutputCount(const VernonStageExecutable *stage) {
    return stage ? stage->bindingProjection.outputs.size() : 0;
}

VernonStatus vernonRuntimeStageExecutableGetOutputByIndex(const VernonStageExecutable *stage, size_t index,
                                                          VernonProgramOutputView *output) {
    if (!stage || !output || index >= stage->bindingProjection.outputs.size())
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillOutputView(stage->bindingProjection.outputs[index], *output) ? VERNON_STATUS_OK
                                                                            : VERNON_STATUS_PARSE_ERROR;
}

VernonStatus vernonRuntimeStageExecutableFindOutput(const VernonStageExecutable *stage, VernonStringView name,
                                                    VernonProgramOutputView *output) {
    if (!stage || !output || (name.size && !name.data))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const auto found = std::find_if(stage->bindingProjection.outputs.begin(), stage->bindingProjection.outputs.end(),
                                    [&](const Output &value) { return stringViewEquals(name, value.name); });
    return found != stage->bindingProjection.outputs.end() && fillOutputView(*found, *output)
               ? VERNON_STATUS_OK
               : VERNON_STATUS_INVALID_ARGUMENT;
}

namespace {

struct PipelineComputeResource {
    uint32_t value{};
    std::string access;
};

struct PipelineComputePassDescription {
    std::string name;
    uint64_t grid[3]{1, 1, 1};
    std::vector<uint32_t> operands;
    std::vector<uint32_t> results;
    std::vector<PipelineComputeResource> resources;
};

extern "C++" std::string programNodeName(const program::Node &node) {
    return "program." + std::to_string(node.id) + "." + (node.name.empty() ? node.stage : node.name);
}

extern "C++" PipelineComputePassDescription describeProgramNode(const program::Node &node) {
    PipelineComputePassDescription description;
    description.name = programNodeName(node);
    const program::ComputeOperation &compute = program::computeOperation(node);
    for (size_t axis = 0; axis < 3; ++axis)
        description.grid[axis] =
            compute.workgroups[axis].kind == program::ControlKind::Static ? compute.workgroups[axis].value : 0;
    description.operands = node.operands;
    description.results = node.results;
    description.resources.reserve(node.accesses.size());
    for (const program::ResourceAccess &resource : node.accesses) {
        const uint32_t value = resource.kind == program::AccessKind::Read         ? resource.value
                               : resource.kind == program::AccessKind::Initialize ? resource.after
                                                                                  : resource.before;
        const std::string access = resource.kind == program::AccessKind::Read         ? "read"
                                   : resource.kind == program::AccessKind::Initialize ? "write"
                                                                                      : resource.access;
        description.resources.push_back({value, access.empty() ? "read_write" : access});
    }
    return description;
}

bool resolveProgramGrid(vernon::runtime::program::DispatchMapping mapping, const uint64_t staticGrid[3],
                        const std::vector<VernonProgramArgument> &arguments, VernonLaunchSize &grid,
                        std::string &error) {
    grid = {static_cast<uint32_t>(staticGrid[0]), static_cast<uint32_t>(staticGrid[1]),
            static_cast<uint32_t>(staticGrid[2])};
    if (mapping == vernon::runtime::program::DispatchMapping::StaticGrid)
        return true;
    for (const VernonProgramArgument &argument : arguments) {
        if (argument.kind != VERNON_PROGRAM_TENSOR || (argument.tensor.rank && !argument.tensor.shape))
            continue;
        uint64_t count = 1;
        for (uint32_t dimension = 0; dimension < argument.tensor.rank; ++dimension) {
            const uint64_t extent = argument.tensor.shape[dimension];
            if (!extent) {
                count = 0;
                break;
            }
            if (count > std::numeric_limits<uint32_t>::max() / extent) {
                error = "linearized Program dispatch exceeds the portable uint32 range";
                return false;
            }
            count *= extent;
        }
        grid = {static_cast<uint32_t>(count), 1, 1};
        return true;
    }
    error = "linearized Program dispatch has no Tensor extent source";
    return false;
}

class PipelineComputePass final : public vernon::execution::ComputePass {
public:
    PipelineComputePass(const program::Node &node, VernonStageExecutable &pipeline,
                        vernon::runtime::program_execution::MaterializedNodeFrame materialized,
                        const std::vector<vernon::execution::GraphBuffer> &resources,
                        const std::vector<VernonProgramArgument> *arena, const program::Program *program,
                        VernonLaunchSize grid)
        : ComputePass(programNodeName(node)), description_(describeProgramNode(node)), pipeline_(pipeline),
          materialized_(std::move(materialized)), resources_(resources), arena_(arena), program_(program), grid_(grid) {
    }

    void declare() override {
        for (const PipelineComputeResource &use : description_.resources) {
            if (use.access == "read")
                read(resources_.at(use.value));
            else if (use.access == "write")
                write(resources_.at(use.value));
            else
                readWrite(resources_.at(use.value));
        }
        setFlags(vernon::execution::PassSideEffect);
    }

    VernonRhiStatus execute(vernon::execution::ComputeEncoder &,
                            const vernon::execution::ExecutionResources &) override {
        VernonStageInvocationDescriptor invocation{};
        invocation.struct_size = sizeof(invocation);
        invocation.abi_version = VERNON_PROGRAM_VERSION;
        invocation.arguments = materialized_.arguments.data();
        invocation.argument_count = materialized_.arguments.size();
        invocation.compute_grid = grid_;
        PlannedComputeLaunch plan;
        std::string error;
        if (!materialized_.prepareHost(error)) {
            invocationDiagnostic(*pipeline_.context) = description_.name + ": " + error;
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        }
        if (!grid_.x || !grid_.y || !grid_.z)
            return VERNON_RHI_STATUS_OK;
        const uint32_t workgroup[3]{pipeline_.workgroupSize.x, pipeline_.workgroupSize.y, pipeline_.workgroupSize.z};
        if (pipeline_.context && pipeline_.context->backend == VERNON_RUNTIME_CPU) {
            VernonAdTapeAllocator *allocator = nullptr;
            VernonAdRegionHandle root = VERNON_AD_INVALID_REGION_HANDLE;
            const auto bindTape = [&](uint32_t valueId) {
                if (!arena_ || !program_ || valueId >= arena_->size() || valueId >= program_->values.size() ||
                    !program::isTapeValueType(program_->values[valueId].type))
                    return;
                const VernonProgramArgument &argument = (*arena_)[valueId];
                constexpr size_t allocatorDescriptorBytes = sizeof(VernonAdTapeAllocator *);
                constexpr size_t rootDescriptorBytes = sizeof(VernonAdRegionHandle);
                if (argument.kind != VERNON_PROGRAM_TENSOR || !argument.tensor.host_data ||
                    argument.tensor.byte_size < allocatorDescriptorBytes + rootDescriptorBytes)
                    return;
                const auto *bytes = static_cast<const uint8_t *>(argument.tensor.host_data);
                std::memcpy(&allocator, bytes, allocatorDescriptorBytes);
                std::memcpy(&root, bytes + allocatorDescriptorBytes, rootDescriptorBytes);
            };
            for (uint32_t operand : description_.operands)
                bindTape(operand);
            for (uint32_t result : description_.results)
                bindTape(result);
            vernon::runtime::setCpuProgramTape(pipeline_, allocator, root);
        }
        if (!planComputeInvocation(pipeline_.bindingProjection, pipeline_.workgroupSize, invocation, plan, error)) {
            invocationDiagnostic(*pipeline_.context) = std::move(error);
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        }
        const uint32_t grid[3]{plan.grid.x, plan.grid.y, plan.grid.z};
        if (!validateDispatchContract(pipeline_.dispatchContract, grid, workgroup, error)) {
            invocationDiagnostic(*pipeline_.context) = std::move(error);
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        }
        const VernonStatus status = invokeBackendComputePipeline(pipeline_, plan);
        if (status != VERNON_STATUS_OK) {
            std::string &detail = invocationDiagnostic(*pipeline_.context);
            if (detail.empty())
                detail = "pipeline compute failed";
            detail = description_.name + ": " + detail;
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        }
        if (!commitComputeResults(plan, error)) {
            invocationDiagnostic(*pipeline_.context) = description_.name + ": " + error;
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        }
        if (!materialized_.commitHost(error)) {
            invocationDiagnostic(*pipeline_.context) = description_.name + ": " + error;
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        }
        return VERNON_RHI_STATUS_OK;
    }

private:
    PipelineComputePassDescription description_;
    VernonStageExecutable &pipeline_;
    vernon::runtime::program_execution::MaterializedNodeFrame materialized_;
    const std::vector<vernon::execution::GraphBuffer> &resources_;
    const std::vector<VernonProgramArgument> *arena_{};
    const program::Program *program_{};
    VernonLaunchSize grid_{};
};

struct ManagedGraphicsCommandContext {
    VernonStageExecutable &pipeline;
    vernon::runtime::program_execution::MaterializedNodeFrame materialized;
    std::shared_ptr<RuntimeProgramControl> renderPass;
    std::shared_ptr<RuntimeProgramControl> draw;
    std::shared_ptr<RuntimeProgramControl> dynamic;
    VernonStageInvocationDescriptor invocation{};
    PlannedGraphicsInvocation plan;
    VernonGraphicsState graphicsState{};
    std::vector<VernonColorBlendState> colorBlends;
    VernonDrawCommand defaultDraw{};
};

struct ManagedGraphicsCommandBatch {
    std::vector<std::shared_ptr<ManagedGraphicsCommandContext>> draws;
};

void prepareGraphicsPipelineState(const program::GraphicsOperation &operation, ManagedGraphicsCommandContext &context) {
    const program::GraphicsPipelineState &source = operation.pipelineState;
    context.colorBlends.clear();
    context.colorBlends.reserve(source.colorBlends.size());
    for (size_t location = 0; location < source.colorBlends.size(); ++location)
        context.colorBlends.push_back(source.colorBlends.at(static_cast<uint32_t>(location)));
    context.graphicsState = {};
    context.graphicsState.struct_size = sizeof(context.graphicsState);
    context.graphicsState.topology = source.topology;
    context.graphicsState.rasterization = source.rasterization;
    context.graphicsState.depth_stencil = source.depthStencil;
    context.graphicsState.color_blends = context.colorBlends.data();
    context.graphicsState.color_blend_count = context.colorBlends.size();
}

VernonRhiStatus encodeManagedGraphicsBatch(void *opaque, VernonRhiCommandEncoder encoder) {
    auto &batch = *static_cast<ManagedGraphicsCommandBatch *>(opaque);
    if (batch.draws.empty())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const VernonRhiDevice device = batch.draws.front()->pipeline.context->rhiDevice;
    if (!vernon::rhi::beginProviderRendering(device, encoder))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    VernonRhiStatus status = VERNON_RHI_STATUS_OK;
    for (const std::shared_ptr<ManagedGraphicsCommandContext> &context : batch.draws) {
        if (!context || context->pipeline.context->rhiDevice.index != device.index ||
            vernonRuntimeReferenceRhiCommandEncoder(context->pipeline.context, encoder,
                                                    &context->invocation.command_encoder) != VERNON_STATUS_OK) {
            status = VERNON_RHI_STATUS_INVALID_ARGUMENT;
            break;
        }
        if (invokeBackendPipeline(context->pipeline, context->invocation, context->plan) != VERNON_STATUS_OK) {
            status = VERNON_RHI_STATUS_INTERNAL_ERROR;
            break;
        }
    }
    const VernonRhiStatus ended = vernonRhiCommandEncoderEndRendering(device, encoder);
    return status == VERNON_RHI_STATUS_OK ? ended : status;
}

VernonStatus executePipelineProgramGraphImpl(
    VernonProgramExecutable &pipeline, const program::Graph &graph,
    vernon::runtime::program_execution::ProgramInvocationState &arena,
    const vernon::runtime::program_execution::ResolvePhysicalEndpoint &resolvePhysicalEndpoint) {
    const program::ResolvedExecutionPlan &execution = *pipeline.executionPlan;
    const program::ResolvedProgram &resolved = *execution.resolvedProgram;
    const program::Program &canonicalProgram = resolved.program;
    const std::vector<VernonProgramArgument> &valueArguments = arena.arguments();
    if (valueArguments.size() != canonicalProgram.values.size())
        return fail(pipeline.context, "Program graph value set does not match its execution topology");
    const auto canonicalGraph =
        std::find_if(canonicalProgram.graphs.begin(), canonicalProgram.graphs.end(),
                     [&](const program::Graph &candidate) { return candidate.direction == graph.direction; });
    if (canonicalGraph == canonicalProgram.graphs.end())
        return fail(pipeline.context, "Program graph is not part of its resolved owner");
    const ProgramInvocationContext *invocationContext = arena.invocationContext();
    const auto control = [&](uint32_t slot,
                             RuntimeProgramControl::Kind kind) -> std::shared_ptr<RuntimeProgramControl> {
        if (!invocationContext)
            return nullptr;
        const std::shared_ptr<void> *payload = invocationContext->controls.find(programControlKey(slot, kind));
        return payload ? std::static_pointer_cast<RuntimeProgramControl>(*payload) : nullptr;
    };
    std::string controlBindingError;
    if (!bindProgramGraphicsControlResources(
            canonicalProgram, graph, execution, arena,
            [&](uint32_t slot) -> const VernonRenderPass * {
                auto renderPass = control(slot, RuntimeProgramControl::RenderPass);
                if (!renderPass)
                    return nullptr;
                renderPass->refresh();
                return &renderPass->renderPass;
            },
            controlBindingError))
        return fail(pipeline.context, std::move(controlBindingError));
    if (pipeline.context->backend != VERNON_RUNTIME_CPU) {
        vernon::execution::detail::RhiCommandExecutionPlan commandPlan;
        vernon::runtime::program_execution::ResolvedTransferExecutor transfers(*pipeline.context, arena);
        const std::optional<program::GraphDirection> graphDirection = program::graphDirection(graph.direction);
        std::string transferError;
        if (!graphDirection || !transfers.prepareGraph(*graphDirection, transferError))
            return fail(pipeline.context, std::move(transferError));
        std::shared_ptr<ManagedGraphicsCommandBatch> graphicsBatch;
        GraphicsScopePlanner graphicsScopes;
        const auto flushCommands = [&]() {
            if (commandPlan.commands.nodes.empty())
                return VERNON_STATUS_OK;
            graphicsBatch.reset();
            const VernonStatus status =
                vernon::runtime::program_execution::executeCommandPlanAndWait(*pipeline.context, commandPlan);
            commandPlan = {};
            return status;
        };
        std::vector<vernon::runtime::program_execution::MaterializedNodeFrame> materializedNodes;
        materializedNodes.reserve(graph.nodes.size());
        for (const program::Node &node : graph.nodes) {
            const std::optional<program::GraphDirection> direction = program::graphDirection(graph.direction);
            const program::ResolvedNodePlan *resolvedNode = direction ? execution.node(*direction, node.id) : nullptr;
            if (!resolvedNode)
                return fail(pipeline.context, "pipeline node has no resolved kernel stage");
            if (program::executionKind(node) == program::ExecutionKind::Graphics) {
                const program::GraphicsOperation &graphics = program::graphicsOperation(node);
                auto renderPass = control(graphics.renderPassControl, RuntimeProgramControl::RenderPass);
                if (!renderPass)
                    return fail(pipeline.context, "managed graphics node has no bound RenderPass control");
                auto draw = control(graphics.drawCommandControl, RuntimeProgramControl::DrawCommand);
                auto dynamic = control(graphics.dynamicStateControl, RuntimeProgramControl::DynamicState);
                renderPass->refresh();
                if (renderPass->renderPass.color_attachment_count != graphics.colorAttachments.size())
                    return fail(pipeline.context,
                                "managed graphics fragment outputs must exactly match the color attachments");
                if (graphics.depthStencilAttachment.has_value() !=
                    static_cast<bool>(renderPass->renderPass.depth_attachment))
                    return fail(pipeline.context,
                                "managed graphics depth attachment does not match the canonical render pass");
                materializedNodes.emplace_back();
                std::string materializationError;
                if (!vernon::runtime::program_execution::materializeNodeFrame(
                        arena, canonicalProgram, node, *resolvedNode, resolvePhysicalEndpoint, materializedNodes.back(),
                        materializationError))
                    return fail(pipeline.context, std::move(materializationError));
                if (const VernonStatus status = transfers.appendBeforeConsumer(
                        {*direction, node.id}, materializedNodes.back().deviceCopiesBefore, commandPlan, transferError);
                    status != VERNON_STATUS_OK)
                    return status;
                if (draw)
                    draw->refresh();
                auto graphicsContext = std::make_shared<ManagedGraphicsCommandContext>(
                    ManagedGraphicsCommandContext{*resolvedNode->stage, std::move(materializedNodes.back()),
                                                  std::move(renderPass), std::move(draw), std::move(dynamic)});
                prepareGraphicsPipelineState(graphics, *graphicsContext);
                VernonStageInvocationDescriptor &invocation = graphicsContext->invocation;
                invocation.struct_size = sizeof(invocation);
                invocation.abi_version = VERNON_PROGRAM_VERSION;
                invocation.arguments = graphicsContext->materialized.arguments.data();
                invocation.argument_count = graphicsContext->materialized.arguments.size();
                invocation.render_pass = &graphicsContext->renderPass->renderPass;
                graphicsContext->defaultDraw = {sizeof(VernonDrawCommand), nullptr,
                                                static_cast<uint32_t>(graphics.vertexCount),
                                                static_cast<uint32_t>(graphics.instanceCount)};
                invocation.draw_command =
                    graphicsContext->draw ? &graphicsContext->draw->draw : &graphicsContext->defaultDraw;
                invocation.dynamic_state = graphicsContext->dynamic ? &graphicsContext->dynamic->dynamic : nullptr;
                invocation.graphics_state = &graphicsContext->graphicsState;
                std::string graphicsError;
                if (!planGraphicsInvocation(
                        resolvedNode->stage->bindingProjection, invocation,
                        [](void *userData, VernonRuntimeProviderResourceReference resource,
                           VernonRuntimeProviderImageDescription *description) {
                            return describeBackendImage(*static_cast<VernonRuntimeContext *>(userData), resource,
                                                        *description);
                        },
                        pipeline.context, graphicsContext->plan, graphicsError))
                    return fail(pipeline.context, std::move(graphicsError));
                if (!checkProgramGraphicsAttachmentSignature(graphics, graphicsContext->plan, graphicsError))
                    return fail(pipeline.context, std::move(graphicsError));
                if (!graphicsScopes.canAppend(graphicsContext->plan)) {
                    graphicsBatch.reset();
                    graphicsScopes.reset();
                }
                if (!graphicsBatch) {
                    graphicsBatch = std::make_shared<ManagedGraphicsCommandBatch>();
                    vernon::execution::detail::RhiCommandExecutionPlan nodePlan;
                    vernon::execution::detail::CommandNode drawCommand;
                    drawCommand.kind = vernon::execution::detail::CommandNodeKind::Derivative;
                    drawCommand.queue = vernon::execution::detail::CommandQueueClass::Graphics;
                    nodePlan.commands.nodes.push_back(std::move(drawCommand));
                    nodePlan.encoders.push_back({encodeManagedGraphicsBatch, graphicsBatch.get()});
                    nodePlan.retainedContexts.push_back(graphicsBatch);
                    std::string compositionError;
                    if (!vernon::execution::detail::appendRhiCommandExecutionPlan(commandPlan, std::move(nodePlan),
                                                                                  true, compositionError))
                        return fail(pipeline.context, std::move(compositionError));
                }
                graphicsScopes.append(graphicsContext->plan);
                graphicsBatch->draws.push_back(std::move(graphicsContext));
                continue;
            }
            graphicsBatch.reset();
            graphicsScopes.reset();
            materializedNodes.emplace_back();
            std::string materializationError;
            if (!vernon::runtime::program_execution::materializeNodeFrame(
                    arena, canonicalProgram, node, *resolvedNode, resolvePhysicalEndpoint, materializedNodes.back(),
                    materializationError))
                return fail(pipeline.context, std::move(materializationError));
            if (const VernonStatus status = transfers.appendBeforeConsumer(
                    {*direction, node.id}, materializedNodes.back().deviceCopiesBefore, commandPlan, transferError);
                status != VERNON_STATUS_OK)
                return status;
            vernon::execution::detail::RhiCommandExecutionPlan nodePlan;
            VernonLaunchSize grid{};
            std::string gridError;
            const program::ComputeOperation &compute = program::computeOperation(node);
            uint64_t staticGrid[3]{};
            for (size_t axis = 0; axis < 3; ++axis)
                if (!arena.resolveControl(canonicalProgram, compute.workgroups[axis], staticGrid[axis], gridError))
                    return fail(pipeline.context, std::move(gridError));
            const auto *computeControls = std::get_if<program::ResolvedComputeControls>(&resolvedNode->controls);
            if (!computeControls || !resolveProgramGrid(computeControls->dispatchMapping, staticGrid,
                                                        materializedNodes.back().arguments, grid, gridError))
                return fail(pipeline.context, std::move(gridError));
            if (!grid.x || !grid.y || !grid.z)
                continue;
            const VernonStatus planned = vernon::runtime::program_execution::buildPipelineCommandPlan(
                *pipeline.context, {}, {}, *resolvedNode->stage, materializedNodes.back().arguments, grid,
                materializedNodes.back().deviceCopiesAfter, vernon::execution::detail::CommandNodeKind::Derivative,
                nodePlan);
            if (planned != VERNON_STATUS_OK)
                return planned;
            std::string compositionError;
            if (!vernon::execution::detail::appendRhiCommandExecutionPlan(commandPlan, std::move(nodePlan), true,
                                                                          compositionError))
                return fail(pipeline.context, std::move(compositionError));
        }
        return flushCommands();
    }
    vernon::execution::ExecutionGraph executionGraph;
    std::vector<vernon::execution::GraphBuffer> resources;
    resources.reserve(valueArguments.size());
    std::vector<char> used(canonicalProgram.values.size());
    program::markGraphValues(graph, used);
    for (size_t index = 0; index < valueArguments.size(); ++index) {
        if (index >= used.size() || !used[index]) {
            resources.push_back({});
            continue;
        }
        const VernonProgramArgument &argument = valueArguments[index];
        if (argument.kind != VERNON_PROGRAM_TENSOR)
            return fail(pipeline.context, "Program graph requires materialized tensor values");
        if (argument.tensor.storage == VERNON_TENSOR_HOST && argument.tensor.host_data) {
            const uint64_t identity = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(argument.tensor.host_data));
            const bool output =
                std::any_of(graph.outputs.begin(), graph.outputs.end(),
                            [&](const program::GraphOutput &candidate) { return candidate.value == index; });
            resources.push_back(executionGraph.importHostBuffer(identity, output));
            continue;
        }
        const VernonRhiBuffer device = arena.buffer(static_cast<uint32_t>(index));
        if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
            return fail(pipeline.context, "Program graph requires a materialized host or RHI Storage backing");
        const bool output =
            std::any_of(graph.outputs.begin(), graph.outputs.end(),
                        [&](const program::GraphOutput &candidate) { return candidate.value == index; });
        resources.push_back(executionGraph.importBuffer(device, output));
    }
    std::vector<vernon::execution::ExecutionPass *> passes(graph.nodes.size());
    for (const program::Node &node : graph.nodes) {
        const std::optional<program::GraphDirection> direction = program::graphDirection(graph.direction);
        const program::ResolvedNodePlan *resolvedNode = direction ? execution.node(*direction, node.id) : nullptr;
        if (!resolvedNode)
            return fail(pipeline.context, "pipeline node has no resolved kernel stage");
        VernonStageExecutable &stage = *resolvedNode->stage;
        vernon::runtime::program_execution::MaterializedNodeFrame materialized;
        std::string materializationError;
        if (!vernon::runtime::program_execution::materializeNodeFrame(arena, canonicalProgram, node, *resolvedNode,
                                                                      resolvePhysicalEndpoint, materialized,
                                                                      materializationError))
            return fail(pipeline.context, std::move(materializationError));
        uint64_t controlGrid[3]{};
        for (size_t axis = 0; axis < 3; ++axis)
            if (!arena.resolveControl(canonicalProgram, program::computeOperation(node).workgroups[axis],
                                      controlGrid[axis], materializationError))
                return fail(pipeline.context, std::move(materializationError));
        VernonLaunchSize grid{};
        const auto *computeControls = std::get_if<program::ResolvedComputeControls>(&resolvedNode->controls);
        if (!computeControls || !resolveProgramGrid(computeControls->dispatchMapping, controlGrid,
                                                    materialized.arguments, grid, materializationError))
            return fail(pipeline.context, std::move(materializationError));
        auto &pass = executionGraph.emplacePass<PipelineComputePass>(node, stage, std::move(materialized), resources,
                                                                     &valueArguments, &canonicalProgram, grid);
        for (uint32_t dependency : execution.predecessors(*direction, node.id)) {
            if (dependency >= passes.size() || !passes[dependency])
                return fail(pipeline.context, "Program node dependency is not materialized");
            pass.dependsOn(*passes[dependency]);
        }
        passes[node.id] = &pass;
    }
    std::string error;
    std::shared_ptr<vernon::execution::CompiledExecutionGraph> compiled = executionGraph.compile(error);
    if (!compiled)
        return fail(pipeline.context, "cannot compile pipeline ExecutionGraph: " + error);
    vernon::execution::ExecutionSubmission submission = compiled->submit();
    if (submission.wait() != VERNON_RHI_STATUS_OK) {
        const std::string detail = invocationDiagnostic(*pipeline.context);
        return fail(pipeline.context, detail.empty() ? "pipeline ExecutionGraph submission failed"
                                                     : "pipeline ExecutionGraph submission failed: " + detail);
    }
    return VERNON_STATUS_OK;
}

VernonStatus encodeStageInvocation(VernonStageExecutable &pipeline, const VernonStageInvocationDescriptor &invocation) {
    if (!pipeline.bindingProjection.compute.empty()) {
        const uint32_t grid[3]{invocation.compute_grid.x, invocation.compute_grid.y, invocation.compute_grid.z};
        const uint32_t workgroup[3]{pipeline.workgroupSize.x, pipeline.workgroupSize.y, pipeline.workgroupSize.z};
        if (!validateDispatchContract(pipeline.dispatchContract, grid, workgroup,
                                      invocationDiagnostic(*pipeline.context)))
            return VERNON_STATUS_INVALID_ARGUMENT;
        PlannedComputeLaunch plan;
        std::string planningError;
        if (!planComputeInvocation(pipeline.bindingProjection, pipeline.workgroupSize, invocation, plan, planningError))
            return fail(pipeline.context, planningError);
        return invokeBackendComputePipeline(pipeline, plan);
    }

    PlannedGraphicsInvocation plan;
    std::string planningError;
    if (!planGraphicsInvocation(
            pipeline.bindingProjection, invocation,
            [](void *userData, VernonRuntimeProviderResourceReference resource,
               VernonRuntimeProviderImageDescription *description) {
                return describeBackendImage(*static_cast<VernonRuntimeContext *>(userData), resource, *description);
            },
            pipeline.context, plan, planningError))
        return fail(pipeline.context, planningError);
    return invokeBackendPipeline(pipeline, invocation, plan);
}

} // namespace

VernonStatus vernonRuntimeStageSubmit(VernonStageExecutable *pipeline,
                                      const VernonStageInvocationDescriptor *invocation, VernonSubmission **output) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (output)
        *output = nullptr;
    if (!pipeline || !invocation || !output || invocation->struct_size < sizeof(VernonStageInvocationDescriptor) ||
        invocation->abi_version != VERNON_PROGRAM_VERSION || (invocation->argument_count && !invocation->arguments) ||
        invocation->command_encoder.value != 0)
        return fail(pipeline ? pipeline->context : nullptr, "invalid pipeline submission");
    std::unique_ptr<VernonSubmission> submission;
    try {
        submission = std::make_unique<VernonSubmission>();
        submission->contextLease = acquireContextLease(*pipeline->context);
        submission->device = pipeline->context->rhiDevice;
    } catch (const std::bad_alloc &) {
        return fail(pipeline->context, "cannot allocate pipeline submission", VERNON_STATUS_INTERNAL_ERROR);
    }
    if (pipeline->context->rhiDevice.index == VERNON_RHI_INVALID_HANDLE_INDEX) {
        const VernonStatus status = encodeStageInvocation(*pipeline, *invocation);
        if (status != VERNON_STATUS_OK)
            return status;
        submission->state = VERNON_SUBMISSION_SUCCEEDED;
        *output = submission.release();
        return VERNON_STATUS_OK;
    }

    const bool graphics = pipeline->bindingProjection.compute.empty();
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = graphics ? VERNON_RHI_QUEUE_GRAPHICS : VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder native{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    if (vernonRhiDeviceCreateCommandEncoder(pipeline->context->rhiDevice, &descriptor, &native) != VERNON_RHI_STATUS_OK)
        return fail(pipeline->context, "failed to create the immediate command encoder");
    VernonStageInvocationDescriptor encoded = *invocation;
    VernonStatus status = referenceBackendCommandEncoder(*pipeline->context, native, encoded.command_encoder);
    bool rendering = false;
    if (status == VERNON_STATUS_OK && graphics) {
        rendering = vernon::rhi::beginProviderRendering(pipeline->context->rhiDevice, native);
        if (!rendering)
            status = fail(pipeline->context, "failed to begin immediate rendering");
    }
    if (status == VERNON_STATUS_OK)
        status = encodeStageInvocation(*pipeline, encoded);
    if (rendering) {
        const VernonRhiStatus endStatus = vernonRhiCommandEncoderEndRendering(pipeline->context->rhiDevice, native);
        if (status == VERNON_STATUS_OK && endStatus != VERNON_RHI_STATUS_OK)
            status = fail(pipeline->context, "failed to end immediate rendering");
    }
    if (status == VERNON_STATUS_OK &&
        vernonRhiCommandEncoderFinish(pipeline->context->rhiDevice, native) != VERNON_RHI_STATUS_OK)
        status = fail(pipeline->context, "failed to finish the immediate command encoder");
    if (status == VERNON_STATUS_OK &&
        vernonRhiDeviceSubmit(pipeline->context->rhiDevice, native, &submission->completion) != VERNON_RHI_STATUS_OK) {
        const VernonStringView detail = vernonRhiDeviceGetLastError(pipeline->context->rhiDevice);
        std::string error = "failed to submit the immediate command encoder";
        if (detail.data && detail.size)
            error.append(": ").append(detail.data, detail.size);
        status = fail(pipeline->context, std::move(error));
    }
    if (status != VERNON_STATUS_OK) {
        (void)vernonRhiDeviceDestroyCommandEncoder(pipeline->context->rhiDevice, native);
        return status;
    }
    VernonRhiCompletionState completionState{};
    if (vernonRhiCompletionGetState(submission->device, submission->completion, &completionState) !=
        VERNON_RHI_STATUS_OK) {
        (void)vernonRhiDeviceDestroyCompletion(submission->device, submission->completion);
        return fail(pipeline->context, "failed to query pipeline submission", VERNON_STATUS_INTERNAL_ERROR);
    }
    submission->state = completionState == VERNON_RHI_COMPLETION_PENDING  ? VERNON_SUBMISSION_PENDING
                        : completionState == VERNON_RHI_COMPLETION_FAILED ? VERNON_SUBMISSION_FAILED
                                                                          : VERNON_SUBMISSION_SUCCEEDED;
    *output = submission.release();
    return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeStageEncode(VernonRuntimeProviderObject encoder, VernonStageExecutable *pipeline,
                                      const VernonStageInvocationDescriptor *invocation) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !invocation || invocation->struct_size < sizeof(VernonStageInvocationDescriptor) ||
        invocation->abi_version != VERNON_PROGRAM_VERSION || (invocation->argument_count && !invocation->arguments))
        return fail(pipeline ? pipeline->context : nullptr, "invalid pipeline invocation");
    VernonStageInvocationDescriptor encoded = *invocation;
    encoded.command_encoder = encoder;
    return encodeStageInvocation(*pipeline, encoded);
}

VernonProgramInstance *vernonRuntimeProgramInstanceCreate(VernonProgramExecutable *pipeline) {
    try {
        RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
        if (!pipeline || !pipeline->context)
            return nullptr;
        return new VernonProgramInstance(*pipeline);
    } catch (...) {
        return nullptr;
    }
}

void vernonRuntimeProgramInstanceDestroy(VernonProgramInstance *instance) { delete instance; }

VernonProgramInvocation *vernonRuntimeProgramInstanceBeginInvocation(VernonProgramInstance *instance) {
    try {
        if (!instance || !instance->state || !instance->state->pipeline)
            return nullptr;
        return new VernonProgramInvocation(*instance);
    } catch (...) {
        return nullptr;
    }
}

VernonStatus vernonRuntimeProgramInvocationBind(VernonProgramInvocation *invocation,
                                                const VernonProgramBindingToken *token,
                                                const VernonProgramArgument *argument,
                                                const VernonProgramResourceLease *lease, uint64_t uploadBytes,
                                                uint64_t uploadRanges) {
    VernonRuntimeContext *context = invocation && invocation->instance && invocation->instance->pipeline
                                        ? invocation->instance->pipeline->context
                                        : nullptr;
    try {
        RuntimeDiagnosticScope diagnostic(context);
        if (!invocation || invocation->finished || !token || token->struct_size < sizeof(*token) || !token->size ||
            !token->data || !argument || argument->kind < VERNON_PROGRAM_TENSOR ||
            argument->kind > VERNON_PROGRAM_SAMPLER)
            return fail(context, "invalid persistent Program binding update");
        const std::string key(static_cast<const char *>(token->data), token->size);
        if (invocation->transaction->find(argument->slot, key)) {
            invocation->transaction->observeUploads(uploadBytes, uploadRanges);
            return VERNON_STATUS_OK;
        }
        auto binding = std::make_shared<RuntimeProgramBinding>();
        binding->argument = *argument;
        if (argument->kind == VERNON_PROGRAM_TENSOR) {
            const VernonTensorView &tensor = argument->tensor;
            if (tensor.struct_size < sizeof(tensor) ||
                tensor.element_layout.struct_size < sizeof(VernonValueLayoutView) ||
                (tensor.rank && (!tensor.shape || !tensor.byte_strides)) ||
                (tensor.element_layout.layout_hash.size && !tensor.element_layout.layout_hash.data) ||
                (tensor.element_layout.leaf_count && !tensor.element_layout.leaves))
                return fail(context, "persistent Program Tensor binding has incomplete metadata");
            if (tensor.rank) {
                binding->shape.assign(tensor.shape, tensor.shape + tensor.rank);
                binding->strides.assign(tensor.byte_strides, tensor.byte_strides + tensor.rank);
            }
            if (tensor.element_layout.layout_hash.size)
                binding->layoutHash.assign(tensor.element_layout.layout_hash.data,
                                           tensor.element_layout.layout_hash.size);
            if (tensor.element_layout.leaf_count)
                binding->leaves.assign(tensor.element_layout.leaves,
                                       tensor.element_layout.leaves + tensor.element_layout.leaf_count);
        }
        if (lease) {
            if (lease->struct_size < sizeof(*lease) || (lease->retain == nullptr) != (lease->release == nullptr))
                return fail(context, "persistent Program resource lease is invalid");
            if (lease->retain) {
                lease->retain(lease->object);
                binding->lease =
                    std::shared_ptr<void>(lease->object, [release = lease->release](void *object) { release(object); });
            }
        }
        binding->refresh();
        invocation->transaction->stage(argument->slot, key, binding, uploadBytes, uploadRanges);
        return VERNON_STATUS_OK;
    } catch (const std::exception &exception) {
        return fail(context, exception.what(), VERNON_STATUS_INTERNAL_ERROR);
    }
}

extern "C++" {
namespace {
uint32_t programControlKey(uint32_t slot, RuntimeProgramControl::Kind kind) {
    if (slot > (UINT32_MAX - static_cast<uint32_t>(kind)) / 3)
        throw std::invalid_argument("Program control slot is out of range");
    return slot * 3 + static_cast<uint32_t>(kind);
}

std::string programControlToken(const VernonProgramBindingToken *token) {
    if (!token || token->struct_size < sizeof(*token) || !token->data || !token->size)
        throw std::invalid_argument("Program control binding token is invalid");
    return {static_cast<const char *>(token->data), token->size};
}

void retainProgramControlLease(RuntimeProgramControl &control, const VernonProgramResourceLease &lease) {
    if (lease.struct_size < sizeof(lease) || (lease.retain == nullptr) != (lease.release == nullptr))
        throw std::invalid_argument("Program control resource lease is invalid");
    if (!lease.retain)
        return;
    lease.retain(lease.object);
    control.leases.emplace_back(lease.object, [release = lease.release](void *object) { release(object); });
}

VernonStatus stageProgramControl(VernonProgramInvocation *invocation, uint32_t slot,
                                 const VernonProgramBindingToken *token,
                                 std::shared_ptr<RuntimeProgramControl> control) {
    VernonRuntimeContext *context = invocation && invocation->instance && invocation->instance->pipeline
                                        ? invocation->instance->pipeline->context
                                        : nullptr;
    if (!invocation || invocation->finished || !control)
        return fail(context, "invalid persistent Program control update");
    const std::string key = programControlToken(token);
    const uint32_t storageSlot = programControlKey(slot, control->kind);
    if (invocation->controlTransaction->find(storageSlot, key))
        return VERNON_STATUS_OK;
    control->refresh();
    invocation->controlTransaction->stage(storageSlot, key, std::move(control), 0, 0);
    return VERNON_STATUS_OK;
}
} // namespace
} // extern "C++"

VernonStatus vernonRuntimeProgramInvocationBindRenderPass(VernonProgramInvocation *invocation, uint32_t controlSlot,
                                                          const VernonProgramBindingToken *token,
                                                          const VernonRenderPass *renderPass,
                                                          const VernonProgramResourceLease *leases, size_t leaseCount) {
    VernonRuntimeContext *context = invocation && invocation->instance && invocation->instance->pipeline
                                        ? invocation->instance->pipeline->context
                                        : nullptr;
    try {
        RuntimeDiagnosticScope diagnostic(context);
        if (!renderPass || renderPass->struct_size < sizeof(*renderPass) ||
            (renderPass->color_attachment_count && !renderPass->color_attachments) || (leaseCount && !leases))
            return fail(context, "invalid persistent Program RenderPass control");
        auto control = std::make_shared<RuntimeProgramControl>();
        control->kind = RuntimeProgramControl::RenderPass;
        control->renderPass = *renderPass;
        control->colors.assign(renderPass->color_attachments,
                               renderPass->color_attachments + renderPass->color_attachment_count);
        if (renderPass->depth_attachment)
            control->depth = *renderPass->depth_attachment;
        for (size_t index = 0; index < leaseCount; ++index)
            retainProgramControlLease(*control, leases[index]);
        return stageProgramControl(invocation, controlSlot, token, std::move(control));
    } catch (const std::exception &exception) {
        return fail(context, exception.what());
    }
}

VernonStatus vernonRuntimeProgramInvocationBindDrawCommand(VernonProgramInvocation *invocation, uint32_t controlSlot,
                                                           const VernonProgramBindingToken *token,
                                                           const VernonDrawCommand *draw,
                                                           const VernonProgramResourceLease *lease) {
    VernonRuntimeContext *context = invocation && invocation->instance && invocation->instance->pipeline
                                        ? invocation->instance->pipeline->context
                                        : nullptr;
    try {
        RuntimeDiagnosticScope diagnostic(context);
        if (!draw || draw->struct_size < sizeof(*draw) || !draw->instance_count ||
            (draw->index_binding && draw->vertex_count))
            return fail(context, "invalid persistent Program DrawCommand control");
        auto control = std::make_shared<RuntimeProgramControl>();
        control->kind = RuntimeProgramControl::DrawCommand;
        control->draw = *draw;
        if (draw->index_binding)
            control->index = *draw->index_binding;
        if (lease)
            retainProgramControlLease(*control, *lease);
        return stageProgramControl(invocation, controlSlot, token, std::move(control));
    } catch (const std::exception &exception) {
        return fail(context, exception.what());
    }
}

VernonStatus vernonRuntimeProgramInvocationBindDynamicState(VernonProgramInvocation *invocation, uint32_t controlSlot,
                                                            const VernonProgramBindingToken *token,
                                                            const VernonDynamicState *dynamicState) {
    VernonRuntimeContext *context = invocation && invocation->instance && invocation->instance->pipeline
                                        ? invocation->instance->pipeline->context
                                        : nullptr;
    try {
        RuntimeDiagnosticScope diagnostic(context);
        if (!dynamicState || dynamicState->struct_size < sizeof(*dynamicState) ||
            dynamicState->stencil_reference > 0xff)
            return fail(context, "invalid persistent Program DynamicState control");
        auto control = std::make_shared<RuntimeProgramControl>();
        control->kind = RuntimeProgramControl::DynamicState;
        control->dynamic = *dynamicState;
        return stageProgramControl(invocation, controlSlot, token, std::move(control));
    } catch (const std::exception &exception) {
        return fail(context, exception.what());
    }
}

VernonStatus vernonRuntimeProgramInvocationForward(VernonProgramInvocation *invocation,
                                                   VernonPullback **outputPullback) {
    VernonProgramExecutable *pipeline = invocation && invocation->instance ? invocation->instance->pipeline : nullptr;
    try {
        RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
        if (outputPullback)
            *outputPullback = nullptr;
        if (!invocation || invocation->finished || !pipeline)
            return fail(pipeline ? pipeline->context : nullptr, "invalid persistent Program invocation");
        invocation->snapshot = invocation->transaction->snapshot();
        invocation->controlSnapshot = invocation->controlTransaction->snapshot();
        std::vector<VernonProgramArgument> arguments;
        const size_t count = vernonRuntimeProgramExecutableGetParameterCount(pipeline);
        arguments.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            VernonProgramParameterView parameter{};
            if (vernonRuntimeProgramExecutableGetParameterByIndex(pipeline, index, &parameter) != VERNON_STATUS_OK)
                return fail(pipeline->context, "cannot reflect persistent Program boundary slot");
            const std::shared_ptr<void> *payload = invocation->snapshot->find(parameter.slot);
            if (!payload)
                return fail(pipeline->context, "persistent Program invocation has an unbound boundary slot");
            auto binding = std::static_pointer_cast<RuntimeProgramBinding>(*payload);
            binding->refresh();
            arguments.push_back(binding->argument);
        }
        VernonStageInvocationDescriptor frame{};
        frame.struct_size = sizeof(frame);
        frame.abi_version = VERNON_PROGRAM_VERSION;
        frame.arguments = arguments.data();
        frame.argument_count = arguments.size();
        const ProgramInvocationContext programContext{*invocation->controlSnapshot};
        VernonPullback *pullback = nullptr;
        const VernonStatus status =
            vernon::runtime::program_execution::forwardProgramInvocation(*pipeline, frame, pullback, &programContext);
        if (status != VERNON_STATUS_OK) {
            invocation->transaction->rollback();
            invocation->controlTransaction->rollback();
            invocation->finished = true;
            return status;
        }
        invocation->snapshot = invocation->transaction->commit();
        invocation->controlSnapshot = invocation->controlTransaction->commit();
        invocation->finished = true;
        if (pullback)
            vernon::runtime::program_execution::attachProgramSnapshot(*pullback, invocation->snapshot);
        if (outputPullback)
            *outputPullback = pullback;
        else if (pullback)
            vernonPullbackDestroy(pullback);
        return VERNON_STATUS_OK;
    } catch (const std::exception &exception) {
        if (invocation && !invocation->finished) {
            invocation->transaction->rollback();
            invocation->controlTransaction->rollback();
            invocation->finished = true;
        }
        return fail(pipeline ? pipeline->context : nullptr, exception.what(), VERNON_STATUS_INTERNAL_ERROR);
    }
}

void vernonRuntimeProgramInvocationRollback(VernonProgramInvocation *invocation) {
    if (!invocation || invocation->finished)
        return;
    invocation->transaction->rollback();
    invocation->controlTransaction->rollback();
    invocation->finished = true;
}

void vernonRuntimeProgramInvocationDestroy(VernonProgramInvocation *invocation) { delete invocation; }

VernonStatus vernonRuntimeProgramInstanceGetTelemetry(const VernonProgramInstance *instance,
                                                      VernonProgramBindingTelemetry *output) {
    if (!instance || !instance->state || !output || output->struct_size < sizeof(*output))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const vernon::runtime::program::BindingTelemetry telemetry = instance->state->bindings.telemetry();
    *output = {sizeof(*output),         telemetry.prepareCount, telemetry.reuseCount,
               telemetry.rollbackCount, telemetry.uploadBytes,  telemetry.uploadRanges};
    return VERNON_STATUS_OK;
}

VernonStatus vernonSubmissionGetState(const VernonSubmission *submission, VernonSubmissionState *output) {
    if (!submission || !output)
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (submission->completion.index == VERNON_RHI_INVALID_HANDLE_INDEX) {
        *output = submission->state;
        return VERNON_STATUS_OK;
    }
    VernonRhiCompletionState state{};
    if (vernonRhiCompletionGetState(submission->device, submission->completion, &state) != VERNON_RHI_STATUS_OK)
        return VERNON_STATUS_INVALID_ARGUMENT;
    *output = state == VERNON_RHI_COMPLETION_PENDING  ? VERNON_SUBMISSION_PENDING
              : state == VERNON_RHI_COMPLETION_FAILED ? VERNON_SUBMISSION_FAILED
                                                      : VERNON_SUBMISSION_SUCCEEDED;
    return VERNON_STATUS_OK;
}

VernonStatus vernonSubmissionWait(VernonSubmission *submission) {
    if (!submission)
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (submission->completion.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        return submission->status;
    const VernonRhiStatus status = vernonRhiCompletionWait(submission->device, submission->completion);
    submission->status = status == VERNON_RHI_STATUS_OK ? VERNON_STATUS_OK : VERNON_STATUS_INTERNAL_ERROR;
    submission->state = status == VERNON_RHI_STATUS_OK ? VERNON_SUBMISSION_SUCCEEDED : VERNON_SUBMISSION_FAILED;
    return submission->status;
}

void vernonSubmissionDestroy(VernonSubmission *submission) {
    if (!submission)
        return;
    if (submission->completion.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        (void)vernonRhiDeviceDestroyCompletion(submission->device, submission->completion);
    delete submission;
}

VernonStatus vernonRuntimeReferenceRhiBuffer(VernonRuntimeContext *context, VernonRhiBuffer buffer, uint64_t offset,
                                             uint64_t size, VernonRuntimeProviderResourceReference *output) {
    if (!context)
        return VERNON_STATUS_INVALID_ARGUMENT;
    RuntimeDiagnosticScope diagnostic(context);
    if (!output)
        return fail(context, "invalid RHI buffer reference");
    return referenceBackendRhiBuffer(*context, buffer, offset, size, *output);
}

VernonStatus vernonRuntimeReferenceRhiImageView(VernonRuntimeContext *context, VernonRhiImageView view,
                                                VernonRuntimeProviderResourceReference *output) {
    if (!context)
        return VERNON_STATUS_INVALID_ARGUMENT;
    RuntimeDiagnosticScope diagnostic(context);
    if (!output)
        return fail(context, "invalid RHI image view reference");
    return referenceBackendRhiImageView(*context, view, *output);
}

VernonStatus vernonRuntimeReferenceRhiSampler(VernonRuntimeContext *context, VernonRhiSampler sampler,
                                              VernonRuntimeProviderResourceReference *output) {
    if (!context)
        return VERNON_STATUS_INVALID_ARGUMENT;
    RuntimeDiagnosticScope diagnostic(context);
    if (!output)
        return fail(context, "invalid RHI sampler reference");
    return referenceBackendRhiSampler(*context, sampler, *output);
}

VernonStatus vernonRuntimeReferenceRhiCommandEncoder(VernonRuntimeContext *context, VernonRhiCommandEncoder encoder,
                                                     VernonRuntimeProviderObject *output) {
    if (!context)
        return VERNON_STATUS_INVALID_ARGUMENT;
    RuntimeDiagnosticScope diagnostic(context);
    if (!output)
        return fail(context, "invalid RHI command encoder reference");
    return referenceBackendCommandEncoder(*context, encoder, *output);
}

} // extern "C"

VernonStatus vernon::runtime::executePipelineProgramGraph(
    VernonProgramExecutable &pipeline, const program::Graph &graph, program_execution::ProgramInvocationState &arena,
    const program_execution::ResolvePhysicalEndpoint &resolvePhysicalEndpoint) {
    return executePipelineProgramGraphImpl(pipeline, graph, arena, resolvePhysicalEndpoint);
}

VernonStageExecutable::~VernonStageExecutable() {
    if (backendState)
        vernon::runtime::destroyBackendPipeline(*this);
}
