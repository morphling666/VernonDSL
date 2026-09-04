#include "VernonRuntime.h"
#include "VernonExecutionGraph.h"
#include "execution_graph/execution_graph_internal.h"
#include "rhi/rhi_internal.h"
#include "runtime/autodiff/program_value_arena.h"
#include "runtime/autodiff/runtime_autodiff_internal.h"
#include "runtime/autodiff/runtime_gpu_commands.h"
#include "runtime/autodiff/tape_allocator_abi.h"
#include "runtime/backend_cpu.h"
#include "runtime/compute_launch_planner.h"
#include "runtime/graphics_invocation_planner.h"
#include "runtime/graphics_scope_planner.h"
#include "runtime/pipeline_bundle.h"
#include "runtime/pipeline_manifest.h"
#include "runtime/pipeline_metadata.h"
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
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

struct RuntimeProgramBinding {
    VernonPipelineArgument argument{};
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;
    std::string layoutHash;
    std::vector<VernonValueLeafView> leaves;
    std::shared_ptr<void> lease;

    void refresh() {
        if (argument.kind != VERNON_PIPELINE_TENSOR)
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
    explicit RuntimeProgramInstanceState(VernonLoadedPipeline &value)
        : pipeline(&value), bindings(&value), controls(&value) {}
    VernonLoadedPipeline *pipeline;
    vernon::runtime::program::ProgramInstance bindings;
    vernon::runtime::program::ProgramInstance controls;
};

struct VernonProgramInstance {
    explicit VernonProgramInstance(VernonLoadedPipeline &value)
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

const program::Program *managedProgram(const VernonLoadedPipeline &pipeline) {
    return pipeline.topology && pipeline.topology->resolvedProgram ? &pipeline.topology->resolvedProgram->program
                                                                   : nullptr;
}

bool publicBoundarySlot(const program::BoundarySlot &slot) {
    return slot.role == program::BoundaryRole::Input || slot.role == program::BoundaryRole::Output;
}

size_t publicBoundaryCount(const program::Program &program) {
    return static_cast<size_t>(
        std::count_if(program.abi.boundarySlots.begin(), program.abi.boundarySlots.end(), publicBoundarySlot));
}

const program::BoundarySlot *publicBoundaryAt(const program::Program &program, size_t publicIndex) {
    for (const program::BoundarySlot &slot : program.abi.boundarySlots)
        if (publicBoundarySlot(slot) && publicIndex-- == 0)
            return &slot;
    return nullptr;
}

const program::BoundarySlot *findPublicBoundary(const program::Program &program, VernonStringView name) {
    const auto found = std::find_if(
        program.abi.boundarySlots.begin(), program.abi.boundarySlots.end(), [&](const program::BoundarySlot &slot) {
            return publicBoundarySlot(slot) && name.size == slot.path.size() &&
                   (name.size == 0 || std::equal(name.data, name.data + name.size, slot.path.data()));
        });
    return found == program.abi.boundarySlots.end() ? nullptr : &*found;
}

const ValueLayout *boundaryLayoutView(const VernonLoadedPipeline &pipeline, const program::BoundarySlot &slot) {
    const program::Program *program = managedProgram(pipeline);
    if (!program)
        return nullptr;
    const auto slotIndex = static_cast<size_t>(&slot - program->abi.boundarySlots.data());
    return slotIndex < pipeline.topology->boundaryLayoutViews.size()
               ? &pipeline.topology->boundaryLayoutViews[slotIndex]
               : nullptr;
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

VernonLoadedPipeline *vernonRuntimeLoadCpuEntry(VernonRuntimeContext *context, VernonCpuEntryPoint entryPoint,
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

VernonLoadedPipeline *vernonRuntimeLoadArtifact(VernonRuntimeContext *context, const void *artifact,
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

VernonStatus vernonRuntimeExecutableBundleInspectKind(const void *bundleData, size_t bundleSize,
                                                      VernonExecutableBundleKind *kind) {
    if (!bundleData || !bundleSize || !kind)
        return VERNON_STATUS_INVALID_ARGUMENT;
    try {
        const nlohmann::json root = nlohmann::json::parse(
            static_cast<const char *>(bundleData), static_cast<const char *>(bundleData) + bundleSize, nullptr, false);
        if (root.is_discarded() || !root.is_object())
            return VERNON_STATUS_PARSE_ERROR;
        const std::string type = root.value("type", "");
        if (type == "pipeline" && root.value("pipeline_version", 0) == VERNON_PIPELINE_VERSION)
            *kind = VERNON_EXECUTABLE_BUNDLE_PIPELINE;
        else if (type == "program_bundle" && root.value("pipeline_version", 0) == VERNON_PIPELINE_VERSION)
            *kind = VERNON_EXECUTABLE_BUNDLE_PROGRAM;
        else
            return VERNON_STATUS_PARSE_ERROR;
        return VERNON_STATUS_OK;
    } catch (...) {
        return VERNON_STATUS_PARSE_ERROR;
    }
}

VernonStatus vernonRuntimePipelineBundleInspectTarget(const void *bundleData, size_t bundleSize,
                                                      VernonRuntimeBackend *target) {
    if (!bundleData || !bundleSize || !target)
        return VERNON_STATUS_INVALID_ARGUMENT;
    try {
        const nlohmann::json root = nlohmann::json::parse(
            static_cast<const char *>(bundleData), static_cast<const char *>(bundleData) + bundleSize, nullptr, false);
        if (root.is_discarded() || !root.is_object())
            return VERNON_STATUS_PARSE_ERROR;
        const bool pipelineSchema =
            root.value("pipeline_version", 0) == VERNON_PIPELINE_VERSION && root.value("type", "") == "pipeline";
        std::string manifestError;
        if (!pipelineSchema || !validatePipelineRootSchema(root, manifestError) ||
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

VernonPipelineBundle *vernonRuntimeLoadPipelineBundleWithOptions(VernonRuntimeContext *context, const void *bundleData,
                                                                 size_t bundleSize,
                                                                 const VernonPipelineBundleLoadOptions *options) {
    if (!context)
        return nullptr;
    RuntimeDiagnosticScope diagnostic(context);
    if ((context->backend != VERNON_RUNTIME_CPU && context->backend != VERNON_RUNTIME_OPENGL &&
         context->backend != VERNON_RUNTIME_OPENGL_ES && context->backend != VERNON_RUNTIME_VULKAN &&
         context->backend != VERNON_RUNTIME_CUDA && context->backend != VERNON_RUNTIME_DIRECTX12 &&
         context->backend != VERNON_RUNTIME_METAL) ||
        !bundleData || !bundleSize || (options && options->struct_size < sizeof(VernonPipelineBundleLoadOptions))) {
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
        const bool pipelineSchema = root.is_object() && root.value("pipeline_version", 0) == VERNON_PIPELINE_VERSION &&
                                    root.value("type", "") == "pipeline";
        if (!pipelineSchema || !validatePipelineRootSchema(root, invocationDiagnostic(*context)) ||
            pipelineTargetKind(root) != expectedTarget || !root.contains("stage_artifacts") ||
            !root["stage_artifacts"].is_object() || !root.contains("variants") || !root["variants"].is_array()) {
            fail(context, "unsupported or invalid pipeline bundle");
            return nullptr;
        }
        if (!validateManifestHash(root, true, invocationDiagnostic(*context)))
            return nullptr;
        RuntimeRequirements requirements;
        if (!parseRuntimeRequirements(root, expectedTarget, requirements, invocationDiagnostic(*context)) ||
            !validateRuntimeRequirements(*context, requirements))
            return nullptr;
        auto bundle = std::make_unique<VernonPipelineBundle>();
        bundle->context = context;
        bundle->id = root.value("id", "");
        AutodiffManifest autodiff;
        if (!parseAutodiffManifest(root, autodiff, invocationDiagnostic(*context)))
            return nullptr;
        if (root.contains("autodiff"))
            bundle->autodiff = std::move(autodiff);
        for (const auto &[id, value] : root["stage_artifacts"].items()) {
            const auto validStageKeys = [&](const nlohmann::json &stage) {
                for (const auto &[key, unused] : stage.items())
                    if (key != "stage" && key != "entry" && key != "artifact" && key != "reflection" &&
                        (context->backend != VERNON_RUNTIME_CPU || key != "symbol"))
                        return false;
                return true;
            };
            if (!value.is_object() || !validStageKeys(value) || !value.contains("stage") ||
                !value["stage"].is_string() || !value.contains("entry") || !value["entry"].is_string() ||
                !value.contains("artifact") || !value["artifact"].is_object() || !value.contains("reflection") ||
                !value["reflection"].is_object() ||
                (context->backend == VERNON_RUNTIME_CPU && (!value.contains("symbol") || !value["symbol"].is_string() ||
                                                            value["symbol"].get_ref<const std::string &>().empty()))) {
                fail(context, "pipeline stage id is invalid");
                return nullptr;
            }
            Stage stage;
            stage.stage = value.value("stage", "");
            stage.entry = value.value("entry", "");
            if (value["reflection"].contains("entries")) {
                stage.reflection = value["reflection"].dump();
                const auto implementation = value["reflection"].find("implementation");
                if (implementation == value["reflection"].end() || !implementation->is_object() ||
                    implementation->size() != 2 || !implementation->contains("target") ||
                    !(*implementation)["target"].is_string() || (*implementation)["target"] != expectedTarget ||
                    !implementation->contains("metadata")) {
                    fail(context, "pipeline stage has invalid target implementation");
                    return nullptr;
                }
                if (!parseTargetImplementationMetadata(expectedTarget, (*implementation)["metadata"], stage.nativeSlots,
                                                       invocationDiagnostic(*context)))
                    return nullptr;
                for (const nlohmann::json &entry : value["reflection"]["entries"]) {
                    if (entry.value("name", "") != stage.entry)
                        continue;
                    const auto workgroup = entry.value("workgroup_size", nlohmann::json::array());
                    if (workgroup.size() == 3)
                        for (size_t index = 0; index < 3; ++index) {
                            stage.workgroup[index] = workgroup[index].get<uint32_t>();
                            if (!stage.workgroup[index]) {
                                fail(context, "compute workgroup dimensions must be non-zero");
                                return nullptr;
                            }
                        }
                    if (stage.stage == "compute" &&
                        !parseDispatchContract(entry, stage.dispatchContract, invocationDiagnostic(*context)))
                        return nullptr;
                }
                if (stage.stage == "compute") {
                    ReflectedEntry reflected;
                    if (!parseReflection(value["reflection"], stage.entry, reflected, context->backend,
                                         invocationDiagnostic(*context)))
                        return nullptr;
                    std::copy_n(reflected.workgroup, 3, stage.workgroup);
                    stage.dispatchContract = reflected.dispatchContract;
                    stage.readFootprints = std::move(reflected.readFootprints);
                    stage.writeFootprints = std::move(reflected.writeFootprints);
                }
            }
            if (!value.contains("artifact")) {
                fail(context, "pipeline stage artifact descriptor is missing");
                return nullptr;
            }
            ResolvedArtifact resolved;
            const ArtifactResolution resolution = artifactResolutionFor(*context);
            if (!resolveArtifact(value["artifact"], bundleDirectory, resolved, invocationDiagnostic(*context),
                                 resolution))
                return nullptr;
            const char *expectedFormat = context->backend == VERNON_RUNTIME_CUDA        ? "ptx"
                                         : context->backend == VERNON_RUNTIME_VULKAN    ? "spirv"
                                         : context->backend == VERNON_RUNTIME_DIRECTX12 ? "dxil"
                                         : context->backend == VERNON_RUNTIME_METAL     ? "msl"
                                         : context->backend == VERNON_RUNTIME_OPENGL_ES ? "gles"
                                                                                        : "glsl";
            const std::string encoding = value["artifact"].value("encoding", "");
            const bool validCpuFormat =
                context->backend == VERNON_RUNTIME_CPU &&
                (resolved.format == "native_library" || resolved.format == "relocatable_object");
            const bool validFormat =
                context->backend == VERNON_RUNTIME_CPU ? validCpuFormat : resolved.format == expectedFormat;
            if (!validFormat || (resolved.external && value["artifact"].contains("encoding")) ||
                (!resolved.external &&
                 (((resolved.format == "spirv" || resolved.format == "dxil") && encoding != "base64") ||
                  (resolved.format != "spirv" && resolved.format != "dxil" && encoding != "utf8")))) {
                fail(context, "pipeline stage artifact format is invalid for target");
                return nullptr;
            }
            if (context->backend == VERNON_RUNTIME_CPU) {
                if (!resolved.external || (resolved.format == "native_library" && !bundleDirectory)) {
                    fail(context, "CPU pipeline artifacts require an external bundle directory");
                    return nullptr;
                }
                const nlohmann::json &nativeArtifact = value["artifact"];
                CpuNativeArtifact artifact;
                if (bundleDirectory)
                    artifact.root = *bundleDirectory;
                artifact.relativeLibrary = std::filesystem::u8path(nativeArtifact.value("path", ""));
                artifact.entry = stage.entry;
                artifact.format = resolved.format;
                artifact.symbol = value.value("symbol", "");
                artifact.targetTriple = requirements.targetTriple;
                artifact.objectFormat = requirements.objectFormat;
                artifact.size = nativeArtifact.value("size", uint64_t{0});
                artifact.sha256 = nativeArtifact.value("sha256", "");
                artifact.reflection = value["reflection"];
                artifact.staticallyLinked = resolved.format == "relocatable_object";
                std::filesystem::path validatedPath;
                const std::string format = resolved.format;
                if (format != "native_library" && format != "relocatable_object")
                    return nullptr;
                // Native libraries are runtime artifacts and are authenticated
                // here. Relocatable objects have already been linked by the
                // host application, so only their metadata is validated.
                if (format == "native_library" &&
                    !resolveCpuNativeArtifact(artifact, validatedPath, nullptr, invocationDiagnostic(*context)))
                    return nullptr;
                stage.cpuArtifact = std::move(artifact);
            }
            if (context->backend == VERNON_RUNTIME_VULKAN || context->backend == VERNON_RUNTIME_DIRECTX12)
                stage.binary = std::move(resolved.bytes);
            else if (context->backend != VERNON_RUNTIME_CPU)
                stage.source.assign(resolved.bytes.begin(), resolved.bytes.end());
            const bool hasArtifact =
                context->backend == VERNON_RUNTIME_CPU ? stage.cpuArtifact.has_value()
                : (context->backend == VERNON_RUNTIME_VULKAN || context->backend == VERNON_RUNTIME_DIRECTX12)
                    ? !stage.binary.empty() && stage.binary.size() % sizeof(uint32_t) == 0 && !stage.reflection.empty()
                : (context->backend == VERNON_RUNTIME_CUDA || context->backend == VERNON_RUNTIME_METAL)
                    ? !stage.source.empty() && !stage.reflection.empty()
                    : !stage.source.empty();
            if (stage.stage.empty() || stage.entry.empty() || !hasArtifact) {
                fail(context, "pipeline stage artifact is invalid");
                return nullptr;
            }
            bundle->stages.emplace(id, std::move(stage));
        }
        for (const nlohmann::json &value : root["variants"]) {
            Variant variant;
            if (!parseVariant(value, variant, invocationDiagnostic(*context)))
                return nullptr;
            auto validStage = [&](const std::string &id, const std::string &kind) {
                if (id.empty())
                    return true;
                const auto found = bundle->stages.find(id);
                return found != bundle->stages.end() && found->second.stage == kind;
            };
            for (const auto &[stage, id] : variant.program) {
                if (!validStage(id, stage)) {
                    fail(context, "pipeline variant references an invalid program stage");
                    return nullptr;
                }
            }
            const bool computeProgram = variant.program.size() == 1 && !variant.compute.empty();
            const bool graphicsProgram =
                !variant.vertex.empty() && !variant.fragment.empty() && variant.program.size() == 2;
            if (!computeProgram && !graphicsProgram) {
                fail(context, "pipeline program stage topology is not supported by this runtime",
                     VERNON_STATUS_UNSUPPORTED_TARGET);
                return nullptr;
            }
            if ((context->backend == VERNON_RUNTIME_CPU || context->backend == VERNON_RUNTIME_CUDA) &&
                !computeProgram) {
                fail(context, std::string(expectedTarget) + " pipeline bundles support compute programs only",
                     VERNON_STATUS_UNSUPPORTED_TARGET);
                return nullptr;
            }
            bundle->variants.push_back(std::move(variant));
        }
        std::sort(bundle->variants.begin(), bundle->variants.end(),
                  [](const Variant &left, const Variant &right) { return left.key < right.key; });
        if (std::adjacent_find(bundle->variants.begin(), bundle->variants.end(),
                               [](const Variant &left, const Variant &right) { return left.key == right.key; }) !=
            bundle->variants.end()) {
            fail(context, "pipeline bundle contains duplicate feature variants");
            return nullptr;
        }
        if (bundle->autodiff) {
            if (bundle->autodiff->profiles.size() != bundle->variants.size()) {
                fail(context, "autodiff profiles do not cover every pipeline variant");
                return nullptr;
            }
            for (size_t index = 0; index < bundle->variants.size(); ++index) {
                const Variant &variant = bundle->variants[index];
                const AutodiffProfile &profiles = bundle->autodiff->profiles[index];
                auto profileStage = [&](const std::string &id) {
                    const auto found = bundle->stages.find(id);
                    return found != bundle->stages.end() && found->second.stage == "compute";
                };
                if (profiles.key != variant.key || profiles.primal != variant.compute ||
                    profiles.primal == profiles.forwardWithTape || profiles.primal == profiles.backward ||
                    profiles.forwardWithTape == profiles.backward || !profileStage(profiles.forwardWithTape) ||
                    !profileStage(profiles.backward)) {
                    fail(context, "autodiff profile references are inconsistent with pipeline variants");
                    return nullptr;
                }
            }
        }
        if (bundle->id.empty()) {
            fail(context, "pipeline bundle id is missing or empty");
            return nullptr;
        }
        if (bundle->variants.empty()) {
            fail(context, "pipeline bundle contains no variants");
            return nullptr;
        }
        ++context->liveBundles;
        return bundle.release();
    } catch (const nlohmann::json::exception &error) {
        fail(context, std::string("invalid pipeline bundle: ") + error.what());
        return nullptr;
    } catch (const std::exception &error) {
        fail(context, std::string("failed to load pipeline bundle: ") + error.what(), VERNON_STATUS_INTERNAL_ERROR);
        return nullptr;
    } catch (...) {
        fail(context, "failed to load pipeline bundle", VERNON_STATUS_INTERNAL_ERROR);
        return nullptr;
    }
}

VernonLoadedPipeline *vernonRuntimeLoadProgramBundleWithOptions(VernonRuntimeContext *context, const void *bundleData,
                                                                size_t bundleSize, VernonFeatureSetView features,
                                                                const VernonPipelineBundleLoadOptions *options) {
    if (!context)
        return nullptr;
    RuntimeDiagnosticScope diagnostic(context);
    if (!bundleData || !bundleSize || (features.count && !features.names) ||
        (options && options->struct_size < sizeof(VernonPipelineBundleLoadOptions))) {
        fail(context, "invalid Program bundle load invocation");
        return nullptr;
    }
    try {
        const auto *bytes = static_cast<const uint8_t *>(bundleData);
        const nlohmann::json document = nlohmann::json::parse(bytes, bytes + bundleSize);
        if (!document.is_object() || document.value("type", std::string()) != "program_bundle" ||
            !document.contains("variants") || !document["variants"].is_array()) {
            fail(context, "cooked Module asset is not a Program bundle", VERNON_STATUS_PARSE_ERROR);
            return nullptr;
        }
        std::vector<std::string> requested;
        requested.reserve(features.count);
        for (size_t index = 0; index < features.count; ++index) {
            if (!features.names[index] || !*features.names[index]) {
                fail(context, "Program bundle feature names must not be empty");
                return nullptr;
            }
            requested.emplace_back(features.names[index]);
        }
        std::sort(requested.begin(), requested.end());
        requested.erase(std::unique(requested.begin(), requested.end()), requested.end());
        const nlohmann::json *selected = nullptr;
        for (const nlohmann::json &variant : document["variants"]) {
            if (!variant.is_object() || !variant.contains("key") || !variant["key"].is_array())
                continue;
            std::vector<std::string> key;
            bool valid = true;
            for (const nlohmann::json &feature : variant["key"]) {
                if (!feature.is_string()) {
                    valid = false;
                    break;
                }
                key.push_back(feature.get<std::string>());
            }
            std::sort(key.begin(), key.end());
            key.erase(std::unique(key.begin(), key.end()), key.end());
            if (valid && key == requested) {
                if (selected) {
                    fail(context, "Program bundle contains duplicate feature variants", VERNON_STATUS_PARSE_ERROR);
                    return nullptr;
                }
                selected = &variant;
            }
        }
        if (!selected || !selected->contains("program") || !(*selected)["program"].is_object() ||
            !selected->contains("artifact_system") || !(*selected)["artifact_system"].is_object() ||
            !selected->contains("stage_bindings") || !(*selected)["stage_bindings"].is_object()) {
            fail(context, selected ? "Program bundle variant is incomplete" : "Program bundle has no matching variant",
                 VERNON_STATUS_PARSE_ERROR);
            return nullptr;
        }
        std::map<std::string, std::string> stageBindings;
        for (const auto &[request, artifact] : (*selected)["stage_bindings"].items()) {
            if (!artifact.is_string()) {
                fail(context, "Program bundle stage binding is not a string", VERNON_STATUS_PARSE_ERROR);
                return nullptr;
            }
            stageBindings.emplace(request, artifact.get<std::string>());
        }
        std::filesystem::path bundleRoot;
        if (options && options->bundle_directory)
            bundleRoot = options->bundle_directory;
        const std::string programJson = (*selected)["program"].dump();
        const std::string artifactSystemJson = (*selected)["artifact_system"].dump();
        std::string error;
        VernonLoadedPipeline *pipeline = program::loadBackendProgramPipeline(
            *context, programJson.data(), programJson.size(), artifactSystemJson.data(), artifactSystemJson.size(),
            stageBindings, bundleRoot, error);
        if (!pipeline)
            fail(context, error.empty() ? "cannot load Program bundle" : error, VERNON_STATUS_PARSE_ERROR);
        return pipeline;
    } catch (const std::exception &exception) {
        fail(context, std::string("failed to load Program bundle: ") + exception.what(), VERNON_STATUS_PARSE_ERROR);
        return nullptr;
    }
}

VernonStringView vernonRuntimePipelineBundleGetId(const VernonPipelineBundle *bundle) {
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

bool fillParameterView(const Parameter &source, VernonPipelineParameterView &destination) {
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

bool fillBoundaryParameterView(const VernonLoadedPipeline &pipeline, const program::BoundarySlot &source,
                               VernonPipelineParameterView &destination) {
    VernonPipelineArgumentKind kind = VERNON_PIPELINE_TENSOR;
    if (source.category == program::BoundaryCategory::Texture)
        kind = VERNON_PIPELINE_IMAGE;
    else if (source.category == program::BoundaryCategory::Sampler)
        kind = VERNON_PIPELINE_SAMPLER;
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

VernonStatus fillImageConstraintView(const Parameter &source, VernonPipelineImageConstraintView &destination) {
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
                                             VernonPipelineImageConstraintView &destination) {
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

bool fillOutputView(const Output &source, VernonPipelineOutputView &destination) {
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

void destroyPipelineImplementations(VernonLoadedPipeline &pipeline) {
    pipeline.topology.reset();
    if (pipeline.backendState)
        destroyBackendPipeline(pipeline);
}

} // namespace

void vernonRuntimePipelineBundleDestroy(VernonPipelineBundle *bundle) {
    if (!bundle)
        return;
    RuntimeDiagnosticScope diagnostic(bundle->context);
    --bundle->context->liveBundles;
    delete bundle;
}

VernonLoadedPipeline *vernonRuntimeResolvePipeline(VernonPipelineBundle *bundle, VernonFeatureSetView features) {
    if (!bundle)
        return nullptr;
    try {
        RuntimeDiagnosticScope diagnostic(bundle->context);
        if (features.count && !features.names) {
            invocationDiagnostic(*bundle->context) = "invalid pipeline feature set";
            return nullptr;
        }
        std::vector<std::string> key;
        for (size_t index = 0; index < features.count; ++index) {
            if (!features.names[index]) {
                invocationDiagnostic(*bundle->context) = "invalid pipeline feature set";
                return nullptr;
            }
            key.emplace_back(features.names[index]);
        }
        std::sort(key.begin(), key.end());
        const auto found = std::find_if(bundle->variants.begin(), bundle->variants.end(),
                                        [&](const Variant &variant) { return variant.key == key; });
        if (found == bundle->variants.end()) {
            fail(bundle->context, "pipeline bundle has no exact feature variant");
            return nullptr;
        }
        auto pipeline = std::make_unique<VernonLoadedPipeline>();
        pipeline->context = bundle->context;
        pipeline->variant = *found;
        if (!found->compute.empty()) {
            const Stage &stage = bundle->stages.at(found->compute);
            pipeline->workgroupSize = {stage.workgroup[0], stage.workgroup[1], stage.workgroup[2]};
            pipeline->dispatchContract = stage.dispatchContract;
            pipeline->readFootprints = stage.readFootprints;
            pipeline->writeFootprints = stage.writeFootprints;
        }
        rebuildVariantLayoutViews(pipeline->variant);
        const bool nativeProgramAutodiff = pipeline->topology && pipeline->topology->resolvedProgram &&
                                           program::findGraph(pipeline->topology->resolvedProgram->program, "backward");
        if (nativeProgramAutodiff) {
            const std::vector<AutodiffDerivativeGroup> derivativeGroups =
                bundle->autodiff ? bundle->autodiff->derivativeGroups : std::vector<AutodiffDerivativeGroup>{};
            if (!vernon::runtime::ad::resolveProgramAutodiff(*pipeline, derivativeGroups)) {
                const std::string message = invocationDiagnostic(*bundle->context);
                destroyPipelineImplementations(*pipeline);
                fail(bundle->context, message);
                return nullptr;
            }
        } else if (bundle->autodiff) {
            const auto profiles = std::find_if(bundle->autodiff->profiles.begin(), bundle->autodiff->profiles.end(),
                                               [&](const AutodiffProfile &candidate) { return candidate.key == key; });
            if (profiles == bundle->autodiff->profiles.end()) {
                fail(bundle->context, "autodiff profiles have no exact feature variant");
                destroyPipelineImplementations(*pipeline);
                return nullptr;
            }
            if (!vernon::runtime::ad::resolvePipelineAutodiff(*bundle, *profiles, *pipeline)) {
                const std::string message = invocationDiagnostic(*bundle->context);
                destroyPipelineImplementations(*pipeline);
                fail(bundle->context, message);
                return nullptr;
            }
        }
        if (!pipeline->topology && !resolveBackendPipeline(*bundle, *found, *pipeline)) {
            destroyPipelineImplementations(*pipeline);
            return nullptr;
        }
        ++bundle->context->livePipelines;
        return pipeline.release();
    } catch (const std::bad_alloc &) {
        fail(bundle->context, "cannot allocate resolved pipeline", VERNON_STATUS_INTERNAL_ERROR);
    } catch (const std::length_error &) {
        fail(bundle->context, "resolved pipeline allocation is too large", VERNON_STATUS_INTERNAL_ERROR);
    } catch (...) {
        fail(bundle->context, "unexpected pipeline resolution failure", VERNON_STATUS_INTERNAL_ERROR);
    }
    return nullptr;
}

size_t vernonRuntimeLoadedPipelineGetParameterCount(const VernonLoadedPipeline *pipeline) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline)
        return 0;
    if (const program::Program *program = managedProgram(*pipeline))
        return publicBoundaryCount(*program);
    return pipeline->variant.parameters.size();
}

VernonStatus vernonRuntimeLoadedPipelineGetParameterByIndex(const VernonLoadedPipeline *pipeline, size_t index,
                                                            VernonPipelineParameterView *parameter) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !parameter)
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (const program::Program *program = managedProgram(*pipeline)) {
        const program::BoundarySlot *slot = publicBoundaryAt(*program, index);
        if (!slot)
            return VERNON_STATUS_INVALID_ARGUMENT;
        return fillBoundaryParameterView(*pipeline, *slot, *parameter) ? VERNON_STATUS_OK : VERNON_STATUS_PARSE_ERROR;
    }
    if (index >= pipeline->variant.parameters.size())
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillParameterView(pipeline->variant.parameters[index], *parameter) ? VERNON_STATUS_OK
                                                                              : VERNON_STATUS_PARSE_ERROR;
}

VernonStatus vernonRuntimeLoadedPipelineFindParameter(const VernonLoadedPipeline *pipeline, VernonStringView name,
                                                      VernonPipelineParameterView *parameter) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !parameter || (name.size && !name.data))
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (const program::Program *program = managedProgram(*pipeline)) {
        const program::BoundarySlot *slot = findPublicBoundary(*program, name);
        if (!slot)
            return VERNON_STATUS_INVALID_ARGUMENT;
        return fillBoundaryParameterView(*pipeline, *slot, *parameter) ? VERNON_STATUS_OK : VERNON_STATUS_PARSE_ERROR;
    }
    const auto found = std::find_if(pipeline->variant.parameters.begin(), pipeline->variant.parameters.end(),
                                    [&](const Parameter &candidate) { return stringViewEquals(name, candidate.name); });
    if (found == pipeline->variant.parameters.end())
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillParameterView(*found, *parameter) ? VERNON_STATUS_OK : VERNON_STATUS_PARSE_ERROR;
}

VernonStatus vernonRuntimeLoadedPipelineGetParameterValueLeaf(const VernonLoadedPipeline *pipeline,
                                                              VernonStringView parameterName, size_t leafIndex,
                                                              VernonPipelineValueLeafView *leaf) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !leaf || leaf->struct_size < sizeof(VernonPipelineValueLeafView) ||
        (parameterName.size && !parameterName.data))
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (const program::Program *program = managedProgram(*pipeline)) {
        const program::BoundarySlot *slot = findPublicBoundary(*program, parameterName);
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
    const auto parameter =
        std::find_if(pipeline->variant.parameters.begin(), pipeline->variant.parameters.end(),
                     [&](const Parameter &candidate) { return stringViewEquals(parameterName, candidate.name); });
    if (parameter == pipeline->variant.parameters.end() || parameter->kind != "tensor")
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

VernonStatus vernonRuntimeLoadedPipelineGetImageConstraintByParameterIndex(
    const VernonLoadedPipeline *pipeline, size_t parameterIndex, VernonPipelineImageConstraintView *constraint) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !constraint || constraint->struct_size < sizeof(VernonPipelineImageConstraintView))
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (const program::Program *program = managedProgram(*pipeline)) {
        const program::BoundarySlot *slot = publicBoundaryAt(*program, parameterIndex);
        return slot ? fillBoundaryImageConstraintView(*slot, *constraint) : VERNON_STATUS_INVALID_ARGUMENT;
    }
    if (parameterIndex >= pipeline->variant.parameters.size())
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillImageConstraintView(pipeline->variant.parameters[parameterIndex], *constraint);
}

VernonStatus vernonRuntimeLoadedPipelineFindImageConstraint(const VernonLoadedPipeline *pipeline,
                                                            VernonStringView parameterName,
                                                            VernonPipelineImageConstraintView *constraint) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !constraint || constraint->struct_size < sizeof(VernonPipelineImageConstraintView) ||
        (parameterName.size && !parameterName.data))
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (const program::Program *program = managedProgram(*pipeline)) {
        const program::BoundarySlot *slot = findPublicBoundary(*program, parameterName);
        return slot ? fillBoundaryImageConstraintView(*slot, *constraint) : VERNON_STATUS_INVALID_ARGUMENT;
    }
    const auto found =
        std::find_if(pipeline->variant.parameters.begin(), pipeline->variant.parameters.end(),
                     [&](const Parameter &parameter) { return stringViewEquals(parameterName, parameter.name); });
    if (found == pipeline->variant.parameters.end())
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillImageConstraintView(*found, *constraint);
}

size_t vernonRuntimeLoadedPipelineGetOutputCount(const VernonLoadedPipeline *pipeline) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    return pipeline ? pipeline->variant.outputs.size() : 0;
}

VernonStatus vernonRuntimeLoadedPipelineGetOutputByIndex(const VernonLoadedPipeline *pipeline, size_t index,
                                                         VernonPipelineOutputView *output) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !output || index >= pipeline->variant.outputs.size())
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillOutputView(pipeline->variant.outputs[index], *output) ? VERNON_STATUS_OK : VERNON_STATUS_PARSE_ERROR;
}

VernonStatus vernonRuntimeLoadedPipelineFindOutput(const VernonLoadedPipeline *pipeline, VernonStringView name,
                                                   VernonPipelineOutputView *output) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !output || (name.size && !name.data))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const auto found = std::find_if(pipeline->variant.outputs.begin(), pipeline->variant.outputs.end(),
                                    [&](const Output &value) { return stringViewEquals(name, value.name); });
    if (found == pipeline->variant.outputs.end())
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillOutputView(*found, *output) ? VERNON_STATUS_OK : VERNON_STATUS_PARSE_ERROR;
}

void vernonRuntimeLoadedPipelineDestroy(VernonLoadedPipeline *pipeline) {
    if (!pipeline)
        return;
    RuntimeDiagnosticScope diagnostic(pipeline->context);
    destroyPipelineImplementations(*pipeline);
    --pipeline->context->livePipelines;
    delete pipeline;
}

uint8_t vernonRuntimeLoadedPipelineIsManagedProgram(const VernonLoadedPipeline *pipeline) {
    return pipeline && pipeline->topology && pipeline->topology->resolvedProgram;
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
                        const std::vector<VernonPipelineArgument> &arguments, VernonLaunchSize &grid,
                        std::string &error) {
    grid = {static_cast<uint32_t>(staticGrid[0]), static_cast<uint32_t>(staticGrid[1]),
            static_cast<uint32_t>(staticGrid[2])};
    if (mapping == vernon::runtime::program::DispatchMapping::StaticGrid)
        return true;
    for (const VernonPipelineArgument &argument : arguments) {
        if (argument.kind != VERNON_PIPELINE_TENSOR || (argument.tensor.rank && !argument.tensor.shape))
            continue;
        uint64_t count = 1;
        for (uint32_t dimension = 0; dimension < argument.tensor.rank; ++dimension) {
            const uint64_t extent = argument.tensor.shape[dimension];
            if (!extent || count > std::numeric_limits<uint32_t>::max() / extent) {
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
    PipelineComputePass(const program::Node &node, VernonLoadedPipeline &pipeline,
                        std::vector<VernonPipelineArgument> arguments,
                        std::vector<std::vector<uint64_t>> argumentShapes,
                        std::vector<std::vector<int64_t>> argumentStrides,
                        const std::vector<vernon::execution::GraphBuffer> &resources,
                        const std::vector<VernonPipelineArgument> *arena, const program::Program *program,
                        VernonLaunchSize grid)
        : ComputePass(programNodeName(node)), description_(describeProgramNode(node)), pipeline_(pipeline),
          arguments_(std::move(arguments)), argumentShapes_(std::move(argumentShapes)),
          argumentStrides_(std::move(argumentStrides)), resources_(resources), arena_(arena), program_(program),
          grid_(grid) {}

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
        VernonPipelineInvocation invocation{};
        invocation.struct_size = sizeof(invocation);
        invocation.abi_version = VERNON_PIPELINE_VERSION;
        invocation.arguments = arguments_.data();
        invocation.argument_count = arguments_.size();
        invocation.compute_grid = grid_;
        PlannedComputeLaunch plan;
        std::string error;
        const uint32_t workgroup[3]{pipeline_.workgroupSize.x, pipeline_.workgroupSize.y, pipeline_.workgroupSize.z};
        if (pipeline_.context && pipeline_.context->backend == VERNON_RUNTIME_CPU) {
            VernonAdTapeAllocator *allocator = nullptr;
            VernonAdRegionHandle root = VERNON_AD_INVALID_REGION_HANDLE;
            const auto bindTape = [&](uint32_t valueId) {
                if (!arena_ || !program_ || valueId >= arena_->size() || valueId >= program_->values.size() ||
                    !program::isTapeValueType(program_->values[valueId].type))
                    return;
                const VernonPipelineArgument &argument = (*arena_)[valueId];
                constexpr size_t allocatorDescriptorBytes = sizeof(VernonAdTapeAllocator *);
                constexpr size_t rootDescriptorBytes = sizeof(VernonAdRegionHandle);
                if (argument.kind != VERNON_PIPELINE_TENSOR || !argument.tensor.host_data ||
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
        if (!planComputeInvocation(pipeline_.variant, pipeline_.workgroupSize, invocation, plan, error)) {
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
        return VERNON_RHI_STATUS_OK;
    }

private:
    PipelineComputePassDescription description_;
    VernonLoadedPipeline &pipeline_;
    std::vector<VernonPipelineArgument> arguments_;
    std::vector<std::vector<uint64_t>> argumentShapes_;
    std::vector<std::vector<int64_t>> argumentStrides_;
    const std::vector<vernon::execution::GraphBuffer> &resources_;
    const std::vector<VernonPipelineArgument> *arena_{};
    const program::Program *program_{};
    VernonLaunchSize grid_{};
};

struct ManagedGraphicsCommandContext {
    VernonLoadedPipeline &pipeline;
    vernon::runtime::ad::MaterializedProgramArguments materialized;
    std::shared_ptr<RuntimeProgramControl> renderPass;
    std::shared_ptr<RuntimeProgramControl> draw;
    std::shared_ptr<RuntimeProgramControl> dynamic;
    VernonPipelineInvocation invocation{};
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

VernonStatus executePipelineProgramGraphImpl(VernonLoadedPipeline &pipeline, const program::Graph &graph,
                                             vernon::runtime::ad::ProgramInvocationFrame &arena) {
    if (!pipeline.topology || !pipeline.topology->resolvedProgram)
        return fail(pipeline.context, "Program graph has no resolved canonical owner");
    const program::ResolvedProgram &resolved = *pipeline.topology->resolvedProgram;
    const program::Program &canonicalProgram = resolved.program;
    const std::vector<VernonPipelineArgument> &valueArguments = arena.logicalArguments();
    if (valueArguments.size() != canonicalProgram.values.size())
        return fail(pipeline.context, "Program graph value set does not match its execution topology");
    const auto canonicalGraph =
        std::find_if(canonicalProgram.graphs.begin(), canonicalProgram.graphs.end(),
                     [&](const program::Graph &candidate) { return candidate.direction == graph.direction; });
    if (canonicalGraph == canonicalProgram.graphs.end())
        return fail(pipeline.context, "Program graph is not part of its resolved owner");
    const size_t graphIndex = static_cast<size_t>(canonicalGraph - canonicalProgram.graphs.begin());
    if (graphIndex >= resolved.graphs.size())
        return fail(pipeline.context, "Program graph has no resolved dependency plan");
    const program::ResolvedGraph &resolvedGraph = resolved.graphs[graphIndex];
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
            canonicalProgram, graph, resolvedGraph, arena,
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
        std::vector<vernon::runtime::ad::gpu::DeviceBufferUpload> storageUploads;
        storageUploads.reserve(arena.deviceUploads().size());
        for (const vernon::runtime::ad::ProgramDeviceUpload &upload : arena.deviceUploads())
            storageUploads.push_back({upload.destination, 0, upload.source, upload.size});
        vernon::execution::detail::RhiCommandExecutionPlan uploadPlan;
        if (const VernonStatus status =
                vernon::runtime::ad::gpu::buildBufferUploadCommandPlan(*pipeline.context, storageUploads, uploadPlan);
            status != VERNON_STATUS_OK)
            return status;
        if (!uploadPlan.commands.nodes.empty()) {
            std::string uploadError;
            if (!vernon::execution::detail::appendRhiCommandExecutionPlan(commandPlan, std::move(uploadPlan), true,
                                                                          uploadError))
                return fail(pipeline.context, std::move(uploadError));
        }
        std::shared_ptr<ManagedGraphicsCommandBatch> graphicsBatch;
        GraphicsScopePlanner graphicsScopes;
        const auto flushCommands = [&]() {
            if (commandPlan.commands.nodes.empty())
                return VERNON_STATUS_OK;
            graphicsBatch.reset();
            const VernonStatus status =
                vernon::runtime::ad::gpu::executeCommandPlanAndWait(*pipeline.context, commandPlan);
            commandPlan = {};
            return status;
        };
        std::vector<vernon::runtime::ad::MaterializedProgramArguments> materializedNodes;
        materializedNodes.reserve(graph.nodes.size());
        for (const program::Node &node : graph.nodes) {
            const auto stageIndex = pipeline.topology->stageIndices.find(node.stage);
            if (stageIndex == pipeline.topology->stageIndices.end())
                return fail(pipeline.context, "pipeline node has no resolved kernel stage");
            VernonResolvedProgramStage &resolvedStage = pipeline.topology->stages[stageIndex->second];
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
                if (!arena.materializeNodeArguments(canonicalProgram, node, resolvedStage, materializedNodes.back(),
                                                    materializationError))
                    return fail(pipeline.context, std::move(materializationError));
                if (draw)
                    draw->refresh();
                auto graphicsContext = std::make_shared<ManagedGraphicsCommandContext>(
                    ManagedGraphicsCommandContext{*resolvedStage.pipeline, std::move(materializedNodes.back()),
                                                  std::move(renderPass), std::move(draw), std::move(dynamic)});
                prepareGraphicsPipelineState(graphics, *graphicsContext);
                VernonPipelineInvocation &invocation = graphicsContext->invocation;
                invocation.struct_size = sizeof(invocation);
                invocation.abi_version = VERNON_PIPELINE_VERSION;
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
                        resolvedStage.pipeline->variant, invocation,
                        [](void *userData, VernonRuntimeProviderResourceReference resource,
                           VernonRuntimeProviderImageDescription *description) {
                            return describeBackendImage(*static_cast<VernonRuntimeContext *>(userData), resource,
                                                        *description);
                        },
                        pipeline.context, graphicsContext->plan, graphicsError))
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
            if (!arena.materializeNodeArguments(canonicalProgram, node, resolvedStage, materializedNodes.back(),
                                                materializationError))
                return fail(pipeline.context, std::move(materializationError));
            vernon::execution::detail::RhiCommandExecutionPlan nodePlan;
            VernonLaunchSize grid{};
            std::string gridError;
            const program::ComputeOperation &compute = program::computeOperation(node);
            uint64_t staticGrid[3]{};
            for (size_t axis = 0; axis < 3; ++axis)
                if (!arena.resolveControl(canonicalProgram, compute.workgroups[axis], staticGrid[axis], gridError))
                    return fail(pipeline.context, std::move(gridError));
            if (!resolveProgramGrid(resolvedStage.dispatchMapping, staticGrid, materializedNodes.back().arguments, grid,
                                    gridError))
                return fail(pipeline.context, std::move(gridError));
            const VernonStatus planned = vernon::runtime::ad::gpu::buildPipelineCommandPlan(
                *pipeline.context, {}, {}, *resolvedStage.pipeline, materializedNodes.back().arguments, grid, {},
                vernon::execution::detail::CommandNodeKind::Derivative, nodePlan);
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
        const VernonPipelineArgument &argument = valueArguments[index];
        if (argument.kind != VERNON_PIPELINE_TENSOR)
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
        const auto stageIndex = pipeline.topology->stageIndices.find(node.stage);
        if (stageIndex == pipeline.topology->stageIndices.end())
            return fail(pipeline.context, "pipeline node has no resolved kernel stage");
        VernonResolvedProgramStage &resolvedStage = pipeline.topology->stages[stageIndex->second];
        VernonLoadedPipeline &stage = *resolvedStage.pipeline;
        vernon::runtime::ad::MaterializedProgramArguments materialized;
        std::string materializationError;
        if (!arena.materializeNodeArguments(canonicalProgram, node, resolvedStage, materialized, materializationError))
            return fail(pipeline.context, std::move(materializationError));
        uint64_t controlGrid[3]{};
        for (size_t axis = 0; axis < 3; ++axis)
            if (!arena.resolveControl(canonicalProgram, program::computeOperation(node).workgroups[axis],
                                      controlGrid[axis], materializationError))
                return fail(pipeline.context, std::move(materializationError));
        VernonLaunchSize grid{};
        if (!resolveProgramGrid(resolvedStage.dispatchMapping, controlGrid, materialized.arguments, grid,
                                materializationError))
            return fail(pipeline.context, std::move(materializationError));
        auto &pass = executionGraph.emplacePass<PipelineComputePass>(
            node, stage, std::move(materialized.arguments), std::move(materialized.shapes),
            std::move(materialized.strides), resources, &valueArguments, &canonicalProgram, grid);
        if (node.id >= resolvedGraph.predecessors.size())
            return fail(pipeline.context, "Program node has no resolved dependency entry");
        for (uint32_t dependency : resolvedGraph.predecessors[node.id]) {
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

VernonStatus executeManagedProgram(VernonLoadedPipeline &pipeline, const VernonPipelineInvocation &invocation) {
    VernonPullback *pullback = nullptr;
    const VernonStatus status = vernon::runtime::ad::forwardProgramInvocation(pipeline, invocation, pullback);
    if (pullback)
        vernonPullbackDestroy(pullback);
    return status;
}

VernonStatus encodePipelineInvocation(VernonLoadedPipeline &pipeline, const VernonPipelineInvocation &invocation) {
    if (pipeline.topology)
        return executeManagedProgram(pipeline, invocation);
    if (!pipeline.variant.compute.empty()) {
        const uint32_t grid[3]{invocation.compute_grid.x, invocation.compute_grid.y, invocation.compute_grid.z};
        const uint32_t workgroup[3]{pipeline.workgroupSize.x, pipeline.workgroupSize.y, pipeline.workgroupSize.z};
        if (!validateDispatchContract(pipeline.dispatchContract, grid, workgroup,
                                      invocationDiagnostic(*pipeline.context)))
            return VERNON_STATUS_INVALID_ARGUMENT;
        PlannedComputeLaunch plan;
        std::string planningError;
        if (!planComputeInvocation(pipeline.variant, pipeline.workgroupSize, invocation, plan, planningError))
            return fail(pipeline.context, planningError);
        return invokeBackendComputePipeline(pipeline, plan);
    }

    PlannedGraphicsInvocation plan;
    std::string planningError;
    if (!planGraphicsInvocation(
            pipeline.variant, invocation,
            [](void *userData, VernonRuntimeProviderResourceReference resource,
               VernonRuntimeProviderImageDescription *description) {
                return describeBackendImage(*static_cast<VernonRuntimeContext *>(userData), resource, *description);
            },
            pipeline.context, plan, planningError))
        return fail(pipeline.context, planningError);
    return invokeBackendPipeline(pipeline, invocation, plan);
}

} // namespace

VernonStatus vernonRuntimePipelineSubmit(VernonLoadedPipeline *pipeline, const VernonPipelineInvocation *invocation,
                                         VernonSubmission **output) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (output)
        *output = nullptr;
    if (!pipeline || !invocation || !output || invocation->struct_size < sizeof(VernonPipelineInvocation) ||
        invocation->abi_version != VERNON_PIPELINE_VERSION || (invocation->argument_count && !invocation->arguments) ||
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
        const VernonStatus status = encodePipelineInvocation(*pipeline, *invocation);
        if (status != VERNON_STATUS_OK)
            return status;
        submission->state = VERNON_SUBMISSION_SUCCEEDED;
        *output = submission.release();
        return VERNON_STATUS_OK;
    }

    const bool graphics = pipeline->variant.compute.empty();
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = graphics ? VERNON_RHI_QUEUE_GRAPHICS : VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder native{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    if (vernonRhiDeviceCreateCommandEncoder(pipeline->context->rhiDevice, &descriptor, &native) != VERNON_RHI_STATUS_OK)
        return fail(pipeline->context, "failed to create the immediate command encoder");
    VernonPipelineInvocation encoded = *invocation;
    VernonStatus status = referenceBackendCommandEncoder(*pipeline->context, native, encoded.command_encoder);
    bool rendering = false;
    if (status == VERNON_STATUS_OK && graphics) {
        rendering = vernon::rhi::beginProviderRendering(pipeline->context->rhiDevice, native);
        if (!rendering)
            status = fail(pipeline->context, "failed to begin immediate rendering");
    }
    if (status == VERNON_STATUS_OK)
        status = encodePipelineInvocation(*pipeline, encoded);
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

VernonStatus vernonRuntimePipelineEncode(VernonRuntimeProviderObject encoder, VernonLoadedPipeline *pipeline,
                                         const VernonPipelineInvocation *invocation) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !invocation || invocation->struct_size < sizeof(VernonPipelineInvocation) ||
        invocation->abi_version != VERNON_PIPELINE_VERSION || (invocation->argument_count && !invocation->arguments))
        return fail(pipeline ? pipeline->context : nullptr, "invalid pipeline invocation");
    VernonPipelineInvocation encoded = *invocation;
    encoded.command_encoder = encoder;
    return encodePipelineInvocation(*pipeline, encoded);
}

VernonProgramInstance *vernonRuntimeProgramInstanceCreate(VernonLoadedPipeline *pipeline) {
    try {
        RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
        if (!pipeline || !pipeline->context || !pipeline->topology || !pipeline->topology->resolvedProgram)
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
                                                const VernonPipelineArgument *argument,
                                                const VernonProgramResourceLease *lease, uint64_t uploadBytes,
                                                uint64_t uploadRanges) {
    VernonRuntimeContext *context = invocation && invocation->instance && invocation->instance->pipeline
                                        ? invocation->instance->pipeline->context
                                        : nullptr;
    try {
        RuntimeDiagnosticScope diagnostic(context);
        if (!invocation || invocation->finished || !token || token->struct_size < sizeof(*token) || !token->size ||
            !token->data || !argument || argument->kind < VERNON_PIPELINE_TENSOR ||
            argument->kind > VERNON_PIPELINE_SAMPLER)
            return fail(context, "invalid persistent Program binding update");
        const std::string key(static_cast<const char *>(token->data), token->size);
        if (invocation->transaction->find(argument->slot, key)) {
            invocation->transaction->observeUploads(uploadBytes, uploadRanges);
            return VERNON_STATUS_OK;
        }
        auto binding = std::make_shared<RuntimeProgramBinding>();
        binding->argument = *argument;
        if (argument->kind == VERNON_PIPELINE_TENSOR) {
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
    VernonLoadedPipeline *pipeline = invocation && invocation->instance ? invocation->instance->pipeline : nullptr;
    try {
        RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
        if (outputPullback)
            *outputPullback = nullptr;
        if (!invocation || invocation->finished || !pipeline)
            return fail(pipeline ? pipeline->context : nullptr, "invalid persistent Program invocation");
        invocation->snapshot = invocation->transaction->snapshot();
        invocation->controlSnapshot = invocation->controlTransaction->snapshot();
        std::vector<VernonPipelineArgument> arguments;
        const size_t count = vernonRuntimeLoadedPipelineGetParameterCount(pipeline);
        arguments.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            VernonPipelineParameterView parameter{};
            if (vernonRuntimeLoadedPipelineGetParameterByIndex(pipeline, index, &parameter) != VERNON_STATUS_OK)
                return fail(pipeline->context, "cannot reflect persistent Program boundary slot");
            const std::shared_ptr<void> *payload = invocation->snapshot->find(parameter.slot);
            if (!payload)
                return fail(pipeline->context, "persistent Program invocation has an unbound boundary slot");
            auto binding = std::static_pointer_cast<RuntimeProgramBinding>(*payload);
            binding->refresh();
            arguments.push_back(binding->argument);
        }
        VernonPipelineInvocation frame{};
        frame.struct_size = sizeof(frame);
        frame.abi_version = VERNON_PIPELINE_VERSION;
        frame.arguments = arguments.data();
        frame.argument_count = arguments.size();
        frame.compute_grid = {1, 1, 1};
        const ProgramInvocationContext programContext{*invocation->controlSnapshot};
        VernonPullback *pullback = nullptr;
        const VernonStatus status =
            vernon::runtime::ad::forwardProgramInvocation(*pipeline, frame, pullback, &programContext);
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
            pullback->programSnapshot = invocation->snapshot;
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

VernonStatus vernon::runtime::executePipelineProgramGraph(VernonLoadedPipeline &pipeline, const program::Graph &graph,
                                                          vernon::runtime::ad::ProgramInvocationFrame &arena) {
    if (!pipeline.topology)
        return fail(pipeline.context, "resolved pipeline has no execution topology");
    return executePipelineProgramGraphImpl(pipeline, graph, arena);
}

VernonPipelineTopology::~VernonPipelineTopology() {
    for (const auto &stage : stages)
        if (stage.pipeline && stage.pipeline->backendState)
            vernon::runtime::destroyBackendPipeline(*stage.pipeline);
}
