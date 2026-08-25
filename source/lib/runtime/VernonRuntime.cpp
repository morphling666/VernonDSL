#include "VernonRuntime.h"
#include "VernonExecutionGraph.h"
#include "rhi/rhi_internal.h"
#include "runtime/autodiff/runtime_autodiff_internal.h"
#include "runtime/compute_launch_planner.h"
#include "runtime/graphics_invocation_planner.h"
#include "runtime/pipeline_bundle.h"
#include "runtime/pipeline_manifest.h"
#include "runtime/pipeline_metadata.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_state.h"
#include "runtime/tensor_bridge.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
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

ArtifactResolution artifactResolutionFor(const VernonRuntimeContext &context, const RuntimeRequirements &requirements) {
#if defined(VERNON_RUNTIME_PROFILE_WEB)
    if (context.backend == VERNON_RUNTIME_CPU && requirements.objectFormat == "wasm")
        return ArtifactResolution::MetadataOnly;
#else
    (void)context;
    (void)requirements;
#endif
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
            const ArtifactResolution resolution = artifactResolutionFor(*context, requirements);
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
                if (!resolved.external || (resolution == ArtifactResolution::LoadBytes && !bundleDirectory)) {
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
                artifact.staticallyLinked = resolution == ArtifactResolution::MetadataOnly;
                std::filesystem::path validatedPath;
                const std::string format = resolved.format;
                if ((format != "native_library" && format != "relocatable_object") ||
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
                std::string artifactKind = stage;
                if (variant.executable) {
                    artifactKind.clear();
                    for (const ProgramGraph &graph : variant.executable->graphs)
                        for (const ProgramNode &node : graph.nodes)
                            if (node.stage == stage) {
                                const std::string nodeKind = node.kind == "compute" ? "compute" : "graphics";
                                if (artifactKind.empty())
                                    artifactKind = nodeKind;
                                else if (artifactKind != nodeKind) {
                                    fail(context, "executable program stage is used with incompatible node kinds");
                                    return nullptr;
                                }
                            }
                }
                if (artifactKind.empty() || !validStage(id, artifactKind)) {
                    fail(context, "pipeline variant references an invalid program stage");
                    return nullptr;
                }
            }
            if (variant.executable) {
                bundle->variants.push_back(std::move(variant));
                continue;
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

bool resolvePipelineTopology(VernonPipelineBundle &bundle, const Variant &variant, VernonLoadedPipeline &pipeline) {
    if (!variant.executable)
        return false;
    auto topology = std::make_shared<VernonPipelineTopology>();
    topology->execution = *variant.executable;
    topology->residualValues = topology->execution.backwardCaptures();

    for (const ProgramGraph &graph : variant.executable->graphs) {
        for (const ProgramNode &node : graph.nodes) {
            if (node.kind != "compute") {
                fail(bundle.context, "executable Program graphics nodes are not implemented",
                     VERNON_STATUS_UNSUPPORTED_TARGET);
                return false;
            }
            if (topology->stageIndices.find(node.stage) != topology->stageIndices.end())
                continue;
            const auto artifact = variant.program.find(node.stage);
            if (artifact == variant.program.end()) {
                fail(bundle.context, "Program node '" + node.name + "' has no implementation artifact");
                return false;
            }
            auto child = std::make_unique<VernonLoadedPipeline>();
            child->context = bundle.context;
            child->variant.compute = artifact->second;
            struct StageUse {
                const Parameter *owner{};
                const ParameterUse *use{};
            };
            std::vector<StageUse> stageUses;
            const auto collectUses = [&](const std::vector<Parameter> &parameters) {
                for (const Parameter &parameter : parameters)
                    for (const ParameterUse &use : parameter.uses)
                        if (use.stage == node.stage)
                            stageUses.push_back(StageUse{&parameter, &use});
            };
            collectUses(variant.parameters);
            collectUses(variant.internalParameters);
            std::sort(stageUses.begin(), stageUses.end(),
                      [](const StageUse &left, const StageUse &right) { return left.use->index < right.use->index; });
            if (stageUses.size() != node.bindings.size()) {
                fail(bundle.context, "Program node '" + node.name + "' stage ABI does not match its value bindings");
                return false;
            }
            std::vector<VernonProgramStageBinding> resolvedBindings;
            resolvedBindings.reserve(stageUses.size());
            std::vector<bool> consumedBindings(node.bindings.size());
            for (size_t index = 0; index < stageUses.size(); ++index) {
                const StageUse &stageUse = stageUses[index];
                const auto value = std::find_if(
                    topology->execution.values.begin(), topology->execution.values.end(),
                    [&](const ProgramValueSlot &candidate) { return candidate.name == stageUse.owner->name; });
                const auto binding =
                    std::find_if(node.bindings.begin(), node.bindings.end(), [&](const ProgramValueBinding &candidate) {
                        const size_t bindingIndex = static_cast<size_t>(&candidate - node.bindings.data());
                        return !consumedBindings[bindingIndex] && value != topology->execution.values.end() &&
                               candidate.value == value->id;
                    });
                if (value == topology->execution.values.end() || binding == node.bindings.end()) {
                    fail(bundle.context,
                         "Program node '" + node.name + "' stage ABI is not bound to its canonical root value");
                    return false;
                }
                consumedBindings[static_cast<size_t>(&*binding - node.bindings.data())] = true;
                Parameter parameter = *stageUse.owner;
                parameter.slot = static_cast<uint32_t>(index);
                parameter.name = binding->parameter;
                parameter.shape = stageUse.use->shape;
                parameter.uses = {*stageUse.use};
                parameter.uses.front().stage = "compute";
                if (stageUse.use->valueLayout) {
                    parameter.valueLayout.reset();
                    parameter.elementLayout = *stageUse.use->valueLayout;
                }
                child->variant.parameters.push_back(std::move(parameter));

                const size_t aliases = static_cast<size_t>(
                    std::count_if(stageUses.begin(), stageUses.end(),
                                  [&](const StageUse &candidate) { return candidate.owner == stageUse.owner; }));
                std::optional<size_t> leaf;
                if (aliases > 1) {
                    const size_t leafIndex = static_cast<size_t>(
                        std::count_if(stageUses.begin(), stageUses.begin() + static_cast<std::ptrdiff_t>(index),
                                      [&](const StageUse &candidate) { return candidate.owner == stageUse.owner; }));
                    const ProgramValueSlot &slot = topology->execution.values[binding->value];
                    if (!slot.valueLayout || slot.valueLayout->leaves.size() != aliases ||
                        leafIndex >= slot.valueLayout->leaves.size() ||
                        slot.valueLayout->leaves[leafIndex].shape != stageUse.use->shape) {
                        fail(bundle.context,
                             "Program node '" + node.name + "' aggregate leaf ABI does not match its root value");
                        return false;
                    }
                    leaf = leafIndex;
                }
                resolvedBindings.push_back(VernonProgramStageBinding{binding->value, leaf});
            }
            const Stage &stage = bundle.stages.at(artifact->second);
            child->workgroupSize = {stage.workgroup[0], stage.workgroup[1], stage.workgroup[2]};
            child->dispatchContract = stage.dispatchContract;
            child->readFootprints = stage.readFootprints;
            child->writeFootprints = stage.writeFootprints;
            if (!resolveBackendPipeline(bundle, child->variant, *child)) {
                const std::string detail = invocationDiagnostic(*bundle.context);
                fail(bundle.context, "cannot resolve Program stage '" + node.stage + "'" +
                                         (detail.empty() ? std::string{} : ": " + detail));
                return false;
            }
            topology->stageIndices.emplace(node.stage, topology->stages.size());
            topology->stages.push_back(VernonResolvedProgramStage{std::move(child), std::move(resolvedBindings)});
        }
    }
    pipeline.topology = std::move(topology);
    return true;
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
        for (Parameter &parameter : pipeline->variant.parameters) {
            if (parameter.valueLayout)
                rebuildValueLayoutPathViews(*parameter.valueLayout);
            rebuildValueLayoutPathViews(parameter.elementLayout);
        }
        for (Parameter &parameter : pipeline->variant.internalParameters) {
            if (parameter.valueLayout)
                rebuildValueLayoutPathViews(*parameter.valueLayout);
            rebuildValueLayoutPathViews(parameter.elementLayout);
        }
        if (isDirectPipelineTopology(*found)) {
            if (!initializeDirectPipelineTopology(*found, *pipeline, invocationDiagnostic(*bundle->context)))
                return nullptr;
        } else if (found->executable) {
            if (!resolvePipelineTopology(*bundle, *found, *pipeline))
                return nullptr;
        }
        const bool nativeProgramAutodiff =
            pipeline->topology &&
            std::any_of(pipeline->topology->execution.graphs.begin(), pipeline->topology->execution.graphs.end(),
                        [](const ProgramGraph &graph) { return graph.direction == "backward"; });
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
        if ((!pipeline->topology || pipeline->topology->directDispatch) &&
            !resolveBackendPipeline(*bundle, *found, *pipeline)) {
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
    return pipeline ? pipeline->variant.parameters.size() : 0;
}

VernonStatus vernonRuntimeLoadedPipelineGetParameterByIndex(const VernonLoadedPipeline *pipeline, size_t index,
                                                            VernonPipelineParameterView *parameter) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !parameter || index >= pipeline->variant.parameters.size())
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillParameterView(pipeline->variant.parameters[index], *parameter) ? VERNON_STATUS_OK
                                                                              : VERNON_STATUS_PARSE_ERROR;
}

VernonStatus vernonRuntimeLoadedPipelineFindParameter(const VernonLoadedPipeline *pipeline, VernonStringView name,
                                                      VernonPipelineParameterView *parameter) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !parameter || (name.size && !name.data))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const auto found = std::find_if(pipeline->variant.parameters.begin(), pipeline->variant.parameters.end(),
                                    [&](const Parameter &p) { return stringViewEquals(name, p.name); });
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
    if (!pipeline || !constraint || constraint->struct_size < sizeof(VernonPipelineImageConstraintView) ||
        parameterIndex >= pipeline->variant.parameters.size())
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

namespace {

struct PipelineOwnedTensor {
    std::vector<uint8_t> bytes;
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;
};

class PipelineComputePass final : public vernon::execution::ComputePass {
public:
    PipelineComputePass(const ProgramNode &node, VernonLoadedPipeline &pipeline,
                        std::vector<VernonPipelineArgument> arguments,
                        std::vector<std::vector<int64_t>> argumentStrides,
                        const std::vector<vernon::execution::GraphBuffer> &resources)
        : ComputePass(node.name), node_(node), pipeline_(pipeline), arguments_(std::move(arguments)),
          argumentStrides_(std::move(argumentStrides)), resources_(resources) {}

    void declare() override {
        for (const ProgramResourceUse &use : node_.resources) {
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
        invocation.compute_grid = {static_cast<uint32_t>(node_.grid[0]), static_cast<uint32_t>(node_.grid[1]),
                                   static_cast<uint32_t>(node_.grid[2])};
        PlannedComputeLaunch plan;
        std::string error;
        const uint32_t grid[3]{invocation.compute_grid.x, invocation.compute_grid.y, invocation.compute_grid.z};
        const uint32_t workgroup[3]{pipeline_.workgroupSize.x, pipeline_.workgroupSize.y, pipeline_.workgroupSize.z};
        if (!validateDispatchContract(pipeline_.dispatchContract, grid, workgroup, error) ||
            !planComputeInvocation(pipeline_.variant, pipeline_.workgroupSize, invocation, plan, error)) {
            invocationDiagnostic(*pipeline_.context) = std::move(error);
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        }
        return invokeBackendComputePipeline(pipeline_, plan) == VERNON_STATUS_OK ? VERNON_RHI_STATUS_OK
                                                                                 : VERNON_RHI_STATUS_INTERNAL_ERROR;
    }

private:
    const ProgramNode &node_;
    VernonLoadedPipeline &pipeline_;
    std::vector<VernonPipelineArgument> arguments_;
    std::vector<std::vector<int64_t>> argumentStrides_;
    const std::vector<vernon::execution::GraphBuffer> &resources_;
};

bool contiguousTensorStorage(const Parameter &parameter, PipelineOwnedTensor &owned, VernonPipelineArgument &argument) {
    const ValueLayout &layout = parameter.valueLayout ? *parameter.valueLayout : parameter.elementLayout;
    if (!layout.byteSize)
        return false;
    uint64_t elements = 1;
    for (uint64_t extent : parameter.shape) {
        if (!extent || elements > std::numeric_limits<uint64_t>::max() / extent)
            return false;
        elements *= extent;
    }
    if (elements > std::numeric_limits<size_t>::max() / layout.byteSize)
        return false;
    owned.bytes.resize(static_cast<size_t>(elements * layout.byteSize));
    owned.shape = parameter.shape;
    owned.strides.resize(owned.shape.size());
    uint64_t stride = layout.byteSize;
    for (size_t index = owned.shape.size(); index-- > 0;) {
        if (stride > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            return false;
        owned.strides[index] = static_cast<int64_t>(stride);
        stride *= owned.shape[index];
    }
    argument = {};
    argument.kind = VERNON_PIPELINE_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_HOST;
    argument.tensor.host_data = owned.bytes.data();
    argument.tensor.element_layout = valueLayoutView(layout);
    argument.tensor.access = VERNON_ACCESS_READ_WRITE;
    argument.tensor.rank = static_cast<uint32_t>(owned.shape.size());
    argument.tensor.shape = owned.shape.data();
    argument.tensor.byte_strides = owned.strides.data();
    argument.tensor.byte_size = owned.bytes.size();
    return true;
}

VernonStatus executePipelineProgramGraphImpl(VernonLoadedPipeline &pipeline, const ProgramGraph &graph,
                                             const std::vector<VernonPipelineArgument> &valueArguments) {
    const ExecutableProgram &execution = pipeline.topology->execution;
    if (valueArguments.size() != execution.values.size())
        return fail(pipeline.context, "Program graph value set does not match its execution topology");
    vernon::execution::ExecutionGraph executionGraph;
    std::vector<vernon::execution::GraphBuffer> resources;
    resources.reserve(valueArguments.size());
    for (size_t index = 0; index < valueArguments.size(); ++index) {
        const VernonPipelineArgument &argument = valueArguments[index];
        if (argument.kind != VERNON_PIPELINE_TENSOR || argument.tensor.storage != VERNON_TENSOR_HOST ||
            !argument.tensor.host_data)
            return fail(pipeline.context, "Program graph requires materialized host tensor values");
        const uint64_t identity = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(argument.tensor.host_data));
        resources.push_back(executionGraph.importHostBuffer(identity, execution.values[index].output));
    }
    std::vector<vernon::execution::ExecutionPass *> passes(graph.nodes.size());
    for (const ProgramNode &node : graph.nodes) {
        const auto stageIndex = pipeline.topology->stageIndices.find(node.stage);
        if (stageIndex == pipeline.topology->stageIndices.end())
            return fail(pipeline.context, "pipeline node has no resolved kernel stage");
        VernonResolvedProgramStage &resolvedStage = pipeline.topology->stages[stageIndex->second];
        VernonLoadedPipeline &stage = *resolvedStage.pipeline;
        if (resolvedStage.bindings.size() != stage.variant.parameters.size())
            return fail(pipeline.context, "resolved Program stage has an invalid binding plan");
        std::vector<VernonPipelineArgument> arguments;
        std::vector<std::vector<int64_t>> argumentStrides;
        arguments.reserve(stage.variant.parameters.size());
        argumentStrides.reserve(stage.variant.parameters.size());
        for (size_t parameterIndex = 0; parameterIndex < stage.variant.parameters.size(); ++parameterIndex) {
            const Parameter &parameter = stage.variant.parameters[parameterIndex];
            const VernonProgramStageBinding &binding = resolvedStage.bindings[parameterIndex];
            const uint32_t valueId = binding.value;
            if (valueId >= valueArguments.size())
                return fail(pipeline.context, "resolved Program stage binding exceeds the value arena");
            VernonPipelineArgument argument = valueArguments[valueId];
            argument.slot = parameter.slot;
            argumentStrides.emplace_back();
            if (binding.leaf) {
                const ProgramValueSlot &slot = execution.values[valueId];
                if (!slot.valueLayout || *binding.leaf >= slot.valueLayout->leaves.size())
                    return fail(pipeline.context, "resolved Program stage leaf exceeds the canonical Value ABI");
                const ValueLeaf &leaf = slot.valueLayout->leaves[*binding.leaf];
                const ValueLayout &parameterLayout =
                    parameter.valueLayout ? *parameter.valueLayout : parameter.elementLayout;
                const std::vector<uint64_t> &parameterShape = parameter.shape;
                argument.tensor.byte_offset += leaf.byteOffset;
                argument.tensor.element_layout = valueLayoutView(parameterLayout);
                argument.tensor.rank = static_cast<uint32_t>(parameterShape.size());
                argument.tensor.shape = parameterShape.empty() ? nullptr : parameterShape.data();
                std::vector<int64_t> &strides = argumentStrides.back();
                strides.resize(parameterShape.size());
                uint64_t stride = parameterLayout.byteSize;
                for (size_t dimension = parameterShape.size(); dimension-- > 0;) {
                    if (stride > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
                        return fail(pipeline.context, "Program aggregate leaf stride overflows");
                    strides[dimension] = static_cast<int64_t>(stride);
                    if (parameterShape[dimension] &&
                        stride > std::numeric_limits<uint64_t>::max() / parameterShape[dimension])
                        return fail(pipeline.context, "Program aggregate leaf footprint overflows");
                    stride *= parameterShape[dimension];
                }
                argument.tensor.byte_strides = strides.empty() ? nullptr : strides.data();
            }
            arguments.push_back(argument);
        }
        auto &pass = executionGraph.emplacePass<PipelineComputePass>(node, stage, std::move(arguments),
                                                                     std::move(argumentStrides), resources);
        for (uint32_t dependency : node.dependencies) {
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

VernonStatus executePipelineTopology(VernonLoadedPipeline &pipeline, const VernonPipelineInvocation &invocation) {
    if (!pipeline.topology)
        return fail(pipeline.context, "resolved pipeline has no execution topology");
    if (pipeline.context->backend != VERNON_RUNTIME_CPU)
        return fail(pipeline.context, "pipeline ExecutionGraph currently supports CPU compute execution",
                    VERNON_STATUS_UNSUPPORTED_TARGET);
    const ExecutableProgram &execution = pipeline.topology->execution;
    const ProgramGraph *forward = nullptr;
    for (const ProgramGraph &graph : execution.graphs)
        if (graph.direction == "forward") {
            forward = &graph;
            break;
        }
    if (!forward)
        return fail(pipeline.context, "executable Program has no forward graph");
    std::unordered_map<uint32_t, const VernonPipelineArgument *> supplied;
    for (size_t index = 0; index < invocation.argument_count; ++index)
        if (!supplied.emplace(invocation.arguments[index].slot, &invocation.arguments[index]).second)
            return fail(pipeline.context, "Program invocation contains duplicate argument slots");
    if (supplied.size() != pipeline.variant.parameters.size())
        return fail(pipeline.context, "Program invocation does not cover graph boundary values");

    const size_t valueCount = execution.values.size();
    std::vector<VernonPipelineArgument> valueArguments(valueCount);
    std::vector<uint8_t> populated(valueCount);
    std::vector<PipelineOwnedTensor> owned(valueCount);
    const auto valueIdByName = [&](const std::string &name) -> std::optional<uint32_t> {
        const auto found = std::find_if(execution.values.begin(), execution.values.end(),
                                        [&](const ProgramValueSlot &value) { return value.name == name; });
        return found == execution.values.end() ? std::nullopt : std::optional<uint32_t>(found->id);
    };
    for (const Parameter &parameter : pipeline.variant.parameters) {
        const auto value = valueIdByName(parameter.name);
        const auto argument = supplied.find(parameter.slot);
        if (!value || argument == supplied.end() || argument->second->kind != VERNON_PIPELINE_TENSOR)
            return fail(pipeline.context, "Program boundary parameter does not match an executable value");
        valueArguments[*value] = *argument->second;
        populated[*value] = 1;
    }
    for (const Parameter &parameter : pipeline.variant.internalParameters) {
        if (parameter.source != "program_value")
            continue;
        const auto value = valueIdByName(parameter.name);
        if (!value || !contiguousTensorStorage(parameter, owned[*value], valueArguments[*value]))
            return fail(pipeline.context, "cannot allocate internal Program value");
        populated[*value] = 1;
    }
    if (std::find(populated.begin(), populated.end(), 0) != populated.end())
        return fail(pipeline.context, "Program invocation has an unmaterialized value");

    return executePipelineProgramGraphImpl(pipeline, *forward, valueArguments);
}

VernonStatus encodePipelineInvocation(VernonLoadedPipeline &pipeline, const VernonPipelineInvocation &invocation) {
    if (pipeline.topology && !pipeline.topology->directDispatch)
        return executePipelineTopology(pipeline, invocation);
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

VernonStatus vernon::runtime::executePipelineProgramGraph(VernonLoadedPipeline &pipeline, const ProgramGraph &graph,
                                                          const std::vector<VernonPipelineArgument> &values) {
    if (!pipeline.topology)
        return fail(pipeline.context, "resolved pipeline has no execution topology");
    if (pipeline.context->backend != VERNON_RUNTIME_CPU)
        return fail(pipeline.context, "pipeline Program graph currently supports CPU compute execution",
                    VERNON_STATUS_UNSUPPORTED_TARGET);
    return executePipelineProgramGraphImpl(pipeline, graph, values);
}

VernonPipelineTopology::~VernonPipelineTopology() {
    for (const auto &stage : stages)
        if (stage.pipeline && stage.pipeline->backendState)
            vernon::runtime::destroyBackendPipeline(*stage.pipeline);
}
