#include "VernonRuntime.h"
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

VernonLoadedPipeline *vernonRuntimeLoadComputeBundle(VernonRuntimeContext *context, const char *directory) {
    RuntimeDiagnosticScope diagnostic(context);
    if (!context || context->backend != VERNON_RUNTIME_CPU || !directory)
        return nullptr;
    try {
        const std::filesystem::path root = std::filesystem::u8path(directory);
        CpuNativeArtifact artifact;
        if (!parseCpuComputeBundle(root, artifact, invocationDiagnostic(*context)))
            return nullptr;
        return loadBackendCpuNativePipeline(*context, artifact);
    } catch (const std::exception &error) {
        fail(context, std::string("failed to load CPU AOT bundle: ") + error.what(), VERNON_STATUS_INTERNAL_ERROR);
        return nullptr;
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
        if (!pipelineSchema || !validateManifestHash(root, true, manifestError))
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
        if (!pipelineSchema || pipelineTargetKind(root) != expectedTarget || !root.contains("stage_artifacts") ||
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
        if (root.contains("program_transform"))
            bundle->autodiff = std::move(autodiff);
        for (const nlohmann::json &feature : root.value("features", nlohmann::json::array()))
            bundle->features.push_back(feature.get<std::string>());
        if (!std::is_sorted(bundle->features.begin(), bundle->features.end()) ||
            std::adjacent_find(bundle->features.begin(), bundle->features.end()) != bundle->features.end()) {
            fail(context, "pipeline feature table is not canonical");
            return nullptr;
        }
        for (const auto &[id, value] : root["stage_artifacts"].items()) {
            if (!value.is_object() ||
                (value.contains("id") && (!value["id"].is_string() || value["id"].get<std::string>() != id))) {
                fail(context, "pipeline stage id is invalid");
                return nullptr;
            }
            Stage stage;
            stage.stage = value.value("stage", "");
            stage.entry = value.value("entry", "");
            if (value.contains("autodiff_profile") || value.contains("autodiff_protocol") ||
                value.contains("autodiff_profiles_identity"))
                stage.autodiff =
                    AutodiffStageMetadata{value.value("autodiff_profile", ""), value.value("autodiff_protocol", ""),
                                          value.value("autodiff_profiles_identity", "")};
            if (value.contains("reflection") && value["reflection"].contains("entries")) {
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
                    if (entry.contains("serial_dispatch") && !entry["serial_dispatch"].is_boolean()) {
                        fail(context, "compute serial-dispatch metadata must be Boolean");
                        return nullptr;
                    }
                    stage.serialDispatch = entry.value("serial_dispatch", false);
                }
            }
            if (!value.contains("artifact")) {
                fail(context, "pipeline stage artifact descriptor is missing");
                return nullptr;
            }
            ResolvedArtifact resolved;
            if (!resolveArtifact(value["artifact"], bundleDirectory, resolved, invocationDiagnostic(*context)))
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
                  (resolved.format != "spirv" && resolved.format != "dxil" && encoding != "utf8"))) ||
                value.value("target", "") != expectedTarget) {
                fail(context, "pipeline stage artifact format is invalid for target");
                return nullptr;
            }
            if (context->backend == VERNON_RUNTIME_CPU) {
                if (!bundleDirectory || !resolved.external) {
                    fail(context, "CPU pipeline artifacts require an external bundle directory");
                    return nullptr;
                }
                const nlohmann::json &nativeArtifact =
                    value.contains("artifact") && value["artifact"].is_object() ? value["artifact"] : value;
                CpuNativeArtifact artifact;
                artifact.root = *bundleDirectory;
                artifact.relativeLibrary =
                    std::filesystem::u8path(nativeArtifact.value("path", value.value("native_library", "")));
                artifact.entry = stage.entry;
                artifact.format = resolved.format;
                artifact.symbol = value.value("symbol", "");
                artifact.operatingSystem = value.value("operating_system", "");
                artifact.architecture = value.value("architecture", "");
                artifact.targetTriple = value.value("target_triple", "");
                artifact.objectFormat = value.value("object_format", "");
                artifact.size = nativeArtifact.value("size", uint64_t{0});
                artifact.sha256 = nativeArtifact.value("sha256", "");
                if (value.contains("reflection"))
                    artifact.reflection = value["reflection"];
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
            auto validStage = [&](const std::string &id, const char *kind) {
                if (id.empty())
                    return true;
                const auto found = bundle->stages.find(id);
                return found != bundle->stages.end() && found->second.stage == kind;
            };
            for (const auto &[stage, id] : variant.program)
                if (!validStage(id, stage.c_str())) {
                    fail(context, "pipeline variant references an invalid program stage");
                    return nullptr;
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
            if (bundle->autodiff->variants.size() != bundle->variants.size()) {
                fail(context, "autodiff profiles do not cover every pipeline variant");
                return nullptr;
            }
            for (size_t index = 0; index < bundle->variants.size(); ++index) {
                const Variant &variant = bundle->variants[index];
                const AutodiffVariant &profiles = bundle->autodiff->variants[index];
                auto profileStage = [&](const std::string &id, const char *profile) {
                    const auto found = bundle->stages.find(id);
                    return found != bundle->stages.end() && found->second.stage == "compute" &&
                           found->second.autodiff && found->second.autodiff->profile == profile &&
                           found->second.autodiff->profilesIdentity == profiles.planIdentity;
                };
                if (profiles.key != variant.key || profiles.primal != variant.compute ||
                    profiles.primal == profiles.forwardWithTape || profiles.primal == profiles.backward ||
                    profiles.forwardWithTape == profiles.backward ||
                    !profileStage(profiles.forwardWithTape, "forward_with_tape") ||
                    !profileStage(profiles.backward, "backward")) {
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

VernonStatus fillTextureConstraintView(const Parameter &source, VernonPipelineTextureConstraintView &destination) {
    if (source.kind != "texture")
        return VERNON_STATUS_INVALID_ARGUMENT;
    const auto dimension = pipelineTextureDimension(source.dimension);
    const auto format = source.textureFormat.empty() ? std::optional<VernonTextureFormat>{}
                                                     : pipelineTextureFormat(source.textureFormat);
    if (!dimension || (!source.textureFormat.empty() && !format))
        return VERNON_STATUS_PARSE_ERROR;
    destination.dimension = *dimension;
    destination.has_format_constraint = format.has_value() ? 1u : 0u;
    destination.format = format.value_or(static_cast<VernonTextureFormat>(0));
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
        if (bundle->autodiff) {
            const auto profiles = std::find_if(bundle->autodiff->variants.begin(), bundle->autodiff->variants.end(),
                                               [&](const AutodiffVariant &candidate) { return candidate.key == key; });
            if (profiles == bundle->autodiff->variants.end()) {
                fail(bundle->context, "autodiff profiles have no exact feature variant");
                return nullptr;
            }
            if (!vernon::runtime::ad::resolvePipelineAutodiff(*bundle, *profiles, *pipeline)) {
                const std::string message = invocationDiagnostic(*bundle->context);
                pipeline.reset();
                fail(bundle->context, message);
                return nullptr;
            }
        }
        for (Parameter &parameter : pipeline->variant.parameters) {
            if (parameter.valueLayout)
                rebuildValueLayoutPathViews(*parameter.valueLayout);
            rebuildValueLayoutPathViews(parameter.elementLayout);
        }
        if (!resolveBackendPipeline(*bundle, *found, *pipeline))
            return nullptr;
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
    const ValueLayout &layout = parameter->valueLayout ? *parameter->valueLayout : parameter->elementLayout;
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

VernonStatus vernonRuntimeLoadedPipelineGetTextureConstraintByParameterIndex(
    const VernonLoadedPipeline *pipeline, size_t parameterIndex, VernonPipelineTextureConstraintView *constraint) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !constraint || constraint->struct_size < sizeof(VernonPipelineTextureConstraintView) ||
        parameterIndex >= pipeline->variant.parameters.size())
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillTextureConstraintView(pipeline->variant.parameters[parameterIndex], *constraint);
}

VernonStatus vernonRuntimeLoadedPipelineFindTextureConstraint(const VernonLoadedPipeline *pipeline,
                                                              VernonStringView parameterName,
                                                              VernonPipelineTextureConstraintView *constraint) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !constraint || constraint->struct_size < sizeof(VernonPipelineTextureConstraintView) ||
        (parameterName.size && !parameterName.data))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const auto found =
        std::find_if(pipeline->variant.parameters.begin(), pipeline->variant.parameters.end(),
                     [&](const Parameter &parameter) { return stringViewEquals(parameterName, parameter.name); });
    if (found == pipeline->variant.parameters.end())
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillTextureConstraintView(*found, *constraint);
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
    destroyBackendPipeline(*pipeline);
    --pipeline->context->livePipelines;
    delete pipeline;
}

VernonStatus vernonRuntimePipelineInvoke(VernonLoadedPipeline *pipeline, const VernonPipelineInvocation *invocation) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !invocation || invocation->struct_size < sizeof(VernonPipelineInvocation) ||
        invocation->abi_version != VERNON_PIPELINE_VERSION || (invocation->argument_count && !invocation->arguments))
        return fail(pipeline ? pipeline->context : nullptr, "invalid pipeline invocation");
    auto encode = [&](const VernonPipelineInvocation &encoded) {
        if (!pipeline->variant.compute.empty()) {
            PlannedComputeLaunch plan;
            std::string planningError;
            if (!planComputeInvocation(pipeline->variant, encoded, plan, planningError))
                return fail(pipeline->context, planningError);
            return invokeBackendComputePipeline(*pipeline, plan);
        }

        PlannedGraphicsInvocation plan;
        std::string planningError;
        if (!planGraphicsInvocation(pipeline->variant, encoded, plan, planningError))
            return fail(pipeline->context, planningError);
        return invokeBackendPipeline(*pipeline, encoded, plan);
    };
    if (invocation->command_encoder.value != 0 || pipeline->context->rhiDevice.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        return encode(*invocation);

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
        status = encode(encoded);
    if (rendering) {
        const VernonRhiStatus endStatus = vernonRhiCommandEncoderEndRendering(pipeline->context->rhiDevice, native);
        if (status == VERNON_STATUS_OK && endStatus != VERNON_RHI_STATUS_OK)
            status = fail(pipeline->context, "failed to end immediate rendering");
    }
    if (status == VERNON_STATUS_OK &&
        vernonRhiCommandEncoderFinish(pipeline->context->rhiDevice, native) != VERNON_RHI_STATUS_OK)
        status = fail(pipeline->context, "failed to finish the immediate command encoder");
    if (status == VERNON_STATUS_OK &&
        vernonRhiDeviceSubmit(pipeline->context->rhiDevice, native) != VERNON_RHI_STATUS_OK) {
        const VernonStringView detail = vernonRhiDeviceGetLastError(pipeline->context->rhiDevice);
        std::string error = "failed to submit the immediate command encoder";
        if (detail.data && detail.size)
            error.append(": ").append(detail.data, detail.size);
        status = fail(pipeline->context, std::move(error));
    }
    vernonRhiDeviceDestroyCommandEncoder(pipeline->context->rhiDevice, native);
    return status;
}

VernonStatus vernonRuntimePipelineEncode(VernonRuntimeProviderObject encoder, VernonLoadedPipeline *pipeline,
                                         const VernonPipelineInvocation *invocation) {
    if (!invocation)
        return fail(pipeline ? pipeline->context : nullptr, "invalid pipeline invocation");
    VernonPipelineInvocation encoded = *invocation;
    encoded.command_encoder = encoder;
    return vernonRuntimePipelineInvoke(pipeline, &encoded);
}

VernonStatus vernonRuntimeSynchronize(VernonRuntimeContext *context) {
    if (!context)
        return VERNON_STATUS_INVALID_ARGUMENT;
    RuntimeDiagnosticScope diagnostic(context);
    return synchronizeBackend(*context);
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

VernonStatus vernonRuntimeReferenceRhiImage(VernonRuntimeContext *context, VernonRhiImage image,
                                            VernonRuntimeProviderResourceReference *output) {
    if (!context)
        return VERNON_STATUS_INVALID_ARGUMENT;
    RuntimeDiagnosticScope diagnostic(context);
    if (!output)
        return fail(context, "invalid RHI image reference");
    return referenceBackendRhiImage(*context, image, *output);
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
