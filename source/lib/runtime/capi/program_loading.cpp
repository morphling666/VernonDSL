#include "VernonFormats.hpp"
#include "VernonRuntime.h"
#include "runtime/capi/boundary.h"
#include "runtime/capi/program_state.h"
#include "runtime/program_execution/program_boundary_contract.h"
#include "runtime/program_execution_backend.h"
#include "runtime/program_graph_linker.h"
#include "runtime/stage_artifact.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace vernon::runtime;
using namespace vernon::runtime::capi;

namespace {

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
        for (const nlohmann::json &assignment : variant["key"]) {
            if (!assignment.is_object() || assignment.size() != 2 || !assignment.contains("name") ||
                !assignment["name"].is_string() || assignment["name"].get_ref<const std::string &>().empty() ||
                !assignment.contains("value") || !assignment["value"].is_object() || assignment["value"].size() != 2 ||
                !assignment["value"].contains("tag") || !assignment["value"]["tag"].is_string() ||
                !assignment["value"].contains("value")) {
                fail(&context, "Program bundle specialization assignment is invalid", VERNON_STATUS_PARSE_ERROR);
                return false;
            }
            vernon::runtime::ProgramSpecialization specialization;
            specialization.name = assignment["name"].get<std::string>();
            const std::string tag = assignment["value"]["tag"].get<std::string>();
            const nlohmann::json &value = assignment["value"]["value"];
            if (tag == "bool" && value.is_boolean()) {
                specialization.kind = vernon::runtime::ProgramSpecializationKind::Bool;
                specialization.value = value.get<bool>();
            } else if (tag == "i32" && value.is_number_integer()) {
                const int64_t integer = value.get<int64_t>();
                if (integer < std::numeric_limits<int32_t>::min() || integer > std::numeric_limits<int32_t>::max()) {
                    fail(&context, "Program bundle i32 specialization is out of range", VERNON_STATUS_PARSE_ERROR);
                    return false;
                }
                specialization.kind = vernon::runtime::ProgramSpecializationKind::I32;
                specialization.value = static_cast<int32_t>(integer);
            } else if (tag == "u32" && value.is_number_unsigned()) {
                const uint64_t integer = value.get<uint64_t>();
                if (integer > std::numeric_limits<uint32_t>::max()) {
                    fail(&context, "Program bundle u32 specialization is out of range", VERNON_STATUS_PARSE_ERROR);
                    return false;
                }
                specialization.kind = vernon::runtime::ProgramSpecializationKind::U32;
                specialization.value = static_cast<uint32_t>(integer);
            } else if (tag == "f32" && value.is_number_float()) {
                const float scalar = value.get<float>();
                if (!std::isfinite(scalar)) {
                    fail(&context, "Program bundle f32 specialization must be finite", VERNON_STATUS_PARSE_ERROR);
                    return false;
                }
                specialization.kind = vernon::runtime::ProgramSpecializationKind::F32;
                specialization.value = scalar == 0.0f ? 0.0f : scalar;
            } else if (tag == "f64" && value.is_number_float()) {
                const double scalar = value.get<double>();
                if (!std::isfinite(scalar)) {
                    fail(&context, "Program bundle f64 specialization must be finite", VERNON_STATUS_PARSE_ERROR);
                    return false;
                }
                specialization.kind = vernon::runtime::ProgramSpecializationKind::F64;
                specialization.value = scalar == 0.0 ? 0.0 : scalar;
            } else if (tag == "type" && value.is_string() && !value.get_ref<const std::string &>().empty()) {
                specialization.kind = vernon::runtime::ProgramSpecializationKind::Type;
                specialization.value = value.get<std::string>();
            } else if (const vernon::formats::Format *format =
                           tag == "format" && value.is_string()
                               ? vernon::formats::find(value.get_ref<const std::string &>())
                               : nullptr;
                       format && format->aspects() == VERNON_RHI_IMAGE_ASPECT_COLOR) {
                specialization.kind = vernon::runtime::ProgramSpecializationKind::Format;
                specialization.value = format->texture;
            } else {
                fail(&context, "Program bundle specialization value does not match its tag", VERNON_STATUS_PARSE_ERROR);
                return false;
            }
            parsed.key.push_back(std::move(specialization));
        }
        if (!std::is_sorted(parsed.key.begin(), parsed.key.end()) ||
            std::adjacent_find(parsed.key.begin(), parsed.key.end(), [](const auto &left, const auto &right) {
                return left.name == right.name;
            }) != parsed.key.end()) {
            fail(&context, "Program bundle specialization key is not canonical", VERNON_STATUS_PARSE_ERROR);
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
            fail(&context, "Program bundle contains duplicate specialization variants", VERNON_STATUS_PARSE_ERROR);
            return false;
        }
        vernon::runtime::program::Diagnostic diagnostic;
        auto program = vernon::runtime::program::parse(variant["program"], diagnostic);
        if (program.isErr()) {
            fail(&context,
                 diagnostic.code + (diagnostic.path.empty() ? ": " : " at " + diagnostic.path + ": ") +
                     diagnostic.message,
                 VERNON_STATUS_PARSE_ERROR);
            return false;
        }
        parsed.program = std::move(program).value();
        auto artifacts = vernon::runtime::program::parseArtifactSystem(document["target"], document["blobs"],
                                                                       variant["artifact_system"], diagnostic);
        if (artifacts.isErr()) {
            fail(&context,
                 diagnostic.code + (diagnostic.path.empty() ? ": " : " at " + diagnostic.path + ": ") +
                     diagnostic.message,
                 VERNON_STATUS_PARSE_ERROR);
            return false;
        }
        parsed.artifactSystem = std::move(artifacts).value();
        result.push_back(std::move(parsed));
    }
    if (result.empty()) {
        fail(&context, "Program bundle declares no variants", VERNON_STATUS_PARSE_ERROR);
        return false;
    }
    return true;
}

vernon::Result<const vernon::runtime::ProgramVariantDeployment *, vernon::RuntimeError>
selectProgramDeployment(VernonRuntimeContext &context,
                        const std::vector<vernon::runtime::ProgramVariantDeployment> &variants,
                        const VernonProgramVariantSelector *selector) {
    using Result = vernon::Result<const vernon::runtime::ProgramVariantDeployment *, vernon::RuntimeError>;
    std::vector<vernon::runtime::ProgramSpecialization> requested;
    if (selector) {
        if (selector->struct_size < sizeof(*selector) ||
            (selector->specialization_count && !selector->specializations)) {
            fail(&context, "Program variant selector is invalid");
            return Result{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                                           {"Program variant selector is invalid", 0, 0}})};
        }
        requested.reserve(selector->specialization_count);
        for (size_t index = 0; index < selector->specialization_count; ++index) {
            const VernonProgramSpecialization &source = selector->specializations[index];
            if (source.struct_size < sizeof(source) || !source.name.data || !source.name.size) {
                fail(&context, "Program specialization is invalid");
                return Result{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                                               {"Program specialization is invalid", 0, 0}})};
            }
            vernon::runtime::ProgramSpecialization destination;
            destination.name.assign(source.name.data, source.name.size);
            switch (source.kind) {
            case VERNON_PROGRAM_SPECIALIZATION_BOOL:
                if (source.value.boolean_value > 1) {
                    fail(&context, "Program bool specialization must be zero or one");
                    return Result{
                        vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                                         {"Program bool specialization must be zero or one", 0, 0}})};
                }
                destination.kind = vernon::runtime::ProgramSpecializationKind::Bool;
                destination.value = source.value.boolean_value != 0;
                break;
            case VERNON_PROGRAM_SPECIALIZATION_I32:
                destination.kind = vernon::runtime::ProgramSpecializationKind::I32;
                destination.value = source.value.i32_value;
                break;
            case VERNON_PROGRAM_SPECIALIZATION_U32:
                destination.kind = vernon::runtime::ProgramSpecializationKind::U32;
                destination.value = source.value.u32_value;
                break;
            case VERNON_PROGRAM_SPECIALIZATION_F32:
                if (!std::isfinite(source.value.f32_value)) {
                    fail(&context, "Program f32 specialization must be finite");
                    return Result{
                        vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                                         {"Program f32 specialization must be finite", 0, 0}})};
                }
                destination.kind = vernon::runtime::ProgramSpecializationKind::F32;
                destination.value = source.value.f32_value == 0.0f ? 0.0f : source.value.f32_value;
                break;
            case VERNON_PROGRAM_SPECIALIZATION_F64:
                if (!std::isfinite(source.value.f64_value)) {
                    fail(&context, "Program f64 specialization must be finite");
                    return Result{
                        vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                                         {"Program f64 specialization must be finite", 0, 0}})};
                }
                destination.kind = vernon::runtime::ProgramSpecializationKind::F64;
                destination.value = source.value.f64_value == 0.0 ? 0.0 : source.value.f64_value;
                break;
            case VERNON_PROGRAM_SPECIALIZATION_TYPE:
                if (!source.value.type_value.data || !source.value.type_value.size) {
                    fail(&context, "Program type specialization requires a type identity");
                    return Result{vernon::err(
                        vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                             {"Program type specialization requires a type identity", 0, 0}})};
                }
                destination.kind = vernon::runtime::ProgramSpecializationKind::Type;
                destination.value = std::string(source.value.type_value.data, source.value.type_value.size);
                break;
            case VERNON_PROGRAM_SPECIALIZATION_FORMAT:
                if (vernon::formats::aspects(source.value.format_value) != VERNON_RHI_IMAGE_ASPECT_COLOR) {
                    fail(&context, "Program format specialization requires a color format");
                    return Result{vernon::err(
                        vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                             {"Program format specialization requires a color format", 0, 0}})};
                }
                destination.kind = vernon::runtime::ProgramSpecializationKind::Format;
                destination.value = source.value.format_value;
                break;
            default:
                fail(&context, "Program specialization kind is invalid");
                return Result{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                                               {"Program specialization kind is invalid", 0, 0}})};
            }
            requested.push_back(std::move(destination));
        }
    }
    std::sort(requested.begin(), requested.end());
    if (std::adjacent_find(requested.begin(), requested.end(), [](const auto &left, const auto &right) {
            return left.name == right.name;
        }) != requested.end()) {
        fail(&context, "Program variant selector contains duplicate specialization names");
        return Result{vernon::err(
            vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                 {"Program variant selector contains duplicate specialization names", 0, 0}})};
    }
    const auto found =
        std::find_if(variants.begin(), variants.end(), [&](const vernon::runtime::ProgramVariantDeployment &variant) {
            return variant.key == requested;
        });
    if (found == variants.end()) {
        fail(&context, "Program bundle has no matching variant", VERNON_STATUS_PARSE_ERROR);
        return Result{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::ParseFailure,
                                                       {"Program bundle has no matching variant", 0, 0}})};
    }
    return Result{vernon::ok(&*found)};
}

vernon::RuntimeErrorCode programLoadErrorCode(const vernon::runtime::program::ProgramLoadError &error) noexcept {
    if (error.code == "PROGRAM_RUNTIME_LIFECYCLE")
        return error.message.find("cannot publish") != std::string::npos ? vernon::RuntimeErrorCode::LifecycleFailure
                                                                         : vernon::RuntimeErrorCode::ResourceExhausted;
    if (error.code.find("UNSUPPORTED") != std::string::npos || error.code == "PROGRAM_ARTIFACT_TARGET")
        return vernon::RuntimeErrorCode::Unsupported;
    return vernon::RuntimeErrorCode::VerificationFailure;
}

vernon::Result<VernonProgramExecutable *, vernon::RuntimeError>
resolveProgramDeployment(VernonRuntimeContext &context, const vernon::runtime::ProgramVariantDeployment &variant,
                         const std::filesystem::path &bundleRoot) {
    using Result = vernon::Result<VernonProgramExecutable *, vernon::RuntimeError>;
    auto loaded = vernon::runtime::program::loadBackendProgramPipeline(context, variant.program, variant.artifactSystem,
                                                                       bundleRoot);
    if (loaded.isErr()) {
        const vernon::RuntimeErrorCode code = programLoadErrorCode(loaded.error());
        fail(&context, vernon::runtime::program::renderProgramLoadError(loaded.error()),
             vernon::toVernonStatus(vernon::RuntimeError{code, {}}));
        return Result{vernon::err(vernon::RuntimeError{code, {"Program executable resolution", 0, 0}})};
    }
    return Result{vernon::ok(std::move(loaded).value().release())};
}

} // namespace

VernonStatus vernonRuntimeProgramBundleInspectTarget(const void *bundleData, size_t bundleSize,
                                                     VernonRuntimeBackend *target) {
    return runtimeStatusBoundary(nullptr, [&] {
        if (!bundleData || !bundleSize || !target)
            return VERNON_STATUS_INVALID_ARGUMENT;
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
        if (actual != members || root.value("compiler_contract_version", 0) != VERNON_COMPILER_CONTRACT_VERSION ||
            root.value("program_version", 0) != VERNON_PROGRAM_VERSION || root.value("type", "") != "program" ||
            !root.contains("id") || !root["id"].is_string() || root["id"].get_ref<const std::string &>().empty() ||
            !root.contains("target") || !root["target"].is_object() || root["target"].size() != 2 ||
            !root["target"].contains("kind") || !root["target"]["kind"].is_string() ||
            !root["target"].contains("options") || !root["target"]["options"].is_object() || !root.contains("blobs") ||
            !root["blobs"].is_object() || !root.contains("variants") || !root["variants"].is_array() ||
            validateProgramBundleHash(root, true).isErr())
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
    });
}

namespace {
using ProgramBundleHandleResult = vernon::Result<VernonProgramBundle *, vernon::RuntimeError>;

ProgramBundleHandleResult loadProgramBundleOperation(VernonRuntimeContext *context, const void *bundleData,
                                                     size_t bundleSize, const VernonProgramBundleLoadOptions *options) {
    if (!context)
        return ProgramBundleHandleResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"invalid Program bundle load invocation", 0, 0}})};
    auto contextPin = context->operations.tryPin();
    if (contextPin.isErr()) {
        fail(context, "runtime context is closing");
        return ProgramBundleHandleResult{vernon::err(vernon::toRuntimeError(contextPin.error()))};
    }
    auto child = RuntimeChildLifecycle::reserve(context->owner);
    if (child.isErr())
        return ProgramBundleHandleResult{vernon::err(std::move(child).error())};
    if (context->backend != VERNON_RUNTIME_CPU && context->backend != VERNON_RUNTIME_OPENGL &&
        context->backend != VERNON_RUNTIME_OPENGL_ES && context->backend != VERNON_RUNTIME_VULKAN &&
        context->backend != VERNON_RUNTIME_CUDA && context->backend != VERNON_RUNTIME_DIRECTX12 &&
        context->backend != VERNON_RUNTIME_METAL) {
        fail(context, "Program bundle backend is unsupported", VERNON_STATUS_UNSUPPORTED_TARGET);
        return ProgramBundleHandleResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::Unsupported, {"Program bundle backend is unsupported", 0, 0}})};
    }
    if (!bundleData || !bundleSize || (options && options->struct_size < sizeof(VernonProgramBundleLoadOptions))) {
        invocationDiagnostic(*context) = "invalid Program bundle load invocation";
        return ProgramBundleHandleResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"invalid Program bundle load invocation", 0, 0}})};
    }
    std::optional<std::filesystem::path> bundleDirectory;
    if (options && options->bundle_directory && options->bundle_directory[0] != '\0')
        bundleDirectory = std::filesystem::u8path(options->bundle_directory);
    const nlohmann::json root = nlohmann::json::parse(
        static_cast<const char *>(bundleData), static_cast<const char *>(bundleData) + bundleSize, nullptr, false);
    if (root.is_discarded()) {
        fail(context, "invalid Program bundle: malformed JSON", VERNON_STATUS_PARSE_ERROR);
        return ProgramBundleHandleResult{
            vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::ParseFailure, {"Program bundle parse", 0, 0}})};
    }
    const char *expectedTarget = context->backend == VERNON_RUNTIME_CPU         ? "cpu"
                                 : context->backend == VERNON_RUNTIME_CUDA      ? "cuda"
                                 : context->backend == VERNON_RUNTIME_VULKAN    ? "vulkan"
                                 : context->backend == VERNON_RUNTIME_DIRECTX12 ? "directx"
                                 : context->backend == VERNON_RUNTIME_METAL
                                     ? "metal"
                                     : (context->backend == VERNON_RUNTIME_OPENGL_ES ? "opengles" : "opengl");
    static const std::set<std::string> canonicalMembers{
        "compiler_contract_version", "program_version", "type", "id", "target", "blobs", "variants", "content_hash"};
    std::set<std::string> members;
    if (root.is_object())
        for (const auto &[name, unused] : root.items())
            members.insert(name);
    if (members != canonicalMembers || root.value("compiler_contract_version", 0) != VERNON_COMPILER_CONTRACT_VERSION ||
        root.value("program_version", 0) != VERNON_PROGRAM_VERSION || root.value("type", "") != "program" ||
        !root["id"].is_string() || root["id"].get_ref<const std::string &>().empty() ||
        !root["content_hash"].is_string() || root["content_hash"].get_ref<const std::string &>().empty() ||
        !root["target"].is_object() || root["target"].size() != 2 || !root["target"].contains("kind") ||
        !root["target"]["kind"].is_string() || !root["target"].contains("options") ||
        !root["target"]["options"].is_object() || !root["blobs"].is_object() || !root["variants"].is_array()) {
        fail(context, "unsupported or invalid Program bundle", VERNON_STATUS_PARSE_ERROR);
        return ProgramBundleHandleResult{
            vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::ParseFailure, {"Program bundle parse", 0, 0}})};
    }
    if (root["target"]["kind"].get_ref<const std::string &>() != expectedTarget) {
        fail(context, "Program bundle target does not match the Runtime backend", VERNON_STATUS_UNSUPPORTED_TARGET);
        return ProgramBundleHandleResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::Unsupported, {"Program bundle target is unsupported", 0, 0}})};
    }
    auto bundle = std::make_unique<VernonProgramBundle>(std::move(child).value());
    bundle->context = context;
    bundle->id = root["id"].get<std::string>();
    bundle->contentHash = root["content_hash"].get<std::string>();
    if (bundleDirectory)
        bundle->bundleRoot = *bundleDirectory;
    if (!parseProgramDeployments(*context, root, bundle->deployments))
        return ProgramBundleHandleResult{
            vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::ParseFailure, {"Program bundle parse", 0, 0}})};
    for (vernon::runtime::ProgramVariantDeployment &deployment : bundle->deployments)
        for (auto &[unused, blob] : deployment.artifactSystem.blobs)
            blob.bundleRoot = bundle->bundleRoot;
    auto bundleHash = validateProgramBundleHash(root, true);
    if (bundleHash.isErr()) {
        fail(context, renderStageArtifactError(std::move(bundleHash).error()), VERNON_STATUS_PARSE_ERROR);
        return ProgramBundleHandleResult{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::VerificationFailure,
                                                                          {"Program bundle verification", 0, 0}})};
    }
    auto published = bundle->lifecycle.publish();
    if (published.isErr()) {
        fail(context, "cannot publish Program bundle", vernon::toVernonStatus(published.error()));
        return ProgramBundleHandleResult{vernon::err(std::move(published).error())};
    }
    return ProgramBundleHandleResult{vernon::ok(bundle.release())};
}
} // namespace

VernonRuntimeOperationStatus
vernonRuntimeLoadProgramBundleWithOptionsResult(VernonRuntimeContext *context, const void *bundleData,
                                                size_t bundleSize, const VernonProgramBundleLoadOptions *options,
                                                VernonProgramBundle **output) {
    return runtimeHandleResultBoundary(
        context, output, [&] { return loadProgramBundleOperation(context, bundleData, bundleSize, options); });
}

VernonProgramBundle *vernonRuntimeLoadProgramBundleWithOptions(VernonRuntimeContext *context, const void *bundleData,
                                                               size_t bundleSize,
                                                               const VernonProgramBundleLoadOptions *options) {
    VernonProgramBundle *output = nullptr;
    (void)vernonRuntimeLoadProgramBundleWithOptionsResult(context, bundleData, bundleSize, options, &output);
    return output;
}

VernonStringView vernonRuntimeProgramBundleGetId(const VernonProgramBundle *bundle) {
    return runtimeValueBoundary(bundle ? bundle->context : nullptr, VernonStringView{nullptr, 0}, [&] {
        if (!bundle)
            return VernonStringView{nullptr, 0};
        auto pin = bundle->lifecycle.pin();
        return pin.isOk() ? VernonStringView{bundle->id.data(), bundle->id.size()} : VernonStringView{nullptr, 0};
    });
}

namespace {

const program::BoundarySlot *programGraphBoundary(const VernonProgramGraph &graph,
                                                  const VernonProgramNodeBindingToken *token) {
    if (!token || token->struct_size < sizeof(*token) || token->graph_id != graph.id ||
        token->node >= graph.nodes.size() || token->kind > VERNON_PROGRAM_SAMPLER)
        return nullptr;
    const RuntimeProgramGraphNode &node = graph.nodes[token->node];
    const auto &boundaries = node.deployment.program.abi.boundarySlots;
    return token->local_slot < boundaries.size() && boundaries[token->local_slot].id == token->local_slot
               ? &boundaries[token->local_slot]
               : nullptr;
}

} // namespace

void vernonRuntimeProgramBundleDestroy(VernonProgramBundle *bundle) {
    runtimeVoidBoundary(bundle ? bundle->context : nullptr, [&] {
        if (!bundle)
            return;
        auto destruction = bundle->lifecycle.beginDestroy();
        if (destruction.isErr()) {
            fail(bundle->context, "Program bundle has an active operation",
                 vernon::toVernonStatus(destruction.error()));
            return;
        }
        if (destruction.value().commit().isErr())
            vernon::resultContractViolation();
        delete bundle;
    });
}

namespace {
using ProgramGraphHandleResult = vernon::Result<VernonProgramGraph *, vernon::RuntimeError>;

ProgramGraphHandleResult createProgramGraphOperation(VernonRuntimeContext *context) {
    if (!context)
        return ProgramGraphHandleResult{vernon::err(
            vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument, {"invalid ProgramGraph creation", 0, 0}})};
    auto contextPin = context->operations.tryPin();
    if (contextPin.isErr()) {
        fail(context, "runtime context is closing", vernon::toVernonStatus(contextPin.error()));
        return ProgramGraphHandleResult{vernon::err(vernon::toRuntimeError(contextPin.error()))};
    }
    auto child = RuntimeChildLifecycle::reserve(context->owner);
    if (child.isErr()) {
        fail(context, "runtime context cannot admit a ProgramGraph", vernon::toVernonStatus(child.error()));
        return ProgramGraphHandleResult{vernon::err(std::move(child).error())};
    }
    auto graph = std::make_unique<VernonProgramGraph>(std::move(child).value());
    graph->context = context;
    graph->id = context->nextProgramGraphId++;
    auto published = graph->lifecycle.publish();
    if (published.isErr()) {
        fail(context, "cannot publish ProgramGraph", vernon::toVernonStatus(published.error()));
        return ProgramGraphHandleResult{vernon::err(std::move(published).error())};
    }
    return ProgramGraphHandleResult{vernon::ok(graph.release())};
}
} // namespace

VernonRuntimeOperationStatus vernonRuntimeProgramGraphCreateResult(VernonRuntimeContext *context,
                                                                   VernonProgramGraph **output) {
    return runtimeHandleResultBoundary(context, output, [&] { return createProgramGraphOperation(context); });
}

VernonProgramGraph *vernonRuntimeProgramGraphCreate(VernonRuntimeContext *context) {
    VernonProgramGraph *output = nullptr;
    (void)vernonRuntimeProgramGraphCreateResult(context, &output);
    return output;
}

void vernonRuntimeProgramGraphDestroy(VernonProgramGraph *graph) {
    runtimeVoidBoundary(graph ? graph->context : nullptr, [&] {
        if (!graph)
            return;
        auto destruction = graph->lifecycle.beginDestroy();
        if (destruction.isErr()) {
            fail(graph->context, "ProgramGraph has an active operation", vernon::toVernonStatus(destruction.error()));
            return;
        }
        if (destruction.value().commit().isErr())
            vernon::resultContractViolation();
        delete graph;
    });
}

VernonStatus vernonRuntimeProgramGraphAddProgram(VernonProgramGraph *graph, const VernonProgramBundle *bundle,
                                                 const VernonProgramVariantSelector *selector, VernonStringView label,
                                                 VernonProgramNodeId *node) {
    return runtimeStatusBoundary(graph ? graph->context : nullptr, [&] {
        if (!graph || !bundle || !node || (label.size && !label.data))
            return fail(graph ? graph->context : nullptr, "invalid ProgramGraph node");
        auto graphPin = graph->lifecycle.pin();
        auto bundlePin = bundle->lifecycle.pin();
        if (graphPin.isErr() || bundlePin.isErr())
            return fail(graph->context, "ProgramGraph or Program bundle is closing");
        if (bundle->context != graph->context || graph->nodes.size() >= UINT32_MAX)
            return fail(graph->context, "invalid ProgramGraph node");
        auto deploymentResult = selectProgramDeployment(*graph->context, bundle->deployments, selector);
        if (deploymentResult.isErr())
            return vernon::toVernonStatus(std::move(deploymentResult).error());
        const ProgramVariantDeployment *deployment = std::move(deploymentResult).value();
        RuntimeProgramGraphNode source;
        source.id = static_cast<uint32_t>(graph->nodes.size());
        source.bundleId = bundle->id;
        source.label = label.size ? std::string(label.data, label.size) : bundle->id;
        source.contentHash = bundle->contentHash;
        source.deployment = *deployment;
        source.bundleRoot = bundle->bundleRoot;
        graph->nodes.push_back(std::move(source));
        *node = graph->nodes.back().id;
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramGraphFindBoundary(const VernonProgramGraph *graph, VernonProgramNodeId node,
                                                   VernonProgramBoundaryRole role, VernonStringView name,
                                                   VernonProgramNodeBindingToken *token) {
    return runtimeStatusBoundary(graph ? graph->context : nullptr, [&] {
        const auto expectedRole = reflectedBoundaryRole(role);
        if (!graph)
            return fail(nullptr, "invalid ProgramGraph boundary lookup");
        auto graphPin = graph->lifecycle.pin();
        if (graphPin.isErr())
            return fail(graph->context, "ProgramGraph is closing");
        if (!expectedRole || !token || token->struct_size < sizeof(*token) || (name.size && !name.data) ||
            node >= graph->nodes.size())
            return fail(graph->context, "invalid ProgramGraph boundary lookup");
        const program::Program &program = graph->nodes[node].deployment.program;
        const auto found = std::find_if(program.abi.boundarySlots.begin(), program.abi.boundarySlots.end(),
                                        [&](const program::BoundarySlot &slot) {
                                            return slot.role == *expectedRole && stringViewEquals(name, slot.path);
                                        });
        if (found == program.abi.boundarySlots.end())
            return fail(graph->context, "ProgramGraph node boundary was not found");
        token->graph_id = graph->id;
        token->node = node;
        token->local_slot = found->id;
        token->kind = vernon::runtime::program_execution::boundaryArgumentKind(found->category);
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramGraphCreateValue(VernonProgramGraph *graph,
                                                  const VernonProgramNodeBindingToken *source,
                                                  VernonProgramGraphValue *value) {
    return runtimeStatusBoundary(graph ? graph->context : nullptr, [&] {
        if (!graph)
            return fail(graph ? graph->context : nullptr, "invalid ProgramGraph Value source");
        auto graphPin = graph->lifecycle.pin();
        if (graphPin.isErr())
            return fail(graph->context, "ProgramGraph is closing");
        const program::BoundarySlot *boundary = programGraphBoundary(*graph, source);
        if (!value || value->struct_size < sizeof(*value) || !boundary ||
            boundary->direction != program::BoundaryDirection::Output)
            return fail(graph->context, "invalid ProgramGraph Value source");
        if (graph->values.size() >= UINT32_MAX)
            return fail(graph->context, "ProgramGraph has too many Values");
        value->graph_id = graph->id;
        value->id = static_cast<uint32_t>(graph->values.size());
        value->kind = source->kind;
        graph->values.push_back({{source->node, source->local_slot}, {}});
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramGraphConnectValue(VernonProgramGraph *graph, const VernonProgramGraphValue *value,
                                                   const VernonProgramNodeBindingToken *destination) {
    return runtimeStatusBoundary(graph ? graph->context : nullptr, [&] {
        if (!graph)
            return fail(graph ? graph->context : nullptr, "invalid ProgramGraph Value destination");
        auto graphPin = graph->lifecycle.pin();
        if (graphPin.isErr())
            return fail(graph->context, "ProgramGraph is closing");
        const program::BoundarySlot *boundary = programGraphBoundary(*graph, destination);
        if (!value || value->struct_size < sizeof(*value) || value->graph_id != graph->id ||
            value->id >= graph->values.size() || !boundary ||
            boundary->direction != program::BoundaryDirection::Input || value->kind != destination->kind)
            return fail(graph->context, "invalid ProgramGraph Value destination");
        graph->values[value->id].destinations.push_back({destination->node, destination->local_slot});
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramGraphCreateStorage(VernonProgramGraph *graph,
                                                    const VernonProgramNodeBindingToken *firstVersion,
                                                    VernonProgramGraphStorage *storage) {
    return runtimeStatusBoundary(graph ? graph->context : nullptr, [&] {
        if (!graph)
            return fail(graph ? graph->context : nullptr, "invalid ProgramGraph Storage source");
        auto graphPin = graph->lifecycle.pin();
        if (graphPin.isErr())
            return fail(graph->context, "ProgramGraph is closing");
        const program::BoundarySlot *boundary = programGraphBoundary(*graph, firstVersion);
        if (!storage || storage->struct_size < sizeof(*storage) || !boundary ||
            boundary->direction != program::BoundaryDirection::Output ||
            boundary->aliasOwner.kind != program::ProgramOwnerKind::Storage)
            return fail(graph->context, "invalid ProgramGraph Storage source");
        if (graph->storages.size() >= UINT32_MAX)
            return fail(graph->context, "ProgramGraph has too many Storages");
        storage->graph_id = graph->id;
        storage->id = static_cast<uint32_t>(graph->storages.size());
        storage->kind = firstVersion->kind;
        graph->storages.push_back({firstVersion->kind, {{firstVersion->node, firstVersion->local_slot}}});
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramGraphAppendStorage(VernonProgramGraph *graph, const VernonProgramGraphStorage *storage,
                                                    const VernonProgramNodeBindingToken *nextVersion) {
    return runtimeStatusBoundary(graph ? graph->context : nullptr, [&] {
        if (!graph)
            return fail(graph ? graph->context : nullptr, "invalid ProgramGraph Storage version");
        auto graphPin = graph->lifecycle.pin();
        if (graphPin.isErr())
            return fail(graph->context, "ProgramGraph is closing");
        const program::BoundarySlot *boundary = programGraphBoundary(*graph, nextVersion);
        if (!storage || storage->struct_size < sizeof(*storage) || storage->graph_id != graph->id ||
            storage->id >= graph->storages.size() || !boundary ||
            boundary->direction != program::BoundaryDirection::Output ||
            boundary->aliasOwner.kind != program::ProgramOwnerKind::Storage || storage->kind != nextVersion->kind)
            return fail(graph->context, "invalid ProgramGraph Storage version");
        RuntimeProgramGraphStorage &target = graph->storages[storage->id];
        if (target.versions.back().node >= nextVersion->node)
            return fail(graph->context, "ProgramGraph Storage versions must follow node order");
        target.versions.push_back({nextVersion->node, nextVersion->local_slot});
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramGraphExportBoundary(VernonProgramGraph *graph,
                                                     const VernonProgramNodeBindingToken *boundaryToken,
                                                     VernonStringView graphName) {
    return runtimeStatusBoundary(graph ? graph->context : nullptr, [&] {
        if (!graph || !boundaryToken || boundaryToken->struct_size < sizeof(*boundaryToken) ||
            boundaryToken->graph_id != graph->id || boundaryToken->node >= graph->nodes.size() || !graphName.data ||
            !graphName.size)
            return fail(graph ? graph->context : nullptr, "invalid ProgramGraph boundary export");
        auto graphPin = graph->lifecycle.pin();
        if (graphPin.isErr())
            return fail(graph->context, "ProgramGraph is closing");
        graph->exports.push_back(
            {{boundaryToken->node, boundaryToken->local_slot}, std::string(graphName.data, graphName.size)});
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramGraphExportValue(VernonProgramGraph *graph, const VernonProgramGraphValue *value,
                                                  VernonStringView graphName) {
    return runtimeStatusBoundary(graph ? graph->context : nullptr, [&] {
        if (!graph || !value || value->struct_size < sizeof(*value) || value->graph_id != graph->id ||
            value->id >= graph->values.size())
            return fail(graph ? graph->context : nullptr, "invalid ProgramGraph Value export");
        auto graphPin = graph->lifecycle.pin();
        if (graphPin.isErr())
            return fail(graph->context, "ProgramGraph is closing");
        const ProgramGraphBoundaryKey source = graph->values[value->id].source;
        const VernonProgramNodeBindingToken token{sizeof(VernonProgramNodeBindingToken), graph->id, source.node,
                                                  source.slot, value->kind};
        return vernonRuntimeProgramGraphExportBoundary(graph, &token, graphName);
    });
}

VernonStatus vernonRuntimeProgramGraphExportStorage(VernonProgramGraph *graph, const VernonProgramGraphStorage *storage,
                                                    VernonStringView graphName) {
    return runtimeStatusBoundary(graph ? graph->context : nullptr, [&] {
        if (!graph || !storage || storage->struct_size < sizeof(*storage) || storage->graph_id != graph->id ||
            storage->id >= graph->storages.size() || graph->storages[storage->id].versions.empty())
            return fail(graph ? graph->context : nullptr, "invalid ProgramGraph Storage export");
        auto graphPin = graph->lifecycle.pin();
        if (graphPin.isErr())
            return fail(graph->context, "ProgramGraph is closing");
        const ProgramGraphBoundaryKey final = graph->storages[storage->id].versions.back();
        const VernonProgramNodeBindingToken token{sizeof(VernonProgramNodeBindingToken), graph->id, final.node,
                                                  final.slot, storage->kind};
        return vernonRuntimeProgramGraphExportBoundary(graph, &token, graphName);
    });
}

namespace {
using ProgramGraphExecutableResult = vernon::Result<VernonProgramExecutable *, vernon::RuntimeError>;

ProgramGraphExecutableResult resolveProgramGraphOperation(VernonProgramGraph *graph) {
    if (!graph)
        return ProgramGraphExecutableResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"invalid ProgramGraph resolve invocation", 0, 0}})};
    auto graphPin = graph->lifecycle.pin();
    if (graphPin.isErr()) {
        fail(graph->context, "ProgramGraph is closing", vernon::toVernonStatus(graphPin.error()));
        return ProgramGraphExecutableResult{vernon::err(std::move(graphPin).error())};
    }
    if (graph->nodes.empty()) {
        fail(graph->context, "invalid ProgramGraph resolve invocation");
        return ProgramGraphExecutableResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"invalid ProgramGraph resolve invocation", 0, 0}})};
    }
    std::vector<ProgramGraphNodeSource> sources;
    sources.reserve(graph->nodes.size());
    for (RuntimeProgramGraphNode &node : graph->nodes) {
        sources.push_back({node.id, node.bundleId, node.label, node.contentHash, &node.deployment, node.bundleRoot});
    }
    std::vector<ProgramGraphConnection> connections;
    for (const RuntimeProgramGraphValue &value : graph->values)
        for (ProgramGraphBoundaryKey destination : value.destinations)
            connections.push_back({value.source, destination, false});
    for (const RuntimeProgramGraphStorage &storage : graph->storages)
        for (size_t index = 1; index < storage.versions.size(); ++index)
            connections.push_back({storage.versions[index - 1], storage.versions[index], true});
    std::vector<ProgramGraphBoundaryKey> retainedBoundaries;
    retainedBoundaries.reserve(graph->storages.size());
    for (const RuntimeProgramGraphStorage &storage : graph->storages)
        retainedBoundaries.push_back(storage.versions.back());
    LinkedProgramDeployment linked;
    program::Diagnostic linkDiagnostic;
    if (!linkProgramGraph(sources, connections, retainedBoundaries, graph->exports, linked, linkDiagnostic)) {
        fail(graph->context,
             linkDiagnostic.code + (linkDiagnostic.path.empty() ? ": " : " at " + linkDiagnostic.path + ": ") +
                 linkDiagnostic.message,
             VERNON_STATUS_PARSE_ERROR);
        return ProgramGraphExecutableResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::VerificationFailure, {"ProgramGraph link verification", 0, 0}})};
    }
    auto loaded = program::loadBackendProgramPipeline(*graph->context, linked.deployment.program,
                                                      linked.deployment.artifactSystem, {});
    if (loaded.isErr()) {
        const vernon::RuntimeErrorCode code = programLoadErrorCode(loaded.error());
        fail(graph->context, program::renderProgramLoadError(loaded.error()),
             vernon::toVernonStatus(vernon::RuntimeError{code, {}}));
        return ProgramGraphExecutableResult{
            vernon::err(vernon::RuntimeError{code, {"ProgramGraph executable resolution", 0, 0}})};
    }
    std::unique_ptr<VernonProgramExecutable, decltype(&vernonRuntimeProgramExecutableDestroy)> executable(
        std::move(loaded).value().release(), vernonRuntimeProgramExecutableDestroy);
    for (const RuntimeProgramGraphNode &node : graph->nodes) {
        if (!program::findGraph(node.deployment.program, "backward"))
            continue;
        auto childResult = program::loadBackendProgramPipeline(*graph->context, node.deployment.program,
                                                               node.deployment.artifactSystem, node.bundleRoot);
        if (childResult.isErr()) {
            const vernon::RuntimeErrorCode code = programLoadErrorCode(childResult.error());
            fail(graph->context, program::renderProgramLoadError(childResult.error()),
                 vernon::toVernonStatus(vernon::RuntimeError{code, {}}));
            return ProgramGraphExecutableResult{
                vernon::err(vernon::RuntimeError{code, {"ProgramGraph child resolution", 0, 0}})};
        }
        VernonProgramExecutable *child = std::move(childResult).value().release();
        const auto mapping = linked.nodeMappings.find(node.id);
        if (mapping == linked.nodeMappings.end()) {
            vernonRuntimeProgramExecutableDestroy(child);
            fail(graph->context, "ProgramGraph child remapping is unavailable", VERNON_STATUS_INTERNAL_ERROR);
            return ProgramGraphExecutableResult{vernon::err(vernon::RuntimeError{
                vernon::RuntimeErrorCode::InternalFailure, {"ProgramGraph child remapping is unavailable", 0, 0}})};
        }
        ProgramGraphNodeAutodiffState state;
        state.executable = std::shared_ptr<VernonProgramExecutable>(child, vernonRuntimeProgramExecutableDestroy);
        state.globalValues = mapping->second.values;
        state.globalStorages = mapping->second.storages;
        executable->programGraphNodeAutodiff.emplace(node.id, std::move(state));
    }
    executable->id = linked.id;
    executable->programGraphId = graph->id;
    executable->publicParameterSlots.reserve(graph->exports.size());
    for (const ProgramGraphExport &exported : graph->exports) {
        const uint32_t slot = linked.boundarySlots.at(exported.boundary);
        if (std::find(executable->publicParameterSlots.begin(), executable->publicParameterSlots.end(), slot) ==
            executable->publicParameterSlots.end())
            executable->publicParameterSlots.push_back(slot);
    }
    buildExecutableBindingIndex(*executable);
    return ProgramGraphExecutableResult{vernon::ok(executable.release())};
}
} // namespace

VernonRuntimeOperationStatus vernonRuntimeResolveProgramGraphResult(VernonProgramGraph *graph,
                                                                    VernonProgramExecutable **output) {
    VernonRuntimeContext *context = graph ? graph->context : nullptr;
    return runtimeHandleResultBoundary(context, output, [&] { return resolveProgramGraphOperation(graph); });
}

VernonProgramExecutable *vernonRuntimeResolveProgramGraph(VernonProgramGraph *graph) {
    VernonProgramExecutable *output = nullptr;
    (void)vernonRuntimeResolveProgramGraphResult(graph, &output);
    return output;
}

namespace {
using ProgramExecutableHandleResult = vernon::Result<VernonProgramExecutable *, vernon::RuntimeError>;

ProgramExecutableHandleResult resolveProgramOperation(VernonProgramBundle *bundle,
                                                      const VernonProgramVariantSelector *selector) {
    if (!bundle)
        return ProgramExecutableHandleResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"invalid Program resolve invocation", 0, 0}})};
    auto bundlePin = bundle->lifecycle.pin();
    if (bundlePin.isErr()) {
        fail(bundle->context, "Program bundle is closing", vernon::toVernonStatus(bundlePin.error()));
        return ProgramExecutableHandleResult{vernon::err(std::move(bundlePin).error())};
    }
    auto selected = selectProgramDeployment(*bundle->context, bundle->deployments, selector);
    if (selected.isErr())
        return ProgramExecutableHandleResult{vernon::err(std::move(selected).error())};
    auto executable = resolveProgramDeployment(*bundle->context, *std::move(selected).value(), bundle->bundleRoot);
    if (executable.isErr())
        return executable;
    executable.value()->id = bundle->id;
    buildExecutableBindingIndex(*executable.value());
    return executable;
}
} // namespace

VernonRuntimeOperationStatus vernonRuntimeResolveProgramResult(VernonProgramBundle *bundle,
                                                               const VernonProgramVariantSelector *selector,
                                                               VernonProgramExecutable **output) {
    VernonRuntimeContext *context = bundle ? bundle->context : nullptr;
    return runtimeHandleResultBoundary(context, output, [&] { return resolveProgramOperation(bundle, selector); });
}

VernonProgramExecutable *vernonRuntimeResolveProgram(VernonProgramBundle *bundle,
                                                     const VernonProgramVariantSelector *selector) {
    VernonProgramExecutable *output = nullptr;
    (void)vernonRuntimeResolveProgramResult(bundle, selector, &output);
    return output;
}
