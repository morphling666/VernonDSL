#include "program_execution_backend.h"

#include "backend_cpu.h"
#include "content_hash.h"
#include "pipeline_metadata.h"
#include "program_execution_manifest.h"
#include "resolved_execution_plan.h"
#include "runtime/autodiff/runtime_autodiff_internal.h"
#include "runtime_dispatch.h"
#include "runtime_state.h"
#include "target_binding_plan.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <memory>

namespace vernon::runtime::program {
namespace {

bool reject(Diagnostic &diagnostic, std::string code, std::string path, std::string message) {
    diagnostic = {std::move(code), "resolve", std::move(path), std::move(message)};
    return false;
}

std::string physicalStageIdentity(const StageArtifact &stage) {
    nlohmann::json modules = nlohmann::json::array();
    for (const CodeModule &module : stage.modules)
        modules.push_back({{"role", module.role},
                           {"format", module.format},
                           {"entry_point", module.entryPoint},
                           {"byte_length", module.byteLength},
                           {"sha256", module.sha256}});
    const nlohmann::json identity{{"backend", stage.backend},
                                  {"operation", stage.operation},
                                  {"contract_hash", stage.contractHash},
                                  {"modules", std::move(modules)}};
    const std::string bytes = identity.dump(-1, ' ', false, nlohmann::json::error_handler_t::strict);
    return sha256Hex(bytes.data(), bytes.size());
}

std::string backendName(VernonRuntimeBackend backend) {
    switch (backend) {
    case VERNON_RUNTIME_CPU:
        return "cpu";
    case VERNON_RUNTIME_VULKAN:
        return "vulkan";
    case VERNON_RUNTIME_METAL:
        return "metal";
    case VERNON_RUNTIME_CUDA:
        return "cuda";
    case VERNON_RUNTIME_DIRECTX12:
        return "directx";
    case VERNON_RUNTIME_OPENGL:
        return "opengl";
    default:
        return {};
    }
}

bool loadCodeModuleBytes(const ArtifactSystem &artifacts, const std::string &artifactId,
                         const std::filesystem::path &bundleRoot, const CodeModule &module, std::vector<uint8_t> &bytes,
                         Diagnostic &diagnostic) {
    const auto blob = artifacts.blobs.find(module.blob);
    if (blob == artifacts.blobs.end())
        return reject(diagnostic, "PROGRAM_BLOB_AUTHENTICATION", "/blobs/" + module.blob,
                      "CodeModule references an unknown Blob");

    std::error_code error;
    const std::filesystem::path root = std::filesystem::canonical(bundleRoot, error);
    const std::filesystem::path relative = std::filesystem::u8path(blob->second.uri);
    if (error || !std::filesystem::is_directory(root, error) || relative.empty() || relative.is_absolute() ||
        relative.has_root_path() || relative.lexically_normal() != relative ||
        std::find(relative.begin(), relative.end(), std::filesystem::path("..")) != relative.end())
        return reject(diagnostic, "PROGRAM_BLOB_AUTHENTICATION", "/blobs/" + module.blob + "/location/uri",
                      "external Blob URI is not a normalized path beneath the bundle root");
    const std::filesystem::path path = std::filesystem::canonical(root / relative, error);
    const std::filesystem::path contained = path.lexically_relative(root);
    if (error || !std::filesystem::is_regular_file(path, error) || contained.empty() || contained.is_absolute() ||
        *contained.begin() == std::filesystem::path(".."))
        return reject(diagnostic, "PROGRAM_BLOB_AUTHENTICATION", "/blobs/" + module.blob + "/location/uri",
                      "external Blob cannot be opened beneath the bundle root");
    const uintmax_t fileSize = std::filesystem::file_size(path, error);
    if (error || fileSize != blob->second.byteLength || fileSize > std::numeric_limits<size_t>::max() ||
        fileSize > static_cast<uintmax_t>(std::numeric_limits<std::streamsize>::max()))
        return reject(diagnostic, "PROGRAM_BLOB_AUTHENTICATION", "/blobs/" + module.blob + "/byte_length",
                      "external Blob length does not match its manifest");
    std::vector<uint8_t> blobBytes(static_cast<size_t>(fileSize));
    std::ifstream input(path, std::ios::binary);
    if ((!blobBytes.empty() &&
         !input.read(reinterpret_cast<char *>(blobBytes.data()), static_cast<std::streamsize>(blobBytes.size()))) ||
        sha256Hex(blobBytes.data(), blobBytes.size()) != blob->second.sha256)
        return reject(diagnostic, "PROGRAM_BLOB_AUTHENTICATION", "/blobs/" + module.blob + "/sha256",
                      "external Blob SHA-256 does not match its manifest");
    if (module.offset > blobBytes.size() || module.byteLength > blobBytes.size() - module.offset ||
        sha256Hex(blobBytes.data() + module.offset, static_cast<size_t>(module.byteLength)) != module.sha256)
        return reject(diagnostic, "PROGRAM_BLOB_AUTHENTICATION",
                      "/artifact_system/artifacts/" + artifactId + "/modules/0/sha256",
                      "CodeModule range SHA-256 does not match its manifest");
    bytes.assign(blobBytes.begin() + static_cast<ptrdiff_t>(module.offset),
                 blobBytes.begin() + static_cast<ptrdiff_t>(module.offset + module.byteLength));
    return true;
}

bool loadModuleBytes(const ArtifactSystem &artifacts, const std::string &artifactId,
                     const std::filesystem::path &bundleRoot, std::vector<uint8_t> &bytes, Diagnostic &diagnostic) {
    const auto stage = artifacts.stages.find(artifactId);
    if (stage == artifacts.stages.end() || stage->second.modules.size() != 1)
        return reject(diagnostic, "PROGRAM_ARTIFACT_TARGET", "/artifact_system/artifacts/" + artifactId,
                      "single-compute loading requires exactly one code module");
    return loadCodeModuleBytes(artifacts, artifactId, bundleRoot, stage->second.modules.front(), bytes, diagnostic);
}

VernonStageExecutable *loadGraphicsProgramPipeline(VernonRuntimeContext &context, const ResolvedProgram &program,
                                                   const ArtifactSystem &artifacts,
                                                   const std::filesystem::path &bundleRoot, const Node &node,
                                                   const ResolvedStage &resolvedStage, Diagnostic &diagnostic) {
    const StageArtifact &artifact = resolvedStage.stage;
    if (context.backend == VERNON_RUNTIME_CPU || context.backend == VERNON_RUNTIME_CUDA)
        return reject(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "/graphs/0/nodes/0/operation",
                      "graphics Program requires a GPU backend"),
               nullptr;
    TargetBindingPlan bindingPlan;
    if (!buildTargetBindingPlan(program, node, resolvedStage, context.backend, bindingPlan, diagnostic))
        return nullptr;
    StageBindingPlan stagePlan;
    ReflectedEntry reflection;
    if (!buildStageBindingPlan(bindingPlan, stagePlan, reflection, diagnostic))
        return nullptr;

    BackendStageBuildInputs inputs;
    inputs.context = &context;
    for (const CodeModule &module : artifact.modules) {
        std::vector<uint8_t> bytes;
        if (!loadCodeModuleBytes(artifacts, resolvedStage.artifact, bundleRoot, module, bytes, diagnostic))
            return nullptr;
        LoadedStageArtifact stage;
        stage.entry = module.entryPoint;
        for (const vernon::runtime::NativeResourceSlot &slot : artifact.nativeSlots)
            if (slot.stage == module.role)
                stage.nativeSlots.push_back(slot);
        if (context.backend == VERNON_RUNTIME_METAL || isOpenGLBackend(context.backend))
            stage.source.assign(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        else
            stage.binary = std::move(bytes);
        const std::string key = resolvedStage.artifact + "/" + module.role;
        inputs.artifacts.emplace(key, std::move(stage));
        stagePlan.artifactKeys[module.role] = key;
        if (module.role == "vertex")
            stagePlan.vertex = key;
        else if (module.role == "fragment")
            stagePlan.fragment = key;
    }
    auto pipeline = std::make_unique<VernonStageExecutable>();
    pipeline->context = &context;
    pipeline->bindingProjection = std::move(stagePlan);
    rebuildStageBindingLayoutViews(pipeline->bindingProjection);
    if (!resolveBackendPipeline(inputs, pipeline->bindingProjection, *pipeline))
        return reject(diagnostic, "PROGRAM_BACKEND_LOAD", "/artifact_system/artifacts/" + resolvedStage.artifact,
                      invocationDiagnostic(context)),
               nullptr;
    return pipeline.release();
}

VernonStageExecutable *loadComputeNodePipeline(VernonRuntimeContext &context, const ArtifactSystem &artifacts,
                                               const std::filesystem::path &bundleRoot,
                                               const ResolvedExecutableNode &node, Diagnostic &diagnostic) {
    const ResolvedStage &stage = *node.stage;
    if (stage.stage.modules.size() != 1)
        return reject(diagnostic, "PROGRAM_STAGE_BINDING", "/nodes/" + node.node->name,
                      "compute node has no resolved code module"),
               nullptr;
    StageBindingPlan stagePlan;
    ReflectedEntry reflection;
    if (!buildStageBindingPlan(node.plan, stagePlan, reflection, diagnostic))
        return nullptr;
    std::vector<uint8_t> moduleBytes;
    // CPU code modules are relocatable objects linked by the embedding
    // application. The runtime resolves their registered entry point and must
    // not read or load the original object file. GPU backends still consume
    // their shader module bytes here.
    if (context.backend != VERNON_RUNTIME_CPU &&
        !loadModuleBytes(artifacts, stage.artifact, bundleRoot, moduleBytes, diagnostic))
        return nullptr;
    const std::string &entryName = stage.stage.modules.front().entryPoint;
    VernonCpuEntryPoint cpuEntry{};
    if (context.backend == VERNON_RUNTIME_CPU) {
        std::string error;
        if (!findRegisteredCpuEntry(context, entryName, cpuEntry, error)) {
            reject(diagnostic, "PROGRAM_ARTIFACT_ENTRY_POINT",
                   "/artifact_system/artifacts/" + stage.artifact + "/modules/0/entry_point", std::move(error));
            return nullptr;
        }
    }
    VernonStageExecutable *pipeline = loadBackendTypedComputePipeline(
        context, std::move(stagePlan), std::move(reflection), moduleBytes.empty() ? nullptr : moduleBytes.data(),
        moduleBytes.size(), entryName, cpuEntry, node.plan.nativeSlots);
    if (!pipeline)
        reject(diagnostic, "PROGRAM_BACKEND_LOAD", "/artifact_system/artifacts/" + stage.artifact,
               invocationDiagnostic(context));
    return pipeline;
}

} // namespace

VernonProgramExecutable *loadBackendProgramPipeline(VernonRuntimeContext &context,
                                                    std::shared_ptr<const ResolvedProgram> program,
                                                    const ArtifactSystem &artifacts,
                                                    const std::filesystem::path &bundleRoot, Diagnostic &diagnostic) {
    diagnostic = {};
    ResolvedExecutablePlan executable;
    if (!buildResolvedExecutablePlan(*program, context.backend, executable, diagnostic))
        return nullptr;
    auto plan = std::make_shared<ResolvedExecutionPlan>();
    plan->resolvedProgram = std::move(program);
    plan->boundaryLayoutViews.resize(plan->resolvedProgram->program.abi.boundarySlots.size());
    for (size_t index = 0; index < plan->resolvedProgram->program.abi.boundarySlots.size(); ++index) {
        const auto &slot = plan->resolvedProgram->program.abi.boundarySlots[index];
        if (slot.layout) {
            plan->boundaryLayoutViews[index] = materializeValueLayout(*slot.layout, slot.logicalType);
            rebuildValueLayoutPathViews(plan->boundaryLayoutViews[index]);
        }
    }
    std::unordered_map<std::string, size_t> cachedStages;
    for (const ResolvedExecutableNode &node : executable.nodes) {
        const std::string identity = physicalStageIdentity(node.stage->stage);
        auto cached = cachedStages.find(identity);
        if (cached == cachedStages.end()) {
            std::shared_ptr<VernonStageExecutable> child(
                executionKind(*node.node) == ExecutionKind::Graphics
                    ? loadGraphicsProgramPipeline(context, *plan->resolvedProgram, artifacts, bundleRoot, *node.node,
                                                  *node.stage, diagnostic)
                    : loadComputeNodePipeline(context, artifacts, bundleRoot, node, diagnostic));
            if (!child) {
                if (diagnostic.code.empty())
                    reject(diagnostic, "PROGRAM_BACKEND_LOAD", "/stages/" + node.node->stage,
                           invocationDiagnostic(context).empty() ? "backend did not report a stage load error"
                                                                 : invocationDiagnostic(context));
                return nullptr;
            }
            cached = cachedStages.emplace(identity, plan->stageCache.size()).first;
            plan->stageCache.push_back(std::move(child));
        }
        std::vector<NodeEndpointProjection> projections;
        for (const TargetBinding &binding : node.plan.bindings)
            if (!internalSource(binding.source))
                projections.push_back({binding.projection.value, binding.projection.leaf, binding});
        const std::optional<GraphDirection> direction = graphDirection(node.graph);
        if (!direction)
            return reject(diagnostic, "PROGRAM_GRAPH_DIRECTION", "/graphs",
                          "resolved node has an unsupported graph direction"),
                   nullptr;
        ResolvedOperationControls controls;
        if (executionKind(*node.node) == ExecutionKind::Compute) {
            const ComputeOperation &compute = computeOperation(*node.node);
            controls = ResolvedComputeControls{
                {compute.workgroups[0], compute.workgroups[1], compute.workgroups[2]},
                node.plan.dispatchMapping,
            };
        } else {
            const GraphicsOperation &graphics = graphicsOperation(*node.node);
            ResolvedGraphicsControls resolvedGraphics;
            resolvedGraphics.renderPassControl = graphics.renderPassControl;
            resolvedGraphics.drawCommandControl = graphics.drawCommandControl;
            resolvedGraphics.dynamicStateControl = graphics.dynamicStateControl;
            const auto attachment = [&](const GraphicsAttachmentSignature &signature) {
                const ResourceAccess &access = node.node->accesses[signature.access];
                return ResolvedGraphicsAttachment{
                    signature.access,
                    access.storage,
                    signature.location,
                    signature.aspects,
                };
            };
            for (const GraphicsAttachmentSignature &signature : graphics.colorAttachments)
                resolvedGraphics.colorAttachments.push_back(attachment(signature));
            if (graphics.depthStencilAttachment)
                resolvedGraphics.depthStencilAttachment = attachment(*graphics.depthStencilAttachment);
            controls = std::move(resolvedGraphics);
        }
        const NodeKey key{*direction, node.node->id};
        plan->nodes.emplace(
            key, ResolvedNodePlan{key, plan->stageCache[cached->second], std::move(projections), std::move(controls)});
    }
    if (!buildResolvedExecutionPolicies(*plan, diagnostic) || !validateResolvedExecutionPlan(*plan, diagnostic))
        return nullptr;
    auto pipeline = std::make_unique<VernonProgramExecutable>(context, std::move(plan));
    if (!vernon::runtime::ad::resolveProgramAutodiff(*pipeline, {}))
        return reject(diagnostic, "PROGRAM_ABI_MISMATCH", "/abi",
                      invocationDiagnostic(context).empty() ? "Program execution topology is invalid"
                                                            : invocationDiagnostic(context)),
               nullptr;
    ++context.livePipelines;
    return pipeline.release();
}

VernonProgramExecutable *loadBackendProgramPipeline(VernonRuntimeContext &context, const Program &program,
                                                    const ArtifactSystem &artifacts,
                                                    const std::filesystem::path &bundleRoot, std::string &error) {
    Diagnostic diagnostic;
    const auto failed = [&]() -> VernonProgramExecutable * {
        error =
            diagnostic.code + (diagnostic.path.empty() ? ": " : " at " + diagnostic.path + ": ") + diagnostic.message;
        return nullptr;
    };
    ResolvedProgram resolved;
    if (!resolve(program, artifacts, resolved, diagnostic)) {
        if (diagnostic.code.empty())
            reject(diagnostic, "PROGRAM_RESOLUTION_FAILED", "", "Program resolver rejected input without a diagnostic");
        return failed();
    }
    auto owner = std::make_shared<ResolvedProgram>(std::move(resolved));
    if (VernonProgramExecutable *loaded =
            loadBackendProgramPipeline(context, std::move(owner), artifacts, bundleRoot, diagnostic))
        return loaded;
    return failed();
}

} // namespace vernon::runtime::program
