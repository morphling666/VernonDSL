#include "program_execution_backend.h"

#include "backend_cpu.h"
#include "content_hash.h"
#include "program_execution_manifest.h"
#include "program_manifest.h"
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
        return reject(diagnostic, "PROGRAM_BLOB_AUTHENTICATION", "/artifact_system/blobs/" + module.blob,
                      "CodeModule references an unknown Blob");

    std::error_code error;
    const std::filesystem::path root = std::filesystem::canonical(bundleRoot, error);
    const std::filesystem::path relative = std::filesystem::u8path(blob->second.uri);
    if (error || !std::filesystem::is_directory(root, error) || relative.empty() || relative.is_absolute() ||
        relative.has_root_path() || relative.lexically_normal() != relative ||
        std::find(relative.begin(), relative.end(), std::filesystem::path("..")) != relative.end())
        return reject(diagnostic, "PROGRAM_BLOB_AUTHENTICATION",
                      "/artifact_system/blobs/" + module.blob + "/location/uri",
                      "external Blob URI is not a normalized path beneath the bundle root");
    const std::filesystem::path path = std::filesystem::canonical(root / relative, error);
    const std::filesystem::path contained = path.lexically_relative(root);
    if (error || !std::filesystem::is_regular_file(path, error) || contained.empty() || contained.is_absolute() ||
        *contained.begin() == std::filesystem::path(".."))
        return reject(diagnostic, "PROGRAM_BLOB_AUTHENTICATION",
                      "/artifact_system/blobs/" + module.blob + "/location/uri",
                      "external Blob cannot be opened beneath the bundle root");
    const uintmax_t fileSize = std::filesystem::file_size(path, error);
    if (error || fileSize != blob->second.byteLength || fileSize > std::numeric_limits<size_t>::max() ||
        fileSize > static_cast<uintmax_t>(std::numeric_limits<std::streamsize>::max()))
        return reject(diagnostic, "PROGRAM_BLOB_AUTHENTICATION",
                      "/artifact_system/blobs/" + module.blob + "/byte_length",
                      "external Blob length does not match its manifest");
    std::vector<uint8_t> blobBytes(static_cast<size_t>(fileSize));
    std::ifstream input(path, std::ios::binary);
    if ((!blobBytes.empty() &&
         !input.read(reinterpret_cast<char *>(blobBytes.data()), static_cast<std::streamsize>(blobBytes.size()))) ||
        sha256Hex(blobBytes.data(), blobBytes.size()) != blob->second.sha256)
        return reject(diagnostic, "PROGRAM_BLOB_AUTHENTICATION", "/artifact_system/blobs/" + module.blob + "/sha256",
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

VernonLoadedPipeline *loadGraphicsProgramPipeline(VernonRuntimeContext &context, const ResolvedProgram &program,
                                                  const ArtifactSystem &artifacts,
                                                  const std::filesystem::path &bundleRoot, const Node &node,
                                                  const ResolvedStage &resolvedStage, Diagnostic &diagnostic) {
    const StageArtifact &artifact = resolvedStage.stage;
    if (context.backend == VERNON_RUNTIME_CPU)
        return reject(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "/graphs/0/nodes/0/operation",
                      "graphics Program requires a GPU backend"),
               nullptr;
    TargetBindingPlan bindingPlan;
    if (!buildTargetBindingPlan(program, node, resolvedStage, context.backend, bindingPlan, diagnostic))
        return nullptr;
    Variant variant;
    if (!materializeGraphicsTargetBindingPlan(bindingPlan, variant, diagnostic))
        return nullptr;

    VernonPipelineBundle bundle;
    bundle.context = &context;
    for (const CodeModule &module : artifact.modules) {
        std::vector<uint8_t> bytes;
        if (!loadCodeModuleBytes(artifacts, resolvedStage.artifact, bundleRoot, module, bytes, diagnostic))
            return nullptr;
        Stage stage;
        stage.stage = module.role;
        stage.entry = module.entryPoint;
        for (const vernon::runtime::NativeResourceSlot &slot : artifact.nativeSlots)
            if (slot.stage == module.role)
                stage.nativeSlots.push_back(slot);
        if (context.backend == VERNON_RUNTIME_METAL || isOpenGLBackend(context.backend))
            stage.source.assign(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        else
            stage.binary = std::move(bytes);
        const std::string key = resolvedStage.artifact + "/" + module.role;
        bundle.stages.emplace(key, std::move(stage));
        variant.program[module.role] = key;
        if (module.role == "vertex")
            variant.vertex = key;
        else if (module.role == "fragment")
            variant.fragment = key;
    }
    auto pipeline = std::make_unique<VernonLoadedPipeline>();
    pipeline->context = &context;
    pipeline->variant = std::move(variant);
    for (vernon::runtime::Parameter &parameter : pipeline->variant.parameters) {
        if (parameter.valueLayout)
            rebuildValueLayoutPathViews(*parameter.valueLayout);
        rebuildValueLayoutPathViews(parameter.elementLayout);
    }
    if (!resolveBackendPipeline(bundle, pipeline->variant, *pipeline))
        return reject(diagnostic, "PROGRAM_BACKEND_LOAD", "/artifact_system/artifacts/" + resolvedStage.artifact,
                      invocationDiagnostic(context)),
               nullptr;
    ++context.livePipelines;
    return pipeline.release();
}

VernonLoadedPipeline *loadComputeNodePipeline(VernonRuntimeContext &context, const ArtifactSystem &artifacts,
                                              const std::filesystem::path &bundleRoot,
                                              const ResolvedExecutableNode &node, Diagnostic &diagnostic) {
    const ResolvedStage &stage = *node.stage;
    if (stage.stage.modules.size() != 1)
        return reject(diagnostic, "PROGRAM_STAGE_BINDING", "/nodes/" + node.node->name,
                      "compute node has no resolved code module"),
               nullptr;
    Variant variant;
    ReflectedEntry reflection;
    if (!materializeTargetBindingPlan(node.plan, variant, reflection, diagnostic))
        return nullptr;
    std::vector<uint8_t> moduleBytes;
    if (!loadModuleBytes(artifacts, stage.artifact, bundleRoot, moduleBytes, diagnostic))
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
    VernonLoadedPipeline *pipeline = loadBackendTypedComputePipeline(
        context, std::move(variant), std::move(reflection), moduleBytes.empty() ? nullptr : moduleBytes.data(),
        moduleBytes.size(), entryName, cpuEntry, node.plan.nativeSlots);
    if (!pipeline)
        reject(diagnostic, "PROGRAM_BACKEND_LOAD", "/artifact_system/artifacts/" + stage.artifact,
               invocationDiagnostic(context));
    return pipeline;
}

bool convertExecutableProgram(const ResolvedProgram &program, ExecutableProgram &execution, Diagnostic &diagnostic) {
    execution = {};
    execution.values.resize(program.program.values.size());
    for (const Value &value : program.program.values) {
        if (value.id >= execution.values.size())
            return reject(diagnostic, "PROGRAM_VALUE", "/values", "Program value id exceeds the value arena");
        ProgramValueSlot slot;
        slot.id = value.id;
        slot.name = value.name;
        slot.type = value.type;
        slot.shape = value.shape;
        if (value.layout) {
            auto layout = std::make_shared<vernon::runtime::ValueLayout>();
            layout->layoutHash = value.layout->layoutHash;
            layout->byteSize = static_cast<uint32_t>(value.layout->byteSize);
            layout->alignment = static_cast<uint32_t>(value.layout->alignment);
            layout->logicalType = value.type;
            for (const LayoutLeaf &leaf : value.layout->leaves) {
                ValueLeaf converted(leaf.dtype, static_cast<uint32_t>(leaf.scalarCount),
                                    static_cast<uint32_t>(leaf.byteOffset));
                converted.shape = leaf.shape;
                for (const LayoutPathComponent &component : leaf.path) {
                    ValuePathComponent path;
                    if (!component.field.empty())
                        path.field = component.field;
                    if (component.index)
                        path.index = *component.index;
                    converted.path.push_back(std::move(path));
                }
                layout->leaves.push_back(std::move(converted));
            }
            slot.valueLayout = std::move(layout);
        }
        execution.values[value.id] = std::move(slot);
    }
    for (const Graph &graph : program.program.graphs) {
        ProgramGraph converted;
        converted.name = graph.name;
        converted.direction = graph.direction;
        for (const GraphInput &input : graph.inputs)
            converted.arguments.push_back(input.value);
        for (const GraphOutput &output : graph.outputs)
            converted.results.push_back(output.value);
        for (const Node &node : graph.nodes) {
            ProgramNode convertedNode;
            convertedNode.id = node.id;
            convertedNode.name = node.name;
            convertedNode.kind = node.operation == "graphics" ? "render" : "compute";
            convertedNode.stage = node.stage;
            convertedNode.operands = node.operands;
            convertedNode.results = node.results;
            for (const EndpointBinding &binding : node.bindings)
                convertedNode.bindings.push_back({binding.module + ":" + std::to_string(binding.index), binding.value});
            converted.nodes.push_back(std::move(convertedNode));
        }
        execution.graphs.push_back(std::move(converted));
    }
    return true;
}

} // namespace

VernonLoadedPipeline *loadBackendProgramPipeline(VernonRuntimeContext &context, const ResolvedProgram &program,
                                                 const ArtifactSystem &artifacts,
                                                 const std::filesystem::path &bundleRoot, Diagnostic &diagnostic) {
    diagnostic = {};
    ResolvedExecutablePlan executable;
    if (!buildResolvedExecutablePlan(program, context.backend, executable, diagnostic))
        return nullptr;
    if (executable.nodes.size() == 1) {
        const ResolvedExecutableNode &node = executable.nodes.front();
        if (node.node->operation == "graphics")
            return loadGraphicsProgramPipeline(context, program, artifacts, bundleRoot, *node.node, *node.stage,
                                               diagnostic);
        return loadComputeNodePipeline(context, artifacts, bundleRoot, node, diagnostic);
    }
    for (const ResolvedExecutableNode &node : executable.nodes)
        if (node.node->operation == "graphics")
            return reject(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "/graphs",
                          "multi-node graphics Programs execute through invocation-backed ExecutionGraph passes"),
                   nullptr;
    auto pipeline = std::make_unique<VernonLoadedPipeline>();
    pipeline->context = &context;
    auto topology = std::make_shared<VernonPipelineTopology>();
    if (!convertExecutableProgram(program, topology->execution, diagnostic))
        return nullptr;
    topology->residualValues = topology->execution.backwardCaptures();
    for (const ResolvedExecutableNode &node : executable.nodes) {
        if (topology->stageIndices.find(node.node->stage) != topology->stageIndices.end())
            continue;
        std::unique_ptr<VernonLoadedPipeline> child(
            loadComputeNodePipeline(context, artifacts, bundleRoot, node, diagnostic));
        if (!child)
            return nullptr;
        --context.livePipelines;
        std::vector<VernonProgramStageBinding> bindings;
        for (const TargetBinding &binding : node.plan.bindings)
            if (binding.source != SourceRepresentation::SystemValue)
                bindings.push_back({binding.endpoint.value, std::nullopt});
        topology->stageIndices.emplace(node.node->stage, topology->stages.size());
        topology->stages.push_back(VernonResolvedProgramStage{std::move(child), std::move(bindings)});
    }
    pipeline->topology = std::move(topology);
    ++context.livePipelines;
    return pipeline.release();
}

VernonLoadedPipeline *loadBackendProgramPipeline(VernonRuntimeContext &context, const char *programJson,
                                                 size_t programJsonSize, const char *artifactSystemJson,
                                                 size_t artifactSystemJsonSize,
                                                 const std::map<std::string, std::string> &stageBindings,
                                                 const std::filesystem::path &bundleRoot, std::string &error) {
    Diagnostic diagnostic;
    Program program;
    ArtifactSystem artifacts;
    const auto failed = [&]() -> VernonLoadedPipeline * {
        error =
            diagnostic.code + (diagnostic.path.empty() ? ": " : " at " + diagnostic.path + ": ") + diagnostic.message;
        return nullptr;
    };
    try {
        if (!parse(nlohmann::json::parse(programJson, programJson + programJsonSize), program, diagnostic) ||
            !parseArtifactSystem(nlohmann::json::parse(artifactSystemJson, artifactSystemJson + artifactSystemJsonSize),
                                 artifacts, diagnostic))
            return failed();
    } catch (const nlohmann::json::exception &exception) {
        reject(diagnostic, "PROGRAM_JSON_INVALID", "", exception.what());
        return failed();
    }
    ResolvedProgram resolved;
    if (!resolve(std::move(program), artifacts, stageBindings, resolved, diagnostic))
        return failed();
    if (VernonLoadedPipeline *loaded = loadBackendProgramPipeline(context, resolved, artifacts, bundleRoot, diagnostic))
        return loaded;
    return failed();
}

} // namespace vernon::runtime::program
