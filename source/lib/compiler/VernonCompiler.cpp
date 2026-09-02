#include "VernonCompiler.h"
#include "compiler_dispatch.h"
#include "compiler_frontend.h"
#include "compiler_graphics_bootstrap.h"
#include "compiler_internal.h"
#include "compiler_kernel_bootstrap.h"
#include "compiler_program_finalization.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <vector>

struct VernonCompilerContext {
    vernon::compiler::CompilerFrontend *frontend{};
    VernonCpuRuntimeHelpersV1 cpuRuntimeHelpers{};
};

struct VernonCompileResult {
    VernonStatus status{VERNON_STATUS_INTERNAL_ERROR};
    std::string diagnostics;
    std::vector<vernon::compiler::Artifact> artifacts;
    std::string reflection;
    vernon::compiler::CpuExecutionStatePtr cpuExecution;
};

namespace {

VernonStringView viewOf(const std::string &value) { return VernonStringView{value.data(), value.size()}; }

std::string stringOf(VernonStringView value) {
    return value.data ? std::string(value.data, value.size) : std::string();
}

std::unique_ptr<VernonCompileResult> validate(VernonCompilerContext *context, const char *source, size_t sourceSize,
                                              vernon::compiler::PreparedModulePtr *prepared = nullptr) {
    auto result = std::make_unique<VernonCompileResult>();
    if (!context || (!source && sourceSize != 0)) {
        result->status = VERNON_STATUS_INVALID_ARGUMENT;
        result->diagnostics = "context and source must be valid";
        return result;
    }
    vernon::compiler::PreparedModulePtr ownedPrepared;
    result->status = vernon::compiler::prepareMlir(*context->frontend, source, sourceSize, ownedPrepared,
                                                   result->artifacts, result->reflection, result->diagnostics);
    if (prepared)
        *prepared = std::move(ownedPrepared);
    return result;
}

} // namespace

extern "C" {

VernonStatus vernonCompilerRegisterCpuRuntimeHelpersV1(VernonCompilerContext *context,
                                                       const VernonCpuRuntimeHelpersV1 *helpers) {
    if (!context || !helpers || helpers->struct_size != sizeof(VernonCpuRuntimeHelpersV1) ||
        !helpers->workgroup_address || !helpers->lane_address || !helpers->workgroup_barrier ||
        !helpers->workgroup_is_leader)
        return VERNON_STATUS_INVALID_ARGUMENT;
    context->cpuRuntimeHelpers = *helpers;
    return VERNON_STATUS_OK;
}

VernonCompilerContext *vernonCompilerCreate(void) {
    std::unique_ptr<VernonCompilerContext> context(new (std::nothrow) VernonCompilerContext());
    if (!context)
        return nullptr;
    context->frontend = vernon::compiler::createCompilerFrontend();
    return context->frontend ? context.release() : nullptr;
}

void vernonCompilerDestroy(VernonCompilerContext *context) {
    if (!context)
        return;
    vernon::compiler::destroyCompilerFrontend(context->frontend);
    delete context;
}

VernonTargetCapabilities vernonCompilerGetTargetCapabilities(const VernonCompilerContext *context,
                                                             VernonTarget target) {
    if (!context || target < VERNON_TARGET_CPU || target > VERNON_TARGET_CUDA)
        return VernonTargetCapabilities{0, 0, 0, 0, 0};
    return vernon::compiler::targetCapabilities(target);
}

VernonCompileResult *vernonCompilerValidateMlir(VernonCompilerContext *context, const char *source, size_t sourceSize) {
    return validate(context, source, sourceSize).release();
}

VernonCompileResult *vernonCompilerCompileMlir(VernonCompilerContext *context, const char *source, size_t sourceSize,
                                               VernonTarget target) {
    VernonCompileOptions options{};
    options.struct_size = sizeof(options);
    options.target = target;
    return vernonCompilerCompileMlirWithOptions(context, source, sourceSize, &options);
}

VernonCompileResult *vernonCompilerCompileMlirWithOptions(VernonCompilerContext *context, const char *source,
                                                          size_t sourceSize, const VernonCompileOptions *options) {
    vernon::compiler::PreparedModulePtr prepared;
    auto result = validate(context, source, sourceSize, &prepared);
    if (result->status != VERNON_STATUS_OK)
        return result.release();

    if (!options) {
        result->status = VERNON_STATUS_INVALID_ARGUMENT;
        result->diagnostics = "compile options must select a target";
        result->artifacts.clear();
        return result.release();
    }

    vernon::compiler::CompileOptions parsedOptions = vernon::compiler::defaultCompileOptions(options->target);
    result->status = vernon::compiler::parseCompileOptions(*options, parsedOptions, result->diagnostics);
    if (result->status != VERNON_STATUS_OK) {
        result->artifacts.clear();
        return result.release();
    }
    result->status =
        vernon::compiler::compileTarget(*prepared, parsedOptions, result->artifacts, result->reflection,
                                        result->diagnostics, &context->cpuRuntimeHelpers, result->cpuExecution);
    return result.release();
}

VernonCompileResult *vernonCompilerPlanProgram(VernonCompilerContext *context, const char *program,
                                               size_t programSize) {
    return validate(context, program, programSize).release();
}

VernonCompileResult *vernonCompilerPlanKernel(VernonCompilerContext *context, const char *kernel, size_t kernelSize) {
    auto result = std::make_unique<VernonCompileResult>();
    if (!context || (!kernel && kernelSize != 0)) {
        result->status = VERNON_STATUS_INVALID_ARGUMENT;
        result->diagnostics = "context and kernel source must be valid";
        return result.release();
    }
    result->status = vernon::compiler::planComputeKernel(*context->frontend, kernel, kernelSize, result->artifacts,
                                                         result->reflection, result->diagnostics);
    return result.release();
}

VernonCompileResult *vernonCompilerPlanGraphics(VernonCompilerContext *context, const VernonGraphicsStageSource *stages,
                                                size_t stageCount, const char *topology, size_t topologySize,
                                                const VernonStringView *features, size_t featureCount,
                                                const VernonStringView *attachmentTypes, size_t attachmentCount,
                                                uint32_t colorCount, const VernonGraphicsPlanOperand *operands,
                                                size_t operandCount) {
    auto result = std::make_unique<VernonCompileResult>();
    if (!context || (!stages && stageCount != 0) || (!features && featureCount != 0) ||
        (!operands && operandCount != 0) || (!topology && topologySize != 0) ||
        (!attachmentTypes && attachmentCount != 0) || colorCount == 0 || colorCount > attachmentCount ||
        attachmentCount - colorCount > 1) {
        result->status = VERNON_STATUS_INVALID_ARGUMENT;
        result->diagnostics = "graphics plan sources, topology, and attachments must be valid";
        return result.release();
    }
    std::vector<vernon::compiler::GraphicsStageSource> parsedStages;
    parsedStages.reserve(stageCount);
    for (size_t index = 0; index < stageCount; ++index) {
        const VernonGraphicsStageSource &stage = stages[index];
        if (stage.struct_size != sizeof(VernonGraphicsStageSource) || (!stage.source && stage.source_size != 0)) {
            result->status = VERNON_STATUS_INVALID_ARGUMENT;
            result->diagnostics = "graphics stage source is incomplete";
            return result.release();
        }
        parsedStages.push_back({stage.source, stage.source_size});
    }
    std::vector<std::string> parsedFeatures;
    parsedFeatures.reserve(featureCount);
    for (size_t index = 0; index < featureCount; ++index)
        parsedFeatures.push_back(stringOf(features[index]));
    std::vector<vernon::compiler::GraphicsPlanOperand> parsedOperands;
    parsedOperands.reserve(operandCount);
    for (size_t index = 0; index < operandCount; ++index) {
        const VernonGraphicsPlanOperand &operand = operands[index];
        if (operand.struct_size != sizeof(VernonGraphicsPlanOperand)) {
            result->status = VERNON_STATUS_INVALID_ARGUMENT;
            result->diagnostics = "graphics operand descriptor is incomplete";
            return result.release();
        }
        parsedOperands.push_back({stringOf(operand.name), stringOf(operand.type)});
    }
    std::vector<std::string> parsedAttachments;
    parsedAttachments.reserve(attachmentCount);
    for (size_t index = 0; index < attachmentCount; ++index)
        parsedAttachments.push_back(stringOf(attachmentTypes[index]));
    result->status = vernon::compiler::planGraphicsProgram(
        *context->frontend, parsedStages, std::string(topology ? topology : "", topologySize), parsedFeatures,
        parsedAttachments, colorCount, parsedOperands, result->artifacts, result->reflection, result->diagnostics);
    return result.release();
}

VernonCompileResult *vernonCompilerFinalizeProgram(VernonCompilerContext *context, const char *plan, size_t planSize,
                                                   const VernonCompiledKernel *kernels, size_t kernelCount) {
    return vernonCompilerFinalizeProgramWithShapes(context, plan, planSize, kernels, kernelCount, nullptr, 0);
}

VernonCompileResult *vernonCompilerFinalizeProgramWithShapes(VernonCompilerContext *context, const char *plan,
                                                             size_t planSize, const VernonCompiledKernel *kernels,
                                                             size_t kernelCount,
                                                             const VernonProgramShapeFact *shapeFacts,
                                                             size_t shapeFactCount) {
    auto result = std::make_unique<VernonCompileResult>();
    if (!context || (!plan && planSize != 0) || (!kernels && kernelCount != 0) ||
        (!shapeFacts && shapeFactCount != 0)) {
        result->status = VERNON_STATUS_INVALID_ARGUMENT;
        result->diagnostics = "Program plan, compiled kernels, and shape facts must be valid";
        return result.release();
    }
    llvm::Expected<llvm::json::Value> parsedPlan = llvm::json::parse(llvm::StringRef(plan ? plan : "", planSize));
    llvm::json::Object *root = parsedPlan ? parsedPlan->getAsObject() : nullptr;
    llvm::json::Array *requests = root ? root->getArray("kernel_compile_requests") : nullptr;
    llvm::json::Object *execution = root ? root->getObject("execution") : nullptr;
    llvm::json::Array *values = execution ? execution->getArray("values") : nullptr;
    if (!requests || !values) {
        result->status = VERNON_STATUS_INVALID_ARGUMENT;
        result->diagnostics = "Program plan has no kernel requests or executable values";
        return result.release();
    }
    std::map<std::string, llvm::json::Object *> requestById;
    for (llvm::json::Value &requestValue : *requests) {
        llvm::json::Object *request = requestValue.getAsObject();
        std::optional<llvm::StringRef> id = request ? request->getString("id") : std::nullopt;
        if (!id || !requestById.emplace(id->str(), request).second) {
            result->status = VERNON_STATUS_INVALID_ARGUMENT;
            result->diagnostics = "Program plan contains invalid or duplicate kernel request ids";
            return result.release();
        }
    }
    const auto valueById = [&](int64_t id) -> llvm::json::Object * {
        for (llvm::json::Value &value : *values)
            if (llvm::json::Object *object = value.getAsObject())
                if (object->getInteger("id") == id)
                    return object;
        return nullptr;
    };
    const auto sameShape = [](const llvm::json::Array *left, const llvm::json::Array *right) {
        const size_t leftSize = left ? left->size() : 0;
        const size_t rightSize = right ? right->size() : 0;
        if (leftSize != rightSize)
            return false;
        for (size_t index = 0; index < leftSize; ++index) {
            const std::optional<int64_t> expected = (*left)[index].getAsInteger();
            const std::optional<int64_t> actual = (*right)[index].getAsInteger();
            if (!expected || !actual)
                return false;
            // Dynamic extents are wildcards. A concrete Program value may
            // specialize a dynamic compiled ABI, and a concrete compiled ABI
            // may specialize a dynamic Program value.
            if (*expected > 0 && *actual > 0 && *expected != *actual)
                return false;
        }
        return true;
    };
    const auto suffixShape = [](const llvm::json::Array *logical, const llvm::json::Array *element) {
        if (!logical || !element || logical->size() < element->size())
            return false;
        const size_t offset = logical->size() - element->size();
        for (size_t index = 0; index < element->size(); ++index) {
            const std::optional<int64_t> expected = (*element)[index].getAsInteger();
            const std::optional<int64_t> actual = (*logical)[index + offset].getAsInteger();
            if (!expected || !actual || *expected <= 0 || (*actual > 0 && *expected != *actual))
                return false;
        }
        return true;
    };

    std::map<std::string, std::string> program;
    llvm::json::Array compiledRows;
    llvm::json::Object canonicalProgram;
    llvm::json::Object stageContracts;
    std::vector<vernon::compiler::CanonicalComputeStage> canonicalStages;
    std::map<std::string, size_t> canonicalStageByRequest;
    for (size_t index = 0; index < kernelCount; ++index) {
        const VernonCompiledKernel &kernel = kernels[index];
        const std::string requestId = stringOf(kernel.request_id);
        const std::string stageId = stringOf(kernel.stage_id);
        const std::string entryName = stringOf(kernel.entry);
        auto requestIt = requestById.find(requestId);
        if (requestIt == requestById.end() || stageId.empty() || entryName.empty()) {
            result->status = VERNON_STATUS_INVALID_ARGUMENT;
            result->diagnostics = "compiled kernel has an unknown request id or incomplete identity";
            return result.release();
        }
        llvm::Expected<llvm::json::Value> parsedKernel = llvm::json::parse(
            llvm::StringRef(kernel.reflection.data ? kernel.reflection.data : "", kernel.reflection.size));
        llvm::json::Object *kernelRoot = parsedKernel ? parsedKernel->getAsObject() : nullptr;
        llvm::json::Array *entries = kernelRoot ? kernelRoot->getArray("entries") : nullptr;
        llvm::json::Object *entry = nullptr;
        if (entries)
            for (llvm::json::Value &entryValue : *entries)
                if (llvm::json::Object *candidate = entryValue.getAsObject())
                    if (candidate->getString("name") == entryName) {
                        if (entry) {
                            entry = nullptr;
                            break;
                        }
                        entry = candidate;
                    }
        llvm::json::Object *request = requestIt->second;
        const std::string expectedStage = request->getString("kind") == "compute" ? "compute" : "graphics";
        if (expectedStage == "compute" && canonicalStageByRequest.count(requestId)) {
            result->status = VERNON_STATUS_INVALID_ARGUMENT;
            result->diagnostics = "compute Program request has more than one compiled implementation";
            return result.release();
        }
        const std::optional<llvm::StringRef> compiledStage = entry ? entry->getString("stage") : std::nullopt;
        const bool matchingStage = expectedStage == "compute"
                                       ? compiledStage == "compute"
                                       : compiledStage == "vertex" || compiledStage == "fragment";
        if (!entry || !matchingStage) {
            result->status = VERNON_STATUS_VERIFICATION_ERROR;
            result->diagnostics = "compiled kernel entry or stage does not match Program request '" + requestId + "'";
            return result.release();
        }
        if (expectedStage == "compute" &&
            !vernon::compiler::normalizeProgramImplementationAbi(*execution, *request, *entry, result->diagnostics)) {
            result->status = VERNON_STATUS_VERIFICATION_ERROR;
            return result.release();
        }
        for (size_t factIndex = 0; factIndex < shapeFactCount; ++factIndex) {
            const VernonProgramShapeFact &fact = shapeFacts[factIndex];
            if (stringOf(fact.request_id) != requestId)
                continue;
            const std::string parameterName = stringOf(fact.parameter);
            if (parameterName.empty() || (!fact.extents && fact.rank != 0)) {
                result->status = VERNON_STATUS_INVALID_ARGUMENT;
                result->diagnostics = "Program invocation shape fact has an incomplete identity";
                return result.release();
            }
            llvm::json::Array *requestBindings = request->getArray("bindings");
            std::optional<int64_t> valueId;
            if (requestBindings)
                for (const llvm::json::Value &bindingValue : *requestBindings)
                    if (const llvm::json::Object *binding = bindingValue.getAsObject();
                        binding && binding->getString("parameter") == parameterName) {
                        valueId = binding->getInteger("value");
                        break;
                    }
            if (!valueId)
                for (llvm::json::Value &candidateValue : *values)
                    if (llvm::json::Object *candidate = candidateValue.getAsObject())
                        if (std::optional<llvm::StringRef> name = candidate->getString("name");
                            candidate->getBoolean("external").value_or(false) && name &&
                            (*name == parameterName || name->ends_with("." + parameterName))) {
                            if (valueId) {
                                valueId.reset();
                                break;
                            }
                            valueId = candidate->getInteger("id");
                        }
            llvm::json::Object *value = valueId ? valueById(*valueId) : nullptr;
            llvm::json::Array *plannedShape = value ? value->getArray("shape") : nullptr;
            const bool textureShape = value && value->getString("type").value_or("").starts_with("!vernon.texture<") &&
                                      plannedShape && plannedShape->empty() && fact.rank > 0 && fact.rank <= 3;
            if (!valueId || !value || !plannedShape || (!textureShape && plannedShape->size() != fact.rank)) {
                result->status = VERNON_STATUS_VERIFICATION_ERROR;
                result->diagnostics = "Program invocation shape does not match parameter '" + parameterName + "'";
                return result.release();
            }
            std::vector<int64_t> concreteExtents;
            concreteExtents.reserve(fact.rank);
            for (size_t axis = 0; axis < fact.rank; ++axis) {
                const uint64_t extent = fact.extents[axis];
                const std::optional<int64_t> planned =
                    textureShape ? std::nullopt : (*plannedShape)[axis].getAsInteger();
                if (!extent || extent > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
                    (planned && *planned > 0 && static_cast<uint64_t>(*planned) != extent)) {
                    result->status = VERNON_STATUS_VERIFICATION_ERROR;
                    result->diagnostics =
                        "Program invocation shape violates the declared shape of parameter '" + parameterName + "'";
                    return result.release();
                }
                concreteExtents.push_back(static_cast<int64_t>(extent));
            }
            const auto concreteShape = [&]() {
                llvm::json::Array shape;
                for (int64_t extent : concreteExtents)
                    shape.emplace_back(extent);
                return shape;
            };
            (*value)["shape"] = llvm::json::Value(concreteShape());
            if (llvm::json::Array *graphs = execution->getArray("graphs"))
                for (llvm::json::Value &graphValue : *graphs)
                    if (llvm::json::Object *graph = graphValue.getAsObject())
                        if (llvm::json::Array *nodes = graph->getArray("nodes"))
                            for (llvm::json::Value &nodeValue : *nodes)
                                if (llvm::json::Object *node = nodeValue.getAsObject();
                                    node && node->getString("stage") == requestId)
                                    if (llvm::json::Array *resources = node->getArray("resources"))
                                        for (const llvm::json::Value &resourceValue : *resources)
                                            if (const llvm::json::Object *resource = resourceValue.getAsObject();
                                                resource && resource->getInteger("value") == *valueId)
                                                if (std::optional<int64_t> after = resource->getInteger("after"))
                                                    if (llvm::json::Object *afterValue = valueById(*after))
                                                        (*afterValue)["shape"] = llvm::json::Value(concreteShape());
        }
        std::map<std::string, llvm::json::Object *> interface;
        const auto collectInterface = [&](llvm::json::Array *rows) {
            if (!rows)
                return;
            for (llvm::json::Value &rowValue : *rows)
                if (llvm::json::Object *row = rowValue.getAsObject())
                    if (std::optional<llvm::StringRef> name = row->getString("vernon.source_name")) {
                        interface.emplace(name->str(), row);
                    }
        };
        if (expectedStage == "compute") {
            collectInterface(entry->getArray("arguments"));
            collectInterface(entry->getArray("results"));
        } else if (llvm::json::Array *compiledEntries = kernelRoot->getArray("entries")) {
            for (llvm::json::Value &entryValue : *compiledEntries) {
                llvm::json::Object *graphicsEntry = entryValue.getAsObject();
                if (!graphicsEntry)
                    continue;
                collectInterface(graphicsEntry->getArray("arguments"));
            }
        }
        llvm::json::Array *bindings = request->getArray("bindings");
        if (!bindings) {
            result->status = VERNON_STATUS_VERIFICATION_ERROR;
            result->diagnostics = "Program request has no ABI bindings";
            return result.release();
        }
        std::map<int64_t, size_t> bindingCounts;
        for (const llvm::json::Value &bindingValue : *bindings)
            if (const llvm::json::Object *binding = bindingValue.getAsObject())
                if (std::optional<int64_t> valueId = binding->getInteger("value"))
                    ++bindingCounts[*valueId];
        std::map<int64_t, size_t> bindingOrdinals;
        for (llvm::json::Value &bindingValue : *bindings) {
            if (expectedStage == "graphics")
                break;
            llvm::json::Object *binding = bindingValue.getAsObject();
            std::optional<llvm::StringRef> parameter = binding ? binding->getString("parameter") : std::nullopt;
            std::optional<int64_t> valueId = binding ? binding->getInteger("value") : std::nullopt;
            llvm::json::Object *value = valueId ? valueById(*valueId) : nullptr;
            const std::optional<llvm::StringRef> role = binding ? binding->getString("autodiff_role") : std::nullopt;
            const std::optional<llvm::StringRef> autodiffSource =
                binding ? binding->getString("autodiff_source") : std::nullopt;
            const llvm::StringRef valueType = value ? value->getString("type").value_or("") : "";
            if (role == "tape" || valueType == "!vernon.ad_tape" || valueType.starts_with("!vernon.ad_tape<"))
                continue;
            auto parameterIt = parameter ? interface.find(parameter->str()) : interface.end();
            if (!parameter || !valueId || parameterIt == interface.end() || !value) {
                result->status = VERNON_STATUS_VERIFICATION_ERROR;
                result->diagnostics =
                    "compiled kernel ABI does not bind Program parameter for request '" + requestId + "'";
                return result.release();
            }
            llvm::json::Object *parameterRow = parameterIt->second;
            std::optional<llvm::StringRef> expectedDtype = value->getString("dtype");
            llvm::json::Array *expectedShape = value->getArray("shape");
            llvm::json::Array aggregateExpectedShape;
            if (expectedStage == "compute" && bindingCounts[*valueId] > 1) {
                llvm::json::Object *layout = value->getObject("value_layout");
                llvm::json::Array *leaves = layout ? layout->getArray("leaves") : nullptr;
                llvm::json::Object *leaf = nullptr;
                const bool semanticProjection = autodiffSource && (role == "gradient" || role == "cotangent");
                if (leaves && semanticProjection) {
                    std::optional<size_t> leafIndex;
                    if (const std::optional<int64_t> projected = binding->getInteger("leaf");
                        projected && *projected >= 0)
                        leafIndex = static_cast<size_t>(*projected);
                    else
                        leafIndex =
                            vernon::compiler::resolveProgramValueLeafIndex(*layout, *autodiffSource, *parameter);
                    if (leafIndex && *leafIndex < leaves->size())
                        leaf = (*leaves)[*leafIndex].getAsObject();
                }
                if (!leaf && !semanticProjection) {
                    const size_t ordinal = bindingOrdinals[(*valueId)]++;
                    leaf = leaves && ordinal < leaves->size() ? (*leaves)[ordinal].getAsObject() : nullptr;
                }
                if (!leaf) {
                    result->status = VERNON_STATUS_VERIFICATION_ERROR;
                    result->diagnostics =
                        "Program aggregate binding does not match its canonical Value ABI for request '" + requestId +
                        "'";
                    return result.release();
                }
                expectedDtype = leaf->getString("dtype");
                if (llvm::json::Array *ownerShape = value->getArray("shape"))
                    for (const llvm::json::Value &extent : *ownerShape)
                        aggregateExpectedShape.emplace_back(extent);
                if (llvm::json::Array *leafShape = leaf->getArray("shape"))
                    for (const llvm::json::Value &extent : *leafShape)
                        aggregateExpectedShape.emplace_back(extent);
                expectedShape = &aggregateExpectedShape;
            }
            std::optional<llvm::StringRef> actualDtype = parameterRow->getString("dtype");
            if (!actualDtype)
                actualDtype = parameterRow->getString("vernon.dtype");
            llvm::json::Array *actualShape = parameterRow->getArray("shape");
            if (!actualShape)
                actualShape = parameterRow->getArray("source_shape");
            llvm::json::Array derivativeActualShape;
            if (actualShape && bindingCounts[*valueId] > 1 && (role == "gradient" || role == "cotangent")) {
                llvm::json::Object *layout = parameterRow->getObject("value_layout");
                if (!layout)
                    layout = parameterRow->getObject("element_layout");
                llvm::json::Array *leaves = layout ? layout->getArray("leaves") : nullptr;
                llvm::json::Object *leaf = leaves && leaves->size() == 1 ? (*leaves)[0].getAsObject() : nullptr;
                llvm::json::Array *leafShape = leaf ? leaf->getArray("shape") : nullptr;
                if (leafShape) {
                    for (const llvm::json::Value &extent : *actualShape)
                        derivativeActualShape.emplace_back(extent);
                    for (const llvm::json::Value &extent : *leafShape)
                        derivativeActualShape.emplace_back(extent);
                    actualShape = &derivativeActualShape;
                }
            }
            const llvm::StringRef parameterKind = parameterRow->getString("kind").value_or("");
            const bool opaqueResource = parameterKind == "image" || parameterKind == "sampler";
            const bool compatibleBindingShape =
                expectedStage == "compute"
                    ? vernon::compiler::compatibleProgramBindingShape(
                          role.value_or(""), parameterRow->getString("vernon.autodiff_carrier").value_or(""),
                          expectedShape, actualShape)
                    : sameShape(expectedShape, actualShape);
            const bool graphicsVertexElement = expectedStage == "graphics" &&
                                               parameterRow->getArray("attribute_leaves") &&
                                               suffixShape(expectedShape, actualShape);
            if (!opaqueResource && ((expectedDtype && actualDtype && *expectedDtype != *actualDtype) ||
                                    (!compatibleBindingShape && !graphicsVertexElement))) {
                auto shapeText = [](const llvm::json::Array *shape) {
                    if (!shape)
                        return std::string("[]");
                    std::string text = "[";
                    for (size_t index = 0; index < shape->size(); ++index) {
                        if (index)
                            text += ", ";
                        if (std::optional<int64_t> extent = (*shape)[index].getAsInteger())
                            text += std::to_string(*extent);
                    }
                    text += "]";
                    return text;
                };
                result->status = VERNON_STATUS_VERIFICATION_ERROR;
                result->diagnostics = "compiled kernel ABI type does not match Program value for request '" +
                                      requestId + "' parameter '" + parameter->str() +
                                      "' expected dtype=" + (expectedDtype ? expectedDtype->str() : "<none>") +
                                      " shape=" + shapeText(expectedShape) +
                                      " actual dtype=" + (actualDtype ? actualDtype->str() : "<none>") +
                                      " shape=" + shapeText(actualShape);
                return result.release();
            }
        }
        auto grouped = canonicalStageByRequest.find(requestId);
        if (grouped == canonicalStageByRequest.end()) {
            canonicalStageByRequest[requestId] = canonicalStages.size();
            canonicalStages.push_back(vernon::compiler::CanonicalComputeStage{requestId, stageId, *kernelRoot, *entry});
            program.emplace(requestId, stageId);
        } else {
            vernon::compiler::CanonicalComputeStage &stage = canonicalStages[grouped->second];
            if (stage.implementationStageId != stageId) {
                result->status = VERNON_STATUS_INVALID_ARGUMENT;
                result->diagnostics = "graphics modules for one Program request disagree on stage identity";
                return result.release();
            }
            llvm::json::Array *mergedEntries = stage.compiledReflection.getArray("entries");
            if (!mergedEntries) {
                result->status = VERNON_STATUS_VERIFICATION_ERROR;
                result->diagnostics = "compiled graphics reflection has no entries";
                return result.release();
            }
            for (const llvm::json::Value &moduleEntry : *entries)
                mergedEntries->emplace_back(moduleEntry);
            llvm::json::Array *mergedFeatures = stage.compiledReflection.getArray("required_features");
            if (!mergedFeatures)
                stage.compiledReflection["required_features"] = llvm::json::Array();
            mergedFeatures = stage.compiledReflection.getArray("required_features");
            if (const llvm::json::Array *features = kernelRoot->getArray("required_features"))
                for (const llvm::json::Value &feature : *features)
                    mergedFeatures->emplace_back(feature);
        }
        llvm::json::Object row;
        row["request_id"] = requestId;
        row["stage_id"] = stageId;
        row["entry"] = entryName;
        row["reflection"] = std::move(*kernelRoot);
        compiledRows.emplace_back(std::move(row));
    }
    if (canonicalStageByRequest.size() != requestById.size()) {
        result->status = VERNON_STATUS_INVALID_ARGUMENT;
        result->diagnostics = "compiled kernels do not exactly cover Program kernel requests";
        return result.release();
    }
    llvm::json::Object targetImplementations;
    if (!vernon::compiler::buildCanonicalComputeProgram(*execution, canonicalStages, canonicalProgram, stageContracts,
                                                        targetImplementations, result->diagnostics)) {
        result->status = VERNON_STATUS_VERIFICATION_ERROR;
        return result.release();
    }
    llvm::json::Object programObject;
    for (const auto &[requestId, stageId] : program)
        programObject[requestId] = stageId;
    (*root)["program"] = std::move(programObject);
    (*root)["compiled_kernels"] = std::move(compiledRows);
    (*root)["canonical_program"] = std::move(canonicalProgram);
    (*root)["stage_contracts"] = std::move(stageContracts);
    if (!targetImplementations.empty())
        (*root)["target_implementations"] = std::move(targetImplementations);
    root->erase("kernel_compile_requests");
    std::string finalized;
    llvm::raw_string_ostream stream(finalized);
    stream << llvm::json::Value(std::move(*root));
    result->reflection = std::move(finalized);
    result->status = VERNON_STATUS_OK;
    return result.release();
}

void vernonCompileResultDestroy(VernonCompileResult *result) { delete result; }

VernonStatus vernonCompileResultGetStatus(const VernonCompileResult *result) {
    return result ? result->status : VERNON_STATUS_INVALID_ARGUMENT;
}

VernonStringView vernonCompileResultGetDiagnostics(const VernonCompileResult *result) {
    return result ? viewOf(result->diagnostics) : VernonStringView{nullptr, 0};
}

VernonStringView vernonCompileResultGetArtifact(const VernonCompileResult *result) {
    return result && !result->artifacts.empty() ? viewOf(result->artifacts.front().data) : VernonStringView{nullptr, 0};
}

size_t vernonCompileResultGetArtifactCount(const VernonCompileResult *result) {
    return result ? result->artifacts.size() : 0;
}

VernonStringView vernonCompileResultGetArtifactName(const VernonCompileResult *result, size_t index) {
    return result && index < result->artifacts.size() ? viewOf(result->artifacts[index].name)
                                                      : VernonStringView{nullptr, 0};
}

VernonStringView vernonCompileResultGetArtifactData(const VernonCompileResult *result, size_t index) {
    return result && index < result->artifacts.size() ? viewOf(result->artifacts[index].data)
                                                      : VernonStringView{nullptr, 0};
}

VernonStringView vernonCompileResultGetReflection(const VernonCompileResult *result) {
    return result ? viewOf(result->reflection) : VernonStringView{nullptr, 0};
}

VernonCpuEntryPoint vernonCompileResultGetCpuEntry(const VernonCompileResult *result, const char *entryName,
                                                   size_t entryNameSize) {
    if (!result || result->status != VERNON_STATUS_OK || (!entryName && entryNameSize != 0))
        return nullptr;
    const std::string_view name(entryName ? entryName : "", entryNameSize);
    return vernon::compiler::findCompiledCpuEntry(result->cpuExecution.get(), name);
}

} // extern "C"
