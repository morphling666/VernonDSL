#include "compiler_reflection.h"

#include "compiler_program_reflection.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAttributeAbi.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace vernon::compiler {

mlir::FailureOr<LogicalReflectionModel> buildLogicalReflectionModel(mlir::ModuleOp module) {
    LogicalReflectionModel model;
    model.compilerContractVersion = VERNON_COMPILER_CONTRACT_VERSION;
    const auto collectValue = [](unsigned index, mlir::Type type, mlir::DictionaryAttr attributes, bool result) {
        LogicalValueModel value;
        value.index = index;
        value.sourcePath = (result ? "result." : "argument.") + std::to_string(index);
        if (attributes)
            if (auto sourcePath = attributes.getAs<mlir::StringAttr>("vernon.source_name")) {
                value.sourcePathExplicit = true;
                value.sourcePath = sourcePath.getValue().str();
            }
        if (attributes)
            if (auto dtype = attributes.getAs<mlir::StringAttr>("vernon.dtype")) {
                value.dtypeExplicit = true;
                value.dtype = dtype.getValue().str();
            }
        if (attributes)
            if (auto dtypes = attributes.getAs<mlir::ArrayAttr>("vernon.abi_leaf_dtypes"))
                for (mlir::Attribute dtype : dtypes)
                    if (auto string = mlir::dyn_cast<mlir::StringAttr>(dtype))
                        value.leafDtypes.push_back(string.getValue().str());
        if (value.leafDtypes.empty() && !value.dtype.empty())
            value.leafDtypes.push_back(value.dtype);
        if (value.dtype.empty() && value.leafDtypes.size() == 1)
            value.dtype = value.leafDtypes.front();
        if (value.dtype.empty()) {
            mlir::Type scalar = type;
            if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
                scalar = shaped.getElementType();
            else if (auto tensor = mlir::dyn_cast<mlir::vernon::TensorType>(type))
                scalar = tensor.getElementType();
            else if (auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(type))
                scalar = view.getElementType();
            if (scalar.isF16())
                value.dtype = "f16";
            else if (scalar.isF32())
                value.dtype = "f32";
            else if (scalar.isF64())
                value.dtype = "f64";
            else if (scalar.isInteger(1))
                value.dtype = "bool";
            else if (scalar.isIndex())
                value.dtype = "index";
        }
        if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
            for (int64_t extent : shaped.getShape())
                value.shape.push_back(extent);
        else if (auto tensor = mlir::dyn_cast<mlir::vernon::TensorType>(type))
            for (int64_t extent : tensor.getShape())
                value.shape.push_back(extent);
        else if (auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(type))
            for (int64_t extent : view.getShape())
                value.shape.push_back(extent);
        if (attributes)
            if (auto shape = attributes.getAs<mlir::DenseI64ArrayAttr>("vernon.source_shape")) {
                value.shapeExplicit = true;
                value.shape.clear();
                for (int64_t extent : shape.asArrayRef())
                    value.shape.push_back(extent);
            }
        return value;
    };
    for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
        auto stage = function->getAttrOfType<mlir::StringAttr>("vernon.stage");
        if (!stage)
            continue;
        LogicalEntryModel entry;
        entry.name = function.getSymName().str();
        entry.stage = stage.getValue().str();
        entry.isEntry = function->hasAttr("vernon.entry");
        for (unsigned index = 0; index < function.getNumArguments(); ++index)
            entry.arguments.push_back(
                collectValue(index, function.getArgumentTypes()[index], function.getArgAttrDict(index), false));
        for (unsigned index = 0; index < function.getNumResults(); ++index)
            entry.results.push_back(
                collectValue(index, function.getResultTypes()[index], function.getResultAttrDict(index), true));
        std::set<std::string> argumentPaths;
        for (const LogicalValueModel &value : entry.arguments)
            if (value.sourcePathExplicit && !argumentPaths.insert(value.sourcePath).second) {
                function.emitError() << "contains duplicate logical argument source path '" << value.sourcePath << "'";
                return mlir::failure();
            }
        std::set<std::string> resultPaths;
        for (const LogicalValueModel &value : entry.results)
            if (value.sourcePathExplicit && !resultPaths.insert(value.sourcePath).second) {
                function.emitError() << "contains duplicate logical result source path '" << value.sourcePath << "'";
                return mlir::failure();
            }
        model.entries.push_back(std::move(entry));
    }
    std::map<std::string, LogicalStructLayout> structLayouts;
    for (mlir::vernon::StructDeclOp declaration : module.getOps<mlir::vernon::StructDeclOp>()) {
        auto structure = mlir::vernon::StructType::get(module.getContext(), declaration.getSymName());
        mlir::FailureOr<mlir::vernon::ValueAbiLayout> planned = mlir::vernon::getValueAbiLayout(structure, module);
        if (mlir::failed(planned)) {
            declaration.emitError("cannot derive canonical struct Value ABI layout");
            return mlir::failure();
        }
        LogicalStructLayout layout;
        layout.name = declaration.getSymName().str();
        layout.size = planned->size;
        layout.alignment = planned->alignment;
        llvm::append_range(layout.fieldOffsets, planned->fieldOffsets);
        if (auto values = declaration->getAttrOfType<mlir::ArrayAttr>("fields"))
            for (mlir::Attribute value : values)
                if (auto field = mlir::dyn_cast<mlir::StringAttr>(value))
                    layout.fields.push_back(field.getValue().str());
        structLayouts.emplace(layout.name, std::move(layout));
    }
    for (auto &[name, layout] : structLayouts) {
        (void)name;
        model.structLayouts.push_back(std::move(layout));
    }

    if (auto encoded = module->getAttrOfType<mlir::ArrayAttr>("vernon.source_dependencies"))
        for (mlir::Attribute attribute : encoded)
            if (auto value = mlir::dyn_cast<mlir::StringAttr>(attribute)) {
                auto [path, digest] = value.getValue().split('=');
                model.dependencies.push_back({path.str(), digest.str()});
            }

    std::set<std::string> requiredFeatures;
    module.walk([&](mlir::Operation *operation) {
        if (mlir::isa<mlir::vernon::WorkgroupAllocOp>(operation))
            requiredFeatures.insert("workgroup_storage");
        if (mlir::isa<mlir::vernon::AtomicOp, mlir::vernon::PhysicalAtomicOp>(operation))
            requiredFeatures.insert("atomics");
        if (mlir::isa<mlir::vernon::BarrierOp>(operation))
            requiredFeatures.insert("barriers");
    });
    llvm::append_range(model.requiredFeatures, requiredFeatures);

    std::string canonicalModule;
    llvm::raw_string_ostream moduleStream(canonicalModule);
    module.print(moduleStream, mlir::OpPrintingFlags().enableDebugInfo(false));
    model.moduleHash = llvm::utohexstr(llvm::xxHash64(canonicalModule));
    return model;
}

std::vector<PhysicalEntryModel> buildPhysicalEntryModels(mlir::ModuleOp module,
                                                         const std::vector<PhysicalEntryProvenance> &provenance) {
    std::map<std::string, const PhysicalEntryProvenance *> provenanceEntries;
    for (const PhysicalEntryProvenance &entry : provenance)
        provenanceEntries.emplace(entry.name, &entry);
    std::vector<PhysicalEntryModel> entries;
    for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
        auto stage = function->getAttrOfType<mlir::StringAttr>("vernon.stage");
        if (!stage)
            continue;
        PhysicalEntryModel entry;
        entry.name = function.getSymName().str();
        entry.stage = stage.getValue().str();
        const PhysicalEntryProvenance *entryProvenance = nullptr;
        if (auto found = provenanceEntries.find(entry.name); found != provenanceEntries.end())
            entryProvenance = found->second;
        for (auto [index, type] : llvm::enumerate(function.getArgumentTypes())) {
            PhysicalArgumentModel argument;
            argument.index = index;
            llvm::raw_string_ostream stream(argument.type);
            type.print(stream);
            if (auto builtin = function.getArgAttrOfType<mlir::StringAttr>(index, "vernon.builtin"))
                argument.builtin = builtin.getValue().str();
            if (entryProvenance && index < entryProvenance->arguments.size())
                if (const std::optional<LogicalValueOrigin> &origin = entryProvenance->arguments[index]) {
                    argument.logicalIndex = origin->index;
                    argument.logicalPath = origin->path;
                }
            if (auto owner =
                    function.getArgAttrOfType<mlir::IntegerAttr>(index, mlir::vernon::kTensorDescriptorOwnerAttrName))
                if (owner.getInt() >= 0)
                    argument.descriptorOwner = static_cast<unsigned>(owner.getInt());
            if (auto component = function.getArgAttrOfType<mlir::StringAttr>(
                    index, mlir::vernon::kTensorDescriptorComponentAttrName))
                argument.descriptorComponent = component.getValue().str();
            if (auto dimension = function.getArgAttrOfType<mlir::IntegerAttr>(
                    index, mlir::vernon::kTensorDescriptorDimensionAttrName))
                if (dimension.getInt() >= 0)
                    argument.descriptorDimension = static_cast<unsigned>(dimension.getInt());
            entry.arguments.push_back(std::move(argument));
        }
        for (auto [index, type] : llvm::enumerate(function.getResultTypes())) {
            PhysicalResultModel result;
            result.index = index;
            llvm::raw_string_ostream stream(result.type);
            type.print(stream);
            if (entryProvenance && index < entryProvenance->results.size())
                if (const std::optional<LogicalValueOrigin> &origin = entryProvenance->results[index]) {
                    result.logicalIndex = origin->index;
                    result.logicalPath = origin->path;
                }
            entry.results.push_back(std::move(result));
        }
        entries.push_back(std::move(entry));
    }
    return entries;
}

namespace {

llvm::json::Value attributeToJson(mlir::Attribute attribute) {
    if (!attribute)
        return nullptr;
    if (auto string = mlir::dyn_cast<mlir::StringAttr>(attribute))
        return string.getValue().str();
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attribute))
        return integer.getInt();
    if (auto integers = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attribute)) {
        llvm::json::Array values;
        for (int64_t value : integers.asArrayRef())
            values.emplace_back(value);
        return values;
    }
    if (auto integers = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attribute)) {
        llvm::json::Array values;
        for (int32_t value : integers.asArrayRef())
            values.emplace_back(static_cast<int64_t>(value));
        return values;
    }
    if (auto array = mlir::dyn_cast<mlir::ArrayAttr>(attribute)) {
        llvm::json::Array values;
        for (mlir::Attribute value : array)
            values.emplace_back(attributeToJson(value));
        return values;
    }

    std::string printed;
    llvm::raw_string_ostream stream(printed);
    attribute.print(stream);
    return printed;
}

enum class ResourceKind { None, Texture, Sampler };

ResourceKind resourceKind(mlir::Type type) {
    if (mlir::isa<mlir::vernon::TextureType>(type))
        return ResourceKind::Texture;
    if (mlir::isa<mlir::vernon::SamplerType>(type))
        return ResourceKind::Sampler;
    return ResourceKind::None;
}

struct SampledTextureBinding {
    int64_t descriptorSet;
    int64_t binding;

    bool operator<(const SampledTextureBinding &other) const {
        return std::tie(descriptorSet, binding) < std::tie(other.descriptorSet, other.binding);
    }
};

using ResourceOrigins = std::set<unsigned>;
using ResourceProvenance = llvm::DenseMap<mlir::Value, ResourceOrigins>;

mlir::FailureOr<std::map<unsigned, std::set<SampledTextureBinding>>>
analyzeSampledTextureBindings(mlir::func::FuncOp function) {
    struct ForwardingEdge {
        mlir::Value source;
        mlir::Value destination;
    };

    ResourceProvenance provenance;
    for (auto [index, argument] : llvm::enumerate(function.getArguments()))
        if (resourceKind(argument.getType()) != ResourceKind::None)
            provenance[argument].insert(index);

    llvm::SmallVector<ForwardingEdge> edges;
    auto addEdge = [&](mlir::Value source, mlir::Value destination) {
        ResourceKind sourceKind = resourceKind(source.getType());
        if (sourceKind != ResourceKind::None && sourceKind == resourceKind(destination.getType()))
            edges.push_back({source, destination});
    };

    function.walk([&](mlir::Operation *operation) {
        if (auto select = mlir::dyn_cast<mlir::arith::SelectOp>(operation)) {
            addEdge(select.getTrueValue(), select.getResult());
            addEdge(select.getFalseValue(), select.getResult());
        } else if (mlir::isa<mlir::UnrealizedConversionCastOp>(operation) && operation->getNumOperands() == 1 &&
                   operation->getNumResults() == 1) {
            addEdge(operation->getOperand(0), operation->getResult(0));
        } else if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(operation)) {
            if (!ifOp.getResults().empty()) {
                auto thenYield = mlir::cast<mlir::scf::YieldOp>(ifOp.thenBlock()->getTerminator());
                auto elseYield = mlir::cast<mlir::scf::YieldOp>(ifOp.elseBlock()->getTerminator());
                for (auto [result, thenValue, elseValue] :
                     llvm::zip_equal(ifOp.getResults(), thenYield.getOperands(), elseYield.getOperands())) {
                    addEdge(thenValue, result);
                    addEdge(elseValue, result);
                }
            }
        } else if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(operation)) {
            for (auto [init, iterArgument, yielded, result] : llvm::zip_equal(
                     forOp.getInitArgs(), forOp.getRegionIterArgs(), forOp.getYieldedValues(), forOp.getResults())) {
                addEdge(init, iterArgument);
                addEdge(yielded, iterArgument);
                addEdge(init, result);
                addEdge(yielded, result);
            }
        } else if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(operation)) {
            auto condition = mlir::cast<mlir::scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
            auto yield = mlir::cast<mlir::scf::YieldOp>(whileOp.getAfter().front().getTerminator());
            for (auto [init, beforeArgument, conditionValue, afterArgument, yielded, result] :
                 llvm::zip_equal(whileOp.getInits(), whileOp.getBeforeArguments(), condition.getArgs(),
                                 whileOp.getAfterArguments(), yield.getOperands(), whileOp.getResults())) {
                addEdge(init, beforeArgument);
                addEdge(yielded, beforeArgument);
                addEdge(conditionValue, afterArgument);
                addEdge(conditionValue, result);
            }
        }

        auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(operation);
        if (!branch)
            return;
        for (auto [successorIndex, successor] : llvm::enumerate(operation->getSuccessors())) {
            mlir::SuccessorOperands successorOperands = branch.getSuccessorOperands(successorIndex);
            for (unsigned argumentIndex = successorOperands.getProducedOperandCount();
                 argumentIndex < successorOperands.size() && argumentIndex < successor->getNumArguments();
                 ++argumentIndex)
                addEdge(successorOperands[argumentIndex], successor->getArgument(argumentIndex));
        }
    });

    bool changed = true;
    while (changed) {
        changed = false;
        for (const ForwardingEdge &edge : edges) {
            const ResourceOrigins &source = provenance[edge.source];
            ResourceOrigins &destination = provenance[edge.destination];
            size_t previousSize = destination.size();
            destination.insert(source.begin(), source.end());
            changed |= destination.size() != previousSize;
        }
    }

    std::map<unsigned, std::set<SampledTextureBinding>> samplerBindings;
    std::map<SampledTextureBinding, std::set<unsigned>> bindingSamplers;
    bool invalid = false;
    function.walk([&](mlir::vernon::IntrinsicOp intrinsic) {
        if (intrinsic.getName() != "texture_sample")
            return;
        const ResourceOrigins &textureOrigins = provenance[intrinsic.getOperand(0)];
        const ResourceOrigins &samplerOrigins = provenance[intrinsic.getOperand(1)];
        if (textureOrigins.empty() || samplerOrigins.empty()) {
            intrinsic.emitError() << "cannot resolve texture_sample "
                                  << (textureOrigins.empty() && samplerOrigins.empty() ? "texture and sampler"
                                      : textureOrigins.empty()                         ? "texture"
                                                                                       : "sampler")
                                  << " provenance to entry arguments";
            invalid = true;
            return;
        }

        for (unsigned textureOrigin : textureOrigins) {
            if (textureOrigin >= function.getNumArguments() ||
                resourceKind(function.getArgumentTypes()[textureOrigin]) != ResourceKind::Texture) {
                intrinsic.emitError() << "texture_sample texture provenance includes non-texture entry "
                                         "argument #"
                                      << textureOrigin;
                invalid = true;
                continue;
            }
            auto attrs = function.getArgAttrDict(textureOrigin);
            auto descriptorSet = attrs.getAs<mlir::IntegerAttr>("vernon.set");
            auto binding = attrs.getAs<mlir::IntegerAttr>("vernon.binding");
            if (!descriptorSet || !binding) {
                intrinsic.emitError() << "texture_sample texture entry argument #" << textureOrigin
                                      << " has no descriptor set/binding";
                invalid = true;
                continue;
            }
            SampledTextureBinding sampledBinding{descriptorSet.getInt(), binding.getInt()};
            for (unsigned samplerOrigin : samplerOrigins) {
                if (samplerOrigin >= function.getNumArguments() ||
                    resourceKind(function.getArgumentTypes()[samplerOrigin]) != ResourceKind::Sampler) {
                    intrinsic.emitError() << "texture_sample sampler provenance includes non-sampler "
                                             "entry argument #"
                                          << samplerOrigin;
                    invalid = true;
                    continue;
                }
                samplerBindings[samplerOrigin].insert(sampledBinding);
                bindingSamplers[sampledBinding].insert(samplerOrigin);
            }
        }
    });
    if (invalid)
        return mlir::failure();

    for (const auto &[binding, samplers] : bindingSamplers) {
        if (samplers.size() <= 1)
            continue;
        std::string samplerList;
        llvm::raw_string_ostream stream(samplerList);
        llvm::interleaveComma(samplers, stream, [&](unsigned sampler) { stream << '#' << sampler; });
        function.emitError() << "sampled texture at set " << binding.descriptorSet << ", binding " << binding.binding
                             << " may use multiple sampler entry arguments (" << samplerList
                             << "); the current combined-image SPIR-V model requires one sampler "
                                "origin per sampled texture binding";
        return mlir::failure();
    }
    return samplerBindings;
}

} // namespace

mlir::FailureOr<llvm::json::Object> reflectCanonicalValueLayout(mlir::ModuleOp module, mlir::Type type,
                                                                llvm::ArrayRef<llvm::StringRef> logicalDtypes) {
    mlir::FailureOr<mlir::vernon::ValueAbiLayout> planned =
        mlir::vernon::getValueAbiLayout(type, module, logicalDtypes);
    if (mlir::failed(planned))
        return mlir::failure();
    std::string logicalType;
    llvm::raw_string_ostream typeStream(logicalType);
    type.print(typeStream);
    typeStream.flush();
    llvm::json::Object reflected = reflectCanonicalValueLayout(*planned, logicalType);
    if (auto structure = mlir::dyn_cast<mlir::vernon::StructType>(type))
        reflected["struct_name"] = structure.getName().str();
    return reflected;
}

llvm::json::Object reflectCanonicalValueLayout(const mlir::vernon::ValueAbiLayout &layout,
                                               llvm::StringRef logicalType) {
    llvm::json::Object reflected;
    reflected["logical_type"] = logicalType.str();
    reflected["byte_size"] = static_cast<int64_t>(layout.size);
    reflected["alignment"] = static_cast<int64_t>(layout.alignment);
    reflected["layout_hash"] = layout.layoutHash;
    llvm::json::Array leaves;
    for (const mlir::vernon::ValueAbiLeaf &leaf : layout.leaves) {
        llvm::json::Object reflectedLeaf;
        llvm::json::Array path;
        for (const mlir::vernon::ValueAbiPathComponent &component : leaf.path)
            if (component.field)
                path.emplace_back(*component.field);
            else
                path.emplace_back(static_cast<int64_t>(component.index));
        reflectedLeaf["path"] = std::move(path);
        reflectedLeaf["dtype"] = leaf.dtype;
        reflectedLeaf["byte_offset"] = static_cast<int64_t>(leaf.byteOffset);
        reflectedLeaf["scalar_count"] = static_cast<int64_t>(leaf.scalarCount);
        llvm::json::Array shape;
        for (uint64_t extent : leaf.shape)
            shape.emplace_back(static_cast<int64_t>(extent));
        reflectedLeaf["shape"] = std::move(shape);
        leaves.emplace_back(std::move(reflectedLeaf));
    }
    reflected["leaves"] = std::move(leaves);
    return reflected;
}

mlir::FailureOr<std::string> buildReflection(mlir::ModuleOp module, const LogicalReflectionModel &logical,
                                             const std::vector<PhysicalEntryModel> &physicalEntries,
                                             const std::vector<PhysicalEntryProvenance> &provenance) {
    std::map<std::string, const LogicalEntryModel *> logicalEntries;
    for (const LogicalEntryModel &entry : logical.entries)
        logicalEntries.emplace(entry.name, &entry);
    std::map<std::string, const PhysicalEntryModel *> physicalEntryTable;
    for (const PhysicalEntryModel &entry : physicalEntries)
        if (!physicalEntryTable.emplace(entry.name, &entry).second) {
            module.emitError() << "duplicate physical entry model for '" << entry.name << "'";
            return mlir::failure();
        }
    std::map<std::string, const PhysicalEntryProvenance *> provenanceTable;
    for (const PhysicalEntryProvenance &entry : provenance)
        if (!provenanceTable.emplace(entry.name, &entry).second) {
            module.emitError() << "duplicate logical-to-physical provenance for '" << entry.name << "'";
            return mlir::failure();
        }
    const auto languageDtype = [](llvm::ArrayRef<llvm::StringRef> leaves, mlir::StringAttr sugar) -> std::string {
        if (leaves.size() == 1)
            return leaves.front().str();
        if (sugar && !sugar.getValue().empty() && leaves.empty())
            return sugar.getValue().str();
        return "";
    };
    auto reflectValueLayout = [&](mlir::Type type, llvm::ArrayRef<llvm::StringRef> logicalDtypes = {})
        -> mlir::FailureOr<llvm::json::Object> { return reflectCanonicalValueLayout(module, type, logicalDtypes); };
    std::function<llvm::json::Object(const mlir::vernon::ByteTransportNode &)> reflectTransportNode;
    reflectTransportNode = [&](const mlir::vernon::ByteTransportNode &node) {
        llvm::StringRef kind = node.kind == mlir::vernon::ByteTransportNodeKind::Scalar    ? "scalar"
                               : node.kind == mlir::vernon::ByteTransportNodeKind::Product ? "product"
                                                                                           : "array";
        llvm::json::Object reflected{{"kind", kind},
                                     {"offset", static_cast<int64_t>(node.byteOffset)},
                                     {"size", static_cast<int64_t>(node.size)},
                                     {"alignment", static_cast<int64_t>(node.alignment)}};
        if (!node.representation.empty())
            reflected["representation"] = node.representation;
        if (!node.shape.empty()) {
            llvm::json::Array shape;
            for (uint64_t extent : node.shape)
                shape.emplace_back(static_cast<int64_t>(extent));
            reflected["shape"] = std::move(shape);
        }
        if (!node.byteStrides.empty()) {
            llvm::json::Array strides;
            for (uint64_t stride : node.byteStrides)
                strides.emplace_back(static_cast<int64_t>(stride));
            reflected["byte_strides"] = std::move(strides);
        }
        if (!node.children.empty()) {
            llvm::json::Array children;
            for (const auto &child : node.children)
                children.emplace_back(reflectTransportNode(*child));
            reflected["children"] = std::move(children);
        }
        return reflected;
    };
    auto reflectCpuValuePlan = [&](const mlir::vernon::CpuCallPlan &plan) {
        llvm::json::Object reflected{
            {"kind", "cpu_call"}, {"profile", "host_value"}, {"canonical_layout_hash", plan.layout.layoutHash}};
        if (plan.root)
            reflected["root"] = reflectTransportNode(*plan.root);
        return reflected;
    };
    auto reflectPhysicalPlan = [&](const mlir::vernon::BackendInterfaceAbiPlan &plan,
                                   llvm::StringRef profile) -> llvm::json::Object {
        auto reflectTree = [&](llvm::StringRef kind, llvm::StringRef hash,
                               const std::shared_ptr<const mlir::vernon::ByteTransportNode> &root) {
            return llvm::json::Object{{"kind", kind.str()},
                                      {"profile", profile.str()},
                                      {"canonical_layout_hash", hash.str()},
                                      {"root", reflectTransportNode(*root)}};
        };
        if (const auto *bytes = std::get_if<mlir::vernon::ByteTransportPlan>(&plan))
            return reflectTree("byte_transport", bytes->canonicalLayoutHash, bytes->root);
        if (const auto *uniform = std::get_if<mlir::vernon::NativeUniformPlan>(&plan))
            return reflectTree("native_uniform", uniform->canonicalLayoutHash, uniform->root);
        if (const auto *kernel = std::get_if<mlir::vernon::KernelParameterPlan>(&plan))
            return reflectTree("kernel_parameter", kernel->canonicalLayoutHash, kernel->root);
        llvm::json::Object reflected{{"profile", profile.str()}};
        if (const auto *unsupported = std::get_if<mlir::vernon::UnsupportedBackendInterfaceAbi>(&plan)) {
            reflected["kind"] = "unsupported";
            reflected["unsupported"] = unsupported->reason;
            return reflected;
        }
        const auto &resource = std::get<mlir::vernon::ResourceBindingPlan>(plan);
        reflected["kind"] = "resource_binding";
        switch (resource.kind) {
        case mlir::vernon::PhysicalResourceAbiKind::HostPointer:
            reflected["resource_kind"] = "host_pointer";
            break;
        case mlir::vernon::PhysicalResourceAbiKind::TensorViewDescriptor:
            reflected["resource_kind"] = "tensor_view_descriptor";
            break;
        case mlir::vernon::PhysicalResourceAbiKind::CudaStorageLeaves:
            reflected["resource_kind"] = "strided_memref_storage_leaves";
            break;
        case mlir::vernon::PhysicalResourceAbiKind::GraphicsStorageLeaves:
            reflected["resource_kind"] = "descriptor_storage_leaves";
            break;
        case mlir::vernon::PhysicalResourceAbiKind::GraphicsTexture:
            reflected["resource_kind"] = "image_reference";
            break;
        case mlir::vernon::PhysicalResourceAbiKind::GraphicsSampler:
            reflected["resource_kind"] = "sampler_descriptor";
            break;
        }
        if (resource.handleSize != 0) {
            reflected["size"] = static_cast<int64_t>(resource.handleSize);
            reflected["alignment"] = static_cast<int64_t>(resource.handleAlignment);
        }
        return reflected;
    };
    std::array<llvm::DenseMap<mlir::Type, mlir::vernon::BackendInterfaceAbiPlan>,
               static_cast<size_t>(mlir::vernon::PhysicalAbiProfile::Count)>
        physicalPlanCache;
    auto physicalPlan = [&](mlir::Type type, mlir::vernon::PhysicalAbiProfile profile,
                            llvm::ArrayRef<llvm::StringRef> logicalDtypes = {})
        -> mlir::FailureOr<mlir::vernon::BackendInterfaceAbiPlan> {
        auto &cache = physicalPlanCache[static_cast<size_t>(profile)];
        if (logicalDtypes.empty())
            if (auto found = cache.find(type); found != cache.end())
                return found->second;
        mlir::FailureOr<mlir::vernon::BackendInterfaceAbiPlan> planned =
            mlir::vernon::getBackendInterfaceAbiPlan(type, module, profile, logicalDtypes);
        if (mlir::failed(planned))
            return mlir::failure();
        if (!logicalDtypes.empty())
            return planned;
        auto inserted = cache.try_emplace(type, std::move(*planned));
        return inserted.first->second;
    };
    struct InterfaceExtent {
        uint64_t size;
        uint64_t alignment;
    };
    auto physicalLayout = [&](mlir::Type type, mlir::vernon::PhysicalAbiProfile profile,
                              llvm::ArrayRef<llvm::StringRef> logicalDtypes = {}) -> mlir::FailureOr<InterfaceExtent> {
        mlir::FailureOr<mlir::vernon::BackendInterfaceAbiPlan> plan = physicalPlan(type, profile, logicalDtypes);
        if (mlir::failed(plan))
            return mlir::failure();
        if (const auto *bytes = std::get_if<mlir::vernon::ByteTransportPlan>(&*plan))
            return InterfaceExtent{bytes->root->size, bytes->root->alignment};
        if (const auto *uniform = std::get_if<mlir::vernon::NativeUniformPlan>(&*plan))
            return InterfaceExtent{uniform->root->size, uniform->root->alignment};
        if (const auto *kernel = std::get_if<mlir::vernon::KernelParameterPlan>(&*plan))
            return InterfaceExtent{kernel->root->size, kernel->root->alignment};
        if (const auto *resource = std::get_if<mlir::vernon::ResourceBindingPlan>(&*plan);
            resource && resource->handleSize != 0)
            return InterfaceExtent{resource->handleSize, resource->handleAlignment};
        return mlir::failure();
    };
    llvm::json::Array entries;
    std::set<std::string> requiredFeatures(logical.requiredFeatures.begin(), logical.requiredFeatures.end());
    bool invalid = false;
    uint32_t nextGeneratedBinding = 0;
    module.walk([&](mlir::func::FuncOp function) {
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            mlir::DictionaryAttr attrs = function.getArgAttrDict(index);
            if (!attrs)
                continue;
            auto descriptorSet = attrs.getAs<mlir::IntegerAttr>("vernon.set");
            auto binding = attrs.getAs<mlir::IntegerAttr>("vernon.binding");
            if (binding && (!descriptorSet || descriptorSet.getInt() == 0))
                nextGeneratedBinding = std::max(nextGeneratedBinding, static_cast<uint32_t>(binding.getInt() + 1));
        }
    });
    module.walk([&](mlir::func::FuncOp function) {
        auto stage = function->getAttrOfType<mlir::StringAttr>("vernon.stage");
        if (!stage)
            return;
        auto physicalFound = physicalEntryTable.find(function.getSymName().str());
        if (physicalFound == physicalEntryTable.end()) {
            function.emitError("has no target-prepared physical entry model");
            invalid = true;
            return;
        }
        const PhysicalEntryModel &physicalEntry = *physicalFound->second;
        if (physicalEntry.stage != stage.getValue() || physicalEntry.arguments.size() != function.getNumArguments() ||
            physicalEntry.results.size() != function.getNumResults()) {
            function.emitError("does not match its target-prepared physical entry model");
            invalid = true;
            return;
        }
        auto provenanceFound = provenanceTable.find(function.getSymName().str());
        if (provenanceFound == provenanceTable.end() ||
            provenanceFound->second->arguments.size() != function.getNumArguments() ||
            provenanceFound->second->results.size() != function.getNumResults()) {
            function.emitError("does not match its logical-to-physical provenance");
            invalid = true;
            return;
        }
        const PhysicalEntryProvenance &entryProvenance = *provenanceFound->second;
        const LogicalEntryModel *logicalEntry = nullptr;
        if (auto logicalFound = logicalEntries.find(function.getSymName().str()); logicalFound != logicalEntries.end())
            logicalEntry = logicalFound->second;
        if (stage.getValue() == "compute")
            requiredFeatures.insert("compute");

        mlir::FailureOr<std::map<unsigned, std::set<SampledTextureBinding>>> sampledBindings =
            analyzeSampledTextureBindings(function);
        if (mlir::failed(sampledBindings)) {
            invalid = true;
            return;
        }

        llvm::json::Array arguments;
        std::map<std::string, std::string> reflectedTensorViewOwners;
        uint64_t argumentOffset = 0;
        std::map<unsigned, uint32_t> generatedUniformBindings;
        std::map<unsigned, InterfaceExtent> physicalValueLayouts;
        uint64_t inlineUniformSize = 0;
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            if (stage.getValue() == "compute")
                break;
            mlir::DictionaryAttr attrs = function.getArgAttrDict(index);
            auto interfaceKind = attrs.getAs<mlir::StringAttr>("vernon.interface");
            if (!interfaceKind || interfaceKind.getValue() != "uniform")
                continue;
            mlir::Type type = function.getArgumentTypes()[index];
            llvm::SmallVector<llvm::StringRef> uniformDtypes;
            if (auto dtypes = attrs.getAs<mlir::ArrayAttr>("vernon.abi_leaf_dtypes"))
                for (mlir::Attribute dtype : dtypes)
                    if (auto value = mlir::dyn_cast<mlir::StringAttr>(dtype))
                        uniformDtypes.push_back(value.getValue());
            if (uniformDtypes.empty())
                if (auto sugar = attrs.getAs<mlir::StringAttr>("vernon.dtype"); sugar && !sugar.getValue().empty())
                    uniformDtypes.push_back(sugar.getValue());
            mlir::FailureOr<InterfaceExtent> layout =
                physicalLayout(type, mlir::vernon::PhysicalAbiProfile::VulkanPushConstant, uniformDtypes);
            if (mlir::failed(layout)) {
                function.emitError() << "cannot plan graphics uniform layout for argument #" << index;
                invalid = true;
                return;
            }
            bool requiresBuffer = mlir::isa<mlir::vernon::TensorType>(type);
            if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type))
                requiresBuffer = tensor.getRank() > 2;
            if (attrs.get("vernon.binding")) {
                layout = physicalLayout(type,
                                        mlir::isa<mlir::vernon::TensorType>(type)
                                            ? mlir::vernon::PhysicalAbiProfile::VulkanStd430StorageBuffer
                                            : mlir::vernon::PhysicalAbiProfile::VulkanStd140UniformBuffer,
                                        uniformDtypes);
                if (mlir::failed(layout)) {
                    function.emitError() << "cannot plan descriptor-backed uniform layout for argument #" << index;
                    invalid = true;
                    return;
                }
                physicalValueLayouts.emplace(index, std::move(*layout));
                continue;
            }
            const uint64_t offset = llvm::alignTo(inlineUniformSize, layout->alignment);
            requiresBuffer = requiresBuffer || offset > 128 || layout->size > 128 - std::min<uint64_t>(offset, 128);
            if (requiresBuffer) {
                generatedUniformBindings[index] = nextGeneratedBinding++;
                layout = physicalLayout(type,
                                        mlir::isa<mlir::vernon::TensorType>(type)
                                            ? mlir::vernon::PhysicalAbiProfile::VulkanStd430StorageBuffer
                                            : mlir::vernon::PhysicalAbiProfile::VulkanStd140UniformBuffer,
                                        uniformDtypes);
                if (mlir::failed(layout)) {
                    function.emitError() << "cannot plan generated uniform buffer layout for argument #" << index;
                    invalid = true;
                    return;
                }
            } else {
                inlineUniformSize = offset + layout->size;
            }
            physicalValueLayouts.emplace(index, std::move(*layout));
        }
        llvm::SmallVector<std::optional<uint32_t>> computeBindings(function.getNumArguments());
        llvm::SmallVector<std::optional<uint32_t>> tensorDescriptorBindings(function.getNumArguments());
        if (stage.getValue() == "compute") {
            uint32_t flattenedBinding = 0;
            for (unsigned index = 0; index < function.getNumArguments(); ++index) {
                mlir::DictionaryAttr attrs = function.getArgAttrDict(index);
                if (attrs.get("vernon.builtin") || attrs.get(mlir::vernon::kTensorDescriptorOwnerAttrName))
                    continue;
                computeBindings[index] = flattenedBinding;
                size_t leafCount = 1;
                mlir::Type storageElement;
                if (auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(function.getArgumentTypes()[index]))
                    storageElement = view.getElementType();
                if (storageElement) {
                    mlir::FailureOr<mlir::vernon::ValueAbiLayout> layout =
                        mlir::vernon::getValueStorageLayout(storageElement, module);
                    if (mlir::failed(layout)) {
                        function.emitError("cannot assign flattened compute bindings");
                        invalid = true;
                        return;
                    }
                    leafCount = layout->leaves.size();
                }
                flattenedBinding += static_cast<uint32_t>(std::max<size_t>(leafCount, 1));
            }
            for (unsigned index = 0; index < function.getNumArguments(); ++index) {
                auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(function.getArgumentTypes()[index]);
                if (!view || function.getArgAttrDict(index).get("vernon.builtin"))
                    continue;
                tensorDescriptorBindings[index] = flattenedBinding;
                flattenedBinding += 1 + 2 * static_cast<uint32_t>(view.getShape().size());
            }
        }
        uint32_t nextBinding = 0;
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            mlir::DictionaryAttr attrs = function.getArgAttrDict(index);
            if (auto binding = attrs.getAs<mlir::IntegerAttr>("vernon.binding"))
                nextBinding = std::max(nextBinding, static_cast<uint32_t>(binding.getInt() + 1));
        }
        bool sawTensorDescriptorArgument = false;
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            const PhysicalArgumentModel &physicalArgument = physicalEntry.arguments[index];
            std::string physicalType;
            llvm::raw_string_ostream physicalTypeStream(physicalType);
            function.getArgumentTypes()[index].print(physicalTypeStream);
            physicalTypeStream.flush();
            auto builtin = function.getArgAttrOfType<mlir::StringAttr>(index, "vernon.builtin");
            const auto descriptorOwner =
                function.getArgAttrOfType<mlir::IntegerAttr>(index, mlir::vernon::kTensorDescriptorOwnerAttrName);
            const auto descriptorComponent =
                function.getArgAttrOfType<mlir::StringAttr>(index, mlir::vernon::kTensorDescriptorComponentAttrName);
            const auto descriptorDimension =
                function.getArgAttrOfType<mlir::IntegerAttr>(index, mlir::vernon::kTensorDescriptorDimensionAttrName);
            const std::optional<unsigned> actualDescriptorOwner =
                descriptorOwner && descriptorOwner.getInt() >= 0
                    ? std::optional<unsigned>(static_cast<unsigned>(descriptorOwner.getInt()))
                    : std::nullopt;
            const std::optional<unsigned> actualDescriptorDimension =
                descriptorDimension && descriptorDimension.getInt() >= 0
                    ? std::optional<unsigned>(static_cast<unsigned>(descriptorDimension.getInt()))
                    : std::nullopt;
            if (physicalArgument.index != index || physicalArgument.type != physicalType ||
                physicalArgument.builtin != (builtin ? builtin.getValue().str() : std::string()) ||
                physicalArgument.descriptorOwner != actualDescriptorOwner ||
                physicalArgument.descriptorComponent !=
                    (descriptorComponent ? descriptorComponent.getValue().str() : std::string()) ||
                physicalArgument.descriptorDimension != actualDescriptorDimension) {
                function.emitError() << "argument #" << index << " does not match its physical entry model";
                invalid = true;
                return;
            }
            std::optional<unsigned> expectedLogicalIndex;
            std::string expectedLogicalPath;
            if (const std::optional<LogicalValueOrigin> &origin = entryProvenance.arguments[index]) {
                expectedLogicalIndex = origin->index;
                expectedLogicalPath = origin->path;
                if (!logicalEntry || origin->index >= logicalEntry->arguments.size() ||
                    logicalEntry->arguments[origin->index].sourcePath != origin->path) {
                    function.emitError() << "argument #" << index << " has invalid logical provenance";
                    invalid = true;
                    return;
                }
            }
            if (physicalArgument.logicalIndex != expectedLogicalIndex ||
                physicalArgument.logicalPath != expectedLogicalPath) {
                function.emitError() << "argument #" << index << " has a stale logical origin";
                invalid = true;
                return;
            }
            const LogicalValueModel *logicalValue = nullptr;
            if (logicalEntry && physicalArgument.logicalIndex &&
                *physicalArgument.logicalIndex < logicalEntry->arguments.size())
                logicalValue = &logicalEntry->arguments[*physicalArgument.logicalIndex];
            if (mlir::vernon::containsLogicalAutodiffHandle(function.getArgumentTypes()[index]))
                continue;
            if (function.getArgAttr(index, mlir::vernon::kTensorDescriptorOwnerAttrName) ||
                function.getArgAttr(index, mlir::vernon::kTensorDescriptorComponentAttrName) ||
                function.getArgAttr(index, mlir::vernon::kTensorDescriptorDimensionAttrName)) {
                sawTensorDescriptorArgument = true;
                continue;
            }
            if (sawTensorDescriptorArgument) {
                function.emitError() << "contains a visible argument after projected TensorView descriptor arguments";
                invalid = true;
                return;
            }
            llvm::json::Object argument;
            argument["index"] = static_cast<int64_t>(index);
            argument["kind"] = "scalar";

            std::string type;
            llvm::raw_string_ostream typeStream(type);
            function.getArgumentTypes()[index].print(typeStream);
            argument["type"] = std::move(type);
            mlir::DictionaryAttr argumentAttrs = function.getArgAttrDict(index);
            mlir::Type argumentType = function.getArgumentTypes()[index];
            const auto leafDtypes = [&](llvm::StringRef attribute) {
                llvm::SmallVector<llvm::StringRef> values;
                if (argumentAttrs)
                    if (auto dtypes = argumentAttrs.getAs<mlir::ArrayAttr>(attribute))
                        for (mlir::Attribute dtype : dtypes) {
                            auto value = mlir::dyn_cast<mlir::StringAttr>(dtype);
                            values.push_back(value ? value.getValue() : llvm::StringRef());
                        }
                return values;
            };
            llvm::SmallVector<llvm::StringRef> valueLogicalDtypes = leafDtypes("vernon.abi_leaf_dtypes");
            if (valueLogicalDtypes.empty())
                if (auto sugar = argumentAttrs ? argumentAttrs.getAs<mlir::StringAttr>("vernon.dtype") : nullptr;
                    sugar && !sugar.getValue().empty())
                    valueLogicalDtypes.push_back(sugar.getValue());
            if (valueLogicalDtypes.empty() && logicalValue)
                for (const std::string &dtype : logicalValue->leafDtypes)
                    valueLogicalDtypes.push_back(dtype);
            llvm::SmallVector<llvm::StringRef> explicitElementLogicalDtypes =
                leafDtypes("vernon.element_abi_leaf_dtypes");
            if (explicitElementLogicalDtypes.empty() && valueLogicalDtypes.size() == 1)
                explicitElementLogicalDtypes = valueLogicalDtypes;
            const llvm::ArrayRef<llvm::StringRef> interfaceLogicalDtypes =
                mlir::isa<mlir::vernon::TensorViewType>(argumentType)
                    ? llvm::ArrayRef<llvm::StringRef>(explicitElementLogicalDtypes)
                    : llvm::ArrayRef<llvm::StringRef>(valueLogicalDtypes);
            llvm::json::Object physicalLayouts;
            uint64_t hostAlignment = 1;
            const bool cpuOpaqueBuiltin = argumentType.isIndex() && argumentAttrs.get("vernon.builtin");
            mlir::FailureOr<mlir::vernon::CpuCallPlan> cpuValuePlan =
                mlir::vernon::getCpuCallPlan(argumentType, module, valueLogicalDtypes);
            if (cpuOpaqueBuiltin) {
                hostAlignment = alignof(uintptr_t);
                argumentOffset = llvm::alignTo(argumentOffset, hostAlignment);
                mlir::FailureOr<mlir::vernon::BackendInterfaceAbiPlan> indexPlan =
                    mlir::vernon::getBackendInterfaceAbiPlan(argumentType, module,
                                                             mlir::vernon::PhysicalAbiProfile::HostValue);
                const auto *bytes =
                    mlir::succeeded(indexPlan) ? std::get_if<mlir::vernon::ByteTransportPlan>(&*indexPlan) : nullptr;
                if (!bytes || !bytes->root) {
                    function.emitError() << "cannot reflect CPU packed ABI for argument #" << index;
                    invalid = true;
                    return;
                }
                physicalLayouts["host_value"] =
                    llvm::json::Object{{"kind", "cpu_call"},
                                       {"profile", "host_value"},
                                       {"canonical_layout_hash", bytes->canonicalLayoutHash},
                                       {"frame_offset", static_cast<int64_t>(argumentOffset)},
                                       {"root", reflectTransportNode(*bytes->root)}};
                argumentOffset += sizeof(uintptr_t);
            } else if (mlir::succeeded(cpuValuePlan)) {
                hostAlignment = cpuValuePlan->layout.alignment;
                argumentOffset = llvm::alignTo(argumentOffset, cpuValuePlan->layout.alignment);
                llvm::json::Object reflected = reflectCpuValuePlan(*cpuValuePlan);
                reflected["frame_offset"] = static_cast<int64_t>(argumentOffset);
                physicalLayouts["host_value"] = std::move(reflected);
                argumentOffset += cpuValuePlan->layout.size;
            } else {
                mlir::FailureOr<mlir::vernon::BackendInterfaceAbiPlan> cpuPlan =
                    mlir::vernon::getBackendInterfaceAbiPlan(
                        argumentType, module, mlir::vernon::PhysicalAbiProfile::HostValue, interfaceLogicalDtypes);
                mlir::FailureOr<InterfaceExtent> cpuLayout =
                    physicalLayout(argumentType, mlir::vernon::PhysicalAbiProfile::HostValue, interfaceLogicalDtypes);
                if (mlir::succeeded(cpuPlan) && mlir::succeeded(cpuLayout)) {
                    hostAlignment = cpuLayout->alignment;
                    argumentOffset = llvm::alignTo(argumentOffset, cpuLayout->alignment);
                    llvm::json::Object reflected = reflectPhysicalPlan(*cpuPlan, "host_value");
                    reflected["frame_offset"] = static_cast<int64_t>(argumentOffset);
                    physicalLayouts["host_value"] = std::move(reflected);
                    argumentOffset += cpuLayout->size;
                }
            }
            mlir::FailureOr<mlir::vernon::BackendInterfaceAbiPlan> cudaPlan = mlir::vernon::getBackendInterfaceAbiPlan(
                argumentType, module, mlir::vernon::PhysicalAbiProfile::CudaKernelParameter, interfaceLogicalDtypes);
            if (mlir::succeeded(cudaPlan))
                physicalLayouts["cuda_kernel_parameter"] = reflectPhysicalPlan(*cudaPlan, "cuda_kernel_parameter");
            std::optional<std::string> graphicsStorage;
            if (auto physical = physicalValueLayouts.find(index); physical != physicalValueLayouts.end()) {
                graphicsStorage =
                    generatedUniformBindings.find(index) != generatedUniformBindings.end()
                        ? (mlir::isa<mlir::vernon::TensorType>(argumentType) ? "storage_buffer" : "uniform_buffer")
                    : argumentAttrs.get("vernon.binding") ? "uniform_buffer"
                                                          : "inline";
            } else if (stage.getValue() == "compute" &&
                       mlir::isa<mlir::RankedTensorType, mlir::vernon::TensorType>(argumentType)) {
                graphicsStorage = "storage_buffer";
            } else if (stage.getValue() == "compute" && !argumentAttrs.get("vernon.builtin") &&
                       !mlir::isa<mlir::vernon::TensorViewType, mlir::vernon::TextureType, mlir::vernon::SamplerType>(
                           argumentType)) {
                graphicsStorage = "storage_buffer";
            }
            const auto addPhysicalLayout = [&](mlir::vernon::PhysicalAbiProfile profile, llvm::StringRef name) {
                mlir::FailureOr<mlir::vernon::BackendInterfaceAbiPlan> plan =
                    mlir::vernon::getBackendInterfaceAbiPlan(argumentType, module, profile, interfaceLogicalDtypes);
                if (mlir::succeeded(plan))
                    physicalLayouts[name] = reflectPhysicalPlan(*plan, name);
            };
            addPhysicalLayout(mlir::vernon::PhysicalAbiProfile::VulkanStd140UniformBuffer,
                              "vulkan_std140_uniform_buffer");
            addPhysicalLayout(mlir::vernon::PhysicalAbiProfile::VulkanStd430StorageBuffer,
                              "vulkan_std430_storage_buffer");
            addPhysicalLayout(mlir::vernon::PhysicalAbiProfile::VulkanPushConstant, "vulkan_push_constant");
            addPhysicalLayout(mlir::vernon::PhysicalAbiProfile::OpenGLNativeUniform, "opengl_native_uniform");
            addPhysicalLayout(mlir::vernon::PhysicalAbiProfile::DirectXConstantBuffer, "directx_constant_buffer");
            addPhysicalLayout(mlir::vernon::PhysicalAbiProfile::MetalConstantBuffer, "metal_constant_buffer");
            argument["physical_layouts"] = std::move(physicalLayouts);
            if (graphicsStorage)
                argument["value_transport"] = *graphicsStorage == "inline" ? "push_constant" : *graphicsStorage;

            if (argumentAttrs) {
                for (mlir::NamedAttribute attr : argumentAttrs) {
                    argument[attr.getName().strref().str()] = attributeToJson(attr.getValue());
                    if (attr.getName().strref() == "vernon.instance_divisor")
                        requiredFeatures.insert("instancing");
                }
            }
            if (logicalValue) {
                if (logicalValue->sourcePathExplicit)
                    argument["vernon.source_name"] = logicalValue->sourcePath;
                if (logicalValue->dtypeExplicit)
                    argument["vernon.dtype"] = logicalValue->dtype;
                if (logicalValue->shapeExplicit) {
                    llvm::json::Array shape;
                    for (int64_t extent : logicalValue->shape)
                        shape.emplace_back(extent);
                    argument["vernon.source_shape"] = std::move(shape);
                }
            }
            if (auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(argumentType)) {
                if (logicalValue && logicalValue->sourcePathExplicit) {
                    reflectedTensorViewOwners.emplace(logicalValue->sourcePath, view.getAccess().str());
                } else if (argumentAttrs) {
                    if (auto sourceName = argumentAttrs.getAs<mlir::StringAttr>("vernon.source_name");
                        sourceName && !sourceName.getValue().empty())
                        reflectedTensorViewOwners.emplace(sourceName.getValue().str(), view.getAccess().str());
                }
            }
            if (auto generated = generatedUniformBindings.find(index); generated != generatedUniformBindings.end()) {
                argument["vernon.set"] = int64_t{0};
                argument["vernon.binding"] = static_cast<int64_t>(generated->second);
            }
            if (computeBindings[index]) {
                argument["vernon.set"] = int64_t{0};
                argument["vernon.binding"] = static_cast<int64_t>(*computeBindings[index]);
            }
            if (!argumentType.isIndex() &&
                !mlir::isa<mlir::vernon::TensorViewType, mlir::vernon::TextureType, mlir::vernon::SamplerType>(
                    argumentType)) {
                mlir::FailureOr<llvm::json::Object> valueLayout = reflectValueLayout(argumentType, valueLogicalDtypes);
                if (mlir::failed(valueLayout)) {
                    function.emitError() << "cannot reflect canonical logical ABI for argument #" << index;
                    invalid = true;
                    return;
                }
                argument["value_layout"] = std::move(*valueLayout);
            }
            mlir::Type elementLayoutType;
            if (auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(argumentType))
                elementLayoutType = view.getElementType();
            else if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(argumentType))
                elementLayoutType = tensor.getElementType();
            else if (auto tensor = mlir::dyn_cast<mlir::vernon::TensorType>(argumentType))
                elementLayoutType = tensor.getElementType();
            if (elementLayoutType) {
                if (elementLayoutType != argumentType && !valueLogicalDtypes.empty() &&
                    explicitElementLogicalDtypes.empty()) {
                    function.emitError() << "container argument #" << index
                                         << " has value ABI dtypes but no element ABI dtypes";
                    invalid = true;
                    return;
                }
                llvm::ArrayRef<llvm::StringRef> elementLogicalDtypes = explicitElementLogicalDtypes;
                mlir::FailureOr<llvm::json::Object> elementLayout =
                    reflectValueLayout(elementLayoutType, elementLogicalDtypes);
                if (mlir::failed(elementLayout)) {
                    function.emitError() << "cannot reflect canonical element layout for argument #" << index;
                    invalid = true;
                    return;
                }
                argument["element_layout"] = std::move(*elementLayout);
            }

            const bool aggregateVertexAttribute =
                stage.getValue() == "vertex" && argumentAttrs &&
                argumentAttrs.getAs<mlir::StringAttr>("vernon.interface") &&
                argumentAttrs.getAs<mlir::StringAttr>("vernon.interface").getValue() == "input" &&
                argumentAttrs.getAs<mlir::IntegerAttr>("vernon.location") && !argumentAttrs.get("vernon.builtin");
            if (aggregateVertexAttribute) {
                mlir::FailureOr<mlir::vernon::AttributeAbiLayout> plan =
                    mlir::vernon::getAttributeAbiLayout(argumentType, module, valueLogicalDtypes);
                if (mlir::failed(plan)) {
                    function.emitError() << "cannot reflect vertex attribute layout for argument #" << index;
                    invalid = true;
                    return;
                }
                const int64_t baseLocation = argumentAttrs.getAs<mlir::IntegerAttr>("vernon.location").getInt();
                llvm::json::Array leaves;
                for (const mlir::vernon::AttributeAbiLeaf &leaf : plan->leaves) {
                    llvm::json::Object reflectedLeaf;
                    llvm::json::Array path;
                    for (const mlir::vernon::ValueAbiPathComponent &component : leaf.path) {
                        if (component.field)
                            path.emplace_back(*component.field);
                        else
                            path.emplace_back(static_cast<int64_t>(component.index));
                    }
                    reflectedLeaf["path"] = std::move(path);
                    reflectedLeaf["location"] = baseLocation + leaf.locationOffset;
                    reflectedLeaf["location_offset"] = leaf.locationOffset;
                    reflectedLeaf["dtype"] = leaf.dtype;
                    reflectedLeaf["component_count"] = leaf.componentCount;
                    reflectedLeaf["byte_offset"] = static_cast<int64_t>(leaf.byteOffset);
                    leaves.emplace_back(std::move(reflectedLeaf));
                }
                argument["kind"] = "tensor_value";
                argument["location_span"] = static_cast<int64_t>(plan->getLocationSpan());
                argument["attribute_leaves"] = std::move(leaves);
            }
            if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(argumentType)) {
                const std::string dtype =
                    languageDtype(valueLogicalDtypes,
                                  argumentAttrs ? argumentAttrs.getAs<mlir::StringAttr>("vernon.dtype") : nullptr);
                argument["kind"] = "tensor_value";
                if (!dtype.empty())
                    argument["dtype"] = dtype;
                llvm::json::Array shape;
                for (int64_t extent : tensor.getShape())
                    shape.emplace_back(extent);
                argument["shape"] = std::move(shape);
                argument["rank"] = static_cast<int64_t>(tensor.getRank());
            }
            if (auto vector = mlir::dyn_cast<mlir::VectorType>(argumentType)) {
                argument["kind"] = "tensor_value";
                const std::string dtype =
                    languageDtype(valueLogicalDtypes,
                                  argumentAttrs ? argumentAttrs.getAs<mlir::StringAttr>("vernon.dtype") : nullptr);
                if (!dtype.empty())
                    argument["dtype"] = dtype;
                llvm::json::Array shape;
                for (int64_t extent : vector.getShape())
                    shape.emplace_back(extent);
                argument["shape"] = std::move(shape);
                argument["rank"] = static_cast<int64_t>(vector.getRank());
            }
            if (auto tensor = mlir::dyn_cast<mlir::vernon::TensorType>(argumentType)) {
                mlir::FailureOr<mlir::vernon::ValueAbiLayout> layout =
                    mlir::vernon::getValueAbiLayout(argumentType, module, valueLogicalDtypes);
                if (mlir::failed(layout)) {
                    function.emitError() << "cannot reflect canonical layout for static aggregate Tensor argument #"
                                         << index;
                    invalid = true;
                    return;
                }
                argument["kind"] = "tensor_value";
                llvm::json::Array shape;
                for (int64_t extent : tensor.getShape())
                    shape.emplace_back(extent);
                argument["shape"] = std::move(shape);
                argument["rank"] = static_cast<int64_t>(tensor.getShape().size());
            }
            if (auto texture = mlir::dyn_cast<mlir::vernon::TextureType>(argumentType)) {
                argument["kind"] = "image";
                argument["resource_kind"] = "image";
                argument["dimension"] = texture.getDimension().str();
                if (texture.getAccess() == "sampled") {
                    argument["binding_role"] = "sampled";
                    argument["sample_result_class"] = "float";
                    argument["access"] = "read";
                } else {
                    argument["binding_role"] = "storage";
                    argument["exact_storage_format"] = texture.getFormat().str();
                    argument["access"] = texture.getAccess().str();
                }
            } else if (mlir::isa<mlir::vernon::SamplerType>(argumentType)) {
                argument["kind"] = "sampler";
                argument["access"] = "read";
                auto pairs = sampledBindings->find(index);
                if (pairs != sampledBindings->end()) {
                    llvm::json::Array bindings;
                    for (const SampledTextureBinding &pair : pairs->second) {
                        llvm::json::Object reflectedBinding;
                        reflectedBinding["set"] = pair.descriptorSet;
                        reflectedBinding["binding"] = pair.binding;
                        bindings.emplace_back(std::move(reflectedBinding));
                    }
                    argument["sampled_image_bindings"] = std::move(bindings);
                }
            }
            if (stage.getValue() == "compute") {
                auto attrs = function.getArgAttrDict(index);
                auto builtin = attrs.getAs<mlir::StringAttr>("vernon.builtin");
                auto sourceDtype = attrs.getAs<mlir::StringAttr>("vernon.dtype");
                if (builtin) {
                    argument["kind"] = "builtin";
                    argument["builtin"] = builtin.getValue().str();
                    if (sourceDtype)
                        argument["dtype"] = sourceDtype.getValue().str();
                } else if (mlir::isa<mlir::vernon::TextureType, mlir::vernon::SamplerType>(argumentType)) {
                    // Resource kind and binding metadata are emitted above.
                } else if (auto tensor = mlir::dyn_cast<mlir::vernon::TensorType>(argumentType)) {
                    argument["kind"] = "tensor_value";
                    argument["access"] = "read";
                    mlir::FailureOr<mlir::vernon::ValueAbiLayout> valueLayout =
                        mlir::vernon::getValueAbiLayout(tensor, module, valueLogicalDtypes);
                    if (mlir::failed(valueLayout)) {
                        function.emitError("cannot reflect aggregate Tensor value layout");
                        invalid = true;
                        return;
                    }
                    argument["alignment"] = static_cast<int64_t>(valueLayout->alignment);
                } else if (auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(argumentType)) {
                    argument["kind"] = "tensor";
                    const std::string dtype = languageDtype(interfaceLogicalDtypes, sourceDtype);
                    if (!dtype.empty())
                        argument["dtype"] = dtype;
                    argument["access"] = view.getAccess().str();
                    argument["address_space"] = view.getAddressSpace().str();
                    argument["rank"] = static_cast<int64_t>(view.getShape().size());
                    llvm::json::Array sourceShape;
                    for (int64_t extent : view.getShape())
                        sourceShape.emplace_back(extent);
                    argument["source_shape"] = std::move(sourceShape);
                    mlir::FailureOr<mlir::vernon::ValueAbiLayout> storageLayout =
                        mlir::vernon::getValueStorageLayout(view.getElementType(), module);
                    if (mlir::failed(storageLayout)) {
                        function.emitError("cannot reflect aggregate TensorView storage layout");
                        invalid = true;
                        return;
                    }
                    argument["alignment"] = static_cast<int64_t>(storageLayout->alignment);
                    const uint32_t firstBinding =
                        computeBindings[index] ? *computeBindings[index]
                        : attrs.getAs<mlir::IntegerAttr>("vernon.binding")
                            ? static_cast<uint32_t>(attrs.getAs<mlir::IntegerAttr>("vernon.binding").getInt())
                            : index;
                    llvm::json::Array storageLeaves;
                    for (auto [leafIndex, leaf] : llvm::enumerate(storageLayout->leaves)) {
                        llvm::json::Object reflectedLeaf;
                        reflectedLeaf["element_size"] =
                            static_cast<int64_t>(std::max<uint64_t>(leaf.scalarType.getIntOrFloatBitWidth() / 8, 1));
                        reflectedLeaf["byte_offset"] = static_cast<int64_t>(leaf.byteOffset);
                        reflectedLeaf["binding"] =
                            static_cast<int64_t>(computeBindings[index] ? firstBinding + leafIndex
                                                 : leafIndex == 0       ? firstBinding
                                                                        : nextBinding++);
                        storageLeaves.emplace_back(std::move(reflectedLeaf));
                    }
                    argument["storage_leaves"] = std::move(storageLeaves);
                    if (!tensorDescriptorBindings[index]) {
                        function.emitError("TensorView has no dispatch descriptor binding sequence");
                        invalid = true;
                        return;
                    }
                    const uint32_t descriptorBase = *tensorDescriptorBindings[index];
                    llvm::json::Object descriptor;
                    descriptor["rank"] = static_cast<int64_t>(view.getShape().size());
                    descriptor["offset_binding"] = static_cast<int64_t>(descriptorBase);
                    llvm::json::Array extentBindings;
                    llvm::json::Array strideBindings;
                    for (uint32_t dimension = 0; dimension < view.getShape().size(); ++dimension) {
                        extentBindings.emplace_back(static_cast<int64_t>(descriptorBase + 1 + dimension));
                        strideBindings.emplace_back(
                            static_cast<int64_t>(descriptorBase + 1 + view.getShape().size() + dimension));
                    }
                    descriptor["extent_bindings"] = std::move(extentBindings);
                    descriptor["stride_bindings"] = std::move(strideBindings);
                    argument["tensor_view_descriptor"] = std::move(descriptor);
                } else if (!mlir::isa<mlir::RankedTensorType, mlir::VectorType>(argumentType)) {
                    argument["kind"] = "scalar";
                    if (const std::string dtype = languageDtype(valueLogicalDtypes, sourceDtype); !dtype.empty())
                        argument["dtype"] = dtype;
                    argument["alignment"] = static_cast<int64_t>(hostAlignment);
                }
            }
            if (mlir::isa<mlir::vernon::TensorViewType>(argumentType))
                requiredFeatures.insert("tensor_views");
            if (auto ownership = function.getArgAttrOfType<mlir::StringAttr>(index, "vernon.accumulation_ownership"))
                argument["vernon.accumulation_ownership"] = ownership.getValue();
            if (mlir::isa<mlir::vernon::TextureType>(argumentType))
                requiredFeatures.insert("textures");
            if (mlir::isa<mlir::vernon::SamplerType>(argumentType))
                requiredFeatures.insert("samplers");
            arguments.emplace_back(std::move(argument));
        }

        llvm::json::Array results;
        uint64_t resultSize = 0;
        for (unsigned index = 0; index < function.getNumResults(); ++index) {
            std::string physicalType;
            llvm::raw_string_ostream physicalTypeStream(physicalType);
            function.getResultTypes()[index].print(physicalTypeStream);
            physicalTypeStream.flush();
            const PhysicalResultModel &physicalResult = physicalEntry.results[index];
            if (physicalResult.index != index || physicalResult.type != physicalType) {
                function.emitError() << "result #" << index << " does not match its physical entry model";
                invalid = true;
                return;
            }
            std::optional<unsigned> expectedLogicalIndex;
            std::string expectedLogicalPath;
            if (const std::optional<LogicalValueOrigin> &origin = entryProvenance.results[index]) {
                expectedLogicalIndex = origin->index;
                expectedLogicalPath = origin->path;
                if (!logicalEntry || origin->index >= logicalEntry->results.size() ||
                    logicalEntry->results[origin->index].sourcePath != origin->path) {
                    function.emitError() << "result #" << index << " has invalid logical provenance";
                    invalid = true;
                    return;
                }
            }
            if (physicalResult.logicalIndex != expectedLogicalIndex ||
                physicalResult.logicalPath != expectedLogicalPath) {
                function.emitError() << "result #" << index << " has a stale logical origin";
                invalid = true;
                return;
            }
            const LogicalValueModel *logicalValue = nullptr;
            if (logicalEntry && physicalResult.logicalIndex &&
                *physicalResult.logicalIndex < logicalEntry->results.size())
                logicalValue = &logicalEntry->results[*physicalResult.logicalIndex];
            if (mlir::vernon::containsLogicalAutodiffHandle(function.getResultTypes()[index]))
                continue;
            llvm::json::Object output;
            output["index"] = static_cast<int64_t>(index);

            std::string type;
            llvm::raw_string_ostream typeStream(type);
            function.getResultTypes()[index].print(typeStream);
            output["type"] = std::move(type);
            mlir::DictionaryAttr resultAttrs = function.getResultAttrDict(index);
            mlir::Type resultType = function.getResultTypes()[index];
            llvm::SmallVector<llvm::StringRef> logicalDtypes;
            if (resultAttrs)
                if (auto dtypes = resultAttrs.getAs<mlir::ArrayAttr>("vernon.abi_leaf_dtypes"))
                    for (mlir::Attribute dtype : dtypes) {
                        auto value = mlir::dyn_cast<mlir::StringAttr>(dtype);
                        logicalDtypes.push_back(value ? value.getValue() : llvm::StringRef());
                    }
            if (logicalDtypes.empty())
                if (auto sugar = resultAttrs ? resultAttrs.getAs<mlir::StringAttr>("vernon.dtype") : nullptr;
                    sugar && !sugar.getValue().empty())
                    logicalDtypes.push_back(sugar.getValue());
            if (logicalDtypes.empty() && logicalValue)
                for (const std::string &dtype : logicalValue->leafDtypes)
                    logicalDtypes.push_back(dtype);
            if (logicalValue) {
                if (logicalValue->sourcePathExplicit)
                    output["vernon.source_name"] = logicalValue->sourcePath;
                if (logicalValue->dtypeExplicit)
                    output["vernon.dtype"] = logicalValue->dtype;
                if (logicalValue->shapeExplicit) {
                    llvm::json::Array shape;
                    for (int64_t extent : logicalValue->shape)
                        shape.emplace_back(extent);
                    output["vernon.source_shape"] = std::move(shape);
                }
            }
            llvm::json::Object physicalLayouts;
            mlir::FailureOr<mlir::vernon::CpuCallPlan> cpuValuePlan =
                mlir::vernon::getCpuCallPlan(resultType, module, logicalDtypes);
            if (mlir::succeeded(cpuValuePlan)) {
                resultSize = llvm::alignTo(resultSize, cpuValuePlan->layout.alignment);
                llvm::json::Object reflected = reflectCpuValuePlan(*cpuValuePlan);
                reflected["frame_offset"] = static_cast<int64_t>(resultSize);
                physicalLayouts["host_value"] = std::move(reflected);
                resultSize += cpuValuePlan->layout.size;
            } else {
                mlir::FailureOr<mlir::vernon::BackendInterfaceAbiPlan> cpuPlan =
                    mlir::vernon::getBackendInterfaceAbiPlan(
                        resultType, module, mlir::vernon::PhysicalAbiProfile::HostValue, logicalDtypes);
                mlir::FailureOr<InterfaceExtent> cpuLayout =
                    physicalLayout(resultType, mlir::vernon::PhysicalAbiProfile::HostValue, logicalDtypes);
                if (mlir::succeeded(cpuPlan) && mlir::succeeded(cpuLayout)) {
                    resultSize = llvm::alignTo(resultSize, cpuLayout->alignment);
                    llvm::json::Object reflected = reflectPhysicalPlan(*cpuPlan, "host_value");
                    reflected["frame_offset"] = static_cast<int64_t>(resultSize);
                    physicalLayouts["host_value"] = std::move(reflected);
                    resultSize += cpuLayout->size;
                }
            }
            if (mlir::FailureOr<mlir::vernon::BackendInterfaceAbiPlan> plan = mlir::vernon::getBackendInterfaceAbiPlan(
                    resultType, module, mlir::vernon::PhysicalAbiProfile::CudaKernelParameter, logicalDtypes);
                mlir::succeeded(plan))
                physicalLayouts["cuda_kernel_parameter"] = reflectPhysicalPlan(*plan, "cuda_kernel_parameter");
            if (mlir::FailureOr<mlir::vernon::BackendInterfaceAbiPlan> plan = mlir::vernon::getBackendInterfaceAbiPlan(
                    resultType, module, mlir::vernon::PhysicalAbiProfile::VulkanStd140UniformBuffer, logicalDtypes);
                mlir::succeeded(plan))
                physicalLayouts["vulkan_std140_uniform_buffer"] =
                    reflectPhysicalPlan(*plan, "vulkan_std140_uniform_buffer");
            if (mlir::FailureOr<mlir::vernon::BackendInterfaceAbiPlan> plan = mlir::vernon::getBackendInterfaceAbiPlan(
                    resultType, module, mlir::vernon::PhysicalAbiProfile::VulkanStd430StorageBuffer, logicalDtypes);
                mlir::succeeded(plan))
                physicalLayouts["vulkan_std430_storage_buffer"] =
                    reflectPhysicalPlan(*plan, "vulkan_std430_storage_buffer");
            if (mlir::FailureOr<mlir::vernon::BackendInterfaceAbiPlan> plan = mlir::vernon::getBackendInterfaceAbiPlan(
                    resultType, module, mlir::vernon::PhysicalAbiProfile::VulkanPushConstant, logicalDtypes);
                mlir::succeeded(plan))
                physicalLayouts["vulkan_push_constant"] = reflectPhysicalPlan(*plan, "vulkan_push_constant");
            if (mlir::FailureOr<mlir::vernon::BackendInterfaceAbiPlan> plan = mlir::vernon::getBackendInterfaceAbiPlan(
                    resultType, module, mlir::vernon::PhysicalAbiProfile::OpenGLNativeUniform, logicalDtypes);
                mlir::succeeded(plan))
                physicalLayouts["opengl_native_uniform"] = reflectPhysicalPlan(*plan, "opengl_native_uniform");
            if (mlir::FailureOr<mlir::vernon::BackendInterfaceAbiPlan> plan = mlir::vernon::getBackendInterfaceAbiPlan(
                    resultType, module, mlir::vernon::PhysicalAbiProfile::DirectXConstantBuffer, logicalDtypes);
                mlir::succeeded(plan))
                physicalLayouts["directx_constant_buffer"] = reflectPhysicalPlan(*plan, "directx_constant_buffer");
            if (mlir::FailureOr<mlir::vernon::BackendInterfaceAbiPlan> plan = mlir::vernon::getBackendInterfaceAbiPlan(
                    resultType, module, mlir::vernon::PhysicalAbiProfile::MetalConstantBuffer, logicalDtypes);
                mlir::succeeded(plan))
                physicalLayouts["metal_constant_buffer"] = reflectPhysicalPlan(*plan, "metal_constant_buffer");
            output["physical_layouts"] = std::move(physicalLayouts);

            if (resultAttrs) {
                for (mlir::NamedAttribute attr : resultAttrs)
                    output[attr.getName().strref().str()] = attributeToJson(attr.getValue());
            }
            mlir::FailureOr<llvm::json::Object> valueLayout = reflectValueLayout(resultType, logicalDtypes);
            if (mlir::failed(valueLayout)) {
                function.emitError() << "cannot reflect canonical logical ABI for result #" << index;
                invalid = true;
                return;
            }
            output["value_layout"] = std::move(*valueLayout);
            results.emplace_back(std::move(output));
        }

        llvm::json::Object entry;
        entry["name"] = function.getSymName().str();
        entry["symbol"] = function.getSymName().str();
        entry["stage"] = stage.getValue().str();
        entry["arguments"] = std::move(arguments);
        entry["results"] = std::move(results);
        entry["physical_layouts"] = llvm::json::Object{
            {"host_value", llvm::json::Object{{"profile", "host_value"},
                                              {"packed_arguments_size", static_cast<int64_t>(argumentOffset)},
                                              {"packed_results_size", static_cast<int64_t>(resultSize)}}},
            {"cuda_kernel_parameter",
             llvm::json::Object{{"profile", "cuda_kernel_parameter"}, {"packing", "kernel_parameters"}}},
            {"vulkan_std140_uniform_buffer",
             llvm::json::Object{{"profile", "vulkan_std140_uniform_buffer"}, {"packing", "resource_bindings"}}},
            {"vulkan_std430_storage_buffer",
             llvm::json::Object{{"profile", "vulkan_std430_storage_buffer"}, {"packing", "resource_bindings"}}},
            {"vulkan_push_constant",
             llvm::json::Object{{"profile", "vulkan_push_constant"}, {"packing", "push_constants"}}},
            {"opengl_native_uniform",
             llvm::json::Object{{"profile", "opengl_native_uniform"}, {"packing", "native_uniforms"}}},
            {"directx_constant_buffer",
             llvm::json::Object{{"profile", "directx_constant_buffer"}, {"packing", "constant_buffer"}}},
            {"metal_constant_buffer",
             llvm::json::Object{{"profile", "metal_constant_buffer"}, {"packing", "constant_buffer"}}},
        };
        if (auto workgroup = function->getAttrOfType<mlir::DenseI32ArrayAttr>("vernon.workgroup_size")) {
            llvm::json::Array dimensions;
            for (int32_t dimension : workgroup.asArrayRef())
                dimensions.emplace_back(static_cast<int64_t>(dimension));
            entry["workgroup_size"] = std::move(dimensions);
        }
        if (stage.getValue() == "compute") {
            auto contract = function->getAttrOfType<mlir::DictionaryAttr>(mlir::vernon::kDispatchContractAttrName);
            auto axes = contract ? contract.getAs<mlir::DenseI32ArrayAttr>("unit_grid_axes") : nullptr;
            auto unitWorkgroup = contract ? contract.getAs<mlir::BoolAttr>("requires_unit_workgroup") : nullptr;
            if (!contract || !axes || !unitWorkgroup) {
                function.emitError("has no validated dispatch contract");
                invalid = true;
                return;
            }
            llvm::json::Array reflectedAxes;
            for (int32_t axis : axes.asArrayRef())
                reflectedAxes.emplace_back(static_cast<int64_t>(axis));
            entry["dispatch_contract"] = llvm::json::Object{{"unit_grid_axes", std::move(reflectedAxes)},
                                                            {"requires_unit_workgroup", unitWorkgroup.getValue()}};
        }
        llvm::json::Array effects;
        llvm::json::Array writeFootprints;
        if (auto reflectedEffects = function->getAttrOfType<mlir::ArrayAttr>("vernon.storage_effects")) {
            for (mlir::Attribute reflectedEffect : reflectedEffects) {
                auto effect = mlir::dyn_cast<mlir::DictionaryAttr>(reflectedEffect);
                if (!effect)
                    continue;
                auto kind = effect.getAs<mlir::StringAttr>("kind");
                auto owner = effect.getAs<mlir::StringAttr>("owner");
                auto region = effect.getAs<mlir::StringAttr>("region");
                if (!kind || !owner || !region)
                    continue;
                const auto reflectedOwner = reflectedTensorViewOwners.find(owner.getValue().str());
                if (reflectedOwner == reflectedTensorViewOwners.end()) {
                    function.emitError() << "storage effect owner '" << owner.getValue()
                                         << "' does not name a reflected TensorView argument";
                    invalid = true;
                    return;
                }
                const llvm::StringRef effectKind = kind.getValue();
                const llvm::StringRef declaredAccess = reflectedOwner->second;
                const bool incompatible = (declaredAccess == "read" && effectKind != "read") ||
                                          (declaredAccess == "write" && effectKind == "read") ||
                                          (effectKind == "atomic" && declaredAccess != "read_write");
                if (incompatible) {
                    function.emitError() << "storage effect '" << effectKind << "' for owner '" << owner.getValue()
                                         << "' conflicts with declared access '" << declaredAccess << "'";
                    invalid = true;
                    return;
                }
                llvm::json::Object reflected;
                reflected["kind"] = kind.getValue().str();
                reflected["owner"] = owner.getValue().str();
                reflected["region"] = region.getValue().str();
                if (auto atomic = effect.getAs<mlir::BoolAttr>("atomic"))
                    reflected["atomic"] = atomic.getValue();
                llvm::json::Array indices;
                if (auto values = effect.getAs<mlir::DenseI64ArrayAttr>("indices"))
                    for (int64_t index : values.asArrayRef())
                        indices.emplace_back(index);
                reflected["indices"] = std::move(indices);
                effects.emplace_back(std::move(reflected));
                if (kind.getValue() != "read") {
                    llvm::json::Object footprint;
                    footprint["version"] = int64_t{1};
                    footprint["owner"] = owner.getValue().str();
                    footprint["kind"] = region.getValue() == "element" ? "exact_element" : "whole_view";
                    llvm::json::Array footprintIndices;
                    if (auto values = effect.getAs<mlir::DenseI64ArrayAttr>("indices"))
                        for (int64_t index : values.asArrayRef())
                            footprintIndices.emplace_back(index);
                    footprint["indices"] = std::move(footprintIndices);
                    writeFootprints.emplace_back(std::move(footprint));
                }
            }
        }
        entry["effects"] = std::move(effects);
        entry["tensor_view_write_footprints"] = std::move(writeFootprints);
        entries.emplace_back(std::move(entry));
    });
    if (invalid)
        return mlir::failure();

    llvm::json::Object root;
    root["compiler_contract_version"] = int64_t{VERNON_COMPILER_CONTRACT_VERSION};
    root["pipeline_version"] = int64_t{VERNON_PIPELINE_VERSION};
    root["entries"] = std::move(entries);
    mlir::FailureOr<std::optional<ProgramReflection>> programReflection = buildProgramReflection(module);
    if (mlir::failed(programReflection))
        return mlir::failure();
    if (*programReflection) {
        root["program_plan"] = std::move((*programReflection)->plan);
        root["kernel_compile_requests"] = std::move((*programReflection)->kernelCompileRequests);
    }
    root["artifacts"] = llvm::json::Array();
    llvm::json::Array reflectedStructs;
    for (const LogicalStructLayout &planned : logical.structLayouts) {
        llvm::json::Object layout;
        layout["name"] = planned.name;
        layout["size"] = static_cast<int64_t>(planned.size);
        layout["alignment"] = static_cast<int64_t>(planned.alignment);
        llvm::json::Array offsets;
        for (uint64_t offset : planned.fieldOffsets)
            offsets.emplace_back(static_cast<int64_t>(offset));
        layout["field_offsets"] = std::move(offsets);
        llvm::json::Array fields;
        for (const std::string &field : planned.fields)
            fields.emplace_back(field);
        layout["fields"] = std::move(fields);
        reflectedStructs.emplace_back(std::move(layout));
    }
    root["struct_layouts"] = std::move(reflectedStructs);
    llvm::json::Array dependencies;
    for (const LogicalDependency &value : logical.dependencies) {
        llvm::json::Object dependency;
        dependency["path"] = value.path;
        dependency["sha256"] = value.sha256;
        dependencies.emplace_back(std::move(dependency));
    }
    root["dependencies"] = std::move(dependencies);
    llvm::json::Array features;
    for (const std::string &feature : requiredFeatures)
        features.emplace_back(feature);
    root["required_features"] = std::move(features);
    root["module_hash"] = logical.moduleHash;

    std::string output;
    llvm::raw_string_ostream stream(output);
    stream << llvm::json::Value(std::move(root));
    return output;
}

bool selectTargetPhysicalLayouts(std::string &reflection, VernonTarget target, std::string &diagnostics) {
    std::set<std::string> allowed;
    switch (target) {
    case VERNON_TARGET_CPU:
        allowed.emplace("host_value");
        break;
    case VERNON_TARGET_CUDA:
        allowed.emplace("cuda_kernel_parameter");
        break;
    case VERNON_TARGET_VULKAN:
        allowed.emplace("vulkan_std140_uniform_buffer");
        allowed.emplace("vulkan_std430_storage_buffer");
        allowed.emplace("vulkan_push_constant");
        break;
    case VERNON_TARGET_OPENGL:
    case VERNON_TARGET_OPENGL_ES:
        allowed.emplace("vulkan_std140_uniform_buffer");
        allowed.emplace("vulkan_std430_storage_buffer");
        allowed.emplace("opengl_native_uniform");
        break;
    case VERNON_TARGET_DIRECTX:
        allowed.emplace("vulkan_std430_storage_buffer");
        allowed.emplace("directx_constant_buffer");
        break;
    case VERNON_TARGET_METAL:
        allowed.emplace("vulkan_std430_storage_buffer");
        allowed.emplace("metal_constant_buffer");
        break;
    }

    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(reflection);
    llvm::json::Object *root = parsed ? parsed->getAsObject() : nullptr;
    llvm::json::Array *entries = root ? root->getArray("entries") : nullptr;
    if (!entries) {
        if (!parsed)
            llvm::consumeError(parsed.takeError());
        diagnostics = "compiler reflection has no entry table";
        return false;
    }
    const auto prune = [](llvm::json::Object &layouts, const std::set<std::string> &profiles) {
        std::vector<std::string> removed;
        for (const auto &[name, _] : layouts)
            if (!profiles.count(name.str()))
                removed.push_back(name.str());
        for (const std::string &name : removed)
            layouts.erase(name);
    };
    const auto selectedProfile = [&](const llvm::json::Object &row) -> std::optional<std::string> {
        if (target == VERNON_TARGET_CPU)
            return "host_value";
        if (target == VERNON_TARGET_CUDA)
            return "cuda_kernel_parameter";
        if (target == VERNON_TARGET_METAL)
            return "metal_constant_buffer";
        std::string transport = row.getString("value_transport").value_or("").str();
        if (transport.empty() && row.getString("vernon.interface").value_or("") == "resource")
            transport = "storage_buffer";
        if (transport == "storage_buffer")
            return "vulkan_std430_storage_buffer";
        if (target == VERNON_TARGET_VULKAN) {
            if (transport == "uniform_buffer")
                return "vulkan_std140_uniform_buffer";
            if (transport == "push_constant")
                return "vulkan_push_constant";
        } else if (target == VERNON_TARGET_OPENGL || target == VERNON_TARGET_OPENGL_ES) {
            if (transport == "uniform_buffer")
                return "vulkan_std140_uniform_buffer";
            if (transport == "push_constant" || transport == "native_uniform")
                return "opengl_native_uniform";
        } else if (target == VERNON_TARGET_DIRECTX && (transport == "uniform_buffer" || transport == "push_constant")) {
            return "directx_constant_buffer";
        }
        return std::nullopt;
    };
    for (llvm::json::Value &entryValue : *entries) {
        llvm::json::Object *entry = entryValue.getAsObject();
        llvm::json::Object *entryLayouts = entry ? entry->getObject("physical_layouts") : nullptr;
        if (!entry || !entryLayouts) {
            diagnostics = "compiler reflection entry has no physical layout table";
            return false;
        }
        std::set<std::string> retainedProfiles;
        for (llvm::StringRef field : {"arguments", "results"}) {
            llvm::json::Array *rows = entry->getArray(field);
            if (!rows) {
                diagnostics = "compiler reflection entry has no " + field.str() + " table";
                return false;
            }
            for (llvm::json::Value &rowValue : *rows) {
                llvm::json::Object *row = rowValue.getAsObject();
                llvm::json::Object *layouts = row ? row->getObject("physical_layouts") : nullptr;
                if (!row || !layouts) {
                    diagnostics = "compiler reflection value has no physical layout table";
                    return false;
                }
                const std::optional<std::string> profile = selectedProfile(*row);
                if (profile) {
                    const std::set<std::string> selected{*profile};
                    prune(*layouts, selected);
                } else {
                    prune(*layouts, allowed);
                }
                for (const auto &[name, _] : *layouts)
                    retainedProfiles.insert(name.str());
            }
        }
        prune(*entryLayouts, retainedProfiles.empty() ? allowed : retainedProfiles);
    }
    reflection.clear();
    llvm::raw_string_ostream stream(reflection);
    stream << llvm::json::Value(std::move(*root));
    return true;
}

bool setCpuReflectionSymbols(std::string &reflection, const std::vector<vernon::CpuAbiWrapperMetadata> &metadata) {
    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(reflection);
    if (!parsed)
        return false;
    llvm::json::Object *root = parsed->getAsObject();
    llvm::json::Array *entries = root ? root->getArray("entries") : nullptr;
    if (!entries)
        return false;
    for (llvm::json::Value &entryValue : *entries) {
        llvm::json::Object *entry = entryValue.getAsObject();
        std::optional<llvm::StringRef> name = entry ? entry->getString("name") : std::nullopt;
        if (!name)
            return false;
        auto found =
            llvm::find_if(metadata, [&](const auto &candidate) { return candidate.internalFunctionSymbol == *name; });
        if (found == metadata.end())
            return false;
        (*entry)["symbol"] = found->exportedWrapperSymbol;
    }
    reflection.clear();
    llvm::raw_string_ostream stream(reflection);
    stream << llvm::json::Value(std::move(*root));
    return true;
}

} // namespace vernon::compiler
