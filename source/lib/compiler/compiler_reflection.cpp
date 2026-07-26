#include "compiler_reflection.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonStorageProjection.h"
#include "mlir/Dialect/Vernon/Transforms/VernonTensorShapeSemantics.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace vernon::compiler {

namespace {

uint64_t sourceTypeAlignment(mlir::Type type);

struct StaticTensorPhysicalLayout {
    uint64_t size;
    uint64_t alignment;
    llvm::SmallVector<uint64_t> arrayStrides;
    std::optional<uint64_t> matrixStride;
    std::optional<llvm::StringRef> matrixOrder;
};

mlir::FailureOr<StaticTensorPhysicalLayout> staticTensorPhysicalLayout(mlir::RankedTensorType tensor) {
    if (!tensor.hasStaticShape() || llvm::any_of(tensor.getShape(), [](int64_t extent) { return extent <= 0; }) ||
        !tensor.getElementType().isIntOrFloat())
        return mlir::failure();
    const uint64_t elementSize = std::max<uint64_t>(tensor.getElementType().getIntOrFloatBitWidth() / 8, 1);
    if (tensor.getRank() == 0)
        return StaticTensorPhysicalLayout{elementSize, elementSize, {}, std::nullopt, std::nullopt};
    if (tensor.getRank() == 1 && tensor.getDimSize(0) >= 2 && tensor.getDimSize(0) <= 4) {
        const uint64_t count = tensor.getDimSize(0);
        const uint64_t alignment = (count == 2 ? 2 : 4) * elementSize;
        return StaticTensorPhysicalLayout{count * elementSize, alignment, {}, std::nullopt, std::nullopt};
    }
    if (tensor.getRank() == 2 && tensor.getDimSize(0) >= 2 && tensor.getDimSize(0) <= 4 && tensor.getDimSize(1) >= 2 &&
        tensor.getDimSize(1) <= 4) {
        const uint64_t rows = tensor.getDimSize(0);
        const uint64_t columns = tensor.getDimSize(1);
        const uint64_t columnAlignment = std::max<uint64_t>((rows == 2 ? 2 : 4) * elementSize, 16);
        const uint64_t stride = llvm::alignTo(rows * elementSize, columnAlignment);
        return StaticTensorPhysicalLayout{
            stride * columns, columnAlignment, {}, stride, llvm::StringRef("column_major")};
    }

    uint64_t size = elementSize;
    uint64_t alignment = elementSize;
    llvm::SmallVector<uint64_t> reversedStrides;
    for (int64_t extent : llvm::reverse(tensor.getShape())) {
        alignment = std::max<uint64_t>(alignment, 16);
        const uint64_t stride = llvm::alignTo(size, alignment);
        reversedStrides.push_back(stride);
        size = stride * static_cast<uint64_t>(extent);
    }
    return StaticTensorPhysicalLayout{size, alignment,
                                      llvm::SmallVector<uint64_t>(reversedStrides.rbegin(), reversedStrides.rend()),
                                      std::nullopt, std::nullopt};
}

std::string proposedStorageClass(mlir::StringRef stage, mlir::DictionaryAttr attrs) {
    if (stage == "compute")
        return "StorageBuffer";
    auto kind = attrs ? attrs.getAs<mlir::StringAttr>("vernon.interface") : nullptr;
    if (!kind)
        return "StorageBuffer";
    if (kind.getValue() == "input")
        return "Input";
    if (kind.getValue() == "uniform")
        return attrs.get("vernon.binding") ? "Uniform" : "PushConstant";
    return "StorageBuffer";
}

uint64_t productTypeSize(mlir::TypeRange fields) {
    uint64_t offset = 0;
    uint64_t alignment = 1;
    for (mlir::Type field : fields) {
        uint64_t fieldSize = sourceTypeSize(field);
        uint64_t fieldAlignment = sourceTypeAlignment(field);
        if (!fieldSize || !fieldAlignment)
            return 0;
        offset = llvm::alignTo(offset, fieldAlignment);
        offset += fieldSize;
        alignment = std::max(alignment, fieldAlignment);
    }
    return llvm::alignTo(offset, alignment);
}

uint64_t sourceTypeAlignment(mlir::Type type) {
    if (type.isIntOrFloat())
        return std::max<uint64_t>(type.getIntOrFloatBitWidth() / 8, 1);
    if (type.isIndex() ||
        mlir::isa<mlir::vernon::TensorViewType, mlir::vernon::TextureType, mlir::vernon::SamplerType>(type))
        return sizeof(uintptr_t);
    if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type))
        return sourceTypeAlignment(tensor.getElementType());
    if (auto tuple = mlir::dyn_cast<mlir::TupleType>(type)) {
        uint64_t alignment = 1;
        for (mlir::Type element : tuple.getTypes()) {
            uint64_t elementAlignment = sourceTypeAlignment(element);
            if (!elementAlignment)
                return 0;
            alignment = std::max(alignment, elementAlignment);
        }
        return alignment;
    }
    return 0;
}

} // namespace

uint64_t sourceTypeSize(mlir::Type type) {
    if (type.isIntOrFloat())
        return std::max<uint64_t>(type.getIntOrFloatBitWidth() / 8, 1);
    if (type.isIndex())
        return sizeof(uint64_t);
    if (mlir::isa<mlir::vernon::TensorViewType, mlir::vernon::TextureType, mlir::vernon::SamplerType>(type))
        return sizeof(uintptr_t);
    if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
        if (!tensor.hasStaticShape())
            return 0;
        uint64_t count = 1;
        for (int64_t dimension : tensor.getShape())
            count *= static_cast<uint64_t>(dimension);
        uint64_t elementSize = sourceTypeSize(tensor.getElementType());
        uint64_t elementAlignment = sourceTypeAlignment(tensor.getElementType());
        return elementSize && elementAlignment ? count * llvm::alignTo(elementSize, elementAlignment) : 0;
    }
    if (auto tuple = mlir::dyn_cast<mlir::TupleType>(type))
        return productTypeSize(tuple.getTypes());
    return 0;
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
            auto thenYield = mlir::cast<mlir::scf::YieldOp>(ifOp.thenBlock()->getTerminator());
            auto elseYield = mlir::cast<mlir::scf::YieldOp>(ifOp.elseBlock()->getTerminator());
            for (auto [result, thenValue, elseValue] :
                 llvm::zip_equal(ifOp.getResults(), thenYield.getOperands(), elseYield.getOperands())) {
                addEdge(thenValue, result);
                addEdge(elseValue, result);
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

mlir::FailureOr<std::string> buildReflection(mlir::ModuleOp module, mlir::ModuleOp sourceModule) {
    auto scalarDtype = [](mlir::Type type) -> std::string {
        if (type.isF16())
            return "f16";
        if (type.isF32())
            return "f32";
        if (type.isF64())
            return "f64";
        if (type.isInteger(1))
            return "bool";
        if (type.isInteger(32))
            return "u32";
        if (type.isIndex())
            return "index";
        return "";
    };
    llvm::json::Array entries;
    std::set<std::string> requiredFeatures;
    bool invalid = false;
    uint32_t nextGeneratedBinding = 0;
    module.walk([&](mlir::func::FuncOp function) {
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            mlir::DictionaryAttr attrs = function.getArgAttrDict(index);
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
        if (stage.getValue() == "compute")
            requiredFeatures.insert("compute");

        mlir::FailureOr<std::map<unsigned, std::set<SampledTextureBinding>>> sampledBindings =
            analyzeSampledTextureBindings(function);
        if (mlir::failed(sampledBindings)) {
            invalid = true;
            return;
        }

        llvm::json::Array arguments;
        uint64_t argumentOffset = 0;
        std::map<unsigned, uint32_t> generatedUniformBindings;
        uint64_t inlineUniformSize = 0;
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            if (stage.getValue() == "compute")
                break;
            mlir::DictionaryAttr attrs = function.getArgAttrDict(index);
            auto interfaceKind = attrs.getAs<mlir::StringAttr>("vernon.interface");
            if (!interfaceKind || interfaceKind.getValue() != "uniform" || attrs.get("vernon.binding"))
                continue;
            mlir::Type type = function.getArgumentTypes()[index];
            uint64_t size = sourceTypeSize(type);
            uint64_t alignment = sourceTypeAlignment(type);
            bool requiresBuffer = false;
            if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
                mlir::FailureOr<StaticTensorPhysicalLayout> layout = staticTensorPhysicalLayout(tensor);
                if (mlir::failed(layout)) {
                    function.emitError() << "cannot plan graphics uniform layout for argument #" << index;
                    invalid = true;
                    return;
                }
                size = layout->size;
                alignment = layout->alignment;
                requiresBuffer = !layout->arrayStrides.empty();
            }
            if (!size || !alignment) {
                function.emitError() << "cannot plan graphics uniform layout for argument #" << index;
                invalid = true;
                return;
            }
            const uint64_t offset = llvm::alignTo(inlineUniformSize, alignment);
            requiresBuffer = requiresBuffer || offset > 128 || size > 128 - std::min<uint64_t>(offset, 128);
            if (requiresBuffer) {
                generatedUniformBindings[index] = nextGeneratedBinding++;
            } else {
                inlineUniformSize = offset + size;
            }
        }
        llvm::SmallVector<std::optional<uint32_t>> computeBindings(function.getNumArguments());
        if (stage.getValue() == "compute") {
            uint32_t flattenedBinding = 0;
            for (unsigned index = 0; index < function.getNumArguments(); ++index) {
                mlir::DictionaryAttr attrs = function.getArgAttrDict(index);
                if (attrs.get("vernon.builtin"))
                    continue;
                computeBindings[index] = flattenedBinding;
                size_t leafCount = 1;
                if (auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(function.getArgumentTypes()[index])) {
                    mlir::FailureOr<mlir::vernon::StorageLayout> layout =
                        mlir::vernon::resolveStorageLayout(view.getElementType(), module);
                    if (mlir::failed(layout)) {
                        function.emitError("cannot assign flattened compute bindings");
                        invalid = true;
                        return;
                    }
                    leafCount = layout->leaves.size();
                }
                flattenedBinding += static_cast<uint32_t>(std::max<size_t>(leafCount, 1));
            }
        }
        uint32_t nextBinding = 0;
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            mlir::DictionaryAttr attrs = function.getArgAttrDict(index);
            if (auto binding = attrs.getAs<mlir::IntegerAttr>("vernon.binding"))
                nextBinding = std::max(nextBinding, static_cast<uint32_t>(binding.getInt() + 1));
        }
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            llvm::json::Object argument;
            argument["index"] = static_cast<int64_t>(index);

            std::string type;
            llvm::raw_string_ostream typeStream(type);
            function.getArgumentTypes()[index].print(typeStream);
            argument["type"] = std::move(type);
            mlir::DictionaryAttr argumentAttrs = function.getArgAttrDict(index);
            mlir::IntegerAttr reflectedSize;
            mlir::IntegerAttr reflectedAlignment;
            if (argumentAttrs) {
                reflectedSize = argumentAttrs.getAs<mlir::IntegerAttr>("vernon.abi_size");
                reflectedAlignment = argumentAttrs.getAs<mlir::IntegerAttr>("vernon.abi_alignment");
            }
            uint64_t size = reflectedSize ? static_cast<uint64_t>(reflectedSize.getInt())
                                          : sourceTypeSize(function.getArgumentTypes()[index]);
            if (size) {
                uint64_t alignment = reflectedAlignment ? static_cast<uint64_t>(reflectedAlignment.getInt())
                                     : size >= 16       ? 16
                                     : size >= 8        ? 8
                                                        : 4;
                argumentOffset = llvm::alignTo(argumentOffset, alignment);
                argument["cpu_offset"] = static_cast<int64_t>(argumentOffset);
                argument["cpu_size"] = static_cast<int64_t>(size);
                argumentOffset += size;
            }

            if (argumentAttrs) {
                for (mlir::NamedAttribute attr : argumentAttrs) {
                    argument[attr.getName().strref().str()] = attributeToJson(attr.getValue());
                    if (attr.getName().strref() == "vernon.instance_divisor")
                        requiredFeatures.insert("instancing");
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
            mlir::Type argumentType = function.getArgumentTypes()[index];
            if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(argumentType)) {
                mlir::FailureOr<StaticTensorPhysicalLayout> layout = staticTensorPhysicalLayout(tensor);
                if (mlir::failed(layout)) {
                    function.emitError() << "cannot reflect physical GPU layout for static Tensor argument #" << index;
                    invalid = true;
                    return;
                }
                const std::string dtype = argumentAttrs && argumentAttrs.getAs<mlir::StringAttr>("vernon.dtype")
                                              ? argumentAttrs.getAs<mlir::StringAttr>("vernon.dtype").getValue().str()
                                              : scalarDtype(tensor.getElementType());
                argument["kind"] = "tensor_value";
                argument["dtype"] = dtype;
                llvm::json::Array shape;
                for (int64_t extent : tensor.getShape())
                    shape.emplace_back(extent);
                argument["shape"] = std::move(shape);
                argument["rank"] = static_cast<int64_t>(tensor.getRank());
                const bool vertexAttribute =
                    stage.getValue() == "vertex" && argumentAttrs &&
                    argumentAttrs.getAs<mlir::StringAttr>("vernon.interface") &&
                    argumentAttrs.getAs<mlir::StringAttr>("vernon.interface").getValue() == "input" &&
                    argumentAttrs.getAs<mlir::IntegerAttr>("vernon.location") && !argumentAttrs.get("vernon.builtin");
                if (vertexAttribute) {
                    mlir::FailureOr<mlir::vernon::StaticAttributePlan> plan =
                        mlir::vernon::getStaticAttributePlan(tensor.getElementType(), tensor.getShape());
                    if (mlir::failed(plan)) {
                        function.emitError() << "cannot reflect vertex attribute layout for argument #" << index;
                        invalid = true;
                        return;
                    }
                    llvm::json::Array leaves;
                    for (const mlir::vernon::StaticAttributeLeaf &leaf : plan->leaves) {
                        llvm::json::Object reflectedLeaf;
                        reflectedLeaf["location_offset"] = leaf.locationOffset;
                        reflectedLeaf["component_count"] = leaf.componentCount;
                        reflectedLeaf["byte_offset"] = static_cast<int64_t>(leaf.byteOffset);
                        leaves.emplace_back(std::move(reflectedLeaf));
                    }
                    argument["location_span"] = static_cast<int64_t>(plan->leaves.size());
                    argument["attribute_leaves"] = std::move(leaves);
                } else {
                    argument["physical_size"] = static_cast<int64_t>(layout->size);
                    argument["physical_alignment"] = static_cast<int64_t>(layout->alignment);
                    argument["proposed_storage_class"] =
                        generatedUniformBindings.find(index) != generatedUniformBindings.end()
                            ? "Uniform"
                            : proposedStorageClass(stage.getValue(), argumentAttrs);
                    llvm::json::Array arrayStrides;
                    for (uint64_t stride : layout->arrayStrides)
                        arrayStrides.emplace_back(static_cast<int64_t>(stride));
                    argument["array_strides"] = std::move(arrayStrides);
                    if (layout->matrixStride) {
                        argument["matrix_stride"] = static_cast<int64_t>(*layout->matrixStride);
                        argument["matrix_order"] = layout->matrixOrder->str();
                    }
                }
            }
            if (auto texture = mlir::dyn_cast<mlir::vernon::TextureType>(argumentType)) {
                argument["kind"] = "texture";
                argument["dtype"] = scalarDtype(texture.getElementType());
                argument["dimension"] = texture.getDimension().str();
                argument["access"] = "read";
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
                    argument["sampled_texture_bindings"] = std::move(bindings);
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
                } else if (auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(argumentType)) {
                    argument["kind"] = "tensor";
                    argument["cuda_abi"] = "strided_memref_1d";
                    const std::string dtype =
                        sourceDtype ? sourceDtype.getValue().str() : scalarDtype(view.getElementType());
                    if (!dtype.empty())
                        argument["dtype"] = dtype;
                    argument["access"] = view.getAccess().str();
                    argument["rank"] = static_cast<int64_t>(view.getRank());
                    auto abiSize = attrs.getAs<mlir::IntegerAttr>("vernon.element_abi_size");
                    auto abiAlignment = attrs.getAs<mlir::IntegerAttr>("vernon.element_abi_alignment");
                    uint64_t elementSize =
                        view.getElementType().isIntOrFloat()
                            ? std::max<uint64_t>(view.getElementType().getIntOrFloatBitWidth() / 8, 1)
                        : abiSize ? static_cast<uint64_t>(abiSize.getInt())
                                  : 0;
                    uint64_t elementAlignment =
                        abiAlignment ? static_cast<uint64_t>(abiAlignment.getInt()) : elementSize;
                    argument["element_abi_size"] = static_cast<int64_t>(elementSize);
                    argument["alignment"] = static_cast<int64_t>(std::max<uint64_t>(elementAlignment, 1));
                    mlir::FailureOr<mlir::vernon::StorageLayout> storageLayout =
                        mlir::vernon::resolveStorageLayout(view.getElementType(), module);
                    if (mlir::failed(storageLayout)) {
                        function.emitError("cannot reflect aggregate TensorView storage layout");
                        invalid = true;
                        return;
                    }
                    const uint32_t firstBinding =
                        computeBindings[index] ? *computeBindings[index]
                        : attrs.getAs<mlir::IntegerAttr>("vernon.binding")
                            ? static_cast<uint32_t>(attrs.getAs<mlir::IntegerAttr>("vernon.binding").getInt())
                            : index;
                    llvm::json::Array storageLeaves;
                    for (auto [leafIndex, leaf] : llvm::enumerate(storageLayout->leaves)) {
                        llvm::json::Object reflectedLeaf;
                        reflectedLeaf["element_size"] =
                            static_cast<int64_t>(std::max<uint64_t>(leaf.type.getIntOrFloatBitWidth() / 8, 1));
                        reflectedLeaf["byte_offset"] = static_cast<int64_t>(leaf.byteOffset);
                        reflectedLeaf["binding"] =
                            static_cast<int64_t>(computeBindings[index] ? firstBinding + leafIndex
                                                 : leafIndex == 0       ? firstBinding
                                                                        : nextBinding++);
                        storageLeaves.emplace_back(std::move(reflectedLeaf));
                    }
                    argument["storage_leaves"] = std::move(storageLeaves);
                    if (auto shape = attrs.getAs<mlir::DenseI64ArrayAttr>("vernon.tensor_shape")) {
                        llvm::json::Array dimensions;
                        llvm::json::Array strides;
                        int64_t stride = 1;
                        llvm::SmallVector<int64_t> reversedStrides(shape.size());
                        for (int64_t dimensionIndex = static_cast<int64_t>(shape.size()) - 1; dimensionIndex >= 0;
                             --dimensionIndex) {
                            reversedStrides[dimensionIndex] = stride;
                            stride *= shape[dimensionIndex];
                        }
                        for (auto [dimension, tensorStride] : llvm::zip_equal(shape.asArrayRef(), reversedStrides)) {
                            dimensions.emplace_back(dimension);
                            strides.emplace_back(tensorStride);
                        }
                        argument["rank"] = static_cast<int64_t>(shape.size());
                        argument["shape"] = std::move(dimensions);
                        argument["strides"] = std::move(strides);
                    }
                } else if (!mlir::isa<mlir::RankedTensorType>(argumentType)) {
                    argument["kind"] = "scalar";
                    argument["dtype"] = sourceDtype ? sourceDtype.getValue().str() : scalarDtype(argumentType);
                    argument["alignment"] = static_cast<int64_t>(std::max<uint64_t>(sourceTypeSize(argumentType), 1));
                }
            }
            if (mlir::isa<mlir::vernon::TensorViewType>(argumentType))
                requiredFeatures.insert("tensor_views");
            if (mlir::isa<mlir::vernon::TextureType>(argumentType))
                requiredFeatures.insert("textures");
            if (mlir::isa<mlir::vernon::SamplerType>(argumentType))
                requiredFeatures.insert("samplers");
            arguments.emplace_back(std::move(argument));
        }

        llvm::json::Array results;
        uint64_t resultSize = 0;
        for (unsigned index = 0; index < function.getNumResults(); ++index) {
            llvm::json::Object output;
            output["index"] = static_cast<int64_t>(index);

            std::string type;
            llvm::raw_string_ostream typeStream(type);
            function.getResultTypes()[index].print(typeStream);
            output["type"] = std::move(type);
            mlir::DictionaryAttr resultAttrs = function.getResultAttrDict(index);
            mlir::IntegerAttr reflectedSize;
            if (resultAttrs)
                reflectedSize = resultAttrs.getAs<mlir::IntegerAttr>("vernon.abi_size");
            resultSize = reflectedSize ? static_cast<uint64_t>(reflectedSize.getInt())
                                       : sourceTypeSize(function.getResultTypes()[index]);
            if (resultSize) {
                output["cpu_offset"] = int64_t{0};
                output["cpu_size"] = static_cast<int64_t>(resultSize);
            }

            if (resultAttrs) {
                for (mlir::NamedAttribute attr : resultAttrs)
                    output[attr.getName().strref().str()] = attributeToJson(attr.getValue());
            }
            results.emplace_back(std::move(output));
        }

        llvm::json::Object entry;
        entry["name"] = function.getSymName().str();
        entry["symbol"] = function.getSymName().str();
        entry["stage"] = stage.getValue().str();
        entry["arguments"] = std::move(arguments);
        entry["results"] = std::move(results);
        entry["cpu_arguments_size"] = static_cast<int64_t>(argumentOffset);
        entry["cpu_results_size"] = static_cast<int64_t>(resultSize);
        if (auto workgroup = function->getAttrOfType<mlir::DenseI32ArrayAttr>("vernon.workgroup_size")) {
            llvm::json::Array dimensions;
            for (int32_t dimension : workgroup.asArrayRef())
                dimensions.emplace_back(static_cast<int64_t>(dimension));
            entry["workgroup_size"] = std::move(dimensions);
        }
        llvm::json::Array effects;
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
                llvm::json::Object reflected;
                reflected["kind"] = kind.getValue().str();
                reflected["owner"] = owner.getValue().str();
                reflected["region"] = region.getValue().str();
                llvm::json::Array indices;
                if (auto values = effect.getAs<mlir::DenseI64ArrayAttr>("indices"))
                    for (int64_t index : values.asArrayRef())
                        indices.emplace_back(index);
                reflected["indices"] = std::move(indices);
                effects.emplace_back(std::move(reflected));
            }
        }
        entry["effects"] = std::move(effects);
        entries.emplace_back(std::move(entry));
    });
    if (invalid)
        return mlir::failure();

    llvm::json::Object root;
    root["schema_version"] = int64_t{3};
    root["gpu_launch_abi_version"] = int64_t{1};
    int64_t valueAbiVersion = 0;
    if (auto version = sourceModule->getAttrOfType<mlir::IntegerAttr>("vernon.value_abi_version"))
        valueAbiVersion = version.getInt();
    root["value_abi_version"] = valueAbiVersion;
    root["entries"] = std::move(entries);
    root["artifacts"] = llvm::json::Array();
    std::map<std::string, llvm::json::Object> structLayouts;
    for (mlir::vernon::StructDeclOp declaration : sourceModule.getOps<mlir::vernon::StructDeclOp>()) {
        llvm::json::Object layout;
        layout["name"] = declaration.getSymName().str();
        if (auto size = declaration->getAttrOfType<mlir::IntegerAttr>("abi_size"))
            layout["size"] = size.getInt();
        if (auto alignment = declaration->getAttrOfType<mlir::IntegerAttr>("abi_alignment"))
            layout["alignment"] = alignment.getInt();
        llvm::json::Array offsets;
        if (auto values = declaration->getAttrOfType<mlir::DenseI64ArrayAttr>("abi_field_offsets"))
            for (int64_t offset : values.asArrayRef())
                offsets.emplace_back(offset);
        layout["field_offsets"] = std::move(offsets);
        llvm::json::Array fields;
        if (auto values = declaration->getAttrOfType<mlir::ArrayAttr>("fields"))
            for (mlir::Attribute value : values)
                if (auto field = mlir::dyn_cast<mlir::StringAttr>(value))
                    fields.emplace_back(field.getValue().str());
        layout["fields"] = std::move(fields);
        structLayouts.emplace(declaration.getSymName().str(), std::move(layout));
    }
    llvm::json::Array reflectedStructs;
    for (auto &[name, layout] : structLayouts) {
        (void)name;
        reflectedStructs.emplace_back(std::move(layout));
    }
    root["struct_layouts"] = std::move(reflectedStructs);
    llvm::json::Array dependencies;
    if (auto encoded = sourceModule->getAttrOfType<mlir::ArrayAttr>("vernon.source_dependencies")) {
        for (mlir::Attribute attribute : encoded) {
            auto value = mlir::dyn_cast<mlir::StringAttr>(attribute);
            if (!value)
                continue;
            auto [path, digest] = value.getValue().split('=');
            llvm::json::Object dependency;
            dependency["path"] = path.str();
            dependency["sha256"] = digest.str();
            dependencies.emplace_back(std::move(dependency));
        }
    }
    root["dependencies"] = std::move(dependencies);
    llvm::json::Array features;
    for (const std::string &feature : requiredFeatures)
        features.emplace_back(feature);
    root["required_features"] = std::move(features);
    std::string canonicalModule;
    llvm::raw_string_ostream moduleStream(canonicalModule);
    sourceModule.print(moduleStream, mlir::OpPrintingFlags().enableDebugInfo(false));
    root["module_hash"] = llvm::utohexstr(llvm::xxHash64(canonicalModule));

    std::string output;
    llvm::raw_string_ostream stream(output);
    stream << llvm::json::Value(std::move(root));
    return output;
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
