#include "compiler_reflection.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAttributeAbi.h"
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
#include <array>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace vernon::compiler {

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
        if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
            return integer.isUnsigned() ? "u32" : "i32";
        if (type.isIndex())
            return "index";
        return "";
    };
    llvm::DenseMap<mlir::Type, mlir::vernon::ValueAbiLayout> logicalPlanCache;
    auto reflectValueLayout =
        [&](mlir::Type type,
            llvm::ArrayRef<llvm::StringRef> logicalDtypes = {}) -> mlir::FailureOr<llvm::json::Object> {
        mlir::vernon::ValueAbiLayout planned;
        if (logicalDtypes.empty()) {
            if (auto found = logicalPlanCache.find(type); found != logicalPlanCache.end()) {
                planned = found->second;
            } else {
                mlir::FailureOr<mlir::vernon::ValueAbiLayout> layout = mlir::vernon::getValueAbiLayout(type, module);
                if (mlir::failed(layout))
                    return mlir::failure();
                planned = *layout;
                logicalPlanCache.try_emplace(type, planned);
            }
        } else {
            mlir::FailureOr<mlir::vernon::ValueAbiLayout> layout =
                mlir::vernon::getValueAbiLayout(type, module, logicalDtypes);
            if (mlir::failed(layout))
                return mlir::failure();
            planned = std::move(*layout);
        }
        const mlir::vernon::ValueAbiLayout &layout = planned;
        llvm::json::Object reflected;
        std::string logicalType;
        llvm::raw_string_ostream typeStream(logicalType);
        type.print(typeStream);
        typeStream.flush();
        reflected["logical_type"] = std::move(logicalType);
        if (auto structure = mlir::dyn_cast<mlir::vernon::StructType>(type))
            reflected["struct_name"] = structure.getName().str();
        reflected["byte_size"] = static_cast<int64_t>(layout.size);
        reflected["alignment"] = static_cast<int64_t>(layout.alignment);
        reflected["layout_hash"] = layout.layoutHash;
        llvm::json::Array leaves;
        for (const mlir::vernon::ValueAbiLeaf &leaf : layout.leaves) {
            llvm::json::Object reflectedLeaf;
            llvm::json::Array path;
            for (const mlir::vernon::ValueAbiPathComponent &component : leaf.path) {
                if (component.field)
                    path.emplace_back(*component.field);
                else
                    path.emplace_back(static_cast<int64_t>(component.index));
            }
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
    };
    auto reflectPhysicalLayout = [](const mlir::vernon::PhysicalValueAbiLayout &physical,
                                    llvm::StringRef profile) -> llvm::json::Object {
        llvm::json::Object layout;
        layout["profile"] = profile.str();
        layout["size"] = static_cast<int64_t>(physical.size);
        layout["alignment"] = static_cast<int64_t>(physical.alignment);
        llvm::json::Array byteStrides;
        for (uint64_t stride : physical.byteStrides)
            byteStrides.emplace_back(static_cast<int64_t>(stride));
        layout["byte_strides"] = std::move(byteStrides);
        llvm::json::Array elementLeafOffsets;
        for (uint64_t offset : physical.elementLeafOffsets)
            elementLeafOffsets.emplace_back(static_cast<int64_t>(offset));
        layout["element_leaf_offsets"] = std::move(elementLeafOffsets);
        return layout;
    };
    auto reflectPhysicalPlan = [&](const mlir::vernon::PhysicalValueAbiPlan &plan,
                                   llvm::StringRef profile) -> llvm::json::Object {
        if (const auto *bytes = std::get_if<mlir::vernon::PhysicalValueAbiLayout>(&plan))
            return reflectPhysicalLayout(*bytes, profile);
        llvm::json::Object reflected{{"profile", profile.str()}};
        if (const auto *unsupported = std::get_if<mlir::vernon::UnsupportedPhysicalValueAbi>(&plan)) {
            reflected["unsupported"] = unsupported->reason;
            return reflected;
        }
        const auto &resource = std::get<mlir::vernon::PhysicalResourceAbiLayout>(plan);
        switch (resource.kind) {
        case mlir::vernon::PhysicalResourceAbiKind::HostPointer:
            reflected["kind"] = "host_pointer";
            break;
        case mlir::vernon::PhysicalResourceAbiKind::CudaStorageLeaves:
            reflected["kind"] = "strided_memref_storage_leaves";
            break;
        case mlir::vernon::PhysicalResourceAbiKind::GraphicsStorageLeaves:
            reflected["kind"] = "descriptor_storage_leaves";
            break;
        case mlir::vernon::PhysicalResourceAbiKind::GraphicsTexture:
            reflected["kind"] = "texture_descriptor";
            break;
        case mlir::vernon::PhysicalResourceAbiKind::GraphicsSampler:
            reflected["kind"] = "sampler_descriptor";
            break;
        }
        if (resource.handleSize != 0) {
            reflected["size"] = static_cast<int64_t>(resource.handleSize);
            reflected["alignment"] = static_cast<int64_t>(resource.handleAlignment);
        }
        if (resource.elementLayout) {
            reflected["element_layout_hash"] = resource.elementLayout->layoutHash;
            reflected["element_size"] = static_cast<int64_t>(resource.elementLayout->size);
            reflected["element_alignment"] = static_cast<int64_t>(resource.elementLayout->alignment);
        }
        return reflected;
    };
    std::array<llvm::DenseMap<mlir::Type, mlir::vernon::PhysicalValueAbiPlan>,
               static_cast<size_t>(mlir::vernon::PhysicalAbiProfile::Count)>
        physicalPlanCache;
    auto physicalPlan =
        [&](mlir::Type type,
            mlir::vernon::PhysicalAbiProfile profile) -> mlir::FailureOr<mlir::vernon::PhysicalValueAbiPlan> {
        auto &cache = physicalPlanCache[static_cast<size_t>(profile)];
        if (auto found = cache.find(type); found != cache.end())
            return found->second;
        mlir::FailureOr<mlir::vernon::PhysicalValueAbiPlan> planned =
            mlir::vernon::getPhysicalValueAbiPlan(type, module, profile);
        if (mlir::failed(planned))
            return mlir::failure();
        auto inserted = cache.try_emplace(type, std::move(*planned));
        return inserted.first->second;
    };
    auto physicalLayout =
        [&](mlir::Type type,
            mlir::vernon::PhysicalAbiProfile profile) -> mlir::FailureOr<mlir::vernon::PhysicalValueAbiLayout> {
        mlir::FailureOr<mlir::vernon::PhysicalValueAbiPlan> plan = physicalPlan(type, profile);
        if (mlir::failed(plan))
            return mlir::failure();
        if (const auto *bytes = std::get_if<mlir::vernon::PhysicalValueAbiLayout>(&*plan))
            return *bytes;
        if (const auto *resource = std::get_if<mlir::vernon::PhysicalResourceAbiLayout>(&*plan);
            resource && resource->handleSize != 0)
            return mlir::vernon::PhysicalValueAbiLayout{resource->handleSize, resource->handleAlignment};
        return mlir::failure();
    };
    llvm::json::Array entries;
    std::set<std::string> requiredFeatures;
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
        std::map<unsigned, mlir::vernon::PhysicalValueAbiLayout> physicalValueLayouts;
        uint64_t inlineUniformSize = 0;
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            if (stage.getValue() == "compute")
                break;
            mlir::DictionaryAttr attrs = function.getArgAttrDict(index);
            auto interfaceKind = attrs.getAs<mlir::StringAttr>("vernon.interface");
            if (!interfaceKind || interfaceKind.getValue() != "uniform")
                continue;
            mlir::Type type = function.getArgumentTypes()[index];
            mlir::FailureOr<mlir::vernon::PhysicalValueAbiLayout> layout =
                physicalLayout(type, mlir::vernon::PhysicalAbiProfile::VulkanPushConstant);
            if (mlir::failed(layout)) {
                function.emitError() << "cannot plan graphics uniform layout for argument #" << index;
                invalid = true;
                return;
            }
            bool requiresBuffer = mlir::isa<mlir::vernon::TensorType>(type);
            if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type))
                requiresBuffer = tensor.getRank() > 2;
            if (attrs.get("vernon.binding")) {
                layout = physicalLayout(type, mlir::isa<mlir::vernon::TensorType>(type)
                                                  ? mlir::vernon::PhysicalAbiProfile::VulkanStd430StorageBuffer
                                                  : mlir::vernon::PhysicalAbiProfile::VulkanStd140UniformBuffer);
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
                layout = physicalLayout(type, mlir::isa<mlir::vernon::TensorType>(type)
                                                  ? mlir::vernon::PhysicalAbiProfile::VulkanStd430StorageBuffer
                                                  : mlir::vernon::PhysicalAbiProfile::VulkanStd140UniformBuffer);
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
        if (stage.getValue() == "compute") {
            uint32_t flattenedBinding = 0;
            for (unsigned index = 0; index < function.getNumArguments(); ++index) {
                mlir::DictionaryAttr attrs = function.getArgAttrDict(index);
                if (attrs.get("vernon.builtin"))
                    continue;
                computeBindings[index] = flattenedBinding;
                size_t leafCount = 1;
                if (auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(function.getArgumentTypes()[index])) {
                    mlir::FailureOr<mlir::vernon::ValueAbiLayout> layout =
                        mlir::vernon::getValueAbiLayout(view.getElementType(), module);
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
            mlir::Type argumentType = function.getArgumentTypes()[index];
            llvm::json::Object physicalLayouts;
            mlir::FailureOr<mlir::vernon::PhysicalValueAbiPlan> cpuPlan =
                physicalPlan(argumentType, mlir::vernon::PhysicalAbiProfile::HostValue);
            mlir::FailureOr<mlir::vernon::PhysicalValueAbiLayout> cpuLayout =
                physicalLayout(argumentType, mlir::vernon::PhysicalAbiProfile::HostValue);
            if (mlir::succeeded(cpuPlan) && mlir::succeeded(cpuLayout)) {
                argumentOffset = llvm::alignTo(argumentOffset, cpuLayout->alignment);
                llvm::json::Object reflected = reflectPhysicalPlan(*cpuPlan, "host_value");
                reflected["offset"] = static_cast<int64_t>(argumentOffset);
                physicalLayouts["host_value"] = std::move(reflected);
                argumentOffset += cpuLayout->size;
            }
            mlir::FailureOr<mlir::vernon::PhysicalValueAbiPlan> cudaPlan =
                physicalPlan(argumentType, mlir::vernon::PhysicalAbiProfile::CudaKernelParameter);
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
                mlir::FailureOr<mlir::vernon::PhysicalValueAbiPlan> plan = physicalPlan(argumentType, profile);
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
            if (auto generated = generatedUniformBindings.find(index); generated != generatedUniformBindings.end()) {
                argument["vernon.set"] = int64_t{0};
                argument["vernon.binding"] = static_cast<int64_t>(generated->second);
            }
            if (computeBindings[index]) {
                argument["vernon.set"] = int64_t{0};
                argument["vernon.binding"] = static_cast<int64_t>(*computeBindings[index]);
            }
            llvm::SmallVector<llvm::StringRef> logicalDtypes;
            const llvm::StringRef dtypeAttribute = mlir::isa<mlir::vernon::TensorViewType>(argumentType)
                                                       ? "vernon.element_abi_leaf_dtypes"
                                                       : "vernon.abi_leaf_dtypes";
            if (argumentAttrs)
                if (auto dtypes = argumentAttrs.getAs<mlir::ArrayAttr>(dtypeAttribute))
                    for (mlir::Attribute dtype : dtypes) {
                        auto value = mlir::dyn_cast<mlir::StringAttr>(dtype);
                        logicalDtypes.push_back(value ? value.getValue() : llvm::StringRef());
                    }
            if (!argumentType.isIndex() &&
                !mlir::isa<mlir::vernon::TensorViewType, mlir::vernon::TextureType, mlir::vernon::SamplerType>(
                    argumentType)) {
                mlir::FailureOr<llvm::json::Object> logicalAbi = reflectValueLayout(argumentType, logicalDtypes);
                if (mlir::failed(logicalAbi)) {
                    function.emitError() << "cannot reflect canonical logical ABI for argument #" << index;
                    invalid = true;
                    return;
                }
                argument["logical_abi"] = std::move(*logicalAbi);
            }
            mlir::Type elementLayoutType;
            if (auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(argumentType))
                elementLayoutType = view.getElementType();
            else if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(argumentType))
                elementLayoutType = tensor.getElementType();
            else if (auto tensor = mlir::dyn_cast<mlir::vernon::TensorType>(argumentType))
                elementLayoutType = tensor.getElementType();
            else if (mlir::isa<mlir::vernon::StructType, mlir::TupleType>(argumentType) || argumentType.isInteger(1) ||
                     argumentType.isInteger(32) || argumentType.isF16() || argumentType.isF32() || argumentType.isF64())
                elementLayoutType = argumentType;
            if (elementLayoutType) {
                llvm::ArrayRef<llvm::StringRef> elementLogicalDtypes = logicalDtypes;
                if (elementLayoutType != argumentType && !elementLayoutType.isIntOrFloat())
                    elementLogicalDtypes = {};
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
                    mlir::vernon::getAttributeAbiLayout(argumentType, module, logicalDtypes);
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
            }
            if (auto tensor = mlir::dyn_cast<mlir::vernon::TensorType>(argumentType)) {
                mlir::FailureOr<mlir::vernon::ValueAbiLayout> layout =
                    mlir::vernon::getValueAbiLayout(argumentType, module);
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
                } else if (auto tensor = mlir::dyn_cast<mlir::vernon::TensorType>(argumentType)) {
                    argument["kind"] = "tensor";
                    argument["access"] = "read";
                    mlir::FailureOr<mlir::vernon::ValueAbiLayout> storageLayout =
                        mlir::vernon::getValueAbiLayout(tensor.getElementType(), module);
                    if (mlir::failed(storageLayout)) {
                        function.emitError("cannot reflect aggregate Tensor storage layout");
                        invalid = true;
                        return;
                    }
                    argument["alignment"] = static_cast<int64_t>(storageLayout->alignment);
                    if (computeBindings[index]) {
                        const uint32_t firstBinding = *computeBindings[index];
                        llvm::json::Array storageLeaves;
                        for (auto [leafIndex, leaf] : llvm::enumerate(storageLayout->leaves)) {
                            llvm::json::Object reflectedLeaf;
                            reflectedLeaf["element_size"] = static_cast<int64_t>(
                                std::max<uint64_t>(leaf.scalarType.getIntOrFloatBitWidth() / 8, 1));
                            reflectedLeaf["byte_offset"] = static_cast<int64_t>(leaf.byteOffset);
                            reflectedLeaf["binding"] = static_cast<int64_t>(firstBinding + leafIndex);
                            storageLeaves.emplace_back(std::move(reflectedLeaf));
                        }
                        argument["storage_leaves"] = std::move(storageLeaves);
                    }
                } else if (auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(argumentType)) {
                    argument["kind"] = "tensor";
                    const std::string dtype =
                        sourceDtype ? sourceDtype.getValue().str() : scalarDtype(view.getElementType());
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
                        mlir::vernon::getValueAbiLayout(view.getElementType(), module);
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
                    auto shape = attrs.getAs<mlir::DenseI64ArrayAttr>(mlir::vernon::kTensorShapeAttrName);
                    auto strides = attrs.getAs<mlir::DenseI64ArrayAttr>(mlir::vernon::kTensorStridesAttrName);
                    auto offset = attrs.getAs<mlir::IntegerAttr>(mlir::vernon::kTensorOffsetAttrName);
                    if (strides || offset) {
                        if (!strides || !offset || strides.size() != view.getShape().size() || offset.getInt() < 0 ||
                            (shape && shape.size() != strides.size())) {
                            function.emitError("TensorView layout requires rank-matched signed strides and a "
                                               "non-negative offset");
                            invalid = true;
                            return;
                        }
                        llvm::json::Array elementStrides;
                        for (int64_t tensorStride : strides.asArrayRef())
                            elementStrides.emplace_back(tensorStride);
                        if (shape) {
                            llvm::json::Array dimensions;
                            for (int64_t dimension : shape.asArrayRef())
                                dimensions.emplace_back(dimension);
                            argument["shape"] = std::move(dimensions);
                        }
                        argument["element_strides"] = std::move(elementStrides);
                        argument["element_offset"] = offset.getInt();
                    }
                } else if (!mlir::isa<mlir::RankedTensorType>(argumentType)) {
                    argument["kind"] = "scalar";
                    argument["dtype"] = sourceDtype ? sourceDtype.getValue().str() : scalarDtype(argumentType);
                    argument["alignment"] =
                        static_cast<int64_t>(mlir::succeeded(cpuLayout) ? cpuLayout->alignment : uint64_t{1});
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
            mlir::Type resultType = function.getResultTypes()[index];
            llvm::json::Object physicalLayouts;
            mlir::FailureOr<mlir::vernon::PhysicalValueAbiPlan> cpuPlan =
                physicalPlan(resultType, mlir::vernon::PhysicalAbiProfile::HostValue);
            mlir::FailureOr<mlir::vernon::PhysicalValueAbiLayout> cpuLayout =
                physicalLayout(resultType, mlir::vernon::PhysicalAbiProfile::HostValue);
            if (mlir::succeeded(cpuPlan) && mlir::succeeded(cpuLayout)) {
                resultSize = llvm::alignTo(resultSize, cpuLayout->alignment);
                llvm::json::Object reflected = reflectPhysicalPlan(*cpuPlan, "host_value");
                reflected["offset"] = static_cast<int64_t>(resultSize);
                physicalLayouts["host_value"] = std::move(reflected);
                resultSize += cpuLayout->size;
            }
            if (mlir::FailureOr<mlir::vernon::PhysicalValueAbiPlan> plan =
                    physicalPlan(resultType, mlir::vernon::PhysicalAbiProfile::CudaKernelParameter);
                mlir::succeeded(plan))
                physicalLayouts["cuda_kernel_parameter"] = reflectPhysicalPlan(*plan, "cuda_kernel_parameter");
            if (mlir::FailureOr<mlir::vernon::PhysicalValueAbiPlan> plan =
                    physicalPlan(resultType, mlir::vernon::PhysicalAbiProfile::VulkanStd140UniformBuffer);
                mlir::succeeded(plan))
                physicalLayouts["vulkan_std140_uniform_buffer"] =
                    reflectPhysicalPlan(*plan, "vulkan_std140_uniform_buffer");
            if (mlir::FailureOr<mlir::vernon::PhysicalValueAbiPlan> plan =
                    physicalPlan(resultType, mlir::vernon::PhysicalAbiProfile::VulkanStd430StorageBuffer);
                mlir::succeeded(plan))
                physicalLayouts["vulkan_std430_storage_buffer"] =
                    reflectPhysicalPlan(*plan, "vulkan_std430_storage_buffer");
            if (mlir::FailureOr<mlir::vernon::PhysicalValueAbiPlan> plan =
                    physicalPlan(resultType, mlir::vernon::PhysicalAbiProfile::VulkanPushConstant);
                mlir::succeeded(plan))
                physicalLayouts["vulkan_push_constant"] = reflectPhysicalPlan(*plan, "vulkan_push_constant");
            if (mlir::FailureOr<mlir::vernon::PhysicalValueAbiPlan> plan =
                    physicalPlan(resultType, mlir::vernon::PhysicalAbiProfile::OpenGLNativeUniform);
                mlir::succeeded(plan))
                physicalLayouts["opengl_native_uniform"] = reflectPhysicalPlan(*plan, "opengl_native_uniform");
            if (mlir::FailureOr<mlir::vernon::PhysicalValueAbiPlan> plan =
                    physicalPlan(resultType, mlir::vernon::PhysicalAbiProfile::DirectXConstantBuffer);
                mlir::succeeded(plan))
                physicalLayouts["directx_constant_buffer"] = reflectPhysicalPlan(*plan, "directx_constant_buffer");
            if (mlir::FailureOr<mlir::vernon::PhysicalValueAbiPlan> plan =
                    physicalPlan(resultType, mlir::vernon::PhysicalAbiProfile::MetalConstantBuffer);
                mlir::succeeded(plan))
                physicalLayouts["metal_constant_buffer"] = reflectPhysicalPlan(*plan, "metal_constant_buffer");
            output["physical_layouts"] = std::move(physicalLayouts);

            if (resultAttrs) {
                for (mlir::NamedAttribute attr : resultAttrs)
                    output[attr.getName().strref().str()] = attributeToJson(attr.getValue());
            }
            llvm::SmallVector<llvm::StringRef> logicalDtypes;
            if (resultAttrs)
                if (auto dtypes = resultAttrs.getAs<mlir::ArrayAttr>("vernon.abi_leaf_dtypes"))
                    for (mlir::Attribute dtype : dtypes) {
                        auto value = mlir::dyn_cast<mlir::StringAttr>(dtype);
                        logicalDtypes.push_back(value ? value.getValue() : llvm::StringRef());
                    }
            mlir::FailureOr<llvm::json::Object> logicalAbi = reflectValueLayout(resultType, logicalDtypes);
            if (mlir::failed(logicalAbi)) {
                function.emitError() << "cannot reflect canonical logical ABI for result #" << index;
                invalid = true;
                return;
            }
            output["logical_abi"] = std::move(*logicalAbi);
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
    root["schema_version"] = int64_t{5};
    root["gpu_launch_abi_version"] = int64_t{1};
    auto valueAbiVersion = sourceModule->getAttrOfType<mlir::IntegerAttr>("vernon.value_abi_version");
    if (!valueAbiVersion || valueAbiVersion.getInt() != mlir::vernon::kCurrentValueAbiVersion) {
        sourceModule.emitError("cannot reflect a module with a missing or unsupported Value ABI version");
        return mlir::failure();
    }
    root["value_abi_version"] = valueAbiVersion.getInt();
    root["entries"] = std::move(entries);
    root["artifacts"] = llvm::json::Array();
    std::map<std::string, llvm::json::Object> structLayouts;
    for (mlir::vernon::StructDeclOp declaration : sourceModule.getOps<mlir::vernon::StructDeclOp>()) {
        auto structure = mlir::vernon::StructType::get(sourceModule.getContext(), declaration.getSymName());
        mlir::FailureOr<mlir::vernon::ValueAbiLayout> planned =
            mlir::vernon::getValueAbiLayout(structure, sourceModule);
        if (mlir::failed(planned)) {
            declaration.emitError("cannot derive canonical struct Value ABI layout");
            return mlir::failure();
        }
        llvm::json::Object layout;
        layout["name"] = declaration.getSymName().str();
        layout["size"] = static_cast<int64_t>(planned->size);
        layout["alignment"] = static_cast<int64_t>(planned->alignment);
        llvm::json::Array offsets;
        for (uint64_t offset : planned->fieldOffsets)
            offsets.emplace_back(static_cast<int64_t>(offset));
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
    sourceModule.walk([&](mlir::Operation *operation) {
        if (mlir::isa<mlir::vernon::WorkgroupAllocOp>(operation))
            requiredFeatures.insert("workgroup_storage");
        if (mlir::isa<mlir::vernon::AtomicOp>(operation))
            requiredFeatures.insert("atomics");
        if (mlir::isa<mlir::vernon::BarrierOp>(operation))
            requiredFeatures.insert("barriers");
    });
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
