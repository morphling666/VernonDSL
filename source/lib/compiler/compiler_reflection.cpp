#include "compiler_reflection.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
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

uint64_t sourceTypeSize(mlir::Type type) {
    if (type.isIntOrFloat())
        return std::max<uint64_t>(type.getIntOrFloatBitWidth() / 8, 1);
    if (type.isIndex())
        return sizeof(uint64_t);
    if (mlir::isa<mlir::vernon::BufferType, mlir::vernon::TextureType, mlir::vernon::SamplerType>(type))
        return sizeof(uintptr_t);
    auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape())
        return 0;
    uint64_t count = 1;
    for (int64_t dimension : tensor.getShape())
        count *= static_cast<uint64_t>(dimension);
    return count * std::max<uint64_t>(tensor.getElementType().getIntOrFloatBitWidth() / 8, 1);
}

namespace {

llvm::json::Value attributeToJson(mlir::Attribute attribute) {
    if (!attribute)
        return nullptr;
    if (auto string = mlir::dyn_cast<mlir::StringAttr>(attribute))
        return string.getValue().str();
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attribute))
        return integer.getInt();

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
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            llvm::json::Object argument;
            argument["index"] = static_cast<int64_t>(index);

            std::string type;
            llvm::raw_string_ostream typeStream(type);
            function.getArgumentTypes()[index].print(typeStream);
            argument["type"] = std::move(type);
            uint64_t size = sourceTypeSize(function.getArgumentTypes()[index]);
            if (size) {
                uint64_t alignment = size >= 16 ? 16 : size >= 8 ? 8 : 4;
                argumentOffset = llvm::alignTo(argumentOffset, alignment);
                argument["cpu_offset"] = static_cast<int64_t>(argumentOffset);
                argument["cpu_size"] = static_cast<int64_t>(size);
                argumentOffset += size;
            }

            if (auto attrs = function.getArgAttrDict(index)) {
                for (mlir::NamedAttribute attr : attrs) {
                    argument[attr.getName().strref().str()] = attributeToJson(attr.getValue());
                    if (attr.getName().strref() == "vernon.instance_divisor")
                        requiredFeatures.insert("instancing");
                }
            }
            mlir::Type argumentType = function.getArgumentTypes()[index];
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
                } else if (auto buffer = mlir::dyn_cast<mlir::vernon::BufferType>(argumentType)) {
                    argument["kind"] = "tensor";
                    argument["cuda_abi"] = "strided_memref_1d";
                    argument["dtype"] =
                        sourceDtype ? sourceDtype.getValue().str() : scalarDtype(buffer.getElementType());
                    argument["access"] = buffer.getAccess().str();
                    uint64_t elementSize = std::max<uint64_t>(buffer.getElementType().getIntOrFloatBitWidth() / 8, 1);
                    argument["alignment"] = static_cast<int64_t>(elementSize);
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
                } else {
                    argument["kind"] = "scalar";
                    argument["dtype"] = sourceDtype ? sourceDtype.getValue().str() : scalarDtype(argumentType);
                    argument["alignment"] = static_cast<int64_t>(std::max<uint64_t>(sourceTypeSize(argumentType), 1));
                }
            }
            if (mlir::isa<mlir::vernon::BufferType>(argumentType))
                requiredFeatures.insert("buffers");
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
            resultSize = sourceTypeSize(function.getResultTypes()[index]);
            if (resultSize) {
                output["cpu_offset"] = int64_t{0};
                output["cpu_size"] = static_cast<int64_t>(resultSize);
            }

            if (auto attrs = function.getResultAttrDict(index)) {
                for (mlir::NamedAttribute attr : attrs)
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
        entries.emplace_back(std::move(entry));
    });
    if (invalid)
        return mlir::failure();

    llvm::json::Object root;
    root["schema_version"] = int64_t{2};
    root["gpu_launch_abi_version"] = int64_t{1};
    root["entries"] = std::move(entries);
    root["artifacts"] = llvm::json::Array();
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
