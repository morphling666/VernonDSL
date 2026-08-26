#include "compiler_program_reflection.h"
#include "compiler_reflection.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffUtils.h"
#include "mlir/Dialect/VernonProgram/IR/VernonProgram.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vernon::compiler {

mlir::FailureOr<std::optional<ProgramReflection>> buildProgramReflection(mlir::ModuleOp module) {
    auto valueCount = module->getAttrOfType<mlir::IntegerAttr>("vernon_program.value_count");
    if (!valueCount)
        return std::optional<ProgramReflection>{};
    const int64_t count = valueCount.getInt();
    if (count < 0)
        return module.emitError("executable Program value count must be non-negative");

    std::vector<std::optional<llvm::json::Object>> reflectedValues(static_cast<size_t>(count));
    llvm::json::Array graphs;
    llvm::json::Array kernelCompileRequests;
    llvm::json::Array inputs;
    llvm::json::Array outputs;
    llvm::json::Array cotangents;
    llvm::json::Array gradients;
    llvm::json::Array captures;
    bool invalid = false;
    const auto publicPath = [](llvm::StringRef path) {
        for (llvm::StringRef prefix : {"input.", "output.", "cotangent.", "gradient."})
            if (path.consume_front(prefix))
                break;
        return path;
    };
    const auto signatureBinding = [&](uint32_t value, llvm::StringRef path) {
        path = publicPath(path);
        return llvm::json::Object{{"value", static_cast<int64_t>(value)}, {"path", path.str()}};
    };
    llvm::StringMap<mlir::Type> primalInputs;
    llvm::StringMap<mlir::Type> primalOutputs;
    for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
        auto direction = function->getAttrOfType<mlir::StringAttr>("vernon_program.graph");
        if (!direction || direction.getValue() != "forward")
            continue;
        auto argumentNames = function->getAttrOfType<mlir::ArrayAttr>("vernon_program.argument_names");
        for (auto [index, argument] : llvm::enumerate(function.getArguments())) {
            mlir::StringAttr source = function.getArgAttrOfType<mlir::StringAttr>(index, "vernon.source_name");
            if (!source && argumentNames && index < argumentNames.size())
                source = mlir::dyn_cast<mlir::StringAttr>(argumentNames[index]);
            if (source)
                primalInputs[publicPath(source.getValue())] = argument.getType();
        }
        auto resultNames = function->getAttrOfType<mlir::ArrayAttr>("vernon_program.result_names");
        for (auto [index, type] : llvm::enumerate(function.getResultTypes())) {
            mlir::StringAttr source = function.getResultAttrOfType<mlir::StringAttr>(index, "vernon.source_name");
            if (!source && resultNames && index < resultNames.size())
                source = mlir::dyn_cast<mlir::StringAttr>(resultNames[index]);
            if (source)
                primalOutputs[publicPath(source.getValue())] = type;
        }
    }
    const auto reflectValue = [&](uint32_t id, llvm::StringRef name, mlir::Type type, bool external, bool output,
                                  bool authoritativeName = false, llvm::ArrayRef<llvm::StringRef> logicalDtypes = {},
                                  mlir::Type derivativeOf = {}, std::optional<llvm::StringRef> logicalDtype = {}) {
        if (id >= reflectedValues.size()) {
            module.emitError("executable Program value id exceeds value_count");
            invalid = true;
            return;
        }
        if (reflectedValues[id]) {
            if (authoritativeName)
                (*reflectedValues[id])["name"] = name.str();
            if (external)
                (*reflectedValues[id])["external"] = true;
            if (output)
                (*reflectedValues[id])["output"] = true;
            return;
        }
        std::string spelling;
        llvm::raw_string_ostream stream(spelling);
        type.print(stream);
        mlir::Type element = type;
        mlir::Type layoutType = type;
        llvm::json::Array shape;
        if (auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(type)) {
            element = view.getElementType();
            layoutType = element;
            for (int64_t extent : view.getShape())
                shape.emplace_back(extent);
        } else if (auto texture = mlir::dyn_cast<mlir::vernon::TextureType>(type)) {
            element = texture.getElementType();
            layoutType = {};
        } else if (mlir::isa<mlir::vernon::SamplerType>(type)) {
            layoutType = {};
        } else if (auto tensor = mlir::dyn_cast<mlir::vernon::TensorType>(type)) {
            element = tensor.getElementType();
            for (int64_t extent : tensor.getShape())
                shape.emplace_back(extent);
        } else if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
            element = tensor.getElementType();
            if (!tensor.hasStaticShape()) {
                module.emitError("executable Program reflection currently requires static Tensor shapes");
                invalid = true;
                return;
            }
            for (int64_t extent : tensor.getShape())
                shape.emplace_back(extent);
        }
        std::string dtype;
        if (logicalDtype && !logicalDtype->empty())
            dtype = logicalDtype->str();
        else if (logicalDtypes.size() == 1)
            dtype = logicalDtypes.front().str();
        if (logicalDtype && !logicalDtype->empty() && logicalDtypes.size() == 1 &&
            logicalDtype->str() != logicalDtypes.front()) {
            module.emitError("vernon.dtype does not match vernon.abi_leaf_dtypes");
            invalid = true;
            return;
        }
        reflectedValues[id] = llvm::json::Object{{"id", static_cast<int64_t>(id)},
                                                 {"name", name.str()},
                                                 {"type", std::move(spelling)},
                                                 {"dtype", std::move(dtype)},
                                                 {"shape", std::move(shape)},
                                                 {"external", external},
                                                 {"output", output}};
        if (!layoutType) {
            return;
        }
        llvm::SmallVector<llvm::StringRef> layoutDtypes(logicalDtypes.begin(), logicalDtypes.end());
        if (layoutDtypes.empty() && logicalDtype && !logicalDtype->empty())
            layoutDtypes.push_back(*logicalDtype);
        if (derivativeOf &&
            failed(mlir::vernon::getAutodiffDerivativeValueLayout(derivativeOf, type, module, layoutDtypes))) {
            module.emitError("cannot reflect executable Program derivative Value ABI");
            invalid = true;
            return;
        }
        mlir::FailureOr<llvm::json::Object> valueLayout = reflectCanonicalValueLayout(module, layoutType, layoutDtypes);
        if (mlir::failed(valueLayout)) {
            module.emitError("cannot reflect executable Program Value ABI");
            invalid = true;
            return;
        }
        (*reflectedValues[id])["value_layout"] = std::move(*valueLayout);
    };

    for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
        auto direction = function->getAttrOfType<mlir::StringAttr>("vernon_program.graph");
        if (!direction)
            continue;
        llvm::json::Object graph;
        graph["name"] = function.getSymName().str();
        graph["direction"] = direction.getValue().str();
        llvm::json::Array arguments;
        auto argumentNames = function->getAttrOfType<mlir::ArrayAttr>("vernon_program.argument_names");
        for (auto [index, argument] : llvm::enumerate(function.getArguments())) {
            auto id = function.getArgAttrOfType<mlir::IntegerAttr>(index, "vernon_program.value_id");
            if (!id || id.getInt() < 0) {
                function.emitError("executable Program argument has no value id");
                invalid = true;
                continue;
            }
            const uint32_t valueId = static_cast<uint32_t>(id.getInt());
            const bool capture = static_cast<bool>(
                function.getArgAttrOfType<mlir::IntegerAttr>(index, mlir::vernon::program::kCaptureForwardValueAttr));
            if (capture && direction.getValue() != "backward") {
                function.emitError("only a backward graph may capture a forward value");
                invalid = true;
            }
            if (!capture) {
                arguments.emplace_back(static_cast<int64_t>(valueId));
                llvm::StringRef path =
                    function.getArgAttrOfType<mlir::StringAttr>(index, "vernon.source_name")
                        ? function.getArgAttrOfType<mlir::StringAttr>(index, "vernon.source_name").getValue()
                    : argumentNames && index < argumentNames.size()
                        ? mlir::cast<mlir::StringAttr>(argumentNames[index]).getValue()
                        : llvm::StringRef("argument");
                if (direction.getValue() == "forward")
                    inputs.emplace_back(signatureBinding(valueId, path));
                else if (direction.getValue() == "backward")
                    cotangents.emplace_back(signatureBinding(valueId, path));
            } else {
                captures.emplace_back(static_cast<int64_t>(valueId));
            }
            llvm::StringRef name =
                argumentNames && index < argumentNames.size()
                    ? mlir::cast<mlir::StringAttr>(argumentNames[index]).getValue()
                : function.getArgAttrOfType<mlir::StringAttr>(index, "vernon.source_name")
                    ? function.getArgAttrOfType<mlir::StringAttr>(index, "vernon.source_name").getValue()
                    : llvm::StringRef("argument");
            llvm::SmallVector<llvm::StringRef> logicalDtypes;
            if (auto dtypes = function.getArgAttrOfType<mlir::ArrayAttr>(index, "vernon.abi_leaf_dtypes"))
                for (mlir::Attribute dtype : dtypes)
                    logicalDtypes.push_back(mlir::cast<mlir::StringAttr>(dtype).getValue());
            mlir::Type derivativeOf;
            if (!capture && direction.getValue() == "backward")
                if (auto found = primalOutputs.find(publicPath(name)); found != primalOutputs.end())
                    derivativeOf = found->second;
            std::optional<llvm::StringRef> logicalDtype;
            if (auto dtype = function.getArgAttrOfType<mlir::StringAttr>(index, "vernon.dtype"))
                logicalDtype = dtype.getValue();
            reflectValue(valueId, name, argument.getType(), !capture, false, !capture, logicalDtypes, derivativeOf,
                         logicalDtype);
        }
        graph["arguments"] = std::move(arguments);

        llvm::json::Array nodes;
        for (mlir::Operation &operation : function.getBody().front().without_terminator()) {
            if (mlir::vernon::program::isStorageAllocIntrinsic(&operation)) {
                auto operandIds = operation.getAttrOfType<mlir::DenseI64ArrayAttr>("vernon_program.operand_value_ids");
                auto resultIds = operation.getAttrOfType<mlir::DenseI64ArrayAttr>("vernon_program.result_value_ids");
                if (!resultIds || resultIds.size() != operation.getNumResults()) {
                    operation.emitError("storage alloc has incomplete executable Program value ids");
                    invalid = true;
                    continue;
                }
                for (auto [index, id] : llvm::enumerate(resultIds.asArrayRef())) {
                    if (id < 0) {
                        operation.emitError("storage alloc has an invalid result value id");
                        invalid = true;
                        continue;
                    }
                    auto abi = mlir::vernon::program::getProgramValueLanguageAbi(operation.getResult(index));
                    reflectValue(static_cast<uint32_t>(id), "value." + std::to_string(id),
                                 operation.getResult(index).getType(), false, false, false, abi.leaves, {}, abi.dtype);
                    auto intrinsic = mlir::cast<mlir::vernon::IntrinsicOp>(&operation);
                    if (mlir::vernon::program::isLikeAllocIntrinsicName(intrinsic.getName()) && id >= 0 &&
                        static_cast<size_t>(id) < reflectedValues.size() && reflectedValues[static_cast<size_t>(id)] &&
                        operandIds && !operandIds.empty()) {
                        const int64_t like = operandIds[0];
                        if (like >= 0)
                            (*reflectedValues[static_cast<size_t>(id)])["like"] = like;
                    }
                }
                continue;
            }
            auto nodeId = operation.getAttrOfType<mlir::IntegerAttr>("vernon_program.node_id");
            auto stage = operation.getAttrOfType<mlir::StringAttr>("vernon_program.stage");
            auto operandIds = operation.getAttrOfType<mlir::DenseI64ArrayAttr>("vernon_program.operand_value_ids");
            auto resultIds = operation.getAttrOfType<mlir::DenseI64ArrayAttr>("vernon_program.result_value_ids");
            auto dependencies = operation.getAttrOfType<mlir::DenseI64ArrayAttr>("vernon_program.dependencies");
            auto operandNames = operation.getAttrOfType<mlir::ArrayAttr>("operand_names");
            auto resultNames = operation.getAttrOfType<mlir::ArrayAttr>("result_names");
            auto resultDtypes = operation.getAttrOfType<mlir::ArrayAttr>("vernon_program.result_abi_leaf_dtypes");
            auto gridControlArguments =
                operation.getAttrOfType<mlir::DenseI64ArrayAttr>("vernon_program.grid_control_arguments");
            auto resourceOperandIndices =
                operation.getAttrOfType<mlir::DenseI64ArrayAttr>("vernon_program.resource_operand_indices");
            auto resultResourceSources =
                operation.getAttrOfType<mlir::DenseI64ArrayAttr>("vernon_program.result_resource_sources");
            auto operandAccesses = operation.getAttrOfType<mlir::ArrayAttr>("vernon_program.operand_accesses");
            const size_t operandOffset =
                mlir::isa<mlir::vernon::program::GraphicsOp>(operation) ? operation.getNumResults() : 0;
            if (!nodeId || !stage || !operandIds || !resultIds || !dependencies || !operandNames || !resultNames ||
                operandNames.size() + operandOffset != static_cast<size_t>(operandIds.size()) ||
                resultNames.size() != static_cast<size_t>(resultIds.size())) {
                operation.emitError("has incomplete executable Program metadata");
                invalid = true;
                continue;
            }
            llvm::json::Object node;
            const std::string requestId = function.getSymName().str() + ":" + std::to_string(nodeId.getInt());
            node["id"] = nodeId.getInt();
            node["name"] = operation.getAttrOfType<mlir::StringAttr>("debug_name")
                               ? operation.getAttrOfType<mlir::StringAttr>("debug_name").getValue().str()
                               : stage.getValue().str();
            node["kind"] = mlir::isa<mlir::vernon::program::ComputeOp>(operation) ? "compute" : "render";
            node["stage"] = requestId;
            if (auto graphics = mlir::dyn_cast<mlir::vernon::program::GraphicsOp>(operation)) {
                node["topology"] = graphics.getTopology().str();
                node["color_count"] = graphics.getColorCount();
            }
            llvm::json::Array operands;
            llvm::json::Array results;
            llvm::json::Array reflectedDependencies;
            llvm::json::Array resources;
            llvm::SmallSet<int64_t, 8> resourceOperands;
            if (resourceOperandIndices)
                resourceOperands.insert(resourceOperandIndices.asArrayRef().begin(),
                                        resourceOperandIndices.asArrayRef().end());
            for (auto [index, id] : llvm::enumerate(operandIds.asArrayRef())) {
                operands.emplace_back(id);
                if (!resourceOperandIndices || resourceOperands.count(static_cast<int64_t>(index))) {
                    llvm::StringRef access = "read";
                    if (operandAccesses && index < operandAccesses.size())
                        access = mlir::cast<mlir::StringAttr>(operandAccesses[index]).getValue();
                    llvm::json::Object resource{{"value", id}, {"access", access.str()}};
                    if (resultResourceSources)
                        for (auto [resultIndex, sourceIndex] : llvm::enumerate(resultResourceSources.asArrayRef()))
                            if (sourceIndex == static_cast<int64_t>(index) &&
                                resultIndex < static_cast<size_t>(resultIds.size()))
                                resource["after"] = resultIds[resultIndex];
                    resources.emplace_back(std::move(resource));
                }
                auto abi = mlir::vernon::program::getProgramValueLanguageAbi(operation.getOperand(index));
                reflectValue(static_cast<uint32_t>(id), "value." + std::to_string(id),
                             operation.getOperand(index).getType(), false, false, false, abi.leaves, {}, abi.dtype);
            }
            if (gridControlArguments)
                for (int64_t argumentIndex : gridControlArguments.asArrayRef()) {
                    if (argumentIndex < 0 || static_cast<uint64_t>(argumentIndex) >= function.getNumArguments()) {
                        operation.emitError("has an invalid grid control argument");
                        invalid = true;
                        continue;
                    }
                    auto valueId =
                        function.getArgAttrOfType<mlir::IntegerAttr>(argumentIndex, "vernon_program.value_id");
                    if (!valueId) {
                        operation.emitError("grid control argument has no executable value id");
                        invalid = true;
                        continue;
                    }
                    operands.emplace_back(valueId.getInt());
                }
            for (auto [index, id] : llvm::enumerate(resultIds.asArrayRef())) {
                results.emplace_back(id);
                if (!resultResourceSources || index >= static_cast<size_t>(resultResourceSources.size()) ||
                    resultResourceSources[index] < 0)
                    resources.emplace_back(llvm::json::Object{{"value", id}, {"access", "write"}});
                llvm::SmallVector<std::string> inheritedDtypes;
                llvm::SmallVector<llvm::StringRef> logicalDtypes;
                if (resultDtypes && index < resultDtypes.size())
                    for (mlir::Attribute dtype : mlir::cast<mlir::ArrayAttr>(resultDtypes[index]))
                        logicalDtypes.push_back(mlir::cast<mlir::StringAttr>(dtype).getValue());
                if (logicalDtypes.empty() && resultResourceSources &&
                    index < static_cast<size_t>(resultResourceSources.size())) {
                    const int64_t sourceIndex = resultResourceSources[index];
                    if (sourceIndex >= 0 && static_cast<size_t>(sourceIndex) < static_cast<size_t>(operandIds.size())) {
                        const int64_t sourceId = operandIds[sourceIndex];
                        if (sourceId >= 0 && static_cast<size_t>(sourceId) < reflectedValues.size() &&
                            reflectedValues[static_cast<size_t>(sourceId)]) {
                            if (const llvm::json::Object *layout =
                                    reflectedValues[static_cast<size_t>(sourceId)]->getObject("value_layout"))
                                if (const llvm::json::Array *leaves = layout->getArray("leaves"))
                                    for (const llvm::json::Value &leafValue : *leaves)
                                        if (const llvm::json::Object *leaf = leafValue.getAsObject())
                                            if (std::optional<llvm::StringRef> dtype = leaf->getString("dtype"))
                                                inheritedDtypes.emplace_back(dtype->str());
                        }
                    }
                    for (const std::string &dtype : inheritedDtypes)
                        logicalDtypes.push_back(dtype);
                }
                reflectValue(static_cast<uint32_t>(id), "value." + std::to_string(id),
                             operation.getResult(index).getType(), false, false, false, logicalDtypes);
            }
            for (int64_t dependency : dependencies.asArrayRef())
                reflectedDependencies.emplace_back(dependency);
            node["operands"] = std::move(operands);
            node["results"] = std::move(results);
            node["dependencies"] = std::move(reflectedDependencies);
            llvm::json::Array bindings;
            auto operandAutodiffRoles =
                operation.getAttrOfType<mlir::ArrayAttr>("vernon_program.operand_autodiff_roles");
            auto operandAutodiffSources =
                operation.getAttrOfType<mlir::ArrayAttr>("vernon_program.operand_autodiff_sources");
            auto resultAutodiffRoles = operation.getAttrOfType<mlir::ArrayAttr>("vernon_program.result_autodiff_roles");
            auto resultAutodiffSources =
                operation.getAttrOfType<mlir::ArrayAttr>("vernon_program.result_autodiff_sources");
            const auto binding = [&](mlir::Attribute name, int64_t id, mlir::ArrayAttr roles, mlir::ArrayAttr sources,
                                     size_t index) {
                llvm::json::Object reflected{{"parameter", mlir::cast<mlir::StringAttr>(name).getValue().str()},
                                             {"value", id}};
                if (roles && sources && index < roles.size() && index < sources.size()) {
                    reflected["autodiff_role"] = mlir::cast<mlir::StringAttr>(roles[index]).getValue().str();
                    reflected["autodiff_source"] = mlir::cast<mlir::StringAttr>(sources[index]).getValue().str();
                }
                return reflected;
            };
            for (auto [index, name] : llvm::enumerate(operandNames))
                bindings.emplace_back(
                    llvm::json::Object{{"parameter", mlir::cast<mlir::StringAttr>(name).getValue().str()},
                                       {"value", operandIds[index + operandOffset]}});
            for (auto [index, name] : llvm::enumerate(resultNames))
                if (!resultResourceSources || index >= static_cast<size_t>(resultResourceSources.size()) ||
                    resultResourceSources[index] < 0)
                    bindings.emplace_back(
                        llvm::json::Object{{"parameter", mlir::cast<mlir::StringAttr>(name).getValue().str()},
                                           {"value", resultIds[index]}});
            node["bindings"] = std::move(bindings);
            node["resources"] = std::move(resources);
            llvm::json::Array grid;
            if (gridControlArguments) {
                for (int64_t argumentIndex : gridControlArguments.asArrayRef()) {
                    auto valueId =
                        function.getArgAttrOfType<mlir::IntegerAttr>(argumentIndex, "vernon_program.value_id");
                    grid.emplace_back(llvm::json::Object{
                        {"control", llvm::json::Object{{"argument", valueId ? valueId.getInt() : -1}}}});
                }
            } else if (auto compute = mlir::dyn_cast<mlir::vernon::program::ComputeOp>(operation))
                for (int64_t extent : compute.getGrid())
                    grid.emplace_back(extent);
            else
                for (int64_t extent : {1, 1, 1})
                    grid.emplace_back(extent);
            node["grid"] = std::move(grid);

            llvm::json::Object request;
            request["id"] = requestId;
            request["graph"] = function.getSymName().str();
            request["direction"] = direction.getValue().str();
            request["kind"] = mlir::isa<mlir::vernon::program::ComputeOp>(operation) ? "compute" : "render";
            request["implementation_hint"] = stage.getValue().str();
            if (auto graphics = mlir::dyn_cast<mlir::vernon::program::GraphicsOp>(operation)) {
                request["topology"] = graphics.getTopology().str();
                request["color_count"] = graphics.getColorCount();
            }
            request["nodes"] = llvm::json::Array{nodeId.getInt()};
            llvm::json::Array requestOperands;
            llvm::json::Array requestResults;
            llvm::json::Array requestBindings;
            llvm::json::Array requestGrid;
            for (int64_t id : operandIds.asArrayRef())
                requestOperands.emplace_back(id);
            for (auto [index, id] : llvm::enumerate(resultIds.asArrayRef()))
                if (!resultResourceSources || index >= static_cast<size_t>(resultResourceSources.size()) ||
                    resultResourceSources[index] < 0)
                    requestResults.emplace_back(id);
            for (auto [index, name] : llvm::enumerate(operandNames))
                requestBindings.emplace_back(binding(name, operandIds[index + operandOffset], operandAutodiffRoles,
                                                     operandAutodiffSources, index));
            for (auto [index, name] : llvm::enumerate(resultNames))
                if (!resultResourceSources || index >= static_cast<size_t>(resultResourceSources.size()) ||
                    resultResourceSources[index] < 0)
                    requestBindings.emplace_back(
                        binding(name, resultIds[index], resultAutodiffRoles, resultAutodiffSources, index));
            if (gridControlArguments) {
                for (int64_t argumentIndex : gridControlArguments.asArrayRef()) {
                    auto valueId =
                        function.getArgAttrOfType<mlir::IntegerAttr>(argumentIndex, "vernon_program.value_id");
                    requestGrid.emplace_back(llvm::json::Object{
                        {"control", llvm::json::Object{{"argument", valueId ? valueId.getInt() : -1}}}});
                }
            } else if (auto compute = mlir::dyn_cast<mlir::vernon::program::ComputeOp>(operation))
                for (int64_t extent : compute.getGrid())
                    requestGrid.emplace_back(extent);
            else
                for (int64_t extent : {1, 1, 1})
                    requestGrid.emplace_back(extent);
            std::string region;
            llvm::raw_string_ostream regionStream(region);
            operation.print(regionStream);
            request["operands"] = std::move(requestOperands);
            request["results"] = std::move(requestResults);
            request["bindings"] = std::move(requestBindings);
            request["grid"] = std::move(requestGrid);
            request["region_mlir"] =
                operation.getAttrOfType<mlir::StringAttr>("vernon_program.region_mlir")
                    ? operation.getAttrOfType<mlir::StringAttr>("vernon_program.region_mlir").getValue().str()
                    : std::move(region);
            kernelCompileRequests.emplace_back(std::move(request));
            nodes.emplace_back(std::move(node));
        }
        graph["nodes"] = std::move(nodes);

        llvm::json::Array results;
        auto resultIds = function->getAttrOfType<mlir::DenseI64ArrayAttr>("vernon_program.result_value_ids");
        auto resultNames = function->getAttrOfType<mlir::ArrayAttr>("vernon_program.result_names");
        if (!resultIds || resultIds.size() != function.getNumResults()) {
            function.emitError("has invalid executable Program result ids");
            invalid = true;
        } else {
            for (auto [index, id] : llvm::enumerate(resultIds.asArrayRef())) {
                results.emplace_back(id);
                llvm::StringRef name =
                    resultNames && index < resultNames.size()
                        ? mlir::cast<mlir::StringAttr>(resultNames[index]).getValue()
                    : function.getResultAttrOfType<mlir::StringAttr>(index, "vernon.source_name")
                        ? function.getResultAttrOfType<mlir::StringAttr>(index, "vernon.source_name").getValue()
                        : llvm::StringRef("result");
                llvm::SmallVector<llvm::StringRef> logicalDtypes;
                if (auto dtypes = function.getResultAttrOfType<mlir::ArrayAttr>(index, "vernon.abi_leaf_dtypes"))
                    for (mlir::Attribute dtype : dtypes)
                        logicalDtypes.push_back(mlir::cast<mlir::StringAttr>(dtype).getValue());
                mlir::Type derivativeOf;
                if (direction.getValue() == "backward")
                    if (auto found = primalInputs.find(publicPath(name)); found != primalInputs.end())
                        derivativeOf = found->second;
                std::optional<llvm::StringRef> logicalDtype;
                if (auto dtype = function.getResultAttrOfType<mlir::StringAttr>(index, "vernon.dtype"))
                    logicalDtype = dtype.getValue();
                reflectValue(static_cast<uint32_t>(id), name, function.getResultTypes()[index], false,
                             direction.getValue() == "forward", true, logicalDtypes, derivativeOf, logicalDtype);
                llvm::StringRef path =
                    function.getResultAttrOfType<mlir::StringAttr>(index, "vernon.source_name")
                        ? function.getResultAttrOfType<mlir::StringAttr>(index, "vernon.source_name").getValue()
                        : name;
                if (direction.getValue() == "forward")
                    outputs.emplace_back(signatureBinding(static_cast<uint32_t>(id), path));
                else if (direction.getValue() == "backward")
                    gradients.emplace_back(signatureBinding(static_cast<uint32_t>(id), path));
            }
        }
        graph["results"] = std::move(results);
        graphs.emplace_back(std::move(graph));
    }
    if (invalid)
        return mlir::failure();
    for (auto [id, value] : llvm::enumerate(reflectedValues))
        if (!value) {
            module.emitError() << "executable Program contains unreflected value id " << id;
            return mlir::failure();
        }
    llvm::json::Array values;
    for (auto &value : reflectedValues)
        values.emplace_back(std::move(*value));
    llvm::json::Object signature;
    signature["inputs"] = std::move(inputs);
    signature["outputs"] = std::move(outputs);
    signature["cotangents"] = std::move(cotangents);
    signature["gradients"] = std::move(gradients);
    signature["captures"] = std::move(captures);
    return std::optional<ProgramReflection>(ProgramReflection{llvm::json::Object{{"values", std::move(values)},
                                                                                 {"graphs", std::move(graphs)},
                                                                                 {"signature", std::move(signature)}},
                                                              std::move(kernelCompileRequests)});
}

} // namespace vernon::compiler
