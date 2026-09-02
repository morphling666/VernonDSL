#include "compiler_kernel_bootstrap.h"
#include "compiler_value_metadata.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/VernonProgram/IR/VernonProgram.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <string>

namespace vernon::compiler {
namespace {

struct KernelArgument {
    unsigned sourceIndex{};
    mlir::Type type;
    std::string name;
    std::string access;
    bool resource{};
    bool writable{};
};

std::string sourceName(mlir::func::FuncOp function, unsigned index) {
    if (auto name = function.getArgAttrOfType<mlir::StringAttr>(index, "vernon.source_name");
        name && !name.getValue().empty())
        return name.getValue().str();
    return "argument." + std::to_string(index);
}

std::string resourceAccess(mlir::Type type) {
    if (auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(type))
        return view.getAccess().str();
    if (auto texture = mlir::dyn_cast<mlir::vernon::TextureType>(type)) {
        llvm::StringRef access = texture.getAccess();
        return access == "sampled" ? "read" : access.str();
    }
    return "read";
}

bool isResource(mlir::func::FuncOp function, unsigned index) {
    mlir::Type type = function.getArgument(index).getType();
    if (mlir::isa<mlir::vernon::TensorViewType, mlir::vernon::TextureType, mlir::vernon::SamplerType>(type))
        return true;
    auto interface = function.getArgAttrOfType<mlir::StringAttr>(index, "vernon.interface");
    return interface && interface.getValue() == "resource";
}

mlir::ArrayAttr languageLeafDtypes(mlir::OpBuilder &builder, mlir::func::FuncOp function, unsigned index, bool result) {
    auto attr = [&](llvm::StringRef name) -> mlir::Attribute {
        return result ? function.getResultAttr(index, name) : function.getArgAttr(index, name);
    };
    mlir::Type type = result ? function.getResultTypes()[index] : function.getArgument(index).getType();
    if (mlir::isa<mlir::vernon::TensorViewType>(type))
        if (auto dtypes = mlir::dyn_cast_if_present<mlir::ArrayAttr>(attr("vernon.element_abi_leaf_dtypes")))
            return dtypes;
    if (auto dtypes = mlir::dyn_cast_if_present<mlir::ArrayAttr>(attr("vernon.abi_leaf_dtypes")))
        return dtypes;
    if (auto dtype = mlir::dyn_cast_if_present<mlir::StringAttr>(attr("vernon.dtype")))
        return builder.getArrayAttr({dtype});
    return builder.getArrayAttr({});
}

} // namespace

VernonStatus planComputeKernel(CompilerFrontend &frontend, const char *source, size_t sourceSize,
                               std::vector<Artifact> &artifacts, std::string &reflection, std::string &diagnostics) {
    mlir::MLIRContext &context = compilerMlirContext(frontend);
    context.getOrLoadDialect<mlir::vernon::program::VernonProgramDialect>();
    mlir::ScopedDiagnosticHandler handler(
        &context, [&](mlir::Diagnostic &diagnostic) { appendDiagnostic(diagnostics, diagnostic); });
    llvm::StringRef text(source ? source : "", sourceSize);
    mlir::ParserConfig parserConfig(&context, /*verifyAfterParse=*/false);
    mlir::OwningOpRef<mlir::ModuleOp> kernel = mlir::parseSourceString<mlir::ModuleOp>(text, parserConfig);
    if (!kernel)
        return VERNON_STATUS_PARSE_ERROR;
    if (mlir::failed(mlir::verify(*kernel)))
        return VERNON_STATUS_VERIFICATION_ERROR;

    mlir::func::FuncOp entry;
    bool multipleEntries = false;
    kernel->walk([&](mlir::func::FuncOp function) {
        if (!function->hasAttr("vernon.entry"))
            return;
        multipleEntries |= static_cast<bool>(entry);
        if (!entry)
            entry = function;
    });
    auto stage = entry ? entry->getAttrOfType<mlir::StringAttr>("vernon.stage") : mlir::StringAttr{};
    if (!entry || multipleEntries || !stage || stage.getValue() != "compute") {
        (*kernel).emitError("kernel planning requires exactly one vernon.entry compute function");
        return VERNON_STATUS_VERIFICATION_ERROR;
    }
    if (entry.isExternal() || !llvm::hasSingleElement(entry.getBody())) {
        entry.emitError("kernel planning requires a defined single-block compute function");
        return VERNON_STATUS_VERIFICATION_ERROR;
    }

    llvm::StringMap<std::pair<bool, bool>> effects;
    if (auto rows = entry->getAttrOfType<mlir::ArrayAttr>("vernon.storage_effects"))
        for (mlir::Attribute value : rows)
            if (auto effect = mlir::dyn_cast<mlir::DictionaryAttr>(value)) {
                auto owner = effect.getAs<mlir::StringAttr>("owner");
                auto kind = effect.getAs<mlir::StringAttr>("kind");
                if (!owner || !kind ||
                    (kind.getValue() != "read" && kind.getValue() != "write" && kind.getValue() != "read_write")) {
                    entry.emitError("has a malformed vernon.storage_effects declaration");
                    return VERNON_STATUS_VERIFICATION_ERROR;
                }
                auto &access = effects[owner.getValue()];
                access.first |= kind.getValue() == "read";
                access.second |= kind.getValue() != "read";
            }

    llvm::SmallVector<KernelArgument> arguments;
    llvm::StringSet<> declaredArgumentNames;
    for (auto [index, argument] : llvm::enumerate(entry.getArguments())) {
        if (entry.getArgAttr(index, "vernon.builtin"))
            continue;
        KernelArgument planned;
        planned.sourceIndex = static_cast<unsigned>(index);
        planned.type = argument.getType();
        planned.name = sourceName(entry, planned.sourceIndex);
        planned.resource = isResource(entry, planned.sourceIndex);
        planned.access = planned.resource ? resourceAccess(planned.type) : "";
        declaredArgumentNames.insert(planned.name);
        if (auto found = effects.find(planned.name); found != effects.end()) {
            if (!planned.resource) {
                entry.emitError() << "storage effect owner '" << planned.name << "' is not a resource argument";
                return VERNON_STATUS_VERIFICATION_ERROR;
            }
            const std::string effectAccess = found->second.first && found->second.second ? "read_write"
                                             : found->second.second                      ? "write"
                                                                                         : "read";
            if ((found->second.first && planned.access == "write") ||
                (found->second.second && planned.access == "read")) {
                entry.emitError() << "storage effect access '" << effectAccess << "' conflicts with declared access '"
                                  << planned.access << "' for '" << planned.name << "'";
                return VERNON_STATUS_VERIFICATION_ERROR;
            }
        }
        planned.writable = planned.resource && planned.access != "read";
        arguments.push_back(std::move(planned));
    }
    for (const auto &[owner, unused] : effects) {
        (void)unused;
        if (!declaredArgumentNames.count(owner)) {
            entry.emitError() << "storage effect owner '" << owner << "' is not a kernel argument";
            return VERNON_STATUS_VERIFICATION_ERROR;
        }
    }

    mlir::OpBuilder builder(&context);
    mlir::OwningOpRef<mlir::ModuleOp> program = mlir::ModuleOp::create(entry.getLoc());
    builder.setInsertionPointToStart(program->getBody());
    for (mlir::vernon::StructDeclOp declaration : kernel->getOps<mlir::vernon::StructDeclOp>())
        builder.insert(declaration->clone());

    llvm::SmallVector<mlir::Type> graphInputs;
    for (const KernelArgument &argument : arguments)
        graphInputs.push_back(argument.type);
    mlir::IntegerType controlType = mlir::IntegerType::get(&context, 32, mlir::IntegerType::Unsigned);
    graphInputs.append(3, controlType);
    llvm::SmallVector<mlir::Type> graphResults(entry.getResultTypes().begin(), entry.getResultTypes().end());
    auto graph = mlir::func::FuncOp::create(entry.getLoc(), "forward",
                                            mlir::FunctionType::get(&context, graphInputs, graphResults));
    graph->setAttr("vernon_program.graph", builder.getStringAttr("forward"));

    llvm::SmallVector<mlir::Attribute> argumentNames;
    for (auto [index, argument] : llvm::enumerate(arguments)) {
        argumentNames.push_back(builder.getStringAttr("input." + argument.name));
        copyValueMetadata(entry, argument.sourceIndex, graph, static_cast<unsigned>(index));
        if (!graph.getArgAttr(index, "vernon.source_name"))
            graph.setArgAttr(index, "vernon.source_name", builder.getStringAttr(argument.name));
    }
    const unsigned controlBase = static_cast<unsigned>(arguments.size());
    for (auto [offset, control] :
         llvm::enumerate(llvm::ArrayRef<llvm::StringRef>{"groups_x", "groups_y", "groups_z"})) {
        argumentNames.push_back(builder.getStringAttr(("input." + control).str()));
        const unsigned index = controlBase + static_cast<unsigned>(offset);
        graph.setArgAttr(index, "vernon.source_name", builder.getStringAttr(control));
        graph.setArgAttr(index, "vernon.dtype", builder.getStringAttr("u32"));
        graph.setArgAttr(index, "vernon.abi_leaf_dtypes", builder.getArrayAttr({builder.getStringAttr("u32")}));
    }
    graph->setAttr("vernon_program.argument_names", builder.getArrayAttr(argumentNames));

    llvm::SmallVector<mlir::Attribute> resultNames;
    for (auto [index, type] : llvm::enumerate(entry.getResultTypes())) {
        (void)type;
        std::string name = "result." + std::to_string(index);
        if (auto sourceName = entry.getResultAttrOfType<mlir::StringAttr>(index, "vernon.source_name");
            sourceName && !sourceName.getValue().empty())
            name = sourceName.getValue().str();
        resultNames.push_back(builder.getStringAttr("output." + name));
        if (mlir::Attribute value = entry.getResultAttr(index, "vernon.source_name"))
            graph.setResultAttr(index, "vernon.source_name", value);
        else
            graph.setResultAttr(index, "vernon.source_name", builder.getStringAttr(name));
        for (llvm::StringRef attr : {"vernon.dtype", "vernon.source_shape", "vernon.abi_leaf_dtypes"})
            if (mlir::Attribute value = entry.getResultAttr(index, attr))
                graph.setResultAttr(index, attr, value);
    }
    graph->setAttr("vernon_program.result_names", builder.getArrayAttr(resultNames));
    builder.insert(graph);
    mlir::Block *block = graph.addEntryBlock();
    builder.setInsertionPointToStart(block);

    llvm::SmallVector<mlir::Type> operationResults;
    llvm::SmallVector<mlir::Attribute> operationResultNames;
    llvm::SmallVector<int64_t> resourceResultSources;
    for (auto [index, argument] : llvm::enumerate(arguments))
        if (argument.writable) {
            operationResults.push_back(argument.type);
            operationResultNames.push_back(builder.getStringAttr(argument.name + ".after"));
            resourceResultSources.push_back(static_cast<int64_t>(index));
        }
    const size_t resourceResultCount = operationResults.size();
    operationResults.append(entry.getResultTypes().begin(), entry.getResultTypes().end());
    for (auto [index, name] : llvm::enumerate(resultNames)) {
        llvm::StringRef spelling = mlir::cast<mlir::StringAttr>(name).getValue();
        operationResultNames.push_back(builder.getStringAttr(spelling.consume_front("output.") ? spelling : spelling));
        resourceResultSources.push_back(-1);
    }

    llvm::SmallVector<mlir::Value> operationOperands;
    llvm::SmallVector<mlir::Attribute> operandNames;
    llvm::SmallVector<mlir::Attribute> operandAccesses;
    llvm::SmallVector<int64_t> resourceOperandIndices;
    for (auto [index, argument] : llvm::enumerate(arguments)) {
        operationOperands.push_back(block->getArgument(index));
        operandNames.push_back(builder.getStringAttr(argument.name));
        operandAccesses.push_back(builder.getStringAttr(argument.resource ? argument.access : "read"));
        if (argument.resource)
            resourceOperandIndices.push_back(static_cast<int64_t>(index));
    }

    mlir::vernon::program::ComputeOp compute = mlir::vernon::program::ComputeOp::create(
        builder, entry.getLoc(), operationResults, operationOperands, entry.getSymName(),
        llvm::ArrayRef<int64_t>{1, 1, 1}, builder.getArrayAttr({}), builder.getArrayAttr(operandNames),
        builder.getArrayAttr(operationResultNames));
    compute->setAttr("vernon_program.grid_control_arguments",
                     builder.getDenseI64ArrayAttr({static_cast<int64_t>(arguments.size()),
                                                   static_cast<int64_t>(arguments.size() + 1),
                                                   static_cast<int64_t>(arguments.size() + 2)}));
    compute->setAttr("vernon_program.resource_operand_indices", builder.getDenseI64ArrayAttr(resourceOperandIndices));
    compute->setAttr("vernon_program.result_resource_sources", builder.getDenseI64ArrayAttr(resourceResultSources));
    compute->setAttr("vernon_program.operand_accesses", builder.getArrayAttr(operandAccesses));
    compute->setAttr("vernon_program.region_mlir", builder.getStringAttr(text));
    llvm::SmallVector<mlir::Attribute> resultLeafDtypes;
    for (const KernelArgument &argument : arguments)
        if (argument.writable)
            resultLeafDtypes.push_back(languageLeafDtypes(builder, entry, argument.sourceIndex, /*result=*/false));
    for (auto [index, type] : llvm::enumerate(entry.getResultTypes())) {
        (void)type;
        resultLeafDtypes.push_back(languageLeafDtypes(builder, entry, static_cast<unsigned>(index), /*result=*/true));
    }
    compute->setAttr("vernon_program.result_abi_leaf_dtypes", builder.getArrayAttr(resultLeafDtypes));

    llvm::SmallVector<mlir::Value> returned;
    for (size_t index = resourceResultCount; index < compute->getNumResults(); ++index)
        returned.push_back(compute->getResult(index));
    mlir::func::ReturnOp::create(builder, entry.getLoc(), returned);

    builder.clearInsertionPoint();
    PreparedModulePtr prepared;
    return prepareProgramModule(frontend, std::move(program), prepared, artifacts, reflection, diagnostics);
}

} // namespace vernon::compiler
