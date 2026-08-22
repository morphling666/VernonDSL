#include "compiler_graphics_bootstrap.h"
#include "compiler_value_metadata.h"

#include "mlir/AsmParser/AsmParser.h"
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
#include "llvm/Support/raw_ostream.h"

namespace vernon::compiler {
namespace {

std::string sourceName(mlir::func::FuncOp function, unsigned index) {
    if (auto name = function.getArgAttrOfType<mlir::StringAttr>(index, "vernon.source_name");
        name && !name.getValue().empty())
        return name.getValue().str();
    return {};
}

bool skipLogicalArgument(mlir::func::FuncOp function, unsigned index, llvm::StringRef stage) {
    if (function.getArgAttr(index, "vernon.builtin") || function.getArgAttr(index, "vernon.implicit"))
        return true;
    return stage == "fragment" && function.getArgAttr(index, "vernon.location");
}

} // namespace

VernonStatus planGraphicsProgram(CompilerFrontend &frontend, const std::vector<GraphicsStageSource> &stages,
                                 const std::string &topology, const std::vector<std::string> &features,
                                 const std::vector<std::string> &attachmentTypes, uint32_t colorCount,
                                 const std::vector<GraphicsPlanOperand> &operands, std::vector<Artifact> &artifacts,
                                 std::string &reflection, std::string &diagnostics) {
    mlir::MLIRContext &context = compilerMlirContext(frontend);
    context.getOrLoadDialect<mlir::vernon::program::VernonProgramDialect>();
    mlir::ScopedDiagnosticHandler handler(
        &context, [&](mlir::Diagnostic &diagnostic) { appendDiagnostic(diagnostics, diagnostic); });
    if (stages.empty() || topology.empty() || attachmentTypes.empty() || colorCount == 0 ||
        colorCount > attachmentTypes.size() || attachmentTypes.size() - colorCount > 1) {
        diagnostics = "graphics planning requires vertex/fragment sources, topology, and attachments";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }

    llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
    llvm::SmallVector<mlir::func::FuncOp> entries;
    llvm::StringMap<std::pair<unsigned, mlir::func::FuncOp>> argumentOwners;
    llvm::SmallVector<std::string> collectedNames;
    for (const GraphicsStageSource &stage : stages) {
        llvm::StringRef text(stage.source ? stage.source : "", stage.sourceSize);
        mlir::ParserConfig parserConfig(&context, /*verifyAfterParse=*/false);
        mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(text, parserConfig);
        if (!module)
            return VERNON_STATUS_PARSE_ERROR;
        if (mlir::failed(mlir::verify(*module)))
            return VERNON_STATUS_VERIFICATION_ERROR;
        mlir::func::FuncOp entry;
        bool multipleEntries = false;
        module->walk([&](mlir::func::FuncOp function) {
            if (!function->hasAttr("vernon.entry"))
                return;
            multipleEntries |= static_cast<bool>(entry);
            if (!entry)
                entry = function;
        });
        auto stageName = entry ? entry->getAttrOfType<mlir::StringAttr>("vernon.stage") : mlir::StringAttr{};
        if (!entry || multipleEntries || !stageName ||
            (stageName.getValue() != "vertex" && stageName.getValue() != "fragment")) {
            (*module).emitError("graphics planning requires one vertex or fragment vernon.entry per stage module");
            return VERNON_STATUS_VERIFICATION_ERROR;
        }
        for (auto [index, argument] : llvm::enumerate(entry.getArguments())) {
            (void)argument;
            if (skipLogicalArgument(entry, static_cast<unsigned>(index), stageName.getValue()))
                continue;
            const std::string name = sourceName(entry, static_cast<unsigned>(index));
            if (name.empty()) {
                entry.emitError("graphics argument is missing vernon.source_name");
                return VERNON_STATUS_VERIFICATION_ERROR;
            }
            auto found = argumentOwners.find(name);
            if (found == argumentOwners.end()) {
                argumentOwners.insert({name, {static_cast<unsigned>(index), entry}});
                collectedNames.push_back(name);
            } else if (found->second.second.getArgument(found->second.first).getType() !=
                       entry.getArgument(index).getType()) {
                entry.emitError() << "graphics stages disagree on logical argument '" << name << "'";
                return VERNON_STATUS_VERIFICATION_ERROR;
            }
        }
        entries.push_back(entry);
        modules.push_back(std::move(module));
    }

    llvm::StringSet<> providedNames;
    for (const GraphicsPlanOperand &operand : operands) {
        if (operand.name.empty() || operand.type.empty() || !providedNames.insert(operand.name).second) {
            diagnostics = "graphics operands must have unique names and MLIR types";
            return VERNON_STATUS_INVALID_ARGUMENT;
        }
        if (!argumentOwners.count(operand.name)) {
            diagnostics = "graphics operand '" + operand.name + "' is not a vertex or fragment argument";
            return VERNON_STATUS_INVALID_ARGUMENT;
        }
    }
    for (const std::string &name : collectedNames)
        if (!providedNames.count(name)) {
            diagnostics = "graphics Program is missing logical argument '" + name + "'";
            return VERNON_STATUS_INVALID_ARGUMENT;
        }

    llvm::SmallVector<mlir::Type> parsedAttachments;
    for (const std::string &spelling : attachmentTypes) {
        mlir::Type attachmentType = mlir::parseType(spelling, &context);
        if (!attachmentType || !mlir::isa<mlir::vernon::TextureType>(attachmentType)) {
            diagnostics = "graphics attachment type is not a vernon texture";
            return VERNON_STATUS_PARSE_ERROR;
        }
        parsedAttachments.push_back(attachmentType);
    }
    llvm::SmallVector<mlir::Type> operandTypes;
    for (const GraphicsPlanOperand &operand : operands) {
        auto owner = argumentOwners.find(operand.name);
        mlir::Type frontendType = owner->second.second.getArgument(owner->second.first).getType();
        mlir::Type invocationType = mlir::parseType(operand.type, &context);
        if (!invocationType) {
            diagnostics = "graphics operand '" + operand.name + "' has an invalid MLIR type";
            return VERNON_STATUS_PARSE_ERROR;
        }
        const bool invocationVertexBuffer = mlir::isa<mlir::vernon::TensorViewType>(invocationType) &&
                                            !mlir::isa<mlir::vernon::TensorViewType>(frontendType);
        operandTypes.push_back(invocationVertexBuffer ? invocationType : frontendType);
    }

    mlir::OpBuilder builder(&context);
    mlir::Location loc = entries.front().getLoc();
    mlir::OwningOpRef<mlir::ModuleOp> program = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToStart(program->getBody());
    llvm::StringSet<> structNames;
    for (mlir::OwningOpRef<mlir::ModuleOp> &module : modules)
        for (mlir::vernon::StructDeclOp declaration : module->getOps<mlir::vernon::StructDeclOp>()) {
            if (!structNames.insert(declaration.getSymName()).second)
                continue;
            builder.insert(declaration->clone());
        }

    llvm::SmallVector<mlir::Type> graphInputs(parsedAttachments.begin(), parsedAttachments.end());
    graphInputs.append(operandTypes.begin(), operandTypes.end());
    auto graph =
        mlir::func::FuncOp::create(loc, "forward", mlir::FunctionType::get(&context, graphInputs, parsedAttachments));
    graph->setAttr("vernon_program.graph", builder.getStringAttr("forward"));
    llvm::SmallVector<mlir::Attribute> argumentNames;
    llvm::SmallVector<mlir::Attribute> resultNames;
    llvm::SmallVector<std::string> attachmentNames;
    for (uint32_t index = 0; index < colorCount; ++index) {
        const std::string name = colorCount == 1 ? "target" : "color" + std::to_string(index);
        attachmentNames.push_back(name);
        argumentNames.push_back(builder.getStringAttr("input." + name));
        resultNames.push_back(builder.getStringAttr("output." + name));
    }
    if (parsedAttachments.size() > colorCount) {
        attachmentNames.emplace_back("depth");
        argumentNames.push_back(builder.getStringAttr("input.depth"));
        resultNames.push_back(builder.getStringAttr("output.depth"));
    }
    for (const GraphicsPlanOperand &operand : operands)
        argumentNames.push_back(builder.getStringAttr("input." + operand.name));
    graph->setAttr("vernon_program.argument_names", builder.getArrayAttr(argumentNames));
    graph->setAttr("vernon_program.result_names", builder.getArrayAttr(resultNames));
    for (auto [index, name] : llvm::enumerate(attachmentNames)) {
        graph.setArgAttr(static_cast<unsigned>(index), "vernon.source_name", builder.getStringAttr(name));
        graph.setResultAttr(static_cast<unsigned>(index), "vernon.source_name", builder.getStringAttr(name));
    }
    for (auto [index, operand] : llvm::enumerate(operands)) {
        const unsigned targetIndex = static_cast<unsigned>(index + attachmentNames.size());
        auto owner = argumentOwners.find(operand.name);
        copyValueMetadata(owner->second.second, owner->second.first, graph, targetIndex);
        graph.setArgAttr(targetIndex, "vernon.source_name", builder.getStringAttr(operand.name));
    }
    builder.insert(graph);
    mlir::Block *block = graph.addEntryBlock();
    builder.setInsertionPointToStart(block);

    llvm::SmallVector<mlir::Attribute> operandNameAttrs;
    llvm::SmallVector<mlir::Attribute> resultNameAttrs;
    llvm::SmallVector<mlir::Attribute> featureAttrs;
    for (const GraphicsPlanOperand &operand : operands)
        operandNameAttrs.push_back(builder.getStringAttr(operand.name));
    for (const std::string &name : attachmentNames)
        resultNameAttrs.push_back(builder.getStringAttr(name));
    for (const std::string &feature : features)
        featureAttrs.push_back(builder.getStringAttr(feature));
    llvm::SmallVector<mlir::Value> graphicsOperands(block->getArguments().begin(), block->getArguments().end());
    mlir::vernon::program::GraphicsOp graphics = mlir::vernon::program::GraphicsOp::create(
        builder, loc, parsedAttachments, graphicsOperands, builder.getStringAttr("interactive.graphics"),
        builder.getStringAttr(topology), builder.getArrayAttr(featureAttrs), builder.getArrayAttr(operandNameAttrs),
        builder.getArrayAttr(resultNameAttrs), builder.getI32IntegerAttr(static_cast<int32_t>(colorCount)));
    graphics->setAttr("vernon_program.graphics_bootstrap", builder.getUnitAttr());
    mlir::func::ReturnOp::create(builder, loc, graphics.getResults());

    std::string programText;
    llvm::raw_string_ostream stream(programText);
    program->print(stream);
    builder.clearInsertionPoint();
    PreparedModulePtr prepared;
    return prepareMlir(frontend, programText.data(), programText.size(), prepared, artifacts, reflection, diagnostics);
}

} // namespace vernon::compiler
