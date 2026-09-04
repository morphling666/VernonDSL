#include "compiler_program_aggregation.h"

#include "compiler_json.h"
#include "compiler_program_target_aggregation.h"

#include <set>

namespace vernon::compiler {
namespace {

llvm::json::Array &arrayField(llvm::json::Object &object, llvm::StringRef name) {
    if (!object.getArray(name))
        object[name] = llvm::json::Array();
    return *object.getArray(name);
}

bool appendPortableFeatures(llvm::json::Object &portable, const llvm::json::Object &reflection, std::string &error) {
    llvm::json::Array &features = arrayField(portable, "required_features");
    const llvm::json::Array *incoming = reflection.getArray("required_features");
    if (!incoming)
        return true;
    std::set<std::string> known;
    for (const llvm::json::Value &feature : features)
        if (const std::optional<llvm::StringRef> name = feature.getAsString())
            known.insert(name->str());
    for (const llvm::json::Value &feature : *incoming) {
        const std::optional<llvm::StringRef> name = feature.getAsString();
        if (!name || name->empty()) {
            error = "compiled Program module has an invalid required feature";
            return false;
        }
        if (known.insert(name->str()).second)
            features.emplace_back(name->str());
    }
    return true;
}

} // namespace

bool appendCompiledProgramModule(CanonicalProgramStage &stage, const std::string &requestId,
                                 const std::string &implementationStageId, ProgramStageOperation operation,
                                 const llvm::json::Object &reflection, const llvm::json::Object &entry,
                                 std::string &error) {
    const std::optional<llvm::StringRef> role = entry.getString("stage");
    const std::optional<llvm::StringRef> entryName = entry.getString("name");
    if (!role || !entryName || role->empty() || entryName->empty()) {
        error = "compiled Program module has no entry identity";
        return false;
    }
    if (stage.modules.empty()) {
        stage.requestId = requestId;
        stage.implementationStageId = implementationStageId;
        stage.operation = operation;
        stage.portableReflection["entries"] = llvm::json::Array();
        stage.portableReflection["required_features"] = llvm::json::Array();
    } else if (stage.requestId != requestId || stage.implementationStageId != implementationStageId ||
               stage.operation != operation) {
        error = "compiled Program modules disagree on stage identity";
        return false;
    }
    if (stage.modules.count(role->str())) {
        error = "compiled Program stage has duplicate module role '" + role->str() + "'";
        return false;
    }
    if (!appendProgramTargetModuleMetadata(stage, reflection, error) ||
        !appendPortableFeatures(stage.portableReflection, reflection, error))
        return false;
    llvm::json::Array *entries = stage.portableReflection.getArray("entries");
    if (!entries)
        return error = "compiled Program aggregation lost its entries", false;
    entries->emplace_back(copyJsonObject(entry));
    stage.modules.emplace(role->str(), CompiledProgramModule{entryName->str(), copyJsonObject(entry)});
    return true;
}

bool finalizeProgramStageAggregation(const CanonicalProgramStage &stage, std::string &error) {
    if (stage.operation == ProgramStageOperation::Compute) {
        if (stage.modules.size() != 1 || !stage.modules.count("compute")) {
            error = "compute Program request requires exactly one compute module";
            return false;
        }
        return true;
    }
    if (stage.modules.size() != 2 || !stage.modules.count("fragment") || !stage.modules.count("vertex")) {
        error = "graphics Program request requires exactly one vertex and one fragment module";
        return false;
    }
    return true;
}

} // namespace vernon::compiler
