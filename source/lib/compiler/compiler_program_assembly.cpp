#include "compiler_program_assembly.h"

#include "compiler_program_abi.h"
#include "compiler_program_serializer.h"

#include "llvm/ADT/Twine.h"

#include <optional>
#include <utility>

namespace vernon::compiler {

bool assembleCanonicalProgram(const llvm::json::Object &rawSignature,
                              const std::vector<CanonicalProgramGraph> &selectedGraphs, llvm::json::Object stages,
                              llvm::json::Array storages, llvm::json::Array values,
                              std::map<std::string, llvm::json::Array> canonicalNodesByGraph,
                              const std::map<int64_t, int64_t> &storageByValue, const std::set<int64_t> &capturedValues,
                              llvm::json::Object &program, std::string &error) {
    llvm::json::Array canonicalGraphs;
    for (const CanonicalProgramGraph &view : selectedGraphs) {
        llvm::json::Array graphInputs;
        for (size_t slot = 0; slot < view.arguments.size(); ++slot)
            graphInputs.emplace_back(llvm::json::Object{
                {"tag", "user_input"}, {"value", view.arguments[slot]}, {"slot", static_cast<int64_t>(slot)}});
        for (const llvm::json::Value &rowValue : values) {
            const llvm::json::Object *row = rowValue.getAsObject();
            const llvm::json::Object *origin = row ? row->getObject("origin") : nullptr;
            if (!origin || origin->getString("tag") != "allocation" || origin->getString("graph") != view.name)
                continue;
            const int64_t valueId = row->getInteger("id").value_or(-1);
            const auto storage = storageByValue.find(valueId);
            if (storage == storageByValue.end()) {
                error = ("allocation origin has no Storage for value " + llvm::Twine(valueId) + " '" +
                         row->getString("name").value_or("") + "' type '" + row->getString("type").value_or("") + "'")
                            .str();
                return false;
            }
            graphInputs.emplace_back(
                llvm::json::Object{{"tag", "allocation"}, {"value", valueId}, {"storage", storage->second}});
        }
        llvm::json::Array graphOutputs;
        for (int64_t value : view.results)
            graphOutputs.emplace_back(
                llvm::json::Object{{"tag", "user_output"}, {"value", value}, {"disposition", "transfer"}});
        llvm::json::Array captures;
        if (view.direction == "backward")
            for (int64_t value : capturedValues)
                captures.emplace_back(llvm::json::Object{{"value", value}});
        canonicalGraphs.emplace_back(llvm::json::Object{{"name", view.name},
                                                        {"direction", view.direction},
                                                        {"inputs", std::move(graphInputs)},
                                                        {"captures", std::move(captures)},
                                                        {"outputs", std::move(graphOutputs)},
                                                        {"nodes", std::move(canonicalNodesByGraph[view.name])}});
    }

    const auto signatureRows = [&](llvm::StringRef field, bool output) {
        llvm::json::Array result;
        if (const llvm::json::Array *rows = rawSignature.getArray(field))
            for (const llvm::json::Value &rowValue : *rows)
                if (const llvm::json::Object *row = rowValue.getAsObject()) {
                    llvm::json::Object binding{{"path", row->getString("path").value_or("").str()},
                                               {"value", row->getInteger("value").value_or(-1)}};
                    if (output)
                        binding["disposition"] = "transfer";
                    result.emplace_back(std::move(binding));
                }
        return result;
    };
    llvm::json::Object signature{
        {"inputs", signatureRows("inputs", false)},
        {"outputs", signatureRows("outputs", true)},
        {"cotangents", signatureRows("cotangents", false)},
        {"gradients", signatureRows("gradients", true)},
    };
    llvm::json::Object abi;
    if (!buildCanonicalProgramAbi(signature, values, storages, canonicalGraphs, abi, error))
        return false;
    std::optional<llvm::json::Object> residualContract;
    if (selectedGraphs.size() > 1) {
        llvm::json::Array residualCaptures;
        for (int64_t value : capturedValues)
            residualCaptures.emplace_back(llvm::json::Object{
                {"value", value},
                {"replay", llvm::json::Object{
                               {"legal", false}, {"required_values", llvm::json::Array()}, {"cost", int64_t{0}}}}});
        residualContract =
            llvm::json::Object{{"captures", std::move(residualCaptures)}, {"shape_symbols", llvm::json::Array()}};
    }
    program = serializeCanonicalProgram({std::move(stages), std::move(storages), std::move(values),
                                         std::move(canonicalGraphs), std::move(abi), std::move(residualContract)});
    return true;
}

} // namespace vernon::compiler
