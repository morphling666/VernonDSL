#ifndef VERNON_COMPILER_PROGRAM_IMPLEMENTATION_H
#define VERNON_COMPILER_PROGRAM_IMPLEMENTATION_H

#include "llvm/Support/JSON.h"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace vernon::compiler {

struct ProgramImplementationBindingPlan {
    std::set<std::string> omittedParameters;
    std::map<std::string, std::vector<std::string>> parameterAliases;
};

bool normalizeProgramImplementationAbi(llvm::json::Object &execution, llvm::json::Object &request,
                                       const llvm::json::Object &compiledEntry, std::string &error);

bool applyProgramImplementationBindingPlan(llvm::json::Object &execution, llvm::StringRef graphName,
                                           llvm::StringRef requestId, const llvm::json::Array &requestBindings,
                                           const ProgramImplementationBindingPlan &plan, std::string &error);

} // namespace vernon::compiler

#endif
