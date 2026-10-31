#pragma once

#include "llvm/ADT/StringRef.h"

#include <string>

namespace vernon::compiler {

struct HostObject {
    std::string object;
    std::string triple;
    std::string optimizedIr;
};

// The normalized triple Host objects target when none is given.
std::string defaultHostTriple();

// Compiles textual LLVM IR emitted by vernon_dsl.hostlang to a relocatable object. The IR carries no data layout or
// triple; both come from the selected target so the frontend's aggregate layouts follow the target's C ABI.
bool compileHostObject(llvm::StringRef ir, llvm::StringRef triple, llvm::StringRef cpu, llvm::StringRef features,
                       HostObject &result, std::string &diagnostics);

} // namespace vernon::compiler
