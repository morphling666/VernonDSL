#include "compiler_internal.h"

#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/raw_ostream.h"

namespace vernon::compiler {

void appendDiagnostic(std::string &output, mlir::Diagnostic &diagnostic) {
    llvm::raw_string_ostream stream(output);
    if (!output.empty())
        stream << '\n';
    stream << diagnostic.getLocation() << ": " << diagnostic;
}

} // namespace vernon::compiler
