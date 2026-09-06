#ifndef VERNON_COMPILER_PROGRAM_PUBLICATION_H
#define VERNON_COMPILER_PROGRAM_PUBLICATION_H

#include "compiler_program_boundary.h"

#include "llvm/Support/JSON.h"

namespace vernon::compiler {

llvm::json::Array serializeProgramBoundarySlots(const ProgramBoundaryPlan &boundaries);

} // namespace vernon::compiler

#endif
