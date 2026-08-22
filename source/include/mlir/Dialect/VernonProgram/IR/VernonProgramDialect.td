//===- VernonProgramDialect.td - Program scheduling dialect -----*- tablegen -*-===//

#ifndef VERNON_PROGRAM_DIALECT
#define VERNON_PROGRAM_DIALECT

include "mlir/IR/OpBase.td"

def VernonProgram_Dialect : Dialect {
  let name = "vernon_program";
  let cppNamespace = "::mlir::vernon::program";
  let description = [{
    Scheduling-level Vernon Program IR. It contains calls to compiled kernel or
    graphics implementations and keeps forward and backward graphs independent
    from kernel-function IR.
  }];
}

#endif // VERNON_PROGRAM_DIALECT
