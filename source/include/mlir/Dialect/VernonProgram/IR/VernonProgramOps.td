//===- VernonProgramOps.td - Program operation definitions ------*- tablegen -*-===//

#ifndef VERNON_PROGRAM_OPS
#define VERNON_PROGRAM_OPS

include "mlir/Dialect/VernonProgram/IR/VernonProgramDialect.td"

class VernonProgram_Op<string mnemonic, list<Trait> traits = []>
    : Op<VernonProgram_Dialect, mnemonic, traits>;

def VernonProgram_ComputeOp : VernonProgram_Op<"compute"> {
  let summary = "Invoke a custom compute implementation";
  let arguments = (ins Variadic<AnyType>:$arguments, StrAttr:$callee,
                       DenseI64ArrayAttr:$grid, StrArrayAttr:$features,
                       StrArrayAttr:$operand_names, StrArrayAttr:$result_names);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
}

def VernonProgram_GraphicsOp : VernonProgram_Op<"graphics"> {
  let summary = "Invoke a graphics pipeline with color and optional depth attachments";
  let description = [{
    Graphics operations are independently schedulable but are not currently
    differentiable. Topology is static pipeline state. The first `color_count`
    operands/results are color attachments; one extra operand/result is an
    optional depth/stencil attachment. Remaining operands are logical shader
    arguments named by `operand_names`.
  }];
  let arguments = (ins Variadic<AnyType>:$operands, StrAttr:$callee, StrAttr:$topology,
                       StrArrayAttr:$features, StrArrayAttr:$operand_names,
                       StrArrayAttr:$result_names,
                       DefaultValuedAttr<I32Attr, "1">:$color_count);
  let results = (outs Variadic<AnyType>:$results);
  let extraClassDeclaration = [{
    ::mlir::ValueRange getAttachments() {
      return getOperands().take_front(getNumResults());
    }
    ::mlir::ValueRange getLogicalArguments() {
      return getOperands().drop_front(getNumResults());
    }
  }];
  let hasVerifier = 1;
}

#endif // VERNON_PROGRAM_OPS
