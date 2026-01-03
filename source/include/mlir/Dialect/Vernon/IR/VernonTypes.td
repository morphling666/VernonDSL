//===- VernonTypes.td - Vernon Type Definitions -----------*- tablegen -*-===//
//
// Vernon DSL — MLIR-native design
//
//===----------------------------------------------------------------------===//

#ifndef VERNON_TYPES
#define VERNON_TYPES

include "mlir/Dialect/Vernon/IR/VernonDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Vernon Semantic Attributes
//===----------------------------------------------------------------------===//

// Optional: tag a value as having shader semantic meaning
def Vernon_SemanticAttr : AttrDef<Vernon_Dialect, "Semantic"> {
  let mnemonic = "semantic";
  let summary = "Shader semantic annotation";
  let description = [{
    Annotates a value with shader-level semantics such as
    color, normal, uv, position, etc.
  }];

  let parameters = (ins
    StrAttr:$name
  );

  let assemblyFormat = "`<` $name `>`";
}

#endif // VERNON_TYPES
