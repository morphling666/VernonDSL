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

def Vernon_StructType : TypeDef<Vernon_Dialect, "Struct"> {
  let mnemonic = "struct";
  let summary = "Reference to a named DSL struct declaration";
  let parameters = (ins StringRefParameter<"struct name">:$name);
  let assemblyFormat = "`<` $name `>`";
}

def Vernon_TensorType : TypeDef<Vernon_Dialect, "Tensor"> {
  let mnemonic = "tensor";
  let summary = "Static Tensor Value whose element is an aggregate Value";
  let parameters = (ins
    TypeParameter<"::mlir::Type", "element type">:$elementType,
    ArrayRefParameter<"int64_t", "static shape">:$shape
  );
  let assemblyFormat = "`<` $elementType `,` `[` $shape `]` `>`";
}

def Vernon_TensorViewType : TypeDef<Vernon_Dialect, "TensorView"> {
  let mnemonic = "tensor_view";
  let summary = "Borrowed shaped storage with runtime layout";
  let parameters = (ins
    TypeParameter<"::mlir::Type", "element type">:$elementType,
    ArrayRefParameter<"int64_t", "logical shape">:$shape,
    StringRefParameter<"read, write, or read_write access">:$access,
    StringRefParameter<"device, workgroup, or private address space">:$addressSpace
  );
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` $elementType `,` `[` $shape `]` `,` $access `,` $addressSpace `>`";
}

def Vernon_TextureType : TypeDef<Vernon_Dialect, "Texture"> {
  let mnemonic = "texture";
  let summary = "Sampled or storage texture resource";
  let parameters = (ins
    StringRefParameter<"texture dimension">:$dimension,
    TypeParameter<"::mlir::Type", "sample element type">:$elementType
  );
  let assemblyFormat = "`<` $dimension `,` $elementType `>`";
}

def Vernon_SamplerType : TypeDef<Vernon_Dialect, "Sampler"> {
  let mnemonic = "sampler";
  let summary = "Texture sampling state";
}

//===----------------------------------------------------------------------===//
// Internal logical autodiff types
//===----------------------------------------------------------------------===//

def Vernon_AdTapeType : TypeDef<Vernon_Dialect, "AdTape"> {
  let mnemonic = "ad_tape";
  let summary = "Backend-neutral ownership handle for one autodiff invocation tape";
  let description = [{
    An internal logical handle for the tape owned by one differentiated
    invocation. Its physical representation is intentionally unspecified.
  }];
}

def Vernon_AdRegionHeaderType : TypeDef<Vernon_Dialect, "AdRegionHeader"> {
  let mnemonic = "ad_region_header";
  let summary = "Backend-neutral handle for one dynamic autodiff region";
  let description = [{
    Identifies a checked dynamic-region header in an invocation tape. Region
    headers record logical record location, execution count, exit kind, and
    nested-region identity without exposing a physical backend layout.
  }];
}

def Vernon_AdAdjointBufferType : TypeDef<Vernon_Dialect, "AdAdjointBuffer"> {
  let mnemonic = "ad_adjoint_buffer";
  let summary = "Invocation-local mutable Storage adjoint state";
  let parameters = (ins
    TypeParameter<"::mlir::Type", "scalar derivative type">:$elementType,
    ArrayRefParameter<"int64_t", "flattened runtime-capable shape">:$shape,
    "unsigned":$indexRank
  );
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` $elementType `,` `[` $shape `]` `,` $indexRank `>`";
}

#endif // VERNON_TYPES
