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

def Vernon_ArrayType : TypeDef<Vernon_Dialect, "Array"> {
  let mnemonic = "array";
  let summary = "Fixed-size array used by shader interfaces and structs";
  let parameters = (ins
    AttrOrTypeParameter<"unsigned", "number of elements">:$size,
    TypeParameter<"::mlir::Type", "element type">:$elementType
  );
  let assemblyFormat = "`<` $size `x` $elementType `>`";
}

def Vernon_StructType : TypeDef<Vernon_Dialect, "Struct"> {
  let mnemonic = "struct";
  let summary = "Reference to a named DSL struct declaration";
  let parameters = (ins StringRefParameter<"struct name">:$name);
  let assemblyFormat = "`<` $name `>`";
}

def Vernon_BufferType : TypeDef<Vernon_Dialect, "Buffer"> {
  let mnemonic = "buffer";
  let summary = "Addressable compute or graphics storage buffer";
  let parameters = (ins
    TypeParameter<"::mlir::Type", "element type">:$elementType,
    StringRefParameter<"read, write, or read_write access">:$access
  );
  let assemblyFormat = "`<` $elementType `,` $access `>`";
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

#endif // VERNON_TYPES
