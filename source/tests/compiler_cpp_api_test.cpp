#include "vernon-c/Compiler.h"

#include <cassert>
#include <string_view>

int main() {
  constexpr std::string_view module = R"mlir(
module {
  func.func @fragment_main(
      %color: tensor<4xf32> {
        vernon.interface = "input",
        vernon.location = 0 : i64
      }) -> (tensor<4xf32> {
        vernon.interface = "output",
        vernon.location = 0 : i64
      }) attributes {
        vernon.entry,
        vernon.stage = "fragment"
      } {
    return %color : tensor<4xf32>
  }
}
)mlir";

  VernonCompilerContext *context = vernonCompilerCreate();
  assert(context);

  VernonCompileResult *result =
      vernonCompilerValidateMlir(context, module.data(), module.size());
  assert(result);
  assert(vernonCompileResultGetStatus(result) == VERNON_STATUS_OK);
  assert(vernonCompileResultGetArtifactCount(result) == 1);
  VernonStringView artifactName = vernonCompileResultGetArtifactName(result, 0);
  assert(std::string_view(artifactName.data, artifactName.size) ==
         "module.mlir");

  VernonStringView reflection = vernonCompileResultGetReflection(result);
  std::string_view reflectionView(reflection.data, reflection.size);
  assert(reflectionView.find("\"fragment_main\"") != std::string_view::npos);
  assert(reflectionView.find("\"results\"") != std::string_view::npos);
  assert(reflectionView.find("\"vernon.location\":0") !=
         std::string_view::npos);

  vernonCompileResultDestroy(result);
  vernonCompilerDestroy(context);
  return 0;
}
