#include "VernonCompiler.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>

namespace {

std::optional<VernonTarget> parseTarget(std::string_view name) {
  if (name == "cpu")
    return VERNON_TARGET_CPU;
  if (name == "opengl")
    return VERNON_TARGET_OPENGL;
  if (name == "opengles")
    return VERNON_TARGET_OPENGL_ES;
  if (name == "vulkan")
    return VERNON_TARGET_VULKAN;
  if (name == "metal")
    return VERNON_TARGET_METAL;
  if (name == "directx")
    return VERNON_TARGET_DIRECTX;
  if (name == "cuda")
    return VERNON_TARGET_CUDA;
  return std::nullopt;
}

void writeView(std::ostream &stream, VernonStringView value) {
  if (value.data && value.size)
    stream.write(value.data, static_cast<std::streamsize>(value.size));
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: vernon-compile <module.mlir>\n"
                 "       vernon-compile --target <target> <module.mlir> "
                 "[--output-dir <directory>] [--reflection <file>]\n";
    return 2;
  }

  const bool validateOnly = std::string_view(argv[1]) != "--target";
  std::optional<VernonTarget> target;
  const char *inputPath = argv[1];
  std::optional<std::filesystem::path> outputDirectory;
  std::optional<std::filesystem::path> reflectionPath;
  if (!validateOnly) {
    if (argc < 4 || !(target = parseTarget(argv[2]))) {
      std::cerr << "unknown target\n";
      return 2;
    }
    inputPath = argv[3];
    for (int index = 4; index < argc; index += 2) {
      if (index + 1 >= argc) {
        std::cerr << "missing value for " << argv[index] << '\n';
        return 2;
      }
      std::string_view option = argv[index];
      if (option == "--output-dir")
        outputDirectory = argv[index + 1];
      else if (option == "--reflection")
        reflectionPath = argv[index + 1];
      else {
        std::cerr << "unknown option " << option << '\n';
        return 2;
      }
    }
  } else if (argc != 2) {
    std::cerr << "validation accepts exactly one input file\n";
    return 2;
  }

  std::ifstream input(inputPath, std::ios::binary);
  if (!input) {
    std::cerr << "cannot open " << inputPath << '\n';
    return 2;
  }
  std::string source((std::istreambuf_iterator<char>(input)),
                     std::istreambuf_iterator<char>());

  VernonCompilerContext *context = vernonCompilerCreate();
  if (!context) {
    std::cerr << "cannot create compiler context\n";
    return 1;
  }

  VernonCompileResult *result =
      validateOnly
          ? vernonCompilerValidateMlir(context, source.data(), source.size())
          : vernonCompilerCompileMlir(context, source.data(), source.size(),
                                      *target);
  const VernonStatus status = vernonCompileResultGetStatus(result);
  if (status != VERNON_STATUS_OK) {
    writeView(std::cerr, vernonCompileResultGetDiagnostics(result));
    std::cerr << '\n';
  } else {
    const size_t artifactCount = vernonCompileResultGetArtifactCount(result);
    for (size_t index = 0; index < artifactCount; ++index) {
      if (outputDirectory) {
        std::filesystem::create_directories(*outputDirectory);
        VernonStringView name =
            vernonCompileResultGetArtifactName(result, index);
        std::filesystem::path path =
            *outputDirectory / std::string(name.data, name.size);
        std::ofstream output(path, std::ios::binary);
        writeView(output, vernonCompileResultGetArtifactData(result, index));
      } else {
        if (artifactCount > 1) {
          std::cout << "// artifact: ";
          writeView(std::cout,
                    vernonCompileResultGetArtifactName(result, index));
          std::cout << '\n';
        }
        writeView(std::cout, vernonCompileResultGetArtifactData(result, index));
        if (index + 1 != artifactCount)
          std::cout << '\n';
      }
    }
    VernonStringView reflection = vernonCompileResultGetReflection(result);
    if (reflectionPath) {
      std::ofstream output(*reflectionPath, std::ios::binary);
      writeView(output, reflection);
    } else if (!outputDirectory) {
      std::cout << "\n// reflection\n";
      writeView(std::cout, reflection);
      std::cout << '\n';
    } else {
      std::filesystem::path path = *outputDirectory / "reflection.json";
      std::ofstream output(path, std::ios::binary);
      writeView(output, reflection);
    }
  }

  vernonCompileResultDestroy(result);
  vernonCompilerDestroy(context);
  return status == VERNON_STATUS_OK ? 0 : 1;
}
