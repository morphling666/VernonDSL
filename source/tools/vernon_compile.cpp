#include "VernonCompiler.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

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

std::string_view targetName(VernonTarget target) {
  switch (target) {
  case VERNON_TARGET_OPENGL:
    return "opengl";
  case VERNON_TARGET_OPENGL_ES:
    return "opengles";
  case VERNON_TARGET_METAL:
    return "metal";
  case VERNON_TARGET_CUDA:
    return "cuda";
  case VERNON_TARGET_CPU:
    return "cpu";
  case VERNON_TARGET_VULKAN:
    return "vulkan";
  case VERNON_TARGET_DIRECTX:
    return "directx";
  }
  return "unknown";
}

void writeView(std::ostream &stream, VernonStringView value) {
  if (value.data && value.size)
    stream.write(value.data, static_cast<std::streamsize>(value.size));
}

std::string_view hostOperatingSystem() {
#if defined(_WIN32)
  return "windows";
#elif defined(__APPLE__)
  return "macos";
#elif defined(__linux__)
  return "linux";
#else
  return "unknown";
#endif
}

std::string_view hostArchitecture() {
#if defined(_M_X64) || defined(__x86_64__)
  return "x86_64";
#elif defined(_M_ARM64) || defined(__aarch64__)
  return "aarch64";
#else
  return "unknown";
#endif
}

std::string_view nativeLibraryFilename() {
#if defined(_WIN32)
  return "compute.dll";
#elif defined(__APPLE__)
  return "compute.dylib";
#else
  return "compute.so";
#endif
}

std::optional<std::string>
findClang(const std::optional<std::filesystem::path> &overridePath,
          const char *argv0, std::string &diagnostic) {
  auto checkExplicit =
      [&](const std::filesystem::path &path,
          std::string_view source) -> std::optional<std::string> {
    std::error_code error;
    if (std::filesystem::is_regular_file(path, error))
      return path.string();
    diagnostic = std::string(source) +
                 " does not name a clang executable: " + path.string();
    return std::nullopt;
  };
  if (overridePath)
    return checkExplicit(*overridePath, "--clang");
  if (std::optional<std::string> environment =
          llvm::sys::Process::GetEnv("VERNON_CLANG"))
    return checkExplicit(*environment, "VERNON_CLANG");

#if defined(_WIN32)
  constexpr std::string_view clangName = "clang.exe";
#else
  constexpr std::string_view clangName = "clang";
#endif
  std::vector<std::filesystem::path> toolchainCandidates;
#if defined(VERNON_LLVM_TOOLS_DIR)
  toolchainCandidates.emplace_back(
      std::filesystem::path(VERNON_LLVM_TOOLS_DIR) / clangName);
#endif
  const std::filesystem::path executable =
      llvm::sys::fs::getMainExecutable(argv0, nullptr);
  toolchainCandidates.emplace_back(executable.parent_path() / clangName);
  for (const std::filesystem::path &candidate : toolchainCandidates) {
    std::error_code error;
    if (std::filesystem::is_regular_file(candidate, error))
      return candidate.string();
  }
  if (llvm::ErrorOr<std::string> path = llvm::sys::findProgramByName(clangName))
    return *path;
  diagnostic =
      "cannot find clang from the LLVM toolchain; pass --clang <path> or set "
      "VERNON_CLANG";
  return std::nullopt;
}

bool compileNativeLibrary(const std::string &clang,
                          const std::filesystem::path &llvmIr,
                          const std::filesystem::path &output) {
  std::vector<std::string> storage = {clang, llvmIr.string(), "-shared", "-O2",
                                      "-o",  output.string()};
  llvm::SmallVector<llvm::StringRef> arguments;
  for (const std::string &argument : storage)
    arguments.push_back(argument);
  std::string error;
  bool executionFailed = false;
  const int exitCode = llvm::sys::ExecuteAndWait(
      clang, arguments, std::nullopt, {}, 0, 0, &error, &executionFailed);
  if (executionFailed || exitCode != 0) {
    std::cerr << "clang failed to create " << output.string();
    if (!error.empty())
      std::cerr << ": " << error;
    std::cerr << " (exit code " << exitCode << ")\n"
              << "command:";
    for (const std::string &argument : storage)
      std::cerr << ' ' << argument;
    std::cerr << '\n';
    return false;
  }
#if defined(_WIN32)
  for (std::string_view extension : {".lib", ".exp"}) {
    std::filesystem::path linkerArtifact = output;
    linkerArtifact.replace_extension(extension);
    std::error_code error;
    std::filesystem::remove(linkerArtifact, error);
  }
#endif
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: vernon-compile <module.mlir>\n"
                 "       vernon-compile --target <target> <module.mlir> "
                 "[--output-dir <directory>] [--reflection <file>] "
                 "[--glsl-version <version>] "
                 "[--bundle <directory> --asset-id <id>] "
                 "[--compute-bundle <directory>] [--clang <path>] "
                 "[--keep-llvm-ir]\n"
                 "  --bundle writes a compiled OpenGL shader asset.\n"
                 "  --compute-bundle writes a runtime compute bundle; pipeline "
                 "bundles are assembled by vernon_dsl.shader_asset_cli.\n";
    return 2;
  }

  const bool validateOnly = std::string_view(argv[1]) != "--target";
  std::optional<VernonTarget> target;
  const char *inputPath = argv[1];
  std::optional<std::filesystem::path> outputDirectory;
  std::optional<std::filesystem::path> reflectionPath;
  std::optional<std::filesystem::path> bundlePath;
  std::optional<std::filesystem::path> computeBundlePath;
  std::optional<std::filesystem::path> clangPath;
  std::optional<std::string> assetId;
  std::optional<uint32_t> glslVersion;
  bool keepLlvmIr = false;
  if (!validateOnly) {
    if (argc < 4 || !(target = parseTarget(argv[2]))) {
      std::cerr << "unknown target\n";
      return 2;
    }
    inputPath = argv[3];
    for (int index = 4; index < argc;) {
      std::string_view option = argv[index];
      if (option == "--keep-llvm-ir") {
        keepLlvmIr = true;
        ++index;
        continue;
      }
      if (index + 1 >= argc) {
        std::cerr << "missing value for " << argv[index] << '\n';
        return 2;
      }
      if (option == "--output-dir")
        outputDirectory = argv[index + 1];
      else if (option == "--reflection")
        reflectionPath = argv[index + 1];
      else if (option == "--bundle")
        bundlePath = argv[index + 1];
      else if (option == "--compute-bundle")
        computeBundlePath = argv[index + 1];
      else if (option == "--clang")
        clangPath = argv[index + 1];
      else if (option == "--asset-id")
        assetId = argv[index + 1];
      else if (option == "--glsl-version") {
        std::string_view value = argv[index + 1];
        uint32_t parsed = 0;
        auto [end, error] =
            std::from_chars(value.data(), value.data() + value.size(), parsed);
        if (error != std::errc() || end != value.data() + value.size()) {
          std::cerr << "invalid GLSL version " << value << '\n';
          return 2;
        }
        glslVersion = parsed;
      } else {
        std::cerr << "unknown option " << option << '\n';
        return 2;
      }
      index += 2;
    }
    if (bundlePath.has_value() != assetId.has_value()) {
      std::cerr << "--bundle and --asset-id must be specified together\n";
      return 2;
    }
    if (bundlePath && *target != VERNON_TARGET_OPENGL) {
      std::cerr << "--bundle writes a compiled shader asset and requires "
                   "target opengl; use --compute-bundle for runtime compute "
                   "artifacts\n";
      return 2;
    }
    if (bundlePath && outputDirectory) {
      std::cerr << "--bundle and --output-dir cannot be combined\n";
      return 2;
    }
    if (computeBundlePath && (bundlePath || outputDirectory)) {
      std::cerr << "--compute-bundle cannot be combined with --bundle or "
                   "--output-dir\n";
      return 2;
    }
    if ((clangPath || keepLlvmIr) &&
        (!computeBundlePath || *target != VERNON_TARGET_CPU)) {
      std::cerr << "--clang and --keep-llvm-ir require a CPU "
                   "--compute-bundle\n";
      return 2;
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

  VernonCompileResult *result = nullptr;
  if (validateOnly) {
    result = vernonCompilerValidateMlir(context, source.data(), source.size());
  } else if (glslVersion) {
    VernonCompileOptions options = {};
    options.struct_size = sizeof(options);
    options.glsl_version = *glslVersion;
    result = vernonCompilerCompileMlirWithOptions(
        context, source.data(), source.size(), *target, &options);
  } else {
    result = vernonCompilerCompileMlir(context, source.data(), source.size(),
                                       *target);
  }
  VernonStatus status = vernonCompileResultGetStatus(result);
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
      } else if (!bundlePath && !computeBundlePath) {
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
    if (bundlePath) {
      llvm::json::Object bundle;
      bundle["schema_version"] = int64_t{1};
      bundle["type"] = "compiled_shader_bundle";
      bundle["id"] = *assetId;
      bundle["target"] = std::string(targetName(*target));
      llvm::StringRef reflectionText(reflection.data, reflection.size);
      llvm::Expected<llvm::json::Value> parsedReflection =
          llvm::json::parse(reflectionText);
      if (!parsedReflection) {
        std::cerr << "compiler produced invalid reflection JSON\n";
        status = VERNON_STATUS_INTERNAL_ERROR;
      } else {
        bundle["reflection"] = std::move(*parsedReflection);
        std::error_code error;
        std::filesystem::create_directories(*bundlePath, error);
        if (error || !std::filesystem::is_directory(*bundlePath)) {
          std::cerr << "cannot create bundle directory " << bundlePath->string()
                    << '\n';
          status = VERNON_STATUS_INTERNAL_ERROR;
        } else {
          for (size_t index = 0; index < artifactCount; ++index) {
            VernonStringView name =
                vernonCompileResultGetArtifactName(result, index);
            VernonStringView data =
                vernonCompileResultGetArtifactData(result, index);
            std::filesystem::path artifactPath =
                *bundlePath / std::string(name.data, name.size);
            std::ofstream artifactOutput(artifactPath, std::ios::binary);
            writeView(artifactOutput, data);
            artifactOutput.close();
            if (!artifactOutput) {
              std::cerr << "cannot write bundle artifact "
                        << artifactPath.string() << '\n';
              status = VERNON_STATUS_INTERNAL_ERROR;
              break;
            }
          }
          if (status == VERNON_STATUS_OK) {
            std::filesystem::path manifestPath = *bundlePath / "shader.json";
            std::ofstream output(manifestPath, std::ios::binary);
            std::string encoded;
            llvm::raw_string_ostream stream(encoded);
            stream << llvm::formatv("{0:2}",
                                    llvm::json::Value(std::move(bundle)));
            stream.flush();
            output << encoded << '\n';
            output.close();
            if (!output) {
              std::cerr << "cannot write bundle manifest "
                        << manifestPath.string() << '\n';
              status = VERNON_STATUS_INTERNAL_ERROR;
            }
          }
        }
      }
    } else if (computeBundlePath) {
      llvm::Expected<llvm::json::Value> parsedReflection =
          llvm::json::parse(llvm::StringRef(reflection.data, reflection.size));
      llvm::json::Object *reflectionObject =
          parsedReflection ? parsedReflection->getAsObject() : nullptr;
      llvm::json::Array *entries =
          reflectionObject ? reflectionObject->getArray("entries") : nullptr;
      llvm::json::Object *computeEntry = nullptr;
      if (entries) {
        for (llvm::json::Value &entryValue : *entries) {
          llvm::json::Object *entry = entryValue.getAsObject();
          if (entry && entry->getString("stage").value_or("") == "compute") {
            if (computeEntry) {
              std::cerr
                  << "compute bundles require exactly one compute entry\n";
              status = VERNON_STATUS_INVALID_ARGUMENT;
              break;
            }
            computeEntry = entry;
          }
        }
      }
      if (status == VERNON_STATUS_OK && (!computeEntry || artifactCount != 1)) {
        std::cerr
            << "compute bundle requires one compute entry and one artifact\n";
        status = VERNON_STATUS_INVALID_ARGUMENT;
      }
      if (status == VERNON_STATUS_OK) {
        std::error_code error;
        std::filesystem::create_directories(*computeBundlePath, error);
        if (error || !std::filesystem::is_directory(*computeBundlePath)) {
          std::cerr << "cannot create compute bundle directory "
                    << computeBundlePath->string() << '\n';
          status = VERNON_STATUS_INTERNAL_ERROR;
        }
      }
      if (status == VERNON_STATUS_OK) {
        VernonStringView artifactName =
            vernonCompileResultGetArtifactName(result, 0);
        VernonStringView artifactData =
            vernonCompileResultGetArtifactData(result, 0);
        std::string artifactFilename(artifactName.data, artifactName.size);
        std::string packagedArtifact(artifactData.data, artifactData.size);
        if (*target == VERNON_TARGET_CPU) {
          if (artifactFilename != "module.ll") {
            std::cerr << "CPU compilation did not produce module.ll\n";
            status = VERNON_STATUS_INTERNAL_ERROR;
          } else {
            const std::filesystem::path llvmIrPath =
                *computeBundlePath / "module.ll";
            std::ofstream llvmIrOutput(llvmIrPath, std::ios::binary);
            writeView(llvmIrOutput, artifactData);
            llvmIrOutput.close();
            if (!llvmIrOutput) {
              std::cerr << "cannot write temporary CPU LLVM IR "
                        << llvmIrPath.string() << '\n';
              status = VERNON_STATUS_INTERNAL_ERROR;
            } else {
              std::string clangDiagnostic;
              std::optional<std::string> clang =
                  findClang(clangPath, argv[0], clangDiagnostic);
              if (!clang) {
                std::cerr << clangDiagnostic << '\n';
                status = VERNON_STATUS_INTERNAL_ERROR;
              } else {
                artifactFilename = std::string(nativeLibraryFilename());
                const std::filesystem::path nativePath =
                    *computeBundlePath / artifactFilename;
                if (!compileNativeLibrary(*clang, llvmIrPath, nativePath)) {
                  status = VERNON_STATUS_INTERNAL_ERROR;
                } else {
                  std::ifstream nativeInput(nativePath, std::ios::binary);
                  packagedArtifact.assign(
                      std::istreambuf_iterator<char>(nativeInput),
                      std::istreambuf_iterator<char>());
                  if (!nativeInput || packagedArtifact.empty()) {
                    std::cerr << "clang did not produce a readable native "
                                 "library at "
                              << nativePath.string() << '\n';
                    status = VERNON_STATUS_INTERNAL_ERROR;
                  }
                }
              }
            }
            if (!keepLlvmIr) {
              std::error_code removeError;
              std::filesystem::remove(llvmIrPath, removeError);
              if (removeError && status == VERNON_STATUS_OK) {
                std::cerr << "cannot remove temporary LLVM IR "
                          << llvmIrPath.string() << ": "
                          << removeError.message() << '\n';
                status = VERNON_STATUS_INTERNAL_ERROR;
              }
            }
          }
        } else {
          std::ofstream artifactOutput(*computeBundlePath / artifactFilename,
                                       std::ios::binary);
          writeView(artifactOutput, artifactData);
          artifactOutput.close();
          if (!artifactOutput) {
            std::cerr << "cannot write compute artifact " << artifactFilename
                      << '\n';
            status = VERNON_STATUS_INTERNAL_ERROR;
          }
        }
        if (status != VERNON_STATUS_OK)
          packagedArtifact.clear();
        llvm::ArrayRef<uint8_t> bytes(
            reinterpret_cast<const uint8_t *>(packagedArtifact.data()),
            packagedArtifact.size());
        std::string digest = llvm::toHex(llvm::SHA256::hash(bytes), true);
        llvm::json::Object manifest;
        manifest["schema_version"] =
            *target == VERNON_TARGET_CPU ? int64_t{2} : int64_t{1};
        manifest["compiler_version"] = "0.1.0";
        if (*target == VERNON_TARGET_CPU) {
          manifest["cpu_invocation_abi_version"] = int64_t{1};
          manifest["operating_system"] = std::string(hostOperatingSystem());
          manifest["architecture"] = std::string(hostArchitecture());
        } else {
          manifest["gpu_launch_abi_version"] = int64_t{1};
        }
        manifest["target"] = std::string(targetName(*target));
        manifest["entry"] = computeEntry->getString("name").value_or("").str();
        manifest["symbol"] =
            computeEntry->getString("symbol").value_or("").str();
        manifest["artifact"] = artifactFilename;
        manifest["artifact_format"] =
            *target == VERNON_TARGET_CPU      ? "native_library"
            : *target == VERNON_TARGET_CUDA   ? "ptx"
            : *target == VERNON_TARGET_VULKAN ? "spirv"
            : *target == VERNON_TARGET_METAL  ? "msl"
                                              : "llvm_ir";
        manifest["artifact_size"] =
            static_cast<int64_t>(packagedArtifact.size());
        manifest["artifact_sha256"] = digest;
        manifest["reflection"] = std::move(*parsedReflection);
        if (status == VERNON_STATUS_OK) {
          std::ofstream manifestOutput(*computeBundlePath / "compute.json",
                                       std::ios::binary);
          std::string encoded;
          llvm::raw_string_ostream stream(encoded);
          stream << llvm::formatv("{0:2}",
                                  llvm::json::Value(std::move(manifest)));
          stream.flush();
          manifestOutput << encoded << '\n';
          manifestOutput.close();
          if (!manifestOutput) {
            std::cerr << "cannot write compute bundle manifest\n";
            status = VERNON_STATUS_INTERNAL_ERROR;
          }
        }
      }
    } else if (reflectionPath) {
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
