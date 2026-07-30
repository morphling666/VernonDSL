#include "vernon_compile_packaging.h"

#include "VernonVersions.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>
#include <string_view>

namespace {

void writeView(std::ostream &stream, VernonStringView value) {
    if (value.data && value.size)
        stream.write(value.data, static_cast<std::streamsize>(value.size));
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

std::string objectFormat(const llvm::Triple &triple) {
    if (triple.isOSBinFormatCOFF())
        return "coff";
    if (triple.isOSBinFormatMachO())
        return "macho";
    if (triple.isWasm())
        return "wasm";
    return "elf";
}

bool linkHostLibrary(VernonCompilerContext *context, VernonStringView object, std::string &filename,
                     std::string &library, std::string &diagnostic) {
    VernonCompileResult *linked = vernonCompilerLinkHostObject(context, object.data, object.size);
    if (!linked || vernonCompileResultGetStatus(linked) != VERNON_STATUS_OK) {
        if (linked) {
            VernonStringView error = vernonCompileResultGetDiagnostics(linked);
            diagnostic.assign(error.data ? error.data : "", error.size);
            vernonCompileResultDestroy(linked);
        } else {
            diagnostic = "embedded LLD returned no result";
        }
        return false;
    }
    if (vernonCompileResultGetArtifactCount(linked) != 1) {
        diagnostic = "embedded LLD returned an invalid artifact set";
        vernonCompileResultDestroy(linked);
        return false;
    }
    VernonStringView name = vernonCompileResultGetArtifactName(linked, 0);
    VernonStringView data = vernonCompileResultGetArtifactData(linked, 0);
    filename.assign(name.data, name.size);
    library.assign(data.data, data.size);
    vernonCompileResultDestroy(linked);
    return true;
}

} // namespace

namespace vernon::tools {

VernonStatus packageCompileResult(VernonCompilerContext *context, const VernonCompileResult *result,
                                  VernonTarget target, const PackagingOptions &options, std::ostream &standardOutput,
                                  std::ostream &standardError) {
    VernonStatus status = vernonCompileResultGetStatus(result);
    if (status != VERNON_STATUS_OK)
        return status;

    const size_t artifactCount = vernonCompileResultGetArtifactCount(result);
    for (size_t index = 0; index < artifactCount; ++index) {
        if (options.outputDirectory) {
            std::filesystem::create_directories(*options.outputDirectory);
            VernonStringView name = vernonCompileResultGetArtifactName(result, index);
            std::filesystem::path path = *options.outputDirectory / std::string(name.data, name.size);
            std::ofstream output(path, std::ios::binary);
            writeView(output, vernonCompileResultGetArtifactData(result, index));
        } else if (!options.computeBundlePath) {
            if (artifactCount > 1) {
                standardOutput << "// artifact: ";
                writeView(standardOutput, vernonCompileResultGetArtifactName(result, index));
                standardOutput << '\n';
            }
            writeView(standardOutput, vernonCompileResultGetArtifactData(result, index));
            if (index + 1 != artifactCount)
                standardOutput << '\n';
        }
    }

    VernonStringView reflection = vernonCompileResultGetReflection(result);
    if (options.computeBundlePath) {
        llvm::Expected<llvm::json::Value> parsedReflection =
            llvm::json::parse(llvm::StringRef(reflection.data, reflection.size));
        llvm::json::Object *reflectionObject = parsedReflection ? parsedReflection->getAsObject() : nullptr;
        llvm::json::Array *entries = reflectionObject ? reflectionObject->getArray("entries") : nullptr;
        llvm::json::Object *computeEntry = nullptr;
        if (entries) {
            for (llvm::json::Value &entryValue : *entries) {
                llvm::json::Object *entry = entryValue.getAsObject();
                if (entry && entry->getString("stage").value_or("") == "compute") {
                    if (computeEntry) {
                        standardError << "compute bundles require exactly one compute entry\n";
                        return VERNON_STATUS_INVALID_ARGUMENT;
                    }
                    computeEntry = entry;
                }
            }
        }
        if (!computeEntry || artifactCount != 1) {
            standardError << "compute bundle requires one compute entry and one "
                             "artifact\n";
            return VERNON_STATUS_INVALID_ARGUMENT;
        }
        std::error_code error;
        std::filesystem::create_directories(*options.computeBundlePath, error);
        if (error || !std::filesystem::is_directory(*options.computeBundlePath)) {
            standardError << "cannot create compute bundle directory " << options.computeBundlePath->string() << '\n';
            return VERNON_STATUS_INTERNAL_ERROR;
        }

        VernonStringView artifactName = vernonCompileResultGetArtifactName(result, 0);
        VernonStringView artifactData = vernonCompileResultGetArtifactData(result, 0);
        std::string artifactFilename(artifactName.data, artifactName.size);
        std::string packagedArtifact(artifactData.data, artifactData.size);
        if (target == VERNON_TARGET_CPU) {
            if (artifactFilename != "module.obj" && artifactFilename != "module.o") {
                standardError << "CPU compilation did not produce a relocatable object\n";
                return VERNON_STATUS_INTERNAL_ERROR;
            }
            if (options.hostRuntimeBundle) {
                std::string linkDiagnostic;
                if (!linkHostLibrary(context, artifactData, artifactFilename, packagedArtifact, linkDiagnostic)) {
                    standardError << linkDiagnostic << '\n';
                    return VERNON_STATUS_INTERNAL_ERROR;
                }
            }
        }
        std::ofstream artifactOutput(*options.computeBundlePath / artifactFilename, std::ios::binary);
        artifactOutput.write(packagedArtifact.data(), static_cast<std::streamsize>(packagedArtifact.size()));
        artifactOutput.close();
        if (!artifactOutput) {
            standardError << "cannot write compute artifact " << artifactFilename << '\n';
            return VERNON_STATUS_INTERNAL_ERROR;
        }

        llvm::ArrayRef<uint8_t> bytes(reinterpret_cast<const uint8_t *>(packagedArtifact.data()),
                                      packagedArtifact.size());
        std::string digest = llvm::toHex(llvm::SHA256::hash(bytes), true);
        llvm::json::Object manifest;
        manifest["pipeline_version"] = int64_t{VERNON_PIPELINE_VERSION};
        manifest["release_version"] = VERNON_RELEASE_VERSION;
        if (target == VERNON_TARGET_CPU) {
            if (options.hostRuntimeBundle) {
                manifest["operating_system"] = std::string(hostOperatingSystem());
                manifest["architecture"] = std::string(hostArchitecture());
            } else {
                const llvm::Triple triple(
                    llvm::Triple::normalize(options.targetTriple.value_or(llvm::sys::getDefaultTargetTriple())));
                manifest["target_triple"] = triple.str();
                manifest["object_format"] = objectFormat(triple);
            }
        }
        manifest["target"] = std::string(targetName(target));
        manifest["entry"] = computeEntry->getString("name").value_or("").str();
        manifest["symbol"] = computeEntry->getString("symbol").value_or("").str();
        manifest["artifact"] = artifactFilename;
        manifest["artifact_format"] = target == VERNON_TARGET_CPU
                                          ? (options.hostRuntimeBundle ? "native_library" : "relocatable_object")
                                      : target == VERNON_TARGET_CUDA   ? "ptx"
                                      : target == VERNON_TARGET_VULKAN ? "spirv"
                                      : target == VERNON_TARGET_METAL  ? "msl"
                                                                       : "llvm_ir";
        manifest["artifact_size"] = static_cast<int64_t>(packagedArtifact.size());
        manifest["artifact_sha256"] = digest;
        manifest["reflection"] = std::move(*parsedReflection);
        std::ofstream manifestOutput(*options.computeBundlePath / "compute.json", std::ios::binary);
        std::string encoded;
        llvm::raw_string_ostream stream(encoded);
        stream << llvm::formatv("{0:2}", llvm::json::Value(std::move(manifest)));
        stream.flush();
        manifestOutput << encoded << '\n';
        manifestOutput.close();
        if (!manifestOutput) {
            standardError << "cannot write compute bundle manifest\n";
            return VERNON_STATUS_INTERNAL_ERROR;
        }
    } else if (options.reflectionPath) {
        std::ofstream output(*options.reflectionPath, std::ios::binary);
        writeView(output, reflection);
    } else if (!options.outputDirectory) {
        standardOutput << "\n// reflection\n";
        writeView(standardOutput, reflection);
        standardOutput << '\n';
    } else {
        std::filesystem::path path = *options.outputDirectory / "reflection.json";
        std::ofstream output(path, std::ios::binary);
        writeView(output, reflection);
    }
    return status;
}

} // namespace vernon::tools
