#include "VernonCompiler.h"
#include "vernon_compile_packaging.h"

#include <charconv>
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
                     "[--output-dir <directory>] [--reflection <file>] "
                     "[--opengl-version <version>] "
                     "[--directx-shader-model <model>] [--metal-platform <macos|ios>] "
                     "[--cpu-triple <triple>] [--cpu-name <name>] "
                     "[--cpu-features <features>]\n";
        return 2;
    }

    const bool validateOnly = std::string_view(argv[1]) != "--target";
    std::optional<VernonTarget> target;
    const char *inputPath = argv[1];
    vernon::tools::PackagingOptions packaging;
    std::optional<uint32_t> glslVersion;
    std::optional<uint32_t> hlslShaderModel;
    std::optional<VernonMetalPlatform> metalPlatform;
    std::optional<std::string> cpuName;
    std::optional<std::string> cpuFeatures;
    if (!validateOnly) {
        if (argc < 4 || !(target = parseTarget(argv[2]))) {
            std::cerr << "unknown target\n";
            return 2;
        }
        inputPath = argv[3];
        for (int index = 4; index < argc;) {
            std::string_view option = argv[index];
            if (index + 1 >= argc) {
                std::cerr << "missing value for " << argv[index] << '\n';
                return 2;
            }
            if (option == "--output-dir")
                packaging.outputDirectory = argv[index + 1];
            else if (option == "--reflection")
                packaging.reflectionPath = argv[index + 1];
            else if (option == "--cpu-triple")
                packaging.targetTriple = argv[index + 1];
            else if (option == "--cpu-name")
                cpuName = argv[index + 1];
            else if (option == "--cpu-features")
                cpuFeatures = argv[index + 1];
            else if (option == "--opengl-version") {
                std::string_view value = argv[index + 1];
                uint32_t parsed = 0;
                auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
                if (error != std::errc() || end != value.data() + value.size()) {
                    std::cerr << "invalid GLSL version " << value << '\n';
                    return 2;
                }
                glslVersion = parsed;
            } else if (option == "--directx-shader-model") {
                std::string_view value = argv[index + 1];
                uint32_t parsed = 0;
                auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
                if (error != std::errc() || end != value.data() + value.size()) {
                    std::cerr << "invalid HLSL Shader Model " << value << '\n';
                    return 2;
                }
                hlslShaderModel = parsed;
            } else if (option == "--metal-platform") {
                const std::string_view value = argv[index + 1];
                if (value == "macos")
                    metalPlatform = VERNON_METAL_PLATFORM_MACOS;
                else if (value == "ios")
                    metalPlatform = VERNON_METAL_PLATFORM_IOS;
                else {
                    std::cerr << "invalid Metal platform " << value << '\n';
                    return 2;
                }
            } else {
                std::cerr << "unknown option " << option << '\n';
                return 2;
            }
            index += 2;
        }
        if ((packaging.targetTriple || cpuName || cpuFeatures) && *target != VERNON_TARGET_CPU) {
            std::cerr << "CPU target options require --target cpu\n";
            return 2;
        }
        if (hlslShaderModel && *target != VERNON_TARGET_DIRECTX) {
            std::cerr << "--directx-shader-model requires --target directx\n";
            return 2;
        }
        if (glslVersion && *target != VERNON_TARGET_OPENGL && *target != VERNON_TARGET_OPENGL_ES) {
            std::cerr << "--opengl-version requires --target opengl or opengles\n";
            return 2;
        }
        if (metalPlatform && *target != VERNON_TARGET_METAL) {
            std::cerr << "--metal-platform requires --target metal\n";
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
    std::string source((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());

    VernonCompilerContext *context = vernonCompilerCreate();
    if (!context) {
        std::cerr << "cannot create compiler context\n";
        return 1;
    }

    VernonCompileResult *result = nullptr;
    if (validateOnly) {
        result = vernonCompilerValidateMlir(context, source.data(), source.size());
    } else if (glslVersion || hlslShaderModel || metalPlatform || packaging.targetTriple || cpuName || cpuFeatures) {
        VernonCompileOptions options = {};
        options.struct_size = sizeof(options);
        options.target = *target;
        auto view = [](const std::optional<std::string> &value) {
            return value ? VernonStringView{value->data(), value->size()} : VernonStringView{};
        };
        if (*target == VERNON_TARGET_CPU) {
            options.as.cpu.triple = view(packaging.targetTriple);
            options.as.cpu.processor = view(cpuName);
            options.as.cpu.features = view(cpuFeatures);
        } else if (*target == VERNON_TARGET_OPENGL || *target == VERNON_TARGET_OPENGL_ES) {
            options.as.opengl.version = glslVersion.value_or(0);
        } else if (*target == VERNON_TARGET_METAL) {
            options.as.metal.platform = metalPlatform.value_or(VERNON_METAL_PLATFORM_MACOS);
        } else if (*target == VERNON_TARGET_DIRECTX) {
            options.as.directx.shader_model = hlslShaderModel.value_or(0);
        }
        result = vernonCompilerCompileMlirWithOptions(context, source.data(), source.size(), &options);
    } else {
        result = vernonCompilerCompileMlir(context, source.data(), source.size(), *target);
    }

    VernonStatus status = vernonCompileResultGetStatus(result);
    if (status != VERNON_STATUS_OK) {
        writeView(std::cerr, vernonCompileResultGetDiagnostics(result));
        std::cerr << '\n';
    } else {
        status = vernon::tools::packageCompileResult(result, target.value_or(VERNON_TARGET_CPU), packaging, std::cout,
                                                     std::cerr);
    }

    vernonCompileResultDestroy(result);
    vernonCompilerDestroy(context);
    return status == VERNON_STATUS_OK ? 0 : 1;
}
