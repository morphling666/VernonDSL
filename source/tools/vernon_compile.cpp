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
                     "[--glsl-version <version>] "
                     "[--hlsl-shader-model <model>] "
                     "[--bundle <directory> --asset-id <id>] "
                     "[--compute-bundle <directory>] [--host-runtime-bundle] "
                     "[--target-triple <triple>] [--cpu <name>] "
                     "[--cpu-features <features>]\n"
                     "  --bundle writes a compiled OpenGL shader asset.\n"
                     "  --compute-bundle writes a relocatable CPU object bundle.\n"
                     "  --host-runtime-bundle finalizes that object with embedded "
                     "LLD for immediate host execution.\n";
        return 2;
    }

    const bool validateOnly = std::string_view(argv[1]) != "--target";
    std::optional<VernonTarget> target;
    const char *inputPath = argv[1];
    vernon::tools::LegacyPackagingOptions packaging;
    std::optional<uint32_t> glslVersion;
    std::optional<uint32_t> hlslShaderModel;
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
            if (option == "--host-runtime-bundle") {
                packaging.hostRuntimeBundle = true;
                ++index;
                continue;
            }
            if (index + 1 >= argc) {
                std::cerr << "missing value for " << argv[index] << '\n';
                return 2;
            }
            if (option == "--output-dir")
                packaging.outputDirectory = argv[index + 1];
            else if (option == "--reflection")
                packaging.reflectionPath = argv[index + 1];
            else if (option == "--bundle")
                packaging.shaderBundlePath = argv[index + 1];
            else if (option == "--compute-bundle")
                packaging.computeBundlePath = argv[index + 1];
            else if (option == "--asset-id")
                packaging.assetId = argv[index + 1];
            else if (option == "--target-triple")
                packaging.targetTriple = argv[index + 1];
            else if (option == "--cpu")
                cpuName = argv[index + 1];
            else if (option == "--cpu-features")
                cpuFeatures = argv[index + 1];
            else if (option == "--glsl-version") {
                std::string_view value = argv[index + 1];
                uint32_t parsed = 0;
                auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
                if (error != std::errc() || end != value.data() + value.size()) {
                    std::cerr << "invalid GLSL version " << value << '\n';
                    return 2;
                }
                glslVersion = parsed;
            } else if (option == "--hlsl-shader-model") {
                std::string_view value = argv[index + 1];
                uint32_t parsed = 0;
                auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
                if (error != std::errc() || end != value.data() + value.size()) {
                    std::cerr << "invalid HLSL Shader Model " << value << '\n';
                    return 2;
                }
                hlslShaderModel = parsed;
            } else {
                std::cerr << "unknown option " << option << '\n';
                return 2;
            }
            index += 2;
        }
        if (packaging.shaderBundlePath.has_value() != packaging.assetId.has_value()) {
            std::cerr << "--bundle and --asset-id must be specified together\n";
            return 2;
        }
        if (packaging.shaderBundlePath && *target != VERNON_TARGET_OPENGL) {
            std::cerr << "--bundle writes a compiled shader asset and requires "
                         "target opengl; use --compute-bundle for runtime compute "
                         "artifacts\n";
            return 2;
        }
        if (packaging.shaderBundlePath && packaging.outputDirectory) {
            std::cerr << "--bundle and --output-dir cannot be combined\n";
            return 2;
        }
        if (packaging.computeBundlePath && (packaging.shaderBundlePath || packaging.outputDirectory)) {
            std::cerr << "--compute-bundle cannot be combined with --bundle or "
                         "--output-dir\n";
            return 2;
        }
        if (packaging.hostRuntimeBundle && (!packaging.computeBundlePath || *target != VERNON_TARGET_CPU)) {
            std::cerr << "--host-runtime-bundle requires a CPU --compute-bundle\n";
            return 2;
        }
        if ((packaging.targetTriple || cpuName || cpuFeatures) && *target != VERNON_TARGET_CPU) {
            std::cerr << "CPU target options require --target cpu\n";
            return 2;
        }
        if (hlslShaderModel && *target != VERNON_TARGET_DIRECTX) {
            std::cerr << "--hlsl-shader-model requires --target directx\n";
            return 2;
        }
        if (packaging.hostRuntimeBundle && packaging.targetTriple) {
            std::cerr << "--host-runtime-bundle always uses the compiler host target\n";
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
    } else if (glslVersion || hlslShaderModel || packaging.targetTriple || cpuName || cpuFeatures) {
        VernonCompileOptions options = {};
        options.struct_size = sizeof(options);
        options.glsl_version = glslVersion.value_or(0);
        options.hlsl_shader_model = hlslShaderModel.value_or(0);
        auto view = [](const std::optional<std::string> &value) {
            return value ? VernonStringView{value->data(), value->size()} : VernonStringView{};
        };
        options.cpu_target_triple = view(packaging.targetTriple);
        options.cpu_name = view(cpuName);
        options.cpu_features = view(cpuFeatures);
        result = vernonCompilerCompileMlirWithOptions(context, source.data(), source.size(), *target, &options);
    } else {
        result = vernonCompilerCompileMlir(context, source.data(), source.size(), *target);
    }

    VernonStatus status = vernonCompileResultGetStatus(result);
    if (status != VERNON_STATUS_OK) {
        writeView(std::cerr, vernonCompileResultGetDiagnostics(result));
        std::cerr << '\n';
    } else {
        status = vernon::tools::packageCompileResult(context, result, target.value_or(VERNON_TARGET_CPU), packaging,
                                                     std::cout, std::cerr);
    }

    vernonCompileResultDestroy(result);
    vernonCompilerDestroy(context);
    return status == VERNON_STATUS_OK ? 0 : 1;
}
