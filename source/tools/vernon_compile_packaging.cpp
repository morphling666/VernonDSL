#include "vernon_compile_packaging.h"

#include <filesystem>
#include <fstream>
#include <ostream>

namespace {

void writeView(std::ostream &stream, VernonStringView value) {
    if (value.data && value.size)
        stream.write(value.data, static_cast<std::streamsize>(value.size));
}

} // namespace

namespace vernon::tools {

VernonStatus packageCompileResult(const VernonCompileResult *result, VernonTarget, const PackagingOptions &options,
                                  std::ostream &standardOutput, std::ostream &) {
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
        } else {
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
    if (options.reflectionPath) {
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
