#include "compiler_dxc.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <filesystem>
#include <optional>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

namespace vernon::compiler {
namespace {

struct TemporaryFile {
    llvm::SmallString<256> path;
    ~TemporaryFile() {
        if (!path.empty())
            llvm::sys::fs::remove(path);
    }
};

bool writeTemporaryFile(llvm::StringRef suffix, llvm::StringRef data, TemporaryFile &file, std::string &error) {
    int descriptor = -1;
    if (std::error_code code = llvm::sys::fs::createTemporaryFile("vernon-dxc", suffix, descriptor, file.path)) {
        error = "failed to create DXC temporary file: " + code.message();
        return false;
    }
    llvm::raw_fd_ostream output(descriptor, true);
    output.write(data.data(), data.size());
    output.close();
    if (output.has_error()) {
        error = "failed to write DXC temporary HLSL";
        return false;
    }
    return true;
}

std::string readFile(llvm::StringRef path) {
    auto buffer = llvm::MemoryBuffer::getFile(path, false);
    return buffer ? buffer.get()->getBuffer().str() : std::string();
}

std::optional<std::pair<std::string, std::string>> profileAndEntry(const std::string &name, uint32_t shaderModel) {
    const size_t stageSeparator = name.rfind(".vert.hlsl");
    const size_t fragmentSeparator = name.rfind(".frag.hlsl");
    const size_t computeSeparator = name.rfind(".comp.hlsl");
    const size_t separator = stageSeparator != std::string::npos      ? stageSeparator
                             : fragmentSeparator != std::string::npos ? fragmentSeparator
                                                                      : computeSeparator;
    if (separator == std::string::npos)
        return std::nullopt;
    const char *prefix = stageSeparator != std::string::npos      ? "vs"
                         : fragmentSeparator != std::string::npos ? "ps"
                                                                  : "cs";
    return std::pair<std::string, std::string>{
        std::string(prefix) + "_" + std::to_string(shaderModel / 10) + "_" + std::to_string(shaderModel % 10), "main"};
}

std::string dxcExecutable() {
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&dxcExecutable), &module)) {
        std::array<wchar_t, 32768> path{};
        const DWORD size = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
        if (size && size < path.size()) {
            std::filesystem::path adjacent(path.data());
            adjacent.replace_filename(L"dxc.exe");
            if (std::filesystem::is_regular_file(adjacent))
                return adjacent.u8string();
        }
    }
#endif
#if defined(VERNON_DXC_EXECUTABLE)
    return VERNON_DXC_EXECUTABLE;
#else
    return {};
#endif
}

} // namespace

bool compileHlslToDxil(const std::vector<Artifact> &hlslArtifacts, uint32_t shaderModel,
                       std::vector<Artifact> &dxilArtifacts, std::string &diagnostics) {
    const std::string executable = dxcExecutable();
    if (executable.empty()) {
        diagnostics = "DirectX DXIL compilation is unavailable because DXC was not configured";
        return false;
    }
    dxilArtifacts.clear();
    for (const Artifact &artifact : hlslArtifacts) {
        const auto profile = profileAndEntry(artifact.name, shaderModel);
        if (!profile) {
            diagnostics = "cannot determine DXC stage profile from artifact '" + artifact.name + "'";
            return false;
        }
        TemporaryFile source;
        TemporaryFile output;
        TemporaryFile errors;
        if (!writeTemporaryFile("hlsl", artifact.data, source, diagnostics) ||
            !writeTemporaryFile("dxil", {}, output, diagnostics) || !writeTemporaryFile("log", {}, errors, diagnostics))
            return false;
        const std::array<llvm::StringRef, 14> arguments{executable,
                                                        source.path,
                                                        "-E",
                                                        profile->second,
                                                        "-T",
                                                        profile->first,
                                                        "-Fo",
                                                        output.path,
                                                        "-O3",
                                                        "-Ges",
                                                        "-Qstrip_debug",
                                                        "-Qstrip_reflect",
                                                        "-HV",
                                                        "2021"};
        const std::array<std::optional<llvm::StringRef>, 3> redirects{std::nullopt, std::nullopt,
                                                                      llvm::StringRef(errors.path)};
        std::string executionError;
        bool executionFailed = false;
        const int status = llvm::sys::ExecuteAndWait(executable, arguments, std::nullopt, redirects, 0, 0,
                                                     &executionError, &executionFailed);
        const std::string compilerOutput = readFile(errors.path);
        if (executionFailed || status != 0) {
            diagnostics = compilerOutput.empty() ? "DXC failed: " + executionError : compilerOutput;
            dxilArtifacts.clear();
            return false;
        }
        std::string binary = readFile(output.path);
        if (binary.size() < 4 || binary.compare(0, 4, "DXBC") != 0) {
            diagnostics = "DXC did not produce a valid DXIL container";
            dxilArtifacts.clear();
            return false;
        }
        std::string name = artifact.name.substr(0, artifact.name.size() - 4) + "dxil";
        dxilArtifacts.push_back({std::move(name), std::move(binary)});
    }
    return !dxilArtifacts.empty();
}

} // namespace vernon::compiler
