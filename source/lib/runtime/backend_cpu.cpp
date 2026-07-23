#include "backend_cpu.h"

#include "runtime_state.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vernon::runtime {
namespace {

std::mutex &staticEntriesMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<std::string, VernonCpuEntryPoint> &staticEntries() {
    static std::unordered_map<std::string, VernonCpuEntryPoint> entries;
    return entries;
}

VernonStatus fail(std::string &error, std::string message, VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    error = std::move(message);
    return status;
}

} // namespace

VernonStatus registerStaticCpuEntry(VernonStringView symbol, VernonCpuEntryPoint entry) {
    if (!symbol.data || !symbol.size || !entry)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const std::string name(symbol.data, symbol.size);
    std::lock_guard<std::mutex> lock(staticEntriesMutex());
    auto [found, inserted] = staticEntries().emplace(name, entry);
    return inserted || found->second == entry ? VERNON_STATUS_OK : VERNON_STATUS_INVALID_ARGUMENT;
}

bool loadCpuEntry(VernonCpuEntryPoint entry, const char *reflection, size_t reflectionSize, const char *entryName,
                  size_t entryNameSize, CpuKernelState &state, ReflectedEntry &metadata, std::string &error) {
    try {
        const nlohmann::json parsed = nlohmann::json::parse(reflection, reflection + reflectionSize, nullptr, false);
        if (parsed.is_discarded() || !parseReflection(parsed, std::string(entryName, entryNameSize), metadata, error))
            return false;
        state.entry = entry;
        return true;
    } catch (const std::exception &exception) {
        error = std::string("failed to load CPU entry: ") + exception.what();
        return false;
    }
}

bool loadCpuNativeArtifact(const CpuNativeArtifact &artifact, CpuKernelState &state, ReflectedEntry &metadata,
                           std::string &error) {
    std::filesystem::path libraryPath;
    if (!resolveCpuNativeArtifact(artifact, libraryPath, &metadata, error))
        return false;
    if (artifact.format == "relocatable_object") {
        std::lock_guard<std::mutex> lock(staticEntriesMutex());
        const auto found = staticEntries().find(artifact.symbol);
        if (found != staticEntries().end())
            state.entry = found->second;
    } else {
        const std::string nativePath = libraryPath.u8string();
        if (!state.nativeLibrary.open(nativePath.c_str(), error))
            return false;
        state.entry = reinterpret_cast<VernonCpuEntryPoint>(state.nativeLibrary.symbol(artifact.symbol.c_str()));
    }
    if (state.entry)
        return true;
    error = artifact.format == "relocatable_object"
                ? "CPU AOT object symbol '" + artifact.symbol + "' was not statically registered"
                : "CPU AOT library does not export symbol '" + artifact.symbol + "'";
    return false;
}

bool createCpuBuffer(VernonDeviceBuffer &buffer) {
    auto *state = new CpuBufferState();
    state->storage.resize(buffer.size);
    installRuntimeBackendState(buffer, state);
    return true;
}

VernonStatus copyToCpuBuffer(VernonDeviceBuffer &buffer, size_t offset, const void *source, size_t size) {
    if (!source || offset > buffer.size || size > buffer.size - offset)
        return fail(buffer.context->error, "invalid compute buffer upload");
    std::memcpy(cpuBufferState(buffer).storage.data() + offset, source, size);
    return VERNON_STATUS_OK;
}

VernonStatus copyFromCpuBuffer(const VernonDeviceBuffer &buffer, size_t offset, void *destination, size_t size) {
    if (!destination || offset > buffer.size || size > buffer.size - offset)
        return fail(buffer.context->error, "invalid compute buffer readback");
    std::memcpy(destination, cpuBufferState(buffer).storage.data() + offset, size);
    return VERNON_STATUS_OK;
}

VernonStatus launchCpuKernel(VernonRuntimeContext &context, const CpuKernelState &state, const ReflectedEntry &metadata,
                             VernonLaunchSize globalSize, const VernonLaunchArgument *arguments, size_t argumentCount,
                             std::string &error) {
    (void)argumentCount;
    std::vector<unsigned char> packed(metadata.cpuArgumentsSize);
    size_t supplied = 0;
    std::vector<const ReflectedArgument *> globalInvocationIds;
    for (const ReflectedArgument &reflected : metadata.arguments) {
        if (reflected.kind == "builtin") {
            if (reflected.builtin == "global_invocation_id")
                globalInvocationIds.push_back(&reflected);
            continue;
        }
        const VernonLaunchArgument &argument = arguments[supplied++];
        if (reflected.kind == "tensor") {
            if (argument.kind != VERNON_LAUNCH_TENSOR || !argument.buffer || argument.buffer->context != &context ||
                reflected.cpuSize != sizeof(uintptr_t) ||
                (reflected.tensorBytes && argument.buffer->size < reflected.tensorBytes) ||
                argument.buffer->alignment < reflected.alignment)
                return fail(error, "invalid CPU Tensor launch argument");
            const uintptr_t pointer = reinterpret_cast<uintptr_t>(cpuBufferState(*argument.buffer).storage.data());
            std::memcpy(packed.data() + reflected.cpuOffset, &pointer, sizeof(pointer));
        } else {
            if (argument.kind != VERNON_LAUNCH_SCALAR || !argument.scalar_data ||
                argument.scalar_size != reflected.cpuSize)
                return fail(error, "invalid CPU scalar launch argument");
            std::memcpy(packed.data() + reflected.cpuOffset, argument.scalar_data, argument.scalar_size);
        }
    }

    VernonCpuInvocation invocation{packed.data(), packed.size(), nullptr, 0, nullptr};
    for (uint32_t z = 0; z < globalSize.z; ++z)
        for (uint32_t y = 0; y < globalSize.y; ++y)
            for (uint32_t x = 0; x < globalSize.x; ++x) {
                const uint32_t id[3]{x, y, z};
                for (const ReflectedArgument *builtin : globalInvocationIds)
                    std::memcpy(packed.data() + builtin->cpuOffset, id, std::min(builtin->cpuSize, sizeof(id)));
                const VernonStatus status = state.entry(&invocation);
                if (status != VERNON_STATUS_OK)
                    return fail(error, "CPU AOT entry invocation failed", status);
            }
    return VERNON_STATUS_OK;
}

} // namespace vernon::runtime
