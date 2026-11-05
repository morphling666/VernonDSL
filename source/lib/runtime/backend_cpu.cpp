#include "backend_cpu.h"

#include "runtime_state.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vernon::runtime {
namespace {

struct StaticEntryRegistration {
    VernonCpuEntryPoint entry;
    size_t references;
};

struct StaticEntryRegistry {
    std::mutex mutex;
    // Several loaded images may each carry a copy of one object, so one symbol
    // can map to several live definitions. Lookups use the oldest; a definition
    // leaves when the image that registered it unregisters it on unload.
    std::unordered_map<std::string, std::vector<StaticEntryRegistration>> entries;
};

// Intentionally leaked: object destructors unregister during process exit or
// image unload, possibly after this library's own statics were destroyed.
StaticEntryRegistry &staticEntryRegistry() {
    static auto *registry = new StaticEntryRegistry();
    return *registry;
}

bool validStaticCpuEntries(const VernonStaticCpuEntry *entries, size_t count) {
    if (!entries && count)
        return false;
    for (size_t index = 0; index < count; ++index)
        if (!entries[index].symbol.data || !entries[index].symbol.size || !entries[index].entry_point)
            return false;
    return true;
}

} // namespace

bool initializeCpuContext(VernonRuntimeContext &context, uint32_t deviceIndex) {
    if (deviceIndex != 0)
        return false;
    auto state = std::unique_ptr<CpuContextState>(new (std::nothrow) CpuContextState());
    if (!state)
        return false;
    state->scheduler = CpuWorkgroupScheduler::create(CpuWorkgroupScheduler::defaultConfig(), state->error);
    if (!state->scheduler)
        return false;
    installRuntimeBackendState(context, state.release());
    return true;
}

CpuWorkgroupScheduler &cpuWorkgroupScheduler(VernonRuntimeContext &context) {
    return *runtimeBackendState<CpuContextState>(context).scheduler;
}

VernonStatus registerStaticCpuEntries(const VernonStaticCpuEntry *entries, size_t count) {
    if (!validStaticCpuEntries(entries, count))
        return VERNON_STATUS_INVALID_ARGUMENT;
    StaticEntryRegistry &registry = staticEntryRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    for (size_t index = 0; index < count; ++index) {
        auto &definitions = registry.entries[std::string(entries[index].symbol.data, entries[index].symbol.size)];
        const auto found = std::find_if(definitions.begin(), definitions.end(), [&](const auto &definition) {
            return definition.entry == entries[index].entry_point;
        });
        if (found != definitions.end())
            ++found->references;
        else
            definitions.push_back({entries[index].entry_point, 1});
    }
    return VERNON_STATUS_OK;
}

VernonStatus unregisterStaticCpuEntries(const VernonStaticCpuEntry *entries, size_t count) {
    if (!validStaticCpuEntries(entries, count))
        return VERNON_STATUS_INVALID_ARGUMENT;
    StaticEntryRegistry &registry = staticEntryRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    for (size_t index = 0; index < count; ++index) {
        const auto symbol = registry.entries.find(std::string(entries[index].symbol.data, entries[index].symbol.size));
        if (symbol == registry.entries.end())
            continue;
        auto &definitions = symbol->second;
        const auto found = std::find_if(definitions.begin(), definitions.end(), [&](const auto &definition) {
            return definition.entry == entries[index].entry_point;
        });
        if (found == definitions.end() || --found->references != 0)
            continue;
        definitions.erase(found);
        if (definitions.empty())
            registry.entries.erase(symbol);
    }
    return VERNON_STATUS_OK;
}

VernonCpuEntryPoint findStaticCpuEntry(const std::string &symbol) {
    StaticEntryRegistry &registry = staticEntryRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.entries.find(symbol);
    return found != registry.entries.end() ? found->second.front().entry : nullptr;
}

bool findRegisteredCpuEntry(VernonRuntimeContext &context, const std::string &symbol, VernonCpuEntryPoint &entry,
                            std::string &error) {
    {
        std::lock_guard<std::mutex> lock(context.cpuEntriesMutex);
        const auto found = context.cpuEntries.find(symbol);
        if (found != context.cpuEntries.end())
            entry = found->second.first;
    }
    if (!entry)
        entry = findStaticCpuEntry(symbol);
    if (entry)
        return true;
    error = "CPU AOT object symbol '" + symbol +
            "' was not statically registered; link the Program's CPU objects into the application (whole-archive "
            "when they are in a static library)";
    return false;
}

} // namespace vernon::runtime
