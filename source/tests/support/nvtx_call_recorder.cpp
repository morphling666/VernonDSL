#include "nvtx_call_recorder.h"

#include <nvtx3/nvToolsExt.h>

#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace {

std::mutex callsMutex;
std::vector<std::string> calls;
std::unordered_map<nvtxRangeId_t, std::string> openRanges;
nvtxRangeId_t lastRange = 0;
// Its address is the handle of the "Vernon" domain; other domains get a null handle and are not recorded.
char vernonDomainTag;

nvtxDomainHandle_t vernonDomain() { return reinterpret_cast<nvtxDomainHandle_t>(&vernonDomainTag); }

std::string message(const nvtxEventAttributes_t *attributes) {
    return attributes && attributes->messageType == NVTX_MESSAGE_TYPE_ASCII ? attributes->message.ascii : "";
}

nvtxDomainHandle_t NVTX_API createDomain(const char *name) {
    return std::strcmp(name, "Vernon") == 0 ? vernonDomain() : nullptr;
}

nvtxRangeId_t NVTX_API startRange(nvtxDomainHandle_t domain, const nvtxEventAttributes_t *attributes) {
    std::lock_guard<std::mutex> guard(callsMutex);
    const nvtxRangeId_t range = ++lastRange;
    if (domain == vernonDomain()) {
        calls.push_back("start " + message(attributes));
        openRanges.emplace(range, message(attributes));
    }
    return range;
}

void NVTX_API endRange(nvtxDomainHandle_t domain, nvtxRangeId_t range) {
    std::lock_guard<std::mutex> guard(callsMutex);
    if (domain != vernonDomain())
        return;
    const auto open = openRanges.find(range);
    calls.push_back(open == openRanges.end() ? "end of an unknown range" : "end " + open->second);
    if (open != openRanges.end())
        openRanges.erase(open);
}

void NVTX_API mark(nvtxDomainHandle_t domain, const nvtxEventAttributes_t *attributes) {
    std::lock_guard<std::mutex> guard(callsMutex);
    if (domain == vernonDomain())
        calls.push_back("mark " + message(attributes));
}

template <typename Function> void install(NvtxFunctionTable table, NvtxCallbackIdCore2 id, Function function) {
    *table[id] = reinterpret_cast<NvtxFunctionPointer>(function);
}

} // namespace

std::vector<std::string> vernon::tests::takeNvtxCalls() {
    std::lock_guard<std::mutex> guard(callsMutex);
    return std::exchange(calls, {});
}

extern "C" VERNON_NVTX_CALL_RECORDER_API int NVTX_API InitializeInjectionNvtx2(NvtxGetExportTableFunc_t exportTable) {
    const auto *callbacks = static_cast<const NvtxExportTableCallbacks *>(exportTable(NVTX_ETID_CALLBACKS));
    NvtxFunctionTable table = nullptr;
    unsigned int size = 0;
    if (!callbacks || !callbacks->GetModuleFunctionTable(NVTX_CB_MODULE_CORE2, &table, &size) ||
        size <= NVTX_CBID_CORE2_DomainCreateA)
        return 0;
    install(table, NVTX_CBID_CORE2_DomainCreateA, &createDomain);
    install(table, NVTX_CBID_CORE2_DomainRangeStartEx, &startRange);
    install(table, NVTX_CBID_CORE2_DomainRangeEnd, &endRange);
    install(table, NVTX_CBID_CORE2_DomainMarkEx, &mark);
    return 1;
}
