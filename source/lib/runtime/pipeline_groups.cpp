#include "pipeline_groups.h"

#include <algorithm>

namespace vernon::runtime {

PipelineGroups::~PipelineGroups() {
    for (VernonRuntimeCoreGroupLayout *layout : layouts_)
        vernonRuntimeCoreGroupLayoutDestroy(layout);
}

bool PipelineGroups::prepare(const VernonRuntimeDeviceProvider &provider,
                             const std::vector<VernonRuntimeProviderBindingLayoutEntry> &layouts, std::string &error) {
    uint32_t groupCount = 0;
    for (const VernonRuntimeProviderBindingLayoutEntry &layout : layouts) {
        if (layout.set >= VERNON_RUNTIME_PROVIDER_MAX_GROUPS) {
            error = "binding set " + std::to_string(layout.set) + " exceeds the provider group limit";
            return false;
        }
        groupCount = std::max(groupCount, layout.set + 1);
    }
    std::vector<std::vector<VernonRuntimeProviderBindingLayoutEntry>> entries(groupCount);
    for (const VernonRuntimeProviderBindingLayoutEntry &layout : layouts) {
        std::vector<VernonRuntimeProviderBindingLayoutEntry> &groupEntries = entries[layout.set];
        groupEntries.push_back(layout);
        groupEntries.back().slot = static_cast<uint32_t>(groupEntries.size() - 1);
        groupEntries.back().set = 0;
    }
    for (uint32_t index = 0; index < groupCount; ++index) {
        VernonRuntimeCoreGroupLayout *layout = nullptr;
        if (vernonRuntimeCorePrepareGroupLayout(&provider, entries[index].data(), entries[index].size(), &layout) !=
            VERNON_STATUS_OK) {
            error = "failed to prepare the provider group layout of set " + std::to_string(index);
            return false;
        }
        layouts_.push_back(layout);
    }
    return true;
}

} // namespace vernon::runtime
