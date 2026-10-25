#include "pipeline_groups.h"

#include <algorithm>
#include <cstring>

namespace vernon::runtime {

bool inlineGroupValue(const VernonRuntimeProviderBindingValue &value) {
    return value.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE || value.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
           (value.flags & VERNON_RUNTIME_PROVIDER_BINDING_HOST_STORAGE) != 0;
}

namespace {

void mix(uint64_t &hash, uint64_t value) { hash = (hash ^ value) * 0x100000001b3ull; }

uint64_t keyHash(const VernonRuntimeCoreGroupLayout *layout, const GroupStore::Key &key) {
    uint64_t hash = 0xcbf29ce484222325ull;
    mix(hash, reinterpret_cast<uintptr_t>(layout));
    for (size_t index = 0; index < key.size(); ++index)
        mix(hash, reinterpret_cast<uintptr_t>(key[index]));
    return hash;
}

} // namespace

GroupStore::~GroupStore() {
    for (const auto &[hash, entry] : entries_)
        vernonRuntimeCoreGroupDestroy(entry->group);
}

void GroupStore::collect() {
    for (auto entry = entries_.begin(); entry != entries_.end();) {
        const auto &lifetimes = entry->second->lifetimes;
        if (std::none_of(lifetimes.begin(), lifetimes.end(),
                         [](const std::shared_ptr<const void> &lifetime) { return lifetime.use_count() == 1; })) {
            ++entry;
            continue;
        }
        vernonRuntimeCoreGroupDestroy(entry->second->group);
        entry = entries_.erase(entry);
    }
}

VernonStatus GroupStore::bind(VernonRuntimeCorePackets *packets, uint32_t index, VernonRuntimeCoreGroupLayout *layout,
                              const Key &key, const std::vector<uint32_t> &slots, const GroupValues &values,
                              const GroupSink &made, VernonRuntimeProviderObject encoder, std::string &error) {
    const uint64_t hash = keyHash(layout, key);
    // Binding under the lock keeps a concurrent collection from destroying the group before the recording retains it.
    const std::lock_guard lock(mutex_);
    const auto [first, last] = entries_.equal_range(hash);
    for (auto candidate = first; candidate != last; ++candidate) {
        const Entry &entry = *candidate->second;
        if (entry.layout != layout || entry.identities.size() != key.size())
            continue;
        size_t matched = 0;
        while (matched < key.size() && entry.identities[matched] == key[matched])
            ++matched;
        if (matched == key.size())
            return vernonRuntimeCorePacketsBindGroup(packets, index, entry.group);
    }

    const bool persistent = lifetime_ == VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT;
    const bool transient = lifetime_ == VERNON_RUNTIME_PROVIDER_GROUP_TRANSIENT;
    if (!transient && !persistent && !made.made)
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (persistent)
        collect();
    auto entry = std::make_unique<Entry>();
    entry->layout = layout;
    entry->identities.reserve(key.size());
    for (size_t identity = 0; identity < key.size(); ++identity)
        entry->identities.push_back(key[identity]);
    if (persistent)
        for (size_t parameter = 0; parameter < key.parameterCount; ++parameter)
            if (const std::shared_ptr<void> *lifetime = key.owners[key.parameters[parameter]].lifetime)
                entry->lifetimes.push_back(*lifetime);
    entry->values.resize(slots.size());
    std::vector<std::vector<uint8_t>> storage(slots.size());
    size_t inlineSize = 0;
    for (size_t slot = 0; slot < slots.size(); ++slot) {
        VernonRuntimeProviderBindingValue &value = entry->values[slot];
        if (!values.fill(values.userData, slots[slot], value, storage[slot], error))
            return VERNON_STATUS_INVALID_ARGUMENT;
        value.slot = static_cast<uint32_t>(slot);
        if (inlineGroupValue(value))
            inlineSize += value.payload.inline_value.size;
    }
    entry->inlineBytes.resize(inlineSize);
    size_t offset = 0;
    for (VernonRuntimeProviderBindingValue &value : entry->values) {
        if (!inlineGroupValue(value) || value.payload.inline_value.size == 0)
            continue;
        std::memcpy(entry->inlineBytes.data() + offset, value.payload.inline_value.data,
                    value.payload.inline_value.size);
        value.payload.inline_value.data = entry->inlineBytes.data() + offset;
        offset += value.payload.inline_value.size;
    }
    VernonStatus status =
        vernonRuntimeCoreCreateGroup(layout, entry->values.data(), entry->values.size(), lifetime_,
                                     transient ? encoder : VernonRuntimeProviderObject{}, &entry->group);
    if (status == VERNON_STATUS_OK && !transient && !persistent)
        status = made.made(made.userData, entry->group, slots, entry->values.data(), entry->values.size());
    if (status == VERNON_STATUS_OK)
        status = vernonRuntimeCorePacketsBindGroup(packets, index, entry->group);
    if (status != VERNON_STATUS_OK) {
        vernonRuntimeCoreGroupDestroy(entry->group);
        return status;
    }
    entries_.emplace(hash, std::move(entry));
    return VERNON_STATUS_OK;
}

PipelineGroups::~PipelineGroups() {
    for (VernonRuntimeCoreGroupLayout *layout : layouts_)
        vernonRuntimeCoreGroupLayoutDestroy(layout);
}

bool PipelineGroups::prepare(const VernonRuntimeDeviceProvider &provider,
                             const std::vector<VernonRuntimeProviderBindingLayoutEntry> &layouts,
                             const std::vector<SlotDependencies> &dependencies, std::string &error) {
    if (dependencies.size() != layouts.size()) {
        error = "prepared binding dependencies do not match the provider layouts";
        return false;
    }
    uint32_t groupCount = 0;
    for (const VernonRuntimeProviderBindingLayoutEntry &layout : layouts) {
        if (layout.set >= VERNON_RUNTIME_PROVIDER_MAX_GROUPS) {
            error = "binding set " + std::to_string(layout.set) + " exceeds the provider group limit";
            return false;
        }
        groupCount = std::max(groupCount, layout.set + 1);
    }
    std::vector<std::vector<VernonRuntimeProviderBindingLayoutEntry>> entries(groupCount);
    groups_.resize(groupCount);
    for (size_t index = 0; index < layouts.size(); ++index) {
        const VernonRuntimeProviderBindingLayoutEntry &layout = layouts[index];
        Group &group = groups_[layout.set];
        std::vector<VernonRuntimeProviderBindingLayoutEntry> &groupEntries = entries[layout.set];
        group.slots.push_back(static_cast<uint32_t>(index));
        groupEntries.push_back(layout);
        groupEntries.back().slot = static_cast<uint32_t>(groupEntries.size() - 1);
        groupEntries.back().set = 0;
        group.parameters.insert(group.parameters.end(), dependencies[index].parameters.begin(),
                                dependencies[index].parameters.end());
        group.command = group.command || dependencies[index].command;
    }
    for (uint32_t index = 0; index < groupCount; ++index) {
        Group &group = groups_[index];
        std::sort(group.parameters.begin(), group.parameters.end());
        group.parameters.erase(std::unique(group.parameters.begin(), group.parameters.end()), group.parameters.end());
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

VernonStatus PipelineGroups::bind(VernonRuntimeCorePackets *packets, VernonRuntimeProviderObject encoder,
                                  const GroupSources &sources, std::string &error) const {
    for (uint32_t index = 0; index < groups_.size(); ++index) {
        const Group &group = groups_[index];
        if (group.slots.empty())
            continue;
        GroupStore::Key key;
        GroupStore *store = sources.transient;
        if (!sources.parameters) {
            key.command = sources.command;
        } else {
            key = {group.parameters.data(), group.parameters.size(), group.command ? sources.command : nullptr,
                   sources.parameters};
            GroupStore *owner = group.parameters.empty() ? nullptr : sources.parameters[group.parameters[0]].groups;
            for (uint32_t parameter : group.parameters)
                if (sources.parameters[parameter].groups != owner)
                    owner = nullptr;
            if (owner && !group.command)
                store = owner;
        }
        if (const VernonStatus status = store->bind(packets, index, layouts_[index], key, group.slots, sources.values,
                                                    sources.made, encoder, error);
            status != VERNON_STATUS_OK)
            return status;
    }
    return VERNON_STATUS_OK;
}

GroupStore &InstanceGroups::node(uint64_t key) {
    const std::lock_guard lock(mutex_);
    std::unique_ptr<GroupStore> &store = nodes_[key];
    if (!store)
        store = std::make_unique<GroupStore>(VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT);
    return *store;
}

} // namespace vernon::runtime
