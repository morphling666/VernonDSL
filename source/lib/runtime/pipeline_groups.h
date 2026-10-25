#ifndef VERNON_RUNTIME_PIPELINE_GROUPS_H
#define VERNON_RUNTIME_PIPELINE_GROUPS_H

#include "VernonRuntimeCore.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vernon::runtime {

class GroupStore;

// Who supplies the argument of one Stage parameter to a command. The same identity always supplies the same
// argument, so the groups an owner's arguments alone decide are found by identity and never by value.
struct ArgumentOwner {
    const void *identity{};
    // The persistent groups of the owner, or null for an owner whose groups live in the command stream.
    GroupStore *groups{};
    // Keeps `identity` alive while a persistent group made from it lives; null when `groups` dies with the identity.
    const std::shared_ptr<void> *lifetime{};
};

// Fills the provider value of prepared layout `slot` when a group holding it is made. An inline payload may point into
// `storage`, which outlives the making of the group.
struct GroupValues {
    bool (*fill)(const void *userData, size_t slot, VernonRuntimeProviderBindingValue &value,
                 std::vector<uint8_t> &storage, std::string &error){};
    const void *userData{};
};

// Told of each group an updatable store makes, before a command binds it; its inline values are undefined until
// written.
struct GroupSink {
    VernonStatus (*made)(void *userData, VernonRuntimeCoreGroup *group, const std::vector<uint32_t> &slots,
                         const VernonRuntimeProviderBindingValue *values, size_t valueCount){};
    void *userData{};
};

// Whether a group value's bytes are inline in it rather than in a resource.
bool inlineGroupValue(const VernonRuntimeProviderBindingValue &value);

// What decides the groups of one command.
struct GroupSources {
    // Per Stage parameter, the owner of its argument; null when the command alone owns every argument.
    const ArgumentOwner *parameters{};
    // Identity of the inputs that no parameter decides, such as the viewport resolution; with null `parameters` it
    // identifies every argument as well.
    const void *command{};
    // Required: the groups of the command alone. An updatable store requires `made`.
    GroupStore *transient{};
    GroupValues values;
    GroupSink made;
};

// The provider groups one owner made. A persistent or updatable store outlives command streams and keeps its groups
// until it dies, a persistent one only while their owners live; a transient store belongs to one command stream.
class GroupStore {
public:
    explicit GroupStore(VernonRuntimeProviderGroupLifetime lifetime) : lifetime_(lifetime) {}
    GroupStore(const GroupStore &) = delete;
    GroupStore &operator=(const GroupStore &) = delete;
    ~GroupStore();

    // The identities a group is found by, in the order of the group's dependencies.
    struct Key {
        const uint32_t *parameters{};
        size_t parameterCount{};
        // Appended after the parameters' owners when the group also depends on the command.
        const void *command{};
        const ArgumentOwner *owners{};

        size_t size() const { return parameterCount + (command ? 1 : 0); }
        const void *operator[](size_t index) const {
            return index < parameterCount ? owners[parameters[index]].identity : command;
        }
    };

    // Binds at `index` of `packets` the group of `layout` that `key` finds, made from the values of `slots` (one per
    // group entry, in entry order) on first use. `error` is set only when filling a value fails.
    VernonStatus bind(VernonRuntimeCorePackets *packets, uint32_t index, VernonRuntimeCoreGroupLayout *layout,
                      const Key &key, const std::vector<uint32_t> &slots, const GroupValues &values,
                      const GroupSink &made, VernonRuntimeProviderObject encoder, std::string &error);

private:
    struct Entry {
        VernonRuntimeCoreGroupLayout *layout{};
        std::vector<const void *> identities;
        std::vector<std::shared_ptr<const void>> lifetimes;
        std::vector<VernonRuntimeProviderBindingValue> values;
        // Inline payloads of `values` point into it.
        std::vector<uint8_t> inlineBytes;
        VernonRuntimeCoreGroup *group{};
    };

    // Drops the groups of owners that only this store still keeps alive.
    void collect();

    VernonRuntimeProviderGroupLifetime lifetime_;
    std::mutex mutex_;
    std::unordered_multimap<uint64_t, std::unique_ptr<Entry>> entries_;
};

// The groups of one prepared pipeline: group g holds the prepared layouts in descriptor set g.
class PipelineGroups {
public:
    PipelineGroups() = default;
    PipelineGroups(const PipelineGroups &) = delete;
    PipelineGroups &operator=(const PipelineGroups &) = delete;
    ~PipelineGroups();

    // What one prepared layout's value is a function of.
    struct SlotDependencies {
        // Stage parameters, by index.
        std::vector<uint32_t> parameters;
        // Whether the value also depends on the command, beyond its parameters.
        bool command{};
    };

    bool prepare(const VernonRuntimeDeviceProvider &provider,
                 const std::vector<VernonRuntimeProviderBindingLayoutEntry> &layouts,
                 const std::vector<SlotDependencies> &dependencies, std::string &error);
    VernonRuntimeCoreGroupLayout *const *layouts() const { return layouts_.data(); }
    size_t size() const { return layouts_.size(); }
    // Binds the groups of one command of `encoder`, each kept by the owner of what decides it.
    VernonStatus bind(VernonRuntimeCorePackets *packets, VernonRuntimeProviderObject encoder,
                      const GroupSources &sources, std::string &error) const;

private:
    struct Group {
        // Prepared layouts, in group entry order.
        std::vector<uint32_t> slots;
        std::vector<uint32_t> parameters;
        bool command{};
    };

    std::vector<VernonRuntimeCoreGroupLayout *> layouts_;
    std::vector<Group> groups_;
};

// The persistent groups that the bindings of one Program instance make, one store per Node: a Node's projection
// decides how a binding lowers, so one binding makes different groups in different Nodes.
class InstanceGroups {
public:
    GroupStore &node(uint64_t key);

private:
    std::mutex mutex_;
    std::unordered_map<uint64_t, std::unique_ptr<GroupStore>> nodes_;
};

} // namespace vernon::runtime

#endif
