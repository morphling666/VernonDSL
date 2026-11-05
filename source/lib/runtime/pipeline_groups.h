#ifndef VERNON_RUNTIME_PIPELINE_GROUPS_H
#define VERNON_RUNTIME_PIPELINE_GROUPS_H

#include "VernonRuntimeCore.h"

#include <cstddef>
#include <string>
#include <vector>

namespace vernon::runtime {

// The persistent groups of one owner. A device keeps a group that only the arguments of one store's owner decide while
// that owner lives, and finds it by the owner's identity.
class GroupStore {};

// Who supplies the argument of one Stage parameter to a command. The same identity always supplies the same
// argument, so the groups an owner's arguments alone decide are found by identity and never by value.
struct ArgumentOwner {
    const void *identity{};
    // The persistent groups of the owner, or null for an owner whose groups live in the command stream.
    GroupStore *groups{};
};

// The group layouts of one prepared pipeline: layout g holds the prepared layouts in descriptor set g.
class PipelineGroups {
public:
    PipelineGroups() = default;
    PipelineGroups(const PipelineGroups &) = delete;
    PipelineGroups &operator=(const PipelineGroups &) = delete;
    ~PipelineGroups();

    bool prepare(const VernonRuntimeDeviceProvider &provider,
                 const std::vector<VernonRuntimeProviderBindingLayoutEntry> &layouts, std::string &error);
    VernonRuntimeCoreGroupLayout *const *layouts() const { return layouts_.data(); }
    size_t size() const { return layouts_.size(); }

private:
    std::vector<VernonRuntimeCoreGroupLayout *> layouts_;
};

} // namespace vernon::runtime

#endif
