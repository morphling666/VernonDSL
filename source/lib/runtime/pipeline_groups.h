#ifndef VERNON_RUNTIME_PIPELINE_GROUPS_H
#define VERNON_RUNTIME_PIPELINE_GROUPS_H

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

} // namespace vernon::runtime

#endif
