#ifndef VERNON_RHI_LOGICAL_RESOURCE_RECORD_H
#define VERNON_RHI_LOGICAL_RESOURCE_RECORD_H

#include <cassert>
#include <cstdint>
#include <limits>

namespace vernon::rhi {

template <typename Handle> constexpr uint64_t encodeResourceKey(Handle handle) {
    return (static_cast<uint64_t>(handle.generation) << 32) | (static_cast<uint64_t>(handle.index) + 1);
}

struct LogicalResourceRecord {
    uint32_t bindingReferences{};
    uint32_t generation{1};
    bool publicAlive{};
    bool occupied{};

    bool isPublic(uint32_t expectedGeneration) const {
        return occupied && publicAlive && generation == expectedGeneration;
    }

    bool isRetained(uint32_t expectedGeneration) const { return occupied && generation == expectedGeneration; }

    void publish() {
        assert(!occupied);
        assert(bindingReferences == 0);
        occupied = true;
        publicAlive = true;
    }

    void destroyPublicOwner() {
        assert(occupied && publicAlive);
        publicAlive = false;
    }

    void restorePublicOwner() {
        assert(occupied && !publicAlive);
        publicAlive = true;
    }

    bool retain() {
        if (!occupied || bindingReferences == (std::numeric_limits<uint32_t>::max)())
            return false;
        ++bindingReferences;
        return true;
    }

    bool release() {
        if (!occupied || bindingReferences == 0)
            return false;
        --bindingReferences;
        return !publicAlive && bindingReferences == 0;
    }

    void recycle() {
        assert(occupied);
        assert(!publicAlive);
        assert(bindingReferences == 0);
        occupied = false;
        if (++generation == 0)
            generation = 1;
    }

    void validate() const {
        assert(generation != 0);
        assert(!publicAlive || occupied);
        assert(bindingReferences == 0 || occupied);
    }
};

} // namespace vernon::rhi

#endif
