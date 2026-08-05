#include "host_tape_allocator.h"

#include <algorithm>
#include <new>
#include <stdexcept>
#include <utility>

namespace vernon::runtime::ad {

namespace {

using AllocatorRegistryEntry = std::pair<VernonAdTapeAllocator *, HostTapeAllocator *>;
thread_local std::vector<AllocatorRegistryEntry> allocatorRegistry;

} // namespace

bool HostTapeSnapshot::readRegion(VernonAdRegionHandle handle, const std::byte *&data, size_t &size) const {
    const auto region = std::find_if(regions_.begin(), regions_.end(),
                                     [handle](const Region &candidate) { return candidate.handle == handle; });
    if (region == regions_.end() || region->begin > region->end || region->end > storage_.size())
        return false;
    data = region->begin == region->end ? nullptr : storage_.data() + region->begin;
    size = region->end - region->begin;
    return true;
}

HostTapeAllocator::HostTapeAllocator(size_t capacityBytes) : ownerThread_(std::this_thread::get_id()) {
    descriptor_.struct_size = sizeof(descriptor_);
    descriptor_.abi_version = VERNON_AD_TAPE_ALLOCATOR_ABI_VERSION;
    descriptor_.user_data = this;
    descriptor_.capacity_bytes = capacityBytes;
    descriptor_.reset = reset;
    descriptor_.begin_region = beginRegion;
    descriptor_.reserve = reserve;
    descriptor_.end_region = endRegion;
    descriptor_.seal = seal;
    descriptor_.read_region = readRegion;
    allocatorRegistry.emplace_back(&descriptor_, this);
    reset(&descriptor_);
}

HostTapeAllocator::~HostTapeAllocator() {
    const auto entry =
        std::find_if(allocatorRegistry.begin(), allocatorRegistry.end(),
                     [this](const AllocatorRegistryEntry &candidate) { return candidate.first == &descriptor_; });
    if (entry != allocatorRegistry.end())
        allocatorRegistry.erase(entry);
}

HostTapeAllocator *HostTapeAllocator::owner(VernonAdTapeAllocator *allocator) {
    if (!allocator)
        return nullptr;
    if (allocator->struct_size != sizeof(VernonAdTapeAllocator) ||
        allocator->abi_version != VERNON_AD_TAPE_ALLOCATOR_ABI_VERSION) {
        allocator->status = VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
        return nullptr;
    }
    const auto owner =
        std::find_if(allocatorRegistry.begin(), allocatorRegistry.end(),
                     [allocator](const AllocatorRegistryEntry &candidate) { return candidate.first == allocator; });
    if (owner == allocatorRegistry.end()) {
        allocator->status = VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
        return nullptr;
    }
    return owner->second;
}

bool HostTapeAllocator::callable() const { return ownerThread_ == std::this_thread::get_id(); }

VernonAdTapeAllocatorStatus HostTapeAllocator::fail(VernonAdTapeAllocatorStatus status) {
    descriptor_.status = status;
    return status;
}

HostTapeAllocator::Region *HostTapeAllocator::findRegion(VernonAdRegionHandle handle) {
    const auto region = std::find_if(regions_.begin(), regions_.end(),
                                     [handle](const Region &candidate) { return candidate.handle == handle; });
    return region == regions_.end() ? nullptr : &*region;
}

VernonAdTapeAllocatorStatus HostTapeAllocator::reset(VernonAdTapeAllocator *allocator) {
    HostTapeAllocator *self = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (!self->callable())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    self->storage_.clear();
    self->regions_.clear();
    self->openRegions_.clear();
    self->sealed_ = false;
    allocator->required_bytes = 0;
    allocator->status = VERNON_AD_TAPE_ALLOCATOR_OK;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostTapeAllocator::beginRegion(VernonAdTapeAllocator *allocator,
                                                           VernonAdRegionHandle parent, VernonAdRegionHandle *region) {
    if (region)
        *region = VERNON_AD_INVALID_REGION_HANDLE;
    HostTapeAllocator *self = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (!self->callable())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    if (allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK)
        return allocator->status;
    if (!region || self->sealed_)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    const VernonAdRegionHandle expectedParent =
        self->openRegions_.empty() ? VERNON_AD_INVALID_REGION_HANDLE : self->regions_[self->openRegions_.back()].handle;
    if (parent != expectedParent || self->nextRegionHandle_ == VERNON_AD_INVALID_REGION_HANDLE)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    const VernonAdRegionHandle handle = self->nextRegionHandle_++;
    try {
        self->regions_.push_back({handle, allocator->required_bytes, allocator->required_bytes, true});
    } catch (const std::bad_alloc &) {
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
    } catch (const std::length_error &) {
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
    }
    try {
        self->openRegions_.push_back(self->regions_.size() - 1);
    } catch (const std::bad_alloc &) {
        self->regions_.pop_back();
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
    } catch (const std::length_error &) {
        self->regions_.pop_back();
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
    }
    *region = handle;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostTapeAllocator::reserve(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region,
                                                       size_t byteSize, size_t alignment, size_t *byteOffset,
                                                       void **writeAddress) {
    if (byteOffset)
        *byteOffset = 0;
    if (writeAddress)
        *writeAddress = nullptr;
    HostTapeAllocator *self = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (!self->callable())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    if (allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK &&
        allocator->status != VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED)
        return allocator->status;
    if (!byteOffset || !writeAddress || self->sealed_ || !alignment || (alignment & (alignment - 1)) != 0 ||
        self->openRegions_.empty())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    Region *active = &self->regions_[self->openRegions_.back()];
    if (active->handle != region)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);

    const size_t current = allocator->required_bytes;
    const size_t mask = alignment - 1;
    if (current > std::numeric_limits<size_t>::max() - mask)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW);
    const size_t aligned = (current + mask) & ~mask;
    if (byteSize > std::numeric_limits<size_t>::max() - aligned)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW);
    const size_t end = aligned + byteSize;
    allocator->required_bytes = end;
    for (const size_t openRegion : self->openRegions_)
        self->regions_[openRegion].end = end;
    *byteOffset = aligned;

    if (allocator->status == VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED || end > allocator->capacity_bytes)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED);
    if (end > self->storage_.size()) {
        try {
            self->storage_.resize(end);
        } catch (const std::bad_alloc &) {
            return self->fail(VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
        } catch (const std::length_error &) {
            return self->fail(VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
        }
    }
    *writeAddress = byteSize ? self->storage_.data() + aligned : nullptr;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostTapeAllocator::endRegion(VernonAdTapeAllocator *allocator,
                                                         VernonAdRegionHandle region) {
    HostTapeAllocator *self = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (!self->callable())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    if (allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK)
        return allocator->status;
    if (self->sealed_ || self->openRegions_.empty())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    Region &active = self->regions_[self->openRegions_.back()];
    if (active.handle != region || !active.open)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    active.open = false;
    self->openRegions_.pop_back();
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostTapeAllocator::seal(VernonAdTapeAllocator *allocator) {
    HostTapeAllocator *self = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (!self->callable())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    if (allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK)
        return allocator->status;
    if (self->sealed_ || !self->openRegions_.empty())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    self->sealed_ = true;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostTapeAllocator::readRegion(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region,
                                                          const void **data, size_t *byteSize) {
    if (data)
        *data = nullptr;
    if (byteSize)
        *byteSize = 0;
    HostTapeAllocator *self = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (!self->callable())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    if (allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK)
        return allocator->status;
    if (!self->sealed_ || !data || !byteSize)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    Region *found = self->findRegion(region);
    if (!found || found->open || found->begin > found->end || found->end > self->storage_.size())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    *data = found->begin == found->end ? nullptr : self->storage_.data() + found->begin;
    *byteSize = found->end - found->begin;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

std::shared_ptr<const HostTapeSnapshot> HostTapeAllocator::takeSnapshot() {
    if (!callable() || !sealed_ || descriptor_.status != VERNON_AD_TAPE_ALLOCATOR_OK)
        return {};
    try {
        auto snapshot = std::make_shared<HostTapeSnapshot>();
        snapshot->regions_.reserve(regions_.size());
        for (const Region &region : regions_)
            snapshot->regions_.push_back({region.handle, region.begin, region.end});
        snapshot->storage_ = std::move(storage_);
        descriptor_.status = VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
        return snapshot;
    } catch (const std::bad_alloc &) {
        fail(VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
    } catch (const std::length_error &) {
        fail(VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
    }
    return {};
}

} // namespace vernon::runtime::ad
