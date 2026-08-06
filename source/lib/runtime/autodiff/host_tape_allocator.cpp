#include "host_tape_allocator.h"
#include "VernonRuntime.h"

#include <algorithm>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

namespace vernon::runtime::ad {
namespace {

using TapeRegistryEntry = std::pair<VernonAdTapeAllocator *, HostDynamicTape *>;
thread_local std::vector<TapeRegistryEntry> tapeRegistry;

bool checkedAdd(size_t left, size_t right, size_t &result) {
    if (right > std::numeric_limits<size_t>::max() - left)
        return false;
    result = left + right;
    return true;
}

bool checkedMultiply(size_t left, size_t right, size_t &result) {
    if (left && right > std::numeric_limits<size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

} // namespace

bool HostTapeMemoryPolicy::reserve(size_t currentInvocationBytes, size_t additionalBytes) {
    std::lock_guard lock(mutex_);
    size_t invocationBytes = 0;
    size_t contextBytes = 0;
    if (!checkedAdd(currentInvocationBytes, additionalBytes, invocationBytes) ||
        !checkedAdd(contextBytes_, additionalBytes, contextBytes) || invocationBytes > invocationLimit_ ||
        contextBytes > contextLimit_)
        return false;
    contextBytes_ = contextBytes;
    return true;
}

void HostTapeMemoryPolicy::release(size_t bytes) {
    std::lock_guard lock(mutex_);
    contextBytes_ = bytes > contextBytes_ ? 0 : contextBytes_ - bytes;
}

HostTapeSnapshot::~HostTapeSnapshot() {
    if (policy_ && policyCharge_)
        policy_->release(policyCharge_);
}

const HostTapeSnapshot::Region *HostTapeSnapshot::findRegion(VernonAdRegionHandle handle) const {
    const auto found = regionIndex_.find(handle);
    return found == regionIndex_.end() || found->second >= regions_.size() ? nullptr : &regions_[found->second];
}

VernonAdRegionHandle HostTapeSnapshot::rootRegion() const {
    return regions_.empty() ? VERNON_AD_INVALID_REGION_HANDLE : regions_.front().handle;
}

VernonAdTapeAllocatorStatus HostTapeSnapshot::reject(VernonAdTapeAllocator *allocator) {
    if (allocator)
        allocator->status = VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
    return VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
}

VernonAdTapeAllocatorStatus HostTapeSnapshot::rejectBegin(VernonAdTapeAllocator *allocator, VernonAdRegionHandle,
                                                          VernonAdRegionHandle *) {
    return reject(allocator);
}

VernonAdTapeAllocatorStatus HostTapeSnapshot::rejectReserve(VernonAdTapeAllocator *allocator, VernonAdRegionHandle,
                                                            size_t, size_t, size_t, VernonAdRecordHandle *) {
    return reject(allocator);
}

VernonAdTapeAllocatorStatus HostTapeSnapshot::rejectWrite(VernonAdTapeAllocator *allocator, VernonAdRecordHandle,
                                                          size_t, const void *, size_t) {
    return reject(allocator);
}

VernonAdTapeAllocatorStatus HostTapeSnapshot::rejectChild(VernonAdTapeAllocator *allocator, VernonAdRecordHandle,
                                                          size_t, VernonAdRegionHandle) {
    return reject(allocator);
}

VernonAdTapeAllocatorStatus HostTapeSnapshot::rejectEnd(VernonAdTapeAllocator *allocator, VernonAdRegionHandle, size_t,
                                                        uint32_t) {
    return reject(allocator);
}

VernonAdTapeAllocatorStatus HostTapeSnapshot::readLeaf(VernonAdTapeAllocator *allocator,
                                                       VernonAdRegionHandle regionHandle, size_t recordIndex,
                                                       size_t leafOffset, void *data, size_t byteSize) {
    if (!allocator || allocator->struct_size != sizeof(VernonAdTapeAllocator) ||
        allocator->abi_version != VERNON_AD_TAPE_ALLOCATOR_ABI_VERSION || !allocator->user_data ||
        allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK || (byteSize && !data))
        return reject(allocator);
    const auto *self = static_cast<const HostTapeSnapshot *>(allocator->user_data);
    const Region *region = self->findRegion(regionHandle);
    if (!region || recordIndex >= region->records.size() || region->records[recordIndex] >= self->records_.size())
        return reject(allocator);
    const Record &record = self->records_[region->records[recordIndex]];
    size_t end = 0;
    if (!checkedAdd(leafOffset, byteSize, end) || end > record.payloadSize ||
        record.payloadOffset > self->payload_.size() ||
        record.payloadSize > self->payload_.size() - record.payloadOffset)
        return reject(allocator);
    if (byteSize)
        std::memcpy(data, self->payload_.data() + record.payloadOffset + leafOffset, byteSize);
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostTapeSnapshot::readChild(VernonAdTapeAllocator *allocator,
                                                        VernonAdRegionHandle regionHandle, size_t recordIndex,
                                                        size_t childOrdinal, VernonAdRegionHandle *child) {
    if (child)
        *child = VERNON_AD_INVALID_REGION_HANDLE;
    if (!allocator || !child || allocator->struct_size != sizeof(VernonAdTapeAllocator) ||
        allocator->abi_version != VERNON_AD_TAPE_ALLOCATOR_ABI_VERSION || !allocator->user_data ||
        allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK)
        return reject(allocator);
    const auto *self = static_cast<const HostTapeSnapshot *>(allocator->user_data);
    const Region *region = self->findRegion(regionHandle);
    if (!region || recordIndex >= region->records.size() || region->records[recordIndex] >= self->records_.size())
        return reject(allocator);
    const Record &record = self->records_[region->records[recordIndex]];
    if (childOrdinal >= record.children.size() || !self->findRegion(record.children[childOrdinal]))
        return reject(allocator);
    *child = record.children[childOrdinal];
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostTapeSnapshot::readExecutedCount(VernonAdTapeAllocator *allocator,
                                                                VernonAdRegionHandle regionHandle,
                                                                size_t *executedCount) {
    if (executedCount)
        *executedCount = 0;
    if (!allocator || !executedCount || allocator->struct_size != sizeof(VernonAdTapeAllocator) ||
        allocator->abi_version != VERNON_AD_TAPE_ALLOCATOR_ABI_VERSION || !allocator->user_data ||
        allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK)
        return reject(allocator);
    const auto *self = static_cast<const HostTapeSnapshot *>(allocator->user_data);
    const Region *region = self->findRegion(regionHandle);
    if (!region)
        return reject(allocator);
    *executedCount = region->executedCount;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostTapeSnapshot::readExitKind(VernonAdTapeAllocator *allocator,
                                                           VernonAdRegionHandle regionHandle, uint32_t *exitKind) {
    if (exitKind)
        *exitKind = 0;
    if (!allocator || !exitKind || allocator->struct_size != sizeof(VernonAdTapeAllocator) ||
        allocator->abi_version != VERNON_AD_TAPE_ALLOCATOR_ABI_VERSION || !allocator->user_data ||
        allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK)
        return reject(allocator);
    const auto *self = static_cast<const HostTapeSnapshot *>(allocator->user_data);
    const Region *region = self->findRegion(regionHandle);
    if (!region)
        return reject(allocator);
    *exitKind = region->exitKind;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

void HostTapeSnapshot::initializeDescriptor() {
    descriptor_ = {};
    descriptor_.struct_size = sizeof(descriptor_);
    descriptor_.abi_version = VERNON_AD_TAPE_ALLOCATOR_ABI_VERSION;
    descriptor_.status = VERNON_AD_TAPE_ALLOCATOR_OK;
    descriptor_.user_data = this;
    descriptor_.capacity_bytes = policyCharge_;
    descriptor_.required_bytes = policyCharge_;
    descriptor_.reset = reject;
    descriptor_.begin_region = rejectBegin;
    descriptor_.reserve_record = rejectReserve;
    descriptor_.write_leaf = rejectWrite;
    descriptor_.set_child = rejectChild;
    descriptor_.end_region = rejectEnd;
    descriptor_.seal = reject;
    descriptor_.read_leaf = readLeaf;
    descriptor_.read_child = readChild;
    descriptor_.read_executed_count = readExecutedCount;
    descriptor_.read_exit_kind = readExitKind;
}

HostDynamicTape::HostDynamicTape(size_t capacityBytes, std::shared_ptr<HostTapeMemoryPolicy> policy)
    : policy_(std::move(policy)), ownerThread_(std::this_thread::get_id()) {
    descriptor_.struct_size = sizeof(descriptor_);
    descriptor_.abi_version = VERNON_AD_TAPE_ALLOCATOR_ABI_VERSION;
    descriptor_.user_data = this;
    descriptor_.capacity_bytes = capacityBytes;
    descriptor_.reset = reset;
    descriptor_.begin_region = beginRegion;
    descriptor_.reserve_record = reserveRecord;
    descriptor_.write_leaf = writeLeaf;
    descriptor_.set_child = setChild;
    descriptor_.end_region = endRegion;
    descriptor_.seal = seal;
    descriptor_.read_leaf = readLeaf;
    descriptor_.read_child = readChild;
    descriptor_.read_executed_count = readCount;
    descriptor_.read_exit_kind = readExit;
    tapeRegistry.emplace_back(&descriptor_, this);
    reset(&descriptor_);
}

HostDynamicTape::~HostDynamicTape() {
    const auto entry =
        std::find_if(tapeRegistry.begin(), tapeRegistry.end(),
                     [this](const TapeRegistryEntry &candidate) { return candidate.first == &descriptor_; });
    if (entry != tapeRegistry.end())
        tapeRegistry.erase(entry);
    releaseCharge();
}

HostDynamicTape *HostDynamicTape::owner(VernonAdTapeAllocator *allocator) {
    if (!allocator)
        return nullptr;
    if (allocator->struct_size != sizeof(VernonAdTapeAllocator) ||
        allocator->abi_version != VERNON_AD_TAPE_ALLOCATOR_ABI_VERSION) {
        allocator->status = VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
        return nullptr;
    }
    const auto found = std::find_if(tapeRegistry.begin(), tapeRegistry.end(),
                                    [allocator](const TapeRegistryEntry &entry) { return entry.first == allocator; });
    if (found == tapeRegistry.end()) {
        allocator->status = VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
        return nullptr;
    }
    return found->second;
}

bool HostDynamicTape::callable() const { return ownerThread_ == std::this_thread::get_id(); }

VernonAdTapeAllocatorStatus HostDynamicTape::fail(VernonAdTapeAllocatorStatus status) {
    if (descriptor_.status == VERNON_AD_TAPE_ALLOCATOR_OK ||
        descriptor_.status == VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED)
        descriptor_.status = status;
    return descriptor_.status;
}

void HostDynamicTape::releaseCharge() {
    if (policy_ && policyCharge_)
        policy_->release(std::exchange(policyCharge_, 0));
}

HostDynamicTape::Region *HostDynamicTape::findRegion(VernonAdRegionHandle handle) {
    const auto found = regionIndex_.find(handle);
    return found == regionIndex_.end() || found->second >= regions_.size() ? nullptr : &regions_[found->second];
}

HostDynamicTape::Record *HostDynamicTape::findRecord(VernonAdRecordHandle handle) {
    const auto found = recordIndex_.find(handle);
    return found == recordIndex_.end() || found->second >= records_.size() ? nullptr : &records_[found->second];
}

VernonAdTapeAllocatorStatus HostDynamicTape::reset(VernonAdTapeAllocator *allocator) {
    HostDynamicTape *self = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (!self->callable())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    self->releaseCharge();
    self->payload_.clear();
    self->regions_.clear();
    self->records_.clear();
    self->regionIndex_.clear();
    self->recordIndex_.clear();
    self->openRegions_.clear();
    self->sealed_ = false;
    allocator->required_bytes = 0;
    allocator->status = VERNON_AD_TAPE_ALLOCATOR_OK;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostDynamicTape::beginRegion(VernonAdTapeAllocator *allocator, VernonAdRegionHandle parent,
                                                         VernonAdRegionHandle *region) {
    if (region)
        *region = VERNON_AD_INVALID_REGION_HANDLE;
    HostDynamicTape *self = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (!self->callable())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    if (allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK)
        return allocator->status;
    if (!region || self->sealed_)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    const VernonAdRegionHandle expected =
        self->openRegions_.empty() ? VERNON_AD_INVALID_REGION_HANDLE : self->regions_[self->openRegions_.back()].handle;
    if (parent != expected || self->nextRegionHandle_ == VERNON_AD_INVALID_REGION_HANDLE)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    try {
        const size_t index = self->regions_.size();
        const VernonAdRegionHandle handle = self->nextRegionHandle_++;
        self->regions_.push_back({handle, parent, {}, 0, 0, true, false});
        self->regionIndex_.emplace(handle, index);
        self->openRegions_.push_back(index);
        *region = handle;
        return VERNON_AD_TAPE_ALLOCATOR_OK;
    } catch (const std::bad_alloc &) {
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
    } catch (const std::length_error &) {
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
    }
}

VernonAdTapeAllocatorStatus HostDynamicTape::reserveRecord(VernonAdTapeAllocator *allocator,
                                                           VernonAdRegionHandle regionHandle, size_t payloadSize,
                                                           size_t payloadAlignment, size_t childCount,
                                                           VernonAdRecordHandle *record) {
    if (record)
        *record = VERNON_AD_INVALID_RECORD_HANDLE;
    HostDynamicTape *self = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (!self->callable())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    if (allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK &&
        allocator->status != VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED)
        return allocator->status;
    if (!record || self->sealed_ || !payloadAlignment || (payloadAlignment & (payloadAlignment - 1)) ||
        self->openRegions_.empty() || self->regions_[self->openRegions_.back()].handle != regionHandle)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);

    size_t childBytes = 0;
    size_t afterChildren = 0;
    size_t aligned = 0;
    size_t end = 0;
    if (!checkedMultiply(childCount, sizeof(VernonAdRegionHandle), childBytes) ||
        !checkedAdd(allocator->required_bytes, childBytes, afterChildren) ||
        !checkedAdd(afterChildren, payloadAlignment - 1, aligned)) {
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW);
    }
    aligned &= ~(payloadAlignment - 1);
    if (!checkedAdd(aligned, payloadSize, end))
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW);
    const size_t additional = end - allocator->required_bytes;
    allocator->required_bytes = end;
    if (allocator->status == VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED || end > allocator->capacity_bytes ||
        !self->policy_ || !self->policy_->reserve(self->policyCharge_, additional))
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED);

    const size_t payloadOffset = self->payload_.size();
    try {
        self->payload_.resize(payloadOffset + payloadSize);
        const size_t index = self->records_.size();
        const VernonAdRecordHandle handle = self->nextRecordHandle_++;
        self->records_.push_back({handle, self->openRegions_.back(), payloadOffset, payloadSize,
                                  std::vector<VernonAdRegionHandle>(childCount)});
        self->recordIndex_.emplace(handle, index);
        self->regions_[self->openRegions_.back()].records.push_back(index);
        self->policyCharge_ += additional;
        *record = handle;
        return VERNON_AD_TAPE_ALLOCATOR_OK;
    } catch (const std::bad_alloc &) {
        self->policy_->release(additional);
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
    } catch (const std::length_error &) {
        self->policy_->release(additional);
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
    }
}

VernonAdTapeAllocatorStatus HostDynamicTape::writeLeaf(VernonAdTapeAllocator *allocator,
                                                       VernonAdRecordHandle recordHandle, size_t leafOffset,
                                                       const void *data, size_t byteSize) {
    HostDynamicTape *self = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (!self->callable())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    if (allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK)
        return allocator->status;
    if (self->sealed_ || (byteSize && !data))
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    Record *record = self->findRecord(recordHandle);
    size_t end = 0;
    if (!record || !checkedAdd(leafOffset, byteSize, end) || end > record->payloadSize ||
        record->regionIndex != self->openRegions_.back())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    if (byteSize)
        std::memcpy(self->payload_.data() + record->payloadOffset + leafOffset, data, byteSize);
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostDynamicTape::setChild(VernonAdTapeAllocator *allocator,
                                                      VernonAdRecordHandle recordHandle, size_t childOrdinal,
                                                      VernonAdRegionHandle childHandle) {
    HostDynamicTape *self = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (!self->callable())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    if (allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK)
        return allocator->status;
    if (self->sealed_)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    Record *record = self->findRecord(recordHandle);
    Region *child = self->findRegion(childHandle);
    if (!record || !child || childOrdinal >= record->children.size() ||
        record->children[childOrdinal] != VERNON_AD_INVALID_REGION_HANDLE ||
        child->parent != self->regions_[record->regionIndex].handle)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    record->children[childOrdinal] = childHandle;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostDynamicTape::endRegion(VernonAdTapeAllocator *allocator,
                                                       VernonAdRegionHandle regionHandle, size_t executedCount,
                                                       uint32_t exitKind) {
    HostDynamicTape *self = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (!self->callable())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    if (allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK)
        return allocator->status;
    if (self->sealed_ || self->openRegions_.empty())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    Region &region = self->regions_[self->openRegions_.back()];
    if (region.handle != regionHandle || !region.open)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    region.executedCount = executedCount;
    region.exitKind = exitKind;
    region.open = false;
    region.ended = true;
    self->openRegions_.pop_back();
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostDynamicTape::seal(VernonAdTapeAllocator *allocator) {
    HostDynamicTape *self = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (!self->callable())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    if (allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK)
        return allocator->status;
    if (self->sealed_ || !self->openRegions_.empty() || self->regions_.empty())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    for (const Region &region : self->regions_)
        if (!region.ended)
            return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    for (const Record &record : self->records_)
        for (VernonAdRegionHandle child : record.children)
            if (!child || !self->findRegion(child))
                return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    self->sealed_ = true;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostDynamicTape::readLeaf(VernonAdTapeAllocator *allocator,
                                                      VernonAdRegionHandle regionHandle, size_t recordIndex,
                                                      size_t leafOffset, void *data, size_t byteSize) {
    HostDynamicTape *self = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (!self->callable() || allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK || !self->sealed_ || (byteSize && !data))
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    Region *region = self->findRegion(regionHandle);
    if (!region || recordIndex >= region->records.size() || region->records[recordIndex] >= self->records_.size())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    const Record &record = self->records_[region->records[recordIndex]];
    size_t end = 0;
    if (!checkedAdd(leafOffset, byteSize, end) || end > record.payloadSize)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    if (byteSize)
        std::memcpy(data, self->payload_.data() + record.payloadOffset + leafOffset, byteSize);
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostDynamicTape::readChild(VernonAdTapeAllocator *allocator,
                                                       VernonAdRegionHandle regionHandle, size_t recordIndex,
                                                       size_t childOrdinal, VernonAdRegionHandle *child) {
    if (child)
        *child = VERNON_AD_INVALID_REGION_HANDLE;
    HostDynamicTape *self = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (!self->callable() || allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK || !self->sealed_ || !child)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    Region *region = self->findRegion(regionHandle);
    if (!region || recordIndex >= region->records.size() || region->records[recordIndex] >= self->records_.size())
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    const Record &record = self->records_[region->records[recordIndex]];
    if (childOrdinal >= record.children.size() || !self->findRegion(record.children[childOrdinal]))
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    *child = record.children[childOrdinal];
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostDynamicTape::readCount(VernonAdTapeAllocator *allocator,
                                                       VernonAdRegionHandle regionHandle, size_t *count) {
    if (count)
        *count = 0;
    HostDynamicTape *self = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (!self->callable() || allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK || !self->sealed_ || !count)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    Region *region = self->findRegion(regionHandle);
    if (!region)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    *count = region->executedCount;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostDynamicTape::readExit(VernonAdTapeAllocator *allocator,
                                                      VernonAdRegionHandle regionHandle, uint32_t *exitKind) {
    if (exitKind)
        *exitKind = 0;
    HostDynamicTape *self = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (!self->callable() || allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK || !self->sealed_ || !exitKind)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    Region *region = self->findRegion(regionHandle);
    if (!region)
        return self->fail(VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    *exitKind = region->exitKind;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

std::shared_ptr<const HostTapeSnapshot> HostDynamicTape::takeSnapshot() {
    if (!callable() || !sealed_ || descriptor_.status != VERNON_AD_TAPE_ALLOCATOR_OK)
        return {};
    try {
        auto snapshot = std::make_shared<HostTapeSnapshot>();
        snapshot->payload_ = payload_;
        snapshot->regions_.reserve(regions_.size());
        snapshot->records_.reserve(records_.size());
        for (const Region &region : regions_)
            snapshot->regions_.push_back({region.handle, region.records, region.executedCount, region.exitKind});
        for (const Record &record : records_)
            snapshot->records_.push_back({record.handle, record.payloadOffset, record.payloadSize, record.children});
        snapshot->regionIndex_ = regionIndex_;
        snapshot->policy_ = policy_;
        snapshot->policyCharge_ = std::exchange(policyCharge_, 0);
        snapshot->initializeDescriptor();
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
