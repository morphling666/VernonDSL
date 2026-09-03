#include "program_instance.h"

#include <stdexcept>
#include <utility>

namespace vernon::runtime::program {

const std::shared_ptr<void> *InvocationSnapshot::find(uint32_t slot) const {
    const auto found = entries_.find(slot);
    return found == entries_.end() ? nullptr : &found->second.payload;
}

BindingTransaction::BindingTransaction(PersistentBindingState &state, const void *executable, uint64_t generation,
                                       std::unordered_map<uint32_t, BindingEntry> snapshot)
    : state_(&state), executable_(executable), generation_(generation), snapshot_(std::move(snapshot)) {}

BindingTransaction::~BindingTransaction() {
    if (!finished_)
        rollback();
}

const std::shared_ptr<void> *BindingTransaction::find(uint32_t slot, const std::string &token) {
    if (finished_)
        throw std::logic_error("Program binding transaction is already finished");
    const auto updated = updates_.find(slot);
    if (updated != updates_.end()) {
        if (updated->second.token != token)
            return nullptr;
        ++reuseCount_;
        return &updated->second.payload;
    }
    const auto existing = snapshot_.find(slot);
    if (existing == snapshot_.end() || existing->second.token != token)
        return nullptr;
    ++reuseCount_;
    return &existing->second.payload;
}

void BindingTransaction::observeUploads(uint64_t uploadBytes, uint64_t uploadRanges) {
    if (finished_)
        throw std::logic_error("Program binding transaction is already finished");
    uploadBytes_ += uploadBytes;
    uploadRanges_ += uploadRanges;
}

void BindingTransaction::stage(uint32_t slot, std::string token, std::shared_ptr<void> payload, uint64_t uploadBytes,
                               uint64_t uploadRanges) {
    if (finished_)
        throw std::logic_error("Program binding transaction is already finished");
    if (!payload)
        throw std::invalid_argument("Program binding payload must not be null");
    updates_.insert_or_assign(slot, BindingEntry{std::move(token), std::move(payload)});
    uploadBytes_ += uploadBytes;
    uploadRanges_ += uploadRanges;
}

std::shared_ptr<const InvocationSnapshot> BindingTransaction::snapshot() const {
    if (finished_)
        throw std::logic_error("Program binding transaction is already finished");
    auto entries = snapshot_;
    for (const auto &[slot, entry] : updates_)
        entries.insert_or_assign(slot, entry);
    return std::shared_ptr<const InvocationSnapshot>(new InvocationSnapshot(std::move(entries)));
}

std::shared_ptr<const InvocationSnapshot> BindingTransaction::commit() {
    if (finished_)
        throw std::logic_error("Program binding transaction is already finished");
    std::lock_guard lock(state_->mutex_);
    if (state_->executable_ != executable_ || state_->generation_ != generation_)
        throw std::runtime_error("Program executable changed before binding transaction commit");
    for (auto &[slot, entry] : updates_) {
        snapshot_.insert_or_assign(slot, entry);
        state_->entries_.insert_or_assign(slot, std::move(entry));
        ++state_->telemetry_.prepareCount;
    }
    state_->telemetry_.reuseCount += reuseCount_;
    state_->telemetry_.uploadBytes += uploadBytes_;
    state_->telemetry_.uploadRanges += uploadRanges_;
    finished_ = true;
    return std::shared_ptr<const InvocationSnapshot>(new InvocationSnapshot(std::move(snapshot_)));
}

void BindingTransaction::rollback() {
    if (finished_)
        return;
    std::lock_guard lock(state_->mutex_);
    ++state_->telemetry_.rollbackCount;
    finished_ = true;
}

std::unique_ptr<BindingTransaction> PersistentBindingState::begin(const void *executable) {
    if (!executable)
        throw std::invalid_argument("Program executable identity must not be null");
    std::lock_guard lock(mutex_);
    if (executable_ != executable) {
        executable_ = executable;
        entries_.clear();
        ++generation_;
    }
    return std::unique_ptr<BindingTransaction>(new BindingTransaction(*this, executable, generation_, entries_));
}

void PersistentBindingState::clear() {
    std::lock_guard lock(mutex_);
    executable_ = nullptr;
    entries_.clear();
    ++generation_;
}

BindingTelemetry PersistentBindingState::telemetry() const {
    std::lock_guard lock(mutex_);
    return telemetry_;
}

std::unique_ptr<BindingTransaction> ProgramInstance::beginInvocation() { return bindings_.begin(executable_); }

BindingTelemetry ProgramInstance::telemetry() const { return bindings_.telemetry(); }

void ProgramInstance::clear() { bindings_.clear(); }

} // namespace vernon::runtime::program
