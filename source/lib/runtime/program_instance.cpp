#include "program_instance.h"

#include <utility>

namespace vernon::runtime::program {

const std::shared_ptr<void> *InvocationSnapshot::find(uint64_t slot) const {
    const auto found = entries_.find(slot);
    return found == entries_.end() ? nullptr : &found->second.payload;
}

BindingTransaction::BindingTransaction(PersistentBindingState &state, const void *executable, uint64_t generation,
                                       std::shared_ptr<const InvocationSnapshot> snapshot)
    : state_(&state), executable_(executable), generation_(generation), snapshot_(std::move(snapshot)) {}

BindingTransaction::~BindingTransaction() noexcept {
    if (!finished_)
        rollback();
}

BindingResult<bool> BindingTransaction::matches(uint64_t slot, std::string_view token) const noexcept {
    if (finished_ || frozen_)
        return BindingResult<bool>{vernon::err(BindingError::InvalidState)};
    const auto updated = updates_.find(slot);
    if (updated != updates_.end())
        return BindingResult<bool>{vernon::ok(updated->second.token == token)};
    const auto existing = snapshot_->entries_.find(slot);
    return BindingResult<bool>{vernon::ok(existing != snapshot_->entries_.end() && existing->second.token == token)};
}

BindingResult<void> BindingTransaction::observeReuses(uint64_t count) noexcept {
    if (finished_ || frozen_)
        return BindingResult<void>{vernon::err(BindingError::InvalidState)};
    reuseCount_ += count;
    return BindingResult<void>{vernon::ok()};
}

BindingResult<void> BindingTransaction::observeUploads(uint64_t uploadBytes, uint64_t uploadRanges) noexcept {
    if (finished_ || frozen_)
        return BindingResult<void>{vernon::err(BindingError::InvalidState)};
    uploadBytes_ += uploadBytes;
    uploadRanges_ += uploadRanges;
    return BindingResult<void>{vernon::ok()};
}

BindingResult<void> BindingTransaction::stage(uint64_t slot, std::string token, std::shared_ptr<void> payload,
                                              uint64_t uploadBytes, uint64_t uploadRanges) noexcept {
    try {
        std::vector<std::pair<uint64_t, BindingEntry>> entries;
        entries.emplace_back(slot, BindingEntry{std::move(token), std::move(payload)});
        return stageMany(std::move(entries), uploadBytes, uploadRanges);
    } catch (...) {
        return BindingResult<void>{vernon::err(BindingError::AllocationFailure)};
    }
}

BindingResult<void> BindingTransaction::stageMany(std::vector<std::pair<uint64_t, BindingEntry>> entries,
                                                  uint64_t uploadBytes, uint64_t uploadRanges) noexcept {
    if (finished_ || frozen_)
        return BindingResult<void>{vernon::err(BindingError::InvalidState)};
    try {
        auto candidate = updates_;
        for (auto &[slot, entry] : entries) {
            if (!entry.payload)
                return BindingResult<void>{vernon::err(BindingError::InvalidPayload)};
            candidate.insert_or_assign(slot, std::move(entry));
        }
        updates_.swap(candidate);
        uploadBytes_ += uploadBytes;
        uploadRanges_ += uploadRanges;
        return BindingResult<void>{vernon::ok()};
    } catch (...) {
        return BindingResult<void>{vernon::err(BindingError::AllocationFailure)};
    }
}

BindingResult<std::shared_ptr<const InvocationSnapshot>> BindingTransaction::freeze() noexcept {
    if (finished_ || frozen_)
        return BindingResult<std::shared_ptr<const InvocationSnapshot>>{vernon::err(BindingError::InvalidState)};
    try {
        if (updates_.empty())
            frozenSnapshot_ = snapshot_;
        else {
            BindingMap entries = snapshot_->entries_;
            for (const auto &[slot, entry] : updates_)
                entries.insert_or_assign(slot, entry);
            frozenSnapshot_ = std::shared_ptr<const InvocationSnapshot>(new InvocationSnapshot(std::move(entries)));
        }
        frozen_ = true;
        return BindingResult<std::shared_ptr<const InvocationSnapshot>>{vernon::ok(frozenSnapshot_)};
    } catch (...) {
        return BindingResult<std::shared_ptr<const InvocationSnapshot>>{vernon::err(BindingError::AllocationFailure)};
    }
}

BindingResult<std::shared_ptr<const InvocationSnapshot>> BindingTransaction::commit() noexcept {
    if (finished_)
        return BindingResult<std::shared_ptr<const InvocationSnapshot>>{vernon::err(BindingError::InvalidState)};
    if (!frozen_) {
        auto frozen = freeze();
        if (frozen.isErr())
            return BindingResult<std::shared_ptr<const InvocationSnapshot>>{vernon::err(frozen.error())};
    }
    try {
        std::lock_guard lock(state_->mutex_);
        if (state_->executable_ != executable_ || state_->generation_ != generation_)
            return BindingResult<std::shared_ptr<const InvocationSnapshot>>{vernon::err(BindingError::StaleInstance)};
        BindingTelemetry candidateTelemetry = state_->telemetry_;
        candidateTelemetry.prepareCount += updates_.size();
        candidateTelemetry.reuseCount += reuseCount_;
        candidateTelemetry.uploadBytes += uploadBytes_;
        candidateTelemetry.uploadRanges += uploadRanges_;
        if (!updates_.empty()) {
            if (state_->snapshot_ == snapshot_)
                state_->snapshot_ = frozenSnapshot_;
            else {
                BindingMap entries = state_->snapshot_->entries_;
                for (const auto &[slot, entry] : updates_)
                    entries.insert_or_assign(slot, entry);
                state_->snapshot_ =
                    std::shared_ptr<const InvocationSnapshot>(new InvocationSnapshot(std::move(entries)));
            }
        }
        state_->telemetry_ = candidateTelemetry;
        finished_ = true;
        return BindingResult<std::shared_ptr<const InvocationSnapshot>>{vernon::ok(std::move(frozenSnapshot_))};
    } catch (...) {
        return BindingResult<std::shared_ptr<const InvocationSnapshot>>{vernon::err(BindingError::AllocationFailure)};
    }
}

void BindingTransaction::rollback() noexcept {
    if (finished_)
        return;
    try {
        std::lock_guard lock(state_->mutex_);
        ++state_->telemetry_.rollbackCount;
        finished_ = true;
    } catch (...) {
        vernon::resultContractViolation();
    }
}

PersistentBindingState::PersistentBindingState()
    : snapshot_(std::shared_ptr<const InvocationSnapshot>(new InvocationSnapshot({}))) {}

BindingResult<std::unique_ptr<BindingTransaction>> PersistentBindingState::begin(const void *executable) noexcept {
    if (!executable)
        return BindingResult<std::unique_ptr<BindingTransaction>>{vernon::err(BindingError::InvalidPayload)};
    try {
        std::lock_guard lock(mutex_);
        const bool replacement = executable_ != executable;
        const uint64_t transactionGeneration = replacement ? generation_ + 1 : generation_;
        std::shared_ptr<const InvocationSnapshot> snapshot = snapshot_;
        if (replacement)
            snapshot = std::shared_ptr<const InvocationSnapshot>(new InvocationSnapshot({}));
        auto transaction = std::unique_ptr<BindingTransaction>(
            new BindingTransaction(*this, executable, transactionGeneration, snapshot));
        if (replacement) {
            executable_ = executable;
            snapshot_ = std::move(snapshot);
            generation_ = transactionGeneration;
        }
        return BindingResult<std::unique_ptr<BindingTransaction>>{vernon::ok(std::move(transaction))};
    } catch (...) {
        return BindingResult<std::unique_ptr<BindingTransaction>>{vernon::err(BindingError::AllocationFailure)};
    }
}

void PersistentBindingState::clear() {
    std::lock_guard lock(mutex_);
    executable_ = nullptr;
    snapshot_ = std::shared_ptr<const InvocationSnapshot>(new InvocationSnapshot({}));
    ++generation_;
}

BindingTelemetry PersistentBindingState::telemetry() const {
    std::lock_guard lock(mutex_);
    return telemetry_;
}

BindingResult<std::unique_ptr<BindingTransaction>> ProgramInstance::beginInvocation() noexcept {
    return bindings_.begin(executable_);
}

BindingTelemetry ProgramInstance::telemetry() const { return bindings_.telemetry(); }

void ProgramInstance::clear() { bindings_.clear(); }

} // namespace vernon::runtime::program
