#ifndef VERNON_RUNTIME_PROGRAM_INSTANCE_H
#define VERNON_RUNTIME_PROGRAM_INSTANCE_H

#include "VernonResult.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vernon::runtime::program {

enum class BindingError {
    InvalidState,
    InvalidPayload,
    StaleInstance,
    AllocationFailure,
};

template <typename T> using BindingResult = vernon::Result<T, BindingError>;

struct BindingTelemetry {
    uint64_t prepareCount{};
    uint64_t reuseCount{};
    uint64_t rollbackCount{};
    uint64_t uploadBytes{};
    uint64_t uploadRanges{};
};

struct BindingEntry {
    std::string token;
    std::shared_ptr<void> payload;
};

using BindingMap = std::unordered_map<uint64_t, BindingEntry>;

class InvocationSnapshot {
public:
    const std::shared_ptr<void> *find(uint64_t slot) const;

private:
    friend class BindingTransaction;
    friend class PersistentBindingState;
    explicit InvocationSnapshot(BindingMap entries) : entries_(std::move(entries)) {}

    BindingMap entries_;
};

class PersistentBindingState;

class BindingTransaction {
public:
    ~BindingTransaction() noexcept;
    BindingTransaction(const BindingTransaction &) = delete;
    BindingTransaction &operator=(const BindingTransaction &) = delete;

    BindingResult<bool> matches(uint64_t slot, std::string_view token) const noexcept;
    BindingResult<void> observeReuses(uint64_t count) noexcept;
    BindingResult<void> observeUploads(uint64_t uploadBytes, uint64_t uploadRanges) noexcept;
    BindingResult<void> stage(uint64_t slot, std::string token, std::shared_ptr<void> payload, uint64_t uploadBytes,
                              uint64_t uploadRanges) noexcept;
    BindingResult<void> stageMany(std::vector<std::pair<uint64_t, BindingEntry>> entries, uint64_t uploadBytes,
                                  uint64_t uploadRanges) noexcept;
    BindingResult<std::shared_ptr<const InvocationSnapshot>> freeze() noexcept;
    BindingResult<std::shared_ptr<const InvocationSnapshot>> commit() noexcept;
    void rollback() noexcept;

private:
    friend class PersistentBindingState;
    BindingTransaction(PersistentBindingState &state, const void *executable, uint64_t generation,
                       std::shared_ptr<const InvocationSnapshot> snapshot);

    PersistentBindingState *state_;
    const void *executable_;
    uint64_t generation_;
    std::shared_ptr<const InvocationSnapshot> snapshot_;
    BindingMap updates_;
    uint64_t reuseCount_{};
    uint64_t uploadBytes_{};
    uint64_t uploadRanges_{};
    std::shared_ptr<const InvocationSnapshot> frozenSnapshot_;
    bool frozen_{};
    bool finished_{};
};

class PersistentBindingState {
public:
    PersistentBindingState();
    BindingResult<std::unique_ptr<BindingTransaction>> begin(const void *executable) noexcept;
    void clear();
    BindingTelemetry telemetry() const;

private:
    friend class BindingTransaction;

    mutable std::mutex mutex_;
    const void *executable_{};
    uint64_t generation_{};
    std::shared_ptr<const InvocationSnapshot> snapshot_;
    BindingTelemetry telemetry_;
};

class ProgramInstance {
public:
    explicit ProgramInstance(const void *executable) : executable_(executable) {}
    ProgramInstance(const ProgramInstance &) = delete;
    ProgramInstance &operator=(const ProgramInstance &) = delete;

    BindingResult<std::unique_ptr<BindingTransaction>> beginInvocation() noexcept;
    BindingTelemetry telemetry() const;
    void clear();

private:
    const void *executable_;
    PersistentBindingState bindings_;
};

} // namespace vernon::runtime::program

#endif
