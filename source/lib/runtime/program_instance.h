#ifndef VERNON_RUNTIME_PROGRAM_INSTANCE_H
#define VERNON_RUNTIME_PROGRAM_INSTANCE_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace vernon::runtime::program {

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

class InvocationSnapshot {
public:
    const std::shared_ptr<void> *find(uint32_t slot) const;

private:
    friend class BindingTransaction;
    explicit InvocationSnapshot(std::unordered_map<uint32_t, BindingEntry> entries) : entries_(std::move(entries)) {}

    std::unordered_map<uint32_t, BindingEntry> entries_;
};

class PersistentBindingState;

class BindingTransaction {
public:
    ~BindingTransaction();
    BindingTransaction(const BindingTransaction &) = delete;
    BindingTransaction &operator=(const BindingTransaction &) = delete;

    const std::shared_ptr<void> *find(uint32_t slot, const std::string &token);
    void observeUploads(uint64_t uploadBytes, uint64_t uploadRanges);
    void stage(uint32_t slot, std::string token, std::shared_ptr<void> payload, uint64_t uploadBytes,
               uint64_t uploadRanges);
    std::shared_ptr<const InvocationSnapshot> snapshot() const;
    std::shared_ptr<const InvocationSnapshot> commit();
    void rollback();

private:
    friend class PersistentBindingState;
    BindingTransaction(PersistentBindingState &state, const void *executable, uint64_t generation,
                       std::unordered_map<uint32_t, BindingEntry> snapshot);

    PersistentBindingState *state_;
    const void *executable_;
    uint64_t generation_;
    std::unordered_map<uint32_t, BindingEntry> snapshot_;
    std::unordered_map<uint32_t, BindingEntry> updates_;
    uint64_t reuseCount_{};
    uint64_t uploadBytes_{};
    uint64_t uploadRanges_{};
    bool finished_{};
};

class PersistentBindingState {
public:
    std::unique_ptr<BindingTransaction> begin(const void *executable);
    void clear();
    BindingTelemetry telemetry() const;

private:
    friend class BindingTransaction;

    mutable std::mutex mutex_;
    const void *executable_{};
    uint64_t generation_{};
    std::unordered_map<uint32_t, BindingEntry> entries_;
    BindingTelemetry telemetry_;
};

class ProgramInstance {
public:
    explicit ProgramInstance(const void *executable) : executable_(executable) {}
    ProgramInstance(const ProgramInstance &) = delete;
    ProgramInstance &operator=(const ProgramInstance &) = delete;

    std::unique_ptr<BindingTransaction> beginInvocation();
    BindingTelemetry telemetry() const;
    void clear();

private:
    const void *executable_;
    PersistentBindingState bindings_;
};

} // namespace vernon::runtime::program

#endif
