#ifndef VERNON_RUNTIME_AUTODIFF_HOST_EFFECT_TRANSACTION_H
#define VERNON_RUNTIME_AUTODIFF_HOST_EFFECT_TRANSACTION_H

#include "VernonResult.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace vernon::runtime::ad {

enum class HostEffectError {
    InvalidState,
    InvalidDestination,
};

class HostEffectTransaction {
public:
    explicit HostEffectTransaction(size_t outputSize) : outputSize_(outputSize) {}
    ~HostEffectTransaction() noexcept { discard(); }

    [[nodiscard]] Result<uint8_t *, HostEffectError> stageStorage(void *destination, size_t size, bool preserve,
                                                                  bool commit = true) {
        if (state_ != State::Capturing)
            return Result<uint8_t *, HostEffectError>{err(HostEffectError::InvalidState)};
        if (size && !destination)
            return Result<uint8_t *, HostEffectError>{err(HostEffectError::InvalidDestination)};
        StorageShadow shadow{destination, std::vector<uint8_t>(size), commit};
        if (preserve && size)
            std::memcpy(shadow.bytes.data(), destination, size);
        storage_.push_back(std::move(shadow));
        return Result<uint8_t *, HostEffectError>{ok(storage_.back().bytes.data())};
    }

    [[nodiscard]] Result<uint8_t *, HostEffectError> stagedOutput() {
        if (state_ != State::Capturing)
            return Result<uint8_t *, HostEffectError>{err(HostEffectError::InvalidState)};
        if (output_.empty() && outputSize_)
            output_.resize(outputSize_);
        return Result<uint8_t *, HostEffectError>{ok(output_.data())};
    }

    [[nodiscard]] Result<void, HostEffectError> commit(void *outputDestination) noexcept {
        if (state_ != State::Capturing)
            return Result<void, HostEffectError>{err(HostEffectError::InvalidState)};
        if (outputSize_ && !outputDestination)
            return Result<void, HostEffectError>{err(HostEffectError::InvalidDestination)};
        for (const StorageShadow &shadow : storage_)
            if (shadow.commit && !shadow.bytes.empty())
                std::memcpy(shadow.destination, shadow.bytes.data(), shadow.bytes.size());
        if (!output_.empty())
            std::memcpy(outputDestination, output_.data(), output_.size());
        state_ = State::Committed;
        return Result<void, HostEffectError>{ok()};
    }

    bool discard() noexcept {
        if (state_ != State::Capturing)
            return false;
        storage_.clear();
        output_.clear();
        state_ = State::Discarded;
        return true;
    }

private:
    enum class State { Capturing, Committed, Discarded };

    struct StorageShadow {
        void *destination{};
        std::vector<uint8_t> bytes;
        bool commit{};
    };

    std::vector<StorageShadow> storage_;
    std::vector<uint8_t> output_;
    size_t outputSize_{};
    State state_{State::Capturing};
};

} // namespace vernon::runtime::ad

#endif
