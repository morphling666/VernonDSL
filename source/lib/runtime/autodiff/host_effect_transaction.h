#ifndef VERNON_RUNTIME_AUTODIFF_HOST_EFFECT_TRANSACTION_H
#define VERNON_RUNTIME_AUTODIFF_HOST_EFFECT_TRANSACTION_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace vernon::runtime::ad {

class HostEffectTransaction {
public:
    explicit HostEffectTransaction(size_t outputSize) : outputSize_(outputSize) {}
    ~HostEffectTransaction() { discard(); }

    uint8_t *stageStorage(void *destination, size_t size, bool preserve, bool commit = true) {
        if (state_ != State::Capturing)
            return nullptr;
        try {
            StorageShadow shadow{destination, std::vector<uint8_t>(size), commit};
            if (preserve && size)
                std::memcpy(shadow.bytes.data(), destination, size);
            storage_.push_back(std::move(shadow));
            return storage_.back().bytes.data();
        } catch (...) {
            discard();
            return nullptr;
        }
    }

    uint8_t *stagedOutput() {
        if (state_ != State::Capturing)
            return nullptr;
        try {
            if (output_.empty() && outputSize_)
                output_.resize(outputSize_);
            return output_.data();
        } catch (...) {
            discard();
            return nullptr;
        }
    }

    bool commit(void *outputDestination) {
        if (state_ != State::Capturing)
            return false;
        if (outputSize_ && !outputDestination) {
            discard();
            return false;
        }
        for (const StorageShadow &shadow : storage_)
            if (shadow.commit && !shadow.bytes.empty())
                std::memcpy(shadow.destination, shadow.bytes.data(), shadow.bytes.size());
        if (!output_.empty())
            std::memcpy(outputDestination, output_.data(), output_.size());
        state_ = State::Committed;
        return true;
    }

    bool discard() {
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
