#include "tally/counter.h"

#include <algorithm>
#include <atomic>

namespace tally {

namespace {
std::atomic<int32_t> live_counters{0};
std::atomic<int32_t> live_cursors{0};
std::atomic<uint32_t> next_serial{1};
} // namespace

Counter::Counter(Mode mode) : serial_(next_serial++), mode_(mode) { ++live_counters; }

Counter::~Counter() { --live_counters; }

void Counter::add(int32_t amount) {
    ++count_;
    total_ = mode_ == Mode::Sum ? total_ + amount : std::max<int64_t>(total_, amount);
    recent_ = {static_cast<float>(amount), recent_[0], recent_[1]};
    history_.push_back(amount);
}

int32_t Counter::live() { return live_counters.load(); }

Cursor::Cursor(const Counter &counter) : counter_(counter) { ++live_cursors; }

Cursor::~Cursor() { --live_cursors; }

int32_t Cursor::live() { return live_cursors.load(); }

} // namespace tally
