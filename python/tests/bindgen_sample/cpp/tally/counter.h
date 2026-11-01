// tally: a small C++ library standing in for an existing one. It knows nothing of Host code; tally_api.cpp adapts it
// to the functions vernon_dsl.bindgen generated from tally.py.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace tally {

enum class Mode { Sum, Max };

// An amount and its weight, laid out as the Host Struct Sample is.
struct Sample {
    int32_t amount = 0;
    float weight = 0.0f;
};

// Inclusive bounds, wider than the Host Struct Range holds them.
struct Range {
    int64_t low = 0;
    int64_t high = 0;
};

// A position in a Counter's history, with the counter's serial so tickets of different counters differ.
struct Ticket {
    uint32_t position = 0;
    uint32_t serial = 0;
};

class Counter {
public:
    explicit Counter(Mode mode);
    ~Counter();
    Counter(const Counter &) = delete;
    Counter &operator=(const Counter &) = delete;

    void add(int32_t amount);
    int32_t count() const { return count_; }
    int64_t total() const { return total_; }
    // The last three amounts, newest first.
    const std::array<float, 3> &recent() const { return recent_; }
    // Every amount, oldest first.
    const std::vector<int32_t> &history() const { return history_; }
    const std::string &label() const { return label_; }
    void setLabel(std::string label) { label_ = std::move(label); }
    // Opaque values kept with the counter, by name.
    std::map<std::string, std::string> &stored() { return stored_; }
    const std::map<std::string, std::string> &stored() const { return stored_; }

    uint32_t serial() const { return serial_; }

    static int32_t live();

private:
    uint32_t serial_;
    Mode mode_;
    int32_t count_ = 0;
    int64_t total_ = 0;
    std::array<float, 3> recent_{};
    std::vector<int32_t> history_;
    std::string label_;
    std::map<std::string, std::string> stored_;
};

// Reads a Counter's history in order; it must not outlive the counter.
class Cursor {
public:
    explicit Cursor(const Counter &counter);
    ~Cursor();
    Cursor(const Cursor &) = delete;
    Cursor &operator=(const Cursor &) = delete;

    const Counter &counter() const { return counter_; }
    size_t &position() { return position_; }

    static int32_t live();

private:
    const Counter &counter_;
    size_t position_ = 0;
};

} // namespace tally
