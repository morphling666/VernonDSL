// The tally functions the generated C API calls, declared in generated/tally_api_impl.h.
#include "tally_api_impl.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <string>

namespace tally::api {

namespace {

// A frozen counter refuses what would change it, except a call marked `unfreezes` and storing a Marker.
vh::Status allowFrozen(const VhCall &call, const tally::Counter &counter) {
    if (counter.label() != "frozen" || call.effect < VH_EFFECT_WRITE ||
        vh_find_meta(call.meta, call.meta_count, "unfreezes") != nullptr ||
        (call.type != nullptr && std::strcmp(call.type->name, "Marker") == 0)) {
        return {};
    }
    return vh::fail(std::string(call.symbol) + ": the counter is frozen", 16);
}

} // namespace

vh::Status allow(const VhCall & /*call*/) { return {}; }

vh::Status allow(const VhCall &call, const tally::Counter &counter) { return allowFrozen(call, counter); }

vh::Status allow(const VhCall &call, const tally::Cursor &cursor) { return allowFrozen(call, cursor.counter()); }

vh::Result<std::unique_ptr<tally::Counter>> counter_create(TallyMode mode) {
    if (mode != TallyMode_SUM && mode != TallyMode_MAX) {
        return vh::fail("counter_create: unknown mode " + std::to_string(mode));
    }
    return std::make_unique<tally::Counter>(mode == TallyMode_SUM ? tally::Mode::Sum : tally::Mode::Max);
}

vh::Status counter_destroy(std::unique_ptr<tally::Counter> /*counter*/) { return {}; }

vh::Status counter_add(tally::Counter &counter, int32_t amount) {
    if (amount < 0) {
        return vh::fail("counter_add: amount " + std::to_string(amount) + " is negative", 10);
    }
    counter.add(amount);
    return {};
}

vh::Result<TallyStats> counter_stats(const tally::Counter &counter) {
    TallyStats stats{};
    stats.count = counter.count();
    stats.total = counter.total();
    for (size_t index = 0; index < counter.recent().size(); ++index) {
        stats.recent[index] = counter.recent()[index];
    }
    return stats;
}

vh::Result<int32_t> counter_label(tally::Counter &counter, std::string_view label) {
    counter.setLabel(std::string(label));
    return static_cast<int32_t>(counter.label().size());
}

tally::Range convert(const TallyRange &value) { return tally::Range{value.low, value.high}; }

TallyRange convert(const tally::Range &value) {
    return TallyRange{static_cast<int32_t>(value.low), static_cast<int32_t>(value.high)};
}

vh::Status counter_add_all(tally::Counter &counter, vh::Span<int32_t> amounts) {
    for (int32_t amount : amounts) {
        if (amount < 0) {
            return vh::fail("counter_add_all: amount " + std::to_string(amount) + " is negative", 10);
        }
    }
    for (int32_t amount : amounts) {
        counter.add(amount);
    }
    return {};
}

vh::Result<float> counter_add_weighted(tally::Counter &counter, vh::Span<vh::Entry<int32_t, tally::Sample>> samples) {
    float total = 0.0f;
    float position = 1.0f;
    for (const auto &entry : samples) {
        counter.add(entry.value.amount);
        total += static_cast<float>(entry.key) * entry.value.weight * position;
        position += 1.0f;
    }
    return total;
}

vh::Result<std::string> counter_name(const tally::Counter &counter) {
    if (counter.label().empty()) {
        return vh::fail("counter_name: the counter has no label", 11);
    }
    return counter.label();
}

vh::Result<std::vector<int32_t>> counter_history(const tally::Counter &counter) { return counter.history(); }

vh::Result<std::vector<vh::Entry<int32_t, int32_t>>> counter_histogram(const tally::Counter &counter) {
    std::vector<vh::Entry<int32_t, int32_t>> counts;
    for (int32_t amount : counter.history()) {
        counts.push_back({amount, 0});
    }
    // Every amount is listed once per occurrence, each with its running count: the caller keeps the last entry of
    // each key, in first-occurrence order.
    std::map<int32_t, int32_t> seen;
    for (auto &entry : counts) {
        entry.value = ++seen[entry.key];
    }
    return counts;
}

vh::Result<std::vector<tally::Sample>> counter_samples(const tally::Counter &counter) {
    std::vector<tally::Sample> samples;
    for (int32_t amount : counter.history()) {
        samples.push_back(tally::Sample{amount, static_cast<float>(amount) * 0.5f});
    }
    return samples;
}

vh::Result<tally::Range> counter_range(const tally::Counter &counter) {
    if (counter.history().empty()) {
        return vh::fail("counter_range: nothing was counted", 12);
    }
    tally::Range range{counter.history().front(), counter.history().front()};
    for (int32_t amount : counter.history()) {
        range.low = std::min<int64_t>(range.low, amount);
        range.high = std::max<int64_t>(range.high, amount);
    }
    return range;
}

vh::Result<int32_t> counter_in_range(const tally::Counter &counter, const tally::Range &range) {
    int32_t inside = 0;
    for (int32_t amount : counter.history()) {
        inside += amount >= range.low && amount <= range.high ? 1 : 0;
    }
    return inside;
}

vh::Status counter_store(tally::Counter &counter, const VhType &type, const void *value) {
    counter.stored()[type.name] = std::string(static_cast<const char *>(value), static_cast<size_t>(type.size));
    return {};
}

vh::Status counter_load(const tally::Counter &counter, const VhType &type, void *out) {
    auto found = counter.stored().find(type.name);
    if (found == counter.stored().end() || found->second.size() != type.size) {
        return vh::fail(std::string("counter_load: no ") + type.name + " is stored", 13);
    }
    std::memcpy(out, found->second.data(), found->second.size());
    return {};
}

vh::Result<bool> counter_has(const tally::Counter &counter, const VhType &type) {
    return counter.stored().count(type.name) != 0;
}

vh::Result<std::vector<tally::Range>> counter_clamp(const tally::Counter &counter, vh::Span<tally::Range> ranges) {
    auto bounds = counter_range(counter);
    if (!bounds.ok()) {
        return std::move(bounds.error());
    }
    std::vector<tally::Range> clamped;
    for (const tally::Range &range : ranges) {
        clamped.push_back(
            tally::Range{std::max(range.low, bounds.value().low), std::min(range.high, bounds.value().high)});
    }
    return clamped;
}

tally::Ticket convert(TallyTicket value) {
    return tally::Ticket{static_cast<uint32_t>(value.bits & UINT32_MAX), static_cast<uint32_t>(value.bits >> 32)};
}

TallyTicket convert(const tally::Ticket &value) {
    return TallyTicket{(static_cast<uint64_t>(value.serial) << 32) | value.position};
}

vh::Result<tally::Ticket> counter_ticket(const tally::Counter &counter) {
    if (counter.history().empty()) {
        return vh::fail("counter_ticket: nothing was counted", 12);
    }
    return tally::Ticket{static_cast<uint32_t>(counter.history().size() - 1), counter.serial()};
}

vh::Result<std::vector<tally::Ticket>> counter_tickets(const tally::Counter &counter) {
    std::vector<tally::Ticket> tickets;
    for (size_t position = 0; position < counter.history().size(); ++position) {
        tickets.push_back(tally::Ticket{static_cast<uint32_t>(position), counter.serial()});
    }
    return tickets;
}

vh::Result<int32_t> counter_redeem(const tally::Counter &counter, vh::Span<tally::Ticket> tickets) {
    int32_t sum = 0;
    for (const tally::Ticket &ticket : tickets) {
        if (ticket.serial != counter.serial() || ticket.position >= counter.history().size()) {
            return vh::fail("counter_redeem: ticket " + std::to_string(ticket.position) + " is not this counter's", 14);
        }
        sum += counter.history()[ticket.position];
    }
    return sum;
}

int32_t ticket_position(tally::Ticket ticket) { return static_cast<int32_t>(ticket.position); }

vh::Result<std::unique_ptr<tally::Cursor>> cursor_create(tally::Counter &counter) {
    return std::make_unique<tally::Cursor>(counter);
}

vh::Status cursor_destroy(std::unique_ptr<tally::Cursor> /*cursor*/) { return {}; }

vh::Result<int32_t> cursor_next(tally::Cursor &cursor) {
    const std::vector<int32_t> &history = cursor.counter().history();
    if (cursor.position() >= history.size()) {
        return vh::fail("cursor_next: the cursor is past the history", 15);
    }
    return history[cursor.position()++];
}

vh::Result<int32_t> live_cursors() { return tally::Cursor::live(); }

vh::Result<bool> counter_exists(uint64_t bits) { return handles::find(TallyCounter{bits}) != nullptr; }

vh::Result<uint64_t> cursor_counter_bits(const tally::Cursor &cursor) {
    return handles::handle_of(cursor.counter()).bits;
}

vh::Status counter_destroy_all() { return handles::destroy_all<TallyCounter>(); }

float scaled_total(const TallyStats &stats, float factor) { return static_cast<float>(stats.total) * factor; }

vh::Result<int32_t> live_counters() { return tally::Counter::live(); }

} // namespace tally::api
