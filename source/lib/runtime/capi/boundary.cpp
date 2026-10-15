#include "runtime/capi/boundary.h"
#include "VernonError.hpp"
#include "runtime/runtime_state.h"
#include "runtime/runtime_test_hooks.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <string_view>

namespace {

struct RuntimeDiagnosticState {
    const VernonRuntimeContext *context{};
    uint64_t generation{};
    std::string pending;
    std::string published;
    std::array<char, 192> emergency{};
    size_t emergencySize{};
    size_t depth{};
    uint64_t lastUse{};
    bool retired{};
};

struct RuntimeDiagnosticOverflowEntry {
    RuntimeDiagnosticState state;
    std::unique_ptr<RuntimeDiagnosticOverflowEntry> next;
};

struct RuntimeDiagnosticTable {
    std::array<RuntimeDiagnosticState, 16> entries{};
    std::unique_ptr<RuntimeDiagnosticOverflowEntry> overflow;
    RuntimeDiagnosticState emergency;
    RuntimeDiagnosticState discard;
    uint64_t clock{};
};
thread_local RuntimeDiagnosticTable invocationDiagnostics;
thread_local bool failNextDiagnosticOverflowAllocation;
std::atomic<uint64_t> nextDiagnosticGeneration{1};

void touchDiagnosticState(RuntimeDiagnosticState &state) noexcept {
    if (invocationDiagnostics.clock != std::numeric_limits<uint64_t>::max())
        ++invocationDiagnostics.clock;
    state.lastUse = invocationDiagnostics.clock;
}

RuntimeDiagnosticState *findDiagnosticState(const VernonRuntimeContext *context, uint64_t generation) noexcept {
    for (RuntimeDiagnosticState &state : invocationDiagnostics.entries) {
        if (state.context == context && state.generation == generation) {
            touchDiagnosticState(state);
            return &state;
        }
    }
    for (RuntimeDiagnosticOverflowEntry *entry = invocationDiagnostics.overflow.get(); entry;
         entry = entry->next.get()) {
        if (entry->state.context == context && entry->state.generation == generation) {
            touchDiagnosticState(entry->state);
            return &entry->state;
        }
    }
    RuntimeDiagnosticState &emergency = invocationDiagnostics.emergency;
    if (emergency.context == context && emergency.generation == generation) {
        touchDiagnosticState(emergency);
        return &emergency;
    }
    return nullptr;
}

RuntimeDiagnosticState *findDiagnosticState(const VernonRuntimeContext &context) noexcept {
    return findDiagnosticState(&context, context.diagnosticGeneration);
}

void initializeDiagnosticState(RuntimeDiagnosticState &state, VernonRuntimeContext &context) noexcept {
    state.context = &context;
    state.generation = context.diagnosticGeneration;
    state.pending.clear();
    state.published.clear();
    state.emergencySize = 0;
    state.depth = 0;
    state.retired = false;
    touchDiagnosticState(state);
}

RuntimeDiagnosticState *tryDiagnosticState(VernonRuntimeContext &context) noexcept {
    if (RuntimeDiagnosticState *state = findDiagnosticState(context))
        return state;
    RuntimeDiagnosticState *candidate = &invocationDiagnostics.entries.front();
    for (RuntimeDiagnosticState &state : invocationDiagnostics.entries) {
        if (!state.context) {
            candidate = &state;
            break;
        }
        if (!state.depth && (candidate->depth || state.lastUse < candidate->lastUse))
            candidate = &state;
    }
    if (!candidate->depth) {
        initializeDiagnosticState(*candidate, context);
        return candidate;
    }

    RuntimeDiagnosticOverflowEntry *last = nullptr;
    for (RuntimeDiagnosticOverflowEntry *entry = invocationDiagnostics.overflow.get(); entry;
         entry = entry->next.get()) {
        if (!entry->state.depth) {
            initializeDiagnosticState(entry->state, context);
            return &entry->state;
        }
        last = entry;
    }

    std::unique_ptr<RuntimeDiagnosticOverflowEntry> added;
    if (failNextDiagnosticOverflowAllocation)
        failNextDiagnosticOverflowAllocation = false;
    else
        added.reset(new (std::nothrow) RuntimeDiagnosticOverflowEntry);
    if (added) {
        RuntimeDiagnosticOverflowEntry *result = added.get();
        if (last)
            last->next = std::move(added);
        else
            invocationDiagnostics.overflow = std::move(added);
        initializeDiagnosticState(result->state, context);
        return &result->state;
    }

    RuntimeDiagnosticState &emergency = invocationDiagnostics.emergency;
    if (!emergency.depth) {
        initializeDiagnosticState(emergency, context);
        return &emergency;
    }
    return nullptr;
}

RuntimeDiagnosticState &diagnosticState(VernonRuntimeContext &context) noexcept {
    if (RuntimeDiagnosticState *state = tryDiagnosticState(context))
        return *state;
    return invocationDiagnostics.discard;
}

} // namespace

std::string &vernon::runtime::invocationDiagnostic(VernonRuntimeContext &context) {
    RuntimeDiagnosticState &state = diagnosticState(context);
    return state.depth ? state.pending : state.published;
}

const std::string *vernon::runtime::currentInvocationDiagnostic(const VernonRuntimeContext &context) {
    RuntimeDiagnosticState *state = findDiagnosticState(context);
    if (!state || state->emergencySize)
        return nullptr;
    return state->depth ? &state->pending : &state->published;
}

VernonStringView vernon::runtime::currentInvocationDiagnosticView(const VernonRuntimeContext &context) noexcept {
    RuntimeDiagnosticState *state = findDiagnosticState(context);
    if (!state)
        return {nullptr, 0};
    if (state->emergencySize)
        return {state->emergency.data(), state->emergencySize};
    const std::string &diagnostic = state->depth ? state->pending : state->published;
    return {diagnostic.data(), diagnostic.size()};
}

void vernon::runtime::clearInvocationDiagnostic(const VernonRuntimeContext &context) noexcept {
    RuntimeDiagnosticState *state = findDiagnosticState(context);
    if (state && state->depth)
        state->retired = true;
    else if (state)
        *state = {};
}

uint64_t vernon::runtime::diagnosticClockForTesting() noexcept { return invocationDiagnostics.clock; }

void vernon::runtime::setDiagnosticClockForTesting(uint64_t clock) noexcept {
    invocationDiagnostics.clock = clock;
    for (RuntimeDiagnosticState &state : invocationDiagnostics.entries)
        state.lastUse = clock;
    for (RuntimeDiagnosticOverflowEntry *entry = invocationDiagnostics.overflow.get(); entry; entry = entry->next.get())
        entry->state.lastUse = clock;
    invocationDiagnostics.emergency.lastUse = clock;
}

void vernon::runtime::failNextDiagnosticOverflowAllocationForTesting() noexcept {
    failNextDiagnosticOverflowAllocation = true;
}

uint64_t vernon::runtime::diagnosticGenerationCounterForTesting() noexcept {
    return nextDiagnosticGeneration.load(std::memory_order_relaxed);
}

void vernon::runtime::setDiagnosticGenerationCounterForTesting(uint64_t generation) noexcept {
    nextDiagnosticGeneration.store(generation, std::memory_order_relaxed);
}

vernon::runtime::RuntimeDiagnosticScope::RuntimeDiagnosticScope(const VernonRuntimeContext *context) noexcept
    : context_(context), generation_(context ? context->diagnosticGeneration : 0) {
    if (!context_)
        return;
    RuntimeDiagnosticState *state = tryDiagnosticState(*const_cast<VernonRuntimeContext *>(context_));
    state_ = state;
    if (state && state->depth++ == 0) {
        state->pending.clear();
        state->emergencySize = 0;
    }
}

vernon::runtime::RuntimeDiagnosticScope::~RuntimeDiagnosticScope() noexcept {
    auto *state = static_cast<RuntimeDiagnosticState *>(state_);
    if (!state || state->context != context_ || state->generation != generation_)
        return;
    if (!state->depth)
        return;
    if (--state->depth != 0)
        return;
    if (state->retired)
        *state = {};
    else if (!state->emergencySize)
        state->published = std::move(state->pending);
}

namespace vernon::runtime::capi {

vernon::Result<uint64_t, vernon::RuntimeError> reserveDiagnosticGeneration() noexcept {
    using Result = vernon::Result<uint64_t, vernon::RuntimeError>;
    uint64_t generation = nextDiagnosticGeneration.load(std::memory_order_relaxed);
    do {
        if (generation == std::numeric_limits<uint64_t>::max())
            return Result{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::ResourceExhausted,
                                                           {"runtime diagnostic generation exhausted", 0, 0}})};
    } while (!nextDiagnosticGeneration.compare_exchange_weak(generation, generation + 1, std::memory_order_relaxed,
                                                             std::memory_order_relaxed));
    return Result{vernon::ok(generation)};
}

VernonStatus fail(VernonRuntimeContext *context, std::string_view error, VernonStatus status) noexcept {
    if (!context)
        return status;
    RuntimeDiagnosticState *state = tryDiagnosticState(*context);
    if (!state)
        return status;
    try {
        (state->depth ? state->pending : state->published).assign(error);
        state->emergencySize = 0;
    } catch (...) {
        constexpr std::string_view fallback = "Runtime diagnostic allocation failed";
        state->emergencySize = std::min(fallback.size(), state->emergency.size() - 1);
        std::memcpy(state->emergency.data(), fallback.data(), state->emergencySize);
        state->emergency[state->emergencySize] = '\0';
    }
    return status;
}

VernonStatus publishRuntimeError(VernonRuntimeContext *context, vernon::RuntimeError error) noexcept {
    if (context) {
        RuntimeDiagnosticState *state = tryDiagnosticState(*context);
        if (!state)
            return vernon::toVernonStatus(error);
        try {
            (state->depth ? state->pending : state->published)
                .assign(error.context.operation ? error.context.operation : "Runtime operation failed");
            state->emergencySize = 0;
        } catch (...) {
            const size_t required =
                vernon::renderEmergencyDiagnostic(error, state->emergency.data(), state->emergency.size());
            state->emergencySize = std::min(required, state->emergency.size() - 1);
        }
    }
    return vernon::toVernonStatus(error);
}

VernonRuntimeOperationStatus runtimeOperationStatus(vernon::RuntimeErrorCode code) noexcept {
    return code == vernon::RuntimeErrorCode::NotReady ? VERNON_RUNTIME_OPERATION_INVALID_ARGUMENT
                                                      : static_cast<VernonRuntimeOperationStatus>(code);
}

} // namespace vernon::runtime::capi
