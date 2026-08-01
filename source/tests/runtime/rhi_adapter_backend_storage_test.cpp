#include "../../lib/runtime/rhi_adapter/adapter_common.h"

#include <gtest/gtest.h>

#include <memory>
#include <utility>

namespace {

struct TestBackendState {
    explicit TestBackendState(size_t &destructions) : destructions(destructions) {}
    size_t &destructions;
};

void destroyTestBackend(void *state) noexcept {
    auto *testState = static_cast<TestBackendState *>(state);
    ++testState->destructions;
    delete testState;
}

VernonStatus synchronizeTestBackend(void *, std::string &) noexcept { return VERNON_STATUS_OK; }
uint64_t testBackendIdentity(const void *state) noexcept { return reinterpret_cast<uintptr_t>(state); }
void invalidateTestBackend(void *) noexcept {}

const RhiAdapterBackendOps testBackendOps{destroyTestBackend, synchronizeTestBackend, testBackendIdentity,
                                          invalidateTestBackend};

} // namespace

TEST(RhiAdapterBackendStorage, RejectsIncompleteAndDuplicateState) {
    size_t destructions = 0;
    {
        RhiAdapterBackendStorage storage;
        auto state = std::make_unique<TestBackendState>(destructions);

        EXPECT_FALSE(storage.adopt(nullptr, &testBackendOps));
        EXPECT_FALSE(storage.adopt(state.get(), nullptr));
        ASSERT_TRUE(storage.adopt(state.get(), &testBackendOps));
        [[maybe_unused]] auto *adoptedState = state.release();

        auto duplicate = std::make_unique<TestBackendState>(destructions);
        EXPECT_FALSE(storage.adopt(duplicate.get(), &testBackendOps));
        EXPECT_EQ(destructions, 0u);
    }
    EXPECT_EQ(destructions, 1u);
}

TEST(RhiAdapterBackendStorage, AdapterDestructionDestroysAdoptedStateOnce) {
    size_t destructions = 0;
    {
        VernonRuntimeRhiAdapter adapter;
        auto state = std::make_unique<TestBackendState>(destructions);
        ASSERT_TRUE(adapter.backend.adopt(state.get(), &testBackendOps));
        [[maybe_unused]] auto *adoptedState = state.release();
    }
    EXPECT_EQ(destructions, 1u);
}

TEST(RhiAdapterBackendStorage, MoveConstructionTransfersOwnership) {
    size_t destructions = 0;
    {
        RhiAdapterBackendStorage source;
        auto state = std::make_unique<TestBackendState>(destructions);
        ASSERT_TRUE(source.adopt(state.get(), &testBackendOps));
        [[maybe_unused]] auto *adoptedState = state.release();

        RhiAdapterBackendStorage destination(std::move(source));
        EXPECT_EQ(source.state, nullptr);
        EXPECT_EQ(source.ops, nullptr);
        EXPECT_NE(destination.state, nullptr);
        EXPECT_EQ(destination.ops, &testBackendOps);
    }
    EXPECT_EQ(destructions, 1u);
}

TEST(RhiAdapterBackendStorage, MoveAssignmentReleasesPreviousState) {
    size_t destructions = 0;
    {
        RhiAdapterBackendStorage source;
        auto sourceState = std::make_unique<TestBackendState>(destructions);
        ASSERT_TRUE(source.adopt(sourceState.get(), &testBackendOps));
        [[maybe_unused]] auto *adoptedSourceState = sourceState.release();

        RhiAdapterBackendStorage destination;
        auto destinationState = std::make_unique<TestBackendState>(destructions);
        ASSERT_TRUE(destination.adopt(destinationState.get(), &testBackendOps));
        [[maybe_unused]] auto *adoptedDestinationState = destinationState.release();

        destination = std::move(source);
        EXPECT_EQ(destructions, 1u);
        EXPECT_EQ(source.state, nullptr);
        EXPECT_EQ(source.ops, nullptr);
    }
    EXPECT_EQ(destructions, 2u);
}
