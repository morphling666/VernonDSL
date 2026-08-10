#include "runtime/cpu_workgroup_dispatch.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

class CpuWorkgroupDispatchTest : public ::testing::Test {
protected:
    void SetUp() override {
        scheduler_ = vernon::runtime::CpuWorkgroupScheduler::create({8, 16}, error_);
        ASSERT_NE(scheduler_, nullptr) << error_;
    }

    std::string error_;
    std::unique_ptr<vernon::runtime::CpuWorkgroupScheduler> scheduler_;
};

TEST_F(CpuWorkgroupDispatchTest, VisitsEveryIndependentInvocationExactlyOnceAndReusesWorkers) {
    constexpr uint32_t grid[3]{7, 5, 3};
    constexpr uint32_t workgroup[3]{1, 1, 1};
    std::array<std::atomic<uint32_t>, 7 * 5 * 3> visits{};
    constexpr size_t expectedWorkers = 8;
    std::array<std::set<std::thread::id>, 2> launchThreads;

    for (size_t launch = 0; launch < launchThreads.size(); ++launch) {
        std::mutex threadsMutex;
        std::condition_variable workersReady;
        size_t waitingWorkers = 0;
        bool releaseWorkers = false;
        bool rendezvousTimedOut = false;
        ASSERT_EQ(scheduler_->dispatch(grid, workgroup,
                                       [&](const vernon::runtime::CpuLaneCoordinates &lane) {
                                           EXPECT_EQ(lane.local[0], 0u);
                                           EXPECT_EQ(lane.local[1], 0u);
                                           EXPECT_EQ(lane.local[2], 0u);
                                           EXPECT_EQ(lane.group[0], lane.global[0]);
                                           EXPECT_EQ(lane.group[1], lane.global[1]);
                                           EXPECT_EQ(lane.group[2], lane.global[2]);
                                           visits[lane.linearIndex].fetch_add(1, std::memory_order_relaxed);
                                           std::unique_lock lock(threadsMutex);
                                           launchThreads[launch].insert(std::this_thread::get_id());
                                           if (waitingWorkers < expectedWorkers) {
                                               ++waitingWorkers;
                                               if (waitingWorkers == expectedWorkers) {
                                                   releaseWorkers = true;
                                                   workersReady.notify_all();
                                               } else if (!workersReady.wait_for(lock, std::chrono::seconds(5),
                                                                                 [&] { return releaseWorkers; })) {
                                                   rendezvousTimedOut = true;
                                                   releaseWorkers = true;
                                                   workersReady.notify_all();
                                               }
                                           }
                                           return VERNON_STATUS_OK;
                                       }),
                  VERNON_STATUS_OK)
            << scheduler_->lastDiagnostic();
        EXPECT_FALSE(rendezvousTimedOut);
        EXPECT_EQ(launchThreads[launch].size(), expectedWorkers);
    }

    for (const std::atomic<uint32_t> &visit : visits)
        EXPECT_EQ(visit.load(std::memory_order_relaxed), 2u);
    EXPECT_EQ(launchThreads[0], launchThreads[1]);
    EXPECT_EQ(scheduler_->workerCount(), expectedWorkers);
}

TEST_F(CpuWorkgroupDispatchTest, IsolatesSharedStorageAndSynchronizesLanes) {
    constexpr uint32_t grid[3]{2, 1, 1};
    constexpr uint32_t workgroup[3]{4, 1, 1};
    std::array<int32_t, 10> result{};

    ASSERT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [&](const vernon::runtime::CpuLaneCoordinates &lane) {
                                       const uint64_t address = vernonCpuWorkgroupAddressV1(
                                           0, sizeof(std::atomic<int32_t>), alignof(std::atomic<int32_t>), 0);
                                       if (!address)
                                           return VERNON_STATUS_INTERNAL_ERROR;
                                       auto *shared = reinterpret_cast<std::atomic<int32_t> *>(address);
                                       if (lane.local[0] == 0)
                                           new (shared) std::atomic<int32_t>(static_cast<int32_t>(lane.group[0] * 100));
                                       vernonCpuWorkgroupBarrierV1(0);
                                       result[lane.group[0] * 4 + lane.local[0]] =
                                           shared->fetch_add(1, std::memory_order_relaxed);
                                       vernonCpuWorkgroupBarrierV1(1);
                                       if (lane.local[0] == 0)
                                           result[8 + lane.group[0]] = shared->load(std::memory_order_relaxed);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_OK);

    for (size_t group = 0; group < 2; ++group) {
        std::array<int32_t, 4> previous{result[group * 4], result[group * 4 + 1], result[group * 4 + 2],
                                        result[group * 4 + 3]};
        std::sort(previous.begin(), previous.end());
        const int32_t base = static_cast<int32_t>(group * 100);
        EXPECT_EQ(previous, (std::array<int32_t, 4>{base, base + 1, base + 2, base + 3}));
        EXPECT_EQ(result[8 + group], base + 4);
    }
}

TEST_F(CpuWorkgroupDispatchTest, SupportsRepeatedBarrierGenerations) {
    constexpr uint32_t grid[3]{2, 1, 1};
    constexpr uint32_t workgroup[3]{4, 1, 1};
    std::array<std::atomic<uint32_t>, 2> generations{};

    ASSERT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [&](const vernon::runtime::CpuLaneCoordinates &lane) {
                                       for (uint64_t generation = 0; generation < 32; ++generation) {
                                           if (lane.local[0] == generation % workgroup[0])
                                               generations[lane.group[0]].fetch_add(1, std::memory_order_relaxed);
                                           vernonCpuWorkgroupBarrierV1(generation);
                                           if (generations[lane.group[0]].load(std::memory_order_relaxed) !=
                                               generation + 1)
                                               return VERNON_STATUS_INTERNAL_ERROR;
                                           vernonCpuWorkgroupBarrierV1(32 + generation);
                                       }
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_OK)
        << scheduler_->lastDiagnostic();
    EXPECT_EQ(generations[0].load(std::memory_order_relaxed), 32u);
    EXPECT_EQ(generations[1].load(std::memory_order_relaxed), 32u);
}

TEST_F(CpuWorkgroupDispatchTest, ReusesBoundedWorkersForParallelCooperativeTeams) {
    constexpr uint32_t grid[3]{5, 1, 1};
    constexpr uint32_t workgroup[3]{4, 1, 1};
    std::array<std::set<std::thread::id>, 2> launchThreads;
    std::mutex threadsMutex;

    for (size_t launch = 0; launch < launchThreads.size(); ++launch) {
        ASSERT_EQ(scheduler_->dispatch(grid, workgroup,
                                       [&](const vernon::runtime::CpuLaneCoordinates &) {
                                           {
                                               std::lock_guard lock(threadsMutex);
                                               launchThreads[launch].insert(std::this_thread::get_id());
                                           }
                                           vernonCpuWorkgroupBarrierV1(0);
                                           return VERNON_STATUS_OK;
                                       }),
                  VERNON_STATUS_OK)
            << scheduler_->lastDiagnostic();
        EXPECT_EQ(launchThreads[launch].size(), 8u);
    }

    EXPECT_EQ(launchThreads[0], launchThreads[1]);
    EXPECT_EQ(scheduler_->workerCount(), 8u);
}

TEST_F(CpuWorkgroupDispatchTest, RunsConcurrentDispatchesWithinOneWorkerBudget) {
    constexpr uint32_t grid[3]{1, 1, 1};
    constexpr uint32_t workgroup[3]{4, 1, 1};
    std::mutex rendezvousMutex;
    std::condition_variable rendezvous;
    size_t arrived = 0;
    bool release = false;
    bool timedOut = false;
    std::array<VernonStatus, 2> statuses{VERNON_STATUS_INTERNAL_ERROR, VERNON_STATUS_INTERNAL_ERROR};
    std::array<std::thread, 2> callers;

    for (size_t dispatch = 0; dispatch < callers.size(); ++dispatch) {
        callers[dispatch] = std::thread([&, dispatch] {
            statuses[dispatch] =
                scheduler_->dispatch(grid, workgroup, [&](const vernon::runtime::CpuLaneCoordinates &) {
                    std::unique_lock lock(rendezvousMutex);
                    ++arrived;
                    if (arrived == 8) {
                        release = true;
                        rendezvous.notify_all();
                    } else if (!rendezvous.wait_for(lock, std::chrono::seconds(5), [&] { return release; })) {
                        timedOut = true;
                        release = true;
                        rendezvous.notify_all();
                    }
                    return VERNON_STATUS_OK;
                });
        });
    }
    for (std::thread &caller : callers)
        caller.join();

    EXPECT_FALSE(timedOut);
    EXPECT_EQ(arrived, 8u);
    EXPECT_EQ(statuses[0], VERNON_STATUS_OK);
    EXPECT_EQ(statuses[1], VERNON_STATUS_OK);
    EXPECT_EQ(scheduler_->workerCount(), 8u);
}

TEST_F(CpuWorkgroupDispatchTest, PropagatesLaneFailureWithoutDeadlockingBarriers) {
    constexpr uint32_t grid[3]{1, 1, 1};
    constexpr uint32_t workgroup[3]{4, 1, 1};

    EXPECT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [](const vernon::runtime::CpuLaneCoordinates &lane) {
                                       if (lane.local[0] == 0)
                                           return VERNON_STATUS_INTERNAL_ERROR;
                                       vernonCpuWorkgroupBarrierV1(0);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_NE(scheduler_->lastDiagnostic().find("CPU lane returned an error"), std::string::npos);
    EXPECT_NE(scheduler_->lastDiagnostic().find("workgroup (0,0,0)"), std::string::npos);
}

TEST_F(CpuWorkgroupDispatchTest, RejectsDivergentBarrierSites) {
    constexpr uint32_t grid[3]{1, 1, 1};
    constexpr uint32_t workgroup[3]{2, 1, 1};
    EXPECT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [](const vernon::runtime::CpuLaneCoordinates &lane) {
                                       vernonCpuWorkgroupBarrierV1(lane.local[0]);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_NE(scheduler_->lastDiagnostic().find("barrier site mismatch"), std::string::npos);
    EXPECT_NE(scheduler_->lastDiagnostic().find("local lane"), std::string::npos);
}

TEST_F(CpuWorkgroupDispatchTest, DetectsLaneCompletionBeforeBarrier) {
    constexpr uint32_t grid[3]{1, 1, 1};
    constexpr uint32_t workgroup[3]{2, 1, 1};
    EXPECT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [](const vernon::runtime::CpuLaneCoordinates &lane) {
                                       if (lane.local[0])
                                           vernonCpuWorkgroupBarrierV1(0);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_NE(scheduler_->lastDiagnostic().find("barrier site 0"), std::string::npos);
}

TEST_F(CpuWorkgroupDispatchTest, ContainsLaneExceptions) {
    constexpr uint32_t grid[3]{1, 1, 1};
    constexpr uint32_t workgroup[3]{2, 1, 1};
    EXPECT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [](const vernon::runtime::CpuLaneCoordinates &lane) {
                                       if (!lane.local[0])
                                           throw std::runtime_error("lane failure");
                                       vernonCpuWorkgroupBarrierV1(0);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_NE(scheduler_->lastDiagnostic().find("lane failure"), std::string::npos);
    EXPECT_NE(scheduler_->lastDiagnostic().find("local lane"), std::string::npos);
}

TEST_F(CpuWorkgroupDispatchTest, ReportsWorkgroupArenaOverflowWithoutReturningPoisonAddress) {
    constexpr uint32_t grid[3]{1, 1, 1};
    constexpr uint32_t workgroup[3]{2, 1, 1};
    EXPECT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [](const vernon::runtime::CpuLaneCoordinates &) {
                                       EXPECT_EQ(vernonCpuWorkgroupAddressV1(0, 16 * 1024 + 1, 4, 0), 0u);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_INTERNAL_ERROR);
}

TEST_F(CpuWorkgroupDispatchTest, AllocationPreflightFailureStopsLanesBeforeExternalWrites) {
    constexpr uint32_t grid[3]{1, 1, 1};
    constexpr uint32_t workgroup[3]{2, 1, 1};
    std::atomic<uint32_t> externalWrites{};

    EXPECT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [&](const vernon::runtime::CpuLaneCoordinates &) {
                                       (void)vernonCpuWorkgroupAddressV1(0, 16 * 1024 + 1, 16, 0);
                                       vernonCpuWorkgroupBarrierV1(std::numeric_limits<uint64_t>::max());
                                       externalWrites.fetch_add(1, std::memory_order_relaxed);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(externalWrites.load(std::memory_order_relaxed), 0u);
    EXPECT_NE(scheduler_->lastDiagnostic().find("16 KiB"), std::string::npos);
}

TEST_F(CpuWorkgroupDispatchTest, SeparatesPrimalAndPullbackSharedStorageBudgets) {
    constexpr uint32_t grid[3]{1, 1, 1};
    constexpr uint32_t workgroup[3]{2, 1, 1};
    constexpr uint64_t pullbackSite = uint64_t{1} << 63;

    ASSERT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [](const vernon::runtime::CpuLaneCoordinates &) {
                                       const uint64_t primal = vernonCpuWorkgroupAddressV1(0, 16 * 1024, 16, 0);
                                       const uint64_t pullback =
                                           vernonCpuWorkgroupAddressV1(pullbackSite, 16 * 1024, 16, 0);
                                       if (!primal || !pullback || primal == pullback)
                                           return VERNON_STATUS_INTERNAL_ERROR;
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_OK)
        << scheduler_->lastDiagnostic();
}

TEST_F(CpuWorkgroupDispatchTest, PullbackSharedAdjointsAreNotLimitedByThePrimalArena) {
    constexpr uint32_t grid[3]{1, 1, 1};
    constexpr uint32_t workgroup[3]{2, 1, 1};
    constexpr uint64_t pullbackSite = uint64_t{1} << 63;

    ASSERT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [](const vernon::runtime::CpuLaneCoordinates &) {
                                       if (!vernonCpuWorkgroupAddressV1(pullbackSite, 32 * 1024, 16, 0))
                                           return VERNON_STATUS_INTERNAL_ERROR;
                                       vernonCpuWorkgroupBarrierV1(0);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_OK)
        << scheduler_->lastDiagnostic();
}

TEST_F(CpuWorkgroupDispatchTest, SealsAllocationSitesAtTheFirstBarrier) {
    constexpr uint32_t grid[3]{1, 1, 1};
    constexpr uint32_t workgroup[3]{2, 1, 1};

    EXPECT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [](const vernon::runtime::CpuLaneCoordinates &) {
                                       for (uint64_t site = 0; site < 257; ++site)
                                           if (!vernonCpuWorkgroupAddressV1(site, 1, 1, 0))
                                               return VERNON_STATUS_INTERNAL_ERROR;
                                       vernonCpuWorkgroupBarrierV1(0);
                                       EXPECT_EQ(vernonCpuWorkgroupAddressV1(257, 1, 1, 0), 0u);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_NE(scheduler_->lastDiagnostic().find("not established during preflight"), std::string::npos);
}

TEST_F(CpuWorkgroupDispatchTest, RejectsWorkgroupsAboveConfiguredVolumeBeforeExecution) {
    constexpr uint32_t grid[3]{1, 1, 1};
    constexpr uint32_t workgroup[3]{17, 1, 1};
    std::atomic<bool> executed{};

    EXPECT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [&](const vernon::runtime::CpuLaneCoordinates &) {
                                       executed.store(true, std::memory_order_relaxed);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_FALSE(executed.load(std::memory_order_relaxed));
    EXPECT_NE(scheduler_->lastDiagnostic().find("exceeds configured maximum"), std::string::npos);
}

TEST_F(CpuWorkgroupDispatchTest, SupportsOneCooperativeTeamLargerThanTheParallelBudget) {
    constexpr uint32_t grid[3]{1, 1, 1};
    constexpr uint32_t workgroup[3]{9, 1, 1};
    std::atomic<uint32_t> visits{};

    ASSERT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [&](const vernon::runtime::CpuLaneCoordinates &) {
                                       visits.fetch_add(1, std::memory_order_relaxed);
                                       vernonCpuWorkgroupBarrierV1(0);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_OK)
        << scheduler_->lastDiagnostic();
    EXPECT_EQ(visits.load(std::memory_order_relaxed), 9u);
    EXPECT_EQ(scheduler_->workerCount(), 9u);
}

TEST(CpuWorkgroupSchedulerConfig, RejectsInvalidLimits) {
    std::string error;
    EXPECT_EQ(vernon::runtime::CpuWorkgroupScheduler::create({0, 16}, error), nullptr);
    EXPECT_FALSE(error.empty());
    error.clear();
    EXPECT_EQ(vernon::runtime::CpuWorkgroupScheduler::create({8, 0}, error), nullptr);
    EXPECT_FALSE(error.empty());
}

TEST(CpuWorkgroupHelpers, FailHardOutsideActiveWorkgroup) {
    EXPECT_DEATH_IF_SUPPORTED((void)vernonCpuWorkgroupAddressV1(0, 4, 4, 0),
                              "vernonCpuWorkgroupAddressV1 called outside");
    EXPECT_DEATH_IF_SUPPORTED(vernonCpuWorkgroupBarrierV1(0), "vernonCpuWorkgroupBarrierV1 called outside");
    EXPECT_DEATH_IF_SUPPORTED((void)vernonCpuWorkgroupIsLeaderV1(), "vernonCpuWorkgroupIsLeaderV1 called outside");
}

} // namespace
