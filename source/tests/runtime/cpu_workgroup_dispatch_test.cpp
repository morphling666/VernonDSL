#include "runtime/cpu_workgroup_dispatch.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

template <typename Callback> VernonStatus forEachLane(VernonCpuRangeV1 &range, Callback &&callback) {
    for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
        range.active_lane = localLinear;
        const VernonStatus status = callback(vernon::runtime::cpuRangeCoordinates(range, localLinear));
        if (status != VERNON_STATUS_OK)
            return status;
    }
    return VERNON_STATUS_OK;
}

class CpuWorkgroupDispatchTest : public ::testing::Test {
protected:
    void SetUp() override {
        vernon::runtime::CpuWorkgroupSchedulerConfig config{8, 256};
#if defined(VERNON_RUNTIME_PROFILE_WEB)
        config.executionPolicy = vernon::runtime::CpuSchedulerExecutionPolicy::CallingThread;
#endif
        scheduler_ = vernon::runtime::CpuWorkgroupScheduler::create(config, error_);
        ASSERT_NE(scheduler_, nullptr) << error_;
    }

    static void yield(VernonCpuRangeV1 &range, uint64_t site) {
        range.yielded_site = site;
        range.outcome = VERNON_CPU_RANGE_YIELDED_V1;
    }

    std::string error_;
    std::unique_ptr<vernon::runtime::CpuWorkgroupScheduler> scheduler_;
};

TEST_F(CpuWorkgroupDispatchTest, ExecutesEachInvocationInContiguousGroupRangesAndReusesWorkers) {
    constexpr uint32_t grid[3]{7, 5, 3};
    constexpr uint32_t workgroup[3]{4, 2, 1};
    constexpr size_t invocationCount = 7 * 5 * 3 * 4 * 2;
    std::array<std::atomic<uint32_t>, invocationCount> visits{};
    std::array<std::set<std::thread::id>, 2> launchThreads;

    for (size_t launch = 0; launch < launchThreads.size(); ++launch) {
        std::mutex threadsMutex;
        ASSERT_EQ(
            scheduler_->dispatch(grid, workgroup,
                                 [&](VernonCpuRangeV1 &range) {
                                     {
                                         std::lock_guard lock(threadsMutex);
                                         launchThreads[launch].insert(std::this_thread::get_id());
                                     }
                                     EXPECT_EQ(range.lane_begin, 0u);
                                     EXPECT_EQ(range.lane_end, 8u);
                                     return forEachLane(range, [&](const vernon::runtime::CpuLaneCoordinates &lane) {
                                         visits[lane.linearIndex].fetch_add(1, std::memory_order_relaxed);
                                         EXPECT_EQ(lane.global[0], lane.group[0] * workgroup[0] + lane.local[0]);
                                         EXPECT_EQ(lane.global[1], lane.group[1] * workgroup[1] + lane.local[1]);
                                         return VERNON_STATUS_OK;
                                     });
                                 }),
            VERNON_STATUS_OK)
            << scheduler_->lastDiagnostic();
    }

    for (const std::atomic<uint32_t> &visit : visits)
        EXPECT_EQ(visit.load(std::memory_order_relaxed), 2u);
    EXPECT_FALSE(launchThreads[0].empty());
    EXPECT_FALSE(launchThreads[1].empty());
    EXPECT_LE(launchThreads[0].size(), 8u);
    EXPECT_LE(launchThreads[1].size(), 8u);
    std::set<std::thread::id> workerPool = launchThreads[0];
    workerPool.insert(launchThreads[1].begin(), launchThreads[1].end());
    EXPECT_LE(workerPool.size(), 8u);
}

TEST_F(CpuWorkgroupDispatchTest, CallingThreadPolicyReusesRangePhaseEngineWithoutWorkers) {
    std::string error;
    auto scheduler = vernon::runtime::CpuWorkgroupScheduler::create(
        {1, 256, vernon::runtime::CpuSchedulerExecutionPolicy::CallingThread}, error);
    ASSERT_NE(scheduler, nullptr) << error;
    constexpr uint32_t grid[3]{3, 1, 1};
    constexpr uint32_t workgroup[3]{4, 1, 1};
    std::array<uint32_t, 12> visits{};
    const std::thread::id callingThread = std::this_thread::get_id();

    ASSERT_EQ(scheduler->dispatch(grid, workgroup,
                                  [&](VernonCpuRangeV1 &range) {
                                      EXPECT_EQ(std::this_thread::get_id(), callingThread);
                                      if (range.phase == 0) {
                                          yield(range, 7);
                                          return VERNON_STATUS_OK;
                                      }
                                      return forEachLane(range, [&](const vernon::runtime::CpuLaneCoordinates &lane) {
                                          ++visits[lane.linearIndex];
                                          return VERNON_STATUS_OK;
                                      });
                                  }),
              VERNON_STATUS_OK)
        << scheduler->lastDiagnostic();
    EXPECT_TRUE(std::all_of(visits.begin(), visits.end(), [](uint32_t count) { return count == 1; }));
}

TEST_F(CpuWorkgroupDispatchTest, InlineDispatchBypassesWorkerPoolForMultipleLargeGroups) {
    constexpr uint32_t grid[3]{4, 1, 1};
    constexpr uint32_t workgroup[3]{256, 1, 1};
    const std::thread::id callingThread = std::this_thread::get_id();
    std::atomic<size_t> visits{};

    ASSERT_EQ(scheduler_->dispatchInline(grid, workgroup,
                                         [&](VernonCpuRangeV1 &range) {
                                             EXPECT_EQ(std::this_thread::get_id(), callingThread);
                                             visits.fetch_add(range.lane_end - range.lane_begin,
                                                              std::memory_order_relaxed);
                                             return VERNON_STATUS_OK;
                                         }),
              VERNON_STATUS_OK)
        << scheduler_->lastDiagnostic();
    EXPECT_EQ(visits.load(std::memory_order_relaxed), 1024u);
}

TEST_F(CpuWorkgroupDispatchTest, PersistsSharedStorageAcrossNonblockingPhases) {
    constexpr uint32_t grid[3]{2, 1, 1};
    constexpr uint32_t workgroup[3]{4, 1, 1};
    std::array<int32_t, 10> result{};

    ASSERT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [&](VernonCpuRangeV1 &range) {
                                       const uint64_t address = vernonCpuWorkgroupAddressV1(
                                           0, sizeof(std::atomic<int32_t>), alignof(std::atomic<int32_t>), 0);
                                       if (!address)
                                           return VERNON_STATUS_INTERNAL_ERROR;
                                       auto *shared = reinterpret_cast<std::atomic<int32_t> *>(address);
                                       if (range.phase == 0) {
                                           if (range.lane_begin == 0)
                                               new (shared)
                                                   std::atomic<int32_t>(static_cast<int32_t>(range.group[0] * 100));
                                           yield(range, 0);
                                           return VERNON_STATUS_OK;
                                       }
                                       if (range.phase == 1) {
                                           VernonStatus status =
                                               forEachLane(range, [&](const vernon::runtime::CpuLaneCoordinates &lane) {
                                                   result[lane.group[0] * 4 + lane.local[0]] =
                                                       shared->fetch_add(1, std::memory_order_relaxed);
                                                   return VERNON_STATUS_OK;
                                               });
                                           yield(range, 1);
                                           return status;
                                       }
                                       if (range.lane_begin == 0)
                                           result[8 + range.group[0]] = shared->load(std::memory_order_relaxed);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_OK)
        << scheduler_->lastDiagnostic();

    for (size_t group = 0; group < 2; ++group) {
        std::array<int32_t, 4> previous{result[group * 4], result[group * 4 + 1], result[group * 4 + 2],
                                        result[group * 4 + 3]};
        std::sort(previous.begin(), previous.end());
        const int32_t base = static_cast<int32_t>(group * 100);
        EXPECT_EQ(previous, (std::array<int32_t, 4>{base, base + 1, base + 2, base + 3}));
        EXPECT_EQ(result[8 + group], base + 4);
    }
}

TEST_F(CpuWorkgroupDispatchTest, SupportsRepeatedPhaseGenerations) {
    constexpr uint32_t grid[3]{2, 1, 1};
    constexpr uint32_t workgroup[3]{4, 1, 1};
    std::array<std::atomic<uint32_t>, 2> generations{};

    ASSERT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [&](VernonCpuRangeV1 &range) {
                                       if (range.phase == 64)
                                           return VERNON_STATUS_OK;
                                       if ((range.phase & 1) == 0 && range.lane_begin == 0)
                                           generations[range.group[0]].fetch_add(1, std::memory_order_relaxed);
                                       else if ((range.phase & 1) != 0 &&
                                                generations[range.group[0]].load(std::memory_order_relaxed) !=
                                                    (range.phase + 1) / 2)
                                           return VERNON_STATUS_INTERNAL_ERROR;
                                       yield(range, range.phase);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_OK)
        << scheduler_->lastDiagnostic();
    EXPECT_EQ(generations[0].load(std::memory_order_relaxed), 32u);
    EXPECT_EQ(generations[1].load(std::memory_order_relaxed), 32u);
}

TEST_F(CpuWorkgroupDispatchTest, PersistsLanePrivateScratchAcrossPhases) {
    constexpr uint32_t grid[3]{1, 1, 1};
    constexpr uint32_t workgroup[3]{16, 1, 1};

    ASSERT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [&](VernonCpuRangeV1 &range) {
                                       VernonStatus status =
                                           forEachLane(range, [&](const vernon::runtime::CpuLaneCoordinates &lane) {
                                               const uint64_t address =
                                                   vernonCpuLaneAddressV1(7, sizeof(uint32_t), alignof(uint32_t), 0);
                                               if (!address)
                                                   return VERNON_STATUS_INTERNAL_ERROR;
                                               auto *value = reinterpret_cast<uint32_t *>(address);
                                               if (range.phase == 0)
                                                   *value = lane.local[0] + 100;
                                               else if (*value != lane.local[0] + 100)
                                                   return VERNON_STATUS_INTERNAL_ERROR;
                                               return VERNON_STATUS_OK;
                                           });
                                       if (range.phase == 0)
                                           yield(range, 0);
                                       return status;
                                   }),
              VERNON_STATUS_OK)
        << scheduler_->lastDiagnostic();
}

#if !defined(VERNON_RUNTIME_PROFILE_WEB)
TEST_F(CpuWorkgroupDispatchTest, RunsConcurrentDispatchesWithinOneWorkerBudget) {
    constexpr uint32_t grid[3]{8, 1, 1};
    constexpr uint32_t workgroup[3]{4, 1, 1};
    std::mutex mutex;
    std::condition_variable rendezvous;
    size_t arrived = 0;
    bool release = false;
    bool timedOut = false;
    std::array<VernonStatus, 2> statuses{VERNON_STATUS_INTERNAL_ERROR, VERNON_STATUS_INTERNAL_ERROR};
    std::array<std::thread, 2> callers;

    for (size_t dispatch = 0; dispatch < callers.size(); ++dispatch) {
        callers[dispatch] = std::thread([&, dispatch] {
            statuses[dispatch] = scheduler_->dispatch(grid, workgroup, [&](VernonCpuRangeV1 &) {
                std::unique_lock lock(mutex);
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
    EXPECT_EQ(arrived, 16u);
    EXPECT_EQ(statuses[0], VERNON_STATUS_OK);
    EXPECT_EQ(statuses[1], VERNON_STATUS_OK);
}
#endif

TEST_F(CpuWorkgroupDispatchTest, ContainsRangeFailuresAndInvalidOutcomes) {
    constexpr uint32_t grid[3]{2, 1, 1};
    constexpr uint32_t workgroup[3]{4, 1, 1};

    EXPECT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [](VernonCpuRangeV1 &range) {
                                       return range.group[0] == 0 ? VERNON_STATUS_INTERNAL_ERROR : VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_NE(scheduler_->lastDiagnostic().find("CPU range returned an error"), std::string::npos);

    EXPECT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [](VernonCpuRangeV1 &range) {
                                       range.outcome = UINT32_MAX;
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_NE(scheduler_->lastDiagnostic().find("invalid phase outcome"), std::string::npos);
}

TEST_F(CpuWorkgroupDispatchTest, ReportsArenaOverflowAndSealsAtFirstYield) {
    constexpr uint32_t grid[3]{1, 1, 1};
    constexpr uint32_t workgroup[3]{4, 1, 1};

    EXPECT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [&](VernonCpuRangeV1 &) {
                                       EXPECT_EQ(vernonCpuWorkgroupAddressV1(0, 16 * 1024 + 1, 4, 0), 0u);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [&](VernonCpuRangeV1 &) {
                                       EXPECT_EQ(vernonCpuLaneAddressV1(0, uint64_t{1} << 63, 1, 0), 0u);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_INTERNAL_ERROR);

    EXPECT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [&](VernonCpuRangeV1 &range) {
                                       if (range.phase == 0) {
                                           EXPECT_NE(vernonCpuWorkgroupAddressV1(0, 4, 4, 0), 0u);
                                           yield(range, 0);
                                           return VERNON_STATUS_OK;
                                       }
                                       EXPECT_EQ(vernonCpuWorkgroupAddressV1(1, 4, 4, 0), 0u);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_NE(scheduler_->lastDiagnostic().find("not established during preflight"), std::string::npos);
}

#if !defined(VERNON_RUNTIME_PROFILE_WEB)
TEST_F(CpuWorkgroupDispatchTest, KeepsWorkerCountBoundedForLargeWorkgroups) {
    constexpr uint32_t grid[3]{1, 1, 1};
    constexpr uint32_t workgroup[3]{256, 1, 1};
    std::atomic<uint32_t> visits{};
    std::mutex threadsMutex;
    std::set<std::thread::id> threads;

    ASSERT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [&](VernonCpuRangeV1 &range) {
                                       {
                                           std::lock_guard lock(threadsMutex);
                                           threads.insert(std::this_thread::get_id());
                                       }
                                       visits.fetch_add(static_cast<uint32_t>(range.lane_end - range.lane_begin),
                                                        std::memory_order_relaxed);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_OK)
        << scheduler_->lastDiagnostic();
    EXPECT_EQ(visits.load(std::memory_order_relaxed), 256u);
    EXPECT_GT(threads.size(), 1u);
    EXPECT_LE(threads.size(), 8u);
}
#endif

TEST_F(CpuWorkgroupDispatchTest, PartitionsUnevenLaneChunksWithoutEmptyRanges) {
    constexpr uint32_t grid[3]{2, 1, 1};
    constexpr uint32_t workgroup[3]{6, 1, 1};
    std::array<std::atomic<uint32_t>, 12> visits{};

    ASSERT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [&](VernonCpuRangeV1 &range) {
                                       if (range.lane_begin >= range.lane_end)
                                           return VERNON_STATUS_INTERNAL_ERROR;
                                       return forEachLane(
                                           range, [&](const vernon::runtime::CpuLaneCoordinates &coordinates) {
                                               visits[coordinates.linearIndex].fetch_add(1, std::memory_order_relaxed);
                                               return VERNON_STATUS_OK;
                                           });
                                   }),
              VERNON_STATUS_OK)
        << scheduler_->lastDiagnostic();
    for (const std::atomic<uint32_t> &visit : visits)
        EXPECT_EQ(visit.load(std::memory_order_relaxed), 1u);
}

TEST_F(CpuWorkgroupDispatchTest, ExecutesTinyDispatchInline) {
    constexpr uint32_t grid[3]{1, 1, 1};
    constexpr uint32_t workgroup[3]{8, 1, 1};
    const std::thread::id caller = std::this_thread::get_id();

    ASSERT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [&](VernonCpuRangeV1 &range) {
                                       EXPECT_EQ(std::this_thread::get_id(), caller);
                                       EXPECT_EQ(range.lane_begin, 0u);
                                       EXPECT_EQ(range.lane_end, 8u);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_OK);
}

TEST_F(CpuWorkgroupDispatchTest, AllocatesLaneStateAsOneContiguousGroupSlab) {
    constexpr uint32_t grid[3]{1, 1, 1};
    constexpr uint32_t workgroup[3]{256, 1, 1};
    std::vector<uint64_t> addresses(256);

    ASSERT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [&](VernonCpuRangeV1 &range) {
                                       return forEachLane(range, [&](const vernon::runtime::CpuLaneCoordinates &lane) {
                                           addresses[lane.local[0]] =
                                               vernonCpuLaneAddressV1(42, sizeof(uint64_t), alignof(uint64_t), 0);
                                           return addresses[lane.local[0]] ? VERNON_STATUS_OK
                                                                           : VERNON_STATUS_INTERNAL_ERROR;
                                       });
                                   }),
              VERNON_STATUS_OK)
        << scheduler_->lastDiagnostic();

    for (size_t lane = 1; lane < addresses.size(); ++lane)
        EXPECT_EQ(addresses[lane], addresses[0] + lane * sizeof(uint64_t));
}

TEST_F(CpuWorkgroupDispatchTest, StreamsLargeGridThroughBoundedActiveGroups) {
    constexpr uint32_t grid[3]{100000, 1, 1};
    constexpr uint32_t workgroup[3]{1, 1, 1};
    std::atomic<size_t> completed{};

    ASSERT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [&](VernonCpuRangeV1 &) {
                                       completed.fetch_add(1, std::memory_order_relaxed);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_OK)
        << scheduler_->lastDiagnostic();
    EXPECT_EQ(completed.load(std::memory_order_relaxed), 100000u);
}

TEST_F(CpuWorkgroupDispatchTest, RejectsWorkgroupsAboveConfiguredVolumeBeforeExecution) {
    constexpr uint32_t grid[3]{1, 1, 1};
    constexpr uint32_t workgroup[3]{257, 1, 1};
    std::atomic<bool> executed{};

    EXPECT_EQ(scheduler_->dispatch(grid, workgroup,
                                   [&](VernonCpuRangeV1 &) {
                                       executed.store(true, std::memory_order_relaxed);
                                       return VERNON_STATUS_OK;
                                   }),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_FALSE(executed.load(std::memory_order_relaxed));
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
    EXPECT_DEATH_IF_SUPPORTED((void)vernonCpuLaneAddressV1(0, 4, 4, 0), "vernonCpuLaneAddressV1 called outside");
    EXPECT_DEATH_IF_SUPPORTED(vernonCpuWorkgroupBarrierV1(0), "vernonCpuWorkgroupBarrierV1 called outside");
    EXPECT_DEATH_IF_SUPPORTED((void)vernonCpuWorkgroupIsLeaderV1(), "vernonCpuWorkgroupIsLeaderV1 called outside");
}

} // namespace
