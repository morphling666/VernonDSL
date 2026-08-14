#ifndef VERNON_RUNTIME_CPU_WORKGROUP_DISPATCH_H
#define VERNON_RUNTIME_CPU_WORKGROUP_DISPATCH_H

#include "VernonCommon.h"
#include "VernonCpuWorkgroupABI.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace vernon::runtime {

struct CpuLaneCoordinates {
    uint32_t global[3]{};
    uint32_t local[3]{};
    uint32_t group[3]{};
    size_t linearIndex{};
};

CpuLaneCoordinates cpuRangeCoordinates(const VernonCpuRangeV1 &range, size_t localLinear) noexcept;

using CpuRangeCallback = std::function<VernonStatus(VernonCpuRangeV1 &)>;

enum class CpuSchedulerExecutionPolicy {
    WorkerPool,
    CallingThread,
};

struct CpuWorkgroupSchedulerConfig {
    size_t threadBudget{};
    size_t maxWorkgroupVolume{};
    CpuSchedulerExecutionPolicy executionPolicy{CpuSchedulerExecutionPolicy::WorkerPool};
};

class CpuWorkgroupScheduler {
public:
    static CpuWorkgroupSchedulerConfig defaultConfig() noexcept;
    static std::unique_ptr<CpuWorkgroupScheduler> create(CpuWorkgroupSchedulerConfig config,
                                                         std::string &error) noexcept;

    ~CpuWorkgroupScheduler();
    CpuWorkgroupScheduler(const CpuWorkgroupScheduler &) = delete;
    CpuWorkgroupScheduler &operator=(const CpuWorkgroupScheduler &) = delete;

    VernonStatus dispatch(const uint32_t grid[3], const uint32_t workgroup[3],
                          const CpuRangeCallback &callback) noexcept;
    VernonStatus dispatchInline(const uint32_t grid[3], const uint32_t workgroup[3],
                                const CpuRangeCallback &callback) noexcept;
    const std::string &lastDiagnostic() const noexcept;

private:
    class Impl;
    explicit CpuWorkgroupScheduler(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;
};

} // namespace vernon::runtime

#endif
