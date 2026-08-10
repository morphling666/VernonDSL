#ifndef VERNON_RUNTIME_CPU_WORKGROUP_DISPATCH_H
#define VERNON_RUNTIME_CPU_WORKGROUP_DISPATCH_H

#include "VernonCpuWorkgroupABI.h"
#include "VernonRuntime.h"

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

using CpuLaneCallback = std::function<VernonStatus(const CpuLaneCoordinates &)>;

struct CpuWorkgroupSchedulerConfig {
    size_t threadBudget{};
    size_t maxWorkgroupVolume{};
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
                          const CpuLaneCallback &callback) noexcept;
    const std::string &lastDiagnostic() const noexcept;
    size_t workerCount() const noexcept;
    size_t maxWorkgroupVolume() const noexcept;

private:
    class Impl;
    explicit CpuWorkgroupScheduler(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;
};

} // namespace vernon::runtime

#endif
