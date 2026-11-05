#ifndef VERNON_RUNTIME_BACKEND_CPU_H
#define VERNON_RUNTIME_BACKEND_CPU_H

#include "VernonRuntime.h"
#include "cpu_workgroup_dispatch.h"

#include <cstddef>
#include <memory>
#include <string>

struct VernonRuntimeContext;

namespace vernon::runtime {

struct CpuContextState {
    std::unique_ptr<CpuWorkgroupScheduler> scheduler;
    std::string error;
};

bool initializeCpuContext(VernonRuntimeContext &context, uint32_t deviceIndex);
CpuWorkgroupScheduler &cpuWorkgroupScheduler(VernonRuntimeContext &context);
VernonStatus registerStaticCpuEntries(const VernonStaticCpuEntry *entries, size_t count);
VernonStatus unregisterStaticCpuEntries(const VernonStaticCpuEntry *entries, size_t count);
VernonCpuEntryPoint findStaticCpuEntry(const std::string &symbol);

bool findRegisteredCpuEntry(VernonRuntimeContext &context, const std::string &symbol, VernonCpuEntryPoint &entry,
                            std::string &error);

} // namespace vernon::runtime

#endif
