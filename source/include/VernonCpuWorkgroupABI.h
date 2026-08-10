#ifndef VERNON_CPU_WORKGROUP_ABI_H
#define VERNON_CPU_WORKGROUP_ABI_H

#include <stdbool.h>
#include <stdint.h>

#define VERNON_CPU_WORKGROUP_ADDRESS_V1_SYMBOL "vernonCpuWorkgroupAddressV1"
#define VERNON_CPU_LANE_ADDRESS_V1_SYMBOL "vernonCpuLaneAddressV1"
#define VERNON_CPU_WORKGROUP_BARRIER_V1_SYMBOL "vernonCpuWorkgroupBarrierV1"
#define VERNON_CPU_WORKGROUP_IS_LEADER_V1_SYMBOL "vernonCpuWorkgroupIsLeaderV1"

#ifdef __cplusplus
extern "C" {
#endif

uint64_t vernonCpuWorkgroupAddressV1(uint64_t site, uint64_t size, uint64_t alignment, uint64_t offset);
uint64_t vernonCpuLaneAddressV1(uint64_t site, uint64_t size, uint64_t alignment, uint64_t offset);
void vernonCpuWorkgroupBarrierV1(uint64_t site);
bool vernonCpuWorkgroupIsLeaderV1(void);

#ifdef __cplusplus
}
#endif

#endif
