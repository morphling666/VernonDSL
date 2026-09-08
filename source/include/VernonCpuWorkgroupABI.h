#ifndef VERNON_CPU_WORKGROUP_ABI_H
#define VERNON_CPU_WORKGROUP_ABI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define VERNON_CPU_WORKGROUP_ADDRESS_V1_SYMBOL "vernonCpuWorkgroupAddressV1"
#define VERNON_CPU_LANE_ADDRESS_V1_SYMBOL "vernonCpuLaneAddressV1"
#define VERNON_CPU_WORKGROUP_BARRIER_V1_SYMBOL "vernonCpuWorkgroupBarrierV1"
#define VERNON_CPU_WORKGROUP_IS_LEADER_V1_SYMBOL "vernonCpuWorkgroupIsLeaderV1"
#define VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1 SIZE_MAX
#define VERNON_CPU_RANGE_COMPLETE_V1 0
#define VERNON_CPU_RANGE_YIELDED_V1 1
#define VERNON_CPU_LANE_COROUTINE_HANDLE_SITE_V1 UINT64_C(0xfffffffffffffffe)
#define VERNON_CPU_LANE_COROUTINE_FRAME_SITE_V1 UINT64_C(0xfffffffffffffffd)

typedef struct VernonCpuRangeV1 {
    size_t struct_size;
    const void *arguments;
    size_t arguments_size;
    void *results;
    size_t results_size;
    const void *textures;
    uint32_t grid[3];
    uint32_t workgroup[3];
    uint32_t group[3];
    /* Half-open workgroup-linear lane interval executed by this call. */
    size_t lane_begin;
    size_t lane_end;
    size_t active_lane;
    uint64_t phase;
    uint64_t yielded_site;
    size_t completed_lanes;
    uint32_t outcome;
    /*
     * Optional per-invocation frames, indexed by flattened global invocation.
     * A null table uses the common arguments/results pointer above.
     */
    const void *const *lane_arguments;
    void *const *lane_results;
    size_t lane_table_count;
} VernonCpuRangeV1;

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
