#ifndef VERNON_CPU_WORKGROUP_ABI_H
#define VERNON_CPU_WORKGROUP_ABI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define VERNON_CPU_WORKGROUP_ADDRESS_SYMBOL "vernonCpuWorkgroupAddress"
#define VERNON_CPU_LANE_ADDRESS_SYMBOL "vernonCpuLaneAddress"
#define VERNON_CPU_WORKGROUP_BARRIER_SYMBOL "vernonCpuWorkgroupBarrier"
#define VERNON_CPU_WORKGROUP_IS_LEADER_SYMBOL "vernonCpuWorkgroupIsLeader"
#define VERNON_CPU_RANGE_ARGUMENTS_SIZE SIZE_MAX
#define VERNON_CPU_RANGE_COMPLETE 0
#define VERNON_CPU_RANGE_YIELDED 1
#define VERNON_CPU_LANE_COROUTINE_HANDLE_SITE UINT64_C(0xfffffffffffffffe)
#define VERNON_CPU_LANE_COROUTINE_FRAME_SITE UINT64_C(0xfffffffffffffffd)

typedef struct VernonCpuRange {
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
} VernonCpuRange;

#ifdef __cplusplus
extern "C" {
#endif

uint64_t vernonCpuWorkgroupAddress(uint64_t site, uint64_t size, uint64_t alignment, uint64_t offset);
uint64_t vernonCpuLaneAddress(uint64_t site, uint64_t size, uint64_t alignment, uint64_t offset);
void vernonCpuWorkgroupBarrier(uint64_t site);
bool vernonCpuWorkgroupIsLeader(void);

#ifdef __cplusplus
}
#endif

#endif
