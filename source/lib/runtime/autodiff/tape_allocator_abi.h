#ifndef VERNON_RUNTIME_AUTODIFF_TAPE_ALLOCATOR_ABI_H
#define VERNON_RUNTIME_AUTODIFF_TAPE_ALLOCATOR_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VERNON_AD_TAPE_ALLOCATOR_ABI_VERSION 2u
#define VERNON_AD_TAPE_ALLOCATOR_BUILTIN "ad_tape_allocator"
#define VERNON_AD_TAPE_ROOT_REGION_BUILTIN "ad_tape_root_region"

typedef uint64_t VernonAdRegionHandle;
typedef uint64_t VernonAdRecordHandle;

#define VERNON_AD_INVALID_REGION_HANDLE UINT64_C(0)
#define VERNON_AD_INVALID_RECORD_HANDLE UINT64_C(0)

typedef uint32_t VernonAdTapeAllocatorStatus;

enum {
    VERNON_AD_TAPE_ALLOCATOR_OK = 0,
    VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED = 1,
    VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW = 2,
    VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE = 3,
    VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE = 4,
    VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI = 5
};

enum {
    VERNON_AD_TAPE_ALLOCATOR_FIELD_REQUIRED_BYTES = 5,
    VERNON_AD_TAPE_ALLOCATOR_FIELD_RESET = 6,
    VERNON_AD_TAPE_ALLOCATOR_FIELD_BEGIN_REGION = 7,
    VERNON_AD_TAPE_ALLOCATOR_FIELD_RESERVE_RECORD = 8,
    VERNON_AD_TAPE_ALLOCATOR_FIELD_WRITE_LEAF = 9,
    VERNON_AD_TAPE_ALLOCATOR_FIELD_SET_CHILD = 10,
    VERNON_AD_TAPE_ALLOCATOR_FIELD_END_REGION = 11,
    VERNON_AD_TAPE_ALLOCATOR_FIELD_SEAL = 12,
    VERNON_AD_TAPE_ALLOCATOR_FIELD_READ_LEAF = 13,
    VERNON_AD_TAPE_ALLOCATOR_FIELD_READ_CHILD = 14,
    VERNON_AD_TAPE_ALLOCATOR_FIELD_READ_EXECUTED_COUNT = 15,
    VERNON_AD_TAPE_ALLOCATOR_FIELD_READ_EXIT_KIND = 16
};

typedef struct VernonAdTapeAllocator VernonAdTapeAllocator;

typedef VernonAdTapeAllocatorStatus (*VernonAdTapeResetCallback)(VernonAdTapeAllocator *allocator);
typedef VernonAdTapeAllocatorStatus (*VernonAdTapeBeginRegionCallback)(VernonAdTapeAllocator *allocator,
                                                                       VernonAdRegionHandle parent,
                                                                       VernonAdRegionHandle *region);
typedef VernonAdTapeAllocatorStatus (*VernonAdTapeReserveRecordCallback)(VernonAdTapeAllocator *allocator,
                                                                         VernonAdRegionHandle region,
                                                                         size_t payload_size, size_t payload_alignment,
                                                                         size_t child_count,
                                                                         VernonAdRecordHandle *record);
typedef VernonAdTapeAllocatorStatus (*VernonAdTapeWriteLeafCallback)(VernonAdTapeAllocator *allocator,
                                                                     VernonAdRecordHandle record, size_t leaf_offset,
                                                                     const void *data, size_t byte_size);
typedef VernonAdTapeAllocatorStatus (*VernonAdTapeSetChildCallback)(VernonAdTapeAllocator *allocator,
                                                                    VernonAdRecordHandle record, size_t child_ordinal,
                                                                    VernonAdRegionHandle child);
typedef VernonAdTapeAllocatorStatus (*VernonAdTapeEndRegionCallback)(VernonAdTapeAllocator *allocator,
                                                                     VernonAdRegionHandle region, size_t executed_count,
                                                                     uint32_t exit_kind);
typedef VernonAdTapeAllocatorStatus (*VernonAdTapeSealCallback)(VernonAdTapeAllocator *allocator);
typedef VernonAdTapeAllocatorStatus (*VernonAdTapeReadLeafCallback)(VernonAdTapeAllocator *allocator,
                                                                    VernonAdRegionHandle region, size_t record_index,
                                                                    size_t leaf_offset, void *data, size_t byte_size);
typedef VernonAdTapeAllocatorStatus (*VernonAdTapeReadChildCallback)(VernonAdTapeAllocator *allocator,
                                                                     VernonAdRegionHandle region, size_t record_index,
                                                                     size_t child_ordinal, VernonAdRegionHandle *child);
typedef VernonAdTapeAllocatorStatus (*VernonAdTapeReadExecutedCountCallback)(VernonAdTapeAllocator *allocator,
                                                                             VernonAdRegionHandle region,
                                                                             size_t *executed_count);
typedef VernonAdTapeAllocatorStatus (*VernonAdTapeReadExitKindCallback)(VernonAdTapeAllocator *allocator,
                                                                        VernonAdRegionHandle region,
                                                                        uint32_t *exit_kind);

/*
 * Internal Compiler/Runtime ABI supplied through the hidden
 * "ad_tape_allocator" packed argument. It is not a user function parameter,
 * and VernonCpuInvocation::textures remains exclusively a texture callback
 * table.
 *
 * One descriptor belongs to one Runtime-created invocation and to the thread
 * that created it. Generated code must not copy the descriptor or modify its
 * fields. Callbacks from another thread fail with INVALID_STATE.
 *
 * reset starts a new capture, clears status and required_bytes, and invalidates
 * every region and record handle from the previous capture. Regions are
 * properly nested: parent must be the current open region, and end_region
 * closes the current region with its executed count and exit kind. seal
 * succeeds only after every region is closed and every child slot is set.
 *
 * reserve_record accepts power-of-two payload alignment. Padding, payload, and
 * Runtime-owned child metadata count toward required_bytes. Opaque handles are
 * stable across payload growth. Generated code can only write leaves and child
 * relationships through semantic callbacks; it cannot observe tape storage.
 * INVALID_STATE, arithmetic overflow, and host allocation failure are terminal
 * until reset; callbacks other than reset return the latched failure without
 * further mutation. reset is the only operation that clears these failures.
 *
 * Read callbacks are legal only after a successful seal. A successful CPU
 * entry transfers the sealed, immutable snapshot to its pullback. Snapshot
 * descriptors reject every capture callback, and destroying the pullback
 * releases the transferred storage and its memory-policy charge.
 */
struct VernonAdTapeAllocator {
    size_t struct_size;
    uint32_t abi_version;
    VernonAdTapeAllocatorStatus status;
    void *user_data;
    size_t capacity_bytes;
    size_t required_bytes;
    VernonAdTapeResetCallback reset;
    VernonAdTapeBeginRegionCallback begin_region;
    VernonAdTapeReserveRecordCallback reserve_record;
    VernonAdTapeWriteLeafCallback write_leaf;
    VernonAdTapeSetChildCallback set_child;
    VernonAdTapeEndRegionCallback end_region;
    VernonAdTapeSealCallback seal;
    VernonAdTapeReadLeafCallback read_leaf;
    VernonAdTapeReadChildCallback read_child;
    VernonAdTapeReadExecutedCountCallback read_executed_count;
    VernonAdTapeReadExitKindCallback read_exit_kind;
};

#ifdef __cplusplus
}

#include <type_traits>

static_assert(sizeof(VernonAdTapeAllocatorStatus) == 4);
static_assert(sizeof(VernonAdRegionHandle) == 8);
static_assert(sizeof(VernonAdRecordHandle) == 8);
static_assert(alignof(VernonAdTapeAllocator) == alignof(void *));
static_assert(offsetof(VernonAdTapeAllocator, abi_version) == (sizeof(void *) == 8 ? 8u : 4u));
static_assert(offsetof(VernonAdTapeAllocator, status) == (sizeof(void *) == 8 ? 12u : 8u));
static_assert(offsetof(VernonAdTapeAllocator, user_data) == (sizeof(void *) == 8 ? 16u : 12u));
static_assert(offsetof(VernonAdTapeAllocator, capacity_bytes) == (sizeof(void *) == 8 ? 24u : 16u));
static_assert(offsetof(VernonAdTapeAllocator, required_bytes) == (sizeof(void *) == 8 ? 32u : 20u));
static_assert(offsetof(VernonAdTapeAllocator, reset) == (sizeof(void *) == 8 ? 40u : 24u));
static_assert(offsetof(VernonAdTapeAllocator, begin_region) == (sizeof(void *) == 8 ? 48u : 28u));
static_assert(offsetof(VernonAdTapeAllocator, reserve_record) == (sizeof(void *) == 8 ? 56u : 32u));
static_assert(offsetof(VernonAdTapeAllocator, write_leaf) == (sizeof(void *) == 8 ? 64u : 36u));
static_assert(offsetof(VernonAdTapeAllocator, set_child) == (sizeof(void *) == 8 ? 72u : 40u));
static_assert(offsetof(VernonAdTapeAllocator, end_region) == (sizeof(void *) == 8 ? 80u : 44u));
static_assert(offsetof(VernonAdTapeAllocator, seal) == (sizeof(void *) == 8 ? 88u : 48u));
static_assert(offsetof(VernonAdTapeAllocator, read_leaf) == (sizeof(void *) == 8 ? 96u : 52u));
static_assert(offsetof(VernonAdTapeAllocator, read_child) == (sizeof(void *) == 8 ? 104u : 56u));
static_assert(offsetof(VernonAdTapeAllocator, read_executed_count) == (sizeof(void *) == 8 ? 112u : 60u));
static_assert(offsetof(VernonAdTapeAllocator, read_exit_kind) == (sizeof(void *) == 8 ? 120u : 64u));
static_assert(sizeof(VernonAdTapeAllocator) == (sizeof(void *) == 8 ? 128u : 68u));
static_assert(std::is_same_v<decltype(VernonAdTapeAllocator::reset), VernonAdTapeResetCallback>);
static_assert(std::is_same_v<decltype(VernonAdTapeAllocator::begin_region), VernonAdTapeBeginRegionCallback>);
static_assert(std::is_same_v<decltype(VernonAdTapeAllocator::reserve_record), VernonAdTapeReserveRecordCallback>);
static_assert(std::is_same_v<decltype(VernonAdTapeAllocator::write_leaf), VernonAdTapeWriteLeafCallback>);
static_assert(std::is_same_v<decltype(VernonAdTapeAllocator::set_child), VernonAdTapeSetChildCallback>);
static_assert(std::is_same_v<decltype(VernonAdTapeAllocator::end_region), VernonAdTapeEndRegionCallback>);
static_assert(std::is_same_v<decltype(VernonAdTapeAllocator::seal), VernonAdTapeSealCallback>);
static_assert(std::is_same_v<decltype(VernonAdTapeAllocator::read_leaf), VernonAdTapeReadLeafCallback>);
static_assert(std::is_same_v<decltype(VernonAdTapeAllocator::read_child), VernonAdTapeReadChildCallback>);
static_assert(
    std::is_same_v<decltype(VernonAdTapeAllocator::read_executed_count), VernonAdTapeReadExecutedCountCallback>);
static_assert(std::is_same_v<decltype(VernonAdTapeAllocator::read_exit_kind), VernonAdTapeReadExitKindCallback>);
#endif

#endif
