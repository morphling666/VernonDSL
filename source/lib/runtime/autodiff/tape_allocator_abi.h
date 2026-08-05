#ifndef VERNON_RUNTIME_AUTODIFF_TAPE_ALLOCATOR_ABI_H
#define VERNON_RUNTIME_AUTODIFF_TAPE_ALLOCATOR_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VERNON_AD_TAPE_ALLOCATOR_ABI_VERSION 1u
#define VERNON_AD_TAPE_ALLOCATOR_BUILTIN "ad_tape_allocator"

typedef uint64_t VernonAdRegionHandle;

#define VERNON_AD_INVALID_REGION_HANDLE UINT64_C(0)

typedef uint32_t VernonAdTapeAllocatorStatus;

enum {
    VERNON_AD_TAPE_ALLOCATOR_OK = 0,
    VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED = 1,
    VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW = 2,
    VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE = 3,
    VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE = 4,
    VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI = 5
};

typedef struct VernonAdTapeAllocator VernonAdTapeAllocator;

typedef VernonAdTapeAllocatorStatus (*VernonAdTapeResetCallback)(VernonAdTapeAllocator *allocator);
typedef VernonAdTapeAllocatorStatus (*VernonAdTapeBeginRegionCallback)(VernonAdTapeAllocator *allocator,
                                                                       VernonAdRegionHandle parent,
                                                                       VernonAdRegionHandle *region);
typedef VernonAdTapeAllocatorStatus (*VernonAdTapeReserveCallback)(VernonAdTapeAllocator *allocator,
                                                                   VernonAdRegionHandle region, size_t byte_size,
                                                                   size_t alignment, size_t *byte_offset,
                                                                   void **write_address);
typedef VernonAdTapeAllocatorStatus (*VernonAdTapeEndRegionCallback)(VernonAdTapeAllocator *allocator,
                                                                     VernonAdRegionHandle region);
typedef VernonAdTapeAllocatorStatus (*VernonAdTapeSealCallback)(VernonAdTapeAllocator *allocator);
typedef VernonAdTapeAllocatorStatus (*VernonAdTapeReadRegionCallback)(VernonAdTapeAllocator *allocator,
                                                                      VernonAdRegionHandle region, const void **data,
                                                                      size_t *byte_size);

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
 * every region handle from the previous capture. Regions are properly nested:
 * parent must be the current open region, and end_region closes the current
 * region. A region's byte range includes all reservations made by nested
 * descendants. seal succeeds only after every region is closed.
 *
 * reserve accepts power-of-two alignment. Its padding and byte_size both count
 * toward required_bytes. After CAPACITY_EXHAUSTED, reserve performs no writes
 * but continues exact checked accounting for every attempted reservation.
 * write_address remains valid until the next reserve or reset callback; the
 * caller must write the reservation before making either call.
 * INVALID_STATE, arithmetic overflow, and host allocation failure are terminal
 * until reset; callbacks other than reset return the latched failure without
 * further mutation. reset is the only operation that clears these failures.
 *
 * read_region is legal only after a successful seal and until reset or
 * ownership transfer. A successful CPU entry transfers the sealed, immutable
 * tape storage to its pullback; callbacks then fail with INVALID_STATE until
 * reset, and destroying the pullback releases the transferred storage.
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
    VernonAdTapeReserveCallback reserve;
    VernonAdTapeEndRegionCallback end_region;
    VernonAdTapeSealCallback seal;
    VernonAdTapeReadRegionCallback read_region;
};

#ifdef __cplusplus
}
#endif

#endif
