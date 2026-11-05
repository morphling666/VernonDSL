#ifndef VERNON_RUNTIME_SCHEDULE_LINKER_H
#define VERNON_RUNTIME_SCHEDULE_LINKER_H

// The linker of specs/runtime/architecture_v2.md section 4.2. It runs when a graph's structure changes, never on a
// call. Queues and render scopes come with graphics nodes.

#include "VernonResult.hpp"
#include "runtime/schedule/fragment.h"

#include <string>
#include <vector>

namespace vernon::schedule {

// Slot `slot` of graph node `node`.
struct NodeSlot {
    uint32_t node{};
    uint32_t slot{};
};

// Node slots that are one buffer or one scalar: a Value edge's producer export and consumer imports, or the versions
// of a Storage chain. An internal connection is a buffer only the graph uses, which the schedule allocates in its
// arena from the extents its producer's slot declares; any other is one slot of the linked schedule. A slot in no
// connection is a slot of its own.
struct Connection {
    std::vector<NodeSlot> sites;
    bool internal{};
};

// An operation of the linked schedule, with its ranges in the linked record.
struct LinkedDispatch {
    OperationKind kind{OperationKind::Dispatch};
    uint32_t executable{};
    uint64_t argumentsOffset{};
    uint64_t argumentsSize{};
    uint64_t gridOffset{};
    bool unitGridAxes[3]{};
    std::vector<Extent> gridElements;
    // Their slots are linked slots.
    std::vector<std::pair<uint32_t, uint32_t>> unitGridSlots;
    std::vector<InvocationExtent> invocationExtents;
    TapeUse tape;
    uint64_t resultsSize{};
    // Their slots are linked slots.
    std::vector<ResultUse> results;
    hal::PassOffsets pass;
    std::vector<hal::RecordAddress> recordAddresses;
    hal::LeafCopy copy;
    // The linked slots and transients it accesses.
    std::vector<BufferAccess> accesses;
    // The earlier operations it must follow, ascending: each buffer's last writer before a read, and its last writer
    // and the reads since before a write. The CPU and CUDA run operations in order, so only recordings that issue
    // barriers read these. Arena aliasing adds no entries: transients sharing bytes have disjoint intervals, and a
    // recording that packs the arena orders the last access of one before the first of the next.
    std::vector<uint32_t> after;
};

struct LinkedSchedule {
    std::vector<hal::ExecutableSource> executables;
    std::vector<LinkedDispatch> operations;
    // The nodeOperationEnds of each fragment in node order, in linked operation indices.
    std::vector<uint32_t> nodeOperationEnds;
    // Each node's argument record sits at its node offset, so its slots and fields keep their fragment offsets plus
    // that offset.
    std::vector<uint64_t> nodeOffsets;
    // Per node and node slot, the linked slot it is, or kNoSlot for an internal connection.
    std::vector<std::vector<uint32_t>> nodeSlots;
    std::vector<VernonScheduleSlot> slots;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<std::vector<Extent>> extents;
    std::vector<VernonScheduleField> fields;
    // Per buffer slot and dimension, the field of its first binding that holds the extent the caller wrote.
    std::vector<std::vector<VernonScheduleField>> extentFields;
    std::vector<Transient> transients;
    std::vector<VernonScheduleField> transientFields;
    std::vector<uint8_t> constants;
    std::vector<ConstantAddress> constantAddresses;
    uint64_t recordSize{};
    std::vector<uint8_t> recordTemplate;
    bool callerGrid{};
};

// Links the nodes in order, each node's operations in its fragment's order, in O(operations + fields + sites). Linked
// slots are the slots of each node in order, a connection's at its first site. A caller grid links only in a graph of
// one node.
[[nodiscard]] Result<LinkedSchedule, std::string> link(const std::vector<const Fragment *> &nodes,
                                                       const std::vector<Connection> &connections = {});

// Offsets in one arena for transients of the given byte sizes: two whose operation intervals overlap get disjoint
// ranges, so disjoint ones may share bytes. Largest first, each at the lowest offset clear of the overlapping ones
// already placed; every offset is a multiple of `alignment`.
struct ArenaLayout {
    std::vector<uint64_t> offsets;
    uint64_t size{};
};

[[nodiscard]] ArenaLayout packArena(const std::vector<Transient> &transients, const std::vector<uint64_t> &sizes,
                                    uint64_t alignment);

} // namespace vernon::schedule

#endif
