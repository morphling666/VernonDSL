#include "runtime/schedule/linker.h"

#include <algorithm>
#include <numeric>
#include <utility>

namespace vernon::schedule {
namespace {

using Linked = Result<LinkedSchedule, std::string>;
using Checked = Result<void, std::string>;

constexpr uint64_t kNodeAlignment = 16;
constexpr uint64_t kConstantAlignment = 16;

Checked failCheck(std::string message) { return Checked{err(std::move(message))}; }

uint64_t alignUp(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

bool sameExtent(const Extent &left, const Extent &right) {
    return left.slot == right.slot &&
           (left.slot == kNoSlot ? left.value == right.value : left.dimension == right.dimension);
}

class GraphLinker {
public:
    GraphLinker(const std::vector<const Fragment *> &nodes, const std::vector<Connection> &connections)
        : nodes_(nodes), connections_(connections) {}

    Linked run() {
        if (nodes_.empty())
            return Linked{err(std::string("a graph has no nodes"))};
        for (const Fragment *node : nodes_) {
            if (!node)
                return Linked{err(std::string("a graph node has no fragment"))};
            if (auto valid = node->validate(); valid.isErr())
                return Linked{err(std::move(valid).error())};
            if (node->callerGrid && nodes_.size() != 1)
                return Linked{err(std::string("a kernel whose caller writes its grid links only alone"))};
        }
        // Classes come first: slots, fields, transients and operations all name a node slot through its class.
        // Extents follow the slots, since an extent names the linked slot or the internal connection it resolves to.
        for (auto step : {&GraphLinker::addClasses, &GraphLinker::layOut, &GraphLinker::addSlots,
                          &GraphLinker::addExtents, &GraphLinker::addFields, &GraphLinker::addTransients,
                          &GraphLinker::addOperations, &GraphLinker::orderOperations})
            if (auto done = (this->*step)(); done.isErr())
                return Linked{err(std::move(done).error())};
        linked_.callerGrid = nodes_.size() == 1 && nodes_.front()->callerGrid;
        linked_.extentFields.resize(linked_.slots.size());
        std::vector<std::vector<bool>> found(linked_.slots.size());
        for (size_t slot = 0; slot < linked_.slots.size(); ++slot) {
            linked_.extentFields[slot].resize(linked_.slots[slot].rank);
            found[slot].assign(linked_.slots[slot].rank, false);
        }
        for (const VernonScheduleField &field : linked_.fields) {
            if (field.kind != VERNON_SCHEDULE_FIELD_EXTENT || found[field.slot][field.dimension])
                continue;
            linked_.extentFields[field.slot][field.dimension] = field;
            found[field.slot][field.dimension] = true;
        }
        return Linked{ok(std::move(linked_))};
    }

private:
    // The node slots that are one buffer or scalar. `placed` is the first site in node order, whose offset is the
    // linked slot's; every other site repeats the slot's bytes as a field.
    struct Class {
        std::vector<NodeSlot> sites;
        bool internal{};
        bool assigned{};
        NodeSlot placed{};
        uint32_t slot{kNoSlot};
        uint32_t transient{kNoSlot};
        // 0 before its extents resolve, 1 while they do, 2 after.
        uint8_t resolution{};
        std::vector<Extent> extents;
    };

    const VernonScheduleSlot &slotOf(NodeSlot site) const { return nodes_[site.node]->slots[site.slot]; }
    Class &classOf(uint32_t node, uint32_t slot) { return classes_[classOf_[node][slot]]; }

    Checked addClasses() {
        classOf_.resize(nodes_.size());
        for (size_t node = 0; node < nodes_.size(); ++node)
            classOf_[node].assign(nodes_[node]->slots.size(), kNoSlot);
        for (const Connection &connection : connections_) {
            if (connection.sites.empty())
                return failCheck("a graph connection has no sites");
            for (const NodeSlot &site : connection.sites) {
                if (site.node >= nodes_.size() || site.slot >= nodes_[site.node]->slots.size())
                    return failCheck("a graph connection names no node slot");
                if (classOf_[site.node][site.slot] != kNoSlot)
                    return failCheck("a node slot is in two graph connections");
                classOf_[site.node][site.slot] = static_cast<uint32_t>(classes_.size());
            }
            const VernonScheduleSlot &first = slotOf(connection.sites.front());
            for (const NodeSlot &site : connection.sites) {
                const VernonScheduleSlot &slot = slotOf(site);
                if (slot.kind != first.kind || slot.dtype != first.dtype || slot.rank != first.rank ||
                    slot.element_size != first.element_size || slot.size != first.size)
                    return failCheck("connected node slots differ in kind, type or rank");
            }
            if (first.kind == VERNON_SCHEDULE_SLOT_TAPE && connection.sites.size() > 1)
                return failCheck("autodiff tapes do not connect across nodes");
            if (connection.internal && first.kind != VERNON_SCHEDULE_SLOT_BUFFER)
                return failCheck("an internal graph connection must be a buffer");
            Class added;
            added.sites = connection.sites;
            added.internal = connection.internal;
            classes_.push_back(std::move(added));
        }
        for (uint32_t node = 0; node < nodes_.size(); ++node)
            for (uint32_t slot = 0; slot < classOf_[node].size(); ++slot)
                if (classOf_[node][slot] == kNoSlot) {
                    classOf_[node][slot] = static_cast<uint32_t>(classes_.size());
                    Class added;
                    added.sites = {{node, slot}};
                    classes_.push_back(std::move(added));
                }
        return Checked{ok()};
    }

    // Each node's record at the next aligned offset, its executables and constants after the previous node's.
    Checked layOut() {
        uint64_t end = 0;
        for (const Fragment *node : nodes_) {
            const uint64_t offset = linked_.nodeOffsets.empty() ? 0 : alignUp(end, kNodeAlignment);
            linked_.nodeOffsets.push_back(offset);
            end = offset + node->recordSize;
            linked_.recordTemplate.resize(end, 0);
            std::copy(node->recordTemplate.begin(), node->recordTemplate.end(),
                      linked_.recordTemplate.begin() + static_cast<ptrdiff_t>(offset));
            executableBase_.push_back(static_cast<uint32_t>(linked_.executables.size()));
            linked_.executables.insert(linked_.executables.end(), node->executables.begin(), node->executables.end());
            const uint64_t constants = alignUp(linked_.constants.size(), kConstantAlignment);
            constantBase_.push_back(constants);
            linked_.constants.resize(constants, 0);
            linked_.constants.insert(linked_.constants.end(), node->constants.begin(), node->constants.end());
            operationBase_.push_back(operationCount_);
            for (const uint32_t operationEnd : node->nodeOperationEnds)
                linked_.nodeOperationEnds.push_back(operationCount_ + operationEnd);
            linked_.nodeLabels.insert(linked_.nodeLabels.end(), node->nodeLabels.begin(), node->nodeLabels.end());
            operationCount_ += static_cast<uint32_t>(node->operations.size());
            transientBase_.push_back(nodeTransients_);
            nodeTransients_ += static_cast<uint32_t>(node->transients.size());
        }
        linked_.recordSize = end;
        return Checked{ok()};
    }

    Checked addSlots() {
        uint32_t internal = 0;
        linked_.nodeSlots.resize(nodes_.size());
        for (uint32_t node = 0; node < nodes_.size(); ++node) {
            linked_.nodeSlots[node].resize(classOf_[node].size());
            for (uint32_t slot = 0; slot < classOf_[node].size(); ++slot) {
                Class &owner = classOf(node, slot);
                if (!owner.assigned) {
                    owner.assigned = true;
                    owner.placed = {node, slot};
                    if (owner.internal) {
                        owner.transient = nodeTransients_ + internal++;
                    } else {
                        owner.slot = static_cast<uint32_t>(linked_.slots.size());
                        VernonScheduleSlot placed = slotOf(owner.placed);
                        placed.offset += linked_.nodeOffsets[node];
                        for (const NodeSlot &site : owner.sites)
                            placed.access |= slotOf(site).access;
                        linked_.slots.push_back(placed);
                    }
                }
                linked_.nodeSlots[node][slot] = owner.slot;
            }
        }
        return Checked{ok()};
    }

    // An extent of node `node` in linked terms: static, a linked slot's, or what an internal connection's resolves to.
    Result<Extent, std::string> resolve(uint32_t node, const Extent &extent) {
        using Resolved = Result<Extent, std::string>;
        if (extent.slot == kNoSlot)
            return Resolved{ok(extent)};
        Class &owner = classOf(node, extent.slot);
        if (!owner.internal)
            return Resolved{ok(Extent{0, owner.slot, extent.dimension})};
        if (auto resolved = resolveClass(owner); resolved.isErr())
            return Resolved{err(std::move(resolved).error())};
        return Resolved{ok(owner.extents[extent.dimension])};
    }

    // A connection's extents: what its sites require, which must agree; a dimension no site constrains is free for a
    // slot, and unknown for an internal connection.
    Checked resolveClass(Class &owner) {
        if (owner.resolution == 2)
            return Checked{ok()};
        if (owner.resolution == 1)
            return failCheck("graph connection extents depend on themselves");
        owner.resolution = 1;
        const uint32_t rank = slotOf(owner.sites.front()).rank;
        std::vector<Extent> extents(rank);
        for (uint32_t dimension = 0; dimension < rank; ++dimension) {
            bool constrained = false;
            for (const NodeSlot &site : owner.sites) {
                const Extent &declared = nodes_[site.node]->extents[site.slot][dimension];
                if (declared.slot == site.slot && declared.dimension == dimension)
                    continue;
                auto resolved = resolve(site.node, declared);
                if (resolved.isErr())
                    return failCheck(std::move(resolved).error());
                const Extent extent = resolved.value();
                if (constrained && !sameExtent(extent, extents[dimension]))
                    return failCheck("connected node slots require different extents");
                extents[dimension] = extent;
                constrained = true;
            }
            if (!constrained) {
                if (owner.internal)
                    return failCheck("an internal graph connection has an extent no node slot declares");
                extents[dimension] = Extent{0, owner.slot, dimension};
            }
        }
        owner.extents = std::move(extents);
        owner.resolution = 2;
        return Checked{ok()};
    }

    Checked addExtents() {
        linked_.shapes.resize(linked_.slots.size());
        linked_.extents.resize(linked_.slots.size());
        for (Class &owner : classes_) {
            if (auto resolved = resolveClass(owner); resolved.isErr())
                return resolved;
            if (owner.internal)
                continue;
            std::vector<int64_t> shape = nodes_[owner.placed.node]->shapes[owner.placed.slot];
            for (const NodeSlot &site : owner.sites) {
                const std::vector<int64_t> &declared = nodes_[site.node]->shapes[site.slot];
                for (size_t dimension = 0; dimension < shape.size(); ++dimension) {
                    if (declared[dimension] < 0)
                        continue;
                    if (shape[dimension] >= 0 && shape[dimension] != declared[dimension])
                        return failCheck("connected node slots declare different static extents");
                    shape[dimension] = declared[dimension];
                }
            }
            linked_.shapes[owner.slot] = std::move(shape);
            linked_.extents[owner.slot] = owner.extents;
        }
        return Checked{ok()};
    }

    Checked addFields() {
        for (uint32_t node = 0; node < nodes_.size(); ++node) {
            const uint64_t base = linked_.nodeOffsets[node];
            for (const VernonScheduleField &field : nodes_[node]->fields) {
                const Class &owner = classOf(node, field.slot);
                VernonScheduleField moved = field;
                moved.offset += base;
                if (!owner.internal) {
                    moved.slot = owner.slot;
                    linked_.fields.push_back(moved);
                    continue;
                }
                moved.slot = owner.transient;
                if (field.kind == VERNON_SCHEDULE_FIELD_BYTES)
                    moved.kind = kTransientFieldBytes;
                linked_.transientFields.push_back(moved);
            }
            for (VernonScheduleField field : nodes_[node]->transientFields) {
                field.slot += transientBase_[node];
                field.offset += base;
                linked_.transientFields.push_back(field);
            }
            for (uint32_t slot = 0; slot < classOf_[node].size(); ++slot) {
                const Class &owner = classOf(node, slot);
                const VernonScheduleSlot &declared = nodes_[node]->slots[slot];
                const uint64_t at = base + declared.offset;
                if (owner.internal)
                    linked_.transientFields.push_back({owner.transient, VERNON_SCHEDULE_FIELD_ADDRESS, 0, 8, at});
                else if (owner.placed.node != node || owner.placed.slot != slot)
                    linked_.fields.push_back(
                        {owner.slot,
                         declared.kind == VERNON_SCHEDULE_SLOT_SCALAR || graphicsSlotSize(declared.kind)
                             ? static_cast<uint32_t>(VERNON_SCHEDULE_FIELD_VALUE)
                             : static_cast<uint32_t>(VERNON_SCHEDULE_FIELD_ADDRESS),
                         0, static_cast<uint32_t>(declared.size), at});
            }
            for (const ConstantAddress &address : nodes_[node]->constantAddresses)
                linked_.constantAddresses.push_back(
                    {address.recordOffset + base, address.constant + constantBase_[node]});
        }
        return Checked{ok()};
    }

    Checked addTransients() {
        for (uint32_t node = 0; node < nodes_.size(); ++node)
            for (Transient transient : nodes_[node]->transients) {
                for (Extent &extent : transient.extents) {
                    auto resolved = resolve(node, extent);
                    if (resolved.isErr())
                        return failCheck(std::move(resolved).error());
                    extent = resolved.value();
                }
                transient.first += operationBase_[node];
                transient.last += operationBase_[node];
                linked_.transients.push_back(std::move(transient));
            }
        linked_.transients.resize(nodeTransients_ + std::count_if(classes_.begin(), classes_.end(),
                                                                  [](const Class &owner) { return owner.internal; }));
        for (const Class &owner : classes_) {
            if (!owner.internal)
                continue;
            const VernonScheduleSlot &declared = slotOf(owner.placed);
            Transient &transient = linked_.transients[owner.transient];
            transient.dtype = declared.dtype;
            transient.elementSize = declared.element_size;
            transient.extents = owner.extents;
            transient.first = UINT32_MAX;
        }
        return Checked{ok()};
    }

    Checked addOperations() {
        for (uint32_t node = 0; node < nodes_.size(); ++node) {
            const uint64_t base = linked_.nodeOffsets[node];
            for (const Dispatch &dispatch : nodes_[node]->operations) {
                LinkedDispatch operation;
                operation.kind = dispatch.kind;
                operation.executable = dispatch.executable + executableBase_[node];
                operation.argumentsOffset = dispatch.argumentsOffset + base;
                operation.argumentsSize = dispatch.argumentsSize;
                operation.gridOffset = dispatch.kind == OperationKind::Dispatch ? dispatch.gridOffset + base : 0;
                operation.copy = dispatch.copy;
                for (const hal::RecordAddress &address : dispatch.recordAddresses)
                    operation.recordAddresses.push_back({address.at + base, address.target + base});
                if (dispatch.kind == OperationKind::Pass || dispatch.kind == OperationKind::Copy) {
                    operation.pass = dispatch.pass;
                    for (uint64_t &offset : operation.pass.parameters)
                        if (offset != UINT64_MAX)
                            offset += base;
                    for (hal::AttachmentRead &attachment : operation.pass.attachments)
                        if (attachment.renderPass != UINT64_MAX)
                            attachment.renderPass += base;
                    for (std::vector<hal::InternalExtent> &extents : operation.pass.internalExtents)
                        for (hal::InternalExtent &extent : extents)
                            if (extent.argument != UINT64_MAX)
                                extent.argument += base;
                    for (std::optional<hal::ComputeGrid> &launch : operation.pass.launches)
                        for (size_t axis = 0; launch && axis < 3; ++axis)
                            if (launch->argument[axis] != UINT64_MAX)
                                launch->argument[axis] += base;
                    for (uint64_t *offset :
                         {&operation.pass.renderPass, &operation.pass.draw, &operation.pass.dynamic,
                          &operation.pass.stream, &operation.pass.table, &operation.pass.grid.argument[0],
                          &operation.pass.grid.argument[1], &operation.pass.grid.argument[2]})
                        if (*offset != UINT64_MAX)
                            *offset += base;
                }
                std::copy_n(dispatch.unitGridAxes, 3, operation.unitGridAxes);
                for (const auto &[axis, slot] : dispatch.unitGridSlots)
                    operation.unitGridSlots.emplace_back(axis, classOf(node, slot).slot);
                for (InvocationExtent extent : dispatch.invocationExtents) {
                    extent.offset += base;
                    operation.invocationExtents.push_back(extent);
                }
                for (const Extent &extent : dispatch.gridElements) {
                    auto resolved = resolve(node, extent);
                    if (resolved.isErr())
                        return failCheck(std::move(resolved).error());
                    operation.gridElements.push_back(resolved.value());
                }
                if (dispatch.tape.present()) {
                    if (dispatch.tape.slot == kNoSlot)
                        return failCheck("kernels with autodiff tapes link only as nodes of a Program");
                    operation.tape = dispatch.tape;
                    operation.tape.slot = classOf(node, dispatch.tape.slot).slot;
                    operation.tape.shift(base);
                }
                operation.resultsSize = dispatch.resultsSize;
                for (const ResultUse &result : dispatch.results) {
                    if (result.slot == kNoSlot)
                        return failCheck("kernels with results link only as nodes of a Program");
                    const Class &owner = classOf(node, result.slot);
                    if (owner.internal)
                        return failCheck("kernel results that a later node reads are not linked yet");
                    operation.results.push_back(result);
                    operation.results.back().slot = owner.slot;
                }
                for (const BufferAccess &access : dispatch.accesses) {
                    if (access.transient) {
                        operation.accesses.push_back({true, access.index + transientBase_[node], access.access});
                        continue;
                    }
                    const Class &owner = classOf(node, access.index);
                    operation.accesses.push_back(owner.internal ? BufferAccess{true, owner.transient, access.access}
                                                                : BufferAccess{false, owner.slot, access.access});
                }
                linked_.operations.push_back(std::move(operation));
            }
        }
        // An internal connection lives from its first access to its last, and the first must write it.
        for (uint32_t index = 0; index < linked_.operations.size(); ++index)
            for (const BufferAccess &access : linked_.operations[index].accesses) {
                if (!access.transient || access.index < nodeTransients_)
                    continue;
                Transient &transient = linked_.transients[access.index];
                if (transient.first == UINT32_MAX) {
                    if (!(access.access & VERNON_SCHEDULE_ACCESS_WRITE))
                        return failCheck("an internal graph connection is read before a node writes it");
                    transient.first = index;
                }
                transient.last = index;
            }
        for (uint32_t index = nodeTransients_; index < linked_.transients.size(); ++index)
            if (linked_.transients[index].first == UINT32_MAX)
                return failCheck("an internal graph connection is accessed by no operation");
        return Checked{ok()};
    }

    Checked orderOperations() {
        struct Hazard {
            uint32_t writer{kNoSlot};
            std::vector<uint32_t> readers;
        };
        std::vector<Hazard> slots(linked_.slots.size());
        std::vector<Hazard> transients(linked_.transients.size());
        const auto hazard = [&](const BufferAccess &access) -> Hazard & {
            return access.transient ? transients[access.index] : slots[access.index];
        };
        for (uint32_t index = 0; index < linked_.operations.size(); ++index) {
            LinkedDispatch &operation = linked_.operations[index];
            for (const BufferAccess &access : operation.accesses) {
                const Hazard &state = hazard(access);
                if (state.writer != kNoSlot)
                    operation.after.push_back(state.writer);
                if (access.access & VERNON_SCHEDULE_ACCESS_WRITE)
                    operation.after.insert(operation.after.end(), state.readers.begin(), state.readers.end());
            }
            std::sort(operation.after.begin(), operation.after.end());
            operation.after.erase(std::unique(operation.after.begin(), operation.after.end()), operation.after.end());
            for (const BufferAccess &access : operation.accesses) {
                Hazard &state = hazard(access);
                if (access.access & VERNON_SCHEDULE_ACCESS_WRITE) {
                    state.writer = index;
                    state.readers.clear();
                } else if (state.readers.empty() || state.readers.back() != index) {
                    state.readers.push_back(index);
                }
            }
        }
        return Checked{ok()};
    }

    const std::vector<const Fragment *> &nodes_;
    const std::vector<Connection> &connections_;
    std::vector<Class> classes_;
    std::vector<std::vector<uint32_t>> classOf_;
    std::vector<uint32_t> executableBase_;
    std::vector<uint64_t> constantBase_;
    std::vector<uint32_t> operationBase_;
    std::vector<uint32_t> transientBase_;
    uint32_t operationCount_{};
    uint32_t nodeTransients_{};
    LinkedSchedule linked_;
};

} // namespace

Result<LinkedSchedule, std::string> link(const std::vector<const Fragment *> &nodes,
                                         const std::vector<Connection> &connections) {
    return GraphLinker(nodes, connections).run();
}

ArenaLayout packArena(const std::vector<Transient> &transients, const std::vector<uint64_t> &sizes,
                      uint64_t alignment) {
    ArenaLayout layout;
    layout.offsets.assign(transients.size(), 0);
    std::vector<size_t> order(transients.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) { return sizes[left] > sizes[right]; });
    struct Placed {
        uint64_t begin;
        uint64_t end;
    };
    std::vector<size_t> placed;
    std::vector<Placed> blocking;
    for (size_t index : order) {
        blocking.clear();
        for (size_t other : placed)
            if (transients[other].first <= transients[index].last && transients[index].first <= transients[other].last)
                blocking.push_back({layout.offsets[other], layout.offsets[other] + sizes[other]});
        std::sort(blocking.begin(), blocking.end(),
                  [](const Placed &left, const Placed &right) { return left.begin < right.begin; });
        uint64_t offset = 0;
        for (const Placed &range : blocking) {
            if (offset + sizes[index] <= range.begin)
                break;
            offset = std::max(offset, (range.end + alignment - 1) / alignment * alignment);
        }
        layout.offsets[index] = offset;
        layout.size = std::max(layout.size, offset + sizes[index]);
        placed.push_back(index);
    }
    return layout;
}

} // namespace vernon::schedule
