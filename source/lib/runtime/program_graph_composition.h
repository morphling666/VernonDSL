#ifndef VERNON_RUNTIME_PROGRAM_GRAPH_COMPOSITION_H
#define VERNON_RUNTIME_PROGRAM_GRAPH_COMPOSITION_H

#include "runtime_state.h"

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace vernon::runtime {

struct ProgramGraphBoundaryKey {
    uint32_t node{};
    uint32_t slot{};

    bool operator<(const ProgramGraphBoundaryKey &other) const {
        return node != other.node ? node < other.node : slot < other.slot;
    }
    bool operator==(const ProgramGraphBoundaryKey &other) const { return node == other.node && slot == other.slot; }
};

struct ProgramGraphNodeSource {
    uint32_t id{};
    std::string bundleId;
    std::string contentHash;
    const ProgramVariantDeployment *deployment{};
};

struct ProgramGraphConnection {
    ProgramGraphBoundaryKey source;
    ProgramGraphBoundaryKey destination;
    bool storageConnection{};
};

struct ProgramGraphExport {
    ProgramGraphBoundaryKey boundary;
    std::string path;
};

// The ABI a composed Program graph presents: its nodes' boundary slots no connection internalizes, those it exports
// renamed to their export paths and the others to "node/<id>/<path>". Connected boundaries share Value and Storage
// ids, so slots that alias across nodes alias in the composed ABI too. The Program has no graphs or Stages; its nodes
// run as their own executables.
struct ComposedProgramAbi {
    std::string id;
    program::Program program;
    // Per node boundary, the composed slot; a Storage version a later node continues maps to the chain's final slot.
    std::map<ProgramGraphBoundaryKey, uint32_t> boundarySlots;
};

bool composeProgramGraph(const std::vector<ProgramGraphNodeSource> &nodes,
                         const std::vector<ProgramGraphConnection> &connections,
                         const std::vector<ProgramGraphBoundaryKey> &retainedBoundaries,
                         const std::vector<ProgramGraphExport> &exports, ComposedProgramAbi &composed,
                         program::Diagnostic &diagnostic);

} // namespace vernon::runtime

#endif
