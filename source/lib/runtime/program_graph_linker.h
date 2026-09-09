#ifndef VERNON_RUNTIME_PROGRAM_GRAPH_LINKER_H
#define VERNON_RUNTIME_PROGRAM_GRAPH_LINKER_H

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
    std::filesystem::path bundleRoot;
};

struct ProgramGraphConnection {
    ProgramGraphBoundaryKey source;
    ProgramGraphBoundaryKey destination;
    bool storageConnection{};
};

struct ProgramGraphGraphicsKey {
    uint32_t node{};
    uint32_t localNode{};

    bool operator<(const ProgramGraphGraphicsKey &other) const {
        return node != other.node ? node < other.node : localNode < other.localNode;
    }
};

struct ProgramGraphExport {
    ProgramGraphBoundaryKey boundary;
    std::string path;
};

struct LinkedProgramDeployment {
    std::string id;
    ProgramVariantDeployment deployment;
    std::map<ProgramGraphBoundaryKey, uint32_t> boundarySlots;
    std::map<ProgramGraphGraphicsKey, uint32_t> graphicsNodes;
};

bool sameProgramGraphBoundaryContract(const program::BoundarySlot &left, const program::BoundarySlot &right);

bool linkProgramGraph(const std::vector<ProgramGraphNodeSource> &nodes,
                      const std::vector<ProgramGraphConnection> &connections,
                      const std::vector<ProgramGraphBoundaryKey> &retainedBoundaries,
                      const std::vector<ProgramGraphExport> &exports, LinkedProgramDeployment &linked,
                      program::Diagnostic &diagnostic);

} // namespace vernon::runtime

#endif
