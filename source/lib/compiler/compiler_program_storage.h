#ifndef VERNON_COMPILER_PROGRAM_STORAGE_H
#define VERNON_COMPILER_PROGRAM_STORAGE_H

#include "compiler_program_graph.h"
#include "compiler_program_types.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace vernon::compiler {

struct ProgramArgumentSlot {
    std::string graph;
    int64_t slot{};
};

struct ProgramValueProducer {
    std::string graph;
    int64_t node{-1};
};

struct ProgramLogicalResource {
    int64_t before{};
    std::optional<int64_t> after;
    std::string access;
    int64_t storage{-1};
};

struct ProgramResourceIndex {
    std::map<int64_t, ProgramArgumentSlot> argumentSlots;
    std::map<std::string, std::string> graphDirections;
    std::map<int64_t, ProgramValueProducer> producerByValue;
    std::map<int64_t, std::string> allocationGraph;
    std::map<std::string, const CanonicalComputeStage *> compiledByRequest;
    std::map<std::string, std::map<int64_t, ProgramLogicalResource>> resourcesByStage;
    std::set<int64_t> resourceVersions;
    std::map<int64_t, int64_t> storageParents;
};

struct ProgramStorageOwnerId {
    int64_t value{};

    bool operator<(const ProgramStorageOwnerId &other) const { return value < other.value; }
    bool operator==(const ProgramStorageOwnerId &other) const { return value == other.value; }
};

struct ProgramStorageAliasPlan {
    std::map<int64_t, ProgramStorageOwnerId> ownerByValue;
    std::map<ProgramStorageOwnerId, std::vector<int64_t>> versionsByOwner;
};

struct ProgramStoragePlan {
    llvm::json::Array storages;
    llvm::json::Array values;
    std::map<int64_t, int64_t> storageByValue;
    ProgramStorageAliasPlan aliases;
};

bool isProgramTextureType(llvm::StringRef type);
bool isProgramSamplerType(llvm::StringRef type);
bool isProgramAdTapeType(llvm::StringRef type);
bool isProgramTensorViewType(llvm::StringRef type);
bool isValidProgramResourceAccess(llvm::StringRef access);

bool indexProgramResources(const llvm::json::Array &values, const std::vector<CanonicalProgramGraph> &selectedGraphs,
                           const std::vector<CanonicalComputeStage> &compiledStages, ProgramResourceIndex &index,
                           std::string &error);

bool planProgramStorageAliases(const std::map<int64_t, int64_t> &parents, ProgramStorageAliasPlan &plan,
                               std::string &error);

bool materializeProgramStoragePlan(const llvm::json::Array &rawValues,
                                   const std::vector<CanonicalProgramGraph> &selectedGraphs,
                                   ProgramResourceIndex &index, std::set<int64_t> &capturedValues,
                                   ProgramStoragePlan &plan, std::string &error);

int64_t programStorageRoot(const std::map<int64_t, int64_t> &parents, int64_t value);

bool isDynamicProgramExtent(const llvm::json::Value &value);

bool isCurrentGraphArgument(int64_t value, llvm::StringRef graph,
                            const std::map<int64_t, ProgramArgumentSlot> &argumentSlots);

int64_t resolveOwnedLikeSource(const llvm::json::Array &values, int64_t root, llvm::StringRef allocationGraph,
                               bool captureLegal, const std::map<int64_t, ProgramArgumentSlot> &argumentSlots,
                               const std::set<int64_t> &captures,
                               const std::map<int64_t, ProgramValueProducer> &producerByValue,
                               const std::map<int64_t, int64_t> &parents, std::string &error);

bool planOwnedDynamicExtents(const llvm::json::Array &values, int64_t root, llvm::StringRef allocationGraph,
                             bool captureLegal, const llvm::json::Array &shape,
                             const std::map<int64_t, ProgramArgumentSlot> &argumentSlots,
                             const std::set<int64_t> &captures,
                             const std::map<int64_t, ProgramValueProducer> &producerByValue,
                             const std::map<int64_t, int64_t> &parents, llvm::json::Array &extents, std::string &error);

} // namespace vernon::compiler

#endif
