#ifndef VERNON_RUNTIME_PROGRAM_MANIFEST_H
#define VERNON_RUNTIME_PROGRAM_MANIFEST_H

#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vernon::runtime {

struct ValueLayout;

struct ProgramValueSlot {
    uint32_t id{UINT32_MAX};
    std::string name;
    std::string type;
    std::string dtype;
    std::vector<uint64_t> shape;
    std::shared_ptr<ValueLayout> valueLayout;
    std::optional<uint32_t> storage;
    bool external{};
    bool output{};
};

struct ProgramBufferExtent {
    bool isStatic{true};
    uint64_t staticValue{};
    uint32_t value{UINT32_MAX};
    uint32_t axis{};
};

struct ProgramStorageSlot {
    uint32_t id{UINT32_MAX};
    uint64_t byteLength{};
    uint32_t initialValue{UINT32_MAX};
    bool owned{};
    std::vector<ProgramBufferExtent> byteLengthExtents;
};

struct ProgramResourceUse {
    uint32_t value{UINT32_MAX};
    std::string access;
};

struct ProgramValueBinding {
    std::string parameter;
    uint32_t value{UINT32_MAX};
};

struct ProgramNode {
    uint32_t id{UINT32_MAX};
    std::string name;
    std::string kind;
    std::string stage;
    std::vector<uint32_t> operands;
    std::vector<uint32_t> results;
    std::vector<uint32_t> dependencies;
    std::vector<ProgramValueBinding> bindings;
    std::vector<ProgramResourceUse> resources;
    uint64_t grid[3]{1, 1, 1};
};

struct ProgramGraph {
    std::string name;
    std::string direction;
    std::vector<uint32_t> arguments;
    std::vector<uint32_t> captures;
    std::vector<uint32_t> results;
    std::vector<ProgramNode> nodes;
};

struct ProgramAdSignatureBinding {
    uint32_t value{UINT32_MAX};
    std::string path;
};

struct ProgramAdSignature {
    std::vector<ProgramAdSignatureBinding> inputs;
    std::vector<ProgramAdSignatureBinding> outputs;
    std::vector<ProgramAdSignatureBinding> cotangents;
    std::vector<ProgramAdSignatureBinding> gradients;
    std::vector<uint32_t> captures;
    bool declared{};
};

struct ExecutableProgram {
    std::vector<ProgramValueSlot> values;
    std::vector<ProgramStorageSlot> storages;
    std::vector<ProgramGraph> graphs;
    ProgramAdSignature adSignature;

    bool validate(const std::map<std::string, std::string> &stages, std::string &error) const;
    std::vector<uint32_t> backwardCaptures() const;
    std::vector<uint32_t> residualCaptures() const;
};

void markProgramGraphValues(const ProgramGraph &graph, std::vector<char> &live);

bool parseExecutableProgram(const nlohmann::json &value, ExecutableProgram &program, std::string &error);

} // namespace vernon::runtime

#endif // VERNON_RUNTIME_PROGRAM_MANIFEST_H
