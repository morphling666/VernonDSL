#ifndef VERNON_RUNTIME_DISPATCH_CONTRACT_H
#define VERNON_RUNTIME_DISPATCH_CONTRACT_H

// Residual launch constraints of the ordinary-write injectivity proof:
// specs/compiler/invocation_index_ownership.md §2

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>

namespace vernon::runtime {

struct DispatchContract {
    bool unitGridAxes[3]{false, false, false};
    bool requiresUnitWorkgroup{};
};

inline bool parseDispatchContract(const nlohmann::json &entry, DispatchContract &output, std::string &error) {
    output = {};
    const auto contract = entry.find("dispatch_contract");
    if (contract == entry.end() || !contract->is_object() || contract->size() != 2 ||
        !contract->contains("unit_grid_axes") || !(*contract)["unit_grid_axes"].is_array() ||
        !contract->contains("requires_unit_workgroup") || !(*contract)["requires_unit_workgroup"].is_boolean()) {
        error = "compute entry has a missing or malformed dispatch contract";
        return false;
    }
    int64_t previous = -1;
    for (const nlohmann::json &axisValue : (*contract)["unit_grid_axes"]) {
        if ((!axisValue.is_number_integer() && !axisValue.is_number_unsigned())) {
            error = "compute dispatch contract has an invalid unit grid axis";
            return false;
        }
        const int64_t axis = axisValue.get<int64_t>();
        if (axis < 0 || axis >= 3 || axis <= previous) {
            error = "compute dispatch contract has an invalid unit grid axis";
            return false;
        }
        output.unitGridAxes[axis] = true;
        previous = axis;
    }
    output.requiresUnitWorkgroup = (*contract)["requires_unit_workgroup"].get<bool>();
    if (output.requiresUnitWorkgroup &&
        (!output.unitGridAxes[0] || !output.unitGridAxes[1] || !output.unitGridAxes[2])) {
        error = "unit-workgroup dispatch contract must constrain every grid axis";
        return false;
    }
    return true;
}

inline bool validateDispatchContract(const DispatchContract &contract, const uint32_t grid[3],
                                     const uint32_t workgroup[3], std::string &error) {
    for (unsigned axis = 0; axis < 3; ++axis) {
        if (!grid[axis] || !workgroup[axis]) {
            error = "compute dispatch dimensions must be positive";
            return false;
        }
        if (contract.unitGridAxes[axis] && grid[axis] != 1) {
            error = "compute dispatch grid axis " + std::to_string(axis) + " must equal 1";
            return false;
        }
        if (contract.requiresUnitWorkgroup && workgroup[axis] != 1) {
            error = "compute dispatch workgroup axis " + std::to_string(axis) + " must equal 1";
            return false;
        }
    }
    return true;
}

} // namespace vernon::runtime

#endif
