#ifndef VERNON_RUNTIME_AUTODIFF_DERIVATIVE_GROUPS_H
#define VERNON_RUNTIME_AUTODIFF_DERIVATIVE_GROUPS_H

#include "VernonRuntime.h"
#include "runtime/stage_binding_plan.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct VernonProgramExecutable;

namespace vernon::runtime::ad {

// One scalar leaf of a Program boundary: its dotted path from the boundary, and the bytes it occupies.
struct ValueAbi {
    std::string path;
    VernonDataType dtype{};
    size_t byteSize{};
    size_t alignment{1};
    std::vector<uint64_t> logicalShape;
};

inline std::string canonicalValueLeafPath(const std::string &root, const ValueLeaf &leaf) {
    std::string result = root;
    for (const ValuePathComponent &component : leaf.path) {
        if (!result.empty())
            result.push_back('.');
        result += component.field ? *component.field : std::to_string(component.index);
    }
    return result;
}

size_t dtypeSize(VernonDataType dtype);
bool appendValueLayoutAbi(const ValueLayout &layout, const std::vector<uint64_t> &shape, const std::string &rootPath,
                          std::vector<ValueAbi> &values, std::string &error);

// Fills the executable's derivative groups from the ProgramABI's derivative projections: its gradients sorted by
// declared path, then its cotangents. A Program that declares both checks them against the boundaries' leaf ABIs.
bool resolveDerivativeGroups(VernonProgramExecutable &executable);

} // namespace vernon::runtime::ad

#endif
