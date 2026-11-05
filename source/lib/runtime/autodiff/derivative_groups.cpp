#include "runtime/autodiff/derivative_groups.h"

#include "runtime/autodiff/autodiff_metadata.h"
#include "runtime/pipeline_metadata.h"
#include "runtime/resolved_execution_plan.h"
#include "runtime/runtime_state.h"
#include "runtime/target_binding_plan.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vernon::runtime::ad {

size_t dtypeSize(VernonDataType dtype) {
    switch (dtype) {
    case VERNON_DATA_BOOL:
    case VERNON_DATA_U8:
        return 1;
    case VERNON_DATA_F16:
        return 2;
    case VERNON_DATA_I32:
    case VERNON_DATA_U32:
    case VERNON_DATA_F32:
        return sizeof(float);
    case VERNON_DATA_F64:
        return sizeof(double);
    default:
        return 0;
    }
}

bool appendValueLayoutAbi(const ValueLayout &layout, const std::vector<uint64_t> &shape, const std::string &rootPath,
                          std::vector<ValueAbi> &values, std::string &error) {
    if (layout.leaves.empty())
        return error = "autodiff parameter has no canonical Value layout", false;
    size_t elementCount = 1;
    bool dynamicShape = false;
    for (uint64_t extent : shape) {
        if (!extent) {
            dynamicShape = true;
            continue;
        }
        if (elementCount > std::numeric_limits<size_t>::max() / static_cast<size_t>(extent))
            return error = "autodiff parameter shape overflows", false;
        elementCount *= static_cast<size_t>(extent);
    }
    for (const ValueLeaf &leaf : layout.leaves) {
        const std::optional<VernonDataType> dtype = pipelineDataType(leaf.dtype);
        size_t bytes = dtype ? dtypeSize(*dtype) : 0;
        if (!bytes || !leaf.scalarCount ||
            bytes > std::numeric_limits<size_t>::max() / static_cast<size_t>(leaf.scalarCount))
            return error = "autodiff parameter leaf ABI is invalid", false;
        bytes *= static_cast<size_t>(leaf.scalarCount);
        if (!dynamicShape) {
            if (elementCount && bytes > std::numeric_limits<size_t>::max() / elementCount)
                return error = "autodiff parameter leaf byte size overflows", false;
            bytes *= elementCount;
        }
        std::string path = canonicalValueLeafPath(rootPath, leaf);
        std::vector<uint64_t> logicalShape = shape;
        logicalShape.insert(logicalShape.end(), leaf.shape.begin(), leaf.shape.end());
        ValueAbi candidate{std::move(path), *dtype, bytes, std::max<size_t>(layout.alignment, 1),
                           std::move(logicalShape)};
        const auto existing = std::find_if(values.begin(), values.end(),
                                           [&](const ValueAbi &value) { return value.path == candidate.path; });
        if (existing == values.end()) {
            values.push_back(std::move(candidate));
            continue;
        }
        if (existing->dtype != candidate.dtype || existing->byteSize != candidate.byteSize ||
            existing->alignment != candidate.alignment || existing->logicalShape != candidate.logicalShape)
            return error = "autodiff parameters define conflicting ABI values for '" + candidate.path + "'", false;
    }
    return true;
}

namespace {

bool selects(const std::vector<program::LayoutPathComponent> &projection, const ValueLeaf &leaf) {
    if (leaf.path.size() < projection.size())
        return false;
    for (size_t component = 0; component < projection.size(); ++component) {
        const auto &expected = projection[component];
        const auto &actual = leaf.path[component];
        if (expected.index ? actual.field || *expected.index != actual.index
                           : !actual.field || expected.field != *actual.field)
            return false;
    }
    return true;
}

// The leaves `projection` selects from `slot`, with the projection's path removed, and the leaf ABIs they occupy.
bool projectedLeaves(const program::BoundarySlot &slot, const program::DerivativeProjection &projection,
                     std::vector<std::string> &paths, std::vector<ValueAbi> &values, std::string &error) {
    if (!slot.layout)
        return error = "ProgramABI derivative projection has no canonical ValueLayout", false;
    const ValueLayout layout = program::materializeValueLayout(*slot.layout, slot.logicalType);
    ValueLayout projected = layout;
    projected.leaves.clear();
    for (const ValueLeaf &leaf : layout.leaves) {
        if (!selects(projection.valuePath, leaf))
            continue;
        ValueLeaf selected = leaf;
        selected.path.erase(selected.path.begin(),
                            selected.path.begin() + static_cast<ptrdiff_t>(projection.valuePath.size()));
        paths.push_back(canonicalValueLeafPath(slot.path, selected));
        projected.leaves.push_back(std::move(selected));
    }
    if (projected.leaves.empty())
        return error = "ProgramABI derivative projection selects no canonical Value leaf", false;
    rebuildValueLayoutPathViews(projected);
    const size_t begin = values.size();
    if (!appendValueLayoutAbi(projected, slot.outerShape, slot.path, values, error))
        return false;
    if (values.size() - begin != projected.leaves.size())
        return error = "Program aggregate ABI contains duplicate leaf paths", false;
    return true;
}

} // namespace

bool resolveDerivativeGroups(VernonProgramExecutable &executable) {
    if (!executable.context)
        return false;
    const program::Program &program = executable.executionPlan->resolvedProgram->program;
    std::string error;
    const auto fail = [&]() {
        invocationDiagnostic(*executable.context) = std::move(error);
        return false;
    };
    if (!program::findGraph(program, "forward"))
        return error = "Program execution topology requires a forward graph", fail();
    std::vector<AutodiffDerivativeGroup> groups;
    std::vector<ValueAbi> gradients;
    std::vector<ValueAbi> cotangents;
    for (const program::BoundaryRole role : {program::BoundaryRole::Gradient, program::BoundaryRole::Cotangent}) {
        const size_t first = groups.size();
        for (const program::DerivativeProjection &projection : program.abi.derivativeProjections) {
            const program::BoundarySlot &slot = program.abi.boundarySlots[projection.derivative.slot];
            if (slot.role != program::BoundaryRole::Gradient && slot.role != program::BoundaryRole::Cotangent)
                return error = "DerivativeProjection does not reference a derivative boundary", fail();
            if (slot.role != role)
                continue;
            AutodiffDerivativeGroup group{role == program::BoundaryRole::Gradient ? AutodiffDerivativeRole::Gradient
                                                                                  : AutodiffDerivativeRole::Cotangent,
                                          projection.derivative.path,
                                          {}};
            if (!projectedLeaves(slot, projection, group.leafPaths,
                                 role == program::BoundaryRole::Gradient ? gradients : cotangents, error))
                return fail();
            std::sort(group.leafPaths.begin(), group.leafPaths.end());
            groups.push_back(std::move(group));
        }
        std::sort(groups.begin() + static_cast<std::ptrdiff_t>(first), groups.end(),
                  [](const AutodiffDerivativeGroup &left, const AutodiffDerivativeGroup &right) {
                      return left.declaredPath < right.declaredPath;
                  });
    }
    if (!gradients.empty() && !cotangents.empty() && !validateAutodiffDerivativeGroups(groups, error))
        return fail();
    executable.programAutodiff.derivativeGroups = std::move(groups);
    return true;
}

} // namespace vernon::runtime::ad
