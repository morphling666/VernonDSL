#include "VernonRuntime.h"
#include "runtime/capi/boundary.h"
#include "runtime/capi/program_state.h"
#include "runtime/program_execution/program_boundary_contract.h"
#include "runtime/stage_binding_plan.h"

#include <algorithm>
#include <iterator>
#include <string_view>
#include <vector>

using namespace vernon::runtime;
using namespace vernon::runtime::capi;

namespace {

VernonValueLayoutView valueLayoutView(const ValueLayout &layout) {
    return {sizeof(VernonValueLayoutView),
            layout.byteSize,
            layout.alignment,
            {layout.layoutHash.data(), layout.layoutHash.size()},
            layout.abiLeaves.empty() ? nullptr : layout.abiLeaves.data(),
            layout.abiLeaves.size()};
}

bool fillBoundaryParameterView(const VernonProgramExecutable &pipeline, const program::BoundarySlot &source,
                               VernonProgramParameterView &destination) {
    const ValueLayout *layout = boundaryLayoutView(pipeline, source);
    if ((source.category == program::BoundaryCategory::Value ||
         source.category == program::BoundaryCategory::StorageView) &&
        (!source.layout || !layout))
        return false;
    destination = {source.id,
                   {source.path.data(), source.path.size()},
                   program_execution::boundaryArgumentKind(source.category),
                   layout ? valueLayoutView(*layout) : VernonValueLayoutView{},
                   program_execution::boundaryValueAccess(source.access),
                   static_cast<uint32_t>(source.outerShape.size()),
                   source.outerShape.empty() ? nullptr : source.outerShape.data()};
    return true;
}

VernonStatus fillBoundaryImageConstraintView(const program::BoundarySlot &source,
                                             VernonProgramImageConstraintView &destination) {
    if (source.category != program::BoundaryCategory::Texture || !source.storage ||
        source.storage->descriptorKind != program::StorageDescriptorKind::Image)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const auto dimension = artifactTextureDimension(source.storage->image.dimension);
    if (!dimension)
        return VERNON_STATUS_PARSE_ERROR;
    destination.dimension = *dimension;
    const auto hasUsage = [&](std::string_view usage) {
        return std::find(source.storage->image.usage.begin(), source.storage->image.usage.end(), usage) !=
               source.storage->image.usage.end();
    };
    destination.binding_role = hasUsage("color_attachment")           ? VERNON_IMAGE_BINDING_COLOR_ATTACHMENT
                               : hasUsage("depth_stencil_attachment") ? VERNON_IMAGE_BINDING_DEPTH_STENCIL_ATTACHMENT
                               : source.access == program::BoundaryAccess::Read ? VERNON_IMAGE_BINDING_SAMPLED
                                                                                : VERNON_IMAGE_BINDING_STORAGE;
    destination.sample_result_class = VERNON_IMAGE_SAMPLE_FLOAT;
    if (destination.binding_role == VERNON_IMAGE_BINDING_STORAGE) {
        const auto format = artifactStorageFormat(source.storage->image.format);
        if (!format)
            return VERNON_STATUS_PARSE_ERROR;
        destination.storage_format = *format;
    } else {
        destination.storage_format = static_cast<VernonTextureFormat>(0);
    }
    std::fill(std::begin(destination.reserved), std::end(destination.reserved), 0);
    return VERNON_STATUS_OK;
}

} // namespace

size_t vernonRuntimeProgramExecutableGetParameterCount(const VernonProgramExecutable *pipeline) {
    return runtimeValueBoundary<size_t>(pipeline ? pipeline->context : nullptr, 0, [&] {
        if (!pipeline)
            return size_t{0};
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return size_t{0};
        return pipeline->programGraphId ? pipeline->publicParameterSlots.size()
                                        : publicBoundaryCount(*executableProgram(*pipeline));
    });
}

size_t vernonRuntimeProgramExecutableGetMutationCapacity(const VernonProgramExecutable *pipeline) {
    return runtimeValueBoundary<size_t>(pipeline ? pipeline->context : nullptr, 0, [&] {
        if (!pipeline)
            return size_t{0};
        auto pipelinePin = pipeline->lifecycle.pin();
        return pipelinePin.isOk() ? mutationCapacity(*pipeline) : size_t{0};
    });
}

VernonStatus vernonRuntimeProgramExecutableGetParameterByIndex(const VernonProgramExecutable *pipeline, size_t index,
                                                               VernonProgramParameterView *parameter) {
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (!pipeline || !parameter)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return vernon::toVernonStatus(pipelinePin.error());
        const program::BoundarySlot *slot = parameterBoundaryAt(*pipeline, index);
        if (!slot)
            return VERNON_STATUS_INVALID_ARGUMENT;
        return fillBoundaryParameterView(*pipeline, *slot, *parameter) ? VERNON_STATUS_OK : VERNON_STATUS_PARSE_ERROR;
    });
}

VernonStatus vernonRuntimeProgramExecutableFindParameter(const VernonProgramExecutable *pipeline, VernonStringView name,
                                                         VernonProgramParameterView *parameter) {
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (!pipeline || !parameter || (name.size && !name.data))
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return vernon::toVernonStatus(pipelinePin.error());
        const program::BoundarySlot *slot = findParameterBoundary(*pipeline, name);
        if (!slot)
            return VERNON_STATUS_INVALID_ARGUMENT;
        return fillBoundaryParameterView(*pipeline, *slot, *parameter) ? VERNON_STATUS_OK : VERNON_STATUS_PARSE_ERROR;
    });
}

VernonStatus vernonRuntimeProgramExecutableGetParameterValueLeaf(const VernonProgramExecutable *pipeline,
                                                                 VernonStringView parameterName, size_t leafIndex,
                                                                 VernonProgramValueLeafView *leaf) {
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (!pipeline || !leaf || leaf->struct_size < sizeof(VernonProgramValueLeafView) ||
            (parameterName.size && !parameterName.data))
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return vernon::toVernonStatus(pipelinePin.error());
        const program::BoundarySlot *slot = findParameterBoundary(*pipeline, parameterName);
        const ValueLayout *layout = slot ? boundaryLayoutView(*pipeline, *slot) : nullptr;
        if (!slot || !layout ||
            (slot->category != program::BoundaryCategory::Value &&
             slot->category != program::BoundaryCategory::StorageView) ||
            leafIndex >= layout->leaves.size())
            return VERNON_STATUS_INVALID_ARGUMENT;
        const ValueLeaf &source = layout->leaves[leafIndex];
        leaf->value = layout->abiLeaves[leafIndex];
        leaf->path = source.abiPath.empty() ? nullptr : source.abiPath.data();
        leaf->path_count = source.abiPath.size();
        leaf->static_shape = source.shape.empty() ? nullptr : source.shape.data();
        leaf->static_rank = static_cast<uint32_t>(source.shape.size());
        return VERNON_STATUS_OK;
    });
}

size_t vernonRuntimeProgramExecutableGetBoundaryCount(const VernonProgramExecutable *pipeline,
                                                      VernonProgramBoundaryRole boundary) {
    return runtimeValueBoundary<size_t>(pipeline ? pipeline->context : nullptr, 0, [&] {
        const auto role = reflectedBoundaryRole(boundary);
        if (!pipeline || !role)
            return size_t{0};
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return size_t{0};
        const program::Program &program = *executableProgram(*pipeline);
        return static_cast<size_t>(
            std::count_if(program.abi.boundarySlots.begin(), program.abi.boundarySlots.end(),
                          [&](const program::BoundarySlot &slot) { return slot.role == *role; }));
    });
}

VernonStatus vernonRuntimeProgramExecutableGetBoundaryByIndex(const VernonProgramExecutable *pipeline,
                                                              VernonProgramBoundaryRole boundary, size_t index,
                                                              VernonProgramParameterView *parameter) {
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        const auto role = reflectedBoundaryRole(boundary);
        if (!pipeline || !role || !parameter)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return vernon::toVernonStatus(pipelinePin.error());
        const program::BoundarySlot *slot = boundaryAt(*executableProgram(*pipeline), *role, index);
        if (!slot)
            return VERNON_STATUS_INVALID_ARGUMENT;
        return fillBoundaryParameterView(*pipeline, *slot, *parameter) ? VERNON_STATUS_OK : VERNON_STATUS_PARSE_ERROR;
    });
}

VernonStatus vernonRuntimeProgramExecutableFindBoundary(const VernonProgramExecutable *pipeline,
                                                        VernonProgramBoundaryRole boundary, VernonStringView name,
                                                        VernonProgramParameterView *parameter) {
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        const auto role = reflectedBoundaryRole(boundary);
        if (!pipeline || !role || !parameter || (name.size && !name.data))
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return vernon::toVernonStatus(pipelinePin.error());
        const program::Program &program = *executableProgram(*pipeline);
        const auto found = std::find_if(
            program.abi.boundarySlots.begin(), program.abi.boundarySlots.end(),
            [&](const program::BoundarySlot &slot) { return slot.role == *role && stringViewEquals(name, slot.path); });
        if (found == program.abi.boundarySlots.end())
            return VERNON_STATUS_INVALID_ARGUMENT;
        return fillBoundaryParameterView(*pipeline, *found, *parameter) ? VERNON_STATUS_OK : VERNON_STATUS_PARSE_ERROR;
    });
}

VernonStatus vernonRuntimeProgramExecutableGetBoundaryValueLeaf(const VernonProgramExecutable *pipeline,
                                                                VernonProgramBoundaryRole boundary, uint32_t slotId,
                                                                size_t leafIndex, VernonProgramValueLeafView *leaf) {
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        const auto role = reflectedBoundaryRole(boundary);
        if (!pipeline || !role || !leaf || leaf->struct_size < sizeof(*leaf))
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return vernon::toVernonStatus(pipelinePin.error());
        const program::Program &program = *executableProgram(*pipeline);
        if (slotId >= program.abi.boundarySlots.size())
            return VERNON_STATUS_INVALID_ARGUMENT;
        const program::BoundarySlot &slot = program.abi.boundarySlots[slotId];
        const ValueLayout *layout = boundaryLayoutView(*pipeline, slot);
        if (slot.id != slotId || slot.role != *role || !layout || leafIndex >= layout->leaves.size())
            return VERNON_STATUS_INVALID_ARGUMENT;
        const ValueLeaf &source = layout->leaves[leafIndex];
        leaf->value = layout->abiLeaves[leafIndex];
        leaf->path = source.abiPath.empty() ? nullptr : source.abiPath.data();
        leaf->path_count = source.abiPath.size();
        leaf->static_shape = source.shape.empty() ? nullptr : source.shape.data();
        leaf->static_rank = static_cast<uint32_t>(source.shape.size());
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramExecutableGetImageConstraintByParameterIndex(
    const VernonProgramExecutable *pipeline, size_t parameterIndex, VernonProgramImageConstraintView *constraint) {
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (!pipeline || !constraint || constraint->struct_size < sizeof(VernonProgramImageConstraintView))
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return vernon::toVernonStatus(pipelinePin.error());
        const program::BoundarySlot *slot = parameterBoundaryAt(*pipeline, parameterIndex);
        return slot ? fillBoundaryImageConstraintView(*slot, *constraint) : VERNON_STATUS_INVALID_ARGUMENT;
    });
}

VernonStatus vernonRuntimeProgramExecutableFindImageConstraint(const VernonProgramExecutable *pipeline,
                                                               VernonStringView parameterName,
                                                               VernonProgramImageConstraintView *constraint) {
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (!pipeline || !constraint || constraint->struct_size < sizeof(VernonProgramImageConstraintView) ||
            (parameterName.size && !parameterName.data))
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return vernon::toVernonStatus(pipelinePin.error());
        const program::BoundarySlot *slot = findParameterBoundary(*pipeline, parameterName);
        return slot ? fillBoundaryImageConstraintView(*slot, *constraint) : VERNON_STATUS_INVALID_ARGUMENT;
    });
}

size_t vernonRuntimeProgramExecutableGetGraphicsNodeCount(const VernonProgramExecutable *pipeline) {
    return runtimeValueBoundary<size_t>(pipeline ? pipeline->context : nullptr, 0, [&] {
        if (!pipeline)
            return size_t{0};
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return size_t{0};
        std::vector<const program::Node *> nodes;
        collectForwardGraphicsNodes(pipeline, nodes);
        return nodes.size();
    });
}

VernonStatus vernonRuntimeProgramExecutableGetGraphicsControlsByIndex(const VernonProgramExecutable *pipeline,
                                                                      size_t index,
                                                                      VernonProgramGraphicsControlsView *output) {
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (!pipeline || !output)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return vernon::toVernonStatus(pipelinePin.error());
        std::vector<const program::Node *> nodes;
        collectForwardGraphicsNodes(pipeline, nodes);
        if (index >= nodes.size())
            return VERNON_STATUS_INVALID_ARGUMENT;
        const program::Node &node = *nodes[index];
        const program::GraphicsOperation &graphics = program::graphicsOperation(node);
        *output = {sizeof(VernonProgramGraphicsControlsView), node.id, graphics.renderPassControl,
                   graphics.drawCommandControl, graphics.dynamicStateControl};
        return VERNON_STATUS_OK;
    });
}

void vernonRuntimeProgramExecutableDestroy(VernonProgramExecutable *pipeline) {
    runtimeVoidBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (!pipeline)
            return;
        auto destruction = pipeline->lifecycle.beginDestroy();
        if (destruction.isErr()) {
            fail(pipeline->context, "Program executable has an active operation",
                 vernon::toVernonStatus(destruction.error()));
            return;
        }
        auto close = pipeline->instanceOwner.beginClose();
        if (close.isErr()) {
            (void)destruction.value().rollback();
            fail(pipeline->context, "Program executable still owns live instances",
                 vernon::toVernonStatus(close.error()));
            return;
        }
        if (close.value().commit().isErr() || destruction.value().commit().isErr())
            vernon::resultContractViolation();
        delete pipeline;
    });
}

VernonStringView vernonRuntimeProgramExecutableGetId(const VernonProgramExecutable *pipeline) {
    return runtimeValueBoundary(pipeline ? pipeline->context : nullptr, VernonStringView{nullptr, 0}, [&] {
        if (!pipeline)
            return VernonStringView{nullptr, 0};
        auto pin = pipeline->lifecycle.pin();
        return pin.isOk() ? VernonStringView{pipeline->id.data(), pipeline->id.size()} : VernonStringView{nullptr, 0};
    });
}
