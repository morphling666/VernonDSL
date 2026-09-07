#include "materialized_node_frame.h"

#include "physical_buffer_view.h"
#include "program_tensor_copy.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_state.h"
#include "runtime/tensor_bridge.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace vernon::runtime::program_execution {
namespace {

VernonValueAccess valueAccess(const std::string &access) {
    if (access == "read")
        return VERNON_ACCESS_READ;
    if (access == "write")
        return VERNON_ACCESS_WRITE;
    return VERNON_ACCESS_READ_WRITE;
}

std::optional<shape::DeclaredShape> logicalProjectionShape(const Parameter &parameter,
                                                           const program::TargetBinding *target) {
    const shape::DeclaredShape physical = shape::decodeRuntimeContractShape(parameter.shape);
    if (!target || !target->viewTransform)
        return physical;
    const auto &axes = target->viewTransform->axes;
    if (axes.size() != physical.size())
        return std::nullopt;
    std::vector<std::optional<shape::Extent>> logical;
    for (size_t physicalAxis = 0; physicalAxis < axes.size(); ++physicalAxis) {
        if (axes[physicalAxis].source != program::ViewAxisSource::LogicalAxis)
            continue;
        if (logical.size() <= axes[physicalAxis].logicalAxis)
            logical.resize(axes[physicalAxis].logicalAxis + 1);
        auto &extent = logical[axes[physicalAxis].logicalAxis];
        if (extent && *extent != physical[physicalAxis])
            return std::nullopt;
        extent = physical[physicalAxis];
    }
    shape::DeclaredShape result;
    result.reserve(logical.size());
    for (const auto &extent : logical) {
        if (!extent)
            return std::nullopt;
        result.push_back(*extent);
    }
    return result;
}

} // namespace

bool MaterializedNodeFrame::prepareHost(std::string &error) const {
    for (const HostCopy &copy : copiesBefore) {
        if (!copy.source || !copy.destination)
            return error = "node endpoint input transfer has no host backing", false;
        std::memcpy(copy.destination + copy.destinationOffset, copy.source + copy.sourceOffset, copy.size);
    }
    return true;
}

bool MaterializedNodeFrame::commitHost(std::string &error) const {
    for (const HostCopy &copy : copiesAfter) {
        if (!copy.source || !copy.destination)
            return error = "node endpoint result transfer has no host backing", false;
        std::memcpy(copy.destination + copy.destinationOffset, copy.source + copy.sourceOffset, copy.size);
    }
    return true;
}

bool materializeNodeFrame(const ProgramInvocationState &invocation, const program::Program &program,
                          const program::Node &node, const program::ResolvedNodePlan &nodePlan,
                          const ResolvePhysicalEndpoint &resolvePhysicalEndpoint, MaterializedNodeFrame &output,
                          std::string &error) {
    if (!nodePlan.stage)
        return error = "resolved Program stage has no physical implementation", false;
    const VernonStageExecutable &stage = *nodePlan.stage;
    if (nodePlan.projections.size() != stage.bindingProjection.parameters.size())
        return error = "resolved Program stage has an invalid endpoint plan", false;
    output = {};
    output.arguments.reserve(nodePlan.projections.size());
    output.shapes.reserve(nodePlan.projections.size());
    output.strides.reserve(nodePlan.projections.size());

    uint64_t dispatchInvocations = 1;
    if (program::executionKind(node) == program::ExecutionKind::Compute) {
        for (const program::ControlComponent &control : program::computeOperation(node).workgroups) {
            uint64_t extent{};
            if (!invocation.resolveControl(program, control, extent, error))
                return false;
            if (!extent || dispatchInvocations > std::numeric_limits<uint64_t>::max() / extent)
                return error = "Program dispatch invocation count overflows", false;
            dispatchInvocations *= extent;
        }
    }
    for (uint32_t extent : {stage.workgroupSize.x, stage.workgroupSize.y, stage.workgroupSize.z}) {
        if (!extent || dispatchInvocations > std::numeric_limits<uint64_t>::max() / extent)
            return error = "Program dispatch invocation count overflows", false;
        dispatchInvocations *= extent;
    }

    for (size_t parameterIndex = 0; parameterIndex < nodePlan.projections.size(); ++parameterIndex) {
        const Parameter &parameter = stage.bindingProjection.parameters[parameterIndex];
        const program::NodeEndpointProjection &binding = nodePlan.projections[parameterIndex];
        if (binding.value >= invocation.arguments().size() || binding.value >= program.values.size())
            return error = "resolved Program endpoint exceeds invocation Values", false;

        VernonProgramArgument controlImage{};
        const VernonProgramArgument *source = nullptr;
        const program::Value &programValue = program.values[binding.value];
        const bool requiresHostProjection =
            std::any_of(parameter.uses.begin(), parameter.uses.end(), [](const ParameterUse &use) {
                return !use.tensorViewDescriptor && (use.interfaceKind == "value" || use.interfaceKind == "result");
            });
        const VernonRuntimeProviderResourceReference *image =
            programValue.storage ? invocation.controlImage(*programValue.storage) : nullptr;
        if (image) {
            controlImage.slot = binding.value;
            controlImage.kind = VERNON_PROGRAM_IMAGE;
            controlImage.image.view = *image;
            source = &controlImage;
        } else if (requiresHostProjection && binding.value < invocation.values().size() &&
                   invocation.values()[binding.value].ownership == ProgramValueOwnership::BorrowedHost) {
            source = &invocation.values()[binding.value].argument;
        } else {
            source = resolvePhysicalEndpoint ? resolvePhysicalEndpoint(binding.value, binding.target)
                                             : invocation.argument(binding.value);
        }
        if (!source)
            return error = "resolved Program endpoint has no physical carrier", false;
        VernonProgramArgument materialized = *source;
        materialized.slot = parameter.slot;
        if (requiresHostProjection && materialized.kind == VERNON_PROGRAM_TENSOR &&
            materialized.tensor.storage == VERNON_TENSOR_RHI_RESOURCE)
            return error = "resolved Program host endpoint requires a planned "
                           "transfer before materialization",
                   false;
        if (materialized.kind == VERNON_PROGRAM_TENSOR) {
            materialized.tensor.access = valueAccess(binding.target.access);
            materialized.tensor.element_layout = pipelineValueLayout(binding.target.elementLayout);
        }

        output.shapes.emplace_back();
        output.strides.emplace_back();
        if (materialized.kind == VERNON_PROGRAM_TENSOR && materialized.tensor.rank) {
            const shape::DeclaredShape declared = shape::decodeRuntimeContractShape(programValue.shape);
            if (binding.value < invocation.values().size() && invocation.values()[binding.value].concreteShape) {
                output.shapes.back() = *invocation.values()[binding.value].concreteShape;
                output.strides.back() = invocation.values()[binding.value].strides;
            } else if (!shape::isConcrete(declared)) {
                return error = "resolved Program TensorView shape is not concrete", false;
            } else if (!materialized.tensor.shape || !materialized.tensor.byte_strides) {
                return error = "resolved Program TensorView has incomplete "
                               "logical layout",
                       false;
            } else {
                output.shapes.back().assign(materialized.tensor.shape,
                                            materialized.tensor.shape + materialized.tensor.rank);
                output.strides.back().assign(materialized.tensor.byte_strides,
                                             materialized.tensor.byte_strides + materialized.tensor.rank);
            }
            if (output.shapes.back().size() != output.strides.back().size())
                return error = "resolved Program TensorView shape has no strides", false;
            materialized.tensor.rank = static_cast<uint32_t>(output.shapes.back().size());
            materialized.tensor.shape = output.shapes.back().data();
            materialized.tensor.byte_strides = output.strides.back().data();
        }

        if (binding.target.carrier == program::TargetCarrier::UniformBuffer && !binding.target.endpointProjection &&
            materialized.kind == VERNON_PROGRAM_TENSOR && materialized.tensor.storage == VERNON_TENSOR_RHI_RESOURCE &&
            materialized.tensor.byte_size && nodePlan.stage->context) {
            VernonRhiBuffer canonicalBuffer{};
            if (!resolveBackendRhiBufferReference(*nodePlan.stage->context, materialized.tensor.resource,
                                                  canonicalBuffer))
                return error = "uniform endpoint has no canonical RHI backing", false;
            auto carrier = std::make_shared<DeviceBuffer>(*nodePlan.stage->context, materialized.tensor.byte_size);
            if (!carrier->valid() || !carrier->reference(materialized.tensor.resource))
                return error = "uniform endpoint carrier allocation failed", false;
            size_t canonicalOffset = 0;
            if (!checkedDeviceBufferOffset(materialized.tensor.resource.offset, materialized.tensor.byte_offset,
                                           canonicalOffset))
                return error = "uniform endpoint resource offset exceeds the host address space", false;
            output.deviceCopiesBefore.push_back(
                {canonicalBuffer, carrier->handle(), canonicalOffset, 0, materialized.tensor.byte_size});
            output.deviceEndpointCarriers.push_back(std::move(carrier));
            materialized.tensor.byte_offset = 0;
        }

        if (binding.target.semantic == program::CarrierSemantic::Tape) {
            if (binding.target.tapeCarrier &&
                (*binding.target.tapeCarrier == program_plan::TapeCarrier::ReplayStatus ||
                 *binding.target.tapeCarrier == program_plan::TapeCarrier::LaunchMetadata)) {
                if (const auto concrete = shape::concrete(binding.target.shape)) {
                    size_t elements = 0;
                    if (!shape::checkedElementCount(*concrete, elements) || !binding.target.elementLayout.byteSize ||
                        elements > std::numeric_limits<size_t>::max() / binding.target.elementLayout.byteSize ||
                        elements * binding.target.elementLayout.byteSize > materialized.tensor.byte_size)
                        return error = "resolved fixed tape carrier is smaller "
                                       "than its Stage ABI",
                               false;
                    materialized.tensor.byte_size = elements * binding.target.elementLayout.byteSize;
                }
            }
            PhysicalBufferView view;
            if (materialized.kind != VERNON_PROGRAM_TENSOR ||
                !materializePhysicalBufferView(binding.target.shape, binding.target.elementLayout,
                                               materialized.tensor.byte_size, view))
                return error = "resolved tape carrier has an incompatible "
                               "Stage-local view",
                       false;
            output.shapes.back() = std::move(view.shape);
            output.strides.back() = std::move(view.strides);
            materialized.tensor.rank = static_cast<uint32_t>(output.shapes.back().size());
            materialized.tensor.shape = output.shapes.back().empty() ? nullptr : output.shapes.back().data();
            materialized.tensor.byte_strides = output.strides.back().empty() ? nullptr : output.strides.back().data();
        }

        if (binding.target.endpointProjection) {
            const program::PhysicalEndpointProjection &projection = *binding.target.endpointProjection;
            const program::Value &slot = program.values[binding.value];
            if (materialized.kind != VERNON_PROGRAM_TENSOR || !binding.logicalLeaf || !slot.layout ||
                *binding.logicalLeaf >= slot.layout->leaves.size() || !projection.carrierByteSize ||
                !projection.leafByteSize || projection.leafByteOffset > projection.carrierByteSize ||
                projection.leafByteSize > projection.carrierByteSize - projection.leafByteOffset)
                return error = "resolved endpoint projection has an invalid "
                               "logical or physical carrier",
                       false;
            const ValueLayout logicalLayout = program::materializeValueLayout(*slot.layout, slot.type);
            const ValueLeaf &logicalLeaf = logicalLayout.leaves[*binding.logicalLeaf];
            const auto logicalDtype = pipelineDataType(logicalLeaf.dtype);
            const size_t logicalLeafSize =
                logicalDtype ? dataTypeSize(*logicalDtype) * static_cast<size_t>(logicalLeaf.scalarCount) : 0;
            if (!logicalDtype || logicalLeafSize != projection.leafByteSize)
                return error = "resolved endpoint leaf size disagrees with "
                               "the logical Value",
                       false;

            size_t elements = 0;
            std::vector<int64_t> canonicalStrides = output.strides.back();
            if (!shape::checkedElementCount(output.shapes.back(), elements) ||
                elements > std::numeric_limits<size_t>::max() / projection.carrierByteSize ||
                !shape::rowMajorByteStrides(output.shapes.back(), projection.carrierByteSize, output.strides.back()))
                return error = "physical endpoint carrier size overflows", false;
            const size_t byteSize = elements * projection.carrierByteSize;
            VernonTensorView canonical = materialized.tensor;
            canonical.byte_offset += logicalLeaf.byteOffset;
            canonical.element_layout.byte_size = static_cast<uint32_t>(projection.leafByteSize);
            canonical.element_layout.alignment =
                static_cast<uint32_t>(std::min(projection.leafByteSize, projection.carrierAlignment));
            canonical.byte_strides = canonicalStrides.empty() ? nullptr : canonicalStrides.data();

            std::vector<int64_t> leafCarrierStrides = output.strides.back();
            VernonTensorView leafCarrier = canonical;
            leafCarrier.byte_offset = projection.leafByteOffset;
            leafCarrier.byte_size = byteSize;
            leafCarrier.byte_strides = leafCarrierStrides.empty() ? nullptr : leafCarrierStrides.data();
            std::vector<ProgramTensorCopyRegion> regions;
            if (!planProgramTensorCopy(canonical, leafCarrier, regions, error))
                return false;
            materialized.tensor.byte_offset = 0;
            materialized.tensor.byte_size = byteSize;
            materialized.tensor.byte_strides = output.strides.back().empty() ? nullptr : output.strides.back().data();
            const bool reads = materialized.tensor.access != VERNON_ACCESS_WRITE;
            const bool writes = materialized.tensor.access != VERNON_ACCESS_READ;
            if (!elements) {
                // Empty domains retain canonical identity and have no carrier.
            } else if (canonical.storage == VERNON_TENSOR_HOST) {
                output.hostEndpointCarriers.emplace_back(byteSize);
                materialized.tensor.host_data = output.hostEndpointCarriers.back().data();
                const auto *canonicalBytes = static_cast<const uint8_t *>(canonical.host_data);
                auto *carrierBytes = output.hostEndpointCarriers.back().data();
                for (const ProgramTensorCopyRegion &region : regions) {
                    if (reads)
                        output.copiesBefore.push_back(
                            {canonicalBytes, carrierBytes, region.sourceOffset, region.destinationOffset, region.size});
                    if (writes)
                        output.copiesAfter.push_back({carrierBytes, const_cast<uint8_t *>(canonicalBytes),
                                                      region.destinationOffset, region.sourceOffset, region.size});
                }
            } else if (canonical.storage == VERNON_TENSOR_RHI_RESOURCE && nodePlan.stage->context) {
                auto carrier = std::make_shared<DeviceBuffer>(*nodePlan.stage->context, byteSize);
                if (!carrier->valid() || !carrier->reference(materialized.tensor.resource))
                    return error = "physical endpoint carrier allocation failed", false;
                output.deviceEndpointCarriers.push_back(carrier);
                VernonRhiBuffer canonicalBuffer{};
                if (!resolveBackendRhiBufferReference(*nodePlan.stage->context, canonical.resource, canonicalBuffer))
                    return error = "logical endpoint has no RHI Storage backing", false;
                for (const ProgramTensorCopyRegion &region : regions) {
                    size_t canonicalOffset = 0;
                    if (!checkedDeviceBufferOffset(canonical.resource.offset, region.sourceOffset, canonicalOffset))
                        return error = "logical endpoint resource offset exceeds the host address space", false;
                    if (reads)
                        output.deviceCopiesBefore.push_back({canonicalBuffer, carrier->handle(), canonicalOffset,
                                                             region.destinationOffset, region.size});
                    if (writes)
                        output.deviceCopiesAfter.push_back({carrier->handle(), canonicalBuffer,
                                                            region.destinationOffset, canonicalOffset, region.size});
                }
            } else {
                return error = "endpoint projection has no physical carrier backing", false;
            }
        } else if (binding.logicalLeaf) {
            const program::Value &slot = program.values[binding.value];
            if (materialized.kind != VERNON_PROGRAM_TENSOR || !slot.layout ||
                *binding.logicalLeaf >= slot.layout->leaves.size())
                return error = "resolved endpoint leaf exceeds canonical Value ABI", false;
            const ValueLayout valueLayout = program::materializeValueLayout(*slot.layout, slot.type);
            const ValueLeaf &leaf = valueLayout.leaves[*binding.logicalLeaf];
            const ValueLayout &parameterLayout =
                parameter.valueLayout ? *parameter.valueLayout : parameter.elementLayout;
            materialized.tensor.byte_offset += leaf.byteOffset;
            materialized.tensor.element_layout = pipelineValueLayout(parameterLayout);
            const auto projectionShape = logicalProjectionShape(parameter, &binding.target);
            std::vector<uint64_t> projectedShape;
            std::vector<int64_t> projectedStrides;
            std::vector<uint64_t> compactShape;
            std::vector<int64_t> compactStrides;
            if (!projectionShape || !parameterLayout.byteSize ||
                !shape::materializeLeafProjection(output.shapes.back(), output.strides.back(), *projectionShape,
                                                  parameterLayout.byteSize, projectedShape, projectedStrides) ||
                !shape::materializeCompactProjection(output.shapes.back(), *projectionShape, parameterLayout.byteSize,
                                                     compactShape, compactStrides))
                return error = "aggregate leaf has an incompatible physical shape", false;
            if (projectedShape != compactShape || projectedStrides != compactStrides) {
                size_t elements = 0;
                if (!shape::checkedElementCount(compactShape, elements) ||
                    elements > std::numeric_limits<size_t>::max() / parameterLayout.byteSize)
                    return error = "aggregate leaf endpoint size overflows", false;
                const size_t byteSize = elements * parameterLayout.byteSize;
                if (!byteSize)
                    return error = "aggregate leaf endpoint size is zero", false;
                VernonTensorView canonical = materialized.tensor;
                canonical.rank = static_cast<uint32_t>(projectedShape.size());
                canonical.shape = projectedShape.empty() ? nullptr : projectedShape.data();
                canonical.byte_strides = projectedStrides.empty() ? nullptr : projectedStrides.data();
                output.shapes.back() = std::move(compactShape);
                output.strides.back() = std::move(compactStrides);
                materialized.tensor.byte_offset = 0;
                materialized.tensor.byte_size = byteSize;
                materialized.tensor.rank = static_cast<uint32_t>(output.shapes.back().size());
                materialized.tensor.shape = output.shapes.back().empty() ? nullptr : output.shapes.back().data();
                materialized.tensor.byte_strides =
                    output.strides.back().empty() ? nullptr : output.strides.back().data();
                const bool reads = materialized.tensor.access != VERNON_ACCESS_WRITE;
                const bool writes = materialized.tensor.access != VERNON_ACCESS_READ;
                std::vector<ProgramTensorCopyRegion> regions;
                if (canonical.storage == VERNON_TENSOR_HOST) {
                    output.hostEndpointCarriers.emplace_back(byteSize);
                    materialized.tensor.host_data = output.hostEndpointCarriers.back().data();
                    VernonTensorView compact = materialized.tensor;
                    if (!planProgramTensorCopy(canonical, compact, regions, error))
                        return false;
                    const auto *canonicalBytes = static_cast<const uint8_t *>(canonical.host_data);
                    auto *compactBytes = output.hostEndpointCarriers.back().data();
                    for (const ProgramTensorCopyRegion &region : regions) {
                        if (reads)
                            output.copiesBefore.push_back({canonicalBytes, compactBytes, region.sourceOffset,
                                                           region.destinationOffset, region.size});
                        if (writes)
                            output.copiesAfter.push_back({compactBytes, const_cast<uint8_t *>(canonicalBytes),
                                                          region.destinationOffset, region.sourceOffset, region.size});
                    }
                } else if (canonical.storage == VERNON_TENSOR_RHI_RESOURCE && nodePlan.stage->context) {
                    auto storage = std::make_shared<DeviceBuffer>(*nodePlan.stage->context, byteSize);
                    if (!storage->valid() || !storage->reference(materialized.tensor.resource))
                        return error = "aggregate leaf endpoint allocation failed", false;
                    output.deviceEndpointCarriers.push_back(storage);
                    VernonTensorView compact = materialized.tensor;
                    if (!planProgramTensorCopy(canonical, compact, regions, error))
                        return false;
                    VernonRhiBuffer canonicalBuffer{};
                    if (!resolveBackendRhiBufferReference(*nodePlan.stage->context, canonical.resource,
                                                          canonicalBuffer))
                        return error = "aggregate leaf endpoint has no RHI "
                                       "backing",
                               false;
                    for (const ProgramTensorCopyRegion &region : regions) {
                        size_t canonicalOffset = 0;
                        if (!checkedDeviceBufferOffset(canonical.resource.offset, region.sourceOffset, canonicalOffset))
                            return error = "aggregate leaf resource offset exceeds the host address space", false;
                        if (reads)
                            output.deviceCopiesBefore.push_back({canonicalBuffer, storage->handle(), canonicalOffset,
                                                                 region.destinationOffset, region.size});
                        if (writes)
                            output.deviceCopiesAfter.push_back({storage->handle(), canonicalBuffer,
                                                                region.destinationOffset, canonicalOffset,
                                                                region.size});
                    }
                } else {
                    return error = "aggregate leaf has no materializable backing", false;
                }
            } else {
                output.shapes.back() = std::move(projectedShape);
                output.strides.back() = std::move(projectedStrides);
            }
            materialized.tensor.rank = static_cast<uint32_t>(output.shapes.back().size());
            materialized.tensor.shape = output.shapes.back().empty() ? nullptr : output.shapes.back().data();
            materialized.tensor.byte_strides = output.strides.back().empty() ? nullptr : output.strides.back().data();
        }

        if (binding.target.viewTransform) {
            if (materialized.kind != VERNON_PROGRAM_TENSOR || !materialized.tensor.shape ||
                !materialized.tensor.byte_strides ||
                parameter.shape.size() != binding.target.viewTransform->axes.size())
                return error = "Program view transform has an incompatible logical view", false;
            auto &shape = output.shapes.back();
            auto &strides = output.strides.back();
            const std::vector<uint64_t> logicalShape = shape;
            const std::vector<int64_t> logicalStrides = strides;
            shape.clear();
            strides.clear();
            for (size_t physicalAxis = 0; physicalAxis < binding.target.viewTransform->axes.size(); ++physicalAxis) {
                const auto &axis = binding.target.viewTransform->axes[physicalAxis];
                if (axis.source == program::ViewAxisSource::Constant) {
                    if (!axis.constantExtent || !axis.zeroStride)
                        return error = "constant view axis has no storage mapping", false;
                    shape.push_back(axis.constantExtent);
                    strides.push_back(0);
                } else if (axis.source == program::ViewAxisSource::InvocationLinearCarrier) {
                    if (!axis.zeroStride)
                        return error = "dispatch-derived view axis must "
                                       "broadcast",
                               false;
                    shape.push_back(dispatchInvocations);
                    strides.push_back(0);
                } else {
                    if (axis.logicalAxis >= logicalShape.size())
                        return error = "view transform references an unknown "
                                       "logical axis",
                               false;
                    const uint64_t physicalExtent = parameter.shape[physicalAxis];
                    const uint64_t logicalExtent = logicalShape[axis.logicalAxis];
                    if (physicalExtent && logicalExtent && physicalExtent != logicalExtent)
                        return error = "view transform extent disagrees with "
                                       "logical Value",
                               false;
                    shape.push_back(logicalExtent ? logicalExtent : physicalExtent);
                    strides.push_back(axis.zeroStride ? 0 : logicalStrides[axis.logicalAxis]);
                }
            }
            materialized.tensor.rank = static_cast<uint32_t>(shape.size());
            materialized.tensor.shape = shape.data();
            materialized.tensor.byte_strides = strides.data();
        }
        output.arguments.push_back(materialized);
    }
    return true;
}

} // namespace vernon::runtime::program_execution
