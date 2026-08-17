#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_CPU_PREPARATION_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_CPU_PREPARATION_H

#include "runtime_cpu_program.h"

#include "host_effect_transaction.h"
#include "runtime/cpu_workgroup_dispatch.h"

namespace vernon::runtime::ad::cpu {

VernonStatus fail(VernonRuntimeContext &context, std::string message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT);
const char *allocatorFailure(VernonAdTapeAllocatorStatus status);
bool parseProfile(const Stage &stage, HostProfileLayout &layout, std::string &error);
bool materializeTensorViewShape(const HostArgument &argument, const VernonAdValueSet &inputs,
                                std::vector<uint64_t> &shape);
VernonStatus stageForwardInputs(VernonRuntimeContext &context, const HostProfileLayout &layout,
                                const VernonAdValueSet &inputs, std::vector<uint8_t> &arguments,
                                HostEffectTransaction &transaction, std::vector<StorageRange> &storageRanges,
                                std::vector<StagedTensorView> &tensorViews, bool preserveWriteOnly = false);
void restoreReplayReadWriteShadows(std::vector<StagedTensorView> &tensorViews);
VernonStatus flushStagedTensorViews(VernonRuntimeContext &context, const std::vector<StagedTensorView> &tensorViews);
bool writeInvocationBuiltin(const HostArgument &argument, const CpuLaneCoordinates &coordinates, uint8_t *arguments);
VernonStatus prepareGradientDestinations(VernonRuntimeContext &context, const Signature &signature,
                                         VernonAdValueSet &gradients, std::vector<VernonAdValue *> &destinations,
                                         std::vector<std::vector<uint8_t>> &stagedGradients);
bool writeTensorViewDescriptor(const HostArgument &argument, const std::vector<uint64_t> &shape, void *data,
                               uint8_t *arguments);
bool tensorViewDescriptorShape(const HostArgument &argument, const std::vector<uint64_t> &logicalShape,
                               std::vector<uint64_t> &descriptorShape);
bool isShapeSource(const HostArgument &argument);
bool isPrimalSource(const HostArgument &argument);
std::string tensorOwnerName(const HostArgument &argument);
bool materializeRuntimeSignature(Signature &signature, const VernonAdValueSet &inputs);
VernonStatus accumulateFloatingBytes(VernonDataType dtype, uint8_t *destination, const uint8_t *source,
                                     size_t byteSize);
VernonStatus accumulateBackwardResults(const HostProfileLayout &layout, const Signature &signature,
                                       const uint8_t *results, const std::vector<size_t> &resultGradientIndices,
                                       uint8_t *privateGradients, const std::vector<size_t> &privateGradientOffsets);
VernonStatus accumulateGradientBytes(VernonRuntimeContext &context, const ValueAbi &abi, const uint8_t *source,
                                     std::vector<uint8_t> &destination);
void commitGradientDestinations(const std::vector<VernonAdValue *> &destinations,
                                const std::vector<std::vector<uint8_t>> &stagedGradients);

} // namespace vernon::runtime::ad::cpu

#endif
