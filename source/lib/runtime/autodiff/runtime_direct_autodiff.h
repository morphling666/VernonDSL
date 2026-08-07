#ifndef VERNON_RUNTIME_RUNTIME_DIRECT_AUTODIFF_H
#define VERNON_RUNTIME_RUNTIME_DIRECT_AUTODIFF_H

#include "VernonRuntime.h"
#include "autodiff_metadata.h"

namespace vernon::runtime {

struct AutodiffValueMetadataView {
    VernonStringView path;
    VernonDataType dtype{};
    const uint64_t *shape{};
    size_t rank{};
};

struct AutodiffDerivativeGroupView {
    AutodiffDerivativeRole role{};
    VernonStringView declaredPath;
    const VernonStringView *leafPaths{};
    size_t leafCount{};
};

VERNON_RUNTIME_CAPI VernonLoadedPipeline *loadBackendCpuAutodiffPipeline(
    VernonRuntimeContext &context, VernonCpuEntryPoint primalEntry, VernonStringView primalReflection,
    VernonStringView primalName, VernonCpuEntryPoint forwardEntry, VernonStringView forwardReflection,
    VernonStringView forwardName, VernonCpuEntryPoint backwardEntry, VernonStringView backwardReflection,
    VernonStringView backwardName, VernonStringView forwardProtocol, VernonStringView backwardProtocol,
    const AutodiffDerivativeGroupView *derivativeGroups, size_t derivativeGroupCount);

VERNON_RUNTIME_CAPI size_t getAutodiffOutputMetadataCount(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI size_t getAutodiffCotangentMetadataCount(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI bool getAutodiffOutputMetadata(const VernonLoadedPipeline *pipeline, size_t index,
                                                   AutodiffValueMetadataView &metadata);
VERNON_RUNTIME_CAPI bool getAutodiffCotangentMetadata(const VernonLoadedPipeline *pipeline, size_t index,
                                                      AutodiffValueMetadataView &metadata);
VERNON_RUNTIME_CAPI size_t getAutodiffDerivativeGroupCount(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI bool getAutodiffDerivativeGroupMetadata(const VernonLoadedPipeline *pipeline, size_t groupIndex,
                                                            VernonStringView &role, VernonStringView &declaredPath,
                                                            size_t &leafCount);
VERNON_RUNTIME_CAPI bool getAutodiffDerivativeGroupLeaf(const VernonLoadedPipeline *pipeline, size_t groupIndex,
                                                        size_t leafIndex, VernonStringView &leafPath);
VERNON_RUNTIME_CAPI bool hasAutodiffStorageObjectives(const VernonLoadedPipeline *pipeline);

} // namespace vernon::runtime

#endif
