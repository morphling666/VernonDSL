#ifndef VERNON_RUNTIME_RUNTIME_DIRECT_AUTODIFF_H
#define VERNON_RUNTIME_RUNTIME_DIRECT_AUTODIFF_H

#include "VernonRuntime.h"

namespace vernon::runtime {

struct AutodiffValueMetadataView {
    VernonStringView path;
    VernonDataType dtype{};
    const uint64_t *shape{};
    size_t rank{};
};

VERNON_RUNTIME_CAPI VernonLoadedPipeline *loadBackendCpuAutodiffPipeline(
    VernonRuntimeContext &context, VernonCpuEntryPoint primalEntry, VernonStringView primalReflection,
    VernonStringView primalName, VernonCpuEntryPoint forwardEntry, VernonStringView forwardReflection,
    VernonStringView forwardName, VernonCpuEntryPoint backwardEntry, VernonStringView backwardReflection,
    VernonStringView backwardName, VernonStringView forwardProtocol, VernonStringView backwardProtocol,
    const VernonStringView *gradientPaths, size_t gradientPathCount);

VERNON_RUNTIME_CAPI size_t getAutodiffOutputMetadataCount(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI size_t getAutodiffCotangentMetadataCount(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI bool getAutodiffOutputMetadata(const VernonLoadedPipeline *pipeline, size_t index,
                                                   AutodiffValueMetadataView &metadata);
VERNON_RUNTIME_CAPI bool getAutodiffCotangentMetadata(const VernonLoadedPipeline *pipeline, size_t index,
                                                      AutodiffValueMetadataView &metadata);

} // namespace vernon::runtime

#endif
