#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_DERIVATIVES_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_DERIVATIVES_H

#include "runtime/autodiff/runtime_autodiff_internal.h"
#include "runtime/autodiff/runtime_gpu_bindings.h"

namespace vernon::runtime::ad::gpu {

struct PreparedDerivativeValues {
    const VernonAdValueSet *cotangents{};
    std::vector<uint8_t> implicitCotangentBytes;
    VernonAdValue implicitCotangent{};
    VernonAdValueSet implicitCotangentSet{};
    std::vector<const VernonAdValue *> cotangentSources;
    std::vector<VernonAdValue *> destinations;
    std::vector<std::vector<uint8_t>> stagedGradients;
    std::vector<size_t> gradientSlots;
    DerivativeSlots devices;
    size_t temporaryBytes{};
    VernonStatus failureStatus{VERNON_STATUS_INVALID_ARGUMENT};

    bool prepare(VernonRuntimeContext &context, const Signature &signature, const VernonAdValueSet *sourceCotangents,
                 VernonAdValueSet &gradients, const BindingSpecPlan &bindingSpecs, VernonLaunchSize invocationExtent,
                 size_t baseTemporaryBytes, size_t temporaryLimit, std::string &error);
    bool stageGradients(const Signature &signature, std::string &error);
    bool publishGradients() const;
};

} // namespace vernon::runtime::ad::gpu

#endif
