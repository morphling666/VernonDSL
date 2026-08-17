#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_SIGNATURE_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_SIGNATURE_H

#include "runtime/autodiff/runtime_autodiff_internal.h"

namespace vernon::runtime::ad::gpu {

bool buildSignature(VernonRuntimeContext &context, const Variant &forward, const Variant &backward,
                    const std::vector<std::string> &gradientPaths, Signature &signature);

} // namespace vernon::runtime::ad::gpu

#endif
