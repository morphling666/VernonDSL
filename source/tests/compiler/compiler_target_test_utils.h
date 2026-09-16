#pragma once

#include "VernonCompiler.h"

namespace vernon::tests {

inline bool unavailableDirectXTarget(const VernonCompilerContext *compiler, VernonTarget target) {
    if (target != VERNON_TARGET_DIRECTX)
        return false;
    VernonTargetCapabilities capabilities{};
    capabilities.struct_size = sizeof(capabilities);
    capabilities.abi_version = VERNON_TARGET_CAPABILITIES_VERSION;
    return vernonCompilerQueryTargetCapabilities(compiler, target, &capabilities) != VERNON_STATUS_OK ||
           !capabilities.available;
}

} // namespace vernon::tests
