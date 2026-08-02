#pragma once

#include "VernonCompiler.h"

namespace vernon::tests {

inline bool unavailableDirectXTarget(const VernonCompilerContext *compiler, VernonTarget target) {
    return target == VERNON_TARGET_DIRECTX && !vernonCompilerGetTargetCapabilities(compiler, target).available;
}

} // namespace vernon::tests
