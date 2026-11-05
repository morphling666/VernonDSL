#ifndef VERNON_RUNTIME_RUNTIME_TEST_HOOKS_H
#define VERNON_RUNTIME_RUNTIME_TEST_HOOKS_H

#include "VernonRuntime.h"

#include <cstddef>
#include <cstdint>

namespace vernon::runtime {

bool validateMetalArgumentBufferLimitsForTesting(uint64_t buffers, uint64_t textures, uint64_t samplers,
                                                 bool writableTexture, uint32_t deviceTier);
uint64_t diagnosticClockForTesting() noexcept;
void setDiagnosticClockForTesting(uint64_t clock) noexcept;
void failNextDiagnosticOverflowAllocationForTesting() noexcept;
uint64_t diagnosticGenerationCounterForTesting() noexcept;
void setDiagnosticGenerationCounterForTesting(uint64_t generation) noexcept;

} // namespace vernon::runtime

#endif
