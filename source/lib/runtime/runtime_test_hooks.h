#ifndef VERNON_RUNTIME_RUNTIME_TEST_HOOKS_H
#define VERNON_RUNTIME_RUNTIME_TEST_HOOKS_H

#include "VernonRuntime.h"

#include <cstddef>
#include <cstdint>

namespace vernon::runtime {

uint64_t diagnosticClockForTesting() noexcept;
void setDiagnosticClockForTesting(uint64_t clock) noexcept;
void failNextDiagnosticOverflowAllocationForTesting() noexcept;
uint64_t diagnosticGenerationCounterForTesting() noexcept;
void setDiagnosticGenerationCounterForTesting(uint64_t generation) noexcept;

} // namespace vernon::runtime

#endif
