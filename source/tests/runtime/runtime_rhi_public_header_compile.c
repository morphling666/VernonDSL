#include "VernonRHI.h"
#include "VernonRuntimeCore.h"
#include "VernonRuntimeProvider.h"
#include "VernonRuntimeRHIAdapter.h"

int vernon_runtime_rhi_public_header_compile(void) {
    VernonRhiBuffer buffer = {VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    VernonRuntimeProviderObject object = {0};
    return (int)(buffer.index == VERNON_RHI_INVALID_HANDLE_INDEX && object.value == 0);
}
