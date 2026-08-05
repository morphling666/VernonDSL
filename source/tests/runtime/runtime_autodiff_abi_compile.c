#include "runtime/autodiff/tape_allocator_abi.h"

void vernon_runtime_autodiff_abi_c_header_compile_check(void) {
    VernonAdTapeAllocator allocator = {0};
    allocator.struct_size = sizeof(allocator);
    allocator.abi_version = VERNON_AD_TAPE_ALLOCATOR_ABI_VERSION;
    (void)allocator;
}
