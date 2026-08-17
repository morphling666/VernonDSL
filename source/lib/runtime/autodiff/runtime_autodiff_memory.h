#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_AUTODIFF_MEMORY_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_AUTODIFF_MEMORY_H

#include "autodiff/autodiff_memory_accounting.h"

#include <cstddef>

namespace vernon::runtime::ad {

using MemoryAccounting = vernon::autodiff::MemoryAccounting<size_t>;

} // namespace vernon::runtime::ad

#endif
