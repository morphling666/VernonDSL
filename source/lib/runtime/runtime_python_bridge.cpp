#include "runtime_python_bridge.h"

#include "autodiff/host_tape_allocator.h"
#include "program_boundary_view.h"

#include <string>
#include <vector>

namespace {

VernonStringView viewOf(const std::string &value) { return {value.data(), value.size()}; }

} // namespace

extern "C" {

uint64_t vernonRuntimePrivateGetAutodiffHostTapeContextLimit(const VernonRuntimeContext *context) {
    return context ? vernon::runtime::ad::kDefaultHostTapeContextLimit : 0;
}

VernonStatus vernonRuntimePrivateVisitProgramBoundaries(const VernonProgramExecutable *executable,
                                                        VernonRuntimePrivateProgramBoundaryVisitor visitor,
                                                        void *userData) {
    if (!executable || !visitor)
        return VERNON_STATUS_INVALID_ARGUMENT;
    try {
        for (const vernon::runtime::ProgramBoundaryView &item : vernon::runtime::programBoundaryViews(*executable)) {
            const VernonRuntimePrivateProgramBoundary view{sizeof(VernonRuntimePrivateProgramBoundary),
                                                           item.slot,
                                                           item.value,
                                                           viewOf(item.path),
                                                           viewOf(item.role),
                                                           viewOf(item.category)};
            visitor(userData, &view);
        }
        return VERNON_STATUS_OK;
    } catch (...) {
        return VERNON_STATUS_INTERNAL_ERROR;
    }
}

} // extern "C"
