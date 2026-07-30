#pragma once

#include "VernonCompiler.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VernonPythonValueAbiPlan VernonPythonValueAbiPlan;

typedef struct VernonPythonValueAbiNodeView {
    uint64_t byte_size;
    uint64_t alignment;
    const uint64_t *field_offsets;
    size_t field_count;
    uint64_t element_stride;
    uint8_t has_element_stride;
} VernonPythonValueAbiNodeView;

typedef struct VernonPythonValueAbiPlanView {
    VernonStatus status;
    VernonStringView diagnostics;
    const VernonPythonValueAbiNodeView *nodes;
    size_t node_count;
} VernonPythonValueAbiPlanView;

/*
 * Private bridge for the in-tree Python extension. It is exported from the
 * compiler DLL but is not installed and is not part of the public C ABI.
 */
VERNON_DSL_CAPI VernonPythonValueAbiPlan *vernonCompilerPlanPythonValueAbi(VernonStringView module,
                                                                           const VernonStringView *logical_dtypes,
                                                                           size_t logical_dtype_count);
VERNON_DSL_CAPI void vernonCompilerDestroyPythonValueAbiPlan(VernonPythonValueAbiPlan *plan);
VERNON_DSL_CAPI VernonPythonValueAbiPlanView
vernonCompilerGetPythonValueAbiPlanView(const VernonPythonValueAbiPlan *plan);

#ifdef __cplusplus
}
#endif
