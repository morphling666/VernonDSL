#pragma once

#include "VernonCompiler.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VernonPythonValueAbiPlan VernonPythonValueAbiPlan;
typedef struct VernonPythonStructuredVjp VernonPythonStructuredVjp;

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

typedef struct VernonPythonStructuredVjpView {
    VernonStatus status;
    VernonStringView diagnostics;
    VernonStringView forward_module;
    VernonStringView backward_module;
    uint64_t tape_bytes;
    const VernonStringView *derivative_rules;
    size_t derivative_rule_count;
} VernonPythonStructuredVjpView;

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

VERNON_DSL_CAPI VernonPythonStructuredVjp *
vernonCompilerBuildPythonStructuredVjp(VernonStringView module, VernonStringView entry,
                                       const VernonStringView *wrt_paths, size_t wrt_path_count,
                                       VernonStringView forward_symbol, VernonStringView backward_symbol);
VERNON_DSL_CAPI VernonStatus vernonCompilerFinalizePythonStructuredVjp(VernonPythonStructuredVjp *result,
                                                                       VernonStringView profiles_identity);
VERNON_DSL_CAPI void vernonCompilerDestroyPythonStructuredVjp(VernonPythonStructuredVjp *result);
VERNON_DSL_CAPI VernonPythonStructuredVjpView
vernonCompilerGetPythonStructuredVjpView(const VernonPythonStructuredVjp *result);

#ifdef __cplusplus
}
#endif
