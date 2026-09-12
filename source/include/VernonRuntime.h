#ifndef VERNON_C_RUNTIME_H
#define VERNON_C_RUNTIME_H

#include "VernonCommon.h"
#include "VernonOpenGLContext.h"
#include "VernonRHI.h"
#include "VernonRuntimeProvider.h"
#include "VernonTextureTypes.h"

#if defined(VERNON_RUNTIME_STATIC)
#define VERNON_RUNTIME_CAPI
#elif defined(_WIN32) && defined(VERNON_RUNTIME_BUILD)
#define VERNON_RUNTIME_CAPI __declspec(dllexport)
#elif defined(_WIN32)
#define VERNON_RUNTIME_CAPI __declspec(dllimport)
#else
#define VERNON_RUNTIME_CAPI
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VernonRuntimeContext VernonRuntimeContext;
typedef struct VernonProgramBundle VernonProgramBundle;
typedef struct VernonProgramGraph VernonProgramGraph;
typedef struct VernonProgramExecutable VernonProgramExecutable;
typedef struct VernonPullback VernonPullback;
typedef struct VernonSubmission VernonSubmission;
typedef struct VernonProgramInstance VernonProgramInstance;
typedef struct VernonProgramInvocation VernonProgramInvocation;

typedef enum VernonSubmissionState {
    VERNON_SUBMISSION_PENDING = 0,
    VERNON_SUBMISSION_SUCCEEDED = 1,
    VERNON_SUBMISSION_FAILED = 2
} VernonSubmissionState;

typedef enum VernonRuntimeBackend {
    VERNON_RUNTIME_CPU = 0,
    VERNON_RUNTIME_CUDA = 1,
    VERNON_RUNTIME_VULKAN = 2,
    VERNON_RUNTIME_OPENGL = 3,
    VERNON_RUNTIME_OPENGL_ES = 4,
    VERNON_RUNTIME_DIRECTX12 = 5,
    VERNON_RUNTIME_METAL = 6
} VernonRuntimeBackend;

typedef struct VernonRuntimeCapabilities {
    uint8_t available;
    uint8_t supports_compute;
    uint8_t supports_graphics;
    uint8_t supports_storage_buffers;
    uint16_t api_version_major;
    uint16_t api_version_minor;
    uint32_t graphics_draw_abi_version;
    VernonStringView diagnostic;
} VernonRuntimeCapabilities;

typedef struct VernonRuntimeCreateOptions {
    uint32_t struct_size;
    uint32_t device_index;
    /* Reserved for future use; initialize all elements to zero. */
    uint32_t reserved[4];
} VernonRuntimeCreateOptions;

typedef struct VernonLaunchSize {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} VernonLaunchSize;

VERNON_RUNTIME_CAPI VernonRuntimeCapabilities vernonRuntimeGetCapabilities(VernonRuntimeBackend backend);
VERNON_RUNTIME_CAPI VernonRuntimeCapabilities vernonRuntimeGetContextCapabilities(const VernonRuntimeContext *context);
VERNON_RUNTIME_CAPI VernonRuntimeContext *vernonRuntimeCreateWithOptions(VernonRuntimeBackend backend,
                                                                         const VernonRuntimeCreateOptions *options);
VERNON_RUNTIME_CAPI VernonRuntimeContext *vernonRuntimeCreateForRhiDevice(VernonRuntimeBackend backend,
                                                                          VernonRhiDevice device);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeDestroy(VernonRuntimeContext *context);
VERNON_RUNTIME_CAPI VernonStringView vernonRuntimeGetLastError(const VernonRuntimeContext *context);

/*
 * Registers an AOT entry that was statically linked into the application.
 * Re-registering the same symbol and pointer is idempotent.
 */
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeRegisterStaticCpuEntry(VernonStringView symbol,
                                                                     VernonCpuEntryPoint entry_point);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeRegisterCpuEntry(VernonRuntimeContext *context, VernonStringView symbol,
                                                               VernonCpuEntryPoint entry_point);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeUnregisterCpuEntry(VernonRuntimeContext *context, VernonStringView symbol,
                                                                 VernonCpuEntryPoint entry_point);

typedef enum VernonIndexType { VERNON_INDEX_U32 = 0 } VernonIndexType;

typedef enum VernonPrimitiveTopology {
    VERNON_TOPOLOGY_TRIANGLE_LIST = 0,
    VERNON_TOPOLOGY_LINE_LIST = 1,
    VERNON_TOPOLOGY_POINT_LIST = 2
} VernonPrimitiveTopology;

typedef struct VernonIndexBinding {
    VernonIndexType type;
    size_t offset;
    uint32_t index_count;
    VernonRuntimeProviderResourceReference resource;
} VernonIndexBinding;

typedef struct VernonColorAttachment {
    uint32_t location;
    VernonRuntimeProviderResourceReference view;
    VernonRuntimeProviderLoadOperation load_operation;
    VernonRuntimeProviderStoreOperation store_operation;
    float clear_color[4];
} VernonColorAttachment;

typedef struct VernonDepthAttachment {
    VernonRuntimeProviderResourceReference view;
    VernonRuntimeProviderLoadOperation load_operation;
    VernonRuntimeProviderStoreOperation store_operation;
    float clear_depth;
    VernonRuntimeProviderLoadOperation stencil_load_operation;
    VernonRuntimeProviderStoreOperation stencil_store_operation;
    uint32_t clear_stencil;
} VernonDepthAttachment;

typedef struct VernonGraphicsState {
    uint32_t struct_size;
    VernonPrimitiveTopology topology;
    VernonRasterizationState rasterization;
    VernonDepthStencilState depth_stencil;
    const VernonColorBlendState *color_blends;
    size_t color_blend_count;
    uint32_t reserved[4];
} VernonGraphicsState;

typedef struct VernonRenderPass {
    uint32_t struct_size;
    const VernonColorAttachment *color_attachments;
    size_t color_attachment_count;
    const VernonDepthAttachment *depth_attachment;
    uint32_t render_area[4];
} VernonRenderPass;

typedef struct VernonDrawCommand {
    uint32_t struct_size;
    const VernonIndexBinding *index_binding;
    uint32_t vertex_count;
    uint32_t instance_count;
} VernonDrawCommand;

typedef struct VernonDynamicState {
    uint32_t struct_size;
    uint32_t viewport[4];
    uint32_t scissor[4];
    uint32_t stencil_reference;
} VernonDynamicState;

VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeReferenceRhiBuffer(VernonRuntimeContext *context, VernonRhiBuffer buffer,
                                                                 uint64_t offset, uint64_t size,
                                                                 VernonRuntimeProviderResourceReference *output);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeReferenceRhiImageView(VernonRuntimeContext *context,
                                                                    VernonRhiImageView view,
                                                                    VernonRuntimeProviderResourceReference *output);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeReferenceRhiSampler(VernonRuntimeContext *context,
                                                                  VernonRhiSampler sampler,
                                                                  VernonRuntimeProviderResourceReference *output);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeReferenceRhiCommandEncoder(VernonRuntimeContext *context,
                                                                         VernonRhiCommandEncoder encoder,
                                                                         VernonRuntimeProviderObject *output);

typedef enum VernonDataType {
    VERNON_DATA_BOOL = 0,
    VERNON_DATA_I32 = 1,
    VERNON_DATA_U32 = 2,
    VERNON_DATA_F16 = 3,
    VERNON_DATA_F32 = 4,
    VERNON_DATA_F64 = 5,
    VERNON_DATA_U8 = 6
} VernonDataType;

typedef enum VernonProgramSpecializationKind {
    VERNON_PROGRAM_SPECIALIZATION_BOOL = 0,
    VERNON_PROGRAM_SPECIALIZATION_I32 = 1,
    VERNON_PROGRAM_SPECIALIZATION_U32 = 2,
    VERNON_PROGRAM_SPECIALIZATION_F32 = 3,
    VERNON_PROGRAM_SPECIALIZATION_F64 = 4
} VernonProgramSpecializationKind;

typedef union VernonProgramSpecializationValue {
    uint8_t boolean_value;
    int32_t i32_value;
    uint32_t u32_value;
    float f32_value;
    double f64_value;
} VernonProgramSpecializationValue;

typedef struct VernonProgramSpecialization {
    uint32_t struct_size;
    VernonStringView name;
    VernonProgramSpecializationKind kind;
    VernonProgramSpecializationValue value;
    uint32_t reserved[4];
} VernonProgramSpecialization;

/*
 * Selects one exact cooked Program variant. Names must be non-empty and
 * unique; order is irrelevant. A null selector selects the empty key.
 * Floating-point values must be finite.
 */
typedef struct VernonProgramVariantSelector {
    uint32_t struct_size;
    const VernonProgramSpecialization *specializations;
    size_t specialization_count;
    uint32_t reserved[4];
} VernonProgramVariantSelector;

typedef enum VernonValueAccess {
    VERNON_ACCESS_READ = 0,
    VERNON_ACCESS_WRITE = 1,
    VERNON_ACCESS_READ_WRITE = 2
} VernonValueAccess;

typedef enum VernonTensorStorage { VERNON_TENSOR_HOST = 0, VERNON_TENSOR_RHI_RESOURCE = 1 } VernonTensorStorage;

typedef struct VernonValueLeafView {
    uint32_t dtype;
    uint32_t scalar_count;
    uint32_t byte_offset;
} VernonValueLeafView;

typedef enum VernonValuePathComponentKind {
    VERNON_VALUE_PATH_FIELD = 0,
    VERNON_VALUE_PATH_INDEX = 1
} VernonValuePathComponentKind;

typedef struct VernonValuePathComponentView {
    VernonValuePathComponentKind kind;
    VernonStringView field;
    uint64_t index;
} VernonValuePathComponentView;

typedef struct VernonProgramValueLeafView {
    uint32_t struct_size;
    VernonValueLeafView value;
    /* Struct field path of this payload leaf. Empty for a scalar Tensor cell. */
    const VernonValuePathComponentView *path;
    size_t path_count;
    /* Leaf-internal packed extents. Empty for a scalar cell. Outer Tensor rank
     * lives on VernonProgramParameterView.rank / static_shape, not here. */
    const uint64_t *static_shape;
    uint32_t static_rank;
} VernonProgramValueLeafView;

typedef struct VernonValueLayoutView {
    uint32_t struct_size;
    uint32_t byte_size;
    uint32_t alignment;
    VernonStringView layout_hash;
    const VernonValueLeafView *leaves;
    size_t leaf_count;
} VernonValueLayoutView;

VERNON_RUNTIME_CAPI VernonValueLayoutView vernonRuntimeGetScalarValueLayout(VernonDataType dtype);

typedef struct VernonTensorView {
    uint32_t struct_size;
    VernonTensorStorage storage;
    union {
        const void *host_data;
        VernonRuntimeProviderResourceReference resource;
    };
    VernonValueLayoutView element_layout;
    VernonValueAccess access;
    uint32_t rank;
    const uint64_t *shape;
    const int64_t *byte_strides;
    size_t byte_offset;
    size_t byte_size;
} VernonTensorView;

typedef struct VernonImageArgument {
    VernonRuntimeProviderResourceReference view;
} VernonImageArgument;

typedef enum VernonProgramArgumentKind {
    VERNON_PROGRAM_TENSOR = 0,
    VERNON_PROGRAM_IMAGE = 1,
    VERNON_PROGRAM_SAMPLER = 2
} VernonProgramArgumentKind;

typedef struct VernonProgramArgument {
    uint32_t slot;
    VernonProgramArgumentKind kind;
    union {
        VernonTensorView tensor;
        VernonImageArgument image;
        VernonRuntimeProviderResourceReference resource;
    };
} VernonProgramArgument;

typedef struct VernonProgramBindingToken {
    uint32_t struct_size;
    const void *data;
    size_t size;
} VernonProgramBindingToken;

typedef struct VernonProgramResourceLease {
    uint32_t struct_size;
    void *object;
    void (*retain)(void *object);
    void (*release)(void *object);
} VernonProgramResourceLease;

typedef struct VernonProgramBindingTelemetry {
    uint32_t struct_size;
    uint64_t prepare_count;
    uint64_t reuse_count;
    uint64_t rollback_count;
    uint64_t upload_bytes;
    uint64_t upload_ranges;
} VernonProgramBindingTelemetry;

typedef struct VernonProgramParameterView {
    uint32_t slot;
    VernonStringView name;
    VernonProgramArgumentKind kind;
    /* Packed host-binding ABI of the whole argument (call-frame / copy plan).
     * Named element_layout for historical reasons; for a shaped Tensor this
     * is the packed value, not one cell. Cells/fields: GetParameterValueLeaf. */
    VernonValueLayoutView element_layout;
    VernonValueAccess access;
    /* Outer Tensor extents. Empty for a scalar or struct parameter. */
    uint32_t rank;
    const uint64_t *static_shape;
} VernonProgramParameterView;

typedef struct VernonProgramImageConstraintView {
    uint32_t struct_size;
    VernonTextureDimension dimension;
    VernonImageBindingRole binding_role;
    VernonImageSampleResultClass sample_result_class;
    VernonTextureFormat storage_format;
    /* Reserved for future constraints; initialize all elements to zero. */
    uint32_t reserved[4];
} VernonProgramImageConstraintView;

typedef enum VernonProgramBoundaryRole {
    VERNON_PROGRAM_BOUNDARY_INPUT = 0,
    VERNON_PROGRAM_BOUNDARY_OUTPUT = 1,
    VERNON_PROGRAM_BOUNDARY_COTANGENT = 2,
    VERNON_PROGRAM_BOUNDARY_GRADIENT = 3
} VernonProgramBoundaryRole;

typedef uint32_t VernonProgramNodeId;

typedef struct VernonProgramNodeBindingToken {
    uint32_t struct_size;
    uint64_t graph_id;
    VernonProgramNodeId node;
    uint32_t local_slot;
    VernonProgramArgumentKind kind;
} VernonProgramNodeBindingToken;

typedef struct VernonProgramGraphValue {
    uint32_t struct_size;
    uint64_t graph_id;
    uint32_t id;
    VernonProgramArgumentKind kind;
} VernonProgramGraphValue;

typedef struct VernonProgramGraphStorage {
    uint32_t struct_size;
    uint64_t graph_id;
    uint32_t id;
    VernonProgramArgumentKind kind;
} VernonProgramGraphStorage;

typedef enum VernonAdDerivativeRole {
    VERNON_AD_DERIVATIVE_GRADIENT = 0,
    VERNON_AD_DERIVATIVE_COTANGENT = 1
} VernonAdDerivativeRole;

typedef struct VernonAdDerivativeGroupView {
    uint32_t struct_size;
    VernonAdDerivativeRole role;
    VernonStringView declared_path;
    size_t leaf_count;
    /* Reserved for future use; initialize all elements to zero. */
    uint32_t reserved[4];
} VernonAdDerivativeGroupView;

#define VERNON_PULLBACK_APPLY_OPTIONS_VERSION 1

typedef struct VernonPullbackApplyOptions {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t maximum_temporary_bytes;
    /* Reserved for future use; initialize all elements to zero. */
    uint32_t reserved[4];
} VernonPullbackApplyOptions;

typedef enum VernonInvocationSubmissionState {
    VERNON_INVOCATION_NOT_SUBMITTED = 0,
    VERNON_INVOCATION_COMPLETED = 1,
    VERNON_INVOCATION_INDETERMINATE = 2
} VernonInvocationSubmissionState;

typedef enum VernonBoundaryMutationState {
    VERNON_BOUNDARY_MUTATION_UNCHANGED = 0,
    VERNON_BOUNDARY_MUTATION_COMMITTED = 1,
    VERNON_BOUNDARY_MUTATION_IN_PLACE_COMMITTED = 2,
    VERNON_BOUNDARY_MUTATION_INDETERMINATE = 3
} VernonBoundaryMutationState;

typedef enum VernonMutationBoundaryKind {
    VERNON_MUTATION_PROGRAM_BOUNDARY = 0,
    VERNON_MUTATION_RENDER_PASS_CONTROL = 1
} VernonMutationBoundaryKind;

typedef struct VernonBoundaryMutation {
    VernonMutationBoundaryKind kind;
    uint32_t slot;
    VernonBoundaryMutationState state;
} VernonBoundaryMutation;

typedef struct VernonInvocationMutationOutcome {
    uint32_t struct_size;
    VernonInvocationSubmissionState submission;
    VernonBoundaryMutation *mutations;
    size_t mutation_capacity;
    size_t mutation_count;
    uint32_t reserved[4];
} VernonInvocationMutationOutcome;

#define VERNON_PROGRAM_AUTODIFF_INVOCATION_OPTIONS_VERSION 1

typedef struct VernonProgramAutodiffInvocationOptions {
    uint32_t struct_size;
    uint32_t abi_version;
    uint8_t has_checkpoint_memory_budget;
    uint8_t reserved_bytes[7];
    uint64_t checkpoint_memory_budget;
    VernonStringView checkpoint_policy;
    /* Reserved for future use; initialize all elements to zero. */
    uint32_t reserved[4];
} VernonProgramAutodiffInvocationOptions;

typedef struct VernonProgramBundleLoadOptions {
    uint32_t struct_size;
    /*
     * UTF-8 directory containing the manifest. Required by external artifact
     * descriptors and CPU native-library sidecars.
     */
    const char *bundle_directory;
    /* Reserved for future use; initialize all elements to zero. */
    uint32_t reserved[4];
} VernonProgramBundleLoadOptions;

VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramBundleInspectTarget(const void *bundle, size_t bundle_size,
                                                                         VernonRuntimeBackend *target);
/* Loads a cooked single-kernel pipeline or Module Program bundle. CPU
 * relocatable objects must already be linked and registered by the host. */
VERNON_RUNTIME_CAPI VernonProgramBundle *
vernonRuntimeLoadProgramBundleWithOptions(VernonRuntimeContext *context, const void *bundle, size_t bundle_size,
                                          const VernonProgramBundleLoadOptions *options);
VERNON_RUNTIME_CAPI VernonStringView vernonRuntimeProgramBundleGetId(const VernonProgramBundle *bundle);
VERNON_RUNTIME_CAPI void vernonRuntimeProgramBundleDestroy(VernonProgramBundle *bundle);
VERNON_RUNTIME_CAPI VernonProgramGraph *vernonRuntimeProgramGraphCreate(VernonRuntimeContext *context);
VERNON_RUNTIME_CAPI void vernonRuntimeProgramGraphDestroy(VernonProgramGraph *graph);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramGraphAddProgram(VernonProgramGraph *graph,
                                                                     const VernonProgramBundle *bundle,
                                                                     const VernonProgramVariantSelector *selector,
                                                                     VernonProgramNodeId *node);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramGraphFindBoundary(const VernonProgramGraph *graph,
                                                                       VernonProgramNodeId node,
                                                                       VernonProgramBoundaryRole role,
                                                                       VernonStringView name,
                                                                       VernonProgramNodeBindingToken *token);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramGraphCreateValue(VernonProgramGraph *graph,
                                                                      const VernonProgramNodeBindingToken *source,
                                                                      VernonProgramGraphValue *value);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramGraphConnectValue(
    VernonProgramGraph *graph, const VernonProgramGraphValue *value, const VernonProgramNodeBindingToken *destination);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramGraphCreateStorage(
    VernonProgramGraph *graph, const VernonProgramNodeBindingToken *first_version, VernonProgramGraphStorage *storage);
VERNON_RUNTIME_CAPI VernonStatus
vernonRuntimeProgramGraphAppendStorage(VernonProgramGraph *graph, const VernonProgramGraphStorage *storage,
                                       const VernonProgramNodeBindingToken *next_version);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramGraphExportBoundary(VernonProgramGraph *graph,
                                                                         const VernonProgramNodeBindingToken *boundary,
                                                                         VernonStringView graph_name);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramGraphExportValue(VernonProgramGraph *graph,
                                                                      const VernonProgramGraphValue *value,
                                                                      VernonStringView graph_name);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramGraphExportStorage(VernonProgramGraph *graph,
                                                                        const VernonProgramGraphStorage *storage,
                                                                        VernonStringView graph_name);
VERNON_RUNTIME_CAPI VernonProgramExecutable *vernonRuntimeResolveProgramGraph(VernonProgramGraph *graph);
VERNON_RUNTIME_CAPI VernonProgramExecutable *vernonRuntimeResolveProgram(VernonProgramBundle *bundle,
                                                                         const VernonProgramVariantSelector *selector);
VERNON_RUNTIME_CAPI void vernonRuntimeProgramExecutableDestroy(VernonProgramExecutable *pipeline);
VERNON_RUNTIME_CAPI VernonStringView vernonRuntimeProgramExecutableGetId(const VernonProgramExecutable *pipeline);
VERNON_RUNTIME_CAPI size_t vernonRuntimeProgramExecutableGetParameterCount(const VernonProgramExecutable *pipeline);
VERNON_RUNTIME_CAPI size_t vernonRuntimeProgramExecutableGetMutationCapacity(const VernonProgramExecutable *pipeline);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramExecutableGetParameterByIndex(
    const VernonProgramExecutable *pipeline, size_t index, VernonProgramParameterView *parameter);
/*
 * Parameter identity and host binding: name, access, outer Tensor shape
 * (rank / static_shape), and packed host_value ABI in element_layout.
 *
 * This is not the cell/field leaf table. A Tensor[f32,(2,)] reports
 * rank=1, static_shape=[2], and element_layout of the whole packed value
 * (scalar_count=2). Walk payload cells/fields with GetParameterValueLeaf.
 */
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramExecutableFindParameter(const VernonProgramExecutable *pipeline,
                                                                             VernonStringView name,
                                                                             VernonProgramParameterView *parameter);
/*
 * Logical payload leaf of one tensor parameter: one cell of a shaped Tensor,
 * or one field of a struct. Query this for AD packing / reflection dtype
 * and leaf-internal shape.
 *
 * Tensor[f32,(2,)] has one leaf: f32, scalar_count=1, static_rank=0. Outer
 * extent [2] is FindParameter().static_shape, not this leaf. Packed
 * host_value ABI (the whole tensor as one layout) is FindParameter().
 * element_layout, not this API.
 */
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramExecutableGetParameterValueLeaf(
    const VernonProgramExecutable *pipeline, VernonStringView parameter_name, size_t leaf_index,
    VernonProgramValueLeafView *leaf);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramExecutableGetImageConstraintByParameterIndex(
    const VernonProgramExecutable *pipeline, size_t parameter_index, VernonProgramImageConstraintView *constraint);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramExecutableFindImageConstraint(
    const VernonProgramExecutable *pipeline, VernonStringView parameter_name,
    VernonProgramImageConstraintView *constraint);
/* The control slots a graphics node reads its per-invocation controls from.
 *
 * A Program assigns each control its own slot when it is compiled, so a caller cannot assume any particular number
 * and must ask before binding. `node` is the graph node id the controls belong to. */
typedef struct VernonProgramGraphicsControlsView {
    uint32_t struct_size;
    uint32_t node;
    uint32_t render_pass_control;
    uint32_t draw_command_control;
    uint32_t dynamic_state_control;
} VernonProgramGraphicsControlsView;

VERNON_RUNTIME_CAPI size_t vernonRuntimeProgramExecutableGetGraphicsNodeCount(const VernonProgramExecutable *pipeline);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramExecutableGetGraphicsControlsByIndex(
    const VernonProgramExecutable *pipeline, size_t index, VernonProgramGraphicsControlsView *output);
VERNON_RUNTIME_CAPI VernonProgramInstance *vernonRuntimeProgramInstanceCreate(VernonProgramExecutable *pipeline);
VERNON_RUNTIME_CAPI void vernonRuntimeProgramInstanceDestroy(VernonProgramInstance *instance);
VERNON_RUNTIME_CAPI VernonProgramInvocation *
vernonRuntimeProgramInstanceBeginInvocation(VernonProgramInstance *instance);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramInvocationSetAutodiffOptions(
    VernonProgramInvocation *invocation, const VernonProgramAutodiffInvocationOptions *options);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramInvocationTryReuse(VernonProgramInvocation *invocation,
                                                                        uint32_t slot,
                                                                        const VernonProgramBindingToken *token,
                                                                        uint64_t upload_bytes, uint64_t upload_ranges,
                                                                        uint8_t *reused);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramInvocationBind(VernonProgramInvocation *invocation,
                                                                    const VernonProgramBindingToken *token,
                                                                    const VernonProgramArgument *argument,
                                                                    const VernonProgramResourceLease *lease,
                                                                    uint64_t upload_bytes, uint64_t upload_ranges);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramInvocationBindRenderPass(
    VernonProgramInvocation *invocation, uint32_t control_slot, const VernonProgramBindingToken *token,
    const VernonRenderPass *render_pass, const VernonProgramResourceLease *leases, size_t lease_count);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramInvocationBindDrawCommand(VernonProgramInvocation *invocation,
                                                                               uint32_t control_slot,
                                                                               const VernonProgramBindingToken *token,
                                                                               const VernonDrawCommand *draw,
                                                                               const VernonProgramResourceLease *lease);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramInvocationBindDynamicState(
    VernonProgramInvocation *invocation, uint32_t control_slot, const VernonProgramBindingToken *token,
    const VernonDynamicState *dynamic_state);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramInvocationExecute(VernonProgramInvocation *invocation,
                                                                       uint8_t retain_pullback,
                                                                       VernonInvocationMutationOutcome *outcome);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramInvocationCommit(VernonProgramInvocation *invocation,
                                                                      VernonPullback **output_pullback);
/*
 * For a ProgramGraph, commit returns no composite pullback. Retrieve each
 * differentiated child's pullback exactly once after commit. The returned
 * pullback owns its retained state and may outlive the invocation.
 */
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramInvocationGetNodePullback(VernonProgramInvocation *invocation,
                                                                               VernonProgramNodeId node,
                                                                               VernonPullback **output_pullback);
VERNON_RUNTIME_CAPI void vernonRuntimeProgramInvocationRollback(VernonProgramInvocation *invocation);
VERNON_RUNTIME_CAPI void vernonRuntimeProgramInvocationDestroy(VernonProgramInvocation *invocation);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramInstanceGetTelemetry(const VernonProgramInstance *instance,
                                                                          VernonProgramBindingTelemetry *output);
VERNON_RUNTIME_CAPI VernonStatus vernonSubmissionGetState(const VernonSubmission *submission,
                                                          VernonSubmissionState *output);
VERNON_RUNTIME_CAPI VernonStatus vernonSubmissionWait(VernonSubmission *submission);
VERNON_RUNTIME_CAPI void vernonSubmissionDestroy(VernonSubmission *submission);
VERNON_RUNTIME_CAPI uint8_t vernonRuntimeProgramExecutableHasProgramAutodiff(const VernonProgramExecutable *pipeline);
VERNON_RUNTIME_CAPI size_t vernonRuntimeProgramExecutableGetBoundaryCount(const VernonProgramExecutable *pipeline,
                                                                          VernonProgramBoundaryRole boundary);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramExecutableGetBoundaryByIndex(
    const VernonProgramExecutable *pipeline, VernonProgramBoundaryRole boundary, size_t index,
    VernonProgramParameterView *parameter);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramExecutableFindBoundary(const VernonProgramExecutable *pipeline,
                                                                            VernonProgramBoundaryRole boundary,
                                                                            VernonStringView name,
                                                                            VernonProgramParameterView *parameter);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramExecutableGetBoundaryValueLeaf(
    const VernonProgramExecutable *pipeline, VernonProgramBoundaryRole boundary, uint32_t slot, size_t leaf_index,
    VernonProgramValueLeafView *leaf);
VERNON_RUNTIME_CAPI size_t
vernonRuntimeProgramExecutableGetAdDerivativeGroupCount(const VernonProgramExecutable *pipeline);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramExecutableGetAdDerivativeGroupByIndex(
    const VernonProgramExecutable *pipeline, size_t group_index, VernonAdDerivativeGroupView *group);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramExecutableGetAdDerivativeGroupLeaf(
    const VernonProgramExecutable *pipeline, size_t group_index, size_t leaf_index, VernonStringView *leaf_path);
VERNON_RUNTIME_CAPI VernonStatus vernonProgramPullbackApplyWithOptions(VernonPullback *pullback,
                                                                       const VernonProgramArgument *arguments,
                                                                       size_t argument_count,
                                                                       const VernonPullbackApplyOptions *options,
                                                                       VernonInvocationMutationOutcome *outcome);
VERNON_RUNTIME_CAPI VernonStatus vernonProgramPullbackApply(VernonPullback *pullback,
                                                            const VernonProgramArgument *arguments,
                                                            size_t argument_count,
                                                            VernonInvocationMutationOutcome *outcome);
VERNON_RUNTIME_CAPI void vernonProgramPullbackDestroy(VernonPullback *pullback);

#ifdef __cplusplus
}
#endif

#endif
