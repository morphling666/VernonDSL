#ifndef VERNON_C_RUNTIME_H
#define VERNON_C_RUNTIME_H

#include "VernonCommon.h"
#include "VernonOpenGLContext.h"
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

typedef enum VernonRuntimeOperationStatus {
    VERNON_RUNTIME_OPERATION_OK = 0,
    VERNON_RUNTIME_OPERATION_INVALID_ARGUMENT = 1,
    VERNON_RUNTIME_OPERATION_PARSE_FAILURE = 2,
    VERNON_RUNTIME_OPERATION_VERIFICATION_FAILURE = 3,
    VERNON_RUNTIME_OPERATION_UNSUPPORTED = 4,
    VERNON_RUNTIME_OPERATION_RESOURCE_EXHAUSTED = 5,
    VERNON_RUNTIME_OPERATION_DEVICE_FAILURE = 6,
    VERNON_RUNTIME_OPERATION_LIFECYCLE_FAILURE = 7,
    VERNON_RUNTIME_OPERATION_INTERNAL_FAILURE = 8
} VernonRuntimeOperationStatus;

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
    /* Selects the device of the backend. */
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
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeDestroy(VernonRuntimeContext *context);
VERNON_RUNTIME_CAPI VernonStringView vernonRuntimeGetLastError(const VernonRuntimeContext *context);

/*
 * Registers AOT entries that are statically linked into the application.
 * Every cooked CPU Program object calls this from its own static constructor,
 * so applications only link the objects. Objects placed in a static library
 * must be linked whole-archive, or the linker drops them with their
 * constructor. Registrations are reference counted per symbol and pointer, and
 * each object's static destructor unregisters its own, so unloading a shared
 * library withdraws exactly the definitions it contributed. Lookups use the
 * oldest live definition of a symbol.
 */
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeRegisterStaticCpuEntries(const VernonStaticCpuEntry *entries,
                                                                       size_t count);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeUnregisterStaticCpuEntries(const VernonStaticCpuEntry *entries,
                                                                         size_t count);
/*
 * Host modules (`vernon_host/reflect.h` tables) cooked as Program assets. A cooked
 * object registers its table under the asset id from a static constructor and
 * unregisters it from its static destructor; Python mode registers the table it
 * builds the same way, so applications find a module by id in both modes.
 * Registrations are reference counted per id and table. A lookup returns the
 * newest live registration of the id, so a Python-mode or reloaded table shadows
 * the linked one until it is unregistered. Returns null for an unknown id.
 */
typedef struct VernonStaticHostModule {
    VernonStringView id;
    const struct VhModule *module;
} VernonStaticHostModule;

VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeRegisterStaticHostModule(const VernonStaticHostModule *module);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeUnregisterStaticHostModule(const VernonStaticHostModule *module);
VERNON_RUNTIME_CAPI const struct VhModule *vernonRuntimeFindHostModule(VernonStringView id);
/*
 * The container runtime of Host code (`vernon_host/reflect.h` VhList and
 * VhAllocator), which compiled Host objects and Python mode both call. A null
 * allocator is the default one, over malloc. Reserve grows `list` to hold at
 * least `capacity` elements, at least doubling, and keeps its elements; it
 * returns 0, or the Host OUT_OF_MEMORY status (-4) and leaves `list` unchanged.
 * Release returns the list's memory and zeroes it.
 */
struct VhList;
struct VhDict;
struct VhAllocator;
VERNON_RUNTIME_CAPI int32_t vernonRuntimeHostListReserve(struct VhList *list, uint64_t capacity, uint64_t element_size,
                                                         uint64_t alignment, const struct VhAllocator *allocator);
VERNON_RUNTIME_CAPI void vernonRuntimeHostListRelease(struct VhList *list, uint64_t element_size, uint64_t alignment,
                                                      const struct VhAllocator *allocator);
/*
 * The default allocator as a VhAllocator, over this runtime's malloc and free.
 * Callers hand it to code that fills containers the runtime later frees, such as
 * an extern's owned result, so the memory comes from the heap that frees it even
 * where each shared library links a C runtime of its own.
 */
VERNON_RUNTIME_CAPI const struct VhAllocator *vernonRuntimeHostDefaultAllocator(void);
/*
 * A `dict[K, V]` is a `vernon_host/reflect.h` VhDict. Its entries are
 * `{uint64_t hash; K key; V value;}` under C layout rules, which this record
 * describes. A key passed in points at the key's bytes, or for a str key at
 * `{uint64_t data; uint64_t size}`, the prefix of a VhList; an entry holds a str
 * key as a VhList the dict owns.
 */
typedef enum VernonHostDictKeyKind {
    VERNON_HOST_DICT_KEY_BYTES = 0,
    VERNON_HOST_DICT_KEY_STR = 1
} VernonHostDictKeyKind;

typedef struct VernonHostDictLayout {
    uint64_t entry_size;
    uint64_t entry_alignment;
    uint64_t key_offset;
    uint64_t key_size;
    uint64_t value_offset;
    /* A VernonHostDictKeyKind. */
    uint64_t key_kind;
} VernonHostDictLayout;

/*
 * Find returns the index of the entry holding `key`, or -1. Insert finds or
 * appends it, a new entry's value zeroed, and writes its index to `*entry`; it
 * returns 0 or OUT_OF_MEMORY (-4), leaving the dict's contents unchanged.
 * Remove returns 1 when it removed `key`, 0 when the dict has none. Entry
 * indices are stable until an insert of a new key grows or compacts the dict.
 * Clear removes every entry and keeps the memory; Release returns it all and
 * zeroes the record.
 */
VERNON_RUNTIME_CAPI int64_t vernonRuntimeHostDictFind(const struct VhDict *dict, const void *key,
                                                      const VernonHostDictLayout *layout);
VERNON_RUNTIME_CAPI int32_t vernonRuntimeHostDictInsert(struct VhDict *dict, const void *key,
                                                        const VernonHostDictLayout *layout,
                                                        const struct VhAllocator *allocator, uint64_t *entry);
VERNON_RUNTIME_CAPI int32_t vernonRuntimeHostDictRemove(struct VhDict *dict, const void *key,
                                                        const VernonHostDictLayout *layout,
                                                        const struct VhAllocator *allocator);
VERNON_RUNTIME_CAPI void vernonRuntimeHostDictClear(struct VhDict *dict, const VernonHostDictLayout *layout,
                                                    const struct VhAllocator *allocator);
VERNON_RUNTIME_CAPI void vernonRuntimeHostDictRelease(struct VhDict *dict, const VernonHostDictLayout *layout,
                                                      const struct VhAllocator *allocator);
/*
 * A replacement field's format spec in a Host f-string,
 * `[[fill]align][sign][width][.precision][type]`, parsed by the compiler,
 * which also turns a `0` before the width into fill '0' and align '='. Every
 * member zero means no spec.
 */
typedef struct VernonHostFormatSpec {
    /* A code point; 0 is a space. */
    uint32_t fill;
    /* 0 for the value's default (left for text, right for numbers), or '<', '>', '^', '='. */
    uint8_t align;
    /* 0 or '-' for a sign on negatives only, '+', or ' '. */
    uint8_t sign;
    /* 0, 'd', 'x', 'X', 'o' or 'b' for integers; 0, 'f', 'e' or 'E' for floats. */
    uint8_t type;
    uint8_t reserved;
    /* The minimum width in code points. */
    uint32_t width;
    /* Digits after the point for 'f', 'e' and 'E', at most 64; -1 for the default of 6. */
    int32_t precision;
} VernonHostFormatSpec;

/*
 * Append text to a `str` record, formatted as Python's format() formats the
 * value under `spec` (NULL for none): a str's UTF-8 bytes; an integer's
 * two's-complement bits, signed or not; a float as Python's float, whose repr
 * (no type) is the shortest text that reads back as the same double. Text may
 * lie in `str` itself. They return 0, or OUT_OF_MEMORY (-4) and leave `str`
 * unchanged.
 */
VERNON_RUNTIME_CAPI int32_t vernonRuntimeHostStrAppend(struct VhList *str, const char *data, uint64_t size,
                                                       const VernonHostFormatSpec *spec,
                                                       const struct VhAllocator *allocator);
VERNON_RUNTIME_CAPI int32_t vernonRuntimeHostStrAppendInteger(struct VhList *str, uint64_t bits, int32_t is_signed,
                                                              const VernonHostFormatSpec *spec,
                                                              const struct VhAllocator *allocator);
VERNON_RUNTIME_CAPI int32_t vernonRuntimeHostStrAppendFloat(struct VhList *str, double value,
                                                            const VernonHostFormatSpec *spec,
                                                            const struct VhAllocator *allocator);

/*
 * Host TensorStorage records (`vernon_host/reflect.h` VhStorage): `{uint64_t data; uint64_t shape[rank]}`, dense
 * row-major elements of `element_size` bytes. Allocate fills an empty record with zeroed elements of `shape`; it
 * returns 0, the Host INVALID_SHAPE status (-7) for a negative extent or a rank above
 * VERNON_HOST_STORAGE_MAX_RANK, or OUT_OF_MEMORY (-4), and leaves the record unchanged on failure. A storage of
 * no elements holds no memory. Release returns the memory and zeroes the record. Fill copies `value`, one element,
 * into every element.
 */
#define VERNON_HOST_STORAGE_MAX_RANK 8
VERNON_RUNTIME_CAPI int32_t vernonRuntimeHostStorageAllocate(void *storage, uint32_t rank, const int64_t *shape,
                                                             uint64_t element_size, uint64_t alignment,
                                                             const struct VhAllocator *allocator);
VERNON_RUNTIME_CAPI void vernonRuntimeHostStorageRelease(void *storage, uint32_t rank, uint64_t element_size,
                                                         uint64_t alignment, const struct VhAllocator *allocator);
VERNON_RUNTIME_CAPI void vernonRuntimeHostStorageFill(void *storage, uint32_t rank, const void *value,
                                                      uint64_t element_size);
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
    /* Byte offset of the first index. */
    size_t offset;
    uint32_t index_count;
    VernonRuntimeProviderResourceReference resource;
    /* Added to each index before vertex fetch; `vertex_index` includes it on every backend. */
    int32_t base_vertex;
} VernonIndexBinding;

typedef struct VernonColorAttachment {
    uint32_t location;
    VernonRuntimeProviderResourceReference view;
    VernonRuntimeProviderLoadOperation load_operation;
    VernonRuntimeProviderStoreOperation store_operation;
    VernonClearColor clear_color;
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
    /* A draw of no instances draws nothing and reads no instance records. */
    uint32_t instance_count;
    /* `instance_index` equals first_instance plus the instance ordinal on every backend. */
    uint32_t first_instance;
} VernonDrawCommand;

typedef struct VernonDynamicState {
    uint32_t struct_size;
    uint32_t viewport[4];
    uint32_t scissor[4];
    uint32_t stencil_reference;
} VernonDynamicState;

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
    VERNON_PROGRAM_SPECIALIZATION_F64 = 4,
    VERNON_PROGRAM_SPECIALIZATION_TYPE = 5,
    VERNON_PROGRAM_SPECIALIZATION_FORMAT = 6
} VernonProgramSpecializationKind;

typedef union VernonProgramSpecializationValue {
    uint8_t boolean_value;
    int32_t i32_value;
    uint32_t u32_value;
    float f32_value;
    double f64_value;
    /* Canonical type identity, `<module>:<Class>` or a scalar name; compared bytewise. */
    VernonStringView type_value;
    /* A color format; the specialization selects a graphics pipeline's color target format. */
    VernonTextureFormat format_value;
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
 * Floating-point values must be finite and type identities non-empty.
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

typedef enum VernonTensorStorage { VERNON_TENSOR_HOST = 0, VERNON_TENSOR_DEVICE_RESOURCE = 1 } VernonTensorStorage;

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

typedef struct VernonProgramResourceLease {
    uint32_t struct_size;
    void *object;
    void (*retain)(void *object);
    void (*release)(void *object);
} VernonProgramResourceLease;

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

/* specs/runtime/gpu_profiling.md section 3. A scope is one native render scope, holding `node_count` fused graphics
 * nodes from `first_node`, or one compute node. Begin and end count from the begin of the call's first scope. */
typedef struct VernonScopeTiming {
    uint32_t first_node;
    uint32_t node_count;
    uint32_t reserved[2];
    uint64_t begin_nanoseconds;
    uint64_t end_nanoseconds;
} VernonScopeTiming;

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
VERNON_RUNTIME_CAPI VernonRuntimeOperationStatus vernonRuntimeLoadProgramBundleWithOptionsResult(
    VernonRuntimeContext *context, const void *bundle, size_t bundle_size,
    const VernonProgramBundleLoadOptions *options, VernonProgramBundle **output);
VERNON_RUNTIME_CAPI VernonStringView vernonRuntimeProgramBundleGetId(const VernonProgramBundle *bundle);
VERNON_RUNTIME_CAPI void vernonRuntimeProgramBundleDestroy(VernonProgramBundle *bundle);
VERNON_RUNTIME_CAPI VernonProgramGraph *vernonRuntimeProgramGraphCreate(VernonRuntimeContext *context);
VERNON_RUNTIME_CAPI VernonRuntimeOperationStatus vernonRuntimeProgramGraphCreateResult(VernonRuntimeContext *context,
                                                                                       VernonProgramGraph **output);
VERNON_RUNTIME_CAPI void vernonRuntimeProgramGraphDestroy(VernonProgramGraph *graph);
/* `label` names the child's Nodes in debug groups and diagnostics: the label itself for a child with one Node,
 * otherwise `label/<Node name>`. An empty label is the child's bundle id. Labels never enter identity. */
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeProgramGraphAddProgram(VernonProgramGraph *graph,
                                                                     const VernonProgramBundle *bundle,
                                                                     const VernonProgramVariantSelector *selector,
                                                                     VernonStringView label, VernonProgramNodeId *node);
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
VERNON_RUNTIME_CAPI VernonRuntimeOperationStatus
vernonRuntimeResolveProgramGraphResult(VernonProgramGraph *graph, VernonProgramExecutable **output);
VERNON_RUNTIME_CAPI VernonProgramExecutable *vernonRuntimeResolveProgram(VernonProgramBundle *bundle,
                                                                         const VernonProgramVariantSelector *selector);
VERNON_RUNTIME_CAPI VernonRuntimeOperationStatus vernonRuntimeResolveProgramResult(
    VernonProgramBundle *bundle, const VernonProgramVariantSelector *selector, VernonProgramExecutable **output);
VERNON_RUNTIME_CAPI void vernonRuntimeProgramExecutableDestroy(VernonProgramExecutable *pipeline);
VERNON_RUNTIME_CAPI VernonStringView vernonRuntimeProgramExecutableGetId(const VernonProgramExecutable *pipeline);
VERNON_RUNTIME_CAPI size_t vernonRuntimeProgramExecutableGetParameterCount(const VernonProgramExecutable *pipeline);
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
/* The control slots a graphics node reads its per-call controls from.
 *
 * A Program assigns each control its own slot when it is compiled, so a caller cannot assume any particular number
 * and must ask before binding. `node` is the graph node id the controls belong to; in a composed Program graph it
 * counts the forward nodes of every earlier Program, as the graph's linked schedule numbers them, and the control
 * slots are those of the Program that draws the node. */
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
/*
 * Recorded draws (specs/program/recorded_draws.md). A pipeline draws inside graphics Node `node` of `executable`;
 * `source` is a resolved one-node graphics Program on the same context that conforms to the Node. A null `source`
 * returns pipeline 0, the Node's own Stage; a composed Program graph takes a `source`. Pipelines and binding sets
 * retain the executables they refer to, so those
 * executables cannot be destroyed while either is alive.
 */
typedef struct VernonGraphicsPipeline VernonGraphicsPipeline;
typedef struct VernonBindingSet VernonBindingSet;

VERNON_RUNTIME_CAPI VernonRuntimeOperationStatus
vernonRuntimeGraphicsPipelineCreate(VernonProgramExecutable *executable, VernonProgramNodeId node,
                                    VernonProgramExecutable *source, VernonGraphicsPipeline **output);
VERNON_RUNTIME_CAPI void vernonRuntimeGraphicsPipelineDestroy(VernonGraphicsPipeline *pipeline);

/* `argument->slot` is the slot of a per-draw parameter of the pipeline's Program, found through
 * vernonRuntimeProgramExecutableFindParameter on that Program (`executable` for pipeline 0, otherwise `source`). The
 * argument must match the parameter's boundary contract and must not need a Runtime upload. A host Tensor's bytes are
 * copied when bound; device resources stay borrowed. */
VERNON_RUNTIME_CAPI VernonRuntimeOperationStatus vernonRuntimeBindingSetCreate(VernonGraphicsPipeline *pipeline,
                                                                               VernonBindingSet **output);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeBindingSetBind(VernonBindingSet *set,
                                                             const VernonProgramArgument *argument,
                                                             const VernonProgramResourceLease *lease);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeBindingSetSeal(VernonBindingSet *set);
VERNON_RUNTIME_CAPI void vernonRuntimeBindingSetDestroy(VernonBindingSet *set);

typedef enum VernonGraphicsCommandKind {
    VERNON_GRAPHICS_COMMAND_BIND_PIPELINE = 0,
    VERNON_GRAPHICS_COMMAND_BIND_SET = 1,
    VERNON_GRAPHICS_COMMAND_SET_VIEWPORT = 2,
    VERNON_GRAPHICS_COMMAND_SET_SCISSOR = 3,
    VERNON_GRAPHICS_COMMAND_SET_STENCIL_REFERENCE = 4,
    VERNON_GRAPHICS_COMMAND_DRAW = 5,
    /* Debug groups and markers name the draws between them in capture tools and have no effect on rendering. */
    VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP = 6,
    VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP = 7,
    VERNON_GRAPHICS_COMMAND_INSERT_DEBUG_MARKER = 8
} VernonGraphicsCommandKind;

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

#ifdef __cplusplus
}
#endif

#endif
