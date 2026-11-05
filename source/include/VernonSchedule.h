#ifndef VERNON_C_SCHEDULE_H
#define VERNON_C_SCHEDULE_H

/*
 * Linked schedules (specs/runtime/architecture_v2.md). A linked schedule is recorded once on its context's device; a
 * call writes the schedule's argument record and submits it. The record's layout is fixed when the schedule is
 * linked and described by its slots and fields, so a caller packs arguments at known offsets without the Runtime
 * planning anything per call.
 */

#include "VernonRuntime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VernonLinkedSchedule VernonLinkedSchedule;

typedef enum VernonScheduleSlotKind {
    /* The address of a TensorView's buffer, pointer-sized; the view's offset is a field. */
    VERNON_SCHEDULE_SLOT_BUFFER = 1,
    /* The bytes of a scalar argument. */
    VERNON_SCHEDULE_SLOT_SCALAR = 2,
    /*
     * A VernonScheduleTape pointer, pointer-sized: the autodiff tape a CPU forward kernel writes and its VJP kernel
     * reads. A forward call writes it anew; a backward call reads what the forward call wrote. VERNON_SCHEDULE_ACCESS_
     * WRITE marks the slot of a forward graph, VERNON_SCHEDULE_ACCESS_READ that of a backward graph.
     */
    VERNON_SCHEDULE_SLOT_TAPE = 3,
    /*
     * A graphics node's argument, a VernonProgramArgument by value: a buffer, image or sampler resource, or a host
     * Tensor whose bytes the call packs into the pipeline's layout. Its pointers are read during the call only.
     */
    VERNON_SCHEDULE_SLOT_ARGUMENT = 4,
    /* A `const VernonRenderPass *`, read during the call: a graphics node's attachments and their operations. */
    VERNON_SCHEDULE_SLOT_PASS = 5,
    /* A VernonDrawCommand by value; its index binding is read during the call. */
    VERNON_SCHEDULE_SLOT_DRAW = 6,
    /*
     * A VernonDynamicState by value: the viewport, scissor and stencil reference of a graphics node's draws. Zero
     * struct_size, as in the record template, draws with the render area as viewport and scissor.
     */
    VERNON_SCHEDULE_SLOT_DYNAMIC = 7,
    /*
     * A VernonScheduleDrawStream by value. Zero struct_size, as in the record template, draws the DRAW slot once;
     * otherwise the node draws the stream's records with the whole index buffer of the DRAW slot's index binding.
     */
    VERNON_SCHEDULE_SLOT_STREAM = 8,
    /*
     * A `const VernonSchedulePassTable *`, read during the call. Null, as in the record template, draws the DRAW and
     * STREAM slots; otherwise the node draws the table's entries in order instead.
     */
    VERNON_SCHEDULE_SLOT_TABLE = 9
} VernonScheduleSlotKind;

/*
 * The indexed draws of a graphics node: the first min(count, max_records) records from first_record on, where count
 * is the uint32_t at count_index of the counts. A host stream points at records and counts the call reads before it
 * returns; a device stream names buffers of them, written on the device before the call's passes run. Writers zero
 * the records past the count they write, so that a record a stale count names draws nothing; a device stream may
 * therefore leave `counts` zero and draw all max_records records. A device stream needs VK_KHR_draw_indirect_count on
 * Vulkan with counts (and multiDrawIndirect always), and OpenGL 4.3, or 4.6 with counts; on OpenGL its
 * index binding starts at offset 0 and its pipelines read `instance_index` only through gl_BaseInstance.
 */
typedef struct VernonScheduleDrawStream {
    uint32_t struct_size;
    uint32_t first_record;
    uint32_t max_records;
    uint32_t count_index;
    const VernonRuntimeProviderDrawRecord *host_records;
    const uint32_t *host_counts;
    VernonRuntimeProviderResourceReference records;
    VernonRuntimeProviderResourceReference counts;
} VernonScheduleDrawStream;

/*
 * A debug command a pass table encodes: a VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP, POP_DEBUG_GROUP or
 * INSERT_DEBUG_MARKER and its label, which the call reads before it returns. A table pops every group it pushes.
 */
typedef struct VernonScheduleDebugCommand {
    uint32_t kind;
    VernonStringView label;
} VernonScheduleDebugCommand;

/*
 * One draw command of a pass table, drawn with its own pipeline and arguments into the node's attachments.
 * `pipeline` was created for the node's Program and Node by vernonRuntimeGraphicsPipelineCreate; null draws with the
 * node's own Stage. `sets` are the sealed binding sets bound since the previous entry of the table, in bind order: as
 * in a graphics recording, a set's entries stay bound for later entries, and across a pipeline change to the
 * per-draw parameters of the same path. `arguments` are arguments of the pipeline's Program by boundary slot, as
 * vernonRuntimeBindingSetBind takes them, which take precedence over the sets for this entry only. A per-draw
 * parameter neither supplies reads the node's ARGUMENT slot when the pass Node's own Stage reads it there. Zero
 * `stream.struct_size` draws `draw` once; otherwise the stream's records index the whole buffer of
 * `draw.index_binding`, which must be set, and the counts of `draw` are unused. Zero `dynamic.struct_size` draws with
 * the node's DYNAMIC slot. `debug_commands` are encoded before the entry draws.
 */
typedef struct VernonSchedulePassEntry {
    uint32_t struct_size;
    const VernonGraphicsPipeline *pipeline;
    const VernonProgramArgument *arguments;
    size_t argument_count;
    VernonDrawCommand draw;
    VernonScheduleDrawStream stream;
    VernonDynamicState dynamic;
    const VernonBindingSet *const *sets;
    size_t set_count;
    const VernonScheduleDebugCommand *debug_commands;
    size_t debug_command_count;
} VernonSchedulePassEntry;

/*
 * Nonzero `revision` names the table's contents: a call may reuse what an earlier call planned for a table at the same
 * address with the same revision, as long as the node's arguments keep their resources and shapes and its attachments
 * their formats and extent. A caller gives each table it builds a revision no earlier table had, and while it keeps
 * one only the host stream records and counts, and the host bytes of the node's arguments, change. Zero plans the table
 * at every call. `debug_commands` are encoded after the last entry.
 */
typedef struct VernonSchedulePassTable {
    uint32_t struct_size;
    const VernonSchedulePassEntry *entries;
    size_t entry_count;
    uint64_t revision;
    const VernonScheduleDebugCommand *debug_commands;
    size_t debug_command_count;
} VernonSchedulePassTable;

/* An autodiff tape that a caller holds from a forward call until the backward call that reads it. */
typedef struct VernonScheduleTape VernonScheduleTape;

/* An empty tape; null when it cannot be allocated. */
VERNON_RUNTIME_CAPI VernonScheduleTape *vernonScheduleTapeCreate(void);
VERNON_RUNTIME_CAPI void vernonScheduleTapeDestroy(VernonScheduleTape *tape);

typedef enum VernonScheduleAccess {
    VERNON_SCHEDULE_ACCESS_READ = 1,
    VERNON_SCHEDULE_ACCESS_WRITE = 2,
    VERNON_SCHEDULE_ACCESS_READ_WRITE = 3
} VernonScheduleAccess;

/* One launch argument, in the order the kernel declares them, builtins excluded. */
typedef struct VernonScheduleSlot {
    uint32_t kind;
    /* VernonDataType of a scalar, or of a TensorView's elements. */
    uint32_t dtype;
    /* A TensorView's rank, or zero. */
    uint32_t rank;
    /* VernonScheduleAccess of a TensorView, or zero. */
    uint32_t access;
    uint64_t element_size;
    uint64_t offset;
    uint64_t size;
} VernonScheduleSlot;

typedef enum VernonScheduleFieldKind {
    /* The view's first element, in elements from its buffer's start. */
    VERNON_SCHEDULE_FIELD_OFFSET = 1,
    VERNON_SCHEDULE_FIELD_EXTENT = 2,
    /* In elements. */
    VERNON_SCHEDULE_FIELD_STRIDE = 3,
    /* The buffer's address again, as 8 bytes, for an ABI that passes it twice. */
    VERNON_SCHEDULE_FIELD_ADDRESS = 4,
    /* The buffer's length in elements. */
    VERNON_SCHEDULE_FIELD_CAPACITY = 5,
    /* A scalar slot's bytes again, at a further binding of the slot; `size` is the slot's size. */
    VERNON_SCHEDULE_FIELD_VALUE = 6,
    /* The buffer's length in bytes, as 8 bytes, for the fill that zeroes a buffer the schedule starts as zeros. */
    VERNON_SCHEDULE_FIELD_BYTES = 7
} VernonScheduleFieldKind;

/* An extent a caller supplies: `value` when `slot` is UINT32_MAX, else extent `dimension` of buffer slot `slot`. */
typedef struct VernonScheduleExtent {
    int64_t value;
    uint32_t slot;
    uint32_t dimension;
} VernonScheduleExtent;

/*
 * One integer of a TensorView's metadata, or a repeated address or scalar. A 4-byte field holds a value that must fit
 * in int32_t. A slot bound several times has these fields once per binding.
 */
typedef struct VernonScheduleField {
    /* Index of the slot. */
    uint32_t slot;
    uint32_t kind;
    /* The dimension of an extent or stride, or zero. */
    uint32_t dimension;
    uint32_t size;
    uint64_t offset;
} VernonScheduleField;

/* One compiled kernel Stage: its reflection, and its device code. */
typedef struct VernonScheduleKernelDescriptor {
    uint32_t struct_size;
    /* The Stage's reflection JSON and the entry in it. */
    VernonStringView reflection;
    VernonStringView entry;
    /* The entry of a CPU Stage, which the caller keeps loaded for the schedule's lifetime. */
    VernonCpuEntryPoint cpu_entry;
    /* The PTX of a CUDA Stage; the schedule keeps its own copy. */
    const void *cuda_artifact;
    size_t cuda_artifact_size;
    /* The CPU Stage's chunk entry, when it has one; a dispatch then runs spans of whole workgroups through it. */
    VernonCpuChunkEntryPoint cpu_chunk_entry;
} VernonScheduleKernelDescriptor;

/*
 * Links one kernel as a one-node graph on the context's device and records it. Returns null with the context's last
 * error set when the kernel uses a feature linked schedules do not support yet; such a kernel runs as a Program.
 */
VERNON_RUNTIME_CAPI VernonLinkedSchedule *vernonScheduleLinkKernel(VernonRuntimeContext *context,
                                                                   const VernonScheduleKernelDescriptor *descriptor);

/* One logical Stage of a Program variant, compiled: its name in the variant's `stages`, and its kernel. */
typedef struct VernonScheduleStage {
    VernonStringView name;
    VernonScheduleKernelDescriptor kernel;
} VernonScheduleStage;

typedef struct VernonScheduleProgramDescriptor {
    uint32_t struct_size;
    /* The variant's `program` JSON, and the graph to link, such as "forward". */
    VernonStringView program;
    VernonStringView graph;
    const VernonScheduleStage *stages;
    size_t stage_count;
    /* The owned Storages that start as zeros, by id; the manifest leaves that to the caller's allocation recipe. */
    const uint32_t *zeroed_storages;
    size_t zeroed_storage_count;
    /*
     * Storages, by id, that the graph shares with another graph of the Program and exposes as slots after its boundary
     * slots, in this order: an allocation of the graph is an output the caller supplies and keeps (a VJP forward's
     * residual), and any other is an input (a value the backward graph captures).
     */
    const uint32_t *exposed_storages;
    size_t exposed_storage_count;
    /*
     * The resolved Program, whose Stages it prepared; `program`, `stages` and `zeroed_storages` are then ignored, and
     * it must outlive the schedule. vernonScheduleSlotBoundary names the boundary slot of each ARGUMENT slot.
     *
     * On a graphics device the slots are the graph's input boundary slots, and its output boundary slots of buffer
     * Storages no input boundary slot has, as ARGUMENT slots in boundary slot order, then a PASS, a DRAW, a DYNAMIC, a
     * STREAM and a TABLE slot per graphics node; a compute node, which dispatches between the passes in graph order,
     * has no slot of its own. The caller allocates each such output; the attachments a node renders to are its other
     * outputs.
     *
     * On the CPU and CUDA the graph is compute only. Its slots are an ARGUMENT slot per public boundary slot of the
     * graph, outputs included, in boundary slot order; a call ignores the argument of an output that republishes an
     * input's Storage. The schedule has no fields, shapes or extents: each call reads the arguments' TensorViews.
     */
    VernonProgramExecutable *executable;
} VernonScheduleProgramDescriptor;

/*
 * Links one graph of a Program as a one-node graph and records it. Its slots are the ABI's boundary slots of that
 * graph in order, outputs included, then its exposed Storages: the caller supplies every output, sizing a fresh one
 * from vernonScheduleSlotExtents. Returns null with the context's last error set when the Program uses a feature
 * linked schedules do not support yet; it then runs as a Program.
 */
VERNON_RUNTIME_CAPI VernonLinkedSchedule *vernonScheduleLinkProgram(VernonRuntimeContext *context,
                                                                    const VernonScheduleProgramDescriptor *descriptor);

/* Slot `slot` of graph node `node`, numbered as vernonScheduleLinkProgram numbers that node's slots. */
typedef struct VernonScheduleNodeSlot {
    uint32_t node;
    uint32_t slot;
} VernonScheduleNodeSlot;

/*
 * Node slots that are one buffer or one scalar: a Value edge's producer and consumers, or a Storage chain's versions.
 * Their kinds, element types and ranks must match. An `internal` connection is a buffer only the graph uses: the
 * schedule allocates it from the extents its slots declare, and it is no slot of the schedule.
 */
typedef struct VernonScheduleConnection {
    const VernonScheduleNodeSlot *sites;
    size_t site_count;
    uint32_t internal;
} VernonScheduleConnection;

typedef struct VernonScheduleGraphDescriptor {
    uint32_t struct_size;
    /* The nodes in the order they run. */
    const VernonScheduleProgramDescriptor *nodes;
    size_t node_count;
    const VernonScheduleConnection *connections;
    size_t connection_count;
} VernonScheduleGraphDescriptor;

/*
 * Links graph nodes, each one graph of a Program, into one schedule and records it. Its slots are each node's slots
 * in node order, a connection's at its first site, and none for an internal connection; vernonScheduleNodeSlot maps
 * them. Returns null with the context's last error set when a node does not link or the connections disagree.
 */
VERNON_RUNTIME_CAPI VernonLinkedSchedule *vernonScheduleLinkGraph(VernonRuntimeContext *context,
                                                                  const VernonScheduleGraphDescriptor *descriptor);
/* The number of slots of graph node `node`; a schedule linked from one kernel or Program is node 0. */
VERNON_RUNTIME_CAPI size_t vernonScheduleNodeSlotCount(const VernonLinkedSchedule *schedule, size_t node);
/* The schedule slot of slot `slot` of node `node`; UINT32_MAX for an internal connection or no such slot. */
VERNON_RUNTIME_CAPI uint32_t vernonScheduleNodeSlot(const VernonLinkedSchedule *schedule, size_t node, size_t slot);
/*
 * The Program boundary slot (VernonProgramArgument::slot) an ARGUMENT slot of a schedule linked from an executable
 * takes; UINT32_MAX for any other slot.
 */
VERNON_RUNTIME_CAPI uint32_t vernonScheduleSlotBoundary(const VernonLinkedSchedule *schedule, size_t slot);
VERNON_RUNTIME_CAPI void vernonScheduleDestroy(VernonLinkedSchedule *schedule);

VERNON_RUNTIME_CAPI size_t vernonScheduleRecordSize(const VernonLinkedSchedule *schedule);
/* The record's initial bytes, which hold the constants no slot or field writes; a caller starts each record from it. */
VERNON_RUNTIME_CAPI const uint8_t *vernonScheduleRecordTemplate(const VernonLinkedSchedule *schedule);
/* Offset of the three uint32_t workgroup counts of the call, or UINT64_MAX when the schedule fixes its grids. */
VERNON_RUNTIME_CAPI uint64_t vernonScheduleGridOffset(const VernonLinkedSchedule *schedule);
VERNON_RUNTIME_CAPI size_t vernonScheduleSlotCount(const VernonLinkedSchedule *schedule);
VERNON_RUNTIME_CAPI const VernonScheduleSlot *vernonScheduleSlots(const VernonLinkedSchedule *schedule);
/*
 * The `rank` declared extents of a TensorView slot, -1 where dynamic; for an ARGUMENT slot, those of the linked slot
 * vernonScheduleArgumentSlot describes.
 */
VERNON_RUNTIME_CAPI const int64_t *vernonScheduleSlotShape(const VernonLinkedSchedule *schedule, size_t slot);
/*
 * The linked slot an ARGUMENT slot of a schedule linked from a compute Program executable writes: its kind (BUFFER or
 * SCALAR), element type and size and rank, with the union of the accesses of every node slot the argument feeds. Kind
 * zero for an argument that writes no slot itself, an output that republishes an input's Storage. Null for any other
 * slot.
 */
VERNON_RUNTIME_CAPI const VernonScheduleSlot *vernonScheduleArgumentSlot(const VernonLinkedSchedule *schedule,
                                                                         size_t slot);
/*
 * The `rank` extents a TensorView slot's argument must have: each is static, the slot's own (free), or another slot's,
 * so a caller sizes an output it allocates from its inputs. The call checks them.
 */
VERNON_RUNTIME_CAPI const VernonScheduleExtent *vernonScheduleSlotExtents(const VernonLinkedSchedule *schedule,
                                                                          size_t slot);
VERNON_RUNTIME_CAPI size_t vernonScheduleFieldCount(const VernonLinkedSchedule *schedule);
VERNON_RUNTIME_CAPI const VernonScheduleField *vernonScheduleFields(const VernonLinkedSchedule *schedule);

/*
 * Submits the schedule with `record`, which must be vernonScheduleRecordSize bytes with every slot and field written.
 * The caller checks its arguments against the slots; the call checks only the extents each slot must have and the
 * grid, and places the schedule's own allocations. Calls of one schedule run one at a time. On the CPU the call returns
 * after the work completes. On CUDA it returns once the work is enqueued on the device's stream, which the device's
 * copies also use, so reading a written buffer back orders after it; the record may be reused at once. On Vulkan and
 * OpenGL it returns once the passes are submitted to the device's queue; the RenderPass, index bindings and host
 * argument bytes the record points to are read during the call only. `point`, when not null, receives the point on the
 * context's device timeline that the call's submission signals.
 */
VERNON_RUNTIME_CAPI VernonStatus vernonScheduleCall(VernonLinkedSchedule *schedule, const void *record, size_t size,
                                                    uint64_t *point);

/* What the last call of a graphics schedule drew. */
typedef struct VernonScheduleGraphicsTelemetry {
    uint32_t struct_size;
    /* Adjacent passes into the same attachments, where the later one preserves them, share a render scope. */
    uint64_t render_scopes;
    /* Draw commands the passes encoded; a draw stream is one. */
    uint64_t draws;
} VernonScheduleGraphicsTelemetry;

/* Zero for a schedule that draws nothing or was not called yet. */
VERNON_RUNTIME_CAPI VernonStatus vernonScheduleGetGraphicsTelemetry(const VernonLinkedSchedule *schedule,
                                                                    VernonScheduleGraphicsTelemetry *telemetry);

/*
 * Nonzero `enabled` makes later calls of the schedule time each render scope and each compute dispatch
 * (specs/runtime/gpu_profiling.md): on Vulkan and OpenGL with device timers, on CUDA with events, which keep the
 * call off its CUDA graph, and on the CPU with the host clock. Zero, the default, times nothing.
 */
VERNON_RUNTIME_CAPI VernonStatus vernonScheduleSetTimings(VernonLinkedSchedule *schedule, uint32_t enabled);

/*
 * The timed scopes of the call that signalled `point`, in execution order, in nanoseconds from the begin of its first
 * scope; a call is one device submission. Null `scopes` queries `count`; a smaller `capacity` than `count` is
 * INVALID_ARGUMENT. NOT_READY until a vernonScheduleWait for the point returned; UNSUPPORTED_TARGET when the call was
 * not timed, the device could not time it, or the device dropped its timings, keeping only its latest few calls'.
 */
VERNON_RUNTIME_CAPI VernonStatus vernonScheduleGetScopeTimings(const VernonLinkedSchedule *schedule, uint64_t point,
                                                               VernonScopeTiming *scopes, size_t capacity,
                                                               size_t *count);

/*
 * Blocks until the context's device has completed the submission that signalled `point`, and every earlier one. Zero
 * and points of a context that has linked no schedule complete at once; a point not yet signalled is an error.
 */
VERNON_RUNTIME_CAPI VernonStatus vernonScheduleWait(VernonRuntimeContext *context, uint64_t point);

/*
 * A linked VJP: the forward and backward graphs of a resolved Program linked as two schedules, which run through
 * vernonScheduleVjpForward and vernonScheduleVjpBackward only. The forward call writes the Values the backward graph
 * reads, other than the forward arguments, into a residual block the backward call reads.
 */
typedef struct VernonScheduleVjp VernonScheduleVjp;
typedef struct VernonScheduleResiduals VernonScheduleResiduals;

/*
 * Links a Program resolved from a bundle whose variant has exactly a forward and a backward graph, or a composed
 * Program graph of such Programs, on any device that links schedules. A composed graph's backward schedule runs its
 * nodes' backward graphs in reverse, each node's input gradient feeding the cotangent of the output it reads; a Value
 * read by several
 * nodes, or by a node and the caller, is not linked yet. Returns null with the context's last error set when a graph
 * does not link; the executable must outlive the VJP.
 */
VERNON_RUNTIME_CAPI VernonScheduleVjp *vernonScheduleLinkVjp(VernonRuntimeContext *context,
                                                             VernonProgramExecutable *executable);

/*
 * How a checkpointed VJP picks the forward nodes its pullback runs again. Each forward call keeps the residuals of the
 * forward nodes from some k on and the pullback runs nodes 0 through k - 1 again for the others; k covers every node
 * that records an autodiff tape, whose size is known only once it ran. BALANCED and MIN_RUNTIME take the smallest k
 * whose held bytes, the forward inputs and the kept residuals, fit the memory budget; MIN_MEMORY runs every node
 * again. A call no k fits fails with VERNON_STATUS_INVALID_ARGUMENT.
 */
typedef enum VernonScheduleCheckpointPolicy {
    VERNON_SCHEDULE_CHECKPOINT_BALANCED = 0,
    VERNON_SCHEDULE_CHECKPOINT_MIN_MEMORY = 1,
    VERNON_SCHEDULE_CHECKPOINT_MIN_RUNTIME = 2,
} VernonScheduleCheckpointPolicy;

typedef struct VernonScheduleCheckpointOptions {
    size_t struct_size;
    uint64_t memory_budget;
    VernonScheduleCheckpointPolicy policy;
} VernonScheduleCheckpointOptions;

/*
 * vernonScheduleLinkVjp of a Program resolved from a bundle whose forward calls hold at most `memory_budget` bytes
 * until their pullback. The forward graph must not read and write one argument, which a pullback reads again; a
 * pullback that runs forward nodes again writes their outputs to buffers of its own.
 */
VERNON_RUNTIME_CAPI VernonScheduleVjp *
vernonScheduleLinkCheckpointedVjp(VernonRuntimeContext *context, VernonProgramExecutable *executable,
                                  const VernonScheduleCheckpointOptions *options);
VERNON_RUNTIME_CAPI void vernonScheduleVjpDestroy(VernonScheduleVjp *vjp);
/*
 * The forward (`backward` zero) or backward schedule, owned by the VJP, for its slot and record queries. Their slots
 * are ARGUMENT slots, as a compute Program's from vernonScheduleLinkProgram are: the forward schedule's are the forward
 * graph's parameters, and the backward's are a cotangent per differentiated output and a gradient per differentiated
 * input, in boundary slot order. vernonScheduleSlotBoundary names the primal boundary slot a backward slot
 * differentiates, and vernonScheduleVjpSlotRole whether it takes a cotangent or a gradient.
 */
VERNON_RUNTIME_CAPI const VernonLinkedSchedule *vernonScheduleVjpSchedule(const VernonScheduleVjp *vjp,
                                                                          uint32_t backward);
/* VERNON_PROGRAM_BOUNDARY_COTANGENT or _GRADIENT for a backward ARGUMENT slot, and _INPUT for no such slot. */
VERNON_RUNTIME_CAPI VernonProgramBoundaryRole vernonScheduleVjpSlotRole(const VernonScheduleVjp *vjp, size_t slot);
/*
 * Calls the forward schedule with `record` and returns the call's residual block. The forward arguments the backward
 * graph reads must stay valid and unchanged until the backward call, and the block until both calls completed.
 */
VERNON_RUNTIME_CAPI VernonStatus vernonScheduleVjpForward(VernonScheduleVjp *vjp, const void *record, size_t size,
                                                          VernonScheduleResiduals **residuals, uint64_t *point);
/*
 * Calls the backward schedule with `record` and the residuals of the forward call it differentiates. It overwrites
 * each gradient argument, whatever the buffer held before; a scalar gradient is a buffer of one element.
 */
VERNON_RUNTIME_CAPI VernonStatus vernonScheduleVjpBackward(VernonScheduleVjp *vjp, const void *record, size_t size,
                                                           const VernonScheduleResiduals *residuals, uint64_t *point);
/* Blocks until every call that read `residuals` completed, then frees them; a failed call is reported only by
 * vernonScheduleWait of its point. */
VERNON_RUNTIME_CAPI void vernonScheduleResidualsDestroy(VernonScheduleResiduals *residuals);
/* The bytes of the residuals the block keeps, its autodiff tapes excluded; zero for null. */
VERNON_RUNTIME_CAPI uint64_t vernonScheduleResidualsGetByteSize(const VernonScheduleResiduals *residuals);

/*
 * What the forward call of a checkpointed VJP kept, `present` zero otherwise. `held_bytes` is its inputs
 * (`initial_state_bytes`) and kept residuals (`kept_bytes`); its pullback runs `recomputed_nodes` of the
 * `forward_nodes` forward nodes again to produce `recomputed_bytes` of residuals and every autodiff tape.
 */
typedef struct VernonScheduleCheckpoint {
    size_t struct_size;
    uint32_t present;
    VernonScheduleCheckpointPolicy policy;
    uint64_t memory_budget;
    uint64_t held_bytes;
    uint64_t initial_state_bytes;
    uint64_t kept_bytes;
    uint64_t recomputed_bytes;
    uint32_t recomputed_nodes;
    uint32_t forward_nodes;
} VernonScheduleCheckpoint;

VERNON_RUNTIME_CAPI VernonStatus vernonScheduleResidualsGetCheckpoint(const VernonScheduleResiduals *residuals,
                                                                      VernonScheduleCheckpoint *checkpoint);

/*
 * What one forward node of a linked VJP call left in the call's residual block: the autodiff tapes it recorded and the
 * residual Storages it wrote that the backward graph reads. A composed graph's passes are its nodes' forward nodes, in
 * node order. `name` lives as long as the VJP.
 */
typedef struct VernonScheduleVjpPass {
    size_t struct_size;
    uint32_t node;
    VernonStringView name;
    uint32_t tape_count;
    uint32_t dynamic_tape;
    uint32_t control_history;
    uint64_t tape_logical_bytes;
    uint64_t tape_resident_bytes;
    uint64_t tape_allocated_bytes;
    uint64_t residual_bytes;
    /* Nonzero when the call's pullback runs the node again rather than reading what it left. */
    uint32_t recomputed;
} VernonScheduleVjpPass;

VERNON_RUNTIME_CAPI size_t vernonScheduleVjpPassCount(const VernonScheduleVjp *vjp);
VERNON_RUNTIME_CAPI VernonStatus vernonScheduleVjpGetPass(const VernonScheduleVjp *vjp,
                                                          const VernonScheduleResiduals *residuals, size_t pass,
                                                          VernonScheduleVjpPass *output);

#ifdef __cplusplus
}
#endif

#endif
