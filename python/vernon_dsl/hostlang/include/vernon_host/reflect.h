/* The reflection table of a compiled Host module, the same in ahead-of-time objects and Python mode.
 *
 * A module exports one `const VhModule <prefix>module`. Every Host type its functions and classes use has a
 * `VhType`; a class lists its fields with their offsets and metadata, its methods, and its initializer, so an
 * application can allocate, initialize, inspect, edit and call instances knowing only this table. Pointers in it
 * stay valid while the object is loaded.
 *
 * Every function in the table has the Host convention: `int32_t f(params..., Result *out)`, returning zero or a
 * nonzero status; scalars and handles pass by value, aggregates by pointer, a str as a `const VhStrArg *` the callee
 * copies, and methods take the instance first.
 *
 * Records that may gain members start with their `struct_size`, and later versions only append members. Code built
 * against this header reads a newer table by indexing record arrays through the `vh_*_at` accessors, which stride by
 * the table's `struct_size`, and by reading only members below it.
 */
#ifndef VERNON_HOST_REFLECT_H
#define VERNON_HOST_REFLECT_H

#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VH_REFLECT_VERSION 3u

typedef enum VhKind {
    VH_KIND_BOOL = 1,
    VH_KIND_I32 = 2,
    VH_KIND_U32 = 3,
    VH_KIND_I64 = 4,
    VH_KIND_U64 = 5,
    VH_KIND_F32 = 6,
    VH_KIND_F64 = 7,
    /* A struct holding one uint64_t. */
    VH_KIND_HANDLE = 8,
    /* `count` elements of `element`, row-major over `shape`. */
    VH_KIND_ARRAY = 9,
    /* An immutable value type. */
    VH_KIND_STRUCT = 10,
    /* Fields named item0, item1, ... */
    VH_KIND_TUPLE = 11,
    /* Mutable state with methods; instances are initialized by `init`, not by copying. */
    VH_KIND_CLASS = 12,
    /* `list[element]`, a VhList record of `element` values. */
    VH_KIND_LIST = 13,
    /* `str`, a VhList record of UTF-8 bytes. */
    VH_KIND_STR = 14,
    /* `dict[key, element]`, a VhDict record. */
    VH_KIND_DICT = 15,
    /* `TensorStorage[element, (e0, ...)]`, a VH_STORAGE record of `rank` axes; `shape` holds each axis's static
     * extent, 0 for a dyn one. */
    VH_KIND_STORAGE = 16,
    /* An IntEnum, an int32_t; `element` is the i32 type and `enumerators` lists its members. */
    VH_KIND_ENUM = 17
} VhKind;

/* The memory of a `list[T]` or `str` class field: `capacity` elements at the address `data`, the first `size` of
 * them live. The address is a uint64_t so layouts are the same on every target. A zeroed record is empty and owns
 * nothing; a nonzero `capacity` is memory the module's allocator gave, which the owning instance's `fini` returns. */
typedef struct VhList {
    uint64_t data;
    uint64_t size;
    uint64_t capacity;
} VhList;

static inline void *vh_list_data(const VhList *list) { return (void *)(uintptr_t)list->data; }

/* The memory of a `dict[K, V]` class field. `entries` holds `capacity` entries `{uint64_t hash; K key; V value;}`
 * under C layout rules (vh_dict_layout gives the offsets), of which the first `used` were written, in insertion
 * order; `size` of them are live, and a removed one has the hash VH_DICT_REMOVED. A str key is a VhList the dict
 * owns. `slots` is the dict's open-addressing index of `slot_count` uint64_t. A zeroed record is empty and owns
 * nothing. Only VernonRuntime's vernonRuntimeHostDict* functions change one. */
typedef struct VhDict {
    uint64_t entries;
    uint64_t size;
    uint64_t used;
    uint64_t capacity;
    uint64_t slots;
    uint64_t slot_count;
} VhDict;

#define VH_DICT_REMOVED UINT64_MAX

/* The memory of a `TensorStorage[T, (e0, ...)]` field: `{uint64_t data; uint64_t shape[rank];}`, the dense row-major
 * elements at the address `data` and the extent of each axis. A zeroed record has no elements and owns nothing; a
 * record with elements owns memory the module's allocator gave, which the owning instance's `fini` returns. Elements
 * are bool, int32_t, uint32_t, float or double, aligned to their size. In Python mode a `vd.TensorStorage` reads the
 * field, so synchronize it before C reads elements a device wrote. */
static inline void *vh_storage_data(const void *storage) { return (void *)(uintptr_t)*(const uint64_t *)storage; }
static inline const uint64_t *vh_storage_shape(const void *storage) { return (const uint64_t *)storage + 1; }
static inline uint64_t vh_storage_count(const void *storage, uint32_t rank) {
    uint64_t count = 1;
    for (uint32_t axis = 0; axis < rank; ++axis) {
        count *= vh_storage_shape(storage)[axis];
    }
    return count;
}

/* How generated headers spell container fields: the records above in C, and in C++ the typed vh::Str, vh::List,
 * vh::Dict and vh::TensorStorage of vernon_host/containers.hpp, which have the same layout. */
#ifndef __cplusplus
#define VH_STR VhList
#define VH_LIST(T) VhList
#define VH_DICT(K, V) VhDict
#define VH_STORAGE(T, RANK)                                                                                            \
    struct {                                                                                                           \
        uint64_t data;                                                                                                 \
        uint64_t shape[RANK];                                                                                          \
    }
#endif

/* A `str` argument of an extern, passed by value: UTF-8 bytes, not NUL-terminated, that the caller keeps alive only
 * for the call. */
typedef struct VhStrArg {
    const char *data;
    uint64_t size;
} VhStrArg;

/* Where a module's containers get memory. `allocate` returns NULL when it cannot; `release` receives the size and
 * alignment `allocate` was asked for. */
typedef struct VhAllocator {
    uint32_t struct_size;
    void *user;
    void *(*allocate)(void *user, uint64_t size, uint64_t alignment);
    void (*release)(void *user, void *pointer, uint64_t size, uint64_t alignment);
} VhAllocator;

/* A `list[T]` or `dict[K, V]` argument of an extern, passed by value like a VhStrArg: `size` elements, or entries
 * `{K key; V value;}` in insertion order, that the caller keeps alive only for the call. */
typedef struct VhSpan {
    const void *data;
    uint64_t size;
} VhSpan;

/* The status an extern returns when `allocator` cannot give it memory for an owned result. */
#define VH_STATUS_OUT_OF_MEMORY (-4)

/* Fills the empty `*out` of an extern's owned `str`, `list[T]` or `dict[K, V]` result with `count` items of
 * `item_size` bytes (UTF-8 bytes, elements, or entries `{K key; V value;}`) copied from `data`, in memory from
 * `allocator`, the one the extern received. Returns 0 or VH_STATUS_OUT_OF_MEMORY, leaving `*out` empty. The caller
 * releases the record through that allocator, which is never NULL: with no module allocator installed it is
 * VernonRuntime's default, so a library linking another C runtime still allocates from the heap that frees it. */
static inline int32_t vh_list_assign(VhList *out, const void *data, uint64_t count, uint64_t item_size,
                                     uint64_t alignment, const VhAllocator *allocator) {
    if (count == 0) {
        return 0;
    }
    void *memory = allocator->allocate(allocator->user, count * item_size, alignment);
    if (memory == NULL) {
        return VH_STATUS_OUT_OF_MEMORY;
    }
    memcpy(memory, data, (size_t)(count * item_size));
    out->data = (uint64_t)(uintptr_t)memory;
    out->size = count;
    out->capacity = count;
    return 0;
}

typedef enum VhEffect { VH_EFFECT_PURE = 0, VH_EFFECT_READ = 1, VH_EFFECT_WRITE = 2, VH_EFFECT_IO = 3 } VhEffect;

typedef struct VhType VhType;

/* Uninterpreted metadata from `vd.meta(key=value)`; `value` is JSON text. */
typedef struct VhMeta {
    const char *key;
    const char *value;
} VhMeta;

/* A member of an enum. Two members may share a value, as aliases do. */
typedef struct VhEnumerator {
    const char *name;
    int64_t value;
} VhEnumerator;

typedef struct VhField {
    uint32_t struct_size;
    const char *name;
    const VhType *type;
    uint64_t offset;
    /* The docstring following the field; NULL without one. */
    const char *doc;
    const VhMeta *meta;
    uint32_t meta_count;
} VhField;

typedef struct VhParam {
    const char *name;
    const VhType *type;
} VhParam;

/* An extern a function can call, directly or through the functions it calls. */
typedef struct VhCall {
    uint32_t struct_size;
    /* The extern's declared VhEffect. */
    uint32_t effect;
    /* The C symbol the extern binds. */
    const char *symbol;
    /* The Struct a generic extern is bound to at this call, passed as its `type[T]` argument or as the type of its
     * `T` value; NULL for other externs. */
    const VhType *type;
    /* The extern's `meta=` entries. Tables built before these members existed end at `type`: read them only when
     * `struct_size` covers them. */
    const VhMeta *meta;
    uint32_t meta_count;
} VhCall;

typedef struct VhFunction {
    uint32_t struct_size;
    const char *name;
    /* The exported C symbol; NULL for a class's private, leading-underscore methods. */
    const char *symbol;
    const void *address;
    /* A method's parameters exclude the instance it takes first. */
    const VhParam *params;
    uint32_t param_count;
    /* The declared VhEffect bound. */
    uint32_t effect;
    /* NULL for none. */
    const VhType *result;
    /* Every extern the function can reach, one entry per (symbol, type), ordered by symbol then type name. */
    const VhCall *calls;
    uint32_t call_count;
} VhFunction;

typedef int32_t (*VhInit)(void *self);
typedef void (*VhFini)(void *self);

struct VhType {
    uint32_t struct_size;
    uint32_t kind;
    uint32_t alignment;
    uint64_t size;
    /* The Host spelling: a scalar's kind, or the declared name. */
    const char *name;
    /* FNV-1a 64 of the canonical layout: two types with equal hashes hold interchangeable bytes. */
    uint64_t layout_hash;
    const VhType *element;
    const uint32_t *shape;
    uint32_t rank;
    uint32_t field_count;
    const VhField *fields;
    const VhFunction *methods;
    uint32_t method_count;
    VhInit init;
    const char *doc;
    const VhMeta *meta;
    uint32_t meta_count;
    /* Releases what a class instance, list, str, dict or storage owns, leaving it empty; NULL for types that own
     * nothing but
     * their bytes. An application calls it on an instance before freeing the instance's memory. */
    VhFini fini;
    /* A dict's key type, whose values `element` maps; NULL for other kinds. */
    const VhType *key;
    /* An enum's members in declaration order; NULL for other kinds. Read them only when `struct_size` covers
     * them, as for every appended member. */
    const VhEnumerator *enumerators;
    uint32_t enumerator_count;
};

static inline uint64_t vh_align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

/* The byte offsets of a `dict` type's entry members and the entry size, which strides `VhDict.entries`. */
static inline void vh_dict_layout(const VhType *type, uint64_t *key_offset, uint64_t *value_offset,
                                  uint64_t *entry_size) {
    uint64_t alignment = type->key->alignment > 8 ? type->key->alignment : 8;
    alignment = type->element->alignment > alignment ? type->element->alignment : alignment;
    *key_offset = vh_align_up(sizeof(uint64_t), type->key->alignment);
    *value_offset = vh_align_up(*key_offset + type->key->size, type->element->alignment);
    *entry_size = vh_align_up(*value_offset + type->element->size, alignment);
}

typedef struct VhModule {
    uint32_t struct_size;
    uint32_t version;
    uint32_t type_count;
    const char *name;
    /* Of every type layout and function signature, so code built against one revision can check it runs with it. */
    uint64_t hash;
    /* Every struct, tuple, handle and class, dependencies first. */
    const VhType *const *types;
    /* The exported free functions. */
    const VhFunction *functions;
    uint32_t function_count;
    /* The module's allocator slot; a NULL allocator is VernonRuntime's default. Set it through vh_set_allocator. */
    const VhAllocator **allocator;
} VhModule;

/* Installs the allocator `module` takes container memory from, NULL for the default. Change it only while no
 * instance of the module holds container memory, because memory goes back to the allocator current at release.
 * Tables sharing instances (a reload's old and new, or the Python-mode and linked tables) must share it. */
static inline void vh_set_allocator(const VhModule *module, const VhAllocator *allocator) {
    if (module->struct_size > offsetof(VhModule, allocator) && module->allocator != NULL) {
        *module->allocator = allocator;
    }
}

/* The allocator `module` takes container memory from; NULL is the default. Code changing a module's containers
 * outside Host code grows and releases them through this one. */
static inline const VhAllocator *vh_get_allocator(const VhModule *module) {
    if (module->struct_size > offsetof(VhModule, allocator) && module->allocator != NULL) {
        return *module->allocator;
    }
    return NULL;
}

static inline const VhField *vh_field_at(const VhType *type, uint32_t index) {
    return (const VhField *)((const char *)type->fields + (size_t)index * type->fields->struct_size);
}

static inline const VhFunction *vh_method_at(const VhType *type, uint32_t index) {
    return (const VhFunction *)((const char *)type->methods + (size_t)index * type->methods->struct_size);
}

static inline const VhFunction *vh_function_at(const VhModule *module, uint32_t index) {
    return (const VhFunction *)((const char *)module->functions + (size_t)index * module->functions->struct_size);
}

static inline const VhCall *vh_call_at(const VhFunction *function, uint32_t index) {
    return (const VhCall *)((const char *)function->calls + (size_t)index * function->calls->struct_size);
}

static inline const VhType *vh_find_type(const VhModule *module, const char *name) {
    for (uint32_t index = 0; index < module->type_count; ++index) {
        if (strcmp(module->types[index]->name, name) == 0) {
            return module->types[index];
        }
    }
    return NULL;
}

static inline const VhField *vh_find_field(const VhType *type, const char *name) {
    for (uint32_t index = 0; index < type->field_count; ++index) {
        const VhField *field = vh_field_at(type, index);
        if (strcmp(field->name, name) == 0) {
            return field;
        }
    }
    return NULL;
}

static inline const VhFunction *vh_find_method(const VhType *type, const char *name) {
    for (uint32_t index = 0; index < type->method_count; ++index) {
        const VhFunction *method = vh_method_at(type, index);
        if (strcmp(method->name, name) == 0) {
            return method;
        }
    }
    return NULL;
}

static inline const VhFunction *vh_find_function(const VhModule *module, const char *name) {
    for (uint32_t index = 0; index < module->function_count; ++index) {
        const VhFunction *function = vh_function_at(module, index);
        if (strcmp(function->name, name) == 0) {
            return function;
        }
    }
    return NULL;
}

static inline const char *vh_find_meta(const VhMeta *meta, uint32_t count, const char *key) {
    for (uint32_t index = 0; index < count; ++index) {
        if (strcmp(meta[index].key, key) == 0) {
            return meta[index].value;
        }
    }
    return NULL;
}

/* Zeroes the container records of a `type` value whose memory moved elsewhere, so its owner's fini skips them. */
static inline void vh_forget(const VhType *type, void *value) {
    if (type->kind == VH_KIND_LIST || type->kind == VH_KIND_STR || type->kind == VH_KIND_DICT ||
        type->kind == VH_KIND_STORAGE) {
        memset(value, 0, (size_t)type->size);
    } else if (type->kind == VH_KIND_CLASS && type->fini != NULL) {
        for (uint32_t index = 0; index < type->field_count; ++index) {
            const VhField *field = vh_field_at(type, index);
            vh_forget(field->type, (char *)value + field->offset);
        }
    }
}

/* Moves into `to`, an instance of class `to_type` its `init` already ran on, the fields of `from`, an instance of
 * `from_type`, that keep their name and a layout of equal hash, recursing into class fields by name; other fields
 * keep their defaults. Container memory moves rather than copies: the target's default is released, and the
 * source's record is zeroed, so both instances can be finalized. Both tables must use one allocator. This carries
 * live state across a reload whose code changed the class. */
static inline void vh_migrate(const VhType *from_type, void *from, const VhType *to_type, void *to) {
    if (from_type->layout_hash == to_type->layout_hash) {
        if (to_type->fini != NULL) {
            to_type->fini(to);
        }
        memcpy(to, from, (size_t)to_type->size);
        vh_forget(from_type, from);
        return;
    }
    for (uint32_t index = 0; index < to_type->field_count; ++index) {
        const VhField *target = vh_field_at(to_type, index);
        const VhField *source = vh_find_field(from_type, target->name);
        if (source == NULL) {
            continue;
        }
        char *source_bytes = (char *)from + source->offset;
        char *target_bytes = (char *)to + target->offset;
        if (source->type->layout_hash == target->type->layout_hash ||
            (source->type->kind == VH_KIND_CLASS && target->type->kind == VH_KIND_CLASS)) {
            vh_migrate(source->type, source_bytes, target->type, target_bytes);
        }
    }
}

typedef struct VhJsonWriter {
    char *out;
    size_t capacity;
    size_t length;
} VhJsonWriter;

static inline void vh_json_text(VhJsonWriter *writer, const char *text) {
    for (; *text; ++text, ++writer->length) {
        if (writer->length + 1 < writer->capacity) {
            writer->out[writer->length] = *text;
        }
    }
}

static inline void vh_json_float(VhJsonWriter *writer, double value, int digits) {
    char buffer[32];
    if (isnan(value)) {
        vh_json_text(writer, "\"nan\"");
    } else if (isinf(value)) {
        vh_json_text(writer, value > 0 ? "\"inf\"" : "\"-inf\"");
    } else {
        snprintf(buffer, sizeof buffer, "%.*g", digits, value);
        vh_json_text(writer, buffer);
    }
}

static inline void vh_json_value(VhJsonWriter *writer, const VhType *type, const char *bytes);

/* Writes axis `axis` of an array onward, advancing `*element` through its row-major elements. */
static inline void vh_json_axis(VhJsonWriter *writer, const VhType *type, const char *bytes, uint32_t axis,
                                uint64_t *element) {
    vh_json_text(writer, "[");
    for (uint32_t index = 0; index < type->shape[axis]; ++index) {
        if (index) {
            vh_json_text(writer, ",");
        }
        if (axis + 1 < type->rank) {
            vh_json_axis(writer, type, bytes, axis + 1, element);
        } else {
            vh_json_value(writer, type->element, bytes + (*element)++ * type->element->size);
        }
    }
    vh_json_text(writer, "]");
}

/* The same for a storage of `type`, over the shape its record holds. */
static inline void vh_json_storage_axis(VhJsonWriter *writer, const VhType *type, const void *storage, uint32_t axis,
                                        uint64_t *element) {
    const char *data = (const char *)vh_storage_data(storage);
    vh_json_text(writer, "[");
    for (uint64_t index = 0; index < vh_storage_shape(storage)[axis]; ++index) {
        if (index) {
            vh_json_text(writer, ",");
        }
        if (axis + 1 < type->rank) {
            vh_json_storage_axis(writer, type, storage, axis + 1, element);
        } else {
            vh_json_value(writer, type->element, data + (*element)++ * type->element->size);
        }
    }
    vh_json_text(writer, "]");
}

/* The name of an enum's first member with `value`, or NULL if no member has it. */
static inline const char *vh_enum_name(const VhType *type, int32_t value) {
    for (uint32_t index = 0; index < type->enumerator_count; ++index) {
        if (type->enumerators[index].value == value) {
            return type->enumerators[index].name;
        }
    }
    return NULL;
}

static inline void vh_json_value(VhJsonWriter *writer, const VhType *type, const char *bytes) {
    char buffer[32];
    switch (type->kind) {
    case VH_KIND_BOOL:
        vh_json_text(writer, *(const uint8_t *)bytes ? "true" : "false");
        return;
    case VH_KIND_I32:
        snprintf(buffer, sizeof buffer, "%" PRId32, *(const int32_t *)bytes);
        break;
    case VH_KIND_ENUM: {
        /* A value no member has stays a number. */
        const int32_t value = *(const int32_t *)bytes;
        const char *name = vh_enum_name(type, value);
        if (name != NULL) {
            vh_json_text(writer, "\"");
            vh_json_text(writer, name);
            vh_json_text(writer, "\"");
            return;
        }
        snprintf(buffer, sizeof buffer, "%" PRId32, value);
        break;
    }
    case VH_KIND_U32:
        snprintf(buffer, sizeof buffer, "%" PRIu32, *(const uint32_t *)bytes);
        break;
    case VH_KIND_I64:
        snprintf(buffer, sizeof buffer, "%" PRId64, *(const int64_t *)bytes);
        break;
    case VH_KIND_U64:
    case VH_KIND_HANDLE:
        snprintf(buffer, sizeof buffer, "%" PRIu64, *(const uint64_t *)bytes);
        break;
    case VH_KIND_F32:
        vh_json_float(writer, *(const float *)bytes, 9);
        return;
    case VH_KIND_F64:
        vh_json_float(writer, *(const double *)bytes, 17);
        return;
    case VH_KIND_ARRAY: {
        uint64_t element = 0;
        vh_json_axis(writer, type, bytes, 0, &element);
        return;
    }
    case VH_KIND_STORAGE: {
        /* Nested lists over the storage's own shape, as numpy's tolist gives. */
        uint64_t element = 0;
        vh_json_storage_axis(writer, type, bytes, 0, &element);
        return;
    }
    case VH_KIND_LIST: {
        const VhList *list = (const VhList *)bytes;
        const char *data = (const char *)vh_list_data(list);
        vh_json_text(writer, "[");
        for (uint64_t index = 0; index < list->size; ++index) {
            if (index) {
                vh_json_text(writer, ",");
            }
            vh_json_value(writer, type->element, data + index * type->element->size);
        }
        vh_json_text(writer, "]");
        return;
    }
    case VH_KIND_STR: {
        const VhList *text = (const VhList *)bytes;
        const unsigned char *data = (const unsigned char *)vh_list_data(text);
        vh_json_text(writer, "\"");
        for (uint64_t index = 0; index < text->size; ++index) {
            const unsigned char byte = data[index];
            char escaped[8] = {(char)byte, 0};
            if (byte == '"' || byte == '\\') {
                escaped[0] = '\\';
                escaped[1] = (char)byte;
            } else if (byte < 0x20) {
                snprintf(escaped, sizeof escaped, "\\u%04x", byte);
            }
            vh_json_text(writer, escaped);
        }
        vh_json_text(writer, "\"");
        return;
    }
    case VH_KIND_DICT: {
        /* An object of the live entries in insertion order; a key that is not a str is written as its JSON text in
         * a string, "7" or "true". */
        const VhDict *dict = (const VhDict *)bytes;
        uint64_t key_offset, value_offset, entry_size;
        vh_dict_layout(type, &key_offset, &value_offset, &entry_size);
        const char *entries = (const char *)(uintptr_t)dict->entries;
        int first = 1;
        vh_json_text(writer, "{");
        for (uint64_t index = 0; index < dict->used; ++index) {
            const char *entry = entries + index * entry_size;
            uint64_t hash;
            memcpy(&hash, entry, sizeof hash);
            if (hash == VH_DICT_REMOVED) {
                continue;
            }
            vh_json_text(writer, first ? "" : ",");
            first = 0;
            /* str keys and enum member names write their own quotes; other keys are the text of their JSON. */
            const int quoted = type->key->kind != VH_KIND_STR &&
                               (type->key->kind != VH_KIND_ENUM ||
                                vh_enum_name(type->key, *(const int32_t *)(entry + key_offset)) == NULL);
            vh_json_text(writer, quoted ? "\"" : "");
            vh_json_value(writer, type->key, entry + key_offset);
            vh_json_text(writer, quoted ? "\":" : ":");
            vh_json_value(writer, type->element, entry + value_offset);
        }
        vh_json_text(writer, "}");
        return;
    }
    default: {
        const int object = type->kind != VH_KIND_TUPLE;
        vh_json_text(writer, object ? "{" : "[");
        for (uint32_t index = 0; index < type->field_count; ++index) {
            const VhField *field = vh_field_at(type, index);
            if (index) {
                vh_json_text(writer, ",");
            }
            if (object) {
                /* Field names are identifiers, so they need no escaping. */
                vh_json_text(writer, "\"");
                vh_json_text(writer, field->name);
                vh_json_text(writer, "\":");
            }
            vh_json_value(writer, field->type, bytes + field->offset);
        }
        vh_json_text(writer, object ? "}" : "]");
        return;
    }
    }
    vh_json_text(writer, buffer);
}

/* Writes the JSON form of the `type` value at `value` (the form `vernon_dsl.hostlang.value_json` reads and writes)
 * into `out`, truncated to `capacity - 1` bytes and NUL-terminated when `capacity` is nonzero. Returns the full
 * length, as snprintf does, so a caller can size a buffer by calling it with a capacity of zero. */
static inline size_t vh_write_json(const VhType *type, const void *value, char *out, size_t capacity) {
    VhJsonWriter writer = {out, capacity, 0};
    vh_json_value(&writer, type, (const char *)value);
    if (capacity) {
        out[writer.length < capacity ? writer.length : capacity - 1] = '\0';
    }
    return writer.length;
}

#ifdef __cplusplus
}
#endif

#endif /* VERNON_HOST_REFLECT_H */
