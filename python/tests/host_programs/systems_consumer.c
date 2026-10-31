/* Links the AOT object of systems.py and drives it as an application would: through its header, and generically
 * through its reflection table. Prints one `name=value` line per check for the test to compare. */
#include "systems.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static DemoTransform g_transform = {{0.0f, 0.0f, 0.0f}, 1.0f};

int32_t demo_get_transform(DemoEntity entity, DemoTransform *out) {
    (void)entity;
    *out = g_transform;
    return 0;
}

int32_t demo_set_transform(DemoEntity entity, const DemoTransform *value) {
    (void)entity;
    g_transform = *value;
    return 0;
}

/* A store of any reflected Struct, keyed by type and entity, as an engine keeps components it never saw. */
typedef struct Row {
    const VhType *type;
    uint64_t entity;
    unsigned char bytes[64];
} Row;
static Row g_rows[8];
static int g_row_count = 0;

static Row *find_row(const VhType *type, uint64_t entity) {
    for (int index = 0; index < g_row_count; ++index) {
        if (g_rows[index].type->layout_hash == type->layout_hash && g_rows[index].entity == entity) {
            return &g_rows[index];
        }
    }
    return NULL;
}

int32_t demo_read(DemoEntity entity, const VhType *type, void *out) {
    Row *row = find_row(type, entity.bits);
    if (row == NULL) {
        return 7;
    }
    memcpy(out, row->bytes, (size_t)type->size);
    return 0;
}

int32_t demo_write(DemoEntity entity, const VhType *type, const void *value) {
    Row *row = find_row(type, entity.bits);
    if (row == NULL) {
        row = &g_rows[g_row_count++];
        row->type = type;
        row->entity = entity.bits;
    }
    memcpy(row->bytes, value, (size_t)type->size);
    return 0;
}

static const VhField *field(const VhType *type, const char *name) {
    const VhField *found = vh_find_field(type, name);
    if (found == NULL) {
        printf("missing_field=%s\n", name);
        exit(1);
    }
    return found;
}

int main(void) {
    const VhModule *module = &demo_module;
    printf("version=%u\n", module->version);
    printf("name=%s\n", module->name);
    printf("types=%u\n", module->type_count);
    printf("functions=%u\n", module->function_count);

    const VhType *mover = vh_find_type(module, "Mover");
    printf("mover_kind=%u size=%llu align=%u\n", mover->kind, (unsigned long long)mover->size, mover->alignment);
    printf("mover_doc=%s\n", mover->doc);
    printf("mover_role=%s\n", vh_find_meta(mover->meta, mover->meta_count, "role"));
    const VhField *speed = field(mover, "speed");
    printf("speed_offset=%llu doc=%s max=%s\n", (unsigned long long)speed->offset, speed->doc,
           vh_find_meta(speed->meta, speed->meta_count, "max"));
    const VhField *history = field(mover, "history");
    printf("history_kind=%u count=%u element=%s\n", history->type->kind, history->type->shape[0],
           history->type->element->name);

    /* Generic instantiation: allocate by the table, initialize, edit a field by offset, call a method. */
    void *instance = calloc(1, (size_t)mover->size);
    printf("init_status=%d\n", mover->init(instance));
    printf("speed_default=%g\n", *(float *)((char *)instance + speed->offset));
    *(float *)((char *)instance + speed->offset) = 4.0f;
    const VhFunction *update = vh_find_method(mover, "update");
    printf("update_symbol=%s effect=%u params=%u\n", update->symbol, update->effect, update->param_count);
    typedef int32_t (*Update)(void *, double, int32_t *);
    int32_t updates = 0;
    for (int step = 0; step < 3; ++step) {
        ((Update)update->address)(instance, 0.25, &updates);
    }
    printf("updates=%d height=%g\n", updates, g_transform.position[1]);
    printf("private_symbol=%s\n", vh_find_method(mover, "_record")->symbol == NULL ? "null" : "set");

    /* The typed header path reaches the same code. */
    DemoMover typed;
    demo_Mover__init(&typed);
    float height = 0.0f;
    demo_Mover_height_after(&typed, 2.0, &height);
    printf("height_after=%g\n", height);

    /* Generic externs: the C store sees each Struct's descriptor. */
    const VhType *health = vh_find_type(module, "Health");
    DemoHealth start = {4.0f, 5.0f};
    DemoEntity target = {3};
    demo_write(target, health, &start);
    demo_write(target, vh_find_type(module, "Transform"), &g_transform);
    DemoRegen regen;
    demo_Regen__init(&regen);
    float healed = 0.0f;
    int32_t status = demo_Regen_update(&regen, target, 0.5, &healed);
    DemoHealth after;
    demo_read(target, health, &after);
    printf("regen_status=%d healed=%g stored=%g\n", status, healed, after.current);
    printf("health_meta=%s\n", vh_find_meta(health->meta, health->meta_count, "name"));

    /* Migration onto a type of equal layout copies every byte. */
    void *copy = calloc(1, (size_t)mover->size);
    mover->init(copy);
    vh_migrate(mover, instance, mover, copy);
    printf("migrated_speed=%g\n", *(float *)((char *)copy + speed->offset));

    float scaled_value = 0.0f;
    demo_scaled(1.5f, 3, &scaled_value);
    printf("scaled=%g function=%s\n", scaled_value, vh_function_at(module, 0)->name);

    /* Call-site reflection: the externs each method reaches, with the Struct a generic extern receives. */
    const VhFunction *regen_update = vh_find_method(vh_find_type(module, "Regen"), "update");
    for (uint32_t index = 0; index < regen_update->call_count; ++index) {
        const VhCall *call = vh_call_at(regen_update, index);
        printf("regen_call=%s:%u:%s\n", call->symbol, call->effect, call->type == NULL ? "-" : call->type->name);
    }
    for (uint32_t index = 0; index < update->call_count; ++index) {
        const VhCall *call = vh_call_at(update, index);
        printf("mover_call=%s:%u:%s\n", call->symbol, call->effect, call->type == NULL ? "-" : call->type->name);
    }
    /* The JSON form, sized by a first call with no buffer. */
    char json[512];
    size_t needed = vh_write_json(health, &after, NULL, 0);
    vh_write_json(health, &after, json, sizeof json);
    printf("health_json=%s needed=%u\n", json, (unsigned)needed);
    vh_write_json(mover, instance, json, sizeof json);
    printf("mover_json=%s\n", json);
    char small[8];
    vh_write_json(health, &after, small, sizeof small);
    printf("truncated=%s\n", small);
    printf("sizes=%d fini=%s\n",
           module->struct_size == sizeof(VhModule) && mover->struct_size == sizeof(VhType) &&
               speed->struct_size == sizeof(VhField) && update->struct_size == sizeof(VhFunction),
           mover->fini == NULL ? "null" : "set");
    free(copy);
    free(instance);
    return 0;
}
