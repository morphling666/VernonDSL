/* The externs of transfers.py in C, as a library implements them: spans read in place, owned results filled with
 * vh_list_assign through the allocator the caller passes. They compute what the Python bindings of
 * test_host_transfers.py compute. */
#include <stdio.h>

#include "vernon_host/reflect.h"

typedef struct Sample {
    float weight;
    int32_t count;
} Sample;

typedef struct WeightEntry {
    int32_t key;
    float value;
} WeightEntry;

typedef struct SampleEntry {
    int32_t key;
    Sample value;
} SampleEntry;

VH_EXPORT float transfers_total(VhSpan values, VhSpan weights) {
    float total = 0.0f;
    const int32_t *items = (const int32_t *)values.data;
    for (uint64_t index = 0; index < values.size; ++index) {
        total += (float)items[index];
    }
    const WeightEntry *entries = (const WeightEntry *)weights.data;
    for (uint64_t index = 0; index < weights.size; ++index) {
        total += entries[index].value * (float)(index + 1);
    }
    return total;
}

VH_EXPORT int32_t transfers_samples(VhSpan samples) {
    int32_t total = 0;
    const Sample *items = (const Sample *)samples.data;
    for (uint64_t index = 0; index < samples.size; ++index) {
        total += items[index].count * (int32_t)(index + 1);
    }
    return total;
}

VH_EXPORT int32_t transfers_name(int32_t id, VhList *out, const VhAllocator *allocator) {
    if (id < 0) {
        return 7;
    }
    char text[32];
    int length = snprintf(text, sizeof text, "item-%d", (int)id);
    return vh_list_assign(out, text, (uint64_t)length, 1, 1, allocator);
}

VH_EXPORT int32_t transfers_range(int32_t count, VhList *out, const VhAllocator *allocator) {
    int32_t items[64];
    if (count < 0 || count > 64) {
        return 8;
    }
    for (int32_t index = 0; index < count; ++index) {
        items[index] = index * 3;
    }
    return vh_list_assign(out, items, (uint64_t)count, sizeof(int32_t), 4, allocator);
}

/* Keys repeat, index % 3, so the caller keeps each key's last entry. */
VH_EXPORT int32_t transfers_table(int32_t count, VhList *out, const VhAllocator *allocator) {
    SampleEntry entries[64];
    if (count < 0 || count > 64) {
        return 8;
    }
    for (int32_t index = 0; index < count; ++index) {
        entries[index].key = index % 3;
        entries[index].value.weight = (float)index;
        entries[index].value.count = index + 1;
    }
    return vh_list_assign(out, entries, (uint64_t)count, sizeof(SampleEntry), 4, allocator);
}
