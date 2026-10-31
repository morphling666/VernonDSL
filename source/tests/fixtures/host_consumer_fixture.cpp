// A C++ application consuming the Host module asset python/tests/host_programs/systems_asset.py ("demo/systems"). It
// defines the module's externs and drives the table VernonRuntime finds under the asset id, generically, by reflection
// alone: the cooked object it links registers itself, and a table Python mode builds can be registered over it. Both
// tables must produce the same report.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "VernonRuntime.h"
#include "host_systems.h"

namespace nb = nanobind;

namespace {

DemoTransform g_transform = {{0.0f, 0.0f, 0.0f}, 1.0f};

// A store of any reflected Struct, keyed by layout and entity, as an engine keeps components it never saw.
struct Row {
    uint64_t layout_hash;
    uint64_t entity;
    std::vector<unsigned char> bytes;
};
std::vector<Row> g_rows;

Row *find_row(const VhType *type, uint64_t entity) {
    for (Row &row : g_rows) {
        if (row.layout_hash == type->layout_hash && row.entity == entity) {
            return &row;
        }
    }
    return nullptr;
}

constexpr int32_t kMissingComponent = 7;

} // namespace

extern "C" {

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

int32_t demo_read(DemoEntity entity, const VhType *type, void *out) {
    const Row *row = find_row(type, entity.bits);
    if (row == nullptr) {
        return kMissingComponent;
    }
    std::memcpy(out, row->bytes.data(), static_cast<size_t>(type->size));
    return 0;
}

int32_t demo_write(DemoEntity entity, const VhType *type, const void *value) {
    Row *row = find_row(type, entity.bits);
    if (row == nullptr) {
        row = &g_rows.emplace_back(Row{type->layout_hash, entity.bits, {}});
    }
    const auto *bytes = static_cast<const unsigned char *>(value);
    row->bytes.assign(bytes, bytes + type->size);
    return 0;
}

} // extern "C"

namespace {

class Report {
public:
    void line(const char *format, ...) {
        char buffer[1024];
        va_list arguments;
        va_start(arguments, format);
        std::vsnprintf(buffer, sizeof buffer, format, arguments);
        va_end(arguments);
        lines_.emplace_back(buffer);
    }

    std::vector<std::string> take() { return std::move(lines_); }

private:
    std::vector<std::string> lines_;
};

const char *text(const char *value) { return value == nullptr ? "-" : value; }

std::string json(const VhType *type, const void *value) {
    std::string out(vh_write_json(type, value, nullptr, 0), '\0');
    vh_write_json(type, value, out.data(), out.size() + 1);
    return out;
}

// Counts what Host containers take from and give back to the module's allocator.
struct CountingAllocator {
    int allocations = 0;
    int releases = 0;
    long long live_bytes = 0;

    VhAllocator allocator() {
        VhAllocator result = {};
        result.struct_size = sizeof(VhAllocator);
        result.user = this;
        result.allocate = [](void *user, uint64_t size, uint64_t) -> void * {
            auto *self = static_cast<CountingAllocator *>(user);
            ++self->allocations;
            self->live_bytes += static_cast<long long>(size);
            return std::malloc(static_cast<size_t>(size));
        };
        result.release = [](void *user, void *pointer, uint64_t size, uint64_t) {
            auto *self = static_cast<CountingAllocator *>(user);
            ++self->releases;
            self->live_bytes -= static_cast<long long>(size);
            std::free(pointer);
        };
        return result;
    }
};

// Zeroed, max-aligned storage of one instance, released through the type's fini when it has one.
class Instance {
public:
    explicit Instance(const VhType *type)
        : type_(type),
          storage_((static_cast<size_t>(type->size) + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t)) {}
    Instance(const Instance &) = delete;
    Instance &operator=(const Instance &) = delete;
    ~Instance() {
        if (type_->fini != nullptr) {
            type_->fini(data());
        }
    }

    void *data() { return storage_.data(); }

    template <typename T> T &at(const char *field) {
        return *reinterpret_cast<T *>(static_cast<char *>(data()) + vh_find_field(type_, field)->offset);
    }

private:
    const VhType *type_;
    std::vector<std::max_align_t> storage_;
};

void describe(Report &report, const VhModule *module) {
    report.line("version=%u sizes=%d", module->version,
                module->struct_size == sizeof(VhModule) && module->type_count > 0 &&
                    module->types[0]->struct_size == sizeof(VhType));
    report.line("module=%s types=%u functions=%u", module->name, module->type_count, module->function_count);
    for (uint32_t index = 0; index < module->type_count; ++index) {
        const VhType *type = module->types[index];
        report.line("type=%s kind=%u size=%llu align=%u fields=%u methods=%u init=%d fini=%d", type->name, type->kind,
                    static_cast<unsigned long long>(type->size), type->alignment, type->field_count, type->method_count,
                    type->init != nullptr, type->fini != nullptr);
        for (uint32_t field_index = 0; field_index < type->field_count; ++field_index) {
            const VhField *field = vh_field_at(type, field_index);
            report.line("  field=%s type=%s offset=%llu doc=%s max=%s", field->name, field->type->name,
                        static_cast<unsigned long long>(field->offset), text(field->doc),
                        text(vh_find_meta(field->meta, field->meta_count, "max")));
        }
        for (uint32_t method_index = 0; method_index < type->method_count; ++method_index) {
            const VhFunction *method = vh_method_at(type, method_index);
            report.line("  method=%s symbol=%s effect=%u params=%u callable=%d", method->name, text(method->symbol),
                        method->effect, method->param_count, method->address != nullptr);
            for (uint32_t call_index = 0; call_index < method->call_count; ++call_index) {
                const VhCall *call = vh_call_at(method, call_index);
                report.line("    call=%s effect=%u type=%s", call->symbol, call->effect,
                            call->type == nullptr ? "-" : call->type->name);
            }
        }
    }
    for (uint32_t index = 0; index < module->function_count; ++index) {
        const VhFunction *function = vh_function_at(module, index);
        report.line("function=%s symbol=%s effect=%u", function->name, text(function->symbol), function->effect);
    }
}

void run(Report &report, const VhModule *module) {
    const VhType *mover_type = vh_find_type(module, "Mover");
    Instance mover(mover_type);
    const int32_t init_status = mover_type->init(mover.data());
    report.line("mover_init=%d speed_default=%g", init_status, mover.at<float>("speed"));
    mover.at<float>("speed") = 4.0f;
    using Update = int32_t (*)(void *, double, int32_t *);
    const auto update = reinterpret_cast<Update>(vh_find_method(mover_type, "update")->address);
    for (int step = 0; step < 3; ++step) {
        int32_t updates = 0;
        const int32_t status = update(mover.data(), 0.25, &updates);
        report.line("update status=%d updates=%d height=%g", status, updates, g_transform.position[1]);
    }
    report.line("mover_json=%s", json(mover_type, mover.data()).c_str());

    using HeightAfter = int32_t (*)(void *, double, float *);
    float height = 0.0f;
    const int32_t height_status =
        reinterpret_cast<HeightAfter>(vh_find_method(mover_type, "height_after")->address)(mover.data(), 2.0, &height);
    report.line("height_after status=%d value=%g", height_status, height);

    Instance copy(mover_type);
    mover_type->init(copy.data());
    vh_migrate(mover_type, mover.data(), mover_type, copy.data());
    report.line("migrated_json=%s", json(mover_type, copy.data()).c_str());

    // Generic externs see the descriptor of the table being driven; the store keys rows by layout, so the Python
    // and AOT descriptors of one Struct find the same row.
    const VhType *health_type = vh_find_type(module, "Health");
    const DemoEntity target = {3};
    const DemoHealth start = {4.0f, 5.0f};
    demo_write(target, health_type, &start);
    demo_write(target, vh_find_type(module, "Transform"), &g_transform);
    const VhType *regen_type = vh_find_type(module, "Regen");
    Instance regen(regen_type);
    regen_type->init(regen.data());
    using RegenUpdate = int32_t (*)(void *, DemoEntity, double, float *);
    const auto regen_update = reinterpret_cast<RegenUpdate>(vh_find_method(regen_type, "update")->address);
    float healed = 0.0f;
    const int32_t regen_status = regen_update(regen.data(), target, 0.5, &healed);
    DemoHealth stored = {};
    demo_read(target, health_type, &stored);
    report.line("regen status=%d healed=%g stored=%s", regen_status, healed, json(health_type, &stored).c_str());
    float missing = 0.0f;
    report.line("regen_missing status=%d", regen_update(regen.data(), DemoEntity{99}, 0.5, &missing));

    // Poison's str and list take memory from the allocator installed here, and give all of it back through the
    // type's fini when the instances go out of scope.
    CountingAllocator counting;
    const VhAllocator allocator = counting.allocator();
    vh_set_allocator(module, &allocator);
    {
        // Poison comes from the asset's second module, abilities.py, and holds a Timer imported from systems.py;
        // its private helper `scaled` doubles, while the exported systems `scaled` below does not.
        const VhType *poison_type = vh_find_type(module, "Poison");
        Instance poison(poison_type);
        poison_type->init(poison.data());
        report.line("poison_timer_type=%s same_timer=%d", vh_find_field(poison_type, "timer")->type->name,
                    vh_find_field(poison_type, "timer")->type == vh_find_type(module, "Timer"));
        const VhType *drains_type = vh_find_field(poison_type, "drains")->type;
        report.line("drains kind=%u element=%s fini=%d name_kind=%u", drains_type->kind, drains_type->element->name,
                    drains_type->fini != nullptr, vh_find_field(poison_type, "name")->type->kind);
        const VhType *hits_type = vh_find_field(poison_type, "hits")->type;
        report.line("hits kind=%u key=%s element=%s size=%llu", hits_type->kind, hits_type->key->name,
                    hits_type->element->name, static_cast<unsigned long long>(hits_type->size));
        using PoisonUpdate = int32_t (*)(void *, DemoEntity, double, float *);
        const auto poison_update = reinterpret_cast<PoisonUpdate>(vh_find_method(poison_type, "update")->address);
        for (int step = 0; step < 2; ++step) {
            float drained = 0.0f;
            const int32_t status = poison_update(poison.data(), target, 0.25, &drained);
            demo_read(target, health_type, &stored);
            report.line("poison status=%d drained=%g current=%g ticks=%d", status, drained, stored.current,
                        poison.at<int32_t>("ticks"));
        }
        // The generated header types the container fields, so C++ reads what Host code wrote and edits what Host code
        // reads next, through the module's allocator.
        auto *typed = static_cast<DemoPoison *>(poison.data());
        const vh::Allocator containers(module);
        const float *hit = typed->hits.find(target);
        float drained_sum = 0.0f;
        for (const float drained : typed->drains) {
            drained_sum += drained;
        }
        report.line("typed name=%s drains=%g hit=%g missing=%d", std::string(typed->name.view()).c_str(), drained_sum,
                    hit == nullptr ? -1.0f : *hit, typed->hits.find(DemoEntity{99}) == nullptr);
        const bool edited = typed->drains.push_back(containers, 0.5f) && typed->levels.assign(containers, "venom", 9);
        report.line("typed_edit=%d", edited);
        using Total = int32_t (*)(void *, float *);
        float total = 0.0f;
        const int32_t total_status =
            reinterpret_cast<Total>(vh_find_method(poison_type, "total_drained")->address)(poison.data(), &total);
        report.line("total_drained status=%d value=%g", total_status, total);
        using Strengthen = int32_t (*)(void *, int32_t *);
        const auto strengthen = reinterpret_cast<Strengthen>(vh_find_method(poison_type, "strengthen")->address);
        for (int step = 0; step < 2; ++step) {
            int32_t level = 0;
            const int32_t status = strengthen(poison.data(), &level);
            report.line("strengthen status=%d level=%d", status, level);
        }
        std::string levels;
        for (auto [name, level] : typed->levels) {
            levels += std::string(name) + "=" + std::to_string(level) + " ";
        }
        report.line("typed levels=%s", levels.c_str());
        report.line("poison_json=%s", json(poison_type, poison.data()).c_str());
        Instance moved(poison_type);
        poison_type->init(moved.data());
        vh_migrate(poison_type, poison.data(), poison_type, moved.data());
        report.line("poison_moved=%s", json(poison_type, moved.data()).c_str());
        report.line("poison_left=%s", json(poison_type, poison.data()).c_str());
    }
    {
        // Trail's storages are typed records over the allocator's memory: C++ indexes what Host code wrote, sizes a
        // storage anew, and hands the elements on as a VernonTensorView without copying them.
        const VhType *trail_type = vh_find_type(module, "Trail");
        Instance trail(trail_type);
        const int32_t init_status = trail_type->init(trail.data());
        const VhType *cells_type = vh_find_field(trail_type, "cells")->type;
        const VhType *marks_type = vh_find_field(trail_type, "marks")->type;
        report.line("trail_init=%d cells kind=%u element=%s rank=%u shape=%llu,%llu fini=%d marks=%llu", init_status,
                    cells_type->kind, cells_type->element->name, cells_type->rank,
                    static_cast<unsigned long long>(cells_type->shape[0]),
                    static_cast<unsigned long long>(cells_type->shape[1]), cells_type->fini != nullptr,
                    static_cast<unsigned long long>(marks_type->shape[0]));
        using Spread = int32_t (*)(void *, int32_t, int32_t, int32_t *);
        int32_t rows = 0;
        const int32_t spread_status =
            reinterpret_cast<Spread>(vh_find_method(trail_type, "spread")->address)(trail.data(), 2, 3, &rows);
        auto *typed = static_cast<DemoTrail *>(trail.data());
        report.line("spread status=%d rows=%d cell=%g marks=%d,%d missing=%d", spread_status, rows, typed->cells(1, 2),
                    typed->marks(0), typed->marks(1), typed->cells.get({2, 0}) == nullptr);
        typed->cells(0, 0) = 100.0f;
        using Total = int32_t (*)(void *, float *);
        const auto trail_total = reinterpret_cast<Total>(vh_find_method(trail_type, "total")->address);
        float total = 0.0f;
        const int32_t total_status = trail_total(trail.data(), &total);
        report.line("trail_total status=%d value=%g", total_status, total);

        int64_t strides[2] = {};
        const VernonTensorView view = typed->cells.view(strides);
        report.line("view rank=%u shape=%lld,%lld strides=%lld,%lld bytes=%llu shared=%d", view.rank,
                    static_cast<long long>(view.shape[0]), static_cast<long long>(view.shape[1]),
                    static_cast<long long>(view.byte_strides[0]), static_cast<long long>(view.byte_strides[1]),
                    static_cast<unsigned long long>(view.byte_size), view.host_data == typed->cells.data());

        const vh::Allocator containers(module);
        const bool sized = typed->cells.allocate(containers, {1, 2});
        typed->cells.fill(2.5f);
        trail_total(trail.data(), &total);
        report.line("trail_resized=%d count=%llu c_count=%llu total=%g", sized,
                    static_cast<unsigned long long>(typed->cells.size()),
                    static_cast<unsigned long long>(vh_storage_count(&typed->cells, 2)), total);
        report.line("trail_json=%s", json(trail_type, trail.data()).c_str());
        Instance moved(trail_type);
        trail_type->init(moved.data());
        vh_migrate(trail_type, trail.data(), trail_type, moved.data());
        report.line("trail_moved=%s", json(trail_type, moved.data()).c_str());
        report.line("trail_left=%s", json(trail_type, trail.data()).c_str());
    }
    // Only the balance is compared: compiled code builds str expressions in temporaries Python mode does not need.
    report.line("allocator used=%d balanced=%d", counting.allocations > 0,
                counting.allocations == counting.releases && counting.live_bytes == 0);
    vh_set_allocator(module, nullptr);

    using Scaled = int32_t (*)(float, int32_t, float *);
    float scaled = 0.0f;
    const int32_t scaled_status =
        reinterpret_cast<Scaled>(vh_find_function(module, "scaled")->address)(1.5f, 3, &scaled);
    report.line("scaled status=%d value=%g", scaled_status, scaled);
}

VernonStringView view(const std::string &text) { return {text.data(), text.size()}; }

uintptr_t find(const std::string &id) { return reinterpret_cast<uintptr_t>(vernonRuntimeFindHostModule(view(id))); }

int register_table(const std::string &id, uintptr_t address) {
    const VernonStaticHostModule record{view(id), reinterpret_cast<const VhModule *>(address)};
    return vernonRuntimeRegisterStaticHostModule(&record);
}

int unregister_table(const std::string &id, uintptr_t address) {
    const VernonStaticHostModule record{view(id), reinterpret_cast<const VhModule *>(address)};
    return vernonRuntimeUnregisterStaticHostModule(&record);
}

std::vector<std::string> drive(const std::string &id) {
    const VhModule *module = vernonRuntimeFindHostModule(view(id));
    if (module == nullptr) {
        throw std::runtime_error("no Host module is registered as " + id);
    }
    Report report;
    describe(report, module);
    run(report, module);
    return report.take();
}

void reset() {
    g_transform = {{0.0f, 0.0f, 0.0f}, 1.0f};
    g_rows.clear();
}

} // namespace

NB_MODULE(vernon_host_consumer_fixture, module) {
    module.def("drive", &drive,
               "Drives the VhModule table registered as `id` by reflection; returns the report lines.");
    module.def("find", &find, "The address of the table registered as `id`, or zero.");
    module.def("register_table", &register_table, "Registers the table at `address` as `id`; returns the status.");
    module.def("unregister_table", &unregister_table, "Withdraws a registration; returns the status.");
    module.def("reset", &reset, "Clears the extern state, the transform and the component store.");
    module.def("aot_table", [] { return reinterpret_cast<uintptr_t>(&demo_module); });
    module.def("extern_symbols", [] {
        nb::dict symbols;
        symbols["demo_get_transform"] = reinterpret_cast<uintptr_t>(&demo_get_transform);
        symbols["demo_set_transform"] = reinterpret_cast<uintptr_t>(&demo_set_transform);
        symbols["demo_read"] = reinterpret_cast<uintptr_t>(&demo_read);
        symbols["demo_write"] = reinterpret_cast<uintptr_t>(&demo_write);
        return symbols;
    });
}
