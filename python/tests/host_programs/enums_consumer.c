/* Links the AOT object of enums.py's free functions and reads its enums through the header and the reflection
 * table. Prints one `name=value` line per check for the test to compare. */
#include "enums.h"

#include <stdio.h>

float demo_mode_speed(DemoMode mode) { return (float)mode + 0.5f; }

DemoMode demo_next_mode(DemoMode mode) { return mode == DemoMode_WALK ? DemoMode_RUN : DemoMode_IDLE; }

int main(void) {
    const VhModule *module = &demo_module;
    const VhType *mode = vh_find_type(module, "Mode");
    printf("mode_kind=%u size=%llu count=%u doc=%s cpp=%s\n", mode->kind, (unsigned long long)mode->size,
           mode->enumerator_count, mode->doc, vh_find_meta(mode->meta, mode->meta_count, "cpp"));
    printf("names=%s,%s,%s\n", vh_enum_name(mode, 1), vh_enum_name(mode, -2),
           vh_enum_name(mode, 7) == NULL ? "null" : "set");

    DemoMode faster = DemoMode_IDLE;
    demo_faster(DemoMode_WALK, &faster);
    int64_t code = 0;
    demo_code(DemoMode_BACK, &code);
    bool from_code = false;
    demo_from_code(4, &from_code);
    printf("faster=%d code=%lld from_code=%d\n", faster, (long long)code, from_code);

    char json[128];
    DemoBody body;
    demo_walk(2.0f, &body);
    vh_write_json(vh_find_type(module, "Body"), &body, json, sizeof json);
    printf("body_json=%s\n", json);
    body.mode = 7;
    vh_write_json(vh_find_type(module, "Body"), &body, json, sizeof json);
    printf("unknown_json=%s\n", json);
    float speed = 0.0f;
    body.mode = DemoMode_BACK;
    demo_body_speed(&body, &speed);
    printf("body_speed=%g\n", speed);
    return 0;
}
