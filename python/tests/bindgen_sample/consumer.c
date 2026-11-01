/* A C program linking tally_user's AOT object with tally's generated C API, as a packaged build would. */
#include <stdio.h>

#include "tally_api.h"
#include "tally_user.h"

int main(void) {
    int64_t total = 0;
    int32_t status = tally_user_run(4, &total);
    printf("run %d %lld\n", status, (long long)total);

    float peak = 0.0f;
    status = tally_user_peak(5, &peak);
    printf("peak %d %.1f\n", status, peak);

    int32_t ignored = 0;
    status = tally_user_stale(&ignored);
    char message[128];
    tally_v1_last_error(message, sizeof message);
    printf("stale %d %s\n", status, message);

    float spans = 0.0f;
    status = tally_user_spans(4, &spans);
    printf("spans %d %.1f\n", status, spans);

    int64_t owned = 0;
    status = tally_user_owned(7, &owned);
    printf("owned %d %lld\n", status, (long long)owned);

    status = tally_user_nameless(&ignored);
    tally_v1_last_error(message, sizeof message);
    printf("nameless %d %s\n", status, message);

    int32_t ranged = 0;
    status = tally_user_ranged(&ranged);
    printf("ranged %d %d\n", status, ranged);

    float stored = 0.0f;
    status = tally_user_stored(7, &stored);
    printf("stored %d %.1f\n", status, stored);

    int32_t alive = -1;
    status = tally_user_alive(&alive);
    printf("alive %d %d\n", status, alive);
    printf("schema %s\n", tally_v1_schema());
    return 0;
}
