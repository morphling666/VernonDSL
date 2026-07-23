#include <vernon-c/Runtime.h>

int main(void) {
    VernonRuntimeCapabilities capabilities = vernonRuntimeGetCapabilities(VERNON_RUNTIME_CPU);
    return capabilities.available ? 0 : 1;
}
