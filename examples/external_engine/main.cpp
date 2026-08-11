#include "engine_demo.h"

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>

EM_JS(int, hasBrowserEnvironment, (), { return 'window' in globalThis && 'document' in globalThis; });
#endif

#include <chrono>
#include <iostream>
#include <memory>

namespace {

struct Application {
    std::unique_ptr<ExternalEngineDemo> demo;
    std::chrono::steady_clock::time_point start;
    bool failed{};
};

bool renderApplicationFrame(Application &application) {
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - application.start).count();
    if (!application.demo->renderFrame(elapsed)) {
        application.failed = true;
        return false;
    }
    return !application.demo->shouldClose();
}

#if defined(__EMSCRIPTEN__)
void renderWebFrame(void *userData) {
    auto *application = static_cast<Application *>(userData);
    if (!renderApplicationFrame(*application)) {
        emscripten_cancel_main_loop();
        application->demo->shutdown();
        delete application;
    }
}
#endif

} // namespace

int main() {
    auto application = std::make_unique<Application>();
    application->demo = createExternalEngineDemo();
#if defined(__EMSCRIPTEN__)
    const bool headless = !hasBrowserEnvironment();
#else
    constexpr bool headless = false;
#endif
    if (!application->demo || !application->demo->initialize(headless)) {
        std::cerr << "failed to initialize the Vernon external-engine demo\n";
        return 1;
    }
    application->start = std::chrono::steady_clock::now();

#if defined(__EMSCRIPTEN__)
    if (!headless) {
        emscripten_set_main_loop_arg(&renderWebFrame, application.release(), 0, 1);
    }
#endif

    do {
        if (!renderApplicationFrame(*application))
            break;
    } while (!headless);
    application->demo->shutdown();
    return application->failed ? 1 : 0;
}
