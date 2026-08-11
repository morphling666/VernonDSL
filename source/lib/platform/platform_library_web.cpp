#include "platform_library.h"

#include <utility>

namespace vernon::platform {

PlatformLibrary::~PlatformLibrary() = default;

PlatformLibrary::PlatformLibrary(PlatformLibrary &&other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}

PlatformLibrary &PlatformLibrary::operator=(PlatformLibrary &&other) noexcept {
    if (this != &other)
        handle_ = std::exchange(other.handle_, nullptr);
    return *this;
}

bool PlatformLibrary::open(const char *, std::string &error) {
    handle_ = nullptr;
    error = "dynamic libraries are unavailable in the web Runtime profile";
    return false;
}

void *PlatformLibrary::symbol(const char *) const { return nullptr; }

void PlatformLibrary::close() { handle_ = nullptr; }

} // namespace vernon::platform
