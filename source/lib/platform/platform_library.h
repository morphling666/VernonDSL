#ifndef VERNON_PLATFORM_LIBRARY_H
#define VERNON_PLATFORM_LIBRARY_H

#include <string>

namespace vernon::platform {

class PlatformLibrary {
public:
    PlatformLibrary() = default;
    ~PlatformLibrary();

    PlatformLibrary(const PlatformLibrary &) = delete;
    PlatformLibrary &operator=(const PlatformLibrary &) = delete;
    PlatformLibrary(PlatformLibrary &&other) noexcept;
    PlatformLibrary &operator=(PlatformLibrary &&other) noexcept;

    bool open(const char *path, std::string &error);
    void *symbol(const char *name) const;
    explicit operator bool() const { return handle_ != nullptr; }

private:
    void close();

    void *handle_{};
};

} // namespace vernon::platform

namespace vernon::runtime {
using PlatformLibrary = platform::PlatformLibrary;
}

#endif
