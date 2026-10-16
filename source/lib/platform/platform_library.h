#ifndef VERNON_PLATFORM_LIBRARY_H
#define VERNON_PLATFORM_LIBRARY_H

#include <string>
#include <string_view>

namespace vernon::platform {

#if defined(_WIN32)
// Empty when `value` is not valid UTF-8.
std::wstring utf8ToWide(std::string_view value);
#endif

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
