#include "platform_library.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <utility>

namespace vernon::runtime {
namespace {

#if defined(_WIN32)
std::string windowsError(DWORD code) {
  char *message = nullptr;
  const DWORD size = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, code, 0, reinterpret_cast<char *>(&message), 0, nullptr);
  std::string result =
      size && message ? std::string(message, size) : "unknown Windows error";
  if (message)
    LocalFree(message);
  while (!result.empty() && (result.back() == '\r' || result.back() == '\n'))
    result.pop_back();
  return result;
}
#endif

} // namespace

PlatformLibrary::~PlatformLibrary() { close(); }

PlatformLibrary::PlatformLibrary(PlatformLibrary &&other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)) {}

PlatformLibrary &PlatformLibrary::operator=(PlatformLibrary &&other) noexcept {
  if (this != &other) {
    close();
    handle_ = std::exchange(other.handle_, nullptr);
  }
  return *this;
}

bool PlatformLibrary::open(const char *path, std::string &error) {
  close();
  if (!path || !*path) {
    error = "dynamic library path is empty";
    return false;
  }
#if defined(_WIN32)
  handle_ = LoadLibraryA(path);
  if (!handle_)
    error = "cannot load dynamic library '" + std::string(path) +
            "': " + windowsError(GetLastError());
#else
  dlerror();
  handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!handle_) {
    const char *message = dlerror();
    error = "cannot load dynamic library '" + std::string(path) +
            "': " + (message ? message : "unknown loader error");
  }
#endif
  return handle_ != nullptr;
}

void *PlatformLibrary::symbol(const char *name) const {
  if (!handle_ || !name)
    return nullptr;
#if defined(_WIN32)
  return reinterpret_cast<void *>(
      GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
  return dlsym(handle_, name);
#endif
}

void PlatformLibrary::close() {
  if (!handle_)
    return;
#if defined(_WIN32)
  FreeLibrary(static_cast<HMODULE>(handle_));
#else
  dlclose(handle_);
#endif
  handle_ = nullptr;
}

} // namespace vernon::runtime
