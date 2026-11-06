#include "hal/vulkan/vulkan_driver.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <vulkan/vulkan_win32.h>
#endif

namespace vernon::hal::vulkan {

bool createPlatformSurface(VkInstance instance, void *window, VkSurfaceKHR &surface, std::string &error) {
#if defined(_WIN32)
    const auto create = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(driver().createPlatformSurface);
    if (!create || !window) {
        error = "Vulkan surfaces need VK_KHR_win32_surface and a window";
        return false;
    }
    VkWin32SurfaceCreateInfoKHR info{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    info.hinstance = GetModuleHandleW(nullptr);
    info.hwnd = static_cast<HWND>(window);
    const VkResult result = create(instance, &info, nullptr, &surface);
    if (result != VK_SUCCESS) {
        error = "vkCreateWin32SurfaceKHR failed with Vulkan error " + std::to_string(static_cast<int>(result));
        return false;
    }
    return true;
#else
    (void)instance;
    (void)window;
    (void)surface;
    error = "Vulkan surfaces are not supported on this platform";
    return false;
#endif
}

} // namespace vernon::hal::vulkan
