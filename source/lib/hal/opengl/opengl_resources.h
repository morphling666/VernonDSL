#ifndef VERNON_HAL_OPENGL_RESOURCES_H
#define VERNON_HAL_OPENGL_RESOURCES_H

#include "VernonHal.h"
#include "VernonRHI.h"

#include <cstdint>
#include <memory>

namespace vernon::rhi::opengl {
struct Buffer;
}

namespace vernon::hal {

class DeviceBuffers;
class Resources;

// What a native view reference of an OpenGL device names (hal/native_images.h): the texture passes bind and its
// target. A texture view's own texture starts at the view's first mip and layer; an identity view binds its image's
// texture.
struct OpenGLViewTexture {
    uint32_t name{};
    uint32_t target{};
};

uint32_t openGLTarget(VernonTextureDimension dimension);

// The resources of an OpenGL or OpenGL ES HAL device: buffers, textures, samplers and surfaces outside the RHI's
// registry, named by the native references of hal/native_buffers.h and hal/native_images.h. Null for a device of
// another backend; the RHI device must outlive them.
std::unique_ptr<Resources> createOpenGLResources(VernonRhiDevice device);

// The buffers an OpenGL HAL device allocates for its schedules, named the same way. Null for a device of another
// backend; the RHI device must outlive them.
std::unique_ptr<DeviceBuffers> createOpenGLDeviceBuffers(VernonRhiDevice device);

// The buffer of `serial`; null once it is destroyed. The caller keeps the buffer alive while it uses the result.
rhi::opengl::Buffer *openGLBuffer(uint64_t serial);

} // namespace vernon::hal

#endif
