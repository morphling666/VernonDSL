#ifndef VERNON_HAL_OPENGL_RESOURCES_H
#define VERNON_HAL_OPENGL_RESOURCES_H

#include "VernonHal.h"

#include <cstdint>
#include <memory>

namespace vernon::hal::opengl {
struct Buffer;
struct DeviceState;
} // namespace vernon::hal::opengl

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

// The resources of an OpenGL or OpenGL ES HAL device over `state`: buffers, textures, samplers and surfaces, named by
// the native references of hal/native_buffers.h and hal/native_images.h.
std::unique_ptr<Resources> createOpenGLResources(std::shared_ptr<opengl::DeviceState> state);

// The buffers an OpenGL HAL device allocates for its schedules, named the same way.
std::unique_ptr<DeviceBuffers> createOpenGLDeviceBuffers(std::shared_ptr<opengl::DeviceState> state);

// The buffer of `serial`; null once it is destroyed. The caller keeps the buffer alive while it uses the result.
opengl::Buffer *openGLBuffer(uint64_t serial);

} // namespace vernon::hal

#endif
