#ifndef VERNON_HAL_OPENGL_STATE_H
#define VERNON_HAL_OPENGL_STATE_H

#include "VernonOpenGLContext.h"
#include "hal/opengl/opengl_driver.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vernon::hal::opengl {

struct Buffer {
    Uint name{};
    bool imported{};
};

struct Image {
    Uint name{};
    bool imported{};
};

struct Sampler {
    Uint name{};
    bool imported{};
};

constexpr size_t kMaxColorAttachments = 8;

// The texture names a rendering scope attaches, by color location; `stencil` attaches `depth` at the depth-stencil
// point.
struct FramebufferAttachments {
    std::array<Uint, kMaxColorAttachments> colors{};
    Uint depth{};
    bool stencil{};

    bool operator==(const FramebufferAttachments &other) const {
        return colors == other.colors && depth == other.depth && stencil == other.stencil;
    }
    bool attaches(Uint texture) const {
        return depth == texture || std::find(colors.begin(), colors.end(), texture) != colors.end();
    }
};

struct AttachmentFramebuffer {
    FramebufferAttachments attachments;
    Uint name{};
};

enum class DepthStencilAspect { Depth, Stencil };

// The OpenGL context a HAL device adopts, which its device, resources and runtime context share.
struct DeviceState {
    DeviceState() = default;
    // Removes the debug callback of a device destroyed without shutdown.
    ~DeviceState();
    DeviceState(const DeviceState &) = delete;
    DeviceState &operator=(const DeviceState &) = delete;

    // `flags` are the device's VernonHalDeviceFlagBits; validation implies labels.
    bool initialize(const VernonOpenGLContextCallbacks &callbacks, bool embeddedProfile, uint32_t flags,
                    std::string &error);
    // Removes the debug callback and deletes pooled queries; the context must be current.
    void shutdown();
    bool makeCurrent() const noexcept;
    bool supportsCompute() const;
    bool apiVersionAtLeast(uint32_t major, uint32_t minor) const;
    bool hasExtension(std::string_view name) const;

    bool createBuffer(Buffer &buffer, size_t size, std::string &error);
    void importBuffer(Buffer &buffer, Uint name);
    void destroyBuffer(Buffer &buffer);
    bool uploadBuffer(const Buffer &buffer, size_t offset, const void *source, size_t size, std::string &error);
    bool downloadBuffer(const Buffer &buffer, size_t offset, void *destination, size_t size, std::string &error);
    bool copyBuffer(const Buffer &source, size_t sourceOffset, const Buffer &destination, size_t destinationOffset,
                    size_t size, std::string &error);

    bool createImage2D(Image &image, Int internalFormat, Size width, Size height, Enum externalFormat, Enum type,
                       std::string &error);
    void importImage(Image &image, Uint name);
    void destroyImage(Image &image);
    bool uploadImage2D(const Image &image, Size width, Size height, Enum externalFormat, Enum type, const void *source);
    // `layer` selects the layer of 2D-array and cube-array targets; other targets ignore it.
    bool downloadImage2D(const Image &image, Enum target, Int mipLevel, Int layer, Int x, Int y, Size width,
                         Size height, Enum externalFormat, Enum type, void *destination, std::string &error);
    // The format and type glReadPixels reads a color image in besides the pair its component class guarantees; OpenGL
    // ES reads in no other pair. Zeros when the image is not readable as a color attachment.
    std::pair<Enum, Enum> implementationColorReadPair(const Image &image, Enum target, Int mipLevel, Int layer);
    bool downloadImage3D(const Image &image, Int mipLevel, Int x, Int y, Int z, Size width, Size height, Size depth,
                         Enum externalFormat, Enum type, size_t layerSize, void *destination, std::string &error);
    // Writes one 32-bit word per texel of the region into `destination` from offset 0, row by row from (x, y): the
    // float bits of depth, or the stencil value. OpenGL ES reads neither aspect with glReadPixels, so this is a compute
    // pass and needs OpenGL ES 3.1. `target` is the texture's own target; `layer` selects the layer, or the layer-face
    // of cube targets.
    bool copyDepthStencilToBuffer(const Image &image, Enum target, DepthStencilAspect aspect, Int mipLevel, Int layer,
                                  Int x, Int y, Size width, Size height, const Buffer &destination, std::string &error);
    void destroyDepthStencilCopyObjects();

    bool createSampler(Sampler &sampler, Int wrapU, Int wrapV, Int wrapW, Int minFilter, Int magFilter,
                       std::string &error);
    void importSampler(Sampler &sampler, Uint name);
    void destroySampler(Sampler &sampler);

    Uint compileShader(Enum kind, const std::string &source, std::string &error);
    Uint linkProgram(const std::vector<Uint> &shaders, std::string &error);
    void destroyProgram(Uint program);
    bool createVertexArray(Uint &vertexArray, std::string &error);
    void destroyVertexArray(Uint vertexArray);
    bool createFramebuffer(Uint &framebuffer, std::string &error);
    void destroyFramebuffer(Uint framebuffer);
    void destroyAttachmentFramebuffers();

    // No-ops without debug labels. A push beyond the context's group stack limit is skipped, and so is its pop, so
    // that labels never raise a GL error. Labels are cut to the context's message length.
    void pushDebugGroup(std::string_view label);
    void popDebugGroup();
    void popDebugGroups();
    void insertDebugMarker(std::string_view label);

    // Reads and clears the context's error flag. On a threaded driver every read waits until the driver thread has
    // run the calls queued before it, so only validation mode and synchronous operations read it per operation.
    Enum readError();
    // Outside validation mode, a synchronous operation checks its own errors between these calls. Earlier errors stay
    // queued for takeErrors; endSynchronous returns the operation's first error and drops its errors from the queue,
    // since the operation reports them itself.
    void beginSynchronous();
    Enum endSynchronous();
    // Takes the errors raised on the context since the last call. Call when a submission is checked after its fence
    // signalled.
    std::optional<std::string> takeErrors();

    // Timer queries are pooled: released queries are reused instead of deleted. releaseQueries pools all of them or,
    // when it throws, none.
    Uint acquireQuery();
    void releaseQueries(const std::vector<Uint> &queries);

    // Every OpenGL call on the context holds it, with the context current.
    std::mutex mutex;
    // initialize succeeded, so the state has objects on the context to tear down.
    bool initialized{};
    VernonOpenGLContextCallbacks callbacks{};
    bool embeddedProfile{};
    // VERNON_HAL_DEVICE_VALIDATION: error scopes around every operation.
    bool validation{};
    // VERNON_HAL_DEVICE_LABELS: debug groups and markers reach the context.
    bool labels{};
    // GL_GPU_DISJOINT_EXT can be read; without it, timers cannot tell that a disjoint event invalidated them.
    bool disjointQueryable{};
    std::vector<std::string> extensions;
    // A complete framebuffer per attachment set that a rendering scope used. Deleting a texture destroys the
    // framebuffers attaching it: GL may reuse its name, and a framebuffer that is not bound keeps it alive.
    std::vector<AttachmentFramebuffer> attachmentFramebuffers;
    Driver driver{};
    // Created on first use; programs are indexed by aspect and by 2D, 2D-array, cube and cube-array target.
    Uint depthStencilCopyPrograms[2][4]{};
    Uint depthStencilCopySampler{};

private:
    // A new framebuffer, bound, that reads `image` at `attachment`; zero, deleted, when it is incomplete.
    Uint readFramebuffer(const Image &image, Enum target, Int mipLevel, Int layer, Enum attachment);
    Uint depthStencilCopyProgram(Enum target, DepthStencilAspect aspect, std::string &error);
    Size debugLabelLength(std::string_view label) const;

    // Errors not yet reported by a submission. The debug callback may add to it on any thread. The first message is
    // kept with the number of errors.
    struct ErrorQueue {
        std::mutex mutex;
        std::string first;
        size_t count{};
        void push(std::string message);
    };
    // Clears the context's error flag, queueing what it held unless the debug callback already received it.
    void deferErrors();
    void removeDebugOutput();

    Int maxDebugMessageLength{};
    Int maxDebugGroupDepth{};
    // One entry per open group, telling whether its push reached the context.
    std::vector<bool> debugGroupsPushed;
    std::unique_ptr<ErrorQueue> errors = std::make_unique<ErrorQueue>();
    bool debugOutput{};
    // The queued error count when the running synchronous operation began.
    size_t synchronousMark{};
    std::vector<Uint> freeQueries;
};

// The state that a HAL device, its resources and its runtime context share. Its last owner deletes the objects the
// state made on the context, so the host's context must outlive every one of them.
std::shared_ptr<DeviceState> createDeviceState();

} // namespace vernon::hal::opengl

#endif
