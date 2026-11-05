#ifndef VERNON_PYTHON_NATIVE_HAL_H
#define VERNON_PYTHON_NATIVE_HAL_H

#include "VernonFormats.hpp"
#include "VernonHal.h"
#include "VernonOpenGLContext.h"
#include "VernonTextureTypes.h"

#include <nanobind/nanobind.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace nb = nanobind;

struct RuntimeState;

// The message of this thread's last failed HAL call.
std::string halError();

struct HalContextOwner {
    virtual ~HalContextOwner() = default;
};

// A HAL device and the context an OpenGL device adopted. The device owns its runtime, so it outlives every Runtime,
// resource and schedule made on it, and the context outlives the device.
struct HalDeviceState {
    HalDeviceState(VernonRuntimeBackend backend, uint32_t deviceIndex);
    HalDeviceState(VernonRuntimeBackend backend, const VernonOpenGLContextCallbacks &callbacks,
                   std::shared_ptr<HalContextOwner> contextOwner = {});
    ~HalDeviceState();

    VernonRuntimeContext *runtimeContext() const noexcept { return vernonHalDeviceGetRuntime(device); }

    VernonRuntimeBackend backend;
    std::shared_ptr<HalContextOwner> contextOwner;
    VernonHalDevice *device{};
    // The one RuntimeState of the device's runtime, shared by every Runtime made on the device.
    std::weak_ptr<RuntimeState> runtime;
};

struct HalBuffer {
    HalBuffer(std::shared_ptr<HalDeviceState> host, size_t size);
    ~HalBuffer();

    void upload(const nb::bytes &data, size_t offset);
    void uploadRanges(const nb::list &ranges);
    nb::bytes download() const;
    nb::list downloadRanges(const nb::list &ranges) const;
    // The address kernels take the buffer by on the CPU and CUDA devices, looked up on first use.
    void *address();

    std::shared_ptr<HalDeviceState> host;
    VernonHalBuffer *handle{};
    size_t size{};

private:
    void *address_{};
};

struct HalImageView;

struct HalImage {
    HalImage(std::shared_ptr<HalDeviceState> host, uint32_t width, uint32_t height, uint32_t depth,
             VernonTextureFormat format, VernonTextureDimension dimension, uint32_t mipLevels, uint32_t usage,
             uint32_t arrayLayers);
    ~HalImage();

    void upload(const nb::bytes &data, uint32_t mipLevel, uint32_t offsetX, uint32_t offsetY, uint32_t offsetZ,
                uint32_t uploadWidth, uint32_t uploadHeight, uint32_t uploadDepth, uint32_t baseArrayLayer,
                uint32_t arrayLayerCount, uint32_t aspects);
    void uploadRegions(const nb::list &regions);
    nb::bytes download(uint32_t mipLevel, uint32_t offsetX, uint32_t offsetY, uint32_t offsetZ, uint32_t downloadWidth,
                       uint32_t downloadHeight, uint32_t downloadDepth, uint32_t baseArrayLayer,
                       uint32_t arrayLayerCount, uint32_t aspects) const;
    nb::list downloadRegions(const nb::list &regions) const;
    void generateMipmaps();

    std::shared_ptr<HalDeviceState> host;
    VernonHalImage *handle{};
    uint32_t width{};
    uint32_t height{};
    uint32_t depth{1};
    VernonTextureFormat format{};
    VernonTextureDimension dimension{};
    uint32_t mipLevels{1};
    uint32_t usage{};
    uint32_t layers{1};

private:
    struct Layout {
        size_t pixelSize;
        VernonHalImageDataFormat format;
        VernonHalImageDataType type;
    };
    struct Region {
        uint32_t mipLevel;
        uint32_t offsetX;
        uint32_t offsetY;
        uint32_t offsetZ;
        uint32_t width;
        uint32_t height;
        uint32_t depth;
        uint32_t baseArrayLayer;
        uint32_t arrayLayerCount;
        uint32_t aspects;
    };

    static uint32_t mipExtent(uint32_t extent, uint32_t level);
    static Layout dataLayout(VernonTextureFormat format, uint32_t aspects);
    static size_t checkedByteSize(uint32_t width, uint32_t height, uint32_t depth, size_t pixelSize);
    static VernonHalImageRegion layerRegion(const Region &region, uint32_t layer, const Layout &layout);
    Region parseRegion(const nb::dict &region) const;
    Region checkedRegion(Region region, const char *operation) const;
};

struct HalImageView {
    HalImageView(HalImage *image, VernonTextureFormat format, VernonTextureDimension dimension, uint32_t baseMipLevel,
                 uint32_t mipLevelCount, uint32_t baseArrayLayer, uint32_t arrayLayerCount, uint32_t aspects);
    ~HalImageView();

    HalImage *image{};
    std::shared_ptr<HalDeviceState> host;
    VernonHalImageView *handle{};
    VernonTextureFormat format{};
    VernonTextureDimension dimension{};
    uint32_t baseMipLevel{};
    uint32_t mipLevelCount{};
    uint32_t baseArrayLayer{};
    uint32_t arrayLayerCount{};
    uint32_t aspects{};
    uint32_t width{};
    uint32_t height{};
    uint32_t layers{1};
};

struct HalSampler {
    HalSampler(std::shared_ptr<HalDeviceState> host, VernonHalAddressMode address);
    ~HalSampler();

    std::shared_ptr<HalDeviceState> host;
    VernonHalSampler *handle{};
};

struct HalDevice {
    HalDevice(VernonRuntimeBackend backend, uint32_t deviceIndex);
    explicit HalDevice(std::shared_ptr<HalDeviceState> state);

    static std::unique_ptr<HalDevice> createExternalOpenGL(VernonRuntimeBackend backend, uintptr_t userData,
                                                           uintptr_t makeCurrent, uintptr_t getProcAddress,
                                                           uint16_t apiMajor, uint16_t apiMinor);
    static std::unique_ptr<HalDevice> createOwnedOpenGL(VernonRuntimeBackend backend, uint16_t apiMajor,
                                                        uint16_t apiMinor);
    std::unique_ptr<HalBuffer> createBuffer(size_t size);
    // `arrayLayers` counts layers, so a cube array of N cubes takes 6 * N; zero selects the dimension's
    // single-element layer count (6 for a cube, 1 otherwise).
    std::unique_ptr<HalImage> createImage(uint32_t width, uint32_t height, VernonTextureFormat format,
                                          VernonTextureDimension dimension, uint32_t depth, uint32_t mipLevels,
                                          uint32_t usage, uint32_t arrayLayers);
    std::unique_ptr<HalImage> createAttachmentImage(uint32_t width, uint32_t height, VernonTextureFormat format,
                                                    uint32_t usage);
    std::unique_ptr<HalSampler> createSampler(VernonHalAddressMode address);

    std::shared_ptr<HalDeviceState> state;
};

#endif
