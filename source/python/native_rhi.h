#ifndef VERNON_PYTHON_NATIVE_RHI_H
#define VERNON_PYTHON_NATIVE_RHI_H

#include "VernonOpenGLContext.h"
#include "VernonRHI.h"
#include "VernonTextureTypes.h"

#include <nanobind/nanobind.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace nb = nanobind;

struct RhiHostState {
    RhiHostState(VernonRhiBackend backend, uint32_t deviceIndex);
    RhiHostState(VernonRhiBackend backend, const VernonOpenGLContextCallbacks &callbacks);
    ~RhiHostState();

    VernonRhiBackend backend;
    VernonRhiDevice device{};
};

struct RhiBuffer {
    RhiBuffer(std::shared_ptr<RhiHostState> host, size_t size);
    ~RhiBuffer();

    void upload(const nb::bytes &data, size_t offset);
    void uploadRanges(const nb::list &ranges);
    nb::bytes download() const;

    std::shared_ptr<RhiHostState> host;
    VernonRhiBuffer handle{};
    size_t size{};
};

struct RhiImageView;

struct RhiImage {
    RhiImage(std::shared_ptr<RhiHostState> host, uint32_t width, uint32_t height, uint32_t depth,
             VernonTextureFormat format, VernonTextureDimension dimension, uint32_t mipLevels, uint32_t usage);
    ~RhiImage();

    void upload(const nb::bytes &data, uint32_t mipLevel, uint32_t offsetX, uint32_t offsetY, uint32_t offsetZ,
                uint32_t uploadWidth, uint32_t uploadHeight, uint32_t uploadDepth);
    nb::bytes download(uint32_t mipLevel, uint32_t offsetX, uint32_t offsetY, uint32_t offsetZ, uint32_t downloadWidth,
                       uint32_t downloadHeight, uint32_t downloadDepth) const;
    void generateMipmaps();

    std::shared_ptr<RhiHostState> host;
    VernonRhiImage handle{};
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
        VernonRhiImageDataFormat format;
        VernonRhiImageDataType type;
    };

    static uint32_t mipExtent(uint32_t extent, uint32_t level);
    static Layout dataLayout(VernonTextureFormat format);
    static size_t checkedByteSize(uint32_t width, uint32_t height, uint32_t depth, size_t pixelSize);
};

struct RhiImageView {
    RhiImageView(RhiImage *image, VernonTextureFormat format, VernonTextureDimension dimension, uint32_t baseMipLevel,
                 uint32_t mipLevelCount, uint32_t baseArrayLayer, uint32_t arrayLayerCount, uint32_t aspects);
    ~RhiImageView();

    RhiImage *image{};
    std::shared_ptr<RhiHostState> host;
    VernonRhiImageView handle{};
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

struct RhiSampler {
    RhiSampler(std::shared_ptr<RhiHostState> host, VernonRhiSamplerAddressMode address);
    ~RhiSampler();

    std::shared_ptr<RhiHostState> host;
    VernonRhiSampler handle{};
};

struct RhiHost {
    RhiHost(VernonRhiBackend backend, uint32_t deviceIndex);
    explicit RhiHost(std::shared_ptr<RhiHostState> state);

    static std::unique_ptr<RhiHost> createExternalOpenGL(VernonRhiBackend backend, uintptr_t userData,
                                                         uintptr_t makeCurrent, uintptr_t getProcAddress,
                                                         uint16_t apiMajor, uint16_t apiMinor);
    std::unique_ptr<RhiBuffer> createBuffer(size_t size);
    std::unique_ptr<RhiImage> createImage(uint32_t width, uint32_t height, VernonTextureFormat format,
                                          VernonTextureDimension dimension, uint32_t depth, uint32_t mipLevels,
                                          uint32_t usage);
    std::unique_ptr<RhiImage> createAttachmentImage(uint32_t width, uint32_t height, VernonTextureFormat format,
                                                    uint32_t usage);
    std::unique_ptr<RhiSampler> createSampler(VernonRhiSamplerAddressMode address);

    std::shared_ptr<RhiHostState> state;
};

#endif
