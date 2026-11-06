#include "hal/opengl/opengl_resources.h"

#include "VernonFormats.hpp"
#include "hal/native_buffers.h"
#include "hal/native_images.h"
#include "hal/opengl/opengl_state.h"
#include "hal/resources.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vernon::hal {
namespace {

namespace gl = opengl;
using Access = std::shared_ptr<gl::DeviceState>;

constexpr gl::Enum kUnpackAlignment = 0x0CF5;
constexpr gl::Enum kDepthComponent = 0x1902;
constexpr gl::Enum kStencilIndex = 0x1901;
constexpr gl::Enum kFloat = 0x1406;
constexpr gl::Enum kRgba = 0x1908;
constexpr gl::Enum kRgbaInteger = 0x8D99;
constexpr gl::Enum kByte = 0x1400;
constexpr gl::Enum kUnsignedByte = 0x1401;
constexpr gl::Enum kShort = 0x1402;
constexpr gl::Enum kUnsignedShort = 0x1403;
constexpr gl::Enum kInt = 0x1404;
constexpr gl::Enum kUnsignedInt = 0x1405;
constexpr gl::Enum kInvalidOperation = 0x0502;

Serials<gl::Buffer> &serials() {
    static Serials<gl::Buffer> instance;
    return instance;
}

template <typename T = void> Done<T> internal(const char *what, const std::string &detail) {
    return fail<T>(VERNON_HAL_INTERNAL_ERROR, detail.empty() ? std::string(what) : std::string(what) + ": " + detail);
}

uint32_t mipExtent(uint32_t size, uint32_t mip) { return std::max<uint32_t>(1, size >> mip); }

// OpenGL addresses 3D slices and array layers (cube-array layer-faces included) as the z coordinate of the 3D texture
// entry points; cube maps instead use one 2D target per face.
bool volumeTarget(VernonTextureDimension dimension) {
    return dimension == VERNON_TEXTURE_3D || dimension == VERNON_TEXTURE_2D_ARRAY ||
           dimension == VERNON_TEXTURE_CUBE_ARRAY;
}

uint32_t volumeDepth(const VernonImageDescriptor &descriptor, uint32_t mip) {
    return descriptor.dimension == VERNON_TEXTURE_3D ? mipExtent(descriptor.extent.depth, mip)
                                                     : descriptor.array_layer_count;
}

struct TransferLayout {
    gl::Enum format{};
    gl::Enum type{};
};

// A region transfers whole texels in the table's layout or one aspect of a depth-stencil image. The packed
// depth-stencil layout is GL_FLOAT_32_UNSIGNED_INT_24_8_REV: the depth float, then a word whose low byte is the
// stencil.
TransferLayout transferLayout(VernonTextureFormat format, uint32_t aspect) {
    const formats::Format *entry = formats::find(format);
    if (!entry)
        return {};
    if (aspect == entry->aspects())
        return {entry->glFormat, entry->glType};
    if (aspect == VERNON_IMAGE_ASPECT_DEPTH)
        return {kDepthComponent, kFloat};
    if (aspect == VERNON_IMAGE_ASPECT_STENCIL)
        return {kStencilIndex, kUnsignedByte};
    return {};
}

struct EmbeddedRead {
    gl::Enum format{};
    gl::Enum type{};
    size_t componentBytes{};
};

// OpenGL ES reads a color buffer as four components in the format and type its component class guarantees on every
// implementation, and in any other pair only where the implementation names it, so an image whose own layout is not
// that pair reads through it and is repacked. Packed unorm and float layouts are their own guaranteed pair or are not
// renderable, so they never read through RGBA.
std::optional<EmbeddedRead> embeddedRead(const formats::Format &entry) {
    if (entry.aspects() != VERNON_IMAGE_ASPECT_COLOR)
        return std::nullopt;
    const bool wide = entry.dataType == VERNON_HAL_IMAGE_DATA_UINT16 || entry.dataType == VERNON_HAL_IMAGE_DATA_INT16;
    EmbeddedRead read;
    switch (entry.componentClass) {
    case formats::ComponentClass::Float:
        if (entry.dataType != VERNON_HAL_IMAGE_DATA_FLOAT16 && entry.dataType != VERNON_HAL_IMAGE_DATA_FLOAT32)
            return std::nullopt;
        read = {kRgba, kFloat, sizeof(float)};
        break;
    case formats::ComponentClass::Unorm:
        if (entry.dataType == VERNON_HAL_IMAGE_DATA_UINT32)
            return std::nullopt;
        read = wide ? EmbeddedRead{kRgba, kUnsignedShort, 2} : EmbeddedRead{kRgba, kUnsignedByte, 1};
        break;
    case formats::ComponentClass::Snorm:
        read = wide ? EmbeddedRead{kRgba, kShort, 2} : EmbeddedRead{kRgba, kByte, 1};
        break;
    case formats::ComponentClass::Uint:
        read = {kRgbaInteger, kUnsignedInt, sizeof(uint32_t)};
        break;
    case formats::ComponentClass::Sint:
        read = {kRgbaInteger, kInt, sizeof(int32_t)};
        break;
    }
    if (read.format == entry.glFormat && read.type == entry.glType)
        return std::nullopt;
    return read;
}

// The half-float bits of `value`, which must be a half-float's value, as every texel of a half-float image is.
uint16_t exactHalf(float value) {
    uint32_t bits{};
    std::memcpy(&bits, &value, sizeof(bits));
    const auto sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t exponent = (bits >> 23) & 0xFFu;
    uint32_t mantissa = bits & 0x7FFFFFu;
    if (exponent == 0xFFu)
        return static_cast<uint16_t>(sign | 0x7C00u | (mantissa != 0 ? 0x200u : 0u));
    const int biased = static_cast<int>(exponent) - 127 + 15;
    if (biased >= 31)
        return static_cast<uint16_t>(sign | 0x7C00u);
    if (biased <= 0) {
        if (exponent == 0 || biased < -10)
            return sign;
        mantissa |= 0x800000u;
        return static_cast<uint16_t>(sign | (mantissa >> (14 - biased)));
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(biased) << 10) | (mantissa >> 13));
}

// Repacks `texels` read as `read` into the image's whole-texel layout. An integer narrows to its low bytes, which hold
// its value on the little-endian hosts OpenGL ES runs on.
void repackEmbeddedRead(const formats::Format &entry, const EmbeddedRead &read, const uint8_t *rgba, size_t texels,
                        uint8_t *output) {
    const size_t stride = 4 * read.componentBytes;
    if (entry.texture == VERNON_TEXTURE_RGB10A2_UINT) {
        for (size_t texel = 0; texel < texels; ++texel) {
            uint32_t component[4]{};
            std::memcpy(component, rgba + texel * stride, sizeof(component));
            const uint32_t word = (component[0] & 0x3FFu) | ((component[1] & 0x3FFu) << 10) |
                                  ((component[2] & 0x3FFu) << 20) | ((component[3] & 0x3u) << 30);
            std::memcpy(output + texel * sizeof(word), &word, sizeof(word));
        }
        return;
    }
    const uint32_t components = entry.components();
    const size_t elementBytes = entry.pixelBytes / components;
    for (size_t texel = 0; texel < texels; ++texel)
        for (uint32_t component = 0; component < components; ++component) {
            const uint8_t *from = rgba + texel * stride + component * read.componentBytes;
            uint8_t *to = output + (texel * components + component) * elementBytes;
            if (entry.dataType == VERNON_HAL_IMAGE_DATA_FLOAT16) {
                float value{};
                std::memcpy(&value, from, sizeof(value));
                const uint16_t bits = exactHalf(value);
                std::memcpy(to, &bits, sizeof(bits));
            } else {
                std::memcpy(to, from, elementBytes);
            }
        }
}

void storePackedDepthStencil(uint8_t *texel, float depth, uint8_t stencil) {
    std::memcpy(texel, &depth, sizeof(depth));
    texel[4] = stencil;
    std::memset(texel + 5, 0, kPackedDepthStencilBytes - 5);
}

// Outside validation mode an operation's errors surface when its completion is checked. A synchronous operation
// already waits for the driver, so it checks its own errors in every mode, and every open scope is closed by
// scopeError.
enum class Scope { Validation, Always };

std::optional<std::string> openScope(gl::DeviceState &state, const char *operation, Scope scope = Scope::Validation) {
    if (!state.validation) {
        if (scope == Scope::Always)
            state.beginSynchronous();
        return std::nullopt;
    }
    const gl::Enum error = state.readError();
    if (!error)
        return std::nullopt;
    return "the OpenGL context had error " + std::to_string(error) + " before " + operation;
}

gl::Enum scopeError(gl::DeviceState &state, Scope scope = Scope::Validation) {
    if (state.validation)
        return state.readError();
    return scope == Scope::Always ? state.endSynchronous() : 0;
}

Done<void> closeScope(gl::DeviceState &state, const char *operation, Scope scope = Scope::Validation) {
    const gl::Enum error = scopeError(state, scope);
    if (!error)
        return Done<void>{ok()};
    return fail(VERNON_HAL_INTERNAL_ERROR,
                std::string(operation) + " failed with OpenGL error " + std::to_string(error));
}

// glBlitFramebuffer honors the scissor test, which holds the last scope's or draw's state, and on desktop GL
// GL_FRAMEBUFFER_SRGB, which a render scope leaves enabled.
void prepareBlits(gl::DeviceState &state) {
    state.driver.disable(gl::kScissorTest);
    if (!state.embeddedProfile)
        state.driver.disable(gl::kFramebufferSrgb);
}

// One OpenGL buffer, named by its serial. Every OpenGL call holds the device's mutex and makes its context current
// first.
class NativeBuffer {
public:
    NativeBuffer(Access access, gl::Buffer buffer, uint64_t size) : access_(access), buffer_(buffer), size_(size) {
        serial_ = serials().add(&buffer_);
    }
    NativeBuffer(const NativeBuffer &) = delete;
    NativeBuffer &operator=(const NativeBuffer &) = delete;
    ~NativeBuffer() {
        serials().remove(serial_);
        std::lock_guard<std::mutex> guard(access_->mutex);
        access_->makeCurrent();
        access_->destroyBuffer(buffer_);
    }

    const gl::Buffer &buffer() const { return buffer_; }
    uint64_t size() const { return size_; }
    VernonRuntimeProviderResourceReference reference(uint64_t offset, uint64_t size) const {
        VernonRuntimeProviderResourceReference reference{};
        reference.identity = nativeBufferIdentity(access_.get());
        reference.resource.value = serial_;
        reference.offset = offset;
        reference.size = size;
        return reference;
    }

private:
    Access access_;
    gl::Buffer buffer_;
    uint64_t size_;
    uint64_t serial_{};
};

struct OpenGLHalBuffer final : VernonHalBuffer {
    OpenGLHalBuffer(Access access, gl::Buffer buffer, uint64_t size) : native(access, buffer, size) {}

    NativeBuffer native;
};

Result<gl::Buffer, std::string> createNative(Access access, uint64_t size) {
    using Created = Result<gl::Buffer, std::string>;
    if (size > (std::numeric_limits<size_t>::max)())
        return Created{err(std::string("the buffer is too large"))};
    gl::Buffer native{};
    std::lock_guard<std::mutex> guard(access->mutex);
    access->makeCurrent();
    std::string error;
    if (!access->createBuffer(native, static_cast<size_t>(size), error))
        return Created{err("cannot create the buffer: " + error)};
    return Created{ok(native)};
}

void destroyNative(Access access, gl::Buffer &native) {
    std::lock_guard<std::mutex> guard(access->mutex);
    access->makeCurrent();
    access->destroyBuffer(native);
}

struct OpenGLImage final : VernonHalImage {
    explicit OpenGLImage(Access access) : access(access) {}
    OpenGLImage(const OpenGLImage &) = delete;
    OpenGLImage &operator=(const OpenGLImage &) = delete;
    ~OpenGLImage() override {
        forgetUndefined(&serial, 1);
        std::lock_guard<std::mutex> guard(access->mutex);
        access->destroyImage(native);
    }

    Access access;
    gl::Image native;
    gl::Enum target{};
    uint64_t serial{nextImageSerial()};
};

struct OpenGLView final : VernonHalImageView {
    explicit OpenGLView(Access access) : access(access) {}
    OpenGLView(const OpenGLView &) = delete;
    OpenGLView &operator=(const OpenGLView &) = delete;
    ~OpenGLView() override {
        if (serial)
            nativeViews().remove(serial);
        std::lock_guard<std::mutex> guard(access->mutex);
        access->destroyImage(owned);
    }

    Access access;
    // A texture view's own texture; empty for an identity view.
    gl::Image owned;
    OpenGLViewTexture texture;
    NativeView entry;
    uint64_t serial{};
};

struct OpenGLSampler final : VernonHalSampler {
    explicit OpenGLSampler(Access access) : access(access) {}
    OpenGLSampler(const OpenGLSampler &) = delete;
    OpenGLSampler &operator=(const OpenGLSampler &) = delete;
    ~OpenGLSampler() override {
        if (serial)
            nativeSamplers().remove(serial);
        std::lock_guard<std::mutex> guard(access->mutex);
        access->makeCurrent();
        access->destroySampler(sampler);
    }

    Access access;
    gl::Sampler sampler;
    uint64_t serial{};
};

// A window's default framebuffer, and the ring of images that passes render to and presents blit to it.
struct OpenGLSurface final : VernonHalSurface {
    VernonOpenGLSurfaceCallbacks callbacks{};
    VernonHalSurfaceConfiguration configuration{};
    // Index-aligned; each view covers its image, and the views are destroyed first.
    std::vector<std::unique_ptr<OpenGLImage>> images;
    std::vector<std::unique_ptr<OpenGLView>> views;
    size_t next{};
    // The swap interval last set on the window; -1 before the first present.
    int32_t interval{-1};
};

// OpenGL ES differs from desktop OpenGL in these rules: float color rendering needs ES 3.2 or an extension, 32-bit
// float filtering and blending need extensions, 16-bit normalized formats need an extension and snorm rendering
// another, BGRA images are not core, depth textures are never linearly filtered, and image units accept only the
// formats listed in ES 3.1.
uint32_t embeddedFormatCapabilities(const gl::DeviceState &device, const formats::Format &entry) {
    constexpr uint32_t rendering = VERNON_HAL_FORMAT_RENDERABLE | VERNON_HAL_FORMAT_BLENDABLE |
                                   VERNON_HAL_FORMAT_MULTISAMPLE | VERNON_HAL_FORMAT_RESOLVABLE;
    uint32_t result = entry.baseline;
    const bool color = entry.aspects() == VERNON_IMAGE_ASPECT_COLOR;
    const bool floatColor = color && entry.componentClass == formats::ComponentClass::Float;
    const bool float32 = floatColor && entry.dataType == VERNON_HAL_IMAGE_DATA_FLOAT32;
    const bool float16 = floatColor && entry.dataType == VERNON_HAL_IMAGE_DATA_FLOAT16;
    const bool normalized = entry.componentClass == formats::ComponentClass::Unorm ||
                            entry.componentClass == formats::ComponentClass::Snorm;
    const bool norm16 =
        color && normalized &&
        (entry.dataType == VERNON_HAL_IMAGE_DATA_UINT16 || entry.dataType == VERNON_HAL_IMAGE_DATA_INT16);
    if (entry.dataFormat == VERNON_HAL_IMAGE_DATA_BGRA || (norm16 && !device.hasExtension("GL_EXT_texture_norm16")))
        return 0;
    if (floatColor && !device.apiVersionAtLeast(3, 2) && !device.hasExtension("GL_EXT_color_buffer_float") &&
        !(float16 && device.hasExtension("GL_EXT_color_buffer_half_float")))
        result &= ~rendering;
    if (entry.componentClass == formats::ComponentClass::Snorm && !device.hasExtension("GL_EXT_render_snorm"))
        result &= ~rendering;
    if (float32 && !device.hasExtension("GL_OES_texture_float_linear"))
        result &= ~VERNON_HAL_FORMAT_FILTERABLE;
    if (float32 && !device.hasExtension("GL_EXT_float_blend"))
        result &= ~VERNON_HAL_FORMAT_BLENDABLE;
    if (!color)
        result &= ~VERNON_HAL_FORMAT_FILTERABLE;
    switch (entry.texture) {
    case VERNON_TEXTURE_RGBA8_UNORM:
    case VERNON_TEXTURE_RGBA8_SNORM:
    case VERNON_TEXTURE_RGBA8_UINT:
    case VERNON_TEXTURE_RGBA8_SINT:
    case VERNON_TEXTURE_RGBA16_FLOAT:
    case VERNON_TEXTURE_RGBA16_UINT:
    case VERNON_TEXTURE_RGBA16_SINT:
    case VERNON_TEXTURE_R32_FLOAT:
    case VERNON_TEXTURE_R32_UINT:
    case VERNON_TEXTURE_R32_SINT:
    case VERNON_TEXTURE_RGBA32_FLOAT:
    case VERNON_TEXTURE_RGBA32_UINT:
    case VERNON_TEXTURE_RGBA32_SINT:
        break;
    default:
        result &= ~VERNON_HAL_FORMAT_STORAGE;
    }
    return result;
}

class OpenGLDeviceBuffers final : public DeviceBuffers {
public:
    explicit OpenGLDeviceBuffers(Access access)
        : access_(access), pool_(std::make_shared<BufferPool<NativeBuffer>>([access](uint64_t size) {
              using Created = Result<std::unique_ptr<NativeBuffer>, std::string>;
              auto created = createNative(access, size);
              if (created.isErr())
                  return Created{err(std::move(created).error())};
              gl::Buffer native = created.value();
              auto buffer = std::unique_ptr<NativeBuffer>(new (std::nothrow) NativeBuffer(access, native, size));
              if (!buffer) {
                  destroyNative(access, native);
                  return Created{err(std::string("cannot allocate a buffer"))};
              }
              return Created{ok(std::move(buffer))};
          })) {}

    Result<std::unique_ptr<Buffer>, std::string> allocate(uint64_t size) override { return pool_->allocate(size); }

    Result<void, std::string> upload(Buffer &buffer, const void *bytes, size_t size) override {
        const NativeBuffer &native = static_cast<BufferPool<NativeBuffer>::Pooled &>(buffer).native();
        if (size > native.size())
            return Result<void, std::string>{err(std::string("the upload overruns the buffer"))};
        std::lock_guard<std::mutex> guard(access_->mutex);
        access_->makeCurrent();
        std::string error;
        if (!access_->uploadBuffer(native.buffer(), 0, bytes, size, error))
            return Result<void, std::string>{err("cannot upload to an OpenGL buffer: " + error)};
        return Result<void, std::string>{ok()};
    }

    Result<void, std::string> download(const Buffer &buffer, void *bytes, size_t size) override {
        const NativeBuffer &native = static_cast<const BufferPool<NativeBuffer>::Pooled &>(buffer).native();
        if (size > native.size())
            return Result<void, std::string>{err(std::string("the download overruns the buffer"))};
        std::lock_guard<std::mutex> guard(access_->mutex);
        access_->makeCurrent();
        std::string error;
        if (!access_->downloadBuffer(native.buffer(), 0, bytes, size, error))
            return Result<void, std::string>{err("cannot read an OpenGL buffer back: " + error)};
        return Result<void, std::string>{ok()};
    }

private:
    Access access_;
    std::shared_ptr<BufferPool<NativeBuffer>> pool_;
};

class OpenGLResources final : public Resources {
public:
    explicit OpenGLResources(Access access) : access_(access), state_(*access_) {}
    OpenGLResources(const OpenGLResources &) = delete;
    OpenGLResources &operator=(const OpenGLResources &) = delete;
    ~OpenGLResources() override {
        if (!presentFramebuffer_)
            return;
        std::lock_guard<std::mutex> guard(access_->mutex);
        state_.makeCurrent();
        state_.destroyFramebuffer(presentFramebuffer_);
    }

    Done<uint32_t> formatCapabilities(VernonTextureFormat format) override {
        const formats::Format *entry = formats::find(format);
        if (!entry)
            return fail<uint32_t>(VERNON_HAL_INVALID_ARGUMENT, "the format is unknown");
        std::lock_guard<std::mutex> guard(access_->mutex);
        state_.makeCurrent();
        uint32_t result = state_.embeddedProfile ? embeddedFormatCapabilities(state_, *entry) : entry->baseline;
        if (!state_.supportsCompute())
            result &= ~VERNON_HAL_FORMAT_STORAGE;
        // 2D copies blit through framebuffers, so they need a color-renderable format; OpenGL renders RGB8, which the
        // table baseline leaves unrenderable for the other backends.
        if (!(result & VERNON_HAL_FORMAT_RENDERABLE) && entry->texture != VERNON_TEXTURE_RGB8_UNORM)
            result &= ~VERNON_HAL_FORMAT_COPYABLE;
        gl::Int sampleCounts = 0;
        if ((result & VERNON_HAL_FORMAT_RENDERABLE) && state_.driver.getInternalformativ)
            state_.driver.getInternalformativ(gl::kRenderbuffer, static_cast<gl::Enum>(entry->glInternalFormat),
                                              gl::kNumSampleCounts, 1, &sampleCounts);
        if (sampleCounts <= 0)
            result &= ~(VERNON_HAL_FORMAT_MULTISAMPLE | VERNON_HAL_FORMAT_RESOLVABLE);
        return Done<uint32_t>{ok(result)};
    }

    Done<std::unique_ptr<VernonHalBuffer>> createBuffer(const VernonHalBufferDescriptor &descriptor) override {
        using Created = Done<std::unique_ptr<VernonHalBuffer>>;
        if (descriptor.size > (std::numeric_limits<size_t>::max)())
            return fail<std::unique_ptr<VernonHalBuffer>>(VERNON_HAL_RESOURCE_EXHAUSTED, "the buffer is too large");
        auto created = createNative(access_, descriptor.size);
        if (created.isErr())
            return fail<std::unique_ptr<VernonHalBuffer>>(VERNON_HAL_INTERNAL_ERROR, created.error());
        gl::Buffer native = created.value();
        auto buffer =
            std::unique_ptr<OpenGLHalBuffer>(new (std::nothrow) OpenGLHalBuffer(access_, native, descriptor.size));
        if (!buffer) {
            destroyNative(access_, native);
            return fail<std::unique_ptr<VernonHalBuffer>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a buffer");
        }
        return Created{ok(std::unique_ptr<VernonHalBuffer>(std::move(buffer)))};
    }

    Done<void> write(VernonHalBuffer &buffer, const VernonHalBufferWrite *writes, size_t count) override {
        const gl::Buffer &native = static_cast<OpenGLHalBuffer &>(buffer).native.buffer();
        std::lock_guard<std::mutex> guard(access_->mutex);
        state_.makeCurrent();
        std::string error;
        for (size_t index = 0; index < count; ++index)
            if (!state_.uploadBuffer(native, static_cast<size_t>(writes[index].offset), writes[index].source,
                                     static_cast<size_t>(writes[index].size), error))
                return fail(VERNON_HAL_INTERNAL_ERROR, "cannot write the buffer: " + error);
        return Done<void>{ok()};
    }

    Done<void> read(VernonHalBuffer &buffer, const VernonHalBufferRead *reads, size_t count) override {
        const gl::Buffer &native = static_cast<OpenGLHalBuffer &>(buffer).native.buffer();
        std::lock_guard<std::mutex> guard(access_->mutex);
        state_.makeCurrent();
        std::string error;
        for (size_t index = 0; index < count; ++index)
            if (!state_.downloadBuffer(native, static_cast<size_t>(reads[index].offset), reads[index].destination,
                                       static_cast<size_t>(reads[index].size), error))
                return fail(VERNON_HAL_INTERNAL_ERROR, "cannot read the buffer: " + error);
        return Done<void>{ok()};
    }

    Done<void *> address(const VernonHalBuffer &) override {
        return fail<void *>(VERNON_HAL_UNSUPPORTED, "only CPU and CUDA buffers have an address");
    }

    Done<std::unique_ptr<VernonHalImage>> createImage(const VernonImageDescriptor &descriptor) override {
        auto made = makeImage(descriptor);
        if (made.isErr())
            return Done<std::unique_ptr<VernonHalImage>>{err(std::move(made).error())};
        return Done<std::unique_ptr<VernonHalImage>>{ok(std::unique_ptr<VernonHalImage>(std::move(made).value()))};
    }

    Done<void> write(VernonHalImage &image, const VernonHalImageWrite *writes, size_t count) override {
        if (!count)
            return Done<void>{ok()};
        const VernonImageDescriptor &descriptor = image.descriptor;
        for (size_t index = 0; index < count; ++index) {
            if (!regionBytes(descriptor, writes[index].region))
                return fail(VERNON_HAL_INVALID_ARGUMENT, "a write's data layout does not match the image");
            // glTexSubImage has no format that writes one aspect of a packed depth-stencil texture.
            if (formats::hasStencil(descriptor.format) &&
                writes[index].region.aspect != (VERNON_IMAGE_ASPECT_DEPTH | VERNON_IMAGE_ASPECT_STENCIL))
                return fail(VERNON_HAL_UNSUPPORTED, "OpenGL writes the depth and stencil of a packed image together");
        }
        const auto &target = static_cast<OpenGLImage &>(image);
        std::lock_guard<std::mutex> guard(access_->mutex);
        state_.makeCurrent();
        if (auto opened = openScope(state_, "writing an image"))
            return fail(VERNON_HAL_INTERNAL_ERROR, *opened);
        gl::Driver &driver = state_.driver;
        driver.bindTexture(target.target, target.native.name);
        driver.pixelStorei(kUnpackAlignment, 1);
        for (size_t index = 0; index < count; ++index) {
            const VernonHalImageRegion &region = writes[index].region;
            const TransferLayout layout = transferLayout(descriptor.format, region.aspect);
            const auto mip = static_cast<gl::Int>(region.mip_level);
            const auto x = static_cast<gl::Int>(region.offset_x);
            const auto y = static_cast<gl::Int>(region.offset_y);
            if (volumeTarget(descriptor.dimension))
                driver.texSubImage3D(
                    target.target, mip, x, y,
                    static_cast<gl::Int>(descriptor.dimension == VERNON_TEXTURE_3D ? region.offset_z
                                                                                   : region.array_layer),
                    static_cast<gl::Size>(region.width), static_cast<gl::Size>(region.height),
                    static_cast<gl::Size>(region.depth), layout.format, layout.type, writes[index].source);
            else
                driver.texSubImage2D(
                    descriptor.dimension == VERNON_TEXTURE_CUBE ? gl::kTextureCubeMapPositiveX + region.array_layer
                                                                : target.target,
                    mip, x, y, static_cast<gl::Size>(region.width), static_cast<gl::Size>(region.height), layout.format,
                    layout.type, writes[index].source);
        }
        driver.bindTexture(target.target, 0);
        Done<void> closed = closeScope(state_, "writing an image");
        if (!closed.isErr())
            forgetUndefined(&target.serial, 1);
        return closed;
    }

    Done<void> read(VernonHalImage &image, const VernonHalImageRead *reads, size_t count) override {
        for (size_t index = 0; index < count; ++index)
            if (!regionBytes(image.descriptor, reads[index].region))
                return fail(VERNON_HAL_INVALID_ARGUMENT, "a read's data layout does not match the image");
        const auto &source = static_cast<OpenGLImage &>(image);
        if (anyUndefined(&source.serial, 1))
            return fail(VERNON_HAL_INVALID_ARGUMENT, "the image's contents are undefined");
        std::lock_guard<std::mutex> guard(access_->mutex);
        state_.makeCurrent();
        for (size_t index = 0; index < count; ++index) {
            // glReadPixels reports an unsupported format or type only through glGetError and leaves the destination
            // untouched, so the check runs in every mode.
            if (auto opened = openScope(state_, "reading an image", Scope::Always))
                return fail(VERNON_HAL_INTERNAL_ERROR, *opened);
            auto read = readRegion(source, reads[index].region, reads[index].destination);
            const gl::Enum error = scopeError(state_, Scope::Always);
            if (read.isErr())
                return read;
            if (error)
                return fail(VERNON_HAL_INTERNAL_ERROR,
                            "reading an image failed with OpenGL error " + std::to_string(error));
        }
        return Done<void>{ok()};
    }

    Done<void> generateMipmaps(VernonHalImage &image) override {
        const VernonImageDescriptor &descriptor = image.descriptor;
        if (descriptor.mip_level_count < 2 || formats::hasDepth(descriptor.format))
            return fail(VERNON_HAL_INVALID_ARGUMENT, "the image has no mips to generate");
        const auto &target = static_cast<OpenGLImage &>(image);
        std::lock_guard<std::mutex> guard(access_->mutex);
        state_.makeCurrent();
        // glGenerateMipmap reports an unsupported format only through glGetError, so the check runs in every mode.
        if (auto opened = openScope(state_, "generating mipmaps", Scope::Always))
            return fail(VERNON_HAL_INTERNAL_ERROR, *opened);
        state_.driver.bindTexture(target.target, target.native.name);
        state_.driver.generateMipmap(target.target);
        state_.driver.bindTexture(target.target, 0);
        const gl::Enum error = scopeError(state_, Scope::Always);
        if (error == kInvalidOperation)
            return fail(VERNON_HAL_UNSUPPORTED, "OpenGL cannot generate the mipmaps of the image's format");
        if (error)
            return fail(VERNON_HAL_INTERNAL_ERROR,
                        "generating mipmaps failed with OpenGL error " + std::to_string(error));
        return Done<void>{ok()};
    }

    Done<std::unique_ptr<VernonHalImageView>> createView(VernonHalImage &image,
                                                         const VernonImageViewDescriptor &descriptor) override {
        auto made = makeView(static_cast<OpenGLImage &>(image), descriptor);
        if (made.isErr())
            return Done<std::unique_ptr<VernonHalImageView>>{err(std::move(made).error())};
        return Done<std::unique_ptr<VernonHalImageView>>{
            ok(std::unique_ptr<VernonHalImageView>(std::move(made).value()))};
    }

    // OpenGL samples isotropically, so `max_anisotropy` changes nothing.
    Done<std::unique_ptr<VernonHalSampler>> createSampler(const VernonHalSamplerDescriptor &descriptor) override {
        using Created = Done<std::unique_ptr<VernonHalSampler>>;
        if (descriptor.min_filter > VERNON_HAL_FILTER_LINEAR || descriptor.mag_filter > VERNON_HAL_FILTER_LINEAR ||
            descriptor.mip_filter > VERNON_HAL_FILTER_LINEAR ||
            descriptor.address_u > VERNON_HAL_ADDRESS_MIRRORED_REPEAT ||
            descriptor.address_v > VERNON_HAL_ADDRESS_MIRRORED_REPEAT ||
            descriptor.address_w > VERNON_HAL_ADDRESS_MIRRORED_REPEAT || !(descriptor.max_anisotropy <= 16.0f))
            return fail<std::unique_ptr<VernonHalSampler>>(VERNON_HAL_INVALID_ARGUMENT,
                                                           "the sampler descriptor is invalid");
        constexpr gl::Int kAddress[] = {0x2901, 0x812F, 0x8370};
        // GL_NEAREST_MIPMAP_NEAREST, GL_LINEAR_MIPMAP_NEAREST, GL_NEAREST_MIPMAP_LINEAR, GL_LINEAR_MIPMAP_LINEAR.
        const gl::Int minFilter = 0x2700 + (descriptor.min_filter == VERNON_HAL_FILTER_LINEAR ? 1 : 0) +
                                  (descriptor.mip_filter == VERNON_HAL_FILTER_LINEAR ? 2 : 0);
        const gl::Int magFilter = descriptor.mag_filter == VERNON_HAL_FILTER_LINEAR ? 0x2601 : 0x2600;
        auto sampler = std::unique_ptr<OpenGLSampler>(new (std::nothrow) OpenGLSampler(access_));
        if (!sampler)
            return fail<std::unique_ptr<VernonHalSampler>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a sampler");
        std::string error;
        {
            std::lock_guard<std::mutex> guard(access_->mutex);
            state_.makeCurrent();
            if (!state_.createSampler(sampler->sampler, kAddress[descriptor.address_u], kAddress[descriptor.address_v],
                                      kAddress[descriptor.address_w], minFilter, magFilter, error))
                return internal<std::unique_ptr<VernonHalSampler>>("cannot create the sampler", error);
        }
        sampler->serial = nativeSamplers().add(&sampler->sampler);
        return Created{ok(std::unique_ptr<VernonHalSampler>(std::move(sampler)))};
    }

    Done<std::unique_ptr<VernonHalSurface>> createSurface(const VernonHalSurfaceDescriptor &descriptor) override {
        using Created = Done<std::unique_ptr<VernonHalSurface>>;
        const VernonOpenGLSurfaceCallbacks *callbacks = descriptor.native.opengl;
        if (!callbacks || callbacks->struct_size < sizeof(*callbacks) || !callbacks->make_current ||
            !callbacks->swap_interval || !callbacks->swap_buffers || !callbacks->drawable_size)
            return fail<std::unique_ptr<VernonHalSurface>>(VERNON_HAL_INVALID_ARGUMENT,
                                                           "the surface's callbacks are incomplete");
        if (!state_.driver.blitFramebuffer)
            return fail<std::unique_ptr<VernonHalSurface>>(VERNON_HAL_UNSUPPORTED, "the context cannot blit");
        auto surface = std::unique_ptr<OpenGLSurface>(new (std::nothrow) OpenGLSurface());
        if (!surface)
            return fail<std::unique_ptr<VernonHalSurface>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a surface");
        surface->callbacks = *callbacks;
        return Created{ok(std::unique_ptr<VernonHalSurface>(std::move(surface)))};
    }

    Done<VernonHalSurfaceCapabilities> capabilities(VernonHalSurface &surface) override {
        const VernonOpenGLSurfaceCallbacks &callbacks = static_cast<OpenGLSurface &>(surface).callbacks;
        VernonHalSurfaceCapabilities capabilities{};
        capabilities.struct_size = sizeof(capabilities);
        capabilities.formats[0] = VERNON_TEXTURE_RGBA8_UNORM;
        capabilities.format_count = 1;
        capabilities.present_modes =
            (1u << VERNON_HAL_PRESENT_FIFO) | (1u << VERNON_HAL_PRESENT_MAILBOX) | (1u << VERNON_HAL_PRESENT_IMMEDIATE);
        // Without sampled or storage usage, an acquired image is read only by a preserving render scope or a read,
        // which a submission rejects while its contents are undefined.
        capabilities.usage = VERNON_IMAGE_COLOR_ATTACHMENT | VERNON_IMAGE_TRANSFER_SOURCE;
        capabilities.min_image_count = 1;
        callbacks.drawable_size(callbacks.user_data, &capabilities.current_width, &capabilities.current_height);
        return Done<VernonHalSurfaceCapabilities>{ok(capabilities)};
    }

    // The new images replace the old ones only once all of them exist.
    Done<void> configure(VernonHalSurface &surface, const VernonHalSurfaceConfiguration &configuration) override {
        auto &target = static_cast<OpenGLSurface &>(surface);
        const VernonImageDescriptor descriptor{VERNON_TEXTURE_2D,
                                               {configuration.width, configuration.height, 1},
                                               configuration.format,
                                               1,
                                               1,
                                               1,
                                               configuration.usage | VERNON_IMAGE_TRANSFER_SOURCE};
        const VernonImageViewDescriptor viewDescriptor{
            VERNON_TEXTURE_2D, configuration.format, {0, 1, 0, 1, VERNON_IMAGE_ASPECT_COLOR}};
        std::vector<std::unique_ptr<OpenGLImage>> images;
        std::vector<std::unique_ptr<OpenGLView>> views;
        for (uint32_t index = 0; index < configuration.image_count; ++index) {
            auto image = makeImage(descriptor);
            if (image.isErr())
                return Done<void>{err(std::move(image).error())};
            std::unique_ptr<OpenGLImage> made = std::move(image).value();
            made->device = surface.device;
            made->descriptor = descriptor;
            made->surface = &surface;
            auto view = makeView(*made, viewDescriptor);
            if (view.isErr())
                return Done<void>{err(std::move(view).error())};
            std::unique_ptr<OpenGLView> madeView = std::move(view).value();
            madeView->image = made.get();
            madeView->descriptor = viewDescriptor;
            images.push_back(std::move(made));
            views.push_back(std::move(madeView));
        }
        target.views = std::move(views);
        target.images = std::move(images);
        target.configuration = configuration;
        target.next = 0;
        return Done<void>{ok()};
    }

    // A window whose drawable is no longer the configured size reports out of date, so the host configures again.
    Done<VernonHalSurfaceImage> acquire(VernonHalSurface &surface) override {
        auto &source = static_cast<OpenGLSurface &>(surface);
        if (source.images.empty())
            return fail<VernonHalSurfaceImage>(VERNON_HAL_INVALID_ARGUMENT, "the surface is not configured");
        uint32_t width = 0;
        uint32_t height = 0;
        source.callbacks.drawable_size(source.callbacks.user_data, &width, &height);
        if (width != source.configuration.width || height != source.configuration.height)
            return fail<VernonHalSurfaceImage>(VERNON_HAL_SURFACE_OUT_OF_DATE, "the surface is out of date");
        OpenGLImage &image = *source.images[source.next];
        markUndefined(image.serial);
        return Done<VernonHalSurfaceImage>{ok(VernonHalSurfaceImage{&image, source.views[source.next].get(), 0})};
    }

    Done<void> present(VernonHalSurface &surface) override {
        auto &target = static_cast<OpenGLSurface &>(surface);
        const OpenGLImage &presented = *target.images[target.next];
        target.next = (target.next + 1) % target.images.size();
        const VernonOpenGLSurfaceCallbacks &callbacks = target.callbacks;
        std::lock_guard<std::mutex> guard(access_->mutex);
        callbacks.make_current(callbacks.user_data);
        const int32_t interval = target.configuration.present_mode == VERNON_HAL_PRESENT_FIFO ? 1 : 0;
        if (target.interval != interval) {
            callbacks.swap_interval(callbacks.user_data, interval);
            target.interval = interval;
        }
        auto blitted = blitToWindow(presented, target.configuration.width, target.configuration.height);
        if (blitted.isOk())
            callbacks.swap_buffers(callbacks.user_data);
        state_.makeCurrent();
        return blitted;
    }

    Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &, const VernonHalBuffer &buffer,
                                                           uint64_t offset, uint64_t size) override {
        return Done<VernonRuntimeProviderResourceReference>{
            ok(static_cast<const OpenGLHalBuffer &>(buffer).native.reference(offset, size))};
    }
    Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &,
                                                           const VernonHalImageView &view) override {
        VernonRuntimeProviderResourceReference reference{};
        reference.identity = nativeIdentity(&state_, kNativeViewReference);
        reference.resource.value = static_cast<const OpenGLView &>(view).serial;
        return Done<VernonRuntimeProviderResourceReference>{ok(reference)};
    }
    Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &,
                                                           const VernonHalSampler &sampler) override {
        VernonRuntimeProviderResourceReference reference{};
        reference.identity = nativeIdentity(&state_, kNativeSamplerReference);
        reference.resource.value = static_cast<const OpenGLSampler &>(sampler).serial;
        return Done<VernonRuntimeProviderResourceReference>{ok(reference)};
    }

private:
    Done<std::unique_ptr<OpenGLImage>> makeImage(const VernonImageDescriptor &descriptor) {
        using Made = Done<std::unique_ptr<OpenGLImage>>;
        if (!validImage(descriptor))
            return fail<std::unique_ptr<OpenGLImage>>(VERNON_HAL_INVALID_ARGUMENT, "the image descriptor is invalid");
        const formats::Format *entry = formats::find(descriptor.format);
        const gl::Enum target = openGLTarget(descriptor.dimension);
        if (!entry || !target)
            return fail<std::unique_ptr<OpenGLImage>>(VERNON_HAL_UNSUPPORTED, "OpenGL cannot create the image");
        auto image = std::unique_ptr<OpenGLImage>(new (std::nothrow) OpenGLImage(access_));
        if (!image)
            return fail<std::unique_ptr<OpenGLImage>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate an image");
        image->target = target;
        const auto internalFormat = static_cast<gl::Int>(entry->glInternalFormat);
        std::lock_guard<std::mutex> guard(access_->mutex);
        state_.makeCurrent();
        gl::Driver &driver = state_.driver;
        driver.genTextures(1, &image->native.name);
        if (!image->native.name)
            return fail<std::unique_ptr<OpenGLImage>>(VERNON_HAL_INTERNAL_ERROR, "glGenTextures failed");
        const bool volume = volumeTarget(descriptor.dimension);
        const bool immutable = driver.texStorage2D && (!volume || driver.texStorage3D);
        if ((descriptor.usage & VERNON_IMAGE_STORAGE) && !immutable)
            return fail<std::unique_ptr<OpenGLImage>>(VERNON_HAL_UNSUPPORTED,
                                                      "OpenGL storage images need immutable texture storage");
        if (auto opened = openScope(state_, "creating an image"))
            return fail<std::unique_ptr<OpenGLImage>>(VERNON_HAL_INTERNAL_ERROR, *opened);
        driver.bindTexture(target, image->native.name);
        const auto levels = static_cast<gl::Size>(descriptor.mip_level_count);
        const auto width = static_cast<gl::Size>(descriptor.extent.width);
        const auto height = static_cast<gl::Size>(descriptor.extent.height);
        if (immutable && volume) {
            driver.texStorage3D(target, levels, static_cast<gl::Enum>(internalFormat), width, height,
                                static_cast<gl::Size>(volumeDepth(descriptor, 0)));
        } else if (immutable) {
            driver.texStorage2D(target, levels, static_cast<gl::Enum>(internalFormat), width, height);
        } else {
            for (uint32_t mip = 0; mip < descriptor.mip_level_count; ++mip) {
                const auto mipWidth = static_cast<gl::Size>(mipExtent(descriptor.extent.width, mip));
                const auto mipHeight = static_cast<gl::Size>(mipExtent(descriptor.extent.height, mip));
                const auto level = static_cast<gl::Int>(mip);
                if (volume)
                    driver.texImage3D(target, level, internalFormat, mipWidth, mipHeight,
                                      static_cast<gl::Size>(volumeDepth(descriptor, mip)), 0, entry->glFormat,
                                      entry->glType, nullptr);
                else if (descriptor.dimension == VERNON_TEXTURE_CUBE)
                    for (uint32_t face = 0; face < 6; ++face)
                        driver.texImage2D(gl::kTextureCubeMapPositiveX + face, level, internalFormat, mipWidth,
                                          mipHeight, 0, entry->glFormat, entry->glType, nullptr);
                else
                    driver.texImage2D(target, level, internalFormat, mipWidth, mipHeight, 0, entry->glFormat,
                                      entry->glType, nullptr);
            }
        }
        driver.bindTexture(target, 0);
        auto closed = closeScope(state_, "creating an image");
        if (closed.isErr())
            return Made{err(std::move(closed).error())};
        return Made{ok(std::move(image))};
    }

    Done<std::unique_ptr<OpenGLView>> makeView(OpenGLImage &image, const VernonImageViewDescriptor &descriptor) {
        using Made = Done<std::unique_ptr<OpenGLView>>;
        if (!validView(image.descriptor, descriptor))
            return fail<std::unique_ptr<OpenGLView>>(VERNON_HAL_INVALID_ARGUMENT, "the view descriptor is invalid");
        const formats::Format *entry = formats::find(descriptor.format);
        const gl::Enum target = openGLTarget(descriptor.dimension);
        if (!entry || !target)
            return fail<std::unique_ptr<OpenGLView>>(VERNON_HAL_UNSUPPORTED, "OpenGL cannot view the image");
        const VernonImageDescriptor &parent = image.descriptor;
        const VernonImageSubresourceRange &range = descriptor.subresources;
        const bool identity = descriptor.format == parent.format && descriptor.dimension == parent.dimension &&
                              range.aspects == formats::aspects(parent.format) && range.base_mip_level == 0 &&
                              range.mip_level_count == parent.mip_level_count && range.base_array_layer == 0 &&
                              range.array_layer_count == parent.array_layer_count;
        auto view = std::unique_ptr<OpenGLView>(new (std::nothrow) OpenGLView(access_));
        if (!view)
            return fail<std::unique_ptr<OpenGLView>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a view");
        view->texture = {image.native.name, target};
        if (!identity) {
            if (state_.embeddedProfile || !state_.driver.textureView)
                return fail<std::unique_ptr<OpenGLView>>(VERNON_HAL_UNSUPPORTED,
                                                         "the OpenGL context views only whole images");
            std::lock_guard<std::mutex> guard(access_->mutex);
            state_.makeCurrent();
            state_.driver.genTextures(1, &view->owned.name);
            if (!view->owned.name)
                return fail<std::unique_ptr<OpenGLView>>(VERNON_HAL_INTERNAL_ERROR, "glGenTextures failed");
            state_.driver.textureView(view->owned.name, target, image.native.name,
                                      static_cast<gl::Enum>(entry->glInternalFormat), range.base_mip_level,
                                      range.mip_level_count, range.base_array_layer, range.array_layer_count);
            view->texture.name = view->owned.name;
        }
        view->entry = {image.descriptor, descriptor, image.serial, &view->texture};
        view->serial = nativeViews().add(&view->entry);
        return Made{ok(std::move(view))};
    }

    // Runs with the device's mutex held and the context current.
    Done<void> readRegion(const OpenGLImage &image, const VernonHalImageRegion &region, void *destination) {
        const VernonImageDescriptor &descriptor = image.descriptor;
        // OpenGL ES reads neither depth nor stencil with glReadPixels.
        if (state_.embeddedProfile && formats::hasDepth(descriptor.format))
            return readDepthStencil(image, region, destination);
        const TransferLayout layout = transferLayout(descriptor.format, region.aspect);
        if (!layout.format || !layout.type)
            return fail(VERNON_HAL_INVALID_ARGUMENT, "the read's aspect cannot be read");
        const size_t bytes = *regionBytes(descriptor, region);
        const auto mip = static_cast<gl::Int>(region.mip_level);
        const auto x = static_cast<gl::Int>(region.offset_x);
        const auto y = static_cast<gl::Int>(region.offset_y);
        const auto width = static_cast<gl::Size>(region.width);
        const auto height = static_cast<gl::Size>(region.height);
        std::string error;
        if (descriptor.dimension == VERNON_TEXTURE_3D) {
            if (!state_.downloadImage3D(image.native, mip, x, y, static_cast<gl::Int>(region.offset_z), width, height,
                                        static_cast<gl::Size>(region.depth), layout.format, layout.type,
                                        bytes / region.depth, destination, error))
                return internal("cannot read the image", error);
            return Done<void>{ok()};
        }
        const gl::Enum target = descriptor.dimension == VERNON_TEXTURE_CUBE
                                    ? gl::kTextureCubeMapPositiveX + region.array_layer
                                    : image.target;
        const auto layer = static_cast<gl::Int>(region.array_layer);
        const bool packed = region.aspect == (VERNON_IMAGE_ASPECT_DEPTH | VERNON_IMAGE_ASPECT_STENCIL);
        const formats::Format &entry = *formats::find(descriptor.format);
        std::optional<EmbeddedRead> through = state_.embeddedProfile && !packed ? embeddedRead(entry) : std::nullopt;
        // Reading through the guaranteed pair converts on the implementation, and some lose the sign of a 16-bit snorm
        // read as RGBA, so the image's own layout is read whenever the implementation names it.
        if (through && state_.implementationColorReadPair(image.native, target, mip, layer) ==
                           std::pair<gl::Enum, gl::Enum>{layout.format, layout.type})
            through.reset();
        if (through) {
            const size_t texels = size_t{region.width} * region.height;
            std::vector<uint8_t> rgba(texels * 4 * through->componentBytes);
            if (!state_.downloadImage2D(image.native, target, mip, layer, x, y, width, height, through->format,
                                        through->type, rgba.data(), error))
                return internal("cannot read the image", error);
            repackEmbeddedRead(entry, *through, rgba.data(), texels, static_cast<uint8_t *>(destination));
            return Done<void>{ok()};
        }
        if (!packed) {
            if (!state_.downloadImage2D(image.native, target, mip, layer, x, y, width, height, layout.format,
                                        layout.type, destination, error))
                return internal("cannot read the image", error);
            return Done<void>{ok()};
        }
        std::vector<uint8_t> native(bytes);
        if (!state_.downloadImage2D(image.native, target, mip, layer, x, y, width, height, layout.format, layout.type,
                                    native.data(), error))
            return internal("cannot read the image", error);
        auto *output = static_cast<uint8_t *>(destination);
        const size_t texels = size_t{region.width} * region.height;
        for (size_t index = 0; index < texels; ++index) {
            const uint8_t *texel = native.data() + index * kPackedDepthStencilBytes;
            float depth{};
            uint32_t stencil{};
            std::memcpy(&depth, texel, sizeof(depth));
            std::memcpy(&stencil, texel + sizeof(depth), sizeof(stencil));
            storePackedDepthStencil(output + index * kPackedDepthStencilBytes, depth,
                                    static_cast<uint8_t>(stencil & 0xff));
        }
        return Done<void>{ok()};
    }

    // A compute pass copies each aspect to a buffer of one 32-bit word a texel, which needs OpenGL ES 3.1.
    Done<void> readDepthStencil(const OpenGLImage &image, const VernonHalImageRegion &region, void *destination) {
        using gl::DepthStencilAspect;
        const bool readDepth = (region.aspect & VERNON_IMAGE_ASPECT_DEPTH) != 0;
        const bool readStencil = (region.aspect & VERNON_IMAGE_ASPECT_STENCIL) != 0;
        const size_t texels = size_t{region.width} * region.height;
        std::string error;
        gl::Buffer words;
        if (!state_.createBuffer(words, texels * sizeof(uint32_t), error))
            return internal("cannot read the image", error);
        std::vector<uint32_t> depth(readDepth ? texels : 0);
        std::vector<uint32_t> stencil(readStencil ? texels : 0);
        const auto copy = [&](DepthStencilAspect aspect, std::vector<uint32_t> &values) {
            return state_.copyDepthStencilToBuffer(
                       image.native, image.target, aspect, static_cast<gl::Int>(region.mip_level),
                       static_cast<gl::Int>(region.array_layer), static_cast<gl::Int>(region.offset_x),
                       static_cast<gl::Int>(region.offset_y), static_cast<gl::Size>(region.width),
                       static_cast<gl::Size>(region.height), words, error) &&
                   state_.downloadBuffer(words, 0, values.data(), values.size() * sizeof(uint32_t), error);
        };
        const bool copied = (!readDepth || copy(DepthStencilAspect::Depth, depth)) &&
                            (!readStencil || copy(DepthStencilAspect::Stencil, stencil));
        state_.destroyBuffer(words);
        if (!copied)
            return state_.supportsCompute() ? internal("cannot read the image", error)
                                            : fail(VERNON_HAL_UNSUPPORTED, "OpenGL ES reads depth with compute");
        auto *output = static_cast<uint8_t *>(destination);
        // The compute pass reads depth as a float, which holds a 16-bit unorm depth exactly.
        const bool unorm16 = formats::find(image.descriptor.format)->dataType == VERNON_HAL_IMAGE_DATA_UINT16;
        for (size_t index = 0; index < texels; ++index) {
            float value{};
            if (readDepth)
                std::memcpy(&value, &depth[index], sizeof(value));
            if (readDepth && readStencil) {
                storePackedDepthStencil(output + index * kPackedDepthStencilBytes, value,
                                        static_cast<uint8_t>(stencil[index]));
            } else if (readDepth && unorm16) {
                const auto unorm = static_cast<uint16_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 65535.0f));
                std::memcpy(output + index * sizeof(unorm), &unorm, sizeof(unorm));
            } else if (readDepth) {
                std::memcpy(output + index * sizeof(value), &value, sizeof(value));
            } else {
                output[index] = static_cast<uint8_t>(stencil[index]);
            }
        }
        return Done<void>{ok()};
    }

    // Runs with the device's mutex held and the window's context current. Surface images hold the image's top row at
    // row 0, and the window's framebuffer holds its bottom row there, so the blit reverses the rows.
    Done<void> blitToWindow(const OpenGLImage &image, uint32_t width, uint32_t height) {
        gl::Driver &driver = state_.driver;
        if (auto opened = openScope(state_, "presenting a surface"))
            return fail(VERNON_HAL_INTERNAL_ERROR, *opened);
        if (!presentFramebuffer_)
            driver.genFramebuffers(1, &presentFramebuffer_);
        if (!presentFramebuffer_)
            return fail(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot create the present framebuffer");
        prepareBlits(state_);
        driver.bindFramebuffer(gl::kReadFramebuffer, presentFramebuffer_);
        driver.framebufferTexture2D(gl::kReadFramebuffer, gl::kColorAttachment0, gl::kTexture2D, image.native.name, 0);
        driver.readBuffer(gl::kColorAttachment0);
        driver.bindFramebuffer(gl::kDrawFramebuffer, 0);
        const auto right = static_cast<gl::Int>(width);
        const auto top = static_cast<gl::Int>(height);
        driver.blitFramebuffer(0, 0, right, top, 0, top, right, 0, gl::kColorBufferBit, gl::kNearest);
        driver.bindFramebuffer(gl::kFramebuffer, 0);
        return closeScope(state_, "presenting a surface");
    }

    Access access_;
    gl::DeviceState &state_;
    // Reads a presented image; made on the first present.
    gl::Uint presentFramebuffer_{};
};

} // namespace

uint32_t openGLTarget(VernonTextureDimension dimension) {
    switch (dimension) {
    case VERNON_TEXTURE_2D:
        return gl::kTexture2D;
    case VERNON_TEXTURE_3D:
        return gl::kTexture3D;
    case VERNON_TEXTURE_CUBE:
        return gl::kTextureCubeMap;
    case VERNON_TEXTURE_2D_ARRAY:
        return gl::kTexture2DArray;
    case VERNON_TEXTURE_CUBE_ARRAY:
        return gl::kTextureCubeMapArray;
    }
    return 0;
}

gl::Buffer *openGLBuffer(uint64_t serial) { return serials().find(serial); }

std::unique_ptr<Resources> createOpenGLResources(std::shared_ptr<opengl::DeviceState> state) {
    if (!state)
        return nullptr;
    return std::unique_ptr<Resources>(new (std::nothrow) OpenGLResources(std::move(state)));
}

std::unique_ptr<DeviceBuffers> createOpenGLDeviceBuffers(std::shared_ptr<opengl::DeviceState> state) {
    if (!state)
        return nullptr;
    return std::unique_ptr<DeviceBuffers>(new (std::nothrow) OpenGLDeviceBuffers(std::move(state)));
}

} // namespace vernon::hal
