#ifndef VERNON_HAL_H
#define VERNON_HAL_H

#include "VernonOpenGLContext.h"
#include "VernonRuntime.h"
#include "VernonTextureTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The device API of specs/runtime/architecture_v2.md section 16.3: devices, the buffers, images, views, samplers and
 * surfaces linked schedules read and write, host transfers ordered on the device's timeline, and its timeline points.
 * Commands are recorded only from linked schedules, so there is no command encoder. A handle is a pointer to the
 * backend's object; a destroyed one is freed once the device completes the submissions made before its destruction.
 *
 * A device created on an OpenGL or OpenGL ES context is bound to the thread that owns the context: every call on the
 * device and its objects must come from that thread. Other devices accept calls from any thread, but an object's
 * calls must not race its destruction.
 */

typedef struct VernonHalDevice VernonHalDevice;
typedef struct VernonHalBuffer VernonHalBuffer;
typedef struct VernonHalImage VernonHalImage;
typedef struct VernonHalImageView VernonHalImageView;
typedef struct VernonHalSampler VernonHalSampler;
typedef struct VernonHalSurface VernonHalSurface;

typedef enum VernonHalStatus {
    VERNON_HAL_OK = 0,
    VERNON_HAL_INVALID_ARGUMENT = 1,
    /* The backend or this device cannot do it. */
    VERNON_HAL_UNSUPPORTED = 2,
    VERNON_HAL_INTERNAL_ERROR = 3,
    VERNON_HAL_RESOURCE_EXHAUSTED = 4,
    /* The surface no longer matches its window; configure it again, then retry. */
    VERNON_HAL_SURFACE_OUT_OF_DATE = 5,
    /* The surface's window is gone; destroy the surface and create it again. */
    VERNON_HAL_SURFACE_LOST = 6
} VernonHalStatus;

/*
 * VALIDATION checks native errors around every operation that can raise one and fails that operation, at the cost of a
 * driver round trip per check; it implies LABELS. Without it a native error fails the next submission waited for.
 * LABELS sends debug groups and markers to the native API, for capture tools. OPENGL_OWNS_DEBUG_OUTPUT lets the device
 * install the context's KHR_debug callback, which then receives every error of the context, including the host's.
 * FORCE_SOFTWARE selects a software adapter where the backend has one.
 */
typedef enum VernonHalDeviceFlagBits {
    VERNON_HAL_DEVICE_VALIDATION = 1u << 0,
    VERNON_HAL_DEVICE_LABELS = 1u << 1,
    VERNON_HAL_DEVICE_OPENGL_OWNS_DEBUG_OUTPUT = 1u << 2,
    VERNON_HAL_DEVICE_FORCE_SOFTWARE = 1u << 3
} VernonHalDeviceFlagBits;

typedef struct VernonHalDeviceDescriptor {
    uint32_t struct_size;
    VernonRuntimeBackend backend;
    uint32_t device_index;
    /* OpenGL and OpenGL ES only, and required there: the caller's context, which the device adopts. */
    const VernonOpenGLContextCallbacks *opengl;
    /* VernonHalDeviceFlagBits */
    uint32_t flags;
    uint32_t reserved[4];
} VernonHalDeviceDescriptor;

typedef enum VernonHalFormatCapabilityBits {
    VERNON_HAL_FORMAT_RENDERABLE = 1u << 0,
    VERNON_HAL_FORMAT_BLENDABLE = 1u << 1,
    VERNON_HAL_FORMAT_FILTERABLE = 1u << 2,
    VERNON_HAL_FORMAT_STORAGE = 1u << 3,
    VERNON_HAL_FORMAT_RESOLVABLE = 1u << 4,
    VERNON_HAL_FORMAT_MULTISAMPLE = 1u << 5,
    VERNON_HAL_FORMAT_COPYABLE = 1u << 6
} VernonHalFormatCapabilityBits;

/* Every buffer may be the source and destination of host transfers. */
typedef enum VernonHalBufferUsageBits {
    VERNON_HAL_BUFFER_UNIFORM = 1u << 0,
    VERNON_HAL_BUFFER_STORAGE = 1u << 1,
    VERNON_HAL_BUFFER_VERTEX = 1u << 2,
    VERNON_HAL_BUFFER_INDEX = 1u << 3,
    VERNON_HAL_BUFFER_INDIRECT = 1u << 4
} VernonHalBufferUsageBits;

/* A buffer's one home. UPLOAD memory is host-written and device-read, READBACK device-written and host-read. */
typedef enum VernonHalMemory {
    VERNON_HAL_MEMORY_DEVICE = 0,
    VERNON_HAL_MEMORY_UPLOAD = 1,
    VERNON_HAL_MEMORY_READBACK = 2
} VernonHalMemory;

typedef struct VernonHalBufferDescriptor {
    uint32_t struct_size;
    uint64_t size;
    /* VernonHalBufferUsageBits */
    uint32_t usage;
    VernonHalMemory memory;
    uint32_t reserved[4];
} VernonHalBufferDescriptor;

typedef struct VernonHalBufferWrite {
    uint64_t offset;
    const void *source;
    uint64_t size;
} VernonHalBufferWrite;

typedef struct VernonHalBufferRead {
    uint64_t offset;
    void *destination;
    uint64_t size;
} VernonHalBufferRead;

/* The host layout of image transfer data, tightly packed rows of texels; values match the format table's columns. */
typedef enum VernonHalImageDataFormat {
    VERNON_HAL_IMAGE_DATA_RED = 0,
    VERNON_HAL_IMAGE_DATA_RG = 1,
    VERNON_HAL_IMAGE_DATA_RGB = 2,
    VERNON_HAL_IMAGE_DATA_BGR = 3,
    VERNON_HAL_IMAGE_DATA_RGBA = 4,
    VERNON_HAL_IMAGE_DATA_BGRA = 5,
    VERNON_HAL_IMAGE_DATA_DEPTH = 6,
    VERNON_HAL_IMAGE_DATA_DEPTH_STENCIL = 7,
    VERNON_HAL_IMAGE_DATA_STENCIL = 8
} VernonHalImageDataFormat;

typedef enum VernonHalImageDataType {
    VERNON_HAL_IMAGE_DATA_UINT8 = 0,
    VERNON_HAL_IMAGE_DATA_FLOAT16 = 1,
    VERNON_HAL_IMAGE_DATA_FLOAT32 = 2,
    VERNON_HAL_IMAGE_DATA_UINT32 = 3
} VernonHalImageDataType;

/* A box of one mip level and layer, the aspects it transfers (VernonImageAspectBits), and the host layout of its
 * data. */
typedef struct VernonHalImageRegion {
    uint32_t mip_level;
    uint32_t array_layer;
    uint32_t aspect;
    uint32_t offset_x;
    uint32_t offset_y;
    uint32_t offset_z;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    VernonHalImageDataFormat data_format;
    VernonHalImageDataType data_type;
} VernonHalImageRegion;

typedef struct VernonHalImageWrite {
    VernonHalImageRegion region;
    const void *source;
} VernonHalImageWrite;

typedef struct VernonHalImageRead {
    VernonHalImageRegion region;
    void *destination;
    size_t size;
} VernonHalImageRead;

typedef enum VernonHalFilter { VERNON_HAL_FILTER_NEAREST = 0, VERNON_HAL_FILTER_LINEAR = 1 } VernonHalFilter;

typedef enum VernonHalAddressMode {
    VERNON_HAL_ADDRESS_REPEAT = 0,
    VERNON_HAL_ADDRESS_CLAMP_TO_EDGE = 1,
    VERNON_HAL_ADDRESS_MIRRORED_REPEAT = 2
} VernonHalAddressMode;

typedef struct VernonHalSamplerDescriptor {
    uint32_t struct_size;
    VernonHalFilter min_filter;
    VernonHalFilter mag_filter;
    VernonHalFilter mip_filter;
    VernonHalAddressMode address_u;
    VernonHalAddressMode address_v;
    VernonHalAddressMode address_w;
    /* In [0, 16]; 1 or less samples isotropically. */
    float max_anisotropy;
    uint32_t reserved[4];
} VernonHalSamplerDescriptor;

typedef enum VernonHalPresentMode {
    /* Waits for vertical blank; every surface supports it. */
    VERNON_HAL_PRESENT_FIFO = 0,
    VERNON_HAL_PRESENT_MAILBOX = 1,
    VERNON_HAL_PRESENT_IMMEDIATE = 2
} VernonHalPresentMode;

/* The member of `native` the device's backend reads: a window on Vulkan and D3D12, a layer on Metal, callbacks on
 * OpenGL and OpenGL ES, which are copied at creation. */
typedef struct VernonHalSurfaceDescriptor {
    uint32_t struct_size;
    union {
        void *win32_window;                         /* HWND */
        void *metal_layer;                          /* CAMetalLayer */
        const VernonOpenGLSurfaceCallbacks *opengl; /* copied at creation */
    } native;
    uint32_t reserved[4];
} VernonHalSurfaceDescriptor;

typedef struct VernonHalSurfaceCapabilities {
    uint32_t struct_size;
    VernonTextureFormat formats[8];
    uint32_t format_count;
    /* 1u << VernonHalPresentMode for each supported mode. */
    uint32_t present_modes;
    /* The VernonImageUsageBits a configuration may request; COLOR_ATTACHMENT is always among them. */
    uint32_t usage;
    uint32_t min_image_count;
    /* 0 when unbounded. */
    uint32_t max_image_count;
    /* The window's size in pixels; UINT32_MAX when the configuration decides it. */
    uint32_t current_width;
    uint32_t current_height;
    uint32_t reserved[4];
} VernonHalSurfaceCapabilities;

typedef struct VernonHalSurfaceConfiguration {
    uint32_t struct_size;
    VernonTextureFormat format;
    /* VernonImageUsageBits; must include COLOR_ATTACHMENT. */
    uint32_t usage;
    uint32_t width;
    uint32_t height;
    VernonHalPresentMode present_mode;
    uint32_t image_count;
    uint32_t reserved[4];
} VernonHalSurfaceConfiguration;

/*
 * A 2D image with one mip level, layer and sample in the configured format, extent and usage, and a view of all of
 * it. The surface owns both, which stay valid until the next present. The contents are undefined: the first use must
 * write the image, so a scope that preserves it and a read of it fail until one does.
 */
typedef struct VernonHalSurfaceImage {
    VernonHalImage *image;
    VernonHalImageView *view;
    /* Still presentable, but configuring again would match the window better. */
    uint32_t suboptimal;
} VernonHalSurfaceImage;

/* The message of this thread's last failed HAL call, valid until its next failed one; empty when none failed. */
VERNON_RUNTIME_CAPI VernonStringView vernonHalGetLastError(void);

VERNON_RUNTIME_CAPI VernonHalStatus vernonHalCreateDevice(const VernonHalDeviceDescriptor *descriptor,
                                                          VernonHalDevice **output);
/* Returns VERNON_HAL_INVALID_ARGUMENT, leaving the device valid, while it has objects or its runtime has schedules. */
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalDestroyDevice(VernonHalDevice *device);
VERNON_RUNTIME_CAPI VernonRuntimeBackend vernonHalDeviceGetBackend(const VernonHalDevice *device);
/* The runtime that links schedules on the device; the device owns it. */
VERNON_RUNTIME_CAPI VernonRuntimeContext *vernonHalDeviceGetRuntime(VernonHalDevice *device);
/* Writes the VernonHalFormatCapabilityBits the device supports for `format`; 0 on a device without images. */
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalDeviceGetFormatCapabilities(VernonHalDevice *device,
                                                                         VernonTextureFormat format,
                                                                         uint32_t *capabilities);
/* Waits until the device completed `point`, a point a schedule call on its runtime returned. */
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalDeviceWait(VernonHalDevice *device, uint64_t point);

/*
 * Host transfers are ordered on the device's timeline: a write takes effect after every submission made before it and
 * before every later one, and its source bytes are free when it returns; a read waits for the submissions made before
 * it, then copies. Ranges must lie inside the buffer.
 */
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalCreateBuffer(VernonHalDevice *device,
                                                          const VernonHalBufferDescriptor *descriptor,
                                                          VernonHalBuffer **output);
VERNON_RUNTIME_CAPI void vernonHalDestroyBuffer(VernonHalBuffer *buffer);
VERNON_RUNTIME_CAPI VernonHalDevice *vernonHalBufferGetDevice(const VernonHalBuffer *buffer);
VERNON_RUNTIME_CAPI uint64_t vernonHalBufferGetSize(const VernonHalBuffer *buffer);
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalBufferWrite(VernonHalBuffer *buffer, const VernonHalBufferWrite *writes,
                                                         size_t write_count);
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalBufferRead(VernonHalBuffer *buffer, const VernonHalBufferRead *reads,
                                                        size_t read_count);
/*
 * The buffer's address where kernels take buffers by address: host memory on the CPU device, a device pointer on
 * CUDA. It does not change while the buffer lives. Other devices return VERNON_HAL_UNSUPPORTED.
 */
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalBufferGetAddress(const VernonHalBuffer *buffer, void **output);

/*
 * Image transfers follow the ordering of buffer transfers. OpenGL and OpenGL ES write the depth and stencil of a
 * depth-stencil image only together, and OpenGL ES reads depth and stencil from OpenGL ES 3.1. The CPU device
 * transfers only in each format's whole-texel layout and generates no mipmaps.
 */
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalCreateImage(VernonHalDevice *device,
                                                         const VernonImageDescriptor *descriptor,
                                                         VernonHalImage **output);
/* Returns VERNON_HAL_INVALID_ARGUMENT, leaving the image valid, while it has views or belongs to a surface. */
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalDestroyImage(VernonHalImage *image);
VERNON_RUNTIME_CAPI VernonHalDevice *vernonHalImageGetDevice(const VernonHalImage *image);
VERNON_RUNTIME_CAPI const VernonImageDescriptor *vernonHalImageGetDescriptor(const VernonHalImage *image);
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalImageWrite(VernonHalImage *image, const VernonHalImageWrite *writes,
                                                        size_t write_count);
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalImageRead(VernonHalImage *image, const VernonHalImageRead *reads,
                                                       size_t read_count);
/* Fills every mip level below the first from the one above it. */
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalImageGenerateMipmaps(VernonHalImage *image);

VERNON_RUNTIME_CAPI VernonHalStatus vernonHalCreateImageView(VernonHalImage *image,
                                                             const VernonImageViewDescriptor *descriptor,
                                                             VernonHalImageView **output);
/* Returns VERNON_HAL_INVALID_ARGUMENT, leaving the view valid, for a surface image's view. */
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalDestroyImageView(VernonHalImageView *view);
VERNON_RUNTIME_CAPI VernonHalImage *vernonHalImageViewGetImage(const VernonHalImageView *view);

VERNON_RUNTIME_CAPI VernonHalStatus vernonHalCreateSampler(VernonHalDevice *device,
                                                           const VernonHalSamplerDescriptor *descriptor,
                                                           VernonHalSampler **output);
VERNON_RUNTIME_CAPI void vernonHalDestroySampler(VernonHalSampler *sampler);
VERNON_RUNTIME_CAPI VernonHalDevice *vernonHalSamplerGetDevice(const VernonHalSampler *sampler);

/*
 * A surface presents a host window: the host owns the window, and the surface owns the images it presents. It has at
 * most one acquired image; acquire after acquire, present without acquire, and configure or destroy while an image is
 * acquired return VERNON_HAL_INVALID_ARGUMENT. Present is ordered after every submission made before it. A device
 * that cannot present returns VERNON_HAL_UNSUPPORTED from creation.
 */
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalCreateSurface(VernonHalDevice *device,
                                                           const VernonHalSurfaceDescriptor *descriptor,
                                                           VernonHalSurface **output);
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalDestroySurface(VernonHalSurface *surface);
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalSurfaceGetCapabilities(VernonHalSurface *surface,
                                                                    VernonHalSurfaceCapabilities *output);
/* Replaces the surface's images. A zero extent, which a minimized window reports, returns INVALID_ARGUMENT. */
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalSurfaceConfigure(VernonHalSurface *surface,
                                                              const VernonHalSurfaceConfiguration *configuration);
/* May block until an image is available; a failed acquire leaves no image acquired. */
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalSurfaceAcquire(VernonHalSurface *surface, VernonHalSurfaceImage *output);
/* Releases the acquired image even when it fails. */
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalSurfacePresent(VernonHalSurface *surface);

/*
 * The references a schedule argument names a resource by: a TensorView's resource, an index binding, a draw stream's
 * records and counts, an image argument, an attachment, a sampler. A reference stays valid while its resource lives,
 * and a surface image's view only until the present. The CPU device's arguments are host tensors, so it returns
 * VERNON_HAL_UNSUPPORTED.
 */
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalReferenceBuffer(const VernonHalBuffer *buffer, uint64_t offset,
                                                             uint64_t size,
                                                             VernonRuntimeProviderResourceReference *output);
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalReferenceImageView(const VernonHalImageView *view,
                                                                VernonRuntimeProviderResourceReference *output);
VERNON_RUNTIME_CAPI VernonHalStatus vernonHalReferenceSampler(const VernonHalSampler *sampler,
                                                              VernonRuntimeProviderResourceReference *output);

#ifdef __cplusplus
}
#endif

#endif
