#ifndef VERNON_RHI_H
#define VERNON_RHI_H

#include "VernonCommon.h"
#include "VernonOpenGLContext.h"

#if defined(VERNON_RHI_STATIC)
#define VERNON_RHI_CAPI
#elif defined(_WIN32) && defined(VERNON_RHI_BUILD)
#define VERNON_RHI_CAPI __declspec(dllexport)
#elif defined(_WIN32)
#define VERNON_RHI_CAPI __declspec(dllimport)
#else
#define VERNON_RHI_CAPI
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum { VERNON_RHI_INVALID_HANDLE_INDEX = UINT32_MAX };

typedef enum VernonRhiStatus {
    VERNON_RHI_STATUS_OK = 0,
    VERNON_RHI_STATUS_INVALID_ARGUMENT = 1,
    VERNON_RHI_STATUS_UNSUPPORTED = 2,
    VERNON_RHI_STATUS_INTERNAL_ERROR = 3
} VernonRhiStatus;

#define VERNON_RHI_DECLARE_HANDLE(Name)                                                                                \
    typedef struct Name {                                                                                              \
        uint32_t index;                                                                                                \
        uint32_t generation;                                                                                           \
    } Name

VERNON_RHI_DECLARE_HANDLE(VernonRhiAdapter);
VERNON_RHI_DECLARE_HANDLE(VernonRhiDevice);
VERNON_RHI_DECLARE_HANDLE(VernonRhiQueue);
VERNON_RHI_DECLARE_HANDLE(VernonRhiBuffer);
VERNON_RHI_DECLARE_HANDLE(VernonRhiImage);
VERNON_RHI_DECLARE_HANDLE(VernonRhiImageView);
VERNON_RHI_DECLARE_HANDLE(VernonRhiSampler);
VERNON_RHI_DECLARE_HANDLE(VernonRhiShaderModule);
VERNON_RHI_DECLARE_HANDLE(VernonRhiPipelineLayout);
VERNON_RHI_DECLARE_HANDLE(VernonRhiComputePipeline);
VERNON_RHI_DECLARE_HANDLE(VernonRhiGraphicsPipeline);
VERNON_RHI_DECLARE_HANDLE(VernonRhiBindingSet);
VERNON_RHI_DECLARE_HANDLE(VernonRhiCommandEncoder);
VERNON_RHI_DECLARE_HANDLE(VernonRhiCompletion);
VERNON_RHI_DECLARE_HANDLE(VernonRhiNativeDescriptorRange);

#undef VERNON_RHI_DECLARE_HANDLE

typedef enum VernonRhiCapabilityBits {
    VERNON_RHI_CAPABILITY_COMPUTE = 1u << 0,
    VERNON_RHI_CAPABILITY_GRAPHICS = 1u << 1,
    VERNON_RHI_CAPABILITY_NATIVE_INTEROP = 1u << 2
} VernonRhiCapabilityBits;

typedef enum VernonRhiBackend {
    VERNON_RHI_BACKEND_CUDA = 0,
    VERNON_RHI_BACKEND_VULKAN = 1,
    VERNON_RHI_BACKEND_DIRECTX12 = 2,
    VERNON_RHI_BACKEND_OPENGL = 3,
    VERNON_RHI_BACKEND_OPENGL_ES = 4,
    VERNON_RHI_BACKEND_METAL = 5
} VernonRhiBackend;

typedef enum VernonRhiOwnedDeviceFlagBits {
    VERNON_RHI_OWNED_DEVICE_FORCE_SOFTWARE = 1u << 0
} VernonRhiOwnedDeviceFlagBits;

typedef enum VernonRhiQueueCapabilityBits {
    VERNON_RHI_QUEUE_TRANSFER = 1u << 0,
    VERNON_RHI_QUEUE_COMPUTE = 1u << 1,
    VERNON_RHI_QUEUE_GRAPHICS = 1u << 2
} VernonRhiQueueCapabilityBits;

typedef enum VernonRhiBufferUsageBits {
    VERNON_RHI_BUFFER_TRANSFER_SOURCE = 1u << 0,
    VERNON_RHI_BUFFER_TRANSFER_DESTINATION = 1u << 1,
    VERNON_RHI_BUFFER_UNIFORM = 1u << 2,
    VERNON_RHI_BUFFER_STORAGE = 1u << 3,
    VERNON_RHI_BUFFER_VERTEX = 1u << 4,
    VERNON_RHI_BUFFER_INDEX = 1u << 5,
    VERNON_RHI_BUFFER_INDIRECT = 1u << 6
} VernonRhiBufferUsageBits;

typedef enum VernonRhiImageUsageBits {
    VERNON_RHI_IMAGE_TRANSFER_SOURCE = 1u << 0,
    VERNON_RHI_IMAGE_TRANSFER_DESTINATION = 1u << 1,
    VERNON_RHI_IMAGE_SAMPLED = 1u << 2,
    VERNON_RHI_IMAGE_STORAGE = 1u << 3,
    VERNON_RHI_IMAGE_COLOR_ATTACHMENT = 1u << 4,
    VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT = 1u << 5
} VernonRhiImageUsageBits;

typedef enum VernonRhiAccessBits {
    VERNON_RHI_ACCESS_NONE = 0,
    VERNON_RHI_ACCESS_TRANSFER_READ = 1u << 0,
    VERNON_RHI_ACCESS_TRANSFER_WRITE = 1u << 1,
    VERNON_RHI_ACCESS_SHADER_READ = 1u << 2,
    VERNON_RHI_ACCESS_SHADER_WRITE = 1u << 3,
    VERNON_RHI_ACCESS_COLOR_READ = 1u << 4,
    VERNON_RHI_ACCESS_COLOR_WRITE = 1u << 5,
    VERNON_RHI_ACCESS_DEPTH_STENCIL_READ = 1u << 6,
    VERNON_RHI_ACCESS_DEPTH_STENCIL_WRITE = 1u << 7,
    VERNON_RHI_ACCESS_VERTEX_READ = 1u << 8,
    VERNON_RHI_ACCESS_INDEX_READ = 1u << 9,
    VERNON_RHI_ACCESS_INDIRECT_READ = 1u << 10,
    VERNON_RHI_ACCESS_HOST_READ = 1u << 11,
    VERNON_RHI_ACCESS_HOST_WRITE = 1u << 12
} VernonRhiAccessBits;

typedef enum VernonRhiResourceState {
    VERNON_RHI_STATE_UNDEFINED = 0,
    VERNON_RHI_STATE_COMMON = 1,
    VERNON_RHI_STATE_TRANSFER_SOURCE = 2,
    VERNON_RHI_STATE_TRANSFER_DESTINATION = 3,
    VERNON_RHI_STATE_SHADER_READ = 4,
    VERNON_RHI_STATE_SHADER_WRITE = 5,
    VERNON_RHI_STATE_COLOR_ATTACHMENT = 6,
    VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT = 7,
    VERNON_RHI_STATE_PRESENT = 8
} VernonRhiResourceState;

typedef enum VernonRhiMemoryClass {
    VERNON_RHI_MEMORY_DEVICE = 0,
    VERNON_RHI_MEMORY_UPLOAD = 1,
    VERNON_RHI_MEMORY_READBACK = 2
} VernonRhiMemoryClass;

typedef enum VernonRhiShaderStageBits {
    VERNON_RHI_STAGE_COMPUTE = 1u << 0,
    VERNON_RHI_STAGE_VERTEX = 1u << 1,
    VERNON_RHI_STAGE_FRAGMENT = 1u << 2
} VernonRhiShaderStageBits;

typedef enum VernonRhiFormat {
    VERNON_RHI_FORMAT_UNDEFINED = 0,
    VERNON_RHI_FORMAT_R8_UNORM = 1,
    VERNON_RHI_FORMAT_RG8_UNORM = 2,
    VERNON_RHI_FORMAT_RGBA8_UNORM = 3,
    VERNON_RHI_FORMAT_RGBA8_SRGB = 4,
    VERNON_RHI_FORMAT_R16_FLOAT = 5,
    VERNON_RHI_FORMAT_RGBA16_FLOAT = 6,
    VERNON_RHI_FORMAT_R32_FLOAT = 7,
    VERNON_RHI_FORMAT_RGBA32_FLOAT = 8,
    VERNON_RHI_FORMAT_D32_FLOAT = 9,
    VERNON_RHI_FORMAT_RGB8_UNORM = 10,
    VERNON_RHI_FORMAT_RG32_FLOAT = 11,
    VERNON_RHI_FORMAT_RGB32_FLOAT = 12,
    VERNON_RHI_FORMAT_R11G11B10_FLOAT = 13
} VernonRhiFormat;

typedef enum VernonRhiImageDimension {
    VERNON_RHI_IMAGE_2D = 0,
    VERNON_RHI_IMAGE_3D = 1,
    VERNON_RHI_IMAGE_CUBE = 2
} VernonRhiImageDimension;

typedef enum VernonRhiImageDataType {
    VERNON_RHI_IMAGE_DATA_UINT8 = 0,
    VERNON_RHI_IMAGE_DATA_FLOAT32 = 1
} VernonRhiImageDataType;

typedef enum VernonRhiImageDataFormat {
    VERNON_RHI_IMAGE_DATA_RED = 0,
    VERNON_RHI_IMAGE_DATA_RG = 1,
    VERNON_RHI_IMAGE_DATA_RGB = 2,
    VERNON_RHI_IMAGE_DATA_BGR = 3,
    VERNON_RHI_IMAGE_DATA_RGBA = 4,
    VERNON_RHI_IMAGE_DATA_BGRA = 5,
    VERNON_RHI_IMAGE_DATA_DEPTH = 6,
    VERNON_RHI_IMAGE_DATA_DEPTH_STENCIL = 7
} VernonRhiImageDataFormat;

typedef enum VernonRhiSamplerFilter {
    VERNON_RHI_FILTER_NEAREST = 0,
    VERNON_RHI_FILTER_LINEAR = 1,
    VERNON_RHI_FILTER_NEAREST_MIPMAP_NEAREST = 2,
    VERNON_RHI_FILTER_LINEAR_MIPMAP_LINEAR = 3,
    VERNON_RHI_FILTER_LINEAR_MIPMAP_NEAREST = 4,
    VERNON_RHI_FILTER_NEAREST_MIPMAP_LINEAR = 5
} VernonRhiSamplerFilter;

typedef enum VernonRhiSamplerAddressMode {
    VERNON_RHI_ADDRESS_REPEAT = 0,
    VERNON_RHI_ADDRESS_CLAMP_TO_EDGE = 1,
    VERNON_RHI_ADDRESS_MIRRORED_REPEAT = 2
} VernonRhiSamplerAddressMode;

typedef enum VernonRhiLoadOperation {
    VERNON_RHI_LOAD_CLEAR = 0,
    VERNON_RHI_LOAD_PRESERVE = 1,
    VERNON_RHI_LOAD_DISCARD = 2
} VernonRhiLoadOperation;

typedef enum VernonRhiStoreOperation {
    VERNON_RHI_STORE_PRESERVE = 0,
    VERNON_RHI_STORE_DISCARD = 1
} VernonRhiStoreOperation;

typedef enum VernonRhiCompareOperation {
    VERNON_RHI_COMPARE_NEVER = 0,
    VERNON_RHI_COMPARE_LESS = 1,
    VERNON_RHI_COMPARE_EQUAL = 2,
    VERNON_RHI_COMPARE_LESS_EQUAL = 3,
    VERNON_RHI_COMPARE_GREATER = 4,
    VERNON_RHI_COMPARE_NOT_EQUAL = 5,
    VERNON_RHI_COMPARE_GREATER_EQUAL = 6,
    VERNON_RHI_COMPARE_ALWAYS = 7
} VernonRhiCompareOperation;

typedef enum VernonRhiCullMode {
    VERNON_RHI_CULL_NONE = 0,
    VERNON_RHI_CULL_FRONT = 1,
    VERNON_RHI_CULL_BACK = 2
} VernonRhiCullMode;

typedef enum VernonRhiFrontFace {
    VERNON_RHI_FRONT_FACE_COUNTER_CLOCKWISE = 0,
    VERNON_RHI_FRONT_FACE_CLOCKWISE = 1
} VernonRhiFrontFace;

typedef enum VernonRhiBlendFactor {
    VERNON_RHI_BLEND_ZERO = 0,
    VERNON_RHI_BLEND_ONE = 1,
    VERNON_RHI_BLEND_SOURCE_COLOR = 2,
    VERNON_RHI_BLEND_ONE_MINUS_SOURCE_COLOR = 3,
    VERNON_RHI_BLEND_DESTINATION_COLOR = 4,
    VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_COLOR = 5,
    VERNON_RHI_BLEND_SOURCE_ALPHA = 6,
    VERNON_RHI_BLEND_ONE_MINUS_SOURCE_ALPHA = 7,
    VERNON_RHI_BLEND_DESTINATION_ALPHA = 8,
    VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA = 9
} VernonRhiBlendFactor;

typedef enum VernonRhiBlendOperation {
    VERNON_RHI_BLEND_ADD = 0,
    VERNON_RHI_BLEND_SUBTRACT = 1,
    VERNON_RHI_BLEND_REVERSE_SUBTRACT = 2,
    VERNON_RHI_BLEND_MINIMUM = 3,
    VERNON_RHI_BLEND_MAXIMUM = 4
} VernonRhiBlendOperation;

typedef enum VernonRhiColorWriteBits {
    VERNON_RHI_COLOR_WRITE_RED = 1u << 0,
    VERNON_RHI_COLOR_WRITE_GREEN = 1u << 1,
    VERNON_RHI_COLOR_WRITE_BLUE = 1u << 2,
    VERNON_RHI_COLOR_WRITE_ALPHA = 1u << 3,
    VERNON_RHI_COLOR_WRITE_ALL = (1u << 4) - 1
} VernonRhiColorWriteBits;

typedef enum VernonRhiAttachmentAspectBits {
    VERNON_RHI_ATTACHMENT_DEPTH = 1u << 0,
    VERNON_RHI_ATTACHMENT_STENCIL = 1u << 1
} VernonRhiAttachmentAspectBits;

typedef struct VernonRhiDeviceIdentity {
    uint64_t adapter_id;
    uint64_t device_id;
    uint64_t driver_version;
} VernonRhiDeviceIdentity;

typedef struct VernonRhiAdapterDescriptor {
    uint32_t struct_size;
    uint32_t required_capabilities;
    uint32_t preferred_adapter_index;
    uint32_t reserved[4];
} VernonRhiAdapterDescriptor;

typedef struct VernonRhiDeviceDescriptor {
    uint32_t struct_size;
    uint32_t required_capabilities;
    uint32_t adapter_index;
    uint32_t device_index;
    uint32_t reserved[4];
} VernonRhiDeviceDescriptor;

typedef struct VernonRhiOwnedDeviceDescriptor {
    uint32_t struct_size;
    VernonRhiBackend backend;
    uint32_t device_index;
    const VernonOpenGLContextCallbacks *opengl_callbacks;
    uint32_t flags;
    uint32_t reserved[3];
} VernonRhiOwnedDeviceDescriptor;

typedef struct VernonRhiQueueDescriptor {
    uint32_t struct_size;
    uint32_t required_capabilities;
    uint32_t priority;
    uint32_t reserved[4];
} VernonRhiQueueDescriptor;

typedef struct VernonRhiBufferDescriptor {
    uint32_t struct_size;
    uint64_t size;
    uint64_t alignment;
    uint32_t usage;
    VernonRhiMemoryClass memory_class;
    uint32_t reserved[4];
} VernonRhiBufferDescriptor;

typedef struct VernonRhiImageDescriptor {
    uint32_t struct_size;
    VernonRhiImageDimension dimension;
    VernonRhiFormat format;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t mip_levels;
    uint32_t array_layers;
    uint32_t sample_count;
    uint32_t usage;
    uint32_t reserved[4];
} VernonRhiImageDescriptor;

typedef struct VernonRhiImageViewDescriptor {
    uint32_t struct_size;
    VernonRhiImage image;
    VernonRhiFormat format;
    uint32_t base_mip_level;
    uint32_t mip_level_count;
    uint32_t base_array_layer;
    uint32_t array_layer_count;
    uint32_t reserved[4];
} VernonRhiImageViewDescriptor;

typedef struct VernonRhiSamplerDescriptor {
    uint32_t struct_size;
    uint32_t min_filter;
    uint32_t mag_filter;
    uint32_t mip_filter;
    uint32_t address_u;
    uint32_t address_v;
    uint32_t address_w;
    float max_anisotropy;
    uint32_t reserved[4];
} VernonRhiSamplerDescriptor;

typedef struct VernonRhiImageUploadDescriptor {
    uint32_t struct_size;
    uint32_t mip_level;
    uint32_t array_layer;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    VernonRhiImageDataFormat source_format;
    VernonRhiImageDataType source_type;
    const void *data;
    uint32_t reserved[4];
} VernonRhiImageUploadDescriptor;

typedef struct VernonRhiShaderModuleDescriptor {
    uint32_t struct_size;
    uint32_t stage;
    VernonStringView format;
    const void *data;
    size_t size;
    VernonStringView entry;
    uint32_t reserved[4];
} VernonRhiShaderModuleDescriptor;

typedef enum VernonRhiBindingKind {
    VERNON_RHI_BINDING_UNIFORM_BUFFER = 0,
    VERNON_RHI_BINDING_STORAGE_BUFFER = 1,
    VERNON_RHI_BINDING_SAMPLED_IMAGE = 2,
    VERNON_RHI_BINDING_STORAGE_IMAGE = 3,
    VERNON_RHI_BINDING_SAMPLER = 4
} VernonRhiBindingKind;

typedef struct VernonRhiBindingLayoutEntry {
    uint32_t set;
    uint32_t binding;
    VernonRhiBindingKind kind;
    uint32_t stage_mask;
    uint32_t access;
    uint32_t array_count;
} VernonRhiBindingLayoutEntry;

typedef struct VernonRhiPipelineLayoutDescriptor {
    uint32_t struct_size;
    const VernonRhiBindingLayoutEntry *bindings;
    size_t binding_count;
    uint32_t push_constant_size;
    uint32_t reserved[4];
} VernonRhiPipelineLayoutDescriptor;

typedef struct VernonRhiComputePipelineDescriptor {
    uint32_t struct_size;
    VernonRhiPipelineLayout layout;
    VernonRhiShaderModule shader;
    uint32_t reserved[4];
} VernonRhiComputePipelineDescriptor;

typedef struct VernonRhiRasterizationState {
    VernonRhiCullMode cull_mode;
    VernonRhiFrontFace front_face;
    uint32_t depth_clamp;
    uint32_t depth_bias_enabled;
    float depth_bias_constant;
    float depth_bias_slope;
} VernonRhiRasterizationState;

typedef struct VernonRhiDepthStencilState {
    uint32_t depth_test;
    uint32_t depth_write;
    VernonRhiCompareOperation depth_compare;
    uint32_t stencil_test;
} VernonRhiDepthStencilState;

typedef struct VernonRhiColorBlendState {
    uint32_t blend_enabled;
    VernonRhiBlendFactor source_color_factor;
    VernonRhiBlendFactor destination_color_factor;
    VernonRhiBlendOperation color_operation;
    VernonRhiBlendFactor source_alpha_factor;
    VernonRhiBlendFactor destination_alpha_factor;
    VernonRhiBlendOperation alpha_operation;
    uint32_t write_mask;
} VernonRhiColorBlendState;

typedef struct VernonRhiGraphicsPipelineDescriptor {
    uint32_t struct_size;
    VernonRhiPipelineLayout layout;
    VernonRhiShaderModule vertex_shader;
    VernonRhiShaderModule fragment_shader;
    uint32_t topology;
    const VernonRhiFormat *color_formats;
    size_t color_format_count;
    VernonRhiFormat depth_stencil_format;
    uint32_t sample_count;
    VernonRhiRasterizationState rasterization;
    VernonRhiDepthStencilState depth_stencil;
    const VernonRhiColorBlendState *color_blends;
    size_t color_blend_count;
    uint32_t reserved[4];
} VernonRhiGraphicsPipelineDescriptor;

typedef struct VernonRhiBindingValue {
    uint32_t set;
    uint32_t binding;
    VernonRhiBindingKind kind;
    uint64_t offset;
    uint64_t size;
    union {
        VernonRhiBuffer buffer;
        VernonRhiImageView image_view;
        VernonRhiSampler sampler;
    };
} VernonRhiBindingValue;

typedef struct VernonRhiBindingSetDescriptor {
    uint32_t struct_size;
    VernonRhiPipelineLayout layout;
    const VernonRhiBindingValue *values;
    size_t value_count;
    uint32_t reserved[4];
} VernonRhiBindingSetDescriptor;

typedef struct VernonRhiColorAttachment {
    VernonRhiImageView view;
    uint32_t location;
    VernonRhiResourceState initial_state;
    VernonRhiResourceState final_state;
    VernonRhiLoadOperation load_operation;
    VernonRhiStoreOperation store_operation;
    float clear_color[4];
} VernonRhiColorAttachment;

typedef struct VernonRhiDepthStencilAttachment {
    VernonRhiImageView view;
    VernonRhiResourceState initial_state;
    VernonRhiResourceState final_state;
    VernonRhiLoadOperation depth_load_operation;
    VernonRhiStoreOperation depth_store_operation;
    float clear_depth;
    VernonRhiLoadOperation stencil_load_operation;
    VernonRhiStoreOperation stencil_store_operation;
    uint32_t clear_stencil;
    uint32_t read_only_depth;
    uint32_t read_only_stencil;
} VernonRhiDepthStencilAttachment;

typedef struct VernonRhiRenderingDescriptor {
    uint32_t struct_size;
    const VernonRhiColorAttachment *color_attachments;
    size_t color_attachment_count;
    const VernonRhiDepthStencilAttachment *depth_stencil_attachment;
    uint32_t offset_x;
    uint32_t offset_y;
    uint32_t width;
    uint32_t height;
    uint32_t layers;
    uint32_t view_mask;
    uint32_t reserved[4];
} VernonRhiRenderingDescriptor;

typedef struct VernonRhiCommandEncoderDescriptor {
    uint32_t struct_size;
    VernonRhiQueue queue;
    uint32_t required_capabilities;
    uint32_t reserved[4];
} VernonRhiCommandEncoderDescriptor;

typedef struct VernonRhiCompletionDescriptor {
    uint32_t struct_size;
    uint64_t initial_value;
    uint32_t host_visible;
    uint32_t reserved[4];
} VernonRhiCompletionDescriptor;

typedef struct VernonRhiBarrier {
    uint32_t struct_size;
    uint32_t source_stage_mask;
    uint32_t destination_stage_mask;
    uint32_t source_access;
    uint32_t destination_access;
    VernonRhiResourceState old_state;
    VernonRhiResourceState new_state;
    union {
        VernonRhiBuffer buffer;
        VernonRhiImage image;
    };
    uint32_t is_image;
    uint32_t reserved[4];
} VernonRhiBarrier;

typedef struct VernonRhiCommandEncoderStats {
    uint32_t rendering_scope_count;
    uint32_t barrier_count;
    uint32_t clear_count;
    uint32_t draw_count;
    uint32_t dispatch_count;
    uint32_t submission_count;
} VernonRhiCommandEncoderStats;

typedef enum VernonRhiNativeDescriptorHeapType {
    VERNON_RHI_NATIVE_DESCRIPTOR_RESOURCE = 0,
    VERNON_RHI_NATIVE_DESCRIPTOR_SAMPLER = 1,
    VERNON_RHI_NATIVE_DESCRIPTOR_RENDER_TARGET = 2,
    VERNON_RHI_NATIVE_DESCRIPTOR_DEPTH_STENCIL = 3
} VernonRhiNativeDescriptorHeapType;

typedef struct VernonRhiDirectX12BorrowedDeviceDescriptor {
    uint32_t struct_size;
    void *device;
    void *queue;
    void *command_list;
    uint32_t queue_capabilities;
    uint32_t reserved[4];
} VernonRhiDirectX12BorrowedDeviceDescriptor;

typedef struct VernonRhiDirectX12BorrowedBufferDescriptor {
    uint32_t struct_size;
    void *resource;
    uint64_t size;
    uint32_t usage;
    VernonRhiResourceState state;
    uint32_t reserved[4];
} VernonRhiDirectX12BorrowedBufferDescriptor;

typedef struct VernonRhiDirectX12BorrowedImageDescriptor {
    uint32_t struct_size;
    void *resource;
    VernonRhiImageDescriptor image;
    VernonRhiResourceState state;
    uint32_t reserved[4];
} VernonRhiDirectX12BorrowedImageDescriptor;

typedef struct VernonRhiDirectX12BorrowedDescriptorRangeDescriptor {
    uint32_t struct_size;
    void *heap;
    uint64_t cpu_handle;
    uint64_t gpu_handle;
    uint32_t descriptor_count;
    VernonRhiNativeDescriptorHeapType heap_type;
    uint32_t reserved[4];
} VernonRhiDirectX12BorrowedDescriptorRangeDescriptor;

typedef struct VernonRhiVulkanBorrowedDeviceDescriptor {
    uint32_t struct_size;
    void *instance;
    void *physical_device;
    void *device;
    void *queue;
    void *command_buffer;
    uint32_t queue_family_index;
    uint32_t queue_capabilities;
    uint32_t reserved[4];
} VernonRhiVulkanBorrowedDeviceDescriptor;

typedef struct VernonRhiVulkanBorrowedBufferDescriptor {
    uint32_t struct_size;
    uint64_t buffer;
    uint64_t size;
    uint32_t usage;
    VernonRhiResourceState state;
    uint32_t reserved[4];
} VernonRhiVulkanBorrowedBufferDescriptor;

typedef struct VernonRhiVulkanBorrowedImageDescriptor {
    uint32_t struct_size;
    uint64_t image;
    VernonRhiImageDescriptor descriptor;
    VernonRhiResourceState state;
    uint32_t reserved[4];
} VernonRhiVulkanBorrowedImageDescriptor;

typedef struct VernonRhiVulkanBorrowedImageViewDescriptor {
    uint32_t struct_size;
    uint64_t image_view;
    VernonRhiImageViewDescriptor descriptor;
    uint32_t reserved[4];
} VernonRhiVulkanBorrowedImageViewDescriptor;

VERNON_RHI_CAPI uint32_t vernonRhiGetApiVersion(void);
VERNON_RHI_CAPI VernonRhiDevice vernonRhiCreateDevice(const VernonRhiOwnedDeviceDescriptor *descriptor);
VERNON_RHI_CAPI VernonRhiDevice vernonRhiCreateOpenGLDevice(const VernonOpenGLContextCallbacks *callbacks,
                                                            uint32_t embedded_profile);
VERNON_RHI_CAPI void vernonRhiDestroyDevice(VernonRhiDevice device);
VERNON_RHI_CAPI VernonStringView vernonRhiDeviceGetLastError(VernonRhiDevice device);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceSynchronize(VernonRhiDevice device);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceCreateCommandEncoder(VernonRhiDevice device,
                                                                    const VernonRhiCommandEncoderDescriptor *descriptor,
                                                                    VernonRhiCommandEncoder *output);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceDestroyCommandEncoder(VernonRhiDevice device,
                                                                     VernonRhiCommandEncoder encoder);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiCommandEncoderBarrier(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                                               const VernonRhiBarrier *barriers, size_t barrier_count);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiCommandEncoderBeginRendering(VernonRhiDevice device,
                                                                      VernonRhiCommandEncoder encoder,
                                                                      const VernonRhiRenderingDescriptor *descriptor);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiCommandEncoderEndRendering(VernonRhiDevice device,
                                                                    VernonRhiCommandEncoder encoder);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiCommandEncoderClearColorAttachment(VernonRhiDevice device,
                                                                            VernonRhiCommandEncoder encoder,
                                                                            uint32_t location,
                                                                            const float clear_color[4]);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiCommandEncoderClearDepthStencilAttachment(VernonRhiDevice device,
                                                                                   VernonRhiCommandEncoder encoder,
                                                                                   float clear_depth,
                                                                                   uint32_t clear_stencil,
                                                                                   uint32_t aspects);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiCommandEncoderFinish(VernonRhiDevice device, VernonRhiCommandEncoder encoder);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceSubmit(VernonRhiDevice device, VernonRhiCommandEncoder encoder);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiCommandEncoderGetStats(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                                                VernonRhiCommandEncoderStats *output);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceCreateBuffer(VernonRhiDevice device,
                                                            const VernonRhiBufferDescriptor *descriptor,
                                                            VernonRhiBuffer *output);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceUploadBuffer(VernonRhiDevice device, VernonRhiBuffer buffer,
                                                            uint64_t offset, const void *source, uint64_t size);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceDownloadBuffer(VernonRhiDevice device, VernonRhiBuffer buffer,
                                                              uint64_t offset, void *destination, uint64_t size);
VERNON_RHI_CAPI uint32_t vernonRhiDeviceIsBufferValid(VernonRhiDevice device, VernonRhiBuffer buffer);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceCreateImage(VernonRhiDevice device,
                                                           const VernonRhiImageDescriptor *descriptor,
                                                           VernonRhiImage *output);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceSetImageSampler(VernonRhiDevice device, VernonRhiImage image,
                                                               const VernonRhiSamplerDescriptor *descriptor);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceUploadImage(VernonRhiDevice device, VernonRhiImage image,
                                                           const VernonRhiImageUploadDescriptor *uploads,
                                                           size_t upload_count);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceDownloadImage(VernonRhiDevice device, VernonRhiImage image,
                                                             void *destination, size_t size);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceGenerateImageMipmaps(VernonRhiDevice device, VernonRhiImage image);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceBindImage(VernonRhiDevice device, VernonRhiImage image,
                                                         uint32_t texture_unit);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceDestroyImage(VernonRhiDevice device, VernonRhiImage image);
VERNON_RHI_CAPI uint32_t vernonRhiDeviceIsImageValid(VernonRhiDevice device, VernonRhiImage image);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceCreateSampler(VernonRhiDevice device,
                                                             const VernonRhiSamplerDescriptor *descriptor,
                                                             VernonRhiSampler *output);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceDestroySampler(VernonRhiDevice device, VernonRhiSampler sampler);
VERNON_RHI_CAPI uint32_t vernonRhiDeviceIsSamplerValid(VernonRhiDevice device, VernonRhiSampler sampler);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceGetImageNativeHandle(VernonRhiDevice device, VernonRhiImage image,
                                                                    uint64_t *output);
VERNON_RHI_CAPI VernonRhiDevice
vernonRhiCreateBorrowedDirectX12Device(const VernonRhiDirectX12BorrowedDeviceDescriptor *descriptor);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDirectX12DeviceGetBorrowedQueue(VernonRhiDevice device, void **output);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDirectX12DeviceGetBorrowedCommandList(VernonRhiDevice device, void **output);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDirectX12DeviceImportBorrowedBuffer(
    VernonRhiDevice device, const VernonRhiDirectX12BorrowedBufferDescriptor *descriptor, VernonRhiBuffer *output);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDirectX12DeviceImportBorrowedImage(
    VernonRhiDevice device, const VernonRhiDirectX12BorrowedImageDescriptor *descriptor, VernonRhiImage *output);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDirectX12DeviceImportBorrowedDescriptorRange(
    VernonRhiDevice device, const VernonRhiDirectX12BorrowedDescriptorRangeDescriptor *descriptor,
    VernonRhiNativeDescriptorRange *output);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceDestroyBuffer(VernonRhiDevice device, VernonRhiBuffer buffer);
VERNON_RHI_CAPI VernonRhiStatus
vernonRhiDeviceDestroyNativeDescriptorRange(VernonRhiDevice device, VernonRhiNativeDescriptorRange descriptor_range);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceGetBufferNativeHandle(VernonRhiDevice device, VernonRhiBuffer buffer,
                                                                     void **output);
VERNON_RHI_CAPI VernonRhiStatus
vernonRhiDeviceGetNativeDescriptorRange(VernonRhiDevice device, VernonRhiNativeDescriptorRange descriptor_range,
                                        VernonRhiDirectX12BorrowedDescriptorRangeDescriptor *output);
VERNON_RHI_CAPI VernonRhiDevice
vernonRhiCreateBorrowedVulkanDevice(const VernonRhiVulkanBorrowedDeviceDescriptor *descriptor);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiVulkanDeviceGetBorrowedQueue(VernonRhiDevice device, void **output);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiVulkanDeviceGetBorrowedCommandBuffer(VernonRhiDevice device, void **output);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiVulkanDeviceImportBorrowedBuffer(
    VernonRhiDevice device, const VernonRhiVulkanBorrowedBufferDescriptor *descriptor, VernonRhiBuffer *output);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiVulkanDeviceImportBorrowedImage(
    VernonRhiDevice device, const VernonRhiVulkanBorrowedImageDescriptor *descriptor, VernonRhiImage *output);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiVulkanDeviceImportBorrowedImageView(
    VernonRhiDevice device, const VernonRhiVulkanBorrowedImageViewDescriptor *descriptor, VernonRhiImageView *output);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiVulkanDeviceGetBufferNativeHandle(VernonRhiDevice device,
                                                                           VernonRhiBuffer buffer, uint64_t *output);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceDestroyImageView(VernonRhiDevice device, VernonRhiImageView image_view);
VERNON_RHI_CAPI VernonRhiStatus vernonRhiDeviceGetImageViewNativeHandle(VernonRhiDevice device,
                                                                        VernonRhiImageView image_view,
                                                                        uint64_t *output);

#ifdef __cplusplus
}
#endif

#endif
