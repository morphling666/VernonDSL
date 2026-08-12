#include "VernonCompiler.h"
#include "VernonExecutionGraph.h"
#include "VernonRHI.h"
#include "VernonRuntime.h"
#include "compiler_python_bridge.h"
#include "runtime/autodiff/runtime_direct_autodiff.h"
#include "runtime/tensor_bridge.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <future>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

std::string stringView(VernonStringView view) { return view.data ? std::string(view.data, view.size) : std::string(); }

struct Compiler {
    Compiler() : context(vernonCompilerCreate()) {
        if (!context)
            throw std::runtime_error("cannot create Vernon compiler");
        const VernonCpuRuntimeHelpersV1 cpuHelpers{sizeof(VernonCpuRuntimeHelpersV1), &vernonCpuWorkgroupAddressV1,
                                                   &vernonCpuLaneAddressV1, &vernonCpuWorkgroupBarrierV1,
                                                   &vernonCpuWorkgroupIsLeaderV1};
        if (vernonCompilerRegisterCpuRuntimeHelpersV1(context, &cpuHelpers) != VERNON_STATUS_OK) {
            vernonCompilerDestroy(context);
            context = nullptr;
            throw std::runtime_error("cannot register CPU workgroup helpers with the Vernon compiler");
        }
    }
    ~Compiler() { vernonCompilerDestroy(context); }

    VernonCompilerContext *context{};
};

bool targetAvailable(VernonTarget target) {
    Compiler compiler;
    return vernonCompilerGetTargetCapabilities(compiler.context, target).available != 0;
}

nb::dict targetCapabilities(VernonTarget target) {
    Compiler compiler;
    const VernonTargetCapabilities capabilities = vernonCompilerGetTargetCapabilities(compiler.context, target);
    nb::dict result;
    result["available"] = capabilities.available != 0;
    result["graphics"] = capabilities.supports_graphics != 0;
    result["compute"] = capabilities.supports_compute != 0;
    result["device_storage_atomics"] = capabilities.supports_device_storage_atomics != 0;
    result["f32_device_atomic_add"] = capabilities.supports_f32_device_atomic_add != 0;
    return result;
}

nb::list planValueAbi(const std::string &moduleText, const std::vector<std::string> &logicalDtypes) {
    std::vector<VernonStringView> dtypes;
    dtypes.reserve(logicalDtypes.size());
    for (const std::string &dtype : logicalDtypes)
        dtypes.push_back({dtype.data(), dtype.size()});
    std::unique_ptr<VernonPythonValueAbiPlan, decltype(&vernonCompilerDestroyPythonValueAbiPlan)> plan(
        vernonCompilerPlanPythonValueAbi({moduleText.data(), moduleText.size()}, dtypes.data(), dtypes.size()),
        &vernonCompilerDestroyPythonValueAbiPlan);
    if (!plan)
        throw std::bad_alloc();
    const VernonPythonValueAbiPlanView view = vernonCompilerGetPythonValueAbiPlanView(plan.get());
    if (view.status != VERNON_STATUS_OK) {
        const std::string diagnostics = stringView(view.diagnostics);
        throw std::invalid_argument(diagnostics.empty() ? "native Value ABI planning failed" : diagnostics);
    }

    nb::list nodes;
    for (size_t index = 0; index < view.node_count; ++index) {
        const VernonPythonValueAbiNodeView &node = view.nodes[index];
        std::vector<uint64_t> offsets;
        if (node.field_count)
            offsets.assign(node.field_offsets, node.field_offsets + node.field_count);
        nb::object elementStride = nb::none();
        if (node.has_element_stride)
            elementStride = nb::int_(node.element_stride);
        nodes.append(nb::make_tuple(node.byte_size, node.alignment, std::move(offsets), std::move(elementStride)));
    }
    return nodes;
}

struct StructuredVjp {
    using ResultPtr = std::unique_ptr<VernonPythonStructuredVjp, decltype(&vernonCompilerDestroyPythonStructuredVjp)>;

    explicit StructuredVjp(VernonPythonStructuredVjp *result)
        : result(result, &vernonCompilerDestroyPythonStructuredVjp) {}

    VernonPythonStructuredVjpView current() const { return vernonCompilerGetPythonStructuredVjpView(result.get()); }

    uint64_t tapeBytes() const { return current().tape_bytes; }

    nb::list derivativeRules() const {
        VernonPythonStructuredVjpView transformed = current();
        nb::list rules;
        for (size_t index = 0; index < transformed.derivative_rule_count; ++index)
            rules.append(stringView(transformed.derivative_rules[index]));
        return rules;
    }

    nb::dict profiles(const std::string &identity) {
        if (vernonCompilerFinalizePythonStructuredVjp(result.get(), {identity.data(), identity.size()}) !=
            VERNON_STATUS_OK) {
            const std::string diagnostics = stringView(current().diagnostics);
            throw std::invalid_argument(diagnostics.empty() ? "cannot finalize structured VJP profiles" : diagnostics);
        }
        VernonPythonStructuredVjpView transformed = current();
        nb::dict profiles;
        profiles["forward_with_tape"] = stringView(transformed.forward_module);
        profiles["backward"] = stringView(transformed.backward_module);
        return profiles;
    }

    ResultPtr result;
};

std::unique_ptr<StructuredVjp> buildStructuredVjp(const std::string &moduleText, const std::string &entry,
                                                  const std::vector<std::string> &wrtPaths,
                                                  const std::vector<std::string> &outputPaths,
                                                  const std::string &forwardSymbol, const std::string &backwardSymbol) {
    std::vector<VernonStringView> paths;
    paths.reserve(wrtPaths.size());
    for (const std::string &path : wrtPaths)
        paths.push_back({path.data(), path.size()});
    std::vector<VernonStringView> outputs;
    outputs.reserve(outputPaths.size());
    for (const std::string &path : outputPaths)
        outputs.push_back({path.data(), path.size()});
    auto view = [](const std::string &value) { return VernonStringView{value.data(), value.size()}; };
    VernonPythonStructuredVjp *result = vernonCompilerBuildPythonStructuredVjp(
        view(moduleText), view(entry), paths.data(), paths.size(), outputs.data(), outputs.size(), view(forwardSymbol),
        view(backwardSymbol));
    if (!result)
        throw std::bad_alloc();
    std::unique_ptr<StructuredVjp> transformedResult = std::make_unique<StructuredVjp>(result);
    VernonPythonStructuredVjpView transformed = transformedResult->current();
    if (transformed.status != VERNON_STATUS_OK) {
        std::string diagnostics = stringView(transformed.diagnostics);
        throw std::invalid_argument(diagnostics.empty() ? "structured VJP transform failed" : diagnostics);
    }
    return transformedResult;
}

struct RhiHostState;
struct Runtime;
RhiHostState *runtimeRhiHost(const Runtime *runtime);
struct CompiledProgram;
struct LoadedPipeline;
struct PipelineInvocationBuilder;
struct PythonExecutionGraph;

struct RhiHostState {
    RhiHostState(VernonRhiBackend backend, uint32_t deviceIndex) : backend(backend) {
        VernonRhiOwnedDeviceDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.backend = backend;
        descriptor.device_index = deviceIndex;
        device = vernonRhiCreateDevice(&descriptor);
        if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
            throw std::runtime_error("cannot create Vernon RHI device");
    }
    RhiHostState(VernonRhiBackend backend, const VernonOpenGLContextCallbacks &callbacks) : backend(backend) {
        VernonRhiOwnedDeviceDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.backend = backend;
        descriptor.opengl_callbacks = &callbacks;
        device = vernonRhiCreateDevice(&descriptor);
        if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
            throw std::runtime_error("cannot create external OpenGL RHI device");
    }
    ~RhiHostState() { vernonRhiDestroyDevice(device); }

    VernonRhiBackend backend;
    VernonRhiDevice device{};
};

struct RhiBuffer {
    RhiBuffer(std::shared_ptr<RhiHostState> host, size_t size) : host(std::move(host)), size(size) {
        VernonRhiBufferDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.size = size;
        descriptor.usage = VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION |
                           VERNON_RHI_BUFFER_UNIFORM | VERNON_RHI_BUFFER_STORAGE | VERNON_RHI_BUFFER_VERTEX |
                           VERNON_RHI_BUFFER_INDEX;
        descriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
        if (vernonRhiDeviceCreateBuffer(this->host->device, &descriptor, &handle) != VERNON_RHI_STATUS_OK)
            throw std::runtime_error("cannot create Vernon RHI buffer");
    }
    ~RhiBuffer() { vernonRhiDeviceDestroyBuffer(host->device, handle); }

    void upload(const nb::bytes &data, size_t offset) {
        if (offset > size || data.size() > size - offset ||
            vernonRhiDeviceUploadBuffer(host->device, handle, offset, data.c_str(), data.size()) !=
                VERNON_RHI_STATUS_OK)
            throw std::runtime_error("RHI buffer upload failed");
    }
    void uploadRanges(const nb::list &ranges) {
        std::vector<std::pair<size_t, nb::bytes>> uploads;
        std::vector<VernonRhiBufferUploadRange> nativeRanges;
        uploads.reserve(nb::len(ranges));
        nativeRanges.reserve(nb::len(ranges));
        for (nb::handle item : ranges) {
            nb::tuple range = nb::cast<nb::tuple>(item);
            if (nb::len(range) != 2)
                throw std::invalid_argument("RHI buffer upload range must contain an offset and bytes");
            const size_t offset = nb::cast<size_t>(range[0]);
            nb::bytes data = nb::cast<nb::bytes>(range[1]);
            if (offset > size || data.size() > size - offset)
                throw std::invalid_argument("RHI buffer upload range is outside the buffer");
            uploads.emplace_back(offset, std::move(data));
        }
        for (const auto &[offset, data] : uploads)
            nativeRanges.push_back({offset, data.c_str(), data.size()});
        if (vernonRhiDeviceUploadBufferRanges(host->device, handle, nativeRanges.data(), nativeRanges.size()) !=
            VERNON_RHI_STATUS_OK)
            throw std::runtime_error("RHI buffer range upload failed");
    }
    nb::bytes download() const {
        std::string data(size, '\0');
        if (vernonRhiDeviceDownloadBuffer(host->device, handle, 0, data.data(), data.size()) != VERNON_RHI_STATUS_OK)
            throw std::runtime_error("RHI buffer download failed");
        return nb::bytes(data.data(), data.size());
    }

    std::shared_ptr<RhiHostState> host;
    VernonRhiBuffer handle{};
    size_t size{};
};

VernonRhiFormat rhiFormat(VernonTextureFormat format) {
    switch (format) {
    case VERNON_TEXTURE_RGBA8_UNORM:
        return VERNON_RHI_FORMAT_RGBA8_UNORM;
    case VERNON_TEXTURE_RGBA8_SRGB:
        return VERNON_RHI_FORMAT_RGBA8_SRGB;
    case VERNON_TEXTURE_RGBA16_FLOAT:
        return VERNON_RHI_FORMAT_RGBA16_FLOAT;
    case VERNON_TEXTURE_RGBA32_FLOAT:
        return VERNON_RHI_FORMAT_RGBA32_FLOAT;
    case VERNON_TEXTURE_R8_UNORM:
        return VERNON_RHI_FORMAT_R8_UNORM;
    case VERNON_TEXTURE_R16_FLOAT:
        return VERNON_RHI_FORMAT_R16_FLOAT;
    case VERNON_TEXTURE_R32_FLOAT:
        return VERNON_RHI_FORMAT_R32_FLOAT;
    case VERNON_TEXTURE_RG8_UNORM:
        return VERNON_RHI_FORMAT_RG8_UNORM;
    case VERNON_TEXTURE_RGB8_UNORM:
        return VERNON_RHI_FORMAT_RGB8_UNORM;
    case VERNON_TEXTURE_R11G11B10_FLOAT:
        return VERNON_RHI_FORMAT_R11G11B10_FLOAT;
    case VERNON_TEXTURE_D32_FLOAT:
        return VERNON_RHI_FORMAT_D32_FLOAT;
    case VERNON_TEXTURE_D32_FLOAT_S8_UINT:
        return VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT;
    }
    throw std::invalid_argument("unsupported attachment image format");
}

VernonRhiImageDimension rhiDimension(VernonTextureDimension dimension) {
    switch (dimension) {
    case VERNON_TEXTURE_2D:
        return VERNON_RHI_IMAGE_2D;
    case VERNON_TEXTURE_3D:
        return VERNON_RHI_IMAGE_3D;
    case VERNON_TEXTURE_CUBE:
        return VERNON_RHI_IMAGE_CUBE;
    }
    throw std::invalid_argument("unsupported texture dimension");
}

struct RhiImageView;

struct RhiImage {
    RhiImage(std::shared_ptr<RhiHostState> host, uint32_t width, uint32_t height, uint32_t depth,
             VernonTextureFormat format, VernonTextureDimension dimension, uint32_t mipLevels, uint32_t usage)
        : host(std::move(host)), width(width), height(height), depth(depth), format(format), dimension(dimension),
          mipLevels(mipLevels), usage(usage), layers(dimension == VERNON_TEXTURE_CUBE ? 6u : 1u) {
        if (!width || !height || !depth || !mipLevels || !usage)
            throw std::invalid_argument("RHI image extent, mip count, and usage must be non-zero");
        if (dimension != VERNON_TEXTURE_3D && depth != 1)
            throw std::invalid_argument("only a 3D RHI image may have depth greater than one");
        if (dimension == VERNON_TEXTURE_CUBE && width != height)
            throw std::invalid_argument("RHI cube image faces must be square");
        VernonRhiImageDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.dimension = rhiDimension(dimension);
        descriptor.format = rhiFormat(format);
        descriptor.width = width;
        descriptor.height = height;
        descriptor.depth = depth;
        descriptor.mip_levels = mipLevels;
        descriptor.array_layers = layers;
        descriptor.sample_count = 1;
        descriptor.usage = usage;
        if (vernonRhiDeviceCreateImage(this->host->device, &descriptor, &handle) != VERNON_RHI_STATUS_OK)
            throw std::runtime_error("cannot create Vernon RHI image");
    }
    ~RhiImage() { vernonRhiDeviceDestroyImage(host->device, handle); }

    void upload(const nb::bytes &data, uint32_t mipLevel, uint32_t offsetX, uint32_t offsetY, uint32_t offsetZ,
                uint32_t uploadWidth, uint32_t uploadHeight, uint32_t uploadDepth) {
        if (!(usage & VERNON_RHI_IMAGE_TRANSFER_DESTINATION))
            throw std::runtime_error("image format does not support upload");
        if (mipLevel >= mipLevels)
            throw std::invalid_argument("RHI image upload mip level is out of range");
        const uint32_t mipWidth = mipExtent(width, mipLevel);
        const uint32_t mipHeight = mipExtent(height, mipLevel);
        const uint32_t mipDepth = dimension == VERNON_TEXTURE_3D ? mipExtent(depth, mipLevel) : depth;
        uploadWidth = uploadWidth ? uploadWidth : mipWidth - offsetX;
        uploadHeight = uploadHeight ? uploadHeight : mipHeight - offsetY;
        uploadDepth = uploadDepth ? uploadDepth : mipDepth - offsetZ;
        if (offsetX >= mipWidth || offsetY >= mipHeight || offsetZ >= mipDepth || uploadWidth > mipWidth - offsetX ||
            uploadHeight > mipHeight - offsetY || uploadDepth > mipDepth - offsetZ)
            throw std::invalid_argument("RHI image upload region is out of range");
        const Layout layout = dataLayout(format);
        const size_t layerSize = checkedByteSize(uploadWidth, uploadHeight, uploadDepth, layout.pixelSize);
        if (data.size() != layerSize * layers)
            throw std::runtime_error("RHI image upload size does not match its extent");
        std::vector<VernonRhiImageUploadDescriptor> descriptors(layers);
        for (uint32_t layer = 0; layer < layers; ++layer) {
            VernonRhiImageUploadDescriptor &descriptor = descriptors[layer];
            descriptor.struct_size = sizeof(descriptor);
            descriptor.mip_level = mipLevel;
            descriptor.array_layer = layer;
            descriptor.offset_x = offsetX;
            descriptor.offset_y = offsetY;
            descriptor.offset_z = offsetZ;
            descriptor.width = uploadWidth;
            descriptor.height = uploadHeight;
            descriptor.depth = uploadDepth;
            descriptor.source_format = layout.format;
            descriptor.source_type = layout.type;
            descriptor.data = data.c_str() + layerSize * layer;
        }
        if (vernonRhiDeviceUploadImage(host->device, handle, descriptors.data(), descriptors.size()) !=
            VERNON_RHI_STATUS_OK)
            throw std::runtime_error("RHI image upload failed: " +
                                     stringView(vernonRhiDeviceGetLastError(host->device)));
    }
    nb::bytes download(uint32_t mipLevel, uint32_t offsetX, uint32_t offsetY, uint32_t offsetZ, uint32_t downloadWidth,
                       uint32_t downloadHeight, uint32_t downloadDepth) const {
        if (!(usage & VERNON_RHI_IMAGE_TRANSFER_SOURCE))
            throw std::runtime_error("image format does not support download");
        if (mipLevel >= mipLevels)
            throw std::invalid_argument("RHI image download mip level is out of range");
        const uint32_t mipWidth = mipExtent(width, mipLevel);
        const uint32_t mipHeight = mipExtent(height, mipLevel);
        const uint32_t mipDepth = dimension == VERNON_TEXTURE_3D ? mipExtent(depth, mipLevel) : depth;
        downloadWidth = downloadWidth ? downloadWidth : mipWidth - offsetX;
        downloadHeight = downloadHeight ? downloadHeight : mipHeight - offsetY;
        downloadDepth = downloadDepth ? downloadDepth : mipDepth - offsetZ;
        if (offsetX >= mipWidth || offsetY >= mipHeight || offsetZ >= mipDepth || downloadWidth > mipWidth - offsetX ||
            downloadHeight > mipHeight - offsetY || downloadDepth > mipDepth - offsetZ)
            throw std::invalid_argument("RHI image download region is out of range");
        const Layout layout = dataLayout(format);
        const size_t layerSize = checkedByteSize(downloadWidth, downloadHeight, downloadDepth, layout.pixelSize);
        std::string data(layerSize * layers, '\0');
        for (uint32_t layer = 0; layer < layers; ++layer) {
            VernonRhiImageDownloadDescriptor descriptor{};
            descriptor.struct_size = sizeof(descriptor);
            descriptor.mip_level = mipLevel;
            descriptor.array_layer = layer;
            descriptor.offset_x = offsetX;
            descriptor.offset_y = offsetY;
            descriptor.offset_z = offsetZ;
            descriptor.width = downloadWidth;
            descriptor.height = downloadHeight;
            descriptor.depth = downloadDepth;
            descriptor.destination_format = layout.format;
            descriptor.destination_type = layout.type;
            if (vernonRhiDeviceDownloadImage(host->device, handle, &descriptor, data.data() + layer * layerSize,
                                             layerSize) != VERNON_RHI_STATUS_OK)
                throw std::runtime_error("RHI image download failed: " +
                                         stringView(vernonRhiDeviceGetLastError(host->device)));
        }
        return nb::bytes(data.data(), data.size());
    }

    void generateMipmaps() {
        if (mipLevels < 2)
            throw std::runtime_error("image has no mip chain to generate");
        if (vernonRhiDeviceGenerateImageMipmaps(host->device, handle) != VERNON_RHI_STATUS_OK)
            throw std::runtime_error("RHI image mipmap generation failed: " +
                                     stringView(vernonRhiDeviceGetLastError(host->device)));
    }

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

    static uint32_t mipExtent(uint32_t extent, uint32_t level) {
        const uint32_t value = extent >> level;
        return value ? value : 1;
    }

    static Layout dataLayout(VernonTextureFormat format) {
        switch (format) {
        case VERNON_TEXTURE_R8_UNORM:
            return {1, VERNON_RHI_IMAGE_DATA_RED, VERNON_RHI_IMAGE_DATA_UINT8};
        case VERNON_TEXTURE_RG8_UNORM:
            return {2, VERNON_RHI_IMAGE_DATA_RG, VERNON_RHI_IMAGE_DATA_UINT8};
        case VERNON_TEXTURE_RGB8_UNORM:
            return {3, VERNON_RHI_IMAGE_DATA_RGB, VERNON_RHI_IMAGE_DATA_UINT8};
        case VERNON_TEXTURE_RGBA8_UNORM:
        case VERNON_TEXTURE_RGBA8_SRGB:
            return {4, VERNON_RHI_IMAGE_DATA_RGBA, VERNON_RHI_IMAGE_DATA_UINT8};
        case VERNON_TEXTURE_R16_FLOAT:
            return {2, VERNON_RHI_IMAGE_DATA_RED, VERNON_RHI_IMAGE_DATA_FLOAT16};
        case VERNON_TEXTURE_RGBA16_FLOAT:
            return {8, VERNON_RHI_IMAGE_DATA_RGBA, VERNON_RHI_IMAGE_DATA_FLOAT16};
        case VERNON_TEXTURE_R32_FLOAT:
            return {4, VERNON_RHI_IMAGE_DATA_RED, VERNON_RHI_IMAGE_DATA_FLOAT32};
        case VERNON_TEXTURE_RGBA32_FLOAT:
            return {16, VERNON_RHI_IMAGE_DATA_RGBA, VERNON_RHI_IMAGE_DATA_FLOAT32};
        case VERNON_TEXTURE_R11G11B10_FLOAT:
            return {4, VERNON_RHI_IMAGE_DATA_RGB, VERNON_RHI_IMAGE_DATA_UINT32};
        case VERNON_TEXTURE_D32_FLOAT:
            return {4, VERNON_RHI_IMAGE_DATA_DEPTH, VERNON_RHI_IMAGE_DATA_FLOAT32};
        case VERNON_TEXTURE_D32_FLOAT_S8_UINT:
            return {8, VERNON_RHI_IMAGE_DATA_DEPTH_STENCIL, VERNON_RHI_IMAGE_DATA_FLOAT32};
        }
        throw std::invalid_argument("unsupported RHI image data layout");
    }

    static size_t checkedByteSize(uint32_t width, uint32_t height, uint32_t depth, size_t pixelSize) {
        size_t size = width;
        for (const size_t extent : {static_cast<size_t>(height), static_cast<size_t>(depth), pixelSize}) {
            if (extent && size > std::numeric_limits<size_t>::max() / extent)
                throw std::overflow_error("RHI image byte size overflows");
            size *= extent;
        }
        return size;
    }
};

struct RhiImageView {
    RhiImageView(RhiImage *image, VernonTextureFormat format, VernonTextureDimension dimension, uint32_t baseMipLevel,
                 uint32_t mipLevelCount, uint32_t baseArrayLayer, uint32_t arrayLayerCount, uint32_t aspects)
        : image(image), host(image ? image->host : nullptr), format(format), dimension(dimension),
          baseMipLevel(baseMipLevel), mipLevelCount(mipLevelCount), baseArrayLayer(baseArrayLayer),
          arrayLayerCount(arrayLayerCount), aspects(aspects) {
        if (!image || !mipLevelCount || !arrayLayerCount || !aspects)
            throw std::invalid_argument("RHI image view requires an image and non-empty subresources");
        VernonRhiImageViewDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.image = image->handle;
        descriptor.dimension = rhiDimension(dimension);
        descriptor.format = rhiFormat(format);
        descriptor.base_mip_level = baseMipLevel;
        descriptor.mip_level_count = mipLevelCount;
        descriptor.base_array_layer = baseArrayLayer;
        descriptor.array_layer_count = arrayLayerCount;
        descriptor.aspects = aspects;
        if (vernonRhiDeviceCreateImageView(host->device, &descriptor, &handle) != VERNON_RHI_STATUS_OK)
            throw std::invalid_argument("cannot create Vernon RHI image view: " +
                                        stringView(vernonRhiDeviceGetLastError(host->device)));
        width = std::max(image->width >> baseMipLevel, 1u);
        height = std::max(image->height >> baseMipLevel, 1u);
        layers = dimension == VERNON_TEXTURE_3D ? 1u : arrayLayerCount;
    }

    ~RhiImageView() { vernonRhiDeviceDestroyImageView(host->device, handle); }

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
    RhiSampler(std::shared_ptr<RhiHostState> host, VernonRhiSamplerAddressMode address) : host(std::move(host)) {
        if (address > VERNON_RHI_ADDRESS_MIRRORED_REPEAT)
            throw std::invalid_argument("unsupported sampler address mode");
        VernonRhiSamplerDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.min_filter = VERNON_RHI_FILTER_LINEAR;
        descriptor.mag_filter = VERNON_RHI_FILTER_LINEAR;
        descriptor.mip_filter = VERNON_RHI_FILTER_LINEAR;
        descriptor.address_u = address;
        descriptor.address_v = address;
        descriptor.address_w = address;
        descriptor.max_anisotropy = 1.0f;
        if (vernonRhiDeviceCreateSampler(this->host->device, &descriptor, &handle) != VERNON_RHI_STATUS_OK)
            throw std::runtime_error("cannot create Vernon RHI sampler");
    }
    ~RhiSampler() { vernonRhiDeviceDestroySampler(host->device, handle); }

    std::shared_ptr<RhiHostState> host;
    VernonRhiSampler handle{};
};

struct RhiHost {
    RhiHost(VernonRhiBackend backend, uint32_t deviceIndex)
        : state(std::make_shared<RhiHostState>(backend, deviceIndex)) {}
    explicit RhiHost(std::shared_ptr<RhiHostState> state) : state(std::move(state)) {}

    static std::unique_ptr<RhiHost> createExternalOpenGL(VernonRhiBackend backend, uintptr_t userData,
                                                         uintptr_t makeCurrent, uintptr_t getProcAddress,
                                                         uint16_t apiMajor, uint16_t apiMinor) {
        if (backend != VERNON_RHI_BACKEND_OPENGL && backend != VERNON_RHI_BACKEND_OPENGL_ES)
            throw std::invalid_argument("external OpenGL host requires an OpenGL backend");
        VernonOpenGLContextCallbacks callbacks{};
        callbacks.struct_size = sizeof(callbacks);
        callbacks.user_data = reinterpret_cast<void *>(userData);
        callbacks.make_current = reinterpret_cast<VernonOpenGLMakeCurrentFn>(makeCurrent);
        callbacks.get_proc_address = reinterpret_cast<VernonOpenGLGetProcAddressFn>(getProcAddress);
        callbacks.api_version_major = apiMajor;
        callbacks.api_version_minor = apiMinor;
        return std::make_unique<RhiHost>(std::make_shared<RhiHostState>(backend, callbacks));
    }

    std::unique_ptr<RhiBuffer> createBuffer(size_t size) { return std::make_unique<RhiBuffer>(state, size); }
    std::unique_ptr<RhiImage> createImage(uint32_t width, uint32_t height, VernonTextureFormat format,
                                          VernonTextureDimension dimension, uint32_t depth, uint32_t mipLevels,
                                          uint32_t usage) {
        if (dimension == VERNON_TEXTURE_3D &&
            (format == VERNON_TEXTURE_D32_FLOAT || format == VERNON_TEXTURE_D32_FLOAT_S8_UINT))
            throw std::invalid_argument("3D depth textures are unsupported");
        if (!usage) {
            usage = VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION | VERNON_RHI_IMAGE_SAMPLED;
            if (dimension != VERNON_TEXTURE_3D)
                usage |= format == VERNON_TEXTURE_D32_FLOAT || format == VERNON_TEXTURE_D32_FLOAT_S8_UINT
                             ? VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT
                             : VERNON_RHI_IMAGE_COLOR_ATTACHMENT;
        }
        return std::make_unique<RhiImage>(state, width, height, depth, format, dimension, mipLevels, usage);
    }
    std::unique_ptr<RhiImage> createAttachmentImage(uint32_t width, uint32_t height, VernonTextureFormat format,
                                                    uint32_t usage) {
        constexpr uint32_t attachmentUsages =
            VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT;
        if (!usage || (usage & ~attachmentUsages))
            throw std::invalid_argument("attachment image usage must contain only attachment roles");
        const bool depthFormat = format == VERNON_TEXTURE_D32_FLOAT || format == VERNON_TEXTURE_D32_FLOAT_S8_UINT;
        const bool depthUsage = (usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT) != 0;
        if (depthFormat != depthUsage || (depthUsage && (usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT)))
            throw std::invalid_argument("attachment image format does not match its usage");
        return std::make_unique<RhiImage>(state, width, height, 1, format, VERNON_TEXTURE_2D, 1, usage);
    }
    std::unique_ptr<RhiSampler> createSampler(VernonRhiSamplerAddressMode address) {
        return std::make_unique<RhiSampler>(state, address);
    }
    std::unique_ptr<Runtime> createRuntime();
    std::unique_ptr<PythonExecutionGraph> createExecutionGraph();

    std::shared_ptr<RhiHostState> state;
};

struct PythonGraphResource {
    explicit PythonGraphResource(vernon::execution::GraphResource value) : resource(value) {}
    explicit PythonGraphResource(vernon::execution::GraphBuffer value) : resource(value), buffer(value) {}
    explicit PythonGraphResource(vernon::execution::GraphImage value) : resource(value), image(value), isImage(true) {}

    vernon::execution::GraphResource resource;
    vernon::execution::GraphBuffer buffer;
    vernon::execution::GraphImage image;
    bool isImage{};
};

struct PythonExecutionParameter {
    explicit PythonExecutionParameter(vernon::execution::ExecutionParameter value) : parameter(value) {}
    vernon::execution::ExecutionParameter parameter;
};

struct PythonExecutionBindingValue final : vernon::execution::ExecutionBindingValue {
    explicit PythonExecutionBindingValue(nb::object value) : value(std::move(value)) {}
    nb::object value;
};

struct PythonExecutionBindingsBuilder {
    explicit PythonExecutionBindingsBuilder(vernon::execution::ExecutionBindingsBuilder value)
        : builder(std::move(value)) {}

    void set(const PythonExecutionParameter &parameter, nb::object value) {
        builder.set(parameter.parameter, std::make_shared<PythonExecutionBindingValue>(std::move(value)));
    }

    vernon::execution::ExecutionBindingsBuilder builder;
};

struct PythonExecutionBindingsView {
    explicit PythonExecutionBindingsView(const vernon::execution::ExecutionResources &value) : resources(value) {}

    nb::object get(const PythonExecutionParameter &parameter) const {
        auto value =
            std::dynamic_pointer_cast<const PythonExecutionBindingValue>(resources.binding(parameter.parameter));
        if (!value)
            throw std::runtime_error("execution parameter binding has an incompatible native value type");
        return value->value;
    }

    uintptr_t token(const PythonExecutionParameter &parameter) const {
        return reinterpret_cast<uintptr_t>(resources.binding(parameter.parameter).get());
    }

    const vernon::execution::ExecutionResources &resources;
};

struct PythonExecutionGraph;

struct PythonGraphCallbackState {
    std::exception_ptr exception;
};

struct PythonRenderPass final : vernon::execution::RenderPass {
    PythonRenderPass(std::shared_ptr<PythonGraphCallbackState> callbackState, std::string name, PyObject *owner)
        : RenderPass(std::move(name)), callbackState(std::move(callbackState)), owner(owner) {}

    void declare() override;
    VernonRhiStatus execute(vernon::execution::GraphicsEncoder &encoder,
                            const vernon::execution::ExecutionResources &) override;

    void use(const PythonGraphResource &resource, vernon::execution::AccessMode access, VernonRhiResourceState state,
             uint32_t stageMask) {
        if (access == vernon::execution::AccessMode::Read) {
            if (resource.isImage)
                read(resource.image, state, stageMask);
            else
                read(resource.resource, state, stageMask);
        } else if (access == vernon::execution::AccessMode::Write) {
            if (resource.isImage)
                write(resource.image, state, stageMask);
            else
                write(resource.resource, state, stageMask);
        } else if (resource.isImage) {
            readWrite(resource.image, state, stageMask);
        } else {
            readWrite(resource.resource, state, stageMask);
        }
    }

    void addColor(uint32_t location, const PythonGraphResource &resource, VernonRhiLoadOperation load,
                  VernonRhiStoreOperation store, const std::array<float, 4> &clear) {
        if (!resource.isImage)
            throw std::invalid_argument("color attachment must be an image graph resource");
        vernon::execution::ColorAttachmentUse attachment{};
        attachment.image = resource.image;
        attachment.load = load;
        attachment.store = store;
        std::copy(clear.begin(), clear.end(), attachment.clear);
        color(location, attachment);
    }

    void setDepth(const PythonGraphResource &resource, VernonRhiLoadOperation depthLoad,
                  VernonRhiStoreOperation depthStore, float clearDepth, VernonRhiLoadOperation stencilLoad,
                  VernonRhiStoreOperation stencilStore, uint32_t clearStencil, bool readOnlyDepth,
                  bool readOnlyStencil) {
        if (!resource.isImage)
            throw std::invalid_argument("depth attachment must be an image graph resource");
        vernon::execution::DepthStencilAttachmentUse attachment{};
        attachment.image = resource.image;
        attachment.depthLoad = depthLoad;
        attachment.depthStore = depthStore;
        attachment.clearDepth = clearDepth;
        attachment.stencilLoad = stencilLoad;
        attachment.stencilStore = stencilStore;
        attachment.clearStencil = clearStencil;
        attachment.readOnlyDepth = readOnlyDepth;
        attachment.readOnlyStencil = readOnlyStencil;
        depth(attachment);
    }

    void setRenderArea(uint32_t x, uint32_t y, uint32_t width, uint32_t height) { renderArea(x, y, width, height); }

    std::shared_ptr<PythonGraphCallbackState> callbackState;
    PyObject *owner{};
};

struct PythonComputePass final : vernon::execution::ComputePass {
    PythonComputePass(std::shared_ptr<PythonGraphCallbackState> callbackState, std::string name, PyObject *owner)
        : ComputePass(std::move(name)), callbackState(std::move(callbackState)), owner(owner) {}

    void declare() override;
    VernonRhiStatus execute(vernon::execution::ComputeEncoder &encoder,
                            const vernon::execution::ExecutionResources &) override;

    void use(const PythonGraphResource &resource, vernon::execution::AccessMode access, VernonRhiResourceState state,
             uint32_t stageMask) {
        if (access == vernon::execution::AccessMode::Read) {
            if (resource.isImage)
                read(resource.image, state, stageMask);
            else
                read(resource.resource, state, stageMask);
        } else if (access == vernon::execution::AccessMode::Write) {
            if (resource.isImage)
                write(resource.image, state, stageMask);
            else
                write(resource.resource, state, stageMask);
        } else if (resource.isImage) {
            readWrite(resource.image, state, stageMask);
        } else {
            readWrite(resource.resource, state, stageMask);
        }
    }

    std::shared_ptr<PythonGraphCallbackState> callbackState;
    PyObject *owner{};
};

struct PythonCompiledBarrier {
    uint32_t sourceStageMask{};
    uint32_t destinationStageMask{};
    uint32_t sourceAccess{};
    uint32_t destinationAccess{};
    uint32_t oldState{};
    uint32_t newState{};
    bool isImage{};
    uint32_t baseMipLevel{};
    uint32_t mipLevelCount{};
    uint32_t baseArrayLayer{};
    uint32_t arrayLayerCount{};
    uint32_t aspects{};
};

struct PythonCompiledScope {
    bool rendering{};
    std::vector<uint32_t> passIndices;
    std::vector<PythonCompiledBarrier> barriers;
};

struct PythonExecutionSubmission {
    PythonExecutionSubmission(vernon::execution::ExecutionSubmission value,
                              std::shared_ptr<std::vector<nb::object>> retainedOwners)
        : submission(std::move(value)), owners(std::move(retainedOwners)) {}

    void wait() {
        const VernonRhiStatus status = submission.wait();
        if (status != VERNON_RHI_STATUS_OK)
            throw std::runtime_error("execution submission failed with provider status " +
                                     std::to_string(static_cast<uint32_t>(status)));
    }

    uint32_t state() const { return static_cast<uint32_t>(submission.state()); }
    vernon::execution::ExecutionSubmission submission;
    std::shared_ptr<std::vector<nb::object>> owners;
};

struct PythonCompiledExecutionGraph {
    PythonCompiledExecutionGraph(std::shared_ptr<vernon::execution::CompiledExecutionGraph> value,
                                 std::shared_ptr<PythonGraphCallbackState> state,
                                 std::shared_ptr<std::vector<nb::object>> retainedOwners)
        : plan(std::move(value)), callbackState(std::move(state)), owners(std::move(retainedOwners)) {}

    std::unique_ptr<PythonExecutionBindingsBuilder> createBindings(const nb::list &initial) {
        std::vector<vernon::execution::ExecutionBinding> bindings;
        bindings.reserve(nb::len(initial));
        for (nb::handle item : initial) {
            nb::tuple entry = nb::cast<nb::tuple>(item);
            if (nb::len(entry) != 2)
                throw std::invalid_argument("execution binding entries must contain a parameter and value");
            const auto &parameter = nb::cast<const PythonExecutionParameter &>(entry[0]);
            bindings.push_back(
                {parameter.parameter, std::make_shared<PythonExecutionBindingValue>(nb::borrow<nb::object>(entry[1]))});
        }
        return std::make_unique<PythonExecutionBindingsBuilder>(plan->createBindings(bindings));
    }

    std::unique_ptr<PythonExecutionSubmission> submit(PythonExecutionBindingsBuilder *bindings) {
        callbackState->exception = nullptr;
        auto snapshot =
            bindings ? bindings->builder.snapshot() : std::shared_ptr<const vernon::execution::ExecutionBindings>{};
        auto submission = std::make_unique<PythonExecutionSubmission>(plan->submit(std::move(snapshot)), owners);
        if (callbackState->exception)
            std::rethrow_exception(std::exchange(callbackState->exception, nullptr));
        return submission;
    }

    std::vector<PythonCompiledScope> scopes() const {
        std::vector<PythonCompiledScope> result;
        result.reserve(plan->scopes().size());
        for (const auto &scope : plan->scopes()) {
            PythonCompiledScope compiled{scope.rendering, scope.passIndices, {}};
            compiled.barriers.reserve(scope.barriers.size());
            for (const VernonRhiBarrier &barrier : scope.barriers)
                compiled.barriers.push_back(
                    {barrier.source_stage_mask, barrier.destination_stage_mask, barrier.source_access,
                     barrier.destination_access, static_cast<uint32_t>(barrier.old_state),
                     static_cast<uint32_t>(barrier.new_state), barrier.is_image != 0,
                     barrier.image_subresources.base_mip_level, barrier.image_subresources.mip_level_count,
                     barrier.image_subresources.base_array_layer, barrier.image_subresources.array_layer_count,
                     barrier.image_subresources.aspects});
            result.push_back(std::move(compiled));
        }
        return result;
    }

    std::shared_ptr<vernon::execution::CompiledExecutionGraph> plan;
    std::shared_ptr<PythonGraphCallbackState> callbackState;
    std::shared_ptr<std::vector<nb::object>> owners;
};

struct PythonExecutionGraph {
    PythonExecutionGraph() = default;
    explicit PythonExecutionGraph(std::shared_ptr<RhiHostState> host)
        : host(std::move(host)), graph(this->host->device) {}

    PythonRenderPass *addRenderPass(const std::string &name, const nb::object &owner) {
        owners->push_back(owner);
        return &graph.emplacePass<PythonRenderPass>(callbackState, name, owner.ptr());
    }

    PythonComputePass *addComputePass(const std::string &name, const nb::object &owner) {
        owners->push_back(owner);
        return &graph.emplacePass<PythonComputePass>(callbackState, name, owner.ptr());
    }

    PythonGraphResource importBuffer(RhiBuffer &buffer, bool exported) {
        if (buffer.host != host)
            throw std::invalid_argument("buffer belongs to another execution graph device");
        return PythonGraphResource(graph.importBuffer(buffer.handle, exported));
    }

    PythonGraphResource importHostBuffer(uint64_t identity, bool exported) {
        const vernon::execution::GraphBuffer resource = graph.importHostBuffer(identity, exported);
        if (resource.id == UINT32_MAX)
            throw std::invalid_argument("host execution graph resource identity must be non-zero");
        return PythonGraphResource(resource);
    }

    PythonGraphResource importImage(RhiImageView &view, bool exported) {
        if (view.host != host)
            throw std::invalid_argument("image belongs to another execution graph device");
        return PythonGraphResource(graph.importImage(view.image->handle, view.handle, exported));
    }

    PythonExecutionParameter parameter(const std::string &name) {
        return PythonExecutionParameter(graph.parameter(name));
    }

    std::unique_ptr<PythonCompiledExecutionGraph> compile() {
        std::string error;
        auto plan = graph.compile(error);
        if (!plan)
            throw std::invalid_argument(error);
        return std::make_unique<PythonCompiledExecutionGraph>(std::move(plan), callbackState, owners);
    }

    void validate() const {
        std::string error;
        if (!graph.validate(error))
            throw std::invalid_argument(error);
    }

    std::shared_ptr<RhiHostState> host;
    std::shared_ptr<PythonGraphCallbackState> callbackState{std::make_shared<PythonGraphCallbackState>()};
    std::shared_ptr<std::vector<nb::object>> owners{std::make_shared<std::vector<nb::object>>()};
    vernon::execution::ExecutionGraph graph;
};

void PythonRenderPass::declare() { nb::borrow<nb::object>(owner).attr("_native_declare")(); }

VernonRhiStatus PythonRenderPass::execute(vernon::execution::GraphicsEncoder &encoder,
                                          const vernon::execution::ExecutionResources &resources) {
    try {
        if (resources.hasBindings()) {
            PythonExecutionBindingsView bindings(resources);
            nb::borrow<nb::object>(owner).attr("_native_execute")(nb::cast(encoder),
                                                                  nb::cast(&bindings, nb::rv_policy::reference));
        } else {
            nb::borrow<nb::object>(owner).attr("_native_execute")(nb::cast(encoder), nb::none());
        }
        return VERNON_RHI_STATUS_OK;
    } catch (...) {
        callbackState->exception = std::current_exception();
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
}

void PythonComputePass::declare() { nb::borrow<nb::object>(owner).attr("_native_declare")(); }

VernonRhiStatus PythonComputePass::execute(vernon::execution::ComputeEncoder &encoder,
                                           const vernon::execution::ExecutionResources &resources) {
    try {
        if (resources.hasBindings()) {
            PythonExecutionBindingsView bindings(resources);
            nb::borrow<nb::object>(owner).attr("_native_execute")(nb::cast(encoder),
                                                                  nb::cast(&bindings, nb::rv_policy::reference));
        } else {
            nb::borrow<nb::object>(owner).attr("_native_execute")(nb::cast(encoder), nb::none());
        }
        return VERNON_RHI_STATUS_OK;
    } catch (...) {
        callbackState->exception = std::current_exception();
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
}

std::unique_ptr<PythonExecutionGraph> RhiHost::createExecutionGraph() {
    return std::make_unique<PythonExecutionGraph>(state);
}

using SharedCompileResult = std::shared_ptr<VernonCompileResult>;

struct CompiledProgram {
    CompiledProgram(VernonCompileResult *result, VernonTarget target)
        : result(result, &vernonCompileResultDestroy), target(target) {
        if (!this->result)
            throw std::runtime_error("compiler returned no result");
    }

    bool ok() const { return vernonCompileResultGetStatus(result.get()) == VERNON_STATUS_OK; }

    VernonStatus status() const { return vernonCompileResultGetStatus(result.get()); }

    std::string diagnostics() const { return stringView(vernonCompileResultGetDiagnostics(result.get())); }

    std::string reflection() const { return stringView(vernonCompileResultGetReflection(result.get())); }

    nb::list artifacts() const {
        nb::list values;
        for (size_t index = 0; index < vernonCompileResultGetArtifactCount(result.get()); ++index) {
            VernonStringView name = vernonCompileResultGetArtifactName(result.get(), index);
            VernonStringView data = vernonCompileResultGetArtifactData(result.get(), index);
            values.append(nb::make_tuple(stringView(name), nb::bytes(data.data, data.size)));
        }
        return values;
    }

    bool hasCpuEntry(const std::string &entry) const {
        return vernonCompileResultGetCpuEntry(result.get(), entry.data(), entry.size()) != nullptr;
    }

    void requireSuccess() const {
        if (!ok()) {
            std::string message = diagnostics();
            throw std::runtime_error(message.empty() ? "compilation failed" : message);
        }
    }

    SharedCompileResult result;
    VernonTarget target;
};

struct CpuTargetOptionStrings {
    std::string triple;
    std::string processor;
    std::string features;
};

CpuTargetOptionStrings parseCpuTargetOptions(const nb::dict &targetOptions) {
    for (auto item : targetOptions) {
        const std::string key = nb::cast<std::string>(item.first);
        if (key != "triple" && key != "processor" && key != "features")
            throw std::invalid_argument("unknown option '" + key + "' for CPU target");
    }
    auto readString = [&](const char *name) {
        return targetOptions.contains(name) ? nb::cast<std::string>(targetOptions[name]) : std::string();
    };
    return {readString("triple"), readString("processor"), readString("features")};
}

std::unique_ptr<CompiledProgram> compileProgramResult(Compiler &compiler, const std::string &mlir, VernonTarget target,
                                                      const nb::dict &targetOptions) {
    auto readString = [&](const char *name) {
        return targetOptions.contains(name) ? nb::cast<std::string>(targetOptions[name]) : std::string();
    };
    auto rejectUnknown = [&](std::initializer_list<std::string_view> allowed) {
        for (auto item : targetOptions) {
            const std::string key = nb::cast<std::string>(item.first);
            if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
                throw std::invalid_argument("unknown option '" + key + "' for selected target");
        }
    };
    VernonCompileOptions options{};
    options.struct_size = sizeof(options);
    options.target = target;
    CpuTargetOptionStrings cpu;
    if (target == VERNON_TARGET_CPU) {
        cpu = parseCpuTargetOptions(targetOptions);
        options.as.cpu.triple = VernonStringView{cpu.triple.data(), cpu.triple.size()};
        options.as.cpu.processor = VernonStringView{cpu.processor.data(), cpu.processor.size()};
        options.as.cpu.features = VernonStringView{cpu.features.data(), cpu.features.size()};
    } else if (target == VERNON_TARGET_OPENGL || target == VERNON_TARGET_OPENGL_ES) {
        rejectUnknown({"version"});
        options.as.opengl.version =
            targetOptions.contains("version") ? nb::cast<uint32_t>(targetOptions["version"]) : 0;
    } else if (target == VERNON_TARGET_METAL) {
        rejectUnknown({"platform"});
        const std::string platform = readString("platform");
        options.as.metal.platform = platform.empty() || platform == "macos" ? VERNON_METAL_PLATFORM_MACOS
                                    : platform == "ios"                     ? VERNON_METAL_PLATFORM_IOS
                                                        : static_cast<VernonMetalPlatform>(UINT32_MAX);
    } else if (target == VERNON_TARGET_DIRECTX) {
        rejectUnknown({"shader_model"});
        options.as.directx.shader_model =
            targetOptions.contains("shader_model") ? nb::cast<uint32_t>(targetOptions["shader_model"]) : 0;
    } else {
        rejectUnknown({});
    }
    return std::make_unique<CompiledProgram>(
        vernonCompilerCompileMlirWithOptions(compiler.context, mlir.data(), mlir.size(), &options), target);
}

std::vector<std::unique_ptr<CompiledProgram>> compileCpuProgramResults(const std::vector<std::string> &modules,
                                                                       const nb::dict &targetOptions) {
    const CpuTargetOptionStrings cpu = parseCpuTargetOptions(targetOptions);

    std::vector<std::future<std::unique_ptr<CompiledProgram>>> futures;
    std::vector<std::unique_ptr<CompiledProgram>> results;
    futures.reserve(modules.size());
    results.reserve(modules.size());
    {
        nb::gil_scoped_release release;
        for (const std::string &module : modules) {
            futures.push_back(std::async(std::launch::async, [module, cpu] {
                Compiler compiler;
                VernonCompileOptions options{};
                options.struct_size = sizeof(options);
                options.target = VERNON_TARGET_CPU;
                options.as.cpu.triple = VernonStringView{cpu.triple.data(), cpu.triple.size()};
                options.as.cpu.processor = VernonStringView{cpu.processor.data(), cpu.processor.size()};
                options.as.cpu.features = VernonStringView{cpu.features.data(), cpu.features.size()};
                return std::make_unique<CompiledProgram>(
                    vernonCompilerCompileMlirWithOptions(compiler.context, module.data(), module.size(), &options),
                    VERNON_TARGET_CPU);
            }));
        }
        for (auto &future : futures)
            results.push_back(future.get());
    }
    return results;
}

struct PipelineParameterMetadata {
    uint32_t slot{};
    std::string name;
    VernonPipelineArgumentKind kind{};
    uint32_t elementByteSize{};
    uint32_t elementAlignment{};
    std::string layoutHash;
    std::vector<VernonValueLeafView> elementLeaves;
    VernonValueAccess access{};
    VernonImageBindingRole imageBindingRole{VERNON_IMAGE_BINDING_SAMPLED};
    VernonTextureFormat storageImageFormat{};
    std::vector<uint64_t> shape;
};

struct PipelineOutputMetadata {
    std::string name;
    VernonPipelineArgumentKind kind{};
    VernonDataType dtype{};
    VernonValueAccess access{};
    std::vector<uint64_t> shape;
    uint32_t location{};
};

PipelineParameterMetadata parameterMetadata(const VernonPipelineParameterView &view) {
    PipelineParameterMetadata result;
    result.slot = view.slot;
    result.name = stringView(view.name);
    result.kind = view.kind;
    result.elementByteSize = view.element_layout.byte_size;
    result.elementAlignment = view.element_layout.alignment;
    result.layoutHash = stringView(view.element_layout.layout_hash);
    if (view.element_layout.leaf_count)
        result.elementLeaves.assign(view.element_layout.leaves,
                                    view.element_layout.leaves + view.element_layout.leaf_count);
    result.access = view.access;
    if (view.rank)
        result.shape.assign(view.static_shape, view.static_shape + view.rank);
    return result;
}

PipelineOutputMetadata outputMetadata(const VernonPipelineOutputView &view) {
    PipelineOutputMetadata result;
    result.name = stringView(view.name);
    result.kind = view.kind;
    result.dtype = view.dtype;
    result.access = view.access;
    result.location = view.location;
    if (view.rank)
        result.shape.assign(view.static_shape, view.static_shape + view.rank);
    return result;
}

struct PythonRuntimeSubmission {
    explicit PythonRuntimeSubmission(VernonSubmission *value) : handle(value) {}
    ~PythonRuntimeSubmission() { vernonSubmissionDestroy(handle); }
    PythonRuntimeSubmission(const PythonRuntimeSubmission &) = delete;
    PythonRuntimeSubmission &operator=(const PythonRuntimeSubmission &) = delete;

    void wait() {
        if (vernonSubmissionWait(handle) != VERNON_STATUS_OK)
            throw std::runtime_error("pipeline submission failed");
    }

    uint32_t state() const {
        VernonSubmissionState value{};
        if (vernonSubmissionGetState(handle, &value) != VERNON_STATUS_OK)
            throw std::runtime_error("cannot query pipeline submission");
        return static_cast<uint32_t>(value);
    }

    VernonSubmission *handle{};
};

struct PreparedPipelineArgument {
    VernonPipelineArgument value{};
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;
    std::string layoutHash;
    std::vector<VernonValueLeafView> elementLeaves;
    nb::object owner;
};

struct PipelineInvocationBuilder {
    PipelineInvocationBuilder(Runtime *owner, VernonRuntimeContext *runtime, VernonLoadedPipeline *pipeline)
        : owner(owner), runtime(runtime), pipeline(pipeline) {}
    PipelineInvocationBuilder(const PipelineInvocationBuilder &) = delete;
    PipelineInvocationBuilder &operator=(const PipelineInvocationBuilder &) = delete;

    PipelineParameterMetadata resolveParameter(const nb::object &identifier) {
        VernonPipelineParameterView view{};
        if (nb::isinstance<nb::str>(identifier)) {
            const std::string name = nb::cast<std::string>(identifier);
            if (vernonRuntimeLoadedPipelineFindParameter(pipeline, {name.data(), name.size()}, &view) !=
                VERNON_STATUS_OK)
                throw std::invalid_argument("unknown pipeline parameter '" + name + "'");
            PipelineParameterMetadata metadata = parameterMetadata(view);
            if (view.kind == VERNON_PIPELINE_IMAGE) {
                VernonPipelineImageConstraintView constraint{};
                constraint.struct_size = sizeof(constraint);
                if (vernonRuntimeLoadedPipelineFindImageConstraint(pipeline, {name.data(), name.size()}, &constraint) !=
                    VERNON_STATUS_OK)
                    throw std::runtime_error("cannot query pipeline image constraint");
                metadata.imageBindingRole = constraint.binding_role;
                metadata.storageImageFormat = constraint.storage_format;
            }
            return metadata;
        }
        if (!nb::isinstance<nb::int_>(identifier))
            throw std::invalid_argument("pipeline parameter must be a name or slot");
        const uint32_t slot = nb::cast<uint32_t>(identifier);
        const size_t count = vernonRuntimeLoadedPipelineGetParameterCount(pipeline);
        for (size_t index = 0; index < count; ++index) {
            if (vernonRuntimeLoadedPipelineGetParameterByIndex(pipeline, index, &view) == VERNON_STATUS_OK &&
                view.slot == slot) {
                PipelineParameterMetadata metadata = parameterMetadata(view);
                if (view.kind == VERNON_PIPELINE_IMAGE) {
                    VernonPipelineImageConstraintView constraint{};
                    constraint.struct_size = sizeof(constraint);
                    if (vernonRuntimeLoadedPipelineGetImageConstraintByParameterIndex(pipeline, index, &constraint) !=
                        VERNON_STATUS_OK)
                        throw std::runtime_error("cannot query pipeline image constraint");
                    metadata.imageBindingRole = constraint.binding_role;
                    metadata.storageImageFormat = constraint.storage_format;
                }
                return metadata;
            }
        }
        throw std::invalid_argument("unknown pipeline parameter slot " + std::to_string(slot));
    }

    std::unique_ptr<PreparedPipelineArgument> createArgument(const PipelineParameterMetadata &parameter,
                                                             VernonPipelineArgumentKind kind) {
        if (parameter.kind != kind)
            throw std::invalid_argument("pipeline parameter '" + parameter.name + "' has a different reflected kind");
        auto prepared = std::make_unique<PreparedPipelineArgument>();
        PreparedPipelineArgument &result = *prepared;
        result.value.slot = parameter.slot;
        result.value.kind = kind;
        if (kind == VERNON_PIPELINE_TENSOR) {
            result.layoutHash = parameter.layoutHash;
            result.elementLeaves = parameter.elementLeaves;
            result.value.tensor.element_layout = {
                sizeof(VernonValueLayoutView), parameter.elementByteSize,
                parameter.elementAlignment,    {result.layoutHash.data(), result.layoutHash.size()},
                result.elementLeaves.data(),   result.elementLeaves.size(),
            };
        }
        return prepared;
    }

    PipelineInvocationBuilder &preparedArgument(PreparedPipelineArgument &prepared) {
        if (!slots.insert(prepared.value.slot).second)
            throw std::invalid_argument("pipeline parameter was already bound");
        arguments.push_back(&prepared);
        return *this;
    }

    PipelineInvocationBuilder &ownedArgument(std::unique_ptr<PreparedPipelineArgument> prepared) {
        PreparedPipelineArgument &value = *prepared;
        ownedArguments.push_back(std::move(prepared));
        return preparedArgument(value);
    }

    static VernonDataType numpyDataType(const nb::object &array) {
        const std::string name = nb::cast<std::string>(array.attr("dtype").attr("name"));
        if (name == "bool")
            return VERNON_DATA_BOOL;
        if (name == "uint8")
            return VERNON_DATA_U8;
        if (name == "int32")
            return VERNON_DATA_I32;
        if (name == "uint32")
            return VERNON_DATA_U32;
        if (name == "float16")
            return VERNON_DATA_F16;
        if (name == "float32")
            return VERNON_DATA_F32;
        if (name == "float64")
            return VERNON_DATA_F64;
        throw std::invalid_argument("unsupported NumPy pipeline dtype '" + name + "'");
    }

    std::unique_ptr<PreparedPipelineArgument> prepareHostTensor(const nb::object &identifier, const nb::object &array) {
        if (!nb::isinstance(array, nb::module_::import_("numpy").attr("ndarray")))
            throw std::invalid_argument("host tensor must be a NumPy ndarray");
        const PipelineParameterMetadata parameter = resolveParameter(identifier);
        auto prepared = createArgument(parameter, VERNON_PIPELINE_TENSOR);
        PreparedPipelineArgument &argument = *prepared;
        const std::vector<uint64_t> arrayShape = nb::cast<std::vector<uint64_t>>(array.attr("shape"));
        const std::vector<int64_t> arrayStrides = nb::cast<std::vector<int64_t>>(array.attr("strides"));
        if (arrayStrides.size() != arrayShape.size())
            throw std::invalid_argument("NumPy Tensor shape/stride mismatch");
        const size_t elementSize = nb::cast<size_t>(array.attr("dtype").attr("itemsize"));
        size_t rank = arrayShape.size();
        if (elementSize != parameter.elementByteSize) {
            size_t trailingSize = elementSize;
            while (rank && trailingSize < parameter.elementByteSize) {
                const size_t dimension = --rank;
                if (arrayStrides[dimension] != static_cast<int64_t>(trailingSize) ||
                    arrayShape[dimension] > std::numeric_limits<size_t>::max() / trailingSize)
                    throw std::invalid_argument("host tensor aggregate element storage must be contiguous");
                trailingSize *= static_cast<size_t>(arrayShape[dimension]);
            }
            if (trailingSize != parameter.elementByteSize)
                throw std::invalid_argument("host tensor element size does not match pipeline reflection");
        }
        argument.shape.assign(arrayShape.begin(), arrayShape.begin() + static_cast<std::ptrdiff_t>(rank));
        argument.strides.assign(arrayStrides.begin(), arrayStrides.begin() + static_cast<std::ptrdiff_t>(rank));
        for (uint64_t extent : argument.shape)
            if (!extent)
                throw std::invalid_argument("NumPy Tensor dimensions must be positive");
        VernonTensorView layoutProbe{};
        layoutProbe.element_layout = argument.value.tensor.element_layout;
        layoutProbe.rank = static_cast<uint32_t>(rank);
        layoutProbe.shape = argument.shape.data();
        layoutProbe.byte_strides = argument.strides.data();
        size_t before = 0;
        size_t after = 0;
        size_t span = 0;
        if (!vernon::runtime::tensorRelativeByteBounds(layoutProbe, before, after) ||
            !vernon::runtime::tensorRequiredSpan(layoutProbe, span))
            throw std::invalid_argument("NumPy Tensor byte span overflows");
        if (array.attr("dtype").attr("fields").is_none() && parameter.elementLeaves.size() == 1 &&
            parameter.elementLeaves[0].scalar_count == 1 && parameter.elementLeaves[0].byte_offset == 0 &&
            numpyDataType(array) != static_cast<VernonDataType>(parameter.elementLeaves[0].dtype))
            throw std::invalid_argument("host tensor dtype does not match pipeline reflection");
        argument.owner = array;
        const uintptr_t data = nb::cast<uintptr_t>(array.attr("ctypes").attr("data"));
        uintptr_t allocation = data - before;
        size_t allocationSize = span;
        nb::object base = array.attr("base");
        while (!base.is_none() && nb::isinstance(base, nb::module_::import_("numpy").attr("ndarray"))) {
            const uintptr_t candidate = nb::cast<uintptr_t>(base.attr("ctypes").attr("data"));
            const size_t candidateSize = nb::cast<size_t>(base.attr("nbytes"));
            if (candidate > data || data - candidate > candidateSize || before > data - candidate ||
                after + parameter.elementByteSize > candidateSize - (data - candidate))
                break;
            allocation = candidate;
            allocationSize = candidateSize;
            base = base.attr("base");
        }
        argument.value.tensor.struct_size = sizeof(VernonTensorView);
        argument.value.tensor.storage = VERNON_TENSOR_HOST;
        argument.value.tensor.host_data = reinterpret_cast<const void *>(allocation);
        argument.value.tensor.access = parameter.access;
        argument.value.tensor.rank = static_cast<uint32_t>(argument.shape.size());
        argument.value.tensor.shape = argument.shape.data();
        argument.value.tensor.byte_strides = argument.strides.data();
        argument.value.tensor.byte_offset = data - allocation;
        argument.value.tensor.byte_size = allocationSize;
        if (argument.value.tensor.access != VERNON_ACCESS_READ &&
            !vernon::runtime::tensorByteLayoutInjective(argument.value.tensor))
            throw std::invalid_argument("writable NumPy Tensor must have an internally injective layout");
        return prepared;
    }

    PipelineInvocationBuilder &hostTensor(const nb::object &identifier, const nb::object &array) {
        return ownedArgument(prepareHostTensor(identifier, array));
    }

    std::unique_ptr<PreparedPipelineArgument> prepareRhiTensor(const nb::object &identifier, RhiBuffer *buffer,
                                                               uint32_t access, const std::vector<uint64_t> &shape,
                                                               const std::vector<int64_t> &strides, size_t offset) {
        if (!buffer || shape.size() != strides.size() || shape.empty())
            throw std::invalid_argument("RHI Tensor shape and strides must have equal non-zero rank");
        const PipelineParameterMetadata parameter = resolveParameter(identifier);
        auto prepared = createArgument(parameter, VERNON_PIPELINE_TENSOR);
        PreparedPipelineArgument &argument = *prepared;
        argument.owner = nb::cast(buffer, nb::rv_policy::reference);
        argument.shape = shape;
        argument.strides = strides;
        argument.value.tensor.struct_size = sizeof(VernonTensorView);
        argument.value.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
        if (vernonRuntimeReferenceRhiBuffer(runtime, buffer->handle, 0, buffer->size,
                                            &argument.value.tensor.resource) != VERNON_STATUS_OK)
            throw std::invalid_argument("RHI buffer belongs to another Runtime device");
        argument.value.tensor.access = static_cast<VernonValueAccess>(access);
        argument.value.tensor.rank = static_cast<uint32_t>(shape.size());
        argument.value.tensor.shape = argument.shape.data();
        argument.value.tensor.byte_strides = argument.strides.data();
        argument.value.tensor.byte_offset = offset;
        argument.value.tensor.byte_size = buffer->size;
        if (argument.value.tensor.access != VERNON_ACCESS_READ &&
            !vernon::runtime::tensorByteLayoutInjective(argument.value.tensor))
            throw std::invalid_argument("writable RHI Tensor must have an internally injective layout");
        return prepared;
    }

    PipelineInvocationBuilder &rhiTensor(const nb::object &identifier, RhiBuffer *buffer, uint32_t access,
                                         const std::vector<uint64_t> &shape, const std::vector<int64_t> &strides,
                                         size_t offset) {
        return ownedArgument(prepareRhiTensor(identifier, buffer, access, shape, strides, offset));
    }

    std::unique_ptr<PreparedPipelineArgument> prepareRhiTexture(const nb::object &identifier, RhiImageView *view) {
        const PipelineParameterMetadata parameter = resolveParameter(identifier);
        const bool storage = parameter.imageBindingRole == VERNON_IMAGE_BINDING_STORAGE;
        const uint32_t requiredUsage = storage ? VERNON_RHI_IMAGE_STORAGE : VERNON_RHI_IMAGE_SAMPLED;
        if (!view || !(view->image->usage & requiredUsage))
            throw std::invalid_argument("RHI texture usage does not match the pipeline parameter");
        if (storage && view->format != parameter.storageImageFormat)
            throw std::invalid_argument("RHI texture format does not match the storage texture parameter");
        auto prepared = createArgument(parameter, VERNON_PIPELINE_IMAGE);
        PreparedPipelineArgument &argument = *prepared;
        argument.owner = nb::cast(view, nb::rv_policy::reference);
        if (vernonRuntimeReferenceRhiImageView(runtime, view->handle, &argument.value.image.view) != VERNON_STATUS_OK)
            throw std::invalid_argument("RHI image view belongs to another Runtime device");
        return prepared;
    }

    PipelineInvocationBuilder &rhiTexture(const nb::object &identifier, RhiImageView *view) {
        return ownedArgument(prepareRhiTexture(identifier, view));
    }

    std::unique_ptr<PreparedPipelineArgument> prepareRhiSampler(const nb::object &identifier, RhiSampler *sampler) {
        if (!sampler)
            throw std::invalid_argument("RHI sampler is null");
        const PipelineParameterMetadata parameter = resolveParameter(identifier);
        auto prepared = createArgument(parameter, VERNON_PIPELINE_SAMPLER);
        PreparedPipelineArgument &argument = *prepared;
        argument.owner = nb::cast(sampler, nb::rv_policy::reference);
        if (vernonRuntimeReferenceRhiSampler(runtime, sampler->handle, &argument.value.resource) != VERNON_STATUS_OK)
            throw std::invalid_argument("RHI sampler belongs to another Runtime device");
        return prepared;
    }

    PipelineInvocationBuilder &rhiSampler(const nb::object &identifier, RhiSampler *sampler) {
        return ownedArgument(prepareRhiSampler(identifier, sampler));
    }

    PipelineInvocationBuilder &rhiColorAttachment(uint32_t location, RhiImageView *view, uint32_t loadOperation,
                                                  uint32_t storeOperation, const std::array<float, 4> &clearColor) {
        if (!view || !(view->image->usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT) ||
            (view->format == VERNON_TEXTURE_D32_FLOAT || view->format == VERNON_TEXTURE_D32_FLOAT_S8_UINT) ||
            loadOperation > VERNON_RHI_LOAD_DISCARD || storeOperation > VERNON_RHI_STORE_DISCARD)
            throw std::invalid_argument("RHI color attachment is null");
        VernonColorAttachment attachment{};
        attachment.location = location;
        if (vernonRuntimeReferenceRhiImageView(runtime, view->handle, &attachment.view) != VERNON_STATUS_OK)
            throw std::invalid_argument("RHI color attachment belongs to another Runtime device");
        attachment.load_operation = static_cast<VernonRuntimeProviderLoadOperation>(loadOperation);
        attachment.store_operation = static_cast<VernonRuntimeProviderStoreOperation>(storeOperation);
        std::copy(clearColor.begin(), clearColor.end(), attachment.clear_color);
        attachments.push_back(attachment);
        return *this;
    }

    PipelineInvocationBuilder &rhiDepthAttachment(RhiImageView *view, uint32_t loadOperation, uint32_t storeOperation,
                                                  float clearDepth) {
        if (!view || !(view->image->usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT) ||
            (view->format != VERNON_TEXTURE_D32_FLOAT && view->format != VERNON_TEXTURE_D32_FLOAT_S8_UINT) ||
            loadOperation > VERNON_RHI_LOAD_DISCARD || storeOperation > VERNON_RHI_STORE_DISCARD || clearDepth < 0.0f ||
            clearDepth > 1.0f)
            throw std::invalid_argument("RHI depth attachment must use D32 format");
        depthAttachment = {};
        if (vernonRuntimeReferenceRhiImageView(runtime, view->handle, &depthAttachment.view) != VERNON_STATUS_OK)
            throw std::invalid_argument("RHI depth attachment belongs to another Runtime device");
        depthAttachment.load_operation = static_cast<VernonRuntimeProviderLoadOperation>(loadOperation);
        depthAttachment.store_operation = static_cast<VernonRuntimeProviderStoreOperation>(storeOperation);
        depthAttachment.clear_depth = clearDepth;
        hasDepthAttachment = true;
        return *this;
    }

    PipelineInvocationBuilder &rhiIndexBinding(RhiBuffer *buffer, uint32_t count, size_t offset) {
        if (!buffer || offset > buffer->size)
            throw std::invalid_argument("RHI index buffer is null");
        index = {};
        if (vernonRuntimeReferenceRhiBuffer(runtime, buffer->handle, offset, buffer->size - offset, &index.resource) !=
            VERNON_STATUS_OK)
            throw std::invalid_argument("RHI index buffer belongs to another Runtime device");
        index.type = VERNON_INDEX_U32;
        index.offset = offset;
        index.index_count = count;
        hasIndex = true;
        return *this;
    }

    PipelineInvocationBuilder &setTopology(uint32_t value) {
        if (value > static_cast<uint32_t>(VERNON_TOPOLOGY_POINT_LIST))
            throw std::invalid_argument("invalid primitive topology");
        topology = static_cast<VernonPrimitiveTopology>(value);
        return *this;
    }

    PipelineInvocationBuilder &counts(uint32_t vertices, uint32_t instances) {
        vertexCount = vertices;
        instanceCount = instances;
        return *this;
    }

    PipelineInvocationBuilder &grid(uint32_t x, uint32_t y, uint32_t z) {
        computeGrid = {x, y, z};
        return *this;
    }

    PipelineInvocationBuilder &setViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
        viewport[0] = x;
        viewport[1] = y;
        viewport[2] = width;
        viewport[3] = height;
        return *this;
    }

    PipelineInvocationBuilder &setScissor(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
        scissor[0] = x;
        scissor[1] = y;
        scissor[2] = width;
        scissor[3] = height;
        return *this;
    }

    std::unique_ptr<PythonRuntimeSubmission> submit(VernonRuntimeProviderObject *encoder) {
        std::vector<VernonPipelineArgument> values;
        values.reserve(arguments.size());
        for (const PreparedPipelineArgument *argument : arguments)
            values.push_back(argument->value);
        VernonPipelineInvocation invocation{};
        invocation.struct_size = sizeof(invocation);
        invocation.abi_version = VERNON_PIPELINE_VERSION;
        invocation.arguments = values.empty() ? nullptr : values.data();
        invocation.argument_count = values.size();
        invocation.index_binding = hasIndex ? &index : nullptr;
        invocation.color_attachments = attachments.empty() ? nullptr : attachments.data();
        invocation.color_attachment_count = attachments.size();
        invocation.depth_attachment = hasDepthAttachment ? &depthAttachment : nullptr;
        invocation.topology = topology;
        invocation.vertex_count = vertexCount;
        invocation.instance_count = instanceCount;
        invocation.compute_grid = computeGrid;
        std::memcpy(invocation.viewport, viewport, sizeof(viewport));
        std::memcpy(invocation.scissor, scissor, sizeof(scissor));
        if (encoder) {
            if (vernonRuntimePipelineEncode(*encoder, pipeline, &invocation) != VERNON_STATUS_OK)
                throw std::runtime_error("pipeline encoding failed: " + stringView(vernonRuntimeGetLastError(runtime)));
            return {};
        }
        VernonSubmission *submission{};
        if (vernonRuntimePipelineSubmit(pipeline, &invocation, &submission) != VERNON_STATUS_OK)
            throw std::runtime_error("pipeline submission failed: " + stringView(vernonRuntimeGetLastError(runtime)));
        return std::make_unique<PythonRuntimeSubmission>(submission);
    }

    std::unique_ptr<PythonRuntimeSubmission> submit() { return submit(nullptr); }

    template <typename Encoder> void encode(const Encoder &encoder) {
        VernonRuntimeProviderObject providerEncoder{};
        if (vernonRuntimeReferenceRhiCommandEncoder(runtime, encoder.native(), &providerEncoder) != VERNON_STATUS_OK)
            throw std::invalid_argument("command encoder belongs to another Runtime device");
        (void)submit(&providerEncoder);
    }

    Runtime *owner{};
    VernonRuntimeContext *runtime{};
    VernonLoadedPipeline *pipeline{};
    std::vector<PreparedPipelineArgument *> arguments;
    std::vector<std::unique_ptr<PreparedPipelineArgument>> ownedArguments;
    std::unordered_set<uint32_t> slots;
    std::vector<VernonColorAttachment> attachments;
    VernonDepthAttachment depthAttachment{};
    VernonIndexBinding index{};
    bool hasDepthAttachment{};
    bool hasIndex{};
    VernonPrimitiveTopology topology{VERNON_TOPOLOGY_TRIANGLE_LIST};
    uint32_t vertexCount{};
    uint32_t instanceCount{};
    VernonLaunchSize computeGrid{};
    uint32_t viewport[4]{};
    uint32_t scissor[4]{};
};

const char *numpyDtypeName(VernonDataType dtype) {
    if (dtype == VERNON_DATA_BOOL)
        return "bool_";
    if (dtype == VERNON_DATA_U8)
        return "uint8";
    if (dtype == VERNON_DATA_I32)
        return "int32";
    if (dtype == VERNON_DATA_U32)
        return "uint32";
    if (dtype == VERNON_DATA_F16)
        return "float16";
    if (dtype == VERNON_DATA_F32)
        return "float32";
    if (dtype == VERNON_DATA_F64)
        return "float64";
    throw std::invalid_argument("Python autodiff Value dtype is unsupported");
}

size_t autodiffDtypeSize(VernonDataType dtype) {
    if (dtype == VERNON_DATA_BOOL || dtype == VERNON_DATA_U8)
        return 1;
    if (dtype == VERNON_DATA_F16)
        return 2;
    if (dtype == VERNON_DATA_I32 || dtype == VERNON_DATA_U32 || dtype == VERNON_DATA_F32)
        return 4;
    if (dtype == VERNON_DATA_F64)
        return 8;
    throw std::invalid_argument("Python autodiff Value dtype is unsupported");
}

VernonDataType autodiffTangentDtype(VernonDataType primal) {
    return primal == VERNON_DATA_F16 ? VERNON_DATA_F32 : primal;
}

std::string formatShape(const std::vector<uint64_t> &shape) {
    std::string result = "[";
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index)
            result += ", ";
        result += std::to_string(shape[index]);
    }
    return result + "]";
}

struct PythonAdViewDescriptor {
    uintptr_t allocationBegin{};
    size_t allocationSize{};
    size_t byteOffset{};
    VernonDataType dtype{};
    bool writable{};
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;

    VernonTensorView tensorView() const {
        VernonTensorView tensor{};
        tensor.struct_size = sizeof(VernonTensorView);
        tensor.storage = VERNON_TENSOR_HOST;
        tensor.host_data = reinterpret_cast<const void *>(allocationBegin);
        tensor.element_layout = vernonRuntimeGetScalarValueLayout(dtype);
        tensor.access = writable ? VERNON_ACCESS_READ_WRITE : VERNON_ACCESS_READ;
        tensor.rank = static_cast<uint32_t>(shape.size());
        tensor.shape = shape.data();
        tensor.byte_strides = strides.data();
        tensor.byte_offset = byteOffset;
        tensor.byte_size = allocationSize;
        return tensor;
    }
};

PythonAdViewDescriptor validatePythonAdOriginalView(const std::string &path, VernonDataType dtype,
                                                    const std::vector<uint64_t> &expectedShape, const nb::object &array,
                                                    bool writable) {
    std::vector<uint64_t> shape = nb::cast<std::vector<uint64_t>>(array.attr("shape"));
    if (shape != expectedShape)
        throw std::invalid_argument("Python autodiff Value '" + path + "' shape " + formatShape(shape) +
                                    " does not match reflection " + formatShape(expectedShape));
    const size_t itemSize = nb::cast<size_t>(array.attr("dtype").attr("itemsize"));
    if (itemSize != autodiffDtypeSize(dtype) ||
        !nb::cast<bool>(
            array.attr("dtype").attr("__eq__")(nb::module_::import_("numpy").attr("dtype")(numpyDtypeName(dtype)))))
        throw std::invalid_argument("Python autodiff Value '" + path + "' dtype does not match reflection");
    std::vector<int64_t> strides = nb::cast<std::vector<int64_t>>(array.attr("strides"));
    if (strides.size() != shape.size())
        throw std::invalid_argument("Python autodiff Value '" + path + "' has an invalid stride rank");

    nb::object allocation = array;
    std::unordered_set<PyObject *> visited;
    visited.insert(allocation.ptr());
    while (nb::hasattr(allocation, "base")) {
        nb::object base = allocation.attr("base");
        if (base.is_none() || !visited.insert(base.ptr()).second)
            break;
        allocation = std::move(base);
    }
    nb::object numpy = nb::module_::import_("numpy");
    nb::object allocationArray = numpy.attr("asarray")(allocation);
    nb::tuple allocationBounds = nb::cast<nb::tuple>(numpy.attr("byte_bounds")(allocationArray));
    const uintptr_t allocationBegin = nb::cast<uintptr_t>(allocationBounds[0]);
    const uintptr_t allocationEnd = nb::cast<uintptr_t>(allocationBounds[1]);
    const uintptr_t data = nb::cast<uintptr_t>(array.attr("ctypes").attr("data"));
    if (allocationEnd < allocationBegin || data < allocationBegin || data > allocationEnd)
        throw std::invalid_argument("Python autodiff Value '" + path + "' has an invalid allocation base");
    const uintptr_t allocationSize = allocationEnd - allocationBegin;
    if (allocationSize > std::numeric_limits<size_t>::max())
        throw std::invalid_argument("Python autodiff Value '" + path + "' allocation size overflows");

    PythonAdViewDescriptor descriptor;
    descriptor.allocationBegin = allocationBegin;
    descriptor.allocationSize = static_cast<size_t>(allocationSize);
    descriptor.byteOffset = static_cast<size_t>(data - allocationBegin);
    descriptor.dtype = dtype;
    descriptor.writable = writable;
    descriptor.shape = std::move(shape);
    descriptor.strides = std::move(strides);
    const VernonTensorView tensor = descriptor.tensorView();
    if (!vernon::runtime::tensorElementCount(tensor))
        throw std::invalid_argument("Python autodiff Value '" + path + "' shape overflows");
    if (!vernon::runtime::tensorLogicalByteSize(tensor))
        throw std::invalid_argument("Python autodiff Value '" + path + "' byte size overflows");
    if (!vernon::runtime::tensorFitsAllocation(tensor))
        throw std::invalid_argument("Python autodiff Value '" + path + "' layout is outside its owner allocation");
    if (writable && !vernon::runtime::tensorByteLayoutInjective(tensor))
        throw std::invalid_argument("writable Python autodiff Value '" + path +
                                    "' must have an internally injective layout");
    return descriptor;
}

bool pythonAdViewsOverlap(const PythonAdViewDescriptor &left, const PythonAdViewDescriptor &right) {
    return vernon::runtime::tensorViewsHaveWritableOverlap(left.tensorView(), right.tensorView());
}

struct PythonAdValue {
    std::string path;
    nb::object source;
    nb::object array;
    std::vector<uint64_t> shape;
    bool writable{};
    PythonAdViewDescriptor originalView;
    VernonAdValue value{};

    PythonAdValue(std::string path, VernonDataType dtype, std::vector<uint64_t> shape, const nb::object &source,
                  uint32_t access = VERNON_ACCESS_READ)
        : path(std::move(path)), source(nb::module_::import_("numpy").attr("asarray")(source)), array(this->source),
          shape(std::move(shape)), writable(access != VERNON_ACCESS_READ) {
        originalView = validatePythonAdOriginalView(this->path, dtype, this->shape, this->source, writable);
        if (!nb::cast<bool>(array.attr("flags").attr("c_contiguous"))) {
            nb::object numpy = nb::module_::import_("numpy");
            array =
                access == VERNON_ACCESS_WRITE
                    ? numpy.attr("empty")(this->shape, nb::arg("dtype") = numpy.attr("dtype")(numpyDtypeName(dtype)))
                    : numpy.attr("ascontiguousarray")(array);
        }
        size_t scalarCount = 1;
        for (uint64_t extent : this->shape) {
            if (scalarCount && extent > std::numeric_limits<size_t>::max() / scalarCount)
                throw std::invalid_argument("Python autodiff Value shape overflows");
            scalarCount *= static_cast<size_t>(extent);
        }
        const size_t scalarSize = autodiffDtypeSize(dtype);
        if (!scalarSize || (scalarCount && scalarSize > std::numeric_limits<size_t>::max() / scalarCount))
            throw std::invalid_argument("Python autodiff Value byte size overflows");
        if (this->shape.size() > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("Python autodiff Value rank overflows");
        value.struct_size = sizeof(value);
        value.path = {this->path.data(), this->path.size()};
        value.dtype = dtype;
        value.data = reinterpret_cast<void *>(nb::cast<uintptr_t>(array.attr("ctypes").attr("data")));
        value.size = scalarCount * scalarSize;
        value.rank = static_cast<uint32_t>(this->shape.size());
        value.shape = this->shape.empty() ? nullptr : this->shape.data();
    }

    PythonAdValue(PythonAdValue &&other) noexcept
        : path(std::move(other.path)), source(std::move(other.source)), array(std::move(other.array)),
          shape(std::move(other.shape)), writable(other.writable), originalView(std::move(other.originalView)),
          value(other.value) {
        value.path = {path.data(), path.size()};
        value.shape = shape.empty() ? nullptr : shape.data();
    }
    void commit() {
        if (writable && source.ptr() != array.ptr())
            nb::module_::import_("numpy").attr("copyto")(source, array);
    }
    PythonAdValue(const PythonAdValue &) = delete;
    PythonAdValue &operator=(const PythonAdValue &) = delete;
};

struct PythonAdMetadata {
    std::string path;
    std::string binding;
    VernonDataType dtype{};
    std::vector<uint64_t> shape;
};

PythonAdMetadata adInputLeafMetadata(VernonLoadedPipeline *pipeline, const PipelineParameterMetadata &parameter,
                                     size_t leafIndex, VernonPipelineValueLeafView *reflected = nullptr) {
    VernonPipelineValueLeafView leaf{};
    leaf.struct_size = sizeof(leaf);
    const VernonStringView parameterName{parameter.name.data(), parameter.name.size()};
    if (vernonRuntimeLoadedPipelineGetParameterValueLeaf(pipeline, parameterName, leafIndex, &leaf) != VERNON_STATUS_OK)
        throw std::runtime_error("cannot read autodiff input leaf metadata");
    PythonAdMetadata result{parameter.name, parameter.name, static_cast<VernonDataType>(leaf.value.dtype), {}};
    for (size_t index = 0; index < leaf.path_count; ++index) {
        const VernonValuePathComponentView &component = leaf.path[index];
        result.path += ".";
        if (component.kind == VERNON_VALUE_PATH_FIELD)
            result.path += stringView(component.field);
        else if (component.kind == VERNON_VALUE_PATH_INDEX)
            result.path += std::to_string(component.index);
        else
            throw std::runtime_error("autodiff input leaf has an invalid path");
    }
    if (parameter.kind == VERNON_PIPELINE_TENSOR) {
        result.shape = parameter.shape;
        if (leaf.static_rank)
            result.shape.insert(result.shape.end(), leaf.static_shape, leaf.static_shape + leaf.static_rank);
    } else if (leaf.static_rank) {
        result.shape.assign(leaf.static_shape, leaf.static_shape + leaf.static_rank);
    }
    if (reflected)
        *reflected = leaf;
    return result;
}

nb::object resolveAdInputLeaf(const nb::dict &bindings, const PipelineParameterMetadata &parameter,
                              const VernonPipelineValueLeafView &leaf) {
    nb::str root(parameter.name.c_str());
    if (!bindings.contains(root))
        throw std::invalid_argument("missing autodiff binding '" + parameter.name + "'");
    nb::object value = nb::borrow<nb::object>(bindings[root]);
    if (nb::hasattr(value, "_native_host_array")) {
        nb::object array = value.attr("_native_host_array")();
        nb::object fields = array.attr("dtype").attr("fields");
        if (!fields.is_none()) {
            nb::str key("__value");
            if (nb::cast<bool>(fields.attr("__contains__")(key)))
                array = array.attr("__getitem__")(key);
        }
        value = std::move(array);
        if (leaf.path_count == 0)
            return value;
    }
    for (size_t index = 0; index < leaf.path_count; ++index) {
        const VernonValuePathComponentView &component = leaf.path[index];
        if (component.kind == VERNON_VALUE_PATH_FIELD) {
            const std::string field = stringView(component.field);
            if (nb::isinstance<nb::dict>(value)) {
                nb::dict mapping = nb::cast<nb::dict>(value);
                nb::str key(field.c_str());
                if (!mapping.contains(key))
                    throw std::invalid_argument("autodiff Struct binding is missing field '" + field + "'");
                value = nb::borrow<nb::object>(mapping[key]);
            } else {
                if (nb::hasattr(value, field.c_str()))
                    value = value.attr(field.c_str());
                else if (nb::hasattr(value, "__getitem__"))
                    value = value.attr("__getitem__")(field);
                else
                    throw std::invalid_argument("autodiff Struct binding has no field '" + field + "'");
            }
        } else if (component.kind == VERNON_VALUE_PATH_INDEX) {
            nb::object fields = nb::hasattr(value, "dtype") ? value.attr("dtype").attr("fields") : nb::none();
            const std::string index = std::to_string(component.index);
            nb::str key(index.c_str());
            if (!fields.is_none() && nb::cast<bool>(fields.attr("__contains__")(key)))
                value = value.attr("__getitem__")(key);
            else
                value = nb::module_::import_("numpy").attr("take")(value, component.index,
                                                                   nb::arg("axis") = parameter.shape.size());
        } else {
            throw std::runtime_error("autodiff input leaf has an invalid path");
        }
    }
    return value;
}

struct PythonPullback {
    PythonPullback(VernonRuntimeContext *runtime, VernonPullback *handle, std::vector<PythonAdMetadata> gradients,
                   std::vector<PythonAdMetadata> cotangents, nb::object pipelineOwner, nb::dict bindings)
        : runtime(runtime), handle(handle), gradients(std::move(gradients)), cotangents(std::move(cotangents)),
          pipelineOwner(std::move(pipelineOwner)), bindings(std::move(bindings)) {}
    ~PythonPullback() { vernonPullbackDestroy(handle); }

    nb::dict apply(const nb::object &cotangent) {
        std::deque<PythonAdValue> gradientValues;
        std::vector<VernonAdValue> gradientViews;
        nb::dict result;
        for (const PythonAdMetadata &gradient : gradients) {
            nb::object zeros = nb::module_::import_("numpy").attr("zeros")(
                gradient.shape, nb::module_::import_("numpy").attr(numpyDtypeName(gradient.dtype)));
            gradientValues.emplace_back(gradient.path, gradient.dtype, gradient.shape, zeros);
            gradientViews.push_back(gradientValues.back().value);
        }
        VernonAdValueSet gradientSet{sizeof(VernonAdValueSet), gradientViews.data(), gradientViews.size(), {}};

        std::deque<PythonAdValue> seeds;
        std::vector<VernonAdValue> seedViews;
        VernonAdValueSet seedSet{};
        const VernonAdValueSet *seedView = nullptr;
        if (!cotangent.is_none()) {
            if (cotangents.size() == 1) {
                const PythonAdMetadata &metadata = cotangents.front();
                seeds.emplace_back(metadata.path, metadata.dtype, metadata.shape, cotangent);
            } else {
                if (!nb::isinstance<nb::dict>(cotangent))
                    throw std::invalid_argument("aggregate pullback cotangent must be a leaf-path dictionary");
                nb::dict values = nb::cast<nb::dict>(cotangent);
                if (values.size() != cotangents.size())
                    throw std::invalid_argument("aggregate pullback requires every output cotangent leaf");
                for (const PythonAdMetadata &metadata : cotangents) {
                    nb::str path(metadata.path.c_str());
                    if (!values.contains(path))
                        throw std::invalid_argument("missing output cotangent leaf '" + metadata.path + "'");
                    seeds.emplace_back(metadata.path, metadata.dtype, metadata.shape,
                                       nb::borrow<nb::object>(values[path]));
                }
            }
            for (PythonAdValue &seed : seeds)
                seedViews.push_back(seed.value);
            seedSet = {sizeof(VernonAdValueSet), seedViews.data(), seedViews.size(), {}};
            seedView = &seedSet;
        }
        if (vernonPullbackApply(handle, seedView, &gradientSet) != VERNON_STATUS_OK)
            throw std::runtime_error("pullback application failed: " + stringView(vernonRuntimeGetLastError(runtime)));
        std::unordered_map<PyObject *, nb::object> gradientsByOwner;
        for (size_t index = 0; index < gradientValues.size(); ++index) {
            PythonAdValue &gradient = gradientValues[index];
            const PythonAdMetadata &metadata = gradients[index];
            nb::str path(gradient.path.c_str());
            nb::str bindingPath(metadata.binding.c_str());
            nb::object binding =
                bindings.contains(bindingPath) ? nb::borrow<nb::object>(bindings[bindingPath]) : nb::none();
            if (binding.is_none() || !nb::hasattr(binding, "_materialize_gradient")) {
                result[path] = gradient.array;
                continue;
            }
            nb::object owner = nb::hasattr(binding, "owner") ? binding.attr("owner") : binding;
            auto existing = gradientsByOwner.find(owner.ptr());
            nb::object materialized =
                existing == gradientsByOwner.end()
                    ? binding.attr("_materialize_gradient")(gradient.array, path)
                    : binding.attr("_materialize_gradient")(gradient.array, path, existing->second);
            if (existing == gradientsByOwner.end())
                gradientsByOwner.emplace(owner.ptr(), materialized);
            result[path] = std::move(materialized);
        }
        return result;
    }

    VernonRuntimeContext *runtime{};
    VernonPullback *handle{};
    std::vector<PythonAdMetadata> gradients;
    std::vector<PythonAdMetadata> cotangents;
    nb::object pipelineOwner;
    nb::dict bindings;
};

struct LoadedPipeline {
    LoadedPipeline(Runtime *owner, VernonRuntimeContext *runtime, VernonPipelineBundle *bundle,
                   VernonLoadedPipeline *pipeline, std::vector<SharedCompileResult> retainedResults = {})
        : owner(owner), runtime(runtime), bundle(bundle), pipeline(pipeline),
          retainedResults(std::move(retainedResults)) {}
    ~LoadedPipeline() {
        vernonRuntimeLoadedPipelineDestroy(pipeline);
        vernonRuntimePipelineBundleDestroy(bundle);
    }

    std::unique_ptr<PipelineInvocationBuilder> invocationBuilder() {
        return std::make_unique<PipelineInvocationBuilder>(owner, runtime, pipeline);
    }

    std::array<uint32_t, 3> workgroupSize() const {
        const VernonLaunchSize size = vernon::runtime::autodiffWorkgroupSize(pipeline);
        return {size.x, size.y, size.z};
    }

    nb::tuple vjp(uint32_t gridX, uint32_t gridY, uint32_t gridZ, const nb::dict &bindings, nb::object pipelineOwner) {
        if (!gridX || !gridY || !gridZ)
            throw std::invalid_argument("autodiff grid dimensions must be positive");
        const std::vector<PipelineParameterMetadata> metadata = parameters();
        if (bindings.size() != metadata.size())
            throw std::invalid_argument("autodiff bindings do not match pipeline parameters");
        std::deque<PythonAdValue> inputValues;
        std::vector<VernonAdValue> inputViews;
        std::vector<PythonAdMetadata> inputLeafMetadata;
        for (const PipelineParameterMetadata &parameter : metadata) {
            for (size_t leafIndex = 0; leafIndex < parameter.elementLeaves.size(); ++leafIndex) {
                VernonPipelineValueLeafView reflected{};
                PythonAdMetadata leaf = adInputLeafMetadata(pipeline, parameter, leafIndex, &reflected);
                nb::object value = resolveAdInputLeaf(bindings, parameter, reflected);
                const std::vector<uint64_t> actualShape = nb::cast<std::vector<uint64_t>>(value.attr("shape"));
                if (actualShape.size() == leaf.shape.size())
                    for (size_t dimension = 0; dimension < leaf.shape.size(); ++dimension)
                        if (!leaf.shape[dimension])
                            leaf.shape[dimension] = actualShape[dimension];
                inputValues.emplace_back(leaf.path, leaf.dtype, leaf.shape, value, parameter.access);
                inputViews.push_back(inputValues.back().value);
                inputLeafMetadata.push_back(std::move(leaf));
            }
        }
        for (size_t left = 0; left < inputValues.size(); ++left)
            for (size_t right = left + 1; right < inputValues.size(); ++right) {
                const PythonAdValue &leftValue = inputValues[left];
                const PythonAdValue &rightValue = inputValues[right];
                if (pythonAdViewsOverlap(leftValue.originalView, rightValue.originalView))
                    throw std::invalid_argument("Python autodiff Values '" + leftValue.path + "' and '" +
                                                rightValue.path + "' have incompatible overlapping views");
            }
        VernonAdValueSet inputSet{sizeof(VernonAdValueSet), inputViews.data(), inputViews.size(), {}};

        const VernonLaunchSize workgroup = vernon::runtime::autodiffWorkgroupSize(pipeline);
        if (gridX > UINT32_MAX / workgroup.x || gridY > UINT32_MAX / workgroup.y || gridZ > UINT32_MAX / workgroup.z)
            throw std::invalid_argument("autodiff invocation extent overflows");
        const uint32_t extentX = gridX * workgroup.x;
        const uint32_t extentY = gridY * workgroup.y;
        const uint32_t extentZ = gridZ * workgroup.z;
        if (extentX > SIZE_MAX / extentY || static_cast<size_t>(extentX) * extentY > SIZE_MAX / extentZ)
            throw std::invalid_argument("autodiff grid size overflows");
        const size_t carrierCount = static_cast<size_t>(extentX) * extentY * extentZ;
        const size_t outputCount = vernonRuntimeLoadedPipelineGetAdOutputCount(pipeline);
        const size_t cotangentCount = vernonRuntimeLoadedPipelineGetAdCotangentCount(pipeline);
        if (!outputCount || !cotangentCount)
            throw std::runtime_error("pipeline has no consistent autodiff output/cotangent signature");
        auto prependCarrierDimensions = [&](std::vector<uint64_t> &shape) {
            if (carrierCount > 1)
                shape.insert(shape.begin(), {extentZ, extentY, extentX});
        };
        const bool storageObjectives = vernon::runtime::hasAutodiffStorageObjectives(pipeline);
        auto materializeMetadata = [&](const VernonAdValueMetadataView &value) {
            const std::string path = stringView(value.path);
            PythonAdMetadata result{path, path, value.dtype, {}};
            if (value.rank)
                result.shape.assign(value.shape, value.shape + value.rank);
            prependCarrierDimensions(result.shape);
            return result;
        };
        std::vector<PythonAdMetadata> outputMetadata;
        std::vector<PythonAdMetadata> cotangentMetadata;
        outputMetadata.reserve(outputCount);
        cotangentMetadata.reserve(cotangentCount);
        std::deque<PythonAdValue> outputValues;
        std::vector<VernonAdValue> outputViews;
        nb::dict aggregateOutput;
        for (size_t index = 0; index < outputCount; ++index) {
            VernonAdValueMetadataView outputValue{};
            outputValue.struct_size = sizeof(outputValue);
            if (vernonRuntimeLoadedPipelineGetAdOutputByIndex(pipeline, index, &outputValue) != VERNON_STATUS_OK)
                throw std::runtime_error("pipeline has no consistent autodiff output/cotangent signature");
            outputMetadata.push_back(materializeMetadata(outputValue));
            const PythonAdMetadata &metadata = outputMetadata.back();
            if (storageObjectives)
                continue;
            nb::object zeros = nb::module_::import_("numpy").attr("zeros")(
                metadata.shape, nb::module_::import_("numpy").attr(numpyDtypeName(metadata.dtype)));
            outputValues.emplace_back(metadata.path, metadata.dtype, metadata.shape, zeros);
            outputViews.push_back(outputValues.back().value);
            aggregateOutput[nb::str(metadata.path.c_str())] = outputValues.back().array;
        }
        for (size_t index = 0; index < cotangentCount; ++index) {
            VernonAdValueMetadataView cotangentValue{};
            cotangentValue.struct_size = sizeof(cotangentValue);
            if (vernonRuntimeLoadedPipelineGetAdCotangentByIndex(pipeline, index, &cotangentValue) != VERNON_STATUS_OK)
                throw std::runtime_error("pipeline has no consistent autodiff output/cotangent signature");
            PythonAdMetadata cotangent = materializeMetadata(cotangentValue);
            const auto input =
                std::find_if(inputLeafMetadata.begin(), inputLeafMetadata.end(),
                             [&](const PythonAdMetadata &candidate) { return candidate.path == cotangent.path; });
            if (input != inputLeafMetadata.end()) {
                cotangent.binding = input->binding;
                cotangent.shape = input->shape;
                prependCarrierDimensions(cotangent.shape);
            }
            cotangentMetadata.push_back(std::move(cotangent));
        }
        if (storageObjectives) {
            outputValues.clear();
            outputViews.clear();
            aggregateOutput.clear();
        }
        VernonAdValueSet outputSet{sizeof(VernonAdValueSet), outputViews.data(), outputViews.size(), {}};
        VernonPullback *pullback = nullptr;
        if (vernonAdPipelineForward(pipeline, {gridX, gridY, gridZ}, &inputSet, &outputSet, &pullback) !=
            VERNON_STATUS_OK)
            throw std::runtime_error("autodiff forward invocation failed: " +
                                     stringView(vernonRuntimeGetLastError(runtime)));
        for (PythonAdValue &input : inputValues)
            input.commit();

        std::vector<PythonAdMetadata> gradients;
        const size_t gradientCount = vernonRuntimeLoadedPipelineGetAdGradientCount(pipeline);
        gradients.reserve(gradientCount);
        for (size_t index = 0; index < gradientCount; ++index) {
            VernonAdValueMetadataView gradient{};
            gradient.struct_size = sizeof(gradient);
            if (vernonRuntimeLoadedPipelineGetAdGradientByIndex(pipeline, index, &gradient) != VERNON_STATUS_OK) {
                vernonPullbackDestroy(pullback);
                throw std::runtime_error("cannot read autodiff gradient reflection");
            }
            const std::string name = stringView(gradient.path);
            const auto reflectedInput =
                std::find_if(inputLeafMetadata.begin(), inputLeafMetadata.end(),
                             [&](const PythonAdMetadata &candidate) { return candidate.path == name; });
            if (reflectedInput == inputLeafMetadata.end()) {
                vernonPullbackDestroy(pullback);
                throw std::runtime_error("autodiff gradient path does not match an input Value leaf");
            }
            PythonAdMetadata inputLeaf = *reflectedInput;
            if (autodiffTangentDtype(inputLeaf.dtype) != gradient.dtype) {
                vernonPullbackDestroy(pullback);
                throw std::runtime_error("autodiff gradient dtype does not match its input Value leaf tangent type");
            }
            inputLeaf.dtype = gradient.dtype;
            gradients.push_back(std::move(inputLeaf));
        }
        nb::object output = storageObjectives          ? nb::none()
                            : outputValues.size() == 1 ? outputValues.front().array
                                                       : nb::borrow<nb::object>(aggregateOutput);
        return nb::make_tuple(output, std::make_unique<PythonPullback>(runtime, pullback, std::move(gradients),
                                                                       std::move(cotangentMetadata),
                                                                       std::move(pipelineOwner), nb::dict(bindings)));
    }

    static size_t dataTypeSize(VernonDataType type) {
        switch (type) {
        case VERNON_DATA_BOOL:
        case VERNON_DATA_U8:
            return 1;
        case VERNON_DATA_F16:
            return 2;
        case VERNON_DATA_I32:
        case VERNON_DATA_U32:
        case VERNON_DATA_F32:
            return 4;
        case VERNON_DATA_F64:
            return 8;
        }
        throw std::invalid_argument("unsupported compute pipeline data type");
    }

    std::unique_ptr<PythonRuntimeSubmission> submitCompute(uint32_t x, uint32_t y, uint32_t z, const nb::list &values) {
        const std::vector<PipelineParameterMetadata> metadata = parameters();
        if (values.size() != metadata.size())
            throw std::invalid_argument("compute pipeline value count does not match reflection");
        std::deque<std::string> scalarStorage;
        std::deque<std::vector<uint64_t>> shapes;
        std::deque<std::vector<int64_t>> strides;
        std::vector<VernonPipelineArgument> arguments;
        arguments.reserve(metadata.size());
        for (size_t index = 0; index < metadata.size(); ++index) {
            const PipelineParameterMetadata &parameter = metadata[index];
            if (parameter.kind != VERNON_PIPELINE_TENSOR)
                throw std::invalid_argument("direct compute pipelines accept only Tensor/value parameters");
            shapes.push_back(parameter.shape);
            strides.emplace_back(parameter.shape.size());
            size_t stride = parameter.elementByteSize;
            for (size_t dimension = parameter.shape.size(); dimension-- != 0;) {
                strides.back()[dimension] = static_cast<int64_t>(stride);
                stride *= static_cast<size_t>(parameter.shape[dimension]);
            }
            VernonPipelineArgument argument{};
            argument.slot = parameter.slot;
            argument.kind = VERNON_PIPELINE_TENSOR;
            argument.tensor.struct_size = sizeof(VernonTensorView);
            argument.tensor.element_layout = {
                sizeof(VernonValueLayoutView),  parameter.elementByteSize,
                parameter.elementAlignment,     {parameter.layoutHash.data(), parameter.layoutHash.size()},
                parameter.elementLeaves.data(), parameter.elementLeaves.size(),
            };
            argument.tensor.access = parameter.access;
            argument.tensor.rank = static_cast<uint32_t>(shapes.back().size());
            argument.tensor.shape = shapes.back().empty() ? nullptr : shapes.back().data();
            argument.tensor.byte_strides = strides.back().empty() ? nullptr : strides.back().data();
            nb::handle value = values[index];
            if (nb::isinstance<RhiBuffer>(value)) {
                RhiBuffer *buffer = nb::cast<RhiBuffer *>(value);
                if (!buffer || runtimeRhiHost(owner) != buffer->host.get())
                    throw std::invalid_argument("compute pipeline RHI buffer belongs to another device");
                argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
                if (vernonRuntimeReferenceRhiBuffer(runtime, buffer->handle, 0, buffer->size,
                                                    &argument.tensor.resource) != VERNON_STATUS_OK)
                    throw std::invalid_argument("compute pipeline RHI buffer is invalid");
                argument.tensor.byte_size = buffer->size;
            } else if (nb::isinstance<nb::bytes>(value)) {
                nb::bytes bytes = nb::borrow<nb::bytes>(value);
                scalarStorage.emplace_back(bytes.c_str(), bytes.size());
                argument.tensor.storage = VERNON_TENSOR_HOST;
                argument.tensor.host_data = scalarStorage.back().data();
                argument.tensor.byte_size = scalarStorage.back().size();
            } else {
                throw std::invalid_argument("compute pipeline values must be RhiBuffer or bytes");
            }
            arguments.push_back(argument);
        }
        VernonPipelineInvocation invocation{};
        invocation.struct_size = sizeof(invocation);
        invocation.abi_version = VERNON_PIPELINE_VERSION;
        invocation.arguments = arguments.data();
        invocation.argument_count = arguments.size();
        invocation.compute_grid = {x, y, z};
        VernonSubmission *submission{};
        if (vernonRuntimePipelineSubmit(pipeline, &invocation, &submission) != VERNON_STATUS_OK)
            throw std::runtime_error("compute pipeline submission failed: " +
                                     stringView(vernonRuntimeGetLastError(runtime)));
        return std::make_unique<PythonRuntimeSubmission>(submission);
    }

    nb::list derivativeGroups() const {
        nb::list result;
        const size_t groupCount = vernonRuntimeLoadedPipelineGetAdDerivativeGroupCount(pipeline);
        for (size_t groupIndex = 0; groupIndex < groupCount; ++groupIndex) {
            VernonAdDerivativeGroupView group{};
            group.struct_size = sizeof(group);
            if (vernonRuntimeLoadedPipelineGetAdDerivativeGroupByIndex(pipeline, groupIndex, &group) !=
                VERNON_STATUS_OK)
                throw std::runtime_error("cannot read autodiff derivative group metadata");
            nb::list leaves;
            for (size_t leafIndex = 0; leafIndex < group.leaf_count; ++leafIndex) {
                VernonStringView leafPath{};
                if (vernonRuntimeLoadedPipelineGetAdDerivativeGroupLeaf(pipeline, groupIndex, leafIndex, &leafPath) !=
                    VERNON_STATUS_OK)
                    throw std::runtime_error("cannot read autodiff derivative group leaf");
                leaves.append(stringView(leafPath));
            }
            result.append(nb::make_tuple(group.role == VERNON_AD_DERIVATIVE_GRADIENT ? "gradient" : "cotangent",
                                         stringView(group.declared_path), leaves));
        }
        return result;
    }

    std::vector<PipelineParameterMetadata> parameters() const {
        std::vector<PipelineParameterMetadata> result;
        const size_t count = vernonRuntimeLoadedPipelineGetParameterCount(pipeline);
        result.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            VernonPipelineParameterView view{};
            if (vernonRuntimeLoadedPipelineGetParameterByIndex(pipeline, index, &view) != VERNON_STATUS_OK)
                throw std::runtime_error("cannot read loaded pipeline parameter");
            result.push_back(parameterMetadata(view));
        }
        return result;
    }

    std::vector<PipelineOutputMetadata> outputs() const {
        std::vector<PipelineOutputMetadata> result;
        const size_t count = vernonRuntimeLoadedPipelineGetOutputCount(pipeline);
        result.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            VernonPipelineOutputView view{};
            if (vernonRuntimeLoadedPipelineGetOutputByIndex(pipeline, index, &view) != VERNON_STATUS_OK)
                throw std::runtime_error("cannot read loaded pipeline output");
            result.push_back(outputMetadata(view));
        }
        return result;
    }

    Runtime *owner{};
    VernonRuntimeContext *runtime{};
    VernonPipelineBundle *bundle{};
    VernonLoadedPipeline *pipeline{};
    // ORC entry pointers are valid only while their compile result owns the JIT.
    std::vector<SharedCompileResult> retainedResults;
};
struct Runtime {
    explicit Runtime(VernonRuntimeContext *handle, std::shared_ptr<RhiHostState> rhiHost = {})
        : handle(handle), rhiHost(std::move(rhiHost)) {}
    explicit Runtime(VernonRuntimeBackend backend) {
        VernonRuntimeCreateOptions options{};
        options.struct_size = sizeof(options);
        handle = vernonRuntimeCreateWithOptions(backend, &options);
        if (!handle)
            throw std::runtime_error("requested runtime backend is unavailable");
    }
    ~Runtime() { vernonRuntimeDestroy(handle); }

    std::unique_ptr<PythonExecutionGraph> createExecutionGraph() {
        if (rhiHost)
            return std::make_unique<PythonExecutionGraph>(rhiHost);
        return std::make_unique<PythonExecutionGraph>();
    }

    std::unique_ptr<LoadedPipeline> load(const nb::bytes &artifact, const std::string &reflection,
                                         const std::string &entry) {
        VernonLoadedPipeline *pipeline =
            vernonRuntimeLoadArtifact(handle, artifact.c_str(), artifact.size(), reflection.data(), reflection.size(),
                                      entry.data(), entry.size());
        if (!pipeline)
            throw std::runtime_error("cannot load compute pipeline: " + stringView(vernonRuntimeGetLastError(handle)));
        return std::make_unique<LoadedPipeline>(this, handle, nullptr, pipeline);
    }

    std::unique_ptr<LoadedPipeline> loadCpuEntry(const CompiledProgram &program, const std::string &entry) {
        program.requireSuccess();
        if (program.target != VERNON_TARGET_CPU)
            throw std::runtime_error("CPU entries can be loaded only from CPU compiled programs");
        if (entry.empty())
            throw std::runtime_error("CPU entry name must not be empty");
        VernonCpuEntryPoint entryPoint =
            vernonCompileResultGetCpuEntry(program.result.get(), entry.data(), entry.size());
        if (!entryPoint)
            throw std::runtime_error("CPU entry '" + entry + "' was not found in compiled program");
        const std::string reflection = program.reflection();
        VernonLoadedPipeline *pipeline = vernonRuntimeLoadCpuEntry(handle, entryPoint, reflection.data(),
                                                                   reflection.size(), entry.data(), entry.size());
        if (!pipeline)
            throw std::runtime_error("cannot load CPU entry: " + stringView(vernonRuntimeGetLastError(handle)));
        return std::make_unique<LoadedPipeline>(this, handle, nullptr, pipeline,
                                                std::vector<SharedCompileResult>{program.result});
    }

    std::unique_ptr<LoadedPipeline> loadCpuAutodiff(const CompiledProgram &primal, const std::string &primalName,
                                                    const CompiledProgram &forward, const std::string &forwardName,
                                                    const CompiledProgram &backward, const std::string &backwardName,
                                                    const std::string &forwardProtocol,
                                                    const std::string &backwardProtocol,
                                                    const nb::list &groupMetadata) {
        const CompiledProgram *programs[] = {&primal, &forward, &backward};
        const std::string *names[] = {&primalName, &forwardName, &backwardName};
        VernonCpuEntryPoint entries[3]{};
        std::string reflections[3];
        for (size_t index = 0; index < 3; ++index) {
            programs[index]->requireSuccess();
            if (programs[index]->target != VERNON_TARGET_CPU)
                throw std::runtime_error("direct autodiff profiles require CPU compiled programs");
            if (names[index]->empty())
                throw std::runtime_error("direct autodiff profile entry names must not be empty");
            entries[index] = vernonCompileResultGetCpuEntry(programs[index]->result.get(), names[index]->data(),
                                                            names[index]->size());
            if (!entries[index])
                throw std::runtime_error("direct autodiff profile entry '" + *names[index] + "' was not found");
            reflections[index] = programs[index]->reflection();
        }
        auto view = [](const std::string &value) { return VernonStringView{value.data(), value.size()}; };
        std::vector<vernon::runtime::AutodiffDerivativeGroup> derivativeGroups;
        derivativeGroups.reserve(groupMetadata.size());
        for (nb::handle item : groupMetadata) {
            nb::tuple metadata = nb::cast<nb::tuple>(item);
            if (metadata.size() != 3)
                throw std::invalid_argument("direct autodiff derivative group metadata is invalid");
            const std::string role = nb::cast<std::string>(metadata[0]);
            vernon::runtime::AutodiffDerivativeRole derivativeRole;
            if (role == "gradient")
                derivativeRole = vernon::runtime::AutodiffDerivativeRole::Gradient;
            else if (role == "cotangent")
                derivativeRole = vernon::runtime::AutodiffDerivativeRole::Cotangent;
            else
                throw std::invalid_argument("direct autodiff derivative group role is invalid");
            derivativeGroups.push_back(
                {derivativeRole, nb::cast<std::string>(metadata[1]), nb::cast<std::vector<std::string>>(metadata[2])});
        }
        std::vector<std::vector<VernonStringView>> derivativeLeafViews;
        std::vector<vernon::runtime::AutodiffDerivativeGroupView> derivativeGroupViews;
        derivativeLeafViews.reserve(derivativeGroups.size());
        derivativeGroupViews.reserve(derivativeGroups.size());
        for (const vernon::runtime::AutodiffDerivativeGroup &group : derivativeGroups) {
            std::vector<VernonStringView> &leaves = derivativeLeafViews.emplace_back();
            leaves.reserve(group.leafPaths.size());
            for (const std::string &leaf : group.leafPaths)
                leaves.push_back(view(leaf));
            derivativeGroupViews.push_back({group.role, view(group.declaredPath), leaves.data(), leaves.size()});
        }
        VernonLoadedPipeline *pipeline = vernon::runtime::loadBackendCpuAutodiffPipeline(
            *handle, entries[0], view(reflections[0]), view(primalName), entries[1], view(reflections[1]),
            view(forwardName), entries[2], view(reflections[2]), view(backwardName), view(forwardProtocol),
            view(backwardProtocol), derivativeGroupViews.data(), derivativeGroupViews.size());
        if (!pipeline)
            throw std::runtime_error("cannot load direct CPU autodiff profiles: " +
                                     stringView(vernonRuntimeGetLastError(handle)));
        return std::make_unique<LoadedPipeline>(
            this, handle, nullptr, pipeline,
            std::vector<SharedCompileResult>{primal.result, forward.result, backward.result});
    }

    std::unique_ptr<LoadedPipeline> loadPipeline(const nb::bytes &data, const std::vector<std::string> &features) {
        VernonPipelineBundle *bundle =
            vernonRuntimeLoadPipelineBundleWithOptions(handle, data.c_str(), data.size(), nullptr);
        if (!bundle)
            throw std::runtime_error("cannot load pipeline bundle: " + stringView(vernonRuntimeGetLastError(handle)));
        std::vector<const char *> names;
        for (const std::string &feature : features)
            names.push_back(feature.c_str());
        VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {names.data(), names.size()});
        if (!pipeline) {
            vernonRuntimePipelineBundleDestroy(bundle);
            throw std::runtime_error("cannot resolve pipeline bundle: " +
                                     stringView(vernonRuntimeGetLastError(handle)));
        }
        return std::make_unique<LoadedPipeline>(this, handle, bundle, pipeline);
    }

    std::unique_ptr<LoadedPipeline> loadPipelineAsset(const nb::bytes &data, const std::string &directory,
                                                      const std::vector<std::string> &features) {
        VernonPipelineBundleLoadOptions options{};
        options.struct_size = sizeof(options);
        options.bundle_directory = directory.c_str();
        VernonPipelineBundle *bundle =
            vernonRuntimeLoadPipelineBundleWithOptions(handle, data.c_str(), data.size(), &options);
        if (!bundle)
            throw std::runtime_error("cannot load pipeline bundle: " + stringView(vernonRuntimeGetLastError(handle)));
        std::vector<const char *> names;
        for (const std::string &feature : features)
            names.push_back(feature.c_str());
        VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {names.data(), names.size()});
        if (!pipeline) {
            vernonRuntimePipelineBundleDestroy(bundle);
            throw std::runtime_error("cannot resolve pipeline bundle: " +
                                     stringView(vernonRuntimeGetLastError(handle)));
        }
        return std::make_unique<LoadedPipeline>(this, handle, bundle, pipeline);
    }

    VernonRuntimeContext *handle{};
    std::shared_ptr<RhiHostState> rhiHost;
};

RhiHostState *runtimeRhiHost(const Runtime *runtime) { return runtime ? runtime->rhiHost.get() : nullptr; }

std::unique_ptr<Runtime> RhiHost::createRuntime() {
    VernonRuntimeBackend backend;
    switch (state->backend) {
    case VERNON_RHI_BACKEND_CUDA:
        backend = VERNON_RUNTIME_CUDA;
        break;
    case VERNON_RHI_BACKEND_VULKAN:
        backend = VERNON_RUNTIME_VULKAN;
        break;
    case VERNON_RHI_BACKEND_DIRECTX12:
        backend = VERNON_RUNTIME_DIRECTX12;
        break;
    case VERNON_RHI_BACKEND_METAL:
        backend = VERNON_RUNTIME_METAL;
        break;
    case VERNON_RHI_BACKEND_OPENGL:
        backend = VERNON_RUNTIME_OPENGL;
        break;
    case VERNON_RHI_BACKEND_OPENGL_ES:
        backend = VERNON_RUNTIME_OPENGL_ES;
        break;
    }
    VernonRuntimeContext *runtime = vernonRuntimeCreateForRhiDevice(backend, state->device);
    if (!runtime)
        throw std::runtime_error("cannot create Runtime for Vernon RHI device");
    return std::make_unique<Runtime>(runtime, state);
}

} // namespace

NB_MODULE(_native, module) {
    module.doc() = "VernonDSL native compiler and kernel runtime bindings";
    nb::enum_<VernonTarget>(module, "Target")
        .value("CPU", VERNON_TARGET_CPU)
        .value("CUDA", VERNON_TARGET_CUDA)
        .value("VULKAN", VERNON_TARGET_VULKAN)
        .value("METAL", VERNON_TARGET_METAL)
        .value("DIRECTX", VERNON_TARGET_DIRECTX)
        .value("OPENGL", VERNON_TARGET_OPENGL)
        .value("OPENGL_ES", VERNON_TARGET_OPENGL_ES);
    module.def("target_available", &targetAvailable, nb::arg("target"));
    module.def("target_capabilities", &targetCapabilities, nb::arg("target"));
    nb::enum_<VernonStatus>(module, "Status")
        .value("OK", VERNON_STATUS_OK)
        .value("INVALID_ARGUMENT", VERNON_STATUS_INVALID_ARGUMENT)
        .value("PARSE_ERROR", VERNON_STATUS_PARSE_ERROR)
        .value("VERIFICATION_ERROR", VERNON_STATUS_VERIFICATION_ERROR)
        .value("UNSUPPORTED_TARGET", VERNON_STATUS_UNSUPPORTED_TARGET)
        .value("INTERNAL_ERROR", VERNON_STATUS_INTERNAL_ERROR);
    nb::enum_<VernonRuntimeBackend>(module, "RuntimeBackend")
        .value("CPU", VERNON_RUNTIME_CPU)
        .value("CUDA", VERNON_RUNTIME_CUDA)
        .value("VULKAN", VERNON_RUNTIME_VULKAN)
        .value("OPENGL", VERNON_RUNTIME_OPENGL)
        .value("OPENGL_ES", VERNON_RUNTIME_OPENGL_ES)
        .value("DIRECTX12", VERNON_RUNTIME_DIRECTX12)
        .value("METAL", VERNON_RUNTIME_METAL);
    nb::enum_<VernonRhiBackend>(module, "RhiBackend")
        .value("CUDA", VERNON_RHI_BACKEND_CUDA)
        .value("VULKAN", VERNON_RHI_BACKEND_VULKAN)
        .value("DIRECTX12", VERNON_RHI_BACKEND_DIRECTX12)
        .value("OPENGL", VERNON_RHI_BACKEND_OPENGL)
        .value("OPENGL_ES", VERNON_RHI_BACKEND_OPENGL_ES)
        .value("METAL", VERNON_RHI_BACKEND_METAL);
    nb::enum_<VernonPrimitiveTopology>(module, "PrimitiveTopology")
        .value("TRIANGLE_LIST", VERNON_TOPOLOGY_TRIANGLE_LIST)
        .value("LINE_LIST", VERNON_TOPOLOGY_LINE_LIST)
        .value("POINT_LIST", VERNON_TOPOLOGY_POINT_LIST);
    nb::enum_<VernonTextureFormat>(module, "TextureFormat")
        .value("RGBA8_UNORM", VERNON_TEXTURE_RGBA8_UNORM)
        .value("RGBA8_SRGB", VERNON_TEXTURE_RGBA8_SRGB)
        .value("RGBA16_FLOAT", VERNON_TEXTURE_RGBA16_FLOAT)
        .value("RGBA32_FLOAT", VERNON_TEXTURE_RGBA32_FLOAT)
        .value("R8_UNORM", VERNON_TEXTURE_R8_UNORM)
        .value("R16_FLOAT", VERNON_TEXTURE_R16_FLOAT)
        .value("R32_FLOAT", VERNON_TEXTURE_R32_FLOAT)
        .value("RG8_UNORM", VERNON_TEXTURE_RG8_UNORM)
        .value("RGB8_UNORM", VERNON_TEXTURE_RGB8_UNORM)
        .value("R11G11B10_FLOAT", VERNON_TEXTURE_R11G11B10_FLOAT)
        .value("D32_FLOAT", VERNON_TEXTURE_D32_FLOAT)
        .value("D32_FLOAT_S8_UINT", VERNON_TEXTURE_D32_FLOAT_S8_UINT);
    nb::enum_<VernonTextureDimension>(module, "TextureDimension")
        .value("TEXTURE_2D", VERNON_TEXTURE_2D)
        .value("TEXTURE_3D", VERNON_TEXTURE_3D)
        .value("CUBE", VERNON_TEXTURE_CUBE);
    nb::enum_<VernonRhiSamplerAddressMode>(module, "SamplerAddressMode")
        .value("REPEAT", VERNON_RHI_ADDRESS_REPEAT)
        .value("CLAMP_TO_EDGE", VERNON_RHI_ADDRESS_CLAMP_TO_EDGE)
        .value("MIRRORED_REPEAT", VERNON_RHI_ADDRESS_MIRRORED_REPEAT);
    module.attr("IMAGE_COLOR_ATTACHMENT") = static_cast<uint32_t>(VERNON_RHI_IMAGE_COLOR_ATTACHMENT);
    module.attr("IMAGE_DEPTH_STENCIL_ATTACHMENT") = static_cast<uint32_t>(VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT);
    module.attr("IMAGE_TRANSFER_SOURCE") = static_cast<uint32_t>(VERNON_RHI_IMAGE_TRANSFER_SOURCE);
    module.attr("IMAGE_TRANSFER_DESTINATION") = static_cast<uint32_t>(VERNON_RHI_IMAGE_TRANSFER_DESTINATION);
    module.attr("IMAGE_SAMPLED") = static_cast<uint32_t>(VERNON_RHI_IMAGE_SAMPLED);
    module.attr("IMAGE_STORAGE") = static_cast<uint32_t>(VERNON_RHI_IMAGE_STORAGE);
    module.attr("IMAGE_ASPECT_COLOR") = static_cast<uint32_t>(VERNON_RHI_IMAGE_ASPECT_COLOR);
    module.attr("IMAGE_ASPECT_DEPTH") = static_cast<uint32_t>(VERNON_RHI_IMAGE_ASPECT_DEPTH);
    module.attr("IMAGE_ASPECT_STENCIL") = static_cast<uint32_t>(VERNON_RHI_IMAGE_ASPECT_STENCIL);
    module.def("_plan_value_abi", &planValueAbi, nb::arg("module"), nb::arg("logical_dtypes"));
    nb::class_<StructuredVjp>(module, "_StructuredVjp")
        .def_prop_ro("tape_bytes", &StructuredVjp::tapeBytes)
        .def_prop_ro("derivative_rules", &StructuredVjp::derivativeRules)
        .def("profiles", &StructuredVjp::profiles, nb::arg("identity"));
    module.def("_build_structured_vjp", &buildStructuredVjp, nb::arg("module"), nb::arg("entry"), nb::arg("wrt_paths"),
               nb::arg("output_paths"), nb::arg("forward_symbol"), nb::arg("backward_symbol"));
    nb::class_<Compiler>(module, "Compiler")
        .def(nb::init<>())
        .def("compile_program_result", &compileProgramResult, nb::arg("mlir"), nb::arg("target"),
             nb::arg("options") = nb::dict());
    module.def("_compile_cpu_program_results", &compileCpuProgramResults, nb::arg("modules"),
               nb::arg("options") = nb::dict());
    nb::class_<CompiledProgram>(module, "CompiledProgram")
        .def_prop_ro("ok", &CompiledProgram::ok)
        .def_prop_ro("status", &CompiledProgram::status)
        .def_prop_ro("diagnostics", &CompiledProgram::diagnostics)
        .def_prop_ro("artifacts", &CompiledProgram::artifacts)
        .def_prop_ro("reflection", &CompiledProgram::reflection)
        .def_prop_ro("target", [](const CompiledProgram &value) { return value.target; })
        .def("has_cpu_entry", &CompiledProgram::hasCpuEntry);
    nb::class_<RhiHost>(module, "RhiHost")
        .def(nb::init<VernonRhiBackend, uint32_t>(), nb::arg("backend"), nb::arg("device_index") = 0)
        .def_static("create_external_opengl", &RhiHost::createExternalOpenGL, nb::arg("backend"), nb::arg("user_data"),
                    nb::arg("make_current"), nb::arg("get_proc_address"), nb::arg("api_major"), nb::arg("api_minor"))
        .def("create_buffer", &RhiHost::createBuffer)
        .def("create_image", &RhiHost::createImage, nb::arg("width"), nb::arg("height"),
             nb::arg("format") = VERNON_TEXTURE_RGBA8_UNORM, nb::arg("dimension") = VERNON_TEXTURE_2D,
             nb::arg("depth") = 1, nb::arg("mip_levels") = 1, nb::arg("usage") = 0)
        .def("create_attachment_image", &RhiHost::createAttachmentImage)
        .def("create_sampler", &RhiHost::createSampler, nb::arg("address") = VERNON_RHI_ADDRESS_REPEAT)
        .def("create_runtime", &RhiHost::createRuntime, nb::keep_alive<0, 1>())
        .def("create_execution_graph", &RhiHost::createExecutionGraph);
    nb::class_<RhiBuffer>(module, "RhiBuffer")
        .def_prop_ro("size", [](const RhiBuffer &value) { return value.size; })
        .def("upload", &RhiBuffer::upload, nb::arg("data"), nb::arg("offset") = 0)
        .def("upload_ranges", &RhiBuffer::uploadRanges, nb::arg("ranges"))
        .def("download", &RhiBuffer::download);
    nb::class_<RhiImage>(module, "RhiImage")
        .def_prop_ro("width", [](const RhiImage &value) { return value.width; })
        .def_prop_ro("height", [](const RhiImage &value) { return value.height; })
        .def_prop_ro("depth", [](const RhiImage &value) { return value.depth; })
        .def_prop_ro("mip_levels", [](const RhiImage &value) { return value.mipLevels; })
        .def("upload", &RhiImage::upload, nb::arg("data"), nb::arg("mip_level") = 0, nb::arg("offset_x") = 0,
             nb::arg("offset_y") = 0, nb::arg("offset_z") = 0, nb::arg("width") = 0, nb::arg("height") = 0,
             nb::arg("depth") = 0)
        .def("download", &RhiImage::download, nb::arg("mip_level") = 0, nb::arg("offset_x") = 0,
             nb::arg("offset_y") = 0, nb::arg("offset_z") = 0, nb::arg("width") = 0, nb::arg("height") = 0,
             nb::arg("depth") = 0)
        .def("generate_mipmaps", &RhiImage::generateMipmaps)
        .def(
            "create_view",
            [](RhiImage &image, const nb::object &format, const nb::object &dimension, uint32_t baseMipLevel,
               uint32_t mipLevelCount, uint32_t baseArrayLayer, uint32_t arrayLayerCount, uint32_t aspects) {
                const VernonTextureFormat viewFormat =
                    format.is_none() ? image.format : nb::cast<VernonTextureFormat>(format);
                const VernonTextureDimension viewDimension =
                    dimension.is_none() ? image.dimension : nb::cast<VernonTextureDimension>(dimension);
                if (baseMipLevel >= image.mipLevels || baseArrayLayer >= image.layers)
                    throw std::invalid_argument("RHI image view base subresource is out of range");
                mipLevelCount = mipLevelCount ? mipLevelCount : image.mipLevels - baseMipLevel;
                arrayLayerCount = arrayLayerCount ? arrayLayerCount : image.layers - baseArrayLayer;
                if (!aspects)
                    aspects = image.format == VERNON_TEXTURE_D32_FLOAT ? VERNON_RHI_IMAGE_ASPECT_DEPTH
                              : image.format == VERNON_TEXTURE_D32_FLOAT_S8_UINT
                                  ? VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL
                                  : VERNON_RHI_IMAGE_ASPECT_COLOR;
                return std::make_unique<RhiImageView>(&image, viewFormat, viewDimension, baseMipLevel, mipLevelCount,
                                                      baseArrayLayer, arrayLayerCount, aspects);
            },
            nb::arg("format") = nb::none(), nb::arg("dimension") = nb::none(), nb::arg("base_mip_level") = 0,
            nb::arg("mip_level_count") = 0, nb::arg("base_array_layer") = 0, nb::arg("array_layer_count") = 0,
            nb::arg("aspects") = 0, nb::keep_alive<0, 1>());
    nb::class_<RhiImageView>(module, "RhiImageView")
        .def_prop_ro("width", [](const RhiImageView &value) { return value.width; })
        .def_prop_ro("height", [](const RhiImageView &value) { return value.height; })
        .def_prop_ro("base_mip_level", [](const RhiImageView &value) { return value.baseMipLevel; })
        .def_prop_ro("mip_level_count", [](const RhiImageView &value) { return value.mipLevelCount; })
        .def_prop_ro("base_array_layer", [](const RhiImageView &value) { return value.baseArrayLayer; })
        .def_prop_ro("array_layer_count", [](const RhiImageView &value) { return value.arrayLayerCount; })
        .def_prop_ro("aspects", [](const RhiImageView &value) { return value.aspects; });
    nb::class_<RhiSampler>(module, "RhiSampler");
    nb::class_<PythonGraphResource>(module, "_GraphResource")
        .def_prop_ro("id", [](const PythonGraphResource &value) { return value.resource.id; })
        .def_prop_ro("is_image", [](const PythonGraphResource &value) { return value.isImage; });
    nb::class_<vernon::execution::ExecutionPass>(module, "_ExecutionPass")
        .def("depends_on", &vernon::execution::ExecutionPass::dependsOn)
        .def("set_flags", &vernon::execution::ExecutionPass::setFlags)
        .def_prop_ro("flags", &vernon::execution::ExecutionPass::flags);
    nb::class_<PythonRenderPass, vernon::execution::ExecutionPass>(module, "_RenderPass")
        .def(
            "use",
            [](PythonRenderPass &pass, const PythonGraphResource &resource, uint32_t access, uint32_t state,
               uint32_t stageMask) {
                if (access > static_cast<uint32_t>(vernon::execution::AccessMode::ReadWrite) ||
                    state > static_cast<uint32_t>(VERNON_RHI_STATE_PRESENT) ||
                    (stageMask & ~(VERNON_RHI_STAGE_VERTEX | VERNON_RHI_STAGE_FRAGMENT)) != 0)
                    throw std::invalid_argument("invalid execution graph resource use");
                pass.use(resource, static_cast<vernon::execution::AccessMode>(access),
                         static_cast<VernonRhiResourceState>(state), stageMask);
            },
            nb::arg("resource"), nb::arg("access"), nb::arg("state"), nb::arg("stage_mask"))
        .def(
            "color",
            [](PythonRenderPass &pass, uint32_t location, const PythonGraphResource &resource, uint32_t load,
               uint32_t store, const std::array<float, 4> &clear) {
                if (load > static_cast<uint32_t>(VERNON_RHI_LOAD_DISCARD) ||
                    store > static_cast<uint32_t>(VERNON_RHI_STORE_DISCARD))
                    throw std::invalid_argument("invalid color attachment operation");
                pass.addColor(location, resource, static_cast<VernonRhiLoadOperation>(load),
                              static_cast<VernonRhiStoreOperation>(store), clear);
            },
            nb::arg("location"), nb::arg("resource"), nb::arg("load"), nb::arg("store"), nb::arg("clear"))
        .def(
            "depth",
            [](PythonRenderPass &pass, const PythonGraphResource &resource, uint32_t depthLoad, uint32_t depthStore,
               float clearDepth, uint32_t stencilLoad, uint32_t stencilStore, uint32_t clearStencil, bool readOnlyDepth,
               bool readOnlyStencil) {
                if (depthLoad > static_cast<uint32_t>(VERNON_RHI_LOAD_DISCARD) ||
                    stencilLoad > static_cast<uint32_t>(VERNON_RHI_LOAD_DISCARD) ||
                    depthStore > static_cast<uint32_t>(VERNON_RHI_STORE_DISCARD) ||
                    stencilStore > static_cast<uint32_t>(VERNON_RHI_STORE_DISCARD) || clearDepth < 0.0f ||
                    clearDepth > 1.0f)
                    throw std::invalid_argument("invalid depth attachment operation");
                pass.setDepth(resource, static_cast<VernonRhiLoadOperation>(depthLoad),
                              static_cast<VernonRhiStoreOperation>(depthStore), clearDepth,
                              static_cast<VernonRhiLoadOperation>(stencilLoad),
                              static_cast<VernonRhiStoreOperation>(stencilStore), clearStencil, readOnlyDepth,
                              readOnlyStencil);
            },
            nb::arg("resource"), nb::arg("depth_load"), nb::arg("depth_store"), nb::arg("clear_depth"),
            nb::arg("stencil_load"), nb::arg("stencil_store"), nb::arg("clear_stencil"), nb::arg("read_only_depth"),
            nb::arg("read_only_stencil"))
        .def("render_area", &PythonRenderPass::setRenderArea);
    nb::class_<PythonComputePass, vernon::execution::ExecutionPass>(module, "_ComputePass")
        .def(
            "use",
            [](PythonComputePass &pass, const PythonGraphResource &resource, uint32_t access, uint32_t state,
               uint32_t stageMask) {
                if (access > static_cast<uint32_t>(vernon::execution::AccessMode::ReadWrite) ||
                    state > static_cast<uint32_t>(VERNON_RHI_STATE_PRESENT) ||
                    (stageMask & ~VERNON_RHI_STAGE_COMPUTE) != 0)
                    throw std::invalid_argument("invalid execution graph resource use");
                pass.use(resource, static_cast<vernon::execution::AccessMode>(access),
                         static_cast<VernonRhiResourceState>(state), stageMask);
            },
            nb::arg("resource"), nb::arg("access"), nb::arg("state"), nb::arg("stage_mask"));
    nb::class_<PythonCompiledBarrier>(module, "_CompiledBarrier")
        .def_ro("source_stage_mask", &PythonCompiledBarrier::sourceStageMask)
        .def_ro("destination_stage_mask", &PythonCompiledBarrier::destinationStageMask)
        .def_ro("source_access", &PythonCompiledBarrier::sourceAccess)
        .def_ro("destination_access", &PythonCompiledBarrier::destinationAccess)
        .def_ro("old_state", &PythonCompiledBarrier::oldState)
        .def_ro("new_state", &PythonCompiledBarrier::newState)
        .def_ro("is_image", &PythonCompiledBarrier::isImage)
        .def_ro("base_mip_level", &PythonCompiledBarrier::baseMipLevel)
        .def_ro("mip_level_count", &PythonCompiledBarrier::mipLevelCount)
        .def_ro("base_array_layer", &PythonCompiledBarrier::baseArrayLayer)
        .def_ro("array_layer_count", &PythonCompiledBarrier::arrayLayerCount)
        .def_ro("aspects", &PythonCompiledBarrier::aspects);
    nb::class_<PythonCompiledScope>(module, "_CompiledScope")
        .def_ro("rendering", &PythonCompiledScope::rendering)
        .def_ro("pass_indices", &PythonCompiledScope::passIndices)
        .def_ro("barriers", &PythonCompiledScope::barriers);
    nb::class_<PythonExecutionParameter>(module, "_ExecutionParameter")
        .def_prop_ro("id", [](const PythonExecutionParameter &value) { return value.parameter.id; });
    nb::class_<PythonExecutionBindingsBuilder>(module, "_ExecutionBindings")
        .def("set", &PythonExecutionBindingsBuilder::set);
    nb::class_<PythonExecutionBindingsView>(module, "_ExecutionBindingsView")
        .def("get", &PythonExecutionBindingsView::get)
        .def("token", &PythonExecutionBindingsView::token);
    nb::class_<PythonExecutionGraph>(module, "_ExecutionGraph")
        .def("add_render_pass", &PythonExecutionGraph::addRenderPass, nb::rv_policy::reference)
        .def("add_compute_pass", &PythonExecutionGraph::addComputePass, nb::rv_policy::reference)
        .def("import_buffer", &PythonExecutionGraph::importBuffer, nb::arg("buffer"), nb::arg("exported") = false)
        .def("import_host_buffer", &PythonExecutionGraph::importHostBuffer, nb::arg("identity"),
             nb::arg("exported") = false)
        .def("import_image", &PythonExecutionGraph::importImage, nb::arg("image"), nb::arg("exported") = false)
        .def("parameter", &PythonExecutionGraph::parameter)
        .def("compile", &PythonExecutionGraph::compile)
        .def("validate", &PythonExecutionGraph::validate);
    nb::class_<PythonCompiledExecutionGraph>(module, "_CompiledExecutionGraph")
        .def("create_bindings", &PythonCompiledExecutionGraph::createBindings)
        .def("submit", &PythonCompiledExecutionGraph::submit, nb::arg("bindings") = nb::none())
        .def_prop_ro(
            "schedule",
            [](const PythonCompiledExecutionGraph &value) -> const std::vector<uint32_t> & {
                return value.plan->schedule();
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro("scopes", &PythonCompiledExecutionGraph::scopes);
    nb::class_<PythonExecutionSubmission>(module, "_ExecutionSubmission")
        .def("wait", &PythonExecutionSubmission::wait, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("state", &PythonExecutionSubmission::state);
    nb::class_<vernon::execution::GraphicsEncoder>(module, "_GraphicsEncoder");
    nb::class_<vernon::execution::ComputeEncoder>(module, "_ComputeEncoder");
    nb::class_<Runtime>(module, "Runtime")
        .def(nb::init<VernonRuntimeBackend>(), nb::arg("backend"))
        .def("load", &Runtime::load, nb::keep_alive<0, 1>())
        .def("load_cpu_entry", &Runtime::loadCpuEntry, nb::keep_alive<0, 1>())
        .def("load_cpu_autodiff", &Runtime::loadCpuAutodiff, nb::keep_alive<0, 1>())
        .def("load_pipeline", &Runtime::loadPipeline, nb::keep_alive<0, 1>())
        .def("load_pipeline_asset", &Runtime::loadPipelineAsset, nb::keep_alive<0, 1>())
        .def("create_execution_graph", &Runtime::createExecutionGraph);
    nb::class_<PipelineParameterMetadata>(module, "PipelineParameter")
        .def_ro("slot", &PipelineParameterMetadata::slot)
        .def_ro("name", &PipelineParameterMetadata::name)
        .def_prop_ro("kind", [](const PipelineParameterMetadata &value) { return static_cast<uint32_t>(value.kind); })
        .def_ro("element_byte_size", &PipelineParameterMetadata::elementByteSize)
        .def_ro("element_alignment", &PipelineParameterMetadata::elementAlignment)
        .def_ro("layout_hash", &PipelineParameterMetadata::layoutHash)
        .def_prop_ro("element_leaves",
                     [](const PipelineParameterMetadata &value) {
                         nb::list leaves;
                         for (const VernonValueLeafView &leaf : value.elementLeaves)
                             leaves.append(nb::make_tuple(leaf.dtype, leaf.scalar_count, leaf.byte_offset));
                         return leaves;
                     })
        .def_prop_ro("access",
                     [](const PipelineParameterMetadata &value) { return static_cast<uint32_t>(value.access); })
        .def_ro("shape", &PipelineParameterMetadata::shape);
    nb::class_<PipelineOutputMetadata>(module, "PipelineOutput")
        .def_ro("name", &PipelineOutputMetadata::name)
        .def_prop_ro("kind", [](const PipelineOutputMetadata &value) { return static_cast<uint32_t>(value.kind); })
        .def_prop_ro("dtype", [](const PipelineOutputMetadata &value) { return static_cast<uint32_t>(value.dtype); })
        .def_prop_ro("access", [](const PipelineOutputMetadata &value) { return static_cast<uint32_t>(value.access); })
        .def_ro("shape", &PipelineOutputMetadata::shape)
        .def_ro("location", &PipelineOutputMetadata::location);
    nb::class_<PythonRuntimeSubmission>(module, "Submission")
        .def("wait", &PythonRuntimeSubmission::wait, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("state", &PythonRuntimeSubmission::state);
    nb::class_<PreparedPipelineArgument>(module, "_PreparedPipelineArgument");
    nb::class_<PipelineInvocationBuilder>(module, "PipelineInvocationBuilder")
        .def("prepare_host_tensor", &PipelineInvocationBuilder::prepareHostTensor, nb::arg("parameter"),
             nb::arg("array"))
        .def("prepare_rhi_tensor", &PipelineInvocationBuilder::prepareRhiTensor, nb::arg("parameter"),
             nb::arg("buffer"), nb::arg("access"), nb::arg("shape"), nb::arg("strides"), nb::arg("offset") = 0)
        .def("prepare_rhi_texture", &PipelineInvocationBuilder::prepareRhiTexture, nb::arg("parameter"),
             nb::arg("texture"))
        .def("prepare_rhi_sampler", &PipelineInvocationBuilder::prepareRhiSampler, nb::arg("parameter"),
             nb::arg("sampler"))
        .def("prepared_argument", &PipelineInvocationBuilder::preparedArgument, nb::arg("argument"),
             nb::rv_policy::reference_internal, nb::keep_alive<1, 2>())
        .def("host_tensor", &PipelineInvocationBuilder::hostTensor, nb::arg("parameter"), nb::arg("array"),
             nb::rv_policy::reference_internal)
        .def("rhi_tensor", &PipelineInvocationBuilder::rhiTensor, nb::arg("parameter"), nb::arg("buffer"),
             nb::arg("access"), nb::arg("shape"), nb::arg("strides"), nb::arg("offset") = 0,
             nb::rv_policy::reference_internal, nb::keep_alive<1, 3>())
        .def("rhi_texture", &PipelineInvocationBuilder::rhiTexture, nb::arg("parameter"), nb::arg("texture"),
             nb::rv_policy::reference_internal, nb::keep_alive<1, 3>())
        .def("rhi_sampler", &PipelineInvocationBuilder::rhiSampler, nb::arg("parameter"), nb::arg("sampler"),
             nb::rv_policy::reference_internal, nb::keep_alive<1, 3>())
        .def("rhi_color_attachment", &PipelineInvocationBuilder::rhiColorAttachment, nb::arg("location"),
             nb::arg("texture"), nb::arg("load_operation"), nb::arg("store_operation"), nb::arg("clear_color"),
             nb::rv_policy::reference_internal, nb::keep_alive<1, 3>())
        .def("rhi_depth_attachment", &PipelineInvocationBuilder::rhiDepthAttachment, nb::arg("texture"),
             nb::arg("load_operation"), nb::arg("store_operation"), nb::arg("clear_depth"),
             nb::rv_policy::reference_internal, nb::keep_alive<1, 2>())
        .def("rhi_index_binding", &PipelineInvocationBuilder::rhiIndexBinding, nb::arg("buffer"), nb::arg("count"),
             nb::arg("offset") = 0, nb::rv_policy::reference_internal, nb::keep_alive<1, 2>())
        .def("topology", &PipelineInvocationBuilder::setTopology, nb::arg("topology"),
             nb::rv_policy::reference_internal)
        .def("counts", &PipelineInvocationBuilder::counts, nb::arg("vertex_count") = 0, nb::arg("instance_count") = 0,
             nb::rv_policy::reference_internal)
        .def("grid", &PipelineInvocationBuilder::grid, nb::arg("x"), nb::arg("y"), nb::arg("z"),
             nb::rv_policy::reference_internal)
        .def("viewport", &PipelineInvocationBuilder::setViewport, nb::arg("x"), nb::arg("y"), nb::arg("width"),
             nb::arg("height"), nb::rv_policy::reference_internal)
        .def("scissor", &PipelineInvocationBuilder::setScissor, nb::arg("x"), nb::arg("y"), nb::arg("width"),
             nb::arg("height"), nb::rv_policy::reference_internal)
        .def("encode", [](PipelineInvocationBuilder &builder,
                          const vernon::execution::GraphicsEncoder &encoder) { builder.encode(encoder); })
        .def("encode", [](PipelineInvocationBuilder &builder,
                          const vernon::execution::ComputeEncoder &encoder) { builder.encode(encoder); })
        .def("submit", [](PipelineInvocationBuilder &builder) { return builder.submit(); });
    nb::class_<PythonPullback>(module, "Pullback")
        .def("__call__", &PythonPullback::apply, nb::arg("cotangent") = nb::none());
    nb::class_<LoadedPipeline>(module, "LoadedPipeline")
        .def("invocation_builder", &LoadedPipeline::invocationBuilder, nb::keep_alive<0, 1>())
        .def(
            "submit",
            [](LoadedPipeline &pipeline, uint32_t x, uint32_t y, uint32_t z, const nb::list &values) {
                return pipeline.submitCompute(x, y, z, values);
            },
            nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("values"))
        .def(
            "submit",
            [](LoadedPipeline &pipeline, PipelineInvocationBuilder &builder) {
                if (builder.pipeline != pipeline.pipeline)
                    throw std::invalid_argument("invocation builder belongs to another pipeline");
                return builder.submit();
            },
            nb::arg("builder"))
        .def(
            "vjp",
            [](LoadedPipeline &pipeline, const nb::dict &bindings, const nb::tuple &grid) {
                if (grid.size() != 3)
                    throw std::invalid_argument("autodiff grid must contain three dimensions");
                const auto dimension = [&](size_t index) {
                    if (PyBool_Check(grid[index].ptr()))
                        throw std::invalid_argument("autodiff grid dimensions must be positive integers");
                    const uint64_t value = nb::cast<uint64_t>(grid[index]);
                    if (!value || value > UINT32_MAX)
                        throw std::invalid_argument("autodiff grid dimensions must be positive uint32 values");
                    return static_cast<uint32_t>(value);
                };
                const uint32_t x = dimension(0);
                const uint32_t y = dimension(1);
                const uint32_t z = dimension(2);
                return pipeline.vjp(x, y, z, bindings, nb::cast(&pipeline, nb::rv_policy::reference));
            },
            nb::arg("bindings"), nb::arg("grid"))
        .def_prop_ro("derivative_groups", &LoadedPipeline::derivativeGroups)
        .def_prop_ro("workgroup_size", &LoadedPipeline::workgroupSize)
        .def_prop_ro("parameters", &LoadedPipeline::parameters)
        .def_prop_ro("outputs", &LoadedPipeline::outputs);
    module.attr("DATA_BOOL") = static_cast<uint32_t>(VERNON_DATA_BOOL);
    module.attr("DATA_I32") = static_cast<uint32_t>(VERNON_DATA_I32);
    module.attr("DATA_U32") = static_cast<uint32_t>(VERNON_DATA_U32);
    module.attr("DATA_F16") = static_cast<uint32_t>(VERNON_DATA_F16);
    module.attr("DATA_F32") = static_cast<uint32_t>(VERNON_DATA_F32);
    module.attr("DATA_F64") = static_cast<uint32_t>(VERNON_DATA_F64);
    module.attr("ACCESS_READ") = static_cast<uint32_t>(VERNON_ACCESS_READ);
    module.attr("ACCESS_WRITE") = static_cast<uint32_t>(VERNON_ACCESS_WRITE);
    module.attr("ACCESS_READ_WRITE") = static_cast<uint32_t>(VERNON_ACCESS_READ_WRITE);
    module.attr("PIPELINE_TENSOR") = static_cast<uint32_t>(VERNON_PIPELINE_TENSOR);
    module.attr("PIPELINE_IMAGE") = static_cast<uint32_t>(VERNON_PIPELINE_IMAGE);
    module.attr("PIPELINE_SAMPLER") = static_cast<uint32_t>(VERNON_PIPELINE_SAMPLER);
    module.attr("TOPOLOGY_TRIANGLE_LIST") = static_cast<uint32_t>(VERNON_TOPOLOGY_TRIANGLE_LIST);
    module.attr("ATTACHMENT_CLEAR") = static_cast<uint32_t>(VERNON_RHI_LOAD_CLEAR);
    module.attr("ATTACHMENT_PRESERVE") = static_cast<uint32_t>(VERNON_RHI_LOAD_PRESERVE);
    module.attr("ATTACHMENT_DISCARD") = static_cast<uint32_t>(VERNON_RHI_LOAD_DISCARD);
    module.attr("ATTACHMENT_STORE") = static_cast<uint32_t>(VERNON_RHI_STORE_PRESERVE);
    module.attr("ATTACHMENT_DONT_CARE") = static_cast<uint32_t>(VERNON_RHI_STORE_DISCARD);
    module.attr("GRAPH_READ") = static_cast<uint32_t>(vernon::execution::AccessMode::Read);
    module.attr("GRAPH_WRITE") = static_cast<uint32_t>(vernon::execution::AccessMode::Write);
    module.attr("GRAPH_READ_WRITE") = static_cast<uint32_t>(vernon::execution::AccessMode::ReadWrite);
    module.attr("GRAPH_SHADER_READ") = static_cast<uint32_t>(VERNON_RHI_STATE_SHADER_READ);
    module.attr("GRAPH_SHADER_WRITE") = static_cast<uint32_t>(VERNON_RHI_STATE_SHADER_WRITE);
    module.attr("GRAPH_STAGE_COMPUTE") = static_cast<uint32_t>(VERNON_RHI_STAGE_COMPUTE);
    module.attr("GRAPH_STAGE_VERTEX") = static_cast<uint32_t>(VERNON_RHI_STAGE_VERTEX);
    module.attr("GRAPH_STAGE_FRAGMENT") = static_cast<uint32_t>(VERNON_RHI_STAGE_FRAGMENT);
    module.attr("GRAPH_PASS_NEVER_CULL") = static_cast<uint32_t>(vernon::execution::PassNeverCull);
    module.attr("GRAPH_PASS_NO_MERGE") = static_cast<uint32_t>(vernon::execution::PassNoMerge);
    module.attr("GRAPH_PASS_SIDE_EFFECT") = static_cast<uint32_t>(vernon::execution::PassSideEffect);
    module.attr("TOPOLOGY_LINE_LIST") = static_cast<uint32_t>(VERNON_TOPOLOGY_LINE_LIST);
    module.attr("TOPOLOGY_POINT_LIST") = static_cast<uint32_t>(VERNON_TOPOLOGY_POINT_LIST);
    module.def("runtime_available",
               [](VernonRuntimeBackend backend) { return vernonRuntimeGetCapabilities(backend).available != 0; });
}
