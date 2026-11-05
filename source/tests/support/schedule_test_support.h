#ifndef VERNON_TESTS_SUPPORT_SCHEDULE_TEST_SUPPORT_H
#define VERNON_TESTS_SUPPORT_SCHEDULE_TEST_SUPPORT_H

#include "VernonHal.h"
#include "VernonRuntime.h"
#include "VernonSchedule.h"
#include "backend_test_matrix.h"
#include "program_fixture_manifest_table.h"

#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE)
#include "glfw_context_owner.h"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vernon::tests {

// The fixture's manifests for each target that cooked it; the CPU's only when `includeCpu`.
inline std::vector<ProgramFixtureManifest> programFixtureCases(std::string_view fixtureId, bool includeCpu = false) {
    std::vector<ProgramFixtureManifest> result;
    for (const ProgramFixtureManifest &entry : programFixtureManifestTable)
        if (entry.fixtureId == fixtureId && (includeCpu || entry.runtime != VERNON_RUNTIME_CPU))
            result.push_back(entry);
    return result;
}

inline std::string halError() {
    const VernonStringView error = vernonHalGetLastError();
    return error.data ? std::string(error.data, error.size) : std::string();
}

inline std::string runtimeError(VernonRuntimeContext *runtime) {
    const VernonStringView error = vernonRuntimeGetLastError(runtime);
    return error.data ? std::string(error.data, error.size) : std::string();
}

inline std::string readManifest(const std::filesystem::path &manifestPath) {
    std::ifstream input(manifestPath, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

// A cooked bundle, loaded with its blobs from the manifest's directory, and one executable resolved from it.
class ResolvedProgram {
public:
    ResolvedProgram(VernonRuntimeContext *runtime, const std::filesystem::path &manifestPath,
                    const VernonProgramVariantSelector *selector = nullptr) {
        const std::string manifest = readManifest(manifestPath);
        if (!runtime || manifest.empty())
            return;
        const std::string directory = manifestPath.parent_path().u8string();
        VernonProgramBundleLoadOptions options{};
        options.struct_size = sizeof(options);
        options.bundle_directory = directory.c_str();
        bundle_ = vernonRuntimeLoadProgramBundleWithOptions(runtime, manifest.data(), manifest.size(), &options);
        if (bundle_)
            executable_ = vernonRuntimeResolveProgram(bundle_, selector);
    }
    ResolvedProgram(const ResolvedProgram &) = delete;
    ResolvedProgram &operator=(const ResolvedProgram &) = delete;
    ~ResolvedProgram() {
        vernonRuntimeProgramExecutableDestroy(executable_);
        vernonRuntimeProgramBundleDestroy(bundle_);
    }

    explicit operator bool() const { return executable_ != nullptr; }
    VernonProgramExecutable *get() const { return executable_; }
    VernonProgramBundle *bundle() const { return bundle_; }

private:
    VernonProgramBundle *bundle_{};
    VernonProgramExecutable *executable_{};
};

// A VernonHal.h device of one backend: an OpenGL or OpenGL ES device adopts a hidden GLFW window's context.
class HalTestDevice {
public:
    HalTestDevice() = default;
    HalTestDevice(const HalTestDevice &) = delete;
    HalTestDevice &operator=(const HalTestDevice &) = delete;
    ~HalTestDevice() {
        if (device_)
            EXPECT_EQ(vernonHalDestroyDevice(device_), VERNON_HAL_OK) << halError();
    }

    BackendProbeResult initialize(VernonRuntimeBackend backend, const BackendTestRequirements &requirements) {
        const auto row = std::find_if(backendTestMatrix.begin(), backendTestMatrix.end(),
                                      [&](const BackendTestRow &candidate) { return candidate.runtime == backend; });
        if (row == backendTestMatrix.end())
            return {BackendProbeKind::ProbeFailure, "the backend is not in the test matrix"};
        backend_ = backend;
        VernonHalDeviceDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.backend = backend;
        if (backend == VERNON_RUNTIME_OPENGL || backend == VERNON_RUNTIME_OPENGL_ES) {
#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE)
            const bool desktop = backend == VERNON_RUNTIME_OPENGL;
            const uint16_t major =
                requirements.minimumApiMajor ? requirements.minimumApiMajor : static_cast<uint16_t>(desktop ? 4 : 3);
            const uint16_t minor =
                requirements.minimumApiMajor ? requirements.minimumApiMinor : static_cast<uint16_t>(desktop ? 1 : 0);
            try {
                context_ = std::make_unique<host::GlfwContextOwner>(
                    desktop ? host::GlfwContextApi::OpenGL : host::GlfwContextApi::OpenGLES, major, minor,
                    requirements.windowWidth, requirements.windowHeight);
            } catch (const std::runtime_error &error) {
                return {BackendProbeKind::DeviceOrContextUnavailable,
                        std::string(row->name) + " context is unavailable: " + error.what()};
            }
            callbacks_ = context_->callbacks();
            descriptor.opengl = &callbacks_;
#else
            return {BackendProbeKind::PlatformNotBuilt, std::string(row->name) + " GLFW context owner is not built"};
#endif
        }
        if (vernonHalCreateDevice(&descriptor, &device_) != VERNON_HAL_OK)
            return {BackendProbeKind::DeviceOrContextUnavailable,
                    std::string(row->name) + " device is unavailable: " + halError()};
        return probeRuntimeBackend(*row, requirements, runtime());
    }

    // Adopts the caller's OpenGL or OpenGL ES context, such as a test's fake driver, instead of creating one.
    bool adoptOpenGL(VernonRuntimeBackend backend, const VernonOpenGLContextCallbacks &callbacks, uint32_t flags = 0) {
        backend_ = backend;
        adopted_ = callbacks;
        VernonHalDeviceDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.backend = backend;
        descriptor.opengl = &adopted_;
        descriptor.flags = flags;
        return vernonHalCreateDevice(&descriptor, &device_) == VERNON_HAL_OK;
    }

    VernonHalDevice *get() const { return device_; }
    VernonRuntimeContext *runtime() const { return vernonHalDeviceGetRuntime(device_); }
    VernonRuntimeBackend backend() const { return backend_; }
    // The CPU device's schedules take host tensors; the others' take HAL resource references.
    bool host() const { return backend_ == VERNON_RUNTIME_CPU; }

    // The OpenGL context's window as a surface; empty for other backends.
    std::optional<VernonOpenGLSurfaceCallbacks> openglSurface() const {
#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE)
        if (context_)
            return context_->surfaceCallbacks();
#endif
        return std::nullopt;
    }

private:
    VernonRuntimeBackend backend_{VERNON_RUNTIME_CPU};
    VernonHalDevice *device_{};
    VernonOpenGLContextCallbacks adopted_{};
#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE)
    std::unique_ptr<host::GlfwContextOwner> context_;
    VernonOpenGLContextCallbacks callbacks_{};
#endif
};

// A buffer of `size` bytes in device memory, written with `data` when given.
class HalBuffer {
public:
    HalBuffer(VernonHalDevice *device, uint64_t size, const void *data = nullptr,
              uint32_t usage = VERNON_HAL_BUFFER_STORAGE)
        : size_(size) {
        VernonHalBufferDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.size = size;
        descriptor.usage = usage;
        descriptor.memory = VERNON_HAL_MEMORY_DEVICE;
        EXPECT_EQ(vernonHalCreateBuffer(device, &descriptor, &buffer_), VERNON_HAL_OK) << halError();
        if (buffer_ && data)
            write(data, size);
    }
    HalBuffer(const HalBuffer &) = delete;
    HalBuffer &operator=(const HalBuffer &) = delete;
    ~HalBuffer() { vernonHalDestroyBuffer(buffer_); }

    explicit operator bool() const { return buffer_ != nullptr; }
    VernonHalBuffer *get() const { return buffer_; }
    uint64_t size() const { return size_; }

    void write(const void *data, uint64_t size, uint64_t offset = 0) {
        const VernonHalBufferWrite write{offset, data, size};
        EXPECT_EQ(vernonHalBufferWrite(buffer_, &write, 1), VERNON_HAL_OK) << halError();
    }
    void read(void *data, uint64_t size, uint64_t offset = 0) const {
        const VernonHalBufferRead read{offset, data, size};
        EXPECT_EQ(vernonHalBufferRead(buffer_, &read, 1), VERNON_HAL_OK) << halError();
    }
    template <typename T> std::vector<T> values(size_t count) const {
        std::vector<T> result(count);
        read(result.data(), count * sizeof(T));
        return result;
    }

    VernonRuntimeProviderResourceReference reference() const {
        VernonRuntimeProviderResourceReference result{};
        EXPECT_EQ(vernonHalReferenceBuffer(buffer_, 0, size_, &result), VERNON_HAL_OK) << halError();
        return result;
    }
    void *address() const {
        void *result = nullptr;
        EXPECT_EQ(vernonHalBufferGetAddress(buffer_, &result), VERNON_HAL_OK) << halError();
        return result;
    }

private:
    VernonHalBuffer *buffer_{};
    uint64_t size_{};
};

// A tensor in a device buffer: the view a schedule argument passes, which points at this object's shape and strides.
class DeviceTensor {
public:
    // `byteStrides` defaults to the dense row-major strides of `elementSize`-byte elements.
    DeviceTensor(const HalTestDevice &device, const void *data, uint64_t bytes, std::vector<uint64_t> shape,
                 uint64_t elementSize, std::vector<int64_t> byteStrides = {}, uint64_t byteOffset = 0,
                 uint32_t usage = VERNON_HAL_BUFFER_STORAGE)
        : buffer_(device.get(), bytes, data, usage), host_(device.host()), shape_(std::move(shape)),
          strides_(std::move(byteStrides)), offset_(byteOffset) {
        if (strides_.empty()) {
            strides_.resize(shape_.size());
            int64_t stride = static_cast<int64_t>(elementSize);
            for (size_t axis = shape_.size(); axis-- > 0;) {
                strides_[axis] = stride;
                stride *= static_cast<int64_t>(shape_[axis]);
            }
        }
    }
    template <typename T>
    DeviceTensor(const HalTestDevice &device, const std::vector<T> &values, std::vector<uint64_t> shape = {},
                 std::vector<int64_t> byteStrides = {}, uint64_t byteOffset = 0)
        : DeviceTensor(device, values.data(), values.size() * sizeof(T),
                       shape.empty() ? std::vector<uint64_t>{values.size()} : std::move(shape), sizeof(T),
                       std::move(byteStrides), byteOffset) {}

    HalBuffer &buffer() { return buffer_; }
    const HalBuffer &buffer() const { return buffer_; }
    template <typename T> std::vector<T> values() const { return buffer_.values<T>(buffer_.size() / sizeof(T)); }

    // The view without its element layout and access, which the record fills from the boundary it binds.
    VernonTensorView view() const {
        VernonTensorView result{};
        result.struct_size = sizeof(result);
        if (host_) {
            result.storage = VERNON_TENSOR_HOST;
            result.host_data = buffer_.address();
        } else {
            result.storage = VERNON_TENSOR_RHI_RESOURCE;
            result.resource = buffer_.reference();
        }
        result.rank = static_cast<uint32_t>(shape_.size());
        result.shape = shape_.empty() ? nullptr : shape_.data();
        result.byte_strides = strides_.empty() ? nullptr : strides_.data();
        result.byte_offset = static_cast<size_t>(offset_);
        result.byte_size = static_cast<size_t>(buffer_.size());
        return result;
    }

private:
    HalBuffer buffer_;
    bool host_;
    std::vector<uint64_t> shape_;
    std::vector<int64_t> strides_;
    uint64_t offset_;
};

// A host tensor view of `data`; the record fills its element layout and access.
inline VernonTensorView hostView(void *data, size_t byteSize, uint32_t rank = 0, const uint64_t *shape = nullptr,
                                 const int64_t *byteStrides = nullptr, size_t byteOffset = 0) {
    VernonTensorView result{};
    result.struct_size = sizeof(result);
    result.storage = VERNON_TENSOR_HOST;
    result.host_data = data;
    result.rank = rank;
    result.shape = shape;
    result.byte_strides = byteStrides;
    result.byte_offset = byteOffset;
    result.byte_size = byteSize;
    return result;
}

// Dense host values a CPU schedule argument views; the view points at this object, which must outlive the call. An
// empty shape is a scalar.
template <typename T> struct HostTensor {
    explicit HostTensor(std::vector<T> initial)
        : values(std::move(initial)), shape{values.size()}, strides{static_cast<int64_t>(sizeof(T))} {}
    HostTensor(std::vector<T> initial, std::vector<uint64_t> dimensions)
        : values(std::move(initial)), shape(std::move(dimensions)), strides(shape.size()) {
        int64_t stride = sizeof(T);
        for (size_t axis = shape.size(); axis-- > 0;) {
            strides[axis] = stride;
            stride *= static_cast<int64_t>(shape[axis]);
        }
    }
    HostTensor(const HostTensor &) = delete;
    HostTensor &operator=(const HostTensor &) = delete;

    VernonTensorView view() {
        return hostView(values.data(), values.size() * sizeof(T), static_cast<uint32_t>(shape.size()),
                        shape.empty() ? nullptr : shape.data(), strides.empty() ? nullptr : strides.data());
    }

    std::vector<T> values;
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;
};

// A 2D or cube image of one mip level, and a view of all its layers and the aspects `viewAspects`.
class HalImage {
public:
    HalImage(VernonHalDevice *device, VernonTextureFormat format, uint32_t width, uint32_t height, uint32_t usage,
             uint32_t viewAspects = VERNON_IMAGE_ASPECT_COLOR, VernonTextureDimension dimension = VERNON_TEXTURE_2D)
        : width_(width), height_(height) {
        const uint32_t layers = dimension == VERNON_TEXTURE_CUBE ? 6 : 1;
        VernonImageDescriptor descriptor{};
        descriptor.dimension = dimension;
        descriptor.extent = {width, height, 1};
        descriptor.format = format;
        descriptor.mip_level_count = 1;
        descriptor.array_layer_count = layers;
        descriptor.sample_count = 1;
        descriptor.usage = usage | VERNON_IMAGE_TRANSFER_SOURCE | VERNON_IMAGE_TRANSFER_DESTINATION;
        EXPECT_EQ(vernonHalCreateImage(device, &descriptor, &image_), VERNON_HAL_OK) << halError();
        if (!image_)
            return;
        VernonImageViewDescriptor view{};
        view.dimension = dimension;
        view.format = format;
        view.subresources = {0, 1, 0, layers, viewAspects};
        EXPECT_EQ(vernonHalCreateImageView(image_, &view, &view_), VERNON_HAL_OK) << halError();
        if (view_)
            EXPECT_EQ(vernonHalReferenceImageView(view_, &reference_), VERNON_HAL_OK) << halError();
    }
    HalImage(const HalImage &) = delete;
    HalImage &operator=(const HalImage &) = delete;
    ~HalImage() {
        if (view_)
            EXPECT_EQ(vernonHalDestroyImageView(view_), VERNON_HAL_OK) << halError();
        if (image_)
            EXPECT_EQ(vernonHalDestroyImage(image_), VERNON_HAL_OK) << halError();
    }

    explicit operator bool() const { return view_ != nullptr; }
    VernonRuntimeProviderResourceReference reference() const { return reference_; }

    VernonHalStatus write(const void *data, VernonHalImageDataFormat format, VernonHalImageDataType type,
                          uint32_t aspect = VERNON_IMAGE_ASPECT_COLOR, uint32_t layer = 0) {
        VernonHalImageWrite write{};
        write.region = region(format, type, aspect);
        write.region.array_layer = layer;
        write.source = data;
        return vernonHalImageWrite(image_, &write, 1);
    }

    // The whole image's texels in `format` and `type`, `texelBytes` each.
    template <typename T>
    std::vector<T> read(VernonHalImageDataFormat format, VernonHalImageDataType type, size_t texelBytes,
                        uint32_t aspect = VERNON_IMAGE_ASPECT_COLOR) const {
        const size_t bytes = static_cast<size_t>(width_) * height_ * texelBytes;
        std::vector<T> result((bytes + sizeof(T) - 1) / sizeof(T));
        VernonHalImageRead read{};
        read.region = region(format, type, aspect);
        read.destination = result.data();
        read.size = bytes;
        EXPECT_EQ(vernonHalImageRead(image_, &read, 1), VERNON_HAL_OK) << halError();
        return result;
    }
    std::vector<uint8_t> rgba8() const {
        return read<uint8_t>(VERNON_HAL_IMAGE_DATA_RGBA, VERNON_HAL_IMAGE_DATA_UINT8, 4);
    }

private:
    VernonHalImageRegion region(VernonHalImageDataFormat format, VernonHalImageDataType type, uint32_t aspect) const {
        VernonHalImageRegion result{};
        result.aspect = aspect;
        result.width = width_;
        result.height = height_;
        result.depth = 1;
        result.data_format = format;
        result.data_type = type;
        return result;
    }

    VernonHalImage *image_{};
    VernonHalImageView *view_{};
    VernonRuntimeProviderResourceReference reference_{};
    uint32_t width_;
    uint32_t height_;
};

class HalSampler {
public:
    HalSampler(VernonHalDevice *device, VernonHalFilter filter,
               VernonHalAddressMode address = VERNON_HAL_ADDRESS_CLAMP_TO_EDGE) {
        VernonHalSamplerDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.min_filter = filter;
        descriptor.mag_filter = filter;
        descriptor.mip_filter = VERNON_HAL_FILTER_NEAREST;
        descriptor.address_u = address;
        descriptor.address_v = address;
        descriptor.address_w = address;
        EXPECT_EQ(vernonHalCreateSampler(device, &descriptor, &sampler_), VERNON_HAL_OK) << halError();
        if (sampler_)
            EXPECT_EQ(vernonHalReferenceSampler(sampler_, &reference_), VERNON_HAL_OK) << halError();
    }
    HalSampler(const HalSampler &) = delete;
    HalSampler &operator=(const HalSampler &) = delete;
    ~HalSampler() { vernonHalDestroySampler(sampler_); }

    explicit operator bool() const { return sampler_ != nullptr; }
    VernonRuntimeProviderResourceReference reference() const { return reference_; }

private:
    VernonHalSampler *sampler_{};
    VernonRuntimeProviderResourceReference reference_{};
};

// The caller record of a schedule linked from a Program executable, whose ARGUMENT slots bind by the name of the
// boundary each takes: a name binds every slot of a boundary so named, so an output that republishes an input binds
// with it. A backward schedule's slot takes the cotangent or gradient boundary of the primal its slot names.
class ScheduleRecord {
public:
    ScheduleRecord() = default;
    ScheduleRecord(VernonProgramExecutable *executable, const VernonLinkedSchedule *schedule,
                   const VernonScheduleVjp *backwardOf = nullptr)
        : schedule_(schedule) {
        const uint8_t *bytes = vernonScheduleRecordTemplate(schedule);
        record_.assign(bytes, bytes + vernonScheduleRecordSize(schedule));
        const VernonScheduleSlot *slots = vernonScheduleSlots(schedule);
        for (size_t index = 0; index < vernonScheduleSlotCount(schedule); ++index) {
            const uint32_t primal = vernonScheduleSlotBoundary(schedule, index);
            if (slots[index].kind != VERNON_SCHEDULE_SLOT_ARGUMENT || primal == UINT32_MAX)
                continue;
            Argument argument;
            argument.offset = slots[index].offset;
            if (backwardOf) {
                argument.role = vernonScheduleVjpSlotRole(backwardOf, index);
                const std::string primalName = boundaryName(executable, primal);
                // A composed Program graph declares no derivative boundaries: its slot takes the primal's layout.
                if (!findDerivative(executable, argument.role, primalName, argument.view)) {
                    VernonProgramBoundaryRole primalRole{};
                    if (!findBoundary(executable, primal, primalRole, argument.view))
                        throw std::logic_error("a backward slot differentiates no Program boundary");
                    argument.view.access =
                        argument.role == VERNON_PROGRAM_BOUNDARY_COTANGENT ? VERNON_ACCESS_READ : VERNON_ACCESS_WRITE;
                }
            } else if (!findBoundary(executable, primal, argument.role, argument.view)) {
                throw std::logic_error("a schedule slot takes no Program boundary");
            }
            argument.name.assign(argument.view.name.data, argument.view.name.size);
            arguments_.push_back(std::move(argument));
        }
    }

    // Binds the tensor slots named `name`; `role` limits them to one boundary role.
    bool bind(std::string_view name, const VernonTensorView &view,
              std::optional<VernonProgramBoundaryRole> role = std::nullopt) {
        return bindEach(name, role, [&](const Argument &argument) {
            VernonProgramArgument written{};
            written.slot = argument.view.slot;
            written.kind = VERNON_PROGRAM_TENSOR;
            written.tensor = view;
            written.tensor.element_layout = argument.view.element_layout;
            written.tensor.access = argument.view.access;
            return written;
        });
    }
    bool bind(std::string_view name, const DeviceTensor &tensor,
              std::optional<VernonProgramBoundaryRole> role = std::nullopt) {
        return bind(name, tensor.view(), role);
    }
    // Binds a scalar, which the record keeps.
    template <typename T> bool bindScalar(std::string_view name, T value) {
        scalars_.emplace_back();
        std::memcpy(scalars_.back().data(), &value, sizeof(value));
        void *data = scalars_.back().data();
        return bindEach(name, std::nullopt, [&](const Argument &argument) {
            VernonProgramArgument written{};
            written.slot = argument.view.slot;
            written.kind = VERNON_PROGRAM_TENSOR;
            written.tensor = hostView(data, sizeof(value));
            written.tensor.element_layout = argument.view.element_layout;
            written.tensor.access = VERNON_ACCESS_READ;
            return written;
        });
    }
    bool bindImage(std::string_view name, VernonRuntimeProviderResourceReference view) {
        return bindEach(name, std::nullopt, [&](const Argument &argument) {
            VernonProgramArgument written{};
            written.slot = argument.view.slot;
            written.kind = VERNON_PROGRAM_IMAGE;
            written.image = {view};
            return written;
        });
    }
    bool bindSampler(std::string_view name, VernonRuntimeProviderResourceReference sampler) {
        return bindEach(name, std::nullopt, [&](const Argument &argument) {
            VernonProgramArgument written{};
            written.slot = argument.view.slot;
            written.kind = VERNON_PROGRAM_SAMPLER;
            written.resource = sampler;
            return written;
        });
    }
    // Binds the grid axes the schedule takes as `__grid_x`, `__grid_y` and `__grid_z`; a fixed grid takes none.
    void bindGrid(uint32_t x, uint32_t y = 1, uint32_t z = 1) {
        const uint32_t axes[]{x, y, z};
        const char *names[]{"__grid_x", "__grid_y", "__grid_z"};
        for (size_t axis = 0; axis < 3; ++axis)
            if (takes(names[axis]))
                bindScalar(names[axis], axes[axis]);
    }
    bool takes(std::string_view name) const {
        return std::any_of(arguments_.begin(), arguments_.end(),
                           [&](const Argument &argument) { return argument.name == name; });
    }

    // The names of the slots nothing bound.
    std::vector<std::string> unbound() const {
        std::vector<std::string> result;
        for (const Argument &argument : arguments_)
            if (!argument.bound)
                result.push_back(argument.name);
        return result;
    }

    template <typename T> void write(size_t slot, const T &value) {
        ASSERT_LT(slot, vernonScheduleSlotCount(schedule_));
        const VernonScheduleSlot &linked = vernonScheduleSlots(schedule_)[slot];
        ASSERT_EQ(linked.size, sizeof(value));
        std::memcpy(record_.data() + linked.offset, &value, sizeof(value));
    }
    template <typename T> void writeNode(size_t node, size_t index, const T &value) {
        const uint32_t linked = vernonScheduleNodeSlot(schedule_, node, index);
        ASSERT_NE(linked, UINT32_MAX);
        write(linked, value);
    }
    // The PASS slot of the `graphicsNode`th graphics node, or SIZE_MAX.
    size_t passSlot(size_t graphicsNode) const {
        const VernonScheduleSlot *slots = vernonScheduleSlots(schedule_);
        for (size_t index = 0; index < vernonScheduleSlotCount(schedule_); ++index)
            if (slots[index].kind == VERNON_SCHEDULE_SLOT_PASS && graphicsNode-- == 0)
                return index;
        return SIZE_MAX;
    }

    const uint8_t *data() const { return record_.data(); }
    size_t size() const { return record_.size(); }

private:
    struct Argument {
        uint64_t offset{};
        VernonProgramBoundaryRole role{};
        VernonProgramParameterView view{};
        std::string name;
        bool bound{};
    };

    static bool findBoundary(VernonProgramExecutable *executable, uint32_t slot, VernonProgramBoundaryRole &role,
                             VernonProgramParameterView &view) {
        for (VernonProgramBoundaryRole candidate : {VERNON_PROGRAM_BOUNDARY_INPUT, VERNON_PROGRAM_BOUNDARY_OUTPUT})
            for (size_t index = 0; index < vernonRuntimeProgramExecutableGetBoundaryCount(executable, candidate);
                 ++index)
                if (vernonRuntimeProgramExecutableGetBoundaryByIndex(executable, candidate, index, &view) ==
                        VERNON_STATUS_OK &&
                    view.slot == slot) {
                    role = candidate;
                    return true;
                }
        return false;
    }
    static std::string boundaryName(VernonProgramExecutable *executable, uint32_t slot) {
        VernonProgramBoundaryRole role{};
        VernonProgramParameterView view{};
        return findBoundary(executable, slot, role, view) ? std::string(view.name.data, view.name.size) : std::string();
    }
    // The derivative boundary of `role` named as the primal, or as a field path into it.
    static bool findDerivative(VernonProgramExecutable *executable, VernonProgramBoundaryRole role,
                               const std::string &primal, VernonProgramParameterView &view) {
        std::optional<VernonProgramParameterView> field;
        for (size_t index = 0;
             !primal.empty() && index < vernonRuntimeProgramExecutableGetBoundaryCount(executable, role); ++index) {
            VernonProgramParameterView candidate{};
            if (vernonRuntimeProgramExecutableGetBoundaryByIndex(executable, role, index, &candidate) !=
                VERNON_STATUS_OK)
                continue;
            const std::string_view name(candidate.name.data, candidate.name.size);
            if (name == primal) {
                view = candidate;
                return true;
            }
            if (name.size() > primal.size() && name[primal.size()] == '.' && name.substr(0, primal.size()) == primal)
                field = candidate;
        }
        if (field)
            view = *field;
        return field.has_value();
    }

    template <typename Make>
    bool bindEach(std::string_view name, std::optional<VernonProgramBoundaryRole> role, Make make) {
        bool found = false;
        for (Argument &argument : arguments_) {
            if (argument.name != name || (role && argument.role != *role))
                continue;
            const VernonProgramArgument written = make(argument);
            std::memcpy(record_.data() + argument.offset, &written, sizeof(written));
            argument.bound = true;
            found = true;
        }
        EXPECT_TRUE(found) << "no schedule slot takes '" << name << "'";
        return found;
    }

    const VernonLinkedSchedule *schedule_{};
    std::vector<uint8_t> record_;
    std::vector<Argument> arguments_;
    std::deque<std::array<uint8_t, 16>> scalars_;
};

struct ScheduleDeleter {
    void operator()(VernonLinkedSchedule *schedule) const { vernonScheduleDestroy(schedule); }
};

// A Program's forward graph linked as one schedule, and the record it is called with.
class LinkedProgram {
public:
    LinkedProgram(VernonRuntimeContext *runtime, VernonProgramExecutable *executable) : runtime_(runtime) {
        VernonScheduleProgramDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.graph = {"forward", 7};
        descriptor.executable = executable;
        schedule_.reset(vernonScheduleLinkProgram(runtime, &descriptor));
        if (schedule_)
            record_ = ScheduleRecord(executable, schedule_.get());
    }

    explicit operator bool() const { return schedule_ != nullptr; }
    VernonLinkedSchedule *get() const { return schedule_.get(); }
    ScheduleRecord &record() { return record_; }
    std::string error() const { return runtimeError(runtime_); }

    // Calls and, when the call succeeds, waits for it.
    VernonStatus call(uint64_t *point = nullptr) {
        uint64_t submitted = 0;
        const VernonStatus status = vernonScheduleCall(schedule_.get(), record_.data(), record_.size(), &submitted);
        if (status == VERNON_STATUS_OK)
            EXPECT_EQ(vernonScheduleWait(runtime_, submitted), VERNON_STATUS_OK) << error();
        if (point)
            *point = submitted;
        return status;
    }

    VernonScheduleGraphicsTelemetry telemetry() const {
        VernonScheduleGraphicsTelemetry result{};
        result.struct_size = sizeof(result);
        EXPECT_EQ(vernonScheduleGetGraphicsTelemetry(schedule_.get(), &result), VERNON_STATUS_OK);
        return result;
    }

private:
    VernonRuntimeContext *runtime_;
    std::unique_ptr<VernonLinkedSchedule, ScheduleDeleter> schedule_;
    ScheduleRecord record_;
};

// A Program's forward and backward graphs linked as a VJP, with a record for each; a forward call keeps the residuals
// the next backward call reads.
class LinkedVjp {
public:
    LinkedVjp(VernonRuntimeContext *runtime, VernonProgramExecutable *executable,
              const VernonScheduleCheckpointOptions *checkpoint = nullptr)
        : runtime_(runtime) {
        vjp_.reset(checkpoint ? vernonScheduleLinkCheckpointedVjp(runtime, executable, checkpoint)
                              : vernonScheduleLinkVjp(runtime, executable));
        if (!vjp_)
            return;
        forward_ = ScheduleRecord(executable, vernonScheduleVjpSchedule(vjp_.get(), 0));
        backward_ = ScheduleRecord(executable, vernonScheduleVjpSchedule(vjp_.get(), 1), vjp_.get());
    }

    explicit operator bool() const { return vjp_ != nullptr; }
    VernonScheduleVjp *get() const { return vjp_.get(); }
    ScheduleRecord &forwardRecord() { return forward_; }
    ScheduleRecord &backwardRecord() { return backward_; }
    const VernonScheduleResiduals *residuals() const { return residuals_.get(); }
    std::string error() const { return runtimeError(runtime_); }

    // What every pass of the last forward call left, summed; `dynamic_tape` counts the passes with one.
    VernonScheduleVjpPass passTotals() const {
        VernonScheduleVjpPass totals{};
        for (size_t index = 0; index < vernonScheduleVjpPassCount(vjp_.get()); ++index) {
            VernonScheduleVjpPass pass{};
            pass.struct_size = sizeof(pass);
            EXPECT_EQ(vernonScheduleVjpGetPass(vjp_.get(), residuals_.get(), index, &pass), VERNON_STATUS_OK)
                << error();
            totals.tape_count += pass.tape_count;
            totals.dynamic_tape += pass.dynamic_tape ? 1 : 0;
            totals.tape_logical_bytes += pass.tape_logical_bytes;
            totals.tape_resident_bytes += pass.tape_resident_bytes;
            totals.tape_allocated_bytes += pass.tape_allocated_bytes;
            totals.residual_bytes += pass.residual_bytes;
        }
        return totals;
    }

    VernonStatus forward() {
        VernonScheduleResiduals *residuals = nullptr;
        uint64_t point = 0;
        const VernonStatus status =
            vernonScheduleVjpForward(vjp_.get(), forward_.data(), forward_.size(), &residuals, &point);
        if (status != VERNON_STATUS_OK)
            return status;
        residuals_.reset(residuals);
        EXPECT_EQ(vernonScheduleWait(runtime_, point), VERNON_STATUS_OK) << error();
        return status;
    }
    VernonStatus backward() {
        uint64_t point = 0;
        const VernonStatus status =
            vernonScheduleVjpBackward(vjp_.get(), backward_.data(), backward_.size(), residuals_.get(), &point);
        if (status == VERNON_STATUS_OK)
            EXPECT_EQ(vernonScheduleWait(runtime_, point), VERNON_STATUS_OK) << error();
        return status;
    }

private:
    struct VjpDeleter {
        void operator()(VernonScheduleVjp *vjp) const { vernonScheduleVjpDestroy(vjp); }
    };
    struct ResidualsDeleter {
        void operator()(VernonScheduleResiduals *residuals) const { vernonScheduleResidualsDestroy(residuals); }
    };

    VernonRuntimeContext *runtime_;
    std::unique_ptr<VernonScheduleVjp, VjpDeleter> vjp_;
    ScheduleRecord forward_;
    ScheduleRecord backward_;
    std::unique_ptr<VernonScheduleResiduals, ResidualsDeleter> residuals_;
};

// A clearing, preserving color attachment and the render pass and dynamic state of a full `extent` target.
struct PassControls {
    VernonColorAttachment attachment{};
    VernonRenderPass renderPass{};
    VernonDynamicState dynamic{};
    const VernonRenderPass *pointer{&renderPass};

    explicit PassControls(VernonRuntimeProviderResourceReference view, uint32_t extent) {
        attachment.view = view;
        attachment.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
        attachment.store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
        attachment.clear_color.f32[3] = 1.0f;
        renderPass.struct_size = sizeof(renderPass);
        renderPass.color_attachments = &attachment;
        renderPass.color_attachment_count = 1;
        renderPass.render_area[2] = extent;
        renderPass.render_area[3] = extent;
        dynamic.struct_size = sizeof(dynamic);
        dynamic.viewport[2] = static_cast<float>(extent);
        dynamic.viewport[3] = static_cast<float>(extent);
        dynamic.scissor[2] = extent;
        dynamic.scissor[3] = extent;
    }
    PassControls(const PassControls &) = delete;
    PassControls &operator=(const PassControls &) = delete;

    // Writes the render pass and dynamic state of the `graphicsNode`th graphics node of `record`'s schedule, and
    // `draw` when given; the node's slots follow its PASS slot as PASS, DRAW, DYNAMIC, STREAM and TABLE.
    void writeTo(ScheduleRecord &record, size_t graphicsNode = 0, const VernonDrawCommand *draw = nullptr) const {
        const size_t pass = record.passSlot(graphicsNode);
        ASSERT_NE(pass, SIZE_MAX);
        record.write(pass, pointer);
        if (draw)
            record.write(pass + 1, *draw);
        record.write(pass + 2, dynamic);
    }
};

// A Program fixture's test on a VernonHal.h device of its target; a device the machine lacks skips it.
class HalProgramTest : public testing::TestWithParam<ProgramFixtureManifest> {
protected:
    void SetUp() override {
        const BackendProbeResult probe = device_.initialize(GetParam().runtime, requirements());
        if (!probe.available()) {
            if (probe.skippable())
                GTEST_SKIP() << probe.reason;
            FAIL() << probe.reason;
        }
    }

    virtual BackendTestRequirements requirements() const = 0;

    const ProgramFixtureManifest &fixture(std::string_view fixtureId) const {
        const ProgramFixtureManifest *result = findProgramFixtureManifest(fixtureId, GetParam().target);
        if (!result)
            throw std::logic_error("missing fixture '" + std::string(fixtureId) + "' for target '" +
                                   std::string(GetParam().target) + "'");
        return *result;
    }

    VernonRuntimeContext *runtime() const { return device_.runtime(); }

    HalTestDevice device_;
};

} // namespace vernon::tests

#endif
