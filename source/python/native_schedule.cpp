#include "native_schedule.h"
#include "VernonSchedule.h"
#include "native_hal.h"
#include "native_program_autodiff.h"
#include "native_runtime.h"
#include "runtime/dirty_range_set.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// The attribute names a call reads and writes, interned once: a call touches a dozen attributes per buffer argument.
struct Names {
    PyObject *control, *poisoned, *claims, *ioReservations, *residencies, *entries, *version, *hostVersion, *coherence,
        *dirtyRanges, *native, *array, *handle, *owner, *components, *access, *offset, *shape, *strides, *timeline,
        *writePoint, *deviceDirty, *deviceDirtyRanges, *region, *ranges, *preserved;
};

const Names &names() {
    static const Names interned = [] {
        const auto intern = [](const char *text) { return PyUnicode_InternFromString(text); };
        return Names{intern("_control"),
                     intern("poisoned"),
                     intern("_claims"),
                     intern("_io_reservations"),
                     intern("residencies"),
                     intern("_entries"),
                     intern("version"),
                     intern("host_version"),
                     intern("coherence"),
                     intern("dirty_ranges"),
                     intern("_native"),
                     intern("_array"),
                     intern("handle"),
                     intern("_owner"),
                     intern("_components"),
                     intern("_access"),
                     intern("_offset"),
                     intern("_shape"),
                     intern("_strides"),
                     intern("timeline"),
                     intern("write_point"),
                     intern("device_dirty"),
                     intern("device_dirty_ranges"),
                     intern("_region"),
                     intern("ranges"),
                     intern("preserved")};
    }();
    return interned;
}

nb::object attribute(PyObject *object, PyObject *name) {
    PyObject *value = PyObject_GetAttr(object, name);
    if (!value)
        throw nb::python_error();
    return nb::steal(value);
}

void setAttribute(PyObject *object, PyObject *name, PyObject *value) {
    if (PyObject_SetAttr(object, name, value))
        throw nb::python_error();
}

struct ScheduleDeleter {
    void operator()(VernonLinkedSchedule *schedule) const noexcept { vernonScheduleDestroy(schedule); }
};

class BufferViews {
public:
    BufferViews() = default;
    BufferViews(const BufferViews &) = delete;
    BufferViews &operator=(const BufferViews &) = delete;
    ~BufferViews() {
        for (Py_buffer &view : views_)
            PyBuffer_Release(&view);
    }

    Py_buffer *acquire(PyObject *array, bool writable) {
        Py_buffer view{};
        if (PyObject_GetBuffer(array, &view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | (writable ? PyBUF_WRITABLE : 0))) {
            PyErr_Clear();
            return nullptr;
        }
        views_.push_back(view);
        return &views_.back();
    }

    void reserve(size_t count) { views_.reserve(count); }

private:
    std::vector<Py_buffer> views_;
};

bool formatMatches(uint32_t dtype, const Py_buffer &view) {
    const char *format = view.format ? view.format : "B";
    while (*format == '@' || *format == '=' || *format == '<')
        ++format;
    if (!format[0] || format[1])
        return false;
    switch (dtype) {
    case VERNON_DATA_BOOL:
        return format[0] == '?';
    case VERNON_DATA_I32:
        return view.itemsize == 4 && (format[0] == 'i' || format[0] == 'l');
    case VERNON_DATA_U32:
        return view.itemsize == 4 && (format[0] == 'I' || format[0] == 'L');
    case VERNON_DATA_F16:
        return format[0] == 'e';
    case VERNON_DATA_F32:
        return format[0] == 'f';
    case VERNON_DATA_F64:
        return format[0] == 'd';
    case VERNON_DATA_U8:
        return format[0] == 'B';
    default:
        return false;
    }
}

bool structuredFormat(const Py_buffer &view) {
    const char *format = view.format ? view.format : "B";
    while (*format == '@' || *format == '=' || *format == '<')
        ++format;
    return format[0] == 'T' && format[1] == '{';
}

// The element type formatMatches accepts for `view`, or UINT32_MAX.
uint32_t formatDataType(const Py_buffer &view) {
    for (uint32_t dtype : {VERNON_DATA_BOOL, VERNON_DATA_I32, VERNON_DATA_U32, VERNON_DATA_F16, VERNON_DATA_F32,
                           VERNON_DATA_F64, VERNON_DATA_U8})
        if (formatMatches(dtype, view))
            return dtype;
    return UINT32_MAX;
}

// The NumPy scalar types a scalar slot accepts besides Python's own numbers.
struct NumpyScalars {
    nb::object integer;
    nb::object floating;

    bool isInteger(PyObject *value) const { return PyObject_IsInstance(value, integer.ptr()) == 1; }
    bool isFloating(PyObject *value) const { return PyObject_IsInstance(value, floating.ptr()) == 1; }
};

bool packScalar(PyObject *value, const VernonScheduleSlot &slot, const NumpyScalars &numpy, uint8_t *out) {
    std::memset(out, 0, slot.size);
    if (PyBool_Check(value)) {
        if (slot.dtype != VERNON_DATA_BOOL)
            return false;
        out[0] = value == Py_True ? 1 : 0;
        return true;
    }
    switch (slot.dtype) {
    case VERNON_DATA_F32:
    case VERNON_DATA_F64: {
        if (!PyFloat_Check(value) && !PyLong_Check(value) && !numpy.isFloating(value) && !numpy.isInteger(value))
            return false;
        const double converted = PyFloat_AsDouble(value);
        if (converted == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            return false;
        }
        if (slot.dtype == VERNON_DATA_F64 && slot.size == sizeof(double)) {
            std::memcpy(out, &converted, sizeof(double));
            return true;
        }
        if (slot.dtype == VERNON_DATA_F32 && slot.size == sizeof(float)) {
            const auto narrowed = static_cast<float>(converted);
            std::memcpy(out, &narrowed, sizeof(float));
            return true;
        }
        return false;
    }
    case VERNON_DATA_I32:
    case VERNON_DATA_U32:
    case VERNON_DATA_U8: {
        nb::object integer;
        if (PyLong_Check(value)) {
            integer = nb::borrow(value);
        } else if (numpy.isInteger(value)) {
            integer = nb::steal(PyNumber_Index(value));
            if (!integer.is_valid()) {
                PyErr_Clear();
                return false;
            }
        } else {
            return false;
        }
        int overflow = 0;
        const long long converted = PyLong_AsLongLongAndOverflow(integer.ptr(), &overflow);
        if (overflow || (converted == -1 && PyErr_Occurred())) {
            PyErr_Clear();
            return false;
        }
        long long lowest = 0;
        long long highest = std::numeric_limits<uint8_t>::max();
        if (slot.dtype == VERNON_DATA_I32) {
            lowest = std::numeric_limits<int32_t>::min();
            highest = std::numeric_limits<int32_t>::max();
        } else if (slot.dtype == VERNON_DATA_U32) {
            highest = std::numeric_limits<uint32_t>::max();
        }
        const size_t width = slot.dtype == VERNON_DATA_U8 ? 1 : 4;
        if (converted < lowest || converted > highest || slot.size != width)
            return false;
        if (width == 1) {
            out[0] = static_cast<uint8_t>(converted);
        } else {
            const auto narrowed = static_cast<uint32_t>(converted);
            std::memcpy(out, &narrowed, sizeof(narrowed));
        }
        return true;
    }
    default:
        return false;
    }
}

bool readIndex(PyObject *value, int64_t &out) {
    if (!PyLong_Check(value) || PyBool_Check(value))
        return false;
    int overflow = 0;
    const long long converted = PyLong_AsLongLongAndOverflow(value, &overflow);
    if (overflow || (converted == -1 && PyErr_Occurred())) {
        PyErr_Clear();
        return false;
    }
    out = converted;
    return true;
}

// Reads a tuple of Python ints into `out`; false for anything else.
bool readIndices(PyObject *tuple, std::vector<int64_t> &out) {
    if (!PyTuple_CheckExact(tuple))
        return false;
    const Py_ssize_t size = PyTuple_GET_SIZE(tuple);
    out.resize(static_cast<size_t>(size));
    for (Py_ssize_t index = 0; index < size; ++index)
        if (!readIndex(PyTuple_GET_ITEM(tuple, index), out[static_cast<size_t>(index)]))
            return false;
    return true;
}

// The element-unit form of tensorByteLayoutInjective: no two indices of a writable view name one element.
bool injective(const std::vector<int64_t> &shape, const std::vector<int64_t> &strides) {
    std::vector<std::pair<uint64_t, uint64_t>> dimensions;
    dimensions.reserve(shape.size());
    for (size_t dimension = 0; dimension < shape.size(); ++dimension) {
        if (!shape[dimension])
            return true;
        if (shape[dimension] > 1)
            dimensions.emplace_back(strides[dimension] < 0 ? 0 - static_cast<uint64_t>(strides[dimension])
                                                           : static_cast<uint64_t>(strides[dimension]),
                                    static_cast<uint64_t>(shape[dimension]));
    }
    std::sort(dimensions.begin(), dimensions.end());
    uint64_t covered = 0;
    for (auto [stride, extent] : dimensions) {
        if (stride <= covered || stride > (std::numeric_limits<uint64_t>::max() - covered) / (extent - 1))
            return false;
        covered += stride * (extent - 1);
    }
    return true;
}

struct BufferArgument {
    PyObject *owner{};
    // The session's residency of the owner, on a GPU, and its buffer, which a Program argument references.
    PyObject *residency{};
    HalBuffer *device{};
    VernonRuntimeProviderResourceReference reference{};
    void *data{};
    uint32_t dtype{};
    // An element of named fields, which the executable lays out; `dtype` is then its first field's.
    bool structured{};
    int64_t itemSize{};
    int64_t capacity{};
    int64_t offset{};
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    // The host bytes the view spans, on the CPU.
    const uint8_t *spanBegin{};
    size_t spanBytes{};
    // The owner's version when the call bound it.
    int64_t version{};
    bool writes{};
};

// A bound buffer as a call last bound it. Every write to an owner, by the host, a device or an upload, advances its
// control version, so while the version, the host array and the session's residency are unchanged the binding holds,
// and a call checks only those and the owner's borrows.
struct BoundBinding {
    bool valid{};
    nb::object control;
    // Held, so an object that replaces one cannot take its address.
    nb::object array;
    nb::object residency;
    // The instance dicts of the control block and the owner, and the control's residency set and its entries, which
    // revalidate the binding without attribute lookups; null when an object has no instance dict.
    nb::object controlDict;
    nb::object ownerDict;
    nb::object residencies;
    nb::object entries;
    // A device write's publication: the control's coherence and the owner's region ranges, which a Storage's never
    // change; each publication sets a copy of the ranges.
    nb::object coherence;
    nb::object ranges;
    BufferArgument buffer;
};

// The slots of a bound view whose values the view holds, each with its object, and the slots a call's arguments fill,
// in order.
struct BoundSlots {
    std::vector<nb::object> values;
    std::vector<uint32_t> unbound;
    mutable std::vector<BoundBinding> bindings;
};

// The residuals of one linked VJP call in one buffer: the forward schedule writes each at its entry's offset and the
// backward schedule reads it there, so a call holds one block rather than a tensor per residual. Nothing else can bind
// a block's bytes, so they need no version or residency. An autodiff tape residual is a VernonScheduleTape the block
// owns rather than bytes in it.
class ResidualBlock {
public:
    struct Entry {
        uint64_t offset{};
        int64_t elements{};
        uint32_t dtype{};
        uint64_t elementSize{};
        std::vector<int64_t> shape;
        bool tape{};
    };

    static constexpr uint64_t kAlignment = 256;

    // Lays the entries out at aligned offsets, then allocates the block on the device of `host`, or on the host.
    ResidualBlock(std::vector<Entry> entries, const std::shared_ptr<HalDeviceState> &host)
        : entries_(std::move(entries)) {
        if (entries_.empty())
            return;
        uint64_t size = 0;
        for (Entry &entry : entries_) {
            if (entry.tape) {
                entry.offset = tapes_.size();
                tapes_.emplace_back(vernonScheduleTapeCreate());
                if (!tapes_.back())
                    throw std::bad_alloc();
                continue;
            }
            entry.offset = size;
            const uint64_t bytes = static_cast<uint64_t>(entry.elements) * entry.elementSize;
            size += (bytes + kAlignment - 1) / kAlignment * kAlignment;
        }
        size = std::max(size, kAlignment);
        if (host) {
            device_ = std::make_unique<HalBuffer>(host, static_cast<size_t>(size));
            base_ = static_cast<uint8_t *>(device_->address());
            if (!base_)
                throw std::runtime_error("cannot address a residual block: " + halError());
        } else {
            host_.resize(static_cast<size_t>(size + kAlignment));
            const auto address = reinterpret_cast<uintptr_t>(host_.data());
            base_ = host_.data() + ((kAlignment - address % kAlignment) % kAlignment);
        }
    }

    const std::vector<Entry> &entries() const { return entries_; }
    uint8_t *address(const Entry &entry) const { return base_ + entry.offset; }
    VernonScheduleTape *tape(const Entry &entry) const { return tapes_[entry.offset].get(); }

private:
    struct TapeDeleter {
        void operator()(VernonScheduleTape *tape) const { vernonScheduleTapeDestroy(tape); }
    };

    std::vector<Entry> entries_;
    std::vector<std::unique_ptr<VernonScheduleTape, TapeDeleter>> tapes_;
    std::unique_ptr<HalBuffer> device_;
    std::vector<uint8_t> host_;
    uint8_t *base_{};
};

// The device timeline of one Runtime. A linked call stores it with the point it signalled on each residency it writes,
// so a tensor can wait for its last write without the Runtime that made it.
class ScheduleTimeline {
public:
    explicit ScheduleTimeline(std::shared_ptr<RuntimeState> state) : state_(std::move(state)) {}

    void wait(uint64_t point) const {
        VernonStatus status = VERNON_STATUS_OK;
        {
            nb::gil_scoped_release released;
            status = vernonScheduleWait(state_->handle(), point);
        }
        if (status != VERNON_STATUS_OK)
            throw std::runtime_error(nativeStringView(vernonRuntimeGetLastError(state_->handle())));
    }

private:
    std::shared_ptr<RuntimeState> state_;
};

// Binds TensorStorage and TensorView arguments of a linked call and publishes the call's writes to them. On the CPU a
// buffer argument is an owner's host array, and the owner must have no residency. On a GPU it is the owner's residency
// in the session `residencyKey` names, which must hold the owner's current version with no host writes pending. A
// binding that finds an argument outside these returns false, so the call runs nothing.
class BufferBinder {
protected:
    // `references` binds a GPU buffer by the resource reference a Program argument takes rather than by its address.
    BufferBinder(std::shared_ptr<RuntimeState> state, nb::object storageType, nb::object viewType,
                 nb::object residencyKey, bool references)
        : state_(std::move(state)), storageType_(std::move(storageType)), viewType_(std::move(viewType)),
          residencyKey_(std::move(residencyKey)), references_(references) {
        nb::module_ numpy = nb::module_::import_("numpy");
        numpy_.integer = numpy.attr("integer");
        numpy_.floating = numpy.attr("floating");
        device_ = state_->device->backend != VERNON_RUNTIME_CPU;
    }

    // Whether a bound buffer's last binding still holds; see BoundBinding.
    bool stillBound(const BoundBinding &binding) const {
        const Names &name = names();
        if (binding.controlDict.is_valid() && binding.ownerDict.is_valid())
            return stillBoundByDict(binding);
        PyObject *control = binding.control.ptr();
        int64_t version = 0;
        if (!readIndex(attribute(control, name.version).ptr(), version) || version != binding.buffer.version ||
            PyObject_IsTrue(attribute(control, name.poisoned).ptr()) != 0 ||
            PyObject_Length(attribute(control, name.claims).ptr()) != 0 ||
            PyObject_Length(attribute(control, name.ioReservations).ptr()) != 0 ||
            attribute(binding.buffer.owner, name.array).ptr() != binding.array.ptr() ||
            (binding.buffer.writes && PyObject_IsTrue(attribute(control, name.preserved).ptr()) != 0))
            return false;
        if (!device_)
            return true;
        nb::object entries = attribute(attribute(control, name.residencies).ptr(), name.entries);
        if (!PyDict_CheckExact(entries.ptr()))
            return false;
        PyObject *residency = PyDict_GetItemWithError(entries.ptr(), residencyKey_.ptr());
        if (!residency)
            PyErr_Clear();
        return residency == binding.residency.ptr();
    }

    // `stillBound` over the cached instance dicts. The checks are the same; an entry a dict lacks fails the binding,
    // which then binds again through attributes.
    bool stillBoundByDict(const BoundBinding &binding) const {
        const Names &name = names();
        PyObject *control = binding.controlDict.ptr();
        PyObject *version = PyDict_GetItem(control, name.version);
        PyObject *claims = PyDict_GetItem(control, name.claims);
        PyObject *reservations = PyDict_GetItem(control, name.ioReservations);
        int64_t current = 0;
        if (!version || !readIndex(version, current) || current != binding.buffer.version ||
            PyDict_GetItem(control, name.poisoned) != Py_False || !claims || !PyList_CheckExact(claims) ||
            PyList_GET_SIZE(claims) != 0 || !reservations || !PyList_CheckExact(reservations) ||
            PyList_GET_SIZE(reservations) != 0 ||
            PyDict_GetItem(binding.ownerDict.ptr(), name.array) != binding.array.ptr())
            return false;
        if (binding.buffer.writes) {
            PyObject *preserved = PyDict_GetItem(control, name.preserved);
            if (preserved && PyObject_IsTrue(preserved) != 0)
                return false;
        }
        if (!device_)
            return true;
        if (PyDict_GetItem(control, name.residencies) != binding.residencies.ptr() || !binding.entries.is_valid())
            return false;
        return PyDict_GetItem(binding.entries.ptr(), residencyKey_.ptr()) == binding.residency.ptr();
    }

    static nb::object instanceDict(PyObject *object) {
        PyObject *dict = PyObject_GenericGetDict(object, nullptr);
        if (!dict) {
            PyErr_Clear();
            return {};
        }
        return nb::steal(dict);
    }

    bool bindBuffer(PyObject *value, const VernonScheduleSlot &slot, const std::vector<int64_t> &declared,
                    BufferViews &views, BufferArgument &buffer, BoundBinding *binding = nullptr) {
        if (!bindBufferNow(value, slot, declared, views, buffer))
            return false;
        if (binding) {
            const Names &name = names();
            binding->control = attribute(buffer.owner, name.control);
            binding->array = attribute(buffer.owner, name.array);
            binding->residency = device_ ? nb::borrow(buffer.residency) : nb::object();
            binding->controlDict = instanceDict(binding->control.ptr());
            binding->ownerDict = instanceDict(buffer.owner);
            binding->residencies = attribute(binding->control.ptr(), name.residencies);
            binding->entries =
                binding->residencies.is_valid() ? instanceDict(binding->residencies.ptr()) : nb::object();
            if (binding->entries.is_valid()) {
                PyObject *entries = PyDict_GetItem(binding->entries.ptr(), name.entries);
                binding->entries = entries && PyDict_CheckExact(entries) ? nb::borrow(entries) : nb::object();
            }
            binding->coherence = nb::object();
            binding->ranges = nb::object();
            if (device_ && buffer.writes) {
                binding->coherence = attribute(binding->control.ptr(), name.coherence);
                nb::object region = attribute(buffer.owner, name.region);
                nb::object ranges = region.is_valid() ? attribute(region.ptr(), name.ranges) : nb::object();
                if (ranges.is_valid())
                    binding->ranges = nb::steal(PySequence_List(ranges.ptr()));
                if (!binding->ranges.is_valid())
                    PyErr_Clear();
            }
            binding->buffer = buffer;
            binding->valid = true;
        }
        return true;
    }

    // A slot of kind zero takes a buffer of any element type and extents: a Program output that republishes an input's
    // Storage, which the input's slot checks.
    bool bindBufferNow(PyObject *value, const VernonScheduleSlot &slot, const std::vector<int64_t> &declared,
                       BufferViews &views, BufferArgument &buffer) {
        const bool typed = slot.kind != 0;
        const bool writes = slot.access & VERNON_SCHEDULE_ACCESS_WRITE;
        buffer.writes = writes;
        const Names &name = names();
        bool fullView = false;
        if (Py_TYPE(value) == reinterpret_cast<PyTypeObject *>(storageType_.ptr())) {
            buffer.owner = value;
            fullView = true;
        } else if (Py_TYPE(value) == reinterpret_cast<PyTypeObject *>(viewType_.ptr())) {
            nb::object owner = attribute(value, name.owner);
            if (Py_TYPE(owner.ptr()) != reinterpret_cast<PyTypeObject *>(storageType_.ptr()) ||
                !attribute(value, name.components).is_none() ||
                !accessSatisfies(attribute(value, name.access), slot.access))
                return false;
            // The view holds its owner, so the borrowed pointer outlives the call.
            buffer.owner = owner.ptr();
        } else {
            return false;
        }
        PyObject *owner = buffer.owner;
        nb::object control = attribute(owner, name.control);
        if (PyObject_IsTrue(attribute(control.ptr(), name.poisoned).ptr()) != 0 ||
            PyObject_Length(attribute(control.ptr(), name.claims).ptr()) != 0 ||
            PyObject_Length(attribute(control.ptr(), name.ioReservations).ptr()) != 0 ||
            (writes && PyObject_IsTrue(attribute(control.ptr(), name.preserved).ptr()) != 0))
            return false;
        nb::object version = attribute(control.ptr(), name.version);
        if (!readIndex(version.ptr(), buffer.version))
            return false;
        nb::object entries = attribute(attribute(control.ptr(), name.residencies).ptr(), name.entries);
        if (!PyDict_CheckExact(entries.ptr()))
            return false;
        if (!device_) {
            if (PyDict_GET_SIZE(entries.ptr()) != 0)
                return false;
        } else {
            PyObject *residency = PyDict_GetItemWithError(entries.ptr(), residencyKey_.ptr());
            if (!residency) {
                PyErr_Clear();
                return false;
            }
            buffer.residency = residency;
            if (!attribute(residency, name.version).equal(version) ||
                !hostClean(attribute(control.ptr(), name.coherence)))
                return false;
        }
        // The host array gives a GPU argument its element type and extents too, and the device copy its size.
        Py_buffer *view = views.acquire(attribute(owner, name.array).ptr(), writes && !device_);
        if (!view)
            return false;
        // A structured element is checked by its size only, as a Program invocation checks it.
        buffer.structured = structuredFormat(*view);
        if (typed && (static_cast<uint64_t>(view->itemsize) != slot.element_size ||
                      (!buffer.structured && !formatMatches(slot.dtype, *view))))
            return false;
        buffer.dtype = typed ? slot.dtype : formatDataType(*view);
        if (buffer.dtype == UINT32_MAX)
            return false;
        buffer.itemSize = view->itemsize;
        buffer.capacity = view->len / view->itemsize;
        if (device_) {
            buffer.device = deviceBuffer(attribute(buffer.residency, name.handle), static_cast<size_t>(view->len));
            if (!buffer.device)
                return false;
            if (references_) {
                buffer.data = nullptr;
                if (vernonHalReferenceBuffer(buffer.device->handle, 0, buffer.device->size, &buffer.reference) !=
                    VERNON_HAL_OK)
                    return false;
            } else if (!(buffer.data = buffer.device->address())) {
                return false;
            }
        } else {
            buffer.data = view->buf;
        }
        if (fullView) {
            buffer.offset = 0;
            buffer.shape.assign(view->shape, view->shape + view->ndim);
            buffer.strides.assign(static_cast<size_t>(view->ndim), 1);
            for (int dimension = view->ndim - 1; dimension > 0; --dimension)
                buffer.strides[dimension - 1] = buffer.strides[dimension] * buffer.shape[dimension];
        } else if (!readIndex(attribute(value, name.offset).ptr(), buffer.offset) ||
                   !readIndices(attribute(value, name.shape).ptr(), buffer.shape) ||
                   !readIndices(attribute(value, name.strides).ptr(), buffer.strides) ||
                   buffer.strides.size() != buffer.shape.size()) {
            return false;
        }
        if ((typed && buffer.shape.size() != declared.size()) || (writes && !injective(buffer.shape, buffer.strides)))
            return false;
        int64_t lowest = buffer.offset;
        int64_t highest = buffer.offset;
        bool empty = false;
        for (size_t dimension = 0; dimension < buffer.shape.size(); ++dimension) {
            const int64_t extent = buffer.shape[dimension];
            if (extent < 0 || (typed && declared[dimension] >= 0 && extent != declared[dimension]))
                return false;
            empty = empty || !extent;
            const int64_t span = extent ? (extent - 1) * buffer.strides[dimension] : 0;
            (span < 0 ? lowest : highest) += span;
        }
        buffer.spanBytes = 0;
        if (empty)
            return true;
        if (lowest < 0 || highest >= static_cast<int64_t>(view->len / view->itemsize))
            return false;
        buffer.spanBegin = static_cast<const uint8_t *>(view->buf) + lowest * view->itemsize;
        buffer.spanBytes = static_cast<size_t>(highest - lowest + 1) * static_cast<size_t>(view->itemsize);
        return true;
    }

    // Publishes a call's write to `buffer`: its owner's version advances, and on a GPU this residency becomes
    // authoritative over the whole owner as of `point`, which the host downloads before it next reads, as after a
    // Program's write; binding found the host's dirty ranges empty. A failed call poisons the owner instead.
    void publishWrite(const BufferArgument &buffer, BoundBinding *binding, bool failed, PyObject *point,
                      PyObject *timeline) {
        const Names &name = names();
        nb::object control = binding ? binding->control : attribute(buffer.owner, name.control);
        if (failed) {
            setAttribute(control.ptr(), name.poisoned, Py_True);
            return;
        }
        PyObject *controlDict = binding && binding->controlDict.is_valid() ? binding->controlDict.ptr() : nullptr;
        PyObject *current = controlDict ? PyDict_GetItem(controlDict, name.version) : nullptr;
        const int64_t advanced = (current ? nb::cast<int64_t>(nb::handle(current))
                                          : nb::cast<int64_t>(attribute(control.ptr(), name.version))) +
                                 1;
        nb::object version = nb::int_(advanced);
        if (!current || PyDict_SetItem(controlDict, name.version, version.ptr()) != 0)
            setAttribute(control.ptr(), name.version, version.ptr());
        // The call's own write keeps its binding.
        if (binding)
            binding->buffer.version = advanced;
        if (!device_) {
            setAttribute(control.ptr(), name.hostVersion, version.ptr());
            return;
        }
        PyObject *residency = buffer.residency;
        setAttribute(residency, name.version, version.ptr());
        setAttribute(residency, name.timeline, timeline);
        setAttribute(residency, name.writePoint, point);
        const bool cached = binding && binding->coherence.is_valid() && binding->ranges.is_valid();
        nb::object coherence = cached ? binding->coherence : attribute(control.ptr(), name.coherence);
        setAttribute(coherence.ptr(), name.deviceDirty, Py_True);
        nb::object ranges =
            cached
                ? nb::steal(PyList_GetSlice(binding->ranges.ptr(), 0, PY_SSIZE_T_MAX))
                : nb::steal(PySequence_List(attribute(attribute(buffer.owner, name.region).ptr(), name.ranges).ptr()));
        if (!ranges.is_valid())
            throw nb::python_error();
        setAttribute(coherence.ptr(), name.deviceDirtyRanges, ranges.ptr());
    }

    // Whether the host holds no writes the device has not seen.
    static bool hostClean(const nb::object &coherence) {
        nb::object dirty = attribute(coherence.ptr(), names().dirtyRanges);
        PyObject *native = PyObject_GetAttr(dirty.ptr(), names().native);
        if (!native) {
            PyErr_Clear();
            return PyObject_IsTrue(dirty.ptr()) == 0;
        }
        nb::object held = nb::steal(native);
        const vernon::runtime::DirtyRangeSet *ranges = nullptr;
        if (!nb::try_cast<const vernon::runtime::DirtyRangeSet *>(held, ranges) || !ranges)
            return PyObject_IsTrue(dirty.ptr()) == 0;
        return ranges->empty();
    }

    // A residency's buffer, which must belong to this Runtime's device and hold `size` bytes.
    HalBuffer *deviceBuffer(nb::handle handle, size_t size) const {
        HalBuffer *buffer = nullptr;
        if (!nb::try_cast<HalBuffer *>(handle, buffer) || !buffer || !buffer->host || buffer->size < size ||
            buffer->host != state_->device)
            return nullptr;
        return buffer;
    }

    static bool accessSatisfies(nb::handle access, uint32_t declared) {
        if (!PyUnicode_Check(access.ptr()))
            return false;
        uint32_t granted = 0;
        if (!PyUnicode_CompareWithASCIIString(access.ptr(), "read"))
            granted = VERNON_SCHEDULE_ACCESS_READ;
        else if (!PyUnicode_CompareWithASCIIString(access.ptr(), "write"))
            granted = VERNON_SCHEDULE_ACCESS_WRITE;
        else if (!PyUnicode_CompareWithASCIIString(access.ptr(), "read_write"))
            granted = VERNON_SCHEDULE_ACCESS_READ_WRITE;
        return granted && (granted & declared) == declared;
    }

    std::shared_ptr<RuntimeState> state_;
    nb::object storageType_;
    nb::object viewType_;
    nb::object residencyKey_;
    bool references_{};
    bool device_{};
    NumpyScalars numpy_;
};

// One linked kernel bound to one Runtime: `call` writes the schedule's record from Python arguments and submits it.
// A call returns false, without running anything, when an argument is outside what the record packs natively; the
// caller then runs the kernel as a Program, which reports the argument's error or handles its residencies. Buffer
// arguments bind as BufferBinder's; the Program path makes a GPU argument current, so a kernel's first call on fresh
// data falls back once.
class ScheduleKernel : BufferBinder {
public:
    // A CPU kernel links from its compiled `program`; a GPU kernel from its `reflection` JSON and `artifact`.
    ScheduleKernel(const Runtime &runtime, nb::object program, const std::string &entry, nb::object storageType,
                   nb::object viewType, nb::object reflection, nb::object artifact, nb::object residencyKey)
        : ScheduleKernel(runtime, std::move(storageType), std::move(viewType), std::move(residencyKey)) {
        KernelSource source;
        VernonScheduleKernelDescriptor descriptor{};
        describe(program, entry, reflection, artifact, source, descriptor);
        schedule_.reset(vernonScheduleLinkKernel(state_->handle(), &descriptor));
        adopt();
    }

    // One graph of a Program variant from its `program` JSON. Each of `stages` is a logical Stage's name and entry,
    // with its compiled program on the CPU, or its reflection JSON and artifact on a GPU; `zeroed` lists the Storages
    // that start as zeros and `exposed` the Storages that follow the boundary slots as slots.
    static ScheduleKernel linkProgram(const Runtime &runtime, const std::string &program, const std::string &graph,
                                      const nb::list &stages, const std::vector<uint32_t> &zeroed,
                                      const std::vector<uint32_t> &exposed, nb::object storageType, nb::object viewType,
                                      nb::object residencyKey) {
        ScheduleKernel linked(runtime, std::move(storageType), std::move(viewType), std::move(residencyKey));
        const size_t count = nb::len(stages);
        std::vector<std::string> names(count);
        std::vector<KernelSource> sources(count);
        std::vector<VernonScheduleStage> descriptors(count);
        nb::list programs;
        for (size_t index = 0; index < count; ++index) {
            nb::tuple stage = nb::cast<nb::tuple>(stages[index]);
            if (stage.size() != 5)
                throw std::invalid_argument("a Program Stage is a name, an entry, a program, a reflection and an "
                                            "artifact");
            names[index] = nb::cast<std::string>(stage[0]);
            programs.append(stage[2]);
            VernonScheduleStage &descriptor = descriptors[index];
            descriptor.name = {names[index].data(), names[index].size()};
            linked.describe(stage[2], nb::cast<std::string>(stage[1]), stage[3], stage[4], sources[index],
                            descriptor.kernel);
        }
        linked.program_ = programs;
        VernonScheduleProgramDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.program = {program.data(), program.size()};
        descriptor.graph = {graph.data(), graph.size()};
        descriptor.stages = descriptors.data();
        descriptor.stage_count = descriptors.size();
        descriptor.zeroed_storages = zeroed.data();
        descriptor.zeroed_storage_count = zeroed.size();
        descriptor.exposed_storages = exposed.data();
        descriptor.exposed_storage_count = exposed.size();
        linked.schedule_.reset(vernonScheduleLinkProgram(linked.state_->handle(), &descriptor));
        linked.adopt();
        return linked;
    }

    // Graph nodes, each a `(program, graph, stages, zeroed, exposed)` of link_program, in the order they run, and
    // connections, each `(sites, internal)` with `sites` a list of `(node, slot)`; see vernonScheduleLinkGraph.
    static ScheduleKernel linkGraph(const Runtime &runtime, const nb::list &nodes, const nb::list &connections,
                                    nb::object storageType, nb::object viewType, nb::object residencyKey) {
        ScheduleKernel linked(runtime, std::move(storageType), std::move(viewType), std::move(residencyKey));
        const size_t count = nb::len(nodes);
        std::vector<std::string> programs(count);
        std::vector<std::string> graphs(count);
        std::vector<std::vector<std::string>> names(count);
        std::vector<std::vector<KernelSource>> sources(count);
        std::vector<std::vector<VernonScheduleStage>> stages(count);
        std::vector<std::vector<uint32_t>> zeroed(count);
        std::vector<std::vector<uint32_t>> exposed(count);
        std::vector<VernonScheduleProgramDescriptor> descriptors(count);
        nb::list compiled;
        for (size_t node = 0; node < count; ++node) {
            nb::tuple row = nb::cast<nb::tuple>(nodes[node]);
            if (row.size() != 5)
                throw std::invalid_argument("a graph node is a program, a graph, its stages, its zeroed and its "
                                            "exposed Storages");
            programs[node] = nb::cast<std::string>(row[0]);
            graphs[node] = nb::cast<std::string>(row[1]);
            nb::list stageRows = nb::cast<nb::list>(row[2]);
            zeroed[node] = nb::cast<std::vector<uint32_t>>(row[3]);
            exposed[node] = nb::cast<std::vector<uint32_t>>(row[4]);
            const size_t stageCount = nb::len(stageRows);
            names[node].resize(stageCount);
            sources[node].resize(stageCount);
            stages[node].resize(stageCount);
            for (size_t index = 0; index < stageCount; ++index) {
                nb::tuple stage = nb::cast<nb::tuple>(stageRows[index]);
                if (stage.size() != 5)
                    throw std::invalid_argument("a Program Stage is a name, an entry, a program, a reflection and an "
                                                "artifact");
                names[node][index] = nb::cast<std::string>(stage[0]);
                compiled.append(stage[2]);
                VernonScheduleStage &descriptor = stages[node][index];
                descriptor.name = {names[node][index].data(), names[node][index].size()};
                linked.describe(stage[2], nb::cast<std::string>(stage[1]), stage[3], stage[4], sources[node][index],
                                descriptor.kernel);
            }
            VernonScheduleProgramDescriptor &descriptor = descriptors[node];
            descriptor.struct_size = sizeof(descriptor);
            descriptor.program = {programs[node].data(), programs[node].size()};
            descriptor.graph = {graphs[node].data(), graphs[node].size()};
            descriptor.stages = stages[node].data();
            descriptor.stage_count = stages[node].size();
            descriptor.zeroed_storages = zeroed[node].data();
            descriptor.zeroed_storage_count = zeroed[node].size();
            descriptor.exposed_storages = exposed[node].data();
            descriptor.exposed_storage_count = exposed[node].size();
        }
        linked.program_ = compiled;
        const size_t connectionCount = nb::len(connections);
        std::vector<std::vector<VernonScheduleNodeSlot>> sites(connectionCount);
        std::vector<VernonScheduleConnection> rows(connectionCount);
        for (size_t index = 0; index < connectionCount; ++index) {
            nb::tuple row = nb::cast<nb::tuple>(connections[index]);
            if (row.size() != 2)
                throw std::invalid_argument("a graph connection is its sites and whether it is internal");
            for (nb::handle site : nb::cast<nb::list>(row[0])) {
                nb::tuple pair = nb::cast<nb::tuple>(site);
                sites[index].push_back({nb::cast<uint32_t>(pair[0]), nb::cast<uint32_t>(pair[1])});
            }
            rows[index] = {sites[index].data(), sites[index].size(), nb::cast<bool>(row[1]) ? 1u : 0u};
        }
        VernonScheduleGraphDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.nodes = descriptors.data();
        descriptor.node_count = descriptors.size();
        descriptor.connections = rows.data();
        descriptor.connection_count = rows.size();
        linked.schedule_.reset(vernonScheduleLinkGraph(linked.state_->handle(), &descriptor));
        linked.adopt();
        linked.nodeSlots_.resize(count);
        for (size_t node = 0; node < count; ++node)
            for (size_t slot = 0; slot < vernonScheduleNodeSlotCount(linked.schedule_.get(), node); ++slot)
                linked.nodeSlots_[node].push_back(vernonScheduleNodeSlot(linked.schedule_.get(), node, slot));
        return linked;
    }

    // Per graph node, the schedule slot of each of its slots, None for an internal connection.
    nb::list nodeSlots() const {
        nb::list rows;
        for (const std::vector<uint32_t> &node : nodeSlots_) {
            nb::list row;
            for (uint32_t slot : node)
                row.append(slot == UINT32_MAX ? nb::none() : nb::int_(slot));
            rows.append(row);
        }
        return rows;
    }

    // Per TensorView slot, its extents as (value, slot, dimension): `value` when `slot` is None, else extent
    // `dimension` of slot `slot`, which is the slot itself for a free extent.
    nb::list slotExtents() const {
        nb::list extents;
        for (size_t index = 0; index < slots_.size(); ++index) {
            nb::list row;
            for (const VernonScheduleExtent &extent : extents_[index])
                row.append(extent.slot == UINT32_MAX ? nb::make_tuple(extent.value, nb::none(), 0)
                                                     : nb::make_tuple(0, extent.slot, extent.dimension));
            extents.append(row);
        }
        return extents;
    }

private:
    // The strings and bytes a kernel descriptor points into, which must outlive the link call.
    struct KernelSource {
        std::string entry;
        std::string reflection;
        nb::bytes ptx;
    };

    ScheduleKernel(const Runtime &runtime, nb::object storageType, nb::object viewType, nb::object residencyKey)
        : BufferBinder(runtime.state, std::move(storageType), std::move(viewType), std::move(residencyKey), false) {}

    void describe(nb::handle program, const std::string &entry, nb::handle reflection, nb::handle artifact,
                  KernelSource &source, VernonScheduleKernelDescriptor &descriptor) {
        descriptor.struct_size = sizeof(descriptor);
        source.entry = entry;
        descriptor.entry = {source.entry.data(), source.entry.size()};
        if (device_) {
            if (!nb::try_cast<std::string>(reflection, source.reflection) ||
                !nb::try_cast<nb::bytes>(artifact, source.ptx) || residencyKey_.is_none())
                throw std::invalid_argument("a GPU kernel links from its reflection, artifact and session");
            descriptor.cuda_artifact = source.ptx.c_str();
            descriptor.cuda_artifact_size = source.ptx.size();
        } else {
            program_ = nb::borrow(program);
            const CompiledProgram &compiled = nb::cast<const CompiledProgram &>(program);
            compiled.requireSuccess();
            source.reflection = compiled.reflection();
            descriptor.cpu_entry = vernonCompileResultGetCpuEntry(compiled.result.get(), entry.data(), entry.size());
            if (!descriptor.cpu_entry)
                throw std::runtime_error("CPU entry '" + entry + "' was not found");
            descriptor.cpu_chunk_entry =
                vernonCompileResultGetCpuChunkEntry(compiled.result.get(), entry.data(), entry.size());
        }
        descriptor.reflection = {source.reflection.data(), source.reflection.size()};
    }

    // Takes the slots, fields and record layout of the schedule just linked.
    void adopt() {
        if (!schedule_)
            throw std::invalid_argument(nativeStringView(vernonRuntimeGetLastError(state_->handle())));
        const VernonScheduleSlot *slots = vernonScheduleSlots(schedule_.get());
        slots_.assign(slots, slots + vernonScheduleSlotCount(schedule_.get()));
        for (size_t index = 0; index < slots_.size(); ++index) {
            const int64_t *shape = vernonScheduleSlotShape(schedule_.get(), index);
            shapes_.emplace_back(shape, shape + slots_[index].rank);
            const VernonScheduleExtent *extents = vernonScheduleSlotExtents(schedule_.get(), index);
            extents_.emplace_back(extents, extents + slots_[index].rank);
        }
        const VernonScheduleField *fields = vernonScheduleFields(schedule_.get());
        fields_.assign(fields, fields + vernonScheduleFieldCount(schedule_.get()));
        const uint8_t *initial = vernonScheduleRecordTemplate(schedule_.get());
        record_.assign(initial, initial + vernonScheduleRecordSize(schedule_.get()));
        gridOffset_ = vernonScheduleGridOffset(schedule_.get());
        blockIndex_.assign(slots_.size(), -1);
        if (device_)
            timeline_ = nb::cast(ScheduleTimeline(state_));
    }

public:
    // The kinds of the launch arguments in order: "buffer", "scalar" or "tape".
    nb::list slotKinds() const {
        nb::list kinds;
        for (const VernonScheduleSlot &slot : slots_)
            kinds.append(slot.kind == VERNON_SCHEDULE_SLOT_BUFFER ? "buffer"
                         : slot.kind == VERNON_SCHEDULE_SLOT_TAPE ? "tape"
                                                                  : "scalar");
        return kinds;
    }

    // Makes `slots`, in residual order, the slots a call binds from a ResidualBlock rather than its arguments, where it
    // passes None. A `producing` schedule sizes each residual from its extents, which must be static or another
    // argument's.
    void setBlockSlots(const std::vector<uint32_t> &slots, bool producing) {
        std::vector<int32_t> blockIndex(slots_.size(), -1);
        for (size_t index = 0; index < slots.size(); ++index) {
            const uint32_t slot = slots[index];
            if (slot >= slots_.size() || blockIndex[slot] >= 0 ||
                (slots_[slot].kind != VERNON_SCHEDULE_SLOT_BUFFER && slots_[slot].kind != VERNON_SCHEDULE_SLOT_TAPE))
                throw std::invalid_argument("a residual block slot must be a distinct buffer or tape slot");
            blockIndex[slot] = static_cast<int32_t>(index);
        }
        for (uint32_t slot : slots) {
            for (const VernonScheduleExtent &extent : extents_[slot]) {
                if (!producing || extent.slot == UINT32_MAX)
                    continue;
                if (extent.slot >= slots_.size() || slots_[extent.slot].kind != VERNON_SCHEDULE_SLOT_BUFFER ||
                    blockIndex[extent.slot] >= 0 || extent.dimension >= slots_[extent.slot].rank)
                    throw std::invalid_argument("a residual's extents must be static or another argument's");
            }
        }
        blockIndex_ = std::move(blockIndex);
        blockSlots_ = slots;
    }

    // `grid` is the three workgroup counts of a kernel, and None for a Program, whose grids are fixed.
    bool call(nb::handle arguments, nb::handle grid, const BoundSlots *bound = nullptr) {
        return run(arguments, grid, nullptr, nullptr, bound);
    }

    // A forward call that writes its block slots into a new ResidualBlock and returns it; None when an argument is
    // outside what the record packs.
    nb::object callProducing(nb::handle arguments, const BoundSlots *bound = nullptr) {
        std::unique_ptr<ResidualBlock> block;
        if (!run(arguments, nb::none(), &block, nullptr, bound))
            return nb::none();
        return nb::cast(block.release(), nb::rv_policy::take_ownership);
    }

    // A backward call that reads its block slots from the block of the forward call it differentiates.
    bool callConsuming(nb::handle arguments, const ResidualBlock &block, const BoundSlots *bound = nullptr) {
        return run(arguments, nb::none(), nullptr, &block, bound);
    }

    // The slots of a bound view of this schedule: `slots` hold `values`, in order, and a call passes the others.
    BoundSlots boundSlots(const std::vector<uint32_t> &slots, nb::tuple values) const {
        if (slots.size() != values.size())
            throw std::invalid_argument("a bound view binds one value per slot");
        BoundSlots bound;
        bound.values.resize(slots_.size());
        for (size_t index = 0; index < slots.size(); ++index) {
            const uint32_t slot = slots[index];
            if (slot >= slots_.size() || bound.values[slot].is_valid() || blockIndex_[slot] >= 0 ||
                values[index].is_none())
                throw std::invalid_argument("a bound slot must be a distinct argument slot with a value");
            bound.values[slot] = nb::borrow(values[index]);
        }
        // The residual block supplies its slots, which every call passes None, so a bound view passes them itself.
        for (uint32_t slot = 0; slot < slots_.size(); ++slot)
            if (blockIndex_[slot] >= 0)
                bound.values[slot] = nb::none();
            else if (!bound.values[slot].is_valid())
                bound.unbound.push_back(slot);
        bound.bindings.resize(slots_.size());
        return bound;
    }

private:
    bool run(nb::handle argumentsHandle, nb::handle gridHandle, std::unique_ptr<ResidualBlock> *produce,
             const ResidualBlock *consume, const BoundSlots *bound) {
        PyObject *arguments = argumentsHandle.ptr();
        PyObject *grid = gridHandle.ptr();
        const bool callerGrid = gridOffset_ != UINT64_MAX;
        const size_t passed = bound ? bound->unbound.size() : slots_.size();
        if (!PyTuple_CheckExact(arguments) || static_cast<size_t>(PyTuple_GET_SIZE(arguments)) != passed ||
            (callerGrid ? !PyTuple_Check(grid) || PyTuple_GET_SIZE(grid) != 3 : grid != Py_None))
            return false;
        if (bound) {
            values_.assign(slots_.size(), nullptr);
            for (size_t slot = 0; slot < slots_.size(); ++slot)
                values_[slot] = bound->values[slot].ptr();
            for (size_t index = 0; index < passed; ++index)
                values_[bound->unbound[index]] = PyTuple_GET_ITEM(arguments, static_cast<Py_ssize_t>(index));
        }
        uint32_t workgroups[3]{};
        for (Py_ssize_t axis = 0; callerGrid && axis < 3; ++axis) {
            int64_t count = 0;
            if (!readIndex(PyTuple_GET_ITEM(grid, axis), count) || count <= 0 ||
                count > std::numeric_limits<uint32_t>::max())
                return false;
            workgroups[axis] = static_cast<uint32_t>(count);
        }
        BufferViews views;
        views.reserve(slots_.size());
        buffers_.resize(slots_.size());
        uint8_t *record = record_.data();
        for (size_t index = 0; index < slots_.size(); ++index) {
            const VernonScheduleSlot &slot = slots_[index];
            PyObject *value = bound ? values_[index] : PyTuple_GET_ITEM(arguments, static_cast<Py_ssize_t>(index));
            if (slot.kind == VERNON_SCHEDULE_SLOT_SCALAR) {
                if (!packScalar(value, slot, numpy_, record + slot.offset))
                    return false;
                continue;
            }
            BufferArgument &buffer = buffers_[index];
            if (slot.kind == VERNON_SCHEDULE_SLOT_TAPE && blockIndex_[index] < 0)
                return false;
            if (blockIndex_[index] >= 0) {
                if (value != Py_None)
                    return false;
                buffer.owner = nullptr;
                buffer.writes = false;
                buffer.spanBytes = 0;
                continue;
            }
            BoundBinding *binding = bound && bound->values[index].is_valid() ? &bound->bindings[index] : nullptr;
            if (binding && binding->valid && stillBound(*binding)) {
                buffer = binding->buffer;
            } else if (!bindBuffer(value, slot, shapes_[index], views, buffer, binding)) {
                if (binding)
                    binding->valid = false;
                return false;
            }
            std::memcpy(record + slot.offset, &buffer.data, sizeof(void *));
        }
        if (!blockSlots_.empty()) {
            if (produce && !(*produce = makeBlock()))
                return false;
            if (!bindBlock(produce ? produce->get() : consume, record))
                return false;
        } else if (produce) {
            // A forward call without residuals still returns a block, which its pullback consumes.
            *produce = std::make_unique<ResidualBlock>(std::vector<ResidualBlock::Entry>{}, nullptr);
        }
        if (!disjointWrites())
            return false;
        for (const VernonScheduleField &field : fields_) {
            if (field.kind == VERNON_SCHEDULE_FIELD_VALUE) {
                std::memcpy(record + field.offset, record + slots_[field.slot].offset, field.size);
                continue;
            }
            const BufferArgument &buffer = buffers_[field.slot];
            int64_t value = buffer.offset;
            if (field.kind == VERNON_SCHEDULE_FIELD_EXTENT)
                value = buffer.shape[field.dimension];
            else if (field.kind == VERNON_SCHEDULE_FIELD_STRIDE)
                value = buffer.strides[field.dimension];
            else if (field.kind == VERNON_SCHEDULE_FIELD_CAPACITY)
                value = buffer.capacity;
            else if (field.kind == VERNON_SCHEDULE_FIELD_BYTES)
                value = buffer.capacity * static_cast<int64_t>(slots_[field.slot].element_size);
            else if (field.kind == VERNON_SCHEDULE_FIELD_ADDRESS)
                value = static_cast<int64_t>(reinterpret_cast<uintptr_t>(buffer.data));
            if (field.size == sizeof(int32_t)) {
                if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max())
                    return false;
                const auto narrowed = static_cast<int32_t>(value);
                std::memcpy(record + field.offset, &narrowed, sizeof(narrowed));
            } else {
                std::memcpy(record + field.offset, &value, sizeof(value));
            }
        }
        if (callerGrid)
            std::memcpy(record + gridOffset_, workgroups, sizeof(workgroups));

        uint64_t point = 0;
        const VernonStatus status = vernonScheduleCall(schedule_.get(), record, record_.size(), &point);
        nb::object pointObject = nb::int_(point);
        for (size_t index = 0; index < slots_.size(); ++index) {
            if (slots_[index].kind != VERNON_SCHEDULE_SLOT_BUFFER || !buffers_[index].writes)
                continue;
            BoundBinding *binding = bound && bound->bindings[index].valid ? &bound->bindings[index] : nullptr;
            publishWrite(buffers_[index], binding, status != VERNON_STATUS_OK, pointObject.ptr(), timeline_.ptr());
        }
        if (status != VERNON_STATUS_OK)
            throw std::runtime_error(nativeStringView(vernonRuntimeGetLastError(state_->handle())));
        return true;
    }

private:
    // The Program path rejects a written view that overlaps another argument, and reports why, so a call falls back
    // when a written owner is bound twice. CPU entries mark their buffers `noalias`, so on the CPU distinct owners
    // whose host spans overlap a written one fall back too. Both checks sort, so a call with hundreds of buffers
    // stays O(n log n).
    bool disjointWrites() {
        owners_.clear();
        spans_.clear();
        for (size_t index = 0; index < slots_.size(); ++index) {
            if (slots_[index].kind != VERNON_SCHEDULE_SLOT_BUFFER || blockIndex_[index] >= 0)
                continue;
            const BufferArgument &buffer = buffers_[index];
            owners_.push_back({reinterpret_cast<uintptr_t>(buffer.owner), buffer.writes});
            if (!device_ && buffer.spanBytes)
                spans_.push_back({reinterpret_cast<uintptr_t>(buffer.spanBegin),
                                  reinterpret_cast<uintptr_t>(buffer.spanBegin) + buffer.spanBytes, buffer.writes});
        }
        std::sort(owners_.begin(), owners_.end());
        for (size_t index = 1; index < owners_.size(); ++index)
            if (owners_[index].first == owners_[index - 1].first &&
                (owners_[index].second || owners_[index - 1].second))
                return false;
        std::sort(spans_.begin(), spans_.end(),
                  [](const Span &left, const Span &right) { return left.begin < right.begin; });
        // Sorted by start, a span overlaps an earlier one exactly when it starts before the furthest end so far.
        uintptr_t furthest = 0;
        uintptr_t furthestWritten = 0;
        for (const Span &span : spans_) {
            if (span.begin < (span.writes ? furthest : furthestWritten))
                return false;
            furthest = std::max(furthest, span.end);
            if (span.writes)
                furthestWritten = std::max(furthestWritten, span.end);
        }
        return true;
    }

    // The block of a producing call, each residual sized from the arguments already bound; null when an extent
    // contradicts the residual's static one.
    std::unique_ptr<ResidualBlock> makeBlock() const {
        std::vector<ResidualBlock::Entry> entries(blockSlots_.size());
        for (size_t index = 0; index < blockSlots_.size(); ++index) {
            const uint32_t slot = blockSlots_[index];
            ResidualBlock::Entry &entry = entries[index];
            if (slots_[slot].kind == VERNON_SCHEDULE_SLOT_TAPE) {
                entry.tape = true;
                continue;
            }
            entry.dtype = slots_[slot].dtype;
            entry.elementSize = slots_[slot].element_size;
            entry.elements = 1;
            for (size_t dimension = 0; dimension < extents_[slot].size(); ++dimension) {
                const VernonScheduleExtent &extent = extents_[slot][dimension];
                const int64_t value =
                    extent.slot == UINT32_MAX ? extent.value : buffers_[extent.slot].shape[extent.dimension];
                if (value < 0 || (shapes_[slot][dimension] >= 0 && value != shapes_[slot][dimension]) ||
                    (value && entry.elements > std::numeric_limits<int64_t>::max() / value))
                    return nullptr;
                entry.shape.push_back(value);
                entry.elements *= value;
            }
        }
        return std::make_unique<ResidualBlock>(std::move(entries), device_ ? state_->device : nullptr);
    }

    // Binds each block slot to its residual, which must have the slot's element type and declared extents.
    bool bindBlock(const ResidualBlock *block, uint8_t *record) {
        if (!block || block->entries().size() != blockSlots_.size())
            return false;
        for (size_t index = 0; index < blockSlots_.size(); ++index) {
            const uint32_t slot = blockSlots_[index];
            const ResidualBlock::Entry &entry = block->entries()[index];
            if (entry.tape != (slots_[slot].kind == VERNON_SCHEDULE_SLOT_TAPE))
                return false;
            if (entry.tape) {
                VernonScheduleTape *tape = block->tape(entry);
                std::memcpy(record + slots_[slot].offset, &tape, sizeof(tape));
                continue;
            }
            const std::vector<int64_t> &declared = shapes_[slot];
            if (entry.dtype != slots_[slot].dtype || entry.elementSize != slots_[slot].element_size ||
                entry.shape.size() != declared.size())
                return false;
            for (size_t dimension = 0; dimension < declared.size(); ++dimension)
                if (declared[dimension] >= 0 && entry.shape[dimension] != declared[dimension])
                    return false;
            BufferArgument &buffer = buffers_[slot];
            buffer.data = block->address(entry);
            buffer.capacity = entry.elements;
            buffer.offset = 0;
            buffer.shape = entry.shape;
            buffer.strides.assign(entry.shape.size(), 1);
            for (size_t dimension = entry.shape.size(); dimension > 1; --dimension)
                buffer.strides[dimension - 2] = buffer.strides[dimension - 1] * entry.shape[dimension - 1];
            std::memcpy(record + slots_[slot].offset, &buffer.data, sizeof(void *));
        }
        return true;
    }

    nb::object program_;
    std::unique_ptr<VernonLinkedSchedule, ScheduleDeleter> schedule_;
    std::vector<VernonScheduleSlot> slots_;
    std::vector<std::vector<int64_t>> shapes_;
    std::vector<std::vector<VernonScheduleExtent>> extents_;
    std::vector<VernonScheduleField> fields_;
    std::vector<BufferArgument> buffers_;
    // Per slot, the value a bound call binds it to.
    std::vector<PyObject *> values_;
    // The block slots in residual order, and per slot its residual or -1.
    std::vector<uint32_t> blockSlots_;
    std::vector<int32_t> blockIndex_;
    std::vector<std::vector<uint32_t>> nodeSlots_;
    struct Span {
        uintptr_t begin;
        uintptr_t end;
        bool writes;
    };
    std::vector<std::pair<uintptr_t, bool>> owners_;
    std::vector<Span> spans_;
    std::vector<uint8_t> record_;
    uint64_t gridOffset_{};
    nb::object timeline_;
};

// A linked schedule with some slots bound once (language contract section 8, `step.bind`): a call passes only the
// other slots' arguments, in slot order, and binds the bound values as if they were passed.
class BoundSchedule {
public:
    BoundSchedule(nb::object kernel, const std::vector<uint32_t> &slots, nb::tuple values)
        : kernel_(std::move(kernel)), schedule_(nb::cast<ScheduleKernel *>(kernel_)),
          bound_(schedule_->boundSlots(slots, std::move(values))) {}

    bool call(nb::handle arguments, nb::handle grid) { return schedule_->call(arguments, grid, &bound_); }
    nb::object callProducing(nb::handle arguments) { return schedule_->callProducing(arguments, &bound_); }
    bool callConsuming(nb::handle arguments, const ResidualBlock &block) {
        return schedule_->callConsuming(arguments, block, &bound_);
    }

private:
    nb::object kernel_;
    ScheduleKernel *schedule_;
    BoundSlots bound_;
};

// Where the caller record of a schedule linked from a Program executable takes each boundary slot's argument.
struct ArgumentOffsets {
    // Per boundary slot, the offset of its ARGUMENT slot, or UINT64_MAX.
    std::vector<uint64_t> offsets;
    size_t count{};
};

// Writes each argument the builder prepared at its slot's offset. A prepared argument of no slot is an output that
// republishes an input's Storage, which the input's slot already binds.
void packArguments(const ProgramInvocationBuilder &arguments, const ArgumentOffsets &layout,
                   std::vector<uint8_t> &record) {
    std::vector<bool> bound(layout.offsets.size());
    size_t written = 0;
    for (const PreparedProgramArgument *argument : arguments.arguments) {
        const uint32_t slot = argument->value.slot;
        if (slot >= layout.offsets.size() || layout.offsets[slot] == UINT64_MAX)
            continue;
        if (bound[slot])
            throw std::invalid_argument("the Program schedule call binds a slot twice");
        bound[slot] = true;
        ++written;
        std::memcpy(record.data() + layout.offsets[slot], &argument->value, sizeof(argument->value));
    }
    if (written != layout.count)
        throw std::invalid_argument("the Program schedule call leaves an argument unbound");
}

[[noreturn]] void throwScheduleError(VernonStatus status, VernonRuntimeContext *runtime) {
    const std::string error = nativeStringView(vernonRuntimeGetLastError(runtime));
    if (status == VERNON_STATUS_INVALID_ARGUMENT)
        throw std::invalid_argument(error);
    throw std::runtime_error(error);
}

// The forward graph of a Program executable linked as one schedule (VernonScheduleProgramDescriptor::executable). A
// call packs the arguments a builder prepared into the ARGUMENT slots, and a control builder per graphics node, in
// graph order, into that node's PASS, DRAW and DYNAMIC slots.
class ProgramSchedule {
public:
    explicit ProgramSchedule(nb::object executable)
        : executableObject_(std::move(executable)),
          executable_(nb::cast<PythonProgramExecutable *>(executableObject_)) {
        VernonScheduleProgramDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.graph = {"forward", 7};
        descriptor.executable = executable_->executable;
        schedule_.reset(vernonScheduleLinkProgram(executable_->runtime, &descriptor));
        if (!schedule_)
            throw std::invalid_argument(nativeStringView(vernonRuntimeGetLastError(executable_->runtime)));
        const VernonScheduleSlot *slots = vernonScheduleSlots(schedule_.get());
        const size_t count = vernonScheduleSlotCount(schedule_.get());
        size_t at = 0;
        for (; at < count && slots[at].kind == VERNON_SCHEDULE_SLOT_ARGUMENT; ++at) {
            const uint32_t boundary = vernonScheduleSlotBoundary(schedule_.get(), at);
            if (boundary == UINT32_MAX)
                throw std::invalid_argument("the schedule has an argument of no Program boundary slot");
            if (arguments_.offsets.size() <= boundary)
                arguments_.offsets.resize(boundary + 1, UINT64_MAX);
            arguments_.offsets[boundary] = slots[at].offset;
        }
        arguments_.count = at;
        constexpr VernonScheduleSlotKind passKinds[]{VERNON_SCHEDULE_SLOT_PASS, VERNON_SCHEDULE_SLOT_DRAW,
                                                     VERNON_SCHEDULE_SLOT_DYNAMIC, VERNON_SCHEDULE_SLOT_STREAM,
                                                     VERNON_SCHEDULE_SLOT_TABLE};
        while (at < count) {
            if (count - at < std::size(passKinds))
                throw std::invalid_argument("the schedule's graphics slots are incomplete");
            uint64_t offsets[std::size(passKinds)]{};
            for (size_t kind = 0; kind < std::size(passKinds); ++kind, ++at) {
                if (slots[at].kind != static_cast<uint32_t>(passKinds[kind]))
                    throw std::invalid_argument("the schedule's graphics slots are out of order");
                offsets[kind] = slots[at].offset;
            }
            passes_.push_back({offsets[0], offsets[1], offsets[2]});
        }
        const uint8_t *bytes = vernonScheduleRecordTemplate(schedule_.get());
        template_.assign(bytes, bytes + vernonScheduleRecordSize(schedule_.get()));
        timeline_ = nb::cast(ScheduleTimeline(executable_->owner));
    }

    std::unique_ptr<ProgramInvocationBuilder> builder() const {
        return std::make_unique<ProgramInvocationBuilder>(executable_->owner, executable_->runtime,
                                                          executable_->executable);
    }

    // The point on the device timeline that the call's submission signals.
    uint64_t call(const ProgramInvocationBuilder &arguments, const std::vector<ProgramInvocationBuilder *> &passes) {
        if (passes.size() != passes_.size())
            throw std::invalid_argument("the Program schedule takes a control builder per graphics node");
        record_ = template_;
        packArguments(arguments, arguments_, record_);
        for (size_t index = 0; index < passes_.size(); ++index) {
            const ProgramInvocationBuilder *control = passes[index];
            if (!control || !control->hasGraphicsState)
                throw std::invalid_argument("Program graphics node control builder has no typed graphics state");
            control->refreshGraphicsControls();
            const PassSlots &pass = passes_[index];
            const VernonRenderPass *renderPass = &control->renderPass;
            std::memcpy(record_.data() + pass.renderPass, &renderPass, sizeof(renderPass));
            // No vertex count and no index binding draw the vertices the Program's graphics operation declares.
            VernonDrawCommand draw = control->drawCommand;
            if (!control->vertexCount && !control->hasIndex) {
                VernonDrawCommand declared{};
                std::memcpy(&declared, template_.data() + pass.draw, sizeof(declared));
                draw.vertex_count = declared.vertex_count;
            }
            std::memcpy(record_.data() + pass.draw, &draw, sizeof(draw));
            std::memcpy(record_.data() + pass.dynamic, &control->dynamicState, sizeof(control->dynamicState));
        }
        uint64_t point = 0;
        VernonStatus status = VERNON_STATUS_OK;
        {
            nb::gil_scoped_release released;
            status = vernonScheduleCall(schedule_.get(), record_.data(), record_.size(), &point);
        }
        if (status != VERNON_STATUS_OK)
            throwScheduleError(status, executable_->runtime);
        return point;
    }

    size_t passCount() const { return passes_.size(); }
    const nb::object &timeline() const { return timeline_; }

private:
    struct PassSlots {
        uint64_t renderPass;
        uint64_t draw;
        uint64_t dynamic;
    };

    nb::object executableObject_;
    PythonProgramExecutable *executable_;
    std::unique_ptr<VernonLinkedSchedule, ScheduleDeleter> schedule_;
    ArgumentOffsets arguments_;
    std::vector<PassSlots> passes_;
    std::vector<uint8_t> template_;
    std::vector<uint8_t> record_;
    nb::object timeline_;
};

// The caller record of a schedule linked from a Program executable, packed from Python values: per ARGUMENT slot, a
// TensorStorage or TensorView for a TensorView argument or a number for a scalar, written as the VernonProgramArgument
// the slot takes. Buffers bind as BufferBinder's, so a value outside what packs natively fails the pack before anything
// runs, and the caller then prepares the arguments with a ProgramInvocationBuilder, which reports the value's error or
// makes it current.
class ArgumentRecord : BufferBinder {
public:
    ArgumentRecord(std::shared_ptr<RuntimeState> state, VernonProgramExecutable *executable,
                   const VernonLinkedSchedule *schedule, nb::object storageType, nb::object viewType,
                   nb::object residencyKey)
        : BufferBinder(std::move(state), std::move(storageType), std::move(viewType), std::move(residencyKey), true) {
        const size_t count = vernonScheduleSlotCount(schedule);
        const VernonScheduleSlot *outer = vernonScheduleSlots(schedule);
        arguments_.resize(count);
        buffers_.resize(count);
        for (size_t index = 0; index < count; ++index) {
            Argument &argument = arguments_[index];
            const VernonScheduleSlot *inner = vernonScheduleArgumentSlot(schedule, index);
            if (outer[index].kind != VERNON_SCHEDULE_SLOT_ARGUMENT || !inner)
                throw std::invalid_argument("the schedule takes an argument that is no Program argument");
            argument.slot = *inner;
            argument.offset = outer[index].offset;
            argument.boundary = vernonScheduleSlotBoundary(schedule, index);
            if (inner->kind == VERNON_SCHEDULE_SLOT_BUFFER) {
                const int64_t *shape = vernonScheduleSlotShape(schedule, index);
                argument.declared.assign(shape, shape + inner->rank);
                argument.parameter = slotParameterMetadata(executable, argument.boundary);
            }
        }
    }

    size_t size() const { return arguments_.size(); }

    // The slots of a bound view of this record: `slots` hold `values`, in order, and a call passes the others.
    BoundSlots boundSlots(const std::vector<uint32_t> &slots, const nb::tuple &values) const {
        if (slots.size() != values.size())
            throw std::invalid_argument("a bound view binds one value per slot");
        BoundSlots bound;
        bound.values.resize(arguments_.size());
        for (size_t index = 0; index < slots.size(); ++index) {
            const uint32_t slot = slots[index];
            if (slot >= arguments_.size() || bound.values[slot].is_valid() || values[index].is_none())
                throw std::invalid_argument("a bound slot must be a distinct argument slot with a value");
            bound.values[slot] = nb::borrow(values[index]);
        }
        for (uint32_t slot = 0; slot < arguments_.size(); ++slot)
            if (!bound.values[slot].is_valid())
                bound.unbound.push_back(slot);
        bound.bindings.resize(arguments_.size());
        return bound;
    }

    // Writes `arguments`, the values of the slots `bound` leaves unbound in order, into `record`, a copy of the
    // schedule's record template. The record points into this object and `views` until the call completes.
    bool pack(PyObject *arguments, const BoundSlots *bound, uint8_t *record, BufferViews &views) {
        const size_t passed = bound ? bound->unbound.size() : arguments_.size();
        if (!PyTuple_CheckExact(arguments) || static_cast<size_t>(PyTuple_GET_SIZE(arguments)) != passed)
            return false;
        if (bound) {
            values_.assign(arguments_.size(), nullptr);
            for (size_t slot = 0; slot < arguments_.size(); ++slot)
                values_[slot] = bound->values[slot].ptr();
            for (size_t index = 0; index < passed; ++index)
                values_[bound->unbound[index]] = PyTuple_GET_ITEM(arguments, static_cast<Py_ssize_t>(index));
        }
        views.reserve(arguments_.size());
        for (size_t index = 0; index < arguments_.size(); ++index) {
            Argument &argument = arguments_[index];
            BufferArgument &buffer = buffers_[index];
            PyObject *value = bound ? values_[index] : PyTuple_GET_ITEM(arguments, static_cast<Py_ssize_t>(index));
            VernonProgramArgument written{};
            written.slot = argument.boundary;
            written.kind = VERNON_PROGRAM_TENSOR;
            VernonTensorView &view = written.tensor;
            view.struct_size = sizeof(view);
            if (argument.slot.kind == VERNON_SCHEDULE_SLOT_SCALAR) {
                buffer.owner = nullptr;
                buffer.writes = false;
                if (argument.slot.size > sizeof(argument.scalar) ||
                    !packScalar(value, argument.slot, numpy_, argument.scalar))
                    return false;
                view.storage = VERNON_TENSOR_HOST;
                view.host_data = argument.scalar;
                view.element_layout =
                    vernonRuntimeGetScalarValueLayout(static_cast<VernonDataType>(argument.slot.dtype));
                view.access = VERNON_ACCESS_READ;
                view.byte_size = argument.slot.size;
                std::memcpy(record + argument.offset, &written, sizeof(written));
                continue;
            }
            BoundBinding *binding = bound && bound->values[index].is_valid() ? &bound->bindings[index] : nullptr;
            if (binding && binding->valid && stillBound(*binding)) {
                buffer = binding->buffer;
            } else if (!bindBuffer(value, argument.slot, argument.declared, views, buffer, binding)) {
                if (binding)
                    binding->valid = false;
                return false;
            }
            const size_t rank = buffer.shape.size();
            argument.shape.resize(rank);
            argument.strides.resize(rank);
            for (size_t dimension = 0; dimension < rank; ++dimension) {
                argument.shape[dimension] = static_cast<uint64_t>(buffer.shape[dimension]);
                argument.strides[dimension] = buffer.strides[dimension] * buffer.itemSize;
            }
            if (device_) {
                view.storage = VERNON_TENSOR_RHI_RESOURCE;
                view.resource = buffer.reference;
            } else {
                view.storage = VERNON_TENSOR_HOST;
                view.host_data = buffer.data;
            }
            const uint32_t access = argument.slot.access;
            view.access = access == VERNON_SCHEDULE_ACCESS_READ_WRITE ? VERNON_ACCESS_READ_WRITE
                          : access == VERNON_SCHEDULE_ACCESS_WRITE    ? VERNON_ACCESS_WRITE
                                                                      : VERNON_ACCESS_READ;
            view.element_layout = buffer.structured
                                      ? argument.elementLayout()
                                      : vernonRuntimeGetScalarValueLayout(static_cast<VernonDataType>(buffer.dtype));
            view.rank = static_cast<uint32_t>(rank);
            view.shape = rank ? argument.shape.data() : nullptr;
            view.byte_strides = rank ? argument.strides.data() : nullptr;
            view.byte_offset = static_cast<size_t>(buffer.offset * buffer.itemSize);
            view.byte_size = static_cast<size_t>(buffer.capacity * buffer.itemSize);
            std::memcpy(record + argument.offset, &written, sizeof(written));
        }
        return true;
    }

    // Publishes the writes of the call that last packed this record, which signalled `point`, or poisons what it wrote
    // when it failed.
    void publish(VernonStatus status, uint64_t point, const BoundSlots *bound, const nb::object &timeline) {
        nb::object pointObject = nb::int_(point);
        for (size_t index = 0; index < arguments_.size(); ++index) {
            if (!buffers_[index].owner || !buffers_[index].writes)
                continue;
            BoundBinding *binding = bound && bound->bindings[index].valid ? &bound->bindings[index] : nullptr;
            publishWrite(buffers_[index], binding, status != VERNON_STATUS_OK, pointObject.ptr(), timeline.ptr());
        }
    }

private:
    struct Argument {
        // The linked slot the argument writes, with every node's access; kind zero for one that writes none itself.
        VernonScheduleSlot slot{};
        uint64_t offset{};
        uint32_t boundary{};
        std::vector<int64_t> declared;
        // The boundary's element layout, which a structured buffer is passed with.
        ProgramParameterMetadata parameter;
        // What the packed TensorView points at.
        std::vector<uint64_t> shape;
        std::vector<int64_t> strides;
        alignas(16) uint8_t scalar[16]{};

        VernonValueLayoutView elementLayout() const {
            return {sizeof(VernonValueLayoutView),  parameter.elementByteSize,
                    parameter.elementAlignment,     {parameter.layoutHash.data(), parameter.layoutHash.size()},
                    parameter.elementLeaves.data(), parameter.elementLeaves.size()};
        }
    };

    std::vector<Argument> arguments_;
    std::vector<BufferArgument> buffers_;
    std::vector<PyObject *> values_;
};

struct VjpDeleter {
    void operator()(VernonScheduleVjp *vjp) const noexcept { vernonScheduleVjpDestroy(vjp); }
};

struct ResidualsDeleter {
    void operator()(VernonScheduleResiduals *residuals) const noexcept { vernonScheduleResidualsDestroy(residuals); }
};

// The derivative boundary slot of `role` that differentiates boundary slot `primal`; a derivative boundary is named
// as the input or output it differentiates, or as the field path into it that the derivative was taken by. A primal
// has at most one derivative boundary of each role.
uint32_t derivativeBoundary(VernonProgramExecutable *executable, uint32_t primal, VernonProgramBoundaryRole role) {
    VernonProgramParameterView view{};
    std::string name;
    for (VernonProgramBoundaryRole primalRole : {VERNON_PROGRAM_BOUNDARY_INPUT, VERNON_PROGRAM_BOUNDARY_OUTPUT})
        for (size_t index = 0;
             name.empty() && index < vernonRuntimeProgramExecutableGetBoundaryCount(executable, primalRole); ++index)
            if (vernonRuntimeProgramExecutableGetBoundaryByIndex(executable, primalRole, index, &view) ==
                    VERNON_STATUS_OK &&
                view.slot == primal)
                name = nativeStringView(view.name);
    uint32_t fieldPath = UINT32_MAX;
    for (size_t index = 0; !name.empty() && index < vernonRuntimeProgramExecutableGetBoundaryCount(executable, role);
         ++index) {
        if (vernonRuntimeProgramExecutableGetBoundaryByIndex(executable, role, index, &view) != VERNON_STATUS_OK)
            continue;
        const std::string derivative = nativeStringView(view.name);
        if (derivative == name)
            return view.slot;
        if (derivative.size() > name.size() && derivative[name.size()] == '.' &&
            derivative.compare(0, name.size(), name) == 0)
            fieldPath = view.slot;
    }
    if (fieldPath != UINT32_MAX)
        return fieldPath;
    throw std::invalid_argument("a backward schedule slot differentiates no named boundary");
}

// The ARGUMENT slots of `schedule` by the boundary slot `boundaryOf` names for each.
template <typename BoundaryOf>
ArgumentOffsets argumentOffsets(const VernonLinkedSchedule *schedule, BoundaryOf boundaryOf) {
    ArgumentOffsets layout;
    const VernonScheduleSlot *slots = vernonScheduleSlots(schedule);
    layout.count = vernonScheduleSlotCount(schedule);
    for (size_t at = 0; at < layout.count; ++at) {
        if (slots[at].kind != VERNON_SCHEDULE_SLOT_ARGUMENT)
            throw std::invalid_argument("a linked VJP schedule takes only Program arguments");
        const uint32_t boundary = boundaryOf(at);
        if (layout.offsets.size() <= boundary)
            layout.offsets.resize(boundary + 1, UINT64_MAX);
        layout.offsets[boundary] = slots[at].offset;
    }
    return layout;
}

// What a linked VJP shares with the pullbacks of its calls, which may outlive its Python object.
struct LinkedVjpState {
    nb::object executableObject;
    PythonProgramExecutable *executable{};
    std::unique_ptr<VernonScheduleVjp, VjpDeleter> vjp;
    ArgumentOffsets forward;
    ArgumentOffsets backward;
    std::vector<uint8_t> forwardTemplate;
    std::vector<uint8_t> backwardTemplate;
    nb::object timeline;
    // The records calls from Python values pack, when the VJP takes them; a call packs, runs and publishes with the
    // GIL held, so one record serves every call.
    std::unique_ptr<ArgumentRecord> forwardRecord;
    std::unique_ptr<ArgumentRecord> backwardRecord;
    // A call writes its schedule's inner record, so the calls of one VJP are serialized.
    std::mutex mutex;
};

// Slots of one of a VJP's records bound once (`step.bind`).
struct VjpBinding {
    std::shared_ptr<LinkedVjpState> state;
    bool backward{};
    BoundSlots slots;
};

// The backward call of one forward call. Its residual block lives until every call that reads it has completed.
class LinkedVjpCall final : public LinkedBackward {
public:
    LinkedVjpCall(std::shared_ptr<LinkedVjpState> state, VernonScheduleResiduals *residuals)
        : state_(std::move(state)), residuals_(residuals) {}
    ~LinkedVjpCall() override {
        nb::gil_scoped_release released;
        residuals_.reset();
    }

    uint64_t call(const ProgramInvocationBuilder &arguments) override {
        std::vector<uint8_t> record = state_->backwardTemplate;
        packArguments(arguments, state_->backward, record);
        uint64_t point = 0;
        VernonStatus status = VERNON_STATUS_OK;
        {
            nb::gil_scoped_release released;
            std::lock_guard<std::mutex> lock(state_->mutex);
            status =
                vernonScheduleVjpBackward(state_->vjp.get(), record.data(), record.size(), residuals_.get(), &point);
        }
        if (status != VERNON_STATUS_OK)
            throwScheduleError(status, state_->executable->runtime);
        return point;
    }

    // The backward call from Python values, a cotangent or gradient per slot of the backward record that `bound`
    // leaves unbound; false, having run nothing, when a value does not pack.
    bool callValues(nb::handle arguments, const VjpBinding *bound) {
        LinkedVjpState &state = *state_;
        if (!state.backwardRecord || (bound && (!bound->backward || bound->state != state_)))
            return false;
        const BoundSlots *slots = bound ? &bound->slots : nullptr;
        std::vector<uint8_t> record = state.backwardTemplate;
        BufferViews views;
        if (!state.backwardRecord->pack(arguments.ptr(), slots, record.data(), views))
            return false;
        uint64_t point = 0;
        VernonStatus status = VERNON_STATUS_OK;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            status = vernonScheduleVjpBackward(state.vjp.get(), record.data(), record.size(), residuals_.get(), &point);
        }
        state.backwardRecord->publish(status, point, slots, state.timeline);
        if (status != VERNON_STATUS_OK)
            throwScheduleError(status, state.executable->runtime);
        return true;
    }

    void wait(uint64_t point) const override {
        VernonStatus status = VERNON_STATUS_OK;
        {
            nb::gil_scoped_release released;
            status = vernonScheduleWait(state_->executable->runtime, point);
        }
        if (status != VERNON_STATUS_OK)
            throwScheduleError(status, state_->executable->runtime);
    }
    const nb::object &timeline() const override { return state_->timeline; }
    uint64_t residualBytes() const override { return vernonScheduleResidualsGetByteSize(residuals_.get()); }
    nb::object checkpointPlan() const override {
        const VernonScheduleCheckpoint plan = checkpoint();
        if (!plan.present)
            return nb::none();
        nb::dict value;
        value["peak_bytes"] = plan.held_bytes;
        value["memory_budget"] = plan.memory_budget;
        value["logical_residual_bytes"] = plan.kept_bytes;
        value["retained_allocation_bytes"] = plan.kept_bytes;
        value["initial_state_bytes"] = plan.initial_state_bytes;
        value["restoration_bytes"] = plan.recomputed_bytes;
        value["transaction_bytes"] = 0;
        value["persistent_checkpoint_bytes"] = plan.initial_state_bytes;
        value["backward_value_bytes"] = 0;
        value["replay_cost"] = plan.recomputed_nodes;
        value["recomputation_cost"] = plan.recomputed_nodes;
        value["selected_policy"] = plan.policy == VERNON_SCHEDULE_CHECKPOINT_MIN_MEMORY    ? "min_memory"
                                   : plan.policy == VERNON_SCHEDULE_CHECKPOINT_MIN_RUNTIME ? "min_runtime"
                                                                                           : "balanced";
        return value;
    }
    double recomputationFactor() const override {
        const VernonScheduleCheckpoint plan = checkpoint();
        return plan.present && plan.forward_nodes
                   ? 1.0 + static_cast<double>(plan.recomputed_nodes) / static_cast<double>(plan.forward_nodes)
                   : 1.0;
    }
    // A pass's logical, resident and allocated tape bytes count only a tape with control history.
    nb::list passTelemetry() const override {
        nb::list result;
        for (size_t index = 0; index < vernonScheduleVjpPassCount(state_->vjp.get()); ++index) {
            VernonScheduleVjpPass pass{};
            pass.struct_size = sizeof(pass);
            if (vernonScheduleVjpGetPass(state_->vjp.get(), residuals_.get(), index, &pass) != VERNON_STATUS_OK)
                throw std::runtime_error("cannot query linked pullback pass telemetry");
            std::string name = nativeStringView(pass.name);
            for (const std::string_view suffix : {std::string_view(".forward_with_tape"), std::string_view(".vjp")})
                if (name.size() >= suffix.size() &&
                    name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    name.resize(name.size() - suffix.size());
                    break;
                }
            const bool tape = pass.tape_count != 0;
            const bool history = tape && pass.control_history;
            nb::dict telemetry;
            telemetry["schedule_offset"] = pass.node;
            telemetry["pass_name"] = name;
            telemetry["residual_source_kind"] = tape ? (pass.dynamic_tape ? "dynamic_capture" : "static_capture")
                                                : pass.residual_bytes ? "static_capture"
                                                                      : "none";
            telemetry["control_history_kind"] = history ? "dynamic_capture" : "none";
            telemetry["estimated_tape_bytes"] = pass.tape_logical_bytes;
            telemetry["logical_residual_bytes"] = history ? pass.tape_logical_bytes : 0;
            telemetry["resident_tape_bytes"] = history ? pass.tape_resident_bytes : 0;
            telemetry["allocated_tape_bytes"] = history ? pass.tape_allocated_bytes : 0;
            telemetry["retained_allocation_bytes"] = pass.tape_allocated_bytes + pass.residual_bytes;
            telemetry["peak_temporary_tape_bytes"] =
                tape ? std::max<uint64_t>({pass.tape_allocated_bytes, pass.tape_logical_bytes, 1}) : 0;
            telemetry["checkpoint_bytes"] = 0;
            telemetry["active_operation_count"] = 0;
            telemetry["recomputation_cost"] = pass.recomputed;
            result.append(std::move(telemetry));
        }
        return result;
    }
    std::shared_ptr<HalDeviceState> device() const override { return state_->executable->owner->device; }

private:
    VernonScheduleCheckpoint checkpoint() const {
        VernonScheduleCheckpoint plan{};
        plan.struct_size = sizeof(plan);
        if (vernonScheduleResidualsGetCheckpoint(residuals_.get(), &plan) != VERNON_STATUS_OK)
            throw std::runtime_error("cannot query linked pullback checkpoint plan");
        return plan;
    }

    std::shared_ptr<LinkedVjpState> state_;
    std::unique_ptr<VernonScheduleResiduals, ResidualsDeleter> residuals_;
};

// The residuals of one forward call from Python values, whose backward call takes Python values too; `pullback` makes
// the general pullback over the same residuals, for a backward call whose values do not pack.
class VjpCall {
public:
    VjpCall(std::shared_ptr<LinkedVjpState> state, std::shared_ptr<LinkedVjpCall> call)
        : state_(std::move(state)), call_(std::move(call)) {}

    bool backward(nb::handle arguments, const VjpBinding *bound) { return call_->callValues(arguments, bound); }

    // The pullback whose gradients take the shapes of `bindings`, the forward call's inputs and outputs by path.
    std::unique_ptr<PythonPullback> pullback(const nb::dict &bindings) const {
        PythonProgramExecutable &executable = *state_->executable;
        auto [gradients, cotangents] = executable.derivativeLeaves(bindings);
        return std::make_unique<PythonPullback>(executable.owner, executable.runtime, executable.executable, call_,
                                                std::move(gradients), std::move(cotangents), state_->executableObject,
                                                nb::dict(bindings));
    }

private:
    std::shared_ptr<LinkedVjpState> state_;
    std::shared_ptr<LinkedVjpCall> call_;
};

// The name of boundary slot `slot` among the boundaries of `roles`; empty when none has it.
std::string boundaryName(VernonProgramExecutable *executable, uint32_t slot,
                         std::initializer_list<VernonProgramBoundaryRole> roles,
                         VernonProgramBoundaryRole *found = nullptr) {
    VernonProgramParameterView view{};
    for (VernonProgramBoundaryRole role : roles)
        for (size_t index = 0; index < vernonRuntimeProgramExecutableGetBoundaryCount(executable, role); ++index)
            if (vernonRuntimeProgramExecutableGetBoundaryByIndex(executable, role, index, &view) == VERNON_STATUS_OK &&
                view.slot == slot) {
                if (found)
                    *found = role;
                return nativeStringView(view.name);
            }
    return {};
}

// A Program executable's forward and backward graphs linked as a VJP (vernonScheduleLinkVjp). A forward call packs
// the arguments a builder prepared, as ProgramSchedule::call does, and returns its pullback, which runs the backward
// schedule over that call's residuals. A memory budget or policy links it checkpointed
// (vernonScheduleLinkCheckpointedVjp). Given the TensorStorage and TensorView types, and on a GPU the session's
// residency key, it also takes calls from Python values in slot order (`call`), as a ScheduleKernel does.
class ProgramVjp {
public:
    ProgramVjp(nb::object executable, std::optional<uint64_t> budget, const std::string &policy, nb::object storageType,
               nb::object viewType, nb::object residencyKey)
        : state_(std::make_shared<LinkedVjpState>()) {
        LinkedVjpState &state = *state_;
        state.executableObject = std::move(executable);
        state.executable = nb::cast<PythonProgramExecutable *>(state.executableObject);
        VernonRuntimeContext *runtime = state.executable->runtime;
        if (budget || !policy.empty()) {
            VernonScheduleCheckpointOptions options{};
            options.struct_size = sizeof(options);
            options.memory_budget = budget.value_or(UINT64_MAX);
            if (policy.empty() || policy == "balanced")
                options.policy = VERNON_SCHEDULE_CHECKPOINT_BALANCED;
            else if (policy == "min_memory")
                options.policy = VERNON_SCHEDULE_CHECKPOINT_MIN_MEMORY;
            else if (policy == "min_runtime")
                options.policy = VERNON_SCHEDULE_CHECKPOINT_MIN_RUNTIME;
            else
                throw std::invalid_argument("checkpoint policy '" + policy +
                                            "' is not min_memory, balanced or "
                                            "min_runtime");
            state.vjp.reset(vernonScheduleLinkCheckpointedVjp(runtime, state.executable->executable, &options));
        } else {
            state.vjp.reset(vernonScheduleLinkVjp(runtime, state.executable->executable));
        }
        if (!state.vjp)
            throw std::invalid_argument(nativeStringView(vernonRuntimeGetLastError(runtime)));
        const VernonLinkedSchedule *forward = vernonScheduleVjpSchedule(state.vjp.get(), 0);
        const VernonLinkedSchedule *backward = vernonScheduleVjpSchedule(state.vjp.get(), 1);
        state.forward = argumentOffsets(forward, [&](size_t slot) {
            const uint32_t boundary = vernonScheduleSlotBoundary(forward, slot);
            if (boundary == UINT32_MAX)
                throw std::invalid_argument("the forward schedule has an argument of no Program boundary slot");
            return boundary;
        });
        state.backward = argumentOffsets(backward, [&](size_t slot) {
            return derivativeBoundary(state.executable->executable, vernonScheduleSlotBoundary(backward, slot),
                                      vernonScheduleVjpSlotRole(state.vjp.get(), slot));
        });
        const uint8_t *bytes = vernonScheduleRecordTemplate(forward);
        state.forwardTemplate.assign(bytes, bytes + vernonScheduleRecordSize(forward));
        bytes = vernonScheduleRecordTemplate(backward);
        state.backwardTemplate.assign(bytes, bytes + vernonScheduleRecordSize(backward));
        state.timeline = nb::cast(ScheduleTimeline(state.executable->owner));
        if (!storageType.is_none()) {
            VernonProgramExecutable *executable = state.executable->executable;
            state.forwardRecord = std::make_unique<ArgumentRecord>(state.executable->owner, executable, forward,
                                                                   storageType, viewType, residencyKey);
            state.backwardRecord = std::make_unique<ArgumentRecord>(state.executable->owner, executable, backward,
                                                                    storageType, viewType, residencyKey);
        }
    }

    bool calls() const { return state_->forwardRecord != nullptr; }

    // Per slot of a call from Python values, the path of the boundary it takes and whether that is an input.
    nb::list forwardSlots() const {
        const VernonLinkedSchedule *forward = vernonScheduleVjpSchedule(state_->vjp.get(), 0);
        nb::list slots;
        for (size_t slot = 0; slot < vernonScheduleSlotCount(forward); ++slot) {
            VernonProgramBoundaryRole role = VERNON_PROGRAM_BOUNDARY_INPUT;
            const std::string name =
                boundaryName(state_->executable->executable, vernonScheduleSlotBoundary(forward, slot),
                             {VERNON_PROGRAM_BOUNDARY_INPUT, VERNON_PROGRAM_BOUNDARY_OUTPUT}, &role);
            if (name.empty())
                throw std::invalid_argument("the forward schedule takes an argument of no named boundary");
            slots.append(nb::make_tuple(name, role == VERNON_PROGRAM_BOUNDARY_INPUT));
        }
        return slots;
    }

    // Per slot of a backward call from Python values, "cotangent" or "gradient" and the path of its derivative
    // boundary, which is a field path into the input or output it differentiates when the derivative was taken by one.
    nb::list backwardSlots() const {
        const VernonLinkedSchedule *backward = vernonScheduleVjpSchedule(state_->vjp.get(), 1);
        nb::list slots;
        for (size_t slot = 0; slot < vernonScheduleSlotCount(backward); ++slot) {
            const VernonProgramBoundaryRole role = vernonScheduleVjpSlotRole(state_->vjp.get(), slot);
            VernonProgramExecutable *executable = state_->executable->executable;
            const uint32_t derivative =
                derivativeBoundary(executable, vernonScheduleSlotBoundary(backward, slot), role);
            slots.append(nb::make_tuple(role == VERNON_PROGRAM_BOUNDARY_COTANGENT ? "cotangent" : "gradient",
                                        boundaryName(executable, derivative, {role})));
        }
        return slots;
    }

    // The forward call from Python values, those of the forward slots `bound` leaves unbound, in order; its residuals,
    // or None, having run nothing, when a value does not pack.
    nb::object call(nb::handle arguments, const VjpBinding *bound) {
        LinkedVjpState &state = *state_;
        if (!state.forwardRecord || (bound && (bound->backward || bound->state != state_)))
            return nb::none();
        const BoundSlots *slots = bound ? &bound->slots : nullptr;
        std::vector<uint8_t> record = state.forwardTemplate;
        BufferViews views;
        if (!state.forwardRecord->pack(arguments.ptr(), slots, record.data(), views))
            return nb::none();
        VernonScheduleResiduals *residuals = nullptr;
        uint64_t point = 0;
        VernonStatus status = VERNON_STATUS_OK;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            status = vernonScheduleVjpForward(state.vjp.get(), record.data(), record.size(), &residuals, &point);
        }
        state.forwardRecord->publish(status, point, slots, state.timeline);
        if (status != VERNON_STATUS_OK)
            throwScheduleError(status, state.executable->runtime);
        return nb::cast(VjpCall(state_, std::make_shared<LinkedVjpCall>(state_, residuals)));
    }

    // Binds `values` to the slots `positions` of the forward record, or of the backward record when `backward`.
    std::unique_ptr<VjpBinding> bind(bool backward, const std::vector<uint32_t> &positions,
                                     const nb::tuple &values) const {
        const ArgumentRecord *record = backward ? state_->backwardRecord.get() : state_->forwardRecord.get();
        if (!record)
            throw std::invalid_argument("the VJP takes no calls from Python values");
        auto binding = std::make_unique<VjpBinding>();
        binding->state = state_;
        binding->backward = backward;
        binding->slots = record->boundSlots(positions, values);
        return binding;
    }

    std::unique_ptr<ProgramInvocationBuilder> builder() const {
        const PythonProgramExecutable &executable = *state_->executable;
        return std::make_unique<ProgramInvocationBuilder>(executable.owner, executable.runtime, executable.executable);
    }

    // The call's pullback, whose gradients take the shapes of `bindings`, and the point its submission signals.
    nb::tuple forward(const ProgramInvocationBuilder &arguments, const nb::dict &bindings) {
        PythonProgramExecutable &executable = *state_->executable;
        auto [gradients, cotangents] = executable.derivativeLeaves(bindings);
        std::vector<uint8_t> record = state_->forwardTemplate;
        packArguments(arguments, state_->forward, record);
        VernonScheduleResiduals *residuals = nullptr;
        uint64_t point = 0;
        VernonStatus status = VERNON_STATUS_OK;
        {
            nb::gil_scoped_release released;
            std::lock_guard<std::mutex> lock(state_->mutex);
            status = vernonScheduleVjpForward(state_->vjp.get(), record.data(), record.size(), &residuals, &point);
        }
        if (status != VERNON_STATUS_OK)
            throwScheduleError(status, executable.runtime);
        auto call = std::make_shared<LinkedVjpCall>(state_, residuals);
        return nb::make_tuple(std::make_unique<PythonPullback>(executable.owner, executable.runtime,
                                                               executable.executable, std::move(call),
                                                               std::move(gradients), std::move(cotangents),
                                                               state_->executableObject, nb::dict(bindings)),
                              point);
    }

    const nb::object &timeline() const { return state_->timeline; }

private:
    std::shared_ptr<LinkedVjpState> state_;
};

} // namespace

void bindNativeSchedule(nb::module_ &module) {
    nb::class_<ScheduleTimeline>(module, "_ScheduleTimeline").def("wait", &ScheduleTimeline::wait, nb::arg("point"));
    nb::class_<ResidualBlock>(module, "_ResidualBlock");
    nb::class_<ScheduleKernel>(module, "_ScheduleKernel")
        .def(nb::init<const Runtime &, nb::object, const std::string &, nb::object, nb::object, nb::object, nb::object,
                      nb::object>(),
             nb::arg("runtime"), nb::arg("program").none(), nb::arg("entry"), nb::arg("storage_type"),
             nb::arg("view_type"), nb::arg("reflection").none() = nb::none(), nb::arg("artifact").none() = nb::none(),
             nb::arg("residency_key").none() = nb::none())
        .def_static("link_program", &ScheduleKernel::linkProgram, nb::arg("runtime"), nb::arg("program"),
                    nb::arg("graph"), nb::arg("stages"), nb::arg("zeroed"), nb::arg("exposed"), nb::arg("storage_type"),
                    nb::arg("view_type"), nb::arg("residency_key").none() = nb::none())
        .def_static("link_graph", &ScheduleKernel::linkGraph, nb::arg("runtime"), nb::arg("nodes"),
                    nb::arg("connections"), nb::arg("storage_type"), nb::arg("view_type"),
                    nb::arg("residency_key").none() = nb::none())
        .def_prop_ro("node_slots", &ScheduleKernel::nodeSlots)
        .def_prop_ro("slot_kinds", &ScheduleKernel::slotKinds)
        .def_prop_ro("slot_extents", &ScheduleKernel::slotExtents)
        .def(
            "call",
            [](ScheduleKernel &self, nb::handle arguments, nb::handle grid) { return self.call(arguments, grid); },
            nb::arg("arguments"), nb::arg("grid").none())
        .def("set_block_slots", &ScheduleKernel::setBlockSlots, nb::arg("slots"), nb::arg("producing"))
        .def(
            "call_producing", [](ScheduleKernel &self, nb::handle arguments) { return self.callProducing(arguments); },
            nb::arg("arguments"))
        .def(
            "call_consuming",
            [](ScheduleKernel &self, nb::handle arguments, const ResidualBlock &block) {
                return self.callConsuming(arguments, block);
            },
            nb::arg("arguments"), nb::arg("block"));
    nb::class_<BoundSchedule>(module, "_BoundSchedule")
        .def(nb::init<nb::object, const std::vector<uint32_t> &, nb::tuple>(), nb::arg("schedule"), nb::arg("slots"),
             nb::arg("values"))
        .def("call", &BoundSchedule::call, nb::arg("arguments"), nb::arg("grid").none())
        .def("call_producing", &BoundSchedule::callProducing, nb::arg("arguments"))
        .def("call_consuming", &BoundSchedule::callConsuming, nb::arg("arguments"), nb::arg("block"));
    nb::class_<ProgramSchedule>(module, "_ProgramSchedule")
        .def(nb::init<nb::object>(), nb::arg("executable"))
        .def("builder", &ProgramSchedule::builder, nb::keep_alive<0, 1>())
        .def("call", &ProgramSchedule::call, nb::arg("arguments"), nb::arg("passes"))
        .def_prop_ro("pass_count", &ProgramSchedule::passCount)
        .def_prop_ro("timeline", &ProgramSchedule::timeline);
    nb::class_<VjpBinding>(module, "_VjpBinding");
    nb::class_<VjpCall>(module, "_VjpCall")
        .def("backward", &VjpCall::backward, nb::arg("arguments"), nb::arg("bound").none() = nb::none())
        .def("pullback", &VjpCall::pullback, nb::arg("bindings"));
    nb::class_<ProgramVjp>(module, "_ProgramVjp")
        .def(nb::init<nb::object, std::optional<uint64_t>, const std::string &, nb::object, nb::object, nb::object>(),
             nb::arg("executable"), nb::arg("checkpoint_memory_budget").none() = nb::none(),
             nb::arg("checkpoint_policy") = "", nb::arg("storage_type").none() = nb::none(),
             nb::arg("view_type").none() = nb::none(), nb::arg("residency_key").none() = nb::none())
        .def("builder", &ProgramVjp::builder, nb::keep_alive<0, 1>())
        .def("forward", &ProgramVjp::forward, nb::arg("arguments"), nb::arg("bindings"))
        .def("call", &ProgramVjp::call, nb::arg("arguments"), nb::arg("bound").none() = nb::none())
        .def("bind", &ProgramVjp::bind, nb::arg("backward"), nb::arg("positions"), nb::arg("values"))
        .def_prop_ro("calls", &ProgramVjp::calls)
        .def_prop_ro("forward_slots", &ProgramVjp::forwardSlots)
        .def_prop_ro("backward_slots", &ProgramVjp::backwardSlots)
        .def_prop_ro("timeline", &ProgramVjp::timeline);
}
