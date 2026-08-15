#ifndef VERNON_RUNTIME_HPP
#define VERNON_RUNTIME_HPP

#include "VernonRuntime.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace vernon::runtime {

struct Float16 {
    uint16_t bits{};
};

namespace detail {

template <typename T> struct DataType;
template <> struct DataType<bool> {
    static constexpr VernonDataType value = VERNON_DATA_BOOL;
};
template <> struct DataType<int32_t> {
    static constexpr VernonDataType value = VERNON_DATA_I32;
};
template <> struct DataType<uint32_t> {
    static constexpr VernonDataType value = VERNON_DATA_U32;
};
template <> struct DataType<Float16> {
    static constexpr VernonDataType value = VERNON_DATA_F16;
};
template <> struct DataType<float> {
    static constexpr VernonDataType value = VERNON_DATA_F32;
};
template <> struct DataType<double> {
    static constexpr VernonDataType value = VERNON_DATA_F64;
};
template <> struct DataType<uint8_t> {
    static constexpr VernonDataType value = VERNON_DATA_U8;
};

template <typename T, typename = void> struct HasDataAndSize : std::false_type {};
template <typename T>
struct HasDataAndSize<
    T, std::void_t<decltype(std::declval<const T &>().data()), decltype(std::declval<const T &>().size())>>
    : std::true_type {};

inline size_t checkedElementCount(const std::vector<uint64_t> &shape) {
    size_t count = 1;
    for (uint64_t extent : shape) {
        if (!extent || extent > std::numeric_limits<size_t>::max() / count)
            throw std::invalid_argument("Tensor shape must be positive and non-overflowing");
        count *= static_cast<size_t>(extent);
    }
    return count;
}

} // namespace detail

class StructuredValue {
public:
    struct PathComponent {
        std::string field;
        uint64_t index{};
        bool isField{};

        bool operator==(const PathComponent &other) const {
            return isField == other.isField && (isField ? field == other.field : index == other.index);
        }
    };

    static StructuredValue product() {
        StructuredValue result;
        result.kind_ = Kind::Product;
        return result;
    }

    template <typename T> static StructuredValue scalar(const T &value) {
        using Value = std::remove_cv_t<T>;
        return array(detail::DataType<Value>::value, &value, 1);
    }

    template <typename T> static StructuredValue array(const T *data, size_t count) {
        using Value = std::remove_cv_t<T>;
        return array(detail::DataType<Value>::value, data, count);
    }

    StructuredValue &addField(std::string name, StructuredValue value) {
        requireProduct();
        if (name.empty())
            throw std::invalid_argument("structured Value field name must not be empty");
        addChild({std::move(name), 0, true}, std::move(value));
        return *this;
    }

    StructuredValue &addElement(uint64_t index, StructuredValue value) {
        requireProduct();
        addChild({"", index, false}, std::move(value));
        return *this;
    }

    StructuredValue &withShape(std::vector<uint64_t> shape) {
        if (kind_ != Kind::Leaf)
            throw std::invalid_argument("only a scalar leaf view can carry a static shape");
        if (detail::checkedElementCount(shape) != scalarCount_)
            throw std::invalid_argument("shaped scalar view size does not match its shape");
        shape_ = std::move(shape);
        return *this;
    }

private:
    enum class Kind { Leaf, Product };

    struct Child {
        PathComponent component;
        std::shared_ptr<StructuredValue> value;
    };

    Kind kind_{Kind::Leaf};
    VernonDataType dtype_{VERNON_DATA_BOOL};
    size_t scalarCount_{};
    std::vector<uint8_t> bytes_;
    std::vector<uint64_t> shape_;
    std::vector<Child> children_;

    static size_t dataTypeSize(VernonDataType dtype) {
        switch (dtype) {
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
        return 0;
    }

    static StructuredValue array(VernonDataType dtype, const void *data, size_t count) {
        const size_t scalarSize = dataTypeSize(dtype);
        if (!data || !count || scalarSize > std::numeric_limits<size_t>::max() / count)
            throw std::invalid_argument("structured Value scalar view is empty or too large");
        StructuredValue result;
        result.kind_ = Kind::Leaf;
        result.dtype_ = dtype;
        result.scalarCount_ = count;
        result.bytes_.resize(scalarSize * count);
        std::memcpy(result.bytes_.data(), data, result.bytes_.size());
        return result;
    }

    void requireProduct() const {
        if (kind_ != Kind::Product)
            throw std::invalid_argument("structured Value node is not a field/element product");
    }

    void addChild(PathComponent component, StructuredValue value) {
        const auto duplicate = std::find_if(children_.begin(), children_.end(),
                                            [&](const Child &child) { return child.component == component; });
        if (duplicate != children_.end())
            throw std::invalid_argument("structured Value contains a duplicate field or element");
        children_.push_back({std::move(component), std::make_shared<StructuredValue>(std::move(value))});
    }

    friend class PipelineInvocationBuilder;
};

struct NamedField {
    std::string name;
    StructuredValue value;
};

inline StructuredValue value(StructuredValue input) { return input; }

template <typename T, std::enable_if_t<!detail::HasDataAndSize<T>::value, int> = 0>
StructuredValue value(const T &input) {
    return StructuredValue::scalar(input);
}

template <typename Range, std::enable_if_t<detail::HasDataAndSize<Range>::value, int> = 0>
StructuredValue value(const Range &input) {
    using Element = std::remove_cv_t<std::remove_pointer_t<decltype(input.data())>>;
    return StructuredValue::array<Element>(input.data(), static_cast<size_t>(input.size()));
}

template <typename T, size_t Size> StructuredValue value(const T (&input)[Size]) {
    return StructuredValue::array<T>(input, Size);
}

template <typename T> StructuredValue values(const T *data, size_t count) {
    return StructuredValue::array<T>(data, count);
}

template <typename T> NamedField field(std::string name, T &&input) {
    return {std::move(name), value(std::forward<T>(input))};
}

template <typename... Fields> StructuredValue fields(Fields &&...input) {
    StructuredValue result = StructuredValue::product();
    (result.addField(std::forward<Fields>(input).name, std::forward<Fields>(input).value), ...);
    return result;
}

template <typename... Values> StructuredValue elements(Values &&...input) {
    StructuredValue result = StructuredValue::product();
    uint64_t index = 0;
    (result.addElement(index++, value(std::forward<Values>(input))), ...);
    return result;
}

inline StructuredValue shaped(std::vector<uint64_t> shape, StructuredValue input) {
    return std::move(input.withShape(std::move(shape)));
}

template <typename T> StructuredValue shaped(std::vector<uint64_t> shape, T &&input) {
    return shaped(std::move(shape), value(std::forward<T>(input)));
}

class PipelineInvocationBuilder {
public:
    explicit PipelineInvocationBuilder(VernonLoadedPipeline *pipeline) : pipeline_(pipeline) {
        if (!pipeline_)
            throw std::invalid_argument("pipeline invocation builder requires a loaded pipeline");
    }

    PipelineInvocationBuilder &bindValue(std::string_view parameterName, const StructuredValue &source) {
        Parameter parameter = reflect(parameterName);
        if (!parameter.shape.empty())
            throw std::invalid_argument("pipeline parameter '" + parameter.name + "' is a Tensor, not a Value");
        OwnedArgument &argument = addArgument(parameter);
        argument.data = pack(parameter, source, {}, 1);
        argument.shape.clear();
        return *this;
    }

    template <typename T, std::enable_if_t<!std::is_same_v<std::decay_t<T>, StructuredValue>, int> = 0>
    PipelineInvocationBuilder &bindValue(std::string_view parameterName, T &&source) {
        return bindValue(parameterName, value(std::forward<T>(source)));
    }

    PipelineInvocationBuilder &bindTensor(std::string_view parameterName, std::vector<uint64_t> shape,
                                          const StructuredValue &source) {
        Parameter parameter = reflect(parameterName);
        validateOuterShape(parameter, shape);
        const size_t count = detail::checkedElementCount(shape);
        OwnedArgument &argument = addArgument(parameter);
        argument.data = pack(parameter, source, shape, count);
        argument.shape = std::move(shape);
        return *this;
    }

    PipelineInvocationBuilder &bindTensor(std::string_view parameterName, std::initializer_list<uint64_t> shape,
                                          const StructuredValue &source) {
        return bindTensor(parameterName, std::vector<uint64_t>(shape), source);
    }

    class ElementWriter {
    public:
        template <typename T> ElementWriter &field(std::string name, T &&input) {
            root_.addField(std::move(name), value(std::forward<T>(input)));
            return *this;
        }

        template <typename T> ElementWriter &element(uint64_t index, T &&input) {
            root_.addElement(index, value(std::forward<T>(input)));
            return *this;
        }

        StructuredValue &root() { return root_; }

    private:
        StructuredValue root_ = StructuredValue::product();
        friend class PipelineInvocationBuilder;
    };

    template <typename Callback>
    PipelineInvocationBuilder &bindTensor(std::string_view parameterName, std::vector<uint64_t> shape,
                                          Callback callback) {
        Parameter parameter = reflect(parameterName);
        validateOuterShape(parameter, shape);
        const size_t count = detail::checkedElementCount(shape);
        OwnedArgument &argument = addArgument(parameter);
        if (parameter.layout.byte_size > std::numeric_limits<size_t>::max() / count)
            throw std::invalid_argument("packed Tensor byte size overflows");
        argument.data.assign(parameter.layout.byte_size * count, uint8_t{0});
        std::vector<uint64_t> coordinate(shape.size());
        for (size_t linear = 0; linear < count; ++linear) {
            size_t remainder = linear;
            for (size_t dimension = shape.size(); dimension-- > 0;) {
                coordinate[dimension] = remainder % static_cast<size_t>(shape[dimension]);
                remainder /= static_cast<size_t>(shape[dimension]);
            }
            ElementWriter writer;
            if constexpr (std::is_invocable_v<Callback &, ElementWriter &, const std::vector<uint64_t> &>)
                callback(writer, coordinate);
            else if constexpr (std::is_invocable_v<Callback &, ElementWriter &>)
                callback(writer);
            else
                static_assert(std::is_invocable_v<Callback &, ElementWriter &>,
                              "Tensor callback must accept ElementWriter& and optionally the logical coordinate");
            std::vector<uint8_t> packed = pack(parameter, writer.root_, {}, 1);
            std::memcpy(argument.data.data() + linear * parameter.layout.byte_size, packed.data(), packed.size());
        }
        argument.shape = std::move(shape);
        return *this;
    }

    template <typename Callback>
    PipelineInvocationBuilder &bindTensor(std::string_view parameterName, std::initializer_list<uint64_t> shape,
                                          Callback callback) {
        return bindTensor(parameterName, std::vector<uint64_t>(shape), std::move(callback));
    }

    VernonPipelineInvocation invocation() {
        arguments_.clear();
        arguments_.reserve(owned_.size());
        for (OwnedArgument &source : owned_) {
            source.strides.resize(source.shape.size());
            size_t stride = source.parameter.layout.byte_size;
            for (size_t dimension = source.shape.size(); dimension-- > 0;) {
                if (stride > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
                    throw std::invalid_argument("packed Tensor stride exceeds the runtime ABI");
                source.strides[dimension] = static_cast<int64_t>(stride);
                if (source.shape[dimension] > std::numeric_limits<size_t>::max() / stride)
                    throw std::invalid_argument("packed Tensor stride overflows");
                stride *= static_cast<size_t>(source.shape[dimension]);
            }
            VernonTensorView tensor{};
            tensor.struct_size = sizeof(VernonTensorView);
            tensor.storage = VERNON_TENSOR_HOST;
            tensor.host_data = source.data.data();
            tensor.element_layout = source.parameter.layout;
            tensor.access = source.parameter.access;
            tensor.rank = static_cast<uint32_t>(source.shape.size());
            tensor.shape = source.shape.empty() ? nullptr : source.shape.data();
            tensor.byte_strides = source.strides.empty() ? nullptr : source.strides.data();
            tensor.byte_size = source.data.size();
            VernonPipelineArgument argument{};
            argument.slot = source.parameter.slot;
            argument.kind = VERNON_PIPELINE_TENSOR;
            argument.tensor = tensor;
            arguments_.push_back(argument);
        }
        VernonPipelineInvocation result{};
        result.struct_size = sizeof(VernonPipelineInvocation);
        result.abi_version = VERNON_PIPELINE_VERSION;
        result.arguments = arguments_.empty() ? nullptr : arguments_.data();
        result.argument_count = arguments_.size();
        result.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
        result.compute_grid = {1, 1, 1};
        return result;
    }

private:
    struct ReflectedLeaf {
        VernonValueLeafView value{};
        std::vector<StructuredValue::PathComponent> path;
        std::vector<uint64_t> shape;
    };

    struct Parameter {
        uint32_t slot{};
        std::string name;
        VernonValueLayoutView layout{};
        VernonValueAccess access{};
        std::vector<uint64_t> shape;
        std::vector<ReflectedLeaf> leaves;
    };

    struct OwnedArgument {
        Parameter parameter;
        std::vector<uint8_t> data;
        std::vector<uint64_t> shape;
        std::vector<int64_t> strides;
    };

    struct SourceLeaf {
        std::vector<StructuredValue::PathComponent> path;
        const StructuredValue *value{};
    };

    static std::string displayPath(const std::vector<StructuredValue::PathComponent> &path) {
        std::string result;
        for (const auto &component : path) {
            if (component.isField) {
                if (!result.empty())
                    result += '.';
                result += component.field;
            } else {
                result += '[' + std::to_string(component.index) + ']';
            }
        }
        return result.empty() ? std::string("<value>") : result;
    }

    static void flatten(const StructuredValue &source, std::vector<StructuredValue::PathComponent> &path,
                        std::vector<SourceLeaf> &leaves) {
        if (source.kind_ == StructuredValue::Kind::Leaf) {
            leaves.push_back({path, &source});
            return;
        }
        for (const StructuredValue::Child &child : source.children_) {
            path.push_back(child.component);
            flatten(*child.value, path, leaves);
            path.pop_back();
        }
    }

    static std::vector<uint8_t> pack(const Parameter &parameter, const StructuredValue &source,
                                     const std::vector<uint64_t> &outerShape, size_t recordCount) {
        if (parameter.layout.byte_size > std::numeric_limits<size_t>::max() / recordCount)
            throw std::invalid_argument("packed Tensor byte size overflows");
        std::vector<uint8_t> result(parameter.layout.byte_size * recordCount, uint8_t{0});
        std::vector<SourceLeaf> sources;
        std::vector<StructuredValue::PathComponent> path;
        flatten(source, path, sources);
        if (sources.size() != parameter.leaves.size())
            throw std::invalid_argument("structured binding for '" + parameter.name +
                                        "' is incomplete or contains extra fields");
        for (const ReflectedLeaf &leaf : parameter.leaves) {
            const auto found = std::find_if(sources.begin(), sources.end(),
                                            [&](const SourceLeaf &candidate) { return candidate.path == leaf.path; });
            if (found == sources.end())
                throw std::invalid_argument("structured binding for '" + parameter.name + "' is missing '" +
                                            displayPath(leaf.path) + "'");
            const StructuredValue &input = *found->value;
            if (input.dtype_ != static_cast<VernonDataType>(leaf.value.dtype))
                throw std::invalid_argument("structured binding dtype mismatch at '" + displayPath(leaf.path) + "'");
            const size_t expectedScalars = recordCount * static_cast<size_t>(leaf.value.scalar_count);
            if (input.scalarCount_ != expectedScalars)
                throw std::invalid_argument("structured binding scalar count mismatch at '" + displayPath(leaf.path) +
                                            "'");
            if (!input.shape_.empty()) {
                std::vector<uint64_t> expectedShape = outerShape;
                expectedShape.insert(expectedShape.end(), leaf.shape.begin(), leaf.shape.end());
                if (input.shape_ != expectedShape)
                    throw std::invalid_argument("structured binding static shape mismatch at '" +
                                                displayPath(leaf.path) + "'");
            }
            const size_t scalarSize = StructuredValue::dataTypeSize(input.dtype_);
            const size_t recordBytes = scalarSize * leaf.value.scalar_count;
            if (leaf.value.byte_offset > parameter.layout.byte_size ||
                recordBytes > parameter.layout.byte_size - leaf.value.byte_offset)
                throw std::invalid_argument("reflected Value leaf exceeds the canonical element layout");
            for (size_t record = 0; record < recordCount; ++record)
                std::memcpy(result.data() + record * parameter.layout.byte_size + leaf.value.byte_offset,
                            input.bytes_.data() + record * recordBytes, recordBytes);
        }
        return result;
    }

    Parameter reflect(std::string_view name) const {
        VernonPipelineParameterView view{};
        const VernonStringView nameView{name.data(), name.size()};
        if (vernonRuntimeLoadedPipelineFindParameter(pipeline_, nameView, &view) != VERNON_STATUS_OK)
            throw std::invalid_argument("unknown pipeline parameter '" + std::string(name) + "'");
        if (view.kind != VERNON_PIPELINE_TENSOR)
            throw std::invalid_argument("pipeline parameter '" + std::string(name) + "' is not a Value or Tensor");
        Parameter result;
        result.slot = view.slot;
        result.name.assign(view.name.data, view.name.size);
        result.layout = view.element_layout;
        result.access = view.access;
        if (view.rank)
            result.shape.assign(view.static_shape, view.static_shape + view.rank);
        result.leaves.reserve(view.element_layout.leaf_count);
        for (size_t index = 0; index < view.element_layout.leaf_count; ++index) {
            VernonPipelineValueLeafView leaf{};
            leaf.struct_size = sizeof(VernonPipelineValueLeafView);
            if (vernonRuntimeLoadedPipelineGetParameterValueLeaf(pipeline_, nameView, index, &leaf) != VERNON_STATUS_OK)
                throw std::invalid_argument("pipeline parameter has invalid structured Value reflection");
            ReflectedLeaf reflected;
            reflected.value = leaf.value;
            for (size_t componentIndex = 0; componentIndex < leaf.path_count; ++componentIndex) {
                const VernonValuePathComponentView &component = leaf.path[componentIndex];
                if (component.kind == VERNON_VALUE_PATH_FIELD)
                    reflected.path.push_back({std::string(component.field.data, component.field.size), 0, true});
                else if (component.kind == VERNON_VALUE_PATH_INDEX)
                    reflected.path.push_back({"", component.index, false});
                else
                    throw std::invalid_argument("pipeline parameter has an invalid Value leaf path");
            }
            if (leaf.static_rank)
                reflected.shape.assign(leaf.static_shape, leaf.static_shape + leaf.static_rank);
            result.leaves.push_back(std::move(reflected));
        }
        return result;
    }

    static void validateOuterShape(const Parameter &parameter, const std::vector<uint64_t> &shape) {
        detail::checkedElementCount(shape);
        if (shape.size() != parameter.shape.size())
            throw std::invalid_argument("Tensor rank does not match pipeline reflection");
        for (size_t dimension = 0; dimension < shape.size(); ++dimension)
            if (parameter.shape[dimension] && parameter.shape[dimension] != shape[dimension])
                throw std::invalid_argument("Tensor shape does not match pipeline reflection");
    }

    OwnedArgument &addArgument(Parameter parameter) {
        if (std::any_of(owned_.begin(), owned_.end(),
                        [&](const OwnedArgument &argument) { return argument.parameter.slot == parameter.slot; }))
            throw std::invalid_argument("pipeline parameter '" + parameter.name + "' was already bound");
        owned_.push_back({std::move(parameter), {}, {}, {}});
        return owned_.back();
    }

    VernonLoadedPipeline *pipeline_{};
    std::vector<OwnedArgument> owned_;
    std::vector<VernonPipelineArgument> arguments_;
};

class Pullback {
public:
    Pullback() = default;
    explicit Pullback(VernonPullback *handle) : handle_(handle) {}
    Pullback(const Pullback &) = delete;
    Pullback &operator=(const Pullback &) = delete;
    Pullback(Pullback &&other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    Pullback &operator=(Pullback &&other) noexcept {
        if (this != &other) {
            vernonPullbackDestroy(handle_);
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }
    ~Pullback() { vernonPullbackDestroy(handle_); }

    void apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients, uint64_t maximumTemporaryBytes,
               uint64_t maximumReusableConstructionBytes = 0) const {
        if (!handle_)
            throw std::logic_error("pullback is empty");
        const VernonPullbackApplyOptions options{sizeof(VernonPullbackApplyOptions),
                                                 VERNON_PULLBACK_APPLY_OPTIONS_VERSION,
                                                 maximumTemporaryBytes,
                                                 maximumReusableConstructionBytes,
                                                 {}};
        if (vernonPullbackApplyWithOptions(handle_, cotangents, &gradients, &options) != VERNON_STATUS_OK)
            throw std::runtime_error("pullback application failed");
    }

    void apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients) const {
        apply(cotangents, gradients, std::numeric_limits<uint64_t>::max());
    }

    explicit operator bool() const noexcept { return handle_ != nullptr; }
    VernonPullback *get() const noexcept { return handle_; }

private:
    VernonPullback *handle_{};
};

inline Pullback vjp(VernonLoadedPipeline *pipeline, const VernonAdValueSet &inputs, VernonAdValueSet &outputs,
                    VernonLaunchSize computeGrid) {
    VernonPullback *pullback = nullptr;
    if (vernonAdPipelineForward(pipeline, computeGrid, &inputs, &outputs, &pullback) != VERNON_STATUS_OK)
        throw std::runtime_error("autodiff forward invocation failed");
    return Pullback(pullback);
}

} // namespace vernon::runtime

#endif
