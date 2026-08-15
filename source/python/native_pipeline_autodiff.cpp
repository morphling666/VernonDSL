#include "native_pipeline_autodiff.h"

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

namespace {

std::vector<std::string> derivativeGroupLeaves(const nb::handle &group) {
    return nb::cast<std::vector<std::string>>(group.attr("leaf_paths"));
}

nb::object storageCotangentLeaf(const nb::object &value, const nb::handle &group, const std::string &leafPath,
                                const std::vector<uint64_t> &carrierShape, const nb::dict &bindings, bool logical,
                                bool aggregate) {
    if (!aggregate)
        return value.attr("to_numpy")();
    const std::string root = nb::cast<std::string>(group.attr("parameter_root"));
    const std::string suffix = leafPath == root ? std::string{} : leafPath.substr(root.size() + 1);
    nb::object primal = nb::borrow<nb::object>(bindings[nb::str(root.c_str())]);
    nb::object tensorViewType = nb::module_::import_("vernon_dsl._runtime.resources").attr("TensorView");
    if (!nb::isinstance(primal, tensorViewType)) {
        nb::object projection = value.attr("__getitem__")(suffix);
        return projection.attr("to_numpy")();
    }

    std::vector<uint64_t> shape = logical ? std::vector<uint64_t>{} : carrierShape;
    const std::vector<uint64_t> primalShape = nb::cast<std::vector<uint64_t>>(primal.attr("shape"));
    shape.insert(shape.end(), primalShape.begin(), primalShape.end());
    std::vector<int64_t> strides;
    if (!logical) {
        const std::vector<int64_t> byteStrides = nb::cast<std::vector<int64_t>>(value.attr("_array").attr("strides"));
        const int64_t elementSize = nb::cast<int64_t>(value.attr("element_layout").attr("size"));
        for (size_t index = 0; index < carrierShape.size(); ++index)
            strides.push_back(byteStrides[index] / elementSize);
    }
    const std::vector<int64_t> primalStrides = nb::cast<std::vector<int64_t>>(primal.attr("_strides"));
    strides.insert(strides.end(), primalStrides.begin(), primalStrides.end());
    nb::object projection =
        value.attr("_tangent_view")(suffix, nb::arg("shape") = shape, nb::arg("strides") = strides,
                                    nb::arg("offset") = primal.attr("_offset"), nb::arg("access") = "read");
    return projection.attr("to_numpy")();
}

} // namespace

nb::dict PythonPullback::applyGroupedWithOptions(const nb::object &cotangent, const nb::object &gradientGroups,
                                                 const nb::object &cotangentGroups, const nb::object &carrierShape,
                                                 bool logical, const VernonPullbackApplyOptions *options) {
    const size_t cotangentGroupCount = nb::len(cotangentGroups);
    nb::object nativeCotangent = nb::none();
    if (!cotangent.is_none()) {
        nb::dict supplied;
        if (nb::isinstance<nb::dict>(cotangent))
            supplied = nb::cast<nb::dict>(cotangent);
        else {
            if (cotangentGroupCount != 1)
                throw std::invalid_argument("pullback requires exactly one cotangent per declared output path");
            nb::object group = cotangentGroups.attr("__getitem__")(0);
            supplied[nb::str(nb::cast<std::string>(group.attr("declared_path")).c_str())] = cotangent;
        }
        if (supplied.size() != cotangentGroupCount)
            throw std::invalid_argument("pullback requires exactly one cotangent per declared output path");

        const std::vector<uint64_t> carrier = nb::cast<std::vector<uint64_t>>(carrierShape);
        nb::object tensorStorageType = nb::module_::import_("vernon_dsl._runtime.resources").attr("TensorStorage");
        nb::object tangentLayoutType = nb::module_::import_("vernon_dsl.host_values").attr("TangentLayout");
        nb::dict leaves;
        for (nb::handle group : nb::iter(cotangentGroups)) {
            const std::string declaredPath = nb::cast<std::string>(group.attr("declared_path"));
            nb::str declaredKey(declaredPath.c_str());
            if (!supplied.contains(declaredKey))
                throw std::invalid_argument("pullback requires exactly one cotangent per declared output path");
            nb::object value = nb::borrow<nb::object>(supplied[declaredKey]);
            const std::vector<std::string> paths = derivativeGroupLeaves(group);
            const bool storage = nb::isinstance(value, tensorStorageType);
            const bool aggregate = storage && nb::isinstance(value.attr("element_layout"), tangentLayoutType);
            for (const std::string &path : paths) {
                nb::object leaf;
                if (storage)
                    leaf = storageCotangentLeaf(value, group, path, carrier, bindings, logical, aggregate);
                else if (paths.size() == 1)
                    leaf = value;
                else if (nb::isinstance<nb::dict>(value))
                    leaf = nb::borrow<nb::object>(nb::cast<nb::dict>(value)[nb::str(path.c_str())]);
                else
                    throw std::invalid_argument("aggregate pullback cotangent must provide every leaf");
                leaves[nb::str(path.c_str())] = std::move(leaf);
            }
        }
        if (leaves.size() == 1) {
            for (auto item : leaves) {
                nativeCotangent = nb::borrow<nb::object>(item.second);
                break;
            }
        } else {
            nativeCotangent = std::move(leaves);
        }
    }

    nb::dict leafResults = applyImpl(nativeCotangent, logical, options);
    nb::dict grouped;
    for (nb::handle group : nb::iter(gradientGroups)) {
        const std::vector<std::string> paths = derivativeGroupLeaves(group);
        if (paths.empty())
            throw std::runtime_error("gradient group has no derivative leaves");
        nb::object first = nb::borrow<nb::object>(leafResults[nb::str(paths.front().c_str())]);
        for (size_t index = 1; index < paths.size(); ++index)
            if (leafResults[nb::str(paths[index].c_str())].ptr() != first.ptr())
                throw std::runtime_error("gradient leaves did not materialize into one owner");
        const std::string declaredPath = nb::cast<std::string>(group.attr("declared_path"));
        grouped[nb::str(declaredPath.c_str())] = std::move(first);
    }
    return grouped;
}

PythonAdMetadata adInputLeafMetadata(VernonLoadedPipeline *pipeline, const PipelineParameterMetadata &parameter,
                                     size_t leafIndex, VernonPipelineValueLeafView *reflected) {
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
            result.path += nativeStringView(component.field);
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
            const std::string field = nativeStringView(component.field);
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
