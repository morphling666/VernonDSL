#include "native_program_autodiff.h"

#include <algorithm>
#include <cctype>

namespace {

std::vector<std::string> derivativeGroupLeaves(const nb::handle &group) {
    return nb::cast<std::vector<std::string>>(group.attr("leaf_paths"));
}

} // namespace

nb::dict PythonPullback::applyGroupedWithOptions(const nb::object &cotangent, const nb::object &gradientGroups,
                                                 const nb::object &cotangentGroups, const nb::object &carrierShape,
                                                 bool logical, const VernonPullbackApplyOptions *options) {
    (void)carrierShape;
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

        for (nb::handle group : nb::iter(cotangentGroups)) {
            const std::string declaredPath = nb::cast<std::string>(group.attr("declared_path"));
            nb::str declaredKey(declaredPath.c_str());
            if (!supplied.contains(declaredKey))
                throw std::invalid_argument("pullback requires exactly one cotangent per declared output path");
        }
        nativeCotangent = std::move(supplied);
    }

    nb::dict leafResults = applyImpl(nativeCotangent, logical, options);
    nb::dict grouped;
    for (nb::handle group : nb::iter(gradientGroups)) {
        const std::vector<std::string> paths = derivativeGroupLeaves(group);
        if (paths.empty())
            throw std::runtime_error("gradient group has no derivative leaves");
        nb::object first = nb::borrow<nb::object>(leafResults[nb::str(paths.front().c_str())]);
        bool sharedOwner = true;
        for (size_t index = 1; index < paths.size(); ++index)
            if (leafResults[nb::str(paths[index].c_str())].ptr() != first.ptr())
                sharedOwner = false;
        if (!sharedOwner) {
            for (const std::string &path : paths)
                grouped[nb::str(path.c_str())] = nb::borrow<nb::object>(leafResults[nb::str(path.c_str())]);
            continue;
        }
        const std::string declaredPath = nb::cast<std::string>(group.attr("declared_path"));
        grouped[nb::str(declaredPath.c_str())] = std::move(first);
    }
    return grouped;
}

PythonAdMetadata adInputLeafMetadata(VernonProgramExecutable *executable, const ProgramParameterMetadata &parameter,
                                     size_t leafIndex, VernonProgramValueLeafView *reflected) {
    VernonProgramValueLeafView leaf{};
    leaf.struct_size = sizeof(leaf);
    const VernonStringView parameterName{parameter.name.data(), parameter.name.size()};
    if (vernonRuntimeProgramExecutableGetParameterValueLeaf(executable, parameterName, leafIndex, &leaf) !=
        VERNON_STATUS_OK)
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
    if (parameter.kind == VERNON_PROGRAM_TENSOR) {
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

nb::object resolveProgramInputLeaf(const nb::dict &inputs, const std::string &leafPath) {
    std::string root;
    nb::list keys(inputs.attr("keys")());
    for (size_t index = 0; index < keys.size(); ++index) {
        const std::string key = nb::cast<std::string>(nb::str(keys[index]));
        if (leafPath == key || (leafPath.size() > key.size() && leafPath.compare(0, key.size(), key) == 0 &&
                                leafPath[key.size()] == '.')) {
            if (key.size() > root.size())
                root = key;
        }
    }
    if (root.empty())
        throw std::invalid_argument("missing Program autodiff input '" + leafPath + "'");
    nb::object value = nb::borrow<nb::object>(inputs[nb::str(root.c_str())]);
    if (nb::hasattr(value, "_native_host_array")) {
        nb::object array = value.attr("_native_host_array")();
        nb::object fields = array.attr("dtype").attr("fields");
        if (!fields.is_none()) {
            nb::str wrapped("__value");
            if (nb::cast<bool>(fields.attr("__contains__")(wrapped)))
                array = array.attr("__getitem__")(wrapped);
        }
        value = std::move(array);
        if (leafPath == root)
            return value;
    }
    if (leafPath == root)
        return value;
    const size_t tensorRank =
        nb::hasattr(value, "shape") ? nb::cast<std::vector<uint64_t>>(value.attr("shape")).size() : 0;
    size_t begin = root.size() + 1;
    while (begin <= leafPath.size()) {
        const size_t end = std::min(leafPath.find('.', begin), leafPath.size());
        const std::string component = leafPath.substr(begin, end - begin);
        const bool index =
            !component.empty() && std::all_of(component.begin(), component.end(),
                                              [](unsigned char character) { return std::isdigit(character) != 0; });
        if (index) {
            nb::object fields = nb::hasattr(value, "dtype") ? value.attr("dtype").attr("fields") : nb::none();
            nb::str key(component.c_str());
            if (!fields.is_none() && nb::cast<bool>(fields.attr("__contains__")(key)))
                value = value.attr("__getitem__")(key);
            else
                value = nb::module_::import_("numpy").attr("take")(value, std::stoul(component),
                                                                   nb::arg("axis") = tensorRank);
        } else if (nb::isinstance<nb::dict>(value)) {
            nb::dict mapping = nb::cast<nb::dict>(value);
            nb::str key(component.c_str());
            if (!mapping.contains(key))
                throw std::invalid_argument("autodiff Struct binding is missing field '" + component + "'");
            value = nb::borrow<nb::object>(mapping[key]);
        } else if (nb::hasattr(value, component.c_str())) {
            value = value.attr(component.c_str());
        } else if (nb::hasattr(value, "__getitem__")) {
            value = value.attr("__getitem__")(component);
        } else {
            throw std::invalid_argument("autodiff Struct binding has no field '" + component + "'");
        }
        if (end == leafPath.size())
            break;
        begin = end + 1;
    }
    return value;
}
