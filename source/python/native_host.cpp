#include "native_host.h"

#include "VernonRuntime.h"
#include "compiler_python_bridge.h"

#include <nanobind/stl/string.h>

#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace {

VernonStringView viewOf(const std::string &value) { return {value.data(), value.size()}; }

std::string stringOf(VernonStringView value) {
    return value.data ? std::string(value.data, value.size) : std::string();
}

nb::dict compileHost(const std::string &ir, const std::string &triple, const std::string &cpu,
                     const std::string &features) {
    VernonPythonHostObject *result =
        vernonCompilerCompilePythonHost(viewOf(ir), viewOf(triple), viewOf(cpu), viewOf(features));
    if (!result)
        throw std::bad_alloc();
    const VernonPythonHostObjectView view = vernonCompilerGetPythonHostObjectView(result);
    const std::string diagnostics = stringOf(view.diagnostics);
    nb::dict compiled;
    if (view.status == VERNON_STATUS_OK) {
        compiled["object"] = nb::bytes(view.object.data, view.object.size);
        compiled["triple"] = stringOf(view.triple);
        compiled["optimized_ir"] = stringOf(view.optimized_ir);
    }
    vernonCompilerDestroyPythonHostObject(result);
    if (view.status != VERNON_STATUS_OK)
        throw std::invalid_argument(diagnostics.empty() ? "Host compilation failed" : diagnostics);
    return compiled;
}

} // namespace

void bindNativeHost(nb::module_ &module) {
    module.def("_host_runtime_symbols", []() {
        nb::dict symbols;
        symbols["vernonRuntimeHostListReserve"] = reinterpret_cast<uintptr_t>(&vernonRuntimeHostListReserve);
        symbols["vernonRuntimeHostListRelease"] = reinterpret_cast<uintptr_t>(&vernonRuntimeHostListRelease);
        symbols["vernonRuntimeHostDefaultAllocator"] = reinterpret_cast<uintptr_t>(&vernonRuntimeHostDefaultAllocator);
        symbols["vernonRuntimeHostDictFind"] = reinterpret_cast<uintptr_t>(&vernonRuntimeHostDictFind);
        symbols["vernonRuntimeHostDictInsert"] = reinterpret_cast<uintptr_t>(&vernonRuntimeHostDictInsert);
        symbols["vernonRuntimeHostDictRemove"] = reinterpret_cast<uintptr_t>(&vernonRuntimeHostDictRemove);
        symbols["vernonRuntimeHostDictClear"] = reinterpret_cast<uintptr_t>(&vernonRuntimeHostDictClear);
        symbols["vernonRuntimeHostDictRelease"] = reinterpret_cast<uintptr_t>(&vernonRuntimeHostDictRelease);
        symbols["vernonRuntimeHostStrAppend"] = reinterpret_cast<uintptr_t>(&vernonRuntimeHostStrAppend);
        symbols["vernonRuntimeHostStrAppendInteger"] = reinterpret_cast<uintptr_t>(&vernonRuntimeHostStrAppendInteger);
        symbols["vernonRuntimeHostStrAppendFloat"] = reinterpret_cast<uintptr_t>(&vernonRuntimeHostStrAppendFloat);
        symbols["vernonRuntimeHostStorageAllocate"] = reinterpret_cast<uintptr_t>(&vernonRuntimeHostStorageAllocate);
        symbols["vernonRuntimeHostStorageRelease"] = reinterpret_cast<uintptr_t>(&vernonRuntimeHostStorageRelease);
        symbols["vernonRuntimeHostStorageFill"] = reinterpret_cast<uintptr_t>(&vernonRuntimeHostStorageFill);
        return symbols;
    });
    module.def("_default_host_triple", []() { return stringOf(vernonCompilerDefaultHostTriple()); });
    module.def("_compile_host", &compileHost, nb::arg("ir"), nb::arg("triple") = "", nb::arg("cpu") = "",
               nb::arg("features") = "");
}
