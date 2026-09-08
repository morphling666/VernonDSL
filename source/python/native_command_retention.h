#ifndef VERNON_PYTHON_NATIVE_COMMAND_RETENTION_H
#define VERNON_PYTHON_NATIVE_COMMAND_RETENTION_H

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace vernon::execution::detail {
class RhiCommandPlanSink;
}

void retainPythonCommandCompletion(vernon::execution::detail::RhiCommandPlanSink &sink, nb::object transaction);
void retainPythonObject(vernon::execution::detail::RhiCommandPlanSink &sink, nb::object value);

#endif
