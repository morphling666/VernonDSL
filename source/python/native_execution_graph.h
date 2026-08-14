#ifndef VERNON_PYTHON_NATIVE_EXECUTION_GRAPH_H
#define VERNON_PYTHON_NATIVE_EXECUTION_GRAPH_H

#include "native_rhi.h"

#include <nanobind/nanobind.h>

#include <memory>

void bindNativeExecutionGraph(nanobind::module_ &module);
nanobind::object createNativeExecutionGraph(std::shared_ptr<RhiHostState> host);

#endif
