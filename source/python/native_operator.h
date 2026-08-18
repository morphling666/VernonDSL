#ifndef VERNON_PYTHON_NATIVE_OPERATOR_H
#define VERNON_PYTHON_NATIVE_OPERATOR_H

#include <nanobind/nanobind.h>

#include <memory>

namespace nb = nanobind;

struct PipelineInvocationBuilder;
namespace vernon::execution::detail {
class RhiCommandPlanSink;
}

class PythonOperatorDagBuilder {
public:
    explicit PythonOperatorDagBuilder(PipelineInvocationBuilder &builder);
    ~PythonOperatorDagBuilder();
    PythonOperatorDagBuilder(const PythonOperatorDagBuilder &) = delete;
    PythonOperatorDagBuilder &operator=(const PythonOperatorDagBuilder &) = delete;

    void addElementwiseAdd(PipelineInvocationBuilder &builder, const nb::object &output, const nb::object &left,
                           const nb::object &right);
    void execute(vernon::execution::detail::RhiCommandPlanSink *sink = nullptr, nb::object retained = nb::none());

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

std::unique_ptr<PythonOperatorDagBuilder> createPythonOperatorDagBuilder(PipelineInvocationBuilder &builder);
void retainPythonCommandCompletion(vernon::execution::detail::RhiCommandPlanSink &sink, nb::object transaction);
void retainPythonObject(vernon::execution::detail::RhiCommandPlanSink &sink, nb::object value);

#endif
