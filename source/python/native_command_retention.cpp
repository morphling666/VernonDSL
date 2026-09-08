#include "native_command_retention.h"

#include "execution_graph/execution_graph_internal.h"

#include <memory>
#include <utility>

namespace {

class PythonCommandCompletion final : public vernon::execution::detail::RhiCommandCompletion {
public:
    explicit PythonCommandCompletion(nb::object transaction) : transaction_(std::move(transaction)) {}

    void complete(bool succeeded, const vernon::execution::detail::RhiCommandDagExecutionStats &) override {
        nb::gil_scoped_acquire acquire;
        if (transaction_.is_none())
            return;
        transaction_.attr(succeeded ? "_commit_planned_state" : "_rollback_planned_state")();
        transaction_ = nb::none();
    }

private:
    nb::object transaction_;
};

} // namespace

void retainPythonCommandCompletion(vernon::execution::detail::RhiCommandPlanSink &sink, nb::object transaction) {
    sink.onCompletion(std::make_shared<PythonCommandCompletion>(std::move(transaction)));
}

void retainPythonObject(vernon::execution::detail::RhiCommandPlanSink &sink, nb::object value) {
    sink.retain(std::make_shared<nb::object>(std::move(value)));
}
