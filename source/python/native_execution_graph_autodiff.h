#ifndef VERNON_PYTHON_NATIVE_EXECUTION_GRAPH_AUTODIFF_H
#define VERNON_PYTHON_NATIVE_EXECUTION_GRAPH_AUTODIFF_H

#include "VernonExecutionGraph.h"

#include <nanobind/nanobind.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace nb = nanobind;

struct PythonGraphCallbackState {
    struct OperationFrame {
        std::exception_ptr exception;
        uint64_t callbacks{};
        bool reverse{};
    };

    std::exception_ptr exception;
    std::vector<OperationFrame> operationFrames;

    void beginOperation(bool reverse) { operationFrames.push_back({nullptr, 0, reverse}); }
    OperationFrame endOperation() {
        OperationFrame result = std::move(operationFrames.back());
        operationFrames.pop_back();
        return result;
    }
    void recordReverseCallback() {
        if (!operationFrames.empty() && operationFrames.back().reverse)
            ++operationFrames.back().callbacks;
    }
    void captureException(std::exception_ptr value) {
        std::exception_ptr &destination = operationFrames.empty() ? exception : operationFrames.back().exception;
        if (!destination)
            destination = std::move(value);
    }
};

std::shared_ptr<vernon::execution::GraphCheckpointResource> makePythonCheckpointResource(nb::object value);
std::shared_ptr<vernon::execution::GraphAutodiffValue>
makePythonGraphAutodiffValue(nb::object value, std::shared_ptr<PythonGraphCallbackState> callbackState = {});
nb::object pythonGraphAutodiffValue(const std::shared_ptr<vernon::execution::GraphAutodiffValue> &value);
std::unique_ptr<vernon::execution::PassPullback>
makePythonPassPullback(nb::object value, std::shared_ptr<PythonGraphCallbackState> callbackState);

struct PythonGraphBackwardSubmission {
    PythonGraphBackwardSubmission(std::shared_ptr<vernon::execution::GraphBackwardSubmission> value,
                                  std::exception_ptr exception);

    void wait();
    uint32_t state() const;
    nb::dict gradients() const;

    std::shared_ptr<vernon::execution::GraphBackwardSubmission> submission;
    std::exception_ptr exception;
};

struct PythonGraphPullback {
    PythonGraphPullback(std::shared_ptr<vernon::execution::GraphPullback> value,
                        std::shared_ptr<PythonGraphCallbackState> callbackState,
                        std::vector<std::string> objectiveNames,
                        std::shared_ptr<std::vector<nb::object>> retainedOwners);

    std::unique_ptr<PythonGraphBackwardSubmission> submit(const nb::object &cotangent);
    uint32_t forwardState() const;
    void waitForward();
    uint64_t reversePythonCallbackCount() const;

    std::shared_ptr<vernon::execution::GraphPullback> pullback;
    std::shared_ptr<PythonGraphCallbackState> callbackState;
    std::vector<std::string> objectiveNames;
    std::shared_ptr<std::vector<nb::object>> owners;
    uint64_t lastReversePythonCallbackCount{};
};

void bindExecutionGraphAutodiff(nb::module_ &module);

#endif
