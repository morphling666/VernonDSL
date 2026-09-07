#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_PROGRAM_TENSOR_COPY_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_PROGRAM_TENSOR_COPY_H

#include "VernonRuntime.h"

#include <string>
#include <vector>

namespace vernon::runtime::program_execution {

struct ProgramTensorCopyRegion {
    size_t sourceOffset{};
    size_t destinationOffset{};
    size_t size{};
};

bool planProgramTensorCopy(const VernonTensorView &source, const VernonTensorView &destination,
                           std::vector<ProgramTensorCopyRegion> &regions, std::string &error);

} // namespace vernon::runtime::program_execution

#endif
