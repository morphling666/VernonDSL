#ifndef VERNON_RUNTIME_TENSOR_COPY_PLAN_H
#define VERNON_RUNTIME_TENSOR_COPY_PLAN_H

#include "VernonRuntime.h"

#include <string>
#include <vector>

namespace vernon::runtime {

struct ProgramTensorCopyRegion {
    size_t sourceOffset{};
    size_t destinationOffset{};
    size_t size{};
};

bool planProgramTensorCopy(const VernonTensorView &source, const VernonTensorView &destination,
                           std::vector<ProgramTensorCopyRegion> &regions, std::string &error);

} // namespace vernon::runtime

#endif
