#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_PROGRAM_IMAGE_BINDING_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_PROGRAM_IMAGE_BINDING_H

#include "runtime/program_execution_manifest.h"

#include <array>

struct VernonRuntimeContext;

namespace vernon::runtime::program_execution {

struct BoundProgramImage {
    VernonRhiImageViewDescriptor view{};
    VernonRhiImageDescriptor image{};
    VernonRhiImage parent{};
};

bool resolveBorrowedProgramImage(VernonRuntimeContext &context, const program::Storage &storage,
                                 VernonRuntimeProviderResourceReference reference, BoundProgramImage &resolved,
                                 std::string &error);
bool materializeOwnedProgramImageDescriptor(const program::Storage &storage, const std::array<uint32_t, 3> &extent,
                                            VernonRhiImageDescriptor &descriptor, std::string &error);

} // namespace vernon::runtime::program_execution

#endif
