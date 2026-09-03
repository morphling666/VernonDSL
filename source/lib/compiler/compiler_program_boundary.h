#ifndef VERNON_COMPILER_PROGRAM_BOUNDARY_H
#define VERNON_COMPILER_PROGRAM_BOUNDARY_H

#include "compiler_program_derivative.h"

#include "llvm/Support/JSON.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vernon::compiler {

enum class ProgramBoundaryDirection {
    Input,
    Output,
};

enum class ProgramBoundaryCategory {
    Value,
    StorageView,
    Texture,
    Sampler,
};

enum class ProgramBoundaryAccess {
    Read,
    Write,
    ReadWrite,
};

struct ProgramBoundaryOwnerId {
    bool storage{};
    int64_t id{};
};

struct ProgramBoundarySlotPlan {
    ProgramBoundaryIdentity identity;
    int64_t value{};
    ProgramBoundaryDirection direction{ProgramBoundaryDirection::Input};
    ProgramBoundaryCategory category{ProgramBoundaryCategory::Value};
    ProgramBoundaryAccess access{ProgramBoundaryAccess::Read};
    ProgramBoundaryOwnerId owner;
    std::string logicalType;
    llvm::json::Array outerShape;
    std::optional<int64_t> storage;
    std::optional<llvm::json::Object> storageDescriptor;
    std::optional<llvm::json::Object> valueLayout;
};

struct ProgramBoundaryPlan {
    std::vector<ProgramBoundarySlotPlan> slots;
};

bool planProgramBoundaries(const llvm::json::Object &signature, const llvm::json::Array &values,
                           const llvm::json::Array &storages, const llvm::json::Array &graphs,
                           ProgramBoundaryPlan &plan, std::string &error);

} // namespace vernon::compiler

#endif
