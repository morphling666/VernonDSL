#include "compiler_program_derivative.h"

#include "llvm/ADT/StringRef.h"

namespace vernon::compiler {
namespace {

bool isDerivative(ProgramBoundaryRole role) {
    return role == ProgramBoundaryRole::Cotangent || role == ProgramBoundaryRole::Gradient;
}

ProgramBoundaryRole primalRole(ProgramBoundaryRole derivative) {
    return derivative == ProgramBoundaryRole::Cotangent ? ProgramBoundaryRole::Output : ProgramBoundaryRole::Input;
}

} // namespace

bool planProgramDerivativeProjections(const std::vector<ProgramBoundaryIdentity> &boundaries,
                                      std::vector<ProgramDerivativeProjectionPlan> &projections, std::string &error) {
    projections.clear();
    for (const ProgramBoundaryIdentity &derivative : boundaries) {
        if (!isDerivative(derivative.role))
            continue;
        const ProgramBoundaryIdentity *primal = nullptr;
        for (const ProgramBoundaryIdentity &candidate : boundaries) {
            if (candidate.role != primalRole(derivative.role))
                continue;
            const bool exact = candidate.path == derivative.path;
            const bool prefix = derivative.path.size() > candidate.path.size() &&
                                llvm::StringRef(derivative.path).starts_with(candidate.path) &&
                                derivative.path[candidate.path.size()] == '.';
            if ((exact || prefix) && (!primal || candidate.path.size() > primal->path.size()))
                primal = &candidate;
        }
        if (!primal) {
            error = "canonical ProgramABI cannot project derivative boundary '" + derivative.path +
                    "' to a primal boundary";
            return false;
        }
        ProgramDerivativeProjectionPlan projection;
        projection.derivative = derivative;
        projection.primal = *primal;
        llvm::StringRef suffix(derivative.path);
        suffix = suffix.drop_front(primal->path.size());
        if (suffix.consume_front("."))
            while (!suffix.empty()) {
                auto [component, remainder] = suffix.split('.');
                if (component.empty()) {
                    error = "canonical ProgramABI derivative path contains an empty component";
                    return false;
                }
                uint32_t index = 0;
                if (!component.getAsInteger(10, index))
                    projection.valuePath.emplace_back(index);
                else
                    projection.valuePath.emplace_back(component.str());
                suffix = remainder;
            }
        projections.push_back(std::move(projection));
    }
    return true;
}

llvm::json::Array
serializeProgramDerivativeProjections(const std::vector<ProgramDerivativeProjectionPlan> &projections) {
    llvm::json::Array result;
    for (const ProgramDerivativeProjectionPlan &projection : projections) {
        llvm::json::Array valuePath;
        for (const ProgramValuePathComponent &component : projection.valuePath)
            if (const auto *index = std::get_if<uint32_t>(&component))
                valuePath.emplace_back(static_cast<int64_t>(*index));
            else
                valuePath.emplace_back(std::get<std::string>(component));
        result.emplace_back(llvm::json::Object{
            {"derivative",
             llvm::json::Object{{"slot", projection.derivative.slot}, {"path", projection.derivative.path}}},
            {"primal", llvm::json::Object{{"slot", projection.primal.slot}, {"path", projection.primal.path}}},
            {"value_path", std::move(valuePath)},
        });
    }
    return result;
}

} // namespace vernon::compiler
