#include "program_shape_resolver.h"

#include "runtime/runtime_state.h"

namespace vernon::runtime::ad {
using program_execution::ProgramValueState;
namespace {

template <typename Shape> std::string formatShape(const Shape &shape) {
    std::string result = "[";
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index)
            result += ", ";
        result += std::to_string(shape[index]);
    }
    return result + "]";
}

class Resolver {
public:
    Resolver(const program::Program &program, const program::ResolvedExecutionPlan *topology,
             std::vector<ProgramValueState> &values)
        : program_(program), topology_(topology), values_(values) {}

    bool resolve(std::string &error) {
        for (const program::Value &value : program_.values) {
            const shape::DeclaredShape &declared = declaredShape(value);
            if (value.id < values_.size() && values_[value.id].concreteShape &&
                !shape::matches(declared, *values_[value.id].concreteShape))
                return error = "bound shape for Program value " + std::to_string(value.id) +
                               " conflicts with the declared Program shape: value '" + value.name + "' bound " +
                               formatShape(*values_[value.id].concreteShape) + ", declared " + formatShape(value.shape),
                       false;
            if (topology_ && value.id < topology_->preparedValueShapes.size()) {
                const std::optional<shape::ConcreteShape> &concrete = topology_->preparedValueShapes[value.id];
                if (concrete && !bind(value.id, *concrete, "Program declaration", error))
                    return false;
            } else if (const std::optional<shape::ConcreteShape> concrete = shape::concrete(declared);
                       concrete && !bind(value.id, *concrete, "Program declaration", error)) {
                return false;
            }
        }
        return propagateStorageAliases();
    }

private:
    const shape::DeclaredShape &declaredShape(const program::Value &value) const {
        if (topology_ && value.id < topology_->declaredValueShapes.size())
            return topology_->declaredValueShapes[value.id];
        fallbackDeclaredShape_ = shape::decodeRuntimeContractShape(value.shape);
        return fallbackDeclaredShape_;
    }

    bool propagateStorageAliases() {
        for (const program::Storage &storage : program_.storages) {
            const shape::ConcreteShape *resolved = nullptr;
            for (const program::Value &value : program_.values)
                if (value.storage && *value.storage == storage.id && value.id < values_.size() &&
                    values_[value.id].concreteShape) {
                    resolved = &*values_[value.id].concreteShape;
                    break;
                }
            if (!resolved)
                continue;
            for (const program::Value &value : program_.values)
                if (value.storage && *value.storage == storage.id && value.id < values_.size() &&
                    !values_[value.id].concreteShape && shape::matches(declaredShape(value), *resolved))
                    values_[value.id].concreteShape = *resolved;
        }
        return true;
    }

    bool bind(uint32_t value, const shape::ConcreteShape &concrete, const char *source, std::string &error) {
        if (value >= program_.values.size() || value >= values_.size())
            return error = std::string(source) + " refers to an invalid Program value", false;
        if (!shape::matches(declaredShape(program_.values[value]), concrete)) {
            return error = std::string(source) + " for value '" + program_.values[value].name + "' (" +
                           program_.values[value].type +
                           (program_.values[value].layout ? ", canonical layout" : ", no canonical layout") +
                           ") has shape " + formatShape(concrete) + " but the Program declares " +
                           formatShape(program_.values[value].shape),
                   false;
        }
        if (values_[value].concreteShape && *values_[value].concreteShape != concrete)
            return error = std::string(source) + " conflicts with another concrete Program shape", false;
        values_[value].concreteShape = concrete;
        return true;
    }

    const program::Program &program_;
    const program::ResolvedExecutionPlan *topology_;
    std::vector<ProgramValueState> &values_;
    mutable shape::DeclaredShape fallbackDeclaredShape_;
};

} // namespace

bool resolveProgramShapes(const program::Program &program, const program::ResolvedExecutionPlan *topology,
                          std::vector<ProgramValueState> &values, std::string &error) {
    return Resolver(program, topology, values).resolve(error);
}

} // namespace vernon::runtime::ad
