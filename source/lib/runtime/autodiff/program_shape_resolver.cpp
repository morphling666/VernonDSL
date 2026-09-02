#include "program_shape_resolver.h"

#include "program_value_arena.h"
#include "runtime/runtime_state.h"

namespace vernon::runtime::ad {
namespace {

class Resolver {
public:
    Resolver(const program::Program &program, const VernonPipelineTopology *topology,
             std::vector<ProgramHostValue> &values)
        : program_(program), topology_(topology), values_(values) {}

    bool resolve(std::string &error) {
        for (const program::Value &value : program_.values) {
            const shape::DeclaredShape declared = shape::decodeRuntimeContractShape(value.shape);
            if (value.id < values_.size() && values_[value.id].concreteShape &&
                !shape::matches(declared, *values_[value.id].concreteShape))
                return error = "bound shape conflicts with the declared Program shape", false;
            if (const std::optional<shape::ConcreteShape> concrete = shape::concrete(declared);
                concrete && !bind(value.id, *concrete, "Program declaration", error))
                return false;
        }
        if (!seedCompiledStages(error))
            return false;
        return propagateStorageAliases();
    }

private:
    bool seedCompiledStages(std::string &error) {
        if (!topology_)
            return true;
        for (const VernonResolvedProgramStage &stage : topology_->stages)
            for (size_t index = 0;
                 stage.pipeline && index < stage.bindings.size() && index < stage.pipeline->variant.parameters.size();
                 ++index) {
                const VernonProgramStageBinding &binding = stage.bindings[index];
                const Parameter &parameter = stage.pipeline->variant.parameters[index];
                if (binding.leaf || parameter.invocationCarrier || (binding.target && binding.target->viewTransform))
                    continue;
                const std::optional<shape::ConcreteShape> concrete =
                    shape::concrete(shape::decodeRuntimeContractShape(parameter.shape));
                if (!concrete)
                    continue;
                const uint32_t value = binding.value;
                if (value >= program_.values.size())
                    return error = "compiled stage binding refers to an invalid Program value", false;
                if (!bind(value, *concrete, "compiled stage binding", error))
                    return false;
            }
        return true;
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
                    !values_[value.id].concreteShape &&
                    shape::matches(shape::decodeRuntimeContractShape(value.shape), *resolved))
                    values_[value.id].concreteShape = *resolved;
        }
        return true;
    }

    bool bind(uint32_t value, const shape::ConcreteShape &concrete, const char *source, std::string &error) {
        if (value >= program_.values.size() || value >= values_.size())
            return error = std::string(source) + " refers to an invalid Program value", false;
        if (!shape::matches(shape::decodeRuntimeContractShape(program_.values[value].shape), concrete))
            return error = std::string(source) + " conflicts with the declared Program shape", false;
        if (values_[value].concreteShape && *values_[value].concreteShape != concrete)
            return error = std::string(source) + " conflicts with another concrete Program shape", false;
        values_[value].concreteShape = concrete;
        return true;
    }

    const program::Program &program_;
    const VernonPipelineTopology *topology_;
    std::vector<ProgramHostValue> &values_;
};

} // namespace

bool resolveProgramShapes(const program::Program &program, const VernonPipelineTopology *topology,
                          std::vector<ProgramHostValue> &values, std::string &error) {
    return Resolver(program, topology, values).resolve(error);
}

} // namespace vernon::runtime::ad
