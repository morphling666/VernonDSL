#ifndef VERNON_EXTERNAL_ENGINE_CANONICAL_PROGRAM_CALL_H
#define VERNON_EXTERNAL_ENGINE_CANONICAL_PROGRAM_CALL_H

#include "VernonRuntime.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace vernon_external_engine {

struct ProgramGraphicsInvocation {
    const VernonRenderPass *renderPass{};
    const VernonDrawCommand *drawCommand{};
    const VernonDynamicState *dynamicState{};
    uint64_t revision{};
};

inline std::string programRuntimeError(VernonRuntimeContext *runtime) {
    const VernonStringView error = vernonRuntimeGetLastError(runtime);
    return error.data ? std::string(error.data, error.size) : std::string();
}

inline bool invokeProgram(VernonRuntimeContext *runtime, vernon::runtime::ProgramExecutable &executable,
                          vernon::runtime::ProgramInstance &instance, const VernonProgramArgument *arguments,
                          size_t argumentCount, const ProgramGraphicsInvocation *graphics, std::string &error) {
    try {
        vernon::runtime::ProgramInvocation invocation = instance.begin();
        // These example arguments keep their descriptor and backing address stable for the lifetime of the instance.
        // Mutable host contents are read through that stable binding on every invocation.
        std::vector<VernonProgramArgument> boundaryBindings(arguments, arguments + argumentCount);
        const size_t parameterCount = vernonRuntimeProgramExecutableGetParameterCount(executable.get());
        std::vector<VernonProgramParameterView> parameters(parameterCount);
        for (size_t index = 0; index < parameterCount; ++index)
            if (vernonRuntimeProgramExecutableGetParameterByIndex(executable.get(), index, &parameters[index]) !=
                VERNON_STATUS_OK)
                throw std::runtime_error("cannot reflect canonical Program boundaries");
        boundaryBindings.reserve(parameterCount);
        for (const VernonProgramParameterView &parameter : parameters) {
            if (std::any_of(boundaryBindings.begin(), boundaryBindings.end(),
                            [&](const VernonProgramArgument &argument) { return argument.slot == parameter.slot; }))
                continue;
            const auto source =
                std::find_if(boundaryBindings.begin(), boundaryBindings.end(), [&](const VernonProgramArgument &item) {
                    const auto sourceParameter = std::find_if(
                        parameters.begin(), parameters.end(),
                        [&](const VernonProgramParameterView &candidate) { return candidate.slot == item.slot; });
                    return sourceParameter != parameters.end() && sourceParameter->name.size == parameter.name.size &&
                           std::memcmp(sourceParameter->name.data, parameter.name.data, parameter.name.size) == 0;
                });
            if (source != boundaryBindings.end()) {
                VernonProgramArgument alias = *source;
                alias.slot = parameter.slot;
                boundaryBindings.push_back(alias);
            }
        }
        for (const VernonProgramArgument &argument : boundaryBindings) {
            const std::array<uint32_t, 2> tokenBytes{0, argument.slot};
            const VernonProgramBindingToken token{sizeof(token), tokenBytes.data(), sizeof(tokenBytes)};
            invocation.bind(token, argument);
        }
        if (graphics) {
            VernonProgramGraphicsControlsView controls{};
            controls.struct_size = sizeof(controls);
            if (vernonRuntimeProgramExecutableGetGraphicsControlsByIndex(executable.get(), 0, &controls) !=
                VERNON_STATUS_OK)
                throw std::runtime_error("Program has no canonical graphics controls");
            const std::array<uint64_t, 3> renderTokenBytes{1, controls.render_pass_control, graphics->revision};
            const std::array<uint64_t, 3> drawTokenBytes{2, controls.draw_command_control, graphics->revision};
            const std::array<uint64_t, 3> dynamicTokenBytes{3, controls.dynamic_state_control, graphics->revision};
            const VernonProgramBindingToken renderToken{sizeof(renderToken), renderTokenBytes.data(),
                                                        sizeof(renderTokenBytes)};
            const VernonProgramBindingToken drawToken{sizeof(drawToken), drawTokenBytes.data(), sizeof(drawTokenBytes)};
            const VernonProgramBindingToken dynamicToken{sizeof(dynamicToken), dynamicTokenBytes.data(),
                                                         sizeof(dynamicTokenBytes)};
            if (!graphics->renderPass || !graphics->drawCommand || !graphics->dynamicState)
                throw std::invalid_argument("canonical graphics invocation controls are incomplete");
            invocation.bindRenderPass(controls.render_pass_control, renderToken, *graphics->renderPass)
                .bindDrawCommand(controls.draw_command_control, drawToken, *graphics->drawCommand)
                .bindDynamicState(controls.dynamic_state_control, dynamicToken, *graphics->dynamicState);
        }
        invocation.forward(false);
        return true;
    } catch (const std::exception &exception) {
        error = programRuntimeError(runtime);
        if (error.empty())
            error = exception.what();
        return false;
    }
}

} // namespace vernon_external_engine

#endif
