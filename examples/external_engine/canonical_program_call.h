#ifndef VERNON_EXTERNAL_ENGINE_CANONICAL_PROGRAM_CALL_H
#define VERNON_EXTERNAL_ENGINE_CANONICAL_PROGRAM_CALL_H

#include "VernonRuntime.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace vernon_external_engine {

struct ProgramGraphicsInvocation {
    const VernonRenderPass *renderPass{};
    const VernonDrawCommand *drawCommand{};
    const VernonDynamicState *dynamicState{};
    uint64_t revision{};
};

inline std::string programRuntimeError(VernonRuntimeContext *runtime, vernon::RuntimeError runtimeError) {
    const VernonStringView error = vernonRuntimeGetLastError(runtime);
    if (error.data && error.size)
        return std::string(error.data, error.size);
    std::array<char, 192> diagnostic{};
    const size_t size = vernon::renderEmergencyDiagnostic(runtimeError, diagnostic.data(), diagnostic.size());
    return std::string(diagnostic.data(), std::min(size, diagnostic.size() - 1));
}

inline vernon::Result<void, vernon::RuntimeError>
invokeProgram(vernon::runtime::ProgramExecutable &executable, vernon::runtime::ProgramInstance &instance,
              const VernonProgramArgument *arguments, size_t argumentCount, const ProgramGraphicsInvocation *graphics) {
    auto invocationResult = instance.begin();
    if (invocationResult.isErr())
        return vernon::Result<void, vernon::RuntimeError>{vernon::err(invocationResult.error())};
    vernon::runtime::ProgramInvocation invocation = std::move(invocationResult).value();

    // These example arguments keep their descriptor and backing address stable for the lifetime of the instance.
    // Mutable host contents are read through that stable binding on every invocation.
    std::vector<VernonProgramArgument> boundaryBindings(arguments, arguments + argumentCount);
    const size_t parameterCount = vernonRuntimeProgramExecutableGetParameterCount(executable.get());
    std::vector<VernonProgramParameterView> parameters(parameterCount);
    for (size_t index = 0; index < parameterCount; ++index) {
        const VernonStatus status =
            vernonRuntimeProgramExecutableGetParameterByIndex(executable.get(), index, &parameters[index]);
        if (status != VERNON_STATUS_OK)
            return vernon::Result<void, vernon::RuntimeError>{
                vernon::err(vernon::runtimeErrorFromStatus(status, {"invokeProgram.reflectParameters"}))};
    }
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
        auto bindResult = invocation.bind(token, argument);
        if (bindResult.isErr())
            return vernon::Result<void, vernon::RuntimeError>{vernon::err(bindResult.error())};
    }
    if (graphics) {
        if (!graphics->renderPass || !graphics->drawCommand || !graphics->dynamicState)
            return vernon::Result<void, vernon::RuntimeError>{vernon::err(
                vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument, {"invokeProgram.graphicsControls"}})};
        VernonProgramGraphicsControlsView controls{};
        controls.struct_size = sizeof(controls);
        const VernonStatus status =
            vernonRuntimeProgramExecutableGetGraphicsControlsByIndex(executable.get(), 0, &controls);
        if (status != VERNON_STATUS_OK)
            return vernon::Result<void, vernon::RuntimeError>{
                vernon::err(vernon::runtimeErrorFromStatus(status, {"invokeProgram.reflectGraphicsControls"}))};
        const std::array<uint64_t, 3> renderTokenBytes{1, controls.render_pass_control, graphics->revision};
        const std::array<uint64_t, 3> drawTokenBytes{2, controls.draw_command_control, graphics->revision};
        const std::array<uint64_t, 3> dynamicTokenBytes{3, controls.dynamic_state_control, graphics->revision};
        const VernonProgramBindingToken renderToken{sizeof(renderToken), renderTokenBytes.data(),
                                                    sizeof(renderTokenBytes)};
        const VernonProgramBindingToken drawToken{sizeof(drawToken), drawTokenBytes.data(), sizeof(drawTokenBytes)};
        const VernonProgramBindingToken dynamicToken{sizeof(dynamicToken), dynamicTokenBytes.data(),
                                                     sizeof(dynamicTokenBytes)};
        auto renderPassResult =
            invocation.bindRenderPass(controls.render_pass_control, renderToken, *graphics->renderPass);
        if (renderPassResult.isErr())
            return vernon::Result<void, vernon::RuntimeError>{vernon::err(renderPassResult.error())};
        auto drawCommandResult =
            invocation.bindDrawCommand(controls.draw_command_control, drawToken, *graphics->drawCommand);
        if (drawCommandResult.isErr())
            return vernon::Result<void, vernon::RuntimeError>{vernon::err(drawCommandResult.error())};
        auto dynamicStateResult =
            invocation.bindDynamicState(controls.dynamic_state_control, dynamicToken, *graphics->dynamicState);
        if (dynamicStateResult.isErr())
            return vernon::Result<void, vernon::RuntimeError>{vernon::err(dynamicStateResult.error())};
    }
    auto executeResult = invocation.execute(false);
    if (executeResult.isErr())
        return vernon::Result<void, vernon::RuntimeError>{vernon::err(executeResult.error())};
    auto commitResult = invocation.commit();
    if (commitResult.isErr())
        return vernon::Result<void, vernon::RuntimeError>{vernon::err(commitResult.error())};
    vernon::runtime::Pullback pullback = std::move(commitResult).value();
    (void)pullback;
    auto telemetryResult = instance.telemetry();
    if (telemetryResult.isErr())
        return vernon::Result<void, vernon::RuntimeError>{vernon::err(telemetryResult.error())};
    (void)telemetryResult.value();
    return vernon::Result<void, vernon::RuntimeError>{vernon::ok()};
}

} // namespace vernon_external_engine

#endif
