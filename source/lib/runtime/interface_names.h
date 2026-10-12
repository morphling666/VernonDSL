#ifndef VERNON_RUNTIME_INTERFACE_NAMES_H
#define VERNON_RUNTIME_INTERFACE_NAMES_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace vernon::runtime {

// SPIR-V lowering names shader interfaces with this spelling, and OpenGL-family Runtime binds named uniforms and
// textures by the same spelling, so both sides must share it. Binding-group leaves make it non-trivial: `m.albedo`
// is spelled `m_albedo`.
inline std::string sanitizeInterfaceName(std::string_view sourceName, std::string_view fallback, bool varying) {
    const auto isAlpha = [](char character) {
        return (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z');
    };
    const auto isAlnum = [&](char character) { return isAlpha(character) || (character >= '0' && character <= '9'); };
    const std::string_view input = sourceName.empty() ? fallback : sourceName;
    std::string result;
    bool uppercaseNext = varying && input.find('_') != std::string_view::npos;
    for (char character : input) {
        if (isAlnum(character)) {
            result.push_back(uppercaseNext && character >= 'a' && character <= 'z'
                                 ? static_cast<char>(character - 'a' + 'A')
                                 : character);
            uppercaseNext = false;
        } else if (character == '_' && varying) {
            uppercaseNext = true;
        } else {
            result.push_back('_');
            uppercaseNext = false;
        }
    }
    if (result.empty())
        result = "interface_value";
    if (!isAlpha(result.front()) && result.front() != '_')
        result.insert(result.begin(), '_');
    return result;
}

// GLSL 3.30 cannot declare a uniform block binding, so OpenGL-family Runtime finds each block by this name after
// linking and assigns the binding itself. Deriving the name from the binding, which SPIR-V and the manifest both
// carry, keeps it independent of source names and of interface renaming.
inline std::string openGLUniformBlockName(uint32_t binding) {
    return "vernon_uniform_binding_" + std::to_string(binding);
}

// Scalars, 1-4 element vectors, and 2-4 by 2-4 f32 matrices are the only Tensor shapes OpenGL Runtime can upload as
// native uniforms. The graphics binding planner places every other uniform Tensor in a uniform block on every target,
// so the planner and OpenGL Runtime must share this rule.
inline bool isNativeUniformShape(const uint64_t *shape, size_t rank, bool f32Elements) {
    if (rank == 0)
        return true;
    if (rank == 1)
        return shape[0] >= 1 && shape[0] <= 4;
    return rank == 2 && f32Elements && shape[0] >= 2 && shape[0] <= 4 && shape[1] >= 2 && shape[1] <= 4;
}

// Each graphics stage's inline uniforms form one root-constant cbuffer at this register, which the HLSL compiler
// declares and D3D12 Runtime binds as 32-bit root constants.
inline constexpr uint32_t kDirectXRootConstantRegister = 0;
inline constexpr uint32_t kDirectXRootConstantRegisterSpace = 0;

// Descriptor set N uses register space N + 1, leaving space 0 to the root constants alone: otherwise the uniform block
// planned at set 0 binding 0 would take b0 space0 as well.
inline constexpr uint32_t kDirectXDescriptorSetRegisterSpaceOffset = kDirectXRootConstantRegisterSpace + 1;

} // namespace vernon::runtime

#endif
