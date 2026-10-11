#ifndef VERNON_RUNTIME_INTERFACE_NAMES_H
#define VERNON_RUNTIME_INTERFACE_NAMES_H

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

} // namespace vernon::runtime

#endif
