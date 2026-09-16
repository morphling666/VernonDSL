if(NOT DEFINED INPUT
   OR NOT DEFINED OUTPUT
   OR NOT DEFINED SYMBOL)
    message(FATAL_ERROR "EmbedBinary.cmake requires INPUT, OUTPUT, and SYMBOL")
endif()

file(
    READ
    "${INPUT}"
    _vernon_binary
    HEX)
string(
    REGEX
    REPLACE "(..)"
            "0x\\1,"
            _vernon_bytes
            "${_vernon_binary}")
file(
    WRITE "${OUTPUT}"
    "#pragma once\n#include <cstddef>\n#include <cstdint>\ninline constexpr std::uint8_t ${SYMBOL}[] = {${_vernon_bytes}};\ninline constexpr std::size_t ${SYMBOL}_size = sizeof(${SYMBOL});\n"
)
