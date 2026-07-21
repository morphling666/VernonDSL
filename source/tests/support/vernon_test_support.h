#ifndef VERNON_TEST_SUPPORT_H
#define VERNON_TEST_SUPPORT_H

#include "vernon-c/Common.h"

#include <string>
#include <string_view>

namespace vernon::test {

inline std::string_view view(VernonStringView value) {
  return value.data ? std::string_view(value.data, value.size)
                    : std::string_view{};
}

inline bool contains(VernonStringView value, std::string_view expected) {
  return view(value).find(expected) != std::string_view::npos;
}

inline std::string text(VernonStringView value) {
  return std::string(view(value));
}

} // namespace vernon::test

#endif
