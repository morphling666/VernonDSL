#ifndef VERNON_RUNTIME_CONTENT_HASH_H
#define VERNON_RUNTIME_CONTENT_HASH_H

#include <cstddef>
#include <string>
#include <string_view>

namespace vernon::runtime {

bool isSha256Hex(std::string_view value);
std::string sha256Hex(const void *data, size_t size);

} // namespace vernon::runtime

#endif
