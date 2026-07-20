#ifndef VERNON_RUNTIME_CONTENT_HASH_H
#define VERNON_RUNTIME_CONTENT_HASH_H

#include <cstddef>
#include <string>

namespace vernon::runtime {

std::string sha256Hex(const void *data, size_t size);

} // namespace vernon::runtime

#endif
