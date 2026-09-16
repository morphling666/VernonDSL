#ifndef VERNON_RUNTIME_TRANSPORT_NODE_H
#define VERNON_RUNTIME_TRANSPORT_NODE_H

#include <cstdint>
#include <string>
#include <vector>

namespace vernon::runtime {

enum class TransportNodeKind {
    Scalar,
    Product,
    Array,
};

struct TransportNode {
    TransportNodeKind kind{TransportNodeKind::Scalar};
    std::string representation;
    uint64_t offset{};
    uint64_t size{};
    uint64_t alignment{1};
    std::vector<uint64_t> shape;
    std::vector<uint64_t> byteStrides;
    std::vector<TransportNode> children;
};

} // namespace vernon::runtime

#endif
