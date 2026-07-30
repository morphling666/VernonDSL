#ifndef VERNON_DIALECT_VERNON_TRANSFORMS_VERNONSPIRVMARKERS_H
#define VERNON_DIALECT_VERNON_TRANSFORMS_VERNONSPIRVMARKERS_H

#include <cstdint>

namespace mlir::vernon {

// MLIR's SPIR-V dialect has no OpImageQuerySizeLod operation. These values
// make the temporary, valid-SPIR-V instruction sequence unambiguous after
// serialization.
inline constexpr uint32_t kImageQueryLodMarker = 0x564C4F44u;
inline constexpr uint32_t kImageQueryResultMarkerA = 0x56525341u;
inline constexpr uint32_t kImageQueryResultMarkerB = 0x56525342u;
inline constexpr char kImageQueryExpectedCountAttr[] = "vernon.image_query_size_lod_count";

} // namespace mlir::vernon

#endif
