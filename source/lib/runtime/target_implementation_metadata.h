#ifndef VERNON_RUNTIME_TARGET_IMPLEMENTATION_METADATA_H
#define VERNON_RUNTIME_TARGET_IMPLEMENTATION_METADATA_H

#include "pipeline_bundle.h"

#include <nlohmann/json_fwd.hpp>

#include <string>
#include <string_view>
#include <vector>

namespace vernon::runtime {

bool parseTargetImplementationMetadata(std::string_view target, const nlohmann::json &metadata,
                                       std::vector<NativeResourceSlot> &nativeSlots, std::string &error);

} // namespace vernon::runtime

#endif
