#ifndef VERNON_PROGRAM_PLAN_TYPES_H
#define VERNON_PROGRAM_PLAN_TYPES_H

#include <array>
#include <optional>
#include <string_view>

namespace vernon::program_plan {

enum class TapeCarrier {
    TapeData,
    ReplaySegment,
    LaunchMetadata,
    ReplayStatus,
};

inline constexpr std::array<TapeCarrier, 4> tapeCarriers{
    TapeCarrier::TapeData,
    TapeCarrier::ReplaySegment,
    TapeCarrier::LaunchMetadata,
    TapeCarrier::ReplayStatus,
};

constexpr std::string_view tapeCarrierPlanName(TapeCarrier carrier) {
    switch (carrier) {
    case TapeCarrier::TapeData:
        return "tape_data";
    case TapeCarrier::ReplaySegment:
        return "replay_segment";
    case TapeCarrier::LaunchMetadata:
        return "launch_metadata";
    case TapeCarrier::ReplayStatus:
        return "replay_status";
    }
    return {};
}

constexpr std::string_view tapeCarrierRoleName(TapeCarrier carrier) {
    return carrier == TapeCarrier::TapeData ? "tape" : tapeCarrierPlanName(carrier);
}

inline std::optional<TapeCarrier> tapeCarrierFromPlanName(std::string_view name) {
    for (TapeCarrier carrier : tapeCarriers)
        if (name == tapeCarrierPlanName(carrier))
            return carrier;
    return std::nullopt;
}

inline std::optional<TapeCarrier> tapeCarrierFromRoleName(std::string_view name) {
    if (name == "tape")
        return TapeCarrier::TapeData;
    for (TapeCarrier carrier : tapeCarriers)
        if (name == tapeCarrierRoleName(carrier))
            return carrier;
    return std::nullopt;
}

} // namespace vernon::program_plan

#endif
