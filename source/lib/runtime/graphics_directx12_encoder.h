#ifndef VERNON_RUNTIME_GRAPHICS_DIRECTX12_ENCODER_H
#define VERNON_RUNTIME_GRAPHICS_DIRECTX12_ENCODER_H

#include "graphics_invocation_planner.h"

#include <string>

namespace vernon::runtime {

struct DirectX12PipelineState;

VernonStatus encodeAndSubmitDirectX12Graphics(VernonRuntimeContext &context, DirectX12PipelineState &state,
                                              const Variant &variant, const VernonPipelineInvocation &invocation,
                                              const PlannedGraphicsInvocation &plan, std::string &error);

} // namespace vernon::runtime

#endif
