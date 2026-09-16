#ifndef VERNON_PROGRAM_CAPABILITIES_H
#define VERNON_PROGRAM_CAPABILITIES_H

#include <array>
#include <string_view>

namespace vernon::program_capabilities {

enum class Id {
    ComputeForward,
    GraphicsForward,
    ComputeTextureBinding,
    GraphicsTextureSampling,
    ComputeSamplerBinding,
    ComputeVjp,
    GraphicsVjp,
    OpaqueResourceVjp,
    CpuF16,
    GpuF16,
};

struct Entry {
    Id id;
    std::string_view name;
    bool supported;
    std::string_view diagnosticCode;
    std::string_view diagnostic;
};

inline constexpr std::array<Entry, 10> matrix{{
    {Id::ComputeForward, "compute_forward", true, {}, {}},
    {Id::GraphicsForward, "graphics_forward", true, {}, {}},
    {Id::ComputeTextureBinding, "compute_texture_binding", true, {}, {}},
    {Id::GraphicsTextureSampling, "graphics_texture_sampling", true, {}, {}},
    {Id::ComputeSamplerBinding, "compute_sampler_binding", false, "PROGRAM_COMPUTE_SAMPLER_UNSUPPORTED",
     "compute Program sampler bindings are unsupported"},
    {Id::ComputeVjp, "compute_vjp", true, {}, {}},
    {Id::GraphicsVjp, "graphics_vjp", false, "PROGRAM_GRAPHICS_VJP_UNSUPPORTED", "graphics Program VJP is unsupported"},
    {Id::OpaqueResourceVjp, "opaque_resource_vjp", false, "PROGRAM_OPAQUE_RESOURCE_VJP_UNSUPPORTED",
     "Texture and Sampler are forward-only resources"},
    {Id::CpuF16, "cpu_f16", true, {}, {}},
    {Id::GpuF16, "gpu_f16", false, "PROGRAM_GPU_F16_UNSUPPORTED", "GPU targets do not currently support f16"},
}};

constexpr const Entry &get(Id id) {
    for (const Entry &entry : matrix)
        if (entry.id == id)
            return entry;
    return matrix.front();
}

} // namespace vernon::program_capabilities

#endif
