# pyright: reportMissingImports=false

from typing import Annotated

import vernon_dsl as vd


@vd.struct
class Layers:
    density: vd.Vector[vd.f32, 4]
    scale: vd.f32
    steps: vd.u32


@vd.struct
class Sample:
    total: vd.Vector[vd.f32, 3]
    extinction: vd.Vector[vd.f32, 3]
    low: vd.Vector[vd.f32, 3]
    high: vd.Vector[vd.f32, 3]


@vd.func
def sample_layers(layers: Layers, height: vd.f32) -> Sample:
    low = layers.density.xyz * vd.exp(layers.scale * height)
    high = layers.density.zyx * vd.exp(layers.density.w * height)
    return Sample(low + high, low + high * 2.0, low, high)


@vd.vertex
def fullscreen_vertex(
    vertex: Annotated[vd.u32, vd.builtin("vertex_index")],
    extent: Annotated[vd.Vector[vd.f32, 2], vd.uniform()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    x = -1.0 if vertex == 0 else 3.0 if vertex == 1 else -1.0
    y = -1.0 if vertex == 0 else -1.0 if vertex == 1 else 3.0
    return vd.Vector([x * extent.x, y * extent.y, 0.0, 1.0])


# The loop builds a struct each iteration and reads it after a branch, so SPIRV-Cross would hoist the struct out of
# the loop and assign it an initializer list, which HLSL rejects, if the compiler did not read its members directly.
@vd.fragment
def layered_fragment(
    layers: Annotated[Layers, vd.uniform()],
    lookup: vd.Texture["2d", vd.f32],  # noqa: F722
    lookup_sampler: vd.Sampler,
) -> vd.Vector[vd.f32, 4]:
    luminance = vd.Vector([0.0, 0.0, 0.0])
    throughput = vd.Vector([1.0, 1.0, 1.0])
    for index in range(vd.i32(layers.steps)):
        sample = sample_layers(layers, vd.f32(index) / vd.f32(layers.steps))
        extinction = vd.max(sample.extinction, vd.Vector([1e-9, 1e-9, 1e-9]))
        light = vd.texture_sample(lookup, lookup_sampler, vd.Vector([vd.f32(index) * 0.1, 0.5]), 0.0).xyz
        if layers.scale > 0.0:
            light = light * sample.low + sample.high
        luminance = luminance + throughput * sample.total * light / extinction
        throughput = throughput * vd.exp(extinction * -0.1)
    return vd.Vector([luminance, 1.0])


layered_asset = vd.program_asset(
    id="pipelines/layered",
    program=vd.pipeline(
        fullscreen_vertex,
        layered_fragment,
        targets=vd.target_formats(colors={0: vd.rgba16_float}),
    ),
    variants=({},),
)
