from __future__ import annotations

from typing import Annotated

import vernon_dsl as vd


@vd.struct(shared=True)
class SurfaceParams:
    tint: vd.Vector[vd.f32, 4]
    gain: vd.f32


@vd.binding_group
class Lookup:
    weights: vd.TensorView[vd.f32, (vd.dyn,), vd.read]


@vd.binding_group
class Surface:
    params: SurfaceParams
    albedo: vd.Texture["2d", vd.f32]  # noqa: F722
    linear: vd.Sampler
    lookup: Lookup

    @vd.func
    def shade(self, uv: vd.Vector[vd.f32, 2]) -> vd.Vector[vd.f32, 4]:
        texel = vd.texture_sample(self.albedo, self.linear, uv)
        return texel * self.params.tint * self.lookup.weights[0]


@vd.binding_group
class Flat:
    params: SurfaceParams

    @vd.func
    def shade(self, uv: vd.Vector[vd.f32, 2]) -> vd.Vector[vd.f32, 4]:
        return self.params.tint * (self.params.gain + uv[0] * 0.0)


@vd.func
def apply(material, uv):
    return material.shade(uv)


@vd.vertex
def surface_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
    surface: Surface,
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([position * surface.lookup.weights[1], 0.0, 1.0])


@vd.fragment
def surface_fragment(surface: Surface, flat: Flat) -> vd.Vector[vd.f32, 4]:
    uv = vd.Vector([0.5, 0.5])
    return apply(surface, uv) + apply(flat, uv)


@vd.binding_group
class Accumulator:
    params: SurfaceParams
    lookup: Lookup
    bias: vd.TensorView[vd.f32, (vd.dyn,), vd.read]


@vd.kernel(workgroup_size=(1, 1, 1))
def accumulate(
    terms: Accumulator,
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    output[x] = terms.lookup.weights[x] * terms.params.gain + terms.bias[x] + terms.params.tint[0]


@vd.binding_group
class Affine:
    gain: vd.f32
    offset: vd.f32
    lookup: Lookup
    bias: vd.TensorView[vd.f32, (vd.dyn,), vd.read]

    @vd.func
    def term(self, x: vd.u32) -> vd.f32:
        return self.lookup.weights[x] * self.gain + self.bias[x] + self.offset


@vd.kernel(workgroup_size=(1, 1, 1))
def affine(
    terms: Affine,
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    output[x] = apply_term(terms, x)


@vd.func
def apply_term(terms, x):
    return terms.term(x)


@vd.binding_group
class Volume:
    image: vd.Texture["3d", vd.rgba32_float, vd.read_write]  # noqa: F722
    gain: vd.f32


@vd.kernel(workgroup_size=(1, 1, 1))
def sample_volume(
    volume: Volume,
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    texel = vd.texture_load(volume.image, vd.Vector([vd.i32(x), 0, 0]))
    output[x] = texel[0] * volume.gain
