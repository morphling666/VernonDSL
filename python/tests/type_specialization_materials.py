from __future__ import annotations

import vernon_dsl as vd


@vd.binding_group
class Lookup:
    weights: vd.TensorView[vd.f32, (vd.dyn,), vd.read]


@vd.binding_group
class Matte:
    gain: vd.f32
    lookup: Lookup

    @vd.func
    def term(self, x: vd.u32) -> vd.f32:
        return self.lookup.weights[x] * self.gain


@vd.binding_group
class Glossy:
    gain: vd.f32
    offset: vd.f32
    lookup: Lookup

    @vd.func
    def term(self, x: vd.u32) -> vd.f32:
        return self.lookup.weights[x] * self.gain + self.offset


@vd.struct(shared=True)
class SurfaceParams:
    tint: vd.Vector[vd.f32, 4]
    gain: vd.f32


@vd.binding_group
class Flat:
    params: SurfaceParams

    @vd.func
    def shade(self) -> vd.Vector[vd.f32, 4]:
        return self.params.tint


@vd.binding_group
class Scaled:
    params: SurfaceParams

    @vd.func
    def shade(self) -> vd.Vector[vd.f32, 4]:
        return self.params.tint * self.params.gain
