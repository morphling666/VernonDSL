"""Aurora shaders adapted from MIT-licensed jagajaga/coaurora.

Original source: https://github.com/jagajaga/coaurora
Copyright (c) 2026 Arseniy Seroka (jagajaga)
Full MIT notice: examples/THIRD_PARTY_LICENSES.md
"""

from typing import Annotated

import vernon_dsl as vd


@vd.func
def _fract(value: vd.f32) -> vd.f32:
    return value - vd.floor(value)


@vd.func
def _hash(value: vd.f32) -> vd.f32:
    return _fract(vd.sin(value * 127.1 + 311.7) * 43758.5453)


@vd.func
def _palette(phase: vd.f32) -> vd.Vector[vd.f32, 3]:
    return vd.Vector(
        [
            0.48 + 0.32 * vd.cos(phase),
            0.58 + 0.38 * vd.cos(phase - 2.05),
            0.62 + 0.34 * vd.cos(phase + 2.05),
        ]
    )


@vd.func
def _sample_field(
    field: vd.Texture["2d", vd.f32],  # pyright: ignore[reportInvalidTypeForm]  # noqa: F722
    field_sampler: vd.Sampler,
    uv: vd.Vector[vd.f32, 2],
) -> vd.Vector[vd.f32, 4]:
    return vd.texture_sample(field, field_sampler, uv)


@vd.fragment
def aurora_curtains(
    time: Annotated[vd.f32, vd.uniform()],
) -> vd.Vector[vd.f32, 4]:
    coordinates = vd.fragment_coord().xy
    size = vd.resolution()
    uv = coordinates / size
    vertical = vd.clamp(uv.y, 0.0, 1.0)
    color = vd.Vector([0.008, 0.016, 0.055]) * (0.55 + vertical * 0.75)
    for index in range(12):
        curtain_index = vd.f32(index)
        fi = curtain_index / 11.0
        depth = _hash(curtain_index + 0.3)
        base_x = -0.05 + 1.10 * fi + 0.05 * (_hash(curtain_index * 1.7) - 0.5)
        amplitude = (0.028 + 0.03 * _hash(curtain_index * 2.3)) * (0.6 + 0.4 * vd.sin(time * 0.5 + fi * 3.0))
        frequency = (0.35 + 0.5 * _hash(curtain_index * 0.7)) * 6.2831853
        phase = time * 0.6 + _hash(curtain_index * 3.1) * 6.2831853
        radius = 0.0112 + 0.0308 * depth
        center = (
            base_x
            + amplitude * vd.sin(uv.y * frequency + phase)
            + amplitude * 0.22 * vd.sin(uv.y * frequency * 1.7 - time * 0.7 + fi * 3.1)
        )
        delta = uv.x - center
        gaussian = vd.exp(-(delta * delta) / (2.0 * radius * radius))
        fold = 0.5 + 0.5 * vd.sin(uv.y * 7.0 - time * 1.3 + fi * 4.0)
        brightness = (0.11 + 0.17 * depth) * (0.3 + 0.7 * fold * fold) * gaussian * 2.35
        hue = time * 0.35 + fi * 2.0 + uv.y * 2.5
        color = color + _palette(hue) * brightness
    vignette = vd.clamp(1.25 - vd.norm(uv - vd.Vector([0.5, 0.52])) * 0.72, 0.0, 1.0)
    return vd.Vector([color * vignette, 1.0])


@vd.fragment
def aurora_compose(
    field: Annotated[
        vd.Texture["2d", vd.f32],  # pyright: ignore[reportInvalidTypeForm]  # noqa: F722
        vd.resource(set=0, binding=0),
    ],
    field_sampler: Annotated[vd.Sampler, vd.resource(set=0, binding=1)],
    time: Annotated[vd.f32, vd.uniform()],
    blur_radius: Annotated[vd.f32, vd.uniform()],
) -> vd.Vector[vd.f32, 4]:
    pixel = vd.fragment_coord().xy
    size = vd.resolution()
    color = vd.Vector([0.0, 0.0, 0.0])
    for y_index in range(3):
        y_offset = vd.f32(y_index) - 1.0
        for x_index in range(3):
            x_offset = vd.f32(x_index) - 1.0
            offset = vd.Vector([x_offset, y_offset]) * blur_radius
            color = color + _sample_field(field, field_sampler, (pixel + offset) / size).xyz
    color = color / 9.0

    cell = vd.floor(pixel / 3.0)
    star_noise = _hash(cell.x * 41.0 + cell.y * 289.0)
    star = vd.clamp((star_noise - 0.9965) / 0.0035, 0.0, 1.0)
    sky_mask = vd.clamp((pixel.y / size.y - 0.22) * 1.28, 0.0, 1.0)
    color = color + vd.Vector([0.75, 0.86, 1.0]) * star * star * sky_mask

    temporal = _fract(time * 50.0)
    noise_a = _fract(52.9829189 * _fract(vd.dot(pixel, vd.Vector([0.06711056, 0.00583715]))) + temporal)
    noise_b = _fract(
        52.9829189 * _fract(vd.dot(pixel + vd.Vector([97.0, 71.0]), vd.Vector([0.06711056, 0.00583715])))
        + temporal
        + 0.5
    )
    color = color + (noise_a + noise_b - 1.0) * (1.6 / 255.0)
    mapped = color / (color + vd.Vector([1.0, 1.0, 1.0]))
    gamma = vd.Vector(
        [
            vd.pow(vd.clamp(mapped.x, 0.0, 1.0), 0.45454545),
            vd.pow(vd.clamp(mapped.y, 0.0, 1.0), 0.45454545),
            vd.pow(vd.clamp(mapped.z, 0.0, 1.0), 0.45454545),
        ]
    )
    return vd.Vector([gamma, 1.0])
