"""Ray-marched terrain adapted from Kevin Roast's MIT-licensed webglshaders.

Original source: https://github.com/kevinroast/webglshaders/blob/master/terrain1.html
Copyright (c) 2015 Kevin Roast
Full MIT notice: examples/THIRD_PARTY_LICENSES.md
"""

from typing import Annotated

import vernon_dsl as vd


@vd.func
def _smoothstep(edge0: vd.f32, edge1: vd.f32, value: vd.f32) -> vd.f32:
    t = vd.clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@vd.func
def _noise_lod(
    noise_texture: vd.Texture["2d", vd.f32],  # pyright: ignore[reportInvalidTypeForm]  # noqa: F722
    noise_sampler: vd.Sampler,
    position: vd.Vector[vd.f32, 2],
) -> vd.f32:
    return vd.texture_sample(noise_texture, noise_sampler, (position + 0.5) / 512.0, 0.0).x


@vd.func
def _noise(
    noise_texture: vd.Texture["2d", vd.f32],  # pyright: ignore[reportInvalidTypeForm]  # noqa: F722
    noise_sampler: vd.Sampler,
    position: vd.Vector[vd.f32, 2],
) -> vd.f32:
    cell = vd.floor(position)
    fraction = position - cell
    blend = fraction * fraction * (3.0 - 2.0 * fraction)
    index = cell.x + cell.y * 57.0
    a = _noise_lod(noise_texture, noise_sampler, vd.Vector([index, 0.0]))
    b = _noise_lod(noise_texture, noise_sampler, vd.Vector([index + 1.0, 0.0]))
    c = _noise_lod(noise_texture, noise_sampler, vd.Vector([index + 57.0, 0.0]))
    d = _noise_lod(noise_texture, noise_sampler, vd.Vector([index + 58.0, 0.0]))
    lower = a * (1.0 - blend.x) + b * blend.x
    upper = c * (1.0 - blend.x) + d * blend.x
    return lower * (1.0 - blend.y) + upper * blend.y


@vd.func
def _octave_transform(position: vd.Vector[vd.f32, 2]) -> vd.Vector[vd.f32, 2]:
    return vd.Vector(
        [
            position.x * 2.0 + position.y * 1.4,
            position.x * -1.4 + position.y * 2.0,
        ]
    )


@vd.func
def _fractal_noise(
    noise_texture: vd.Texture["2d", vd.f32],  # pyright: ignore[reportInvalidTypeForm]  # noqa: F722
    noise_sampler: vd.Sampler,
    position: vd.Vector[vd.f32, 2],
    octave_count: vd.i32,
) -> vd.f32:
    amplitude = vd.f32(0.75)
    result = vd.f32(0.0)
    for octave_index in range(7):
        if vd.i32(octave_index) >= octave_count:
            break
        ridge = 1.5 - _noise(noise_texture, noise_sampler, position)
        result = result + ridge * ridge * amplitude
        amplitude = amplitude * 0.5
        position = _octave_transform(position)
    return result


@vd.func
def _terrain_height(
    noise_texture: vd.Texture["2d", vd.f32],  # pyright: ignore[reportInvalidTypeForm]  # noqa: F722
    noise_sampler: vd.Sampler,
    position: vd.Vector[vd.f32, 2],
    octave_count: vd.i32,
) -> vd.f32:
    ridge = _fractal_noise(noise_texture, noise_sampler, position.yx * 0.571, octave_count) * 0.693
    detail_position = vd.cos(position.yx * 0.6) + vd.sin(position * 0.6)
    return ridge - _noise(noise_texture, noise_sampler, detail_position)


@vd.func
def _terrain_distance(
    noise_texture: vd.Texture["2d", vd.f32],  # pyright: ignore[reportInvalidTypeForm]  # noqa: F722
    noise_sampler: vd.Sampler,
    position: vd.Vector[vd.f32, 3],
) -> vd.f32:
    return position.y - _terrain_height(noise_texture, noise_sampler, position.xz, 5)


@vd.func
def _terrain_normal(
    noise_texture: vd.Texture["2d", vd.f32],  # pyright: ignore[reportInvalidTypeForm]  # noqa: F722
    noise_sampler: vd.Sampler,
    position: vd.Vector[vd.f32, 3],
    distance_along_ray: vd.f32,
) -> vd.Vector[vd.f32, 3]:
    epsilon = vd.max(0.001 * distance_along_ray, 0.0005)
    offset = vd.Vector([epsilon, 0.0])
    left = _terrain_height(noise_texture, noise_sampler, position.xz - offset, 7)
    right = _terrain_height(noise_texture, noise_sampler, position.xz + offset, 7)
    back = _terrain_height(noise_texture, noise_sampler, position.xz - offset.yx, 7)
    front = _terrain_height(noise_texture, noise_sampler, position.xz + offset.yx, 7)
    return vd.normalize(vd.Vector([left - right, 2.0 * epsilon, back - front]))


@vd.func
def _sky(ray: vd.Vector[vd.f32, 3], sun_direction: vd.Vector[vd.f32, 3]) -> vd.Vector[vd.f32, 3]:
    sun_amount = vd.max(vd.dot(ray, sun_direction), 0.0)
    horizon = vd.pow(1.0 - vd.max(ray.y, 0.0), 6.0)
    sky_color = vd.Vector([0.10, 0.20, 0.30])
    horizon_color = vd.Vector([0.38, 0.36, 0.32])
    sun_color = vd.Vector([1.70, 1.00, 0.60])
    sky = sky_color * (1.0 - horizon) + horizon_color * horizon
    sky = sky + sun_color * sun_amount * sun_amount * 0.25
    sky = sky + sun_color * vd.min(vd.pow(sun_amount, 800.0) * 1.5, 0.3)
    return vd.clamp(sky, 0.0, 1.0)


@vd.func
def _depth_fog(
    color: vd.Vector[vd.f32, 3],
    distance: vd.f32,
    ray: vd.Vector[vd.f32, 3],
    origin: vd.Vector[vd.f32, 3],
    sun_direction: vd.Vector[vd.f32, 3],
) -> vd.Vector[vd.f32, 3]:
    ray_y = ray.y
    if vd.abs(ray_y) < 0.0001:
        ray_y = vd.f32(0.0001)
    fog_amount = 0.30 * vd.exp(-origin.y * 0.15) * (1.0 - vd.exp(-distance * ray_y * 0.15)) / ray_y
    fog_amount = vd.clamp(fog_amount, 0.0, 1.0)
    return color * (1.0 - fog_amount) + _sky(ray, sun_direction) * fog_amount


@vd.func
def _soft_shadow(
    noise_texture: vd.Texture["2d", vd.f32],  # pyright: ignore[reportInvalidTypeForm]  # noqa: F722
    noise_sampler: vd.Sampler,
    origin: vd.Vector[vd.f32, 3],
    direction: vd.Vector[vd.f32, 3],
    shadow_steps: vd.i32,
) -> vd.f32:
    visibility = vd.f32(1.0)
    distance_along_ray = vd.f32(0.05)
    for step_index in range(48):
        if vd.i32(step_index) >= shadow_steps or distance_along_ray >= 16.0:
            break
        distance = _terrain_distance(noise_texture, noise_sampler, origin + direction * distance_along_ray)
        visibility = vd.min(visibility, 64.0 * distance / distance_along_ray)
        distance_along_ray = distance_along_ray + vd.max(distance, 0.005)
    return vd.clamp(visibility, 0.0, 1.0)


@vd.func
def _ambient_occlusion(
    noise_texture: vd.Texture["2d", vd.f32],  # pyright: ignore[reportInvalidTypeForm]  # noqa: F722
    noise_sampler: vd.Sampler,
    position: vd.Vector[vd.f32, 3],
    normal: vd.Vector[vd.f32, 3],
    ao_samples: vd.i32,
) -> vd.f32:
    occlusion = vd.f32(0.0)
    weight = vd.f32(1.0)
    for sample_index in range(4):
        if vd.i32(sample_index) >= ao_samples:
            break
        distance = (vd.f32(sample_index) + 1.0) * 0.2
        occlusion = occlusion + weight * (
            distance - _terrain_distance(noise_texture, noise_sampler, position + normal * distance)
        )
        weight = weight * 0.5
    return 1.0 - vd.clamp(occlusion, 0.0, 1.0)


@vd.func
def _shade_terrain(
    noise_texture: vd.Texture["2d", vd.f32],  # pyright: ignore[reportInvalidTypeForm]  # noqa: F722
    noise_sampler: vd.Sampler,
    position: vd.Vector[vd.f32, 3],
    ray: vd.Vector[vd.f32, 3],
    normal: vd.Vector[vd.f32, 3],
    origin: vd.Vector[vd.f32, 3],
    sun_direction: vd.Vector[vd.f32, 3],
    ambient: vd.f32,
    shadow_steps: vd.i32,
    ao_samples: vd.i32,
) -> vd.Vector[vd.f32, 3]:
    sun_color = vd.Vector([1.70, 1.00, 0.60])
    sun = vd.clamp(vd.dot(normal, sun_direction), 0.0, 1.0)
    half_direction = vd.normalize(-ray + sun_direction)
    snow_noise = _noise_lod(noise_texture, noise_sampler, normal.yz * 2.111)
    snow_mix = _smoothstep(0.05, 0.4, position.y - snow_noise + 0.65)
    grass_noise = _noise_lod(noise_texture, noise_sampler, normal.xy * 0.973) - 0.3
    grass_mix = 1.0 - vd.clamp(position.y + 0.45 + grass_noise, 0.0, 1.0)

    rock_color = vd.Vector([1.0, 0.8, 0.8])
    snow_color = vd.Vector([1.5, 1.5, 1.5])
    grass_color = vd.Vector([0.3, 0.7, 0.3])
    material = rock_color * (1.0 - snow_mix) + snow_color * snow_mix
    material = material * (1.0 - grass_mix) + grass_color * grass_mix
    diffuse_strength = vd.f32(0.5) * (1.0 - snow_mix) + 0.9 * snow_mix
    diffuse_strength = diffuse_strength * (1.0 - grass_mix) + 0.8 * grass_mix
    specular = vd.pow(vd.max(vd.dot(half_direction, normal), 0.0), 64.0) * snow_mix

    shadow = _soft_shadow(
        noise_texture,
        noise_sampler,
        position + normal * 0.003,
        sun_direction,
        shadow_steps,
    )
    light = (sun * sun_color * diffuse_strength + specular) * vd.Vector(
        [shadow, vd.pow(shadow, 1.2), vd.pow(shadow, 1.5)]
    )
    occlusion = _ambient_occlusion(noise_texture, noise_sampler, position, normal, ao_samples) * ambient
    sky_light = vd.clamp(0.5 + 0.5 * normal.y, 0.0, 1.0)
    light = light + sky_light * vd.Vector([0.24, 0.20, 0.24]) * occlusion
    indirect = vd.clamp(
        vd.dot(normal, vd.normalize(sun_direction * vd.Vector([-1.0, 0.0, -1.0]))),
        0.0,
        1.0,
    )
    light = light + indirect * material * 0.3 * occlusion
    color = material * light
    return _depth_fog(color, vd.norm(origin - position), ray, origin, sun_direction)


@vd.func
def _post_process(color: vd.Vector[vd.f32, 3], uv: vd.Vector[vd.f32, 2]) -> vd.Vector[vd.f32, 3]:
    color = vd.Vector(
        [
            vd.pow(vd.max(color.x, 0.0), 0.8),
            vd.pow(vd.max(color.y, 0.0), 0.8),
            vd.pow(vd.max(color.z, 0.0), 0.8),
        ]
    )
    brightness = color * 1.4
    luminance = vd.dot(vd.Vector([0.2125, 0.7154, 0.0721]), brightness)
    saturated = vd.Vector([luminance, luminance, luminance]) * -0.4 + brightness * 1.4
    contrasted = vd.Vector([0.5, 0.5, 0.5]) * -0.2 + saturated * 1.2
    vignette = 0.4 + 0.5 * vd.pow(40.0 * uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y), 0.2)
    return contrasted * vignette


@vd.fragment
def terrain_fragment(
    noise_texture: Annotated[
        vd.Texture["2d", vd.f32],  # pyright: ignore[reportInvalidTypeForm]  # noqa: F722
        vd.resource(set=0, binding=0),
    ],
    noise_sampler: Annotated[vd.Sampler, vd.resource(set=0, binding=1)],
    time: Annotated[vd.f32, vd.uniform()],
    camera_position: Annotated[vd.Vector[vd.f32, 3], vd.uniform()],
    camera_target: Annotated[vd.Vector[vd.f32, 3], vd.uniform()],
    sun_direction: Annotated[vd.Vector[vd.f32, 3], vd.uniform()],
    screen_y_sign: Annotated[vd.f32, vd.uniform()],
    ambient: Annotated[vd.f32, vd.uniform()],
    max_steps: Annotated[vd.i32, vd.uniform()],
    shadow_steps: Annotated[vd.i32, vd.uniform()],
    ao_samples: Annotated[vd.i32, vd.uniform()],
) -> vd.Vector[vd.f32, 4]:
    size = vd.resolution()
    pixel = vd.fragment_coord().xy
    uv = pixel / size
    camera_offset = vd.Vector([0.0, vd.cos(time * 0.5) * 0.5, -time])
    origin = camera_position + camera_offset
    forward = vd.normalize(camera_target - camera_position)
    right = vd.normalize(vd.cross(vd.Vector([0.0, 1.0, 0.0]), forward))
    up = vd.normalize(vd.cross(forward, right))
    screen = uv - 0.5
    screen = vd.Vector([screen.x * size.x / size.y, screen.y * screen_y_sign])
    ray = vd.normalize(forward + right * screen.x + up * screen.y)

    distance_along_ray = vd.f32(0.0)
    position = origin
    for step_index in range(150):
        if vd.i32(step_index) >= max_steps:
            break
        position = origin + ray * distance_along_ray
        distance = _terrain_distance(noise_texture, noise_sampler, position)
        if distance < 0.002:
            break
        distance_along_ray = distance_along_ray + distance * 0.5
        if distance_along_ray > 100.0:
            break

    color = _sky(ray, sun_direction)
    if _terrain_distance(noise_texture, noise_sampler, position) < 0.01:
        normal = _terrain_normal(noise_texture, noise_sampler, position, distance_along_ray)
        color = _shade_terrain(
            noise_texture,
            noise_sampler,
            position,
            ray,
            normal,
            origin,
            sun_direction,
            ambient,
            shadow_steps,
            ao_samples,
        )
    color = _post_process(color, uv)
    return vd.Vector([vd.clamp(color, 0.0, 1.0), 1.0])
