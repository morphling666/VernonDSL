"""Mandelbulb shader adapted from the MIT-licensed WebGL-Mandelbulb.

Original source: https://github.com/matt-k-wong/WebGL-Mandelbulb
Copyright (c) 2026
Full MIT notice: examples/THIRD_PARTY_LICENSES.md
"""

from typing import Annotated

import vernon_dsl as vd


@vd.struct
class DistanceEstimate:
    distance: vd.f32
    orbit_trap: vd.f32


@vd.func
def mandelbulb_distance(
    position: vd.Vector[vd.f32, 3],
    power: vd.f32,
    max_iterations: vd.i32,
) -> DistanceEstimate:
    z = position
    derivative = vd.f32(1.0)
    radius = vd.f32(0.0)
    orbit_trap = vd.f32(100000.0)
    for iteration_index in range(24):
        if vd.i32(iteration_index) >= max_iterations:
            break
        radius = vd.norm(z)
        if radius > 2.0:
            break
        safe_radius = vd.max(radius, 0.000001)
        orbit_trap = vd.min(orbit_trap, safe_radius)
        theta = vd.acos(vd.clamp(z.z / safe_radius, -1.0, 1.0))
        phi = vd.atan2(z.y, z.x)
        derivative = vd.pow(safe_radius, power - 1.0) * power * derivative + 1.0
        raised = vd.pow(safe_radius, power)
        theta = theta * power
        phi = phi * power
        z = (
            raised
            * vd.Vector(
                [
                    vd.sin(theta) * vd.cos(phi),
                    vd.sin(theta) * vd.sin(phi),
                    vd.cos(theta),
                ]
            )
            + position
        )
    safe_radius = vd.max(radius, 0.000001)
    return DistanceEstimate(
        0.5 * vd.log(safe_radius) * safe_radius / derivative,
        orbit_trap,
    )


@vd.func
def mandelbulb_normal(
    position: vd.Vector[vd.f32, 3],
    power: vd.f32,
    max_iterations: vd.i32,
) -> vd.Vector[vd.f32, 3]:
    epsilon = vd.f32(0.0012)
    x = vd.Vector([epsilon, 0.0, 0.0])
    y = vd.Vector([0.0, epsilon, 0.0])
    z = vd.Vector([0.0, 0.0, epsilon])
    return vd.normalize(
        vd.Vector(
            [
                mandelbulb_distance(position + x, power, max_iterations).distance
                - mandelbulb_distance(position - x, power, max_iterations).distance,
                mandelbulb_distance(position + y, power, max_iterations).distance
                - mandelbulb_distance(position - y, power, max_iterations).distance,
                mandelbulb_distance(position + z, power, max_iterations).distance
                - mandelbulb_distance(position - z, power, max_iterations).distance,
            ]
        )
    )


@vd.func
def mandelbulb_shadow(
    origin: vd.Vector[vd.f32, 3],
    direction: vd.Vector[vd.f32, 3],
    power: vd.f32,
    max_iterations: vd.i32,
    shadow_steps: vd.i32,
) -> vd.f32:
    visibility = vd.f32(1.0)
    distance_along_ray = vd.f32(0.02)
    for step_index in range(40):
        if vd.i32(step_index) >= shadow_steps:
            break
        distance = mandelbulb_distance(
            origin + direction * distance_along_ray,
            power,
            max_iterations,
        ).distance
        visibility = vd.min(visibility, 18.0 * distance / distance_along_ray)
        distance_along_ray = distance_along_ray + vd.clamp(distance, 0.02, 0.20)
        if distance < 0.001 or distance_along_ray > 5.0:
            break
    return vd.clamp(visibility, 0.0, 1.0)


@vd.func
def orbit_palette(value: vd.f32, time: vd.f32) -> vd.Vector[vd.f32, 3]:
    phase = value * 6.2831853 + time * 0.18
    return vd.Vector(
        [
            0.52 + 0.48 * vd.cos(phase + 0.15),
            0.50 + 0.50 * vd.cos(phase + 2.25),
            0.55 + 0.45 * vd.cos(phase + 4.25),
        ]
    )


@vd.fragment
def mandelbulb_fragment(
    camera_position: Annotated[vd.Vector[vd.f32, 3], vd.uniform()],
    camera_target: Annotated[vd.Vector[vd.f32, 3], vd.uniform()],
    time: Annotated[vd.f32, vd.uniform()],
    power: Annotated[vd.f32, vd.uniform()],
    max_iterations: Annotated[vd.i32, vd.uniform()],
    max_steps: Annotated[vd.i32, vd.uniform()],
    shadow_steps: Annotated[vd.i32, vd.uniform()],
) -> vd.Vector[vd.f32, 4]:
    size = vd.resolution()
    pixel = vd.fragment_coord().xy
    uv = (pixel - size * 0.5) / size.y
    origin = camera_position
    forward = vd.normalize(camera_target - origin)
    right = vd.normalize(vd.cross(forward, vd.Vector([0.0, 1.0, 0.0])))
    up = vd.normalize(vd.cross(right, forward))
    ray = vd.normalize(right * uv.x + up * uv.y + forward * 1.32)

    distance_along_ray = vd.f32(0.0)
    step_count = vd.f32(0.0)
    orbit_trap = vd.f32(0.0)
    for step_index in range(128):
        if vd.i32(step_index) >= max_steps:
            break
        estimate = mandelbulb_distance(
            origin + ray * distance_along_ray,
            power,
            max_iterations,
        )
        distance = estimate.distance
        orbit_trap = estimate.orbit_trap
        if distance < 0.006:
            break
        distance_along_ray = distance_along_ray + distance
        step_count = vd.f32(step_index) + 1.0
        if distance_along_ray > 12.0:
            break

    horizon = vd.clamp(ray.y * 0.5 + 0.5, 0.0, 1.0)
    background = vd.Vector([0.008, 0.012, 0.030]) * (0.7 + horizon * 1.1)
    color = background
    if distance_along_ray < 4.5:
        position = origin + ray * distance_along_ray
        normal = mandelbulb_normal(position, power, max_iterations)
        light_direction = vd.normalize(vd.Vector([1.2, 1.0, -1.0]))
        diffuse = vd.max(vd.dot(normal, light_direction), 0.0)
        sky_light = vd.max(vd.dot(normal, vd.Vector([0.0, 1.0, 0.0])), 0.0)
        shadow = mandelbulb_shadow(
            position + normal * 0.003,
            light_direction,
            power,
            max_iterations,
            shadow_steps,
        )
        ambient_occlusion = 0.35 + 0.65 * vd.clamp(
            1.0 - step_count / vd.f32(max_steps),
            0.0,
            1.0,
        )
        material = orbit_palette(orbit_trap * 1.5 + vd.norm(position) * 0.2, time)
        ambient = vd.Vector([0.055, 0.08, 0.12]) * (0.28 + sky_light * 0.72)
        color = (ambient * material + material * diffuse * shadow * 1.35) * ambient_occlusion
        view_direction = vd.normalize(origin - position)
        reflected = vd.reflect(-light_direction, normal)
        specular = vd.pow(vd.max(vd.dot(view_direction, reflected), 0.0), 32.0)
        color = color + vd.Vector([1.0, 0.88, 0.72]) * specular * shadow * ambient_occlusion

    fog = 1.0 - vd.exp(-0.018 * distance_along_ray * distance_along_ray)
    fog = vd.clamp(fog, 0.0, 1.0)
    color = color * (1.0 - fog) + background * fog
    gamma = vd.Vector(
        [
            vd.pow(vd.clamp(color.x, 0.0, 1.0), 0.45454545),
            vd.pow(vd.clamp(color.y, 0.0, 1.0), 0.45454545),
            vd.pow(vd.clamp(color.z, 0.0, 1.0), 0.45454545),
        ]
    )
    return vd.Vector([gamma, 1.0])
