from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vernon_dsl as vd
from shader_lib.mesh import expand_indexed_mesh
from shader_lib.pbr import pbr_fragment, pbr_vertex, shadow_fragment, shadow_vertex
from shader_lib.showcase import sky_fragment, sky_vertex
from shader_lib.terrain_erosion import apply_erosion_flow, build_terrain_mesh, compute_erosion_flow
from showcase_common import (
    BatchRenderPass,
    ComputeInvocationPass,
    FramePresenter,
    InvocationBatch,
    InvocationSlot,
    architecture_from_name,
    create_sky_cube,
    load_equirectangular_environment,
    load_rgba_texture,
    look_at,
    perspective,
)


@dataclass(frozen=True)
class TerrainGrid:
    height: np.ndarray
    water: np.ndarray
    sediment: np.ndarray
    rock_detail: np.ndarray
    indices: np.ndarray


def value_noise(u: np.ndarray, v: np.ndarray, frequency: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lattice = rng.random((frequency + 1, frequency + 1), dtype=np.float32)
    sample_x = u * np.float32(frequency)
    sample_y = v * np.float32(frequency)
    x0 = np.floor(sample_x).astype(np.int32)
    y0 = np.floor(sample_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, frequency)
    y1 = np.minimum(y0 + 1, frequency)
    tx = sample_x - x0
    ty = sample_y - y0
    tx = tx * tx * (np.float32(3.0) - np.float32(2.0) * tx)
    ty = ty * ty * (np.float32(3.0) - np.float32(2.0) * ty)
    lower = lattice[y0, x0] * (np.float32(1.0) - tx) + lattice[y0, x1] * tx
    upper = lattice[y1, x0] * (np.float32(1.0) - tx) + lattice[y1, x1] * tx
    return lower * (np.float32(1.0) - ty) + upper * ty


def fractal_noise(u: np.ndarray, v: np.ndarray, seed: int) -> np.ndarray:
    result = np.zeros_like(u, dtype=np.float32)
    amplitude = np.float32(0.5)
    amplitude_sum = np.float32(0.0)
    for octave, frequency in enumerate((2, 4, 8, 16, 32)):
        result += value_noise(u, v, frequency, seed + octave * 19) * amplitude
        amplitude_sum += amplitude
        amplitude *= np.float32(0.5)
    return result / amplitude_sum


def create_terrain_grid(grid: int) -> TerrainGrid:
    if grid < 16:
        raise ValueError("terrain grid must be at least 16")
    coordinate = np.linspace(-1.0, 1.0, grid, dtype=np.float32)
    xx, zz = np.meshgrid(coordinate, coordinate, indexing="xy")
    uu = (xx + np.float32(1.0)) * np.float32(0.5)
    vv = (zz + np.float32(1.0)) * np.float32(0.5)
    warp_u = value_noise(uu, vv, 3, 71) - np.float32(0.5)
    warp_v = value_noise(uu, vv, 3, 113) - np.float32(0.5)
    warped_u = np.clip(uu + warp_u * np.float32(0.16), 0.0, 1.0)
    warped_v = np.clip(vv + warp_v * np.float32(0.16), 0.0, 1.0)
    macro_noise = fractal_noise(warped_u, warped_v, 191)
    rock_detail = fractal_noise(uu, vv, 419)
    macro_centered = macro_noise - np.float32(0.5)
    detail_centered = rock_detail - np.float32(0.5)
    plateau = (
        np.float32(0.98)
        + macro_centered * np.float32(0.52)
        + detail_centered * np.float32(0.09)
        - zz * np.float32(0.07)
    )
    channel_center = np.sin(zz * np.float32(2.55) - np.float32(0.35)) * np.float32(0.19) + np.sin(
        zz * np.float32(7.8) + np.float32(0.6)
    ) * np.float32(0.055)
    channel_distance = np.abs(xx - channel_center + macro_centered * np.float32(0.1) + warp_u * np.float32(0.035))
    canyon_width = np.clip(
        np.float32(0.195) + detail_centered * np.float32(0.055),
        np.float32(0.145),
        np.float32(0.245),
    )
    gorge_width = np.clip(
        np.float32(0.06) + detail_centered * np.float32(0.016),
        np.float32(0.045),
        np.float32(0.078),
    )
    broad_canyon = np.exp(-((channel_distance / canyon_width) ** 2))
    inner_gorge = np.exp(-((channel_distance / gorge_width) ** 2))
    canyon_depth = broad_canyon * (
        np.float32(0.92) + np.sin(zz * np.float32(4.4)) * np.float32(0.09)
    ) + inner_gorge * np.float32(0.42)
    fractured_wall = broad_canyon * (np.float32(1.0) - inner_gorge) * detail_centered * np.float32(0.28)

    left_branch_center = -np.float32(0.82) + (zz + np.float32(0.82)) * np.float32(0.76)
    left_branch_window = np.clip((zz + np.float32(0.88)) * np.float32(2.0), 0.0, 1.0) * np.clip(
        (np.float32(0.38) - zz) * np.float32(2.1),
        0.0,
        1.0,
    )
    left_branch = (
        np.exp(-(((xx - left_branch_center) / np.float32(0.075)) ** 2)) * left_branch_window * np.float32(0.48)
    )
    right_branch_center = np.float32(0.78) - (zz + np.float32(0.22)) * np.float32(0.68)
    right_branch_window = np.clip((zz + np.float32(0.28)) * np.float32(2.4), 0.0, 1.0) * np.clip(
        (np.float32(0.9) - zz) * np.float32(1.8),
        0.0,
        1.0,
    )
    right_branch = (
        np.exp(-(((xx - right_branch_center) / np.float32(0.068)) ** 2)) * right_branch_window * np.float32(0.4)
    )
    edge_distance = np.maximum(np.abs(xx), np.abs(zz))
    edge_falloff = np.clip((np.float32(1.0) - edge_distance) * np.float32(9.0), 0.0, 1.0)
    eroded_plateau = plateau - canyon_depth - left_branch - right_branch + fractured_wall
    height = (np.float32(-0.42) + (eroded_plateau + np.float32(0.42)) * edge_falloff).astype(np.float32)
    main_channel = np.exp(-((channel_distance / np.float32(0.026)) ** 2))
    left_runoff = np.exp(-(((xx - left_branch_center) / np.float32(0.027)) ** 2)) * left_branch_window
    right_runoff = np.exp(-(((xx - right_branch_center) / np.float32(0.025)) ** 2)) * right_branch_window
    water = (
        main_channel * np.float32(0.036) + left_runoff * np.float32(0.018) + right_runoff * np.float32(0.016)
    ).astype(np.float32)
    water *= edge_falloff
    sediment = (water * np.float32(0.42)).astype(np.float32)
    cells_y, cells_x = np.meshgrid(
        np.arange(grid - 1, dtype=np.uint32),
        np.arange(grid - 1, dtype=np.uint32),
        indexing="ij",
    )
    top_left = cells_y * np.uint32(grid) + cells_x
    indices = np.stack(
        (
            top_left,
            top_left + np.uint32(grid),
            top_left + np.uint32(1),
            top_left + np.uint32(1),
            top_left + np.uint32(grid),
            top_left + np.uint32(grid + 1),
        ),
        axis=-1,
    ).reshape(-1)
    return TerrainGrid(
        np.ascontiguousarray(height.reshape(-1)),
        np.ascontiguousarray(water.reshape(-1)),
        np.ascontiguousarray(sediment.reshape(-1)),
        np.ascontiguousarray(rock_detail.reshape(-1)),
        np.ascontiguousarray(indices),
    )


def create_basalt_stage(extent: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half = extent * 0.53
    floor_half = extent * 1.35
    top = np.float32(-0.42)
    bottom = np.float32(-0.92)
    positions = np.array(
        (
            (-half, top, -half),
            (-half, bottom, -half),
            (half, bottom, -half),
            (-half, top, -half),
            (half, bottom, -half),
            (half, top, -half),
            (half, top, -half),
            (half, bottom, -half),
            (half, bottom, half),
            (half, top, -half),
            (half, bottom, half),
            (half, top, half),
            (half, top, half),
            (half, bottom, half),
            (-half, bottom, half),
            (half, top, half),
            (-half, bottom, half),
            (-half, top, half),
            (-half, top, half),
            (-half, bottom, half),
            (-half, bottom, -half),
            (-half, top, half),
            (-half, bottom, -half),
            (-half, top, -half),
            (-floor_half, bottom, -floor_half),
            (-floor_half, bottom, floor_half),
            (floor_half, bottom, floor_half),
            (-floor_half, bottom, -floor_half),
            (floor_half, bottom, floor_half),
            (floor_half, bottom, -floor_half),
        ),
        dtype=np.float32,
    )
    normals = np.repeat(
        np.array(
            (
                (0.0, 0.0, -1.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            dtype=np.float32,
        ),
        6,
        axis=0,
    )
    colors = np.tile(np.array((0.012, 0.009, 0.008), dtype=np.float32), (positions.shape[0], 1))
    materials = np.tile(np.array((0.78, 0.08, 0.0), dtype=np.float32), (positions.shape[0], 1))
    return positions, normals, colors, materials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute-driven hydraulic terrain erosion with opaque PBR runoff.")
    parser.add_argument("--arch", choices=("vulkan", "directx", "opengl"), default="vulkan")
    parser.add_argument("--size", type=int, default=720)
    parser.add_argument("--grid", type=int, default=224)
    parser.add_argument("--extent", type=float, default=13.0)
    parser.add_argument("--rainfall", type=float, default=0.012)
    parser.add_argument("--frames", type=int, default=0, help="zero runs until Escape or Q")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size <= 0 or args.grid < 16 or args.extent <= 0.0 or args.frames < 0 or args.fps <= 0:
        raise ValueError("size/fps/extent must be positive, grid >= 16, and frames cannot be negative")
    if args.rainfall < 0.0:
        raise ValueError("rainfall cannot be negative")
    if args.headless and args.frames == 0:
        args.frames = 1

    vd.init(
        arch=architecture_from_name(args.arch),
        api_version=(4, 3) if args.arch == "opengl" else None,
    )
    terrain = create_terrain_grid(args.grid)
    vertex_count = args.grid * args.grid
    height_a = vd.storage.from_numpy(terrain.height)
    height_b = vd.storage.from_numpy(terrain.height)
    water_a = vd.storage.from_numpy(terrain.water)
    water_b = vd.storage.from_numpy(terrain.water)
    sediment_a = vd.storage.from_numpy(terrain.sediment)
    sediment_b = vd.storage.from_numpy(terrain.sediment)
    flow_a = vd.storage.zeros(dtype=vd.f32, shape=(vertex_count, 4))
    flow_b = vd.storage.zeros(dtype=vd.f32, shape=(vertex_count, 4))
    rock_detail = vd.storage.from_numpy(terrain.rock_detail)
    positions = vd.storage.zeros(dtype=vd.f32, shape=(vertex_count, 3))
    normals = vd.storage.zeros(dtype=vd.f32, shape=(vertex_count, 3))
    colors = vd.storage.zeros(dtype=vd.f32, shape=(vertex_count, 3))
    materials = vd.storage.zeros(dtype=vd.f32, shape=(vertex_count, 3))
    indices = vd.storage.from_numpy(terrain.indices)
    draw_count = terrain.indices.shape[0]
    draw_positions = vd.storage.zeros(dtype=vd.f32, shape=(draw_count, 3))
    draw_normals = vd.storage.zeros(dtype=vd.f32, shape=(draw_count, 3))
    draw_colors = vd.storage.zeros(dtype=vd.f32, shape=(draw_count, 3))
    draw_materials = vd.storage.zeros(dtype=vd.f32, shape=(draw_count, 3))
    stage = create_basalt_stage(args.extent)
    stage_positions = vd.storage.from_numpy(stage[0])
    stage_normals = vd.storage.from_numpy(stage[1])
    stage_colors = vd.storage.from_numpy(stage[2])
    stage_materials = vd.storage.from_numpy(stage[3])

    output = vd.Texture.zeros(shape=(args.size, args.size))
    target = vd.RenderTarget(shape=output.shape).attach_color(0, output).attach_depth(format=vd.depth32)
    shadow_size = args.size * 2
    shadow_map = vd.Texture.zeros(shape=(shadow_size, shadow_size), format=vd.depth32)
    shadow_color = vd.Texture.zeros(shape=(shadow_size, shadow_size))
    shadow_target = (
        vd.RenderTarget(shape=shadow_map.shape).attach_color(0, shadow_color).attach_depth(texture=shadow_map)
    )
    shadow_sampler = vd.sampler(address="clamp_to_edge")
    environment_map = load_equirectangular_environment(
        Path(__file__).resolve().parent / "assets" / "environment" / "belfast_sunset_puresky_1k.hdr",
        face_size=256,
        exposure=0.68,
        rotation=-0.48,
    )
    environment_sampler = vd.sampler()
    rock_material = load_rgba_texture(
        Path(__file__).resolve().parent / "assets" / "rock" / "rock_boulder_cracked_packed_1k.png"
    )
    rock_sampler = vd.sampler(address="repeat")
    sky_positions = vd.storage.from_numpy(create_sky_cube())
    render_sky = vd.pipeline(sky_vertex, sky_fragment)
    render = vd.pipeline(pbr_vertex, pbr_fragment, features={"SHADOW", "ENVIRONMENT"})
    render_terrain = vd.pipeline(
        pbr_vertex,
        pbr_fragment,
        features={"SHADOW", "ROCK_TEXTURE"},
    )
    render_shadow = vd.pipeline(shadow_vertex, shadow_fragment)

    flow_a_slot = InvocationSlot()
    erosion_ab_slot = InvocationSlot()
    flow_b_slot = InvocationSlot()
    erosion_ba_slot = InvocationSlot()
    mesh_slot = InvocationSlot()
    expand_slot = InvocationSlot()
    shadow_batch = InvocationBatch()
    main_batch = InvocationBatch()
    graph = vd.ExecutionGraph()
    graph.add_pass(ComputeInvocationPass("water-flow-a", flow_a_slot))
    graph.add_pass(ComputeInvocationPass("erosion-a-to-b", erosion_ab_slot))
    graph.add_pass(ComputeInvocationPass("water-flow-b", flow_b_slot))
    graph.add_pass(ComputeInvocationPass("erosion-b-to-a", erosion_ba_slot))
    graph.add_pass(ComputeInvocationPass("terrain-normal-material", mesh_slot))
    graph.add_pass(ComputeInvocationPass("terrain-draw-expand", expand_slot))
    graph.add_pass(
        BatchRenderPass(
            "terrain-shadow",
            shadow_target,
            shadow_batch,
            clear_color=(1.0, 1.0, 1.0, 1.0),
        )
    )
    graph.add_pass(
        BatchRenderPass(
            "terrain-pbr",
            target,
            main_batch,
            clear_color=(0.006, 0.003, 0.002, 1.0),
        )
    )

    projection = perspective(
        math.radians(40.0),
        1.0,
        0.1,
        45.0,
        zero_to_one=args.arch != "opengl",
    )
    light_position = np.array((-13.5, 7.2, -10.0), dtype=np.float32)
    light_view_projection = np.ascontiguousarray(
        perspective(
            math.radians(49.0),
            1.0,
            1.0,
            38.0,
            zero_to_one=args.arch != "opengl",
        )
        @ look_at(light_position, np.array((0.0, 0.35, 0.0), dtype=np.float32))
    )
    presenter = FramePresenter(
        output,
        architecture=args.arch,
        title="VernonDSL Dynamic Terrain Erosion",
        headless=args.headless,
        fps=args.fps,
    )
    frame = 0
    start = time.perf_counter()
    half_step = np.float32(0.5 / args.fps)
    common_shadow = {
        "light_view_projection": light_view_projection,
        "topology": vd.triangles,
    }
    try:
        while args.frames == 0 or frame < args.frames:
            phase = np.float32(time.perf_counter() - start if args.frames == 0 else frame / args.fps)
            erosion_arguments = (
                np.uint32(args.grid),
                half_step,
                phase,
                np.float32(args.rainfall),
            )
            flow_a_slot.value = compute_erosion_flow.invocation(
                height_a,
                water_a,
                flow_a,
                *erosion_arguments,
                grid=(vertex_count, 1, 1),
            )
            erosion_ab_slot.value = apply_erosion_flow.invocation(
                height_a,
                water_a,
                sediment_a,
                flow_a,
                height_b,
                water_b,
                sediment_b,
                *erosion_arguments,
                grid=(vertex_count, 1, 1),
            )
            flow_b_slot.value = compute_erosion_flow.invocation(
                height_b,
                water_b,
                flow_b,
                *erosion_arguments,
                grid=(vertex_count, 1, 1),
            )
            erosion_ba_slot.value = apply_erosion_flow.invocation(
                height_b,
                water_b,
                sediment_b,
                flow_b,
                height_a,
                water_a,
                sediment_a,
                *erosion_arguments,
                grid=(vertex_count, 1, 1),
            )
            mesh_slot.value = build_terrain_mesh.invocation(
                height_a,
                water_a,
                sediment_a,
                rock_detail,
                positions,
                normals,
                colors,
                materials,
                np.uint32(args.grid),
                np.float32(args.extent),
                phase,
                grid=(vertex_count, 1, 1),
            )
            expand_slot.value = expand_indexed_mesh.invocation(
                positions,
                normals,
                colors,
                materials,
                indices,
                draw_positions,
                draw_normals,
                draw_colors,
                draw_materials,
                grid=(draw_count, 1, 1),
            )
            angle = float(phase) * 0.025 - 1.88
            camera = np.array((11.8 * math.cos(angle), 6.15, 11.8 * math.sin(angle)), dtype=np.float32)
            view_projection = np.ascontiguousarray(
                projection @ look_at(camera, np.array((0.0, -1.42, 0.2), dtype=np.float32))
            )
            shadow_batch.values = [
                render_shadow.invocation(position=draw_positions, **common_shadow),
                render_shadow.invocation(position=stage_positions, **common_shadow),
            ]
            common_lighting: dict[str, object] = {
                "view_projection": view_projection,
                "light_view_projection": light_view_projection,
                "camera_position": camera,
                "light_position": light_position,
                "shadow_depth_scale": np.float32(0.5 if args.arch == "opengl" else 1.0),
                "shadow_depth_bias": np.float32(0.5 if args.arch == "opengl" else 0.0),
                "shadow_uv_scale": np.array((0.5, 0.5 if args.arch == "opengl" else -0.5), dtype=np.float32),
                "shadow_texel_size": np.array((1.0 / shadow_size, 1.0 / shadow_size), dtype=np.float32),
                "shadow_map": shadow_map,
                "shadow_sampler": shadow_sampler,
                "topology": vd.triangles,
            }
            common_pbr: dict[str, object] = {
                **common_lighting,
                "environment_map": environment_map,
                "environment_sampler": environment_sampler,
            }
            main_batch.values = [
                render_sky.invocation(
                    direction=sky_positions,
                    view_projection=view_projection,
                    camera_position=camera,
                    environment_map=environment_map,
                    environment_sampler=environment_sampler,
                    topology=vd.triangles,
                ),
                render_terrain.invocation(
                    position=draw_positions,
                    normal=draw_normals,
                    base_color=draw_colors,
                    material=draw_materials,
                    rock_material=rock_material,
                    rock_sampler=rock_sampler,
                    **common_lighting,
                ),
                render.invocation(
                    position=stage_positions,
                    normal=stage_normals,
                    base_color=stage_colors,
                    material=stage_materials,
                    **common_pbr,
                ),
            ]
            graph.execute()
            frame += 1
            if not presenter.present():
                break
    finally:
        presenter.close()
    presenter.write(args.output)
    barrier_count = sum(len(scope.barriers) for scope in graph.scopes)
    print(
        f"backend={args.arch} frames={frame} grid={args.grid} rainfall={args.rainfall} "
        f"passes={len(graph.schedule)} barriers={barrier_count}"
    )


if __name__ == "__main__":
    main()
