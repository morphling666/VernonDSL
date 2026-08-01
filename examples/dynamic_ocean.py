from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vernon_dsl as vd
from shader_lib.mesh import expand_indexed_mesh
from shader_lib.ocean import (
    build_ocean_mesh,
    ocean_fragment,
    ocean_vertex,
    step_ocean,
)
from shader_lib.pbr import pbr_fragment, pbr_vertex, shadow_fragment, shadow_vertex
from shader_lib.showcase import sky_fragment, sky_vertex
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
class OceanGrid:
    height: np.ndarray
    positions: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    materials: np.ndarray
    indices: np.ndarray


def create_ocean_grid(grid: int, extent: float) -> OceanGrid:
    if grid < 8 or extent <= 0.0:
        raise ValueError("ocean grid must be at least 8 and extent must be positive")
    coordinate = np.linspace(-0.5, 0.5, grid, dtype=np.float32)
    xx, zz = np.meshgrid(coordinate, coordinate, indexing="xy")
    radius = np.sqrt(xx * xx + zz * zz)
    height = (
        np.sin(xx * np.float32(19.0) + zz * np.float32(7.0)) * np.float32(0.07)
        + np.cos(xx * np.float32(-8.0) + zz * np.float32(15.0)) * np.float32(0.05)
        + np.sin(radius * np.float32(38.0)) * np.exp(-radius * np.float32(2.8)) * np.float32(0.08)
    ).astype(np.float32)
    edge = np.minimum.reduce((xx + 0.5, 0.5 - xx, zz + 0.5, 0.5 - zz))
    height *= np.clip(edge * np.float32(10.0), 0.0, 1.0)
    positions = np.column_stack(
        (
            xx.reshape(-1) * np.float32(extent),
            height.reshape(-1),
            zz.reshape(-1) * np.float32(extent),
        )
    ).astype(np.float32)
    normals = np.tile(np.array((0.0, 1.0, 0.0), dtype=np.float32), (grid * grid, 1))
    colors = np.tile(np.array((0.02, 0.2, 0.33), dtype=np.float32), (grid * grid, 1))
    materials = np.tile(np.array((0.12, 0.72, 0.0), dtype=np.float32), (grid * grid, 1))
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
    return OceanGrid(
        np.ascontiguousarray(height.reshape(-1)),
        np.ascontiguousarray(positions),
        np.ascontiguousarray(normals),
        np.ascontiguousarray(colors),
        np.ascontiguousarray(materials),
        np.ascontiguousarray(indices),
    )


def create_seabed(extent: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half = extent * 0.49
    positions = np.array(
        ((-half, -0.62, -half), (-half, -0.62, half), (half, -0.62, half), (half, -0.62, -half)),
        dtype=np.float32,
    )
    normals = np.tile(np.array((0.0, 1.0, 0.0), dtype=np.float32), (4, 1))
    colors = np.tile(np.array((0.012, 0.025, 0.065), dtype=np.float32), (4, 1))
    materials = np.tile(np.array((0.7, 0.08, 0.9), dtype=np.float32), (4, 1))
    indices = np.array((0, 1, 2, 0, 2, 3), dtype=np.uint32)
    return positions, normals, colors, materials, indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute-driven holographic ocean with PBR, shadows, and cubemap IBL.")
    parser.add_argument("--arch", choices=("vulkan", "directx", "metal", "opengl"), default="vulkan")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--grid", type=int, default=192)
    parser.add_argument("--extent", type=float, default=14.0)
    parser.add_argument("--wave-speed", type=float, default=48.0)
    parser.add_argument("--damping", type=float, default=0.992)
    parser.add_argument("--frames", type=int, default=0, help="zero runs until Escape or Q")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size <= 0 or args.grid < 8 or args.extent <= 0.0 or args.frames < 0 or args.fps <= 0:
        raise ValueError("size/fps/extent must be positive, grid >= 8, and frames cannot be negative")
    if args.wave_speed <= 0.0 or not 0.0 < args.damping <= 1.0:
        raise ValueError("wave speed must be positive and damping must be in (0, 1]")
    if args.headless and args.frames == 0:
        args.frames = 1

    vd.init(
        arch=architecture_from_name(args.arch),
        api_version=(4, 3) if args.arch == "opengl" else None,
    )
    ocean = create_ocean_grid(args.grid, args.extent)
    vertex_count = args.grid * args.grid
    height_a = vd.storage.from_numpy(ocean.height)
    height_b = vd.storage.from_numpy(ocean.height)
    velocity_a = vd.storage.zeros(dtype=vd.f32, shape=(vertex_count,))
    velocity_b = vd.storage.zeros(dtype=vd.f32, shape=(vertex_count,))
    positions = vd.storage.from_numpy(ocean.positions)
    normals = vd.storage.from_numpy(ocean.normals)
    colors = vd.storage.from_numpy(ocean.colors)
    materials = vd.storage.from_numpy(ocean.materials)
    indices = vd.storage.from_numpy(ocean.indices)
    draw_count = ocean.indices.shape[0]
    draw_positions = vd.storage.zeros(dtype=vd.f32, shape=(draw_count, 3))
    draw_normals = vd.storage.zeros(dtype=vd.f32, shape=(draw_count, 3))
    draw_colors = vd.storage.zeros(dtype=vd.f32, shape=(draw_count, 3))
    draw_materials = vd.storage.zeros(dtype=vd.f32, shape=(draw_count, 3))
    seabed = create_seabed(args.extent)
    seabed_draw_order = seabed[4]
    seabed_positions = vd.storage.from_numpy(np.ascontiguousarray(seabed[0][seabed_draw_order]))
    seabed_normals = vd.storage.from_numpy(np.ascontiguousarray(seabed[1][seabed_draw_order]))
    seabed_colors = vd.storage.from_numpy(np.ascontiguousarray(seabed[2][seabed_draw_order]))
    seabed_materials = vd.storage.from_numpy(np.ascontiguousarray(seabed[3][seabed_draw_order]))

    output = vd.Texture.zeros(shape=(args.size, args.size))
    target = vd.RenderTarget(shape=output.shape).attach_color(0, output).attach_depth(format=vd.depth32)
    shadow_map = vd.Texture.zeros(shape=(args.size, args.size), format=vd.depth32)
    shadow_color = vd.Texture.zeros(shape=(args.size, args.size))
    shadow_target = (
        vd.RenderTarget(shape=shadow_map.shape).attach_color(0, shadow_color).attach_depth(texture=shadow_map)
    )
    shadow_sampler = vd.sampler(address="clamp_to_edge")
    environment_map = load_equirectangular_environment(
        Path(__file__).resolve().parent / "assets" / "environment" / "belfast_sunset_puresky_1k.hdr",
        face_size=256,
        exposure=0.52,
        rotation=2.78,
    )
    environment_sampler = vd.sampler()
    normal_map = load_rgba_texture(Path(__file__).resolve().parent / "assets" / "water" / "Foam003_1K_NormalGL.jpg")
    normal_sampler = vd.sampler(address="repeat")
    sky_positions = vd.storage.from_numpy(create_sky_cube())
    render_sky = vd.pipeline(sky_vertex, sky_fragment)
    render_ocean = vd.pipeline(ocean_vertex, ocean_fragment)
    render = vd.pipeline(pbr_vertex, pbr_fragment, features={"SHADOW", "ENVIRONMENT"})
    render_shadow = vd.pipeline(shadow_vertex, shadow_fragment)

    wave_ab_slot = InvocationSlot()
    wave_ba_slot = InvocationSlot()
    mesh_slot = InvocationSlot()
    expand_slot = InvocationSlot()
    shadow_batch = InvocationBatch()
    main_batch = InvocationBatch()
    graph = vd.ExecutionGraph()
    graph.add_pass(ComputeInvocationPass("ocean-wave-a-to-b", wave_ab_slot))
    graph.add_pass(ComputeInvocationPass("ocean-wave-b-to-a", wave_ba_slot))
    graph.add_pass(ComputeInvocationPass("ocean-mesh-rebuild", mesh_slot))
    graph.add_pass(ComputeInvocationPass("ocean-draw-expand", expand_slot))
    graph.add_pass(
        BatchRenderPass(
            "ocean-shadow",
            shadow_target,
            shadow_batch,
            clear_color=(1.0, 1.0, 1.0, 1.0),
        )
    )
    graph.add_pass(
        BatchRenderPass(
            "ocean-pbr",
            target,
            main_batch,
            clear_color=(0.001, 0.004, 0.013, 1.0),
        )
    )

    projection = perspective(
        math.radians(44.0),
        1.0,
        0.1,
        40.0,
        zero_to_one=args.arch != "opengl",
    )
    # This direction matches the sun in belfast_sunset_puresky after the
    # equirectangular rotation above, keeping direct and reflected highlights aligned.
    light_position = np.array((-10.49, 0.68, -17.01), dtype=np.float32)
    light_view_projection = np.ascontiguousarray(
        perspective(
            math.radians(54.0),
            1.0,
            1.0,
            24.0,
            zero_to_one=args.arch != "opengl",
        )
        @ look_at(light_position, np.array((0.0, -0.1, 0.0), dtype=np.float32))
    )
    presenter = FramePresenter(
        output,
        architecture=args.arch,
        title="VernonDSL Holographic Ocean",
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
            wave_arguments = (
                np.uint32(args.grid),
                half_step,
                phase,
                np.float32(args.wave_speed),
                np.float32(args.damping),
            )
            wave_ab_slot.value = step_ocean.invocation(
                height_a,
                velocity_a,
                height_b,
                velocity_b,
                *wave_arguments,
                grid=(vertex_count, 1, 1),
            )
            wave_ba_slot.value = step_ocean.invocation(
                height_b,
                velocity_b,
                height_a,
                velocity_a,
                *wave_arguments,
                grid=(vertex_count, 1, 1),
            )
            mesh_slot.value = build_ocean_mesh.invocation(
                height_a,
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
            angle = float(phase) * 0.13 + 0.72
            camera = np.array((5.15 * math.cos(angle), 1.12, 5.15 * math.sin(angle)), dtype=np.float32)
            view_projection = np.ascontiguousarray(
                projection @ look_at(camera, np.array((0.0, 0.02, 0.0), dtype=np.float32))
            )
            shadow_batch.values = [
                render_shadow.invocation(position=draw_positions, **common_shadow),
                render_shadow.invocation(position=seabed_positions, **common_shadow),
            ]
            common_pbr: dict[str, object] = {
                "view_projection": view_projection,
                "light_view_projection": light_view_projection,
                "camera_position": camera,
                "light_position": light_position,
                "shadow_depth_scale": np.float32(0.5 if args.arch == "opengl" else 1.0),
                "shadow_depth_bias": np.float32(0.5 if args.arch == "opengl" else 0.0),
                "shadow_uv_scale": np.array((0.5, 0.5 if args.arch == "opengl" else -0.5), dtype=np.float32),
                "shadow_texel_size": np.array((1.0 / args.size, 1.0 / args.size), dtype=np.float32),
                "shadow_map": shadow_map,
                "shadow_sampler": shadow_sampler,
                "environment_map": environment_map,
                "environment_sampler": environment_sampler,
                "topology": vd.triangles,
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
                render_ocean.invocation(
                    position=draw_positions,
                    normal=draw_normals,
                    surface_data=draw_colors,
                    view_projection=view_projection,
                    camera_position=camera,
                    light_position=light_position,
                    environment_map=environment_map,
                    environment_sampler=environment_sampler,
                    normal_map=normal_map,
                    normal_sampler=normal_sampler,
                    phase=phase,
                    topology=vd.triangles,
                ),
                render.invocation(
                    position=seabed_positions,
                    normal=seabed_normals,
                    base_color=seabed_colors,
                    material=seabed_materials,
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
    print(f"backend={args.arch} frames={frame} grid={args.grid} passes={len(graph.schedule)} barriers={barrier_count}")


if __name__ == "__main__":
    main()
