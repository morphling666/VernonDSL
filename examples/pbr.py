from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np
import vernon_dsl as vd
from shader_lib.pbr import pbr_fragment, pbr_vertex, shadow_fragment, shadow_vertex


@dataclass(frozen=True)
class SceneMesh:
    positions: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    materials: np.ndarray
    triangles: np.ndarray


def _append_quad(
    positions: list[tuple[float, float, float]],
    normals: list[tuple[float, float, float]],
    colors: list[tuple[float, float, float]],
    materials: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    corners: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    normal: tuple[float, float, float],
    color: tuple[float, float, float],
    material: tuple[float, float, float],
) -> None:
    first = len(positions)
    positions.extend(corners)
    normals.extend((normal,) * 4)
    colors.extend((color,) * 4)
    materials.extend((material,) * 4)
    triangles.extend(((first, first + 1, first + 2), (first, first + 2, first + 3)))


def create_scene_mesh(plane_subdivisions: int = 14) -> SceneMesh:
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float]] = []
    materials: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []

    cube_color = (0.92, 0.28, 0.08)
    cube_material = (0.22, 0.78, 1.0)
    x0, x1 = -0.55, 0.55
    y0, y1 = 0.08, 1.18
    z0, z1 = -0.55, 0.55
    cube_faces = (
        (((x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1)), (1.0, 0.0, 0.0)),
        (((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)), (-1.0, 0.0, 0.0)),
        (((x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)), (0.0, 1.0, 0.0)),
        (((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)), (0.0, -1.0, 0.0)),
        (((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)), (0.0, 0.0, 1.0)),
        (((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0)), (0.0, 0.0, -1.0)),
    )
    for corners, face_normal in cube_faces:
        _append_quad(
            positions,
            normals,
            colors,
            materials,
            triangles,
            corners,
            face_normal,
            cube_color,
            cube_material,
        )

    plane_color = (0.34, 0.37, 0.40)
    plane_material = (0.82, 0.02, 0.0)
    extent = 3.2
    coordinates = np.linspace(-extent, extent, plane_subdivisions + 1)
    for z_index in range(plane_subdivisions):
        for x_index in range(plane_subdivisions):
            left = float(coordinates[x_index])
            right = float(coordinates[x_index + 1])
            near = float(coordinates[z_index])
            far = float(coordinates[z_index + 1])
            _append_quad(
                positions,
                normals,
                colors,
                materials,
                triangles,
                (
                    (left, 0.0, near),
                    (left, 0.0, far),
                    (right, 0.0, far),
                    (right, 0.0, near),
                ),
                (0.0, 1.0, 0.0),
                plane_color,
                plane_material,
            )

    return SceneMesh(
        np.ascontiguousarray(positions, dtype=np.float32),
        np.ascontiguousarray(normals, dtype=np.float32),
        np.ascontiguousarray(colors, dtype=np.float32),
        np.ascontiguousarray(materials, dtype=np.float32),
        np.ascontiguousarray(triangles, dtype=np.uint32),
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = _normalize(target - eye)
    right = _normalize(np.cross(forward, np.array((0.0, 1.0, 0.0), dtype=np.float32)))
    up = np.cross(right, forward)
    result = np.eye(4, dtype=np.float32)
    result[0, :3] = right
    result[1, :3] = up
    result[2, :3] = -forward
    result[0, 3] = -np.dot(right, eye)
    result[1, 3] = -np.dot(up, eye)
    result[2, 3] = np.dot(forward, eye)
    return result


def perspective(vertical_fov: float, aspect: float, near: float, far: float, *, zero_to_one: bool) -> np.ndarray:
    focal_length = 1.0 / math.tan(vertical_fov * 0.5)
    result = np.zeros((4, 4), dtype=np.float32)
    result[0, 0] = focal_length / aspect
    result[1, 1] = focal_length
    if zero_to_one:
        result[2, 2] = far / (near - far)
        result[2, 3] = far * near / (near - far)
    else:
        result[2, 2] = (far + near) / (near - far)
        result[2, 3] = 2.0 * far * near / (near - far)
    result[3, 2] = -1.0
    return result


def create_environment_cube(size: int = 16) -> vd.Texture:
    faces = np.empty((6, size, size, 4), dtype=np.uint8)
    colors = (
        (180, 205, 245, 255),
        (52, 65, 82, 255),
        (110, 155, 235, 255),
        (22, 25, 31, 255),
        (100, 135, 190, 255),
        (44, 52, 66, 255),
    )
    for index, color in enumerate(colors):
        faces[index] = color
    return vd.Texture.cube(faces)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a procedural metallic-roughness PBR cube and plane.")
    parser.add_argument(
        "--arch",
        "--architecture",
        dest="arch",
        choices=("vulkan", "directx", "opengl", "metal"),
        default="vulkan",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="zero runs until Escape or Q")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-shadow", action="store_true")
    parser.add_argument("--no-cubemap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size <= 0 or args.frames < 0 or args.fps <= 0:
        raise ValueError("size and fps must be positive, and frames cannot be negative")
    if args.headless and args.frames == 0:
        args.frames = 1

    architecture = {
        "vulkan": vd.vulkan,
        "directx": vd.directx,
        "opengl": vd.opengl,
        "metal": vd.metal,
    }[args.arch]
    window_name = "VernonDSL PBR"
    vd.init(
        arch=architecture,
        api_version=(3, 3) if args.arch == "opengl" else None,
    )

    mesh = create_scene_mesh()
    draw_order = np.ascontiguousarray(mesh.triangles.reshape(-1))
    positions = vd.storage.from_numpy(np.ascontiguousarray(mesh.positions[draw_order]))
    normals = vd.storage.from_numpy(np.ascontiguousarray(mesh.normals[draw_order]))
    colors = vd.storage.from_numpy(np.ascontiguousarray(mesh.colors[draw_order]))
    materials = vd.storage.from_numpy(np.ascontiguousarray(mesh.materials[draw_order]))
    light_value = np.array((3.0, 5.0, 4.0), dtype=np.float32)
    color = vd.Texture.zeros(shape=(args.size, args.size))
    target = vd.RenderTarget.from_attachments(
        colors={0: color},
        depth=vd.Texture.device(shape=color.shape, format=vd.d32_float),
    )
    shadow_enabled = not args.no_shadow
    environment_enabled = not args.no_cubemap
    shadow_target: vd.RenderTarget | None = None
    shadow_map: object | None = None
    shadow_sampler: object | None = None
    environment_map: object | None = None
    environment_sampler: object | None = None
    features = {
        feature
        for feature, enabled in (
            ("SHADOW", shadow_enabled),
            ("ENVIRONMENT", environment_enabled),
        )
        if enabled
    }
    if shadow_enabled:
        shadow_color = vd.Texture.zeros(shape=(args.size, args.size))
        shadow_target = vd.RenderTarget.from_attachments(
            colors={0: shadow_color},
            depth=vd.Texture.device(shape=shadow_color.shape, format=vd.d32_float),
        )
        shadow_map = shadow_target.depth_texture
        shadow_sampler = vd.sampler(address="clamp_to_edge")
    if environment_enabled:
        environment_map = create_environment_cube()
        environment_sampler = vd.sampler()
    opaque_state = vd.graphics_state(
        rasterization=vd.RasterizationState(cull_mode=vd.CullMode.BACK),
        depth_stencil=vd.DepthStencilState(depth_test=True, depth_write=True),
    )
    render = vd.pipeline(pbr_vertex, pbr_fragment, state=opaque_state, features=features)
    render_shadow = vd.pipeline(shadow_vertex, shadow_fragment, state=opaque_state) if shadow_enabled else None

    projection = perspective(
        math.radians(48.0),
        1.0,
        0.1,
        30.0,
        zero_to_one=args.arch != "opengl",
    )
    light_view_projection: np.ndarray | None = None
    if shadow_enabled:
        light_view = look_at(light_value, np.array((0.0, 0.35, 0.0), dtype=np.float32))
        light_projection = perspective(
            math.radians(62.0),
            1.0,
            0.8,
            16.0,
            zero_to_one=args.arch != "opengl",
        )
        light_view_projection = np.ascontiguousarray(light_projection @ light_view)
        if render_shadow is None:
            raise RuntimeError("shadow pipeline was not prepared")
    render_arguments: dict[str, object] = {
        "position": positions,
        "normal": normals,
        "base_color": colors,
        "material": materials,
        "light_position": light_value,
    }
    if shadow_enabled:
        assert light_view_projection is not None
        assert shadow_map is not None
        assert shadow_sampler is not None
        render_arguments.update(
            light_view_projection=light_view_projection,
            shadow_depth_scale=np.float32(0.5 if args.arch == "opengl" else 1.0),
            shadow_depth_bias=np.float32(0.5 if args.arch == "opengl" else 0.0),
            shadow_uv_scale=np.array(
                (0.5, 0.5 if args.arch == "opengl" else -0.5),
                dtype=np.float32,
            ),
            shadow_texel_size=np.array((1.0 / args.size, 1.0 / args.size), dtype=np.float32),
            shadow_map=shadow_map,
            shadow_sampler=shadow_sampler,
        )
    if environment_enabled:
        assert environment_map is not None
        assert environment_sampler is not None
        render_arguments.update(
            environment_map=environment_map,
            environment_sampler=environment_sampler,
        )
    start_time = time.perf_counter()
    delay_ms = max(1, round(1000 / args.fps))
    frame = 0
    image: np.ndarray | None = None
    try:
        while args.frames == 0 or frame < args.frames:
            elapsed = time.perf_counter() - start_time
            angle = elapsed * 0.28 + 0.65
            camera = np.array(
                (4.8 * math.cos(angle), 3.15, 4.8 * math.sin(angle)),
                dtype=np.float32,
            )
            view = look_at(camera, np.array((0.0, 0.45, 0.0), dtype=np.float32))
            view_projection = projection @ view
            if shadow_enabled:
                assert render_shadow is not None and shadow_target is not None and light_view_projection is not None
                render_shadow(
                    position=positions,
                    light_view_projection=light_view_projection,
                    render_pass=vd.render_pass(
                        shadow_target,
                        color=vd.clear((1.0, 1.0, 1.0, 1.0)),
                        depth=vd.clear_depth(1.0),
                    ),
                )
            render(
                **render_arguments,
                view_projection=np.ascontiguousarray(view_projection),
                camera_position=np.ascontiguousarray(camera),
                render_pass=vd.render_pass(
                    target,
                    color=vd.clear((0.02, 0.025, 0.04, 1.0)),
                    depth=vd.clear_depth(1.0),
                ),
            )
            rgba = color.to_numpy()
            if args.arch == "opengl":
                rgba = np.flipud(rgba)
            image = cv2.cvtColor(np.ascontiguousarray(rgba), cv2.COLOR_RGBA2BGRA)
            frame += 1
            if not args.headless:
                cv2.imshow(window_name, image)
                if cv2.waitKey(delay_ms) & 0xFF in (27, ord("q")):
                    break
    finally:
        if not args.headless:
            cv2.destroyAllWindows()

    if args.output is not None and image is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), image):
            raise RuntimeError(f"cannot write {args.output}")
    print(
        f"backend={args.arch} frames={frame} "
        f"shadow={'off' if args.no_shadow else 'map'} "
        f"environment={'off' if args.no_cubemap else 'cubemap'} "
        f"passes={2 if shadow_enabled else 1} barriers=0 "
        f"compiled={render.compile_count + (render_shadow.compile_count if render_shadow is not None else 0)}"
    )


if __name__ == "__main__":
    main()
