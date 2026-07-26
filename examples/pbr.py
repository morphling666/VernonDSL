from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np
import vernon_dsl as vd
from shader_lib.pbr import pbr_fragment, pbr_vertex


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a procedural metallic-roughness PBR cube and plane.")
    parser.add_argument(
        "--arch",
        choices=("vulkan", "directx", "opengl"),
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
    }[args.arch]
    window_name = "VernonDSL PBR"
    vd.init(
        arch=architecture,
        api_version=(4, 3) if args.arch == "opengl" else None,
    )

    mesh = create_scene_mesh()
    draw_vertex_count = mesh.triangles.size
    draw_order = np.ascontiguousarray(mesh.triangles.reshape(-1))
    positions = vd.storage.from_numpy(np.ascontiguousarray(mesh.positions[draw_order]))
    normals = vd.storage.from_numpy(np.ascontiguousarray(mesh.normals[draw_order]))
    colors = vd.storage.from_numpy(np.ascontiguousarray(mesh.colors[draw_order]))
    materials = vd.storage.from_numpy(np.ascontiguousarray(mesh.materials[draw_order]))
    light_value = np.array((3.0, 5.0, 4.0), dtype=np.float32)
    target = vd.Texture.zeros(shape=(args.size, args.size))
    depth = vd.DepthTexture.zeros(shape=(args.size, args.size))
    features = set()
    if not args.no_shadow:
        features.add("SHADOW")
    if not args.no_cubemap:
        features.add("ENVIRONMENT")
    render = vd.pipeline(pbr_vertex, pbr_fragment, features=features)

    projection = perspective(
        math.radians(48.0),
        1.0,
        0.1,
        30.0,
        zero_to_one=args.arch != "opengl",
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
            render(
                position=positions,
                normal=normals,
                base_color=colors,
                material=materials,
                view_projection=np.ascontiguousarray(view_projection),
                camera_position=np.ascontiguousarray(camera),
                light_position=light_value,
                topology=vd.triangles,
                target=target,
                depth=depth,
            )
            rgba = target.to_numpy()
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
        f"shadow={'off' if args.no_shadow else 'analytic'} "
        f"environment={'off' if args.no_cubemap else 'procedural'} "
        f"compiled={render.compile_count}"
    )


if __name__ == "__main__":
    main()
