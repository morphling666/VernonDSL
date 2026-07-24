from __future__ import annotations

import argparse
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import vernon_dsl as vd

PICKING = vd.feature("PICKING")


@vd.struct
class VertexData:
    position: Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]
    local_color: Annotated[vd.Vector[vd.f32, 2], vd.location(0)]


@vd.struct
class GBuffer:
    color: Annotated[vd.Vector[vd.f32, 4], vd.location(0)]
    object_id: Annotated[vd.Vector[vd.f32, 4], vd.location(1)]


@vd.vertex
def vertex_main(
    position: Annotated[vd.Vector[vd.f32, 2], vd.location(0)],
    offset: Annotated[vd.Vector[vd.f32, 2], vd.instance(location=1)],
) -> VertexData:
    clip_position = vd.Vector([position + offset, 0.0, 1.0])
    local_color = position + vd.Vector([0.5, 0.5])
    return VertexData(clip_position, local_color)


@vd.fragment
def fragment_main(
    local_color: Annotated[vd.Vector[vd.f32, 2], vd.varying(), vd.location(0)],
) -> GBuffer:
    color = vd.Vector([local_color, 1.0, 1.0])
    object_id = vd.Vector([0.0, 0.0, 0.0, 1.0])
    if PICKING:
        object_id = vd.Vector([1.0, 0.25, 0.0, 1.0])
    return GBuffer(color, object_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run indexed, instanced, variant MRT rendering.")
    parser.add_argument(
        "--arch",
        choices=("opengl", "opengles", "vulkan"),
        default="vulkan",
        help=("graphics backend; OpenGL profiles require host context registration"),
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--instances", type=int, default=7)
    parser.add_argument("--frames", type=int, default=0, help="zero runs until Escape or Q")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--no-picking", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--id-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size <= 0 or args.instances <= 0 or args.frames < 0 or args.fps <= 0:
        raise ValueError("size, instances, and fps must be positive")
    architecture = {
        "opengl": vd.opengl,
        "opengles": vd.opengles,
        "vulkan": vd.vulkan,
    }[args.arch]
    vd.init(
        arch=architecture,
        api_version=(3, 3) if args.arch in {"opengl", "opengles"} else None,
    )
    render = vd.pipeline(
        vertex_main,
        fragment_main,
        features=() if args.no_picking else {"PICKING"},
    )

    positions = vd.storage.from_numpy(
        np.array(
            ((-0.09, -0.09), (0.09, -0.09), (0.09, 0.09), (-0.09, 0.09)),
            dtype=np.float32,
        )
    )
    indices = vd.storage.from_numpy(np.array((0, 1, 2, 0, 2, 3), dtype=np.uint32))
    offsets = vd.storage.zeros(dtype=vd.f32, shape=(args.instances, 2))
    color = vd.Texture.zeros(shape=(args.size, args.size))
    object_id = vd.Texture.zeros(shape=(args.size, args.size))
    base_x = np.linspace(-0.75, 0.75, args.instances, dtype=np.float32)

    frame = 0
    delay_ms = max(1, round(1000 / args.fps))
    color_image: np.ndarray | None = None
    id_image: np.ndarray | None = None
    try:
        while args.frames == 0 or frame < args.frames:
            phase = frame / args.fps
            instance_values = np.column_stack(
                (base_x, np.sin(base_x * np.float32(5.0) + np.float32(phase * 2.0)) * np.float32(0.35))
            ).astype(np.float32)
            offsets.copy_from_numpy(np.ascontiguousarray(instance_values, dtype=np.float32))
            render(
                position=positions,
                offset=offsets,
                indices=indices,
                topology=vd.triangles,
                targets={
                    "object_id": object_id,
                    "color": color,
                },
            )
            color_rgba = color.to_numpy()
            id_rgba = object_id.to_numpy()
            if args.arch in {"opengl", "opengles"}:
                color_rgba = np.flipud(color_rgba)
                id_rgba = np.flipud(id_rgba)
            color_rgba = np.ascontiguousarray(color_rgba)
            id_rgba = np.ascontiguousarray(id_rgba)
            color_image = cv2.cvtColor(color_rgba, cv2.COLOR_RGBA2BGRA)
            id_image = cv2.cvtColor(id_rgba, cv2.COLOR_RGBA2BGRA)
            frame += 1
            if not args.headless:
                cv2.imshow("VernonDSL advanced color", color_image)
                if cv2.waitKey(delay_ms) & 0xFF in (27, ord("q")):
                    break
    finally:
        if not args.headless:
            cv2.destroyAllWindows()

    if args.output is not None and color_image is not None:
        if not cv2.imwrite(str(args.output), color_image):
            raise RuntimeError(f"cannot write {args.output}")
    if args.id_output is not None and id_image is not None:
        if not cv2.imwrite(str(args.id_output), id_image):
            raise RuntimeError(f"cannot write {args.id_output}")
    print(
        f"backend={args.arch} frames={frame} instances={args.instances} "
        f"variant={'base' if args.no_picking else 'PICKING'} "
        f"compiled={render.compile_count}"
    )


if __name__ == "__main__":
    main()
