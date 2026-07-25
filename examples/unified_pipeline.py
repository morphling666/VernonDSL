from __future__ import annotations

import argparse
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import vernon_dsl as vd


@vd.kernel(workgroup_size=(2, 1, 1))
def animate_vertices(
    positions: vd.TensorView[vd.f32, 2, vd.write],
    base_positions: vd.TensorView[vd.f32, 2, vd.read],
    phase: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    component = gid[0]
    vertex = gid[1]
    angle = phase + vd.f32(vertex) * 2.1
    if component == 0:
        positions[vertex, component] = base_positions[vertex, component] + vd.sin(angle) * 0.16
    else:
        positions[vertex, component] = base_positions[vertex, component] + vd.cos(angle * 1.3) * 0.1


@vd.vertex
def vertex_main(
    positions: Annotated[vd.Vector[vd.f32, 2], vd.location(0)],
    draw_offset: Annotated[vd.Vector[vd.f32, 2], vd.uniform()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([positions + draw_offset, 0.0, 1.0])


@vd.fragment
def fragment_main(
    color: Annotated[vd.Vector[vd.f32, 4], vd.uniform()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.location(0)]:
    return color


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run separate VernonDSL compute and graphics programs.")
    parser.add_argument(
        "--arch",
        choices=("opengl", "opengles", "vulkan"),
        default="vulkan",
        help="graphics backend; OpenGL uses a hidden packaged GLFW context",
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="number of frames; zero runs until Escape or Q",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, help="optional screenshot path, for example frame.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size <= 0 or args.frames < 0 or args.fps <= 0:
        raise ValueError("--size and --fps must be positive; --frames cannot be negative")
    architecture = {
        "opengl": vd.opengl,
        "opengles": vd.opengles,
        "vulkan": vd.vulkan,
    }[args.arch]
    vd.init(
        arch=architecture,
        api_version=(4, 3) if args.arch == "opengl" else (3, 1) if args.arch == "opengles" else None,
    )

    base_array = np.array(
        (
            (-0.75, -0.7),
            (0.75, -0.7),
            (0.0, 0.75),
        ),
        dtype=np.float32,
    )
    base_positions = vd.storage.from_numpy(base_array)
    positions = vd.storage.from_numpy(base_array)
    draw_offset = vd.storage.from_numpy(np.zeros((2,), dtype=np.float32))
    color = vd.storage.from_numpy(np.array((0.1, 0.65, 1.0, 1.0), dtype=np.float32))
    target = vd.Texture.zeros(shape=(args.size, args.size))
    render = vd.pipeline(vertex_main, fragment_main)

    frame = 0
    image: np.ndarray | None = None
    delay_ms = max(1, round(1000 / args.fps))
    window_name = f"VernonDSL unified pipeline ({args.arch})"
    try:
        while args.frames == 0 or frame < args.frames:
            phase = frame / args.fps
            draw_offset.copy_from_numpy(
                np.array(
                    (
                        np.sin(phase * 0.8) * 0.1,
                        np.cos(phase * 0.6) * 0.05,
                    ),
                    dtype=np.float32,
                )
            )
            color.copy_from_numpy(
                np.array(
                    (
                        0.5 + np.sin(phase * 1.7) * 0.5,
                        0.5 + np.sin(phase * 2.3 + 2.0) * 0.5,
                        0.5 + np.sin(phase * 1.3 + 4.0) * 0.5,
                        1.0,
                    ),
                    dtype=np.float32,
                )
            )
            animate_vertices(positions, base_positions, np.float32(phase))
            render(
                positions=positions,
                draw_offset=draw_offset,
                color=color,
                target=target,
            )
            rgba = target.to_numpy()
            if args.arch in {"opengl", "opengles"}:
                rgba = np.flipud(rgba)
            rgba = np.ascontiguousarray(rgba)
            image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            if not args.headless:
                cv2.imshow(window_name, image)
            frame += 1
            key = cv2.waitKey(delay_ms) & 0xFF if not args.headless else -1
            if key in (27, ord("q")):
                break
    finally:
        if not args.headless:
            cv2.destroyAllWindows()

    if args.output is not None and image is not None:
        if not cv2.imwrite(str(args.output), image):
            raise RuntimeError(f"cannot write screenshot to {args.output}")
    print(f"backend={args.arch} frames={frame}")


if __name__ == "__main__":
    main()
