from __future__ import annotations

import argparse
from functools import cache
from pathlib import Path
from typing import Annotated

import numpy as np
import vernon_dsl as vd

WIDTH = 640
HEIGHT = 320


class RenderDefaults:
    def __init__(self, *, frames: int, time_step: float, fps: int):
        self.frames = frames
        self.time_step = time_step
        self.fps = fps


@cache
def render_defaults() -> RenderDefaults:
    return RenderDefaults(frames=1_000_000, time_step=0.03, fps=30)


@vd.func
def complex_square(z: vd.Vector[vd.f32, 2]) -> vd.Vector[vd.f32, 2]:
    return vd.Vector(
        [
            z[0] ** 2 - z[1] ** 2,
            z[1] * z[0] * 2,
        ]
    )


@vd.kernel(workgroup_size=(16, 16, 1))
def paint(
    pixels: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    time: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    y = gid[1]
    if x < WIDTH and y < HEIGHT:
        c = vd.Vector([-0.8, vd.cos(time) * 0.2])
        z = vd.Vector(
            [
                (vd.f32(x) / vd.f32(HEIGHT) - 1.0) * 2.0,
                (vd.f32(y) / vd.f32(HEIGHT) - 0.5) * 2.0,
            ]
        )
        iterations = 0
        while vd.norm(z) < 20.0 and iterations < 50:
            z = complex_square(z) + c
            iterations += 1
        pixels[y, x] = 1.0 - vd.f32(iterations) * 0.02


def render(time: float = 0.0) -> vd.TensorStorage:
    pixels = vd.storage.zeros(dtype=vd.f32, shape=(HEIGHT, WIDTH))
    paint(pixels, time, grid=(WIDTH // 16, HEIGHT // 16, 1))
    return pixels


def main() -> None:
    defaults = render_defaults()
    parser = argparse.ArgumentParser(description="Render the VernonDSL Julia set")
    parser.add_argument(
        "--arch",
        "--architecture",
        dest="arch",
        choices=("cpu", "cuda", "vulkan", "directx", "opengl", "opengles", "metal"),
        default="cuda",
        help=("execution backend; OpenGL profiles require a host context; DirectX requires Windows"),
    )
    parser.add_argument("--time", type=float, default=0.0)
    parser.add_argument("--frames", type=int, default=defaults.frames)
    parser.add_argument("--time-step", type=float, default=defaults.time_step)
    parser.add_argument("--fps", type=int, default=defaults.fps)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--animation-output", type=Path)
    arguments = parser.parse_args()

    try:
        import cv2  # pyright: ignore[reportMissingImports]
        from showcase_common import write_animation
    except ImportError as error:
        raise SystemExit(
            "Install the optional example dependencies with 'uv sync --extra examples --frozen'."
        ) from error
    if arguments.frames <= 0 or arguments.fps <= 0:
        raise ValueError("frames and fps must be positive")
    architectures = {
        "cpu": vd.cpu,
        "cuda": vd.cuda,
        "vulkan": vd.vulkan,
        "directx": vd.directx,
        "opengl": vd.opengl,
        "opengles": vd.opengles,
        "metal": vd.metal,
    }
    vd.init(arch=architectures[arguments.arch])
    pixels = vd.storage.zeros(dtype=vd.f32, shape=(HEIGHT, WIDTH))
    window = "VernonDSL Julia Set"
    if not arguments.headless:
        cv2.namedWindow(window)
    animation_frames = []
    try:
        for frame in range(arguments.frames):
            paint(
                pixels,
                arguments.time + frame * arguments.time_step,
                grid=(WIDTH // 16, HEIGHT // 16, 1),
            )
            image = pixels.to_numpy()
            if arguments.animation_output is not None:
                gray = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
                animation_frames.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA))
            if not arguments.headless:
                cv2.imshow(window, image)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        if not arguments.headless:
            cv2.destroyWindow(window)
    if arguments.animation_output is not None:
        write_animation(arguments.animation_output, animation_frames, arguments.fps)


if __name__ == "__main__":
    main()
