from __future__ import annotations

import argparse
from pathlib import Path
from typing import Annotated

import vernon_dsl as vd

WIDTH = 640
HEIGHT = 320


@vd.func
def complex_square(z: vd.vec2[vd.f32]) -> vd.vec2[vd.f32]:
    return vd.vec2(
        z[0] ** 2 - z[1] ** 2,
        z[1] * z[0] * 2,
    )


@vd.kernel(workgroup_size=(16, 16, 1))
def paint(
    pixels: vd.Tensor[vd.f32, (None, None)],
    time: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    y = gid[1]
    if x < WIDTH:
        if y < HEIGHT:
            c = vd.vec2(-0.8, vd.cos(time) * 0.2)
            z = vd.vec2(
                (vd.f32(x) / vd.f32(HEIGHT) - 1.0) * 2.0,
                (vd.f32(y) / vd.f32(HEIGHT) - 0.5) * 2.0,
            )
            iterations = 0
            running = vd.norm(z) < 20.0
            while running:
                z = complex_square(z) + c
                iterations += 1
                running = vd.norm(z) < 20.0
                if iterations >= 50:
                    running = False
            pixels[y, x] = 1.0 - vd.f32(iterations) * 0.02


def render(time: float = 0.0) -> vd.Tensor:
    pixels = vd.Tensor.zeros(dtype=vd.f32, shape=(HEIGHT, WIDTH))
    paint(pixels, time, grid=(WIDTH, HEIGHT, 1))
    return pixels


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the VernonDSL Julia set")
    parser.add_argument(
        "--arch",
        choices=("cpu", "cuda", "vulkan", "opengl", "opengles"),
        default="cuda",
        help=("execution backend; OpenGL profiles require a host to register an external context first"),
    )
    parser.add_argument("--time", type=float, default=0.0)
    parser.add_argument("--frames", type=int, default=1_000_000)
    parser.add_argument("--time-step", type=float, default=0.03)
    parser.add_argument("--emit-metal", type=Path)
    arguments = parser.parse_args()
    if arguments.emit_metal is not None:
        pixels = vd.Tensor.zeros(dtype=vd.f32, shape=(HEIGHT, WIDTH))
        source, _ = paint.compile_artifact(pixels, arguments.time, target="metal")
        arguments.emit_metal.write_bytes(source)
        print(f"Wrote {arguments.emit_metal}")
        return

    try:
        import cv2  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise SystemExit("Install the optional example dependencies with 'uv sync --extra examples'.") from error
    architectures = {
        "cpu": vd.cpu,
        "cuda": vd.cuda,
        "vulkan": vd.vulkan,
        "opengl": vd.opengl,
        "opengles": vd.opengles,
    }
    vd.init(arch=architectures[arguments.arch])
    pixels = vd.Tensor.zeros(dtype=vd.f32, shape=(HEIGHT, WIDTH))
    window = "VernonDSL Julia Set"
    cv2.namedWindow(window)
    try:
        for frame in range(arguments.frames):
            paint(
                pixels,
                arguments.time + frame * arguments.time_step,
                grid=(WIDTH, HEIGHT, 1),
            )
            cv2.imshow(window, pixels.to_numpy())
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cv2.destroyWindow(window)


if __name__ == "__main__":
    main()
