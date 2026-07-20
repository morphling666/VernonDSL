from __future__ import annotations

from typing import Annotated

import vernon_dsl as vd

WIDTH = 640
HEIGHT = 320


@vd.kernel(workgroup_size=(16, 16, 1))
def paint(
    pixels: vd.Tensor[vd.f32, (None, None)],
    time: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3, )],
                   vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    y = gid[1]
    if x < WIDTH and y < HEIGHT:
        c = vd.vec2(-0.8, vd.cos(time) * 0.2)
        z = vd.vec2(
            (vd.f32(x) / vd.f32(HEIGHT) - 1.0) * 2.0,
            (vd.f32(y) / vd.f32(HEIGHT) - 0.5) * 2.0,
        )
        iterations = 0
        while vd.norm(z) < 20.0 and iterations < 50:
            z = vd.vec2(
                z[0] * z[0] - z[1] * z[1],
                z[1] * z[0] * 2.0,
            ) + c
            iterations += 1
        pixels[y, x] = 1.0 - vd.f32(iterations) * 0.02


def render(time: float = 0.0) -> vd.Tensor:
    pixels = vd.Tensor.zeros(dtype=vd.f32, shape=(HEIGHT, WIDTH))
    paint(pixels, time, grid=(WIDTH, HEIGHT, 1))
    return pixels


def main() -> None:
    try:
        import cv2  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise SystemExit("Install the optional example dependencies with "
                         "'uv sync --extra examples'.") from error
    image = render().to_numpy()
    cv2.imshow("VernonDSL Julia Set", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
