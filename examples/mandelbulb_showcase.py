from __future__ import annotations

import argparse
import math
import time

import numpy as np
import vernon_dsl as vd
from shader_lib.fullscreen import fullscreen_vertex
from shader_lib.mandelbulb import mandelbulb_fragment
from showcase_common import (
    FramePresenter,
    ShowcasePreset,
    architecture_from_name,
    configure_showcase_parser,
    create_fullscreen_triangle,
    emit_showcase_result,
    resolve_showcase_options,
    write_animation,
)

PRESETS = {
    "smoke": ShowcasePreset(size=128, frames=1, fps=30),
    "showoff": ShowcasePreset(size=640, frames=120, fps=60),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a ray-marched Mandelbulb fractal.")
    configure_showcase_parser(parser, PRESETS)
    return parser.parse_args()


def main() -> None:
    options = resolve_showcase_options(parse_args(), PRESETS)
    vd.init(
        arch=architecture_from_name(options.architecture),
        api_version=(3, 3) if options.architecture == "opengl" else None,
    )
    positions = create_fullscreen_triangle()
    output = vd.Texture.zeros(shape=(options.size, options.size))
    target = vd.RenderTarget.from_attachments(colors={0: output})
    render_mandelbulb = vd.pipeline(fullscreen_vertex, mandelbulb_fragment)
    quality = {
        "smoke": (np.int32(48), np.int32(9), np.int32(12)),
        "showoff": (np.int32(112), np.int32(18), np.int32(32)),
    }[options.preset]
    presenter = FramePresenter(
        output,
        architecture=options.architecture,
        title="VernonDSL Mandelbulb",
        headless=options.headless,
        fps=options.fps,
    )

    def frame_values(phase: float) -> dict[str, object]:
        angle = phase * 0.22 + 0.55
        camera = np.array(
            (3.15 * math.cos(angle), 0.48 + math.sin(phase * 0.17) * 0.12, 3.15 * math.sin(angle)),
            dtype=np.float32,
        )
        return {
            "camera_position": camera,
            "time": np.float32(phase),
            "power": np.float32(8.0 + math.sin(phase * 0.21) * 0.18),
        }

    animation_frames: list[np.ndarray] = []
    frame = 0
    start = time.perf_counter()
    try:
        while options.frames == 0 or frame < options.frames:
            phase = float(time.perf_counter() - start if options.frames == 0 else frame / options.fps)
            render_mandelbulb(
                position=positions,
                camera_target=np.array((0.0, 0.0, 0.0), dtype=np.float32),
                max_steps=quality[0],
                max_iterations=quality[1],
                shadow_steps=quality[2],
                render=vd.render(target, color=vd.clear((0.0, 0.0, 0.0, 1.0))),
                **frame_values(phase),
            )
            frame += 1
            if not presenter.present():
                break
            if options.animation_output is not None and presenter.image is not None:
                animation_frames.append(presenter.image.copy())
    finally:
        presenter.close()
    elapsed = time.perf_counter() - start
    presenter.write(options.output)
    if options.animation_output is not None:
        write_animation(options.animation_output, animation_frames, options.fps)
    if presenter.image is None:
        raise RuntimeError("Mandelbulb showcase did not render an image")
    emit_showcase_result(
        name="mandelbulb",
        options=options,
        image=presenter.image,
        rendered_frames=frame,
        elapsed_seconds=elapsed,
        passes=1,
        barriers=0,
    )


if __name__ == "__main__":
    main()
