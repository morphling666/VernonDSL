from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import vernon_dsl as vd
from shader_lib.fullscreen import fullscreen_vertex
from shader_lib.terrain import terrain_fragment
from showcase_common import (
    BatchRenderPass,
    FramePresenter,
    ShowcasePreset,
    architecture_from_name,
    configure_showcase_parser,
    create_fullscreen_triangle,
    emit_showcase_result,
    load_rgba_texture,
    resolve_showcase_options,
    write_animation,
)

PRESETS = {
    "smoke": ShowcasePreset(size=128, frames=1, fps=30),
    "showoff": ShowcasePreset(size=640, frames=120, fps=60),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a ray-marched procedural mountain landscape.")
    configure_showcase_parser(parser, PRESETS)
    return parser.parse_args()


def main() -> None:
    options = resolve_showcase_options(parse_args(), PRESETS)
    vd.init(
        arch=architecture_from_name(options.architecture),
        api_version=(3, 3) if options.architecture == "opengl" else None,
    )
    positions = create_fullscreen_triangle()
    noise = load_rgba_texture(Path(__file__).parent / "assets" / "noise512.png")
    noise_sampler = vd.sampler(address="repeat")
    output = vd.Texture.zeros(shape=(options.size, options.size))
    target = vd.RenderTarget.from_attachments(colors={0: output})
    render_terrain = vd.pipeline(fullscreen_vertex, terrain_fragment)
    quality = {
        "smoke": (np.int32(64), np.int32(16), np.int32(2)),
        "showoff": (np.int32(150), np.int32(48), np.int32(4)),
    }[options.preset]
    camera_position = np.array((0.0, 2.0, 0.0), dtype=np.float32)
    camera_target = np.array((25.0, -20.0, -65.0), dtype=np.float32)
    sun_direction = np.array((-2.0, 1.6, -4.0), dtype=np.float32)
    sun_direction /= np.linalg.norm(sun_direction)
    graph = vd.ExecutionGraph()
    time_parameter = graph.parameter("time")
    invocation = render_terrain.invocation(
        position=positions,
        noise_texture=noise,
        noise_sampler=noise_sampler,
        time=time_parameter,
        camera_position=camera_position,
        camera_target=camera_target,
        sun_direction=sun_direction,
        screen_y_sign=np.float32(1.0 if options.architecture == "opengl" else -1.0),
        ambient=np.float32(0.2),
        max_steps=quality[0],
        shadow_steps=quality[1],
        ao_samples=quality[2],
        topology=vd.triangles,
    )
    graph.add_pass(
        BatchRenderPass(
            "terrain-raymarch",
            target,
            [invocation],
            clear_color=(0.0, 0.0, 0.0, 1.0),
        )
    )
    presenter = FramePresenter(
        output,
        architecture=options.architecture,
        title="VernonDSL Ray-Marched Terrain",
        headless=options.headless,
        fps=options.fps,
    )
    plan = graph.compile()
    bindings = plan.create_bindings({time_parameter: np.float32(0.0)})
    animation_frames: list[np.ndarray] = []
    frame = 0
    start = time.perf_counter()
    try:
        while options.frames == 0 or frame < options.frames:
            phase = np.float32(time.perf_counter() - start if options.frames == 0 else frame / options.fps)
            bindings.update({time_parameter: phase})
            plan.submit(bindings).wait()
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
        raise RuntimeError("Terrain showcase did not render an image")
    barrier_count = sum(len(scope.barriers) for scope in plan.scopes)
    emit_showcase_result(
        name="terrain",
        options=options,
        image=presenter.image,
        rendered_frames=frame,
        elapsed_seconds=elapsed,
        passes=len(plan.schedule),
        barriers=barrier_count,
    )


if __name__ == "__main__":
    main()
