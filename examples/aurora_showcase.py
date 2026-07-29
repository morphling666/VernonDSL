from __future__ import annotations

import argparse
import time

import numpy as np
import vernon_dsl as vd
from shader_lib.aurora import aurora_compose, aurora_curtains
from shader_lib.fullscreen import fullscreen_vertex
from showcase_common import (
    BatchRenderPass,
    FramePresenter,
    InvocationBatch,
    ShowcasePreset,
    architecture_from_name,
    configure_showcase_parser,
    create_fullscreen_triangle,
    emit_showcase_result,
    resolve_showcase_options,
)

PRESETS = {
    "smoke": ShowcasePreset(size=192, frames=2, fps=30),
    "showoff": ShowcasePreset(size=768, frames=180, fps=60),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a two-pass procedural aurora.")
    configure_showcase_parser(parser, PRESETS)
    return parser.parse_args()


def main() -> None:
    options = resolve_showcase_options(parse_args(), PRESETS)
    vd.init(
        arch=architecture_from_name(options.architecture),
        api_version=(4, 3) if options.architecture == "opengl" else None,
    )
    positions = create_fullscreen_triangle()
    field = vd.Texture.zeros(shape=(options.size, options.size))
    output = vd.Texture.zeros(shape=(options.size, options.size))
    field_target = vd.RenderTarget(shape=field.shape).attach_color(0, field)
    output_target = vd.RenderTarget(shape=output.shape).attach_color(0, output)
    field_sampler = vd.sampler(address="clamp_to_edge")
    render_curtains = vd.pipeline(fullscreen_vertex, aurora_curtains)
    render_compose = vd.pipeline(fullscreen_vertex, aurora_compose)

    curtains_batch = InvocationBatch()
    compose_batch = InvocationBatch()
    graph = vd.ExecutionGraph()
    graph.add_pass(
        BatchRenderPass(
            "aurora-curtains",
            field_target,
            curtains_batch,
            clear_color=(0.0, 0.0, 0.0, 1.0),
        )
    )
    graph.add_pass(
        BatchRenderPass(
            "aurora-compose",
            output_target,
            compose_batch,
            clear_color=(0.0, 0.0, 0.0, 1.0),
        )
    )
    presenter = FramePresenter(
        output,
        architecture=options.architecture,
        title="VernonDSL Aurora",
        headless=options.headless,
        fps=options.fps,
    )
    frame = 0
    start = time.perf_counter()
    try:
        while options.frames == 0 or frame < options.frames:
            phase = np.float32(time.perf_counter() - start if options.frames == 0 else frame / options.fps)
            curtains_batch.values = [
                render_curtains.invocation(
                    position=positions,
                    time=phase,
                    topology=vd.triangles,
                )
            ]
            compose_batch.values = [
                render_compose.invocation(
                    position=positions,
                    field=field,
                    field_sampler=field_sampler,
                    time=phase,
                    blur_radius=np.float32(1.15),
                    topology=vd.triangles,
                )
            ]
            graph.execute()
            frame += 1
            if not presenter.present():
                break
    finally:
        presenter.close()
    elapsed = time.perf_counter() - start
    presenter.write(options.output)
    if presenter.image is None:
        raise RuntimeError("aurora showcase did not render an image")
    barrier_count = sum(len(scope.barriers) for scope in graph.scopes)
    emit_showcase_result(
        name="aurora",
        options=options,
        image=presenter.image,
        rendered_frames=frame,
        elapsed_seconds=elapsed,
        passes=len(graph.schedule),
        barriers=barrier_count,
    )


if __name__ == "__main__":
    main()
