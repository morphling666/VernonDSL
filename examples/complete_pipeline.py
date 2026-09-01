from __future__ import annotations

import argparse
from pathlib import Path
from typing import Annotated

import cv2  # type: ignore[import-not-found]
import numpy as np
import vernon_dsl as vd

ANIMATE = vd.feature("ANIMATE")
PICKING = vd.feature("PICKING")


@vd.func(shared=True)
def mix_color(
    left: vd.Vector[vd.f32, 4],
    right: vd.Vector[vd.f32, 4],
    amount: vd.f32,
) -> vd.Vector[vd.f32, 4]:
    return left * (vd.f32(1.0) - amount) + right * amount


@vd.struct(shared=True)
class Palette:
    warm: vd.Vector[vd.f32, 4]
    cool: vd.Vector[vd.f32, 4]

    @vd.func(shared=True)
    def tint(self, amount: vd.f32) -> vd.Vector[vd.f32, 4]:
        return mix_color(self.warm, self.cool, amount)

    @vd.func
    def device_tint(self, amount: vd.f32) -> vd.Vector[vd.f32, 4]:
        # A shared struct may also expose a device-only method.
        return self.tint(vd.clamp(amount, vd.f32(0.0), vd.f32(1.0)))


@vd.struct
class VertexData:
    position: Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]
    local_color: vd.Vector[vd.f32, 2]


@vd.struct
class GBuffer:
    color: vd.Vector[vd.f32, 4]
    object_id: vd.Vector[vd.f32, 4]


@vd.kernel(workgroup_size=(2, 1, 1))
def animate_instances(
    offset: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    base_offset: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    phase: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    component = gid[0]
    instance_index = gid[1]
    value = base_offset[instance_index, component]
    if ANIMATE:
        if component == 1:
            value = value + vd.sin(phase + vd.f32(instance_index) * vd.f32(0.7)) * vd.f32(0.18)
    offset[instance_index, component] = value


@vd.vertex
def vertex_main(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
    offset: Annotated[vd.Vector[vd.f32, 2], vd.attribute(divisor=1)],
) -> VertexData:
    return VertexData(
        vd.Vector([position + offset, 0.0, 1.0]),
        position + vd.Vector([0.5, 0.5]),
    )


@vd.fragment
def fragment_main(
    local_color: Annotated[vd.Vector[vd.f32, 2], vd.varying()],
    tint: Annotated[vd.Vector[vd.f32, 4], vd.uniform()],
) -> GBuffer:
    source = vd.Vector([local_color, 1.0, 1.0])
    color = source * tint
    object_id = vd.Vector([0.0, 0.0, 0.0, 1.0])
    if PICKING:
        object_id = vd.Vector([1.0, 0.25, 0.0, 1.0])
    return GBuffer(color, object_id)


@vd.fragment
def method_lowering_preview(
    palette: Palette,
    amount: vd.f32,
) -> vd.Vector[vd.f32, 4]:
    # This separate frontend entry makes the device-only method lowering
    # visible without requiring a packed host/device struct ABI.
    return palette.device_tint(amount)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a three-stage variant pipeline with instancing, indexed MRT drawing, and dynamic input rebinding."
        )
    )
    parser.add_argument(
        "--arch",
        "--architecture",
        dest="arch",
        choices=("opengl", "opengles", "vulkan", "directx", "metal"),
        default="vulkan",
        help=("graphics backend; OpenGL profiles require host context registration; DirectX requires Windows"),
    )
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--instances", type=int, default=7)
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="zero runs until Escape or Q; use at least 3 to see every variant",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--id-output", type=Path)
    parser.add_argument("--method-mlir", type=Path)
    return parser.parse_args()


def _write_image(path: Path | None, image: np.ndarray | None) -> None:
    if path is not None and image is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"cannot write {path}")


def main() -> None:
    args = parse_args()
    if args.size <= 0 or args.instances <= 0 or args.frames < 0 or args.fps <= 0:
        raise ValueError("size, instances, and fps must be positive")

    architecture = {
        "opengl": vd.opengl,
        "opengles": vd.opengles,
        "vulkan": vd.vulkan,
        "directx": vd.directx,
        "metal": vd.metal,
    }[args.arch]
    vd.init(
        arch=architecture,
        api_version=(4, 3) if args.arch in {"opengl", "opengles"} else None,
    )

    variants = (
        ("STATIC", (), vd.pipeline(vertex_main, fragment_main)),
        (
            "ANIMATE",
            ("ANIMATE",),
            vd.pipeline(
                vertex_main,
                fragment_main,
                features={"ANIMATE"},
            ),
        ),
        (
            "ANIMATE+PICKING",
            ("ANIMATE", "PICKING"),
            vd.pipeline(
                vertex_main,
                fragment_main,
                features={"ANIMATE", "PICKING"},
            ),
        ),
    )

    packed_positions = vd.storage.from_numpy(
        np.array(
            ((-0.09, -0.09), (0.09, -0.09), (0.09, 0.09), (-0.09, 0.09)),
            dtype=np.float32,
        )
    )
    interleaved_positions = vd.storage.from_numpy(
        np.array(
            (
                (99.0, -0.09, -0.09, 1.0),
                (99.0, 0.09, -0.09, 1.0),
                (99.0, 0.09, 0.09, 1.0),
                (99.0, -0.09, 0.09, 1.0),
            ),
            dtype=np.float32,
        )
    )
    position_bindings = (
        ("packed Tensor", packed_positions),
        ("interleaved TensorView.yz", interleaved_positions.swizzle("yz")),
    )

    index_bindings = (
        vd.storage.from_numpy(np.array((0, 1, 2, 0, 2, 3), dtype=np.uint32)),
        vd.storage.from_numpy(np.array((2, 3, 0, 2, 0, 1), dtype=np.uint32)),
    )
    base_x = np.linspace(-0.75, 0.75, args.instances, dtype=np.float32)
    base_offsets = (
        vd.storage.from_numpy(np.column_stack((base_x, np.zeros_like(base_x))).astype(np.float32)),
        vd.storage.from_numpy(np.column_stack((base_x, np.full_like(base_x, 0.08))).astype(np.float32)),
    )
    offsets = vd.storage.zeros(dtype=vd.f32, shape=(args.instances, 2))

    palette = Palette(
        vd.Vector([1.0, 0.45, 0.15, 1.0]),
        vd.Vector([0.15, 0.65, 1.0, 1.0]),
    )
    tint_bindings = (
        vd.storage.from_numpy(np.asarray(palette.tint(vd.f32(0.15)))),
        vd.storage.from_numpy(np.asarray(palette.tint(vd.f32(0.85)))),
    )

    color = vd.Texture.zeros(shape=(args.size, args.size))
    object_id = vd.Texture.zeros(shape=(args.size, args.size))
    target = vd.RenderTarget.from_attachments(colors={0: color, 1: object_id})
    frame = 0
    color_image: np.ndarray | None = None
    id_image: np.ndarray | None = None
    delay_ms = max(1, round(1000 / args.fps))
    try:
        while args.frames == 0 or frame < args.frames:
            variant_name, features, render = variants[frame % len(variants)]
            binding_name, positions = position_bindings[frame % len(position_bindings)]
            binding_index = frame % 2
            animate_instances(
                offsets,
                base_offsets[binding_index],
                np.float32(frame / args.fps),
                features=features,
            )
            render(
                position=positions,
                offset=offsets,
                tint=tint_bindings[binding_index],
                indices=index_bindings[binding_index],
                topology=vd.triangles,
                target=target,
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
            print(
                f"frame={frame} variant={variant_name} "
                f"position={binding_name} index_buffer={binding_index} "
                f"tint_buffer={binding_index} compiled={render.compile_count}"
            )
            frame += 1
            if not args.headless:
                cv2.imshow("VernonDSL complete color", color_image)
                cv2.imshow("VernonDSL complete object ID", id_image)
                if cv2.waitKey(delay_ms) & 0xFF in (27, ord("q")):
                    break
    finally:
        if not args.headless:
            cv2.destroyAllWindows()

    _write_image(args.output, color_image)
    _write_image(args.id_output, id_image)

    if args.method_mlir is not None:
        mlir = vd.compile_file(Path(__file__), entry="method_lowering_preview")
        args.method_mlir.parent.mkdir(parents=True, exist_ok=True)
        args.method_mlir.write_text(mlir, encoding="utf-8")
    print(f"backend={args.arch} frames={frame} instances={args.instances}")


if __name__ == "__main__":
    main()
