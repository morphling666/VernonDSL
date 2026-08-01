from __future__ import annotations

import argparse
from pathlib import Path
from typing import Annotated

import numpy as np
import vernon_dsl as vd


@vd.func(shared=True)
def inverse_square_falloff(distance: vd.f32) -> vd.f32:
    return vd.clamp(
        vd.f32(1.0) / (vd.f32(1.0) + distance * distance),
        vd.f32(0.0),
        vd.f32(1.0),
    )


@vd.struct(shared=True)
class Light:
    position: vd.Vector[vd.f32, 3]
    color: vd.Vector[vd.f32, 3]
    intensity: vd.f32

    @vd.func(shared=True)
    def distance_to(self, point: vd.Vector[vd.f32, 3]) -> vd.f32:
        return vd.norm(self.position - point)

    @vd.func(shared=True)
    def contribution(self, point: vd.Vector[vd.f32, 3]) -> vd.Vector[vd.f32, 3]:
        falloff = inverse_square_falloff(self.distance_to(point))
        return self.color * self.intensity * falloff

    @vd.func
    def device_lambert(
        self,
        point: vd.Vector[vd.f32, 3],
        normal: vd.Vector[vd.f32, 3],
    ) -> vd.Vector[vd.f32, 3]:
        light_direction = vd.normalize(self.position - point)
        amount = vd.max(vd.dot(normal, light_direction), vd.f32(0.0))
        return self.contribution(point) * amount


@vd.kernel(workgroup_size=(4, 1, 1))
def evaluate_falloff(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    start_distance: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    distance = start_distance + vd.f32(gid[0])
    output[gid[0]] = inverse_square_falloff(distance)


@vd.fragment
def preview_fragment(
    light: Light,
    point: vd.Vector[vd.f32, 3],
    normal: vd.Vector[vd.f32, 3],
) -> vd.Vector[vd.f32, 3]:
    # Device code may call both device-only and shared methods.
    return light.device_lambert(point, normal)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate shared functions and immutable struct methods.")
    parser.add_argument(
        "--mlir",
        type=Path,
        help="optional path for the lowered struct-method frontend MLIR",
    )
    parser.add_argument(
        "--arch",
        "--architecture",
        dest="arch",
        choices=("cpu", "cuda", "vulkan", "directx", "metal"),
        default="cpu",
        help="compute backend; DirectX requires Windows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    light = Light(
        vd.Vector([2.0, 3.0, 4.0]),
        vd.Vector([1.0, 0.6, 0.25]),
        vd.f32(8.0),
    )
    point = vd.Vector([0.0, 0.0, 0.0])
    host_value = light.contribution(point)

    identity = vd.Matrix([[1.0, 0.0], [0.0, 1.0]])
    transformed = vd.matmul(identity, vd.Vector([host_value[0], host_value[1]]))
    reflected = vd.reflect(vd.Vector([0.0, -1.0, 0.0]), vd.Vector([0.0, 1.0, 0.0]))

    print("host contribution:", host_value)
    print("matrix/vector result:", transformed)
    print("reflected direction:", reflected)
    print("position is read-only:", not light.position.flags.writeable)

    try:
        light.device_lambert(point, vd.Vector([0.0, 1.0, 0.0]))
    except TypeError as error:
        print("expected host domain error:", error)

    vd.init(
        arch={
            "cpu": vd.cpu,
            "cuda": vd.cuda,
            "vulkan": vd.vulkan,
            "directx": vd.directx,
            "metal": vd.metal,
        }[args.arch]
    )
    output = vd.storage.zeros(dtype=vd.f32, shape=(4,))
    evaluate_falloff(output, 1.0, grid=(4, 1, 1))
    device_values = output.to_numpy()
    host_values = np.array(
        [inverse_square_falloff(vd.f32(value)) for value in range(1, 5)],
        dtype=np.float32,
    )
    np.testing.assert_allclose(device_values, host_values)
    print("shared helper host/device values:", device_values)

    mlir = vd.compile_file(Path(__file__), entry="preview_fragment")
    if args.mlir is not None:
        args.mlir.parent.mkdir(parents=True, exist_ok=True)
        args.mlir.write_text(mlir, encoding="utf-8")
    print("lowered methods:", "Light__contribution, Light__device_lambert")


if __name__ == "__main__":
    main()
