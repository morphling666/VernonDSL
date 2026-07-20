from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import vernon_dsl as vd

from python.tests.shared_kernel import (  # type: ignore[import-not-found]
    evaluate_shared, shared_polynomial,
)


@vd.func(shared=True)
def shared_length(value: vd.vec3[vd.f32]) -> vd.f32:
    return vd.norm(value)


@vd.struct(shared=True)
class SharedLight:
    position: vd.vec3[vd.f32]
    intensity: vd.f32

    @vd.func(shared=True)
    def contribution(self, point: vd.vec3[vd.f32]) -> vd.f32:
        distance = shared_length(self.position - point)
        return self.intensity / (distance * distance)

    @vd.func
    def device_bias(self, value: vd.f32) -> vd.f32:
        return value + self.intensity


class SharedHostTests(unittest.TestCase):

    def test_shared_function_and_intrinsics_execute_on_host(self) -> None:
        vector = vd.vec3(3.0, 4.0, 0.0)
        self.assertEqual(vector.dtype, np.float32)
        self.assertAlmostEqual(float(shared_length(vector)), 5.0)
        self.assertTrue(
            np.allclose(vd.normalize(vector), np.array([0.6, 0.8, 0.0])))

    def test_shared_struct_is_an_immutable_value(self) -> None:
        source = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        light = SharedLight(source, 8.0)
        source[0] = 100.0

        self.assertEqual(float(light.position[0]), 1.0)
        self.assertFalse(light.position.flags.writeable)
        self.assertAlmostEqual(
            float(light.contribution(vd.vec3(1.0, 2.0, 1.0))), 2.0)
        with self.assertRaisesRegex(AttributeError, "immutable"):
            light.intensity = vd.f32(2.0)
        with self.assertRaises(ValueError):
            light.position[0] = 2.0
        with self.assertRaisesRegex(TypeError, "shader-only"):
            light.device_bias(vd.f32(1.0))

    def test_device_only_definitions_remain_host_forbidden(self) -> None:

        @vd.func
        def helper(value: vd.f32) -> vd.f32:
            return value

        @vd.struct
        class DeviceValue:
            value: vd.f32

        with self.assertRaisesRegex(TypeError, "shader-only"):
            helper(1.0)
        with self.assertRaisesRegex(TypeError, "shader-only"):
            DeviceValue(1.0)

    def test_shared_helper_matches_cpu_compilation(self) -> None:
        vd.init(arch=vd.cpu)
        output = vd.Tensor.zeros(dtype=vd.f32, shape=(3, ))
        evaluate_shared(output, 2.0, grid=(3, 1, 1))
        expected = np.array(
            [shared_polynomial(vd.f32(value)) for value in (2.0, 3.0, 4.0)],
            dtype=np.float32,
        )
        np.testing.assert_allclose(output.to_numpy(), expected)


class SharedCompilerTests(unittest.TestCase):

    def test_methods_lower_to_explicit_private_helpers(self) -> None:
        source = """
from vernon_dsl import *

@struct(shared=True)
class Light:
    position: vec3[f32]
    intensity: f32

    @func(shared=True)
    def distance(self, point: vec3[f32]) -> f32:
        return norm(self.position - point)

    @func(shared=True)
    def contribution(self, point: vec3[f32]) -> f32:
        distance = self.distance(point)
        return self.intensity / (distance * distance)

    @func
    def device_bias(self, value: f32) -> f32:
        return value + self.intensity

@func(shared=True)
def evaluate(light: Light, point: vec3[f32]) -> f32:
    return light.contribution(point)

@func
def make_light(position: vec3[f32], intensity: f32) -> Light:
    return Light(position, intensity)

@func
def evaluate_factory(position: vec3[f32], intensity: f32) -> f32:
    return make_light(position, intensity).device_bias(1.0)

@fragment
def main(light: Light, point: vec3[f32]) -> f32:
    return light.device_bias(evaluate(light, point))
"""
        output = vd.compile_source(source, "shared_methods.py")

        self.assertIn("func.func private @Light__distance", output)
        self.assertIn("func.func private @Light__contribution", output)
        self.assertIn("func.call @Light__distance(%arg0", output)
        self.assertIn("func.call @Light__contribution(%arg0", output)
        self.assertIn("func.call @Light__device_bias(%arg0", output)
        self.assertIn("func.call @Light__device_bias(%", output)
        self.assertIn('"vernon.struct_get"(%arg0)', output)

    def test_domain_and_struct_diagnostics(self) -> None:
        invalid_sources = (
            (
                """
from vernon_dsl import *
@func
def device(value: f32) -> f32:
    return value
@func(shared=True)
def shared(value: f32) -> f32:
    return device(value)
""",
                "cannot call device-only",
            ),
            (
                """
from vernon_dsl import *
@struct
class Value:
    value: f32
    @func(shared=True)
    def get(self) -> f32:
        return self.value
""",
                "requires @struct\\(shared=True\\)",
            ),
            (
                """
from vernon_dsl import *
@struct(shared=True)
class Value:
    value: f32
    @func
    def device(self) -> f32:
        return self.value
    @func(shared=True)
    def shared(self) -> f32:
        return self.device()
""",
                "cannot call device-only",
            ),
            (
                """
from vernon_dsl import *
@struct(shared=True)
class Value:
    value: f32
    @func(shared=True)
    def change(self, value: f32) -> f32:
        self.value = value
        return self.value
""",
                "cannot mutate self",
            ),
            (
                """
from vernon_dsl import *
@struct(shared=True)
class Base:
    value: f32
@struct(shared=True)
class Child(Base):
    other: f32
""",
                "do not support inheritance",
            ),
            (
                """
from vernon_dsl import *
@struct(shared=True)
class Value:
    value: f32
    @func(shared=True)
    def read(self) -> f32:
        return self.value
    @func(shared=True)
    def read(self) -> f32:
        return self.value
""",
                "duplicate struct member",
            ),
            (
                """
from vernon_dsl import *
@struct(shared=True)
class Value:
    value: f32
    @func(shared=True)
    def read(self) -> f32:
        return self.value
    @func(shared=True)
    def method_value(self) -> f32:
        return self.read
""",
                "first-class values",
            ),
            (
                """
from vernon_dsl import *
@func(shared=True)
def sample(image: Texture["2d", f32]) -> f32:
    return 0.0
""",
                "device-only argument",
            ),
        )
        for source, diagnostic in invalid_sources:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(vd.CompileError, diagnostic):
                    vd.compile_source(source, "invalid_shared.py")

    def test_imported_shared_method_is_namespaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lighting.py").write_text(
                "from vernon_dsl import *\n"
                "@struct(shared=True)\n"
                "class Light:\n"
                "    intensity: f32\n"
                "    @func(shared=True)\n"
                "    def scale(self, value: f32) -> f32:\n"
                "        return self.intensity * value\n",
                encoding="utf-8",
            )
            shader = root / "shader.py"
            shader.write_text(
                "from vernon_dsl import *\n"
                "from lighting import Light\n"
                "@fragment\n"
                "def main(light: Light) -> f32:\n"
                "    return light.scale(2.0)\n",
                encoding="utf-8",
            )

            output = vd.compile_file(shader)

        self.assertIn("func.func private @__vernon_lighting__Light__scale",
                      output)
        self.assertIn("func.call @__vernon_lighting__Light__scale", output)

    def test_recursive_methods_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recursive_method.py"
            path.write_text(
                "from vernon_dsl import *\n"
                "@struct(shared=True)\n"
                "class Value:\n"
                "    value: f32\n"
                "    @func(shared=True)\n"
                "    def first(self) -> f32:\n"
                "        return self.second()\n"
                "    @func(shared=True)\n"
                "    def second(self) -> f32:\n"
                "        return self.first()\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(vd.CompileError,
                                        "recursive DSL call graph"):
                vd.compile_file(path)


if __name__ == "__main__":
    unittest.main()
