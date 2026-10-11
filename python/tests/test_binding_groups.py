from __future__ import annotations

import json
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import vernon_dsl as vd
from backend_test_matrix import (
    BackendRequirements,
    BackendRow,
    backend_matrix_test,
    backend_row,
    expand_backend_matrix_tests,
    probe_backend,
    require_available,
)
from binding_group_shader import (
    Accumulator,
    Affine,
    Flat,
    Lookup,
    Surface,
    SurfaceParams,
    Volume,
    accumulate,
    affine,
    sample_volume,
    surface_fragment,
    surface_vertex,
)
from vernon_dsl.diagnostics import CompileError

_GROUPS = """
from vernon_dsl import *

@struct(shared=True)
class Params:
    tint: Vector[f32, 4]
    gain: f32

@binding_group
class Lookup:
    weights: TensorView[f32, (dyn,), read]

@binding_group
class Material:
    params: Params
    albedo: Texture["2d", f32]
    linear: Sampler
    lookup: Lookup

    @func
    def shade(self, uv: Vector[f32, 2]) -> Vector[f32, 4]:
        return texture_sample(self.albedo, self.linear, uv) * self.params.tint * self.lookup.weights[0]
"""

_VERTEX = (
    _GROUPS
    + """
@vertex
def main(
    position: Annotated[Vector[f32, 2], attribute()],
    material: Material,
) -> Annotated[Vector[f32, 4], builtin("position")]:
    return Vector([position * material.lookup.weights[1], 0.0, 1.0])
"""
)

_FRAGMENT = (
    _GROUPS
    + """
@fragment
def main(
    occupied: Annotated[f32, uniform(set=0, binding=1)],
    material: Material,
) -> Vector[f32, 4]:
    return material.shade(Vector([0.5, 0.5])) * occupied
"""
)

_FULL_SCREEN = np.array(((-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)), dtype=np.float32)


def _native() -> Any:
    return import_module("vernon_dsl._native")


def _entry_arguments(mlir: str) -> dict[str, str]:
    (signature,) = [line for line in mlir.splitlines() if "func.func @main" in line]
    arguments: dict[str, str] = {}
    for chunk in signature.split("%arg")[1:]:
        name = chunk.split('vernon.source_name = "', 1)[1].split('"', 1)[0] if "vernon.source_name" in chunk else ""
        arguments[name] = chunk
    return arguments


def _link(modules: list[str]) -> list[dict[str, Any]]:
    linked = _native().Compiler().link_graphics_bindings_result(modules)
    if not linked.ok:
        raise AssertionError(str(linked.diagnostics))
    return list(json.loads(linked.reflection)["layouts"])


def _assignments(layout: dict[str, Any]) -> dict[str, list[int]]:
    return {row["source_name"]: row["bindings"] for row in layout["assignments"] if row["source_name"]}


def _render_pass(texture: vd.Texture) -> vd.RenderPass:
    return vd.render_pass(vd.RenderTarget.from_attachments(colors={0: texture}), color=vd.clear((0.0, 0.0, 0.0, 0.0)))


def _surface() -> Surface:
    return Surface(
        params=SurfaceParams(np.array((1.0, 0.5, 0.25, 1.0), dtype=np.float32), vd.f32(0.0)),
        albedo=vd.Texture.from_numpy(np.full((2, 2, 4), (128, 255, 255, 255), dtype=np.uint8)),
        linear=vd.sampler(address="clamp_to_edge"),
        lookup=Lookup(weights=vd.storage.from_numpy(np.array([0.5, 1.0], dtype=np.float32))),
    )


def _flat() -> Flat:
    return Flat(params=SurfaceParams(np.array((0.25, 0.25, 0.25, 0.5), dtype=np.float32), vd.f32(0.5)))


class BindingGroupCompileTests(unittest.TestCase):
    def test_leaves_are_dotted_entry_parameters_in_field_order(self) -> None:
        arguments = _entry_arguments(vd.compile_source(_FRAGMENT, "material_fragment.py"))
        self.assertEqual(
            [name for name in arguments if name.startswith("material.")],
            ["material.params", "material.albedo", "material.linear", "material.lookup.weights"],
        )
        for name in ("material.params", "material.albedo", "material.linear", "material.lookup.weights"):
            self.assertIn('vernon.binding_group = "material"', arguments[name])
        self.assertIn('vernon.interface = "uniform"', arguments["material.params"])
        self.assertIn('vernon.interface = "resource"', arguments["material.linear"])

    def test_one_contiguous_range_per_occurrence_shared_by_both_stages(self) -> None:
        vertex = vd.compile_source(_VERTEX, "material_vertex.py")
        fragment = vd.compile_source(_FRAGMENT, "material_fragment.py")
        vertex_layout, fragment_layout = (_assignments(layout) for layout in _link([vertex, fragment]))
        leaves = ("material.params", "material.albedo", "material.linear", "material.lookup.weights")
        self.assertEqual(
            {name: vertex_layout[name] for name in leaves}, {name: fragment_layout[name] for name in leaves}
        )
        bindings = [binding for name in leaves for binding in fragment_layout[name]]
        self.assertEqual(bindings, list(range(bindings[0], bindings[0] + len(bindings))))
        self.assertNotIn(1, bindings)

    def test_generic_helper_is_specialized_per_group_type(self) -> None:
        mlir = vd.compile_source(
            _GROUPS
            + """
@binding_group
class Flat:
    params: Params

    @func
    def shade(self, uv: Vector[f32, 2]) -> Vector[f32, 4]:
        return self.params.tint * (self.params.gain + uv[0] * 0.0)

@func
def apply(material, uv):
    return material.shade(uv)

@fragment
def main(material: Material, flat: Flat) -> Vector[f32, 4]:
    uv = Vector([0.5, 0.5])
    return apply(material, uv) + apply(flat, uv)
""",
            "generic_groups.py",
        )
        helpers = {
            line.split("@", 1)[1].split("(", 1)[0] for line in mlir.splitlines() if "func.func private @apply" in line
        }
        self.assertEqual(len(helpers), 2)
        self.assertIn("func.func private @Material__shade", mlir)
        self.assertIn("func.func private @Flat__shade", mlir)

    def test_typed_struct_method_resolves_on_unannotated_receiver(self) -> None:
        source = """
from vernon_dsl import *

@struct
class Scale:
    gain: f32

    @func
    def apply(self, value: f32) -> f32:
        return value * self.gain

@func
def run(scale, value):
    return scale.apply(value)

@kernel
def main(output: TensorView[f32, (dyn,), write], gain: f32) -> None:
    output[0] = run(Scale(gain), 3.0)
"""
        self.assertIn("func.func", vd.compile_source(source, "generic_struct_method.py"))
        with self.assertRaisesRegex(CompileError, r"'Scale' has no method 'missing' \(in helper specialization"):
            vd.compile_source(source.replace("scale.apply(value)", "scale.missing(value)"), "missing_method.py")

    def test_unreachable_methods_are_pruned_without_diagnostics(self) -> None:
        source = _FRAGMENT.replace(
            "    @func\n    def shade",
            "    @func\n    def broken(self) -> f32:\n        return undefined_name\n\n    @func\n    def shade",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pruned_method.py"
            path.write_text(source, encoding="utf-8")
            output = vd.compile_file(path, entry="main")
        self.assertIn("func.func @main", output)
        self.assertNotIn("broken", output)


class BindingGroupDiagnosticTests(unittest.TestCase):
    def _assert_rejected(self, body: str, pattern: str) -> None:
        with self.assertRaisesRegex(CompileError, pattern):
            vd.compile_source(_GROUPS + body, "misuse.py")

    def test_field_metadata_is_rejected(self) -> None:
        with self.assertRaisesRegex(CompileError, r"binding group field cannot declare interface metadata: 'Bad.x'"):
            vd.compile_source(
                """
from vernon_dsl import *

@binding_group
class Bad:
    x: Annotated[f32, uniform()]

@fragment
def main(bad: Bad) -> Vector[f32, 4]:
    return Vector([bad.x, 0.0, 0.0, 1.0])
""",
                "field_metadata.py",
            )

    def test_group_cannot_be_a_struct_field(self) -> None:
        self._assert_rejected(
            """
@struct
class Holder:
    material: Material
""",
            "binding group cannot be a Struct field",
        )

    def test_group_cannot_be_used_as_a_value(self) -> None:
        self._assert_rejected(
            """
@fragment
def main(material: Material) -> Vector[f32, 4]:
    copy = material
    return copy.params.tint
""",
            "binding group cannot be assigned to a local",
        )
        self._assert_rejected(
            """
@func
def leak(material: Material) -> Material:
    return material
""",
            "binding group cannot be returned",
        )

    def test_fields_cannot_be_reassigned(self) -> None:
        self._assert_rejected(
            """
@fragment
def main(material: Material) -> Vector[f32, 4]:
    material.params = Params(Vector([1.0, 1.0, 1.0, 1.0]), 1.0)
    return material.params.tint
""",
            "binding group cannot be reassigned through its fields",
        )

    def test_missing_field_and_method_are_reported(self) -> None:
        self._assert_rejected(
            """
@fragment
def main(material: Material) -> Vector[f32, 4]:
    return material.roughness
""",
            "binding group 'Material' has no field 'roughness'",
        )
        self._assert_rejected(
            """
@fragment
def main(material: Material) -> Vector[f32, 4]:
    return material.glow(Vector([0.5, 0.5]))
""",
            "'Material' has no method 'glow'",
        )

    def test_methods_are_not_first_class(self) -> None:
        self._assert_rejected(
            """
@fragment
def main(material: Material) -> Vector[f32, 4]:
    shade = material.shade
    return shade(Vector([0.5, 0.5]))
""",
            "binding group methods cannot be used as first-class values",
        )


class BindingGroupHostTests(unittest.TestCase):
    def test_container_requires_exactly_its_fields(self) -> None:
        weights = vd.storage.from_numpy(np.zeros(2, dtype=np.float32))
        with self.assertRaisesRegex(TypeError, "weights"):
            Lookup()
        with self.assertRaisesRegex(TypeError, "bias"):
            Lookup(weights=weights, bias=weights)
        lookup = Lookup(weights=weights)
        self.assertIs(lookup.weights, weights)
        with self.assertRaises(AttributeError):
            lookup.weights = weights  # type: ignore[misc]

    def test_argument_must_be_the_declared_group(self) -> None:
        vd.init(arch=vd.cpu)
        weights = vd.storage.from_numpy(np.zeros(3, dtype=np.float32))
        output = vd.storage.zeros(dtype=vd.f32, shape=(3,))
        with self.assertRaisesRegex(TypeError, "Affine"):
            affine(Lookup(weights=weights), output)  # type: ignore[arg-type]

    def test_file_loaded_asset_resolves_group_fields_after_its_module_unloads(self) -> None:
        from vernon_dsl.program_assets import cook_program_asset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "grouped.py"
            source.write_text(
                """
from __future__ import annotations

import vernon_dsl as vd

@vd.binding_group
class Terms:
    gain: vd.f32
    weights: vd.TensorView[vd.f32, (vd.dyn,), vd.read]

@vd.kernel
def scale(terms: Terms, output: vd.TensorView[vd.f32, (vd.dyn,), vd.write]) -> None:
    output[0] = terms.weights[0] * terms.gain

asset = vd.program_asset(id="compute/grouped", program=scale)
""",
                encoding="utf-8",
            )
            manifest = cook_program_asset(program_asset=f"{source}:asset", output=root / "cooked", target="cpu")
            text = manifest.read_text(encoding="utf-8")
        self.assertIn('"terms.gain"', text)
        self.assertIn('"terms.weights"', text)


@expand_backend_matrix_tests
class BindingGroupKernelRuntimeTests(unittest.TestCase):
    @backend_matrix_test(BackendRequirements(compute=True, storage_buffers=True))
    def test_group_kernel_with_method_and_nested_group(self, backend: BackendRow) -> None:
        vd.init(arch=backend.architecture)  # type: ignore[arg-type]
        output = vd.storage.zeros(dtype=vd.f32, shape=(3,))
        terms = Affine(
            gain=2.0,
            offset=0.5,
            lookup=Lookup(weights=vd.storage.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))),
            bias=vd.storage.from_numpy(np.array([10.0, 20.0, 30.0], dtype=np.float32)),
        )
        affine(terms, output)
        np.testing.assert_allclose(output.to_numpy(), [12.5, 24.5, 36.5])

    @backend_matrix_test(BackendRequirements(compute=True, storage_buffers=True))
    def test_group_kernel_with_struct_value(self, backend: BackendRow) -> None:
        vd.init(arch=backend.architecture)  # type: ignore[arg-type]
        output = vd.storage.zeros(dtype=vd.f32, shape=(3,))
        terms = Accumulator(
            params=SurfaceParams(np.array((0.5, 0.0, 0.0, 0.0), dtype=np.float32), vd.f32(2.0)),
            lookup=Lookup(weights=vd.storage.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))),
            bias=vd.storage.from_numpy(np.array([10.0, 20.0, 30.0], dtype=np.float32)),
        )
        accumulate(terms, output)
        np.testing.assert_allclose(output.to_numpy(), [12.5, 24.5, 36.5])


class AffineModule(vd.Module):
    def forward(self, terms: Affine, source: vd.TensorStorage) -> vd.TensorStorage:
        output = vd.empty_like(source)
        affine(terms, output, grid=(3, 1, 1))
        return output


class RegroupedAffine(vd.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = AffineModule()

    def forward(self, terms: Affine, bias: vd.TensorStorage[vd.f32, (3,), vd.read]) -> vd.TensorStorage:
        regrouped = Affine(gain=terms.gain, offset=terms.offset, lookup=terms.lookup, bias=bias)
        return self.inner(regrouped, bias)


def _affine_terms() -> Affine:
    return Affine(
        gain=2.0,
        offset=0.5,
        lookup=Lookup(weights=vd.storage.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))),
        bias=vd.storage.from_numpy(np.array([10.0, 20.0, 30.0], dtype=np.float32)),
    )


@expand_backend_matrix_tests
class BindingGroupModuleTests(unittest.TestCase):
    @backend_matrix_test(BackendRequirements(compute=True, storage_buffers=True))
    def test_forward_passes_a_group_argument_to_a_kernel(self, backend: BackendRow) -> None:
        vd.init(arch=backend.architecture)  # type: ignore[arg-type]
        output = AffineModule()(_affine_terms(), vd.storage.zeros(dtype=vd.f32, shape=(3,)))
        np.testing.assert_allclose(output.to_numpy(), [12.5, 24.5, 36.5])

    def test_forward_builds_a_group_from_leaves_and_passes_it_to_a_child(self) -> None:
        vd.init(arch=vd.cpu)
        bias = vd.storage.from_numpy(np.array([1.0, 1.0, 1.0], dtype=np.float32))
        output = RegroupedAffine()(_affine_terms(), bias)
        np.testing.assert_allclose(output.to_numpy(), [3.5, 5.5, 7.5])

    def test_vjp_by_leaf_and_by_group_root(self) -> None:
        vd.init(arch=vd.cpu)
        for wrt in (("terms.gain", "terms.lookup.weights", "terms.bias"), ("terms",)):
            with self.subTest(wrt=wrt):
                output, pullback = vd.ad.vjp(AffineModule(), wrt=wrt, outputs=("output",))(
                    _affine_terms(), vd.storage.zeros(dtype=vd.f32, shape=(3,))
                )
                np.testing.assert_allclose(output.to_numpy(), [12.5, 24.5, 36.5])
                raw = pullback({"output": np.ones((3,), dtype=np.float32)})
                self.assertIsInstance(raw["terms.lookup.weights"], vd.TensorStorage)
                self.assertIsInstance(raw["terms.bias"], vd.TensorStorage)
                gradients = {
                    path: np.asarray(value.to_numpy() if hasattr(value, "to_numpy") else value)
                    for path, value in raw.items()
                }
                np.testing.assert_allclose(gradients["terms.gain"], 6.0)
                np.testing.assert_allclose(gradients["terms.lookup.weights"], [2.0, 2.0, 2.0])
                np.testing.assert_allclose(gradients["terms.bias"], [1.0, 1.0, 1.0])
                if wrt == ("terms",):
                    np.testing.assert_allclose(gradients["terms.offset"], 3.0)


class BindingGroupVjpTests(unittest.TestCase):
    def setUp(self) -> None:
        vd.init(arch=vd.cpu)

    @staticmethod
    def _affine_terms() -> Affine:
        return Affine(
            gain=2.0,
            offset=0.5,
            lookup=Lookup(weights=vd.storage.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))),
            bias=vd.storage.from_numpy(np.array([10.0, 20.0, 30.0], dtype=np.float32)),
        )

    def test_cotangents_use_dotted_leaf_names(self) -> None:
        transformed = vd.ad.vjp(
            affine,
            wrt=("terms.gain", "terms.lookup.weights", "terms.bias"),
            outputs=("output",),
        )
        output = vd.storage.zeros(dtype=vd.f32, shape=(3,))
        _, pullback = transformed(self._affine_terms(), output, grid=(3, 1, 1))
        np.testing.assert_allclose(output.to_numpy(), [12.5, 24.5, 36.5])
        cotangent = np.zeros((1, 1, 3, 3), dtype=np.float32)
        cotangent[0, 0, np.arange(3), np.arange(3)] = 1.0
        gradients = {
            path: np.asarray(value.to_numpy() if hasattr(value, "to_numpy") else value)
            for path, value in pullback(cotangent).items()
        }
        self.assertEqual(set(gradients), {"terms.gain", "terms.lookup.weights", "terms.bias"})
        np.testing.assert_allclose(gradients["terms.gain"], 6.0)
        np.testing.assert_allclose(gradients["terms.lookup.weights"], [2.0, 2.0, 2.0])
        np.testing.assert_allclose(gradients["terms.bias"], [1.0, 1.0, 1.0])

    def test_group_root_wrt_expands_to_its_differentiable_leaves(self) -> None:
        transformed = vd.ad.vjp(affine, wrt=("terms",), outputs=("output",))
        self.assertEqual(
            transformed.transform.wrt, ("terms.bias", "terms.gain", "terms.lookup.weights", "terms.offset")
        )
        output = vd.storage.zeros(dtype=vd.f32, shape=(3,))
        _, pullback = transformed(self._affine_terms(), output, grid=(3, 1, 1))
        cotangent = np.zeros((1, 1, 3, 3), dtype=np.float32)
        cotangent[0, 0, np.arange(3), np.arange(3)] = 1.0
        raw = pullback(cotangent)
        self.assertIsInstance(raw["terms.lookup.weights"], vd.TensorStorage)
        self.assertIsInstance(raw["terms.bias"], vd.TensorStorage)
        gradients = {
            path: np.asarray(value.to_numpy() if hasattr(value, "to_numpy") else value) for path, value in raw.items()
        }
        self.assertEqual(set(gradients), set(transformed.transform.wrt))
        np.testing.assert_allclose(gradients["terms.gain"], 6.0)
        np.testing.assert_allclose(gradients["terms.offset"], 3.0)
        np.testing.assert_allclose(gradients["terms.lookup.weights"], [2.0, 2.0, 2.0])
        np.testing.assert_allclose(gradients["terms.bias"], [1.0, 1.0, 1.0])

    def test_group_root_expands_struct_value_leaves_to_fields(self) -> None:
        transformed = vd.ad.vjp(accumulate, wrt=("terms",), outputs=("output",))
        self.assertEqual(
            transformed.transform.wrt,
            ("terms.bias", "terms.lookup.weights", "terms.params.gain", "terms.params.tint"),
        )
        output = vd.storage.zeros(dtype=vd.f32, shape=(3,))
        terms = Accumulator(
            params=SurfaceParams(np.array((0.5, 0.0, 0.0, 0.0), dtype=np.float32), vd.f32(2.0)),
            lookup=Lookup(weights=vd.storage.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))),
            bias=vd.storage.from_numpy(np.array([10.0, 20.0, 30.0], dtype=np.float32)),
        )
        _, pullback = transformed(terms, output, grid=(3, 1, 1))
        cotangent = np.zeros((1, 1, 3, 3), dtype=np.float32)
        cotangent[0, 0, np.arange(3), np.arange(3)] = 1.0
        gradients = {
            path: np.asarray(value.to_numpy() if hasattr(value, "to_numpy") else value)
            for path, value in pullback(cotangent).items()
        }
        np.testing.assert_allclose(gradients["terms.params.gain"], 6.0)
        np.testing.assert_allclose(gradients["terms.params.tint"], [3.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(gradients["terms.lookup.weights"], [2.0, 2.0, 2.0])
        np.testing.assert_allclose(gradients["terms.bias"], [1.0, 1.0, 1.0])

    def test_group_without_selectable_leaves_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "'terms' names a binding group with no writable floating TensorView"):
            vd.ad.vjp(affine, wrt=("terms.gain",), outputs=("terms",))

    def test_texture_leaf_is_forward_only(self) -> None:
        image = vd.Texture.from_numpy(
            np.zeros((1, 1, 3, 4), dtype=np.float32),
            dimension="3d",
            format=vd.rgba32_float,
            usage=("storage", "transfer_source", "transfer_destination"),
        )
        transformed = vd.ad.vjp(sample_volume, wrt=("volume.gain",), outputs=("output",))
        output = vd.storage.zeros(dtype=vd.f32, shape=(3,))
        with self.assertRaisesRegex(ValueError, "Texture and Sampler are forward-only resources"):
            transformed(Volume(image=image, gain=2.0), output, grid=(3, 1, 1))


class _GraphicsGroupRuntimeCases:
    arch: Any
    requirements: BackendRequirements

    def setUp(self) -> None:
        require_available(probe_backend(backend_row(self.arch), self.requirements))
        vd.init(arch=self.arch)

    def tearDown(self) -> None:
        vd.init(arch=vd.cpu)

    def test_group_shared_by_both_stages_renders(self) -> None:
        target = vd.Texture.zeros(shape=(8, 8))
        vd.pipeline(surface_vertex, surface_fragment)(
            position=vd.storage.from_numpy(_FULL_SCREEN),
            surface=_surface(),
            flat=_flat(),
            render_pass=_render_pass(target),
        )
        expected = np.round(np.array((0.5 * 128 / 255 + 0.125, 0.375, 0.25, 0.75)) * 255.0)
        np.testing.assert_allclose(target.to_numpy()[4, 4].astype(np.float64), expected, atol=1.0)

    def test_module_forwards_group_arguments_to_a_pipeline(self) -> None:
        target = vd.Texture.zeros(shape=(8, 8))
        SurfaceModule()(vd.storage.from_numpy(_FULL_SCREEN), _surface(), _flat(), _render_pass(target))
        expected = np.round(np.array((0.5 * 128 / 255 + 0.125, 0.375, 0.25, 0.75)) * 255.0)
        np.testing.assert_allclose(target.to_numpy()[4, 4].astype(np.float64), expected, atol=1.0)


class SurfaceModule(vd.Module):
    def forward(
        self,
        position: vd.TensorStorage[vd.f32, (3, 2), vd.read],
        surface: Surface,
        flat: Flat,
        render_pass: vd.RenderPass,
    ) -> None:
        vd.pipeline(surface_vertex, surface_fragment)(
            position=position, surface=surface, flat=flat, render_pass=render_pass
        )


class OpenGLGraphicsGroupRuntimeTests(_GraphicsGroupRuntimeCases, unittest.TestCase):
    arch = vd.opengl
    requirements = BackendRequirements(graphics=True, minimum_api_version=(4, 3))


class OpenGLESGraphicsGroupRuntimeTests(_GraphicsGroupRuntimeCases, unittest.TestCase):
    arch = vd.opengles
    requirements = BackendRequirements(graphics=True, minimum_api_version=(3, 1))


class VulkanGraphicsGroupRuntimeTests(_GraphicsGroupRuntimeCases, unittest.TestCase):
    arch = vd.vulkan
    requirements = BackendRequirements(graphics=True)


class DirectXGraphicsGroupRuntimeTests(_GraphicsGroupRuntimeCases, unittest.TestCase):
    arch = vd.directx
    requirements = BackendRequirements(graphics=True)


if __name__ == "__main__":
    unittest.main()
