from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

from native_support import native_available
from vernon_dsl.program_assets import ProgramCompileError, cook_program_asset

_ASSET = """
from typing import Annotated

import vernon_dsl as vd


@vd.struct
class Varyings:
    position: Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]
    shade: vd.f32
    depth: {vertex_depth}
    identifier: {vertex_identifier}


@vd.struct
class Targets:
    color: vd.Vector[vd.f32, 4]
    identifier: {target_identifier}


@vd.vertex
def vertex_main(vertices: Annotated[vd.Vector[vd.f32, 2], vd.attribute()]) -> Varyings:
    return Varyings(vd.Vector([vertices, 0.0, 1.0]), 0.5, 0.25, {identifier_value})


@vd.fragment
def fragment_main(
    shade: Annotated[vd.f32, vd.varying()],
    depth: {fragment_depth},
    identifier: Annotated[vd.u32, vd.varying()],
) -> Targets:
    return Targets(vd.Vector([shade, depth, 0.0, 1.0]), {target_value})


asset = vd.program_asset(
    id="shaders/interface_rules",
    program=vd.pipeline(
        vertex_main,
        fragment_main,
        targets=vd.target_formats(colors={{0: vd.rgba8_unorm, 1: vd.{target_format}}}),
    ),
)
"""

_LINEAR = "Annotated[vd.f32, vd.varying(vd.Interpolation.LINEAR)]"


class GraphicsInterfaceRuleTests(unittest.TestCase):
    """Stages compile separately, so these rules are checked when the pipeline is planned."""

    def _cook(self, **overrides: str) -> dict[str, Any]:
        if not native_available():
            self.skipTest("native Vernon extension is not built")
        fields = {
            "vertex_depth": _LINEAR,
            "vertex_identifier": "vd.u32",
            "identifier_value": "vd.u32(7)",
            "fragment_depth": _LINEAR,
            "target_identifier": "vd.u32",
            "target_value": "identifier",
            "target_format": "r32_uint",
        } | overrides
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "interface_rules.py"
            source.write_text(_ASSET.format(**fields), encoding="utf-8")
            manifest = cook_program_asset(
                program_asset=f"{source}:asset", output=Path(directory) / "out", target="vulkan"
            )
            return cast(dict[str, Any], json.loads(manifest.read_text(encoding="utf-8")))

    def test_linkage_records_resolved_interpolation(self) -> None:
        artifacts = self._cook()["variants"][0]["artifact_system"]["artifacts"]
        linkage = next(iter(artifacts.values()))["reflection"]["graphics"]["linkage"]
        expected = [
            {"location": 0, "type": "f32", "interpolation": "smooth"},
            {"location": 1, "type": "f32", "interpolation": "noperspective"},
            {"location": 2, "type": "u32", "interpolation": "flat"},
        ]
        self.assertEqual(linkage["fragment_inputs"], expected)
        self.assertEqual(linkage["vertex_outputs"][1:], expected)

    def test_interpolation_must_agree_between_stages(self) -> None:
        with self.assertRaisesRegex(ProgramCompileError, "varying interpolation differs between stages"):
            self._cook(vertex_depth="vd.f32")

    def test_integer_varyings_must_be_flat(self) -> None:
        with self.assertRaisesRegex(ProgramCompileError, "integer varyings must be flat"):
            self._cook(vertex_identifier="Annotated[vd.u32, vd.varying(vd.Interpolation.PERSPECTIVE)]")

    def test_varying_types_must_agree_between_stages(self) -> None:
        with self.assertRaisesRegex(ProgramCompileError, "has type u32, but the vertex output has type i32"):
            self._cook(vertex_identifier="vd.i32", identifier_value="vd.i32(7)")

    def test_fragment_output_follows_the_target_component_class(self) -> None:
        with self.assertRaisesRegex(
            ProgramCompileError, "fragment output type does not match color target format: location 1 writes f32"
        ):
            self._cook(target_identifier="vd.f32", target_value="shade")
        with self.assertRaisesRegex(ProgramCompileError, "writes u32 to rg32_uint, which takes tensor<2xu32>"):
            self._cook(target_format="rg32_uint")
        self._cook(
            target_format="rg32_uint",
            target_identifier="vd.Vector[vd.u32, 2]",
            target_value="vd.Vector([identifier, identifier])",
        )


if __name__ == "__main__":
    unittest.main()
