from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import vernon_dsl as vd
from vernon_dsl._ad_reference import reference_vjp
from vernon_dsl.bundle import BundlePlan, CpuTargetOptions
from vernon_dsl.compiler import Compiler, FrontendCompileRequest
from vernon_dsl.frontend.autodiff import (
    AccessPatternEvidence,
    AccumulationMode,
    OpCode,
    StaticIndicesOp,
)
from vernon_dsl.frontend.autodiff_cpu import execute_program_graph as _execute_program_graph
from vernon_dsl.frontend.autodiff_native import (
    AutodiffNativeLoweringError,
    emit_native_autodiff_modules,
)
from vernon_dsl.pipeline_assets import (
    PipelineCompileError,
    cook_pipeline_asset,
    parse_python_pipeline_asset,
)


def execute_program_graph(program, bindings, grid=(1, 1, 1)):
    return _execute_program_graph(program, bindings, grid)


class AutodiffDeclarationTests(unittest.TestCase):
    def test_compute_vjp_has_deterministic_canonical_identity(self) -> None:
        @vd.kernel
        def compute(value: vd.f32) -> vd.f32:
            return value

        first = vd.ad.vjp(compute, wrt=("z", "value"))
        second = vd.ad.vjp(compute, wrt=["value", "z"])
        self.assertEqual(first.transform.wrt, ("value", "z"))
        self.assertEqual(first.transform.identity, second.transform.identity)

    def test_graphics_vjp_requires_rules(self) -> None:
        @vd.vertex
        def vertex(value: vd.f32) -> vd.f32:
            return value

        @vd.fragment
        def fragment(value: vd.f32) -> vd.f32:
            return value

        with self.assertRaisesRegex(ValueError, "custom rule set"):
            vd.ad.vjp((vertex, fragment), wrt=("value",))

        rules = vd.ad.rule_set(
            id="render/v1",
            rasterization=lambda: None,
            visibility=lambda: None,
            depth=lambda: None,
            blend=lambda: None,
            texture=lambda: None,
        )
        expression = vd.ad.vjp((vertex, fragment), wrt=("value",), rules=rules)
        declaration = vd.pipeline_asset(id="render", program=expression)
        self.assertIs(declaration.program, expression)

    def test_invalid_or_duplicate_wrt_is_rejected(self) -> None:
        @vd.kernel
        def compute(value: vd.f32) -> vd.f32:
            return value

        with self.assertRaisesRegex(ValueError, "canonical source paths"):
            vd.ad.vjp(compute, wrt=("value[0]",))
        with self.assertRaisesRegex(ValueError, "unique"):
            vd.ad.vjp(compute, wrt=("value", "value"))

    def test_bundle_schema_rejects_partial_differentiated_profiles(self) -> None:
        with self.assertRaisesRegex(PipelineCompileError, "require both"):
            BundlePlan(
                "partial",
                CpuTargetOptions(),
                (),
                (),
                (),
                transform={"kind": "vjp"},
            )


class AutodiffParsingTests(unittest.TestCase):
    def test_vjp_pipeline_is_parsed_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "asset.py"
            source.write_text(
                """
import vernon_dsl as vd
raise RuntimeError("must not execute")

@vd.kernel
def compute(value: vd.f32) -> vd.f32:
    return value * value

asset = vd.pipeline_asset(
    id="compute/vjp",
    program=vd.ad.vjp(compute, wrt=("value",)),
)
""",
                encoding="utf-8",
            )
            descriptor = parse_python_pipeline_asset(source, "asset")
            self.assertEqual(descriptor.transform["kind"], "vjp")
            self.assertEqual(descriptor.transform["wrt"], ["value"])
            manifest = json.loads(descriptor.canonical_manifest)
            self.assertEqual(manifest["transform"], descriptor.transform)
            self.assertNotIn("dispatch_grid", manifest)
            self.assertEqual(set(descriptor.stages), {"compute"})

    def test_graphics_vjp_without_rules_is_rejected_statically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "asset.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.vertex
def vertex(value: vd.f32) -> vd.f32:
    return value

@vd.fragment
def fragment(value: vd.f32) -> vd.f32:
    return value

asset = vd.pipeline_asset(
    id="graphics/vjp",
    program=vd.ad.vjp((vertex, fragment), wrt=("value",)),
)
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PipelineCompileError, "custom rule set"):
                parse_python_pipeline_asset(source, "asset")

    def test_static_and_runtime_rule_set_identities_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "graphics.py"
            source.write_text(
                """
import vernon_dsl as vd

def raster_vjp(): pass
def visibility_vjp(): pass
def depth_vjp(): pass
def blend_vjp(): pass
def texture_vjp(): pass

rules = vd.ad.rule_set(
    id="render/v1",
    rasterization=raster_vjp,
    visibility=visibility_vjp,
    depth=depth_vjp,
    blend=blend_vjp,
    texture=texture_vjp,
)

@vd.vertex
def vertex(value: vd.f32) -> vd.f32:
    return value

@vd.fragment
def fragment(value: vd.f32) -> vd.f32:
    return value

asset = vd.pipeline_asset(
    id="graphics/vjp",
    program=vd.ad.vjp((vertex, fragment), wrt=("value",), rules=rules),
)
""",
                encoding="utf-8",
            )
            descriptor = parse_python_pipeline_asset(source, "asset")
            runtime_rules = vd.ad.rule_set(
                id="render/v1",
                rasterization=lambda: None,
                visibility=lambda: None,
                depth=lambda: None,
                blend=lambda: None,
                texture=lambda: None,
            )
            self.assertEqual(descriptor.transform["rule_set_identity"], runtime_rules.digest)

    def test_gpu_cooker_emits_all_differentiated_profiles(self) -> None:
        try:
            from vernon_dsl import _native  # noqa: F401
        except (ImportError, OSError):
            self.skipTest("native Vernon compiler is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "asset.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def compute(value: vd.f32) -> vd.f32:
    return value * value

asset = vd.pipeline_asset(
    id="compute/vjp",
    program=vd.ad.vjp(compute, wrt=("value",)),
)
""",
                encoding="utf-8",
            )
            for target in ("vulkan", "metal", "cuda", "opengl", "opengles"):
                with self.subTest(target=target):
                    manifest = cook_pipeline_asset(
                        pipeline_asset=f"{source}:asset",
                        output=root / f"{target}-build",
                        target=target,
                    )
                    document = json.loads(manifest.read_text(encoding="utf-8"))
                    programs = document["autodiff_profiles"]["variants"][0]["programs"]
                    self.assertEqual(set(programs), {"primal", "forward_with_tape", "backward"})
                    profile_stages = {
                        record["autodiff_profile"]: record
                        for record in document["stage_artifacts"].values()
                        if "autodiff_profile" in record
                    }
                    self.assertEqual(set(profile_stages), {"forward_with_tape", "backward"})
                    expected_roles = {
                        "forward_with_tape": {"input", "output", "tape"},
                        "backward": {"input", "tape", "cotangent", "gradient"},
                    }
                    for profile, record in profile_stages.items():
                        arguments = [
                            argument for entry in record["reflection"]["entries"] for argument in entry["arguments"]
                        ]
                        self.assertIn(
                            "output",
                            {argument["vernon.source_name"] for argument in arguments},
                        )
                        self.assertEqual(
                            {
                                argument["vernon.autodiff_role"]
                                for argument in arguments
                                if "vernon.autodiff_role" in argument
                            },
                            expected_roles[profile],
                        )
                        launch = next(
                            argument
                            for argument in arguments
                            if argument.get("vernon.source_name") == "__vernon_launch"
                        )
                        self.assertEqual(launch["element_layout"]["leaves"][0]["dtype"], "u32")
                        carriers = [
                            argument
                            for argument in arguments
                            if argument.get("vernon.autodiff_role") in {"output", "tape", "cotangent"}
                        ]
                        self.assertTrue(carriers)

    def test_available_opengl_backends_execute_cooked_vjp(self) -> None:
        try:
            from vernon_dsl import _native  # noqa: F401
            from vernon_dsl._runtime import session
        except (ImportError, OSError):
            self.skipTest("native Vernon runtime is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "asset.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def compute(value: vd.Tensor[vd.f32, (2,)]) -> vd.Tensor[vd.f32, (2,)]:
    return value * value

asset = vd.pipeline_asset(
    id="compute/vjp",
    program=vd.ad.vjp(compute, wrt=("value",)),
)
""",
                encoding="utf-8",
            )
            for architecture, target, api_version in (
                (vd.opengl, "opengl", (4, 3)),
                (vd.opengles, "opengles", (3, 1)),
            ):
                with self.subTest(target=target):
                    manifest = cook_pipeline_asset(
                        pipeline_asset=f"{source}:asset",
                        output=root / f"{target}-build",
                        target=target,
                    )
                    try:
                        vd.init(arch=architecture, api_version=api_version)
                    except RuntimeError:
                        continue
                    try:
                        assert session._native_runtime is not None
                        pipeline = session._native_runtime.load_pipeline_asset(
                            manifest.read_bytes(),
                            str(manifest.parent),
                            [],
                        )
                        output, pullback = pipeline.vjp(
                            {"value": np.array([2.0, 3.0], dtype=np.float32)},
                            (1, 1, 1),
                        )
                        np.testing.assert_array_equal(output, np.array([4.0, 9.0], dtype=np.float32))
                        gradients = pullback(np.array([1.0, 2.0], dtype=np.float32))
                        np.testing.assert_array_equal(
                            gradients["value"],
                            np.array([4.0, 12.0], dtype=np.float32),
                        )
                    finally:
                        vd.init(arch=vd.cpu)

    def test_cpu_cooker_emits_all_differentiated_profiles(self) -> None:
        try:
            from vernon_dsl import _native  # noqa: F401
        except (ImportError, OSError):
            self.skipTest("native Vernon compiler is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "asset.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def objective(x: vd.f32, y: vd.f32) -> vd.f32:
    return vd.sin(x * y) + x / y

asset = vd.pipeline_asset(
    id="compute/vjp",
    program=vd.ad.vjp(objective, wrt=("x", "y")),
)
""",
                encoding="utf-8",
            )
            manifest = cook_pipeline_asset(
                pipeline_asset=f"{source}:asset",
                output=root / "adbuild",
                target="cpu",
            )
            first_manifest = manifest.read_bytes()
            repeated = cook_pipeline_asset(
                pipeline_asset=f"{source}:asset",
                output=root / "adbuild",
                target="cpu",
            )
            self.assertEqual(repeated.read_bytes(), first_manifest)
            document = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(document["program_transform"]["kind"], "vjp")
            self.assertEqual(document["program_transform"]["output_cotangents"], ["output"])
            differentiated = document["autodiff_profiles"]
            self.assertRegex(differentiated["identity"], r"^[0-9a-f]{64}$")
            self.assertEqual(len(differentiated["variants"]), 1)
            programs = differentiated["variants"][0]["programs"]
            self.assertEqual(
                set(programs),
                {"primal", "forward_with_tape", "backward"},
            )
            self.assertEqual(
                programs["primal"],
                document["variants"][0]["program"],
            )
            profile_stages = {
                record["autodiff_profile"]: record
                for record in document["stage_artifacts"].values()
                if "autodiff_profile" in record
            }
            self.assertEqual(
                set(profile_stages),
                {"forward_with_tape", "backward"},
            )
            self.assertEqual(len(document["stage_artifacts"]), 3)
            for record in document["stage_artifacts"].values():
                artifact = record["artifact"]
                self.assertEqual(record["format"], "relocatable_object")
                self.assertEqual(artifact["format"], "relocatable_object")
                self.assertTrue((manifest.parent / artifact["path"]).is_file())
            registration = document["cpu_static_registration"]
            self.assertEqual(
                registration["symbols"],
                sorted(record["symbol"] for record in document["stage_artifacts"].values()),
            )
            registration_source = (manifest.parent / registration["source"]).read_text(encoding="utf-8")
            for symbol in registration["symbols"]:
                self.assertIn(f"&{symbol}", registration_source)


class AutodiffReferenceTests(unittest.TestCase):
    def test_scalar_pullback_defaults_to_one_and_is_reusable(self) -> None:
        def objective(x: np.float32, y: np.float32) -> np.float32:
            return x * x + x * y + np.sin(y)

        output, pullback = reference_vjp(
            objective,
            {"x": np.float32(2), "y": np.float32(0.5)},
            ("x", "y"),
        )
        self.assertAlmostEqual(output, 4 + 1 + np.sin(0.5), places=6)
        first = pullback()
        second = pullback(np.float32(2))
        self.assertAlmostEqual(first["x"], 4.5, places=6)
        self.assertAlmostEqual(first["y"], 2 + np.cos(0.5), places=6)
        self.assertAlmostEqual(second["x"], 9, places=6)
        self.assertAlmostEqual(second["y"], 2 * (2 + np.cos(0.5)), places=6)

    def test_tensor_vjp_requires_matching_cotangent(self) -> None:
        def scale(x: np.ndarray) -> np.ndarray:
            return x * x

        _, pullback = reference_vjp(scale, {"x": np.array([1, 2], dtype=np.float16)}, ("x",))
        with self.assertRaisesRegex(TypeError, "explicit cotangent"):
            pullback()
        gradient = pullback(np.array([3, 4], dtype=np.float32))["x"]
        np.testing.assert_array_equal(gradient, np.array([6, 16], dtype=np.float32))
        self.assertEqual(gradient.dtype, np.float32)


class AutodiffFrontendTests(unittest.TestCase):
    def assert_native_modules_compile(self, modules: dict[str, str]) -> None:
        try:
            from vernon_dsl import _native as native
        except (ImportError, OSError):
            self.skipTest("native Vernon compiler is unavailable")
        compiler = native.Compiler()
        for profile, mlir in modules.items():
            program = compiler.compile_program_result(mlir, native.Target.CPU)
            self.assertTrue(program.ok, f"{profile}: {program.diagnostics}")

    def assert_gpu_autodiff_modules_compile(self, graph, profiles) -> None:
        try:
            from vernon_dsl import _native as native
        except (ImportError, OSError):
            self.skipTest("native Vernon compiler is unavailable")
        compiler = native.Compiler()
        for target_name, target, options in (
            ("cuda", native.Target.CUDA, {}),
            ("vulkan", native.Target.VULKAN, {}),
            ("metal", native.Target.METAL, {}),
            ("opengl", native.Target.OPENGL, {"version": 430}),
            ("opengles", native.Target.OPENGL_ES, {"version": 310}),
        ):
            modules = emit_native_autodiff_modules(graph, profiles, target=target_name)
            for profile, mlir in modules.items():
                self.assertIn("!vernon.tensor_view", mlir)
                program = compiler.compile_program_result(mlir, target, options)
                self.assertTrue(program.ok, f"{target_name}/{profile}: {program.diagnostics}")
                self.assertTrue(program.artifacts)
                reflection = json.loads(program.reflection)
                self.assertEqual(len(reflection["entries"]), 1)
                entry = reflection["entries"][0]
                self.assertFalse(entry["results"])
                self.assertTrue(entry["arguments"])
                resources = [
                    argument for argument in entry["arguments"] if argument.get("vernon.interface") == "resource"
                ]
                self.assertTrue(resources)
                self.assertTrue(all(argument["kind"] == "tensor" for argument in resources))
                self.assertTrue(all(argument["type"].startswith("!vernon.tensor_view<") for argument in resources))

    def test_transform_participates_in_frontend_identity_and_mlir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "value.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def square(value: vd.f32) -> vd.f32:
    return value * value
""",
                encoding="utf-8",
            )
            transform = vd.ad.ProgramTransformSpec("vjp", ("value",))
            result = Compiler().compile_request(FrontendCompileRequest(source, "square", program_transform=transform))
            self.assertEqual(result.semantic_inputs["program_transform"], transform.to_dict())
            self.assertNotIn("dispatch_grid", transform.to_dict())
            self.assertIn("vernon.program_transform", result.mlir)
            self.assertIn(transform.identity, result.mlir)
            graph = result.program_graph
            self.assertIsNotNone(graph)
            assert graph is not None
            self.assertEqual(set(graph.to_dict()), {"semantic", "reverse", "tape"})
            self.assertEqual(
                [node.operation for node in graph.semantic.nodes],
                [OpCode.PARAMETER, OpCode.MUL],
            )
            self.assertEqual(graph.reverse.saved_values, (0,))
            self.assertEqual(graph.tape.bytes, 4)
            self.assertEqual(graph.reverse.reverse_order, (1,))
            self.assertEqual(graph.tape.slots[0].to_dict()["size"], 4)
            self.assertEqual(graph.reverse.cotangent_paths, ("output",))
            self.assertEqual(graph.reverse.gradient_paths, ("value",))
            self.assertEqual(graph.reverse.derivative_rules, ("mul",))
            self.assertIn(graph.identity, result.mlir)
            profiles = result.autodiff_profiles
            self.assertIsNotNone(profiles)
            assert profiles is not None
            self.assertNotIn("launch", result.semantic_inputs["program_graph"])
            self.assertEqual(
                result.semantic_inputs["autodiff_profiles"]["launch"],
                graph.launch.to_dict(),
            )
            self.assertEqual(
                [profile.name for profile in profiles.profiles],
                ["primal", "forward_with_tape", "backward"],
            )
            forward = profiles.profiles[1]
            backward = profiles.profiles[2]
            self.assertRegex(forward.symbol, r"^vernon_ad_[0-9a-f]{20}_forward$")
            self.assertRegex(backward.symbol, r"^vernon_ad_[0-9a-f]{20}_backward$")
            self.assertEqual(forward.outputs[-1].role, "tape")
            self.assertEqual(
                backward.inputs[-1].to_dict(),
                {
                    "path": "output",
                    "type": "f32",
                    "role": "cotangent",
                },
            )
            self.assertEqual(
                backward.outputs[0].to_dict(),
                {
                    "path": "value",
                    "type": "f32",
                    "role": "gradient",
                },
            )
            self.assertIn(profiles.identity, result.mlir)
            output, pullback = execute_program_graph(graph, {"value": np.float32(3)})
            self.assertAlmostEqual(output, 9)
            self.assertAlmostEqual(pullback()["value"], 6)
            self.assertAlmostEqual(pullback(np.float32(2))["value"], 12)

    def test_runtime_grid_is_not_part_of_frontend_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "launch_identity.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def square(value: vd.f32) -> vd.f32:
    return value * value
""",
                encoding="utf-8",
            )
            transform = vd.ad.ProgramTransformSpec("vjp", ("value",))
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "square",
                    program_transform=transform,
                )
            )
            assert result.program_graph is not None
            assert result.autodiff_profiles is not None
            identity = result.program_graph.identity
            profile_identity = result.autodiff_profiles.identity
            output, pullback = _execute_program_graph(
                result.program_graph,
                {"value": np.float32(3)},
                (2, 3, 4),
            )
            np.testing.assert_array_equal(output, np.full(24, 9, dtype=np.float32))
            self.assertAlmostEqual(
                pullback(np.ones(24, dtype=np.float32))["value"],
                144,
            )
            self.assertEqual(result.program_graph.identity, identity)
            self.assertEqual(result.autodiff_profiles.identity, profile_identity)
            self.assertNotIn("dispatch_grid", result.semantic_inputs)
            self.assertNotIn(
                "dispatch_grid",
                result.semantic_inputs["autodiff_profiles"]["launch"],
            )
            for invalid_grid in ((1, 1), (0, 1, 1), (1, True, 1)):
                with (
                    self.subTest(grid=invalid_grid),
                    self.assertRaisesRegex(
                        ValueError,
                        "three positive integer extents",
                    ),
                ):
                    _execute_program_graph(
                        result.program_graph,
                        {"value": np.float32(3)},
                        invalid_grid,
                    )

    def test_transform_rejects_unknown_or_integer_wrt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "invalid.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def value(index: vd.i32) -> vd.f32:
    return 1.0
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(vd.CompileError, "no differentiable floating leaves"):
                Compiler().compile_request(
                    FrontendCompileRequest(
                        source,
                        "value",
                        program_transform=vd.ad.ProgramTransformSpec("vjp", ("index",)),
                    )
                )

    def test_compiler_graph_vjp_matches_finite_difference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "objective.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def objective(x: vd.f64, y: vd.f64) -> vd.f64:
    product = x * y
    return vd.sin(product) + x / y
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "objective",
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("x", "y")),
                )
            )
            assert result.program_graph is not None
            assert result.autodiff_profiles is not None
            cuda_modules = emit_native_autodiff_modules(
                result.program_graph,
                result.autodiff_profiles,
                target="cuda",
            )
            cuda_backward = cuda_modules["backward"]
            self.assertIn("vernon.workgroup_size = array<i32: 1, 1, 1>", cuda_backward)
            self.assertGreaterEqual(cuda_backward.count("scf.for"), 3)
            bindings = {"x": np.float64(1.3), "y": np.float64(0.7)}
            output, pullback = execute_program_graph(result.program_graph, bindings)
            gradients = pullback()
            epsilon = 1e-6

            def objective(x: float, y: float) -> float:
                return float(np.sin(x * y) + x / y)

            for name in ("x", "y"):
                positive = dict(bindings)
                negative = dict(bindings)
                positive[name] += epsilon
                negative[name] -= epsilon
                finite_difference = (objective(**positive) - objective(**negative)) / (2 * epsilon)
                self.assertAlmostEqual(gradients[name], finite_difference, places=6)
            self.assertAlmostEqual(output, objective(**bindings), places=12)

    def test_compiler_graph_tensor_pullback_uses_gradient_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tensor.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def square(values: vd.Tensor[vd.f16, (2,)]) -> vd.Tensor[vd.f16, (2,)]:
    return values * values
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "square",
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("values",)),
                )
            )
            assert result.program_graph is not None
            values = np.array([2, 3], dtype=np.float16)
            output, pullback = execute_program_graph(result.program_graph, {"values": values})
            np.testing.assert_array_equal(output, np.array([4, 9], dtype=np.float16))
            with self.assertRaisesRegex(TypeError, "explicit cotangent"):
                pullback()
            gradient = pullback(np.array([3, 4], dtype=np.float32))["values"]
            np.testing.assert_array_equal(gradient, np.array([12, 24], dtype=np.float32))
            self.assertEqual(gradient.dtype, np.float32)

    def test_struct_wrt_path_participates_in_pullback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "struct.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.struct
class Pair:
    x: vd.f32
    y: vd.f32

@vd.kernel
def product(pair: Pair) -> vd.f32:
    return pair.x * pair.y
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "product",
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("pair.x",)),
                )
            )
            assert result.program_graph is not None
            self.assertEqual(result.program_graph.reverse.gradient_paths, ("pair.x",))
            output, pullback = execute_program_graph(
                result.program_graph,
                {"pair": {"x": np.float32(2), "y": np.float32(5)}},
            )
            self.assertAlmostEqual(output, 10)
            self.assertAlmostEqual(pullback()["pair.x"], 5)

    def test_aggregate_wrt_path_requires_a_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "aggregate_wrt.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.struct
class Pair:
    x: vd.f32
    y: vd.f32

@vd.kernel
def product(pair: Pair) -> vd.f32:
    return pair.x * pair.y
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                vd.CompileError,
                "must resolve to one differentiable Scalar or Tensor leaf",
            ):
                Compiler().compile_request(
                    FrontendCompileRequest(
                        source,
                        "product",
                        program_transform=vd.ad.ProgramTransformSpec("vjp", ("pair",)),
                    )
                )

    def test_struct_output_requires_matching_cotangent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "struct_output.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.struct
class Pair:
    linear: vd.f32
    square: vd.f32

@vd.kernel
def split(value: vd.f32) -> Pair:
    return Pair(value, value * value)
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "split",
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("value",)),
                )
            )
            assert result.program_graph is not None
            output, pullback = execute_program_graph(
                result.program_graph,
                {"value": np.float32(4)},
            )
            self.assertEqual(output, {"linear": 4, "square": 16})
            with self.assertRaisesRegex(TypeError, "explicit cotangent"):
                pullback()
            gradient = pullback({"output.linear": np.float32(2), "output.square": np.float32(3)})["value"]
            self.assertAlmostEqual(gradient, 26)

    def test_static_tensor_index_generates_scatter_adjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "index.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def select(values: vd.Tensor[vd.f32, (3,)]) -> vd.f32:
    return values[1] * values[1]
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "select",
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("values",)),
                )
            )
            assert result.program_graph is not None
            index = next(node for node in result.program_graph.semantic.nodes if node.operation is OpCode.INDEX)
            self.assertEqual(index.payload, StaticIndicesOp((1,)))
            _, pullback = execute_program_graph(
                result.program_graph,
                {"values": np.array([2, 3, 4], dtype=np.float32)},
            )
            np.testing.assert_array_equal(
                pullback()["values"],
                np.array([0, 6, 0], dtype=np.float32),
            )
            assert result.autodiff_profiles is not None
            modules = emit_native_autodiff_modules(result.program_graph, result.autodiff_profiles)
            self.assert_native_modules_compile(modules)

    def test_tensor_view_store_returns_fresh_storage_gradient(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "storage_vjp.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def mutate(
    values: vd.TensorView[vd.f32, (3,), vd.read_write],
    scale: vd.f32,
) -> vd.f32:
    old = values[1]
    values[1] = old * scale
    return values[0] + values[1]
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "mutate",
                    program_transform=vd.ad.ProgramTransformSpec(
                        "vjp",
                        ("values", "scale"),
                    ),
                )
            )
            assert result.program_graph is not None
            assert result.autodiff_profiles is not None
            self.assertEqual(
                [(plan.path, plan.mode) for plan in result.program_graph.launch.accumulation_plans],
                [
                    ("scale", AccumulationMode.REDUCE_SUM),
                    ("values", AccumulationMode.SCATTER_ADD),
                ],
            )
            self.assertEqual(
                result.program_graph.launch.accumulation_plans[1].evidence,
                (AccessPatternEvidence.STATIC_INDEX_CONFLICT,),
            )
            backward = next(profile for profile in result.autodiff_profiles.profiles if profile.name == "backward")
            self.assertEqual(
                [(binding.path, binding.type) for binding in backward.outputs],
                [("scale", "f32"), ("values", "tensor<3xf32>")],
            )
            values = np.array([2, 3, 4], dtype=np.float32)
            output, pullback = execute_program_graph(
                result.program_graph,
                {"values": values, "scale": np.float32(2)},
            )
            self.assertAlmostEqual(output, 8)
            np.testing.assert_array_equal(values, np.array([2, 6, 4], dtype=np.float32))
            first = pullback()
            second = pullback(np.float32(2))
            np.testing.assert_array_equal(
                first["values"],
                np.array([1, 2, 0], dtype=np.float32),
            )
            self.assertAlmostEqual(first["scale"], 3)
            np.testing.assert_array_equal(
                second["values"],
                np.array([2, 4, 0], dtype=np.float32),
            )
            self.assertAlmostEqual(second["scale"], 6)
            self.assertIsNot(first["values"], second["values"])
            modules = emit_native_autodiff_modules(result.program_graph, result.autodiff_profiles)
            self.assert_native_modules_compile(modules)
            gpu_modules = emit_native_autodiff_modules(
                result.program_graph,
                result.autodiff_profiles,
                target="metal",
            )
            self.assertIn('vernon.autodiff_role = "storage"', gpu_modules["forward_with_tape"])
            self.assertIn(
                '!vernon.tensor_view<f32, [3], "read_write", "device">',
                gpu_modules["forward_with_tape"],
            )
            self.assertIn(
                '!vernon.tensor_view<f32, [3], "read_write", "device">',
                gpu_modules["backward"],
            )
            self.assertNotIn('name = "construct"', gpu_modules["forward_with_tape"])
            self.assertIn('"vernon.reduce_sum"', gpu_modules["backward"])
            self.assertIn('"vernon.scatter_add"', gpu_modules["backward"])
            self.assert_gpu_autodiff_modules_compile(result.program_graph, result.autodiff_profiles)

    def test_tensor_view_store_executes_when_return_value_does_not_read_storage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "storage_effect_root.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def write_and_square(
    values: vd.TensorView[vd.f32, (2,), vd.write],
    x: vd.f32,
) -> vd.f32:
    values[0] = x
    return x * x
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "write_and_square",
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("x",)),
                )
            )
            assert result.program_graph is not None
            assert result.autodiff_profiles is not None
            values = np.zeros(2, dtype=np.float32)
            output, pullback = execute_program_graph(
                result.program_graph,
                {"values": values, "x": np.float32(3)},
            )
            self.assertAlmostEqual(output, 9)
            np.testing.assert_array_equal(values, np.array([3, 0], dtype=np.float32))
            self.assertAlmostEqual(pullback()["x"], 6)
            modules = emit_native_autodiff_modules(result.program_graph, result.autodiff_profiles)
            forward = modules["forward_with_tape"]
            self.assertIn('"vernon.store"', forward)
            self.assertIn('owner = "values"', forward)
            self.assertIn('kind = "write"', forward)
            self.assert_native_modules_compile(modules)

    def test_dynamic_gather_has_grid_independent_semantic_accumulation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "parallel_gather_vjp.py"
            source.write_text(
                """
from typing import Annotated
import vernon_dsl as vd

@vd.kernel(workgroup_size=(4, 1, 1))
def gather(
    values: vd.TensorView[vd.f32, (3,), vd.read],
    scale: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> vd.f32:
    return values[gid[0]] * scale
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "gather",
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("scale", "values")),
                    workgroup_size=(4, 1, 1),
                )
            )
            assert result.program_graph is not None
            self.assertEqual(
                [(plan.path, plan.mode) for plan in result.program_graph.launch.accumulation_plans],
                [
                    ("scale", AccumulationMode.REDUCE_SUM),
                    ("values", AccumulationMode.SCATTER_ADD),
                ],
            )
            self.assertEqual(
                result.program_graph.launch.accumulation_plans[1].invocation_axes,
                (0,),
            )
            output, pullback = execute_program_graph(
                result.program_graph,
                {
                    "values": np.array([1.0, 2.0, 4.0], dtype=np.float32),
                    "scale": np.float32(2.0),
                },
                (3, 2, 2),
            )
            np.testing.assert_array_equal(
                output,
                np.array([2.0, 4.0, 8.0] * 4, dtype=np.float32),
            )
            gradients = pullback(np.ones(12, dtype=np.float32))
            np.testing.assert_array_equal(
                gradients["values"],
                np.array([8.0, 8.0, 8.0], dtype=np.float32),
            )
            self.assertAlmostEqual(gradients["scale"], 28.0)

    def test_injective_gather_records_disjoint_scatter_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "parallel_disjoint_gather.py"
            source.write_text(
                """
from typing import Annotated
import vernon_dsl as vd

@vd.kernel(workgroup_size=(4, 1, 1))
def gather(
    values: vd.TensorView[vd.f32, (2, 2, 2), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> vd.f32:
    return values[gid[0], gid[1], gid[2]]
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "gather",
                    workgroup_size=(4, 1, 1),
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("values",)),
                )
            )
            assert result.program_graph is not None
            assert result.autodiff_profiles is not None
            accumulation = result.program_graph.launch.accumulation_plans
            self.assertEqual(len(accumulation), 1)
            self.assertEqual(
                accumulation[0].mode,
                AccumulationMode.SCATTER_ADD,
            )
            self.assertEqual(
                accumulation[0].evidence,
                (
                    AccessPatternEvidence.DISJOINT_SCATTER,
                    AccessPatternEvidence.INJECTIVE_GLOBAL_INDEX,
                ),
            )
            self.assertEqual(accumulation[0].invocation_axes, (0, 1, 2))
            modules = emit_native_autodiff_modules(
                result.program_graph,
                result.autodiff_profiles,
                target="cuda",
            )
            self.assertIn("deterministic = false, disjoint", modules["backward"])
            self.assert_gpu_autodiff_modules_compile(result.program_graph, result.autodiff_profiles)

    def test_multiple_injective_maps_do_not_claim_disjoint_scatter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "crossed_gather.py"
            source.write_text(
                """
from typing import Annotated
import vernon_dsl as vd

@vd.kernel(workgroup_size=(4, 1, 1))
def gather(
    values: vd.TensorView[vd.f32, (2, 2, 2), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> vd.f32:
    return values[gid[0], gid[1], gid[2]] + values[gid[1], gid[0], gid[2]]
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "gather",
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("values",)),
                )
            )
            assert result.program_graph is not None
            accumulation = result.program_graph.launch.accumulation_plans[0]
            self.assertNotIn(AccessPatternEvidence.DISJOINT_SCATTER, accumulation.evidence)

    def test_static_gather_records_scatter_add_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "static_scatter.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def gather(values: vd.TensorView[vd.f32, (3,), vd.read]) -> vd.f32:
    return values[1]
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "gather",
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("values",)),
                )
            )
            assert result.program_graph is not None
            accumulation = result.program_graph.launch.accumulation_plans
            self.assertEqual(accumulation[0].mode, AccumulationMode.SCATTER_ADD)
            self.assertEqual(
                accumulation[0].evidence,
                (AccessPatternEvidence.STATIC_INDEX_CONFLICT,),
            )
            self.assertNotIn("required_capabilities", accumulation[0].to_dict())

    def test_gpu_dynamic_storage_lowering_is_constant_size(self) -> None:
        modules_by_extent: dict[int, dict[str, str]] = {}
        with tempfile.TemporaryDirectory() as directory:
            for extent in (3, 1024):
                source = Path(directory) / f"gather_{extent}.py"
                source.write_text(
                    f"""
from typing import Annotated
import vernon_dsl as vd

@vd.kernel(workgroup_size=(4, 1, 1))
def gather(
    values: vd.TensorView[vd.f32, ({extent},), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> vd.f32:
    return values[gid[0]]
""",
                    encoding="utf-8",
                )
                result = Compiler().compile_request(
                    FrontendCompileRequest(
                        source,
                        "gather",
                        workgroup_size=(4, 1, 1),
                        program_transform=vd.ad.ProgramTransformSpec("vjp", ("values",)),
                    )
                )
                assert result.program_graph is not None
                assert result.autodiff_profiles is not None
                modules_by_extent[extent] = emit_native_autodiff_modules(
                    result.program_graph,
                    result.autodiff_profiles,
                    target="metal",
                )
        for profile in ("forward_with_tape", "backward"):
            small = modules_by_extent[3][profile]
            large = modules_by_extent[1024][profile]
            self.assertEqual(small.count('"vernon.load"'), large.count('"vernon.load"'))
            self.assertEqual(small.count('"vernon.store"'), large.count('"vernon.store"'))
            self.assertEqual(small.count('"vernon.atomic"'), large.count('"vernon.atomic"'))
            self.assertNotIn('name = "construct"', large)
            self.assertNotIn("arith.cmpi eq", large)
            self.assertNotIn("arith.select", large)

    def test_gpu_dynamic_storage_store_uses_direct_resource_ops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dynamic_store.py"
            source.write_text(
                """
from typing import Annotated
import vernon_dsl as vd

@vd.kernel(workgroup_size=(4, 1, 1))
def update(
    values: vd.TensorView[vd.f32, (4,), vd.read_write],
    x: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> vd.f32:
    values[gid[0]] = x
    return values[gid[0]]
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "update",
                    workgroup_size=(4, 1, 1),
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("values",)),
                )
            )
            assert result.program_graph is not None
            assert result.autodiff_profiles is not None
            modules = emit_native_autodiff_modules(
                result.program_graph,
                result.autodiff_profiles,
                target="metal",
            )
            self.assertIn('"vernon.store"', modules["forward_with_tape"])
            self.assertIn('"vernon.load"', modules["forward_with_tape"])
            self.assertNotIn('name = "construct"', modules["forward_with_tape"])
            self.assertNotIn("tensor<4xf32>", modules["backward"])
            self.assertNotIn("arith.cmpi eq", modules["backward"])
            self.assertNotIn("arith.select", modules["backward"])
            self.assert_gpu_autodiff_modules_compile(result.program_graph, result.autodiff_profiles)

    def test_dynamic_tensor_view_reports_source_compile_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dynamic_storage_vjp.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def first(values: vd.TensorView[vd.f32, (vd.dyn,), vd.read]) -> vd.f32:
    return values[0]
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                vd.CompileError,
                "positive static shape for TensorView 'values'",
            ):
                Compiler().compile_request(
                    FrontendCompileRequest(
                        source,
                        "first",
                        program_transform=vd.ad.ProgramTransformSpec("vjp", ("values",)),
                    )
                )

    def test_native_storage_control_flow_remains_rejected(self) -> None:
        programs = {
            "loop": """
import vernon_dsl as vd

@vd.kernel
def mutate(values: vd.TensorView[vd.f32, (2,), vd.read_write], x: vd.f32) -> vd.f32:
    for i in range(2):
        values[0] = values[0] + x
    return values[0]
""",
            "early_return": """
import vernon_dsl as vd

@vd.kernel
def mutate(values: vd.TensorView[vd.f32, (2,), vd.read_write], x: vd.f32) -> vd.f32:
    if x > 0.0:
        values[0] = x
        return x
    return values[0]
""",
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, program in programs.items():
                with self.subTest(name=name):
                    source = Path(directory) / f"{name}.py"
                    source.write_text(program, encoding="utf-8")
                    if name == "early_return":
                        with self.assertRaisesRegex(
                            vd.CompileError,
                            "branch-local returns with Storage effects are unavailable",
                        ):
                            Compiler().compile_request(
                                FrontendCompileRequest(
                                    source,
                                    "mutate",
                                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("x",)),
                                )
                            )
                        continue
                    result = Compiler().compile_request(
                        FrontendCompileRequest(
                            source,
                            "mutate",
                            program_transform=vd.ad.ProgramTransformSpec("vjp", ("x",)),
                        )
                    )
                    assert result.program_graph is not None
                    assert result.autodiff_profiles is not None
                    with self.assertRaisesRegex(
                        AutodiffNativeLoweringError,
                        "does not support storage control flow",
                    ):
                        emit_native_autodiff_modules(
                            result.program_graph,
                            result.autodiff_profiles,
                        )

    def test_tensor_view_alias_requires_alias_aware_lowering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "storage_alias.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def copy_first(
    left: vd.TensorView[vd.f32, (2,), vd.read_write],
    right: vd.TensorView[vd.f32, (2,), vd.read],
) -> vd.f32:
    left[0] = right[0]
    return left[0]
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "copy_first",
                    program_transform=vd.ad.ProgramTransformSpec(
                        "vjp",
                        ("left", "right"),
                    ),
                )
            )
            assert result.program_graph is not None
            assert result.autodiff_profiles is not None
            with self.assertRaisesRegex(
                AutodiffNativeLoweringError,
                "cannot prove non-aliasing",
            ):
                emit_native_autodiff_modules(
                    result.program_graph,
                    result.autodiff_profiles,
                )
            shared = np.array([2, 3], dtype=np.float32)
            with self.assertRaisesRegex(TypeError, "cannot prove a legal reverse scatter"):
                execute_program_graph(
                    result.program_graph,
                    {"left": shared, "right": shared},
                )

    def test_read_only_tensor_view_alias_has_independent_formal_gradients(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "readonly_alias.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def product(
    left: vd.TensorView[vd.f32, (2,), vd.read],
    right: vd.TensorView[vd.f32, (2,), vd.read],
) -> vd.f32:
    return left[0] * right[0]
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "product",
                    program_transform=vd.ad.ProgramTransformSpec(
                        "vjp",
                        ("left", "right"),
                    ),
                )
            )
            assert result.program_graph is not None
            shared = np.array([3, 4], dtype=np.float32)
            output, pullback = execute_program_graph(
                result.program_graph,
                {"left": shared, "right": shared},
            )
            self.assertAlmostEqual(output, 9)
            gradients = pullback()
            np.testing.assert_array_equal(
                gradients["left"],
                np.array([3, 0], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                gradients["right"],
                np.array([3, 0], dtype=np.float32),
            )

    def test_missing_derivative_rule_is_rejected_before_lowering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "floor.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def quantize(value: vd.f32) -> vd.f32:
    return vd.floor(value)
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(vd.CompileError, "derivative rule is unavailable for 'floor'"):
                Compiler().compile_request(
                    FrontendCompileRequest(
                        source,
                        "quantize",
                        program_transform=vd.ad.ProgramTransformSpec("vjp", ("value",)),
                    )
                )

    def test_branch_local_return_lowers_to_selected_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "branch.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def choose(value: vd.f32) -> vd.f32:
    if value > 0.0:
        return value
    return -value
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "choose",
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("value",)),
                )
            )
            assert result.program_graph is not None
            positive, positive_pullback = execute_program_graph(
                result.program_graph,
                {"value": np.float32(2)},
            )
            self.assertAlmostEqual(positive, 2)
            self.assertEqual(positive_pullback(), {"value": 1.0})
            negative, negative_pullback = execute_program_graph(
                result.program_graph,
                {"value": np.float32(-3)},
            )
            self.assertAlmostEqual(negative, 3)
            self.assertEqual(negative_pullback(), {"value": -1.0})
            assert result.autodiff_profiles is not None
            modules = emit_native_autodiff_modules(result.program_graph, result.autodiff_profiles)
            self.assertIn("scf.if", modules["forward_with_tape"])
            self.assertIn("scf.if", modules["backward"])
            self.assert_native_modules_compile(modules)
            self.assert_gpu_autodiff_modules_compile(result.program_graph, result.autodiff_profiles)

    def test_tensor_early_return_executes_only_selected_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tensor_return.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def safe(
    values: vd.Tensor[vd.f32, (2,)],
    denominator: vd.f32,
) -> vd.Tensor[vd.f32, (2,)]:
    if denominator == 0.0:
        return values * values
    return values / denominator
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "safe",
                    program_transform=vd.ad.ProgramTransformSpec(
                        "vjp",
                        ("denominator", "values"),
                    ),
                )
            )
            assert result.program_graph is not None
            with np.errstate(divide="raise", invalid="raise"):
                output, pullback = execute_program_graph(
                    result.program_graph,
                    {
                        "values": np.array([2.0, 3.0], dtype=np.float32),
                        "denominator": np.float32(0),
                    },
                )
            np.testing.assert_array_equal(output, np.array([4.0, 9.0], dtype=np.float32))
            gradients = pullback(np.array([1.0, 2.0], dtype=np.float32))
            np.testing.assert_array_equal(
                gradients["values"],
                np.array([4.0, 12.0], dtype=np.float32),
            )
            self.assertAlmostEqual(gradients["denominator"], 0)
            assert result.autodiff_profiles is not None
            modules = emit_native_autodiff_modules(result.program_graph, result.autodiff_profiles)
            self.assert_native_modules_compile(modules)
            self.assert_gpu_autodiff_modules_compile(result.program_graph, result.autodiff_profiles)

    def test_bounded_if_records_condition_and_reverses_selected_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bounded_if.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def objective(x: vd.f32, scale: vd.f32) -> vd.f32:
    result = x * x
    if x > 0.0:
        result = result * scale
    else:
        result = result / scale
    return result
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "objective",
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("scale", "x")),
                )
            )
            assert result.program_graph is not None
            graph = result.program_graph
            self.assertIn(OpCode.CONDITIONAL, [node.operation for node in graph.semantic.nodes])
            condition = next(node.id for node in graph.semantic.nodes if node.operation is OpCode.COMPARE)
            self.assertIn(condition, graph.reverse.saved_values)

            positive, positive_pullback = execute_program_graph(
                graph,
                {"x": np.float32(2), "scale": np.float32(3)},
            )
            self.assertAlmostEqual(positive, 12)
            self.assertEqual(positive_pullback(), {"scale": 4.0, "x": 12.0})

            negative, negative_pullback = execute_program_graph(
                graph,
                {"x": np.float32(-2), "scale": np.float32(4)},
            )
            self.assertAlmostEqual(negative, 1)
            gradients = negative_pullback()
            self.assertAlmostEqual(gradients["x"], -1)
            self.assertAlmostEqual(gradients["scale"], -0.25)

    def test_conditional_expression_executes_only_selected_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "conditional.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def safe(x: vd.f32, denominator: vd.f32) -> vd.f32:
    return x if x > 0.0 else 1.0 / denominator
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "safe",
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("denominator", "x")),
                )
            )
            assert result.program_graph is not None
            with np.errstate(divide="raise", invalid="raise"):
                output, pullback = execute_program_graph(
                    result.program_graph,
                    {"x": np.float32(2), "denominator": np.float32(0)},
                )
            self.assertAlmostEqual(output, 2)
            self.assertEqual(pullback(), {"denominator": 0.0, "x": 1.0})
            assert result.autodiff_profiles is not None
            modules = emit_native_autodiff_modules(result.program_graph, result.autodiff_profiles)
            self.assertIn("scf.if", modules["forward_with_tape"])
            self.assertIn("scf.if", modules["backward"])
            self.assert_native_modules_compile(modules)
            self.assert_gpu_autodiff_modules_compile(result.program_graph, result.autodiff_profiles)

    def test_literal_range_loop_has_statically_bounded_reverse_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "loop.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def fourth_power(x: vd.f32) -> vd.f32:
    result = x
    for i in range(3):
        result = result * x
    return result
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "fourth_power",
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("x",)),
                )
            )
            assert result.program_graph is not None
            graph = result.program_graph
            self.assertEqual(
                [node.operation for node in graph.semantic.nodes].count(OpCode.MUL),
                3,
            )
            output, pullback = execute_program_graph(graph, {"x": np.float32(2)})
            self.assertAlmostEqual(output, 16)
            self.assertAlmostEqual(pullback()["x"], 32)
            assert result.autodiff_profiles is not None
            self.assert_native_modules_compile(emit_native_autodiff_modules(graph, result.autodiff_profiles))

    def test_literal_bound_with_dynamic_break_has_bounded_reverse_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dynamic_loop.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def power(x: vd.f32, count: vd.i32) -> vd.f32:
    result = 1.0
    remaining = count
    for i in range(8):
        if remaining <= 0:
            break
        result = result * x
        remaining = remaining - 1
    return result
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "power",
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("x",)),
                )
            )
            assert result.program_graph is not None
            graph = result.program_graph
            self.assertEqual([node.operation for node in graph.semantic.nodes].count(OpCode.MUL), 8)
            output, pullback = execute_program_graph(
                graph,
                {"x": np.float32(2), "count": np.int32(3)},
            )
            self.assertAlmostEqual(output, 8)
            self.assertAlmostEqual(pullback()["x"], 12)
            empty_output, empty_pullback = execute_program_graph(
                graph,
                {"x": np.float32(2), "count": np.int32(0)},
            )
            self.assertAlmostEqual(empty_output, 1)
            self.assertAlmostEqual(empty_pullback()["x"], 0)
            assert result.autodiff_profiles is not None
            modules = emit_native_autodiff_modules(graph, result.autodiff_profiles)
            self.assertIn("scf.if", modules["forward_with_tape"])
            self.assertIn("scf.if", modules["backward"])
            self.assert_native_modules_compile(modules)
            self.assert_gpu_autodiff_modules_compile(graph, result.autodiff_profiles)

    def test_bounded_dynamic_loop_reverses_tensor_broadcasts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dynamic_tensor_loop.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def repeated_scale(
    values: vd.Tensor[vd.f32, (2,)],
    scale: vd.f32,
    count: vd.i32,
) -> vd.Tensor[vd.f32, (2,)]:
    result = values
    remaining = count
    for i in range(4):
        if remaining <= 0:
            break
        result = result * scale
        remaining = remaining - 1
    return result
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "repeated_scale",
                    program_transform=vd.ad.ProgramTransformSpec(
                        "vjp",
                        ("scale", "values"),
                    ),
                )
            )
            assert result.program_graph is not None
            output, pullback = execute_program_graph(
                result.program_graph,
                {
                    "values": np.array([1.0, 2.0], dtype=np.float32),
                    "scale": np.float32(2),
                    "count": np.int32(3),
                },
            )
            np.testing.assert_array_equal(output, np.array([8.0, 16.0], dtype=np.float32))
            gradients = pullback(np.array([1.0, 2.0], dtype=np.float32))
            np.testing.assert_array_equal(
                gradients["values"],
                np.array([8.0, 16.0], dtype=np.float32),
            )
            self.assertAlmostEqual(gradients["scale"], 60)
            assert result.autodiff_profiles is not None
            modules = emit_native_autodiff_modules(result.program_graph, result.autodiff_profiles)
            self.assert_native_modules_compile(modules)
            self.assert_gpu_autodiff_modules_compile(result.program_graph, result.autodiff_profiles)

    def test_literal_range_loop_enforces_static_iteration_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "large_loop.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def too_large(x: vd.f32) -> vd.f32:
    result = x
    for i in range(1025):
        result = result + x
    return result
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(vd.CompileError, "static iteration cap 1024"):
                Compiler().compile_request(
                    FrontendCompileRequest(
                        source,
                        "too_large",
                        program_transform=vd.ad.ProgramTransformSpec("vjp", ("x",)),
                    )
                )

    def test_monomorphized_helpers_inline_into_reverse_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "helper.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.func
def square(value):
    return value * value

@vd.func
def quartic(value: vd.f32) -> vd.f32:
    return square(square(value))

@vd.kernel
def objective(value: vd.f32) -> vd.f32:
    return quartic(value) + value
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "objective",
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("value",)),
                )
            )
            assert result.program_graph is not None
            operations = [node.operation for node in result.program_graph.semantic.nodes]
            self.assertEqual(operations.count(OpCode.MUL), 2)
            output, pullback = execute_program_graph(
                result.program_graph,
                {"value": np.float32(2)},
            )
            self.assertAlmostEqual(output, 18)
            self.assertAlmostEqual(pullback()["value"], 33)

    def test_native_scalar_profiles_compile_as_independent_artifacts(self) -> None:
        try:
            from vernon_dsl import _native as native
        except (ImportError, OSError):
            self.skipTest("native Vernon compiler is unavailable")
        self.assertTrue(hasattr(native.LoadedPipeline, "vjp"))
        self.assertTrue(hasattr(native, "Pullback"))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "native_ad.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def objective(x: vd.f32, y: vd.f32) -> vd.f32:
    return vd.sin(x * y) + x / y
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "objective",
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("x", "y")),
                )
            )
            assert result.program_graph is not None
            assert result.autodiff_profiles is not None
            modules = emit_native_autodiff_modules(
                result.program_graph,
                result.autodiff_profiles,
            )
            self.assertEqual(set(modules), {"forward_with_tape", "backward"})
            compiler = native.Compiler()
            for profile, mlir in modules.items():
                program = compiler.compile_program_result(mlir, native.Target.CPU)
                self.assertTrue(program.ok, f"{profile}: {program.diagnostics}")
                self.assertTrue(program.artifacts)

    def test_gpu_tensor_reduction_uses_shape_independent_loops(self) -> None:
        modules_by_extent: dict[int, dict[str, str]] = {}
        with tempfile.TemporaryDirectory() as directory:
            for extent in (2, 128):
                source = Path(directory) / f"tensor_atomic_{extent}.py"
                source.write_text(
                    f"""
import vernon_dsl as vd

@vd.kernel(workgroup_size=(2, 1, 1))
def square(values: vd.Tensor[vd.f32, ({extent},)]) -> vd.Tensor[vd.f32, ({extent},)]:
    return values * values
""",
                    encoding="utf-8",
                )
                result = Compiler().compile_request(
                    FrontendCompileRequest(
                        source,
                        "square",
                        workgroup_size=(2, 1, 1),
                        program_transform=vd.ad.ProgramTransformSpec("vjp", ("values",)),
                    )
                )
                assert result.program_graph is not None
                assert result.autodiff_profiles is not None
                self.assertEqual(
                    result.program_graph.launch.accumulation_plans[0].mode,
                    AccumulationMode.REDUCE_SUM,
                )
                modules_by_extent[extent] = emit_native_autodiff_modules(
                    result.program_graph,
                    result.autodiff_profiles,
                    target="cuda",
                )
                if extent == 2:
                    self.assert_gpu_autodiff_modules_compile(result.program_graph, result.autodiff_profiles)
        small = modules_by_extent[2]["backward"]
        large = modules_by_extent[128]["backward"]
        self.assertEqual(small.count("scf.for"), 1)
        self.assertEqual(small.count("scf.for"), large.count("scf.for"))
        self.assertEqual(small.count('"vernon.reduce_sum"'), 1)
        self.assertEqual(small.count('"vernon.reduce_sum"'), large.count('"vernon.reduce_sum"'))

    def test_native_tensor_profiles_compile_as_independent_artifacts(self) -> None:
        try:
            from vernon_dsl import _native as native
        except (ImportError, OSError):
            self.skipTest("native Vernon compiler is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "native_tensor_ad.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def square(values: vd.Tensor[vd.f32, (2,)]) -> vd.Tensor[vd.f32, (2,)]:
    return values * values
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "square",
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("values",)),
                )
            )
            assert result.program_graph is not None
            assert result.autodiff_profiles is not None
            modules = emit_native_autodiff_modules(
                result.program_graph,
                result.autodiff_profiles,
            )
            compiler = native.Compiler()
            for profile, mlir in modules.items():
                program = compiler.compile_program_result(mlir, native.Target.CPU)
                self.assertTrue(program.ok, f"{profile}: {program.diagnostics}")
                self.assertTrue(program.artifacts)

    def test_native_tensor_scalar_broadcast_emits_splat_and_gradient_reduction(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "native_tensor_broadcast_ad.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def scale(values: vd.Tensor[vd.f32, (2,)], factor: vd.f32) -> vd.Tensor[vd.f32, (2,)]:
    return values * factor
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "scale",
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("values", "factor")),
                )
            )
            assert result.program_graph is not None
            assert result.autodiff_profiles is not None
            self.assertIn(
                OpCode.SPLAT,
                [node.operation for node in result.program_graph.semantic.nodes],
            )
            modules = emit_native_autodiff_modules(result.program_graph, result.autodiff_profiles)
            self.assertIn("tensor.splat", modules["forward_with_tape"])
            self.assertIn('name = "reduce_sum_to_shape"', modules["backward"])
            self.assert_native_modules_compile(modules)
            self.assert_gpu_autodiff_modules_compile(result.program_graph, result.autodiff_profiles)

    def test_native_tensor_broadcast_reduces_each_gradient_to_its_source_shape(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "native_tensor_shape_broadcast_ad.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def add(
    rows: vd.Tensor[vd.f32, (2, 1)],
    columns: vd.Tensor[vd.f32, (1, 3)],
) -> vd.Tensor[vd.f32, (2, 3)]:
    return rows + columns
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "add",
                    program_transform=vd.ad.ProgramTransformSpec("vjp", ("rows", "columns")),
                )
            )
            assert result.program_graph is not None
            assert result.autodiff_profiles is not None
            graph = result.program_graph
            self.assertEqual(
                [node.operation for node in graph.semantic.nodes].count(OpCode.BROADCAST),
                2,
            )
            output, pullback = execute_program_graph(
                graph,
                {
                    "rows": np.array([[1.0], [2.0]], dtype=np.float32),
                    "columns": np.array([[10.0, 20.0, 30.0]], dtype=np.float32),
                },
            )
            np.testing.assert_array_equal(output, np.array([[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]))
            gradients = pullback(np.ones((2, 3), dtype=np.float32))
            np.testing.assert_array_equal(gradients["rows"], np.full((2, 1), 3.0, dtype=np.float32))
            np.testing.assert_array_equal(gradients["columns"], np.full((1, 3), 2.0, dtype=np.float32))
            self.assert_native_modules_compile(emit_native_autodiff_modules(graph, result.autodiff_profiles))

    def test_native_bounded_if_emits_forward_and_backward_regions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "native_branch_ad.py"
            source.write_text(
                """
import vernon_dsl as vd

@vd.kernel
def objective(x: vd.f32, scale: vd.f32) -> vd.f32:
    result = x * x
    if x > 0.0:
        result = result * scale
    else:
        result = result / scale
    return result
""",
                encoding="utf-8",
            )
            result = Compiler().compile_request(
                FrontendCompileRequest(
                    source,
                    "objective",
                    program_transform=vd.ad.ProgramTransformSpec(
                        "vjp",
                        ("x", "scale"),
                    ),
                )
            )
            assert result.program_graph is not None
            assert result.autodiff_profiles is not None
            condition = next(
                node.id for node in result.program_graph.semantic.nodes if node.operation is OpCode.COMPARE
            )
            self.assertIn(condition, result.program_graph.reverse.saved_values)
            modules = emit_native_autodiff_modules(
                result.program_graph,
                result.autodiff_profiles,
            )
            self.assertIn("scf.if", modules["forward_with_tape"])
            self.assertIn("scf.if", modules["backward"])
            self.assert_native_modules_compile(modules)


if __name__ == "__main__":
    unittest.main()
