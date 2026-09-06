from __future__ import annotations

# pyright: reportMissingImports=false
import base64
import ctypes
import hashlib
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).parents[3]
PYTHON_TEST_ROOT = PROJECT_ROOT / "python" / "tests"
sys.path.insert(0, str(PYTHON_TEST_ROOT))

from vernon_dsl._versions import COMPILER_CONTRACT_VERSION, PROGRAM_VERSION  # noqa: E402

native = importlib.import_module("vernon_dsl._native")
native_file = native.__file__
if not isinstance(native_file, str):
    raise RuntimeError("vernon_dsl._native has no filesystem location")
NATIVE_PATH = Path(native_file).resolve()
if len(sys.argv) >= 3:
    COMPILER_LIBRARY = Path(sys.argv.pop(1)).resolve()
    COMPILE_CLI = Path(sys.argv.pop(1)).resolve()
else:
    COMPILE_CLI = NATIVE_PATH.with_name("vernon-compile.exe" if sys.platform == "win32" else "vernon-compile")
    COMPILER_LIBRARY = NATIVE_PATH.with_name(
        "VernonDSLCompiler.dll"
        if sys.platform == "win32"
        else "libVernonDSLCompiler.dylib"
        if sys.platform == "darwin"
        else "libVernonDSLCompiler.so"
    )

import vernon_dsl as vd  # noqa: E402
import vernon_dsl._program_assets.compile_orchestration as compile_orchestration_module  # noqa: E402
import vernon_dsl._runtime.session as runtime_module  # noqa: E402
from program_asset_fixture import (  # noqa: E402
    OFFSET,
    scale,
    solid_fragment,
    triangle_vertex,
)
from vernon_dsl.bundle import OpenGLTargetOptions, VulkanTargetOptions, canonical_json  # noqa: E402
from vernon_dsl.compiler import compile_file  # noqa: E402
from vernon_dsl.program_asset_cli import main as program_asset_main  # noqa: E402
from vernon_dsl.program_assets import cook_program_asset  # noqa: E402

FIXTURE = PYTHON_TEST_ROOT / "program_asset_fixture.py"
GOLDEN = json.loads(
    (PROJECT_ROOT / "source" / "tests" / "fixtures" / "compile_parity_golden.json").read_text(encoding="utf-8")
)


class _StringView(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("size", ctypes.c_size_t)]


class _CpuOptions(ctypes.Structure):
    _fields_ = [
        ("triple", _StringView),
        ("processor", _StringView),
        ("features", _StringView),
    ]


class _OpenGLOptions(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32)]


class _MetalOptions(ctypes.Structure):
    _fields_ = [("platform", ctypes.c_uint32)]


class _DirectXOptions(ctypes.Structure):
    _fields_ = [("shader_model", ctypes.c_uint32)]


class _TargetOptions(ctypes.Union):
    _fields_ = [
        ("cpu", _CpuOptions),
        ("opengl", _OpenGLOptions),
        ("metal", _MetalOptions),
        ("directx", _DirectXOptions),
    ]


class _CompileOptions(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("target", ctypes.c_int), ("as_", _TargetOptions)]


def _view_bytes(view: _StringView) -> bytes:
    return ctypes.string_at(view.data, view.size) if view.data else b""


class _DirectCompiler:
    def __init__(self) -> None:
        self.library = ctypes.CDLL(str(COMPILER_LIBRARY))
        self.library.vernonCompilerCreate.restype = ctypes.c_void_p
        self.library.vernonCompilerDestroy.argtypes = [ctypes.c_void_p]
        self.library.vernonCompilerCompileMlirWithOptions.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(_CompileOptions),
        ]
        self.library.vernonCompilerCompileMlirWithOptions.restype = ctypes.c_void_p
        self.library.vernonCompileResultDestroy.argtypes = [ctypes.c_void_p]
        self.library.vernonCompileResultGetStatus.argtypes = [ctypes.c_void_p]
        self.library.vernonCompileResultGetStatus.restype = ctypes.c_int
        self.library.vernonCompileResultGetDiagnostics.argtypes = [ctypes.c_void_p]
        self.library.vernonCompileResultGetDiagnostics.restype = _StringView
        self.library.vernonCompileResultGetReflection.argtypes = [ctypes.c_void_p]
        self.library.vernonCompileResultGetReflection.restype = _StringView
        self.library.vernonCompileResultGetArtifactCount.argtypes = [ctypes.c_void_p]
        self.library.vernonCompileResultGetArtifactCount.restype = ctypes.c_size_t
        for name in ("vernonCompileResultGetArtifactName", "vernonCompileResultGetArtifactData"):
            function = getattr(self.library, name)
            function.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            function.restype = _StringView
        self.context = self.library.vernonCompilerCreate()
        if not self.context:
            raise RuntimeError("direct C API compiler creation failed")

    def close(self) -> None:
        if self.context:
            self.library.vernonCompilerDestroy(self.context)
            self.context = None

    def compile(
        self, mlir: str, target: int, glsl_version: int, hlsl_shader_model: int = 0
    ) -> tuple[dict[str, object], list[tuple[str, bytes]]]:
        source = mlir.encode("utf-8")
        source_buffer = ctypes.create_string_buffer(source)
        options = _CompileOptions()
        options.struct_size = ctypes.sizeof(options)
        options.target = target
        if target in {1, 2}:
            options.as_.opengl.version = glsl_version
        elif target == 5:
            options.as_.directx.shader_model = hlsl_shader_model
        result = self.library.vernonCompilerCompileMlirWithOptions(
            self.context,
            ctypes.cast(source_buffer, ctypes.c_void_p),
            len(source),
            ctypes.byref(options),
        )
        if not result:
            raise RuntimeError("direct C API returned no compile result")
        try:
            status = self.library.vernonCompileResultGetStatus(result)
            diagnostics = _view_bytes(self.library.vernonCompileResultGetDiagnostics(result)).decode()
            if status != 0:
                raise AssertionError(diagnostics)
            reflection = json.loads(_view_bytes(self.library.vernonCompileResultGetReflection(result)))
            artifacts = []
            count = self.library.vernonCompileResultGetArtifactCount(result)
            for index in range(count):
                name = _view_bytes(self.library.vernonCompileResultGetArtifactName(result, index)).decode()
                data = _view_bytes(self.library.vernonCompileResultGetArtifactData(result, index))
                artifacts.append((name, data))
            return reflection, artifacts
        finally:
            self.library.vernonCompileResultDestroy(result)


def _artifact_bytes(bundle: dict[str, object], stage_id: str, root: Path | None = None) -> bytes:
    stage = bundle["stage_artifacts"][stage_id]  # type: ignore[index]
    artifact = stage["artifact"]  # type: ignore[index]
    if artifact["storage"] == "external":
        assert root is not None
        return (root / artifact["path"]).read_bytes()
    if artifact["encoding"] == "base64":
        return base64.b64decode(artifact["data"], validate=True)
    return artifact["data"].encode("utf-8")


def _assert_content_hash(test: unittest.TestCase, bundle: dict[str, object]) -> None:
    unhashed = dict(bundle)
    digest = unhashed.pop("content_hash")
    test.assertEqual(
        digest,
        hashlib.sha256(canonical_json(unhashed).encode("utf-8")).hexdigest(),
    )


class CompileSurfaceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.direct = _DirectCompiler()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.direct.close()

    def test_direct_c_api_owning_program_and_cli_are_byte_identical(self) -> None:
        cases = (
            ("opengl", native.Target.OPENGL, 1, 330, 0),
            ("vulkan", native.Target.VULKAN, 3, 0, 0),
            ("metal", native.Target.METAL, 4, 0, 0),
            ("directx", native.Target.DIRECTX, 5, 0, 60),
        )
        for target_name, native_target, c_target, glsl_version, hlsl_shader_model in cases:
            if target_name == "directx" and not native.target_available(native_target):
                continue
            for entry in ("triangle_vertex", "solid_fragment"):
                with self.subTest(target=target_name, entry=entry):
                    mlir = compile_file(FIXTURE, features=("OFFSET",), entry=entry)
                    direct_reflection, direct_artifacts = self.direct.compile(
                        mlir, c_target, glsl_version, hlsl_shader_model
                    )
                    program = native.Compiler().compile_program_result(
                        mlir,
                        native_target,
                        (
                            {"version": glsl_version}
                            if glsl_version
                            else {"shader_model": hlsl_shader_model}
                            if hlsl_shader_model
                            else {}
                        ),
                    )
                    self.assertTrue(program.ok, program.diagnostics)
                    owning_reflection = json.loads(program.reflection)
                    owning_artifacts = [(str(name), bytes(data)) for name, data in program.artifacts]
                    self.assertEqual(direct_reflection, owning_reflection)
                    self.assertEqual(direct_artifacts, owning_artifacts)

                    with tempfile.TemporaryDirectory() as directory:
                        output = Path(directory)
                        reflection_path = output / "reflection.json"
                        command = [
                            str(COMPILE_CLI),
                            "--target",
                            target_name,
                            str(output / "fixture.mlir"),
                            "--output-dir",
                            str(output / "artifacts"),
                            "--reflection",
                            str(reflection_path),
                        ]
                        if glsl_version:
                            command.extend(["--opengl-version", str(glsl_version)])
                        if hlsl_shader_model:
                            command.extend(["--directx-shader-model", str(hlsl_shader_model)])
                        (output / "fixture.mlir").write_text(mlir, encoding="utf-8")
                        completed = subprocess.run(command, capture_output=True, text=True, check=False)
                        self.assertEqual(completed.returncode, 0, completed.stderr)
                        cli_reflection = json.loads(reflection_path.read_text(encoding="utf-8"))
                        cli_artifacts = [
                            (name, (output / "artifacts" / name).read_bytes()) for name, _ in owning_artifacts
                        ]
                    self.assertEqual(cli_reflection, direct_reflection)
                    self.assertEqual(cli_artifacts, direct_artifacts)
                    self.assertEqual(owning_reflection["compiler_contract_version"], COMPILER_CONTRACT_VERSION)
                    self.assertEqual(owning_reflection["program_version"], PROGRAM_VERSION)
                    self.assertEqual(owning_reflection["target"]["kind"], target_name)
                    if glsl_version:
                        self.assertEqual(owning_reflection["target"]["options"]["version"], glsl_version)
                    else:
                        self.assertNotIn("version", owning_reflection["target"]["options"])
                    if hlsl_shader_model:
                        self.assertEqual(
                            owning_reflection["target"]["options"]["shader_model"],
                            hlsl_shader_model,
                        )
                    self.assertTrue(owning_reflection["module_hash"])
                    self.assertEqual(
                        [(row["name"], row["stage"]) for row in owning_reflection["entries"]],
                        [(entry, "vertex" if entry == "triangle_vertex" else "fragment")],
                    )
                    self.assertEqual(
                        [(row["entry_point"], row["stage"]) for row in owning_reflection["artifacts"]],
                        [(entry, "vertex" if entry == "triangle_vertex" else "fragment")],
                    )
                    for (_, direct_data), (_, owning_data) in zip(direct_artifacts, owning_artifacts, strict=True):
                        self.assertEqual(hashlib.sha256(direct_data).digest(), hashlib.sha256(owning_data).digest())

    def test_interactive_pipeline_and_both_cooker_surfaces_share_plan(self) -> None:
        cases = (
            ("opengl", runtime_module.opengl, (3, 3)),
            ("vulkan", runtime_module.vulkan, None),
        )
        for target_name, architecture, api_version in cases:
            with self.subTest(target=target_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)

                class PipelineCapture:
                    def __init__(self) -> None:
                        self.loads: list[bytes] = []

                    def load_canonical_program(
                        self,
                        manifest: bytes,
                        directory: str,
                        compiled_stages: object,
                    ) -> object:
                        del directory, compiled_stages
                        self.loads.append(bytes(manifest))
                        return object()

                capture = PipelineCapture()
                pipeline = vd.pipeline(triangle_vertex, solid_fragment, features={OFFSET.name})
                position = vd.storage.from_numpy(np.zeros((3, 2), dtype=np.float32))
                target = vd.RenderTarget.from_attachments(colors={0: vd.Texture.zeros(shape=(16, 16))})
                render_pass = vd.render_pass(target)
                with mock.patch.multiple(
                    runtime_module,
                    _architecture=architecture,
                    _native_runtime=capture,
                    _api_version=api_version,
                    _runtime_generation=101,
                ):
                    compiled = pipeline._compile({"position": position}, render_pass, None, None)
                    repeated = pipeline._compile({"position": position}, render_pass, None, None)
                self.assertIs(compiled, repeated)
                self.assertEqual(pipeline.compile_count, 1)
                deployment = compiled.specialization.deployment
                self.assertTrue(deployment.manifest)
                self.assertEqual(capture.loads, [deployment.manifest])

                compile_calls = 0

                class TrackingCompiler:
                    def __init__(self) -> None:
                        self.compiler = native.Compiler()

                    def compile_program_result(self, *args: object, **kwargs: object) -> object:
                        nonlocal compile_calls
                        compile_calls += 1
                        return self.compiler.compile_program_result(*args, **kwargs)

                    def plan_program_result(self, *args: object, **kwargs: object) -> object:
                        return self.compiler.plan_program_result(*args, **kwargs)

                    def finalize_program_result(self, *args: object, **kwargs: object) -> object:
                        return self.compiler.finalize_program_result(*args, **kwargs)

                proxy = SimpleNamespace(
                    Target=native.Target,
                    Compiler=TrackingCompiler,
                    Runtime=mock.Mock(side_effect=AssertionError("offline cooking created a runtime context")),
                )
                cooked_dir = root / "cooked"
                cli_dir = root / "cli"
                target_options = {"version": 330} if target_name == "opengl" else {}
                cli_arguments = [
                    f"{FIXTURE}:triangle_asset",
                    "--target",
                    target_name,
                    "--output",
                    str(cli_dir),
                ]
                if target_name == "opengl":
                    cli_arguments.extend(["--opengl-version", "330"])
                with (
                    mock.patch.object(compile_orchestration_module, "_native_module", return_value=proxy),
                    mock.patch("subprocess.run", side_effect=AssertionError("offline cooking spawned a subprocess")),
                ):
                    manifest_path = cook_program_asset(
                        program_asset=f"{FIXTURE}:triangle_asset",
                        output=cooked_dir,
                        target=(OpenGLTargetOptions(version=330) if target_name == "opengl" else VulkanTargetOptions()),
                    )
                    cli_status = program_asset_main(cli_arguments)
                self.assertEqual(cli_status, 0)
                # Two stage/feature specializations per cook. The unchanged
                # fragment artifact is deduplicated after compilation.
                self.assertEqual(compile_calls, 8)
                cooked = json.loads(manifest_path.read_text(encoding="utf-8"))
                cli_cooked = json.loads((cli_dir / "cli.program.json").read_text(encoding="utf-8"))
                self.assertEqual(cooked, cli_cooked)
                self.assertEqual(cooked["target"], {"kind": target_name, "options": target_options})
                self.assertNotIn("targets", cooked)
                self.assertEqual([variant["key"] for variant in cooked["variants"]], GOLDEN["graphics"]["variants"])
                self.assertNotIn("features", cooked)
                selected = next(variant for variant in cooked["variants"] if variant["key"] == ["OFFSET"])
                self.assertNotIn("parameters", selected)
                self.assertNotIn("outputs", selected)
                self.assertNotIn("stage_artifacts", selected)
                program = selected["program"]
                self.assertIn("abi", program)
                self.assertIn("graphs", program)
                self.assertIn("stages", program)
                self.assertIn("artifacts", selected["artifact_system"])
                _assert_content_hash(self, cooked)

    def test_cpu_owning_program_and_kernel_execution_match_and_cache(self) -> None:
        source = np.array((1.0, 2.0, 3.0, 4.0), dtype=np.float32)
        frontend = scale._lower().frontend
        program = native.Compiler().compile_program_result(frontend.mlir, native.Target.CPU)
        self.assertTrue(program.ok, program.diagnostics)
        reflection = json.loads(program.reflection)
        self.assertEqual(reflection["target"]["kind"], "cpu")
        self.assertTrue(reflection["target"]["options"]["triple"])
        self.assertTrue(program.has_cpu_entry("scale"))

        vd.init(arch=vd.cpu)
        direct_runtime = native.Runtime(native.RuntimeBackend.CPU)
        direct_kernel = direct_runtime.load_cpu_entry(program, "scale")
        direct_values = vd.storage.from_numpy(source)
        direct_builder = direct_kernel.invocation_builder()
        direct_builder.host_tensor(direct_kernel.parameters[0].name, direct_values._native_host_array())
        direct_builder.host_tensor(direct_kernel.parameters[1].name, np.asarray(np.float32(2.5)))
        direct_builder.grid(4, 1, 1).submit().wait()
        direct_result = direct_values.to_numpy()

        runtime_module.Kernel.clear_cache()
        scale.compile_count = 0
        first = vd.storage.from_numpy(source)
        second = vd.storage.from_numpy(source)
        with mock.patch(
            "subprocess.run", side_effect=AssertionError("interactive kernel spawned a compiler subprocess")
        ):
            scale(first, 2.5, grid=(4, 1, 1))
            scale(second, 2.5, grid=(4, 1, 1))
        np.testing.assert_array_equal(first.to_numpy(), direct_result)
        np.testing.assert_array_equal(second.to_numpy(), direct_result)
        np.testing.assert_array_equal(direct_result, source * np.float32(2.5))
        self.assertEqual(scale.compile_count, 1)


if __name__ == "__main__":
    unittest.main()
