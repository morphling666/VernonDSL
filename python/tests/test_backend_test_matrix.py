from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import backend_test_matrix
from backend_test_matrix import (
    BACKEND_TEST_MATRIX,
    BackendRequirements,
    ProbeKind,
    backend_matrix_test,
    expand_backend_matrix_tests,
    probe_backend,
    probe_compiler,
    probe_runtime,
)
from vernon_dsl._runtime.session import RuntimeUnavailableError


class BackendTestMatrixTests(unittest.TestCase):
    def test_matrix_contains_each_canonical_backend_once(self) -> None:
        self.assertEqual(
            [row.name for row in BACKEND_TEST_MATRIX],
            ["CPU", "CUDA", "Vulkan", "DirectX12", "Metal", "OpenGL", "OpenGLES"],
        )
        self.assertEqual(len({row.architecture for row in BACKEND_TEST_MATRIX}), len(BACKEND_TEST_MATRIX))

    def test_compiler_probe_distinguishes_platform_and_capability(self) -> None:
        capabilities = {
            "available": False,
            "compute": False,
            "graphics": False,
            "device_storage_atomics": False,
            "f32_device_atomic_add": False,
        }
        fake = SimpleNamespace(
            Target=SimpleNamespace(CPU=object()),
            target_capabilities=lambda target: capabilities,
        )
        with mock.patch("backend_test_matrix._native", fake):
            result = probe_compiler(BACKEND_TEST_MATRIX[0], BackendRequirements(compute=True))
        self.assertIs(result.kind, ProbeKind.PLATFORM_NOT_BUILT)

        capabilities["available"] = True
        with mock.patch("backend_test_matrix._native", fake):
            result = probe_compiler(BACKEND_TEST_MATRIX[0], BackendRequirements(compute=True))
        self.assertIs(result.kind, ProbeKind.CAPABILITY_UNSUPPORTED)
        self.assertIn("compute", result.reason)

    def test_dynamic_range_step_capability_is_explicit(self) -> None:
        capabilities = {
            "available": True,
            "compute": True,
            "graphics": True,
            "device_storage_atomics": True,
            "f32_device_atomic_add": True,
            "dynamic_range_step": True,
        }
        fake = SimpleNamespace(
            Target=SimpleNamespace(CPU=object(), VULKAN=object()),
            target_capabilities=lambda target: capabilities,
        )
        with mock.patch("backend_test_matrix._native", fake):
            cpu = probe_compiler(BACKEND_TEST_MATRIX[0], BackendRequirements(dynamic_range_step=True))
            capabilities["dynamic_range_step"] = False
            vulkan = probe_compiler(BACKEND_TEST_MATRIX[2], BackendRequirements(dynamic_range_step=True))
        self.assertTrue(cpu.available)
        self.assertIs(vulkan.kind, ProbeKind.CAPABILITY_UNSUPPORTED)
        self.assertIn("dynamic_range_step", vulkan.reason)

    def test_class_decorator_expands_each_matrix_test_per_backend(self) -> None:
        @expand_backend_matrix_tests
        class Example(unittest.TestCase):
            @backend_matrix_test(BackendRequirements(compute=True))
            def test_feature(self, backend) -> None:
                pass

        self.assertFalse(hasattr(Example, "test_feature"))
        self.assertEqual(
            {name for name in vars(Example) if name.startswith("test_feature_")},
            {f"test_feature_{row.runtime_backend.lower()}" for row in BACKEND_TEST_MATRIX},
        )

    def test_matrix_test_restores_cpu_after_capability_skip(self) -> None:
        @expand_backend_matrix_tests
        class Example(unittest.TestCase):
            @backend_matrix_test(BackendRequirements(compute=True))
            def test_feature(self, backend) -> None:
                self.fail("a skipped test must not execute")

        unsupported = backend_test_matrix.ProbeResult(ProbeKind.CAPABILITY_UNSUPPORTED, "unsupported")
        with (
            mock.patch("backend_test_matrix.probe_backend", return_value=unsupported),
            mock.patch("backend_test_matrix.vd.init") as initialize,
            self.assertRaises(unittest.SkipTest),
        ):
            Example("test_feature_cuda").test_feature_cuda()
        initialize.assert_called_once_with(arch=backend_test_matrix.vd.cpu)

    def test_runtime_probe_skips_only_declared_unavailability(self) -> None:
        fake = SimpleNamespace()
        with (
            mock.patch("backend_test_matrix._native", fake),
            mock.patch(
                "backend_test_matrix.vd.init",
                side_effect=[RuntimeUnavailableError("no device"), None],
            ),
        ):
            result = probe_runtime(BACKEND_TEST_MATRIX[1], BackendRequirements(compute=True))
        self.assertIs(result.kind, ProbeKind.DEVICE_OR_CONTEXT_UNAVAILABLE)
        self.assertEqual(result.reason, "no device")

    def test_runtime_probe_does_not_hide_regressions(self) -> None:
        fake = SimpleNamespace()
        with (
            mock.patch("backend_test_matrix._native", fake),
            mock.patch("backend_test_matrix.vd.init", side_effect=RuntimeError("driver regression")),
            self.assertRaisesRegex(RuntimeError, "driver regression"),
        ):
            probe_runtime(BACKEND_TEST_MATRIX[1], BackendRequirements(compute=True))

    def test_runtime_probe_checks_effective_compute_api_version(self) -> None:
        runtime = SimpleNamespace(
            capabilities={
                "available": True,
                "compute": True,
                "graphics": True,
                "storage_buffers": True,
                "api_version": (4, 2),
            }
        )
        with (
            mock.patch("backend_test_matrix._native", SimpleNamespace()),
            mock.patch("backend_test_matrix.vd.init"),
            mock.patch(
                "backend_test_matrix.runtime_session.current_session",
                return_value=SimpleNamespace(native_runtime=runtime),
            ),
        ):
            result = probe_runtime(BACKEND_TEST_MATRIX[5], BackendRequirements(compute=True))
        self.assertIs(result.kind, ProbeKind.CAPABILITY_UNSUPPORTED)
        self.assertIn("api_version>=(4, 3)", result.reason)

    def test_cpu_compute_and_storage_probe_end_to_end(self) -> None:
        if backend_test_matrix._native is None:
            self.skipTest("native Vernon extension is not built")
        result = probe_backend(
            BACKEND_TEST_MATRIX[0],
            BackendRequirements(compute=True, storage_buffers=True),
        )
        self.assertTrue(result.available, result.reason)


if __name__ == "__main__":
    unittest.main()
