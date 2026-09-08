from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import backend_test_matrix
from backend_test_matrix import (
    BACKEND_TEST_MATRIX,
    BackendRequirements,
    ProbeKind,
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
