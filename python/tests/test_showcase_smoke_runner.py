from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

if importlib.util.find_spec("cv2") is None:
    raise unittest.SkipTest("showcase smoke validation requires the optional examples dependency")

import cv2  # type: ignore[import-not-found]  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("run_showcase_smoke", ROOT / "scripts" / "run_showcase_smoke.py")
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
_validate_result = RUNNER._validate_result


class ShowcaseSmokeRunnerTests(unittest.TestCase):
    def test_result_and_png_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            png = root / "terrain.png"
            result_path = root / "terrain.json"
            image = np.zeros((8, 8, 4), dtype=np.uint8)
            image[..., 3] = 255
            image[:4, :, :3] = 180
            self.assertTrue(cv2.imwrite(str(png), image))
            result = {
                "showcase": "terrain",
                "backend": "vulkan",
                "preset": "smoke",
                "frames": 1,
                "size": 8,
                "fps": 30,
                "elapsed_seconds": 0.1,
                "passes": 1,
                "barriers": 0,
                "image": {
                    "shape": [8, 8, 4],
                    "mean": 90.0,
                    "stddev": 90.0,
                    "minimum": 0,
                    "maximum": 180,
                    "nonzero_alpha": 64,
                },
            }
            result_path.write_text(json.dumps(result), encoding="utf-8")

            self.assertEqual(
                _validate_result("terrain", "vulkan", 8, png, result_path),
                result,
            )

    def test_result_schema_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.json"
            result_path.write_text(json.dumps({"showcase": "terrain"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "result fields differ"):
                _validate_result("terrain", "vulkan", 8, root / "missing.png", result_path)


if __name__ == "__main__":
    unittest.main()
