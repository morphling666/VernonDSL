from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

if importlib.util.find_spec("cv2") is None:
    raise unittest.SkipTest("showcase helpers require the optional examples dependency")

EXAMPLES = Path(__file__).parents[2] / "examples"
sys.path.insert(0, str(EXAMPLES))

from showcase_common import (  # noqa: E402
    ShowcasePreset,
    image_statistics,
    resolve_showcase_options,
)


class ShowcaseCommonTests(unittest.TestCase):
    def test_headless_options_resolve_from_preset_and_overrides(self) -> None:
        presets = {
            "smoke": ShowcasePreset(size=64, frames=2, fps=30),
            "showoff": ShowcasePreset(size=512, frames=120, fps=60),
        }
        arguments = argparse.Namespace(
            arch="vulkan",
            preset="smoke",
            size=96,
            frames=None,
            fps=None,
            headless=True,
            output=None,
            result_json=None,
        )
        options = resolve_showcase_options(arguments, presets)
        self.assertEqual((options.size, options.frames, options.fps), (96, 2, 30))

    def test_image_statistics_reject_empty_and_uniform_images(self) -> None:
        with self.assertRaisesRegex(ValueError, "four-channel"):
            image_statistics(np.zeros((4, 4, 3), dtype=np.uint8))
        with self.assertRaisesRegex(RuntimeError, "empty or effectively uniform"):
            image_statistics(np.full((4, 4, 4), 255, dtype=np.uint8))

        image = np.zeros((4, 4, 4), dtype=np.uint8)
        image[..., 3] = 255
        image[1:3, 1:3, :3] = 200
        statistics = image_statistics(image)
        self.assertEqual(statistics["shape"], [4, 4, 4])
        self.assertEqual(statistics["nonzero_alpha"], 16)


if __name__ == "__main__":
    unittest.main()
