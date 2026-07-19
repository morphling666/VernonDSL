from __future__ import annotations

import unittest

import numpy as np


def normalize(value: np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value)


def blinn_phong(
    normal: np.ndarray,
    world_position: np.ndarray,
    albedo: np.ndarray,
    specular_color: np.ndarray,
    ambient_color: np.ndarray,
    light_position: np.ndarray,
    camera_position: np.ndarray,
    shininess: float,
) -> np.ndarray:
    unit_normal = normalize(normal)
    light_direction = normalize(light_position - world_position)
    view_direction = normalize(camera_position - world_position)
    half_direction = normalize(light_direction + view_direction)
    diffuse = max(float(np.dot(unit_normal, light_direction)), 0.0)
    specular = max(float(np.dot(unit_normal, half_direction)), 0.0)**shininess
    return ambient_color + albedo * diffuse + specular_color * specular


class BlinnPhongNumericTests(unittest.TestCase):

    def test_frontal_light_matches_closed_form(self) -> None:
        color = blinn_phong(
            normal=np.array([0.0, 0.0, 1.0]),
            world_position=np.zeros(3),
            albedo=np.array([0.5, 0.25, 0.125]),
            specular_color=np.array([0.2, 0.2, 0.2]),
            ambient_color=np.array([0.1, 0.1, 0.1]),
            light_position=np.array([0.0, 0.0, 2.0]),
            camera_position=np.array([0.0, 0.0, 3.0]),
            shininess=32.0,
        )
        np.testing.assert_allclose(color, [0.8, 0.55, 0.425], rtol=1e-6)

    def test_back_facing_light_has_no_diffuse_or_specular(self) -> None:
        color = blinn_phong(
            normal=np.array([0.0, 0.0, 1.0]),
            world_position=np.zeros(3),
            albedo=np.ones(3),
            specular_color=np.ones(3),
            ambient_color=np.array([0.05, 0.1, 0.15]),
            light_position=np.array([0.0, 0.0, -2.0]),
            camera_position=np.array([1.0, 0.0, 1.0]),
            shininess=16.0,
        )
        np.testing.assert_allclose(color, [0.05, 0.1, 0.15], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
