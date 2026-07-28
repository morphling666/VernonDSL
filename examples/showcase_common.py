from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np
import vernon_dsl as vd


def normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-8:
        raise ValueError("cannot normalize a zero-length vector")
    return vector / length


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = normalize(target - eye)
    right = normalize(np.cross(forward, np.array((0.0, 1.0, 0.0), dtype=np.float32)))
    up = np.cross(right, forward)
    result = np.eye(4, dtype=np.float32)
    result[0, :3] = right
    result[1, :3] = up
    result[2, :3] = -forward
    result[0, 3] = -np.dot(right, eye)
    result[1, 3] = -np.dot(up, eye)
    result[2, 3] = np.dot(forward, eye)
    return result


def perspective(
    vertical_fov: float,
    aspect: float,
    near: float,
    far: float,
    *,
    zero_to_one: bool,
) -> np.ndarray:
    focal_length = 1.0 / math.tan(vertical_fov * 0.5)
    result = np.zeros((4, 4), dtype=np.float32)
    result[0, 0] = focal_length / aspect
    result[1, 1] = focal_length
    if zero_to_one:
        result[2, 2] = far / (near - far)
        result[2, 3] = far * near / (near - far)
    else:
        result[2, 2] = (far + near) / (near - far)
        result[2, 3] = 2.0 * far * near / (near - far)
    result[3, 2] = -1.0
    return result


def create_night_environment(size: int = 64) -> vd.Texture:
    if size <= 0:
        raise ValueError("cubemap size must be positive")
    faces = np.empty((6, size, size, 4), dtype=np.uint8)
    face_colors = np.array(
        (
            (3, 8, 25, 255),
            (3, 8, 25, 255),
            (7, 16, 42, 255),
            (1, 2, 7, 255),
            (3, 8, 25, 255),
            (3, 8, 25, 255),
        ),
        dtype=np.uint8,
    )
    vertical = np.linspace(1.0, 0.42, size, dtype=np.float32)[:, None, None]
    faces[..., :3] = np.clip(
        face_colors[:, None, None, :3].astype(np.float32) * vertical[None, ...],
        0.0,
        255.0,
    ).astype(np.uint8)
    faces[..., 3] = 255
    coordinates = np.arange(size, dtype=np.uint32)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    cloud_x = xx.astype(np.float32) / np.float32(size)
    cloud_y = yy.astype(np.float32) / np.float32(size)
    for face in range(6):
        cloud = np.clip(
            np.sin(cloud_x * np.float32(13.0) + np.float32(face) * 0.9)
            + np.sin(cloud_y * np.float32(9.0) - np.float32(face) * 0.6)
            - np.float32(1.05),
            0.0,
            1.0,
        )
        cloud_color = np.stack((cloud * 5.0, cloud * 9.0, cloud * 16.0), axis=-1)
        faces[face, ..., :3] = np.clip(
            faces[face, ..., :3].astype(np.float32) + cloud_color,
            0.0,
            255.0,
        ).astype(np.uint8)
    return vd.Texture.cube(faces)


def load_equirectangular_environment(
    path: Path,
    *,
    face_size: int = 256,
    exposure: float = 1.35,
    rotation: float = 0.0,
) -> vd.Texture:
    if face_size <= 0 or exposure <= 0.0:
        raise ValueError("cubemap face size and exposure must be positive")
    panorama_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if panorama_bgr is None:
        raise RuntimeError(f"cannot load environment image: {path}")
    if panorama_bgr.ndim != 3 or panorama_bgr.shape[2] < 3:
        raise ValueError("environment image must contain at least three channels")
    panorama = np.ascontiguousarray(panorama_bgr[..., :3][..., ::-1], dtype=np.float32)
    coordinate = (np.arange(face_size, dtype=np.float32) + 0.5) * (2.0 / face_size) - 1.0
    uu, vv = np.meshgrid(coordinate, coordinate, indexing="xy")
    directions = (
        (np.ones_like(uu), -vv, -uu),
        (-np.ones_like(uu), -vv, uu),
        (uu, np.ones_like(uu), vv),
        (uu, -np.ones_like(uu), -vv),
        (uu, -vv, np.ones_like(uu)),
        (-uu, -vv, -np.ones_like(uu)),
    )
    faces = np.empty((6, face_size, face_size, 4), dtype=np.uint8)
    height, width = panorama.shape[:2]
    for face_index, components in enumerate(directions):
        direction = np.stack(components, axis=-1)
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        longitude = np.arctan2(direction[..., 2], direction[..., 0]) + np.float32(rotation)
        latitude = np.arcsin(np.clip(direction[..., 1], -1.0, 1.0))
        map_x = np.mod(longitude / (2.0 * np.pi) + 0.5, 1.0).astype(np.float32) * width
        map_y = (0.5 - latitude / np.pi).astype(np.float32) * (height - 1)
        sampled = cv2.remap(
            panorama,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        linear = sampled * np.float32(exposure)
        tone_mapped = linear / (1.0 + linear)
        faces[face_index, ..., :3] = np.clip(tone_mapped * 255.0, 0.0, 255.0).astype(np.uint8)
        faces[face_index, ..., 3] = 255
    return vd.Texture.cube(faces)


def load_rgba_texture(path: Path) -> vd.Texture:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot load texture image: {path}")
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    return vd.Texture.from_numpy(np.ascontiguousarray(rgba, dtype=np.uint8))


def create_sky_cube() -> np.ndarray:
    corners = np.array(
        (
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ),
        dtype=np.float32,
    )
    quads = np.array(
        (
            (0, 1, 2, 3),
            (5, 4, 7, 6),
            (4, 0, 3, 7),
            (1, 5, 6, 2),
            (3, 2, 6, 7),
            (4, 5, 1, 0),
        ),
        dtype=np.uint32,
    )
    triangles = np.column_stack(
        (
            quads[:, 0],
            quads[:, 1],
            quads[:, 2],
            quads[:, 0],
            quads[:, 2],
            quads[:, 3],
        )
    ).reshape(-1, 3)
    two_sided = np.concatenate((triangles, triangles[:, (0, 2, 1)]), axis=0)
    return np.ascontiguousarray(corners[two_sided.reshape(-1)])


@dataclass
class InvocationSlot:
    value: vd.PipelineInvocation | None = None

    def require(self) -> vd.PipelineInvocation:
        if self.value is None:
            raise RuntimeError("execution graph invocation was not prepared")
        return self.value


class ComputeInvocationPass(vd.ComputePass):
    def __init__(self, name: str, slot: InvocationSlot):
        super().__init__(name)
        self.slot = slot

    def declare(self) -> None:
        self.slot.require().declare(self)

    def execute(self, encoder: vd.ComputeEncoder, resources: vd.ExecutionResources) -> None:
        self.slot.require().encode(encoder, resources)


@dataclass
class InvocationBatch:
    values: list[vd.PipelineInvocation] = field(default_factory=list)

    def require(self) -> list[vd.PipelineInvocation]:
        if not self.values:
            raise RuntimeError("render invocation batch was not prepared")
        return self.values


class BatchRenderPass(vd.RenderPass):
    def __init__(
        self,
        name: str,
        target: vd.RenderTarget,
        batch: InvocationBatch,
        *,
        clear_color: tuple[float, float, float, float],
    ):
        super().__init__(name)
        self.target = target
        self.batch = batch
        self.clear_color = clear_color

    def declare(self) -> None:
        for invocation in self.batch.require():
            invocation.declare(self)
        self.attachments(self.target, clear_color=self.clear_color)

    def execute(self, encoder: vd.GraphicsEncoder, resources: vd.ExecutionResources) -> None:
        for invocation in self.batch.require():
            invocation.encode(encoder, resources)


class FramePresenter:
    def __init__(
        self,
        texture: vd.Texture,
        *,
        architecture: str,
        title: str,
        headless: bool,
        fps: int,
    ):
        self.texture = texture
        self.architecture = architecture
        self.title = title
        self.headless = headless
        self.delay_ms = max(1, round(1000 / fps))
        self.image: np.ndarray | None = None

    def present(self) -> bool:
        rgba = self.texture.to_numpy()
        if self.architecture == "opengl":
            rgba = np.flipud(rgba)
        self.image = cv2.cvtColor(np.ascontiguousarray(rgba), cv2.COLOR_RGBA2BGRA)
        if self.headless:
            return True
        cv2.imshow(self.title, self.image)
        return cv2.waitKey(self.delay_ms) & 0xFF not in (27, ord("q"))

    def close(self) -> None:
        if not self.headless:
            cv2.destroyWindow(self.title)

    def write(self, output: Path | None) -> None:
        if output is None or self.image is None:
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), self.image):
            raise RuntimeError(f"cannot write screenshot to {output}")


def architecture_from_name(name: str) -> object:
    architectures = {
        "vulkan": vd.vulkan,
        "directx": vd.directx,
        "opengl": vd.opengl,
    }
    try:
        return architectures[name]
    except KeyError as error:
        raise ValueError(f"unsupported graphics architecture: {name}") from error
