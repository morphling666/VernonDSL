from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import vernon_dsl as vd

if __package__:
    from .autodiff_smoke_mpc_model import SmokeControlHorizon, smoke_rollout
else:
    from autodiff_smoke_mpc_model import SmokeControlHorizon, smoke_rollout


GRID = 8
CELLS = GRID * GRID


def smoke_transition() -> tuple[np.ndarray, np.ndarray]:
    vertical = np.zeros((GRID, GRID), dtype=np.float32)
    for y in range(GRID):
        vertical[y, y] = np.float32(0.68)
        if y + 1 < GRID:
            vertical[y, y + 1] = np.float32(0.25)
        if y > 0:
            vertical[y, y - 1] = np.float32(0.04)

    horizontal = np.zeros((GRID, GRID), dtype=np.float32)
    for x in range(GRID):
        horizontal[x, x] = np.float32(0.82)
        if x > 0:
            horizontal[x, x - 1] = np.float32(0.075)
        if x + 1 < GRID:
            horizontal[x, x + 1] = np.float32(0.075)
    return vertical, horizontal


def v_target() -> np.ndarray:
    target = np.zeros((GRID, GRID), dtype=np.float32)
    for y in range(GRID):
        left = min(y // 2, GRID // 2 - 1)
        right = GRID - 1 - left
        target[y, left] = 1.0
        target[y, right] = 1.0
    return target


def rollout(
    expression,
    density: np.ndarray,
    vertical_transport: np.ndarray,
    horizontal_diffusion: np.ndarray,
    target: np.ndarray,
    controls: list[np.ndarray],
):
    horizon = SmokeControlHorizon(*controls)
    return expression(
        density,
        vertical_transport,
        horizontal_diffusion,
        target,
        horizon,
        grid=(1, 1, 1),
    )


def optimize(
    *,
    steps: int,
    iterations: int,
    learning_rate: float,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    vd.init(arch=vd.cpu)
    expression = vd.ad.vjp(
        smoke_rollout,
        wrt=("controls.first", "controls.second", "controls.third"),
    )
    vertical_transport, horizontal_diffusion = smoke_transition()
    target = v_target()
    density = np.zeros((GRID, GRID), dtype=np.float32)
    controls = [np.zeros((GRID, GRID), dtype=np.float32) for _ in range(3)]
    applied: list[np.ndarray] = []
    losses: list[float] = []
    zero_seed = np.zeros((GRID, GRID), dtype=np.float32)
    cotangents = {
        "output.loss": np.float32(1.0),
        "output.next_density": zero_seed,
        "output.final_density": zero_seed,
    }

    for step in range(steps):
        for _ in range(iterations):
            output, pullback = rollout(
                expression,
                density,
                vertical_transport,
                horizontal_diffusion,
                target,
                controls,
            )
            gradients = pullback(cotangents)
            for index, name in enumerate(("first", "second", "third")):
                controls[index] = np.clip(
                    controls[index] - np.float32(learning_rate) * gradients[f"controls.{name}"],
                    0.0,
                    1.0,
                ).astype(np.float32)

        output, _ = rollout(
            expression,
            density,
            vertical_transport,
            horizontal_diffusion,
            target,
            controls,
        )
        density = np.asarray(output["output.next_density"], dtype=np.float32).copy()
        loss = float(output["output.loss"])
        error = float(np.linalg.norm(density - target))
        control_norm = float(np.linalg.norm(controls[0]))
        losses.append(loss)
        applied.append(controls[0].copy())
        if verbose:
            print(f"step={step:02d} loss={loss:.6f} density_error={error:.6f} control_norm={control_norm:.6f}")
        controls = [controls[1].copy(), controls[2].copy(), np.zeros((GRID, GRID), dtype=np.float32)]

    return density, np.stack(applied), losses


def save_results(output: Path, density: np.ndarray, controls: np.ndarray, target: np.ndarray) -> None:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("PNG output requires the project examples dependency: pip install '.[examples]'") from error

    output.parent.mkdir(parents=True, exist_ok=True)
    image = np.concatenate((target, density), axis=1)
    image = np.clip(image * np.float32(255.0), 0.0, 255.0).astype(np.uint8)
    image = cv2.resize(image, (image.shape[1] * 48, image.shape[0] * 48), interpolation=cv2.INTER_NEAREST)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"failed to write {output}")
    np.save(output.with_name(output.stem + "_density.npy"), density)
    np.save(output.with_name(output.stem + "_controls.npy"), controls)


def main() -> None:
    parser = argparse.ArgumentParser(description="Differentiable smoke MPC controlled by Vernon structured VJP")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--output", type=Path, default=Path("smoke_mpc.png"))
    arguments = parser.parse_args()
    if arguments.steps <= 0 or arguments.iterations <= 0 or arguments.learning_rate <= 0:
        parser.error("steps, iterations, and learning-rate must be positive")

    density, controls, losses = optimize(
        steps=arguments.steps,
        iterations=arguments.iterations,
        learning_rate=arguments.learning_rate,
    )
    target = v_target()
    save_results(arguments.output, density, controls, target)
    print(
        f"wrote {arguments.output}; first_predicted_loss={losses[0]:.6f} "
        f"last_predicted_loss={losses[-1]:.6f} "
        f"target_error={np.linalg.norm(density - target):.6f}"
    )


if __name__ == "__main__":
    main()
