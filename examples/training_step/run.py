"""Host time of a VJP training step over a chain of elementwise layers, split into the native forward call of the
linked VJP, its native backward call and the Python around them.

Each layer multiplies by its own 64-element weight, and a step is one forward call and its pullback, which returns a
gradient per weight. The Module's source is generated per layer count, because the compiler reads `forward()` from a
file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import platform
import sys
import tempfile
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import vernon_dsl as vd

clock = {"forward": 0.0, "backward": 0.0}


class TimedVjp:
    """A linked VJP whose native forward calls add their time to `clock["forward"]`, and whose calls' backward calls
    add theirs to `clock["backward"]`."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def call(self, arguments: Any, bound: Any = None) -> Any:
        start = time.perf_counter()
        result = self.inner.call(arguments, bound)
        clock["forward"] += time.perf_counter() - start
        return None if result is None else TimedCall(result)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class TimedCall:
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def backward(self, arguments: Any, bound: Any = None) -> Any:
        start = time.perf_counter()
        result = self.inner.backward(arguments, bound)
        clock["backward"] += time.perf_counter() - start
        return result

    def pullback(self, bindings: Any) -> Any:
        return self.inner.pullback(bindings)


def chain(layers: int, directory: pathlib.Path) -> tuple[type, list[str]]:
    names = [f"w{i}" for i in range(layers)]
    view = "vd.TensorView[vd.f32, (vd.dyn,), vd.read]"
    parameters = ", ".join(f"{name}: {view}" for name in names)
    body = ["        x = source"]
    for name in names:
        body += ["        y = vd.empty_like(source)", f"        mul(x, {name}, y, grid=(1, 1, 1))", "        x = y"]
    code = (
        "import vernon_dsl as vd\nfrom kernels import mul\n\n\nclass Net(vd.Module):\n"
        f"    def forward(self, source: {view}, {parameters}) -> vd.TensorStorage:\n"
        + "\n".join(body)
        + "\n        return x\n"
    )
    path = directory / f"chain_{layers}.py"
    path.write_text(code)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Net, names


def measure(layers: int, steps: int, directory: pathlib.Path, bound: bool) -> dict[str, float]:
    net, names = chain(layers, directory)
    source = vd.storage.from_numpy(np.full(64, 1.0, np.float32))
    weights = [vd.storage.from_numpy(np.full(64, 1.0, np.float32)) for _ in names]
    step = vd.ad.vjp(net(), wrt=tuple(names), outputs=None)
    if bound:
        # The weights bind once and their gradients land in the same Storages every step.
        step = step.bind(**dict(zip(names, weights, strict=True)))
        weights = []
    for _ in range(3):
        _, pullback = step(source, *weights)
        pullback()
    linked = getattr(step, "_fast", None) if bound else getattr(pullback, "_specialization", None)
    if linked is None or linked.link.vjp is None:
        raise RuntimeError("the training step did not link")
    linked.link.vjp = TimedVjp(linked.link.vjp)
    clock.update(forward=0.0, backward=0.0)
    start = time.perf_counter()
    for _ in range(steps):
        _, pullback = step(source, *weights)
        pullback()
    total = (time.perf_counter() - start) / steps * 1e6
    forward, backward = clock["forward"] / steps * 1e6, clock["backward"] / steps * 1e6
    # A step started on an idle device never waits for an earlier step's record slot, so its time is the host's.
    idle = []
    for _ in range(min(steps, 200)):
        output, pullback = step(source, *weights)
        gradients = pullback()
        next(iter(gradients.values())).wait()
        output.wait()
        begin = time.perf_counter()
        _, pullback = step(source, *weights)
        pullback()
        idle.append(time.perf_counter() - begin)
    return {
        "step_us": total,
        "forward_us": forward,
        "backward_us": backward,
        "python_us": total - forward - backward,
        "idle_step_us": float(np.median(idle)) * 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 8, 64, 256])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--bound", action="store_true", help="bind the weights once with step.bind")
    parser.add_argument("--result-json", type=pathlib.Path)
    arguments = parser.parse_args()
    vd.init(arch=vd.cuda if arguments.arch == "cuda" else vd.cpu)
    results: dict[str, Any] = {
        "arch": arguments.arch,
        "bound": arguments.bound,
        "platform": platform.platform(),
        "layers": {},
    }
    with tempfile.TemporaryDirectory() as directory:
        for layers in arguments.layers:
            row = measure(layers, arguments.steps, pathlib.Path(directory), arguments.bound)
            results["layers"][str(layers)] = row
            print(
                f"layers {layers:4d}  step {row['step_us']:8.1f} us  forward {row['forward_us']:7.1f}"
                f"  backward {row['backward_us']:7.1f}  python {row['python_us']:7.1f}"
                f"  idle step {row['idle_step_us']:8.1f}",
                flush=True,
            )
    if arguments.result_json is not None:
        arguments.result_json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
