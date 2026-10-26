"""Qwen3 dense decoder: checkpoint loading, a NumPy reference, and VernonDSL decoding.

The VernonDSL decoder runs one token per invocation of `DecodeStep`, a Module, so a step's 8 * layers + 4
dispatches submit as one Program. A prompt is fed through the same step one token at a time.
"""

from __future__ import annotations

import json
import struct
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vernon_dsl as vd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernels  # noqa: E402

MODEL_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "model.safetensors")


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @classmethod
    def load(cls, path: Path) -> Qwen3Config:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{name: raw[name] for name in cls.__dataclass_fields__})


def _fetch(url: str, target: Path, attempts: int = 20) -> None:
    """Downloads `url` to `target`, resuming an interrupted download from its `.part` file."""
    # Each range uses a fresh connection: long-lived hub connections are often throttled to a crawl.
    block = 64 << 20
    partial = target.with_suffix(target.suffix + ".part")
    partial.touch()
    total = None
    failures = 0
    with partial.open("r+b") as file:
        file.seek(0, 2)
        have = file.tell()
        while total is None or have < total:
            request = urllib.request.Request(url, headers={"Range": f"bytes={have}-{have + block - 1}"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206:
                        raise RuntimeError(f"{url} does not support range requests")
                    total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
                    while chunk := response.read(1 << 20):
                        file.write(chunk)
                        have += len(chunk)
                        print(f"\r  {have / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            except OSError as error:
                failures += 1
                if failures == attempts:
                    raise RuntimeError(f"could not download {url}") from error
                print(f"\n  retrying after: {error}", flush=True)
                file.seek(0, 2)
                have = file.tell()
    print()
    partial.replace(target)


def download(repo: str, directory: Path) -> Path:
    """Downloads the files of `repo` from the Hugging Face hub into `directory`, skipping those present."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        target = directory / name
        if not target.exists():
            print(f"downloading {repo}/{name}", flush=True)
            _fetch(f"https://huggingface.co/{repo}/resolve/main/{name}", target)
    return directory


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """The tensors of a safetensors file as f32 arrays. BF16 widens exactly: it is the top half of an f32."""
    with path.open("rb") as file:
        (header_size,) = struct.unpack("<Q", file.read(8))
        header = json.loads(file.read(header_size))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + header_size)
    tensors: dict[str, np.ndarray] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        begin, end = entry["data_offsets"]
        raw = data[begin:end]
        if entry["dtype"] == "BF16":
            values = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif entry["dtype"] == "F32":
            values = np.array(raw.view(np.float32))
        else:
            raise ValueError(f"{name}: unsupported safetensors dtype {entry['dtype']}")
        tensors[name] = values.reshape(entry["shape"])
    return tensors


@dataclass
class Qwen3Weights:
    """Weights stacked over layers, in the layouts the kernels read."""

    embed: np.ndarray  # (1, vocab, hidden); also the output projection when embeddings are tied
    lm_head: np.ndarray  # (1, vocab, hidden)
    final_norm: np.ndarray  # (1, hidden)
    input_norm: np.ndarray  # (layers, hidden)
    post_norm: np.ndarray  # (layers, hidden)
    qkv: np.ndarray  # (layers, (q_heads + 2 * kv_heads) * head_dim, hidden)
    q_norm: np.ndarray  # (layers, head_dim)
    k_norm: np.ndarray  # (layers, head_dim)
    o: np.ndarray  # (layers, hidden, q_heads * head_dim)
    gate_up: np.ndarray  # (layers, 2 * intermediate, hidden), gate rows then up rows
    down: np.ndarray  # (layers, hidden, intermediate)

    @classmethod
    def from_checkpoint(cls, config: Qwen3Config, tensors: dict[str, np.ndarray]) -> Qwen3Weights:
        def stack(*suffixes: str) -> np.ndarray:
            return np.ascontiguousarray(
                np.stack(
                    [
                        np.concatenate([tensors[f"model.layers.{layer}.{suffix}"] for suffix in suffixes], axis=0)
                        for layer in range(config.num_hidden_layers)
                    ]
                ),
                dtype=np.float32,
            )

        embed = np.ascontiguousarray(tensors["model.embed_tokens.weight"][None], dtype=np.float32)
        lm_head = embed if config.tie_word_embeddings else np.ascontiguousarray(tensors["lm_head.weight"][None])
        return cls(
            embed=embed,
            lm_head=lm_head,
            final_norm=np.ascontiguousarray(tensors["model.norm.weight"][None], dtype=np.float32),
            input_norm=stack("input_layernorm.weight"),
            post_norm=stack("post_attention_layernorm.weight"),
            qkv=stack("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
            q_norm=stack("self_attn.q_norm.weight"),
            k_norm=stack("self_attn.k_norm.weight"),
            o=stack("self_attn.o_proj.weight"),
            gate_up=stack("mlp.gate_proj.weight", "mlp.up_proj.weight"),
            down=stack("mlp.down_proj.weight"),
        )

    @classmethod
    def random(cls, config: Qwen3Config, seed: int = 0) -> Qwen3Weights:
        """Weights of the checkpoint's shapes, scaled so activations stay of order one, for tests."""
        rng = np.random.default_rng(seed)
        layers, hidden, inner = config.num_hidden_layers, config.hidden_size, config.intermediate_size
        attention = config.num_attention_heads * config.head_dim
        qkv_rows = (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim

        def normal(*shape: int, fan_in: int) -> np.ndarray:
            return (rng.standard_normal(shape) / np.sqrt(fan_in)).astype(np.float32)

        def norm(*shape: int) -> np.ndarray:
            return (1.0 + 0.1 * rng.standard_normal(shape)).astype(np.float32)

        embed = normal(1, config.vocab_size, hidden, fan_in=1)
        return cls(
            embed=embed,
            lm_head=embed,
            final_norm=norm(1, hidden),
            input_norm=norm(layers, hidden),
            post_norm=norm(layers, hidden),
            qkv=normal(layers, qkv_rows, hidden, fan_in=hidden),
            q_norm=norm(layers, config.head_dim),
            k_norm=norm(layers, config.head_dim),
            o=normal(layers, hidden, attention, fan_in=attention),
            gate_up=normal(layers, 2 * inner, hidden, fan_in=hidden),
            down=normal(layers, hidden, inner, fan_in=inner),
        )


def rope_tables(config: Qwen3Config, positions: int) -> tuple[np.ndarray, np.ndarray]:
    """cos and sin of every position's angles, (positions, head_dim / 2), computed in f64."""
    exponents = np.arange(0, config.head_dim, 2, dtype=np.float64) / config.head_dim
    inverse_frequencies = config.rope_theta**-exponents
    angles = np.arange(positions, dtype=np.float64)[:, None] * inverse_frequencies[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


class ReferenceDecoder:
    """Qwen3 in NumPy f32, one token per step, following Hugging Face's Qwen3 modeling code."""

    def __init__(self, config: Qwen3Config, weights: Qwen3Weights, max_positions: int):
        self.config = config
        self.weights = weights
        self.cos, self.sin = rope_tables(config, max_positions)
        shape = (config.num_hidden_layers, config.num_key_value_heads, max_positions, config.head_dim)
        self.k_cache = np.zeros(shape, dtype=np.float32)
        self.v_cache = np.zeros(shape, dtype=np.float32)

    def _rms_norm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.config.rms_norm_eps) * weight

    def _rope(self, x: np.ndarray, pos: int) -> np.ndarray:
        half = self.config.head_dim // 2
        cos = np.concatenate([self.cos[pos], self.cos[pos]])
        sin = np.concatenate([self.sin[pos], self.sin[pos]])
        rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
        return x * cos + rotated * sin

    def step(self, token: int, pos: int) -> np.ndarray:
        config, weights = self.config, self.weights
        heads, kv_heads, head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        x = weights.embed[0, token].astype(np.float32)
        for layer in range(config.num_hidden_layers):
            qkv = weights.qkv[layer] @ self._rms_norm(x, weights.input_norm[layer])
            q = qkv[: heads * head_dim].reshape(heads, head_dim)
            k = qkv[heads * head_dim : (heads + kv_heads) * head_dim].reshape(kv_heads, head_dim)
            v = qkv[(heads + kv_heads) * head_dim :].reshape(kv_heads, head_dim)
            q = self._rope(self._rms_norm(q, weights.q_norm[layer]), pos)
            k = self._rope(self._rms_norm(k, weights.k_norm[layer]), pos)
            self.k_cache[layer, :, pos] = k
            self.v_cache[layer, :, pos] = v
            keys = np.repeat(self.k_cache[layer, :, : pos + 1], heads // kv_heads, axis=0)
            values = np.repeat(self.v_cache[layer, :, : pos + 1], heads // kv_heads, axis=0)
            scores = np.einsum("hd,htd->ht", q, keys) / np.sqrt(head_dim)
            probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            attention = np.einsum("ht,htd->hd", probabilities, values).reshape(-1)
            x = x + weights.o[layer] @ attention
            gate_up = weights.gate_up[layer] @ self._rms_norm(x, weights.post_norm[layer])
            gate, up = np.split(gate_up, 2)
            x = x + weights.down[layer] @ (gate / (1.0 + np.exp(-gate)) * up)
        return weights.lm_head[0] @ self._rms_norm(x, weights.final_norm[0])


class DecodeStep(vd.Module):
    """One decoding step: the logits of the token and position `step` holds, appending its keys and values."""

    def __init__(self, config: Qwen3Config, max_positions: int):
        super().__init__()
        self.layers = config.num_hidden_layers
        self.eps = np.float32(config.rms_norm_eps)
        self.kv_heads = np.int32(config.num_key_value_heads)
        self.hidden_groups = (config.hidden_size + kernels.LANES - 1) // kernels.LANES
        self.hidden_tiles = (config.hidden_size + kernels.HEAD_DIM - 1) // kernels.HEAD_DIM
        self.qkv_rows = (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim
        self.qk_heads = config.num_attention_heads + config.num_key_value_heads
        self.heads = config.num_attention_heads
        self.hidden = config.hidden_size
        self.gate_up_rows = 2 * config.intermediate_size
        self.vocab = config.vocab_size
        self.position_tiles = (max_positions + kernels.HEAD_DIM - 1) // kernels.HEAD_DIM

    def forward(
        self,
        embed: vd.TensorStorage,
        lm_head: vd.TensorStorage,
        final_norm: vd.TensorStorage,
        input_norm: vd.TensorStorage,
        post_norm: vd.TensorStorage,
        qkv_weight: vd.TensorStorage,
        q_norm: vd.TensorStorage,
        k_norm: vd.TensorStorage,
        o_weight: vd.TensorStorage,
        gate_up_weight: vd.TensorStorage,
        down_weight: vd.TensorStorage,
        cos: vd.TensorStorage,
        sin: vd.TensorStorage,
        k_cache: vd.TensorStorage,
        v_cache: vd.TensorStorage,
        step: vd.TensorStorage,
        hidden: vd.TensorStorage,
        normed: vd.TensorStorage,
        qkv: vd.TensorStorage,
        q: vd.TensorStorage,
        scores: vd.TensorStorage,
        attention: vd.TensorStorage,
        gate_up: vd.TensorStorage,
        logits: vd.TensorStorage,
    ) -> None:
        kernels.embed(embed, step, hidden, grid=(self.hidden_tiles, 1, 1))
        for layer in range(self.layers):
            kernels.rms_norm(hidden, input_norm, normed, layer, self.eps, grid=(self.hidden_groups, 1, 1))
            kernels.gemv(qkv_weight, normed, qkv, layer, grid=(self.qkv_rows, 1, 1))
            kernels.qk_norm_rope_store(
                qkv, q_norm, k_norm, cos, sin, step, q, k_cache, v_cache, layer, self.eps, grid=(1, self.qk_heads, 1)
            )
            kernels.attention_scores(
                q, k_cache, step, scores, layer, self.kv_heads, grid=(self.position_tiles, self.heads, 1)
            )
            kernels.attention_values(scores, v_cache, step, attention, layer, self.kv_heads, grid=(1, self.heads, 1))
            kernels.gemv_residual(o_weight, attention, hidden, layer, grid=(self.hidden, 1, 1))
            kernels.rms_norm(hidden, post_norm, normed, layer, self.eps, grid=(self.hidden_groups, 1, 1))
            kernels.gemv(gate_up_weight, normed, gate_up, layer, grid=(self.gate_up_rows, 1, 1))
            kernels.gemv_swiglu_residual(down_weight, gate_up, hidden, layer, grid=(self.hidden, 1, 1))
        kernels.rms_norm(hidden, final_norm, normed, 0, self.eps, grid=(self.hidden_groups, 1, 1))
        kernels.gemv(lm_head, normed, logits, 0, grid=(self.vocab, 1, 1))


class VernonDecoder:
    """Device storage for the weights, KV cache and activations of `DecodeStep`."""

    def __init__(self, config: Qwen3Config, weights: Qwen3Weights, max_positions: int):
        self.config = config
        self.max_positions = max_positions
        upload = vd.storage.from_numpy
        self.embed = upload(weights.embed)
        self.lm_head = self.embed if weights.lm_head is weights.embed else upload(weights.lm_head)
        self.final_norm = upload(weights.final_norm)
        self.input_norm = upload(weights.input_norm)
        self.post_norm = upload(weights.post_norm)
        self.qkv_weight = upload(weights.qkv)
        self.q_norm = upload(weights.q_norm)
        self.k_norm = upload(weights.k_norm)
        self.o_weight = upload(weights.o)
        self.gate_up_weight = upload(weights.gate_up)
        self.down_weight = upload(weights.down)
        cos, sin = rope_tables(config, max_positions)
        self.cos = upload(cos)
        self.sin = upload(sin)
        heads, kv_heads, head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        cache_shape = (config.num_hidden_layers * kv_heads, max_positions, head_dim)
        self.k_cache = vd.storage.zeros(dtype=vd.f32, shape=cache_shape)
        self.v_cache = vd.storage.zeros(dtype=vd.f32, shape=cache_shape)
        self.step_state = vd.storage.zeros(dtype=vd.i32, shape=(2,))
        self.hidden = vd.storage.zeros(dtype=vd.f32, shape=(config.hidden_size,))
        self.normed = vd.storage.zeros(dtype=vd.f32, shape=(config.hidden_size,))
        self.qkv = vd.storage.zeros(dtype=vd.f32, shape=((heads + 2 * kv_heads) * head_dim,))
        self.q = vd.storage.zeros(dtype=vd.f32, shape=(heads, head_dim))
        self.scores = vd.storage.zeros(dtype=vd.f32, shape=(heads, max_positions))
        self.attention = vd.storage.zeros(dtype=vd.f32, shape=(heads * head_dim,))
        self.gate_up = vd.storage.zeros(dtype=vd.f32, shape=(2 * config.intermediate_size,))
        self.logits = vd.storage.zeros(dtype=vd.f32, shape=(config.vocab_size,))
        self.decode_step = DecodeStep(config, max_positions)

    def reset(self) -> None:
        """Empties the KV cache; the cache is appended to by adding, so it must be zero before a generation."""
        slots = self.config.num_hidden_layers * self.config.num_key_value_heads
        for cache in (self.k_cache, self.v_cache):
            kernels.clear_cache(cache, grid=(1, self.max_positions, slots))

    def step(self, token: int, pos: int, *, read_logits: bool = True) -> np.ndarray | None:
        """Decodes `token` at `pos`; returns its logits unless `read_logits` is false, as while feeding a prompt."""
        if not 0 <= pos < self.max_positions:
            raise ValueError(f"position {pos} is outside the context of {self.max_positions} tokens")
        self.step_state.copy_from_numpy(np.array([pos, token], dtype=np.int32))
        self.decode_step(
            self.embed,
            self.lm_head,
            self.final_norm,
            self.input_norm,
            self.post_norm,
            self.qkv_weight,
            self.q_norm,
            self.k_norm,
            self.o_weight,
            self.gate_up_weight,
            self.down_weight,
            self.cos,
            self.sin,
            self.k_cache,
            self.v_cache,
            self.step_state,
            self.hidden,
            self.normed,
            self.qkv,
            self.q,
            self.scores,
            self.attention,
            self.gate_up,
            self.logits,
        )
        return self.logits.to_numpy() if read_logits else None
