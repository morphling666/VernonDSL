"""Chat with Qwen3 decoded by VernonDSL kernels.

    uv run --frozen --no-sync python examples/qwen3/generate.py --prompt "Why is the sky blue?"
    uv run --frozen --no-sync python examples/qwen3/generate.py            # interactive chat
    uv run --frozen --no-sync python examples/qwen3/generate.py --check    # compare with the NumPy reference

The first run downloads the checkpoint from the Hugging Face hub.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import vernon_dsl as vd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Qwen3Config, Qwen3Weights, ReferenceDecoder, VernonDecoder, download, read_safetensors  # noqa: E402

# Qwen3 ends an assistant turn with <|im_end|>; <|endoftext|> ends a document.
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


def user_turn(message: str, *, think: bool, first: bool) -> str:
    """A user message and the start of the assistant's reply in Qwen3's chat template; an empty thinking block
    turns thinking off, as the template's `enable_thinking=False` does."""
    turn = "" if first else "\n"
    turn += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
    return turn if think else turn + "<think>\n\n</think>\n\n"


class Sampler:
    def __init__(self, *, temperature: float, top_k: int, top_p: float, seed: int):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.rng = np.random.default_rng(seed)

    def __call__(self, logits: np.ndarray) -> int:
        if self.temperature <= 0.0:
            return int(np.argmax(logits))
        candidates = np.argpartition(logits, -self.top_k)[-self.top_k :] if self.top_k > 0 else np.arange(len(logits))
        candidates = candidates[np.argsort(logits[candidates])[::-1]]
        scaled = logits[candidates].astype(np.float64) / self.temperature
        probabilities = np.exp(scaled - scaled.max())
        probabilities /= probabilities.sum()
        # Keep the smallest prefix whose probability reaches top_p.
        kept = int(np.searchsorted(np.cumsum(probabilities), self.top_p)) + 1
        probabilities = probabilities[:kept] / probabilities[:kept].sum()
        return int(candidates[self.rng.choice(kept, p=probabilities)])


class Chat:
    """A conversation whose turns extend one KV cache."""

    def __init__(self, decoder: VernonDecoder, tokenizer, sampler: Sampler, *, think: bool):
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.think = think
        self.stops = {tokenizer.token_to_id(token) for token in STOP_TOKENS}
        self.end_of_turn = tokenizer.token_to_id(STOP_TOKENS[0])
        self.pos = 0
        start = time.perf_counter()
        decoder.reset()
        decoder.step(0, 0)
        decoder.reset()
        print(f"compiled kernels in {time.perf_counter() - start:.1f} s")

    def reply(self, message: str, max_new_tokens: int) -> None:
        prompt = self.tokenizer.encode(user_turn(message, think=self.think, first=self.pos == 0)).ids
        if self.pos + len(prompt) + max_new_tokens + 1 > self.decoder.max_positions:
            print("[context full: starting a new conversation]")
            self.decoder.reset()
            self.pos = 0
            prompt = self.tokenizer.encode(user_turn(message, think=self.think, first=True)).ids
        start = time.perf_counter()
        for index, token in enumerate(prompt):
            logits = self.decoder.step(token, self.pos, read_logits=index == len(prompt) - 1)
            self.pos += 1
        prefill = time.perf_counter() - start
        generated: list[int] = []
        shown = ""
        start = time.perf_counter()
        while len(generated) < max_new_tokens:
            token = self.sampler(logits)
            if token in self.stops:
                break
            generated.append(token)
            text = self.tokenizer.decode(generated)
            # A partial UTF-8 sequence decodes to U+FFFD; wait for the rest of the character.
            if not text.endswith("\ufffd"):
                print(text[len(shown) :], end="", flush=True)
                shown = text
            logits = self.decoder.step(token, self.pos)
            self.pos += 1
        decode = time.perf_counter() - start
        # The next turn's template continues after <|im_end|>, which ends the reply but was sampled, not decoded.
        self.decoder.step(self.end_of_turn, self.pos, read_logits=False)
        self.pos += 1
        print(
            f"\n[{len(prompt)} prompt tokens at {len(prompt) / prefill:.1f} tok/s, "
            f"{len(generated)} generated at {len(generated) / max(decode, 1e-9):.1f} tok/s, "
            f"{self.pos}/{self.decoder.max_positions} context]"
        )


def check(config: Qwen3Config, weights: Qwen3Weights, decoder: VernonDecoder, tokens: list[int]) -> None:
    """Decodes `tokens` with both decoders, reporting how far the VernonDSL logits are from the reference."""
    reference = ReferenceDecoder(config, weights, decoder.max_positions)
    decoder.reset()
    worst = 0.0
    for pos, token in enumerate(tokens):
        got = decoder.step(token, pos)
        want = reference.step(token, pos)
        error = float(np.abs(got - want).max() / np.abs(want).max())
        worst = max(worst, error)
        print(f"pos {pos:3d} token {token:6d}: relative error {error:.2e}, argmax {got.argmax()} / {want.argmax()}")
    print(f"worst relative logit error {worst:.2e}")
    if worst > 1e-3:
        raise SystemExit("the VernonDSL decoder differs from the reference")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Hugging Face repository of a Qwen3 checkpoint")
    parser.add_argument("--model-dir", type=Path, help="checkpoint directory; downloaded into ~/.cache by default")
    parser.add_argument("--arch", choices=("cuda", "vulkan", "directx", "opengl", "opengles", "cpu"), default="cuda")
    parser.add_argument("--prompt", help="answer one prompt and exit; chat interactively without one")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--context", type=int, default=2048, help="KV cache capacity in tokens")
    parser.add_argument("--think", action="store_true", help="let the model reason in a <think> block first")
    parser.add_argument("--temperature", type=float, default=0.7, help="0 decodes greedily")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true", help="compare the decoder with the NumPy reference")
    arguments = parser.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise SystemExit("Install the example dependencies with 'uv sync --extra examples --frozen'.") from error

    directory = arguments.model_dir or Path.home() / ".cache" / "vernon_qwen3" / arguments.model.split("/")[-1]
    download(arguments.model, directory)
    config = Qwen3Config.load(directory / "config.json")
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    start = time.perf_counter()
    weights = Qwen3Weights.from_checkpoint(config, read_safetensors(directory / "model.safetensors"))
    vd.init(arch=getattr(vd, arguments.arch))
    decoder = VernonDecoder(config, weights, arguments.context)
    print(f"loaded {arguments.model} in {time.perf_counter() - start:.1f} s")

    if arguments.check:
        prompt = user_turn("Why is the sky blue?", think=False, first=True)
        check(config, weights, decoder, tokenizer.encode(prompt).ids)
        return
    del weights
    sampler = Sampler(
        temperature=arguments.temperature, top_k=arguments.top_k, top_p=arguments.top_p, seed=arguments.seed
    )
    chat = Chat(decoder, tokenizer, sampler, think=arguments.think)
    if arguments.prompt is not None:
        chat.reply(arguments.prompt, arguments.max_new_tokens)
        return
    print("Type a message; an empty line exits.")
    while message := input("> ").strip():
        chat.reply(message, arguments.max_new_tokens)


if __name__ == "__main__":
    main()
